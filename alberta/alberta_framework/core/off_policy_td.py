"""Off-policy TD learner with importance sampling (Step 3 Phase E).

Implements per-decision importance sampling with optional Retrace-style
ratio clipping for stable off-policy linear value function learning.

Theoretical background:
    TD with linear function approximation is **not** guaranteed to
    converge under off-policy distributions (Baird 1995, Counterexample
    to TD with FA). Several remedies exist:

    1. Per-decision importance sampling (Precup, Sutton, Singh 2000):
       multiply each step's update by rho_t = pi(a_t|s_t) / b(a_t|s_t)
       so that on average we are simulating the on-policy distribution.
       Variance can be very large.
    2. Retrace ratio clipping (Munos et al. 2016): use
       rho_clipped = min(c, rho_t). Convergent for c <= 1; for c > 1 it
       trades bias for variance reduction.
    3. Gradient-TD (TDC, GQ-lambda) (Sutton, Maei, et al. 2009-2010):
       gradient descent on the projected Bellman error.
    4. Emphatic TD (Sutton, Mahmood, White 2016): emphasis traces F_t
       restore on-policy convergence proofs without a secondary weight
       vector.

    This module implements (1), (2), TDC-style Gradient-TD from (3) via
    :class:`GradientTDLinearLearner` (which maintains the required secondary
    weight vector), and ETD(lambda) from (4).

The learner has a simple interface::

    learner = OffPolicyTDLinearLearner(step_size=0.05, retrace_clip=1.0)
    state = learner.init(feature_dim)
    for t in range(T):
        rho_t = pi(a_t | s_t) / b(a_t | s_t)
        result = learner.update(state, obs_t, reward, next_obs, gamma, rho_t)
        state = result.state

Setting ``rho_t = 1.0`` reduces this to standard semi-gradient TD(0).

Use cases (Step 3 DoD-5):
    - Counterfactual prediction: "what would value be under target policy?"
    - Auxiliary Horde demons learning about hand-specified target policies.
    - Baird counterexample / divergence-prevention demonstrations.

Reference:
    Precup, D., Sutton, R.S., & Singh, S. (2000). Eligibility traces for
    off-policy policy evaluation. *ICML*.
    Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. (2016).
    Safe and efficient off-policy reinforcement learning. *NeurIPS*.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.types import Observation

OFF_POLICY_TD_CONFIG_SCHEMA = "alberta.off-policy-td-linear-config.v2"
OFF_POLICY_TD_STATE_SCHEMA = "alberta.off-policy-td-linear-state.v2"
ETD_CONFIG_SCHEMA = "alberta.etd-linear-config.v2"
ETD_STATE_SCHEMA = "alberta.etd-linear-state.v2"
GRADIENT_TD_CONFIG_SCHEMA = "alberta.gradient-td-linear-config.v2"
GRADIENT_TD_STATE_SCHEMA = "alberta.gradient-td-linear-state.v2"
OFF_POLICY_TD_RESOURCE_SCHEMA = "alberta.off-policy-td-resource-budget.v2"
OFF_POLICY_TD_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
OFF_POLICY_TD_MAX_COMMITTED_UPDATES = 2**64 - 1
OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES = 12
OFF_POLICY_TD_LIFETIME_COUNTER_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_FLOAT32_MIN_POSITIVE = float.fromhex("0x1p-149")


def _require_exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Copy a host mapping after rejecting missing and unknown fields."""

    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _require_real(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
    allow_positive_infinity: bool = False,
) -> float:
    """Validate one host hyperparameter without accepting booleans or NaNs."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    scalar = float(value)
    if allow_positive_infinity and scalar == math.inf:
        return scalar
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    if scalar < minimum or (maximum is not None and scalar > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label} must be {interval}; got {value}")
    return scalar


def _require_positive_feature_dim(feature_dim: Any) -> int:
    """Return a positive exact feature dimension."""

    if type(feature_dim) is not int or feature_dim < 1:
        raise ValueError("feature_dim must be a positive exact integer")
    return feature_dim


def _require_float32_vector(value: Any, feature_dim: int, *, label: str) -> None:
    """Validate one public feature-vector boundary without silent coercion."""

    if getattr(value, "shape", None) != (feature_dim,):
        raise ValueError(f"{label} must have shape ({feature_dim},)")
    if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
        raise TypeError(f"{label} must have dtype float32")


def _as_float32_scalar(value: Any, *, label: str) -> Float[Array, ""]:
    """Accept an exact host real or require a scalar float32 array."""

    if isinstance(value, Real) and not isinstance(value, bool):
        return jnp.asarray(float(value), dtype=jnp.float32)
    if getattr(value, "shape", None) != ():
        raise ValueError(f"{label} must be scalar")
    if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
        raise TypeError(f"{label} must have dtype float32")
    return jnp.asarray(value)


def _metadata_valid(value: Any) -> Bool[Array, ""]:
    """Validate legacy timing metadata without including it in bit accounting."""

    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError("timing metadata must be scalar")
    if not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError("timing metadata must use a floating dtype")
    return jnp.isfinite(scalar) & (scalar >= 0.0)


def _learning_arrays_finite(*values: Array) -> Bool[Array, ""]:
    """Return whether all persistent learning arrays are finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for value in values:
        valid = valid & jnp.all(jnp.isfinite(value))
    return valid


def _host_state_mapping(state: Any, *, label: str) -> dict[str, Any]:
    """Return an exact shallow mapping for an explicit legacy migration."""

    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {field.name: getattr(state, field.name) for field in dataclasses.fields(state)}
    raise TypeError(f"legacy {label} must be a mapping or dataclass instance")


def _legacy_step_words(step_count: Any, *, label: str) -> UInt[Array, " 2"]:
    """Migrate one unambiguous pre-v2 int32 counter to an exact identity."""

    counter = jnp.asarray(step_count)
    if counter.shape != () or counter.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {label} step_count must be scalar int32")
    step = int(counter)
    if step < 0:
        raise ValueError(f"negative legacy {label} step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError(f"saturated legacy {label} step_count is ambiguous")
    return jnp.asarray((0, step), dtype=jnp.uint32)


# =============================================================================
# State / result types
# =============================================================================


@chex.dataclass(frozen=True)
class OffPolicyTDState:
    """State for the off-policy linear TD learner.

    Attributes:
        weights: Weight vector for linear value approximation
        bias: Bias term
        eligibility_traces: Per-feature eligibility trace
        bias_eligibility_trace: Bias eligibility trace
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian ``[high, low]`` uint32 lifetime identity.
        birth_timestamp: Non-learning lifecycle metadata, excluded from the
            persistent JAX learning-state bit contract. New and explicitly
            migrated states store it as float32 so rollback remains bitwise.
        uptime_s: Non-learning timing metadata with the same limitation.
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class OffPolicyTDUpdateResult:
    """Result of an off-policy TD update.

    Attributes:
        state: Updated learner state
        prediction: V(s) computed before the update
        td_error: TD error delta = R + gamma * V(s') - V(s)
        rho_clipped: Importance-sampling ratio after clipping (so it can
            be logged for variance diagnostics)
        metrics: Array of shape (5,) with columns
            [squared_td_error, td_error, rho_clipped, mean_alpha, mean_trace]
        pre_step_words: Exact lifetime identity before this transaction.
        post_step_words: Exact identity after commit or rollback.
        lifetime_counter_valid: Whether exact identity and telemetry agreed.
        lifetime_capacity_available: Whether the exact clock could advance.
        source_valid: Whether every transition value was finite and in-domain.
        state_valid: Whether the persistent source learning state was finite.
        candidate_valid: Whether the staged complete state was finite/authentic.
        update_applied: Whether the complete transaction committed atomically.
    """

    state: OffPolicyTDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    rho_clipped: Float[Array, ""]
    metrics: Float[Array, " 5"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ETDState:
    """State for the emphatic TD(lambda) linear learner.

    Attributes:
        weights: Weight vector for linear value approximation
        bias: Bias term
        eligibility_traces: Emphatic eligibility trace
        bias_eligibility_trace: Emphatic eligibility trace for the bias
        follow_on_trace: Scalar follow-on trace ``F_t``
        emphasis: Scalar emphasis ``M_t`` from the latest update
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian ``[high, low]`` uint32 lifetime identity.
        birth_timestamp: Non-learning lifecycle metadata excluded from the
            persistent learning-state bit contract.
        uptime_s: Non-learning timing metadata with the same limitation.
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    follow_on_trace: Float[Array, ""]
    emphasis: Float[Array, ""]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class ETDUpdateResult:
    """Result of an emphatic TD(lambda) update.

    Attributes:
        state: Updated learner state
        prediction: V(s) computed before the update
        td_error: TD error delta = R + gamma * V(s') - V(s)
        follow_on_trace: Updated follow-on trace ``F_t``
        emphasis: Updated scalar emphasis ``M_t``
        metrics: Array of shape (7,) with columns
            [squared_td_error, td_error, rho, mean_alpha, mean_trace,
            follow_on_trace, emphasis]
    """

    state: ETDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    follow_on_trace: Float[Array, ""]
    emphasis: Float[Array, ""]
    metrics: Float[Array, " 7"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientTDState:
    """State for linear off-policy Gradient-TD/TDC prediction.

    The bias is represented by an appended constant feature, so all vectors have
    shape ``feature_dim + 1``.
    """

    weights: Float[Array, " augmented_feature_dim"]
    secondary_weights: Float[Array, " augmented_feature_dim"]
    eligibility_traces: Float[Array, " augmented_feature_dim"]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class GradientTDUpdateResult:
    """Result of one linear Gradient-TD/TDC update."""

    state: GradientTDState
    prediction: Float[Array, " 1"]
    td_error: Float[Array, ""]
    rho_clipped: Float[Array, ""]
    metrics: Float[Array, " 6"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientTDArrayResult:
    """Result from scanning Gradient-TD/TDC over transition arrays."""

    state: GradientTDState
    predictions: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    rho_clipped: Float[Array, " num_steps"]
    metrics: Float[Array, "num_steps 6"]
    updates_applied: Bool[Array, " num_steps"]


@dataclass(frozen=True)
class OffPolicyTDResourceBudget:
    """Exact persistent JAX learning-state accounting for a linear TD learner.

    ``birth_timestamp`` and ``uptime_s`` are deliberately excluded: they are
    lifecycle diagnostics rather than persistent learning state.
    """

    learner_type: str
    feature_dim: int
    parameter_nbytes: int
    trace_nbytes: int
    auxiliary_nbytes: int
    lifetime_counter_nbytes: int
    state_nbytes: int

    def to_dict(self) -> dict[str, int | str]:
        """Serialize this exact resource contract."""

        return {
            "schema": OFF_POLICY_TD_RESOURCE_SCHEMA,
            "learner_type": self.learner_type,
            "feature_dim": self.feature_dim,
            "parameter_nbytes": self.parameter_nbytes,
            "trace_nbytes": self.trace_nbytes,
            "auxiliary_nbytes": self.auxiliary_nbytes,
            "lifetime_counter_nbytes": self.lifetime_counter_nbytes,
            "state_nbytes": self.state_nbytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OffPolicyTDResourceBudget:
        """Restore a resource contract while rejecting schema/tamper drift."""

        fields = _require_exact_manifest(
            payload,
            {
                "schema",
                "learner_type",
                "feature_dim",
                "parameter_nbytes",
                "trace_nbytes",
                "auxiliary_nbytes",
                "lifetime_counter_nbytes",
                "state_nbytes",
            },
            label="off-policy TD resource budget",
        )
        if fields.pop("schema") != OFF_POLICY_TD_RESOURCE_SCHEMA:
            raise ValueError("unsupported off-policy TD resource schema")
        learner_type = fields.get("learner_type")
        if learner_type not in {
            "OffPolicyTDLinearLearner",
            "ETDLinearLearner",
            "GradientTDLinearLearner",
        }:
            raise ValueError("off-policy TD resource learner_type is invalid")
        numeric_names = set(fields) - {"learner_type"}
        if any(type(fields[name]) is not int or fields[name] < 0 for name in numeric_names):
            raise ValueError("off-policy TD resource values must be non-negative integers")
        budget = cls(**fields)
        if budget.feature_dim < 1:
            raise ValueError("off-policy TD resource feature_dim must be positive")
        if budget.lifetime_counter_nbytes != OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES:
            raise ValueError("off-policy TD lifetime byte accounting is inconsistent")
        expected = _resource_budget_for(budget.learner_type, budget.feature_dim)
        if budget != expected:
            raise ValueError("off-policy TD resource byte accounting is inconsistent")
        return budget


def _resource_budget_for(learner_type: str, feature_dim: int) -> OffPolicyTDResourceBudget:
    """Construct one exact history-independent resource contract."""

    feature_dim = _require_positive_feature_dim(feature_dim)
    augmented_nbytes = 4 * (feature_dim + 1)
    if learner_type == "OffPolicyTDLinearLearner":
        parameter_nbytes = augmented_nbytes
        trace_nbytes = augmented_nbytes
        auxiliary_nbytes = 0
    elif learner_type == "ETDLinearLearner":
        parameter_nbytes = augmented_nbytes
        trace_nbytes = augmented_nbytes
        auxiliary_nbytes = 8
    elif learner_type == "GradientTDLinearLearner":
        parameter_nbytes = 2 * augmented_nbytes
        trace_nbytes = augmented_nbytes
        auxiliary_nbytes = 0
    else:
        raise ValueError("unsupported off-policy TD resource learner_type")
    total = (
        parameter_nbytes + trace_nbytes + auxiliary_nbytes + OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES
    )
    return OffPolicyTDResourceBudget(
        learner_type=learner_type,
        feature_dim=feature_dim,
        parameter_nbytes=parameter_nbytes,
        trace_nbytes=trace_nbytes,
        auxiliary_nbytes=auxiliary_nbytes,
        lifetime_counter_nbytes=OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES,
        state_nbytes=total,
    )


def _measure_arrays(*values: Array) -> int:
    """Measure exact bytes in explicit persistent JAX array fields."""

    return sum(int(value.size) * int(value.dtype.itemsize) for value in values)


# =============================================================================
# Learner
# =============================================================================


class OffPolicyTDLinearLearner:
    """Off-policy linear TD(lambda) with per-decision IS and Retrace clipping.

    The update rule is::

        rho_t = pi(a_t|s_t) / b(a_t|s_t)               (provided externally)
        rho_clipped = min(c, rho_t)                     (Retrace clipping)
        delta_t = R_{t+1} + gamma_t * V(s_{t+1}) - V(s_t)
        e_t = gamma_t * lambda_t * rho_clipped * e_{t-1} + phi_t
        w_{t+1} = w_t + alpha * delta_t * rho_clipped * e_t

    Setting ``retrace_clip = inf`` recovers naive per-decision IS.
    Setting ``retrace_clip = 1.0`` gives the Retrace-c=1 update which is
    convergent under standard conditions. Setting ``rho_t = 1`` always
    recovers on-policy semi-gradient TD(lambda).

    Attributes:
        step_size: Learning rate alpha
        trace_decay: Eligibility trace decay lambda
        retrace_clip: Maximum allowed importance ratio (Inf to disable)
    """

    def __init__(
        self,
        step_size: float = 0.05,
        trace_decay: float = 0.0,
        retrace_clip: float = 1.0,
    ):
        """Initialize the off-policy TD learner.

        Args:
            step_size: Learning rate alpha (scalar)
            trace_decay: Eligibility trace decay lambda in [0, 1]
            retrace_clip: Maximum allowed importance ratio (default 1.0
                is the safe Retrace-c=1 choice; pass float("inf") to
                disable clipping).
        """
        self._step_size = _require_real(
            step_size,
            label="step_size",
            minimum=_FLOAT32_MIN_POSITIVE,
        )
        self._trace_decay = _require_real(
            trace_decay,
            label="trace_decay",
            minimum=0.0,
            maximum=1.0,
        )
        self._retrace_clip = _require_real(
            retrace_clip,
            label="retrace_clip",
            minimum=_FLOAT32_MIN_POSITIVE,
            allow_positive_infinity=True,
        )

    @property
    def step_size(self) -> float:
        """Learning rate alpha."""
        return self._step_size

    @property
    def trace_decay(self) -> float:
        """Trace decay lambda."""
        return self._trace_decay

    @property
    def retrace_clip(self) -> float:
        """IS-ratio clip (Retrace c)."""
        return self._retrace_clip

    def resource_budget(self, feature_dim: int) -> OffPolicyTDResourceBudget:
        """Return the exact persistent-state resource contract."""

        return _resource_budget_for("OffPolicyTDLinearLearner", feature_dim)

    def init(self, feature_dim: int) -> OffPolicyTDState:
        """Initialize learner state with zero weights and zero traces."""
        feature_dim = _require_positive_feature_dim(feature_dim)
        return OffPolicyTDState(  # type: ignore[call-arg]
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            birth_timestamp=jnp.asarray(time.time(), dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _validate_state_structure(self, state: OffPolicyTDState) -> Bool[Array, ""]:
        """Validate array layout and return exact-clock authenticity."""

        if not isinstance(state, OffPolicyTDState):
            raise TypeError("state must be an OffPolicyTDState")
        feature_dim = state.weights.shape[0] if state.weights.ndim == 1 else -1
        if feature_dim < 1:
            raise ValueError("state weights must be a non-empty vector")
        _require_float32_vector(state.weights, feature_dim, label="state.weights")
        _require_float32_vector(
            state.eligibility_traces,
            feature_dim,
            label="state.eligibility_traces",
        )
        for name, value in (
            ("state.bias", state.bias),
            ("state.bias_eligibility_trace", state.bias_eligibility_trace),
        ):
            if getattr(value, "shape", None) != ():
                raise ValueError(f"{name} must be scalar")
            if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        _metadata_valid(state.birth_timestamp)
        _metadata_valid(state.uptime_s)
        return _lifetime_counter_valid(state.step_words, state.step_count)

    def state_valid(self, state: OffPolicyTDState) -> Bool[Array, ""]:
        """Return whether a state is finite and has an authentic exact clock."""

        clock_valid = self._validate_state_structure(state)
        return (
            clock_valid
            & _learning_arrays_finite(
                state.weights,
                state.bias,
                state.eligibility_traces,
                state.bias_eligibility_trace,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: OffPolicyTDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute V(s) = w . phi(s) + b."""
        self._validate_state_structure(state)
        _require_float32_vector(
            observation,
            state.weights.shape[0],
            label="observation",
        )
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OffPolicyTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> OffPolicyTDUpdateResult:
        """Apply one off-policy TD update.

        Args:
            state: Current learner state
            observation: Current feature vector phi(s_t)
            reward: Reward R_{t+1}
            next_observation: Next feature vector phi(s_{t+1})
            gamma: State-dependent discount gamma_t (0 at terminal)
            rho: Importance-sampling ratio pi(a_t|s_t) / b(a_t|s_t).
                Pass 1.0 for on-policy data.

        Returns:
            ``OffPolicyTDUpdateResult`` with updated state, prediction,
            TD error, clipped IS ratio, and a metrics array of shape (5,).
        """
        lifetime_counter_valid = self._validate_state_structure(state)
        feature_dim = state.weights.shape[0]
        _require_float32_vector(observation, feature_dim, label="observation")
        _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        clip = jnp.asarray(self._retrace_clip, dtype=jnp.float32)
        gamma_s = _as_float32_scalar(gamma, label="gamma")
        reward_s = _as_float32_scalar(reward, label="reward")
        rho_s = _as_float32_scalar(rho, label="rho")

        rho_clipped = jnp.minimum(rho_s, clip)
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(reward_s)
            & jnp.isfinite(gamma_s)
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
            & (rho_s >= 0.0)
        )
        state_valid = (
            _learning_arrays_finite(
                state.weights,
                state.bias,
                state.eligibility_traces,
                state.bias_eligibility_trace,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        v_t = jnp.dot(state.weights, observation) + state.bias
        v_next = jnp.dot(state.weights, next_observation) + state.bias
        td_error = reward_s + gamma_s * v_next - v_t

        # IS-weighted accumulating eligibility trace
        decay = gamma_s * lam * rho_clipped
        new_e = decay * state.eligibility_traces + observation
        new_e_b = decay * state.bias_eligibility_trace + 1.0

        # Update with rho_clipped * delta * e
        scaled_update = alpha * rho_clipped * td_error
        new_weights = state.weights + scaled_update * new_e
        new_bias = state.bias + scaled_update * new_e_b

        new_state = OffPolicyTDState(  # type: ignore[call-arg]
            weights=new_weights,
            bias=new_bias,
            eligibility_traces=new_e,
            bias_eligibility_trace=new_e_b,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )

        candidate_valid = (
            _lifetime_counter_valid(new_state.step_words, new_state.step_count)
            & _learning_arrays_finite(
                new_state.weights,
                new_state.bias,
                new_state.eligibility_traces,
                new_state.bias_eligibility_trace,
            )
            & _metadata_valid(new_state.birth_timestamp)
            & _metadata_valid(new_state.uptime_s)
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & state_valid
            & candidate_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )

        squared_td = td_error**2
        mean_e = jnp.mean(jnp.abs(new_e))
        metrics = jnp.array(
            [squared_td, td_error, rho_clipped, alpha, mean_e],
            dtype=jnp.float32,
        )

        return OffPolicyTDUpdateResult(  # type: ignore[call-arg]
            state=committed_state,
            prediction=jnp.atleast_1d(v_t),
            td_error=jnp.asarray(td_error),
            rho_clipped=jnp.asarray(rho_clipped),
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_valid=source_valid,
            state_valid=state_valid,
            candidate_valid=candidate_valid,
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "schema": OFF_POLICY_TD_CONFIG_SCHEMA,
            "state_schema": OFF_POLICY_TD_STATE_SCHEMA,
            "type": "OffPolicyTDLinearLearner",
            "step_size": self._step_size,
            "trace_decay": self._trace_decay,
            "retrace_clip": self._retrace_clip,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OffPolicyTDLinearLearner:
        """Reconstruct from dict."""
        fields = _require_exact_manifest(
            config,
            {
                "schema",
                "state_schema",
                "type",
                "step_size",
                "trace_decay",
                "retrace_clip",
            },
            label="off-policy TD config",
        )
        if fields.pop("schema") != OFF_POLICY_TD_CONFIG_SCHEMA:
            raise ValueError("unsupported off-policy TD config schema")
        if fields.pop("state_schema") != OFF_POLICY_TD_STATE_SCHEMA:
            raise ValueError("unsupported off-policy TD state schema")
        if fields.pop("type") != "OffPolicyTDLinearLearner":
            raise ValueError("off-policy TD config type is invalid")
        return cls(**fields)


class ETDLinearLearner:
    """Off-policy linear emphatic TD(lambda).

    ETD(lambda) replaces Retrace's clipped per-decision trace with a
    follow-on trace and scalar emphasis:

    ``F_t = rho_t * gamma_t * F_{t-1} + i_t``
    ``M_t = lambda * i_t + (1 - lambda) * F_t``
    ``e_t = rho_t * (gamma_t * lambda * e_{t-1} + M_t * phi_t)``
    ``w_{t+1} = w_t + alpha * delta_t * e_t``

    The single-step API advances the follow-on trace with the current
    transition's ratio and discount. With ``rho=1``, ``gamma=0``, and
    ``lambda=0``, this reduces to the standard LMS/TD(0) terminating update.

    Attributes:
        step_size: Learning rate alpha
        trace_decay: Eligibility trace decay lambda
    """

    def __init__(
        self,
        step_size: float = 0.05,
        trace_decay: float = 0.0,
    ):
        """Initialize the emphatic TD learner.

        Args:
            step_size: Learning rate alpha (scalar)
            trace_decay: Eligibility trace decay lambda in [0, 1]
        """
        self._step_size = _require_real(
            step_size,
            label="step_size",
            minimum=_FLOAT32_MIN_POSITIVE,
        )
        self._trace_decay = _require_real(
            trace_decay,
            label="trace_decay",
            minimum=0.0,
            maximum=1.0,
        )

    @property
    def step_size(self) -> float:
        """Learning rate alpha."""
        return self._step_size

    @property
    def trace_decay(self) -> float:
        """Trace decay lambda."""
        return self._trace_decay

    def resource_budget(self, feature_dim: int) -> OffPolicyTDResourceBudget:
        """Return the exact persistent-state resource contract."""

        return _resource_budget_for("ETDLinearLearner", feature_dim)

    def init(self, feature_dim: int) -> ETDState:
        """Initialize learner state with zero weights and zero traces."""
        feature_dim = _require_positive_feature_dim(feature_dim)
        return ETDState(  # type: ignore[call-arg]
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            follow_on_trace=jnp.array(0.0, dtype=jnp.float32),
            emphasis=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            birth_timestamp=jnp.asarray(time.time(), dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _validate_state_structure(self, state: ETDState) -> Bool[Array, ""]:
        """Validate array layout and return exact-clock authenticity."""

        if not isinstance(state, ETDState):
            raise TypeError("state must be an ETDState")
        feature_dim = state.weights.shape[0] if state.weights.ndim == 1 else -1
        if feature_dim < 1:
            raise ValueError("state weights must be a non-empty vector")
        _require_float32_vector(state.weights, feature_dim, label="state.weights")
        _require_float32_vector(
            state.eligibility_traces,
            feature_dim,
            label="state.eligibility_traces",
        )
        for name, value in (
            ("state.bias", state.bias),
            ("state.bias_eligibility_trace", state.bias_eligibility_trace),
            ("state.follow_on_trace", state.follow_on_trace),
            ("state.emphasis", state.emphasis),
        ):
            if getattr(value, "shape", None) != ():
                raise ValueError(f"{name} must be scalar")
            if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError(f"{name} must have dtype float32")
        _metadata_valid(state.birth_timestamp)
        _metadata_valid(state.uptime_s)
        return _lifetime_counter_valid(state.step_words, state.step_count)

    def state_valid(self, state: ETDState) -> Bool[Array, ""]:
        """Return whether a state is finite and has an authentic exact clock."""

        return (
            self._validate_state_structure(state)
            & _learning_arrays_finite(
                state.weights,
                state.bias,
                state.eligibility_traces,
                state.bias_eligibility_trace,
                state.follow_on_trace,
                state.emphasis,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: ETDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute V(s) = w . phi(s) + b."""
        self._validate_state_structure(state)
        _require_float32_vector(
            observation,
            state.weights.shape[0],
            label="observation",
        )
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ETDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
        interest: Array | float = 1.0,
    ) -> ETDUpdateResult:
        """Apply one ETD(lambda) update.

        Args:
            state: Current learner state
            observation: Current feature vector phi(s_t)
            reward: Reward R_{t+1}
            next_observation: Next feature vector phi(s_{t+1})
            gamma: State-dependent discount gamma (0 at terminal)
            rho: Importance-sampling ratio pi(a_t|s_t) / b(a_t|s_t).
            interest: State interest i_t. Defaults to 1.0.

        Returns:
            ``ETDUpdateResult`` with updated state, prediction, TD error,
            follow-on trace, emphasis, and a metrics array of shape (7,).
        """
        lifetime_counter_valid = self._validate_state_structure(state)
        feature_dim = state.weights.shape[0]
        _require_float32_vector(observation, feature_dim, label="observation")
        _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        gamma_s = _as_float32_scalar(gamma, label="gamma")
        reward_s = _as_float32_scalar(reward, label="reward")
        rho_s = _as_float32_scalar(rho, label="rho")
        interest_s = _as_float32_scalar(interest, label="interest")
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(reward_s)
            & jnp.isfinite(gamma_s)
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
            & (rho_s >= 0.0)
            & jnp.isfinite(interest_s)
            & (interest_s >= 0.0)
        )
        state_valid = (
            _learning_arrays_finite(
                state.weights,
                state.bias,
                state.eligibility_traces,
                state.bias_eligibility_trace,
                state.follow_on_trace,
                state.emphasis,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        v_t = jnp.dot(state.weights, observation) + state.bias
        v_next = jnp.dot(state.weights, next_observation) + state.bias
        td_error = reward_s + gamma_s * v_next - v_t

        follow_on = rho_s * gamma_s * state.follow_on_trace + interest_s
        emphasis = lam * interest_s + (1.0 - lam) * follow_on

        trace_decay = gamma_s * lam
        new_e = rho_s * (trace_decay * state.eligibility_traces + emphasis * observation)
        new_e_b = rho_s * (trace_decay * state.bias_eligibility_trace + emphasis)

        new_weights = state.weights + alpha * td_error * new_e
        new_bias = state.bias + alpha * td_error * new_e_b

        new_state = ETDState(  # type: ignore[call-arg]
            weights=new_weights,
            bias=new_bias,
            eligibility_traces=new_e,
            bias_eligibility_trace=new_e_b,
            follow_on_trace=follow_on,
            emphasis=emphasis,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )

        candidate_valid = (
            _lifetime_counter_valid(new_state.step_words, new_state.step_count)
            & _learning_arrays_finite(
                new_state.weights,
                new_state.bias,
                new_state.eligibility_traces,
                new_state.bias_eligibility_trace,
                new_state.follow_on_trace,
                new_state.emphasis,
            )
            & _metadata_valid(new_state.birth_timestamp)
            & _metadata_valid(new_state.uptime_s)
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & state_valid
            & candidate_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )

        squared_td = td_error**2
        mean_e = jnp.mean(jnp.abs(new_e))
        metrics = jnp.array(
            [squared_td, td_error, rho_s, alpha, mean_e, follow_on, emphasis],
            dtype=jnp.float32,
        )

        return ETDUpdateResult(  # type: ignore[call-arg]
            state=committed_state,
            prediction=jnp.atleast_1d(v_t),
            td_error=jnp.asarray(td_error),
            follow_on_trace=jnp.asarray(follow_on),
            emphasis=jnp.asarray(emphasis),
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_valid=source_valid,
            state_valid=state_valid,
            candidate_valid=candidate_valid,
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "schema": ETD_CONFIG_SCHEMA,
            "state_schema": ETD_STATE_SCHEMA,
            "type": "ETDLinearLearner",
            "step_size": self._step_size,
            "trace_decay": self._trace_decay,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ETDLinearLearner:
        """Reconstruct from dict."""
        fields = _require_exact_manifest(
            config,
            {"schema", "state_schema", "type", "step_size", "trace_decay"},
            label="ETD config",
        )
        if fields.pop("schema") != ETD_CONFIG_SCHEMA:
            raise ValueError("unsupported ETD config schema")
        if fields.pop("state_schema") != ETD_STATE_SCHEMA:
            raise ValueError("unsupported ETD state schema")
        if fields.pop("type") != "ETDLinearLearner":
            raise ValueError("ETD config type is invalid")
        return cls(**fields)


class GradientTDLinearLearner:
    """Linear off-policy Gradient-TD/TDC learner with secondary weights.

    This implements the linear TDC/GTD(lambda)-style correction with an
    auxiliary weight vector, descending the projected Bellman-error objective in
    the standard linear setting:

    ``delta = r + gamma theta^T phi' - theta^T phi``
    ``e = rho * (phi + gamma * lambda * e)``
    ``theta += alpha * (delta * e - gamma * (1 - lambda) * (h^T e) * phi')``
    ``h += beta * (delta * e - (h^T phi) * phi)``

    The implementation is intentionally linear. Nonlinear shared-trunk GTD is a
    separate approximation problem; this class supplies the exact secondary
    weight correction missing from semi-gradient off-policy TD/Horde.
    """

    def __init__(
        self,
        step_size: float = 0.01,
        secondary_step_size: float = 0.05,
        trace_decay: float = 0.0,
        ratio_clip: float = 10.0,
    ):
        """Initialize the learner."""
        self._step_size = _require_real(
            step_size,
            label="step_size",
            minimum=_FLOAT32_MIN_POSITIVE,
        )
        self._secondary_step_size = _require_real(
            secondary_step_size,
            label="secondary_step_size",
            minimum=0.0,
        )
        self._trace_decay = _require_real(
            trace_decay,
            label="trace_decay",
            minimum=0.0,
            maximum=1.0,
        )
        self._ratio_clip = _require_real(
            ratio_clip,
            label="ratio_clip",
            minimum=_FLOAT32_MIN_POSITIVE,
        )

    @property
    def step_size(self) -> float:
        """Primary learning rate."""
        return self._step_size

    @property
    def secondary_step_size(self) -> float:
        """Secondary-weight learning rate."""
        return self._secondary_step_size

    @property
    def trace_decay(self) -> float:
        """Eligibility trace decay."""
        return self._trace_decay

    @property
    def ratio_clip(self) -> float:
        """Importance-ratio clip."""
        return self._ratio_clip

    def resource_budget(self, feature_dim: int) -> OffPolicyTDResourceBudget:
        """Return the exact persistent-state resource contract."""

        return _resource_budget_for("GradientTDLinearLearner", feature_dim)

    def init(self, feature_dim: int) -> GradientTDState:
        """Initialize primary weights, secondary weights, and traces."""
        feature_dim = _require_positive_feature_dim(feature_dim)
        augmented_dim = feature_dim + 1
        return GradientTDState(  # type: ignore[call-arg]
            weights=jnp.zeros(augmented_dim, dtype=jnp.float32),
            secondary_weights=jnp.zeros(augmented_dim, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(augmented_dim, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            birth_timestamp=jnp.asarray(time.time(), dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _validate_state_structure(self, state: GradientTDState) -> Bool[Array, ""]:
        """Validate array layout and return exact-clock authenticity."""

        if not isinstance(state, GradientTDState):
            raise TypeError("state must be a GradientTDState")
        augmented_dim = state.weights.shape[0] if state.weights.ndim == 1 else -1
        if augmented_dim < 2:
            raise ValueError("state weights must include at least one feature and bias")
        _require_float32_vector(state.weights, augmented_dim, label="state.weights")
        _require_float32_vector(
            state.secondary_weights,
            augmented_dim,
            label="state.secondary_weights",
        )
        _require_float32_vector(
            state.eligibility_traces,
            augmented_dim,
            label="state.eligibility_traces",
        )
        _metadata_valid(state.birth_timestamp)
        _metadata_valid(state.uptime_s)
        return _lifetime_counter_valid(state.step_words, state.step_count)

    def state_valid(self, state: GradientTDState) -> Bool[Array, ""]:
        """Return whether a state is finite and has an authentic exact clock."""

        return (
            self._validate_state_structure(state)
            & _learning_arrays_finite(
                state.weights,
                state.secondary_weights,
                state.eligibility_traces,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )

    @staticmethod
    def _augment(observation: Observation) -> Array:
        """Append the bias feature."""
        return jnp.concatenate(
            (
                jnp.asarray(observation, dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: GradientTDState, observation: Observation) -> Float[Array, " 1"]:
        """Compute ``theta^T phi`` with an appended bias feature."""
        self._validate_state_structure(state)
        _require_float32_vector(
            observation,
            state.weights.shape[0] - 1,
            label="observation",
        )
        return jnp.atleast_1d(jnp.dot(state.weights, self._augment(observation)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: GradientTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
        rho: Array,
    ) -> GradientTDUpdateResult:
        """Apply one off-policy Gradient-TD/TDC update."""
        lifetime_counter_valid = self._validate_state_structure(state)
        feature_dim = state.weights.shape[0] - 1
        _require_float32_vector(observation, feature_dim, label="observation")
        _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        beta = jnp.asarray(self._secondary_step_size, dtype=jnp.float32)
        lam = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        ratio_clip = jnp.asarray(self._ratio_clip, dtype=jnp.float32)
        gamma_s = _as_float32_scalar(gamma, label="gamma")
        reward_s = _as_float32_scalar(reward, label="reward")
        rho_s = _as_float32_scalar(rho, label="rho")
        rho_clipped = jnp.minimum(jnp.maximum(rho_s, 0.0), ratio_clip)
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(reward_s)
            & jnp.isfinite(gamma_s)
            & (gamma_s >= 0.0)
            & (gamma_s <= 1.0)
            & jnp.isfinite(rho_s)
            & (rho_s >= 0.0)
        )
        state_valid = (
            _learning_arrays_finite(
                state.weights,
                state.secondary_weights,
                state.eligibility_traces,
            )
            & _metadata_valid(state.birth_timestamp)
            & _metadata_valid(state.uptime_s)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        phi = self._augment(observation)
        next_phi = self._augment(next_observation)
        prediction = jnp.dot(state.weights, phi)
        next_prediction = jnp.dot(state.weights, next_phi)
        td_error = reward_s + gamma_s * next_prediction - prediction

        traces = rho_clipped * (phi + gamma_s * lam * state.eligibility_traces)
        secondary_dot_trace = jnp.dot(state.secondary_weights, traces)
        secondary_dot_phi = jnp.dot(state.secondary_weights, phi)

        primary_step = alpha * (
            td_error * traces - gamma_s * (1.0 - lam) * secondary_dot_trace * next_phi
        )
        secondary_step = beta * (td_error * traces - secondary_dot_phi * phi)

        new_state = GradientTDState(  # type: ignore[call-arg]
            weights=state.weights + primary_step,
            secondary_weights=state.secondary_weights + secondary_step,
            eligibility_traces=traces,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        candidate_valid = (
            _lifetime_counter_valid(new_state.step_words, new_state.step_count)
            & _learning_arrays_finite(
                new_state.weights,
                new_state.secondary_weights,
                new_state.eligibility_traces,
            )
            & _metadata_valid(new_state.birth_timestamp)
            & _metadata_valid(new_state.uptime_s)
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & state_valid
            & candidate_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )
        metrics = jnp.array(
            [
                td_error**2,
                td_error,
                rho_clipped,
                jnp.sqrt(jnp.mean(new_state.weights**2)),
                jnp.sqrt(jnp.mean(new_state.secondary_weights**2)),
                jnp.mean(jnp.abs(traces)),
            ],
            dtype=jnp.float32,
        )
        return GradientTDUpdateResult(  # type: ignore[call-arg]
            state=committed_state,
            prediction=jnp.atleast_1d(prediction),
            td_error=jnp.asarray(td_error),
            rho_clipped=jnp.asarray(rho_clipped),
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_valid=source_valid,
            state_valid=state_valid,
            candidate_valid=candidate_valid,
            update_applied=update_applied,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "schema": GRADIENT_TD_CONFIG_SCHEMA,
            "state_schema": GRADIENT_TD_STATE_SCHEMA,
            "type": "GradientTDLinearLearner",
            "step_size": self._step_size,
            "secondary_step_size": self._secondary_step_size,
            "trace_decay": self._trace_decay,
            "ratio_clip": self._ratio_clip,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GradientTDLinearLearner:
        """Reconstruct from dict."""
        fields = _require_exact_manifest(
            config,
            {
                "schema",
                "state_schema",
                "type",
                "step_size",
                "secondary_step_size",
                "trace_decay",
                "ratio_clip",
            },
            label="Gradient-TD config",
        )
        if fields.pop("schema") != GRADIENT_TD_CONFIG_SCHEMA:
            raise ValueError("unsupported Gradient-TD config schema")
        if fields.pop("state_schema") != GRADIENT_TD_STATE_SCHEMA:
            raise ValueError("unsupported Gradient-TD state schema")
        if fields.pop("type") != "GradientTDLinearLearner":
            raise ValueError("Gradient-TD config type is invalid")
        return cls(**fields)


def migrate_legacy_off_policy_td_config(
    legacy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate one exact unversioned off-policy TD config to v2."""

    fields = _require_exact_manifest(
        legacy_config,
        {"type", "step_size", "trace_decay", "retrace_clip"},
        label="legacy off-policy TD config",
    )
    if fields.pop("type") != "OffPolicyTDLinearLearner":
        raise ValueError("legacy off-policy TD config type is invalid")
    return OffPolicyTDLinearLearner(**fields).to_config()


def migrate_legacy_etd_config(legacy_config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate one exact unversioned ETD config to v2."""

    fields = _require_exact_manifest(
        legacy_config,
        {"type", "step_size", "trace_decay"},
        label="legacy ETD config",
    )
    if fields.pop("type") != "ETDLinearLearner":
        raise ValueError("legacy ETD config type is invalid")
    return ETDLinearLearner(**fields).to_config()


def migrate_legacy_gradient_td_config(
    legacy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate one exact unversioned Gradient-TD config to v2."""

    fields = _require_exact_manifest(
        legacy_config,
        {
            "type",
            "step_size",
            "secondary_step_size",
            "trace_decay",
            "ratio_clip",
        },
        label="legacy Gradient-TD config",
    )
    if fields.pop("type") != "GradientTDLinearLearner":
        raise ValueError("legacy Gradient-TD config type is invalid")
    return GradientTDLinearLearner(**fields).to_config()


def migrate_legacy_off_policy_td_state(legacy_state: Any) -> OffPolicyTDState:
    """Migrate an exact unsaturated pre-v2 off-policy TD state."""

    fields = _require_exact_manifest(
        _host_state_mapping(legacy_state, label="off-policy TD state"),
        {
            "weights",
            "bias",
            "eligibility_traces",
            "bias_eligibility_trace",
            "step_count",
            "birth_timestamp",
            "uptime_s",
        },
        label="legacy off-policy TD state",
    )
    fields["step_words"] = _legacy_step_words(
        fields["step_count"],
        label="off-policy TD",
    )
    fields["birth_timestamp"] = jnp.asarray(fields["birth_timestamp"], dtype=jnp.float32)
    fields["uptime_s"] = jnp.asarray(fields["uptime_s"], dtype=jnp.float32)
    migrated = OffPolicyTDState(**fields)
    learner = OffPolicyTDLinearLearner()
    if not bool(learner.state_valid(migrated)):
        raise ValueError("legacy off-policy TD state is not finite/authentic")
    return migrated


def migrate_legacy_etd_state(legacy_state: Any) -> ETDState:
    """Migrate an exact unsaturated pre-v2 ETD state."""

    fields = _require_exact_manifest(
        _host_state_mapping(legacy_state, label="ETD state"),
        {
            "weights",
            "bias",
            "eligibility_traces",
            "bias_eligibility_trace",
            "follow_on_trace",
            "emphasis",
            "step_count",
            "birth_timestamp",
            "uptime_s",
        },
        label="legacy ETD state",
    )
    fields["step_words"] = _legacy_step_words(fields["step_count"], label="ETD")
    fields["birth_timestamp"] = jnp.asarray(fields["birth_timestamp"], dtype=jnp.float32)
    fields["uptime_s"] = jnp.asarray(fields["uptime_s"], dtype=jnp.float32)
    migrated = ETDState(**fields)
    learner = ETDLinearLearner()
    if not bool(learner.state_valid(migrated)):
        raise ValueError("legacy ETD state is not finite/authentic")
    return migrated


def migrate_legacy_gradient_td_state(legacy_state: Any) -> GradientTDState:
    """Migrate an exact unsaturated pre-v2 Gradient-TD state."""

    fields = _require_exact_manifest(
        _host_state_mapping(legacy_state, label="Gradient-TD state"),
        {
            "weights",
            "secondary_weights",
            "eligibility_traces",
            "step_count",
            "birth_timestamp",
            "uptime_s",
        },
        label="legacy Gradient-TD state",
    )
    fields["step_words"] = _legacy_step_words(
        fields["step_count"],
        label="Gradient-TD",
    )
    fields["birth_timestamp"] = jnp.asarray(fields["birth_timestamp"], dtype=jnp.float32)
    fields["uptime_s"] = jnp.asarray(fields["uptime_s"], dtype=jnp.float32)
    migrated = GradientTDState(**fields)
    learner = GradientTDLinearLearner()
    if not bool(learner.state_valid(migrated)):
        raise ValueError("legacy Gradient-TD state is not finite/authentic")
    return migrated


def measure_off_policy_td_state_nbytes(state: OffPolicyTDState) -> int:
    """Measure exact persistent JAX learning-state bytes for semi-gradient TD."""

    OffPolicyTDLinearLearner()._validate_state_structure(state)
    return _measure_arrays(
        state.weights,
        state.bias,
        state.eligibility_traces,
        state.bias_eligibility_trace,
        state.step_count,
        state.step_words,
    )


def measure_etd_state_nbytes(state: ETDState) -> int:
    """Measure exact persistent JAX learning-state bytes for ETD."""

    ETDLinearLearner()._validate_state_structure(state)
    return _measure_arrays(
        state.weights,
        state.bias,
        state.eligibility_traces,
        state.bias_eligibility_trace,
        state.follow_on_trace,
        state.emphasis,
        state.step_count,
        state.step_words,
    )


def measure_gradient_td_state_nbytes(state: GradientTDState) -> int:
    """Measure exact persistent JAX learning-state bytes for Gradient-TD."""

    GradientTDLinearLearner()._validate_state_structure(state)
    return _measure_arrays(
        state.weights,
        state.secondary_weights,
        state.eligibility_traces,
        state.step_count,
        state.step_words,
    )


def run_gradient_td_learning_loop(
    learner: GradientTDLinearLearner,
    state: GradientTDState,
    observations: Array,
    rewards: Array,
    next_observations: Array,
    gammas: Array,
    rhos: Array,
) -> GradientTDArrayResult:
    """Run Gradient-TD/TDC over arrays using ``jax.lax.scan``."""

    learner._validate_state_structure(state)
    feature_dim = state.weights.shape[0] - 1
    if getattr(observations, "ndim", None) != 2:
        raise ValueError("observations must be a rank-two array")
    if observations.shape[1] != feature_dim:
        raise ValueError("observations feature dimension does not match state")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape")
    if observations.dtype != jnp.dtype(jnp.float32):
        raise TypeError("observations must have dtype float32")
    if next_observations.dtype != jnp.dtype(jnp.float32):
        raise TypeError("next_observations must have dtype float32")
    num_steps = observations.shape[0]
    for name, value in (("rewards", rewards), ("gammas", gammas), ("rhos", rhos)):
        if getattr(value, "shape", None) != (num_steps,):
            raise ValueError(f"{name} must have shape ({num_steps},)")
        if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError(f"{name} must have dtype float32")

    def step_fn(
        carry: GradientTDState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[GradientTDState, tuple[Array, Array, Array, Array, Array]]:
        obs, reward, next_obs, gamma, rho = inputs
        result = learner.update(carry, obs, reward, next_obs, gamma, rho)
        return (
            result.state,
            (
                result.prediction[0],
                result.td_error,
                result.rho_clipped,
                result.metrics,
                result.update_applied,
            ),
        )

    t0 = time.time()
    final_state, (predictions, td_errors, rho_clipped, metrics, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (observations, rewards, next_observations, gammas, rhos),
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )
    return GradientTDArrayResult(  # type: ignore[call-arg]
        state=final_state,
        predictions=predictions,
        td_errors=td_errors,
        rho_clipped=rho_clipped,
        metrics=metrics,
        updates_applied=updates_applied,
    )
