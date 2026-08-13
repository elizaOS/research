"""Step 2 feature-discovery streams.

These streams are designed for the Alberta Plan's Step 2 setting:
continual supervised learning with vector-valued targets, nonlinear latent
features, and changing feature relevance.  The latent features are known to
the stream but hidden from learners.

Scope caveat: both oracles here (tanh units, pairwise products) lie *inside*
the hypothesis class of the corresponding Step 2 learners, so "discovery"
reduces to selecting and replacing the right features from a representable
pool under a fixed budget.  Feature *construction* proper — targets whose
minimal representation lies outside a one-layer feature bank — is probed by
:mod:`alberta_framework.streams.out_of_class`.
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.types import TimeStep

NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA = (
    "alberta.nonlinear-feature-discovery-stream.config.v2"
)
NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA = (
    "alberta.nonlinear-feature-discovery-stream.state.v2"
)
INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA = (
    "alberta.interaction-feature-discovery-stream.config.v2"
)
INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA = (
    "alberta.interaction-feature-discovery-stream.state.v2"
)
FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA = (
    "alberta.feature-discovery-stream.resource-budget.v2"
)
FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES = 12
FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1

__all__ = [
    "FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES",
    "FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES",
    "FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA",
    "INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA",
    "INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA",
    "NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA",
    "NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA",
    "FeatureDiscoveryStreamResourceBudget",
    "InteractionFeatureDiscoveryState",
    "InteractionFeatureDiscoveryStepResult",
    "InteractionFeatureDiscoveryStream",
    "NonlinearFeatureDiscoveryState",
    "NonlinearFeatureDiscoveryStepResult",
    "NonlinearFeatureDiscoveryStream",
    "collect_feature_discovery_stream",
    "feature_discovery_stream_clock_nbytes",
    "measure_feature_discovery_stream_state_nbytes",
    "migrate_legacy_interaction_feature_discovery_state",
    "migrate_legacy_nonlinear_feature_discovery_state",
]


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    """Require an exact public array contract without silent narrowing."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _require_prng_key(key: Array, *, name: str) -> None:
    """Require a scalar typed or legacy JAX PRNG key."""

    key_data = jr.key_data(key)
    if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


def _tree_floating_arrays_finite(value: Any) -> Bool[Array, ""]:
    """Return whether every persistent floating or complex leaf is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _divmod_lifetime_words(words: Array, divisor: int) -> tuple[Array, Array]:
    """Exact uint64-by-positive-int32 division without enabling JAX x64."""

    _require_array(
        words,
        name="feature-discovery step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    if type(divisor) is not int or not 0 < divisor <= _INT32_MAX:
        raise ValueError("schedule divisor must be a positive int32 integer")
    divisor_array = jnp.asarray(divisor, dtype=jnp.uint32)

    def body(index: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        remainder, quotient_high, quotient_low = carry
        in_high = index < 32
        bit_index = jnp.asarray(31, dtype=jnp.int32) - jnp.mod(index, 32)
        source = jnp.where(in_high, words[0], words[1])
        bit = jnp.bitwise_and(
            jnp.right_shift(source, bit_index.astype(jnp.uint32)),
            jnp.asarray(1, dtype=jnp.uint32),
        )
        doubled = remainder + remainder + bit
        subtract = doubled >= divisor_array
        next_remainder = jnp.where(subtract, doubled - divisor_array, doubled)
        mask = jnp.left_shift(
            jnp.asarray(1, dtype=jnp.uint32),
            bit_index.astype(jnp.uint32),
        )
        next_high = jnp.where(
            in_high & subtract,
            jnp.bitwise_or(quotient_high, mask),
            quotient_high,
        )
        next_low = jnp.where(
            (~in_high) & subtract,
            jnp.bitwise_or(quotient_low, mask),
            quotient_low,
        )
        return next_remainder, next_high, next_low

    zero = jnp.asarray(0, dtype=jnp.uint32)
    remainder, high, low = jax.lax.fori_loop(0, 64, body, (zero, zero, zero))
    return jnp.stack((high, low)).astype(jnp.uint32), remainder


def _context_index_from_words(
    words: Array,
    *,
    context_length: int,
    n_contexts: int,
) -> Int[Array, ""]:
    """Compute ``(event // context_length) % n_contexts`` exactly."""

    context_quotient, _offset = _divmod_lifetime_words(words, context_length)
    _cycles, context_index = _divmod_lifetime_words(context_quotient, n_contexts)
    return context_index.astype(jnp.int32)


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    fields = dict(payload)
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(f"{label} fields are invalid; missing={missing}, extra={extra}")
    return fields


def _host_state_fields(state: Any, *, label: str) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    raise TypeError(f"legacy {label} state must be a mapping or dataclass")


def _legacy_unsaturated_count(fields: Mapping[str, Any], *, label: str) -> int:
    count_array = jnp.asarray(fields["step_count"])
    if count_array.shape != () or count_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {label} step_count must be scalar int32")
    count = int(count_array)
    if count < 0:
        raise ValueError(f"negative legacy {label} step_count indicates wrap")
    if count >= _INT32_MAX:
        raise ValueError(f"saturated legacy {label} step_count is ambiguous")
    return count


def _require_positive_int32(value: Any, *, name: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {_INT32_MAX}]")


def _require_finite_real(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )
    if nonnegative:
        valid = valid and float(value) >= 0.0
    if not valid:
        suffix = " non-negative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{suffix} real")


def _step_input_valid(idx: Array) -> Bool[Array, ""]:
    """Validate the legacy scan index shape and dynamically reject NaN/Inf."""

    idx_array = jnp.asarray(idx)
    if idx_array.shape != ():
        raise ValueError(f"idx must be scalar, got shape {idx_array.shape}")
    if not (
        jnp.issubdtype(idx_array.dtype, jnp.integer)
        or jnp.issubdtype(idx_array.dtype, jnp.floating)
    ):
        raise TypeError("idx must have an integer or floating dtype")
    return jnp.isfinite(idx_array)


@chex.dataclass(frozen=True)
class NonlinearFeatureDiscoveryState:
    """State for ``NonlinearFeatureDiscoveryStream``.

    Attributes:
        key: PRNG key for sample generation.
        latent_weights: Hidden feature weights, shape ``(n_latents, feature_dim)``.
        latent_biases: Hidden feature biases, shape ``(n_latents,)``.
        context_weights: Per-context task weights over latent features,
            shape ``(n_contexts, n_tasks, n_latents)``.
        linear_weights: Small direct linear component, shape
            ``(n_tasks, feature_dim)``.
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian uint32 pair identifying generated events.
    """

    key: PRNGKeyArray
    latent_weights: Float[Array, "n_latents feature_dim"]
    latent_biases: Float[Array, " n_latents"]
    context_weights: Float[Array, "n_contexts n_tasks n_latents"]
    linear_weights: Float[Array, "n_tasks feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class InteractionFeatureDiscoveryState:
    """State for ``InteractionFeatureDiscoveryStream``.

    The hidden oracle features are pairwise products ``x_i * x_j``.  The pair
    list is fixed, while context weights determine which products are useful.
    """

    key: PRNGKeyArray
    pair_left: Int[Array, " n_pairs"]
    pair_right: Int[Array, " n_pairs"]
    context_weights: Float[Array, "n_contexts n_tasks n_pairs"]
    linear_weights: Float[Array, "n_tasks feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class NonlinearFeatureDiscoveryStepResult:
    """One staged nonlinear stream event and its atomic commit diagnostics."""

    timestep: TimeStep
    state: NonlinearFeatureDiscoveryState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    context_index: Int[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class InteractionFeatureDiscoveryStepResult:
    """One staged interaction stream event and its atomic commit diagnostics."""

    timestep: TimeStep
    state: InteractionFeatureDiscoveryState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    context_index: Int[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@dataclass(frozen=True)
class FeatureDiscoveryStreamResourceBudget:
    """Exact persistent-state accounting for one feature-discovery stream."""

    stream_type: str
    state_nbytes: int
    exact_clock_nbytes: int = FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES
    exact_clock_delta_nbytes: int = FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES
    trainable_scalars: int = 0
    replay_capacity: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema": FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA,
            "stream_type": self.stream_type,
            "state_nbytes": self.state_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FeatureDiscoveryStreamResourceBudget":
        expected = {
            "schema",
            "stream_type",
            "state_nbytes",
            "exact_clock_nbytes",
            "exact_clock_delta_nbytes",
            "trainable_scalars",
            "replay_capacity",
        }
        fields = _require_exact_fields(
            payload,
            expected,
            label="feature-discovery resource budget",
        )
        if fields.pop("schema") != FEATURE_DISCOVERY_STREAM_RESOURCE_SCHEMA:
            raise ValueError("feature-discovery resource schema is unsupported")
        if fields["stream_type"] not in {
            "NonlinearFeatureDiscoveryStream",
            "InteractionFeatureDiscoveryStream",
        }:
            raise ValueError("feature-discovery resource stream type is unsupported")
        for name in expected - {"schema", "stream_type"}:
            if type(fields[name]) is not int or fields[name] < 0:
                raise ValueError(f"feature-discovery resource {name} must be non-negative")
        if fields["exact_clock_nbytes"] != FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES:
            raise ValueError("feature-discovery exact clock accounting is invalid")
        if (
            fields["exact_clock_delta_nbytes"]
            != FEATURE_DISCOVERY_STREAM_CLOCK_DELTA_NBYTES
        ):
            raise ValueError("feature-discovery exact clock delta is invalid")
        if fields["trainable_scalars"] != 0 or fields["replay_capacity"] != 0:
            raise ValueError("feature-discovery streams must not own learner or replay state")
        return cls(**fields)


class NonlinearFeatureDiscoveryStream:
    """Non-stationary multitask stream with hidden nonlinear features.

    Observations are raw vectors ``x_t``.  Targets are vector-valued:

    ``y*_t = W_c phi(x_t) + L x_t + noise``

    where ``phi`` is a fixed bank of hidden nonlinear features and ``W_c``
    changes by context.  This creates a controlled Step 2 benchmark: useful
    nonlinear features exist, relevance shifts over time, and the learner has
    only a limited budget of representable features.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int = 4,
        n_latents: int = 32,
        n_contexts: int = 8,
        context_length: int = 500,
        active_latents_per_context: int = 6,
        feature_std: float = 1.0,
        latent_scale: float = 1.0,
        linear_scale: float = 0.05,
        noise_std: float = 0.01,
    ):
        """Initialize the nonlinear feature-discovery stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_latents: Number of hidden oracle nonlinear features.
            n_contexts: Number of recurring relevance contexts.
            context_length: Number of steps before switching context.
            active_latents_per_context: Expected number of useful latent
                features per task/context.
            feature_std: Standard deviation of raw observations.
            latent_scale: Scale of oracle latent weights.
            linear_scale: Scale of the direct linear target component.
            noise_std: Standard deviation of target noise.
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(n_tasks, name="n_tasks")
        _require_positive_int32(n_latents, name="n_latents")
        _require_positive_int32(n_contexts, name="n_contexts")
        _require_positive_int32(context_length, name="context_length")
        _require_positive_int32(
            active_latents_per_context,
            name="active_latents_per_context",
        )
        _require_finite_real(
            feature_std,
            name="feature_std",
            nonnegative=True,
        )
        _require_finite_real(latent_scale, name="latent_scale")
        _require_finite_real(linear_scale, name="linear_scale")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)

        self._feature_dim = feature_dim
        self._n_tasks = n_tasks
        self._n_latents = n_latents
        self._n_contexts = n_contexts
        self._context_length = context_length
        self._active_latents_per_context = active_latents_per_context
        self._feature_std = feature_std
        self._latent_scale = latent_scale
        self._linear_scale = linear_scale
        self._noise_std = noise_std

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_latents(self) -> int:
        """Return the number of hidden oracle features."""
        return self._n_latents

    @property
    def n_contexts(self) -> int:
        """Return the number of recurring contexts."""

        return self._n_contexts

    @property
    def context_length(self) -> int:
        """Return the exact number of events per context."""

        return self._context_length

    @property
    def resource_budget(self) -> FeatureDiscoveryStreamResourceBudget:
        """Return exact persistent-state accounting for this stream."""

        return FeatureDiscoveryStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_feature_discovery_stream_state_nbytes(
                self.init(jr.key(0))
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete nonlinear stream under strict v2 schemas."""

        return {
            "type": type(self).__name__,
            "config_schema": NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA,
            "state_schema": NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "n_tasks": self._n_tasks,
            "n_latents": self._n_latents,
            "n_contexts": self._n_contexts,
            "context_length": self._context_length,
            "active_latents_per_context": self._active_latents_per_context,
            "feature_std": float(self._feature_std),
            "latent_scale": float(self._latent_scale),
            "linear_scale": float(self._linear_scale),
            "noise_std": float(self._noise_std),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "NonlinearFeatureDiscoveryStream":
        """Strictly reconstruct one versioned nonlinear stream."""

        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "n_tasks",
            "n_latents",
            "n_contexts",
            "context_length",
            "active_latents_per_context",
            "feature_std",
            "latent_scale",
            "linear_scale",
            "noise_std",
        }
        fields = _require_exact_fields(
            config,
            expected,
            label="nonlinear feature-discovery config",
        )
        if fields.pop("type") != cls.__name__:
            raise ValueError("nonlinear feature-discovery config type is unsupported")
        if fields.pop("config_schema") != NONLINEAR_FEATURE_DISCOVERY_CONFIG_SCHEMA:
            raise ValueError("nonlinear feature-discovery config schema is unsupported")
        if fields.pop("state_schema") != NONLINEAR_FEATURE_DISCOVERY_STATE_SCHEMA:
            raise ValueError("nonlinear feature-discovery state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: NonlinearFeatureDiscoveryState) -> None:
        """Require every fixed-shape v2 nonlinear stream leaf."""

        _require_prng_key(state.key, name="nonlinear feature-discovery key")
        _require_array(
            state.latent_weights,
            name="nonlinear latent_weights",
            shape=(self._n_latents, self._feature_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.latent_biases,
            name="nonlinear latent_biases",
            shape=(self._n_latents,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.context_weights,
            name="nonlinear context_weights",
            shape=(self._n_contexts, self._n_tasks, self._n_latents),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.linear_weights,
            name="nonlinear linear_weights",
            shape=(self._n_tasks, self._feature_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="nonlinear step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="nonlinear step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: NonlinearFeatureDiscoveryState) -> Bool[Array, ""]:
        """Authenticate exact time and every persistent floating value."""

        self._require_state_contract(state)
        return _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & _tree_floating_arrays_finite(state)

    def init(self, key: Array) -> NonlinearFeatureDiscoveryState:
        """Initialize stream state."""
        key, k_latent, k_bias, k_ctx, k_mask, k_linear = jr.split(key, 6)

        # LeCun-style 1/sqrt(feature_dim) scaling gives each latent a roughly
        # unit-variance preactivation (at latent_scale = feature_std = 1). The
        # bias std of 0.25 is small against that, spreading the tanh operating
        # points across latents without pushing units into saturation.
        latent_weights = (
            self._latent_scale
            * jr.normal(k_latent, (self._n_latents, self._feature_dim), dtype=jnp.float32)
            / jnp.sqrt(float(self._feature_dim))
        )
        latent_biases = 0.25 * jr.normal(k_bias, (self._n_latents,), dtype=jnp.float32)

        dense_context_weights = jr.normal(
            k_ctx,
            (self._n_contexts, self._n_tasks, self._n_latents),
            dtype=jnp.float32,
        )
        keep_prob = min(1.0, self._active_latents_per_context / self._n_latents)
        mask = jr.bernoulli(
            k_mask,
            keep_prob,
            (self._n_contexts, self._n_tasks, self._n_latents),
        )
        # Dividing each head's weights by sqrt(#active latents) keeps target
        # variance roughly constant across contexts: a context switch changes
        # WHICH features matter, not the scale of the regression problem.
        context_weights = dense_context_weights * mask.astype(jnp.float32)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0))
        context_weights = context_weights / norm

        linear_weights = self._linear_scale * jr.normal(
            k_linear, (self._n_tasks, self._feature_dim), dtype=jnp.float32
        )

        return NonlinearFeatureDiscoveryState(
            key=key,
            latent_weights=latent_weights,
            latent_biases=latent_biases,
            context_weights=context_weights,
            linear_weights=linear_weights,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self,
        state: NonlinearFeatureDiscoveryState,
        idx: Array,
    ) -> tuple[TimeStep, NonlinearFeatureDiscoveryState]:
        """Generate one multitask supervised sample."""
        result = self.step_result(state, idx)
        return result.timestep, result.state

    def step_result(
        self,
        state: NonlinearFeatureDiscoveryState,
        idx: Array,
    ) -> NonlinearFeatureDiscoveryStepResult:
        """Stage and atomically commit one exact nonlinear stream event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        words = state.step_words
        lifetime_counter_valid = _lifetime_counter_valid(words, state.step_count)
        state_valid = _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(words)
        )
        context_idx = _context_index_from_words(
            words,
            context_length=self._context_length,
            n_contexts=self._n_contexts,
        )
        key, k_x, k_noise = jr.split(state.key, 3)

        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        latents = jnp.tanh(state.latent_weights @ x + state.latent_biases)

        task_weights = state.context_weights[context_idx]
        target = task_weights @ latents + state.linear_weights @ x
        noise = self._noise_std * jr.normal(k_noise, (self._n_tasks,), dtype=jnp.float32)
        target = target + noise

        candidate_state = NonlinearFeatureDiscoveryState(
            key=key,
            latent_weights=state.latent_weights,
            latent_biases=state.latent_biases,
            context_weights=state.context_weights,
            linear_weights=state.linear_weights,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target))
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & input_valid
            & state_valid
            & output_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                x,
                jnp.full_like(x, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                target,
                jnp.full_like(target, jnp.nan),
            ),
        )
        return NonlinearFeatureDiscoveryStepResult(
            timestep=timestep,
            state=new_state,
            pre_step_words=words,
            post_step_words=cast(Array, new_state.step_words),
            context_index=context_idx,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def collect_feature_discovery_stream(
    stream: Any,
    num_steps: int,
    key: Array,
) -> tuple[Array, Array]:
    """Collect a fixed array view of a feature-discovery stream.

    This helper is for controlled experiments where multiple learners should
    see the exact same stream.  It still uses the one-step stream interface and
    ``jax.lax.scan``; it does not imply experience replay inside a learner.
    """
    import jax

    state = stream.init(key)

    def step_fn(
        carry: Any,
        idx: Array,
    ) -> tuple[Any, tuple[Array, Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, state, jnp.arange(num_steps)
    )
    return observations, targets


class InteractionFeatureDiscoveryStream:
    """Non-stationary stream whose useful features are pairwise products.

    This benchmark gives Step 2 a sharper target than generic MLP learning.
    The useful nonlinear features are literal combinations of existing raw
    features:

    ``phi_ij(x_t) = x_t[i] * x_t[j]``

    The learner observes only ``x_t`` and vector target ``y*_t``.  Contexts
    change which products matter, so a bounded learner must rank and replace
    features rather than merely grow capacity.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int = 4,
        n_contexts: int = 8,
        context_length: int = 500,
        active_pairs_per_context: int = 6,
        feature_std: float = 1.0,
        linear_scale: float = 0.01,
        noise_std: float = 0.01,
        include_squares: bool = False,
    ):
        """Initialize the interaction stream.

        Args:
            feature_dim: Raw observation dimension.
            n_tasks: Number of supervised output heads.
            n_contexts: Number of recurring relevance contexts.
            context_length: Steps before switching context.
            active_pairs_per_context: Expected active pair-products per
                task/context.
            feature_std: Standard deviation of raw observations.
            linear_scale: Scale of the small direct linear component.
            noise_std: Standard deviation of target noise.
            include_squares: Whether to include ``x_i * x_i`` oracle features.
        """
        _require_positive_int32(feature_dim, name="feature_dim", minimum=2)
        _require_positive_int32(n_tasks, name="n_tasks")
        _require_positive_int32(n_contexts, name="n_contexts")
        _require_positive_int32(context_length, name="context_length")
        _require_positive_int32(
            active_pairs_per_context,
            name="active_pairs_per_context",
        )
        _require_finite_real(
            feature_std,
            name="feature_std",
            nonnegative=True,
        )
        _require_finite_real(linear_scale, name="linear_scale")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        if type(include_squares) is not bool:
            raise ValueError("include_squares must be an exact bool")

        self._feature_dim = feature_dim
        self._n_tasks = n_tasks
        self._n_contexts = n_contexts
        self._context_length = context_length
        self._active_pairs_per_context = active_pairs_per_context
        self._feature_std = feature_std
        self._linear_scale = linear_scale
        self._noise_std = noise_std
        self._include_squares = include_squares

    @property
    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the number of supervised tasks."""
        return self._n_tasks

    @property
    def n_contexts(self) -> int:
        """Return the number of recurring contexts."""

        return self._n_contexts

    @property
    def context_length(self) -> int:
        """Return the exact number of events per context."""

        return self._context_length

    @property
    def resource_budget(self) -> FeatureDiscoveryStreamResourceBudget:
        """Return exact persistent-state accounting for this stream."""

        return FeatureDiscoveryStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_feature_discovery_stream_state_nbytes(
                self.init(jr.key(0))
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete interaction stream under strict v2 schemas."""

        return {
            "type": type(self).__name__,
            "config_schema": INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA,
            "state_schema": INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "n_tasks": self._n_tasks,
            "n_contexts": self._n_contexts,
            "context_length": self._context_length,
            "active_pairs_per_context": self._active_pairs_per_context,
            "feature_std": float(self._feature_std),
            "linear_scale": float(self._linear_scale),
            "noise_std": float(self._noise_std),
            "include_squares": self._include_squares,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "InteractionFeatureDiscoveryStream":
        """Strictly reconstruct one versioned interaction stream."""

        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "n_tasks",
            "n_contexts",
            "context_length",
            "active_pairs_per_context",
            "feature_std",
            "linear_scale",
            "noise_std",
            "include_squares",
        }
        fields = _require_exact_fields(
            config,
            expected,
            label="interaction feature-discovery config",
        )
        if fields.pop("type") != cls.__name__:
            raise ValueError("interaction feature-discovery config type is unsupported")
        if fields.pop("config_schema") != INTERACTION_FEATURE_DISCOVERY_CONFIG_SCHEMA:
            raise ValueError("interaction feature-discovery config schema is unsupported")
        if fields.pop("state_schema") != INTERACTION_FEATURE_DISCOVERY_STATE_SCHEMA:
            raise ValueError("interaction feature-discovery state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: InteractionFeatureDiscoveryState) -> None:
        """Require every fixed-shape v2 interaction stream leaf."""

        pair_left, pair_right = self._pairs()
        n_pairs = pair_left.shape[0]
        _require_prng_key(state.key, name="interaction feature-discovery key")
        _require_array(
            state.pair_left,
            name="interaction pair_left",
            shape=(n_pairs,),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.pair_right,
            name="interaction pair_right",
            shape=(n_pairs,),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.context_weights,
            name="interaction context_weights",
            shape=(self._n_contexts, self._n_tasks, n_pairs),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.linear_weights,
            name="interaction linear_weights",
            shape=(self._n_tasks, self._feature_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="interaction step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="interaction step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: InteractionFeatureDiscoveryState) -> Bool[Array, ""]:
        """Authenticate exact time, pair identity, and finite oracle arrays."""

        self._require_state_contract(state)
        expected_left, expected_right = self._pairs()
        pair_identity_valid = jnp.all(state.pair_left == expected_left) & jnp.all(
            state.pair_right == expected_right
        )
        return (
            _lifetime_counter_valid(
                state.step_words,
                state.step_count,
            )
            & pair_identity_valid
            & _tree_floating_arrays_finite(state)
        )

    def _pairs(self) -> tuple[Array, Array]:
        pairs = []
        for i in range(self._feature_dim):
            start = i if self._include_squares else i + 1
            for j in range(start, self._feature_dim):
                pairs.append((i, j))
        arr = jnp.array(pairs, dtype=jnp.int32)
        return arr[:, 0], arr[:, 1]

    def init(self, key: Array) -> InteractionFeatureDiscoveryState:
        """Initialize stream state."""
        key, k_ctx, k_mask, k_linear = jr.split(key, 4)
        pair_left, pair_right = self._pairs()
        n_pairs = pair_left.shape[0]

        dense_context_weights = jr.normal(
            k_ctx,
            (self._n_contexts, self._n_tasks, n_pairs),
            dtype=jnp.float32,
        )
        active_count = min(self._active_pairs_per_context, n_pairs)
        mask_scores = jr.uniform(
            k_mask,
            (self._n_contexts, self._n_tasks, n_pairs),
            dtype=jnp.float32,
        )
        threshold = jnp.sort(mask_scores, axis=-1)[..., active_count - 1 : active_count]
        mask = mask_scores <= threshold
        # Same sqrt(#active) normalization as the nonlinear stream: context
        # switches redirect relevance without changing the target scale.
        context_weights = dense_context_weights * mask.astype(jnp.float32)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0))
        context_weights = context_weights / norm

        linear_weights = self._linear_scale * jr.normal(
            k_linear, (self._n_tasks, self._feature_dim), dtype=jnp.float32
        )

        return InteractionFeatureDiscoveryState(
            key=key,
            pair_left=pair_left,
            pair_right=pair_right,
            context_weights=context_weights,
            linear_weights=linear_weights,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self,
        state: InteractionFeatureDiscoveryState,
        idx: Array,
    ) -> tuple[TimeStep, InteractionFeatureDiscoveryState]:
        """Generate one multitask interaction sample."""
        result = self.step_result(state, idx)
        return result.timestep, result.state

    def step_result(
        self,
        state: InteractionFeatureDiscoveryState,
        idx: Array,
    ) -> InteractionFeatureDiscoveryStepResult:
        """Stage and atomically commit one exact interaction stream event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        words = state.step_words
        lifetime_counter_valid = _lifetime_counter_valid(words, state.step_count)
        expected_left, expected_right = self._pairs()
        pair_identity_valid = jnp.all(state.pair_left == expected_left) & jnp.all(
            state.pair_right == expected_right
        )
        state_valid = pair_identity_valid & _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(words)
        )
        context_idx = _context_index_from_words(
            words,
            context_length=self._context_length,
            n_contexts=self._n_contexts,
        )
        key, k_x, k_noise = jr.split(state.key, 3)
        x = self._feature_std * jr.normal(
            k_x, (self._feature_dim,), dtype=jnp.float32
        )
        interactions = x[state.pair_left] * x[state.pair_right]
        task_weights = state.context_weights[context_idx]
        target = task_weights @ interactions + state.linear_weights @ x
        noise = self._noise_std * jr.normal(k_noise, (self._n_tasks,), dtype=jnp.float32)
        target = target + noise

        candidate_state = InteractionFeatureDiscoveryState(
            key=key,
            pair_left=state.pair_left,
            pair_right=state.pair_right,
            context_weights=state.context_weights,
            linear_weights=state.linear_weights,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target))
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & input_valid
            & state_valid
            & output_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                x,
                jnp.full_like(x, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                target,
                jnp.full_like(target, jnp.nan),
            ),
        )
        return InteractionFeatureDiscoveryStepResult(
            timestep=timestep,
            state=new_state,
            pre_step_words=words,
            post_step_words=cast(Array, new_state.step_words),
            context_index=context_idx,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def feature_discovery_stream_clock_nbytes() -> int:
    """Return bytes owned by saturating telemetry plus exact identity."""

    return FEATURE_DISCOVERY_STREAM_CLOCK_NBYTES


def measure_feature_discovery_stream_state_nbytes(
    state: NonlinearFeatureDiscoveryState | InteractionFeatureDiscoveryState,
) -> int:
    """Measure every persistent JAX-array byte in one concrete stream state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_nonlinear_feature_discovery_state(
    legacy_state: Any,
    *,
    stream: NonlinearFeatureDiscoveryStream,
) -> NonlinearFeatureDiscoveryState:
    """Migrate an exact unsaturated nonlinear stream state to v2."""

    fields = _host_state_fields(legacy_state, label="nonlinear feature-discovery")
    expected = {
        "key",
        "latent_weights",
        "latent_biases",
        "context_weights",
        "linear_weights",
        "step_count",
    }
    fields = _require_exact_fields(
        fields,
        expected,
        label="legacy nonlinear feature-discovery state",
    )
    count = _legacy_unsaturated_count(
        fields,
        label="nonlinear feature-discovery",
    )
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = NonlinearFeatureDiscoveryState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy nonlinear feature-discovery state is invalid")
    return migrated


def migrate_legacy_interaction_feature_discovery_state(
    legacy_state: Any,
    *,
    stream: InteractionFeatureDiscoveryStream,
) -> InteractionFeatureDiscoveryState:
    """Migrate an exact unsaturated interaction stream state to v2."""

    fields = _host_state_fields(legacy_state, label="interaction feature-discovery")
    expected = {
        "key",
        "pair_left",
        "pair_right",
        "context_weights",
        "linear_weights",
        "step_count",
    }
    fields = _require_exact_fields(
        fields,
        expected,
        label="legacy interaction feature-discovery state",
    )
    count = _legacy_unsaturated_count(
        fields,
        label="interaction feature-discovery",
    )
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = InteractionFeatureDiscoveryState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy interaction feature-discovery state is invalid")
    return migrated
