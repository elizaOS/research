# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded layer-level spectral regularization for continual learning.

This module implements the dense-layer objective from Lewandowski et al.,
*Learning Continually by Spectral Regularization* (ICLR 2025), Section 4:

``R(W, b) = (sigma_max(W)**k - 1)**2 + ||b||_2**(2*k)``.

The paper uses one power iteration in its experiments.  Here the power probe is
explicit state: a typed Threefry key owns its initialization, the right singular
probe is checkpointable, and an exact two-word lifetime prevents silent wrap.
The final singular-value estimate differentiates only through ``W``; the power
vectors are stopped, matching the standard power-iteration estimator rather
than backpropagating through its history.

This first mechanism surface is deliberately narrow.  It accepts one float32
dense matrix shaped ``(output_dim, input_dim)`` and its float32 bias.  Callers
must reshape convolution kernels as described in the paper and must handle
normalization parameters separately.  It is not wired into an agent or an
optimizer, and no efficacy, default-selection, or evidence claim follows.

Primary source:
    Lewandowski, A. et al. (2025). Learning Continually by Spectral
    Regularization. ICLR 2025. https://openreview.net/forum?id=Hcb2cgPbMg
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

SPECTRAL_REGULARIZATION_CONFIG_SCHEMA = "alberta.spectral-regularization.config.v1"
SPECTRAL_REGULARIZATION_CHECKPOINT_SCHEMA = (
    "alberta.spectral-regularization.checkpoint.v1"
)
SPECTRAL_REGULARIZATION_MECHANISM_STATUS = "l0-development-only-not-assessed"
SPECTRAL_REGULARIZATION_EVIDENCE_LEVEL = "L0"
SPECTRAL_REGULARIZATION_SCIENTIFIC_PROMOTION_ALLOWED = False
SPECTRAL_REGULARIZATION_DEFAULT_AGENT_INTEGRATION = False
SPECTRAL_REGULARIZATION_SOURCE_PROFILE = (
    "lewandowski-2025-section-4-dense-one-power-probe"
)

_UINT32_MAX = 4_294_967_295
_UINT64_MAX = 18_446_744_073_709_551_615
_MAX_DIMENSION = 16_384
_MAX_PARAMETERS = 16_777_216
_MAX_POWER_ITERATIONS = 64
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _strict_positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return value


def _strict_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    narrowed = float(np.float32(value))
    if not math.isfinite(float(value)) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite in float32")
    if abs(narrowed) > _FLOAT32_MAX or (narrowed != 0.0 and abs(narrowed) < _FLOAT32_TINY):
        raise ValueError(f"{name} must be zero or a normal finite float32 value")
    if narrowed < minimum or (positive and narrowed == 0.0):
        requirement = "positive" if positive else f">= {minimum}"
        raise ValueError(f"{name} must be {requirement}")
    return narrowed


def _typed_threefry_key(key: object) -> bool:
    if not (
        hasattr(key, "shape")
        and tuple(cast(Any, key).shape) == ()
        and jax.dtypes.issubdtype(cast(Any, key).dtype, jax.dtypes.prng_key)
    ):
        return False
    try:
        return str(jr.key_impl(cast(Any, key))) == "threefry2x32"
    except (TypeError, ValueError):
        return False


def _words_from_int(value: int) -> Array:
    return jnp.asarray(
        [(value >> 32) & _UINT32_MAX, value & _UINT32_MAX], dtype=jnp.uint32
    )


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = words[0] + carry
    available = ~jnp.all(words == jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _require_float32_array(value: object, *, name: str, shape: tuple[int, ...]) -> None:
    if not hasattr(value, "shape") or tuple(cast(Any, value).shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if jnp.dtype(cast(Any, value).dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32")


@dataclasses.dataclass(frozen=True)
class SpectralRegularizationConfig:
    """Static dense-layer shape, paper objective, and lifetime bounds."""

    output_dim: int
    input_dim: int
    coefficient: float = 1.0e-3
    exponent: int = 2
    power_iterations: int = 1
    normalization_epsilon: float = 1.0e-8
    maximum_updates: int = _UINT64_MAX

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_dim",
            _strict_positive_int(self.output_dim, name="output_dim", maximum=_MAX_DIMENSION),
        )
        object.__setattr__(
            self,
            "input_dim",
            _strict_positive_int(self.input_dim, name="input_dim", maximum=_MAX_DIMENSION),
        )
        if self.output_dim * self.input_dim > _MAX_PARAMETERS:
            raise ValueError("dense weight exceeds the bounded parameter budget")
        object.__setattr__(
            self,
            "coefficient",
            _strict_float32(self.coefficient, name="coefficient", minimum=0.0),
        )
        if self.exponent not in (1, 2, 4, 8):
            raise ValueError("exponent must be one of the paper-ablation values 1, 2, 4, or 8")
        object.__setattr__(
            self,
            "power_iterations",
            _strict_positive_int(
                self.power_iterations,
                name="power_iterations",
                maximum=_MAX_POWER_ITERATIONS,
            ),
        )
        object.__setattr__(
            self,
            "normalization_epsilon",
            _strict_float32(
                self.normalization_epsilon,
                name="normalization_epsilon",
                minimum=_FLOAT32_TINY,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_updates",
            _strict_positive_int(
                self.maximum_updates, name="maximum_updates", maximum=_UINT64_MAX
            ),
        )

    def to_config(self) -> dict[str, object]:
        """Return a strict JSON-compatible configuration."""

        return {
            "type": "SpectralRegularizer",
            "schema": SPECTRAL_REGULARIZATION_CONFIG_SCHEMA,
            "source_profile": SPECTRAL_REGULARIZATION_SOURCE_PROFILE,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "power_iterations": self.power_iterations,
            "normalization_epsilon": self.normalization_epsilon,
            "maximum_updates": self.maximum_updates,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> SpectralRegularizationConfig:
        """Load only the canonical v1 configuration schema."""

        expected = {
            "type",
            "schema",
            "source_profile",
            "output_dim",
            "input_dim",
            "coefficient",
            "exponent",
            "power_iterations",
            "normalization_epsilon",
            "maximum_updates",
        }
        if set(payload) != expected:
            raise ValueError("spectral regularization config fields are noncanonical")
        if payload["type"] != "SpectralRegularizer":
            raise ValueError("spectral regularization config type is unsupported")
        if payload["schema"] != SPECTRAL_REGULARIZATION_CONFIG_SCHEMA:
            raise ValueError("spectral regularization config schema is unsupported")
        if payload["source_profile"] != SPECTRAL_REGULARIZATION_SOURCE_PROFILE:
            raise ValueError("spectral regularization source profile is unsupported")
        return cls(
            output_dim=cast(int, payload["output_dim"]),
            input_dim=cast(int, payload["input_dim"]),
            coefficient=cast(float, payload["coefficient"]),
            exponent=cast(int, payload["exponent"]),
            power_iterations=cast(int, payload["power_iterations"]),
            normalization_epsilon=cast(float, payload["normalization_epsilon"]),
            maximum_updates=cast(int, payload["maximum_updates"]),
        )


@chex.dataclass(frozen=True)
class SpectralRegularizationState:
    """Power probe, post-initialization key ownership, and exact lifetime."""

    right_probe: Array
    rng_key: Array
    update_count_words: Array


@chex.dataclass(frozen=True)
class SpectralRegularizationResult:
    """One atomic regularizer evaluation and proposed state transition."""

    state: SpectralRegularizationState
    regularizer: Array
    scaled_loss: Array
    spectral_norm_estimate: Array
    weight_gradient: Array
    bias_gradient: Array
    accepted: Array
    exhausted: Array


@dataclasses.dataclass(frozen=True)
class SpectralRegularizationResourceDeclaration:
    """Exact logical persistent and per-evaluation work bounds."""

    parameter_count: int
    persistent_bytes: int
    power_matvecs_per_evaluation: int
    backward_evaluations_per_update: int


class SpectralRegularizer:
    """Stateful one-layer spectral objective with atomic finite rejection."""

    def __init__(self, config: SpectralRegularizationConfig):
        self._config = config
        self._maximum_words = _words_from_int(config.maximum_updates)

    @property
    def config(self) -> SpectralRegularizationConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> SpectralRegularizer:
        return cls(SpectralRegularizationConfig.from_config(payload))

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return metadata for use with the repository checkpoint helper."""

        return {
            "schema": SPECTRAL_REGULARIZATION_CHECKPOINT_SCHEMA,
            "type": "SpectralRegularizer",
            "config": self.to_config(),
        }

    @classmethod
    def from_checkpoint_metadata(cls, payload: Mapping[str, object]) -> SpectralRegularizer:
        if set(payload) != {"schema", "type", "config"}:
            raise ValueError("spectral regularization checkpoint metadata is noncanonical")
        if payload["schema"] != SPECTRAL_REGULARIZATION_CHECKPOINT_SCHEMA:
            raise ValueError("spectral regularization checkpoint schema is unsupported")
        if payload["type"] != "SpectralRegularizer":
            raise ValueError("spectral regularization checkpoint type is unsupported")
        config = payload["config"]
        if not isinstance(config, Mapping):
            raise ValueError("spectral regularization checkpoint config must be a mapping")
        return cls.from_config(config)

    def init(self, key: Array) -> SpectralRegularizationState:
        """Initialize a normalized probe from one explicitly owned key."""

        if not _typed_threefry_key(key):
            raise TypeError("key must be a scalar typed Threefry PRNG key")
        next_key, probe_key = jr.split(key)
        probe = jr.normal(probe_key, (self._config.input_dim,), dtype=jnp.float32)
        norm = jnp.linalg.vector_norm(probe)
        probe = probe / jnp.maximum(
            norm, jnp.asarray(self._config.normalization_epsilon, dtype=jnp.float32)
        )
        return SpectralRegularizationState(
            right_probe=probe,
            rng_key=next_key,
            update_count_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _validate_structure(
        self, state: SpectralRegularizationState, weight: Array, bias: Array
    ) -> None:
        _require_float32_array(
            state.right_probe,
            name="state.right_probe",
            shape=(self._config.input_dim,),
        )
        if not _typed_threefry_key(state.rng_key):
            raise TypeError("state.rng_key must be a scalar typed Threefry PRNG key")
        if tuple(state.update_count_words.shape) != (2,) or jnp.dtype(
            state.update_count_words.dtype
        ) != jnp.dtype(jnp.uint32):
            raise TypeError("state.update_count_words must be uint32[2]")
        _require_float32_array(
            weight,
            name="weight",
            shape=(self._config.output_dim, self._config.input_dim),
        )
        _require_float32_array(bias, name="bias", shape=(self._config.output_dim,))

    def evaluate(
        self,
        state: SpectralRegularizationState,
        weight: Array,
        bias: Array,
    ) -> SpectralRegularizationResult:
        """Evaluate the paper objective and gradient as one atomic transition."""

        self._validate_structure(state, weight, bias)
        epsilon = jnp.asarray(self._config.normalization_epsilon, dtype=jnp.float32)

        safe_weight = jnp.where(jnp.isfinite(weight), weight, 0.0)
        safe_bias = jnp.where(jnp.isfinite(bias), bias, 0.0)
        initial_probe = jnp.where(jnp.isfinite(state.right_probe), state.right_probe, 0.0)
        fallback_left = jnp.full(
            (self._config.output_dim,),
            1.0 / math.sqrt(self._config.output_dim),
            dtype=jnp.float32,
        )

        def power_body(
            _iteration: int, carry: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            _previous_left, right = carry
            left_candidate = safe_weight @ right
            left_norm = jnp.linalg.vector_norm(left_candidate)
            left = jnp.where(left_norm > epsilon, left_candidate / left_norm, fallback_left)
            right_candidate = safe_weight.T @ left
            right_norm = jnp.linalg.vector_norm(right_candidate)
            next_right = jnp.where(
                right_norm > epsilon, right_candidate / right_norm, right
            )
            return left, next_right

        left_probe, right_probe = jax.lax.fori_loop(
            0,
            self._config.power_iterations,
            power_body,
            (fallback_left, initial_probe),
        )
        left_probe = jax.lax.stop_gradient(left_probe)
        right_probe = jax.lax.stop_gradient(right_probe)

        exponent = self._config.exponent
        coefficient = jnp.asarray(self._config.coefficient, dtype=jnp.float32)

        def objective(w: Array, b: Array) -> tuple[Array, tuple[Array, Array]]:
            sigma = left_probe @ w @ right_probe
            # ||b||_2**(2*k) == (sum(b**2))**k.  The latter has the same
            # paper value and a finite derivative at b=0, unlike autodiff
            # through sqrt(sum(b**2)) at the origin.
            bias_squared_norm = jnp.sum(b * b)
            regularizer = (sigma**exponent - 1.0) ** 2 + bias_squared_norm**exponent
            return coefficient * regularizer, (regularizer, sigma)

        (scaled_loss, (regularizer, sigma)), (weight_gradient, bias_gradient) = (
            jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
                safe_weight, safe_bias
            )
        )
        next_words, counter_available = _increment_words(state.update_count_words)
        within_budget = _words_less(state.update_count_words, self._maximum_words)
        source_probe_norm = jnp.linalg.vector_norm(state.right_probe)
        source_probe_valid = jnp.isfinite(source_probe_norm) & (
            jnp.abs(source_probe_norm - 1.0)
            <= jnp.asarray(1.0e-4, dtype=jnp.float32)
        )
        finite = (
            source_probe_valid
            & jnp.all(jnp.isfinite(weight))
            & jnp.all(jnp.isfinite(bias))
            & jnp.isfinite(regularizer)
            & jnp.isfinite(scaled_loss)
            & jnp.isfinite(sigma)
            & jnp.all(jnp.isfinite(weight_gradient))
            & jnp.all(jnp.isfinite(bias_gradient))
        )
        accepted = finite & counter_available & within_budget
        proposed_state = SpectralRegularizationState(
            right_probe=right_probe,
            rng_key=state.rng_key,
            update_count_words=next_words,
        )
        next_state = SpectralRegularizationState(
            right_probe=jnp.where(
                accepted, proposed_state.right_probe, state.right_probe
            ),
            # Evaluation consumes no randomness, so this key never needs a
            # data-dependent selection (typed keys are not numeric arrays).
            rng_key=state.rng_key,
            update_count_words=jnp.where(
                accepted,
                proposed_state.update_count_words,
                state.update_count_words,
            ),
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return SpectralRegularizationResult(
            state=next_state,
            regularizer=jnp.where(accepted, regularizer, zero),
            scaled_loss=jnp.where(accepted, scaled_loss, zero),
            spectral_norm_estimate=jnp.where(accepted, sigma, zero),
            weight_gradient=jnp.where(accepted, weight_gradient, jnp.zeros_like(weight)),
            bias_gradient=jnp.where(accepted, bias_gradient, jnp.zeros_like(bias)),
            accepted=accepted,
            exhausted=~within_budget | ~counter_available,
        )

    def resource_declaration(self) -> SpectralRegularizationResourceDeclaration:
        """Return exact logical state and fixed per-call matrix-vector work."""

        parameter_count = self._config.output_dim * self._config.input_dim
        persistent_bytes = self._config.input_dim * 4 + 8 + 8
        return SpectralRegularizationResourceDeclaration(
            parameter_count=parameter_count + self._config.output_dim,
            persistent_bytes=persistent_bytes,
            power_matvecs_per_evaluation=2 * self._config.power_iterations,
            backward_evaluations_per_update=1,
        )


__all__ = [
    "SPECTRAL_REGULARIZATION_CHECKPOINT_SCHEMA",
    "SPECTRAL_REGULARIZATION_CONFIG_SCHEMA",
    "SPECTRAL_REGULARIZATION_DEFAULT_AGENT_INTEGRATION",
    "SPECTRAL_REGULARIZATION_EVIDENCE_LEVEL",
    "SPECTRAL_REGULARIZATION_MECHANISM_STATUS",
    "SPECTRAL_REGULARIZATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "SPECTRAL_REGULARIZATION_SOURCE_PROFILE",
    "SpectralRegularizationConfig",
    "SpectralRegularizationResourceDeclaration",
    "SpectralRegularizationResult",
    "SpectralRegularizationState",
    "SpectralRegularizer",
]
