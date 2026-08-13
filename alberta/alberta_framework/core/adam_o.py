# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded clean-room AdamO matrix optimizer.

This module implements Equations 16, 19, and 20 from Rosseau, Müller, and
Nowé, *Preserving Plasticity in Continual Learning via Dynamical Isometry*
(arXiv:2606.09762v1).  For a matrix ``W`` the paper's Gram penalty is

``||W.T @ W - I||_F**2`` when ``rows >= columns``, otherwise
``||W @ W.T - I||_F**2``.

AdamO updates Adam's first and second moments from the task gradient alone,
then adds a separately scaled gradient of that Gram penalty.  This separation
is the defining difference from applying Adam to a composite task-plus-
regularizer loss.  The implementation below is a clean-room equation surface;
the paper did not identify an official reusable implementation in the reviewed
v1 source.

The surface accepts one float32 matrix and returns the delta that a caller must
subtract.  Biases, convolution reshaping, parameter groups, schedules, and
agent integration remain caller-owned.  Invalid numeric proposals fail closed,
and an exact two-word lifetime prevents silent counter wrap.  This is bounded
L0 mechanism code, not a benchmark result or a default optimizer selection.

Primary source:
    Rosseau, A., Müller, R., & Nowé, A. (2026). Preserving Plasticity in
    Continual Learning via Dynamical Isometry. arXiv:2606.09762v1,
    Equations 16, 19, and 20. https://arxiv.org/abs/2606.09762
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import cast

import chex
import jax.numpy as jnp
import numpy as np
from jax import Array

ADAMO_CONFIG_SCHEMA = "alberta.adamo.config.v1"
ADAMO_CHECKPOINT_SCHEMA = "alberta.adamo.checkpoint.v1"
ADAMO_MECHANISM_STATUS = "l0-development-only-not-assessed"
ADAMO_EVIDENCE_LEVEL = "L0"
ADAMO_SCIENTIFIC_PROMOTION_ALLOWED = False
ADAMO_DEFAULT_AGENT_INTEGRATION = False
ADAMO_SOURCE_PROFILE = "rosseau-2026-equations-16-19-20-clean-room"

_UINT32_MAX = 4_294_967_295
_UINT64_MAX = 18_446_744_073_709_551_615
_MAX_DIMENSION = 16_384
_MAX_PARAMETERS = 16_777_216
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
    strict_upper: float | None = None,
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
    if strict_upper is not None and narrowed >= strict_upper:
        raise ValueError(f"{name} must be < {strict_upper}")
    return narrowed


def _words_from_int(value: int) -> Array:
    return jnp.asarray(
        [(value >> 32) & _UINT32_MAX, value & _UINT32_MAX], dtype=jnp.uint32
    )


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = words[0] + carry
    available = ~jnp.all(words == jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))
    return jnp.where(
        available, jnp.stack((high, low)).astype(jnp.uint32), words
    ), available


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_to_float32(words: Array) -> Array:
    return words[0].astype(jnp.float32) * jnp.asarray(
        4_294_967_296.0, dtype=jnp.float32
    ) + words[1].astype(jnp.float32)


def _require_matrix(value: object, *, name: str, shape: tuple[int, int]) -> None:
    if not hasattr(value, "shape") or tuple(cast(Array, value).shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if jnp.dtype(cast(Array, value).dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32")


@dataclasses.dataclass(frozen=True)
class AdamOConfig:
    """Static matrix shape and exact Equation 20 coefficients."""

    rows: int
    columns: int
    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    orthogonality_strength: float = 1.0e-3
    isometry_step_size: float | None = None
    maximum_updates: int = _UINT64_MAX

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rows", _strict_positive_int(self.rows, name="rows", maximum=_MAX_DIMENSION)
        )
        object.__setattr__(
            self,
            "columns",
            _strict_positive_int(self.columns, name="columns", maximum=_MAX_DIMENSION),
        )
        if self.rows * self.columns > _MAX_PARAMETERS:
            raise ValueError("AdamO matrix exceeds the bounded parameter budget")
        object.__setattr__(
            self,
            "learning_rate",
            _strict_float32(
                self.learning_rate, name="learning_rate", minimum=_FLOAT32_TINY, positive=True
            ),
        )
        object.__setattr__(
            self,
            "beta1",
            _strict_float32(
                self.beta1, name="beta1", minimum=0.0, strict_upper=1.0
            ),
        )
        object.__setattr__(
            self,
            "beta2",
            _strict_float32(
                self.beta2, name="beta2", minimum=0.0, strict_upper=1.0
            ),
        )
        object.__setattr__(
            self,
            "epsilon",
            _strict_float32(
                self.epsilon, name="epsilon", minimum=_FLOAT32_TINY, positive=True
            ),
        )
        object.__setattr__(
            self,
            "orthogonality_strength",
            _strict_float32(
                self.orthogonality_strength,
                name="orthogonality_strength",
                minimum=0.0,
            ),
        )
        step_size = (
            self.learning_rate
            if self.isometry_step_size is None
            else self.isometry_step_size
        )
        object.__setattr__(
            self,
            "isometry_step_size",
            _strict_float32(
                step_size,
                name="isometry_step_size",
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
        return {
            "type": "AdamO",
            "schema": ADAMO_CONFIG_SCHEMA,
            "source_profile": ADAMO_SOURCE_PROFILE,
            "rows": self.rows,
            "columns": self.columns,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "orthogonality_strength": self.orthogonality_strength,
            "isometry_step_size": self.isometry_step_size,
            "maximum_updates": self.maximum_updates,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> AdamOConfig:
        expected = {
            "type",
            "schema",
            "source_profile",
            "rows",
            "columns",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "orthogonality_strength",
            "isometry_step_size",
            "maximum_updates",
        }
        if set(payload) != expected:
            raise ValueError("AdamO config fields are noncanonical")
        if payload["type"] != "AdamO" or payload["schema"] != ADAMO_CONFIG_SCHEMA:
            raise ValueError("AdamO config type or schema is unsupported")
        if payload["source_profile"] != ADAMO_SOURCE_PROFILE:
            raise ValueError("AdamO source profile is unsupported")
        return cls(
            rows=cast(int, payload["rows"]),
            columns=cast(int, payload["columns"]),
            learning_rate=cast(float, payload["learning_rate"]),
            beta1=cast(float, payload["beta1"]),
            beta2=cast(float, payload["beta2"]),
            epsilon=cast(float, payload["epsilon"]),
            orthogonality_strength=cast(float, payload["orthogonality_strength"]),
            isometry_step_size=cast(float, payload["isometry_step_size"]),
            maximum_updates=cast(int, payload["maximum_updates"]),
        )


@chex.dataclass(frozen=True)
class AdamOState:
    """Task-gradient Adam moments and exact lifetime."""

    first_moment: Array
    second_moment: Array
    update_count_words: Array


@chex.dataclass(frozen=True)
class AdamOResult:
    """One Equation 20 delta and its atomic state transition."""

    state: AdamOState
    parameter_delta: Array
    task_delta: Array
    isometry_delta: Array
    orthogonality_regularizer: Array
    accepted: Array
    exhausted: Array


@dataclasses.dataclass(frozen=True)
class AdamOResourceDeclaration:
    """Exact logical optimizer state and Gram-work declaration."""

    parameter_count: int
    persistent_bytes: int
    gram_dimension: int
    gram_matrix_elements: int
    task_gradient_moment_updates_per_call: int


def orthogonality_regularizer(weight: Array) -> Array:
    """Return paper Equation 16 for one rectangular float32 matrix."""

    rows, columns = weight.shape
    if rows >= columns:
        gram = weight.T @ weight
        identity = jnp.eye(columns, dtype=weight.dtype)
    else:
        gram = weight @ weight.T
        identity = jnp.eye(rows, dtype=weight.dtype)
    deviation = gram - identity
    return jnp.sum(deviation * deviation)


def orthogonality_gradient(weight: Array) -> Array:
    """Return the closed-form gradient of paper Equation 16."""

    rows, columns = weight.shape
    if rows >= columns:
        deviation = weight.T @ weight - jnp.eye(columns, dtype=weight.dtype)
        return 4.0 * weight @ deviation
    deviation = weight @ weight.T - jnp.eye(rows, dtype=weight.dtype)
    return 4.0 * deviation @ weight


class AdamO:
    """One-matrix AdamO transform whose returned delta is subtracted."""

    def __init__(self, config: AdamOConfig):
        self._config = config
        self._shape = (config.rows, config.columns)
        self._maximum_words = _words_from_int(config.maximum_updates)

    @property
    def config(self) -> AdamOConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> AdamO:
        return cls(AdamOConfig.from_config(payload))

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "schema": ADAMO_CHECKPOINT_SCHEMA,
            "type": "AdamO",
            "config": self.to_config(),
        }

    @classmethod
    def from_checkpoint_metadata(cls, payload: Mapping[str, object]) -> AdamO:
        if set(payload) != {"schema", "type", "config"}:
            raise ValueError("AdamO checkpoint metadata is noncanonical")
        if payload["schema"] != ADAMO_CHECKPOINT_SCHEMA or payload["type"] != "AdamO":
            raise ValueError("AdamO checkpoint type or schema is unsupported")
        config = payload["config"]
        if not isinstance(config, Mapping):
            raise ValueError("AdamO checkpoint config must be a mapping")
        return cls.from_config(config)

    def init(self) -> AdamOState:
        return AdamOState(
            first_moment=jnp.zeros(self._shape, dtype=jnp.float32),
            second_moment=jnp.zeros(self._shape, dtype=jnp.float32),
            update_count_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _validate_structure(
        self, state: AdamOState, weight: Array, task_gradient: Array
    ) -> None:
        _require_matrix(state.first_moment, name="state.first_moment", shape=self._shape)
        _require_matrix(state.second_moment, name="state.second_moment", shape=self._shape)
        if tuple(state.update_count_words.shape) != (2,) or jnp.dtype(
            state.update_count_words.dtype
        ) != jnp.dtype(jnp.uint32):
            raise TypeError("state.update_count_words must be uint32[2]")
        _require_matrix(weight, name="weight", shape=self._shape)
        _require_matrix(task_gradient, name="task_gradient", shape=self._shape)

    def update(self, state: AdamOState, weight: Array, task_gradient: Array) -> AdamOResult:
        """Form Equation 20 without mixing the regularizer into Adam moments."""

        self._validate_structure(state, weight, task_gradient)
        safe_weight = jnp.where(jnp.isfinite(weight), weight, 0.0)
        safe_gradient = jnp.where(jnp.isfinite(task_gradient), task_gradient, 0.0)
        beta1 = jnp.asarray(self._config.beta1, dtype=jnp.float32)
        beta2 = jnp.asarray(self._config.beta2, dtype=jnp.float32)
        first = beta1 * state.first_moment + (1.0 - beta1) * safe_gradient
        second = beta2 * state.second_moment + (1.0 - beta2) * safe_gradient**2
        next_words, counter_available = _increment_words(state.update_count_words)
        step = _words_to_float32(next_words)
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        task_delta = (
            jnp.asarray(self._config.learning_rate, dtype=jnp.float32)
            * first_hat
            / (jnp.sqrt(second_hat) + self._config.epsilon)
        )
        regularizer = orthogonality_regularizer(safe_weight)
        regularizer_gradient = orthogonality_gradient(safe_weight)
        isometry_delta = (
            jnp.asarray(cast(float, self._config.isometry_step_size), dtype=jnp.float32)
            * jnp.asarray(self._config.orthogonality_strength, dtype=jnp.float32)
            * regularizer_gradient
        )
        parameter_delta = task_delta + isometry_delta
        within_budget = _words_less(state.update_count_words, self._maximum_words)
        finite = (
            jnp.all(jnp.isfinite(weight))
            & jnp.all(jnp.isfinite(task_gradient))
            & jnp.all(jnp.isfinite(state.first_moment))
            & jnp.all(jnp.isfinite(state.second_moment))
            & jnp.all(state.second_moment >= 0.0)
            & jnp.all(jnp.isfinite(first))
            & jnp.all(jnp.isfinite(second))
            & jnp.all(jnp.isfinite(parameter_delta))
            & jnp.isfinite(regularizer)
        )
        accepted = finite & counter_available & within_budget
        next_state = AdamOState(
            first_moment=jnp.where(accepted, first, state.first_moment),
            second_moment=jnp.where(accepted, second, state.second_moment),
            update_count_words=jnp.where(
                accepted, next_words, state.update_count_words
            ),
        )
        return AdamOResult(
            state=next_state,
            parameter_delta=jnp.where(
                accepted, parameter_delta, jnp.zeros_like(parameter_delta)
            ),
            task_delta=jnp.where(accepted, task_delta, jnp.zeros_like(task_delta)),
            isometry_delta=jnp.where(
                accepted, isometry_delta, jnp.zeros_like(isometry_delta)
            ),
            orthogonality_regularizer=jnp.where(
                accepted, regularizer, jnp.asarray(0.0, dtype=jnp.float32)
            ),
            accepted=accepted,
            exhausted=~within_budget | ~counter_available,
        )

    def resource_declaration(self) -> AdamOResourceDeclaration:
        parameter_count = self._config.rows * self._config.columns
        gram_dimension = min(self._config.rows, self._config.columns)
        return AdamOResourceDeclaration(
            parameter_count=parameter_count,
            persistent_bytes=2 * parameter_count * 4 + 8,
            gram_dimension=gram_dimension,
            gram_matrix_elements=gram_dimension * gram_dimension,
            task_gradient_moment_updates_per_call=2 * parameter_count,
        )


__all__ = [
    "ADAMO_CHECKPOINT_SCHEMA",
    "ADAMO_CONFIG_SCHEMA",
    "ADAMO_DEFAULT_AGENT_INTEGRATION",
    "ADAMO_EVIDENCE_LEVEL",
    "ADAMO_MECHANISM_STATUS",
    "ADAMO_SCIENTIFIC_PROMOTION_ALLOWED",
    "ADAMO_SOURCE_PROFILE",
    "AdamO",
    "AdamOConfig",
    "AdamOResourceDeclaration",
    "AdamOResult",
    "AdamOState",
    "orthogonality_gradient",
    "orthogonality_regularizer",
]
