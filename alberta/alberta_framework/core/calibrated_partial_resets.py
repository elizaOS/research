# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded dense-layer Calibrated Partial Resets (CPR).

This module implements Equations 3, 5--7 and Algorithm 1 from McCutcheon,
Chatzaroulas, and Fallah, *Calibrated Partial Resets: Preventing Policy
Collapse in Continual Reinforcement Learning* (arXiv:2607.24996v1).

For each hidden unit, CPR forms the mean absolute per-example gradient of its
incoming weights, normalizes those scores by the layer mean, and maintains an
EMA.  On scheduled events it computes

``r_i = rho * min(2 * sigmoid(-kappa * (u_i - 1)), 1)``

then pulls incoming weights toward a fresh He-uniform sample by ``r_i`` and
outgoing weights toward zero by the same amount.  Utilities are re-centered to
one after a reset event.

The reviewed official JAX implementation is pinned below.  This clean bounded
surface follows its important operational choices: gradients retain an
explicit sample axis, the source time step is tested before increment, random
state advances only on reset events, and the caller's base optimizer state is
not modified.  It intentionally excludes biases because the paper appendix
and released v1 implementation differ there (the appendix discusses bias
reinitialization while the implementation leaves biases unchanged).

Parameters follow the repository's dense-layer convention: incoming weights
are ``(input_dim, unit_count)`` and outgoing weights are
``(unit_count, output_dim)``.  The caller supplies parameters after its normal
optimizer update.  This is a one-layer L0 mechanism arm, not a full PPO/SAC
optimizer wrapper, agent integration, benchmark result, or promotion claim.

Primary sources:
    McCutcheon, L., Chatzaroulas, E., & Fallah, S. (2026). Calibrated Partial
    Resets: Preventing Policy Collapse in Continual Reinforcement Learning.
    arXiv:2607.24996v1, Equations 3, 5--7 and Algorithm 1.
    https://arxiv.org/abs/2607.24996

    Official JAX reference implementation, commit
    ``6fc2af34783159f5dda50c6915dda32c2d443604``,
    ``continual_learning/optim/cpr.py``:
    https://github.com/LucMc/continual-learning/commit/6fc2af34783159f5dda50c6915dda32c2d443604
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

CPR_CONFIG_SCHEMA = "alberta.calibrated-partial-resets.config.v1"
CPR_CHECKPOINT_SCHEMA = "alberta.calibrated-partial-resets.checkpoint.v1"
CPR_MECHANISM_STATUS = "l0-development-only-not-assessed"
CPR_EVIDENCE_LEVEL = "L0"
CPR_SCIENTIFIC_PROMOTION_ALLOWED = False
CPR_DEFAULT_AGENT_INTEGRATION = False
CPR_BASE_OPTIMIZER_STATE_MUTATED = False
CPR_BIAS_POLICY = "excluded-paper-appendix-and-official-v1-diverge"
CPR_SOURCE_PROFILE = (
    "mccutcheon-2026-equations-3-5-6-7-algorithm-1-official-jax-6fc2af34"
)

_UINT32_MAX = 4_294_967_295
_UINT64_MAX = 18_446_744_073_709_551_615
_MAX_DIMENSION = 16_384
_MAX_PARAMETERS = 16_777_216
_MAX_UPDATE_FREQUENCY = 65_535
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
    maximum: float | None = None,
    strict_maximum: bool = False,
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
    if maximum is not None:
        invalid = narrowed >= maximum if strict_maximum else narrowed > maximum
        if invalid:
            operator = "<" if strict_maximum else "<="
            raise ValueError(f"{name} must be {operator} {maximum}")
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


def _words_at_most(left: Array, right: Array) -> Array:
    return _words_less(left, right) | jnp.all(left == right)


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_mod(words: Array, divisor: int) -> Array:
    """Return uint64-as-two-words modulo a bounded 16-bit divisor."""

    modulus = jnp.asarray(divisor, dtype=jnp.uint32)
    two32_mod = jnp.asarray((1 << 32) % divisor, dtype=jnp.uint32)
    return ((words[0] % modulus) * two32_mod + (words[1] % modulus)) % modulus


def _require_float32_array(value: object, *, name: str, shape: tuple[int, ...]) -> None:
    if not hasattr(value, "shape") or tuple(cast(Any, value).shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if jnp.dtype(cast(Any, value).dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32")


@dataclasses.dataclass(frozen=True)
class CalibratedPartialResetsConfig:
    """Static dense-layer shape, CPR coefficients, schedule, and lifetime."""

    input_dim: int
    unit_count: int
    output_dim: int
    replacement_rate: float = 0.015
    sharpness: float = 16.0
    utility_decay: float = 0.99
    update_frequency: int = 1_000
    utility_normalization_epsilon: float = 1.0e-8
    initialization_scale: float = 1.0
    maximum_updates: int = _UINT64_MAX

    def __post_init__(self) -> None:
        for name in ("input_dim", "unit_count", "output_dim"):
            object.__setattr__(
                self,
                name,
                _strict_positive_int(
                    getattr(self, name), name=name, maximum=_MAX_DIMENSION
                ),
            )
        if (
            self.input_dim * self.unit_count
            + self.unit_count * self.output_dim
            > _MAX_PARAMETERS
        ):
            raise ValueError("CPR layer exceeds the bounded parameter budget")
        object.__setattr__(
            self,
            "replacement_rate",
            _strict_float32(
                self.replacement_rate,
                name="replacement_rate",
                minimum=_FLOAT32_TINY,
                positive=True,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "sharpness",
            _strict_float32(
                self.sharpness, name="sharpness", minimum=_FLOAT32_TINY, positive=True
            ),
        )
        object.__setattr__(
            self,
            "utility_decay",
            _strict_float32(
                self.utility_decay,
                name="utility_decay",
                minimum=0.0,
                maximum=1.0,
                strict_maximum=True,
            ),
        )
        object.__setattr__(
            self,
            "update_frequency",
            _strict_positive_int(
                self.update_frequency,
                name="update_frequency",
                maximum=_MAX_UPDATE_FREQUENCY,
            ),
        )
        object.__setattr__(
            self,
            "utility_normalization_epsilon",
            _strict_float32(
                self.utility_normalization_epsilon,
                name="utility_normalization_epsilon",
                minimum=_FLOAT32_TINY,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "initialization_scale",
            _strict_float32(
                self.initialization_scale,
                name="initialization_scale",
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
            "type": "CalibratedPartialResets",
            "schema": CPR_CONFIG_SCHEMA,
            "source_profile": CPR_SOURCE_PROFILE,
            "bias_policy": CPR_BIAS_POLICY,
            "input_dim": self.input_dim,
            "unit_count": self.unit_count,
            "output_dim": self.output_dim,
            "replacement_rate": self.replacement_rate,
            "sharpness": self.sharpness,
            "utility_decay": self.utility_decay,
            "update_frequency": self.update_frequency,
            "utility_normalization_epsilon": self.utility_normalization_epsilon,
            "initialization_scale": self.initialization_scale,
            "maximum_updates": self.maximum_updates,
        }

    @classmethod
    def from_config(
        cls, payload: Mapping[str, object]
    ) -> CalibratedPartialResetsConfig:
        expected = {
            "type",
            "schema",
            "source_profile",
            "bias_policy",
            "input_dim",
            "unit_count",
            "output_dim",
            "replacement_rate",
            "sharpness",
            "utility_decay",
            "update_frequency",
            "utility_normalization_epsilon",
            "initialization_scale",
            "maximum_updates",
        }
        if set(payload) != expected:
            raise ValueError("CPR config fields are noncanonical")
        if payload["type"] != "CalibratedPartialResets" or payload["schema"] != CPR_CONFIG_SCHEMA:
            raise ValueError("CPR config type or schema is unsupported")
        if payload["source_profile"] != CPR_SOURCE_PROFILE:
            raise ValueError("CPR source profile is unsupported")
        if payload["bias_policy"] != CPR_BIAS_POLICY:
            raise ValueError("CPR bias policy is unsupported")
        return cls(
            input_dim=cast(int, payload["input_dim"]),
            unit_count=cast(int, payload["unit_count"]),
            output_dim=cast(int, payload["output_dim"]),
            replacement_rate=cast(float, payload["replacement_rate"]),
            sharpness=cast(float, payload["sharpness"]),
            utility_decay=cast(float, payload["utility_decay"]),
            update_frequency=cast(int, payload["update_frequency"]),
            utility_normalization_epsilon=cast(
                float, payload["utility_normalization_epsilon"]
            ),
            initialization_scale=cast(float, payload["initialization_scale"]),
            maximum_updates=cast(int, payload["maximum_updates"]),
        )


@chex.dataclass(frozen=True)
class CalibratedPartialResetsParameters:
    """One hidden layer's post-base-optimizer parameters."""

    incoming_weight: Array
    outgoing_weight: Array


@chex.dataclass(frozen=True)
class CalibratedPartialResetsState:
    """Layer-normalized utility EMA, random ownership, and exact clocks."""

    utility: Array
    rng_key: Array
    update_count_words: Array
    reset_event_count_words: Array


@chex.dataclass(frozen=True)
class CalibratedPartialResetsResult:
    """One post-optimizer CPR transaction and complete diagnostics."""

    state: CalibratedPartialResetsState
    parameters: CalibratedPartialResetsParameters
    raw_utility: Array
    normalized_utility: Array
    utility_before_recentering: Array
    reset_fraction: Array
    reset_applied: Array
    accepted: Array
    exhausted: Array


@dataclasses.dataclass(frozen=True)
class CalibratedPartialResetsResourceDeclaration:
    """Exact logical persistent and maximum reset-event work."""

    caller_owned_parameter_count: int
    persistent_bytes: int
    utility_slots: int
    initialization_draws_per_reset_event: int
    base_optimizer_state_bytes_owned: int


class CalibratedPartialResets:
    """One-layer CPR transform applied after a caller-owned optimizer update."""

    def __init__(self, config: CalibratedPartialResetsConfig):
        self._config = config
        self._maximum_words = _words_from_int(config.maximum_updates)
        self._incoming_shape = (config.input_dim, config.unit_count)
        self._outgoing_shape = (config.unit_count, config.output_dim)

    @property
    def config(self) -> CalibratedPartialResetsConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> CalibratedPartialResets:
        return cls(CalibratedPartialResetsConfig.from_config(payload))

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "schema": CPR_CHECKPOINT_SCHEMA,
            "type": "CalibratedPartialResets",
            "config": self.to_config(),
        }

    @classmethod
    def from_checkpoint_metadata(
        cls, payload: Mapping[str, object]
    ) -> CalibratedPartialResets:
        if set(payload) != {"schema", "type", "config"}:
            raise ValueError("CPR checkpoint metadata is noncanonical")
        if (
            payload["schema"] != CPR_CHECKPOINT_SCHEMA
            or payload["type"] != "CalibratedPartialResets"
        ):
            raise ValueError("CPR checkpoint type or schema is unsupported")
        config = payload["config"]
        if not isinstance(config, Mapping):
            raise ValueError("CPR checkpoint config must be a mapping")
        return cls.from_config(config)

    def init(self, key: Array) -> CalibratedPartialResetsState:
        """Initialize utility at its normalized mean and take key ownership."""

        if not _typed_threefry_key(key):
            raise TypeError("key must be a scalar typed Threefry PRNG key")
        return CalibratedPartialResetsState(
            utility=jnp.ones((self._config.unit_count,), dtype=jnp.float32),
            rng_key=key,
            update_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            reset_event_count_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _validate_structure(
        self,
        state: CalibratedPartialResetsState,
        parameters: CalibratedPartialResetsParameters,
        incoming_gradient_samples: Array,
    ) -> None:
        _require_float32_array(
            state.utility,
            name="state.utility",
            shape=(self._config.unit_count,),
        )
        if not _typed_threefry_key(state.rng_key):
            raise TypeError("state.rng_key must be a scalar typed Threefry PRNG key")
        for name, words in (
            ("state.update_count_words", state.update_count_words),
            ("state.reset_event_count_words", state.reset_event_count_words),
        ):
            if tuple(words.shape) != (2,) or jnp.dtype(words.dtype) != jnp.dtype(jnp.uint32):
                raise TypeError(f"{name} must be uint32[2]")
        _require_float32_array(
            parameters.incoming_weight,
            name="parameters.incoming_weight",
            shape=self._incoming_shape,
        )
        _require_float32_array(
            parameters.outgoing_weight,
            name="parameters.outgoing_weight",
            shape=self._outgoing_shape,
        )
        if incoming_gradient_samples.ndim != 3:
            raise ValueError(
                "incoming_gradient_samples must retain (sample, input, unit) axes"
            )
        if incoming_gradient_samples.shape[0] < 1:
            raise ValueError("incoming_gradient_samples must contain at least one sample")
        _require_float32_array(
            incoming_gradient_samples,
            name="incoming_gradient_samples",
            shape=(
                incoming_gradient_samples.shape[0],
                self._config.input_dim,
                self._config.unit_count,
            ),
        )

    def update_after_optimizer(
        self,
        state: CalibratedPartialResetsState,
        parameters: CalibratedPartialResetsParameters,
        incoming_gradient_samples: Array,
    ) -> CalibratedPartialResetsResult:
        """Apply Algorithm 1 after the caller's ordinary optimizer update."""

        self._validate_structure(state, parameters, incoming_gradient_samples)
        safe_gradients = jnp.where(
            jnp.isfinite(incoming_gradient_samples), incoming_gradient_samples, 0.0
        )
        raw_utility = jnp.mean(jnp.abs(safe_gradients), axis=(0, 1))
        normalized_utility = raw_utility / (
            jnp.mean(raw_utility)
            + jnp.asarray(
                self._config.utility_normalization_epsilon, dtype=jnp.float32
            )
        )
        decay = jnp.asarray(self._config.utility_decay, dtype=jnp.float32)
        updated_utility = decay * state.utility + (1.0 - decay) * normalized_utility
        within_budget = _words_less(state.update_count_words, self._maximum_words)
        next_update_words, update_counter_available = _increment_words(
            state.update_count_words
        )
        due = _words_nonzero(state.update_count_words) & (
            _words_mod(state.update_count_words, self._config.update_frequency)
            == jnp.asarray(0, dtype=jnp.uint32)
        )

        def apply_reset(
            _: None,
        ) -> tuple[CalibratedPartialResetsParameters, Array, Array, Array, Array]:
            next_key, initialization_key = jr.split(state.rng_key)
            bound = jnp.asarray(
                self._config.initialization_scale
                * math.sqrt(6.0 / self._config.input_dim),
                dtype=jnp.float32,
            )
            fresh_incoming = jr.uniform(
                initialization_key,
                self._incoming_shape,
                dtype=jnp.float32,
                minval=-bound,
                maxval=bound,
            )
            transformed = jnp.minimum(
                2.0
                * jax.nn.sigmoid(
                    -jnp.asarray(self._config.sharpness, dtype=jnp.float32)
                    * (updated_utility - 1.0)
                ),
                1.0,
            )
            reset_fraction = (
                jnp.asarray(self._config.replacement_rate, dtype=jnp.float32)
                * transformed
            )
            incoming_fraction = reset_fraction[jnp.newaxis, :]
            outgoing_fraction = reset_fraction[:, jnp.newaxis]
            reset_parameters = CalibratedPartialResetsParameters(
                incoming_weight=(1.0 - incoming_fraction)
                * parameters.incoming_weight
                + incoming_fraction * fresh_incoming,
                outgoing_weight=(1.0 - outgoing_fraction)
                * parameters.outgoing_weight,
            )
            next_reset_words, reset_counter_available = _increment_words(
                state.reset_event_count_words
            )
            return (
                reset_parameters,
                jnp.ones_like(updated_utility),
                next_key,
                next_reset_words,
                reset_counter_available,
            )

        def skip_reset(
            _: None,
        ) -> tuple[CalibratedPartialResetsParameters, Array, Array, Array, Array]:
            return (
                parameters,
                updated_utility,
                state.rng_key,
                state.reset_event_count_words,
                jnp.asarray(True),
            )

        (
            proposed_parameters,
            proposed_utility,
            proposed_key,
            proposed_reset_words,
            reset_counter_available,
        ) = jax.lax.cond(due, apply_reset, skip_reset, operand=None)
        finite = (
            jnp.all(jnp.isfinite(state.utility))
            & jnp.all(state.utility >= 0.0)
            & _words_at_most(
                state.reset_event_count_words, state.update_count_words
            )
            & jnp.all(jnp.isfinite(parameters.incoming_weight))
            & jnp.all(jnp.isfinite(parameters.outgoing_weight))
            & jnp.all(jnp.isfinite(incoming_gradient_samples))
            & jnp.all(jnp.isfinite(raw_utility))
            & jnp.all(jnp.isfinite(normalized_utility))
            & jnp.all(jnp.isfinite(updated_utility))
            & jnp.all(jnp.isfinite(proposed_parameters.incoming_weight))
            & jnp.all(jnp.isfinite(proposed_parameters.outgoing_weight))
        )
        accepted = (
            finite
            & within_budget
            & update_counter_available
            & reset_counter_available
        )
        next_state = CalibratedPartialResetsState(
            utility=jnp.where(accepted, proposed_utility, state.utility),
            rng_key=jax.lax.cond(
                accepted, lambda _: proposed_key, lambda _: state.rng_key, operand=None
            ),
            update_count_words=jnp.where(
                accepted, next_update_words, state.update_count_words
            ),
            reset_event_count_words=jnp.where(
                accepted, proposed_reset_words, state.reset_event_count_words
            ),
        )
        next_parameters = CalibratedPartialResetsParameters(
            incoming_weight=jnp.where(
                accepted,
                proposed_parameters.incoming_weight,
                parameters.incoming_weight,
            ),
            outgoing_weight=jnp.where(
                accepted,
                proposed_parameters.outgoing_weight,
                parameters.outgoing_weight,
            ),
        )
        zero_utility = jnp.zeros((self._config.unit_count,), dtype=jnp.float32)
        transformed = jnp.minimum(
            2.0
            * jax.nn.sigmoid(
                -jnp.asarray(self._config.sharpness, dtype=jnp.float32)
                * (updated_utility - 1.0)
            ),
            1.0,
        )
        reset_fraction = jnp.where(
            due & accepted,
            jnp.asarray(self._config.replacement_rate, dtype=jnp.float32)
            * transformed,
            zero_utility,
        )
        return CalibratedPartialResetsResult(
            state=next_state,
            parameters=next_parameters,
            raw_utility=jnp.where(accepted, raw_utility, zero_utility),
            normalized_utility=jnp.where(
                accepted, normalized_utility, zero_utility
            ),
            utility_before_recentering=jnp.where(
                accepted, updated_utility, zero_utility
            ),
            reset_fraction=reset_fraction,
            reset_applied=due & accepted,
            accepted=accepted,
            exhausted=(
                ~within_budget
                | ~update_counter_available
                | (due & ~reset_counter_available)
            ),
        )

    def resource_declaration(self) -> CalibratedPartialResetsResourceDeclaration:
        parameter_count = (
            self._config.input_dim * self._config.unit_count
            + self._config.unit_count * self._config.output_dim
        )
        return CalibratedPartialResetsResourceDeclaration(
            caller_owned_parameter_count=parameter_count,
            persistent_bytes=self._config.unit_count * 4 + 8 + 8 + 8,
            utility_slots=self._config.unit_count,
            initialization_draws_per_reset_event=(
                self._config.input_dim * self._config.unit_count
            ),
            base_optimizer_state_bytes_owned=0,
        )


__all__ = [
    "CPR_BASE_OPTIMIZER_STATE_MUTATED",
    "CPR_BIAS_POLICY",
    "CPR_CHECKPOINT_SCHEMA",
    "CPR_CONFIG_SCHEMA",
    "CPR_DEFAULT_AGENT_INTEGRATION",
    "CPR_EVIDENCE_LEVEL",
    "CPR_MECHANISM_STATUS",
    "CPR_SCIENTIFIC_PROMOTION_ALLOWED",
    "CPR_SOURCE_PROFILE",
    "CalibratedPartialResets",
    "CalibratedPartialResetsConfig",
    "CalibratedPartialResetsParameters",
    "CalibratedPartialResetsResourceDeclaration",
    "CalibratedPartialResetsResult",
    "CalibratedPartialResetsState",
]
