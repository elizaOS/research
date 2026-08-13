# mypy: disable-error-code="call-arg,name-defined"
"""Causal temporal/context features for non-stationary Step 2 streams.

The featurizer augments each observation with cheap context blocks a
downstream linear or shallow learner can exploit under drift:

1. **EMA copy** — a slow exponential average of the observation stream, a
   causal summary of the recent input regime.
2. **Innovation** — the difference between the current observation and that
   EMA, isolating what just changed.
3. **Phase code** — ``sin``/``cos`` of the absolute step count at fixed
   periods (two features per period), a Fourier-style time encoding that lets
   even a linear readout represent target functions that vary periodically in
   time; the sin/cos pair covers arbitrary phase offsets.

Caveat: the phase code depends on the absolute step counter, not on the
observations, so it leaks global time into the feature vector — the same
observation maps to different features at different steps.  It helps only
when the stream's nonstationarity is genuinely periodic near the configured
periods; on aperiodic streams it is a spurious clock signal a downstream
learner can overfit.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_INTEGER_LIMIT = 2**24

TEMPORAL_CONTEXT_STATE_SCHEMA = "alberta.temporal-context-state.v2"
TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES = 12
TEMPORAL_CONTEXT_LIFETIME_COUNTER_DELTA_NBYTES = 8


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact context event without wrapping all-ones words."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("temporal-context step words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("temporal-context step words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_to_int32_telemetry(words: Array) -> Int[Array, ""]:
    """Project an exact identity to saturating compatibility telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("temporal-context step words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("temporal-context step words must have dtype uint32")
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Authenticate exact event identity against compatibility telemetry."""

    count = jnp.asarray(telemetry)
    if count.shape != ():
        raise ValueError("temporal-context step_count must be scalar")
    if count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("temporal-context step_count must have dtype int32")
    return (count >= 0) & (count == _words_to_int32_telemetry(words))


def _words_mod_positive_int(words: Array, divisor: int) -> UInt[Array, ""]:
    """Return exact uint64-word remainder using only portable uint32 ops."""

    if type(divisor) is not int or not 1 <= divisor <= _FLOAT32_INTEGER_LIMIT:
        raise ValueError("temporal-context period must be an exact positive float32 integer")
    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("temporal-context step words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("temporal-context step words must have dtype uint32")
    divisor_u = jnp.asarray(divisor, dtype=jnp.uint32)
    one = jnp.asarray(1, dtype=jnp.uint32)

    def reduce_bit(index: int, remainder: Array) -> Array:
        bit_index = jnp.asarray(63 - index, dtype=jnp.int32)
        from_high = bit_index >= 32
        shift = jnp.where(from_high, bit_index - 32, bit_index)
        source = jnp.where(from_high, array[0], array[1])
        bit = (source >> shift.astype(jnp.uint32)) & one
        doubled = remainder + remainder + bit
        return jnp.where(doubled >= divisor_u, doubled - divisor_u, doubled)

    return cast(
        UInt[Array, ""],
        jax.lax.fori_loop(
            0,
            64,
            reduce_bit,
            jnp.asarray(0, dtype=jnp.uint32),
        ).astype(jnp.uint32),
    )


@dataclass(frozen=True)
class TemporalContextConfig:
    """Configuration for :class:`TemporalContextFeaturizer`.

    The featurizer is causal: features at time ``t`` use the pre-update EMA and
    the current step counter, then the EMA is advanced after the observation is
    exposed.  This is meant for streams whose target changes with slowly moving
    latent context, such as rotating relevant subspaces.

    The default ``periods`` (50, 100, 200) span drift timescales of tens to a
    few hundred steps; set them to the stream's known drift periods when those
    are available.
    """

    input_dim: int
    include_raw: bool = True
    include_ema: bool = True
    include_delta: bool = True
    include_phase_products: bool = False
    ema_decay: float = 0.95
    periods: tuple[float, ...] = (50.0, 100.0, 200.0)

    def output_dim(self) -> int:
        """Return the transformed feature dimensionality."""
        copies = int(self.include_raw) + int(self.include_ema) + int(self.include_delta)
        phase_dim = 2 * len(self.periods)
        product_dim = phase_dim * self.input_dim * int(self.include_phase_products)
        return copies * self.input_dim + phase_dim + product_dim

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        payload["type"] = "TemporalContextConfig"
        payload["state_schema"] = TEMPORAL_CONTEXT_STATE_SCHEMA
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TemporalContextConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""
        payload = dict(config)
        expected = {
            "type",
            "state_schema",
            "input_dim",
            "include_raw",
            "include_ema",
            "include_delta",
            "include_phase_products",
            "ema_decay",
            "periods",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                "temporal-context config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if payload.pop("type") != "TemporalContextConfig":
            raise ValueError("temporal-context config type is unsupported")
        if payload.pop("state_schema") != TEMPORAL_CONTEXT_STATE_SCHEMA:
            raise ValueError("temporal-context state schema is unsupported")
        payload["periods"] = tuple(payload["periods"])
        return cls(**payload)


@chex.dataclass(frozen=True)
class TemporalContextState:
    """State for :class:`TemporalContextFeaturizer`."""

    observation_ema: Float[Array, " input_dim"]
    step_count: Array
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class TemporalContextUpdateResult:
    """Fail-closed diagnostics for one exact context-state update attempt."""

    state: TemporalContextState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class TemporalContextStepResult:
    """Causal features plus the owned atomic update result."""

    state: TemporalContextState
    features: Float[Array, " output_dim"]
    update: TemporalContextUpdateResult


@dataclass(frozen=True)
class TemporalContextResourceBudget:
    """Exact fixed persistent-state accounting."""

    output_scalars: int
    state_scalars: int
    state_bytes: int
    exact_lifetime_counter_bytes: int


def _validate_config(config: TemporalContextConfig) -> None:
    if type(config.input_dim) is not int or config.input_dim < 1:
        raise ValueError("input_dim must be positive")
    if not (config.include_raw or config.include_ema or config.include_delta):
        raise ValueError("at least one observation feature block must be included")
    if not math.isfinite(config.ema_decay) or not 0.0 <= config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not isinstance(config.periods, tuple):
        raise ValueError("periods must be a tuple")
    if any(
        isinstance(period, bool)
        or not isinstance(period, (int, float))
        or not math.isfinite(float(period))
        or float(period) != int(period)
        or not 1 <= int(period) <= _FLOAT32_INTEGER_LIMIT
        for period in config.periods
    ):
        raise ValueError(
            "all temporal periods must be exact positive float32 integers"
        )


class TemporalContextFeaturizer:
    """Causal feature wrapper exposing EMA, innovation, and phase features."""

    def __init__(self, config: TemporalContextConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> TemporalContextConfig:
        """Featurizer configuration."""
        return self._config

    def init(self) -> TemporalContextState:
        """Return an all-zero initial context state."""
        return TemporalContextState(
            observation_ema=jnp.zeros(self._config.input_dim, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def resource_budget(self) -> TemporalContextResourceBudget:
        """Return exact fixed state and output storage."""

        state_scalars = self._config.input_dim + 3
        return TemporalContextResourceBudget(
            output_scalars=self._config.output_dim(),
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
            exact_lifetime_counter_bytes=(
                TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES
            ),
        )

    def _require_state_contract(self, state: TemporalContextState) -> None:
        if not isinstance(state, TemporalContextState):
            raise TypeError("state must be a TemporalContextState")
        if (
            state.observation_ema.shape != (self._config.input_dim,)
            or state.observation_ema.dtype != jnp.dtype(jnp.float32)
        ):
            raise ValueError("temporal-context observation_ema contract is invalid")
        _lifetime_counter_valid(state.step_words, state.step_count)

    def state_valid(self, state: TemporalContextState) -> Bool[Array, ""]:
        """Validate exact lifetime identity and finite EMA state."""

        self._require_state_contract(state)
        return _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & jnp.all(jnp.isfinite(state.observation_ema))

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        """Return current causal context features without advancing state."""
        self._require_state_contract(state)
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((cfg.input_dim,))
        blocks = []
        if cfg.include_raw:
            blocks.append(obs)
        if cfg.include_ema:
            blocks.append(state.observation_ema)
        if cfg.include_delta:
            blocks.append(obs - state.observation_ema)
        if cfg.periods:
            phases: list[Array] = []
            for configured_period in cfg.periods:
                period = int(configured_period)
                remainder = _words_mod_positive_int(state.step_words, period)
                angle = (
                    jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32)
                    * remainder.astype(jnp.float32)
                    / jnp.asarray(period, dtype=jnp.float32)
                )
                phases.extend((jnp.sin(angle), jnp.cos(angle)))
            phase = jnp.stack(phases)
            blocks.append(phase)
            if cfg.include_phase_products:
                blocks.append(jnp.ravel(phase[:, None] * obs[None, :]))
        candidate = jnp.concatenate(blocks, axis=0)
        valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(candidate))
        )
        return jnp.where(valid, candidate, jnp.zeros_like(candidate))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_result(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> TemporalContextUpdateResult:
        """Attempt one atomic context update with explicit acceptance."""
        self._require_state_contract(state)
        decay = jnp.asarray(self._config.ema_decay, dtype=jnp.float32)
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape(
            (self._config.input_dim,)
        )
        candidate_ema = decay * state.observation_ema + (1.0 - decay) * obs
        proposed_words, capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = counter_valid & jnp.all(jnp.isfinite(state.observation_ema))
        input_valid = jnp.all(jnp.isfinite(obs))
        candidate_state_finite = jnp.all(jnp.isfinite(candidate_ema))
        update_applied = (
            state_valid
            & input_valid
            & candidate_state_finite
            & capacity_available
        )
        candidate_state = TemporalContextState(
            observation_ema=candidate_ema,
            step_count=_words_to_int32_telemetry(proposed_words),
            step_words=proposed_words,
        )
        next_state = cast(
            TemporalContextState,
            jax.lax.cond(
                update_applied,
                lambda: candidate_state,
                lambda: state,
            ),
        )
        return TemporalContextUpdateResult(
            state=next_state,
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            lifetime_counter_valid=counter_valid,
            lifetime_capacity_available=capacity_available,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_finite=candidate_state_finite,
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> TemporalContextState:
        """Advance context state, preserving the historical state-only API."""

        return cast(
            TemporalContextState,
            self.update_result(state, observation).state,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> TemporalContextStepResult:
        """Return causal features and the explicit atomic update verdict."""

        features = self.features(state, observation)
        update = self.update_result(state, observation)
        return TemporalContextStepResult(
            state=update.state,
            features=features,
            update=update,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> tuple[TemporalContextState, Float[Array, " output_dim"]]:
        """Return features and then advance context state."""
        result = self.step_result(state, observation)
        return result.state, result.features


def transform_temporal_context_arrays(
    featurizer: TemporalContextFeaturizer,
    observations: Float[Array, "steps input_dim"],
    *,
    state: TemporalContextState | None = None,
) -> tuple[TemporalContextState, Float[Array, "steps output_dim"]]:
    """Transform an observation array with a causal scan."""
    if state is None:
        state = featurizer.init()

    def step_fn(
        carry: TemporalContextState,
        observation: Array,
    ) -> tuple[TemporalContextState, Array]:
        return cast(tuple[TemporalContextState, Array], featurizer.step(carry, observation))

    return cast(
        tuple[TemporalContextState, Float[Array, "steps output_dim"]],
        jax.lax.scan(step_fn, state, observations),
    )


def temporal_context_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by telemetry plus exact lifetime words."""

    return TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES


def measure_temporal_context_state_nbytes(state: TemporalContextState) -> int:
    """Measure persistent JAX-array bytes in one context state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_temporal_context_state(legacy_state: Any) -> TemporalContextState:
    """Migrate an unambiguous pre-v2 state with an unsaturated int32 clock."""

    if isinstance(legacy_state, Mapping):
        state_fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        state_fields = {
            state_field.name: getattr(legacy_state, state_field.name)
            for state_field in fields(legacy_state)
        }
    else:
        raise TypeError("legacy temporal-context state must be a mapping or dataclass")
    if set(state_fields) != {"observation_ema", "step_count"}:
        missing = sorted({"observation_ema", "step_count"} - set(state_fields))
        extra = sorted(set(state_fields) - {"observation_ema", "step_count"})
        raise ValueError(
            "legacy temporal-context state field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_count = jnp.asarray(state_fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy temporal-context step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy temporal-context step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy temporal-context step_count is ambiguous")
    return TemporalContextState(
        observation_ema=jnp.asarray(
            state_fields["observation_ema"],
            dtype=jnp.float32,
        ),
        step_count=step_count,
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )


__all__ = [
    "TEMPORAL_CONTEXT_LIFETIME_COUNTER_DELTA_NBYTES",
    "TEMPORAL_CONTEXT_LIFETIME_COUNTER_NBYTES",
    "TEMPORAL_CONTEXT_STATE_SCHEMA",
    "TemporalContextConfig",
    "TemporalContextFeaturizer",
    "TemporalContextResourceBudget",
    "TemporalContextState",
    "TemporalContextStepResult",
    "TemporalContextUpdateResult",
    "measure_temporal_context_state_nbytes",
    "migrate_legacy_temporal_context_state",
    "temporal_context_lifetime_counter_nbytes",
    "transform_temporal_context_arrays",
]
