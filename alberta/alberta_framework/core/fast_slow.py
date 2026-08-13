# mypy: disable-error-code="call-arg,name-defined"
"""Fast/slow additive learner: two readout timescales over one shared encoder.

The prediction decomposes as
``slow(phi(x)) + sigmoid(g(phi(x))) * fast(phi(x))``: a slow readout carries
durable structure while a per-step-decayed fast readout absorbs transients,
in the spirit of fast-weight memory (Hinton & Plaut 1987, "Using Fast Weights
to Deblur Old Memories").  The module distills the exploratory Step 2 runner
(``d18_simple_universal_resource_basis.py`` under ``examples/The Alberta
Plan/Step2/new_directions/``, a tree vendored out of this checkout — see
``VENDORING.md``) down to this core hypothesis, deliberately
omitting that runner's portfolio machinery — kernel banks, Fourier features,
polynomial caps, hand-routed experts.  What remains is one trainable tanh
encoder, one slow readout, one fast decayed readout, and a learned gate, in a
compact scan-compatible API.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

FAST_SLOW_CONFIG_SCHEMA = "alberta.fast-slow.config.v2"
FAST_SLOW_STATE_SCHEMA = "alberta.fast-slow.state.v2"
FAST_SLOW_RESULT_SCHEMA = "alberta.fast-slow.update-result.v2"
FAST_SLOW_RESOURCE_SCHEMA = "alberta.fast-slow.resource-record.v2"

FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES = 8
FAST_SLOW_LIFETIME_COUNTER_NBYTES = 12

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

_FAST_SLOW_CONFIG_FIELDS = {
    "type",
    "input_dim",
    "output_dim",
    "hidden_dim",
    "encoder_step_size",
    "slow_step_size",
    "fast_step_size",
    "gate_step_size",
    "fast_decay",
    "slow_weight_decay",
    "gate_l2",
    "grad_clip",
    "init_scale",
}
_FAST_SLOW_CONFIG_RECORD_FIELDS = _FAST_SLOW_CONFIG_FIELDS | {
    "schema",
    "state_schema",
    "result_schema",
    "resource_schema",
}


@chex.dataclass(frozen=True)
class FastSlowConfig:
    """Configuration for :class:`FastSlowLearner`.

    Args:
        input_dim: Observation dimensionality.
        output_dim: Prediction dimensionality.
        hidden_dim: Width of the learned tanh encoder.
        encoder_step_size: Step-size for the shared learned encoder.
        slow_step_size: Step-size for the slow readout.
        fast_step_size: Step-size for the fast readout.
        gate_step_size: Step-size for the learned fast/slow gate.
        fast_decay: Per-step decay applied to the fast readout before its new
            update.  This is the only fixed timescale in the learner.
        slow_weight_decay: Optional multiplicative decay for slow readout
            weights.  Defaults to no decay.
        gate_l2: L2 shrinkage on gate weights.  Defaults to no shrinkage.
        grad_clip: Global gradient-norm cap.  Non-positive disables clipping.
        init_scale: Encoder initialization scale.
    """

    input_dim: int
    output_dim: int = 1
    hidden_dim: int = 64
    encoder_step_size: float = 1e-3
    slow_step_size: float = 1e-2
    fast_step_size: float = 5e-2
    gate_step_size: float = 1e-2
    fast_decay: float = 0.98
    slow_weight_decay: float = 1.0
    gate_l2: float = 0.0
    grad_clip: float = 10.0
    init_scale: float = 1.0

    def to_config(self) -> dict[str, Any]:
        """Serialize to a strict versioned plain-data record."""
        return {
            "type": "FastSlowConfig",
            "schema": FAST_SLOW_CONFIG_SCHEMA,
            "state_schema": FAST_SLOW_STATE_SCHEMA,
            "result_schema": FAST_SLOW_RESULT_SCHEMA,
            "resource_schema": FAST_SLOW_RESOURCE_SCHEMA,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "encoder_step_size": self.encoder_step_size,
            "slow_step_size": self.slow_step_size,
            "fast_step_size": self.fast_step_size,
            "gate_step_size": self.gate_step_size,
            "fast_decay": self.fast_decay,
            "slow_weight_decay": self.slow_weight_decay,
            "gate_l2": self.gate_l2,
            "grad_clip": self.grad_clip,
            "init_scale": self.init_scale,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FastSlowConfig:
        """Reconstruct current records or the exact historical record.

        The schema-free record emitted before v2 remains accepted deliberately
        so persisted experimental configurations keep their round-trip.  It is
        accepted only at its exact historical field set; partial and extended
        mappings fail closed.
        """

        if not isinstance(config, Mapping):
            raise TypeError("fast/slow config must be a mapping")
        payload = dict(config)
        fields = set(payload)
        if fields == _FAST_SLOW_CONFIG_RECORD_FIELDS:
            schemas = {
                "type": "FastSlowConfig",
                "schema": FAST_SLOW_CONFIG_SCHEMA,
                "state_schema": FAST_SLOW_STATE_SCHEMA,
                "result_schema": FAST_SLOW_RESULT_SCHEMA,
                "resource_schema": FAST_SLOW_RESOURCE_SCHEMA,
            }
            for name, expected in schemas.items():
                if payload.pop(name) != expected:
                    raise ValueError(f"fast/slow {name} schema value is unsupported")
        elif fields == _FAST_SLOW_CONFIG_FIELDS:
            if payload.pop("type") != "FastSlowConfig":
                raise ValueError("fast/slow config type is unsupported")
        else:
            missing = sorted(_FAST_SLOW_CONFIG_RECORD_FIELDS - fields)
            extra = sorted(fields - _FAST_SLOW_CONFIG_RECORD_FIELDS)
            raise ValueError(
                "fast/slow config fields are invalid; "
                f"missing={missing}, extra={extra}"
            )
        restored = cls(**payload)
        _validate_config(restored)
        return restored


@chex.dataclass(frozen=True)
class FastSlowParams:
    """Trainable parameters for the fast/slow learner."""

    encoder_kernel: Float[Array, "input_dim hidden_dim"]
    encoder_bias: Float[Array, " hidden_dim"]
    slow_kernel: Float[Array, "hidden_dim output_dim"]
    slow_bias: Float[Array, " output_dim"]
    fast_kernel: Float[Array, "hidden_dim output_dim"]
    fast_bias: Float[Array, " output_dim"]
    gate_kernel: Float[Array, "hidden_dim output_dim"]
    gate_bias: Float[Array, " output_dim"]


@chex.dataclass(frozen=True)
class FastSlowState:
    """State for :class:`FastSlowLearner`."""

    params: FastSlowParams
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class FastSlowPredictionParts:
    """Structured forward-pass outputs for diagnostics and updates."""

    prediction: Float[Array, " output_dim"]
    slow_prediction: Float[Array, " output_dim"]
    fast_prediction: Float[Array, " output_dim"]
    gate: Float[Array, " output_dim"]
    features: Float[Array, " hidden_dim"]


@chex.dataclass(frozen=True)
class FastSlowUpdateResult:
    """One staged online transaction plus explicit commit diagnostics."""

    state: FastSlowState
    prediction: Float[Array, " output_dim"]
    error: Float[Array, " output_dim"]
    metrics: Float[Array, " metrics"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    observation_valid: Bool[Array, ""]
    target_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class FastSlowLearningResult:
    """Result of running :func:`run_fast_slow_arrays`."""

    state: FastSlowState
    metrics: Float[Array, "steps metrics"]
    pre_step_words: UInt[Array, "steps 2"]
    post_step_words: UInt[Array, "steps 2"]
    lifetime_counter_valid: Bool[Array, " steps"]
    lifetime_capacity_available: Bool[Array, " steps"]
    state_valid: Bool[Array, " steps"]
    observation_valid: Bool[Array, " steps"]
    target_valid: Bool[Array, " steps"]
    input_valid: Bool[Array, " steps"]
    candidate_state_valid: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class FastSlowStateRecord:
    """Versioned host record describing one persistent learner state."""

    schema: str
    config_schema: str
    parameter_dtype: str
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_nbytes: int
    state_nbytes: int
    step_count: int
    step_words: tuple[int, int]
    state_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data state record."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FastSlowResourceRecord:
    """Exact persistent-state accounting for one configured learner."""

    schema: str
    config_schema: str
    state_schema: str
    result_schema: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    trainable_float32_scalars: int
    parameter_nbytes: int
    exact_lifetime_identity_nbytes: int
    lifetime_counter_nbytes: int
    legacy_state_nbytes: int
    versioned_state_delta_nbytes: int
    state_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    maximum_updates_per_call: int
    replay_capacity: int
    persistent_capacity_growth: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data resource record."""

        return dataclasses.asdict(self)


def _require_config_dimension(value: Any, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [1, {_INT32_MAX}]")


def _require_config_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    parsed = float(value)
    with np.errstate(over="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must be finite in float32")
    if minimum is not None:
        if minimum_inclusive and narrowed < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if not minimum_inclusive and narrowed <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _validate_config(config: FastSlowConfig) -> None:
    if not isinstance(config, FastSlowConfig):
        raise TypeError("config must be a FastSlowConfig")
    _require_config_dimension(config.input_dim, name="input_dim")
    _require_config_dimension(config.output_dim, name="output_dim")
    _require_config_dimension(config.hidden_dim, name="hidden_dim")
    for name in (
        "encoder_step_size",
        "slow_step_size",
        "fast_step_size",
        "gate_step_size",
        "gate_l2",
    ):
        _require_config_float(getattr(config, name), name=name, minimum=0.0)
    _require_config_float(
        config.fast_decay,
        name="fast_decay",
        minimum=0.0,
        maximum=1.0,
    )
    _require_config_float(
        config.slow_weight_decay,
        name="slow_weight_decay",
        minimum=0.0,
        maximum=1.0,
    )
    _require_config_float(config.grad_clip, name="grad_clip")
    _require_config_float(
        config.init_scale,
        name="init_scale",
        minimum=0.0,
        minimum_inclusive=False,
    )


def _require_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected_dtype = jnp.dtype(dtype)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose a big-endian uint32-word successor without wrapping all-ones."""

    _require_array_contract(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_to_saturating_int32(words: Array) -> Int[Array, ""]:
    _require_array_contract(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    _require_array_contract(telemetry, name="step_count", shape=(), dtype=jnp.int32)
    return (telemetry >= 0) & (telemetry == _words_to_saturating_int32(words))


def _saturating_int32_increment(value: Array) -> Int[Array, ""]:
    _require_array_contract(value, name="step_count", shape=(), dtype=jnp.int32)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(value, maximum - jnp.asarray(1, dtype=jnp.int32)) + jnp.asarray(
        1,
        dtype=jnp.int32,
    )


def _floating_tree_is_finite(tree: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _linear_init(key: Array, fan_in: int, fan_out: int, scale: float) -> Array:
    std = scale / jnp.sqrt(jnp.asarray(max(fan_in, 1), dtype=jnp.float32))
    return std * jr.normal(key, (fan_in, fan_out), dtype=jnp.float32)


def init_fast_slow_params(key: Array, config: FastSlowConfig) -> FastSlowParams:
    """Initialize fast/slow learner parameters.

    The encoder starts random but is immediately trainable.  Readouts start at
    zero so the first predictions are neutral and all structure is acquired by
    online updates.
    """
    _validate_config(config)
    encoder_key, gate_key = jr.split(key)
    return FastSlowParams(
        encoder_kernel=_linear_init(
            encoder_key,
            config.input_dim,
            config.hidden_dim,
            config.init_scale,
        ),
        encoder_bias=jnp.zeros(config.hidden_dim, dtype=jnp.float32),
        slow_kernel=jnp.zeros(
            (config.hidden_dim, config.output_dim),
            dtype=jnp.float32,
        ),
        slow_bias=jnp.zeros(config.output_dim, dtype=jnp.float32),
        fast_kernel=jnp.zeros(
            (config.hidden_dim, config.output_dim),
            dtype=jnp.float32,
        ),
        fast_bias=jnp.zeros(config.output_dim, dtype=jnp.float32),
        # Scale 0.01 keeps initial gate logits near zero: the gate starts
        # nearly input-independent at ~0.5, so gating structure is learned
        # rather than imposed by the random init.
        gate_kernel=_linear_init(gate_key, config.hidden_dim, config.output_dim, 0.01),
        gate_bias=jnp.zeros(config.output_dim, dtype=jnp.float32),
    )


def fast_slow_forward(
    params: FastSlowParams,
    observation: Float[Array, " input_dim"],
) -> FastSlowPredictionParts:
    """Run a single-example fast/slow forward pass."""
    features = jnp.tanh(observation @ params.encoder_kernel + params.encoder_bias)
    slow_prediction = features @ params.slow_kernel + params.slow_bias
    fast_prediction = features @ params.fast_kernel + params.fast_bias
    gate = jax.nn.sigmoid(features @ params.gate_kernel + params.gate_bias)
    prediction = slow_prediction + gate * fast_prediction
    return FastSlowPredictionParts(
        prediction=prediction,
        slow_prediction=slow_prediction,
        fast_prediction=fast_prediction,
        gate=gate,
        features=features,
    )


def _loss_and_parts(
    params: FastSlowParams,
    observation: Array,
    target: Array,
) -> tuple[Array, FastSlowPredictionParts]:
    parts = fast_slow_forward(params, observation)
    shaped_target = jnp.reshape(target, parts.prediction.shape)
    error = parts.prediction - shaped_target
    loss = 0.5 * jnp.mean(error**2)
    return loss, parts


def _tree_global_norm(tree: object) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))


def _clip_grads(grads: FastSlowParams, clip: float) -> tuple[FastSlowParams, Array]:
    norm = _tree_global_norm(grads)
    if clip <= 0.0:
        return grads, norm
    scale = jnp.minimum(1.0, jnp.asarray(clip, dtype=jnp.float32) / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: scale * g, grads), norm


class FastSlowLearner:
    """JAX-native fast/slow additive learner.

    Prediction is:

    ``slow(phi(x)) + sigmoid(g(phi(x))) * fast(phi(x))``

    The shared encoder, slow readout, fast readout, and gate all update every
    time step.  The fast readout is decayed before applying its update, giving
    the learner a short-memory path without an external router.
    """

    def __init__(self, config: FastSlowConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> FastSlowConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict versioned learner record."""
        return {
            "type": "FastSlowLearner",
            "schema": FAST_SLOW_CONFIG_SCHEMA,
            "state_schema": FAST_SLOW_STATE_SCHEMA,
            "result_schema": FAST_SLOW_RESULT_SCHEMA,
            "resource_schema": FAST_SLOW_RESOURCE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FastSlowLearner:
        """Reconstruct a current record or the exact historical wrapper."""

        if not isinstance(config, Mapping):
            raise TypeError("fast/slow learner config must be a mapping")
        payload = dict(config)
        legacy_fields = {"type", "config"}
        current_fields = legacy_fields | {
            "schema",
            "state_schema",
            "result_schema",
            "resource_schema",
        }
        if set(payload) == current_fields:
            schemas = {
                "type": "FastSlowLearner",
                "schema": FAST_SLOW_CONFIG_SCHEMA,
                "state_schema": FAST_SLOW_STATE_SCHEMA,
                "result_schema": FAST_SLOW_RESULT_SCHEMA,
                "resource_schema": FAST_SLOW_RESOURCE_SCHEMA,
            }
            for name, expected in schemas.items():
                if payload.pop(name) != expected:
                    raise ValueError(f"fast/slow learner {name} is unsupported")
        elif set(payload) == legacy_fields:
            if payload.pop("type") != "FastSlowLearner":
                raise ValueError("fast/slow learner type is unsupported")
        else:
            raise ValueError("fast/slow learner config fields are invalid")
        inner = payload.pop("config")
        if not isinstance(inner, Mapping):
            raise TypeError("fast/slow learner inner config must be a mapping")
        return cls(FastSlowConfig.from_config(inner))

    def _require_state_contract(self, state: FastSlowState) -> None:
        if not isinstance(state, FastSlowState):
            raise TypeError("state must be a FastSlowState")
        if not isinstance(state.params, FastSlowParams):
            raise TypeError("state.params must be FastSlowParams")
        c = self._config
        contracts = (
            ("encoder_kernel", (c.input_dim, c.hidden_dim)),
            ("encoder_bias", (c.hidden_dim,)),
            ("slow_kernel", (c.hidden_dim, c.output_dim)),
            ("slow_bias", (c.output_dim,)),
            ("fast_kernel", (c.hidden_dim, c.output_dim)),
            ("fast_bias", (c.output_dim,)),
            ("gate_kernel", (c.hidden_dim, c.output_dim)),
            ("gate_bias", (c.output_dim,)),
        )
        for name, shape in contracts:
            _require_array_contract(
                getattr(state.params, name),
                name=f"state.params.{name}",
                shape=shape,
                dtype=jnp.float32,
            )
        _require_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array_contract(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def state_is_valid(self, state: FastSlowState) -> Bool[Array, ""]:
        """Return dynamic validity after enforcing the static state schema."""

        self._require_state_contract(state)
        return _floating_tree_is_finite(state.params) & _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )

    def state_record(self, state: FastSlowState) -> FastSlowStateRecord:
        """Build a strict host-only record for one persistent state."""

        self._require_state_contract(state)
        parameter_shapes = tuple(
            (name, tuple(int(dimension) for dimension in getattr(state.params, name).shape))
            for name in (
                "encoder_kernel",
                "encoder_bias",
                "slow_kernel",
                "slow_bias",
                "fast_kernel",
                "fast_bias",
                "gate_kernel",
                "gate_bias",
            )
        )
        parameter_nbytes = _tree_nbytes(state.params)
        step_words = tuple(
            int(value)
            for value in np.asarray(jax.device_get(state.step_words)).tolist()
        )
        if len(step_words) != 2:
            raise AssertionError("validated fast/slow step_words lost its shape")
        return FastSlowStateRecord(
            schema=FAST_SLOW_STATE_SCHEMA,
            config_schema=FAST_SLOW_CONFIG_SCHEMA,
            parameter_dtype="float32",
            parameter_shapes=parameter_shapes,
            parameter_nbytes=parameter_nbytes,
            state_nbytes=measure_fast_slow_state_nbytes(state),
            step_count=int(jax.device_get(state.step_count)),
            step_words=step_words,
            state_valid=bool(jax.device_get(self.state_is_valid(state))),
        )

    def resource_record(
        self,
        state: FastSlowState | None = None,
    ) -> FastSlowResourceRecord:
        """Return exact persistent bytes and fixed per-call capacity."""

        measured_state = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(measured_state)
        parameter_nbytes = _tree_nbytes(measured_state.params)
        state_nbytes = measure_fast_slow_state_nbytes(measured_state)
        return FastSlowResourceRecord(
            schema=FAST_SLOW_RESOURCE_SCHEMA,
            config_schema=FAST_SLOW_CONFIG_SCHEMA,
            state_schema=FAST_SLOW_STATE_SCHEMA,
            result_schema=FAST_SLOW_RESULT_SCHEMA,
            input_dim=self._config.input_dim,
            output_dim=self._config.output_dim,
            hidden_dim=self._config.hidden_dim,
            trainable_float32_scalars=parameter_nbytes // np.dtype(np.float32).itemsize,
            parameter_nbytes=parameter_nbytes,
            exact_lifetime_identity_nbytes=FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES,
            lifetime_counter_nbytes=FAST_SLOW_LIFETIME_COUNTER_NBYTES,
            legacy_state_nbytes=state_nbytes - FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES,
            versioned_state_delta_nbytes=FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES,
            state_nbytes=state_nbytes,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            maximum_updates_per_call=1,
            replay_capacity=0,
            persistent_capacity_growth=0,
        )

    def init(self, key: Array) -> FastSlowState:
        """Create an initial learner state."""
        return FastSlowState(
            params=init_fast_slow_params(key, self._config),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_parts(
        self,
        state: FastSlowState,
        observation: Float[Array, " input_dim"],
    ) -> FastSlowPredictionParts:
        """Return prediction plus fast/slow/gate diagnostics."""
        self._require_state_contract(state)
        _require_array_contract(
            observation,
            name="observation",
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        return fast_slow_forward(state.params, observation)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: FastSlowState,
        observation: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        """Return the current prediction for one observation."""
        return cast(Array, self.predict_parts(state, observation).prediction)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: FastSlowState,
        observation: Float[Array, " input_dim"],
        target: Float[Array, " output_dim"],
    ) -> FastSlowUpdateResult:
        """Stage one causal update and atomically commit only a valid candidate."""

        self._require_state_contract(state)
        observation_array = _require_array_contract(
            observation,
            name="observation",
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        target_array = _require_array_contract(
            target,
            name="target",
            shape=(self._config.output_dim,),
            dtype=jnp.float32,
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = _floating_tree_is_finite(state.params) & lifetime_counter_valid
        observation_valid = jnp.all(jnp.isfinite(observation_array))
        target_valid = jnp.all(jnp.isfinite(target_array))
        input_valid = observation_valid & target_valid
        safe_observation = jnp.where(
            observation_valid,
            observation_array,
            jnp.zeros_like(observation_array),
        )
        safe_target = jnp.where(
            target_valid,
            target_array,
            jnp.zeros_like(target_array),
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        (loss, parts), grads = jax.value_and_grad(_loss_and_parts, has_aux=True)(
            state.params,
            safe_observation,
            safe_target,
        )
        clipped_grads, grad_norm = _clip_grads(grads, self._config.grad_clip)
        c = self._config

        candidate_params = FastSlowParams(
            encoder_kernel=state.params.encoder_kernel
            - c.encoder_step_size * clipped_grads.encoder_kernel,
            encoder_bias=state.params.encoder_bias
            - c.encoder_step_size * clipped_grads.encoder_bias,
            slow_kernel=c.slow_weight_decay * state.params.slow_kernel
            - c.slow_step_size * clipped_grads.slow_kernel,
            slow_bias=state.params.slow_bias - c.slow_step_size * clipped_grads.slow_bias,
            fast_kernel=c.fast_decay * state.params.fast_kernel
            - c.fast_step_size * clipped_grads.fast_kernel,
            fast_bias=c.fast_decay * state.params.fast_bias
            - c.fast_step_size * clipped_grads.fast_bias,
            gate_kernel=(1.0 - c.gate_step_size * c.gate_l2) * state.params.gate_kernel
            - c.gate_step_size * clipped_grads.gate_kernel,
            gate_bias=state.params.gate_bias - c.gate_step_size * clipped_grads.gate_bias,
        )
        candidate_state = FastSlowState(
            params=candidate_params,
            step_count=_saturating_int32_increment(state.step_count),
            step_words=proposed_words,
        )
        shaped_target = jnp.reshape(safe_target, parts.prediction.shape)
        error = shaped_target - parts.prediction
        candidate_metrics = jnp.asarray(
            [
                loss,
                jnp.mean(error**2),
                jnp.mean(parts.gate),
                grad_norm,
                _tree_global_norm(candidate_params.fast_kernel),
                _tree_global_norm(candidate_params.slow_kernel),
            ],
            dtype=jnp.float32,
        )
        candidate_state_valid = _floating_tree_is_finite(
            (
                candidate_state.params,
                parts,
                grads,
                clipped_grads,
                loss,
                error,
                candidate_metrics,
            )
        ) & _lifetime_counter_valid(
            candidate_state.step_words,
            candidate_state.step_count,
        )
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        prediction = jnp.where(
            update_applied,
            parts.prediction,
            jnp.zeros_like(parts.prediction),
        )
        committed_error = jnp.where(update_applied, error, jnp.zeros_like(error))
        metrics = jnp.where(
            update_applied,
            candidate_metrics,
            jnp.zeros_like(candidate_metrics),
        )
        return FastSlowUpdateResult(
            state=new_state,
            prediction=prediction,
            error=committed_error,
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            observation_valid=observation_valid,
            target_valid=target_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


def run_fast_slow_arrays(
    learner: FastSlowLearner,
    observations: Float[Array, "steps input_dim"],
    targets: Float[Array, "steps output_dim"],
    *,
    state: FastSlowState | None = None,
    key: Array | None = None,
) -> FastSlowLearningResult:
    """Run the learner over arrays with ``jax.lax.scan``.

    Args:
        learner: Fast/slow learner instance.
        observations: Observation array with leading time dimension.
        targets: Target array with matching leading time dimension.
        state: Optional initial state.
        key: Initialization key, required when ``state`` is not supplied.

    Returns:
        Final state and per-step metrics with columns:
        ``loss, mse, mean_gate, grad_norm, fast_norm, slow_norm``.
    """
    observation_array = jnp.asarray(observations)
    if observation_array.ndim != 2 or observation_array.shape[1:] != (
        learner.config.input_dim,
    ):
        raise ValueError(
            "observations must have shape "
            f"(steps, {learner.config.input_dim}), got {observation_array.shape}"
        )
    if observation_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(
            f"observations must have dtype float32, got {observation_array.dtype}"
        )
    target_array = jnp.asarray(targets)
    expected_target_shape = (observation_array.shape[0], learner.config.output_dim)
    if target_array.shape != expected_target_shape:
        raise ValueError(
            f"targets must have shape {expected_target_shape}, got {target_array.shape}"
        )
    if target_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"targets must have dtype float32, got {target_array.dtype}")
    if state is None:
        if key is None:
            raise ValueError("key is required when state is not supplied")
        state = learner.init(key)
    learner._require_state_contract(state)

    def step_fn(
        carry: FastSlowState,
        batch: tuple[Array, Array],
    ) -> tuple[FastSlowState, tuple[Array, ...]]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, (
            result.metrics,
            result.pre_step_words,
            result.post_step_words,
            result.lifetime_counter_valid,
            result.lifetime_capacity_available,
            result.state_valid,
            result.observation_valid,
            result.target_valid,
            result.input_valid,
            result.candidate_state_valid,
            result.update_applied,
        )

    final_state, scan_outputs = jax.lax.scan(
        step_fn,
        state,
        (observation_array, target_array),
    )
    (
        metrics,
        pre_step_words,
        post_step_words,
        lifetime_counter_valid,
        lifetime_capacity_available,
        state_valid,
        observation_valid,
        target_valid,
        input_valid,
        candidate_state_valid,
        update_applied,
    ) = scan_outputs
    return FastSlowLearningResult(
        state=final_state,
        metrics=metrics,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
        lifetime_counter_valid=lifetime_counter_valid,
        lifetime_capacity_available=lifetime_capacity_available,
        state_valid=state_valid,
        observation_valid=observation_valid,
        target_valid=target_valid,
        input_valid=input_valid,
        candidate_state_valid=candidate_state_valid,
        update_applied=update_applied,
    )


def _tree_nbytes(tree: Any) -> int:
    """Measure all persistent array leaves without counting Python objects."""

    return sum(
        int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def measure_fast_slow_state_nbytes(state: FastSlowState) -> int:
    """Measure every persistent array byte in a fast/slow state."""

    if not isinstance(state, FastSlowState):
        raise TypeError("state must be a FastSlowState")
    return _tree_nbytes(state)


def migrate_legacy_fast_slow_state(
    learner: FastSlowLearner,
    state: Mapping[str, Any] | Any,
) -> FastSlowState:
    """Attach exact words to an unambiguous pre-v2 state on the host only.

    Legacy ``int32`` saturation destroys lifetime identity, as do negative
    values produced by wrapped counters.  Neither case is guessed here.
    """

    if not isinstance(learner, FastSlowLearner):
        raise TypeError("learner must be a FastSlowLearner")
    if isinstance(state, Mapping):
        payload = dict(state)
    elif dataclasses.is_dataclass(state) and not isinstance(state, type):
        payload = {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    else:
        raise TypeError("legacy fast/slow state must be a mapping or dataclass")
    expected = {"params", "step_count"}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            "legacy fast/slow state fields are invalid; "
            f"missing={missing}, extra={extra}"
        )
    counter = np.asarray(jax.device_get(payload["step_count"]))
    if counter.shape != () or counter.dtype != np.dtype(np.int32):
        raise TypeError("legacy fast/slow step_count must be a scalar int32")
    count = int(counter)
    if not 0 <= count < _INT32_MAX:
        raise ValueError("legacy fast/slow step_count is ambiguous at the int32 boundary")
    migrated = FastSlowState(
        params=payload["params"],
        step_count=jnp.asarray(count, dtype=jnp.int32),
        step_words=jnp.asarray((0, count), dtype=jnp.uint32),
    )
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy fast/slow state is dynamically invalid")
    return migrated
