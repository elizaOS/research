# mypy: disable-error-code="call-arg,name-defined"
"""Alberta-derived online permanent/transient regression baseline.

This module is deliberately **not** a source-faithful implementation of
Anand & Precup, "Permanent and Transient Representations for Continual
Reinforcement Learning" (ICLR 2026 submission, CC BY 4.0).  The paper's
small-scale algorithms use known task endings, a task buffer, and transient
resets.  Its online Craftax algorithm combines a neural permanent value model,
a specialized MinHash/slot transient memory, periodic buffered consolidation,
and periodic transient decay.  No public implementation of that 2026 method
was found during the design audit.  The authors' earlier NeurIPS 2023 code is
MIT licensed and confirms the buffered consolidation/decay ordering, but it
does not implement the paper's distinct representation scheme.

The bounded baseline here keeps the core causal idea while making departures
needed for one never-resetting scalar regression stream:

* permanent and transient predictions have independent trainable tanh
  encoders and independent heads;
* the transient system learns the current residual while treating the
  permanent prediction as fixed;
* the permanent system distils the post-transient combined prediction while
  treating that target as fixed;
* consolidation occurs on the current sample every step (``k=1``), with no
  replay buffer, task ID, boundary, reset, or growing memory;
* transient decay applies to its head after consolidation, not periodically to
  every transient network parameter; and
* both systems use plain clipped SGD rather than the paper's experiment-
  specific optimizers and architectures.

The source/departure record is available programmatically through
:func:`permanent_transient_design_record`.  These choices make this an honest
mechanistic Alberta baseline, not a reproduction or efficacy claim.
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

PERMANENT_TRANSIENT_DESIGN_SCHEMA = "alberta.permanent-transient.design.v1"
PERMANENT_TRANSIENT_CONFIG_SCHEMA = "alberta.permanent-transient.config.v1"
PERMANENT_TRANSIENT_STATE_SCHEMA = "alberta.permanent-transient.state.v1"
PERMANENT_TRANSIENT_RESULT_SCHEMA = "alberta.permanent-transient.update-result.v1"
PERMANENT_TRANSIENT_RESOURCE_SCHEMA = "alberta.permanent-transient.resource-record.v1"

PERMANENT_TRANSIENT_EXACT_LIFETIME_NBYTES = 8
PERMANENT_TRANSIENT_LIFETIME_COUNTER_NBYTES = 12

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

_CONFIG_VALUE_FIELDS = {
    "input_dim",
    "output_dim",
    "permanent_hidden_dim",
    "transient_hidden_dim",
    "permanent_encoder_step_size",
    "permanent_head_step_size",
    "transient_encoder_step_size",
    "transient_head_step_size",
    "transient_decay",
    "grad_clip",
    "init_scale",
}
_CONFIG_FIELDS = _CONFIG_VALUE_FIELDS | {
    "type",
    "schema",
    "design_schema",
    "state_schema",
    "result_schema",
    "resource_schema",
}
_PARAMETER_NAMES = (
    "permanent_encoder_kernel",
    "permanent_encoder_bias",
    "permanent_head_kernel",
    "permanent_head_bias",
    "transient_encoder_kernel",
    "transient_encoder_bias",
    "transient_head_kernel",
    "transient_head_bias",
)


@dataclasses.dataclass(frozen=True)
class PermanentTransientDesignRecord:
    """Machine-readable attribution and departure boundary."""

    schema: str
    method_name: str
    source_faithful: bool
    primary_paper_title: str
    primary_paper_url: str
    primary_paper_license: str
    reference_code_url: str
    reference_code_license: str
    public_2026_source_located: bool
    departures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data design record."""

        return dataclasses.asdict(self)


def permanent_transient_design_record() -> PermanentTransientDesignRecord:
    """Return the fixed source and non-parity declaration for this baseline."""

    return PermanentTransientDesignRecord(
        schema=PERMANENT_TRANSIENT_DESIGN_SCHEMA,
        method_name="Alberta-derived online permanent/transient regression",
        source_faithful=False,
        primary_paper_title=(
            "Permanent and Transient Representations for Continual Reinforcement Learning"
        ),
        primary_paper_url="https://openreview.net/forum?id=5XfxEQ2SCt",
        primary_paper_license="CC BY 4.0",
        reference_code_url=(
            "https://github.com/NishanthVAnand/"
            "prediction-and-control-in-continual-reinforcement-learning"
        ),
        reference_code_license="MIT",
        public_2026_source_located=False,
        departures=(
            "supervised immediate-target regression replaces TD or Q-learning",
            "two learned tanh encoders replace fixed Fourier partitions and Craftax hashing",
            "no task identity, known task ending, boundary-triggered reset, or episode reset",
            "current-sample k=1 consolidation replaces task and periodic replay buffers",
            "the permanent target uses the current post-transient combined prediction",
            "transient decay is per-step and head-only rather than periodic whole-model decay",
            "plain globally clipped SGD replaces experiment-specific source optimizers",
        ),
    )


@chex.dataclass(frozen=True)
class AlbertaPermanentTransientConfig:
    """Strict configuration for the Alberta-derived two-system learner."""

    input_dim: int
    output_dim: int = 1
    permanent_hidden_dim: int = 32
    transient_hidden_dim: int = 32
    permanent_encoder_step_size: float = 1e-3
    permanent_head_step_size: float = 1e-2
    transient_encoder_step_size: float = 1e-3
    transient_head_step_size: float = 5e-2
    transient_decay: float = 0.98
    grad_clip: float = 10.0
    init_scale: float = 1.0

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact versioned config record."""

        return {
            "type": "AlbertaPermanentTransientConfig",
            "schema": PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            "design_schema": PERMANENT_TRANSIENT_DESIGN_SCHEMA,
            "state_schema": PERMANENT_TRANSIENT_STATE_SCHEMA,
            "result_schema": PERMANENT_TRANSIENT_RESULT_SCHEMA,
            "resource_schema": PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "permanent_hidden_dim": self.permanent_hidden_dim,
            "transient_hidden_dim": self.transient_hidden_dim,
            "permanent_encoder_step_size": self.permanent_encoder_step_size,
            "permanent_head_step_size": self.permanent_head_step_size,
            "transient_encoder_step_size": self.transient_encoder_step_size,
            "transient_head_step_size": self.transient_head_step_size,
            "transient_decay": self.transient_decay,
            "grad_clip": self.grad_clip,
            "init_scale": self.init_scale,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> AlbertaPermanentTransientConfig:
        """Reconstruct only the exact current config schema."""

        if not isinstance(config, Mapping):
            raise TypeError("permanent/transient config must be a mapping")
        payload = dict(config)
        if set(payload) != _CONFIG_FIELDS:
            missing = sorted(_CONFIG_FIELDS - set(payload))
            extra = sorted(set(payload) - _CONFIG_FIELDS)
            raise ValueError(
                "permanent/transient config fields are invalid; "
                f"missing={missing}, extra={extra}"
            )
        schemas = {
            "type": "AlbertaPermanentTransientConfig",
            "schema": PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            "design_schema": PERMANENT_TRANSIENT_DESIGN_SCHEMA,
            "state_schema": PERMANENT_TRANSIENT_STATE_SCHEMA,
            "result_schema": PERMANENT_TRANSIENT_RESULT_SCHEMA,
            "resource_schema": PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
        }
        for name, expected in schemas.items():
            if payload.pop(name) != expected:
                raise ValueError(f"permanent/transient {name} schema value is unsupported")
        restored = cls(**payload)
        _validate_config(restored)
        return restored


@chex.dataclass(frozen=True)
class AlbertaPermanentTransientParams:
    """Independent permanent and transient representation/head parameters."""

    permanent_encoder_kernel: Float[Array, "input_dim permanent_hidden_dim"]
    permanent_encoder_bias: Float[Array, " permanent_hidden_dim"]
    permanent_head_kernel: Float[Array, "permanent_hidden_dim output_dim"]
    permanent_head_bias: Float[Array, " output_dim"]
    transient_encoder_kernel: Float[Array, "input_dim transient_hidden_dim"]
    transient_encoder_bias: Float[Array, " transient_hidden_dim"]
    transient_head_kernel: Float[Array, "transient_hidden_dim output_dim"]
    transient_head_bias: Float[Array, " output_dim"]


@chex.dataclass(frozen=True)
class AlbertaPermanentTransientState:
    """Fixed-shape learner state with an exact lifetime identity."""

    params: AlbertaPermanentTransientParams
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AlbertaPermanentTransientPrediction:
    """Decomposed prediction from truly separate representations."""

    prediction: Float[Array, " output_dim"]
    permanent_prediction: Float[Array, " output_dim"]
    transient_prediction: Float[Array, " output_dim"]
    permanent_features: Float[Array, " permanent_hidden_dim"]
    transient_features: Float[Array, " transient_hidden_dim"]


@chex.dataclass(frozen=True)
class AlbertaPermanentTransientUpdateResult:
    """One atomic update and its fail-closed diagnostics."""

    state: AlbertaPermanentTransientState
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
class AlbertaPermanentTransientLearningResult:
    """Scan result preserving fixed-width metrics and commit facts."""

    state: AlbertaPermanentTransientState
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
class PermanentTransientStateRecord:
    """Strict host description of one persistent state."""

    schema: str
    config_schema: str
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    permanent_parameter_nbytes: int
    transient_parameter_nbytes: int
    parameter_nbytes: int
    state_nbytes: int
    step_count: int
    step_words: tuple[int, int]
    state_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data state record."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PermanentTransientResourceRecord:
    """Exact fixed memory and maximum per-update work declaration."""

    schema: str
    design_schema: str
    config_schema: str
    state_schema: str
    result_schema: str
    input_dim: int
    output_dim: int
    permanent_hidden_dim: int
    transient_hidden_dim: int
    total_hidden_features: int
    permanent_parameter_nbytes: int
    transient_parameter_nbytes: int
    parameter_nbytes: int
    exact_lifetime_identity_nbytes: int
    lifetime_counter_nbytes: int
    state_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    maximum_gradient_evaluations_per_update: int
    maximum_forward_evaluations_per_update: int
    maximum_updates_per_call: int
    replay_capacity: int
    maximum_stored_examples: int
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


def _validate_config(config: AlbertaPermanentTransientConfig) -> None:
    if not isinstance(config, AlbertaPermanentTransientConfig):
        raise TypeError("config must be AlbertaPermanentTransientConfig")
    for name in (
        "input_dim",
        "output_dim",
        "permanent_hidden_dim",
        "transient_hidden_dim",
    ):
        _require_config_dimension(getattr(config, name), name=name)
    for name in (
        "permanent_encoder_step_size",
        "permanent_head_step_size",
        "transient_encoder_step_size",
        "transient_head_step_size",
    ):
        _require_config_float(getattr(config, name), name=name, minimum=0.0)
    _require_config_float(
        config.transient_decay,
        name="transient_decay",
        minimum=0.0,
        maximum=1.0,
    )
    _require_config_float(config.grad_clip, name="grad_clip", minimum=0.0)
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


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    _require_array_contract(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity, proposed, words), capacity


def _words_to_telemetry(words: Array) -> Array:
    _require_array_contract(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    below = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _counter_valid(words: Array, telemetry: Array) -> Array:
    _require_array_contract(telemetry, name="step_count", shape=(), dtype=jnp.int32)
    return (telemetry >= 0) & (telemetry == _words_to_telemetry(words))


def _saturating_increment(value: Array) -> Array:
    _require_array_contract(value, name="step_count", shape=(), dtype=jnp.int32)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(value, maximum - jnp.asarray(1, dtype=jnp.int32)) + jnp.asarray(
        1,
        dtype=jnp.int32,
    )


def _tree_is_finite(tree: Any) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _tree_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.asarray(leaf) ** 2) for leaf in leaves))


def _clip_grads(
    grads: AlbertaPermanentTransientParams,
    clip: float,
) -> tuple[AlbertaPermanentTransientParams, Array]:
    norm = _tree_norm(grads)
    if clip <= 0.0:
        return grads, norm
    scale = jnp.minimum(
        1.0,
        jnp.asarray(clip, dtype=jnp.float32) / (norm + 1e-8),
    )
    return jax.tree_util.tree_map(lambda gradient: scale * gradient, grads), norm


def _linear_init(key: Array, fan_in: int, fan_out: int, scale: float) -> Array:
    std = scale / jnp.sqrt(jnp.asarray(max(fan_in, 1), dtype=jnp.float32))
    return std * jr.normal(key, (fan_in, fan_out), dtype=jnp.float32)


def init_permanent_transient_params(
    key: Array,
    config: AlbertaPermanentTransientConfig,
) -> AlbertaPermanentTransientParams:
    """Initialize independent encoders and zero prediction heads."""

    _validate_config(config)
    permanent_key, transient_key = jr.split(key)
    return AlbertaPermanentTransientParams(
        permanent_encoder_kernel=_linear_init(
            permanent_key,
            config.input_dim,
            config.permanent_hidden_dim,
            config.init_scale,
        ),
        permanent_encoder_bias=jnp.zeros(
            (config.permanent_hidden_dim,),
            dtype=jnp.float32,
        ),
        permanent_head_kernel=jnp.zeros(
            (config.permanent_hidden_dim, config.output_dim),
            dtype=jnp.float32,
        ),
        permanent_head_bias=jnp.zeros((config.output_dim,), dtype=jnp.float32),
        transient_encoder_kernel=_linear_init(
            transient_key,
            config.input_dim,
            config.transient_hidden_dim,
            config.init_scale,
        ),
        transient_encoder_bias=jnp.zeros(
            (config.transient_hidden_dim,),
            dtype=jnp.float32,
        ),
        transient_head_kernel=jnp.zeros(
            (config.transient_hidden_dim, config.output_dim),
            dtype=jnp.float32,
        ),
        transient_head_bias=jnp.zeros((config.output_dim,), dtype=jnp.float32),
    )


def permanent_transient_forward(
    params: AlbertaPermanentTransientParams,
    observation: Float[Array, " input_dim"],
) -> AlbertaPermanentTransientPrediction:
    """Return the sum of two independently represented predictions."""

    permanent_features = jnp.tanh(
        observation @ params.permanent_encoder_kernel + params.permanent_encoder_bias
    )
    transient_features = jnp.tanh(
        observation @ params.transient_encoder_kernel + params.transient_encoder_bias
    )
    permanent_prediction = (
        permanent_features @ params.permanent_head_kernel + params.permanent_head_bias
    )
    transient_prediction = (
        transient_features @ params.transient_head_kernel + params.transient_head_bias
    )
    return AlbertaPermanentTransientPrediction(
        prediction=permanent_prediction + transient_prediction,
        permanent_prediction=permanent_prediction,
        transient_prediction=transient_prediction,
        permanent_features=permanent_features,
        transient_features=transient_features,
    )


def _transient_loss(
    params: AlbertaPermanentTransientParams,
    observation: Array,
    target: Array,
) -> tuple[Array, AlbertaPermanentTransientPrediction]:
    parts = permanent_transient_forward(params, observation)
    residual_prediction = jax.lax.stop_gradient(
        parts.permanent_prediction
    ) + parts.transient_prediction
    error = residual_prediction - target
    return 0.5 * jnp.mean(error**2), parts


def _permanent_loss(
    params: AlbertaPermanentTransientParams,
    observation: Array,
    consolidation_target: Array,
) -> tuple[Array, Array]:
    permanent_features = jnp.tanh(
        observation @ params.permanent_encoder_kernel + params.permanent_encoder_bias
    )
    prediction = permanent_features @ params.permanent_head_kernel + params.permanent_head_bias
    error = prediction - jax.lax.stop_gradient(consolidation_target)
    return 0.5 * jnp.mean(error**2), prediction


def _tree_nbytes(tree: Any) -> int:
    return sum(
        int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _permanent_tree(params: AlbertaPermanentTransientParams) -> tuple[Array, ...]:
    return (
        params.permanent_encoder_kernel,
        params.permanent_encoder_bias,
        params.permanent_head_kernel,
        params.permanent_head_bias,
    )


def _transient_tree(params: AlbertaPermanentTransientParams) -> tuple[Array, ...]:
    return (
        params.transient_encoder_kernel,
        params.transient_encoder_bias,
        params.transient_head_kernel,
        params.transient_head_bias,
    )


class AlbertaPermanentTransientLearner:
    """Two-representation online learner with staged consolidation."""

    def __init__(self, config: AlbertaPermanentTransientConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> AlbertaPermanentTransientConfig:
        """Return the fixed learner config."""

        return self._config

    @property
    def design_record(self) -> PermanentTransientDesignRecord:
        """Return the attribution and non-parity boundary for this learner."""

        return permanent_transient_design_record()

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict versioned learner wrapper."""

        return {
            "type": "AlbertaPermanentTransientLearner",
            "schema": PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            "design_schema": PERMANENT_TRANSIENT_DESIGN_SCHEMA,
            "state_schema": PERMANENT_TRANSIENT_STATE_SCHEMA,
            "result_schema": PERMANENT_TRANSIENT_RESULT_SCHEMA,
            "resource_schema": PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> AlbertaPermanentTransientLearner:
        """Reconstruct only the exact current learner wrapper."""

        if not isinstance(config, Mapping):
            raise TypeError("permanent/transient learner config must be a mapping")
        payload = dict(config)
        expected_fields = {
            "type",
            "schema",
            "design_schema",
            "state_schema",
            "result_schema",
            "resource_schema",
            "config",
        }
        if set(payload) != expected_fields:
            raise ValueError("permanent/transient learner config fields are invalid")
        schemas = {
            "type": "AlbertaPermanentTransientLearner",
            "schema": PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            "design_schema": PERMANENT_TRANSIENT_DESIGN_SCHEMA,
            "state_schema": PERMANENT_TRANSIENT_STATE_SCHEMA,
            "result_schema": PERMANENT_TRANSIENT_RESULT_SCHEMA,
            "resource_schema": PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
        }
        for name, expected in schemas.items():
            if payload.pop(name) != expected:
                raise ValueError(f"permanent/transient learner {name} is unsupported")
        inner = payload.pop("config")
        if not isinstance(inner, Mapping):
            raise TypeError("permanent/transient inner config must be a mapping")
        return cls(AlbertaPermanentTransientConfig.from_config(inner))

    def _require_state_contract(self, state: AlbertaPermanentTransientState) -> None:
        if not isinstance(state, AlbertaPermanentTransientState):
            raise TypeError("state must be AlbertaPermanentTransientState")
        if not isinstance(state.params, AlbertaPermanentTransientParams):
            raise TypeError("state.params must be AlbertaPermanentTransientParams")
        c = self._config
        contracts = (
            ("permanent_encoder_kernel", (c.input_dim, c.permanent_hidden_dim)),
            ("permanent_encoder_bias", (c.permanent_hidden_dim,)),
            ("permanent_head_kernel", (c.permanent_hidden_dim, c.output_dim)),
            ("permanent_head_bias", (c.output_dim,)),
            ("transient_encoder_kernel", (c.input_dim, c.transient_hidden_dim)),
            ("transient_encoder_bias", (c.transient_hidden_dim,)),
            ("transient_head_kernel", (c.transient_hidden_dim, c.output_dim)),
            ("transient_head_bias", (c.output_dim,)),
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

    def state_is_valid(self, state: AlbertaPermanentTransientState) -> Array:
        """Return dynamic validity after enforcing exact shapes and dtypes."""

        self._require_state_contract(state)
        return _tree_is_finite(state.params) & _counter_valid(
            state.step_words,
            state.step_count,
        )

    def init(self, key: Array) -> AlbertaPermanentTransientState:
        """Initialize both representation systems and the exact clock."""

        return AlbertaPermanentTransientState(
            params=init_permanent_transient_params(key, self._config),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_parts(
        self,
        state: AlbertaPermanentTransientState,
        observation: Float[Array, " input_dim"],
    ) -> AlbertaPermanentTransientPrediction:
        """Return permanent, transient, and combined predictions."""

        self._require_state_contract(state)
        checked = _require_array_contract(
            observation,
            name="observation",
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        return permanent_transient_forward(state.params, checked)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: AlbertaPermanentTransientState,
        observation: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        """Return the current combined prediction."""

        return cast(Array, self.predict_parts(state, observation).prediction)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: AlbertaPermanentTransientState,
        observation: Float[Array, " input_dim"],
        target: Float[Array, " output_dim"],
    ) -> AlbertaPermanentTransientUpdateResult:
        """Stage transient learning, consolidation, decay, then one atomic commit."""

        self._require_state_contract(state)
        checked_observation = _require_array_contract(
            observation,
            name="observation",
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        checked_target = _require_array_contract(
            target,
            name="target",
            shape=(self._config.output_dim,),
            dtype=jnp.float32,
        )
        lifetime_counter_valid = _counter_valid(state.step_words, state.step_count)
        state_valid = _tree_is_finite(state.params) & lifetime_counter_valid
        observation_valid = jnp.all(jnp.isfinite(checked_observation))
        target_valid = jnp.all(jnp.isfinite(checked_target))
        input_valid = observation_valid & target_valid
        safe_observation = jnp.where(
            observation_valid,
            checked_observation,
            jnp.zeros_like(checked_observation),
        )
        safe_target = jnp.where(
            target_valid,
            checked_target,
            jnp.zeros_like(checked_target),
        )
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        (transient_loss, parts), transient_grads = jax.value_and_grad(
            _transient_loss,
            has_aux=True,
        )(state.params, safe_observation, safe_target)
        clipped_transient_grads, transient_grad_norm = _clip_grads(
            transient_grads,
            self._config.grad_clip,
        )
        c = self._config
        raw_transient_params = AlbertaPermanentTransientParams(
            permanent_encoder_kernel=state.params.permanent_encoder_kernel,
            permanent_encoder_bias=state.params.permanent_encoder_bias,
            permanent_head_kernel=state.params.permanent_head_kernel,
            permanent_head_bias=state.params.permanent_head_bias,
            transient_encoder_kernel=state.params.transient_encoder_kernel
            - c.transient_encoder_step_size
            * clipped_transient_grads.transient_encoder_kernel,
            transient_encoder_bias=state.params.transient_encoder_bias
            - c.transient_encoder_step_size
            * clipped_transient_grads.transient_encoder_bias,
            transient_head_kernel=state.params.transient_head_kernel
            - c.transient_head_step_size
            * clipped_transient_grads.transient_head_kernel,
            transient_head_bias=state.params.transient_head_bias
            - c.transient_head_step_size * clipped_transient_grads.transient_head_bias,
        )
        post_transient_parts = permanent_transient_forward(
            raw_transient_params,
            safe_observation,
        )
        consolidation_target = jax.lax.stop_gradient(
            parts.permanent_prediction + post_transient_parts.transient_prediction
        )
        (permanent_loss, permanent_prediction), permanent_grads = jax.value_and_grad(
            _permanent_loss,
            has_aux=True,
        )(raw_transient_params, safe_observation, consolidation_target)
        clipped_permanent_grads, permanent_grad_norm = _clip_grads(
            permanent_grads,
            self._config.grad_clip,
        )
        candidate_params = AlbertaPermanentTransientParams(
            permanent_encoder_kernel=state.params.permanent_encoder_kernel
            - c.permanent_encoder_step_size
            * clipped_permanent_grads.permanent_encoder_kernel,
            permanent_encoder_bias=state.params.permanent_encoder_bias
            - c.permanent_encoder_step_size
            * clipped_permanent_grads.permanent_encoder_bias,
            permanent_head_kernel=state.params.permanent_head_kernel
            - c.permanent_head_step_size * clipped_permanent_grads.permanent_head_kernel,
            permanent_head_bias=state.params.permanent_head_bias
            - c.permanent_head_step_size * clipped_permanent_grads.permanent_head_bias,
            transient_encoder_kernel=raw_transient_params.transient_encoder_kernel,
            transient_encoder_bias=raw_transient_params.transient_encoder_bias,
            transient_head_kernel=c.transient_decay
            * raw_transient_params.transient_head_kernel,
            transient_head_bias=c.transient_decay * raw_transient_params.transient_head_bias,
        )
        candidate_state = AlbertaPermanentTransientState(
            params=candidate_params,
            step_count=_saturating_increment(state.step_count),
            step_words=proposed_words,
        )
        error = safe_target - parts.prediction
        metrics = jnp.asarray(
            [
                transient_loss,
                jnp.mean(error**2),
                permanent_loss,
                transient_grad_norm,
                permanent_grad_norm,
                _tree_norm(_permanent_tree(candidate_params)),
                _tree_norm(_transient_tree(candidate_params)),
                jnp.mean(jnp.abs(parts.transient_prediction)),
            ],
            dtype=jnp.float32,
        )
        candidate_state_valid = _tree_is_finite(
            (
                raw_transient_params,
                candidate_params,
                parts,
                post_transient_parts,
                transient_grads,
                clipped_transient_grads,
                permanent_grads,
                clipped_permanent_grads,
                transient_loss,
                permanent_loss,
                permanent_prediction,
                consolidation_target,
                error,
                metrics,
            )
        ) & _counter_valid(candidate_state.step_words, candidate_state.step_count)
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        committed_prediction = jnp.where(
            update_applied,
            parts.prediction,
            jnp.zeros_like(parts.prediction),
        )
        committed_error = jnp.where(update_applied, error, jnp.zeros_like(error))
        committed_metrics = jnp.where(
            update_applied,
            metrics,
            jnp.zeros_like(metrics),
        )
        return AlbertaPermanentTransientUpdateResult(
            state=committed_state,
            prediction=committed_prediction,
            error=committed_error,
            metrics=committed_metrics,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            observation_valid=observation_valid,
            target_valid=target_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    def state_record(
        self,
        state: AlbertaPermanentTransientState,
    ) -> PermanentTransientStateRecord:
        """Build a host-only exact state record."""

        self._require_state_contract(state)
        parameter_shapes = tuple(
            (name, tuple(int(dimension) for dimension in getattr(state.params, name).shape))
            for name in _PARAMETER_NAMES
        )
        permanent_nbytes = _tree_nbytes(_permanent_tree(state.params))
        transient_nbytes = _tree_nbytes(_transient_tree(state.params))
        words_host = np.asarray(jax.device_get(state.step_words))
        words = (int(words_host[0]), int(words_host[1]))
        return PermanentTransientStateRecord(
            schema=PERMANENT_TRANSIENT_STATE_SCHEMA,
            config_schema=PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            parameter_shapes=parameter_shapes,
            permanent_parameter_nbytes=permanent_nbytes,
            transient_parameter_nbytes=transient_nbytes,
            parameter_nbytes=permanent_nbytes + transient_nbytes,
            state_nbytes=measure_permanent_transient_state_nbytes(state),
            step_count=int(jax.device_get(state.step_count)),
            step_words=words,
            state_valid=bool(jax.device_get(self.state_is_valid(state))),
        )

    def resource_record(
        self,
        state: AlbertaPermanentTransientState | None = None,
    ) -> PermanentTransientResourceRecord:
        """Return exact bytes and fixed maximum online work."""

        measured = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(measured)
        permanent_nbytes = _tree_nbytes(_permanent_tree(measured.params))
        transient_nbytes = _tree_nbytes(_transient_tree(measured.params))
        return PermanentTransientResourceRecord(
            schema=PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
            design_schema=PERMANENT_TRANSIENT_DESIGN_SCHEMA,
            config_schema=PERMANENT_TRANSIENT_CONFIG_SCHEMA,
            state_schema=PERMANENT_TRANSIENT_STATE_SCHEMA,
            result_schema=PERMANENT_TRANSIENT_RESULT_SCHEMA,
            input_dim=self._config.input_dim,
            output_dim=self._config.output_dim,
            permanent_hidden_dim=self._config.permanent_hidden_dim,
            transient_hidden_dim=self._config.transient_hidden_dim,
            total_hidden_features=(
                self._config.permanent_hidden_dim + self._config.transient_hidden_dim
            ),
            permanent_parameter_nbytes=permanent_nbytes,
            transient_parameter_nbytes=transient_nbytes,
            parameter_nbytes=permanent_nbytes + transient_nbytes,
            exact_lifetime_identity_nbytes=PERMANENT_TRANSIENT_EXACT_LIFETIME_NBYTES,
            lifetime_counter_nbytes=PERMANENT_TRANSIENT_LIFETIME_COUNTER_NBYTES,
            state_nbytes=measure_permanent_transient_state_nbytes(measured),
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            maximum_gradient_evaluations_per_update=2,
            maximum_forward_evaluations_per_update=3,
            maximum_updates_per_call=1,
            replay_capacity=0,
            maximum_stored_examples=0,
            persistent_capacity_growth=0,
        )


def measure_permanent_transient_state_nbytes(
    state: AlbertaPermanentTransientState,
) -> int:
    """Measure every persistent array byte in one state."""

    if not isinstance(state, AlbertaPermanentTransientState):
        raise TypeError("state must be AlbertaPermanentTransientState")
    return _tree_nbytes(state)


def run_permanent_transient_arrays(
    learner: AlbertaPermanentTransientLearner,
    observations: Float[Array, "steps input_dim"],
    targets: Float[Array, "steps output_dim"],
    *,
    state: AlbertaPermanentTransientState | None = None,
    key: Array | None = None,
) -> AlbertaPermanentTransientLearningResult:
    """Run one fixed-shape stream with ``jax.lax.scan``."""

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
    target_shape = (observation_array.shape[0], learner.config.output_dim)
    if target_array.shape != target_shape:
        raise ValueError(f"targets must have shape {target_shape}, got {target_array.shape}")
    if target_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"targets must have dtype float32, got {target_array.dtype}")
    if state is None:
        if key is None:
            raise ValueError("key is required when state is not supplied")
        state = learner.init(key)
    learner._require_state_contract(state)

    def step_fn(
        carry: AlbertaPermanentTransientState,
        inputs: tuple[Array, Array],
    ) -> tuple[AlbertaPermanentTransientState, tuple[Array, ...]]:
        observation, target = inputs
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

    final_state, outputs = jax.lax.scan(
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
    ) = outputs
    return AlbertaPermanentTransientLearningResult(
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
