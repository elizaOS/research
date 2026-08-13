"""Learning units for continual learning (Alberta Plan Step 1).

Combines function approximation with pluggable step-size optimizers under
temporal uniformity: every component updates at every time step, with no
train/eval phases.  Four learner families live here:

- :class:`LinearLearner` — ``y = w @ x + b`` with any core optimizer
  (LMS/IDBD/Autostep/ObGD) and an optional online feature normalizer.
- :class:`MLPLearner` — small MLP with backprop, sparse initialization, and
  dormant-neuron diagnostics/reset for plasticity studies.
- :class:`TDLinearLearner` — linear TD value prediction with eligibility
  traces and TD-aware step-size adaptation (TDIDBD family; Kearney et al.
  2019).
- :class:`TrueOnlineTDLearner` — True Online TD(lambda) with Dutch traces
  (van Seijen & Sutton 2014).

The ``run_*_learning_loop`` functions drive a learner over a
:class:`~alberta_framework.streams.base.ScanStream` inside ``jax.lax.scan``
so the whole loop JIT-compiles; the ``*_batched`` variants ``jax.vmap`` the
same loop over seeds for multi-seed experiments.
"""

import dataclasses
import functools
import math
import time
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.normalizers import (
    EMANormalizerState,
    Normalizer,
    StreamingBatchNormalizerState,
    WelfordNormalizerState,
)
from alberta_framework.core.optimizers import (
    LMS,
    TDIDBD,
    AutoTDIDBD,
    Bounder,
    Optimizer,
    TDOptimizer,
)
from alberta_framework.core.types import (
    AutostepParamState,
    AutostepState,
    AutoTDIDBDState,
    BatchedLearningResult,
    BatchedMLPResult,
    IDBDParamState,
    IDBDState,
    LMSState,
    MLPParams,
    NormalizerHistory,
    NormalizerTrackingConfig,
    ObGDState,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
    TDIDBDState,
    TDTimeStep,
)
from alberta_framework.core.types import (
    LearnerState as LegacyLearnerState,
)
from alberta_framework.core.types import (
    MLPLearnerState as LegacyMLPLearnerState,
)
from alberta_framework.core.types import (
    TDLearnerState as LegacyTDLearnerState,
)
from alberta_framework.streams.base import ScanStream

LINEAR_LEARNER_CONFIG_SCHEMA = "alberta.linear-learner.config.v2"
LINEAR_LEARNER_STATE_SCHEMA = "alberta.linear-learner.state.v2"
MLP_LEARNER_CONFIG_SCHEMA = "alberta.mlp-learner.config.v2"
MLP_LEARNER_STATE_SCHEMA = "alberta.mlp-learner.state.v2"
TD_LINEAR_LEARNER_CONFIG_SCHEMA = "alberta.td-linear-learner.config.v2"
TD_LINEAR_LEARNER_STATE_SCHEMA = "alberta.td-linear-learner.state.v2"
TRUE_ONLINE_TD_CONFIG_SCHEMA = "alberta.true-online-td.config.v2"
TRUE_ONLINE_TD_STATE_SCHEMA = "alberta.true-online-td.state.v2"

LEARNER_EXACT_LIFETIME_IDENTITY_NBYTES = 8
LEARNER_LIFETIME_COUNTER_NBYTES = 12

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _require_positive_feature_dim(feature_dim: int) -> None:
    if type(feature_dim) is not int or feature_dim <= 0 or feature_dim > _INT32_MAX:
        raise ValueError(f"feature_dim must be a strict integer in [1, {_INT32_MAX}]")


def _require_finite_config_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real scalar")
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


def _require_real_source(
    value: Any,
    *,
    name: str,
    shapes: tuple[tuple[int, ...], ...],
) -> Array:
    array = jnp.asarray(value)
    if array.shape not in shapes:
        raise ValueError(f"{name} must have one of shapes {shapes}, got {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise TypeError(f"{name} must have a real numeric dtype")
    return array.astype(jnp.float32)


def _require_prng_key(value: Any, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape == () and jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
        data = jax.random.key_data(array)
        if data.shape == (2,) and data.dtype == jnp.uint32:
            return array
    if array.shape == (2,) and array.dtype == jnp.uint32:
        return array
    raise ValueError(f"{name} must be one two-word Threefry JAX PRNG key")


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose an exact uint64-word successor without wrapping all-ones."""

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


def _floating_tree_is_finite(tree: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _require_tree_contract(value: Any, template: Any, *, name: str) -> None:
    value_leaves, value_structure = jax.tree_util.tree_flatten(value)
    template_leaves, template_structure = jax.tree_util.tree_flatten(template)
    if str(value_structure) != str(template_structure):
        raise TypeError(f"{name} structure does not match the configured component")
    for index, (leaf, expected) in enumerate(zip(value_leaves, template_leaves, strict=True)):
        array = jnp.asarray(leaf)
        expected_array = jnp.asarray(expected)
        if array.shape != expected_array.shape:
            raise ValueError(
                f"{name} leaf {index} must have shape {expected_array.shape}, got {array.shape}"
            )
        if array.dtype != expected_array.dtype:
            raise TypeError(
                f"{name} leaf {index} must have dtype {expected_array.dtype}, got {array.dtype}"
            )


def _select_state(candidate: Any, source: Any, commit: Array) -> Any:
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(commit, new, old),
        candidate,
        source,
    )


def _tree_nbytes(tree: Any) -> int:
    return sum(np.asarray(jax.device_get(leaf)).nbytes for leaf in jax.tree_util.tree_leaves(tree))


def _legacy_counter_words(step_count: Any, *, label: str) -> UInt[Array, " 2"]:
    counter = np.asarray(step_count)
    if counter.shape != () or counter.dtype != np.dtype(np.int32):
        raise TypeError(f"{label}.step_count must be a scalar int32")
    value = int(counter)
    if value < 0:
        raise ValueError(f"{label}.step_count cannot be negative")
    return jnp.asarray((0, value), dtype=jnp.uint32)


@chex.dataclass(frozen=True)
class LearnerState(LegacyLearnerState):
    """Versioned linear state with an exact 64-bit lifetime identity."""

    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class MLPLearnerState(LegacyMLPLearnerState):
    """Versioned MLP state with an exact 64-bit lifetime identity."""

    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class TDLearnerState(LegacyTDLearnerState):
    """Versioned TD-linear state with an exact 64-bit lifetime identity."""

    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class LearnerResourceBudget:
    """Exact initialized-state and per-update accounting for a learner."""

    learner_type: str
    feature_dim: int
    trainable_float32_scalars: int
    optimizer_state_nbytes: int
    normalizer_state_nbytes: int
    trace_state_nbytes: int
    utility_state_nbytes: int
    lifecycle_counter_nbytes: int
    state_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    max_updates_per_call: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible exact resource record."""

        return dataclasses.asdict(self)

# Type variable for TD stream state
StateT = TypeVar("StateT")

# Type alias for any optimizer type
AnyOptimizer = (
    Optimizer[LMSState]
    | Optimizer[IDBDState]
    | Optimizer[AutostepState]
    | Optimizer[ObGDState]
    | Optimizer[AutostepParamState]
    | Optimizer[IDBDParamState]
)

# Type alias for any TD optimizer type
AnyTDOptimizer = TDOptimizer[TDIDBDState] | TDOptimizer[AutoTDIDBDState]


def _td_optimizer_to_config(optimizer: AnyTDOptimizer) -> dict[str, Any]:
    if isinstance(optimizer, TDIDBD):
        return {
            "type": "TDIDBD",
            "initial_step_size": optimizer._initial_step_size,
            "meta_step_size": optimizer._meta_step_size,
            "trace_decay": optimizer._trace_decay,
            "use_semi_gradient": optimizer._use_semi_gradient,
        }
    if isinstance(optimizer, AutoTDIDBD):
        return {
            "type": "AutoTDIDBD",
            "initial_step_size": optimizer._initial_step_size,
            "meta_step_size": optimizer._meta_step_size,
            "trace_decay": optimizer._trace_decay,
            "normalizer_decay": optimizer._normalizer_decay,
        }
    raise TypeError("TD learner optimizer type is unsupported")


def _td_optimizer_from_config(config: Any) -> AnyTDOptimizer:
    if not isinstance(config, Mapping):
        raise TypeError("TD optimizer config must be a mapping")
    payload = dict(config)
    type_name = payload.get("type")
    expected = (
        {"type", "initial_step_size", "meta_step_size", "trace_decay", "use_semi_gradient"}
        if type_name == "TDIDBD"
        else {"type", "initial_step_size", "meta_step_size", "trace_decay", "normalizer_decay"}
        if type_name == "AutoTDIDBD"
        else set()
    )
    if not expected or set(payload) != expected:
        raise ValueError("TD optimizer config fields or type are unsupported")
    payload.pop("type")
    for field in ("initial_step_size", "meta_step_size"):
        _require_finite_config_float(
            payload[field],
            name=f"optimizer.{field}",
            minimum=0.0,
            minimum_inclusive=False,
        )
    _require_finite_config_float(
        payload["trace_decay"],
        name="optimizer.trace_decay",
        minimum=0.0,
        maximum=1.0,
    )
    if type_name == "TDIDBD":
        if type(payload["use_semi_gradient"]) is not bool:
            raise ValueError("optimizer.use_semi_gradient must be a strict boolean")
        return TDIDBD(**payload)
    _require_finite_config_float(
        payload["normalizer_decay"],
        name="optimizer.normalizer_decay",
        minimum=0.0,
        minimum_inclusive=False,
    )
    return AutoTDIDBD(**payload)


@chex.dataclass(frozen=True)
class UpdateResult:
    """Result of a learner update step.

    Attributes:
        state: Updated learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics -- shape (3,) without normalizer,
            (4,) with normalizer
    """

    state: LearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Array
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class MLPUpdateResult:
    """Result of an MLP learner update step.

    Attributes:
        state: Updated MLP learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics -- shape (3,) without normalizer,
            (4,) with normalizer
    """

    state: MLPLearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Array
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class LinearLearner:
    """Linear function approximator with pluggable optimizer and optional normalizer.

    Computes predictions as: ``y = w @ x + b``

    The learner maintains weights and bias, delegating the adaptation
    of learning rates to the optimizer (e.g., LMS or IDBD).

    This follows the Alberta Plan philosophy of temporal uniformity:
    every component updates at every time step.

    Attributes:
        optimizer: The optimizer to use for weight updates
        normalizer: Optional online feature normalizer
    """

    def __init__(
        self,
        optimizer: AnyOptimizer | None = None,
        normalizer: (
            Normalizer[EMANormalizerState]
            | Normalizer[WelfordNormalizerState]
            | Normalizer[StreamingBatchNormalizerState]
            | None
        ) = None,
    ):
        """Initialize the linear learner.

        Args:
            optimizer: Optimizer for weight updates. Defaults to LMS(0.01)
            normalizer: Optional feature normalizer (e.g. EMANormalizer, WelfordNormalizer)
        """
        self._optimizer: AnyOptimizer = optimizer or LMS(step_size=0.01)
        self._normalizer = normalizer

    @property
    def normalizer(
        self,
    ) -> (
        Normalizer[EMANormalizerState]
        | Normalizer[WelfordNormalizerState]
        | Normalizer[StreamingBatchNormalizerState]
        | None
    ):
        """The feature normalizer, or None if normalization is disabled."""
        return self._normalizer

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict versioned learner configuration."""

        return {
            "schema": LINEAR_LEARNER_CONFIG_SCHEMA,
            "type": "LinearLearner",
            "state_schema": LINEAR_LEARNER_STATE_SCHEMA,
            "optimizer": self._optimizer.to_config(),
            "normalizer": (
                self._normalizer.to_config() if self._normalizer is not None else None
            ),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LinearLearner":
        """Reconstruct only the exact current configuration schema."""

        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import optimizer_from_config

        if not isinstance(config, Mapping):
            raise TypeError("linear learner config must be a mapping")
        payload = dict(config)
        expected = {"schema", "type", "state_schema", "optimizer", "normalizer"}
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy linear learner config requires explicit migration")
            raise ValueError("linear learner config fields do not match v2")
        if payload.pop("schema") != LINEAR_LEARNER_CONFIG_SCHEMA:
            raise ValueError("linear learner config schema is unsupported")
        if payload.pop("type") != "LinearLearner":
            raise ValueError("linear learner config type is unsupported")
        if payload.pop("state_schema") != LINEAR_LEARNER_STATE_SCHEMA:
            raise ValueError("linear learner state schema is unsupported")
        optimizer_config = payload.pop("optimizer")
        if not isinstance(optimizer_config, dict):
            raise TypeError("linear learner optimizer config must be a dictionary")
        normalizer_config = payload.pop("normalizer")
        if normalizer_config is not None and not isinstance(normalizer_config, dict):
            raise TypeError("linear learner normalizer config must be a dictionary or null")
        optimizer = optimizer_from_config(optimizer_config)
        normalizer = (
            normalizer_from_config(normalizer_config)
            if normalizer_config is not None
            else None
        )
        return cls(optimizer=cast(AnyOptimizer, optimizer), normalizer=normalizer)

    def _require_state_contract(self, state: LearnerState) -> int:
        if not isinstance(state, LearnerState):
            if isinstance(state, LegacyLearnerState):
                raise TypeError("legacy linear state requires explicit migration")
            raise TypeError("state must be a LearnerState")
        weights = jnp.asarray(state.weights)
        if weights.ndim != 1 or weights.shape[0] <= 0:
            raise ValueError("state.weights must be a nonempty vector")
        feature_dim = weights.shape[0]
        _require_array_contract(
            state.weights,
            name="state.weights",
            shape=(feature_dim,),
            dtype=jnp.float32,
        )
        _require_array_contract(state.bias, name="state.bias", shape=(), dtype=jnp.float32)
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
        _require_tree_contract(
            state.optimizer_state,
            self._optimizer.init(feature_dim),
            name="state.optimizer_state",
        )
        if self._normalizer is None:
            if state.normalizer_state is not None:
                raise TypeError(
                    "state.normalizer_state must be null when normalization is disabled"
                )
        else:
            if state.normalizer_state is None:
                raise TypeError("state.normalizer_state is required by the configured normalizer")
            _require_tree_contract(
                state.normalizer_state,
                self._normalizer.init(feature_dim),
                name="state.normalizer_state",
            )
        return feature_dim

    def state_is_valid(self, state: LearnerState) -> Bool[Array, ""]:
        """Return dynamic validity after enforcing the static state contract."""

        self._require_state_contract(state)
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (jnp.asarray(state.uptime_s) >= 0.0)
        )

    def resource_budget(self, feature_dim: int) -> LearnerResourceBudget:
        """Return exact initialized-state accounting for this configuration."""

        state = self.init(feature_dim)
        optimizer_nbytes = _tree_nbytes(state.optimizer_state)
        normalizer_nbytes = _tree_nbytes(state.normalizer_state)
        return LearnerResourceBudget(
            learner_type="LinearLearner",
            feature_dim=feature_dim,
            trainable_float32_scalars=feature_dim + 1,
            optimizer_state_nbytes=optimizer_nbytes,
            normalizer_state_nbytes=normalizer_nbytes,
            trace_state_nbytes=0,
            utility_state_nbytes=0,
            lifecycle_counter_nbytes=LEARNER_LIFETIME_COUNTER_NBYTES,
            state_nbytes=_tree_nbytes(state),
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_updates_per_call=1,
            replay_capacity=0,
        )

    def init(self, feature_dim: int) -> LearnerState:
        """Initialize learner state.

        Args:
            feature_dim: Dimension of the input feature vector

        Returns:
            Initial learner state with zero weights and bias
        """
        _require_positive_feature_dim(feature_dim)
        optimizer_state = self._optimizer.init(feature_dim)

        normalizer_state = None
        if self._normalizer is not None:
            normalizer_state = self._normalizer.init(feature_dim)

        return LearnerState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            optimizer_state=optimizer_state,
            normalizer_state=normalizer_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def predict(self, state: LearnerState, observation: Observation) -> Prediction:
        """Compute prediction for an observation.

        Args:
            state: Current learner state
            observation: Input feature vector

        Returns:
            Scalar prediction ``y = w @ x + b``
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        return jnp.atleast_1d(jnp.dot(state.weights, obs) + state.bias)

    def update(
        self,
        state: LearnerState,
        observation: Observation,
        target: Target,
    ) -> UpdateResult:
        """Update learner given observation and target.

        Performs one step of the learning algorithm:
        1. Optionally normalize observation
        2. Compute prediction
        3. Compute error
        4. Get weight updates from optimizer
        5. Apply updates to weights and bias

        Args:
            state: Current learner state
            observation: Input feature vector
            target: Desired output

        Returns:
            UpdateResult with new state, prediction, error, and metrics
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        checked_target = _require_real_source(
            target,
            name="target",
            shapes=((), (1,)),
        )
        source_state_valid = self.state_is_valid(state)
        input_valid = jnp.all(jnp.isfinite(obs)) & jnp.all(jnp.isfinite(checked_target))
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        new_normalizer_state = state.normalizer_state
        if self._normalizer is not None and state.normalizer_state is not None:
            obs, new_normalizer_state = self._normalizer.normalize(
                state.normalizer_state, obs
            )

        prediction = jnp.atleast_1d(jnp.dot(state.weights, obs) + state.bias)

        error = jnp.squeeze(checked_target) - jnp.squeeze(prediction)

        opt_update = self._optimizer.update(
            state.optimizer_state,
            error,
            obs,
        )

        new_weights = state.weights + opt_update.weight_delta
        new_bias = state.bias + opt_update.bias_delta

        candidate_state = LearnerState(
            weights=new_weights,
            bias=new_bias,
            optimizer_state=opt_update.new_state,
            normalizer_state=new_normalizer_state,
            step_count=_words_to_saturating_int32(proposed_words),
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
            step_words=proposed_words,
        )

        # Pack metrics as array for scan compatibility
        squared_error = error**2
        mean_step_size = opt_update.metrics.get("mean_step_size", 0.0)

        if self._normalizer is not None and new_normalizer_state is not None:
            normalizer_mean_var = jnp.mean(new_normalizer_state.var)
            candidate_metrics = jnp.array(
                [squared_error, error, mean_step_size, normalizer_mean_var],
                dtype=jnp.float32,
            )
        else:
            candidate_metrics = jnp.array(
                [squared_error, error, mean_step_size], dtype=jnp.float32
            )

        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = _select_state(candidate_state, state, update_applied)
        metrics = jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics))

        return UpdateResult(
            state=new_state,
            prediction=prediction,
            error=jnp.atleast_1d(error),
            metrics=metrics,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=_lifetime_counter_valid(
                state.step_words, state.step_count
            ),
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


def _maybe_record(
    should_record: Array,
    history: Array | None,
    recording_idx: Array,
    value: Array,
) -> Array | None:
    """Conditionally write *value* into *history* at *recording_idx*.

    Returns *history* unchanged when it is ``None`` or when *should_record* is
    ``False``.  Intended for use inside ``jax.lax.scan`` bodies.
    """
    if history is None:
        return None
    result: Array = jax.lax.cond(
        should_record,
        lambda _: history.at[recording_idx].set(value),
        lambda _: history,
        None,
    )
    return result


def run_learning_loop[StreamStateT](
    learner: LinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: LearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> (
    tuple[LearnerState, Array]
    | tuple[LearnerState, Array, StepSizeHistory]
    | tuple[LearnerState, Array, NormalizerHistory]
    | tuple[LearnerState, Array, StepSizeHistory, NormalizerHistory]
):
    """Run the learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.
    It returns metrics as a fixed-size array rather than a list of dicts.

    Supports both plain and normalized learners. When the learner has a
    normalizer, metrics have 4 columns; otherwise 3 columns.

    Args:
        learner: The learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, returns StepSizeHistory.
        normalizer_tracking: Optional config for recording per-feature normalizer
            state. When provided, returns NormalizerHistory with means and
            variances over time.

    Returns:
        If no tracking:
            Tuple of (final_state, metrics_array) where metrics_array has shape
            (num_steps, 3) or (num_steps, 4) depending on normalizer
        If step_size_tracking only:
            Tuple of (final_state, metrics_array, step_size_history)
        If normalizer_tracking only:
            Tuple of (final_state, metrics_array, normalizer_history)
        If both:
            Tuple of (final_state, metrics_array, step_size_history, normalizer_history)

    Raises:
        ValueError: If tracking interval is invalid
    """
    # Validate tracking configs
    if step_size_tracking is not None:
        if step_size_tracking.interval < 1:
            raise ValueError(
                f"step_size_tracking.interval must be >= 1, got {step_size_tracking.interval}"
            )
        if step_size_tracking.interval > num_steps:
            raise ValueError(
                f"step_size_tracking.interval ({step_size_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    if normalizer_tracking is not None:
        if normalizer_tracking.interval < 1:
            raise ValueError(
                f"normalizer_tracking.interval must be >= 1, got {normalizer_tracking.interval}"
            )
        if normalizer_tracking.interval > num_steps:
            raise ValueError(
                f"normalizer_tracking.interval ({normalizer_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    feature_dim = stream.feature_dim

    # No tracking - simple case
    if step_size_tracking is None and normalizer_tracking is None:

        def step_fn(
            carry: tuple[LearnerState, StreamStateT], idx: Array
        ) -> tuple[tuple[LearnerState, StreamStateT], Array]:
            l_state, s_state = carry
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)
            return (result.state, new_s_state), result.metrics

        t0 = time.time()
        (final_learner, _), metrics = jax.lax.scan(
            step_fn, (learner_state, stream_state), jnp.arange(num_steps)
        )
        elapsed = time.time() - t0
        final_learner = final_learner.replace(uptime_s=final_learner.uptime_s + elapsed)  # type: ignore[attr-defined]

        return final_learner, metrics

    # Tracking enabled - need to set up history arrays
    ss_interval = step_size_tracking.interval if step_size_tracking else num_steps + 1
    norm_interval = normalizer_tracking.interval if normalizer_tracking else num_steps + 1

    ss_num_recordings = num_steps // ss_interval if step_size_tracking else 0
    norm_num_recordings = num_steps // norm_interval if normalizer_tracking else 0

    # Pre-allocate step-size history arrays
    ss_history = (
        jnp.zeros((ss_num_recordings, feature_dim), dtype=jnp.float32)
        if step_size_tracking
        else None
    )
    ss_bias_history = (
        jnp.zeros(ss_num_recordings, dtype=jnp.float32)
        if step_size_tracking and step_size_tracking.include_bias
        else None
    )
    ss_rec_indices = jnp.zeros(ss_num_recordings, dtype=jnp.int32) if step_size_tracking else None

    # Check if we need to track Autostep normalizers
    track_autostep_normalizers = hasattr(learner_state.optimizer_state, "normalizers")
    ss_normalizers = (
        jnp.zeros((ss_num_recordings, feature_dim), dtype=jnp.float32)
        if step_size_tracking and track_autostep_normalizers
        else None
    )

    # Pre-allocate normalizer state history arrays
    norm_means = (
        jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
        if normalizer_tracking
        else None
    )
    norm_vars = (
        jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
        if normalizer_tracking
        else None
    )
    norm_rec_indices = (
        jnp.zeros(norm_num_recordings, dtype=jnp.int32) if normalizer_tracking else None
    )

    def step_fn_with_tracking(
        carry: tuple[
            LearnerState,
            StreamStateT,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
        ],
        idx: Array,
    ) -> tuple[
        tuple[
            LearnerState,
            StreamStateT,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
        ],
        Array,
    ]:
        (
            l_state,
            s_state,
            ss_hist,
            ss_bias_hist,
            ss_rec,
            ss_norm,
            n_means,
            n_vars,
            n_rec,
        ) = carry

        # Perform learning step
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)

        # Step-size tracking
        new_ss_hist = ss_hist
        new_ss_bias_hist = ss_bias_hist
        new_ss_rec = ss_rec
        new_ss_norm = ss_norm

        if ss_hist is not None:
            should_record_ss = (idx % ss_interval) == 0
            recording_idx = idx // ss_interval

            # Extract current step-sizes
            opt_state = result.state.optimizer_state
            if hasattr(opt_state, "log_step_sizes"):
                # IDBD stores log step-sizes
                weight_ss = jnp.exp(opt_state.log_step_sizes)
                bias_ss = opt_state.bias_step_size
            elif hasattr(opt_state, "step_sizes"):
                # Autostep stores step-sizes directly
                weight_ss = opt_state.step_sizes
                bias_ss = opt_state.bias_step_size
            else:
                # LMS has a single fixed step-size
                weight_ss = jnp.full(feature_dim, opt_state.step_size)
                bias_ss = opt_state.step_size

            new_ss_hist = _maybe_record(should_record_ss, ss_hist, recording_idx, weight_ss)
            new_ss_bias_hist = _maybe_record(
                should_record_ss, ss_bias_hist, recording_idx, bias_ss
            )
            new_ss_rec = _maybe_record(should_record_ss, ss_rec, recording_idx, idx)
            # Track Autostep normalizers (v_i) if applicable
            if ss_norm is not None and hasattr(opt_state, "normalizers"):
                new_ss_norm = _maybe_record(
                    should_record_ss, ss_norm, recording_idx, opt_state.normalizers
                )

        # Normalizer state tracking
        new_n_means = n_means
        new_n_vars = n_vars
        new_n_rec = n_rec

        if n_means is not None:
            should_record_norm = (idx % norm_interval) == 0
            norm_recording_idx = idx // norm_interval

            norm_state = result.state.normalizer_state
            new_n_means = _maybe_record(
                should_record_norm, n_means, norm_recording_idx, norm_state.mean
            )
            new_n_vars = _maybe_record(
                should_record_norm, n_vars, norm_recording_idx, norm_state.var
            )
            new_n_rec = _maybe_record(should_record_norm, n_rec, norm_recording_idx, idx)

        return (
            result.state,
            new_s_state,
            new_ss_hist,
            new_ss_bias_hist,
            new_ss_rec,
            new_ss_norm,
            new_n_means,
            new_n_vars,
            new_n_rec,
        ), result.metrics

    initial_carry = (
        learner_state,
        stream_state,
        ss_history,
        ss_bias_history,
        ss_rec_indices,
        ss_normalizers,
        norm_means,
        norm_vars,
        norm_rec_indices,
    )

    t0 = time.time()
    (
        (
            final_learner,
            _,
            final_ss_hist,
            final_ss_bias_hist,
            final_ss_rec,
            final_ss_norm,
            final_n_means,
            final_n_vars,
            final_n_rec,
        ),
        metrics,
    ) = jax.lax.scan(step_fn_with_tracking, initial_carry, jnp.arange(num_steps))
    elapsed = time.time() - t0
    final_learner = final_learner.replace(uptime_s=final_learner.uptime_s + elapsed)  # type: ignore[attr-defined]

    # Build return values based on what was tracked
    ss_history_result = None
    if step_size_tracking is not None and final_ss_hist is not None:
        ss_history_result = StepSizeHistory(
            step_sizes=final_ss_hist,
            bias_step_sizes=final_ss_bias_hist,
            recording_indices=final_ss_rec,
            normalizers=final_ss_norm,
        )

    norm_history_result = None
    if normalizer_tracking is not None and final_n_means is not None:
        norm_history_result = NormalizerHistory(
            means=final_n_means,
            variances=final_n_vars,
            recording_indices=final_n_rec,
        )

    # Return appropriate tuple based on what was tracked
    if ss_history_result is not None and norm_history_result is not None:
        return final_learner, metrics, ss_history_result, norm_history_result
    elif ss_history_result is not None:
        return final_learner, metrics, ss_history_result
    elif norm_history_result is not None:
        return final_learner, metrics, norm_history_result
    else:
        return final_learner, metrics


def run_learning_loop_batched[StreamStateT](
    learner: LinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: LearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> BatchedLearningResult:
    """Run learning loop across multiple seeds in parallel using jax.vmap.

    All seeds share one compiled program, so multi-seed experiments
    parallelize on accelerators instead of running sequentially.

    Supports both plain and normalized learners.

    Args:
        learner: The learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)
        normalizer_tracking: Optional config for recording normalizer state.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)

    Returns:
        BatchedLearningResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, num_cols)
            - step_size_history: Batched history or None if tracking disabled
            - normalizer_history: Batched history or None if tracking disabled

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework import LinearLearner, IDBD, RandomWalkStream
    from alberta_framework import run_learning_loop_batched

    stream = RandomWalkStream(feature_dim=10)
    learner = LinearLearner(optimizer=IDBD())

    # Run 30 seeds in parallel
    keys = jr.split(jr.key(42), 30)
    result = run_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)

    # result.metrics has shape (30, 10000, 3)
    mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average over seeds
    ```
    """

    # Define single-seed function that returns consistent structure
    def single_seed_run(
        key: Array,
    ) -> tuple[LearnerState, Array, StepSizeHistory | None, NormalizerHistory | None]:
        result = run_learning_loop(
            learner, stream, num_steps, key, learner_state,
            step_size_tracking, normalizer_tracking,
        )

        # Unpack based on what tracking was enabled
        if step_size_tracking is not None and normalizer_tracking is not None:
            state, metrics, ss_history, norm_history = cast(
                tuple[LearnerState, Array, StepSizeHistory, NormalizerHistory],
                result,
            )
            return state, metrics, ss_history, norm_history
        elif step_size_tracking is not None:
            state, metrics, ss_history = cast(
                tuple[LearnerState, Array, StepSizeHistory], result
            )
            return state, metrics, ss_history, None
        elif normalizer_tracking is not None:
            state, metrics, norm_history = cast(
                tuple[LearnerState, Array, NormalizerHistory], result
            )
            return state, metrics, None, norm_history
        else:
            state, metrics = cast(tuple[LearnerState, Array], result)
            return state, metrics, None, None

    # vmap over the keys dimension
    t0 = time.time()
    batched_states, batched_metrics, batched_ss_history, batched_norm_history = jax.vmap(
        single_seed_run
    )(keys)
    elapsed = time.time() - t0
    batched_states = batched_states.replace(  # type: ignore[attr-defined]
        uptime_s=batched_states.uptime_s + elapsed
    )

    # Reconstruct batched histories if tracking was enabled
    if step_size_tracking is not None and batched_ss_history is not None:
        batched_step_size_history = StepSizeHistory(
            step_sizes=batched_ss_history.step_sizes,
            bias_step_sizes=batched_ss_history.bias_step_sizes,
            recording_indices=batched_ss_history.recording_indices,
            normalizers=batched_ss_history.normalizers,
        )
    else:
        batched_step_size_history = None

    if normalizer_tracking is not None and batched_norm_history is not None:
        batched_normalizer_history = NormalizerHistory(
            means=batched_norm_history.means,
            variances=batched_norm_history.variances,
            recording_indices=batched_norm_history.recording_indices,
        )
    else:
        batched_normalizer_history = None

    return BatchedLearningResult(
        states=batched_states,
        metrics=batched_metrics,
        step_size_history=batched_step_size_history,
        normalizer_history=batched_normalizer_history,
    )


def metrics_to_dicts(metrics: Array, normalized: bool = False) -> list[dict[str, float]]:
    """Convert metrics array to list of dicts for backward compatibility.

    Args:
        metrics: Array of shape (num_steps, 3) or (num_steps, 4)
        normalized: If True, expects 4 columns including normalizer_mean_var

    Returns:
        List of metric dictionaries
    """
    result = []
    for row in metrics:
        d = {
            "squared_error": float(row[0]),
            "error": float(row[1]),
            "mean_step_size": float(row[2]),
        }
        if normalized and len(row) > 3:
            d["normalizer_mean_var"] = float(row[3])
        result.append(d)
    return result


# =============================================================================
# MLP Learner (Step 2 of Alberta Plan)
# =============================================================================


class MLPLearner:
    """Multi-layer perceptron with composable optimizer, bounder, and normalizer.

    Architecture: ``Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> Dense(1)``

    When ``use_layer_norm=False``, the architecture simplifies to:
    ``Input -> [Dense(H) -> LeakyReLU] x N -> Dense(1)``

    Uses parameterless layer normalization and sparse initialization following
    Elsayed et al. 2024. Accepts a pluggable optimizer (LMS, Autostep), an
    optional bounder (ObGDBounding), and an optional feature normalizer
    (EMANormalizer, WelfordNormalizer).

    The update flow:
    1. If normalizer: normalize observation, update normalizer state
    2. Forward pass + ``jax.grad`` to get per-layer prediction gradients
    3. Update eligibility traces: ``z = gamma * lamda * z + grad``
    4. Per-layer optimizer step: ``step, new_opt = optimizer.update_from_gradient(state, z)``
    5. If bounder: bound all steps globally
    6. Apply: ``param += scale * error * step``

    Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning
    Finally Works"

    Attributes:
        hidden_sizes: Tuple of hidden layer sizes
        optimizer: Optimizer for per-weight step-size adaptation
        bounder: Optional update bounder (e.g. ObGDBounding)
        normalizer: Optional feature normalizer
        use_layer_norm: Whether to apply parameterless layer normalization
        gamma: Discount factor for trace decay
        lamda: Eligibility trace decay parameter
        sparsity: Fraction of weights zeroed out per output neuron
        leaky_relu_slope: Negative slope for LeakyReLU activation

    Single-Step (Daemon) Usage
    --------------------------
    Both ``predict()`` and ``update()`` work with single unbatched
    observations (1D arrays of shape ``(feature_dim,)``). This is the
    intended usage for daemon-style deployments.

    For low-latency daemon use, pre-compile ``predict`` and ``update``
    at startup by running a dummy warmup call:

    ```python
    dummy_obs = jnp.zeros(feature_dim)
    dummy_target = jnp.zeros(1)
    _ = learner.predict(state, dummy_obs)
    result = learner.update(state, dummy_obs, dummy_target)
    ```
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        gamma: float = 0.0,
        lamda: float = 0.0,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        track_neuron_utility: bool = False,
        neuron_utility_decay: float = 0.99,
    ):
        """Initialize MLP learner.

        Args:
            hidden_sizes: Tuple of hidden layer sizes (default: two layers of 128)
            optimizer: Optimizer for weight updates. Defaults to LMS(step_size).
                Must support ``init_for_shape`` and ``update_from_gradient``.
            step_size: Base learning rate (used only when optimizer is None,
                default: 1.0)
            bounder: Optional update bounder (e.g. ObGDBounding for ObGD-style
                bounding). When None, no bounding is applied.
            gamma: Discount factor for trace decay (default: 0.0 for supervised)
            lamda: Eligibility trace decay parameter (default: 0.0 for supervised)
            normalizer: Optional feature normalizer. When provided, features are
                normalized before prediction and learning.
            sparsity: Fraction of weights zeroed out per output neuron (default: 0.9)
            leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
            use_layer_norm: Whether to apply parameterless layer normalization
                between hidden layers (default: True). Set to False for ablation
                studies.
            head_optimizer: Optional separate optimizer for the output (head) layer.
                When None (default), all layers use ``optimizer``. When set, hidden
                layers use ``optimizer`` while the output layer uses
                ``head_optimizer``. This enables hybrid configurations like
                stable LMS for the trunk with adaptive Autostep for the head.
            track_neuron_utility: When True, maintain a per-hidden-unit EMA of
                the gradient L2 norm in ``MLPLearnerState.neuron_utility``.
                Enables dormant-neuron detection for long-running continual agents.
                Default False to avoid overhead when unused.
            neuron_utility_decay: EMA decay for neuron utility (default 0.99).
                Higher values track slower, smoother utility signals.
        """
        if not isinstance(hidden_sizes, tuple) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a nonempty tuple")
        for index, width in enumerate(hidden_sizes):
            if type(width) is not int or width <= 0 or width > _INT32_MAX:
                raise ValueError(
                    f"hidden_sizes[{index}] must be a strict integer in [1, {_INT32_MAX}]"
                )
        _require_finite_config_float(
            step_size,
            name="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite_config_float(gamma, name="gamma", minimum=0.0, maximum=1.0)
        _require_finite_config_float(lamda, name="lamda", minimum=0.0, maximum=1.0)
        _require_finite_config_float(sparsity, name="sparsity", minimum=0.0, maximum=1.0)
        _require_finite_config_float(
            leaky_relu_slope,
            name="leaky_relu_slope",
            minimum=0.0,
        )
        _require_finite_config_float(
            neuron_utility_decay,
            name="neuron_utility_decay",
            minimum=0.0,
            maximum=1.0,
        )
        if type(use_layer_norm) is not bool:
            raise ValueError("use_layer_norm must be a strict boolean")
        if type(track_neuron_utility) is not bool:
            raise ValueError("track_neuron_utility must be a strict boolean")
        self._hidden_sizes = hidden_sizes
        self._optimizer: AnyOptimizer = optimizer or LMS(step_size=step_size)
        self._head_optimizer: AnyOptimizer | None = head_optimizer
        if not self._optimizer.supported_for_mlp():
            raise ValueError(
                f"optimizer {type(self._optimizer).__name__} does not support the MLP "
                "shape-generic update API"
            )
        if self._head_optimizer is not None and not self._head_optimizer.supported_for_mlp():
            raise ValueError(
                f"head_optimizer {type(self._head_optimizer).__name__} does not support the MLP "
                "shape-generic update API"
            )
        self._bounder = bounder
        self._gamma = gamma
        self._lamda = lamda
        self._normalizer = normalizer
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm
        self._track_neuron_utility = track_neuron_utility
        self._neuron_utility_decay = neuron_utility_decay

    @property
    def normalizer(
        self,
    ) -> Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None:
        """The feature normalizer, or None if normalization is disabled."""
        return self._normalizer

    @staticmethod
    def dormant_neuron_fraction(
        state: MLPLearnerState, threshold: float = 0.01
    ) -> float:
        """Fraction of hidden neurons whose utility EMA is below *threshold*.

        Args:
            state: Current MLP learner state (must have neuron_utility tracked).
            threshold: Neurons with utility below this are counted as dormant.

        Returns:
            Scalar fraction in [0, 1].  Returns 0.0 when neuron_utility is None.
        """
        if state.neuron_utility is None:
            return 0.0
        total = 0
        dormant = 0
        for u in state.neuron_utility:
            total += u.shape[0]
            dormant += int(jnp.sum(u < threshold).item())
        return dormant / total if total > 0 else 0.0

    def reset_dormant_neurons(
        self,
        state: MLPLearnerState,
        key: Array,
        threshold: float = 0.01,
    ) -> MLPLearnerState:
        """Re-initialise weights for dormant hidden neurons.

        For each hidden layer, neurons whose utility EMA is below *threshold*
        receive fresh sparse-initialised incoming weights, zero eligibility
        traces, and optimizer state slices re-initialised from the optimizer's
        ``init_for_shape`` values (shared scalar leaves such as meta step-sizes
        are left untouched).  Out-going weights from dormant neurons to
        downstream layers are also zeroed so the reset does not inject a
        sudden large signal.

        This is a Python-level operation (not JIT-compiled) because the
        dormancy mask changes structure per call.  Call periodically, e.g.
        every N environment steps.

        Args:
            state: Current MLP learner state.
            key: JAX random key for re-initialisation.
            threshold: Utility below which a neuron is considered dormant.

        Returns:
            Updated state with dormant neurons re-initialised.
        """
        if state.neuron_utility is None:
            return state

        new_weights = list(state.params.weights)
        new_biases = list(state.params.biases)
        new_traces = list(state.traces)
        new_opt_states = list(state.optimizer_states)
        new_utility = list(state.neuron_utility)

        n_hidden = len(self._hidden_sizes)
        for layer_i in range(n_hidden):
            dormant_mask = state.neuron_utility[layer_i] < threshold
            if not jnp.any(dormant_mask):
                continue
            h_out, h_in = new_weights[layer_i].shape
            key, subkey = jax.random.split(key)
            fresh_w = sparse_init(subkey, (h_out, h_in), sparsity=self._sparsity)
            new_w = jnp.where(dormant_mask[:, None], fresh_w, new_weights[layer_i])
            zero_b = jnp.zeros_like(new_biases[layer_i])
            new_b = jnp.where(dormant_mask, zero_b, new_biases[layer_i])
            new_weights[layer_i] = new_w
            new_biases[layer_i] = new_b
            # Zero traces for incoming weights/biases of reset neurons
            new_traces[2 * layer_i] = jnp.where(dormant_mask[:, None], 0.0, new_traces[2 * layer_i])
            new_traces[2 * layer_i + 1] = jnp.where(dormant_mask, 0.0, new_traces[2 * layer_i + 1])
            # Re-initialise optimizer state slices for reset neurons from the
            # optimizer's own init_for_shape values, not zeros: IDBD stores
            # *log* step-sizes (zero would mean exp(0)=1.0) and Autostep
            # step-sizes adapt multiplicatively (zero would stick at zero).
            # Scalar leaves (shared values like meta_step_size) stay untouched.
            fresh_w_opt = self._optimizer.init_for_shape((h_out, h_in))
            fresh_b_opt = self._optimizer.init_for_shape((h_out,))

            def _reset_by_mask(x: Array, fresh: Array, m: Array = dormant_mask) -> Array:
                if x.ndim == 0:
                    return x
                sel = m[:, None] if x.ndim == 2 else m
                return jnp.where(sel, fresh, x)

            new_opt_states[2 * layer_i] = jax.tree_util.tree_map(
                _reset_by_mask, new_opt_states[2 * layer_i], fresh_w_opt
            )
            new_opt_states[2 * layer_i + 1] = jax.tree_util.tree_map(
                _reset_by_mask, new_opt_states[2 * layer_i + 1], fresh_b_opt
            )
            # Zero outgoing weights from the next layer that feed into this neuron
            if layer_i + 1 < len(new_weights):
                out_w = new_weights[layer_i + 1]  # shape (h_{i+1}, h_i)
                new_weights[layer_i + 1] = jnp.where(dormant_mask[None, :], 0.0, out_w)
            # Reset utility for reset neurons
            new_utility[layer_i] = jnp.where(dormant_mask, 0.0, new_utility[layer_i])

        return state.replace(  # type: ignore[attr-defined, no-any-return]
            params=MLPParams(weights=tuple(new_weights), biases=tuple(new_biases)),
            traces=tuple(new_traces),
            optimizer_states=tuple(new_opt_states),
            neuron_utility=tuple(new_utility),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration to dict.

        Returns:
            Dict with all constructor arguments needed to recreate
            the learner via ``from_config()``.
        """
        config: dict[str, Any] = {
            "schema": MLP_LEARNER_CONFIG_SCHEMA,
            "type": "MLPLearner",
            "state_schema": MLP_LEARNER_STATE_SCHEMA,
            "hidden_sizes": list(self._hidden_sizes),
            "optimizer": self._optimizer.to_config(),
            "bounder": self._bounder.to_config() if self._bounder is not None else None,
            "normalizer": self._normalizer.to_config() if self._normalizer is not None else None,
            "head_optimizer": (
                self._head_optimizer.to_config()
                if self._head_optimizer is not None
                else None
            ),
            "sparsity": self._sparsity,
            "leaky_relu_slope": self._leaky_relu_slope,
            "use_layer_norm": self._use_layer_norm,
            "gamma": self._gamma,
            "lamda": self._lamda,
            "track_neuron_utility": self._track_neuron_utility,
            "neuron_utility_decay": self._neuron_utility_decay,
        }
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MLPLearner":
        """Reconstruct learner from a config dict.

        Args:
            config: Dict as produced by ``to_config()``

        Returns:
            Reconstructed MLPLearner instance
        """
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        if not isinstance(config, Mapping):
            raise TypeError("MLP learner config must be a mapping")
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "state_schema",
            "hidden_sizes",
            "optimizer",
            "bounder",
            "normalizer",
            "head_optimizer",
            "sparsity",
            "leaky_relu_slope",
            "use_layer_norm",
            "gamma",
            "lamda",
            "track_neuron_utility",
            "neuron_utility_decay",
        }
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy MLP learner config requires explicit migration")
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                f"MLP learner config fields do not match v2; missing={missing}, extra={extra}"
            )
        if payload.pop("schema") != MLP_LEARNER_CONFIG_SCHEMA:
            raise ValueError("MLP learner config schema is unsupported")
        if payload.pop("type") != "MLPLearner":
            raise ValueError("MLP learner config type is unsupported")
        if payload.pop("state_schema") != MLP_LEARNER_STATE_SCHEMA:
            raise ValueError("MLP learner state schema is unsupported")

        optimizer_cfg = payload.pop("optimizer")
        if not isinstance(optimizer_cfg, dict):
            raise TypeError("MLP learner optimizer config must be a dictionary")
        optimizer = optimizer_from_config(optimizer_cfg)
        bounder_cfg = payload.pop("bounder")
        if bounder_cfg is not None and not isinstance(bounder_cfg, dict):
            raise TypeError("MLP learner bounder config must be a dictionary or null")
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = payload.pop("normalizer")
        if normalizer_cfg is not None and not isinstance(normalizer_cfg, dict):
            raise TypeError("MLP learner normalizer config must be a dictionary or null")
        normalizer = normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        head_opt_cfg = payload.pop("head_optimizer")
        if head_opt_cfg is not None and not isinstance(head_opt_cfg, dict):
            raise TypeError("MLP head optimizer config must be a dictionary or null")
        head_optimizer = optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None

        hidden_sizes = payload.pop("hidden_sizes")
        if not isinstance(hidden_sizes, list):
            raise TypeError("MLP hidden_sizes must be a list in serialized config")

        return cls(
            hidden_sizes=tuple(hidden_sizes),
            optimizer=cast(AnyOptimizer, optimizer),
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=cast(AnyOptimizer | None, head_optimizer),
            **payload,
        )

    def _require_state_contract(self, state: MLPLearnerState) -> int:
        if not isinstance(state, MLPLearnerState):
            if isinstance(state, LegacyMLPLearnerState):
                raise TypeError("legacy MLP state requires explicit migration")
            raise TypeError("state must be an MLPLearnerState")
        weights = state.params.weights
        biases = state.params.biases
        n_layers = len(self._hidden_sizes) + 1
        if len(weights) != n_layers or len(biases) != n_layers:
            raise ValueError("state.params layer count does not match the learner architecture")
        first = jnp.asarray(weights[0])
        if first.ndim != 2 or first.shape[1] <= 0:
            raise ValueError("state.params.weights[0] must have a positive input width")
        feature_dim = first.shape[1]
        layer_sizes = (feature_dim, *self._hidden_sizes, 1)
        if len(state.traces) != 2 * n_layers:
            raise ValueError("state.traces must contain one weight and bias trace per layer")
        if len(state.optimizer_states) != 2 * n_layers:
            raise ValueError("state.optimizer_states must contain one state per parameter")
        for index in range(n_layers):
            weight_shape = (layer_sizes[index + 1], layer_sizes[index])
            bias_shape = (layer_sizes[index + 1],)
            _require_array_contract(
                weights[index],
                name=f"state.params.weights[{index}]",
                shape=weight_shape,
                dtype=jnp.float32,
            )
            _require_array_contract(
                biases[index],
                name=f"state.params.biases[{index}]",
                shape=bias_shape,
                dtype=jnp.float32,
            )
            _require_array_contract(
                state.traces[2 * index],
                name=f"state.traces[{2 * index}]",
                shape=weight_shape,
                dtype=jnp.float32,
            )
            _require_array_contract(
                state.traces[2 * index + 1],
                name=f"state.traces[{2 * index + 1}]",
                shape=bias_shape,
                dtype=jnp.float32,
            )
            optimizer = (
                self._head_optimizer
                if self._head_optimizer is not None and index == n_layers - 1
                else self._optimizer
            )
            _require_tree_contract(
                state.optimizer_states[2 * index],
                optimizer.init_for_shape(weight_shape),
                name=f"state.optimizer_states[{2 * index}]",
            )
            _require_tree_contract(
                state.optimizer_states[2 * index + 1],
                optimizer.init_for_shape(bias_shape),
                name=f"state.optimizer_states[{2 * index + 1}]",
            )
        if self._normalizer is None:
            if state.normalizer_state is not None:
                raise TypeError(
                    "state.normalizer_state must be null when normalization is disabled"
                )
        else:
            if state.normalizer_state is None:
                raise TypeError("state.normalizer_state is required by the configured normalizer")
            _require_tree_contract(
                state.normalizer_state,
                self._normalizer.init(feature_dim),
                name="state.normalizer_state",
            )
        if self._track_neuron_utility:
            if state.neuron_utility is None or len(state.neuron_utility) != len(
                self._hidden_sizes
            ):
                raise ValueError("state.neuron_utility does not match the hidden architecture")
            for index, width in enumerate(self._hidden_sizes):
                _require_array_contract(
                    state.neuron_utility[index],
                    name=f"state.neuron_utility[{index}]",
                    shape=(width,),
                    dtype=jnp.float32,
                )
        elif state.neuron_utility is not None:
            raise TypeError("state.neuron_utility must be null when utility tracking is disabled")
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
        return feature_dim

    def state_is_valid(self, state: MLPLearnerState) -> Bool[Array, ""]:
        """Return dynamic validity after enforcing the static state contract."""

        self._require_state_contract(state)
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (jnp.asarray(state.uptime_s) >= 0.0)
        )

    def resource_budget(self, feature_dim: int) -> LearnerResourceBudget:
        """Return exact initialized-state accounting for this architecture."""

        state = self.init(feature_dim, jax.random.key(0))
        trainable = sum(array.size for array in state.params.weights) + sum(
            array.size for array in state.params.biases
        )
        return LearnerResourceBudget(
            learner_type="MLPLearner",
            feature_dim=feature_dim,
            trainable_float32_scalars=trainable,
            optimizer_state_nbytes=_tree_nbytes(state.optimizer_states),
            normalizer_state_nbytes=_tree_nbytes(state.normalizer_state),
            trace_state_nbytes=_tree_nbytes(state.traces),
            utility_state_nbytes=_tree_nbytes(state.neuron_utility),
            lifecycle_counter_nbytes=LEARNER_LIFETIME_COUNTER_NBYTES,
            state_nbytes=_tree_nbytes(state),
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_updates_per_call=1,
            replay_capacity=0,
        )

    def init(self, feature_dim: int, key: Array) -> MLPLearnerState:
        """Initialize MLP learner state with sparse weights.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key for weight initialization

        Returns:
            Initial MLP learner state with sparse weights and zero biases
        """
        _require_positive_feature_dim(feature_dim)
        key = _require_prng_key(key, name="key")
        # Build layer sizes: [feature_dim, hidden1, hidden2, ..., 1]
        layer_sizes = [feature_dim, *self._hidden_sizes, 1]

        weights_list = []
        biases_list = []
        traces_list = []
        opt_states_list = []

        n_total_layers = len(layer_sizes) - 1
        for i in range(n_total_layers):
            fan_out = layer_sizes[i + 1]
            fan_in = layer_sizes[i]
            key, subkey = jax.random.split(key)
            w = sparse_init(subkey, (fan_out, fan_in), sparsity=self._sparsity)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            weights_list.append(w)
            biases_list.append(b)
            # Traces for weights and biases (interleaved: w0, b0, w1, b1, ...)
            traces_list.append(jnp.zeros_like(w))
            traces_list.append(jnp.zeros_like(b))
            # Optimizer states: use head_optimizer for the output layer if set
            is_output = i == n_total_layers - 1
            opt = (
                self._head_optimizer
                if (self._head_optimizer is not None and is_output)
                else self._optimizer
            )
            opt_states_list.append(opt.init_for_shape(w.shape))
            opt_states_list.append(opt.init_for_shape(b.shape))

        params = MLPParams(
            weights=tuple(weights_list),
            biases=tuple(biases_list),
        )

        normalizer_state = None
        if self._normalizer is not None:
            normalizer_state = self._normalizer.init(feature_dim)

        neuron_utility: tuple[Array, ...] | None = None
        if self._track_neuron_utility:
            neuron_utility = tuple(
                jnp.zeros(layer_sizes[i + 1], dtype=jnp.float32)
                for i in range(len(layer_sizes) - 2)  # hidden layers only
            )

        return MLPLearnerState(
            params=params,
            optimizer_states=tuple(opt_states_list),
            traces=tuple(traces_list),
            normalizer_state=normalizer_state,
            neuron_utility=neuron_utility,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @staticmethod
    def _forward(
        weights: tuple[Array, ...],
        biases: tuple[Array, ...],
        observation: Array,
        leaky_relu_slope: float,
        use_layer_norm: bool = True,
    ) -> Array:
        """Pure forward pass for use with jax.grad.

        Args:
            weights: Tuple of weight matrices
            biases: Tuple of bias vectors
            observation: Input feature vector
            leaky_relu_slope: Negative slope for LeakyReLU
            use_layer_norm: Whether to apply parameterless layer normalization

        Returns:
            Scalar prediction
        """
        x = observation
        num_layers = len(weights)
        for i in range(num_layers - 1):
            x = weights[i] @ x + biases[i]
            if use_layer_norm:
                # Parameterless layer normalization
                mean = jnp.mean(x)
                var = jnp.var(x)
                x = (x - mean) / jnp.sqrt(var + 1e-5)
            # LeakyReLU
            x = jnp.where(x >= 0, x, leaky_relu_slope * x)
        # Output layer (no activation)
        x = weights[-1] @ x + biases[-1]
        return jnp.squeeze(x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: MLPLearnerState, observation: Observation) -> Prediction:
        """Compute prediction for an observation.

        JIT-compiled automatically. First call triggers tracing; subsequent
        calls with the same learner instance use the cached compilation.

        Args:
            state: Current MLP learner state
            observation: Input feature vector

        Returns:
            Scalar prediction
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        y = self._forward(
            state.params.weights,
            state.params.biases,
            obs,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )
        return jnp.atleast_1d(y)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: MLPLearnerState,
        observation: Observation,
        target: Target,
    ) -> MLPUpdateResult:
        """Update MLP given observation and target.

        JIT-compiled automatically. Performs one step of the learning
        algorithm:

        1. Optionally normalize observation
        2. Compute prediction and error
        3. Compute gradients via jax.grad on the forward pass
        4. Update eligibility traces
        5. Per-layer optimizer step from traces
        6. Optionally bound steps
        7. Apply bounded weight updates

        Args:
            state: Current MLP learner state
            observation: Input feature vector
            target: Desired output

        Returns:
            MLPUpdateResult with new state, prediction, error, and metrics
        """
        feature_dim = self._require_state_contract(state)
        checked_observation = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        checked_target = _require_real_source(
            target,
            name="target",
            shapes=((), (1,)),
        )
        source_state_valid = self.state_is_valid(state)
        input_valid = jnp.all(jnp.isfinite(checked_observation)) & jnp.all(
            jnp.isfinite(checked_target)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        target_scalar = jnp.squeeze(checked_target)

        obs = checked_observation
        new_normalizer_state = state.normalizer_state
        if self._normalizer is not None and state.normalizer_state is not None:
            obs, new_normalizer_state = self._normalizer.normalize(
                state.normalizer_state, observation
            )

        # Forward pass for prediction
        prediction_val = self._forward(
            state.params.weights,
            state.params.biases,
            obs,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )
        prediction = jnp.atleast_1d(prediction_val)
        error = target_scalar - prediction_val

        # Compute gradients w.r.t. prediction
        slope = self._leaky_relu_slope
        ln = self._use_layer_norm

        def pred_fn(weights: tuple[Array, ...], biases: tuple[Array, ...]) -> Array:
            return self._forward(weights, biases, obs, slope, ln)

        weight_grads, bias_grads = jax.grad(pred_fn, argnums=(0, 1))(
            state.params.weights, state.params.biases
        )

        # Update eligibility traces: z = gamma * lamda * z + grad
        gamma_lamda = jnp.array(self._gamma * self._lamda, dtype=jnp.float32)
        n_layers = len(state.params.weights)

        new_traces = []
        for i in range(n_layers):
            # Weight trace (index 2*i)
            new_wt = gamma_lamda * state.traces[2 * i] + weight_grads[i]
            new_traces.append(new_wt)
            # Bias trace (index 2*i + 1)
            new_bt = gamma_lamda * state.traces[2 * i + 1] + bias_grads[i]
            new_traces.append(new_bt)

        # Per-parameter optimizer step from traces
        # Output layer uses head_optimizer if set (last 2 entries: weight + bias)
        n_trace_entries = len(new_traces)
        all_steps = []
        new_opt_states = []
        for j in range(n_trace_entries):
            is_output = self._head_optimizer is not None and j >= n_trace_entries - 2
            opt = self._head_optimizer if is_output else self._optimizer
            step, new_opt = opt.update_from_gradient(
                state.optimizer_states[j], new_traces[j], error=error
            )
            all_steps.append(step)
            new_opt_states.append(new_opt)

        # Bounding (optional)
        bounding_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            all_params = []
            for i in range(n_layers):
                all_params.append(state.params.weights[i])
                all_params.append(state.params.biases[i])
            bounded_steps, bounding_metric = self._bounder.bound(
                tuple(all_steps), error, tuple(all_params)
            )
            all_steps = list(bounded_steps)

        new_weights = []
        new_biases = []
        for i in range(n_layers):
            new_w = state.params.weights[i] + error * all_steps[2 * i]
            new_weights.append(new_w)
            new_b = state.params.biases[i] + error * all_steps[2 * i + 1]
            new_biases.append(new_b)

        new_params = MLPParams(
            weights=tuple(new_weights), biases=tuple(new_biases)
        )

        # Per-hidden-unit gradient utility: EMA of row-wise L2 norm of weight gradients
        new_neuron_utility: tuple[Array, ...] | None = state.neuron_utility
        if state.neuron_utility is not None:
            decay = jnp.array(self._neuron_utility_decay, dtype=jnp.float32)
            new_neuron_utility = tuple(
                decay * state.neuron_utility[i]
                + (1.0 - decay) * jnp.sqrt(jnp.sum(weight_grads[i] ** 2, axis=1) + 1e-12)
                for i in range(len(self._hidden_sizes))
            )

        candidate_state = MLPLearnerState(
            params=new_params,
            optimizer_states=tuple(new_opt_states),
            traces=tuple(new_traces),
            normalizer_state=new_normalizer_state,
            neuron_utility=new_neuron_utility,
            step_count=_words_to_saturating_int32(proposed_words),
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
            step_words=proposed_words,
        )

        squared_error = error**2

        if self._normalizer is not None and new_normalizer_state is not None:
            normalizer_mean_var = jnp.mean(new_normalizer_state.var)
            candidate_metrics = jnp.array(
                [squared_error, error, bounding_metric, normalizer_mean_var],
                dtype=jnp.float32,
            )
        else:
            candidate_metrics = jnp.array(
                [squared_error, error, bounding_metric], dtype=jnp.float32
            )

        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = _select_state(candidate_state, state, update_applied)
        metrics = jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics))

        return MLPUpdateResult(
            state=new_state,
            prediction=prediction,
            error=jnp.atleast_1d(error),
            metrics=metrics,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=_lifetime_counter_valid(
                state.step_words, state.step_count
            ),
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


def run_mlp_learning_loop[StreamStateT](
    learner: MLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: MLPLearnerState | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> (
    tuple[MLPLearnerState, Array]
    | tuple[MLPLearnerState, Array, NormalizerHistory]
):
    """Run the MLP learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.

    Args:
        learner: The MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream and weight initialization
        learner_state: Initial state (if None, will be initialized from stream)
        normalizer_tracking: Optional config for recording per-feature normalizer
            state. When provided, returns NormalizerHistory.

    Returns:
        If no tracking:
            Tuple of (final_state, metrics_array) where metrics_array has shape
            (num_steps, 3) or (num_steps, 4)
        If normalizer_tracking:
            Tuple of (final_state, metrics_array, normalizer_history)

    Raises:
        ValueError: If normalizer_tracking.interval is invalid
    """
    # Validate tracking config
    if normalizer_tracking is not None:
        if normalizer_tracking.interval < 1:
            raise ValueError(
                f"normalizer_tracking.interval must be >= 1, got {normalizer_tracking.interval}"
            )
        if normalizer_tracking.interval > num_steps:
            raise ValueError(
                f"normalizer_tracking.interval ({normalizer_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    # Split key for initialization
    stream_key, init_key = jax.random.split(key)

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim, init_key)
    stream_state = stream.init(stream_key)

    feature_dim = stream.feature_dim

    if normalizer_tracking is None:
        # Simple case without tracking
        def step_fn(
            carry: tuple[MLPLearnerState, StreamStateT], idx: Array
        ) -> tuple[tuple[MLPLearnerState, StreamStateT], Array]:
            l_state, s_state = carry
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)
            return (result.state, new_s_state), result.metrics

        t0 = time.time()
        (final_learner, _), metrics = jax.lax.scan(
            step_fn, (learner_state, stream_state), jnp.arange(num_steps)
        )
        elapsed = time.time() - t0
        final_learner = final_learner.replace(uptime_s=final_learner.uptime_s + elapsed)  # type: ignore[attr-defined]

        return final_learner, metrics

    # Tracking enabled
    norm_interval = normalizer_tracking.interval
    norm_num_recordings = num_steps // norm_interval

    norm_means = jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
    norm_vars = jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
    norm_rec_indices = jnp.zeros(norm_num_recordings, dtype=jnp.int32)

    def step_fn_with_tracking(
        carry: tuple[MLPLearnerState, StreamStateT, Array, Array, Array],
        idx: Array,
    ) -> tuple[
        tuple[MLPLearnerState, StreamStateT, Array, Array, Array],
        Array,
    ]:
        l_state, s_state, n_means, n_vars, n_rec = carry

        # Perform learning step
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)

        # Normalizer state tracking
        should_record = (idx % norm_interval) == 0
        recording_idx = idx // norm_interval

        norm_state = result.state.normalizer_state
        new_n_means = _maybe_record(should_record, n_means, recording_idx, norm_state.mean)
        new_n_vars = _maybe_record(should_record, n_vars, recording_idx, norm_state.var)
        new_n_rec = _maybe_record(should_record, n_rec, recording_idx, idx)

        return (  # type: ignore[return-value]  # n_* are non-None when tracking is enabled
            result.state,
            new_s_state,
            new_n_means,
            new_n_vars,
            new_n_rec,
        ), result.metrics

    initial_carry = (
        learner_state,
        stream_state,
        norm_means,
        norm_vars,
        norm_rec_indices,
    )

    t0 = time.time()
    (
        (final_learner, _, final_n_means, final_n_vars, final_n_rec),
        metrics,
    ) = jax.lax.scan(step_fn_with_tracking, initial_carry, jnp.arange(num_steps))
    elapsed = time.time() - t0
    final_learner = final_learner.replace(uptime_s=final_learner.uptime_s + elapsed)  # type: ignore[attr-defined]

    norm_history = NormalizerHistory(
        means=final_n_means,
        variances=final_n_vars,
        recording_indices=final_n_rec,
    )

    return final_learner, metrics, norm_history


def run_mlp_learning_loop_batched[StreamStateT](
    learner: MLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: MLPLearnerState | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> BatchedMLPResult:
    """Run MLP learning loop across multiple seeds in parallel using jax.vmap.

    All seeds share one compiled program, so multi-seed MLP experiments
    parallelize on accelerators instead of running sequentially.

    Args:
        learner: The MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.
        normalizer_tracking: Optional config for recording normalizer state.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)

    Returns:
        BatchedMLPResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, num_cols)
            - normalizer_history: Batched history or None if tracking disabled

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework import MLPLearner, RandomWalkStream
    from alberta_framework import run_mlp_learning_loop_batched

    stream = RandomWalkStream(feature_dim=10)
    learner = MLPLearner(hidden_sizes=(128, 128))

    # Run 30 seeds in parallel
    keys = jr.split(jr.key(42), 30)
    result = run_mlp_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)

    # result.metrics has shape (30, 10000, 3)
    mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average over seeds
    ```
    """

    def single_seed_run(
        key: Array,
    ) -> tuple[MLPLearnerState, Array, NormalizerHistory | None]:
        result = run_mlp_learning_loop(
            learner, stream, num_steps, key, learner_state, normalizer_tracking
        )

        if normalizer_tracking is not None:
            state, metrics, norm_history = cast(
                tuple[MLPLearnerState, Array, NormalizerHistory], result
            )
            return state, metrics, norm_history
        else:
            state, metrics = cast(tuple[MLPLearnerState, Array], result)
            return state, metrics, None

    t0 = time.time()
    batched_states, batched_metrics, batched_norm_history = jax.vmap(single_seed_run)(keys)
    elapsed = time.time() - t0
    batched_states = batched_states.replace(  # type: ignore[attr-defined]
        uptime_s=batched_states.uptime_s + elapsed
    )

    if normalizer_tracking is not None and batched_norm_history is not None:
        batched_normalizer_history = NormalizerHistory(
            means=batched_norm_history.means,
            variances=batched_norm_history.variances,
            recording_indices=batched_norm_history.recording_indices,
        )
    else:
        batched_normalizer_history = None

    return BatchedMLPResult(
        states=batched_states,
        metrics=batched_metrics,
        normalizer_history=batched_normalizer_history,
    )


# =============================================================================
# TD Learning (for Step 3+ of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class TDUpdateResult:
    """Result of a TD learner update step.

    Attributes:
        state: Updated TD learner state
        prediction: Value prediction V(s) before update
        td_error: TD error delta = R + gamma*V(s') - V(s)
        metrics: Array of metrics [squared_td_error, td_error, mean_step_size, ...]
    """

    state: TDLearnerState
    prediction: Prediction
    td_error: Float[Array, ""]
    metrics: Float[Array, " 4"]
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class TDLinearLearner:
    """Linear function approximator for TD learning.

    Computes value predictions as: ``V(s) = w @ phi(s) + b``

    The learner maintains weights, bias, and eligibility traces, delegating
    the adaptation of learning rates to the TD optimizer (e.g., TDIDBD).

    This follows the Alberta Plan philosophy of temporal uniformity:
    every component updates at every time step.

    Reference: Kearney et al. 2019, "Learning Feature Relevance Through Step Size
    Adaptation in Temporal-Difference Learning"

    Attributes:
        optimizer: The TD optimizer to use for weight updates
    """

    def __init__(self, optimizer: AnyTDOptimizer | None = None):
        """Initialize the TD linear learner.

        Args:
            optimizer: TD optimizer for weight updates. Defaults to TDIDBD()
        """
        self._optimizer: AnyTDOptimizer = optimizer or TDIDBD()
        if not isinstance(self._optimizer, TDIDBD | AutoTDIDBD):
            raise TypeError("optimizer must be TDIDBD or AutoTDIDBD")

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict versioned TD learner configuration."""

        return {
            "schema": TD_LINEAR_LEARNER_CONFIG_SCHEMA,
            "type": "TDLinearLearner",
            "state_schema": TD_LINEAR_LEARNER_STATE_SCHEMA,
            "optimizer": _td_optimizer_to_config(self._optimizer),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TDLinearLearner":
        """Reconstruct only the exact current TD learner schema."""

        if not isinstance(config, Mapping):
            raise TypeError("TD learner config must be a mapping")
        payload = dict(config)
        expected = {"schema", "type", "state_schema", "optimizer"}
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy TD learner config requires explicit migration")
            raise ValueError("TD learner config fields do not match v2")
        if payload.pop("schema") != TD_LINEAR_LEARNER_CONFIG_SCHEMA:
            raise ValueError("TD learner config schema is unsupported")
        if payload.pop("type") != "TDLinearLearner":
            raise ValueError("TD learner config type is unsupported")
        if payload.pop("state_schema") != TD_LINEAR_LEARNER_STATE_SCHEMA:
            raise ValueError("TD learner state schema is unsupported")
        return cls(optimizer=_td_optimizer_from_config(payload.pop("optimizer")))

    def _require_state_contract(self, state: TDLearnerState) -> int:
        if not isinstance(state, TDLearnerState):
            if isinstance(state, LegacyTDLearnerState):
                raise TypeError("legacy TD state requires explicit migration")
            raise TypeError("state must be a TDLearnerState")
        weights = jnp.asarray(state.weights)
        if weights.ndim != 1 or weights.shape[0] <= 0:
            raise ValueError("state.weights must be a nonempty vector")
        feature_dim = weights.shape[0]
        _require_array_contract(
            state.weights,
            name="state.weights",
            shape=(feature_dim,),
            dtype=jnp.float32,
        )
        _require_array_contract(state.bias, name="state.bias", shape=(), dtype=jnp.float32)
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
        _require_tree_contract(
            state.optimizer_state,
            self._optimizer.init(feature_dim),
            name="state.optimizer_state",
        )
        return feature_dim

    def state_is_valid(self, state: TDLearnerState) -> Bool[Array, ""]:
        """Return dynamic validity after enforcing the static state contract."""

        self._require_state_contract(state)
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (jnp.asarray(state.uptime_s) >= 0.0)
        )

    def resource_budget(self, feature_dim: int) -> LearnerResourceBudget:
        """Return exact initialized-state accounting for this TD learner."""

        state = self.init(feature_dim)
        return LearnerResourceBudget(
            learner_type="TDLinearLearner",
            feature_dim=feature_dim,
            trainable_float32_scalars=feature_dim + 1,
            optimizer_state_nbytes=_tree_nbytes(state.optimizer_state),
            normalizer_state_nbytes=0,
            trace_state_nbytes=0,
            utility_state_nbytes=0,
            lifecycle_counter_nbytes=LEARNER_LIFETIME_COUNTER_NBYTES,
            state_nbytes=_tree_nbytes(state),
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_updates_per_call=1,
            replay_capacity=0,
        )

    def init(self, feature_dim: int) -> TDLearnerState:
        """Initialize TD learner state.

        Args:
            feature_dim: Dimension of the input feature vector

        Returns:
            Initial TD learner state with zero weights and bias
        """
        _require_positive_feature_dim(feature_dim)
        optimizer_state = self._optimizer.init(feature_dim)

        return TDLearnerState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            optimizer_state=optimizer_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def predict(self, state: TDLearnerState, observation: Observation) -> Prediction:
        """Compute value prediction for an observation.

        Args:
            state: Current TD learner state
            observation: Input feature vector phi(s)

        Returns:
            Scalar value prediction ``V(s) = w @ phi(s) + b``
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        return jnp.atleast_1d(jnp.dot(state.weights, obs) + state.bias)

    def update(
        self,
        state: TDLearnerState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
    ) -> TDUpdateResult:
        """Update learner given a TD transition.

        Performs one step of TD learning:
        1. Compute V(s) and V(s')
        2. Compute TD error delta = R + gamma*V(s') - V(s)
        3. Get weight updates from TD optimizer
        4. Apply updates to weights and bias

        Args:
            state: Current TD learner state
            observation: Current observation phi(s)
            reward: Reward R received
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal states)

        Returns:
            TDUpdateResult with new state, prediction, TD error, and metrics
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        next_obs = _require_real_source(
            next_observation,
            name="next_observation",
            shapes=((feature_dim,),),
        )
        checked_reward = _require_real_source(
            reward,
            name="reward",
            shapes=((), (1,)),
        )
        checked_gamma = _require_real_source(
            gamma,
            name="gamma",
            shapes=((), (1,)),
        )
        source_state_valid = self.state_is_valid(state)
        gamma_scalar = jnp.squeeze(checked_gamma)
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.isfinite(checked_reward))
            & jnp.isfinite(gamma_scalar)
            & (gamma_scalar >= 0.0)
            & (gamma_scalar <= 1.0)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        prediction = jnp.atleast_1d(jnp.dot(state.weights, obs) + state.bias)
        next_prediction = jnp.atleast_1d(jnp.dot(state.weights, next_obs) + state.bias)

        td_error = (
            jnp.squeeze(checked_reward)
            + gamma_scalar * jnp.squeeze(next_prediction)
            - jnp.squeeze(prediction)
        )

        opt_update = self._optimizer.update(
            state.optimizer_state,
            td_error,
            obs,
            next_obs,
            checked_gamma,
        )

        new_weights = state.weights + opt_update.weight_delta
        new_bias = state.bias + opt_update.bias_delta

        candidate_state = TDLearnerState(
            weights=new_weights,
            bias=new_bias,
            optimizer_state=opt_update.new_state,
            step_count=_words_to_saturating_int32(proposed_words),
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
            step_words=proposed_words,
        )

        # Pack metrics as array for scan compatibility
        squared_td_error = td_error**2
        mean_step_size = opt_update.metrics.get("mean_step_size", 0.0)
        mean_elig_trace = opt_update.metrics.get("mean_eligibility_trace", 0.0)
        candidate_metrics = jnp.array(
            [squared_td_error, td_error, mean_step_size, mean_elig_trace],
            dtype=jnp.float32,
        )

        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = _select_state(candidate_state, state, update_applied)
        metrics = jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics))

        return TDUpdateResult(
            state=new_state,
            prediction=prediction,
            td_error=jnp.atleast_1d(td_error),
            metrics=metrics,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=_lifetime_counter_valid(
                state.step_words, state.step_count
            ),
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


@chex.dataclass(frozen=True)
class TrueOnlineTDState:
    """State for True Online TD(lambda) with an exact lifetime identity."""

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    v_old: Float[Array, ""]
    step_count: Array = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class TrueOnlineTDUpdateResult:
    """Result of one True Online TD(lambda) update.

    ``metrics`` columns: squared TD error, TD error, mean absolute
    eligibility trace, and the stored ``v_old`` bootstrap value.
    """

    state: TrueOnlineTDState
    prediction: Prediction
    next_prediction: Prediction
    td_error: Float[Array, ""]
    metrics: Float[Array, " 4"]
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class TrueOnlineTDLearner:
    """Linear True Online TD(lambda) learner with Dutch traces.

    Implements the true-online update for a linear value function with an
    explicit bias feature. At terminal transitions (``gamma == 0``),
    ``v_old`` is reset to zero so repeated one-step supervised transitions
    reduce exactly to LMS.

    References:
        van Seijen & Sutton (2014). "True Online TD(lambda)." ICML.
        van Seijen et al. (2016). "True Online Temporal-Difference
            Learning." JMLR.
    """

    def __init__(self, step_size: float = 0.05, trace_decay: float = 0.9):
        """Initialize the learner."""
        _require_finite_config_float(
            step_size,
            name="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite_config_float(
            trace_decay,
            name="trace_decay",
            minimum=0.0,
            maximum=1.0,
        )
        self._step_size = step_size
        self._trace_decay = trace_decay

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict versioned learner configuration."""

        return {
            "schema": TRUE_ONLINE_TD_CONFIG_SCHEMA,
            "type": "TrueOnlineTDLearner",
            "state_schema": TRUE_ONLINE_TD_STATE_SCHEMA,
            "step_size": self._step_size,
            "trace_decay": self._trace_decay,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TrueOnlineTDLearner":
        """Reconstruct only the exact current configuration schema."""

        if not isinstance(config, Mapping):
            raise TypeError("true-online TD config must be a mapping")
        payload = dict(config)
        expected = {"schema", "type", "state_schema", "step_size", "trace_decay"}
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy true-online TD config requires explicit migration")
            raise ValueError("true-online TD config fields do not match v2")
        if payload.pop("schema") != TRUE_ONLINE_TD_CONFIG_SCHEMA:
            raise ValueError("true-online TD config schema is unsupported")
        if payload.pop("type") != "TrueOnlineTDLearner":
            raise ValueError("true-online TD config type is unsupported")
        if payload.pop("state_schema") != TRUE_ONLINE_TD_STATE_SCHEMA:
            raise ValueError("true-online TD state schema is unsupported")
        return cls(**payload)

    def _require_state_contract(self, state: TrueOnlineTDState) -> int:
        if not isinstance(state, TrueOnlineTDState):
            raise TypeError("state must be a TrueOnlineTDState")
        weights = jnp.asarray(state.weights)
        if weights.ndim != 1 or weights.shape[0] <= 0:
            raise ValueError("state.weights must be a nonempty vector")
        feature_dim = weights.shape[0]
        fields = {
            "weights": (state.weights, (feature_dim,), jnp.float32),
            "bias": (state.bias, (), jnp.float32),
            "eligibility_traces": (
                state.eligibility_traces,
                (feature_dim,),
                jnp.float32,
            ),
            "bias_eligibility_trace": (
                state.bias_eligibility_trace,
                (),
                jnp.float32,
            ),
            "v_old": (state.v_old, (), jnp.float32),
            "step_count": (state.step_count, (), jnp.int32),
            "step_words": (state.step_words, (2,), jnp.uint32),
        }
        for name, (value, shape, dtype) in fields.items():
            _require_array_contract(
                value,
                name=f"state.{name}",
                shape=shape,
                dtype=dtype,
            )
        return feature_dim

    def state_is_valid(self, state: TrueOnlineTDState) -> Bool[Array, ""]:
        """Return dynamic validity after enforcing the static state contract."""

        self._require_state_contract(state)
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (jnp.asarray(state.uptime_s) >= 0.0)
        )

    def resource_budget(self, feature_dim: int) -> LearnerResourceBudget:
        """Return exact initialized-state accounting for True Online TD."""

        state = self.init(feature_dim)
        return LearnerResourceBudget(
            learner_type="TrueOnlineTDLearner",
            feature_dim=feature_dim,
            trainable_float32_scalars=feature_dim + 1,
            optimizer_state_nbytes=0,
            normalizer_state_nbytes=0,
            trace_state_nbytes=_tree_nbytes(
                (state.eligibility_traces, state.bias_eligibility_trace, state.v_old)
            ),
            utility_state_nbytes=0,
            lifecycle_counter_nbytes=LEARNER_LIFETIME_COUNTER_NBYTES,
            state_nbytes=_tree_nbytes(state),
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_updates_per_call=1,
            replay_capacity=0,
        )

    def init(self, feature_dim: int) -> TrueOnlineTDState:
        """Initialize learner state."""
        _require_positive_feature_dim(feature_dim)
        return TrueOnlineTDState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            v_old=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: TrueOnlineTDState, observation: Observation) -> Prediction:
        """Compute scalar value prediction."""
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        return jnp.atleast_1d(jnp.dot(state.weights, obs) + state.bias)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: TrueOnlineTDState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
    ) -> TrueOnlineTDUpdateResult:
        """Apply one True Online TD(lambda) update."""
        feature_dim = self._require_state_contract(state)
        obs = _require_real_source(
            observation,
            name="observation",
            shapes=((feature_dim,),),
        )
        next_obs = _require_real_source(
            next_observation,
            name="next_observation",
            shapes=((feature_dim,),),
        )
        checked_reward = _require_real_source(
            reward,
            name="reward",
            shapes=((), (1,)),
        )
        checked_gamma = _require_real_source(
            gamma,
            name="gamma",
            shapes=((), (1,)),
        )
        source_state_valid = self.state_is_valid(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        alpha = jnp.asarray(self._step_size, dtype=jnp.float32)
        lamda = jnp.asarray(self._trace_decay, dtype=jnp.float32)
        gamma_scalar = jnp.squeeze(checked_gamma)
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.isfinite(checked_reward))
            & jnp.isfinite(gamma_scalar)
            & (gamma_scalar >= 0.0)
            & (gamma_scalar <= 1.0)
        )

        value = jnp.dot(state.weights, obs) + state.bias
        next_value = jnp.dot(state.weights, next_obs) + state.bias
        td_error = jnp.squeeze(checked_reward) + gamma_scalar * next_value - value

        trace_dot = jnp.dot(state.eligibility_traces, obs)
        trace_dot = trace_dot + state.bias_eligibility_trace
        trace_scale = 1.0 - alpha * gamma_scalar * lamda * trace_dot
        new_traces = gamma_scalar * lamda * state.eligibility_traces + trace_scale * obs
        new_bias_trace = gamma_scalar * lamda * state.bias_eligibility_trace + trace_scale

        correction = value - state.v_old
        update_scale = alpha * (td_error + correction)
        new_weights = (
            state.weights
            + update_scale * new_traces
            - alpha * correction * obs
        )
        new_bias = (
            state.bias
            + update_scale * new_bias_trace
            - alpha * correction
        )

        terminal = gamma_scalar == 0.0
        stored_traces = jnp.where(terminal, jnp.zeros_like(new_traces), new_traces)
        stored_bias_trace = jnp.where(terminal, 0.0, new_bias_trace)
        new_v_old = jnp.where(terminal, 0.0, next_value)
        candidate_state = TrueOnlineTDState(
            weights=new_weights,
            bias=new_bias,
            eligibility_traces=stored_traces,
            bias_eligibility_trace=stored_bias_trace,
            v_old=new_v_old,
            step_count=_words_to_saturating_int32(proposed_words),
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
            step_words=proposed_words,
        )
        candidate_metrics = jnp.array(
            [
                td_error**2,
                td_error,
                jnp.mean(jnp.abs(new_traces)),
                new_v_old,
            ],
            dtype=jnp.float32,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = _select_state(candidate_state, state, update_applied)
        metrics = jnp.where(update_applied, candidate_metrics, jnp.zeros_like(candidate_metrics))
        return TrueOnlineTDUpdateResult(
            state=new_state,
            prediction=jnp.atleast_1d(value),
            next_prediction=jnp.atleast_1d(next_value),
            td_error=jnp.atleast_1d(td_error),
            metrics=metrics,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=_lifetime_counter_valid(
                state.step_words, state.step_count
            ),
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


class TDStream(Protocol[StateT]):
    """Protocol for TD experience streams.

    TD streams produce (s, r, s', gamma) tuples for temporal-difference learning.
    """

    feature_dim: int

    def init(self, key: Array) -> StateT:
        """Initialize stream state."""
        ...

    def step(self, state: StateT, idx: Array) -> tuple[TDTimeStep, StateT]:
        """Generate next TD transition."""
        ...


def run_td_learning_loop[StreamStateT](
    learner: TDLinearLearner,
    stream: TDStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: TDLearnerState | None = None,
) -> tuple[TDLearnerState, Array]:
    """Run the TD learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.
    It returns metrics as a fixed-size array rather than a list of dicts.

    Args:
        learner: The TD learner to train
        stream: TD experience stream providing (s, r, s', gamma) tuples
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)

    Returns:
        Tuple of (final_state, metrics_array) where metrics_array has shape
        (num_steps, 4) with columns [squared_td_error, td_error, mean_step_size,
        mean_eligibility_trace]
    """
    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(
        carry: tuple[TDLearnerState, StreamStateT], idx: Array
    ) -> tuple[tuple[TDLearnerState, StreamStateT], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(
            l_state,
            timestep.observation,
            timestep.reward,
            timestep.next_observation,
            timestep.gamma,
        )
        return (result.state, new_s_state), result.metrics

    t0 = time.time()
    (final_learner, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )
    elapsed = time.time() - t0
    final_learner = final_learner.replace(uptime_s=final_learner.uptime_s + elapsed)  # type: ignore[attr-defined]

    return final_learner, metrics


def run_true_online_td_loop[StreamStateT](
    learner: TrueOnlineTDLearner,
    stream: TDStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: TrueOnlineTDState | None = None,
) -> tuple[TrueOnlineTDState, Array]:
    """Run True Online TD(lambda) over a TD stream with ``jax.lax.scan``."""
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(
        carry: tuple[TrueOnlineTDState, StreamStateT],
        idx: Array,
    ) -> tuple[tuple[TrueOnlineTDState, StreamStateT], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(
            l_state,
            timestep.observation,
            timestep.reward,
            timestep.next_observation,
            timestep.gamma,
        )
        return (result.state, new_s_state), result.metrics

    t0 = time.time()
    (final_learner, _), metrics = jax.lax.scan(
        step_fn,
        (learner_state, stream_state),
        jnp.arange(num_steps),
    )
    elapsed = time.time() - t0
    final_learner = final_learner.replace(  # type: ignore[attr-defined]
        uptime_s=final_learner.uptime_s + elapsed
    )
    return final_learner, metrics


def measure_learner_state_nbytes(state: Any) -> int:
    """Measure the exact bytes of all persistent PyTree leaves in ``state``."""

    return _tree_nbytes(state)


def _legacy_state_fields(raw_state: Any, legacy_type: type[Any], *, label: str) -> dict[str, Any]:
    expected = {field.name for field in dataclasses.fields(legacy_type)}
    if type(raw_state) is legacy_type:
        fields = {name: getattr(raw_state, name) for name in expected}
    elif isinstance(raw_state, Mapping):
        fields = dict(raw_state)
    else:
        raise TypeError(f"{label} must be an exact legacy state or mapping")
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(
            f"{label} fields do not match legacy schema; missing={missing}, extra={extra}"
        )
    return fields


def migrate_legacy_linear_learner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the exact schema-less linear learner payload to v2."""

    if not isinstance(config, Mapping):
        raise TypeError("legacy linear learner config must be a mapping")
    payload = dict(config)
    if set(payload) != {"type", "optimizer", "normalizer"}:
        raise ValueError("legacy linear learner config fields are unsupported")
    migrated = {
        "schema": LINEAR_LEARNER_CONFIG_SCHEMA,
        "state_schema": LINEAR_LEARNER_STATE_SCHEMA,
        **payload,
    }
    LinearLearner.from_config(migrated)
    return migrated


def migrate_legacy_mlp_learner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the exact pre-v2 MLP payload without accepting partial records."""

    if not isinstance(config, Mapping):
        raise TypeError("legacy MLP learner config must be a mapping")
    payload = dict(config)
    expected = {
        "type",
        "hidden_sizes",
        "optimizer",
        "bounder",
        "normalizer",
        "head_optimizer",
        "sparsity",
        "leaky_relu_slope",
        "use_layer_norm",
        "gamma",
        "lamda",
        "track_neuron_utility",
        "neuron_utility_decay",
    }
    if set(payload) != expected:
        raise ValueError("legacy MLP learner config fields are unsupported")
    migrated = {
        "schema": MLP_LEARNER_CONFIG_SCHEMA,
        "state_schema": MLP_LEARNER_STATE_SCHEMA,
        **payload,
    }
    MLPLearner.from_config(migrated)
    return migrated


def migrate_legacy_td_linear_learner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the exact schema-less TD-linear learner payload to v2."""

    if not isinstance(config, Mapping):
        raise TypeError("legacy TD learner config must be a mapping")
    payload = dict(config)
    if set(payload) != {"type", "optimizer"}:
        raise ValueError("legacy TD learner config fields are unsupported")
    migrated = {
        "schema": TD_LINEAR_LEARNER_CONFIG_SCHEMA,
        "state_schema": TD_LINEAR_LEARNER_STATE_SCHEMA,
        **payload,
    }
    TDLinearLearner.from_config(migrated)
    return migrated


def migrate_legacy_true_online_td_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the exact schema-less True Online TD payload to v2."""

    if not isinstance(config, Mapping):
        raise TypeError("legacy true-online TD config must be a mapping")
    payload = dict(config)
    if set(payload) != {"type", "step_size", "trace_decay"}:
        raise ValueError("legacy true-online TD config fields are unsupported")
    migrated = {
        "schema": TRUE_ONLINE_TD_CONFIG_SCHEMA,
        "state_schema": TRUE_ONLINE_TD_STATE_SCHEMA,
        **payload,
    }
    TrueOnlineTDLearner.from_config(migrated)
    return migrated


def migrate_legacy_linear_learner_state(
    learner: LinearLearner,
    state: LegacyLearnerState | Mapping[str, Any],
) -> LearnerState:
    """Migrate a structurally exact, nonnegative legacy linear state."""

    if not isinstance(learner, LinearLearner):
        raise TypeError("learner must be a LinearLearner")
    fields = _legacy_state_fields(state, LegacyLearnerState, label="legacy linear state")
    fields["step_words"] = _legacy_counter_words(
        fields["step_count"], label="legacy linear state"
    )
    migrated = LearnerState(**fields)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy linear state is not dynamically valid")
    return migrated


def migrate_legacy_mlp_learner_state(
    learner: MLPLearner,
    state: LegacyMLPLearnerState | Mapping[str, Any],
) -> MLPLearnerState:
    """Migrate a structurally exact, nonnegative legacy MLP state."""

    if not isinstance(learner, MLPLearner):
        raise TypeError("learner must be an MLPLearner")
    fields = _legacy_state_fields(state, LegacyMLPLearnerState, label="legacy MLP state")
    fields["step_words"] = _legacy_counter_words(
        fields["step_count"], label="legacy MLP state"
    )
    migrated = MLPLearnerState(**fields)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy MLP state is not dynamically valid")
    return migrated


def migrate_legacy_td_linear_learner_state(
    learner: TDLinearLearner,
    state: LegacyTDLearnerState | Mapping[str, Any],
) -> TDLearnerState:
    """Migrate a structurally exact, nonnegative legacy TD-linear state."""

    if not isinstance(learner, TDLinearLearner):
        raise TypeError("learner must be a TDLinearLearner")
    fields = _legacy_state_fields(state, LegacyTDLearnerState, label="legacy TD state")
    fields["step_words"] = _legacy_counter_words(
        fields["step_count"], label="legacy TD state"
    )
    migrated = TDLearnerState(**fields)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy TD state is not dynamically valid")
    return migrated


def migrate_legacy_true_online_td_state(
    learner: TrueOnlineTDLearner,
    state: Mapping[str, Any],
) -> TrueOnlineTDState:
    """Migrate the exact mapping form of the pre-v2 True Online TD state."""

    if not isinstance(learner, TrueOnlineTDLearner):
        raise TypeError("learner must be a TrueOnlineTDLearner")
    if not isinstance(state, Mapping):
        raise TypeError("legacy true-online TD state must be a mapping")
    fields = dict(state)
    expected = {field.name for field in dataclasses.fields(TrueOnlineTDState)} - {
        "step_words"
    }
    if set(fields) != expected:
        raise ValueError("legacy true-online TD state fields are unsupported")
    fields["step_words"] = _legacy_counter_words(
        fields["step_count"], label="legacy true-online TD state"
    )
    migrated = TrueOnlineTDState(**fields)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy true-online TD state is not dynamically valid")
    return migrated
