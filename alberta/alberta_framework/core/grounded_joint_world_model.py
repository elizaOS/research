# mypy: disable-error-code="call-arg"
"""Grounded online joint-action prediction from a learned representation.

This deliberately small mechanism-level (L0, see ``RESEARCH_STATUS.md``)
substrate predicts ordinary next-observation coordinates, reward, and
continuation from the learner's current representation and the *observed
joint action* — the (focal, partner) action pair of the hidden-partner
setting (:mod:`alberta_framework.core.integrated_hidden_partner`).
Conditioning on the partner's observed action lets the model attribute
outcome variation to the partner's choice instead of absorbing it as
environment noise.  The target observation width is independent of the
representation width: a 24-dimensional learned state can therefore be
trained against an 8-dimensional raw observation without using a moving
learned-latent target.

The model is a joint-action-indexed affine map.  Its only persistent
feature-indexed leaf is ``GroundedJointWorldModelState.weights``, whose final
axis is always ``representation_dim``.  There is no optimizer, eligibility
trace, replay buffer, task id, phase id, or hidden feature-indexed state, so
feature-identity routing can relocate ``weights`` columns without leaving
stale per-feature state behind.

Every scored transition is predict-before-update.  Grounded targets are
stop-gradient values.  The uniform all-head fit loss updates every prediction
head, while an optional fixed-total per-head weighting controls only the loss
differentiated with respect to the current representation.  Dynamic numeric
validation is fail-closed: an invalid input, state, gradient, or candidate
parameter update leaves the complete immutable state unchanged and emits no
usable representation gradient.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

GROUNDED_JOINT_WORLD_STATE_SCHEMA = "alberta.grounded-joint-world-state.v3"
GROUNDED_JOINT_WORLD_LIFETIME_COUNTER_NBYTES = 12
GROUNDED_JOINT_WORLD_LIFETIME_COUNTER_DELTA_NBYTES = 8
_CHECKPOINT_SCHEMA = "alberta.grounded_joint_world_model.v3"
_LEGACY_CHECKPOINT_SCHEMA = "alberta.grounded_joint_world_model.v2"
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _saturating_int32_increment(value: Array) -> Int[Array, ""]:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    if getattr(words, "shape", None) != (2,):
        raise ValueError("grounded-world lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("grounded-world lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    if getattr(words, "shape", None) != (2,):
        raise ValueError("grounded-world lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("grounded-world lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("grounded-world update_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("grounded-world update_count must have dtype int32")
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _positive_int(value: Any, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_positive(value: Any, *, name: str) -> float:
    """Return one canonical Python float in the finite normal float32 range."""
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if canonical < _FLOAT32_TINY or canonical > _FLOAT32_MAX:
        raise ValueError(f"{name} must lie in the finite normal float32 range")
    return canonical


def _strict_json_float32_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    """Parse an exactly shaped JSON numeric tree without coercing wire types."""

    def validate(current: Any, remaining: tuple[int, ...], path: str) -> Any:
        if not remaining:
            if type(current) not in (int, float):
                raise ValueError(f"{path} must be a JSON number, not a boolean or string")
            numeric = float(current)
            if not math.isfinite(numeric) or abs(numeric) > _FLOAT32_MAX:
                raise ValueError(f"{path} must be finite and representable in float32")
            return numeric
        if type(current) is not list or len(current) != remaining[0]:
            raise ValueError(f"{path} must be a JSON list of length {remaining[0]}")
        return [
            validate(item, remaining[1:], f"{path}[{index}]") for index, item in enumerate(current)
        ]

    validated = validate(value, shape, name)
    result: npt.NDArray[np.float32] = np.asarray(validated, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _static_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def _real_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if (
        jnp.issubdtype(array.dtype, jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.complexfloating)
        or not (
            jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)
        )
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    return jnp.asarray(array, dtype=jnp.float32)


def _action_scalar(value: Any, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or array.dtype != jnp.int32:
        raise ValueError(f"{name} must be a scalar with dtype int32")
    return array


@dataclasses.dataclass(frozen=True)
class GroundedJointWorldModelConfig:
    """Static construction for :class:`GroundedJointWorldModel`.

    ``representation_dim`` is the learned model input width.
    ``target_observation_dim`` is the independent raw next-observation width.
    The target vector is exactly ``[next_observation, reward, discount]``.
    Optional ``representation_loss_weights`` redistribute a fixed total weight
    of ``target_dim`` without changing the uniform all-head parameter fit.
    ``row_bias_only`` retains identical state shapes while freezing exact-zero
    feature weights as a representation-identifiability control.
    """

    representation_dim: int
    target_observation_dim: int
    n_focal_actions: int
    n_partner_actions: int
    step_size: float = 0.05
    initialization_scale: float = 0.05
    max_input_magnitude: float = 1_000.0
    max_parameter_magnitude: float = 1_000.0
    representation_loss_weights: tuple[float, ...] | None = None
    feature_path_mode: Literal["affine", "row_bias_only"] = "affine"

    def __post_init__(self) -> None:
        """Reject dimensions and bounds that break the fixed-state contract."""
        _positive_int(self.representation_dim, name="representation_dim")
        _positive_int(self.target_observation_dim, name="target_observation_dim")
        _positive_int(self.n_focal_actions, name="n_focal_actions")
        _positive_int(self.n_partner_actions, name="n_partner_actions")
        for name in (
            "step_size",
            "initialization_scale",
            "max_input_magnitude",
            "max_parameter_magnitude",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        if self.initialization_scale > self.max_parameter_magnitude:
            raise ValueError("initialization_scale must not exceed max_parameter_magnitude")
        if self.feature_path_mode not in ("affine", "row_bias_only"):
            raise ValueError("feature_path_mode must be 'affine' or 'row_bias_only'")
        weights = self.representation_loss_weights
        if weights is None:
            return
        if not isinstance(weights, tuple):
            raise ValueError("representation_loss_weights must be an immutable tuple")
        if len(weights) != self.target_dim:
            raise ValueError(
                "representation_loss_weights length must equal target_observation_dim + 2"
            )
        canonical: list[float] = []
        for value in weights:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("representation_loss_weights must not contain a boolean")
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError("representation_loss_weights must contain real numbers")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("representation_loss_weights must be finite")
            if numeric < 0.0:
                raise ValueError("representation_loss_weights must be nonnegative")
            if numeric > _FLOAT32_MAX or (numeric != 0.0 and numeric < _FLOAT32_TINY):
                raise ValueError(
                    "nonzero representation_loss_weights must lie in the finite normal "
                    "float32 range"
                )
            canonical.append(numeric)
        if not any(value > 0.0 for value in canonical):
            raise ValueError("representation_loss_weights must contain a positive entry")
        if math.fsum(canonical) != float(self.target_dim):
            raise ValueError("representation_loss_weights must sum to target_dim")
        float32_total = np.sum(np.asarray(canonical, dtype=np.float32), dtype=np.float32)
        if float32_total != np.float32(self.target_dim):
            raise ValueError(
                "representation_loss_weights must sum to target_dim in deterministic float32"
            )
        object.__setattr__(self, "representation_loss_weights", tuple(canonical))

    @property
    def joint_action_count(self) -> int:
        """Exact Cartesian joint-action cardinality."""
        return self.n_focal_actions * self.n_partner_actions

    @property
    def target_dim(self) -> int:
        """Raw next observation plus reward and continuation heads."""
        return self.target_observation_dim + 2

    def to_config(self) -> dict[str, Any]:
        """Return a strict JSON-compatible construction record."""
        payload = dataclasses.asdict(self)
        if self.representation_loss_weights is not None:
            payload["representation_loss_weights"] = list(self.representation_loss_weights)
        payload["type"] = type(self).__name__
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> GroundedJointWorldModelConfig:
        """Strictly reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
        type_name = payload.pop("type")
        if type_name != cls.__name__:
            raise ValueError(f"unexpected config type: {type_name!r}")
        serialized_weights = payload.get("representation_loss_weights")
        if serialized_weights is not None:
            if not isinstance(serialized_weights, list):
                raise ValueError("serialized representation_loss_weights must be a JSON list")
            payload["representation_loss_weights"] = tuple(serialized_weights)
        return cls(**payload)


@chex.dataclass(frozen=True)
class GroundedJointWorldModelState:
    """Fixed-shape immutable parameters and accepted-update counter.

    ``weights`` is the only representation-feature-indexed persistent leaf;
    its representation axis is explicitly last.  A feature router may produce
    an immutable replacement with ``state.replace(weights=routed_weights)``.
    """

    weights: Float[Array, "joint_actions target_dim representation_dim"]
    bias: Float[Array, "joint_actions target_dim"]
    update_count: Int[Array, ""]
    update_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True)
class GroundedJointWorldResourceBudget:
    """Exact allocation, trainable parameters, and logical update surface."""

    representation_dim: int
    target_observation_dim: int
    joint_action_count: int
    target_dim: int
    allocated_float32_scalars: int
    trainable_float32_scalars: int
    administrative_int32_scalars: int
    administrative_uint32_scalars: int
    state_nbytes: int
    computed_parameter_gradient_float32_scalars_per_update: int
    learned_float32_scalars_touched_per_update: int
    applied_trainable_float32_scalars_per_update: int
    administrative_int32_scalars_touched_per_update: int
    administrative_uint32_scalars_touched_per_update: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact resource record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class GroundedJointWorldPrediction:
    """One read-only prediction for a fully observed joint action.

    ``feature_contribution`` and ``row_bias`` are the exact pre-update affine
    terms used to construct ``raw_predictions``.  Keeping the terms separate
    makes representation-conditioned prediction auditable without exposing or
    adding persistent state.
    """

    joint_action_index: Int[Array, ""]
    next_observation: Float[Array, " target_observation_dim"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    feature_contribution: Float[Array, " target_dim"]
    row_bias: Float[Array, " target_dim"]
    raw_predictions: Float[Array, " target_dim"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GroundedJointWorldInputGradient:
    """Pre-update losses and auditable per-head input derivatives.

    ``loss`` is the backward-compatible alias of ``representation_loss``.
    Both ``*_loss_by_head`` arrays are additive contributions, so summing
    their entries reconstructs the corresponding aggregate up to float32
    reduction order.  Summing ``representation_gradient_by_head`` over its
    leading axis reconstructs ``gradient`` under the same qualification.
    """

    prediction: GroundedJointWorldPrediction
    targets: Float[Array, " target_dim"]
    errors: Float[Array, " target_dim"]
    loss: Float[Array, ""]
    representation_loss: Float[Array, ""]
    fit_loss: Float[Array, ""]
    fit_loss_by_head: Float[Array, " target_dim"]
    representation_loss_by_head: Float[Array, " target_dim"]
    gradient: Float[Array, " representation_dim"]
    representation_gradient_by_head: Float[Array, "target_dim representation_dim"]
    representation_gradient_norm_by_head: Float[Array, " target_dim"]
    gradient_norm: Float[Array, ""]
    feature_path_enabled: Bool[Array, ""]
    representation_credit_enabled: Bool[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GroundedJointWorldDiagnostics:
    """Fail-closed verdicts for one attempted transition."""

    state_valid: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    prediction_valid: Bool[Array, ""]
    target_valid: Bool[Array, ""]
    feature_path_enabled: Bool[Array, ""]
    representation_credit_enabled: Bool[Array, ""]
    representation_gradient_valid: Bool[Array, ""]
    parameter_update_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    row_update_isolated: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GroundedJointWorldUpdateResult:
    """Auditable pre-update losses and the atomically selected next state.

    ``parameter_gradient_norm`` remains a compatibility alias for
    ``computed_parameter_gradient_norm``.  In ``row_bias_only`` mode the
    computed norm includes the deliberately dormant feature-weight gradient,
    while ``applied_parameter_gradient_norm`` includes only the bias gradient
    that can actually change state.  The two proposed-row masks compare the
    accepted candidate and entry state by exact float32 bit pattern.  The
    executed-row deltas are likewise emitted only for an accepted atomic
    update; all four fields are neutral on rejection.
    """

    state: GroundedJointWorldModelState
    prediction: GroundedJointWorldPrediction
    targets: Float[Array, " target_dim"]
    errors: Float[Array, " target_dim"]
    loss: Float[Array, ""]
    representation_loss: Float[Array, ""]
    fit_loss: Float[Array, ""]
    fit_loss_by_head: Float[Array, " target_dim"]
    representation_loss_by_head: Float[Array, " target_dim"]
    representation_gradient: Float[Array, " representation_dim"]
    representation_gradient_by_head: Float[Array, "target_dim representation_dim"]
    representation_gradient_norm_by_head: Float[Array, " target_dim"]
    representation_gradient_norm: Float[Array, ""]
    feature_path_enabled: Bool[Array, ""]
    representation_credit_enabled: Bool[Array, ""]
    gradient_valid: Bool[Array, ""]
    parameter_gradient_norm: Float[Array, ""]
    computed_parameter_gradient_norm: Float[Array, ""]
    applied_parameter_gradient_norm: Float[Array, ""]
    proposed_weight_row_bit_change_mask: Bool[Array, " joint_actions"]
    proposed_bias_row_bit_change_mask: Bool[Array, " joint_actions"]
    executed_weight_row_delta_norm_by_head: Float[Array, " target_dim"]
    executed_bias_row_delta_by_head: Float[Array, " target_dim"]
    pre_update_words: UInt[Array, " 2"]
    post_update_words: UInt[Array, " 2"]
    update_applied: Bool[Array, ""]
    diagnostics: GroundedJointWorldDiagnostics


class GroundedJointWorldModel:
    """Linear bounded model for ``(representation, joint action) -> world``."""

    def __init__(self, config: GroundedJointWorldModelConfig):
        self._config = config

    @property
    def config(self) -> GroundedJointWorldModelConfig:
        """Return the immutable static construction."""
        return self._config

    @property
    def joint_action_count(self) -> int:
        """Return ``n_focal_actions * n_partner_actions`` exactly."""
        return self._config.joint_action_count

    @property
    def target_dim(self) -> int:
        """Return ``target_observation_dim + reward + discount``."""
        return self._config.target_dim

    @property
    def effective_representation_loss_weights(self) -> tuple[float, ...]:
        """Return the exact static per-head representation-credit weights."""
        configured = self._config.representation_loss_weights
        if configured is None:
            return (1.0,) * self.target_dim
        return configured

    def _feature_path_enabled(self) -> Array:
        return jnp.asarray(
            self._config.feature_path_mode == "affine",
            dtype=jnp.bool_,
        )

    def _representation_credit_enabled(self) -> Array:
        return self._feature_path_enabled()

    @property
    def representation_indexed_state_leaves(self) -> tuple[tuple[str, int], ...]:
        """Name every persistent feature-indexed leaf and its feature axis."""
        return (("weights", -1),)

    @property
    def resource_budget(self) -> GroundedJointWorldResourceBudget:
        """Return exact fixed-state and selected-row update accounting."""
        cfg = self._config
        allocated = cfg.joint_action_count * cfg.target_dim * (cfg.representation_dim + 1)
        full_row = cfg.target_dim * (cfg.representation_dim + 1)
        if cfg.feature_path_mode == "row_bias_only":
            trainable = cfg.joint_action_count * cfg.target_dim
            applied = cfg.target_dim
        else:
            trainable = allocated
            applied = full_row
        administrative = 1
        exact_words = 2
        return GroundedJointWorldResourceBudget(
            representation_dim=cfg.representation_dim,
            target_observation_dim=cfg.target_observation_dim,
            joint_action_count=cfg.joint_action_count,
            target_dim=cfg.target_dim,
            allocated_float32_scalars=allocated,
            trainable_float32_scalars=trainable,
            administrative_int32_scalars=administrative,
            administrative_uint32_scalars=exact_words,
            state_nbytes=4 * (allocated + administrative + exact_words),
            computed_parameter_gradient_float32_scalars_per_update=full_row,
            learned_float32_scalars_touched_per_update=applied,
            applied_trainable_float32_scalars_per_update=applied,
            administrative_int32_scalars_touched_per_update=1,
            administrative_uint32_scalars_touched_per_update=2,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete model construction."""
        return {
            "type": type(self).__name__,
            "state_schema": GROUNDED_JOINT_WORLD_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> GroundedJointWorldModel:
        """Strictly reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        if set(payload) != {"type", "state_schema", "config"}:
            raise ValueError("model config fields do not match the serialized schema")
        type_name = payload["type"]
        if type_name != cls.__name__:
            raise ValueError(f"unexpected model type: {type_name!r}")
        if payload["state_schema"] != GROUNDED_JOINT_WORLD_STATE_SCHEMA:
            raise ValueError("grounded joint-world state schema is unsupported")
        nested = payload["config"]
        if not isinstance(nested, Mapping):
            raise ValueError("model config must contain a config mapping")
        return cls(GroundedJointWorldModelConfig.from_config(nested))

    def init(self, key: Array) -> GroundedJointWorldModelState:
        """Initialize bounded affine weights or exact-zero row-bias placeholders."""
        cfg = self._config
        weights = jr.uniform(
            key,
            (cfg.joint_action_count, cfg.target_dim, cfg.representation_dim),
            minval=-cfg.initialization_scale,
            maxval=cfg.initialization_scale,
            dtype=jnp.float32,
        )
        if cfg.feature_path_mode == "row_bias_only":
            weights = jnp.zeros_like(weights)
        return GroundedJointWorldModelState(
            weights=weights,
            bias=jnp.zeros(
                (cfg.joint_action_count, cfg.target_dim),
                dtype=jnp.float32,
            ),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            update_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _validate_state_static_contract(self, state: GroundedJointWorldModelState) -> None:
        if not isinstance(state, GroundedJointWorldModelState):
            raise TypeError("state must be a GroundedJointWorldModelState")
        cfg = self._config
        _static_array_contract(
            state.weights,
            name="state.weights",
            shape=(cfg.joint_action_count, cfg.target_dim, cfg.representation_dim),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.bias,
            name="state.bias",
            shape=(cfg.joint_action_count, cfg.target_dim),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.update_count,
            name="state.update_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.update_words,
            name="state.update_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def _state_valid(self, state: GroundedJointWorldModelState) -> Array:
        bound = jnp.asarray(self._config.max_parameter_magnitude, dtype=jnp.float32)
        valid = (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.abs(state.weights) <= bound)
            & jnp.all(jnp.isfinite(state.bias))
            & jnp.all(jnp.abs(state.bias) <= bound)
            & _lifetime_counter_valid(state.update_words, state.update_count)
        )
        if self._config.feature_path_mode == "row_bias_only":
            return valid & jnp.all(state.weights == 0.0)
        return valid

    @functools.partial(jax.jit, static_argnums=(0,))
    def joint_action_index(
        self,
        focal_action: Array,
        partner_action: Array,
    ) -> Int[Array, ""]:
        """Encode the Cartesian joint action in focal-major order."""
        focal = _action_scalar(focal_action, name="focal_action")
        partner = _action_scalar(partner_action, name="partner_action")
        return focal * self._config.n_partner_actions + partner

    @functools.partial(jax.jit, static_argnums=(0,))
    def joint_action_one_hot(
        self,
        focal_action: Array,
        partner_action: Array,
    ) -> Float[Array, " joint_actions"]:
        """Return the exact one-hot Cartesian joint-action encoding."""
        return jax.nn.one_hot(
            self.joint_action_index(focal_action, partner_action),
            self.joint_action_count,
            dtype=jnp.float32,
        )

    def _zero_prediction(self) -> GroundedJointWorldPrediction:
        return GroundedJointWorldPrediction(
            joint_action_index=jnp.asarray(-1, dtype=jnp.int32),
            next_observation=jnp.zeros(
                (self._config.target_observation_dim,),
                dtype=jnp.float32,
            ),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            feature_contribution=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            row_bias=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            raw_predictions=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _predict_unchecked(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
    ) -> GroundedJointWorldPrediction:
        joint_index = self.joint_action_index(focal_action, partner_action)
        feature_contribution = state.weights[joint_index] @ representation
        row_bias = state.bias[joint_index]
        raw = feature_contribution + row_bias
        observation_dim = self._config.target_observation_dim
        valid = (
            jnp.all(jnp.isfinite(feature_contribution))
            & jnp.all(jnp.isfinite(row_bias))
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.abs(raw) <= self._config.max_input_magnitude)
        )
        return GroundedJointWorldPrediction(
            joint_action_index=joint_index,
            next_observation=raw[:observation_dim],
            reward=raw[observation_dim],
            discount=jnp.clip(raw[observation_dim + 1], 0.0, 1.0),
            feature_contribution=feature_contribution,
            row_bias=row_bias,
            raw_predictions=raw,
            valid=valid,
        )

    def _prediction_input_valid(
        self,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
    ) -> Array:
        bound = jnp.asarray(self._config.max_input_magnitude, dtype=jnp.float32)
        return (
            jnp.all(jnp.isfinite(representation))
            & jnp.all(jnp.abs(representation) <= bound)
            & (focal_action >= 0)
            & (focal_action < self._config.n_focal_actions)
            & (partner_action >= 0)
            & (partner_action < self._config.n_partner_actions)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
    ) -> GroundedJointWorldPrediction:
        """Return a fail-closed read-only joint-world prediction."""
        self._validate_state_static_contract(state)
        rep = _real_array(
            representation,
            (self._config.representation_dim,),
            name="representation",
        )
        focal = _action_scalar(focal_action, name="focal_action")
        partner = _action_scalar(partner_action, name="partner_action")
        valid = self._state_valid(state) & self._prediction_input_valid(rep, focal, partner)
        return cast(
            GroundedJointWorldPrediction,
            jax.lax.cond(
                valid,
                lambda _: self._predict_unchecked(state, rep, focal, partner),
                lambda _: self._zero_prediction(),
                operand=None,
            ),
        )

    def _targets_unchecked(
        self,
        next_observation: Array,
        reward: Array,
        discount: Array,
    ) -> Array:
        targets = jnp.concatenate(
            [
                next_observation,
                jnp.reshape(reward, (1,)),
                jnp.reshape(discount, (1,)),
            ]
        )
        return jax.lax.stop_gradient(targets)

    def _target_input_valid(
        self,
        next_observation: Array,
        reward: Array,
        discount: Array,
    ) -> Array:
        bound = jnp.asarray(self._config.max_input_magnitude, dtype=jnp.float32)
        return (
            jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.abs(next_observation) <= bound)
            & jnp.isfinite(reward)
            & (jnp.abs(reward) <= bound)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
        )

    def _zero_score(self) -> GroundedJointWorldInputGradient:
        return GroundedJointWorldInputGradient(
            prediction=self._zero_prediction(),
            targets=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            errors=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            loss=jnp.asarray(0.0, dtype=jnp.float32),
            representation_loss=jnp.asarray(0.0, dtype=jnp.float32),
            fit_loss=jnp.asarray(0.0, dtype=jnp.float32),
            fit_loss_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            representation_loss_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            gradient=jnp.zeros(
                (self._config.representation_dim,),
                dtype=jnp.float32,
            ),
            representation_gradient_by_head=jnp.zeros(
                (self.target_dim, self._config.representation_dim),
                dtype=jnp.float32,
            ),
            representation_gradient_norm_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            feature_path_enabled=self._feature_path_enabled(),
            representation_credit_enabled=self._representation_credit_enabled(),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _score_unchecked(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
        targets: Array,
    ) -> GroundedJointWorldInputGradient:
        def objective(
            current_representation: Array,
        ) -> tuple[Array, tuple[GroundedJointWorldPrediction, Array, Array]]:
            prediction = self._predict_unchecked(
                state,
                current_representation,
                focal_action,
                partner_action,
            )
            errors = prediction.raw_predictions - targets
            fit_loss = 0.5 * jnp.mean(jnp.square(errors))
            if self._config.representation_loss_weights is None:
                loss = fit_loss
            else:
                weights = jnp.asarray(
                    self._config.representation_loss_weights,
                    dtype=jnp.float32,
                )
                loss = 0.5 * jnp.mean(weights * jnp.square(errors))
            return loss, (prediction, errors, fit_loss)

        (loss_and_aux, gradient) = jax.value_and_grad(objective, has_aux=True)(representation)
        loss, (prediction, errors, fit_loss) = loss_and_aux
        divisor = jnp.asarray(self.target_dim, dtype=jnp.float32)
        fit_loss_by_head = 0.5 * jnp.square(errors) / divisor
        representation_weights = jnp.asarray(
            self.effective_representation_loss_weights,
            dtype=jnp.float32,
        )
        representation_loss_by_head = representation_weights * fit_loss_by_head
        selected_weights = state.weights[prediction.joint_action_index]
        representation_gradient_by_head = (
            selected_weights * (representation_weights * errors / divisor)[:, None]
        )
        if self._config.feature_path_mode == "row_bias_only":
            representation_gradient_by_head = jnp.zeros_like(representation_gradient_by_head)
        representation_gradient_norm_by_head = jnp.linalg.norm(
            representation_gradient_by_head,
            axis=1,
        )
        gradient_norm = jnp.linalg.norm(gradient)
        valid = (
            prediction.valid
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.isfinite(errors))
            & jnp.isfinite(loss)
            & (loss >= 0.0)
            & jnp.isfinite(fit_loss)
            & (fit_loss >= 0.0)
            & jnp.all(jnp.isfinite(fit_loss_by_head))
            & jnp.all(jnp.isfinite(representation_loss_by_head))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(representation_gradient_by_head))
            & jnp.all(jnp.isfinite(representation_gradient_norm_by_head))
            & jnp.isfinite(gradient_norm)
        )
        score = GroundedJointWorldInputGradient(
            prediction=prediction,
            targets=targets,
            errors=errors,
            loss=loss,
            representation_loss=loss,
            fit_loss=fit_loss,
            fit_loss_by_head=fit_loss_by_head,
            representation_loss_by_head=representation_loss_by_head,
            gradient=gradient,
            representation_gradient_by_head=representation_gradient_by_head,
            representation_gradient_norm_by_head=representation_gradient_norm_by_head,
            gradient_norm=gradient_norm,
            feature_path_enabled=self._feature_path_enabled(),
            representation_credit_enabled=self._representation_credit_enabled(),
            valid=valid,
        )
        return cast(
            GroundedJointWorldInputGradient,
            jax.lax.cond(valid, lambda _: score, lambda _: self._zero_score(), operand=None),
        )

    def _validate_transition_static_contract(
        self,
        state: GroundedJointWorldModelState,
        representation: Any,
        focal_action: Any,
        partner_action: Any,
        next_observation: Any,
        reward: Any,
        discount: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        self._validate_state_static_contract(state)
        return (
            _real_array(
                representation,
                (self._config.representation_dim,),
                name="representation",
            ),
            _action_scalar(focal_action, name="focal_action"),
            _action_scalar(partner_action, name="partner_action"),
            _real_array(
                next_observation,
                (self._config.target_observation_dim,),
                name="next_observation",
            ),
            _real_array(reward, (), name="reward"),
            _real_array(discount, (), name="discount"),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_loss_gradient(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
        next_observation: Array,
        reward: Array,
        discount: Array,
    ) -> GroundedJointWorldInputGradient:
        """Differentiate the grounded representation loss through its input.

        This method is read-only.  It does not advance parameters or counters,
        and target gradients are stopped before the objective is formed.  Its
        ``fit_loss`` remains the uniform objective used for all parameter heads.
        """
        rep, focal, partner, next_obs, rew, disc = self._validate_transition_static_contract(
            state,
            representation,
            focal_action,
            partner_action,
            next_observation,
            reward,
            discount,
        )
        input_valid = self._prediction_input_valid(rep, focal, partner) & self._target_input_valid(
            next_obs,
            rew,
            disc,
        )
        valid = self._state_valid(state) & input_valid

        def score(_: None) -> GroundedJointWorldInputGradient:
            targets = self._targets_unchecked(next_obs, rew, disc)
            return self._score_unchecked(state, rep, focal, partner, targets)

        return cast(
            GroundedJointWorldInputGradient,
            jax.lax.cond(valid, score, lambda _: self._zero_score(), operand=None),
        )

    def _rejected_update(
        self,
        state: GroundedJointWorldModelState,
        *,
        state_valid: Array,
        lifetime_counter_valid: Array,
        input_valid: Array,
        capacity_available: Array,
    ) -> GroundedJointWorldUpdateResult:
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = GroundedJointWorldDiagnostics(
            state_valid=state_valid,
            lifetime_counter_valid=lifetime_counter_valid,
            input_valid=input_valid,
            capacity_available=capacity_available,
            prediction_valid=false,
            target_valid=false,
            feature_path_enabled=self._feature_path_enabled(),
            representation_credit_enabled=self._representation_credit_enabled(),
            representation_gradient_valid=false,
            parameter_update_valid=false,
            candidate_state_valid=false,
            row_update_isolated=false,
            applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
        )
        return GroundedJointWorldUpdateResult(
            state=state,
            prediction=self._zero_prediction(),
            targets=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            errors=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            loss=jnp.asarray(0.0, dtype=jnp.float32),
            representation_loss=jnp.asarray(0.0, dtype=jnp.float32),
            fit_loss=jnp.asarray(0.0, dtype=jnp.float32),
            fit_loss_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            representation_loss_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            representation_gradient=jnp.zeros(
                (self._config.representation_dim,),
                dtype=jnp.float32,
            ),
            representation_gradient_by_head=jnp.zeros(
                (self.target_dim, self._config.representation_dim),
                dtype=jnp.float32,
            ),
            representation_gradient_norm_by_head=jnp.zeros((self.target_dim,), dtype=jnp.float32),
            representation_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            feature_path_enabled=self._feature_path_enabled(),
            representation_credit_enabled=self._representation_credit_enabled(),
            gradient_valid=false,
            parameter_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            computed_parameter_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            applied_parameter_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            proposed_weight_row_bit_change_mask=jnp.zeros(
                (self.joint_action_count,),
                dtype=jnp.bool_,
            ),
            proposed_bias_row_bit_change_mask=jnp.zeros(
                (self.joint_action_count,),
                dtype=jnp.bool_,
            ),
            executed_weight_row_delta_norm_by_head=jnp.zeros(
                (self.target_dim,),
                dtype=jnp.float32,
            ),
            executed_bias_row_delta_by_head=jnp.zeros(
                (self.target_dim,),
                dtype=jnp.float32,
            ),
            pre_update_words=state.update_words,
            post_update_words=state.update_words,
            update_applied=false,
            diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
        focal_action: Array,
        partner_action: Array,
        next_observation: Array,
        reward: Array,
        discount: Array,
    ) -> GroundedJointWorldUpdateResult:
        """Atomically learn from one bounded predict-before-update transition."""
        rep, focal, partner, next_obs, rew, disc = self._validate_transition_static_contract(
            state,
            representation,
            focal_action,
            partner_action,
            next_observation,
            reward,
            discount,
        )
        proposed_update_words, capacity_available = (
            _checked_lifetime_words_increment(state.update_words)
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.update_words,
            state.update_count,
        )
        state_valid = self._state_valid(state)
        input_valid = self._prediction_input_valid(rep, focal, partner) & self._target_input_valid(
            next_obs,
            rew,
            disc,
        )
        can_attempt = state_valid & input_valid & capacity_available

        def do_update(_: None) -> GroundedJointWorldUpdateResult:
            targets = self._targets_unchecked(next_obs, rew, disc)
            score = self._score_unchecked(state, rep, focal, partner, targets)
            error_gradient = score.errors / jnp.asarray(self.target_dim, dtype=jnp.float32)
            weight_gradient = error_gradient[:, None] * rep[None, :]
            bias_gradient = error_gradient
            parameter_gradient_norm = jnp.sqrt(
                jnp.sum(jnp.square(weight_gradient)) + jnp.sum(jnp.square(bias_gradient))
            )
            if self._config.feature_path_mode == "row_bias_only":
                applied_parameter_gradient_norm = jnp.linalg.norm(bias_gradient)
            else:
                applied_parameter_gradient_norm = parameter_gradient_norm
            parameter_gradient_valid = (
                score.valid
                & jnp.all(jnp.isfinite(weight_gradient))
                & jnp.all(jnp.isfinite(bias_gradient))
                & jnp.isfinite(parameter_gradient_norm)
                & jnp.isfinite(applied_parameter_gradient_norm)
            )
            safe_weight_gradient = jnp.where(parameter_gradient_valid, weight_gradient, 0.0)
            safe_bias_gradient = jnp.where(parameter_gradient_valid, bias_gradient, 0.0)
            joint_index = score.prediction.joint_action_index
            if self._config.feature_path_mode == "row_bias_only":
                candidate_weight_row = state.weights[joint_index]
            else:
                candidate_weight_row = state.weights[joint_index] - (
                    self._config.step_size * safe_weight_gradient
                )
            candidate_bias_row = state.bias[joint_index] - (
                self._config.step_size * safe_bias_gradient
            )
            candidate_state = GroundedJointWorldModelState(
                weights=state.weights.at[joint_index].set(candidate_weight_row),
                bias=state.bias.at[joint_index].set(candidate_bias_row),
                update_count=_saturating_int32_increment(state.update_count),
                update_words=proposed_update_words,
            )
            proposed_weight_row_bit_change_mask = jnp.any(
                jax.lax.bitcast_convert_type(candidate_state.weights, jnp.uint32)
                != jax.lax.bitcast_convert_type(state.weights, jnp.uint32),
                axis=(1, 2),
            )
            proposed_bias_row_bit_change_mask = jnp.any(
                jax.lax.bitcast_convert_type(candidate_state.bias, jnp.uint32)
                != jax.lax.bitcast_convert_type(state.bias, jnp.uint32),
                axis=1,
            )
            executed_weight_row_delta = candidate_weight_row - state.weights[joint_index]
            executed_weight_row_delta_norm_by_head = jnp.linalg.norm(
                executed_weight_row_delta,
                axis=1,
            )
            executed_bias_row_delta_by_head = candidate_bias_row - state.bias[joint_index]
            parameter_update_valid = (
                parameter_gradient_valid
                & jnp.all(jnp.isfinite(executed_weight_row_delta_norm_by_head))
                & jnp.all(jnp.isfinite(executed_bias_row_delta_by_head))
            )
            executed_row_mask = jax.nn.one_hot(
                joint_index,
                self.joint_action_count,
                dtype=jnp.bool_,
            )
            row_update_isolated = ~jnp.any(
                (proposed_weight_row_bit_change_mask | proposed_bias_row_bit_change_mask)
                & ~executed_row_mask
            )
            candidate_state_valid = self._state_valid(candidate_state)
            applied = (
                score.valid
                & parameter_update_valid
                & row_update_isolated
                & candidate_state_valid
            )
            next_state = cast(
                GroundedJointWorldModelState,
                jax.lax.cond(applied, lambda _: candidate_state, lambda _: state, operand=None),
            )
            diagnostics = GroundedJointWorldDiagnostics(
                state_valid=state_valid,
                lifetime_counter_valid=lifetime_counter_valid,
                input_valid=input_valid,
                capacity_available=capacity_available,
                prediction_valid=score.prediction.valid,
                target_valid=jnp.all(jnp.isfinite(targets)),
                feature_path_enabled=self._feature_path_enabled(),
                representation_credit_enabled=self._representation_credit_enabled(),
                representation_gradient_valid=score.valid,
                parameter_update_valid=parameter_update_valid,
                candidate_state_valid=candidate_state_valid,
                row_update_isolated=jnp.where(applied, row_update_isolated, False),
                applied=applied,
                rejected=~applied,
            )
            zero = self._zero_score()
            prediction = cast(
                GroundedJointWorldPrediction,
                jax.lax.cond(
                    applied,
                    lambda _: score.prediction,
                    lambda _: zero.prediction,
                    operand=None,
                ),
            )
            return GroundedJointWorldUpdateResult(
                state=next_state,
                prediction=prediction,
                targets=jnp.where(applied, targets, zero.targets),
                errors=jnp.where(applied, score.errors, zero.errors),
                loss=jnp.where(applied, score.loss, zero.loss),
                representation_loss=jnp.where(
                    applied,
                    score.representation_loss,
                    zero.representation_loss,
                ),
                fit_loss=jnp.where(applied, score.fit_loss, zero.fit_loss),
                fit_loss_by_head=jnp.where(
                    applied,
                    score.fit_loss_by_head,
                    zero.fit_loss_by_head,
                ),
                representation_loss_by_head=jnp.where(
                    applied,
                    score.representation_loss_by_head,
                    zero.representation_loss_by_head,
                ),
                representation_gradient=jnp.where(applied, score.gradient, zero.gradient),
                representation_gradient_by_head=jnp.where(
                    applied,
                    score.representation_gradient_by_head,
                    zero.representation_gradient_by_head,
                ),
                representation_gradient_norm_by_head=jnp.where(
                    applied,
                    score.representation_gradient_norm_by_head,
                    zero.representation_gradient_norm_by_head,
                ),
                representation_gradient_norm=jnp.where(
                    applied,
                    score.gradient_norm,
                    zero.gradient_norm,
                ),
                feature_path_enabled=self._feature_path_enabled(),
                representation_credit_enabled=self._representation_credit_enabled(),
                gradient_valid=applied & score.valid,
                parameter_gradient_norm=jnp.where(applied, parameter_gradient_norm, 0.0),
                computed_parameter_gradient_norm=jnp.where(
                    applied,
                    parameter_gradient_norm,
                    0.0,
                ),
                applied_parameter_gradient_norm=jnp.where(
                    applied,
                    applied_parameter_gradient_norm,
                    0.0,
                ),
                proposed_weight_row_bit_change_mask=jnp.where(
                    applied,
                    proposed_weight_row_bit_change_mask,
                    jnp.zeros_like(proposed_weight_row_bit_change_mask),
                ),
                proposed_bias_row_bit_change_mask=jnp.where(
                    applied,
                    proposed_bias_row_bit_change_mask,
                    jnp.zeros_like(proposed_bias_row_bit_change_mask),
                ),
                executed_weight_row_delta_norm_by_head=jnp.where(
                    applied,
                    executed_weight_row_delta_norm_by_head,
                    0.0,
                ),
                executed_bias_row_delta_by_head=jnp.where(
                    applied,
                    executed_bias_row_delta_by_head,
                    0.0,
                ),
                pre_update_words=state.update_words,
                post_update_words=next_state.update_words,
                update_applied=applied,
                diagnostics=diagnostics,
            )

        return cast(
            GroundedJointWorldUpdateResult,
            jax.lax.cond(
                can_attempt,
                do_update,
                lambda _: self._rejected_update(
                    state,
                    state_valid=state_valid,
                    lifetime_counter_valid=lifetime_counter_valid,
                    input_valid=input_valid,
                    capacity_available=capacity_available,
                ),
                operand=None,
            ),
        )

    def checkpoint_payload(
        self,
        state: GroundedJointWorldModelState,
    ) -> dict[str, Any]:
        """Return a strict versioned JSON-compatible model/state checkpoint."""
        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("cannot checkpoint an invalid model state")
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "model": self.to_config(),
            "state": {
                "weights": np.asarray(
                    jax.device_get(state.weights),
                    dtype=np.float32,
                ).tolist(),
                "bias": np.asarray(
                    jax.device_get(state.bias),
                    dtype=np.float32,
                ).tolist(),
                "update_count": int(state.update_count),
                "update_words": [int(word) for word in state.update_words],
            },
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> tuple[GroundedJointWorldModel, GroundedJointWorldModelState]:
        """Strictly reconstruct v3, or migrate one unambiguous v2 checkpoint."""
        if set(checkpoint) != {"schema", "model", "state"}:
            raise ValueError("checkpoint fields do not match the supported schema")
        checkpoint_schema = checkpoint.get("schema")
        if checkpoint_schema not in {_CHECKPOINT_SCHEMA, _LEGACY_CHECKPOINT_SCHEMA}:
            raise ValueError("unexpected grounded joint-world checkpoint schema")
        model_payload = checkpoint.get("model")
        state_payload = checkpoint.get("state")
        if not isinstance(model_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint must contain model and state mappings")
        legacy_v2 = checkpoint_schema == _LEGACY_CHECKPOINT_SCHEMA
        expected_state_fields = (
            {"weights", "bias", "update_count"}
            if legacy_v2
            else {"weights", "bias", "update_count", "update_words"}
        )
        if set(state_payload) != expected_state_fields:
            raise ValueError("checkpoint state fields do not match its schema")
        if legacy_v2:
            if set(model_payload) != {"type", "config"}:
                raise ValueError("legacy v2 model fields do not match its schema")
            if model_payload.get("type") != cls.__name__:
                raise ValueError(f"unexpected model type: {model_payload.get('type')!r}")
            legacy_config = model_payload.get("config")
            if not isinstance(legacy_config, Mapping):
                raise ValueError("legacy v2 model config must be a mapping")
            model = cls(GroundedJointWorldModelConfig.from_config(legacy_config))
        else:
            model = cls.from_config(model_payload)
        cfg = model.config
        expected_weights_shape = (
            cfg.joint_action_count,
            cfg.target_dim,
            cfg.representation_dim,
        )
        weights = _strict_json_float32_array(
            state_payload["weights"],
            name="checkpoint state.weights",
            shape=expected_weights_shape,
        )
        expected_bias_shape = (cfg.joint_action_count, cfg.target_dim)
        bias = _strict_json_float32_array(
            state_payload["bias"],
            name="checkpoint state.bias",
            shape=expected_bias_shape,
        )
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("checkpoint parameters must be finite")
        if np.any(np.abs(weights) > cfg.max_parameter_magnitude) or np.any(
            np.abs(bias) > cfg.max_parameter_magnitude
        ):
            raise ValueError("checkpoint parameters exceed max_parameter_magnitude")
        if cfg.feature_path_mode == "row_bias_only" and np.any(weights != 0.0):
            raise ValueError("row_bias_only checkpoint must contain exact zero weights")
        update_count = state_payload["update_count"]
        if type(update_count) is not int or not 0 <= update_count <= _INT32_MAX:
            raise ValueError("checkpoint update_count must be an int32-range integer")
        if legacy_v2:
            if update_count >= _INT32_MAX:
                raise ValueError("saturated legacy v2 update_count is ambiguous")
            update_words = (0, update_count)
        else:
            raw_words = state_payload["update_words"]
            if (
                type(raw_words) is not list
                or len(raw_words) != 2
                or any(type(word) is not int for word in raw_words)
                or any(word < 0 or word > _UINT32_MAX for word in raw_words)
            ):
                raise ValueError(
                    "checkpoint update_words must be two JSON uint32 integers"
                )
            update_words = (raw_words[0], raw_words[1])
        state = GroundedJointWorldModelState(
            weights=jnp.asarray(weights, dtype=jnp.float32),
            bias=jnp.asarray(bias, dtype=jnp.float32),
            update_count=jnp.asarray(update_count, dtype=jnp.int32),
            update_words=jnp.asarray(update_words, dtype=jnp.uint32),
        )
        if not bool(jax.device_get(model._state_valid(state))):
            raise ValueError("checkpoint lifetime counters or parameters are invalid")
        return model, state


def grounded_joint_world_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by compatibility telemetry plus exact identity."""

    return GROUNDED_JOINT_WORLD_LIFETIME_COUNTER_NBYTES


def measure_grounded_joint_world_state_nbytes(
    state: GroundedJointWorldModelState,
) -> int:
    """Measure persistent JAX-array bytes in one grounded-world state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_grounded_joint_world_state(
    legacy_state: Any,
) -> GroundedJointWorldModelState:
    """Migrate one exact pre-v3 state whose int32 clock is unambiguous."""

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy grounded-world state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(GroundedJointWorldModelState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"update_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy grounded-world field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    count = jnp.asarray(fields["update_count"])
    if count.shape != () or count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy grounded-world update_count must be scalar int32")
    count_value = int(count)
    if count_value < 0:
        raise ValueError("negative legacy grounded-world update_count indicates wrap")
    if count_value >= _INT32_MAX:
        raise ValueError("saturated legacy grounded-world update_count is ambiguous")
    fields["update_words"] = jnp.asarray((0, count_value), dtype=jnp.uint32)
    return GroundedJointWorldModelState(**fields)


__all__ = [
    "GROUNDED_JOINT_WORLD_LIFETIME_COUNTER_DELTA_NBYTES",
    "GROUNDED_JOINT_WORLD_LIFETIME_COUNTER_NBYTES",
    "GROUNDED_JOINT_WORLD_STATE_SCHEMA",
    "GroundedJointWorldDiagnostics",
    "GroundedJointWorldInputGradient",
    "GroundedJointWorldModel",
    "GroundedJointWorldModelConfig",
    "GroundedJointWorldModelState",
    "GroundedJointWorldPrediction",
    "GroundedJointWorldResourceBudget",
    "GroundedJointWorldUpdateResult",
    "grounded_joint_world_lifetime_counter_nbytes",
    "measure_grounded_joint_world_state_nbytes",
    "migrate_legacy_grounded_joint_world_state",
]
