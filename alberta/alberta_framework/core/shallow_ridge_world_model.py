# mypy: disable-error-code="attr-defined,call-arg"
"""Bounded L0 shallow world-model reference for discrete online control.

The model is an interpretable action-indexed affine ridge regressor.  For each
discrete action it retains only a Gram matrix, a feature/target cross matrix,
and the corresponding regularized least-squares coefficients.  One accepted
transition recursively adds its sufficient statistics and replaces that
action's coefficients with a direct regularized least-squares solve over the
retained statistics.  This is a regularized follow-the-leader construction;
it is not a claim of reproducing any particular paper's feature map, theorem,
MPC system, or empirical result.

Targets are grounded ``[next_observation, reward, continuation]`` values.  A
transition is always predicted before its target enters the statistics.  The
model has no task identifier, replay, ensemble, latent state, optimizer state,
or RNG state.  Shapes, numeric magnitudes, update capacity, and checkpoint
contents are fixed and validated fail-closed.  Invalid or exhausted updates
leave every state byte unchanged and return no usable prediction signal.

The pure planning surface predicts every discrete action and scores it as
``predicted_reward + predicted_continuation * linear_value(predicted_next_obs)``.
It is a one-step diagnostic baseline only.  This module is development/L0
mechanism code and makes no efficacy, regret, scientific-evidence, or SOTA
claim.
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
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Bool, Float, Int

EVIDENCE_LEVEL = "L0"
SCIENTIFIC_PROMOTION_ALLOWED = False

_CHECKPOINT_SCHEMA = "alberta.shallow_ridge_world_model.v1"
_INT32_MAX = 2**31 - 1
_MAX_EXACT_FLOAT32_INTEGER = 2**24
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_STATE_NBYTES = 256 * 1024 * 1024
_NORMAL_EQUATION_RTOL = 2.0e-3
_GRAM_PSD_RTOL = 2.0e-3


def _positive_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _finite_positive_float32(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical < _FLOAT32_TINY or canonical > _FLOAT32_MAX:
        raise ValueError(f"{name} must be a positive finite normal float32")
    narrowed = float(np.float32(canonical))
    if not math.isfinite(narrowed) or narrowed < _FLOAT32_TINY:
        raise ValueError(f"{name} must remain a positive finite normal float32")
    return narrowed


def _strict_json_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) is float and math.isfinite(actual) and actual == expected
    if isinstance(expected, str):
        return type(actual) is str and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _array_contract(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None:
        try:
            source_dtype = np.asarray(value).dtype
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an array-like numeric value") from exc
    try:
        normalized_dtype = np.dtype(source_dtype)
    except TypeError as exc:
        raise ValueError(f"{name} has an unsupported dtype") from exc
    expected_dtype = np.dtype(dtype)
    if normalized_dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}")
    array = jnp.asarray(value)
    if array.shape != shape or array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have shape {shape} and dtype {expected_dtype}")
    return array


def _strict_json_float32_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    def validate(current: object, remaining: tuple[int, ...], path: str) -> object:
        if not remaining:
            if type(current) not in (int, float):
                raise ValueError(f"{path} must be a JSON number, not a boolean or string")
            numeric = float(cast(int | float, current))
            if not math.isfinite(numeric) or abs(numeric) > _FLOAT32_MAX:
                raise ValueError(f"{path} must be finite and representable in float32")
            return numeric
        if type(current) is not list or len(cast(list[object], current)) != remaining[0]:
            raise ValueError(f"{path} must be a JSON list of length {remaining[0]}")
        return [
            validate(item, remaining[1:], f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], current))
        ]

    validated = validate(value, shape, name)
    array = np.asarray(validated, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite float32 array with shape {shape}")
    return array


def _strict_json_int32_vector(
    value: object,
    *,
    name: str,
    length: int,
) -> npt.NDArray[np.int32]:
    if type(value) is not list or len(cast(list[object], value)) != length:
        raise ValueError(f"{name} must be a JSON list of length {length}")
    items = cast(list[object], value)
    if any(type(item) is not int or not 0 <= item <= _INT32_MAX for item in items):
        raise ValueError(f"{name} must contain nonnegative int32-range integers")
    return np.asarray(items, dtype=np.int32)


@dataclasses.dataclass(frozen=True)
class ShallowRidgeWorldModelConfig:
    """Static construction and hard numeric bounds for the L0 reference."""

    observation_dim: int
    n_actions: int
    ridge: float = 1.0
    max_updates: int = _MAX_EXACT_FLOAT32_INTEGER
    max_input_magnitude: float = 1_000.0
    max_statistic_magnitude: float = 100_000_000.0
    max_parameter_magnitude: float = 1_000_000.0
    max_prediction_magnitude: float = 1_000_000.0

    def __post_init__(self) -> None:
        _positive_int(self.observation_dim, name="observation_dim")
        _positive_int(self.n_actions, name="n_actions")
        _positive_int(
            self.max_updates,
            name="max_updates",
            maximum=_MAX_EXACT_FLOAT32_INTEGER,
        )
        for name in (
            "ridge",
            "max_input_magnitude",
            "max_statistic_magnitude",
            "max_parameter_magnitude",
            "max_prediction_magnitude",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive_float32(getattr(self, name), name=name),
            )
        if self.max_statistic_magnitude < 1.0:
            raise ValueError("max_statistic_magnitude must accommodate the affine intercept")
        if self.state_nbytes > _MAX_STATE_NBYTES:
            raise ValueError(
                f"configured state requires {self.state_nbytes} bytes; "
                f"the implementation limit is {_MAX_STATE_NBYTES}"
            )

    @property
    def feature_dim(self) -> int:
        """Observation coordinates plus one explicit affine intercept."""
        return self.observation_dim + 1

    @property
    def target_dim(self) -> int:
        """Next observation coordinates plus reward and continuation."""
        return self.observation_dim + 2

    @property
    def state_nbytes(self) -> int:
        """Exact configured persistent array bytes."""
        feature_dim = self.feature_dim
        target_dim = self.target_dim
        float32_scalars = self.n_actions * (
            feature_dim * feature_dim + 2 * feature_dim * target_dim
        )
        int32_scalars = self.n_actions + 1
        return 4 * (float32_scalars + int32_scalars)

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible construction record."""
        return {
            "type": type(self).__name__,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "ridge": self.ridge,
            "max_updates": self.max_updates,
            "max_input_magnitude": self.max_input_magnitude,
            "max_statistic_magnitude": self.max_statistic_magnitude,
            "max_parameter_magnitude": self.max_parameter_magnitude,
            "max_prediction_magnitude": self.max_prediction_magnitude,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ShallowRidgeWorldModelConfig:
        """Strictly reconstruct an exact serialized configuration."""
        expected_fields = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(config) != expected_fields:
            raise ValueError("config fields do not match the serialized schema")
        if config.get("type") != cls.__name__:
            raise ValueError("unexpected shallow ridge world-model config type")
        restored = cls(
            observation_dim=cast(int, config["observation_dim"]),
            n_actions=cast(int, config["n_actions"]),
            ridge=cast(float, config["ridge"]),
            max_updates=cast(int, config["max_updates"]),
            max_input_magnitude=cast(float, config["max_input_magnitude"]),
            max_statistic_magnitude=cast(float, config["max_statistic_magnitude"]),
            max_parameter_magnitude=cast(float, config["max_parameter_magnitude"]),
            max_prediction_magnitude=cast(float, config["max_prediction_magnitude"]),
        )
        if not _strict_json_equal(dict(config), restored.to_config()):
            raise ValueError("config contains noncanonical values or JSON scalar types")
        return restored


@chex.dataclass(frozen=True)
class ShallowRidgeWorldModelState:
    """Fixed sufficient statistics, cached coefficients, and exact counters."""

    gram: Float[Array, "n_actions feature_dim feature_dim"]
    cross: Float[Array, "n_actions feature_dim target_dim"]
    weights: Float[Array, "n_actions feature_dim target_dim"]
    action_counts: Int[Array, " n_actions"]
    update_count: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class ShallowRidgeWorldModelResourceBudget:
    """Exact persistent memory and fixed logical work surfaces."""

    observation_dim: int
    n_actions: int
    feature_dim: int
    target_dim: int
    gram_float32_scalars: int
    cross_float32_scalars: int
    cached_weight_float32_scalars: int
    administrative_int32_scalars: int
    state_nbytes: int
    selected_gram_float32_scalars_touched_per_update: int
    selected_cross_float32_scalars_touched_per_update: int
    selected_weight_float32_scalars_solved_per_update: int
    administrative_int32_scalars_touched_per_update: int
    action_predictions_per_planning_call: int
    successor_value_evaluations_per_planning_call: int
    max_updates: int
    state_growth_nbytes_per_transition: int
    replay_capacity: int
    rng_state_nbytes: int

    def to_dict(self) -> dict[str, int]:
        """Return an exact JSON-compatible resource record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ShallowRidgeWorldPrediction:
    """One fail-closed grounded prediction from pre-update coefficients."""

    action: Int[Array, ""]
    features: Float[Array, " feature_dim"]
    raw_outputs: Float[Array, " target_dim"]
    next_observation: Float[Array, " observation_dim"]
    reward: Float[Array, ""]
    continuation: Float[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ShallowRidgeWorldDiagnostics:
    """Dynamic validity and atomic-commit verdicts for one attempted update."""

    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    target_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    prediction_valid: Bool[Array, ""]
    statistics_valid: Bool[Array, ""]
    solve_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ShallowRidgeWorldUpdateResult:
    """Prequential diagnostics and the atomically selected next state."""

    state: ShallowRidgeWorldModelState
    prediction: ShallowRidgeWorldPrediction
    targets: Float[Array, " target_dim"]
    errors: Float[Array, " target_dim"]
    squared_error: Float[Array, ""]
    diagnostics: ShallowRidgeWorldDiagnostics


@chex.dataclass(frozen=True)
class ShallowRidgePlanningResult:
    """Read-only one-step scores for every configured discrete action."""

    actions: Int[Array, " n_actions"]
    next_observations: Float[Array, "n_actions observation_dim"]
    rewards: Float[Array, " n_actions"]
    continuations: Float[Array, " n_actions"]
    successor_values: Float[Array, " n_actions"]
    scores: Float[Array, " n_actions"]
    best_action: Int[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ShallowRidgeWorldLearningResult:
    """Outputs of one uninterrupted predict-before-update scan."""

    state: ShallowRidgeWorldModelState
    next_observation_predictions: Float[Array, "num_steps observation_dim"]
    reward_predictions: Float[Array, " num_steps"]
    continuation_predictions: Float[Array, " num_steps"]
    targets: Float[Array, "num_steps target_dim"]
    errors: Float[Array, "num_steps target_dim"]
    squared_errors: Float[Array, " num_steps"]
    applied: Bool[Array, " num_steps"]
    rejected: Bool[Array, " num_steps"]


class ShallowRidgeWorldModel:
    """Action-indexed recursive ridge reference with fixed lifetime memory."""

    def __init__(self, config: ShallowRidgeWorldModelConfig):
        self._config = config

    @property
    def config(self) -> ShallowRidgeWorldModelConfig:
        """Return the immutable static construction."""
        return self._config

    @property
    def resource_budget(self) -> ShallowRidgeWorldModelResourceBudget:
        """Return exact fixed-state and per-call logical resource bounds."""
        cfg = self._config
        gram = cfg.n_actions * cfg.feature_dim * cfg.feature_dim
        cross = cfg.n_actions * cfg.feature_dim * cfg.target_dim
        weights = cross
        administrative = cfg.n_actions + 1
        return ShallowRidgeWorldModelResourceBudget(
            observation_dim=cfg.observation_dim,
            n_actions=cfg.n_actions,
            feature_dim=cfg.feature_dim,
            target_dim=cfg.target_dim,
            gram_float32_scalars=gram,
            cross_float32_scalars=cross,
            cached_weight_float32_scalars=weights,
            administrative_int32_scalars=administrative,
            state_nbytes=4 * (gram + cross + weights + administrative),
            selected_gram_float32_scalars_touched_per_update=(cfg.feature_dim * cfg.feature_dim),
            selected_cross_float32_scalars_touched_per_update=(cfg.feature_dim * cfg.target_dim),
            selected_weight_float32_scalars_solved_per_update=(cfg.feature_dim * cfg.target_dim),
            administrative_int32_scalars_touched_per_update=2,
            action_predictions_per_planning_call=cfg.n_actions,
            successor_value_evaluations_per_planning_call=cfg.n_actions,
            max_updates=cfg.max_updates,
            state_growth_nbytes_per_transition=0,
            replay_capacity=0,
            rng_state_nbytes=0,
        )

    def to_config(self) -> dict[str, object]:
        """Serialize the complete model construction."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ShallowRidgeWorldModel:
        """Strictly reconstruct the complete model construction."""
        if set(config) != {"type", "config"}:
            raise ValueError("model config fields do not match the serialized schema")
        if config.get("type") != cls.__name__:
            raise ValueError("unexpected shallow ridge world-model type")
        nested = config.get("config")
        if not isinstance(nested, Mapping):
            raise ValueError("model config must contain a config mapping")
        restored = cls(ShallowRidgeWorldModelConfig.from_config(nested))
        if not _strict_json_equal(dict(config), restored.to_config()):
            raise ValueError("model config contains noncanonical values")
        return restored

    def init(self) -> ShallowRidgeWorldModelState:
        """Return the unique deterministic zero state; no RNG is accepted or stored."""
        cfg = self._config
        return ShallowRidgeWorldModelState(
            gram=jnp.zeros(
                (cfg.n_actions, cfg.feature_dim, cfg.feature_dim),
                dtype=jnp.float32,
            ),
            cross=jnp.zeros(
                (cfg.n_actions, cfg.feature_dim, cfg.target_dim),
                dtype=jnp.float32,
            ),
            weights=jnp.zeros(
                (cfg.n_actions, cfg.feature_dim, cfg.target_dim),
                dtype=jnp.float32,
            ),
            action_counts=jnp.zeros((cfg.n_actions,), dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def _validate_state_static_contract(self, state: ShallowRidgeWorldModelState) -> None:
        if not isinstance(state, ShallowRidgeWorldModelState):
            raise TypeError("state must be a ShallowRidgeWorldModelState")
        cfg = self._config
        _array_contract(
            state.gram,
            name="state.gram",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.feature_dim),
            dtype=jnp.float32,
        )
        _array_contract(
            state.cross,
            name="state.cross",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.target_dim),
            dtype=jnp.float32,
        )
        _array_contract(
            state.weights,
            name="state.weights",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.target_dim),
            dtype=jnp.float32,
        )
        _array_contract(
            state.action_counts,
            name="state.action_counts",
            shape=(cfg.n_actions,),
            dtype=jnp.int32,
        )
        _array_contract(
            state.update_count,
            name="state.update_count",
            shape=(),
            dtype=jnp.int32,
        )

    def _counts_valid(self, state: ShallowRidgeWorldModelState) -> Array:
        update_valid = (state.update_count >= 0) & (state.update_count <= self._config.max_updates)

        def add_count(
            carry: tuple[Array, Array],
            value: Array,
        ) -> tuple[tuple[Array, Array], None]:
            total, valid = carry
            value_valid = (value >= 0) & (value <= self._config.max_updates)
            safe_value = jnp.where(value_valid, value, 0)
            room = safe_value <= (_INT32_MAX - total)
            next_total = jnp.where(room, total + safe_value, total)
            return (next_total, valid & value_valid & room), None

        (total, entries_valid), _ = jax.lax.scan(
            add_count,
            (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True, dtype=jnp.bool_),
            ),
            state.action_counts,
        )
        return update_valid & entries_valid & (total == state.update_count)

    def _state_valid(self, state: ShallowRidgeWorldModelState) -> Array:
        cfg = self._config
        statistic_bound = jnp.asarray(cfg.max_statistic_magnitude, dtype=jnp.float32)
        parameter_bound = jnp.asarray(cfg.max_parameter_magnitude, dtype=jnp.float32)
        symmetric = jnp.array_equal(state.gram, jnp.swapaxes(state.gram, -1, -2))
        nonnegative_diagonal = jnp.all(jnp.diagonal(state.gram, axis1=-2, axis2=-1) >= 0.0)
        gram_scale = 1.0 + jnp.max(jnp.abs(state.gram), axis=(-2, -1))
        positive_semidefinite = jnp.all(
            jnp.linalg.eigvalsh(state.gram)
            >= -jnp.asarray(_GRAM_PSD_RTOL, dtype=jnp.float32)
            * gram_scale[:, None]
        )
        intercept_counts_match = jnp.array_equal(
            state.gram[:, -1, -1],
            state.action_counts.astype(jnp.float32),
        )
        empty = state.action_counts == 0
        empty_rows_zero = (
            jnp.all(jnp.where(empty[:, None, None], state.gram == 0.0, True))
            & jnp.all(jnp.where(empty[:, None, None], state.cross == 0.0, True))
            & jnp.all(jnp.where(empty[:, None, None], state.weights == 0.0, True))
        )
        regularized = (
            state.gram
            + jnp.asarray(cfg.ridge, dtype=jnp.float32)
            * jnp.eye(
                cfg.feature_dim,
                dtype=jnp.float32,
            )[None, :, :]
        )
        reconstructed_cross = jnp.einsum("aij,ajt->ait", regularized, state.weights)
        residual = jnp.abs(reconstructed_cross - state.cross)
        residual_scale = 1.0 + jnp.abs(reconstructed_cross) + jnp.abs(state.cross)
        normal_equations_hold = jnp.all(
            residual <= jnp.asarray(_NORMAL_EQUATION_RTOL, dtype=jnp.float32) * residual_scale
        )
        return (
            jnp.all(jnp.isfinite(state.gram))
            & jnp.all(jnp.abs(state.gram) <= statistic_bound)
            & jnp.all(jnp.isfinite(state.cross))
            & jnp.all(jnp.abs(state.cross) <= statistic_bound)
            & jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.abs(state.weights) <= parameter_bound)
            & symmetric
            & nonnegative_diagonal
            & positive_semidefinite
            & intercept_counts_match
            & empty_rows_zero
            & normal_equations_hold
            & self._counts_valid(state)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def state_valid(self, state: ShallowRidgeWorldModelState) -> Bool[Array, ""]:
        """Return the dynamic state-validity verdict after static validation."""
        self._validate_state_static_contract(state)
        return self._state_valid(state)

    def _features(self, observation: Array) -> Array:
        return jnp.concatenate(
            (observation, jnp.ones((1,), dtype=jnp.float32)),
            axis=0,
        )

    def _observation_valid(self, observation: Array) -> Array:
        bound = jnp.asarray(self._config.max_input_magnitude, dtype=jnp.float32)
        return jnp.all(jnp.isfinite(observation)) & jnp.all(jnp.abs(observation) <= bound)

    def _action_valid(self, action: Array) -> Array:
        return (action >= 0) & (action < self._config.n_actions)

    def _target_valid(
        self,
        next_observation: Array,
        reward: Array,
        continuation: Array,
    ) -> Array:
        bound = jnp.asarray(self._config.max_input_magnitude, dtype=jnp.float32)
        return (
            jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.abs(next_observation) <= bound)
            & jnp.isfinite(reward)
            & (jnp.abs(reward) <= bound)
            & jnp.isfinite(continuation)
            & (continuation >= 0.0)
            & (continuation <= 1.0)
        )

    def _zero_prediction(self) -> ShallowRidgeWorldPrediction:
        cfg = self._config
        return ShallowRidgeWorldPrediction(
            action=jnp.asarray(-1, dtype=jnp.int32),
            features=jnp.zeros((cfg.feature_dim,), dtype=jnp.float32),
            raw_outputs=jnp.zeros((cfg.target_dim,), dtype=jnp.float32),
            next_observation=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            continuation=jnp.asarray(0.0, dtype=jnp.float32),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _predict_unchecked(
        self,
        state: ShallowRidgeWorldModelState,
        observation: Array,
        action: Array,
    ) -> ShallowRidgeWorldPrediction:
        cfg = self._config
        safe_action = jnp.clip(action, 0, cfg.n_actions - 1)
        features = self._features(observation)
        raw = features @ state.weights[safe_action]
        output_bound = jnp.asarray(cfg.max_prediction_magnitude, dtype=jnp.float32)
        valid = jnp.all(jnp.isfinite(raw)) & jnp.all(jnp.abs(raw) <= output_bound)
        return ShallowRidgeWorldPrediction(
            action=safe_action,
            features=features,
            raw_outputs=raw,
            next_observation=raw[: cfg.observation_dim],
            reward=raw[cfg.observation_dim],
            continuation=jnp.clip(raw[cfg.observation_dim + 1], 0.0, 1.0),
            valid=valid,
        )

    def _validate_prediction_static_contract(
        self,
        state: ShallowRidgeWorldModelState,
        observation: object,
        action: object,
    ) -> tuple[Array, Array]:
        self._validate_state_static_contract(state)
        return (
            _array_contract(
                observation,
                name="observation",
                shape=(self._config.observation_dim,),
                dtype=jnp.float32,
            ),
            _array_contract(action, name="action", shape=(), dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: ShallowRidgeWorldModelState,
        observation: Array,
        action: Array,
    ) -> ShallowRidgeWorldPrediction:
        """Return a fail-closed grounded prediction without changing state."""
        obs, act = self._validate_prediction_static_contract(state, observation, action)
        candidate = self._predict_unchecked(state, obs, act)
        valid = (
            self._state_valid(state)
            & self._observation_valid(obs)
            & self._action_valid(act)
            & candidate.valid
        )
        return cast(
            ShallowRidgeWorldPrediction,
            jax.lax.cond(
                valid,
                lambda _: candidate,
                lambda _: self._zero_prediction(),
                operand=None,
            ),
        )

    def _targets(
        self,
        next_observation: Array,
        reward: Array,
        continuation: Array,
    ) -> Array:
        return jax.lax.stop_gradient(
            jnp.concatenate(
                (
                    next_observation,
                    jnp.reshape(reward, (1,)),
                    jnp.reshape(continuation, (1,)),
                )
            )
        )

    def _prediction_outputs(self, prediction: ShallowRidgeWorldPrediction) -> Array:
        return jnp.concatenate(
            (
                prediction.next_observation,
                jnp.reshape(prediction.reward, (1,)),
                jnp.reshape(prediction.continuation, (1,)),
            )
        )

    def _rejected_update(
        self,
        state: ShallowRidgeWorldModelState,
        *,
        state_valid: Array,
        input_valid: Array,
        target_valid: Array,
        capacity_available: Array,
    ) -> ShallowRidgeWorldUpdateResult:
        false = jnp.asarray(False, dtype=jnp.bool_)
        return ShallowRidgeWorldUpdateResult(
            state=state,
            prediction=self._zero_prediction(),
            targets=jnp.zeros((self._config.target_dim,), dtype=jnp.float32),
            errors=jnp.zeros((self._config.target_dim,), dtype=jnp.float32),
            squared_error=jnp.asarray(0.0, dtype=jnp.float32),
            diagnostics=ShallowRidgeWorldDiagnostics(
                state_valid=state_valid,
                input_valid=input_valid,
                target_valid=target_valid,
                capacity_available=capacity_available,
                prediction_valid=false,
                statistics_valid=false,
                solve_valid=false,
                candidate_state_valid=false,
                applied=false,
                rejected=jnp.asarray(True, dtype=jnp.bool_),
            ),
        )

    def _validate_update_static_contract(
        self,
        state: ShallowRidgeWorldModelState,
        observation: object,
        action: object,
        next_observation: object,
        reward: object,
        continuation: object,
    ) -> tuple[Array, Array, Array, Array, Array]:
        obs, act = self._validate_prediction_static_contract(state, observation, action)
        return (
            obs,
            act,
            _array_contract(
                next_observation,
                name="next_observation",
                shape=(self._config.observation_dim,),
                dtype=jnp.float32,
            ),
            _array_contract(reward, name="reward", shape=(), dtype=jnp.float32),
            _array_contract(
                continuation,
                name="continuation",
                shape=(),
                dtype=jnp.float32,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ShallowRidgeWorldModelState,
        observation: Array,
        action: Array,
        next_observation: Array,
        reward: Array,
        continuation: Array,
    ) -> ShallowRidgeWorldUpdateResult:
        """Predict first, then atomically add one transition's sufficient statistics."""
        obs, act, next_obs, rew, cont = self._validate_update_static_contract(
            state,
            observation,
            action,
            next_observation,
            reward,
            continuation,
        )
        state_valid = self._state_valid(state)
        input_valid = self._observation_valid(obs) & self._action_valid(act)
        target_valid = self._target_valid(next_obs, rew, cont)
        capacity_available = state.update_count < self._config.max_updates
        can_attempt = state_valid & input_valid & target_valid & capacity_available

        def do_update(_: None) -> ShallowRidgeWorldUpdateResult:
            cfg = self._config
            prediction = self._predict_unchecked(state, obs, act)
            targets = self._targets(next_obs, rew, cont)
            prediction_outputs = self._prediction_outputs(prediction)
            errors = targets - prediction_outputs
            squared_error = jnp.mean(jnp.square(errors))
            features = prediction.features
            safe_action = prediction.action
            gram_row = state.gram[safe_action] + jnp.outer(features, features)
            cross_row = state.cross[safe_action] + jnp.outer(features, targets)
            statistic_bound = jnp.asarray(cfg.max_statistic_magnitude, dtype=jnp.float32)
            statistics_valid = (
                jnp.all(jnp.isfinite(gram_row))
                & jnp.all(jnp.abs(gram_row) <= statistic_bound)
                & jnp.array_equal(gram_row, gram_row.T)
                & jnp.all(jnp.diag(gram_row) >= 0.0)
                & jnp.all(jnp.isfinite(cross_row))
                & jnp.all(jnp.abs(cross_row) <= statistic_bound)
            )
            system = gram_row + jnp.asarray(cfg.ridge, dtype=jnp.float32) * jnp.eye(
                cfg.feature_dim,
                dtype=jnp.float32,
            )
            weight_row = jnp.linalg.solve(system, cross_row)
            parameter_bound = jnp.asarray(cfg.max_parameter_magnitude, dtype=jnp.float32)
            normal_residual = jnp.abs(system @ weight_row - cross_row)
            normal_scale = 1.0 + jnp.abs(system @ weight_row) + jnp.abs(cross_row)
            solve_valid = (
                statistics_valid
                & jnp.all(jnp.isfinite(weight_row))
                & jnp.all(jnp.abs(weight_row) <= parameter_bound)
                & jnp.all(
                    normal_residual
                    <= jnp.asarray(_NORMAL_EQUATION_RTOL, dtype=jnp.float32) * normal_scale
                )
            )
            candidate = ShallowRidgeWorldModelState(
                gram=state.gram.at[safe_action].set(gram_row),
                cross=state.cross.at[safe_action].set(cross_row),
                weights=state.weights.at[safe_action].set(weight_row),
                action_counts=state.action_counts.at[safe_action].add(1),
                update_count=state.update_count + jnp.asarray(1, dtype=jnp.int32),
            )
            candidate_state_valid = self._state_valid(candidate)
            outputs_valid = (
                prediction.valid
                & jnp.all(jnp.isfinite(targets))
                & jnp.all(jnp.isfinite(errors))
                & jnp.isfinite(squared_error)
                & (squared_error >= 0.0)
            )
            applied = outputs_valid & statistics_valid & solve_valid & candidate_state_valid
            next_state = cast(
                ShallowRidgeWorldModelState,
                jax.lax.cond(applied, lambda _: candidate, lambda _: state, operand=None),
            )
            zero_prediction = self._zero_prediction()
            return ShallowRidgeWorldUpdateResult(
                state=next_state,
                prediction=cast(
                    ShallowRidgeWorldPrediction,
                    jax.lax.cond(
                        applied,
                        lambda _: prediction,
                        lambda _: zero_prediction,
                        operand=None,
                    ),
                ),
                targets=jnp.where(applied, targets, 0.0),
                errors=jnp.where(applied, errors, 0.0),
                squared_error=jnp.where(applied, squared_error, 0.0),
                diagnostics=ShallowRidgeWorldDiagnostics(
                    state_valid=state_valid,
                    input_valid=input_valid,
                    target_valid=target_valid,
                    capacity_available=capacity_available,
                    prediction_valid=prediction.valid,
                    statistics_valid=statistics_valid,
                    solve_valid=solve_valid,
                    candidate_state_valid=candidate_state_valid,
                    applied=applied,
                    rejected=~applied,
                ),
            )

        return cast(
            ShallowRidgeWorldUpdateResult,
            jax.lax.cond(
                can_attempt,
                do_update,
                lambda _: self._rejected_update(
                    state,
                    state_valid=state_valid,
                    input_valid=input_valid,
                    target_valid=target_valid,
                    capacity_available=capacity_available,
                ),
                operand=None,
            ),
        )

    def _zero_planning_result(self) -> ShallowRidgePlanningResult:
        cfg = self._config
        return ShallowRidgePlanningResult(
            actions=jnp.arange(cfg.n_actions, dtype=jnp.int32),
            next_observations=jnp.zeros(
                (cfg.n_actions, cfg.observation_dim),
                dtype=jnp.float32,
            ),
            rewards=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            continuations=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            successor_values=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            scores=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            best_action=jnp.asarray(-1, dtype=jnp.int32),
            valid=jnp.asarray(False, dtype=jnp.bool_),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def score_actions(
        self,
        state: ShallowRidgeWorldModelState,
        observation: Array,
        successor_value_weights: Array,
        successor_value_bias: Array,
    ) -> ShallowRidgePlanningResult:
        """Purely score every action with one predicted step and a supplied linear value."""
        self._validate_state_static_contract(state)
        cfg = self._config
        obs = _array_contract(
            observation,
            name="observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        value_weights = _array_contract(
            successor_value_weights,
            name="successor_value_weights",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        value_bias = _array_contract(
            successor_value_bias,
            name="successor_value_bias",
            shape=(),
            dtype=jnp.float32,
        )
        input_bound = jnp.asarray(cfg.max_input_magnitude, dtype=jnp.float32)
        inputs_valid = (
            self._observation_valid(obs)
            & jnp.all(jnp.isfinite(value_weights))
            & jnp.all(jnp.abs(value_weights) <= input_bound)
            & jnp.isfinite(value_bias)
            & (jnp.abs(value_bias) <= input_bound)
        )
        features = self._features(obs)
        raw = jnp.einsum("f,aft->at", features, state.weights)
        next_observations = raw[:, : cfg.observation_dim]
        rewards = raw[:, cfg.observation_dim]
        continuations = jnp.clip(raw[:, cfg.observation_dim + 1], 0.0, 1.0)
        successor_values = next_observations @ value_weights + value_bias
        scores = rewards + continuations * successor_values
        output_bound = jnp.asarray(cfg.max_prediction_magnitude, dtype=jnp.float32)
        outputs_valid = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.abs(raw) <= output_bound)
            & jnp.all(jnp.isfinite(successor_values))
            & jnp.all(jnp.abs(successor_values) <= output_bound)
            & jnp.all(jnp.isfinite(scores))
            & jnp.all(jnp.abs(scores) <= output_bound)
        )
        valid = self._state_valid(state) & inputs_valid & outputs_valid
        candidate = ShallowRidgePlanningResult(
            actions=jnp.arange(cfg.n_actions, dtype=jnp.int32),
            next_observations=next_observations,
            rewards=rewards,
            continuations=continuations,
            successor_values=successor_values,
            scores=scores,
            best_action=jnp.argmax(scores).astype(jnp.int32),
            valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        return cast(
            ShallowRidgePlanningResult,
            jax.lax.cond(
                valid,
                lambda _: candidate,
                lambda _: self._zero_planning_result(),
                operand=None,
            ),
        )

    def checkpoint_payload(
        self,
        state: ShallowRidgeWorldModelState,
    ) -> dict[str, object]:
        """Return a strict versioned JSON checkpoint with no RNG state."""
        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("cannot checkpoint an invalid shallow ridge world-model state")
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "model": self.to_config(),
            "metadata": {
                "evidence_level": EVIDENCE_LEVEL,
                "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
                "resource_budget": self.resource_budget.to_dict(),
                "rng_state_nbytes": 0,
            },
            "state": {
                "gram": np.asarray(jax.device_get(state.gram), dtype=np.float32).tolist(),
                "cross": np.asarray(jax.device_get(state.cross), dtype=np.float32).tolist(),
                "weights": np.asarray(
                    jax.device_get(state.weights),
                    dtype=np.float32,
                ).tolist(),
                "action_counts": np.asarray(
                    jax.device_get(state.action_counts),
                    dtype=np.int32,
                ).tolist(),
                "update_count": int(state.update_count),
            },
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, object],
    ) -> tuple[ShallowRidgeWorldModel, ShallowRidgeWorldModelState]:
        """Strictly restore the exact model/state pair from a v1 checkpoint."""
        if set(checkpoint) != {"schema", "model", "metadata", "state"}:
            raise ValueError("checkpoint fields do not match the v1 schema")
        if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("unexpected shallow ridge world-model checkpoint schema")
        model_payload = checkpoint.get("model")
        metadata = checkpoint.get("metadata")
        state_payload = checkpoint.get("state")
        if (
            not isinstance(model_payload, Mapping)
            or not isinstance(metadata, Mapping)
            or not isinstance(state_payload, Mapping)
        ):
            raise ValueError("checkpoint model, metadata, and state must be mappings")
        model = cls.from_config(model_payload)
        expected_metadata: dict[str, object] = {
            "evidence_level": EVIDENCE_LEVEL,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "resource_budget": model.resource_budget.to_dict(),
            "rng_state_nbytes": 0,
        }
        if not _strict_json_equal(dict(metadata), expected_metadata):
            raise ValueError("checkpoint metadata is not the exact L0 RNG-free resource record")
        if set(state_payload) != {
            "gram",
            "cross",
            "weights",
            "action_counts",
            "update_count",
        }:
            raise ValueError("checkpoint state fields do not match the v1 schema")
        cfg = model.config
        gram = _strict_json_float32_array(
            state_payload["gram"],
            name="checkpoint state.gram",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.feature_dim),
        )
        cross = _strict_json_float32_array(
            state_payload["cross"],
            name="checkpoint state.cross",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.target_dim),
        )
        weights = _strict_json_float32_array(
            state_payload["weights"],
            name="checkpoint state.weights",
            shape=(cfg.n_actions, cfg.feature_dim, cfg.target_dim),
        )
        action_counts = _strict_json_int32_vector(
            state_payload["action_counts"],
            name="checkpoint state.action_counts",
            length=cfg.n_actions,
        )
        update_count = state_payload["update_count"]
        if type(update_count) is not int or not 0 <= update_count <= cfg.max_updates:
            raise ValueError("checkpoint update_count must be an in-capacity int32 integer")
        state = ShallowRidgeWorldModelState(
            gram=jnp.asarray(gram, dtype=jnp.float32),
            cross=jnp.asarray(cross, dtype=jnp.float32),
            weights=jnp.asarray(weights, dtype=jnp.float32),
            action_counts=jnp.asarray(action_counts, dtype=jnp.int32),
            update_count=jnp.asarray(update_count, dtype=jnp.int32),
        )
        if not bool(jax.device_get(model._state_valid(state))):
            raise ValueError("checkpoint contains an invalid or inconsistent model state")
        return model, state


def run_shallow_ridge_world_model(
    model: ShallowRidgeWorldModel,
    state: ShallowRidgeWorldModelState,
    observations: Array,
    actions: Array,
    next_observations: Array,
    rewards: Array,
    continuations: Array,
) -> ShallowRidgeWorldLearningResult:
    """Run one uninterrupted predict-before-update stream with fixed state shape."""
    model._validate_state_static_contract(state)
    observation_shape = getattr(observations, "shape", None)
    if not isinstance(observation_shape, tuple) or len(observation_shape) != 2:
        raise ValueError("observations must be a rank-2 array")
    num_steps = observation_shape[0]
    cfg = model.config
    obs = _array_contract(
        observations,
        name="observations",
        shape=(num_steps, cfg.observation_dim),
        dtype=jnp.float32,
    )
    act = _array_contract(
        actions,
        name="actions",
        shape=(num_steps,),
        dtype=jnp.int32,
    )
    next_obs = _array_contract(
        next_observations,
        name="next_observations",
        shape=(num_steps, cfg.observation_dim),
        dtype=jnp.float32,
    )
    rew = _array_contract(
        rewards,
        name="rewards",
        shape=(num_steps,),
        dtype=jnp.float32,
    )
    cont = _array_contract(
        continuations,
        name="continuations",
        shape=(num_steps,),
        dtype=jnp.float32,
    )

    def scan_step(
        carry: ShallowRidgeWorldModelState,
        transition: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[ShallowRidgeWorldModelState, tuple[Array, ...]]:
        observation, action, next_observation, reward, continuation = transition
        result = model.update(
            carry,
            observation,
            action,
            next_observation,
            reward,
            continuation,
        )
        return result.state, (
            result.prediction.next_observation,
            result.prediction.reward,
            result.prediction.continuation,
            result.targets,
            result.errors,
            result.squared_error,
            result.diagnostics.applied,
            result.diagnostics.rejected,
        )

    final_state, outputs = jax.lax.scan(
        scan_step,
        state,
        (obs, act, next_obs, rew, cont),
    )
    (
        next_observation_predictions,
        reward_predictions,
        continuation_predictions,
        targets,
        errors,
        squared_errors,
        applied,
        rejected,
    ) = outputs
    return ShallowRidgeWorldLearningResult(
        state=final_state,
        next_observation_predictions=next_observation_predictions,
        reward_predictions=reward_predictions,
        continuation_predictions=continuation_predictions,
        targets=targets,
        errors=errors,
        squared_errors=squared_errors,
        applied=applied,
        rejected=rejected,
    )


__all__ = [
    "EVIDENCE_LEVEL",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "ShallowRidgePlanningResult",
    "ShallowRidgeWorldDiagnostics",
    "ShallowRidgeWorldLearningResult",
    "ShallowRidgeWorldModel",
    "ShallowRidgeWorldModelConfig",
    "ShallowRidgeWorldModelResourceBudget",
    "ShallowRidgeWorldModelState",
    "ShallowRidgeWorldPrediction",
    "ShallowRidgeWorldUpdateResult",
    "run_shallow_ridge_world_model",
]
