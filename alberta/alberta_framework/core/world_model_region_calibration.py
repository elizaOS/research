# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded online state/action-region calibration for world-model planning gates.

This module is an isolated L0 mechanism for WP4.3.  It owns no model, planner,
policy, state representation, region assignment, safety decision, or dispatch
authority.  A caller first supplies one exact predict-before-outcome ensemble
record.  The returned immutable receipt binds the lifecycle, decision, model,
representation, action, and region revisions; primitive action and declared
region; every member mean, aleatoric variance, and termination probability;
and the exact pre-update calibration cell.  Exactly one matching real outcome
may then settle that receipt.

Calibration is local to fixed-capacity declared-region × primitive-action
cells.  The channels remain noncompensating and separately typed:

* ensemble disagreement is retained as epistemic evidence and compared with
  realized error through causal excess-error/disagreement factors and
  association diagnostics; realized error is never relabeled as epistemic;
* heteroscedastic variance is assessed through per-head standardized-residual
  coverage and Gaussian NLL, while a separate caller-fixed variance ceiling
  provides an explicit noisy-TV veto;
* grounded next-state and reward errors retain separate empirical quantiles;
* termination uses an uncensored Bernoulli target, support, Brier error, and
  calibration gap.  Time-limit truncation is censored for this channel and is
  never treated as environmental termination, while its grounded next-state
  and reward outcome still calibrate those respective heads.

All thresholds and quantiles in a receipt are computed from the immutable
pre-outcome cell.  ``alpha`` and every absolute ceiling are caller-fixed
configuration facts, not learned safety authority or evidence that the model
is calibrated.  Receipts explicitly carry no planning or safety authority.
Invalid, stale, tampered, non-finite, cross-lifecycle, or exhausted operations
return the complete original state, including all uint64 word-pair clocks.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

WORLD_MODEL_REGION_CALIBRATION_CONFIG_SCHEMA = (
    "alberta.world-model-region-calibration.config.v1"
)
WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA = (
    "alberta.world-model-region-calibration.checkpoint.v1"
)
WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL = "L0"
WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS = "not_assessed"
WORLD_MODEL_REGION_CALIBRATION_SCIENTIFIC_PROMOTION_ALLOWED = False

_UINT64_MAX = 2**64 - 1
_MAX_DIMENSION = 4_096
_MAX_CELLS = 65_536
_MAX_CAPACITY_PER_CELL = 4_096
_MAX_RECORDS = 1_048_576
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_LOG_TWO_PI = float(math.log(2.0 * math.pi))
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_PREDICTION_TAG_SALT = 0x50524544
_CALIBRATION_TAG_SALT = 0x43414C42
_CALIBRATION_STATE_TAG_SALT = 0x53544154
_GATES_TAG_SALT = 0x47415445
_RECEIPT_TAG_SALT = 0x52435054


def _strict_positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [1, {maximum}]")
    return value


def _strict_nonnegative_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [0, {maximum}]")
    return value


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not underflow in float32")
    if narrowed < minimum or (strictly_positive and narrowed == minimum):
        relation = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
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
        return type(actual) is list and len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    return actual == expected


@dataclasses.dataclass(frozen=True)
class WorldModelRegionCalibrationConfig:
    """Static cell geometry, support, caller-fixed alpha, and guard limits."""

    observation_dim: int
    n_actions: int
    n_regions: int
    ensemble_size: int
    capacity_per_cell: int = 64
    min_samples: int = 16
    min_termination_samples: int = 16
    min_termination_class_support: int = 1
    alpha: float = 0.1
    variance_floor: float = 1.0e-6
    disagreement_floor: float = 1.0e-6
    max_aleatoric_variance: float = 100.0
    max_gaussian_nll: float = 100.0
    max_next_state_rmse: float = 100.0
    max_reward_abs_error: float = 100.0
    max_epistemic_next_state_bound: float = 100.0
    max_epistemic_reward_bound: float = 100.0
    max_termination_brier: float = 0.25
    max_prediction_magnitude: float = 1.0e6
    max_outcome_magnitude: float = 1.0e6
    max_events: int = _UINT64_MAX - 1

    def __post_init__(self) -> None:
        _strict_positive_int(
            self.observation_dim,
            name="observation_dim",
            maximum=_MAX_DIMENSION,
        )
        _strict_positive_int(self.n_actions, name="n_actions", maximum=_MAX_CELLS)
        _strict_positive_int(self.n_regions, name="n_regions", maximum=_MAX_CELLS)
        if self.n_actions * self.n_regions > _MAX_CELLS:
            raise ValueError(f"n_actions * n_regions must be <= {_MAX_CELLS}")
        _strict_positive_int(
            self.ensemble_size,
            name="ensemble_size",
            maximum=_MAX_DIMENSION,
        )
        _strict_positive_int(
            self.capacity_per_cell,
            name="capacity_per_cell",
            maximum=_MAX_CAPACITY_PER_CELL,
        )
        if self.n_actions * self.n_regions * self.capacity_per_cell > _MAX_RECORDS:
            raise ValueError(f"configured cells exceed {_MAX_RECORDS} retained records")
        _strict_positive_int(
            self.min_samples,
            name="min_samples",
            maximum=self.capacity_per_cell,
        )
        _strict_positive_int(
            self.min_termination_samples,
            name="min_termination_samples",
            maximum=self.capacity_per_cell,
        )
        _strict_nonnegative_int(
            self.min_termination_class_support,
            name="min_termination_class_support",
            maximum=self.capacity_per_cell,
        )
        if 2 * self.min_termination_class_support > self.capacity_per_cell:
            raise ValueError(
                "twice min_termination_class_support cannot exceed capacity_per_cell"
            )
        _strict_positive_int(self.max_events, name="max_events", maximum=_UINT64_MAX - 1)
        object.__setattr__(
            self,
            "alpha",
            _finite_float32(self.alpha, name="alpha", minimum=0.0, maximum=1.0),
        )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        for name in (
            "variance_floor",
            "disagreement_floor",
            "max_aleatoric_variance",
            "max_gaussian_nll",
            "max_next_state_rmse",
            "max_reward_abs_error",
            "max_epistemic_next_state_bound",
            "max_epistemic_reward_bound",
            "max_termination_brier",
            "max_prediction_magnitude",
            "max_outcome_magnitude",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    strictly_positive=True,
                ),
            )
        if self.max_termination_brier > 1.0:
            raise ValueError("max_termination_brier must be <= 1")
        if self.variance_floor > self.max_aleatoric_variance:
            raise ValueError("variance_floor cannot exceed max_aleatoric_variance")

    @property
    def target_dim(self) -> int:
        """Grounded next-state coordinates plus one reward head."""

        return self.observation_dim + 1

    @property
    def nominal_standardized_residual_limit(self) -> float:
        """Caller-alpha two-sided standard-normal marginal interval limit."""

        return float(NormalDist().inv_cdf(1.0 - 0.5 * self.alpha))

    def to_config(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": type(self).__name__,
            "schema": WORLD_MODEL_REGION_CALIBRATION_CONFIG_SCHEMA,
            "evidence_level": WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL,
            "outcome_status": WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS,
            "scientific_promotion_allowed": (
                WORLD_MODEL_REGION_CALIBRATION_SCIENTIFIC_PROMOTION_ALLOWED
            ),
        }
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> WorldModelRegionCalibrationConfig:
        expected = {field.name for field in dataclasses.fields(cls)} | {
            "type",
            "schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("world-model region calibration config fields are not exact")
        if config.get("type") != cls.__name__:
            raise ValueError("world-model region calibration config type is unsupported")
        if config.get("schema") != WORLD_MODEL_REGION_CALIBRATION_CONFIG_SCHEMA:
            raise ValueError("world-model region calibration config schema is unsupported")
        if config.get("evidence_level") != WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL:
            raise ValueError("world-model region calibration must remain L0")
        if config.get("outcome_status") != WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS:
            raise ValueError("world-model region calibration must remain not_assessed")
        if config.get("scientific_promotion_allowed") is not False:
            raise ValueError("world-model region calibration cannot claim promotion")
        kwargs = {field.name: config[field.name] for field in dataclasses.fields(cls)}
        restored = cls(**kwargs)
        if not _strict_json_equal(dict(config), restored.to_config()):
            raise ValueError("world-model region calibration config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class WorldModelNextStateErrorGate:
    """Separate retained next-state error quantile and fixed ceiling."""

    available: Array
    support_count: Array
    preupdate_rmse_quantile: Array
    caller_max_rmse: Array
    passed: Array


@chex.dataclass(frozen=True)
class WorldModelRewardErrorGate:
    """Separate retained reward error quantile and fixed ceiling."""

    available: Array
    support_count: Array
    preupdate_abs_error_quantile: Array
    caller_max_abs_error: Array
    passed: Array


@chex.dataclass(frozen=True)
class WorldModelEpistemicCalibrationGate:
    """Disagreement-to-realized-excess-error calibration diagnostics."""

    available: Array
    support_count: Array
    current_disagreements: Array
    preupdate_next_state_error_ratio_quantile: Array
    preupdate_reward_error_ratio_quantile: Array
    calibrated_error_bounds: Array
    caller_max_error_bounds: Array
    disagreement_error_correlations: Array
    association_available: Array
    component_passed: Array
    passed: Array


@chex.dataclass(frozen=True)
class WorldModelAleatoricCalibrationGate:
    """Per-head coverage/NLL diagnostics and independent noisy-TV veto."""

    available: Array
    support_count: Array
    caller_alpha: Array
    nominal_standardized_residual_limit: Array
    required_marginal_coverage: Array
    current_mean_variances: Array
    preupdate_variance_quantiles: Array
    preupdate_standardized_residual_quantiles: Array
    preupdate_gaussian_nll_quantiles: Array
    empirical_marginal_coverage: Array
    caller_max_variance: Array
    caller_max_gaussian_nll: Array
    coverage_passed: Array
    nll_passed: Array
    noise_vetoed: Array
    passed: Array


@chex.dataclass(frozen=True)
class WorldModelTerminationCalibrationGate:
    """Uncensored Bernoulli support, Brier, and calibration-gap gate."""

    available: Array
    support_count: Array
    terminal_support_count: Array
    continuing_support_count: Array
    current_mean_termination_probability: Array
    current_member_disagreement: Array
    preupdate_brier_quantile: Array
    preupdate_calibration_gap: Array
    caller_max_brier: Array
    passed: Array


@chex.dataclass(frozen=True)
class WorldModelPlanningCalibrationGates:
    """Noncompensating empirical planning diagnostics from one prestate cell."""

    epistemic: WorldModelEpistemicCalibrationGate
    aleatoric: WorldModelAleatoricCalibrationGate
    next_state_error: WorldModelNextStateErrorGate
    reward_error: WorldModelRewardErrorGate
    termination: WorldModelTerminationCalibrationGate
    descriptive_every_gate_available: Array
    descriptive_every_gate_passed: Array


@chex.dataclass(frozen=True)
class WorldModelPredictBeforeOutcomeReceipt:
    """Content-bound, read-only prediction and pre-update calibration receipt."""

    lifecycle_id_words: Array
    decision_id_words: Array
    model_revision_words: Array
    representation_revision_words: Array
    action_revision_words: Array
    region_revision_words: Array
    calibration_revision_words: Array
    cell_revision_words: Array
    action: Array
    region: Array
    member_mean_predictions: Array
    member_aleatoric_variances: Array
    member_termination_probabilities: Array
    gates: WorldModelPlanningCalibrationGates
    prediction_content_tag: Array
    calibration_content_tag: Array
    gates_content_tag: Array
    receipt_integrity_tag: Array
    planning_authority: Array
    safety_authority: Array
    valid: Array


@chex.dataclass(frozen=True)
class WorldModelCalibrationOutcome:
    """One authoritative real outcome matching an exact prediction receipt."""

    lifecycle_id_words: Array
    decision_id_words: Array
    action: Array
    region: Array
    next_state: Array
    reward: Array
    terminated: Array
    truncated: Array


@chex.dataclass(frozen=True)
class WorldModelRegionCalibrationState:
    """Complete fixed-capacity region/action owner state."""

    lifecycle_id_words: Array
    record_valid: Array
    termination_observed: Array
    terminal_targets: Array
    next_state_squared_errors: Array
    reward_squared_errors: Array
    next_state_epistemic_disagreements: Array
    reward_epistemic_disagreements: Array
    next_state_epistemic_error_ratios: Array
    reward_epistemic_error_ratios: Array
    mean_aleatoric_variances: Array
    absolute_standardized_residuals: Array
    gaussian_nll: Array
    nominal_interval_covered: Array
    mean_termination_probabilities: Array
    termination_brier_errors: Array
    cell_sizes: Array
    cell_write_indices: Array
    termination_support_counts: Array
    terminal_support_counts: Array
    continuing_support_counts: Array
    cell_count_words: Array
    accepted_count_words: Array
    last_decision_id_words: Array


@chex.dataclass(frozen=True)
class WorldModelCalibrationOutcomeDiagnostics:
    """Separately typed realized channels from one accepted real outcome."""

    next_state_squared_error: Array
    reward_squared_error: Array
    next_state_epistemic_disagreement: Array
    reward_epistemic_disagreement: Array
    next_state_mean_aleatoric_variance: Array
    reward_aleatoric_variance: Array
    next_state_epistemic_error_ratio: Array
    reward_epistemic_error_ratio: Array
    absolute_standardized_residuals: Array
    gaussian_nll: Array
    nominal_interval_covered: Array
    mean_termination_probability: Array
    termination_target: Array
    termination_observed: Array
    termination_brier_error: Array


@chex.dataclass(frozen=True)
class WorldModelCalibrationTransactionDiagnostics:
    """Identity, integrity, capacity, and atomic settlement audit."""

    state_static_contract_valid: Array
    receipt_static_contract_valid: Array
    outcome_static_contract_valid: Array
    state_valid: Array
    receipt_valid: Array
    lifecycle_matches: Array
    decision_matches: Array
    cell_identity_matches: Array
    decision_fresh: Array
    outcome_finite: Array
    boundary_semantics_valid: Array
    event_capacity_available: Array
    candidate_state_valid: Array
    applied: Array
    rejected: Array
    pre_accepted_count_words: Array
    post_accepted_count_words: Array
    pre_cell_count_words: Array
    post_cell_count_words: Array


@chex.dataclass(frozen=True)
class WorldModelCalibrationSettlementResult:
    """Owner state plus exact realized and transaction diagnostics."""

    state: WorldModelRegionCalibrationState
    receipt: WorldModelPredictBeforeOutcomeReceipt
    gates: WorldModelPlanningCalibrationGates
    outcome: WorldModelCalibrationOutcomeDiagnostics
    transaction: WorldModelCalibrationTransactionDiagnostics


@dataclasses.dataclass(frozen=True)
class WorldModelRegionCalibrationResourceBudget:
    """Exact logical persistent/receipt sizes and ownership ceilings."""

    persistent_bytes_scope: str
    receipt_bytes_scope: str
    temporary_bytes_scope: str
    region_action_cells: int
    records_per_cell: int
    retained_record_capacity: int
    target_dim: int
    ensemble_size: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    receipt_scalars: int
    receipt_bytes: int
    max_settlements: int
    max_model_updates_per_settlement: int
    max_planner_updates_per_settlement: int
    model_state_owned: int
    representation_state_owned: int
    planning_authority: int
    safety_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _array_contract(value: object, *, shape: tuple[int, ...], dtype: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and cast(Any, value).shape == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not _array_contract(value, shape=shape, dtype=dtype):
        raise TypeError(f"{name} must have exact shape {shape} and dtype {jnp.dtype(dtype)}")
    return jnp.asarray(value)


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _checked_words_add(left: Array, right: Array) -> tuple[Array, Array]:
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_high = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_carry = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    return jnp.stack((high, low)), ~(overflow_high | overflow_carry)


def _words_mod_small(words: Array, modulus: int) -> Array:
    modulus_u = jnp.asarray(modulus, dtype=jnp.uint32)
    two32_mod = jnp.asarray((1 << 32) % modulus, dtype=jnp.uint32)
    high_term = (words[0] % modulus_u) * two32_mod
    return ((high_term + (words[1] % modulus_u)) % modulus_u).astype(jnp.int32)


def _words_leq_limit(words: Array, limit: int) -> Array:
    limit_words = jnp.asarray(
        [(limit >> 32) & 0xFFFFFFFF, limit & 0xFFFFFFFF],
        dtype=jnp.uint32,
    )
    return _words_less_equal(words, limit_words)


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        equal = equal & jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
    return equal


def _tree_content_words(tree: object) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if array.dtype == jnp.dtype(jnp.float32):
            words = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype in (
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported receipt content dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(parts)


def _content_tag(tree: object, *, salt: int) -> Array:
    words = _tree_content_words(tree)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ words[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    tag = jax.lax.fori_loop(
        0,
        words.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(
        tag == jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(salt, dtype=jnp.uint32),
        tag,
    )


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class WorldModelRegionCalibration:
    """Fixed-budget owner of causal state/action-region calibration cells."""

    def __init__(self, config: WorldModelRegionCalibrationConfig) -> None:
        if type(config) is not WorldModelRegionCalibrationConfig:
            raise TypeError("config must be an exact WorldModelRegionCalibrationConfig")
        self._config = config

    @property
    def config(self) -> WorldModelRegionCalibrationConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> WorldModelRegionCalibration:
        return cls(WorldModelRegionCalibrationConfig.from_config(config))

    def _empty_state(self, lifecycle_id_words: Array) -> WorldModelRegionCalibrationState:
        cfg = self._config
        cell_shape = (cfg.n_regions, cfg.n_actions)
        record_shape = (*cell_shape, cfg.capacity_per_cell)
        target_shape = (*record_shape, cfg.target_dim)
        zero_record = jnp.zeros(record_shape, dtype=jnp.float32)
        return WorldModelRegionCalibrationState(
            lifecycle_id_words=lifecycle_id_words,
            record_valid=jnp.zeros(record_shape, dtype=jnp.bool_),
            termination_observed=jnp.zeros(record_shape, dtype=jnp.bool_),
            terminal_targets=jnp.zeros(record_shape, dtype=jnp.bool_),
            next_state_squared_errors=zero_record,
            reward_squared_errors=zero_record,
            next_state_epistemic_disagreements=zero_record,
            reward_epistemic_disagreements=zero_record,
            next_state_epistemic_error_ratios=zero_record,
            reward_epistemic_error_ratios=zero_record,
            mean_aleatoric_variances=jnp.zeros(target_shape, dtype=jnp.float32),
            absolute_standardized_residuals=jnp.zeros(
                target_shape,
                dtype=jnp.float32,
            ),
            gaussian_nll=jnp.zeros(target_shape, dtype=jnp.float32),
            nominal_interval_covered=jnp.zeros(target_shape, dtype=jnp.bool_),
            mean_termination_probabilities=zero_record,
            termination_brier_errors=zero_record,
            cell_sizes=jnp.zeros(cell_shape, dtype=jnp.int32),
            cell_write_indices=jnp.zeros(cell_shape, dtype=jnp.int32),
            termination_support_counts=jnp.zeros(cell_shape, dtype=jnp.int32),
            terminal_support_counts=jnp.zeros(cell_shape, dtype=jnp.int32),
            continuing_support_counts=jnp.zeros(cell_shape, dtype=jnp.int32),
            cell_count_words=jnp.zeros((*cell_shape, 2), dtype=jnp.uint32),
            accepted_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_decision_id_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def init(self, lifecycle_id_words: Array) -> WorldModelRegionCalibrationState:
        """Create a zeroed owner for one caller-declared nonzero lifecycle."""

        lifecycle = _require_array(
            lifecycle_id_words,
            name="lifecycle_id_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        if not bool(jax.device_get(_words_nonzero(lifecycle))):
            raise ValueError("lifecycle_id_words must be nonzero")
        return self._empty_state(lifecycle)

    def _state_static_valid(self, state: object) -> bool:
        if type(state) is not WorldModelRegionCalibrationState:
            return False
        cfg = self._config
        cell = (cfg.n_regions, cfg.n_actions)
        record = (*cell, cfg.capacity_per_cell)
        target = (*record, cfg.target_dim)
        checks = (
            (state.lifecycle_id_words, (2,), jnp.uint32),
            (state.record_valid, record, jnp.bool_),
            (state.termination_observed, record, jnp.bool_),
            (state.terminal_targets, record, jnp.bool_),
            (state.next_state_squared_errors, record, jnp.float32),
            (state.reward_squared_errors, record, jnp.float32),
            (state.next_state_epistemic_disagreements, record, jnp.float32),
            (state.reward_epistemic_disagreements, record, jnp.float32),
            (state.next_state_epistemic_error_ratios, record, jnp.float32),
            (state.reward_epistemic_error_ratios, record, jnp.float32),
            (state.mean_aleatoric_variances, target, jnp.float32),
            (state.absolute_standardized_residuals, target, jnp.float32),
            (state.gaussian_nll, target, jnp.float32),
            (state.nominal_interval_covered, target, jnp.bool_),
            (state.mean_termination_probabilities, record, jnp.float32),
            (state.termination_brier_errors, record, jnp.float32),
            (state.cell_sizes, cell, jnp.int32),
            (state.cell_write_indices, cell, jnp.int32),
            (state.termination_support_counts, cell, jnp.int32),
            (state.terminal_support_counts, cell, jnp.int32),
            (state.continuing_support_counts, cell, jnp.int32),
            (state.cell_count_words, (*cell, 2), jnp.uint32),
            (state.accepted_count_words, (2,), jnp.uint32),
            (state.last_decision_id_words, (2,), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    def _state_valid(self, state: WorldModelRegionCalibrationState) -> Array:
        cfg = self._config
        valid = _words_nonzero(state.lifecycle_id_words)
        valid = (
            valid
            & jnp.all(state.cell_sizes >= 0)
            & jnp.all(state.cell_sizes <= cfg.capacity_per_cell)
            & jnp.all(state.cell_write_indices >= 0)
            & jnp.all(state.cell_write_indices < cfg.capacity_per_cell)
            & jnp.all(state.termination_support_counts >= 0)
            & jnp.all(state.terminal_support_counts >= 0)
            & jnp.all(state.continuing_support_counts >= 0)
            & jnp.all(
                state.termination_support_counts
                == state.terminal_support_counts + state.continuing_support_counts
            )
            & jnp.all(state.termination_support_counts <= state.cell_sizes)
            & _words_leq_limit(state.accepted_count_words, cfg.max_events)
        )

        record_index = jnp.arange(cfg.capacity_per_cell, dtype=jnp.int32)
        expected_record_valid = record_index[None, None, :] < state.cell_sizes[:, :, None]
        expected_record_valid = jnp.where(
            state.cell_sizes[:, :, None] == cfg.capacity_per_cell,
            jnp.ones_like(expected_record_valid),
            expected_record_valid,
        )
        valid = valid & jnp.array_equal(state.record_valid, expected_record_valid)
        valid = valid & jnp.all(~state.termination_observed | state.record_valid)
        valid = valid & jnp.all(~state.terminal_targets | state.termination_observed)

        expected_writes: list[Array] = []
        total_words = jnp.zeros((2,), dtype=jnp.uint32)
        total_ok = jnp.asarray(True, dtype=jnp.bool_)
        for region in range(cfg.n_regions):
            row: list[Array] = []
            for action in range(cfg.n_actions):
                words = state.cell_count_words[region, action]
                row.append(_words_mod_small(words, cfg.capacity_per_cell))
                valid = valid & _words_leq_limit(words, cfg.max_events)
                enough = (words[0] != 0) | (
                    words[1] >= state.cell_sizes[region, action].astype(jnp.uint32)
                )
                valid = valid & enough
                total_words, add_ok = _checked_words_add(total_words, words)
                total_ok = total_ok & add_ok
            expected_writes.append(jnp.stack(row))
        valid = (
            valid
            & total_ok
            & jnp.array_equal(jnp.stack(expected_writes), state.cell_write_indices)
            & jnp.array_equal(total_words, state.accepted_count_words)
        )
        any_event = _words_nonzero(state.accepted_count_words)
        valid = valid & (any_event == _words_nonzero(state.last_decision_id_words))

        termination_counts = jnp.sum(
            state.termination_observed.astype(jnp.int32),
            axis=2,
        )
        terminal_counts = jnp.sum(
            (state.termination_observed & state.terminal_targets).astype(jnp.int32),
            axis=2,
        )
        continuing_counts = jnp.sum(
            (state.termination_observed & ~state.terminal_targets).astype(jnp.int32),
            axis=2,
        )
        valid = (
            valid
            & jnp.array_equal(termination_counts, state.termination_support_counts)
            & jnp.array_equal(terminal_counts, state.terminal_support_counts)
            & jnp.array_equal(continuing_counts, state.continuing_support_counts)
        )

        finite_fields = (
            state.next_state_squared_errors,
            state.reward_squared_errors,
            state.next_state_epistemic_disagreements,
            state.reward_epistemic_disagreements,
            state.next_state_epistemic_error_ratios,
            state.reward_epistemic_error_ratios,
            state.mean_aleatoric_variances,
            state.absolute_standardized_residuals,
            state.gaussian_nll,
            state.mean_termination_probabilities,
            state.termination_brier_errors,
        )
        for value in finite_fields:
            valid = valid & jnp.all(jnp.isfinite(value))
        nonnegative_fields = (
            state.next_state_squared_errors,
            state.reward_squared_errors,
            state.next_state_epistemic_disagreements,
            state.reward_epistemic_disagreements,
            state.next_state_epistemic_error_ratios,
            state.reward_epistemic_error_ratios,
            state.mean_aleatoric_variances,
            state.absolute_standardized_residuals,
            state.termination_brier_errors,
        )
        for value in nonnegative_fields:
            valid = valid & jnp.all(value >= 0.0)
        valid = (
            valid
            & jnp.all(state.mean_termination_probabilities >= 0.0)
            & jnp.all(state.mean_termination_probabilities <= 1.0)
        )

        invalid = ~state.record_valid
        invalid_target = invalid[..., None]
        scalar_buffers = (
            state.next_state_squared_errors,
            state.reward_squared_errors,
            state.next_state_epistemic_disagreements,
            state.reward_epistemic_disagreements,
            state.next_state_epistemic_error_ratios,
            state.reward_epistemic_error_ratios,
            state.mean_termination_probabilities,
            state.termination_brier_errors,
        )
        for value in scalar_buffers:
            valid = valid & jnp.all(jnp.where(invalid, value == 0.0, True))
        target_buffers = (
            state.mean_aleatoric_variances,
            state.absolute_standardized_residuals,
            state.gaussian_nll,
        )
        for value in target_buffers:
            valid = valid & jnp.all(jnp.where(invalid_target, value == 0.0, True))
        valid = (
            valid
            & jnp.all(jnp.where(invalid, ~state.termination_observed, True))
            & jnp.all(jnp.where(invalid, ~state.terminal_targets, True))
            & jnp.all(jnp.where(invalid_target, ~state.nominal_interval_covered, True))
        )
        return jnp.asarray(valid, dtype=jnp.bool_)

    def state_valid(self, state: WorldModelRegionCalibrationState) -> Array:
        """Return dynamic state validity after enforcing exact static shapes."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        return self._state_valid(state)

    def content_tag(self, state: WorldModelRegionCalibrationState) -> Array:
        """Bind every persistent calibration-state array to one read-only tag."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        return _content_tag(state, salt=_CALIBRATION_STATE_TAG_SALT)

    def _masked_upper_quantile(
        self,
        values: Array,
        valid: Array,
        count: Array,
    ) -> Array:
        sentinel = jnp.asarray(jnp.inf, dtype=jnp.float32)
        mask = valid if values.ndim == 1 else valid[:, None]
        ordered = jnp.sort(jnp.where(mask, values, sentinel), axis=0)
        index = jnp.ceil(
            (1.0 - self._config.alpha) * count.astype(jnp.float32)
        ).astype(jnp.int32) - 1
        index = jnp.clip(index, 0, self._config.capacity_per_cell - 1)
        selected = ordered[index]
        return jnp.where(count > 0, selected, jnp.zeros_like(selected))

    def _masked_correlation(
        self,
        left: Array,
        right: Array,
        valid: Array,
        count: Array,
    ) -> tuple[Array, Array]:
        denominator = jnp.maximum(count.astype(jnp.float32), 1.0)
        left_masked = jnp.where(valid, left, 0.0)
        right_masked = jnp.where(valid, right, 0.0)
        left_mean = jnp.sum(left_masked) / denominator
        right_mean = jnp.sum(right_masked) / denominator
        left_centered = jnp.where(valid, left - left_mean, 0.0)
        right_centered = jnp.where(valid, right - right_mean, 0.0)
        covariance = jnp.sum(left_centered * right_centered)
        left_scale = jnp.sum(jnp.square(left_centered))
        right_scale = jnp.sum(jnp.square(right_centered))
        association_available = (
            (count >= 2)
            & (left_scale > self._config.disagreement_floor)
            & (right_scale > self._config.disagreement_floor)
        )
        correlation = covariance / jnp.sqrt(
            jnp.maximum(
                left_scale * right_scale,
                self._config.disagreement_floor,
            )
        )
        return jnp.where(association_available, correlation, 0.0), association_available

    def _cell_content_tag(
        self,
        state: WorldModelRegionCalibrationState,
        region: Array,
        action: Array,
    ) -> Array:
        cell = (
            state.lifecycle_id_words,
            state.accepted_count_words,
            state.cell_count_words[region, action],
            state.cell_sizes[region, action],
            state.cell_write_indices[region, action],
            state.termination_support_counts[region, action],
            state.terminal_support_counts[region, action],
            state.continuing_support_counts[region, action],
            state.record_valid[region, action],
            state.termination_observed[region, action],
            state.terminal_targets[region, action],
            state.next_state_squared_errors[region, action],
            state.reward_squared_errors[region, action],
            state.next_state_epistemic_disagreements[region, action],
            state.reward_epistemic_disagreements[region, action],
            state.next_state_epistemic_error_ratios[region, action],
            state.reward_epistemic_error_ratios[region, action],
            state.mean_aleatoric_variances[region, action],
            state.absolute_standardized_residuals[region, action],
            state.gaussian_nll[region, action],
            state.nominal_interval_covered[region, action],
            state.mean_termination_probabilities[region, action],
            state.termination_brier_errors[region, action],
        )
        return _content_tag(cell, salt=_CALIBRATION_TAG_SALT)

    def _gates(
        self,
        state: WorldModelRegionCalibrationState,
        region: Array,
        action: Array,
        member_means: Array,
        member_variances: Array,
        member_termination: Array,
    ) -> WorldModelPlanningCalibrationGates:
        cfg = self._config
        valid = state.record_valid[region, action]
        count = state.cell_sizes[region, action]
        available = count >= cfg.min_samples
        means = jnp.mean(member_means, axis=0)
        del means
        epistemic_per_head = jnp.var(member_means, axis=0)
        current_next_epistemic = jnp.mean(epistemic_per_head[: cfg.observation_dim])
        current_reward_epistemic = epistemic_per_head[-1]
        current_disagreements = jnp.stack(
            (current_next_epistemic, current_reward_epistemic)
        )

        next_ratio_quantile = self._masked_upper_quantile(
            state.next_state_epistemic_error_ratios[region, action],
            valid,
            count,
        )
        reward_ratio_quantile = self._masked_upper_quantile(
            state.reward_epistemic_error_ratios[region, action],
            valid,
            count,
        )
        error_bounds = current_disagreements * jnp.stack(
            (next_ratio_quantile, reward_ratio_quantile)
        )
        max_error_bounds = jnp.asarray(
            (
                cfg.max_epistemic_next_state_bound,
                cfg.max_epistemic_reward_bound,
            ),
            dtype=jnp.float32,
        )
        next_correlation, next_association = self._masked_correlation(
            state.next_state_epistemic_disagreements[region, action],
            state.next_state_squared_errors[region, action],
            valid,
            count,
        )
        reward_correlation, reward_association = self._masked_correlation(
            state.reward_epistemic_disagreements[region, action],
            state.reward_squared_errors[region, action],
            valid,
            count,
        )
        epistemic_components = available & (error_bounds <= max_error_bounds)
        epistemic = WorldModelEpistemicCalibrationGate(
            available=available,
            support_count=count,
            current_disagreements=current_disagreements,
            preupdate_next_state_error_ratio_quantile=next_ratio_quantile,
            preupdate_reward_error_ratio_quantile=reward_ratio_quantile,
            calibrated_error_bounds=error_bounds,
            caller_max_error_bounds=max_error_bounds,
            disagreement_error_correlations=jnp.stack(
                (next_correlation, reward_correlation)
            ),
            association_available=jnp.stack((next_association, reward_association)),
            component_passed=epistemic_components,
            passed=jnp.all(epistemic_components),
        )

        next_error_quantile = jnp.sqrt(
            self._masked_upper_quantile(
                state.next_state_squared_errors[region, action],
                valid,
                count,
            )
        )
        reward_error_quantile = jnp.sqrt(
            self._masked_upper_quantile(
                state.reward_squared_errors[region, action],
                valid,
                count,
            )
        )
        next_state_error = WorldModelNextStateErrorGate(
            available=available,
            support_count=count,
            preupdate_rmse_quantile=next_error_quantile,
            caller_max_rmse=jnp.asarray(cfg.max_next_state_rmse, dtype=jnp.float32),
            passed=available & (next_error_quantile <= cfg.max_next_state_rmse),
        )
        reward_error = WorldModelRewardErrorGate(
            available=available,
            support_count=count,
            preupdate_abs_error_quantile=reward_error_quantile,
            caller_max_abs_error=jnp.asarray(
                cfg.max_reward_abs_error,
                dtype=jnp.float32,
            ),
            passed=available & (reward_error_quantile <= cfg.max_reward_abs_error),
        )

        current_variances = jnp.mean(member_variances, axis=0)
        variance_quantiles = self._masked_upper_quantile(
            state.mean_aleatoric_variances[region, action],
            valid,
            count,
        )
        standardized_quantiles = self._masked_upper_quantile(
            state.absolute_standardized_residuals[region, action],
            valid,
            count,
        )
        nll_quantiles = self._masked_upper_quantile(
            state.gaussian_nll[region, action],
            valid,
            count,
        )
        coverage = jnp.sum(
            jnp.where(
                valid[:, None],
                state.nominal_interval_covered[region, action].astype(jnp.float32),
                0.0,
            ),
            axis=0,
        ) / jnp.maximum(count.astype(jnp.float32), 1.0)
        required_coverage = jnp.asarray(1.0 - cfg.alpha, dtype=jnp.float32)
        coverage_passed = available & jnp.all(coverage >= required_coverage)
        nll_passed = available & jnp.all(nll_quantiles <= cfg.max_gaussian_nll)
        noise_vetoed = jnp.any(current_variances > cfg.max_aleatoric_variance)
        aleatoric = WorldModelAleatoricCalibrationGate(
            available=available,
            support_count=count,
            caller_alpha=jnp.asarray(cfg.alpha, dtype=jnp.float32),
            nominal_standardized_residual_limit=jnp.asarray(
                cfg.nominal_standardized_residual_limit,
                dtype=jnp.float32,
            ),
            required_marginal_coverage=required_coverage,
            current_mean_variances=current_variances,
            preupdate_variance_quantiles=variance_quantiles,
            preupdate_standardized_residual_quantiles=standardized_quantiles,
            preupdate_gaussian_nll_quantiles=nll_quantiles,
            empirical_marginal_coverage=coverage,
            caller_max_variance=jnp.asarray(
                cfg.max_aleatoric_variance,
                dtype=jnp.float32,
            ),
            caller_max_gaussian_nll=jnp.asarray(
                cfg.max_gaussian_nll,
                dtype=jnp.float32,
            ),
            coverage_passed=coverage_passed,
            nll_passed=nll_passed,
            noise_vetoed=noise_vetoed,
            passed=available & coverage_passed & nll_passed & ~noise_vetoed,
        )

        termination_valid = state.termination_observed[region, action]
        termination_count = state.termination_support_counts[region, action]
        terminal_count = state.terminal_support_counts[region, action]
        continuing_count = state.continuing_support_counts[region, action]
        termination_available = (
            (termination_count >= cfg.min_termination_samples)
            & (terminal_count >= cfg.min_termination_class_support)
            & (continuing_count >= cfg.min_termination_class_support)
        )
        brier_quantile = self._masked_upper_quantile(
            state.termination_brier_errors[region, action],
            termination_valid,
            termination_count,
        )
        probability_sum = jnp.sum(
            jnp.where(
                termination_valid,
                state.mean_termination_probabilities[region, action],
                0.0,
            )
        )
        target_sum = jnp.sum(
            jnp.where(
                termination_valid,
                state.terminal_targets[region, action].astype(jnp.float32),
                0.0,
            )
        )
        termination_denominator = jnp.maximum(
            termination_count.astype(jnp.float32),
            1.0,
        )
        calibration_gap = jnp.abs(
            probability_sum / termination_denominator
            - target_sum / termination_denominator
        )
        current_termination_mean = jnp.mean(member_termination)
        current_termination_disagreement = jnp.var(member_termination)
        termination = WorldModelTerminationCalibrationGate(
            available=termination_available,
            support_count=termination_count,
            terminal_support_count=terminal_count,
            continuing_support_count=continuing_count,
            current_mean_termination_probability=current_termination_mean,
            current_member_disagreement=current_termination_disagreement,
            preupdate_brier_quantile=brier_quantile,
            preupdate_calibration_gap=calibration_gap,
            caller_max_brier=jnp.asarray(
                cfg.max_termination_brier,
                dtype=jnp.float32,
            ),
            passed=(
                termination_available
                & (brier_quantile <= cfg.max_termination_brier)
                & (jnp.square(calibration_gap) <= cfg.max_termination_brier)
            ),
        )
        every_available = (
            epistemic.available
            & aleatoric.available
            & next_state_error.available
            & reward_error.available
            & termination.available
        )
        every_passed = (
            epistemic.passed
            & aleatoric.passed
            & next_state_error.passed
            & reward_error.passed
            & termination.passed
        )
        return WorldModelPlanningCalibrationGates(
            epistemic=epistemic,
            aleatoric=aleatoric,
            next_state_error=next_state_error,
            reward_error=reward_error,
            termination=termination,
            descriptive_every_gate_available=every_available,
            descriptive_every_gate_passed=every_passed,
        )

    def _gates_static_valid(self, gates: object) -> bool:
        if type(gates) is not WorldModelPlanningCalibrationGates:
            return False
        cfg = self._config
        if type(gates.next_state_error) is not WorldModelNextStateErrorGate:
            return False
        if type(gates.reward_error) is not WorldModelRewardErrorGate:
            return False
        if type(gates.epistemic) is not WorldModelEpistemicCalibrationGate:
            return False
        if type(gates.aleatoric) is not WorldModelAleatoricCalibrationGate:
            return False
        if type(gates.termination) is not WorldModelTerminationCalibrationGate:
            return False
        scalar_bool = (
            gates.next_state_error.available,
            gates.next_state_error.passed,
            gates.reward_error.available,
            gates.reward_error.passed,
            gates.epistemic.available,
            gates.epistemic.passed,
            gates.aleatoric.available,
            gates.aleatoric.coverage_passed,
            gates.aleatoric.nll_passed,
            gates.aleatoric.noise_vetoed,
            gates.aleatoric.passed,
            gates.termination.available,
            gates.termination.passed,
            gates.descriptive_every_gate_available,
            gates.descriptive_every_gate_passed,
        )
        scalar_int = (
            gates.next_state_error.support_count,
            gates.reward_error.support_count,
            gates.epistemic.support_count,
            gates.aleatoric.support_count,
            gates.termination.support_count,
            gates.termination.terminal_support_count,
            gates.termination.continuing_support_count,
        )
        scalar_float = (
            gates.next_state_error.preupdate_rmse_quantile,
            gates.next_state_error.caller_max_rmse,
            gates.reward_error.preupdate_abs_error_quantile,
            gates.reward_error.caller_max_abs_error,
            gates.epistemic.preupdate_next_state_error_ratio_quantile,
            gates.epistemic.preupdate_reward_error_ratio_quantile,
            gates.aleatoric.caller_alpha,
            gates.aleatoric.nominal_standardized_residual_limit,
            gates.aleatoric.required_marginal_coverage,
            gates.aleatoric.caller_max_variance,
            gates.aleatoric.caller_max_gaussian_nll,
            gates.termination.current_mean_termination_probability,
            gates.termination.current_member_disagreement,
            gates.termination.preupdate_brier_quantile,
            gates.termination.preupdate_calibration_gap,
            gates.termination.caller_max_brier,
        )
        vectors = (
            (gates.epistemic.current_disagreements, (2,), jnp.float32),
            (gates.epistemic.calibrated_error_bounds, (2,), jnp.float32),
            (gates.epistemic.caller_max_error_bounds, (2,), jnp.float32),
            (gates.epistemic.disagreement_error_correlations, (2,), jnp.float32),
            (gates.epistemic.association_available, (2,), jnp.bool_),
            (gates.epistemic.component_passed, (2,), jnp.bool_),
            (
                gates.aleatoric.current_mean_variances,
                (cfg.target_dim,),
                jnp.float32,
            ),
            (
                gates.aleatoric.preupdate_variance_quantiles,
                (cfg.target_dim,),
                jnp.float32,
            ),
            (
                gates.aleatoric.preupdate_standardized_residual_quantiles,
                (cfg.target_dim,),
                jnp.float32,
            ),
            (
                gates.aleatoric.preupdate_gaussian_nll_quantiles,
                (cfg.target_dim,),
                jnp.float32,
            ),
            (
                gates.aleatoric.empirical_marginal_coverage,
                (cfg.target_dim,),
                jnp.float32,
            ),
        )
        return (
            all(_array_contract(value, shape=(), dtype=jnp.bool_) for value in scalar_bool)
            and all(_array_contract(value, shape=(), dtype=jnp.int32) for value in scalar_int)
            and all(
                _array_contract(value, shape=(), dtype=jnp.float32)
                for value in scalar_float
            )
            and all(
                _array_contract(value, shape=shape, dtype=dtype)
                for value, shape, dtype in vectors
            )
        )

    def _receipt_static_valid(self, receipt: object) -> bool:
        if type(receipt) is not WorldModelPredictBeforeOutcomeReceipt:
            return False
        cfg = self._config
        checks = (
            (receipt.lifecycle_id_words, (2,), jnp.uint32),
            (receipt.decision_id_words, (2,), jnp.uint32),
            (receipt.model_revision_words, (2,), jnp.uint32),
            (receipt.representation_revision_words, (2,), jnp.uint32),
            (receipt.action_revision_words, (2,), jnp.uint32),
            (receipt.region_revision_words, (2,), jnp.uint32),
            (receipt.calibration_revision_words, (2,), jnp.uint32),
            (receipt.cell_revision_words, (2,), jnp.uint32),
            (receipt.action, (), jnp.int32),
            (receipt.region, (), jnp.int32),
            (
                receipt.member_mean_predictions,
                (cfg.ensemble_size, cfg.target_dim),
                jnp.float32,
            ),
            (
                receipt.member_aleatoric_variances,
                (cfg.ensemble_size, cfg.target_dim),
                jnp.float32,
            ),
            (
                receipt.member_termination_probabilities,
                (cfg.ensemble_size,),
                jnp.float32,
            ),
            (receipt.prediction_content_tag, (), jnp.uint32),
            (receipt.calibration_content_tag, (), jnp.uint32),
            (receipt.gates_content_tag, (), jnp.uint32),
            (receipt.receipt_integrity_tag, (), jnp.uint32),
            (receipt.planning_authority, (), jnp.bool_),
            (receipt.safety_authority, (), jnp.bool_),
            (receipt.valid, (), jnp.bool_),
        )
        return self._gates_static_valid(receipt.gates) and all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    def _receipt_tag_payload(
        self,
        *,
        lifecycle: Array,
        decision: Array,
        model_revision: Array,
        representation_revision: Array,
        action_revision: Array,
        region_revision: Array,
        calibration_revision: Array,
        cell_revision: Array,
        action: Array,
        region: Array,
        prediction_tag: Array,
        calibration_tag: Array,
        gates_tag: Array,
        valid: Array,
    ) -> tuple[Array, ...]:
        return (
            lifecycle,
            decision,
            model_revision,
            representation_revision,
            action_revision,
            region_revision,
            calibration_revision,
            cell_revision,
            action,
            region,
            prediction_tag,
            calibration_tag,
            gates_tag,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(False, dtype=jnp.bool_),
            valid,
        )

    def _build_receipt(
        self,
        state: WorldModelRegionCalibrationState,
        *,
        lifecycle: Array,
        decision: Array,
        model_revision: Array,
        representation_revision: Array,
        action_revision: Array,
        region_revision: Array,
        action: Array,
        region: Array,
        member_means: Array,
        member_variances: Array,
        member_termination: Array,
    ) -> WorldModelPredictBeforeOutcomeReceipt:
        cfg = self._config
        safe_action = jnp.clip(action, 0, cfg.n_actions - 1)
        safe_region = jnp.clip(region, 0, cfg.n_regions - 1)
        raw_finite = (
            jnp.all(jnp.isfinite(member_means))
            & jnp.all(jnp.isfinite(member_variances))
            & jnp.all(jnp.isfinite(member_termination))
        )
        predictions_valid = (
            raw_finite
            & jnp.all(jnp.abs(member_means) <= cfg.max_prediction_magnitude)
            & jnp.all(member_variances >= cfg.variance_floor)
            & jnp.all(member_variances <= _FLOAT32_MAX)
            & jnp.all(member_termination >= 0.0)
            & jnp.all(member_termination <= 1.0)
        )
        identity_valid = (
            jnp.array_equal(lifecycle, state.lifecycle_id_words)
            & _words_nonzero(lifecycle)
            & _words_nonzero(decision)
            & _words_nonzero(representation_revision)
            & _words_nonzero(action_revision)
            & _words_nonzero(region_revision)
            & (action >= 0)
            & (action < cfg.n_actions)
            & (region >= 0)
            & (region < cfg.n_regions)
            & _words_less(state.last_decision_id_words, decision)
        )
        capacity = _words_leq_limit(state.accepted_count_words, cfg.max_events - 1)
        base_valid = self._state_valid(state) & identity_valid & predictions_valid & capacity
        safe_means = jnp.clip(
            jnp.nan_to_num(member_means, nan=0.0, posinf=0.0, neginf=0.0),
            -cfg.max_prediction_magnitude,
            cfg.max_prediction_magnitude,
        )
        safe_variances = jnp.clip(
            jnp.nan_to_num(
                member_variances,
                nan=cfg.variance_floor,
                posinf=cfg.variance_floor,
                neginf=cfg.variance_floor,
            ),
            cfg.variance_floor,
            _FLOAT32_MAX,
        )
        safe_termination = jnp.clip(
            jnp.nan_to_num(member_termination, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )
        gates = self._gates(
            state,
            safe_region,
            safe_action,
            safe_means,
            safe_variances,
            safe_termination,
        )
        calibration_revision = state.accepted_count_words
        cell_revision = state.cell_count_words[safe_region, safe_action]
        prediction_payload = (
            lifecycle,
            decision,
            model_revision,
            representation_revision,
            action_revision,
            region_revision,
            action,
            region,
            member_means,
            member_variances,
            member_termination,
        )
        prediction_tag = _content_tag(
            prediction_payload,
            salt=_PREDICTION_TAG_SALT,
        )
        calibration_tag = self._cell_content_tag(state, safe_region, safe_action)
        gates_tag = _content_tag(gates, salt=_GATES_TAG_SALT)
        receipt_tag = _content_tag(
            self._receipt_tag_payload(
                lifecycle=lifecycle,
                decision=decision,
                model_revision=model_revision,
                representation_revision=representation_revision,
                action_revision=action_revision,
                region_revision=region_revision,
                calibration_revision=calibration_revision,
                cell_revision=cell_revision,
                action=action,
                region=region,
                prediction_tag=prediction_tag,
                calibration_tag=calibration_tag,
                gates_tag=gates_tag,
                valid=base_valid,
            ),
            salt=_RECEIPT_TAG_SALT,
        )
        return WorldModelPredictBeforeOutcomeReceipt(
            lifecycle_id_words=lifecycle,
            decision_id_words=decision,
            model_revision_words=model_revision,
            representation_revision_words=representation_revision,
            action_revision_words=action_revision,
            region_revision_words=region_revision,
            calibration_revision_words=calibration_revision,
            cell_revision_words=cell_revision,
            action=action,
            region=region,
            member_mean_predictions=member_means,
            member_aleatoric_variances=member_variances,
            member_termination_probabilities=member_termination,
            gates=gates,
            prediction_content_tag=prediction_tag,
            calibration_content_tag=calibration_tag,
            gates_content_tag=gates_tag,
            receipt_integrity_tag=receipt_tag,
            planning_authority=jnp.asarray(False, dtype=jnp.bool_),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            valid=base_valid,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def issue_prediction(
        self,
        state: WorldModelRegionCalibrationState,
        *,
        lifecycle_id_words: Array,
        decision_id_words: Array,
        model_revision_words: Array,
        representation_revision_words: Array,
        action_revision_words: Array,
        region_revision_words: Array,
        action: Array,
        region: Array,
        member_mean_predictions: Array,
        member_aleatoric_variances: Array,
        member_termination_probabilities: Array,
    ) -> WorldModelPredictBeforeOutcomeReceipt:
        """Issue a pure content-bound receipt from the exact pre-outcome cell."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        cfg = self._config
        lifecycle = _require_array(
            lifecycle_id_words,
            name="lifecycle_id_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        decision = _require_array(
            decision_id_words,
            name="decision_id_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        model_revision = _require_array(
            model_revision_words,
            name="model_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        representation_revision = _require_array(
            representation_revision_words,
            name="representation_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        action_revision = _require_array(
            action_revision_words,
            name="action_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        region_revision = _require_array(
            region_revision_words,
            name="region_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        exact_action = _require_array(
            action,
            name="action",
            shape=(),
            dtype=jnp.int32,
        )
        exact_region = _require_array(
            region,
            name="region",
            shape=(),
            dtype=jnp.int32,
        )
        means = _require_array(
            member_mean_predictions,
            name="member_mean_predictions",
            shape=(cfg.ensemble_size, cfg.target_dim),
            dtype=jnp.float32,
        )
        variances = _require_array(
            member_aleatoric_variances,
            name="member_aleatoric_variances",
            shape=(cfg.ensemble_size, cfg.target_dim),
            dtype=jnp.float32,
        )
        termination = _require_array(
            member_termination_probabilities,
            name="member_termination_probabilities",
            shape=(cfg.ensemble_size,),
            dtype=jnp.float32,
        )
        return self._build_receipt(
            state,
            lifecycle=lifecycle,
            decision=decision,
            model_revision=model_revision,
            representation_revision=representation_revision,
            action_revision=action_revision,
            region_revision=region_revision,
            action=exact_action,
            region=exact_region,
            member_means=means,
            member_variances=variances,
            member_termination=termination,
        )

    def _receipt_valid_dynamic(
        self,
        state: WorldModelRegionCalibrationState,
        receipt: WorldModelPredictBeforeOutcomeReceipt,
    ) -> Array:
        expected = self._build_receipt(
            state,
            lifecycle=receipt.lifecycle_id_words,
            decision=receipt.decision_id_words,
            model_revision=receipt.model_revision_words,
            representation_revision=receipt.representation_revision_words,
            action_revision=receipt.action_revision_words,
            region_revision=receipt.region_revision_words,
            action=receipt.action,
            region=receipt.region,
            member_means=receipt.member_mean_predictions,
            member_variances=receipt.member_aleatoric_variances,
            member_termination=receipt.member_termination_probabilities,
        )
        return receipt.valid & expected.valid & _tree_equal(receipt, expected)

    @functools.partial(jax.jit, static_argnums=(0,))
    def receipt_valid(
        self,
        state: WorldModelRegionCalibrationState,
        receipt: WorldModelPredictBeforeOutcomeReceipt,
    ) -> Array:
        """Recompute receipt content; any settlement makes it globally stale."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        if not self._receipt_static_valid(receipt):
            return jnp.asarray(False, dtype=jnp.bool_)
        return self._receipt_valid_dynamic(state, receipt)

    def _outcome_static_valid(self, outcome: object) -> bool:
        if type(outcome) is not WorldModelCalibrationOutcome:
            return False
        checks = (
            (outcome.lifecycle_id_words, (2,), jnp.uint32),
            (outcome.decision_id_words, (2,), jnp.uint32),
            (outcome.action, (), jnp.int32),
            (outcome.region, (), jnp.int32),
            (outcome.next_state, (self._config.observation_dim,), jnp.float32),
            (outcome.reward, (), jnp.float32),
            (outcome.terminated, (), jnp.bool_),
            (outcome.truncated, (), jnp.bool_),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    def _zero_outcome_diagnostics(self) -> WorldModelCalibrationOutcomeDiagnostics:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return WorldModelCalibrationOutcomeDiagnostics(
            next_state_squared_error=zero,
            reward_squared_error=zero,
            next_state_epistemic_disagreement=zero,
            reward_epistemic_disagreement=zero,
            next_state_mean_aleatoric_variance=zero,
            reward_aleatoric_variance=zero,
            next_state_epistemic_error_ratio=zero,
            reward_epistemic_error_ratio=zero,
            absolute_standardized_residuals=jnp.zeros(
                (self._config.target_dim,),
                dtype=jnp.float32,
            ),
            gaussian_nll=jnp.zeros(
                (self._config.target_dim,),
                dtype=jnp.float32,
            ),
            nominal_interval_covered=jnp.zeros(
                (self._config.target_dim,),
                dtype=jnp.bool_,
            ),
            mean_termination_probability=zero,
            termination_target=false,
            termination_observed=false,
            termination_brier_error=zero,
        )

    def _realized_diagnostics(
        self,
        receipt: WorldModelPredictBeforeOutcomeReceipt,
        outcome: WorldModelCalibrationOutcome,
    ) -> WorldModelCalibrationOutcomeDiagnostics:
        cfg = self._config
        target = jnp.concatenate((outcome.next_state, outcome.reward[None]), axis=0)
        mean_prediction = jnp.mean(receipt.member_mean_predictions, axis=0)
        residual = target - mean_prediction
        squared_residual = jnp.square(residual)
        epistemic = jnp.var(receipt.member_mean_predictions, axis=0)
        aleatoric = jnp.mean(receipt.member_aleatoric_variances, axis=0)
        safe_variance = jnp.maximum(aleatoric, cfg.variance_floor)
        absolute_standardized = jnp.abs(residual) / jnp.sqrt(safe_variance)
        gaussian_nll = 0.5 * (
            jnp.log(jnp.asarray(2.0 * math.pi, dtype=jnp.float32) * safe_variance)
            + squared_residual / safe_variance
        )
        nominal_covered = (
            absolute_standardized <= cfg.nominal_standardized_residual_limit
        )
        next_error = jnp.mean(squared_residual[: cfg.observation_dim])
        reward_error = squared_residual[-1]
        next_epistemic = jnp.mean(epistemic[: cfg.observation_dim])
        reward_epistemic = epistemic[-1]
        next_aleatoric = jnp.mean(aleatoric[: cfg.observation_dim])
        reward_aleatoric = aleatoric[-1]
        next_ratio = jnp.maximum(next_error - next_aleatoric, 0.0) / jnp.maximum(
            next_epistemic,
            cfg.disagreement_floor,
        )
        reward_ratio = jnp.maximum(reward_error - reward_aleatoric, 0.0) / jnp.maximum(
            reward_epistemic,
            cfg.disagreement_floor,
        )
        termination_probability = jnp.mean(receipt.member_termination_probabilities)
        termination_observed = ~outcome.truncated
        termination_target = outcome.terminated & termination_observed
        termination_brier = jnp.where(
            termination_observed,
            jnp.square(
                termination_probability - termination_target.astype(jnp.float32)
            ),
            0.0,
        )
        return WorldModelCalibrationOutcomeDiagnostics(
            next_state_squared_error=next_error,
            reward_squared_error=reward_error,
            next_state_epistemic_disagreement=next_epistemic,
            reward_epistemic_disagreement=reward_epistemic,
            next_state_mean_aleatoric_variance=next_aleatoric,
            reward_aleatoric_variance=reward_aleatoric,
            next_state_epistemic_error_ratio=next_ratio,
            reward_epistemic_error_ratio=reward_ratio,
            absolute_standardized_residuals=absolute_standardized,
            gaussian_nll=gaussian_nll,
            nominal_interval_covered=nominal_covered,
            mean_termination_probability=termination_probability,
            termination_target=termination_target,
            termination_observed=termination_observed,
            termination_brier_error=termination_brier,
        )

    def _candidate_state(
        self,
        state: WorldModelRegionCalibrationState,
        receipt: WorldModelPredictBeforeOutcomeReceipt,
        realized: WorldModelCalibrationOutcomeDiagnostics,
        next_accepted_words: Array,
        next_cell_words: Array,
    ) -> WorldModelRegionCalibrationState:
        cfg = self._config
        region = jnp.clip(receipt.region, 0, cfg.n_regions - 1)
        action = jnp.clip(receipt.action, 0, cfg.n_actions - 1)
        slot = state.cell_write_indices[region, action]
        index = (region, action, slot)
        record_valid = state.record_valid.at[index].set(True)
        termination_observed = state.termination_observed.at[index].set(
            realized.termination_observed
        )
        terminal_targets = state.terminal_targets.at[index].set(
            realized.termination_target
        )
        cell_sizes = state.cell_sizes.at[region, action].set(
            jnp.minimum(
                state.cell_sizes[region, action] + jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(cfg.capacity_per_cell, dtype=jnp.int32),
            )
        )
        cell_write_indices = state.cell_write_indices.at[region, action].set(
            (slot + jnp.asarray(1, dtype=jnp.int32)) % cfg.capacity_per_cell
        )
        cell_count_words = state.cell_count_words.at[region, action].set(next_cell_words)
        termination_support_counts = jnp.sum(
            termination_observed.astype(jnp.int32),
            axis=2,
        )
        terminal_support_counts = jnp.sum(
            (termination_observed & terminal_targets).astype(jnp.int32),
            axis=2,
        )
        continuing_support_counts = jnp.sum(
            (termination_observed & ~terminal_targets).astype(jnp.int32),
            axis=2,
        )
        return WorldModelRegionCalibrationState(
            lifecycle_id_words=state.lifecycle_id_words,
            record_valid=record_valid,
            termination_observed=termination_observed,
            terminal_targets=terminal_targets,
            next_state_squared_errors=state.next_state_squared_errors.at[index].set(
                realized.next_state_squared_error
            ),
            reward_squared_errors=state.reward_squared_errors.at[index].set(
                realized.reward_squared_error
            ),
            next_state_epistemic_disagreements=(
                state.next_state_epistemic_disagreements.at[index].set(
                    realized.next_state_epistemic_disagreement
                )
            ),
            reward_epistemic_disagreements=(
                state.reward_epistemic_disagreements.at[index].set(
                    realized.reward_epistemic_disagreement
                )
            ),
            next_state_epistemic_error_ratios=(
                state.next_state_epistemic_error_ratios.at[index].set(
                    realized.next_state_epistemic_error_ratio
                )
            ),
            reward_epistemic_error_ratios=(
                state.reward_epistemic_error_ratios.at[index].set(
                    realized.reward_epistemic_error_ratio
                )
            ),
            mean_aleatoric_variances=state.mean_aleatoric_variances.at[index].set(
                jnp.mean(receipt.member_aleatoric_variances, axis=0)
            ),
            absolute_standardized_residuals=(
                state.absolute_standardized_residuals.at[index].set(
                    realized.absolute_standardized_residuals
                )
            ),
            gaussian_nll=state.gaussian_nll.at[index].set(realized.gaussian_nll),
            nominal_interval_covered=state.nominal_interval_covered.at[index].set(
                realized.nominal_interval_covered
            ),
            mean_termination_probabilities=(
                state.mean_termination_probabilities.at[index].set(
                    realized.mean_termination_probability
                )
            ),
            termination_brier_errors=state.termination_brier_errors.at[index].set(
                realized.termination_brier_error
            ),
            cell_sizes=cell_sizes,
            cell_write_indices=cell_write_indices,
            termination_support_counts=termination_support_counts,
            terminal_support_counts=terminal_support_counts,
            continuing_support_counts=continuing_support_counts,
            cell_count_words=cell_count_words,
            accepted_count_words=next_accepted_words,
            last_decision_id_words=receipt.decision_id_words,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def settle(
        self,
        state: WorldModelRegionCalibrationState,
        receipt: WorldModelPredictBeforeOutcomeReceipt,
        outcome: WorldModelCalibrationOutcome,
    ) -> WorldModelCalibrationSettlementResult:
        """Settle one exact receipt or return the complete owner state unchanged."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        if not self._receipt_static_valid(receipt):
            raise TypeError("receipt has the wrong static contract")
        if not self._outcome_static_valid(outcome):
            raise TypeError("outcome has the wrong static contract")
        cfg = self._config
        state_valid = self._state_valid(state)
        receipt_valid = self._receipt_valid_dynamic(state, receipt)
        lifecycle_matches = (
            jnp.array_equal(outcome.lifecycle_id_words, state.lifecycle_id_words)
            & jnp.array_equal(outcome.lifecycle_id_words, receipt.lifecycle_id_words)
        )
        decision_matches = jnp.array_equal(
            outcome.decision_id_words,
            receipt.decision_id_words,
        )
        cell_identity_matches = (
            (outcome.action == receipt.action)
            & (outcome.region == receipt.region)
            & (outcome.action >= 0)
            & (outcome.action < cfg.n_actions)
            & (outcome.region >= 0)
            & (outcome.region < cfg.n_regions)
        )
        decision_fresh = _words_less(
            state.last_decision_id_words,
            receipt.decision_id_words,
        )
        outcome_finite = (
            jnp.all(jnp.isfinite(outcome.next_state))
            & jnp.isfinite(outcome.reward)
            & jnp.all(jnp.abs(outcome.next_state) <= cfg.max_outcome_magnitude)
            & (jnp.abs(outcome.reward) <= cfg.max_outcome_magnitude)
        )
        boundary_semantics_valid = ~(outcome.terminated & outcome.truncated)
        event_capacity = _words_leq_limit(
            state.accepted_count_words,
            cfg.max_events - 1,
        )
        next_accepted_words, accepted_increment_valid = _checked_words_increment(
            state.accepted_count_words
        )
        safe_region = jnp.clip(receipt.region, 0, cfg.n_regions - 1)
        safe_action = jnp.clip(receipt.action, 0, cfg.n_actions - 1)
        pre_cell_words = state.cell_count_words[safe_region, safe_action]
        next_cell_words, cell_increment_valid = _checked_words_increment(pre_cell_words)
        realized = self._realized_diagnostics(receipt, outcome)
        realized_finite = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in jax.tree.leaves(realized):
            array = jnp.asarray(leaf)
            if jnp.issubdtype(array.dtype, jnp.inexact):
                realized_finite = realized_finite & jnp.all(jnp.isfinite(array))
        preflight = (
            state_valid
            & receipt_valid
            & lifecycle_matches
            & decision_matches
            & cell_identity_matches
            & decision_fresh
            & outcome_finite
            & boundary_semantics_valid
            & event_capacity
            & accepted_increment_valid
            & cell_increment_valid
            & realized_finite
        )
        candidate = self._candidate_state(
            state,
            receipt,
            realized,
            next_accepted_words,
            next_cell_words,
        )
        candidate_state_valid = self._state_valid(candidate)
        applied = preflight & candidate_state_valid
        next_state = cast(
            WorldModelRegionCalibrationState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, operand=None),
        )
        emitted_outcome = cast(
            WorldModelCalibrationOutcomeDiagnostics,
            jax.lax.cond(
                applied,
                lambda _: realized,
                lambda _: self._zero_outcome_diagnostics(),
                operand=None,
            ),
        )
        post_cell_words = next_state.cell_count_words[safe_region, safe_action]
        return WorldModelCalibrationSettlementResult(
            state=next_state,
            receipt=receipt,
            gates=receipt.gates,
            outcome=emitted_outcome,
            transaction=WorldModelCalibrationTransactionDiagnostics(
                state_static_contract_valid=jnp.asarray(True, dtype=jnp.bool_),
                receipt_static_contract_valid=jnp.asarray(True, dtype=jnp.bool_),
                outcome_static_contract_valid=jnp.asarray(True, dtype=jnp.bool_),
                state_valid=state_valid,
                receipt_valid=receipt_valid,
                lifecycle_matches=lifecycle_matches,
                decision_matches=decision_matches,
                cell_identity_matches=cell_identity_matches,
                decision_fresh=decision_fresh,
                outcome_finite=outcome_finite,
                boundary_semantics_valid=boundary_semantics_valid,
                event_capacity_available=event_capacity,
                candidate_state_valid=candidate_state_valid,
                applied=applied,
                rejected=~applied,
                pre_accepted_count_words=state.accepted_count_words,
                post_accepted_count_words=next_state.accepted_count_words,
                pre_cell_count_words=pre_cell_words,
                post_cell_count_words=post_cell_words,
            ),
        )

    @property
    def resource_budget(self) -> WorldModelRegionCalibrationResourceBudget:
        """Return exact persistent/receipt logical bytes and fixed work ceilings."""

        template = self._empty_state(jnp.asarray((0, 1), dtype=jnp.uint32))
        cfg = self._config
        zero_means = jnp.zeros(
            (cfg.ensemble_size, cfg.target_dim),
            dtype=jnp.float32,
        )
        receipt = self._build_receipt(
            template,
            lifecycle=template.lifecycle_id_words,
            decision=jnp.asarray((0, 1), dtype=jnp.uint32),
            model_revision=jnp.asarray((0, 1), dtype=jnp.uint32),
            representation_revision=jnp.asarray((0, 1), dtype=jnp.uint32),
            action_revision=jnp.asarray((0, 1), dtype=jnp.uint32),
            region_revision=jnp.asarray((0, 1), dtype=jnp.uint32),
            action=jnp.asarray(0, dtype=jnp.int32),
            region=jnp.asarray(0, dtype=jnp.int32),
            member_means=zero_means,
            member_variances=jnp.full_like(zero_means, cfg.variance_floor),
            member_termination=jnp.zeros((cfg.ensemble_size,), dtype=jnp.float32),
        )
        state_scalars, state_bytes = _logical_tree_size(template)
        receipt_scalars, receipt_bytes = _logical_tree_size(receipt)
        return WorldModelRegionCalibrationResourceBudget(
            persistent_bytes_scope="all persistent JAX-array leaves",
            receipt_bytes_scope="one immutable predict-before-outcome receipt",
            temporary_bytes_scope=(
                "source-level fixed cell sorts and one candidate; not a measured device peak"
            ),
            region_action_cells=cfg.n_regions * cfg.n_actions,
            records_per_cell=cfg.capacity_per_cell,
            retained_record_capacity=(
                cfg.n_regions * cfg.n_actions * cfg.capacity_per_cell
            ),
            target_dim=cfg.target_dim,
            ensemble_size=cfg.ensemble_size,
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            receipt_scalars=receipt_scalars,
            receipt_bytes=receipt_bytes,
            max_settlements=cfg.max_events,
            max_model_updates_per_settlement=0,
            max_planner_updates_per_settlement=0,
            model_state_owned=0,
            representation_state_owned=0,
            planning_authority=0,
            safety_authority=0,
            scientific_promotion_allowed=False,
        )


def measure_world_model_region_calibration_state_nbytes(
    state: WorldModelRegionCalibrationState,
) -> int:
    """Measure every persistent JAX-array leaf in one owner state."""

    if type(state) is not WorldModelRegionCalibrationState:
        raise TypeError("state must be an exact WorldModelRegionCalibrationState")
    return _logical_tree_size(state)[1]


def save_world_model_region_calibration_checkpoint(
    owner: WorldModelRegionCalibration,
    state: WorldModelRegionCalibrationState,
    path: str | Path,
) -> None:
    """Persist one exact L0 calibration owner state and construction."""

    if type(owner) is not WorldModelRegionCalibration:
        raise TypeError("owner must be an exact WorldModelRegionCalibration")
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("refusing to save an invalid world-model calibration state")
    config = owner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA,
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": owner.resource_budget.to_config(),
            "evidence_level": WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL,
            "outcome_status": WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "model_state_included": False,
            "planner_state_included": False,
            "planning_authority": False,
            "safety_authority": False,
        },
    )


def load_world_model_region_calibration_checkpoint(
    path: str | Path,
) -> tuple[WorldModelRegionCalibration, WorldModelRegionCalibrationState]:
    """Strictly restore the sole current world-model calibration v1 schema."""

    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "owner_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "outcome_status",
        "scientific_promotion_allowed",
        "model_state_included",
        "planner_state_included",
        "planning_authority",
        "safety_authority",
    }
    if set(metadata) != expected_fields:
        raise ValueError("world-model calibration checkpoint metadata fields are not exact")
    if metadata.get("schema") != WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a world-model region calibration v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("world-model calibration checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("world-model calibration checkpoint config digest does not match")
    owner = WorldModelRegionCalibration.from_config(config)
    if metadata.get("resource_budget") != owner.resource_budget.to_config():
        raise ValueError("world-model calibration checkpoint resource budget does not match")
    if metadata.get("evidence_level") != WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL:
        raise ValueError("world-model calibration checkpoint must remain L0")
    if metadata.get("outcome_status") != WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS:
        raise ValueError("world-model calibration checkpoint must remain not_assessed")
    for name in (
        "scientific_promotion_allowed",
        "model_state_included",
        "planner_state_included",
        "planning_authority",
        "safety_authority",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"world-model calibration checkpoint {name} must be false")
    template = owner._empty_state(jnp.asarray((0, 1), dtype=jnp.uint32))
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("world-model calibration checkpoint metadata changed between reads")
    state = cast(WorldModelRegionCalibrationState, restored)
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("world-model calibration checkpoint restored an invalid state")
    if measure_world_model_region_calibration_state_nbytes(state) != (
        owner.resource_budget.persistent_state_bytes
    ):
        raise ValueError("world-model calibration checkpoint restored a wrong-size state")
    return owner, state


__all__ = [
    "WORLD_MODEL_REGION_CALIBRATION_CHECKPOINT_SCHEMA",
    "WORLD_MODEL_REGION_CALIBRATION_CONFIG_SCHEMA",
    "WORLD_MODEL_REGION_CALIBRATION_EVIDENCE_LEVEL",
    "WORLD_MODEL_REGION_CALIBRATION_OUTCOME_STATUS",
    "WORLD_MODEL_REGION_CALIBRATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "WorldModelAleatoricCalibrationGate",
    "WorldModelCalibrationOutcome",
    "WorldModelCalibrationOutcomeDiagnostics",
    "WorldModelCalibrationSettlementResult",
    "WorldModelCalibrationTransactionDiagnostics",
    "WorldModelEpistemicCalibrationGate",
    "WorldModelNextStateErrorGate",
    "WorldModelPlanningCalibrationGates",
    "WorldModelPredictBeforeOutcomeReceipt",
    "WorldModelRegionCalibration",
    "WorldModelRegionCalibrationConfig",
    "WorldModelRegionCalibrationResourceBudget",
    "WorldModelRegionCalibrationState",
    "WorldModelRewardErrorGate",
    "WorldModelTerminationCalibrationGate",
    "load_world_model_region_calibration_checkpoint",
    "measure_world_model_region_calibration_state_nbytes",
    "save_world_model_region_calibration_checkpoint",
]
