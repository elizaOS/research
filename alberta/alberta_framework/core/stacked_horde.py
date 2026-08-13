# mypy: disable-error-code="call-arg"
"""Stacked linear Horde: thousands of GVF demons as one batched TD update.

The existing Horde implementations (:mod:`~alberta_framework.core.horde`,
:mod:`~alberta_framework.core.independent_demon_horde`) unroll a Python loop
over demons inside ``jit``, so program size — and compile time and memory —
grow linearly with the demon count and throughput collapses well before the
"thousands of demons" regime the Horde architecture calls for (Sutton et al.
2011).  This module takes the other route for the linear case: every demon's
parameters live in one stacked array ``(n_demons, feature_dim)`` and the
entire Horde updates with a handful of batched array operations per step — no
per-demon loop exists anywhere, so program size is *constant* in the demon
count and the demon axis scales like any other array axis (and shards across
devices the same way).

Each demon d learns a linear GVF prediction ``v_d(x) = w_d @ x`` about the
shared behavior stream with its own question functions:

- cumulant ``c_d`` (selected from the observation/target channels or passed
  directly),
- pseudo-termination ``gamma_d``,
- trace decay ``lamda_d``,
- optional per-decision importance ratio ``rho`` for off-policy questions
  (per-decision IS with the ratio composed into the trace,
  ``z_d = rho * (gamma_d * lamda_d * z_d + x)``, matching the repo's
  post-fix off-policy convention).

The update is the textbook TD(lambda) per demon over shared features::

    delta_d = c_d + gamma_d * (w_d @ x') - (w_d @ x)
    z_d     = rho * (gamma_d * lamda_d * z_d + x)
    w_d    += alpha * delta_d * z_d

vectorized across all demons at once.  A ``NaN`` cumulant marks a demon
inactive for that step (its trace still decays; its weights freeze), matching
the NaN-masking convention of the loop-based hordes.
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, UInt

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)

STACKED_HORDE_CONFIG_SCHEMA = "alberta.stacked-horde-config.v2"
STACKED_HORDE_STATE_SCHEMA = "alberta.stacked-horde-state.v2"
STACKED_HORDE_RESOURCE_SCHEMA = "alberta.stacked-horde-resource-budget.v2"
STACKED_HORDE_LIFETIME_COUNTER_NBYTES = 12
STACKED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1

__all__ = [
    "STACKED_HORDE_CONFIG_SCHEMA",
    "STACKED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES",
    "STACKED_HORDE_LIFETIME_COUNTER_NBYTES",
    "STACKED_HORDE_RESOURCE_SCHEMA",
    "STACKED_HORDE_STATE_SCHEMA",
    "StackedHordeConfig",
    "StackedHordeResourceBudget",
    "StackedHordeState",
    "StackedHordeUpdateResult",
    "StackedLinearHorde",
    "measure_stacked_horde_state_nbytes",
    "migrate_legacy_stacked_horde_config",
    "migrate_legacy_stacked_horde_state",
    "nexting_spec",
    "run_stacked_horde_scan",
    "stacked_horde_lifetime_counter_nbytes",
    "stacked_horde_state_nbytes_formula",
]


def _tree_arrays_finite(tree: Any) -> Bool[Array, ""]:
    """Return whether every floating or complex JAX leaf is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _require_exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Copy a host mapping after rejecting missing and unknown fields."""

    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"{label} field manifest is not exact; missing={missing}, extra={extra}"
        )
    return fields


def _host_state_mapping(state: Any, *, label: str) -> dict[str, Any]:
    """Return a shallow host mapping for an explicit migration."""

    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    raise TypeError(f"legacy {label} must be a mapping or dataclass instance")


def _require_float32_array(value: Any, shape: tuple[int, ...], *, label: str) -> None:
    """Validate one public array boundary without silently narrowing it."""

    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != jnp.dtype(jnp.float32):
        raise TypeError(f"{label} must have dtype float32")


@dataclass(frozen=True)
class StackedHordeConfig:
    """Static configuration for :class:`StackedLinearHorde`.

    Attributes:
        n_demons: Number of GVF demons.
        feature_dim: Shared feature dimension.
        gammas: Per-demon pseudo-termination discounts, length ``n_demons``.
        lamdas: Per-demon trace decays, length ``n_demons``.
        cumulant_indices: Per-demon index into the cumulant-source vector
            handed to :meth:`StackedLinearHorde.update`.
        step_size: Shared TD step-size alpha.
    """

    n_demons: int
    feature_dim: int
    gammas: tuple[float, ...]
    lamdas: tuple[float, ...]
    cumulant_indices: tuple[int, ...]
    step_size: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if type(self.n_demons) is not int or self.n_demons < 1:
            raise ValueError("n_demons must be a positive exact integer")
        if type(self.feature_dim) is not int or self.feature_dim < 1:
            raise ValueError("feature_dim must be a positive exact integer")
        for name, seq in (
            ("gammas", self.gammas),
            ("lamdas", self.lamdas),
            ("cumulant_indices", self.cumulant_indices),
        ):
            if not isinstance(seq, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(seq) != self.n_demons:
                raise ValueError(f"{name} must have length n_demons={self.n_demons}")
        if any(
            isinstance(g, bool)
            or not isinstance(g, Real)
            or not math.isfinite(float(g))
            or not 0.0 <= float(g) <= 1.0
            for g in self.gammas
        ):
            raise ValueError("every gamma must be in [0, 1]")
        if any(
            isinstance(lamda, bool)
            or not isinstance(lamda, Real)
            or not math.isfinite(float(lamda))
            or not 0.0 <= float(lamda) <= 1.0
            for lamda in self.lamdas
        ):
            raise ValueError("every lamda must be in [0, 1]")
        if any(type(index) is not int or index < 0 for index in self.cumulant_indices):
            raise ValueError("every cumulant index must be a non-negative exact integer")
        if (
            isinstance(self.step_size, bool)
            or not isinstance(self.step_size, Real)
            or not math.isfinite(float(self.step_size))
            or self.step_size <= 0.0
        ):
            raise ValueError("step_size must be positive and finite")

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "schema": STACKED_HORDE_CONFIG_SCHEMA,
            "state_schema": STACKED_HORDE_STATE_SCHEMA,
            "type": "StackedHordeConfig",
            "n_demons": self.n_demons,
            "feature_dim": self.feature_dim,
            "gammas": [float(value) for value in self.gammas],
            "lamdas": [float(value) for value in self.lamdas],
            "cumulant_indices": [int(value) for value in self.cumulant_indices],
            "step_size": float(self.step_size),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedHordeConfig":
        """Reconstruct a config from :meth:`to_config` output."""
        expected = {
            "schema",
            "state_schema",
            "type",
            "n_demons",
            "feature_dim",
            "gammas",
            "lamdas",
            "cumulant_indices",
            "step_size",
        }
        config = _require_exact_manifest(config, expected, label="stacked Horde config")
        if config.pop("schema") != STACKED_HORDE_CONFIG_SCHEMA:
            raise ValueError("unsupported stacked Horde config schema")
        if config.pop("state_schema") != STACKED_HORDE_STATE_SCHEMA:
            raise ValueError("unsupported stacked Horde state schema")
        if config.pop("type") != "StackedHordeConfig":
            raise ValueError("stacked Horde config type must be 'StackedHordeConfig'")
        for key in ("gammas", "lamdas", "cumulant_indices"):
            if not isinstance(config[key], list):
                raise TypeError(f"serialized {key} must be a list")
            config[key] = tuple(config[key])
        return cls(**config)


def migrate_legacy_stacked_horde_config(
    legacy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate an exact unversioned config to the v2 serialized schema."""

    expected = {
        "type",
        "n_demons",
        "feature_dim",
        "gammas",
        "lamdas",
        "cumulant_indices",
        "step_size",
    }
    fields = _require_exact_manifest(
        legacy_config,
        expected,
        label="legacy stacked Horde config",
    )
    if fields.pop("type") != "StackedHordeConfig":
        raise ValueError("legacy stacked Horde config type is invalid")
    for key in ("gammas", "lamdas", "cumulant_indices"):
        if not isinstance(fields[key], list):
            raise TypeError(f"legacy serialized {key} must be a list")
        fields[key] = tuple(fields[key])
    return StackedHordeConfig(**fields).to_config()


def nexting_spec(
    feature_dim: int,
    cumulant_indices: tuple[int, ...],
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99),
    lamda: float = 0.7,
    step_size: float = 0.05,
) -> StackedHordeConfig:
    """Build a nexting-style config: every cumulant at every timescale.

    Multi-timescale prediction of many sensor channels at once is the
    canonical Horde workload (Modayil, White & Sutton 2014, "nexting").  The
    returned config has ``len(cumulant_indices) * len(gammas)`` demons,
    ordered cumulant-major.

    Args:
        feature_dim: Shared feature dimension.
        cumulant_indices: Which cumulant channels to predict.
        gammas: Prediction timescales for each channel.
        lamda: Shared trace decay.
        step_size: Shared TD step-size.
    """
    idxs: list[int] = []
    gs: list[float] = []
    for c in cumulant_indices:
        for g in gammas:
            idxs.append(c)
            gs.append(g)
    n = len(idxs)
    return StackedHordeConfig(
        n_demons=n,
        feature_dim=feature_dim,
        gammas=tuple(gs),
        lamdas=(lamda,) * n,
        cumulant_indices=tuple(idxs),
        step_size=step_size,
    )


@chex.dataclass(frozen=True)
class StackedHordeState:
    """State for :class:`StackedLinearHorde`.

    Attributes:
        weights: Stacked demon weights, shape ``(n_demons, feature_dim)``.
        traces: Stacked eligibility traces, same shape.
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian ``[high, low]`` uint32 event identity.
    """

    weights: Float[Array, "n_demons feature_dim"]
    traces: Float[Array, "n_demons feature_dim"]
    step_count: Array
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StackedHordeResourceBudget:
    """Exact persistent-state accounting for one stacked Horde."""

    n_demons: int
    feature_dim: int
    parameter_nbytes: int
    trace_nbytes: int
    lifetime_counter_nbytes: int
    state_nbytes: int

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the exact resource contract."""

        return {
            "schema": STACKED_HORDE_RESOURCE_SCHEMA,
            "n_demons": self.n_demons,
            "feature_dim": self.feature_dim,
            "parameter_nbytes": self.parameter_nbytes,
            "trace_nbytes": self.trace_nbytes,
            "lifetime_counter_nbytes": self.lifetime_counter_nbytes,
            "state_nbytes": self.state_nbytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StackedHordeResourceBudget":
        """Load a resource contract while rejecting schema drift."""

        expected = {
            "schema",
            "n_demons",
            "feature_dim",
            "parameter_nbytes",
            "trace_nbytes",
            "lifetime_counter_nbytes",
            "state_nbytes",
        }
        fields = _require_exact_manifest(
            payload,
            expected,
            label="stacked Horde resource budget",
        )
        if fields.pop("schema") != STACKED_HORDE_RESOURCE_SCHEMA:
            raise ValueError("unsupported stacked Horde resource schema")
        if any(type(value) is not int or value < 0 for value in fields.values()):
            raise ValueError("stacked Horde resource values must be non-negative integers")
        budget = cls(**fields)
        expected_parameter_nbytes = 4 * budget.n_demons * budget.feature_dim
        if budget.n_demons < 1 or budget.feature_dim < 1:
            raise ValueError("stacked Horde resource dimensions must be positive")
        if budget.parameter_nbytes != expected_parameter_nbytes:
            raise ValueError("stacked Horde parameter byte accounting is inconsistent")
        if budget.trace_nbytes != expected_parameter_nbytes:
            raise ValueError("stacked Horde trace byte accounting is inconsistent")
        if budget.lifetime_counter_nbytes != STACKED_HORDE_LIFETIME_COUNTER_NBYTES:
            raise ValueError("stacked Horde lifetime byte accounting is inconsistent")
        if budget.state_nbytes != (
            budget.parameter_nbytes
            + budget.trace_nbytes
            + budget.lifetime_counter_nbytes
        ):
            raise ValueError("stacked Horde total byte accounting is inconsistent")
        return budget


@chex.dataclass(frozen=True)
class StackedHordeUpdateResult:
    """Result of one stacked update.

    Attributes:
        state: Updated horde state.
        predictions: Pre-update predictions ``v_d(x)``, shape ``(n_demons,)``.
        td_errors: Per-demon TD errors (NaN where the demon was inactive).
        pre_step_words: Exact event identity before the attempted transaction.
        post_step_words: Exact event identity after commit or rollback.
        lifetime_counter_valid: Whether exact identity and telemetry agreed.
        lifetime_capacity_available: Whether the exact clock could advance.
        source_valid: Whether transition arrays and ratios were admissible.
        state_valid: Whether every persistent floating input-state array was finite.
        per_demon_candidate_valid: Whether every staged demon result was finite.
        candidate_valid: Whether the complete staged state was valid.
        update_applied: Whether every demon committed atomically.
    """

    state: StackedHordeState
    predictions: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    per_demon_candidate_valid: Bool[Array, " n_demons"]
    candidate_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class StackedLinearHorde:
    """Batched linear TD(lambda) over a stacked demon axis (module docstring)."""

    def __init__(self, config: StackedHordeConfig):
        """Initialize the horde around a static config."""
        self._config = config
        self._gammas = jnp.asarray(config.gammas, dtype=jnp.float32)
        self._lamdas = jnp.asarray(config.lamdas, dtype=jnp.float32)
        self._cumulant_idx = jnp.asarray(config.cumulant_indices, dtype=jnp.int32)

    @property
    def config(self) -> StackedHordeConfig:
        """The static configuration."""
        return self._config

    @property
    def resource_budget(self) -> StackedHordeResourceBudget:
        """Exact persistent-state resource contract for this Horde."""

        parameter_nbytes = 4 * self._config.n_demons * self._config.feature_dim
        return StackedHordeResourceBudget(
            n_demons=self._config.n_demons,
            feature_dim=self._config.feature_dim,
            parameter_nbytes=parameter_nbytes,
            trace_nbytes=parameter_nbytes,
            lifetime_counter_nbytes=STACKED_HORDE_LIFETIME_COUNTER_NBYTES,
            state_nbytes=2 * parameter_nbytes + STACKED_HORDE_LIFETIME_COUNTER_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this horde to a dictionary."""
        return {"type": "StackedLinearHorde", "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedLinearHorde":
        """Reconstruct a horde from :meth:`to_config` output."""
        fields = _require_exact_manifest(
            config,
            {"type", "config"},
            label="stacked linear Horde",
        )
        if fields["type"] != "StackedLinearHorde":
            raise ValueError("stacked linear Horde type is invalid")
        nested = fields["config"]
        if not isinstance(nested, Mapping):
            raise TypeError("stacked linear Horde config must be a mapping")
        return cls(StackedHordeConfig.from_config(dict(nested)))

    def init(self) -> StackedHordeState:
        """Initialize zero weights and traces."""
        cfg = self._config
        shape = (cfg.n_demons, cfg.feature_dim)
        return StackedHordeState(
            weights=jnp.zeros(shape, dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _validate_state_structure(
        self,
        state: StackedHordeState,
    ) -> Bool[Array, ""]:
        """Validate static state contracts and return exact-clock validity."""

        shape = (self._config.n_demons, self._config.feature_dim)
        _require_float32_array(state.weights, shape, label="stacked Horde weights")
        _require_float32_array(state.traces, shape, label="stacked Horde traces")
        return _lifetime_counter_valid(state.step_words, state.step_count)

    def state_valid(self, state: StackedHordeState) -> Bool[Array, ""]:
        """Return whether a state is finite and has an authentic exact clock."""

        return self._validate_state_structure(state) & _tree_arrays_finite(state)

    def predict(self, state: StackedHordeState, features: Array) -> Array:
        """All demons' predictions for one feature vector, shape ``(n_demons,)``."""
        self._validate_state_structure(state)
        _require_float32_array(
            features,
            (self._config.feature_dim,),
            label="features",
        )
        return state.weights @ features

    def update(
        self,
        state: StackedHordeState,
        features: Array,
        next_features: Array,
        cumulant_source: Array,
        rho: Array | float = 1.0,
    ) -> StackedHordeUpdateResult:
        """One batched TD(lambda) step for every demon at once.

        Args:
            state: Current horde state.
            features: Feature vector ``x_t``, shape ``(feature_dim,)``.
            next_features: Feature vector ``x_{t+1}``.
            cumulant_source: Vector the demons draw their cumulants from via
                ``config.cumulant_indices`` (e.g. the next observation, or a
                dedicated signal vector).  A ``NaN`` entry deactivates every
                demon reading it for this step: the demon's weights freeze
                and its trace only decays.
            rho: Per-decision importance ratio of the behavior action under
                each demon's target policy.  Scalar (shared) or shape
                ``(n_demons,)``.  On-policy questions use 1.0.

        Returns:
            :class:`StackedHordeUpdateResult` with per-demon predictions and
            TD errors (TD errors are NaN for inactive demons).
        """
        cfg = self._config
        lifetime_counter_valid = self._validate_state_structure(state)
        _require_float32_array(
            features,
            (cfg.feature_dim,),
            label="features",
        )
        _require_float32_array(
            next_features,
            (cfg.feature_dim,),
            label="next_features",
        )
        source_shape = getattr(cumulant_source, "shape", None)
        if source_shape is None or len(source_shape) != 1:
            raise ValueError("cumulant_source must be a rank-one array")
        if getattr(cumulant_source, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("cumulant_source must have dtype float32")
        required_source_dim = max(cfg.cumulant_indices) + 1
        if source_shape[0] < required_source_dim:
            raise ValueError(
                "cumulant_source does not contain every configured cumulant index"
            )

        if isinstance(rho, Real) and not isinstance(rho, bool):
            rho_array = jnp.asarray(float(rho), dtype=jnp.float32)
        else:
            rho_shape = getattr(rho, "shape", None)
            if rho_shape not in {(), (cfg.n_demons,)}:
                raise ValueError(
                    f"rho must be scalar or have shape ({cfg.n_demons},)"
                )
            if getattr(rho, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError("rho arrays must have dtype float32")
            rho_array = jnp.asarray(rho)

        cumulants = cumulant_source[self._cumulant_idx]  # (n_demons,)
        active = ~jnp.isnan(cumulants)
        safe_cumulants = jnp.where(active, cumulants, 0.0)
        rho_vec = jnp.broadcast_to(rho_array, (cfg.n_demons,))
        source_valid = (
            jnp.all(jnp.isfinite(features))
            & jnp.all(jnp.isfinite(next_features))
            & jnp.all(jnp.isfinite(cumulant_source) | jnp.isnan(cumulant_source))
            & jnp.all(jnp.isfinite(rho_vec) & (rho_vec >= 0.0))
        )
        state_valid = _tree_arrays_finite(state)
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )

        v = state.weights @ features  # (n_demons,)
        v_next = state.weights @ next_features
        td_errors = safe_cumulants + self._gammas * v_next - v

        # Per-decision IS with the ratio composed into the trace:
        # z_d = rho_d * (gamma_d * lamda_d * z_d + x).
        decay = (self._gammas * self._lamdas)[:, None]  # (n_demons, 1)
        accumulated = decay * state.traces + features[None, :]
        # Inactive demons: trace decays but the current gradient is withheld.
        decayed_only = decay * state.traces
        new_traces = rho_vec[:, None] * jnp.where(
            active[:, None],
            accumulated,
            decayed_only,
        )

        masked_delta = jnp.where(active, td_errors, 0.0)
        new_weights = state.weights + cfg.step_size * masked_delta[:, None] * new_traces

        candidate_state = StackedHordeState(
            weights=new_weights,
            traces=new_traces,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        reported_td_errors = jnp.where(active, td_errors, jnp.nan)
        per_demon_candidate_valid = (
            jnp.all(jnp.isfinite(new_weights), axis=1)
            & jnp.all(jnp.isfinite(new_traces), axis=1)
            & jnp.isfinite(v)
            & ((~active) | jnp.isfinite(reported_td_errors))
        )
        candidate_valid = (
            jnp.all(per_demon_candidate_valid)
            & _lifetime_counter_valid(
                candidate_state.step_words,
                candidate_state.step_count,
            )
            & _tree_arrays_finite(candidate_state)
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & state_valid
            & candidate_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        return StackedHordeUpdateResult(
            state=new_state,
            predictions=v,
            td_errors=reported_td_errors,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_valid=source_valid,
            state_valid=state_valid,
            per_demon_candidate_valid=per_demon_candidate_valid,
            candidate_valid=candidate_valid,
            update_applied=update_applied,
        )


def stacked_horde_state_nbytes_formula(n_demons: int, feature_dim: int) -> int:
    """Return exact v2 persistent bytes for one stacked state."""

    if type(n_demons) is not int or n_demons < 1:
        raise ValueError("n_demons must be a positive exact integer")
    if type(feature_dim) is not int or feature_dim < 1:
        raise ValueError("feature_dim must be a positive exact integer")
    return 8 * n_demons * feature_dim + STACKED_HORDE_LIFETIME_COUNTER_NBYTES


def measure_stacked_horde_state_nbytes(state: StackedHordeState) -> int:
    """Measure persistent JAX-array bytes in one concrete stacked state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def stacked_horde_lifetime_counter_nbytes() -> int:
    """Return bytes owned by telemetry plus exact event identity."""

    return STACKED_HORDE_LIFETIME_COUNTER_NBYTES


def migrate_legacy_stacked_horde_state(
    legacy_state: Any,
    *,
    config: StackedHordeConfig | None = None,
) -> StackedHordeState:
    """Migrate an exact unsaturated pre-v2 state without inventing history.

    A legacy ``step_count`` at int32 saturation cannot distinguish one event
    from arbitrarily many later events, so it is rejected fail closed.
    """

    fields = _host_state_mapping(legacy_state, label="stacked Horde state")
    fields = _require_exact_manifest(
        fields,
        {"weights", "traces", "step_count"},
        label="legacy stacked Horde state",
    )
    weights = jnp.asarray(fields["weights"])
    traces = jnp.asarray(fields["traces"])
    step_count = jnp.asarray(fields["step_count"])
    if weights.ndim != 2 or not weights.shape[0] or not weights.shape[1]:
        raise ValueError("legacy stacked Horde weights must be a non-empty matrix")
    if traces.shape != weights.shape:
        raise ValueError("legacy stacked Horde traces must match weights")
    if weights.dtype != jnp.dtype(jnp.float32):
        raise TypeError("legacy stacked Horde weights must have dtype float32")
    if traces.dtype != jnp.dtype(jnp.float32):
        raise TypeError("legacy stacked Horde traces must have dtype float32")
    if not bool(jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(traces))):
        raise ValueError("legacy stacked Horde arrays must be finite")
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy stacked Horde step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy stacked Horde step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy stacked Horde step_count is ambiguous")
    if config is not None and weights.shape != (config.n_demons, config.feature_dim):
        raise ValueError("legacy stacked Horde state shape does not match config")
    return StackedHordeState(
        weights=weights,
        traces=traces,
        step_count=step_count,
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )


def run_stacked_horde_scan(
    horde: StackedLinearHorde,
    state: StackedHordeState,
    features: Float[Array, "num_steps feature_dim"],
    cumulant_sources: Float[Array, "num_steps source_dim"],
    rhos: Float[Array, " num_steps"] | None = None,
) -> tuple[StackedHordeState, Float[Array, "num_steps n_demons"]]:
    """Scan the horde over transition arrays.

    ``features[t]`` and ``features[t+1]`` form each transition; the final
    row only provides a bootstrap target, so ``num_steps - 1`` updates run.

    Args:
        horde: The stacked horde.
        state: Initial state.
        features: Feature vectors per step.
        cumulant_sources: Cumulant-source vectors per transition (row ``t``
            is consumed by the ``t -> t+1`` update).
        rhos: Optional per-transition importance ratios (shared across
            demons); defaults to on-policy 1.0.

    Returns:
        ``(final_state, td_errors)`` with td_errors of shape
        ``(num_steps - 1, n_demons)``.
    """
    cfg = horde.config
    horde._validate_state_structure(state)
    feature_shape = getattr(features, "shape", None)
    if (
        feature_shape is None
        or len(feature_shape) != 2
        or feature_shape[1] != cfg.feature_dim
    ):
        raise ValueError(
            f"features must have shape (num_steps, {cfg.feature_dim})"
        )
    if getattr(features, "dtype", None) != jnp.dtype(jnp.float32):
        raise TypeError("features must have dtype float32")
    num_steps = features.shape[0]
    if num_steps < 2:
        raise ValueError("features must contain at least one transition")
    source_shape = getattr(cumulant_sources, "shape", None)
    required_source_dim = max(cfg.cumulant_indices) + 1
    if (
        source_shape is None
        or len(source_shape) != 2
        or source_shape[0] != num_steps
        or source_shape[1] < required_source_dim
    ):
        raise ValueError(
            "cumulant_sources must have shape (num_steps, source_dim) and "
            "contain every configured cumulant index"
        )
    if getattr(cumulant_sources, "dtype", None) != jnp.dtype(jnp.float32):
        raise TypeError("cumulant_sources must have dtype float32")
    if rhos is None:
        rhos = jnp.ones((num_steps,), dtype=jnp.float32)
    else:
        rho_shape = getattr(rhos, "shape", None)
        if rho_shape not in {(num_steps,), (num_steps, cfg.n_demons)}:
            raise ValueError(
                "rhos must have shape (num_steps,) or "
                f"(num_steps, {cfg.n_demons})"
            )
        if getattr(rhos, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("rhos must have dtype float32")

    def step_fn(
        carry: StackedHordeState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[StackedHordeState, Array]:
        x, x_next, c, rho = inputs
        result = horde.update(carry, x, x_next, c, rho)
        return result.state, result.td_errors

    final_state, td_errors = jax.lax.scan(
        step_fn,
        state,
        (features[:-1], features[1:], cumulant_sources[:-1], rhos[:-1]),
    )
    return final_state, td_errors
