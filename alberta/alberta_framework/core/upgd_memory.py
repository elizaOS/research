# mypy: disable-error-code="call-arg,name-defined"
"""Single Step 2 learner combining UPGD with fixed-budget prototype memory.

Two complementary mechanisms form one learner: target-structure UPGD
(:class:`~alberta_framework.core.upgd.UPGDLearner`) provides differentiable
plastic features, and a fixed-budget multi-prototype memory
(:class:`~alberta_framework.core.prototype_memory.PrototypeMemoryLearner`)
retains one-hot class views.  Both components update on every step.  Their
predictions are blended by one learned scalar logit plus causal
confidence/reliability signals, so the deployed object is one learner
rather than a route-selecting portfolio.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.prototype_memory import (
    PROTOTYPE_MEMORY_STATE_SCHEMA,
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    PrototypeMemoryState,
    measure_prototype_memory_state_nbytes,
    migrate_legacy_prototype_memory_state,
)
from alberta_framework.core.upgd import (
    UPGD_STATE_SCHEMA,
    UPGDLearner,
    UPGDState,
    measure_upgd_state_nbytes,
    migrate_legacy_upgd_state,
)

UPGDMemoryReadoutMode = Literal["linear_mse", "softmax_ce"]

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

UPGD_MEMORY_STATE_SCHEMA = "alberta.upgd-memory-state.v2"
UPGD_MEMORY_CONFIG_SCHEMA = "alberta.upgd-memory-config.v2"
UPGD_MEMORY_CHECKPOINT_SCHEMA = "alberta.upgd-memory-checkpoint.v2"
_LEGACY_UPGD_MEMORY_CHECKPOINT_SCHEMA = "alberta.upgd-memory-checkpoint.v1"
UPGD_MEMORY_OUTER_CLOCK_NBYTES = 12
UPGD_MEMORY_OUTER_CLOCK_DELTA_NBYTES = 8

__all__ = [
    "UPGD_MEMORY_CHECKPOINT_SCHEMA",
    "UPGD_MEMORY_CONFIG_SCHEMA",
    "UPGD_MEMORY_OUTER_CLOCK_DELTA_NBYTES",
    "UPGD_MEMORY_OUTER_CLOCK_NBYTES",
    "UPGD_MEMORY_STATE_SCHEMA",
    "UPGDMemoryConfig",
    "UPGDMemoryLearner",
    "UPGDMemoryLearningResult",
    "UPGDMemoryResourceBudget",
    "UPGDMemoryState",
    "UPGDMemoryUpdateResult",
    "load_upgd_memory_checkpoint",
    "measure_upgd_memory_state_nbytes",
    "migrate_legacy_upgd_memory_state",
    "run_upgd_memory_arrays",
    "save_upgd_memory_checkpoint",
]


@dataclass(frozen=True)
class UPGDMemoryConfig:
    """Configuration for :class:`UPGDMemoryLearner`.

    Args:
        feature_dim: Observation dimensionality.
        n_heads: Output dimensionality.  For classification this is the number
            of one-hot classes.
        hidden_sizes: UPGD hidden-layer widths.
        readout_mode: UPGD readout/loss mode.  ``"softmax_ce"`` is the intended
            mode when prototype memory is active.
        upgd_step_size: Base UPGD step-size.
        upgd_head_step_size_multiplier: Fixed multiplier for output-head
            weight and bias updates.
        upgd_head_bias_step_size_multiplier: Extra multiplier for output-head
            bias updates after ``upgd_head_step_size_multiplier``.
        upgd_head_loss_pressure_gate_ratio: Fast/slow loss ratio at which the
            output head receives an additional plasticity multiplier.
        upgd_head_loss_pressure_multiplier: Maximum additional output-head
            plasticity under loss pressure.
        upgd_head_loss_pressure_warmup_steps: Initial updates before
            loss-pressure head plasticity is enabled.
        upgd_head_repetition_multiplier: Maximum additional output-head
            plasticity under repeated-target pressure.
        upgd_head_repetition_decay: EMA decay for repeated-target detection.
        upgd_head_repetition_delta_threshold: Mean absolute target-vector
            change treated as a repeated target.
        upgd_head_repetition_pressure_threshold: Repetition EMA level below
            which repeated-target pressure is ignored.
        upgd_head_repetition_warmup_steps: Initial updates before
            repeated-target head plasticity is enabled.
        slots_per_class: Fixed prototype slots per class.
        memory_update_rate: EMA rate for matched prototypes.
        initial_novelty_threshold: Initial mean-squared distance threshold for
            allocating a fresh prototype.
        memory_bandwidth: Distance-to-logit bandwidth for prototype memory.
        initial_memory_logit: Learned base logit for memory-vs-UPGD blending.
        memory_logit_step_size: Online gradient step-size for the blend logit.
        confidence_logit_scale: Fixed coefficient for memory confidence minus
            UPGD confidence.
        reliability_logit_scale: Fixed coefficient for UPGD loss EMA minus
            memory loss EMA.
        reliability_decay: EMA decay for component losses and allocation rate.
        target_trace_blend_scale: Update-time blend toward the previous
            target vector under repeated-target pressure.  This is a causal
            temporal prior for prequential streams with persistent targets.
            Only ``update`` applies it; ordinary ``predict`` calls stay
            observation-based so held-out batch evaluation is not biased toward
            the last observed target.
        target_trace_pressure_threshold: Repetition EMA level below which the
            target-trace prior is ignored.
        novelty_adaptation_rate: Online log-threshold adaptation step-size.
        target_allocation_rate: Target prototype allocation frequency.  When
            allocation EMA is higher than this, the threshold rises; when lower,
            it falls.
        min_novelty_threshold: Lower threshold clamp.
        max_novelty_threshold: Upper threshold clamp.
    """

    feature_dim: int
    n_heads: int
    hidden_sizes: tuple[int, ...] = (64,)
    readout_mode: UPGDMemoryReadoutMode = "softmax_ce"
    upgd_step_size: float = 0.03
    upgd_head_step_size_multiplier: float = 1.0
    upgd_head_bias_step_size_multiplier: float = 1.0
    upgd_head_loss_pressure_gate_ratio: float = 0.0
    upgd_head_loss_pressure_multiplier: float = 0.0
    upgd_head_loss_pressure_warmup_steps: int = 0
    upgd_head_repetition_multiplier: float = 0.0
    upgd_head_repetition_decay: float = 0.9
    upgd_head_repetition_delta_threshold: float = 0.05
    upgd_head_repetition_pressure_threshold: float = 0.0
    upgd_head_repetition_warmup_steps: int = 0
    slots_per_class: int = 20
    memory_update_rate: float = 0.3
    initial_novelty_threshold: float = 0.08
    memory_bandwidth: float = 0.01
    initial_memory_logit: float = 0.0
    memory_logit_step_size: float = 0.25
    confidence_logit_scale: float = 2.0
    reliability_logit_scale: float = 8.0
    reliability_decay: float = 0.98
    target_trace_blend_scale: float = 0.8
    target_trace_pressure_threshold: float = 0.5
    novelty_adaptation_rate: float = 0.02
    target_allocation_rate: float = 0.18
    min_novelty_threshold: float = 1e-4
    max_novelty_threshold: float = 1.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    def to_config(self) -> dict[str, object]:
        """Serialize to a plain config dictionary."""
        payload = self.to_dict()
        payload["type"] = "UPGDMemoryConfig"
        payload["config_schema"] = UPGD_MEMORY_CONFIG_SCHEMA
        payload["state_schema"] = UPGD_MEMORY_STATE_SCHEMA
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UPGDMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        expected = set(cls(feature_dim=1, n_heads=2).to_config())
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                "UPGD-memory config field manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if payload.pop("type") != "UPGDMemoryConfig":
            raise ValueError("UPGD-memory config type is unsupported")
        if payload.pop("config_schema") != UPGD_MEMORY_CONFIG_SCHEMA:
            raise ValueError("UPGD-memory config schema is unsupported")
        if payload.pop("state_schema") != UPGD_MEMORY_STATE_SCHEMA:
            raise ValueError("UPGD-memory state schema is unsupported")
        if "hidden_sizes" in payload:
            payload["hidden_sizes"] = tuple(payload["hidden_sizes"])
        return cls(**payload)


@chex.dataclass(frozen=True)
class UPGDMemoryState:
    """State for :class:`UPGDMemoryLearner`."""

    upgd_state: UPGDState
    memory_state: PrototypeMemoryState
    memory_logit: Array
    novelty_log_threshold: Array
    upgd_loss_ema: Array
    memory_loss_ema: Array
    blended_loss_ema: Array
    allocation_ema: Array
    step_count: Array
    step_words: Array


@chex.dataclass(frozen=True)
class UPGDMemoryUpdateResult:
    """Result of one UPGD-memory update."""

    state: UPGDMemoryState
    predictions: Float[Array, " n_heads"]
    errors: Float[Array, " n_heads"]
    metrics: Float[Array, " 10"]
    pre_step_words: Array
    post_step_words: Array
    state_valid: Array
    candidate_state_valid: Array
    lifetime_capacity_available: Array
    upgd_update_applied: Array
    memory_update_applied: Array
    blend_update_valid: Array
    update_applied: Array
    update_rejected: Array


@chex.dataclass(frozen=True)
class UPGDMemoryLearningResult:
    """Result from :func:`run_upgd_memory_arrays`."""

    state: UPGDMemoryState
    predictions: Float[Array, "steps n_heads"]
    metrics: Float[Array, "steps 10"]


@dataclass(frozen=True)
class UPGDMemoryResourceBudget:
    """Exact persistent-resource contract for the composite learner."""

    state_nbytes: int
    outer_clock_nbytes: int
    outer_clock_delta_nbytes: int
    upgd_state_nbytes: int
    prototype_memory_state_nbytes: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible resource declaration."""
        return {
            "type": "UPGDMemoryResourceBudget",
            "state_nbytes": self.state_nbytes,
            "outer_clock_nbytes": self.outer_clock_nbytes,
            "outer_clock_delta_nbytes": self.outer_clock_delta_nbytes,
            "upgd_state_nbytes": self.upgd_state_nbytes,
            "prototype_memory_state_nbytes": self.prototype_memory_state_nbytes,
        }


def _validate_config(config: UPGDMemoryConfig) -> None:
    for name, value in (
        ("feature_dim", config.feature_dim),
        ("n_heads", config.n_heads),
        ("upgd_head_loss_pressure_warmup_steps", config.upgd_head_loss_pressure_warmup_steps),
        ("upgd_head_repetition_warmup_steps", config.upgd_head_repetition_warmup_steps),
        ("slots_per_class", config.slots_per_class),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if not isinstance(config.hidden_sizes, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int)
        for size in config.hidden_sizes
    ):
        raise ValueError("hidden_sizes must be a tuple of integers")
    non_real_names = {"feature_dim", "n_heads", "hidden_sizes", "readout_mode"}
    integer_names = {
        "upgd_head_loss_pressure_warmup_steps",
        "upgd_head_repetition_warmup_steps",
        "slots_per_class",
    }
    for field in dataclasses.fields(config):
        if field.name in non_real_names | integer_names:
            continue
        value = getattr(config, field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field.name} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field.name} must be finite")
    if config.feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if config.n_heads < 2:
        raise ValueError("n_heads must be at least 2")
    if any(size < 1 for size in config.hidden_sizes):
        raise ValueError("hidden_sizes must contain only positive widths")
    if config.readout_mode not in {"linear_mse", "softmax_ce"}:
        raise ValueError("readout_mode must be 'linear_mse' or 'softmax_ce'")
    if config.upgd_step_size <= 0.0:
        raise ValueError("upgd_step_size must be positive")
    if config.upgd_head_step_size_multiplier <= 0.0:
        raise ValueError("upgd_head_step_size_multiplier must be positive")
    if config.upgd_head_bias_step_size_multiplier < 0.0:
        raise ValueError("upgd_head_bias_step_size_multiplier must be non-negative")
    if config.upgd_head_loss_pressure_gate_ratio < 0.0:
        raise ValueError("upgd_head_loss_pressure_gate_ratio must be non-negative")
    if config.upgd_head_loss_pressure_multiplier < 0.0:
        raise ValueError("upgd_head_loss_pressure_multiplier must be non-negative")
    if config.upgd_head_loss_pressure_warmup_steps < 0:
        raise ValueError("upgd_head_loss_pressure_warmup_steps must be non-negative")
    if config.upgd_head_repetition_multiplier < 0.0:
        raise ValueError("upgd_head_repetition_multiplier must be non-negative")
    if not 0.0 <= config.upgd_head_repetition_decay < 1.0:
        raise ValueError("upgd_head_repetition_decay must be in [0, 1)")
    if config.upgd_head_repetition_delta_threshold < 0.0:
        raise ValueError("upgd_head_repetition_delta_threshold must be non-negative")
    if not 0.0 <= config.upgd_head_repetition_pressure_threshold < 1.0:
        raise ValueError("upgd_head_repetition_pressure_threshold must be in [0, 1)")
    if config.upgd_head_repetition_warmup_steps < 0:
        raise ValueError("upgd_head_repetition_warmup_steps must be non-negative")
    if config.slots_per_class < 1:
        raise ValueError("slots_per_class must be positive")
    if not 0.0 < config.memory_update_rate <= 1.0:
        raise ValueError("memory_update_rate must be in (0, 1]")
    if config.initial_novelty_threshold <= 0.0:
        raise ValueError("initial_novelty_threshold must be positive")
    if config.memory_bandwidth <= 0.0:
        raise ValueError("memory_bandwidth must be positive")
    if not -8.0 <= config.initial_memory_logit <= 8.0:
        raise ValueError("initial_memory_logit must be in [-8, 8]")
    if config.memory_logit_step_size < 0.0:
        raise ValueError("memory_logit_step_size must be non-negative")
    if not 0.0 <= config.reliability_decay < 1.0:
        raise ValueError("reliability_decay must be in [0, 1)")
    if not 0.0 <= config.target_trace_blend_scale <= 1.0:
        raise ValueError("target_trace_blend_scale must be in [0, 1]")
    if not 0.0 <= config.target_trace_pressure_threshold < 1.0:
        raise ValueError("target_trace_pressure_threshold must be in [0, 1)")
    if config.novelty_adaptation_rate < 0.0:
        raise ValueError("novelty_adaptation_rate must be non-negative")
    if not 0.0 <= config.target_allocation_rate <= 1.0:
        raise ValueError("target_allocation_rate must be in [0, 1]")
    if config.min_novelty_threshold <= 0.0:
        raise ValueError("min_novelty_threshold must be positive")
    if config.max_novelty_threshold < config.min_novelty_threshold:
        raise ValueError("max_novelty_threshold must be >= min_novelty_threshold")
    if not (
        config.min_novelty_threshold
        <= config.initial_novelty_threshold
        <= config.max_novelty_threshold
    ):
        raise ValueError("initial_novelty_threshold must lie within configured bounds")


def _require_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    """Require a persistent/input array's shape and effective dtype."""
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    """Propose an exact outer increment without wrapping all ones."""
    _require_array_contract(
        words,
        name="UPGD-memory step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_to_int32_telemetry(words: Array) -> Array:
    saturated = (words[0] > jnp.asarray(0, dtype=jnp.uint32)) | (
        words[1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        words[1].astype(jnp.int32),
    )


def _active_mse(prediction: Array, target: Array) -> Array:
    active = jnp.isfinite(target)
    safe_target = jnp.where(active, target, 0.0)
    squared = jnp.where(active, (prediction - safe_target) ** 2, 0.0)
    return jnp.sum(squared) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)


def _normalize_simplex(prediction: Array) -> Array:
    clipped = jnp.maximum(prediction, 0.0)
    return clipped / jnp.maximum(jnp.sum(clipped), 1e-12)


class UPGDMemoryLearner:
    """UPGD plus adaptive fixed-budget prototype memory as one learner."""

    def __init__(self, config: UPGDMemoryConfig):
        _validate_config(config)
        self._config = config
        # The UPGD sub-configuration is frozen here; UPGDMemoryConfig exposes
        # only step-size and head-plasticity knobs.  The fixed values match
        # ``UPGDLearner.step2_default``: ObGD bounding at kappa 0.5, init
        # sparsity 0.5, layer norm, Rademacher perturbation of sigma 1e-4
        # every 16 steps, and target-structure loss normalization for one-hot
        # targets.  Relative to raw ``UPGDLearner`` defaults (sigma 1e-3 every
        # step, sparsity 0.9, no update bounding) this perturbs far more
        # gently and keeps more weights alive at init.
        self._upgd = UPGDLearner(
            n_heads=config.n_heads,
            hidden_sizes=config.hidden_sizes,
            step_size=config.upgd_step_size,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=0.5,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            utility_decay=0.995,
            perturbation_beta=2.0,
            perturbation_interval=16,
            loss_normalization="target_structure",
            readout_mode=config.readout_mode,
            track_unit_utilities=False,
            track_gradient_history=False,
            head_step_size_multiplier=config.upgd_head_step_size_multiplier,
            head_bias_step_size_multiplier=(config.upgd_head_bias_step_size_multiplier),
            head_loss_pressure_gate_ratio=(config.upgd_head_loss_pressure_gate_ratio),
            head_loss_pressure_multiplier=(config.upgd_head_loss_pressure_multiplier),
            head_loss_pressure_warmup_steps=(config.upgd_head_loss_pressure_warmup_steps),
            head_repetition_multiplier=config.upgd_head_repetition_multiplier,
            head_repetition_decay=config.upgd_head_repetition_decay,
            head_repetition_delta_threshold=(config.upgd_head_repetition_delta_threshold),
            head_repetition_pressure_threshold=(config.upgd_head_repetition_pressure_threshold),
            head_repetition_warmup_steps=(config.upgd_head_repetition_warmup_steps),
        )
        self._memory = PrototypeMemoryLearner(
            PrototypeMemoryConfig(
                feature_dim=config.feature_dim,
                n_classes=config.n_heads,
                slots_per_class=config.slots_per_class,
                update_rate=config.memory_update_rate,
                novelty_threshold=config.initial_novelty_threshold,
                bandwidth=config.memory_bandwidth,
            )
        )

    @property
    def config(self) -> UPGDMemoryConfig:
        """Learner configuration."""
        return self._config

    @property
    def upgd(self) -> UPGDLearner:
        """Underlying UPGD component."""
        return self._upgd

    @property
    def memory(self) -> PrototypeMemoryLearner:
        """Underlying fixed-budget prototype memory component."""
        return self._memory

    def to_config(self) -> dict[str, object]:
        """Serialize the learner configuration."""
        return {
            "type": "UPGDMemoryLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UPGDMemoryLearner:
        """Reconstruct from :meth:`to_config` output."""
        values = dict(config)
        if set(values) != {"type", "config"}:
            raise ValueError("UPGD-memory learner config fields are invalid")
        if values.pop("type") != "UPGDMemoryLearner":
            raise ValueError("UPGD-memory learner config type is unsupported")
        raw_config = values.pop("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("UPGD-memory inner config is invalid")
        return cls(UPGDMemoryConfig.from_config(dict(raw_config)))

    def init(self, key: Array | None = None) -> UPGDMemoryState:
        """Initialize both components and adaptive blend state."""
        if key is None:
            key = jr.key(0)
        cfg = self._config
        return UPGDMemoryState(
            upgd_state=self._upgd.init(cfg.feature_dim, key),
            memory_state=self._memory.init(),
            memory_logit=jnp.asarray(cfg.initial_memory_logit, dtype=jnp.float32),
            novelty_log_threshold=jnp.log(
                jnp.asarray(cfg.initial_novelty_threshold, dtype=jnp.float32)
            ),
            upgd_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            memory_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            blended_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            allocation_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: UPGDMemoryState) -> None:
        """Require the exact v2 composite state structure."""
        self._upgd._require_state_contract(  # noqa: SLF001
            state.upgd_state,
            feature_dim=self._config.feature_dim,
        )
        self._memory._require_state_contract(state.memory_state)  # noqa: SLF001
        for name, value in (
            ("memory_logit", state.memory_logit),
            ("novelty_log_threshold", state.novelty_log_threshold),
            ("upgd_loss_ema", state.upgd_loss_ema),
            ("memory_loss_ema", state.memory_loss_ema),
            ("blended_loss_ema", state.blended_loss_ema),
            ("allocation_ema", state.allocation_ema),
        ):
            _require_array_contract(
                value,
                name=f"UPGD-memory {name}",
                shape=(),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array_contract(
            state.step_count,
            name="UPGD-memory step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.step_words,
            name="UPGD-memory step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: UPGDMemoryState) -> Array:
        """Authenticate outer, UPGD, memory, and adaptive blend state."""
        self._require_state_contract(state)
        scalars = jnp.stack(
            (
                state.memory_logit,
                state.novelty_log_threshold,
                state.upgd_loss_ema,
                state.memory_loss_ema,
                state.blended_loss_ema,
                state.allocation_ema,
            )
        )
        lower_log = jnp.log(
            jnp.asarray(self._config.min_novelty_threshold, dtype=jnp.float32)
        )
        upper_log = jnp.log(
            jnp.asarray(self._config.max_novelty_threshold, dtype=jnp.float32)
        )
        return (
            self._upgd._transaction_state_valid(state.upgd_state)  # noqa: SLF001
            & self._memory.state_is_valid(state.memory_state)
            & jnp.all(state.step_words == state.upgd_state.step_words)
            & jnp.all(state.step_words == state.memory_state.step_words)
            & (state.step_count == _words_to_int32_telemetry(state.step_words))
            & jnp.all(jnp.isfinite(scalars))
            & (state.memory_logit >= -8.0)
            & (state.memory_logit <= 8.0)
            & (state.novelty_log_threshold >= lower_log)
            & (state.novelty_log_threshold <= upper_log)
            & (state.upgd_loss_ema >= 0.0)
            & (state.memory_loss_ema >= 0.0)
            & (state.blended_loss_ema >= 0.0)
            & (state.allocation_ema >= 0.0)
            & (state.allocation_ema <= 1.0)
        )

    def resource_budget(
        self, state: UPGDMemoryState | None = None
    ) -> UPGDMemoryResourceBudget:
        """Return exact persistent and nested-child resource accounting."""
        if state is None:
            state = self.init(jr.key(0))
        self._require_state_contract(state)
        return UPGDMemoryResourceBudget(
            state_nbytes=measure_upgd_memory_state_nbytes(state),
            outer_clock_nbytes=UPGD_MEMORY_OUTER_CLOCK_NBYTES,
            outer_clock_delta_nbytes=UPGD_MEMORY_OUTER_CLOCK_DELTA_NBYTES,
            upgd_state_nbytes=measure_upgd_state_nbytes(state.upgd_state),
            prototype_memory_state_nbytes=measure_prototype_memory_state_nbytes(
                state.memory_state
            ),
        )

    def _blend_gate(
        self,
        state: UPGDMemoryState,
        upgd_prediction: Array,
        memory_prediction: Array,
    ) -> Array:
        active_memory = (jnp.sum(state.memory_state.counts > 0.0) > 0).astype(jnp.float32)
        confidence_delta = jnp.max(memory_prediction) - jnp.max(upgd_prediction)
        reliability_delta = state.upgd_loss_ema - state.memory_loss_ema
        logit = (
            state.memory_logit
            + self._config.confidence_logit_scale * confidence_delta
            + self._config.reliability_logit_scale * reliability_delta
        )
        return active_memory * jax.nn.sigmoid(logit)

    def _blend_predictions(
        self,
        state: UPGDMemoryState,
        upgd_prediction: Array,
        memory_prediction: Array,
        *,
        include_target_trace: bool,
    ) -> tuple[Array, Array]:
        gate = self._blend_gate(state, upgd_prediction, memory_prediction)
        prediction = (1.0 - gate) * upgd_prediction + gate * memory_prediction
        if self._config.readout_mode == "softmax_ce":
            prediction = _normalize_simplex(prediction)
        trace_scale = jnp.where(
            include_target_trace,
            jnp.asarray(self._config.target_trace_blend_scale, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        threshold = jnp.asarray(
            self._config.target_trace_pressure_threshold,
            dtype=jnp.float32,
        )
        trace_pressure = jnp.clip(
            (state.upgd_state.target_repeat_ema - threshold) / jnp.maximum(1.0 - threshold, 1e-6),
            0.0,
            1.0,
        )
        trace_gate = trace_scale * trace_pressure
        trace_prediction = _normalize_simplex(state.upgd_state.previous_targets)
        prediction = (1.0 - trace_gate) * prediction + trace_gate * trace_prediction
        if self._config.readout_mode == "softmax_ce":
            prediction = _normalize_simplex(prediction)
        return prediction, gate

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_heads"]:
        """Predict with the current learned UPGD-memory blend."""
        self._require_state_contract(state)
        _require_array_contract(
            observation,
            name="UPGD-memory observation",
            shape=(self._config.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        upgd_prediction = self._upgd.predict(state.upgd_state, observation)
        memory_prediction = self._memory.predict(state.memory_state, observation)
        prediction, _gate = self._blend_predictions(
            state,
            upgd_prediction,
            memory_prediction,
            include_target_trace=False,
        )
        return prediction

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_heads"],
    ) -> UPGDMemoryUpdateResult:
        """Stage every child and globally commit one exact atomic transaction."""
        self._require_state_contract(state)
        raw_observation = _require_array_contract(
            observation,
            name="UPGD-memory observation",
            shape=(self._config.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        raw_target = _require_array_contract(
            target,
            name="UPGD-memory target",
            shape=(self._config.n_heads,),
            dtype=jnp.dtype(jnp.float32),
        )
        state_valid = self.state_is_valid(state)
        proposed_step_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        input_valid = jnp.all(jnp.isfinite(raw_observation)) & jnp.all(
            jnp.isfinite(raw_target)
        )
        safe_observation = jnp.where(jnp.isfinite(raw_observation), raw_observation, 0.0)
        safe_target = jnp.where(jnp.isfinite(raw_target), raw_target, 0.0)
        upgd_prediction = self._upgd.predict(state.upgd_state, safe_observation)
        memory_prediction = self._memory.predict(state.memory_state, safe_observation)
        prediction, gate = self._blend_predictions(
            state,
            upgd_prediction,
            memory_prediction,
            include_target_trace=True,
        )
        errors = prediction - safe_target
        blended_loss = _active_mse(prediction, safe_target)
        upgd_loss = _active_mse(upgd_prediction, safe_target)
        memory_loss = _active_mse(memory_prediction, safe_target)

        def blend_loss(memory_logit: Array) -> Array:
            probe_prediction, _probe_gate = self._blend_predictions(
                state.replace(memory_logit=memory_logit),  # type: ignore[attr-defined]
                upgd_prediction,
                memory_prediction,
                include_target_trace=True,
            )
            return _active_mse(probe_prediction, safe_target)

        dloss_dlogit = jax.grad(blend_loss)(state.memory_logit)
        next_memory_logit = state.memory_logit - (
            jnp.asarray(self._config.memory_logit_step_size, dtype=jnp.float32) * dloss_dlogit
        )
        # Bound the learned base logit: sigmoid(+/-8) is already ~0.9997, so
        # the clip costs nothing in gate range but stops unbounded drift during
        # long one-sided regimes, keeping the additive confidence/reliability
        # terms able to reverse the gate after a regime change.
        next_memory_logit = jnp.clip(next_memory_logit, -8.0, 8.0)

        raw_threshold = jnp.exp(state.novelty_log_threshold)
        threshold = jnp.where(
            jnp.isfinite(raw_threshold),
            raw_threshold,
            jnp.asarray(self._config.initial_novelty_threshold, dtype=jnp.float32),
        )
        upgd_result = self._upgd.update(
            state.upgd_state,
            safe_observation,
            safe_target,
        )
        memory_result = self._memory.update_with_novelty_threshold(
            state.memory_state,
            safe_observation,
            safe_target,
            threshold,
        )
        allocated = memory_result.metrics[5]
        decay = jnp.asarray(self._config.reliability_decay, dtype=jnp.float32)
        one_minus_decay = 1.0 - decay
        next_allocation_ema = decay * state.allocation_ema + one_minus_decay * allocated
        allocation_error = next_allocation_ema - jnp.asarray(
            self._config.target_allocation_rate,
            dtype=jnp.float32,
        )
        next_log_threshold = state.novelty_log_threshold + (
            jnp.asarray(self._config.novelty_adaptation_rate, dtype=jnp.float32) * allocation_error
        )
        next_log_threshold = jnp.clip(
            next_log_threshold,
            jnp.log(jnp.asarray(self._config.min_novelty_threshold, dtype=jnp.float32)),
            jnp.log(jnp.asarray(self._config.max_novelty_threshold, dtype=jnp.float32)),
        )

        candidate_state = UPGDMemoryState(
            upgd_state=upgd_result.state,
            memory_state=memory_result.state,
            memory_logit=next_memory_logit,
            novelty_log_threshold=next_log_threshold,
            upgd_loss_ema=decay * state.upgd_loss_ema + one_minus_decay * upgd_loss,
            memory_loss_ema=decay * state.memory_loss_ema + one_minus_decay * memory_loss,
            blended_loss_ema=(decay * state.blended_loss_ema + one_minus_decay * blended_loss),
            allocation_ema=next_allocation_ema,
            step_count=_words_to_int32_telemetry(proposed_step_words),
            step_words=proposed_step_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        blend_update_valid = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        next_memory_logit,
                        next_log_threshold,
                        next_allocation_ema,
                        upgd_loss,
                        memory_loss,
                        blended_loss,
                        gate,
                    )
                )
            )
        )
        children_aligned = (
            jnp.all(upgd_result.post_step_words == proposed_step_words)
            & jnp.all(memory_result.post_step_words == proposed_step_words)
        )
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & upgd_result.update_applied
            & memory_result.update_applied
            & children_aligned
            & blend_update_valid
            & candidate_state_valid
        )
        next_state = jax.tree.map(
            lambda proposed, current: jnp.where(update_applied, proposed, current),
            candidate_state,
            state,
        )
        proposed_metrics = jnp.asarray(
            [
                blended_loss,
                upgd_loss,
                memory_loss,
                gate,
                next_memory_logit,
                threshold,
                next_allocation_ema,
                jnp.sum(memory_result.state.counts > 0.0).astype(jnp.float32),
                jnp.max(upgd_prediction),
                jnp.max(memory_prediction),
            ],
            dtype=jnp.float32,
        )
        metrics = jnp.where(
            update_applied,
            proposed_metrics,
            jnp.zeros_like(proposed_metrics),
        )
        return UPGDMemoryUpdateResult(
            state=next_state,
            predictions=jnp.where(
                update_applied,
                prediction,
                jnp.full_like(prediction, jnp.nan),
            ),
            errors=jnp.where(
                update_applied,
                errors,
                jnp.full_like(errors, jnp.nan),
            ),
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            state_valid=state_valid,
            candidate_state_valid=candidate_state_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            upgd_update_applied=upgd_result.update_applied,
            memory_update_applied=memory_result.update_applied,
            blend_update_valid=blend_update_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def run_upgd_memory_arrays(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    observations: Float[Array, "steps feature_dim"],
    targets: Float[Array, "steps n_heads"],
) -> UPGDMemoryLearningResult:
    """Run a UPGD-memory learner over arrays with ``jax.lax.scan``.

    Metric columns are ``blend_mse, upgd_mse, memory_mse, gate, memory_logit,
    novelty_threshold, allocation_ema, active_prototypes, upgd_conf,
    memory_conf``.
    """

    def step_fn(
        carry: UPGDMemoryState,
        batch: tuple[Array, Array],
    ) -> tuple[UPGDMemoryState, tuple[Array, Array]]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, (result.predictions, result.metrics)

    final_state, (predictions, metrics) = jax.lax.scan(
        step_fn,
        state,
        (observations, targets),
    )
    return UPGDMemoryLearningResult(
        state=final_state,
        predictions=predictions,
        metrics=metrics,
    )


def measure_upgd_memory_state_nbytes(state: UPGDMemoryState) -> int:
    """Measure every persistent JAX-array byte in one composite state."""
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def _zero_sized_array_leaf_indices(state: UPGDMemoryState) -> list[int]:
    """Identify empty persistent leaves that Orbax cannot store directly."""
    return [
        index
        for index, leaf in enumerate(jax.tree.leaves(state))
        if isinstance(leaf, Array) and leaf.size == 0
    ]


def _checkpoint_storage_state(state: UPGDMemoryState) -> UPGDMemoryState:
    """Encode structurally fixed empty arrays as one-element storage sentinels."""
    return cast(
        UPGDMemoryState,
        jax.tree.map(
            lambda leaf: (
                jnp.zeros((1,), dtype=leaf.dtype)
                if isinstance(leaf, Array) and leaf.size == 0
                else leaf
            ),
            state,
        ),
    )


def _restore_checkpoint_empty_arrays(
    restored: UPGDMemoryState,
    template: UPGDMemoryState,
) -> UPGDMemoryState:
    """Decode storage sentinels using the config-derived exact template."""
    return cast(
        UPGDMemoryState,
        jax.tree.map(
            lambda stored, expected: (
                expected
                if isinstance(expected, Array) and expected.size == 0
                else stored
            ),
            restored,
            template,
        ),
    )


def _state_fields(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"{name} must be a mapping or dataclass")


def migrate_legacy_upgd_memory_state(
    legacy_state: Any,
    *,
    config: UPGDMemoryConfig,
) -> UPGDMemoryState:
    """Migrate an exact unsaturated pre-v2 composite and both child clocks."""
    learner = UPGDMemoryLearner(config)
    fields = _state_fields(legacy_state, name="legacy UPGD-memory state")
    current_names = {
        field.name for field in dataclasses.fields(UPGDMemoryState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy UPGD-memory field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    outer_count = _require_array_contract(
        fields["step_count"],
        name="legacy UPGD-memory step_count",
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )
    step = int(outer_count)
    if step < 0:
        raise ValueError("negative legacy UPGD-memory step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy UPGD-memory step_count is ambiguous")

    raw_upgd = fields["upgd_state"]
    upgd_names = set(_state_fields(raw_upgd, name="legacy nested UPGD state"))
    current_upgd_names = {
        field.name for field in dataclasses.fields(UPGDState)  # type: ignore[arg-type]
    }
    if upgd_names == current_upgd_names:
        if not isinstance(raw_upgd, UPGDState):
            raw_upgd = UPGDState(**_state_fields(raw_upgd, name="nested UPGD state"))
        upgd_state = raw_upgd
    else:
        upgd_state = migrate_legacy_upgd_state(
            raw_upgd,
            perturbation_interval=16,
        )

    raw_memory = fields["memory_state"]
    memory_names = set(_state_fields(raw_memory, name="legacy nested memory state"))
    current_memory_names = {
        field.name for field in dataclasses.fields(PrototypeMemoryState)  # type: ignore[arg-type]
    }
    if memory_names == current_memory_names:
        if not isinstance(raw_memory, PrototypeMemoryState):
            raw_memory = PrototypeMemoryState(
                **_state_fields(raw_memory, name="nested memory state")
            )
        memory_state = raw_memory
    else:
        memory_state = migrate_legacy_prototype_memory_state(
            raw_memory,
            config=learner.memory.config,
        )

    fields["upgd_state"] = upgd_state
    fields["memory_state"] = memory_state
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated = UPGDMemoryState(**fields)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy UPGD-memory state violates the v2 contract")
    return migrated


def save_upgd_memory_checkpoint(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    path: str | Path,
) -> None:
    """Persist one globally authenticated v2 composite transaction state."""
    learner._require_state_contract(state)
    if not bool(jax.device_get(learner.state_is_valid(state))):
        raise ValueError("UPGD-memory checkpoint state is invalid")
    save_checkpoint(
        _checkpoint_storage_state(state),
        path,
        metadata={
            "schema": UPGD_MEMORY_CHECKPOINT_SCHEMA,
            "learner_config": learner.to_config(),
            "memory_accounting": learner.resource_budget(state).to_dict(),
            "child_state_schemas": {
                "upgd": UPGD_STATE_SCHEMA,
                "prototype_memory": PROTOTYPE_MEMORY_STATE_SCHEMA,
            },
            "zero_sized_array_leaf_indices": _zero_sized_array_leaf_indices(state),
        },
    )


def load_upgd_memory_checkpoint(
    path: str | Path,
) -> tuple[UPGDMemoryLearner, UPGDMemoryState]:
    """Restore only an authenticated exact-clock v2 composite checkpoint."""
    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "learner_config",
        "memory_accounting",
        "child_state_schemas",
        "zero_sized_array_leaf_indices",
    }
    if set(metadata) != expected:
        raise ValueError("UPGD-memory checkpoint metadata fields are invalid")
    schema = metadata.get("schema")
    if schema == _LEGACY_UPGD_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy UPGD-memory checkpoint v1 lacks exact composite identities; "
            "migrate its state and resave it"
        )
    if schema != UPGD_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError("UPGD-memory checkpoint schema is unsupported")
    schemas = metadata.get("child_state_schemas")
    if schemas != {
        "upgd": UPGD_STATE_SCHEMA,
        "prototype_memory": PROTOTYPE_MEMORY_STATE_SCHEMA,
    }:
        raise ValueError("UPGD-memory child state schemas are unsupported")
    raw_config = metadata.get("learner_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("UPGD-memory checkpoint learner_config is invalid")
    learner = UPGDMemoryLearner.from_config(dict(raw_config))
    template = learner.init(jr.key(0))
    expected_empty_leaves = _zero_sized_array_leaf_indices(template)
    if metadata.get("zero_sized_array_leaf_indices") != expected_empty_leaves:
        raise ValueError("UPGD-memory empty-array storage manifest does not match")
    restored, restored_metadata = load_checkpoint(
        _checkpoint_storage_state(template), path
    )
    if restored_metadata != metadata:
        raise ValueError("UPGD-memory checkpoint metadata changed between reads")
    state = _restore_checkpoint_empty_arrays(restored, template)
    if not isinstance(state, UPGDMemoryState):
        raise TypeError("UPGD-memory checkpoint state type is invalid")
    learner._require_state_contract(state)
    if not bool(jax.device_get(learner.state_is_valid(state))):
        raise ValueError("restored UPGD-memory state is invalid")
    if learner.resource_budget(state).to_dict() != metadata.get("memory_accounting"):
        raise ValueError("UPGD-memory checkpoint resource contract does not match")
    return learner, state
