# mypy: disable-error-code="call-arg,name-defined"
"""Fixed-budget JAX prototype memory for Step 2 retention.

An online nearest-prototype classifier (cf. learning vector quantization,
Kohonen 1990; nearest-class-mean classification, Mensink et al. 2013): each
class owns a fixed number of prototype slots, a matched prototype is
EMA-updated toward the observation, and a sufficiently novel observation
claims an empty slot or recycles the least-used/oldest one (a fixed-budget,
distance-only variant of the allocation rule in Platt 1991).  Prediction is a
softmax over nearest-prototype class logits.

The budget is static, every step can update the memory, and the state is a
JAX PyTree, so the learner runs under ``jax.lax.scan``.  It is the core of
the promoted Step 2 retained-view memory
(:func:`alberta_framework.steps.step2.make_step2_memory_learner`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_CONSECUTIVE_INTEGER_LIMIT = 2**24

PROTOTYPE_MEMORY_STATE_SCHEMA = "alberta.prototype-memory-state.v2"
PROTOTYPE_MEMORY_CONFIG_SCHEMA = "alberta.prototype-memory-config.v2"
PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA = "alberta.prototype-memory-checkpoint.v2"
_LEGACY_PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA = "alberta.prototype-memory-checkpoint.v1"

__all__ = [
    "PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_MEMORY_CONFIG_SCHEMA",
    "PROTOTYPE_MEMORY_STATE_SCHEMA",
    "PrototypeMemoryConfig",
    "PrototypeMemoryLearner",
    "PrototypeMemoryLearningResult",
    "PrototypeMemoryResourceBudget",
    "PrototypeMemoryState",
    "PrototypeMemoryUpdateResult",
    "load_prototype_memory_checkpoint",
    "measure_prototype_memory_state_nbytes",
    "migrate_legacy_prototype_memory_state",
    "prototype_memory_exact_clock_delta_nbytes",
    "prototype_memory_state_delta_nbytes",
    "run_prototype_memory_arrays",
    "save_prototype_memory_checkpoint",
]


@chex.dataclass(frozen=True)
class PrototypeMemoryConfig:
    """Configuration for :class:`PrototypeMemoryLearner`.

    Args:
        feature_dim: Observation dimensionality.
        n_classes: Number of one-hot classes.
        slots_per_class: Fixed prototype budget for each class.
        update_rate: EMA rate for updating a matched prototype.
        novelty_threshold: Mean-squared-distance threshold for allocating a
            new prototype instead of updating the nearest existing one.
        bandwidth: Distance-to-logit bandwidth for softmax prediction.

    ``novelty_threshold`` and ``bandwidth`` are both in units of *mean
    per-dimension* squared distance, so they are insensitive to
    ``feature_dim`` but scale with the square of the input magnitude. The
    defaults are the promoted Step 2 MNIST-pixel calibration
    (:class:`~alberta_framework.steps.step2.Step2MemoryConfig` uses the same
    values); retune them for features on a different scale.
    """

    feature_dim: int
    n_classes: int
    slots_per_class: int = 20
    update_rate: float = 0.3
    novelty_threshold: float = 0.08
    bandwidth: float = 0.01

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "type": "PrototypeMemoryConfig",
            "config_schema": PROTOTYPE_MEMORY_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_MEMORY_STATE_SCHEMA,
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
            "slots_per_class": self.slots_per_class,
            "update_rate": self.update_rate,
            "novelty_threshold": self.novelty_threshold,
            "bandwidth": self.bandwidth,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PrototypeMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        values = dict(config)
        expected = set(cls(feature_dim=1, n_classes=2).to_config())
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                "prototype-memory config field manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if values.pop("type") != "PrototypeMemoryConfig":
            raise ValueError("prototype-memory config type is unsupported")
        if values.pop("config_schema") != PROTOTYPE_MEMORY_CONFIG_SCHEMA:
            raise ValueError("prototype-memory config schema is unsupported")
        if values.pop("state_schema") != PROTOTYPE_MEMORY_STATE_SCHEMA:
            raise ValueError("prototype-memory state schema is unsupported")
        return cls(**values)


@chex.dataclass(frozen=True)
class PrototypeMemoryState:
    """State for :class:`PrototypeMemoryLearner`."""

    means: Float[Array, "n_classes slots_per_class feature_dim"]
    counts: Float[Array, "n_classes slots_per_class"]
    visit_words: Array
    last_update: Array
    last_update_words: Array
    insertion_step: Array
    insertion_words: Array
    step_count: Array
    step_words: Array


@chex.dataclass(frozen=True)
class PrototypeMemoryUpdateResult:
    """Result of one prototype-memory update."""

    state: PrototypeMemoryState
    predictions: Float[Array, " n_classes"]
    errors: Float[Array, " n_classes"]
    metrics: Float[Array, " 6"]
    pre_step_words: Array
    post_step_words: Array
    state_valid: Array
    candidate_state_valid: Array
    lifetime_capacity_available: Array
    update_applied: Array
    update_rejected: Array


@chex.dataclass(frozen=True)
class PrototypeMemoryLearningResult:
    """Result from :func:`run_prototype_memory_arrays`."""

    state: PrototypeMemoryState
    predictions: Float[Array, "steps n_classes"]
    metrics: Float[Array, "steps 6"]


@dataclass(frozen=True)
class PrototypeMemoryResourceBudget:
    """Exact persistent-memory contract for one fixed-capacity memory."""

    state_nbytes: int
    slot_capacity: int
    exact_clock_nbytes: int
    exact_clock_delta_nbytes: int
    state_delta_nbytes: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible resource declaration."""
        return {
            "type": "PrototypeMemoryResourceBudget",
            "state_nbytes": self.state_nbytes,
            "slot_capacity": self.slot_capacity,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "state_delta_nbytes": self.state_delta_nbytes,
        }


def _validate_config(config: PrototypeMemoryConfig) -> None:
    if isinstance(config.feature_dim, bool) or not isinstance(config.feature_dim, int):
        raise ValueError("feature_dim must be an integer")
    if config.feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if isinstance(config.n_classes, bool) or not isinstance(config.n_classes, int):
        raise ValueError("n_classes must be an integer")
    if config.n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if isinstance(config.slots_per_class, bool) or not isinstance(
        config.slots_per_class, int
    ):
        raise ValueError("slots_per_class must be an integer")
    if config.slots_per_class < 1:
        raise ValueError("slots_per_class must be positive")
    for name, value in (
        ("update_rate", config.update_rate),
        ("novelty_threshold", config.novelty_threshold),
        ("bandwidth", config.bandwidth),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < config.update_rate <= 1.0:
        raise ValueError("update_rate must be in (0, 1]")
    if config.novelty_threshold < 0.0:
        raise ValueError("novelty_threshold must be non-negative")
    if config.bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")


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
    """Propose an exact uint64-word increment without wrapping all ones."""
    if words.shape[-1:] != (2,):
        raise ValueError("exact lifetime words must have trailing shape (2,)")
    if words.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("exact lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum, axis=-1)
    low = words[..., 1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((words[..., 0] + carry, low), axis=-1).astype(jnp.uint32)
    return jnp.where(available[..., None], candidate, words), available


def _words_le(left: Array, right: Array) -> Array:
    """Compare big-endian uint32-word identities without enabling x64."""
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] <= right[..., 1])
    )


def _words_to_int32_telemetry(words: Array) -> Array:
    """Project exact identities onto saturating compatibility telemetry."""
    saturated = (words[..., 0] > jnp.asarray(0, dtype=jnp.uint32)) | (
        words[..., 1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        words[..., 1].astype(jnp.int32),
    )


def _words_to_visit_telemetry(words: Array) -> Array:
    """Project exact visits onto non-aliasing saturating float32 telemetry."""
    saturated = (words[..., 0] > jnp.asarray(0, dtype=jnp.uint32)) | (
        words[..., 1]
        >= jnp.asarray(_FLOAT32_CONSECUTIVE_INTEGER_LIMIT, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_FLOAT32_CONSECUTIVE_INTEGER_LIMIT, dtype=jnp.float32),
        words[..., 1].astype(jnp.float32),
    )


def _tree_commit(
    candidate: PrototypeMemoryState,
    state: PrototypeMemoryState,
    gate: Array,
) -> PrototypeMemoryState:
    """Select one whole memory state with no partially committed leaves."""
    return cast(
        PrototypeMemoryState,
        jax.tree.map(lambda new, old: jnp.where(gate, new, old), candidate, state),
    )


def _softmax(logits: Array) -> Array:
    shifted = logits - jnp.max(logits)
    exp = jnp.exp(shifted)
    return exp / jnp.maximum(jnp.sum(exp), 1e-12)


class PrototypeMemoryLearner:
    """Fixed-budget multi-prototype classifier.

    The learner assumes one-hot classification targets.  Non-finite or
    non-simplex targets are ignored by the memory update but still produce a
    prediction and metrics.  This keeps the learner safe in mixed-head streams.
    """

    def __init__(self, config: PrototypeMemoryConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> PrototypeMemoryConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the learner configuration."""
        return {
            "type": "PrototypeMemoryLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PrototypeMemoryLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        values = dict(config)
        if set(values) != {"type", "config"}:
            raise ValueError("prototype-memory learner config fields are invalid")
        if values.pop("type") != "PrototypeMemoryLearner":
            raise ValueError("prototype-memory learner config type is unsupported")
        raw_inner = values.pop("config")
        if not isinstance(raw_inner, Mapping):
            raise ValueError("prototype-memory inner config is invalid")
        inner = dict(raw_inner)
        return cls(PrototypeMemoryConfig.from_config(inner))

    def init(self) -> PrototypeMemoryState:
        """Create an empty fixed-budget memory."""
        c = self._config
        slot_shape = (c.n_classes, c.slots_per_class)
        return PrototypeMemoryState(
            means=jnp.zeros(
                (c.n_classes, c.slots_per_class, c.feature_dim),
                dtype=jnp.float32,
            ),
            counts=jnp.zeros(slot_shape, dtype=jnp.float32),
            visit_words=jnp.zeros((*slot_shape, 2), dtype=jnp.uint32),
            last_update=jnp.zeros(slot_shape, dtype=jnp.int32),
            last_update_words=jnp.zeros((*slot_shape, 2), dtype=jnp.uint32),
            insertion_step=jnp.zeros(slot_shape, dtype=jnp.int32),
            insertion_words=jnp.zeros((*slot_shape, 2), dtype=jnp.uint32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: PrototypeMemoryState) -> None:
        """Require the exact fixed-capacity v2 state manifest."""
        c = self._config
        slots = (c.n_classes, c.slots_per_class)
        _require_array_contract(
            state.means,
            name="prototype-memory means",
            shape=(*slots, c.feature_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array_contract(
            state.counts,
            name="prototype-memory counts",
            shape=slots,
            dtype=jnp.dtype(jnp.float32),
        )
        for name, value in (
            ("visit_words", state.visit_words),
            ("last_update_words", state.last_update_words),
            ("insertion_words", state.insertion_words),
        ):
            _require_array_contract(
                value,
                name=f"prototype-memory {name}",
                shape=(*slots, 2),
                dtype=jnp.dtype(jnp.uint32),
            )
        for name, value in (
            ("last_update", state.last_update),
            ("insertion_step", state.insertion_step),
        ):
            _require_array_contract(
                value,
                name=f"prototype-memory {name}",
                shape=slots,
                dtype=jnp.dtype(jnp.int32),
            )
        _require_array_contract(
            state.step_count,
            name="prototype-memory step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.step_words,
            name="prototype-memory step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: PrototypeMemoryState) -> Array:
        """Authenticate exact ownership, telemetry, and finite slot contents."""
        self._require_state_contract(state)
        used = jnp.any(state.visit_words != jnp.asarray(0, dtype=jnp.uint32), axis=-1)
        timestamp_nonzero = jnp.any(
            state.last_update_words != jnp.asarray(0, dtype=jnp.uint32), axis=-1
        )
        insertion_nonzero = jnp.any(
            state.insertion_words != jnp.asarray(0, dtype=jnp.uint32), axis=-1
        )
        timestamp_order = _words_le(state.insertion_words, state.last_update_words)
        within_lifetime = _words_le(state.last_update_words, state.step_words)
        return (
            jnp.all(jnp.isfinite(state.means))
            & jnp.all(jnp.isfinite(state.counts))
            & jnp.all(state.counts >= 0.0)
            & jnp.all(
                state.counts == _words_to_visit_telemetry(state.visit_words)
            )
            & jnp.all(used == (state.counts > 0.0))
            & jnp.all(_words_le(state.visit_words, state.step_words))
            & jnp.all(timestamp_nonzero == used)
            & jnp.all(insertion_nonzero == used)
            & jnp.all(jnp.where(used, timestamp_order & within_lifetime, True))
            & jnp.all(
                state.last_update
                == _words_to_int32_telemetry(state.last_update_words)
            )
            & jnp.all(
                state.insertion_step
                == _words_to_int32_telemetry(state.insertion_words)
            )
            & (state.step_count == _words_to_int32_telemetry(state.step_words))
        )

    def resource_budget(
        self, state: PrototypeMemoryState | None = None
    ) -> PrototypeMemoryResourceBudget:
        """Return exact fixed-capacity persistent resource accounting."""
        if state is None:
            state = self.init()
        self._require_state_contract(state)
        slots = self._config.n_classes * self._config.slots_per_class
        exact_clock_bytes = 8 + 24 * slots
        return PrototypeMemoryResourceBudget(
            state_nbytes=measure_prototype_memory_state_nbytes(state),
            slot_capacity=slots,
            exact_clock_nbytes=exact_clock_bytes,
            exact_clock_delta_nbytes=prototype_memory_exact_clock_delta_nbytes(
                self._config
            ),
            state_delta_nbytes=prototype_memory_state_delta_nbytes(self._config),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def class_logits(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_classes"]:
        """Return class logits from nearest active prototype distances."""
        self._require_state_contract(state)
        _require_array_contract(
            observation,
            name="prototype-memory observation",
            shape=(self._config.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        x = jnp.asarray(observation, dtype=jnp.float32)
        diffs = state.means - x[None, None, :]
        distances = jnp.mean(diffs * diffs, axis=2)
        slot_logits = -distances / jnp.asarray(self._config.bandwidth, dtype=jnp.float32)
        slot_logits = jnp.where(state.counts > 0.0, slot_logits, -jnp.inf)
        logits = jnp.max(slot_logits, axis=1)
        any_active = jnp.any(state.counts > 0.0, axis=1)
        logits = jnp.where(any_active, logits, -1e9)
        logits = jnp.where(jnp.any(any_active), logits, jnp.zeros_like(logits))
        return logits

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_classes"]:
        """Return class probabilities for one observation."""
        return _softmax(self.class_logits(state, observation))

    @staticmethod
    def valid_one_hot_target(target: Array) -> Array:
        """Return whether ``target`` is a finite one-hot/simplex target."""
        finite = jnp.all(jnp.isfinite(target))
        target_sum = jnp.sum(target)
        max_target = jnp.max(target)
        non_negative = jnp.all(target >= -1e-6)
        return finite & non_negative & (jnp.abs(target_sum - 1.0) <= 1e-5) & (
            max_target >= 0.999
        )

    def _replacement_slot(self, state: PrototypeMemoryState, head: Array) -> Array:
        """Choose least-used, then oldest, slot for a full class budget."""
        visits = state.visit_words[head]
        timestamps = state.last_update_words[head]
        maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        min_visit_high = jnp.min(visits[:, 0])
        tied = visits[:, 0] == min_visit_high
        min_visit_low = jnp.min(jnp.where(tied, visits[:, 1], maximum))
        tied = tied & (visits[:, 1] == min_visit_low)
        min_time_high = jnp.min(jnp.where(tied, timestamps[:, 0], maximum))
        tied = tied & (timestamps[:, 0] == min_time_high)
        min_time_low = jnp.min(jnp.where(tied, timestamps[:, 1], maximum))
        tied = tied & (timestamps[:, 1] == min_time_low)
        return jnp.argmax(tied.astype(jnp.int32))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_novelty_threshold(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_classes"],
        novelty_threshold: Float[Array, ""],
    ) -> PrototypeMemoryUpdateResult:
        """Perform one validated atomic update with an exact event identity."""
        self._require_state_contract(state)
        raw_observation = _require_array_contract(
            observation,
            name="prototype-memory observation",
            shape=(self._config.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        raw_target = _require_array_contract(
            target,
            name="prototype-memory target",
            shape=(self._config.n_classes,),
            dtype=jnp.dtype(jnp.float32),
        )
        raw_threshold = _require_array_contract(
            novelty_threshold,
            name="prototype-memory novelty_threshold",
            shape=(),
            dtype=jnp.dtype(jnp.float32),
        )
        state_valid = self.state_is_valid(state)
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        input_valid = (
            jnp.all(jnp.isfinite(raw_observation))
            & jnp.all(jnp.isfinite(raw_target))
            & jnp.isfinite(raw_threshold)
            & (raw_threshold >= 0.0)
        )
        safe_observation = jnp.where(jnp.isfinite(raw_observation), raw_observation, 0.0)
        safe_target = jnp.where(jnp.isfinite(raw_target), raw_target, 0.0)
        safe_threshold = jnp.where(
            jnp.isfinite(raw_threshold) & (raw_threshold >= 0.0),
            raw_threshold,
            jnp.asarray(self._config.novelty_threshold, dtype=jnp.float32),
        )
        prediction = self.predict(state, safe_observation)
        valid_target = self.valid_one_hot_target(safe_target) & input_valid
        errors = prediction - safe_target
        mse = jnp.mean(errors * errors)
        confidence = jnp.max(prediction)
        correct = jnp.where(
            valid_target,
            (jnp.argmax(prediction) == jnp.argmax(safe_target)).astype(jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )

        def do_update(current: PrototypeMemoryState) -> tuple[PrototypeMemoryState, Array]:
            head = jnp.argmax(safe_target)
            used = current.counts[head] > 0.0
            has_used = jnp.any(used)
            has_empty = jnp.any(~used)
            distances = jnp.mean(
                (current.means[head] - safe_observation[None, :]) ** 2,
                axis=1,
            )
            used_distances = jnp.where(used, distances, jnp.inf)
            nearest_slot = jnp.argmin(used_distances)
            nearest_distance = used_distances[nearest_slot]
            empty_slot = jnp.argmax((~used).astype(jnp.int32))
            replacement_slot = self._replacement_slot(current, head)
            novel = (~has_used) | (nearest_distance > safe_threshold)
            slot = jnp.where(
                ~has_used,
                jnp.array(0, dtype=nearest_slot.dtype),
                jnp.where(
                    novel & has_empty,
                    empty_slot,
                    jnp.where(novel, replacement_slot, nearest_slot),
                ),
            )
            old_mean = current.means[head, slot]
            eta = jnp.asarray(self._config.update_rate, dtype=jnp.float32)
            new_mean = jnp.where(
                novel,
                safe_observation,
                old_mean + eta * (safe_observation - old_mean),
            )
            incremented_visits, visit_capacity = _checked_words_increment(
                current.visit_words[head, slot]
            )
            # A slot cannot exhaust before its owning outer lifetime.  Keep
            # the check load-bearing so corrupted states still fail closed.
            next_visits = jnp.where(
                novel,
                jnp.asarray((0, 1), dtype=jnp.uint32),
                incremented_visits,
            )
            next_insertion = jnp.where(
                novel,
                proposed_words,
                current.insertion_words[head, slot],
            )
            next_state = PrototypeMemoryState(
                means=current.means.at[head, slot].set(new_mean),
                counts=current.counts.at[head, slot].set(
                    _words_to_visit_telemetry(next_visits)
                ),
                visit_words=current.visit_words.at[head, slot].set(next_visits),
                last_update=current.last_update.at[head, slot].set(
                    _words_to_int32_telemetry(proposed_words)
                ),
                last_update_words=current.last_update_words.at[head, slot].set(
                    proposed_words
                ),
                insertion_step=current.insertion_step.at[head, slot].set(
                    _words_to_int32_telemetry(next_insertion)
                ),
                insertion_words=current.insertion_words.at[head, slot].set(
                    next_insertion
                ),
                step_count=_words_to_int32_telemetry(proposed_words),
                step_words=proposed_words,
            )
            return next_state, novel.astype(jnp.float32) * visit_capacity.astype(jnp.float32)

        def skip_update(current: PrototypeMemoryState) -> tuple[PrototypeMemoryState, Array]:
            return (
                PrototypeMemoryState(
                    means=current.means,
                    counts=current.counts,
                    visit_words=current.visit_words,
                    last_update=current.last_update,
                    last_update_words=current.last_update_words,
                    insertion_step=current.insertion_step,
                    insertion_words=current.insertion_words,
                    step_count=_words_to_int32_telemetry(proposed_words),
                    step_words=proposed_words,
                ),
                jnp.array(0.0, dtype=jnp.float32),
            )

        candidate_state, allocated = jax.lax.cond(
            valid_target, do_update, skip_update, state
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = _tree_commit(candidate_state, state, update_applied)
        active = jnp.sum(candidate_state.counts > 0.0).astype(jnp.float32)
        proposed_metrics = jnp.asarray(
            [
                mse,
                correct,
                confidence,
                active,
                valid_target.astype(jnp.float32),
                allocated,
            ],
            dtype=jnp.float32,
        )
        metrics = jnp.where(update_applied, proposed_metrics, jnp.zeros_like(proposed_metrics))
        return PrototypeMemoryUpdateResult(
            state=new_state,
            predictions=jnp.where(update_applied, prediction, jnp.full_like(prediction, jnp.nan)),
            errors=jnp.where(update_applied, errors, jnp.full_like(errors, jnp.nan)),
            metrics=metrics,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            state_valid=state_valid,
            candidate_state_valid=candidate_state_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_classes"],
    ) -> PrototypeMemoryUpdateResult:
        """Perform one causal online memory update."""
        return cast(
            PrototypeMemoryUpdateResult,
            self.update_with_novelty_threshold(
                state,
                observation,
                target,
                jnp.asarray(self._config.novelty_threshold, dtype=jnp.float32),
            ),
        )


def run_prototype_memory_arrays(
    learner: PrototypeMemoryLearner,
    observations: Float[Array, "steps feature_dim"],
    targets: Float[Array, "steps n_classes"],
    *,
    state: PrototypeMemoryState | None = None,
) -> PrototypeMemoryLearningResult:
    """Run the prototype memory over arrays with ``jax.lax.scan``.

    Metric columns are ``mse, correct, confidence, active_prototypes,
    valid_update, allocated``.
    """
    if state is None:
        state = learner.init()

    def step_fn(
        carry: PrototypeMemoryState,
        batch: tuple[Array, Array],
    ) -> tuple[PrototypeMemoryState, tuple[Array, Array]]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, (result.predictions, result.metrics)

    final_state, (predictions, metrics) = jax.lax.scan(
        step_fn,
        state,
        (observations, targets),
    )
    return PrototypeMemoryLearningResult(
        state=final_state,
        predictions=predictions,
        metrics=metrics,
    )


def prototype_memory_exact_clock_delta_nbytes(config: PrototypeMemoryConfig) -> int:
    """Return v2 exact-authority bytes added to the historical state."""
    _validate_config(config)
    slots = config.n_classes * config.slots_per_class
    # Outer identity plus exact visit, last-use, and insertion identities.
    return 8 + 24 * slots


def prototype_memory_state_delta_nbytes(config: PrototypeMemoryConfig) -> int:
    """Return all v2 persistent bytes added to the historical state."""
    _validate_config(config)
    slots = config.n_classes * config.slots_per_class
    # Exact authority plus new saturating insertion telemetry.
    return 8 + 28 * slots


def measure_prototype_memory_state_nbytes(state: PrototypeMemoryState) -> int:
    """Measure every persistent JAX-array byte in one memory state."""
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def _legacy_fields(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"legacy {name} state must be a mapping or dataclass")


def migrate_legacy_prototype_memory_state(
    legacy_state: Any,
    *,
    config: PrototypeMemoryConfig,
) -> PrototypeMemoryState:
    """Migrate only an unsaturated, non-aliased pre-v2 memory snapshot.

    Historical states did not preserve insertion identity.  Migration starts a
    precise ownership epoch at each slot's authenticated last-use event.  This
    does not fabricate an earlier insertion time and insertion identity is not
    used by the eviction rule.
    """
    _validate_config(config)
    fields = _legacy_fields(legacy_state, name="prototype-memory")
    legacy_names = {"means", "counts", "last_update", "step_count"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy prototype-memory field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    slots = (config.n_classes, config.slots_per_class)
    means = _require_array_contract(
        fields["means"],
        name="legacy prototype-memory means",
        shape=(*slots, config.feature_dim),
        dtype=jnp.dtype(jnp.float32),
    )
    counts = _require_array_contract(
        fields["counts"],
        name="legacy prototype-memory counts",
        shape=slots,
        dtype=jnp.dtype(jnp.float32),
    )
    last = _require_array_contract(
        fields["last_update"],
        name="legacy prototype-memory last_update",
        shape=slots,
        dtype=jnp.dtype(jnp.int32),
    )
    step_count = _require_array_contract(
        fields["step_count"],
        name="legacy prototype-memory step_count",
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy prototype-memory step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy prototype-memory step_count is ambiguous")
    if not bool(jnp.all(jnp.isfinite(means))):
        raise ValueError("legacy prototype-memory means must be finite")
    if (
        not bool(jnp.all(jnp.isfinite(counts)))
        or bool(jnp.any(counts < 0.0))
        or bool(jnp.any(counts != jnp.floor(counts)))
        or bool(jnp.any(counts >= _FLOAT32_CONSECUTIVE_INTEGER_LIMIT))
    ):
        raise ValueError(
            "legacy prototype-memory counts are negative, fractional, or at "
            "float32's ambiguous consecutive-integer boundary"
        )
    used = counts > 0.0
    if (
        bool(jnp.any(last < 0))
        or bool(jnp.any(last >= _INT32_MAX))
        or bool(jnp.any(last > step))
        or bool(jnp.any(jnp.where(used, last <= 0, last != 0)))
    ):
        raise ValueError("legacy prototype-memory last-use telemetry is ambiguous")
    visit_words = jnp.zeros((*slots, 2), dtype=jnp.uint32)
    visit_words = visit_words.at[..., 1].set(counts.astype(jnp.uint32))
    last_words = jnp.zeros((*slots, 2), dtype=jnp.uint32)
    last_words = last_words.at[..., 1].set(last.astype(jnp.uint32))
    migrated = PrototypeMemoryState(
        means=means,
        counts=counts,
        visit_words=visit_words,
        last_update=last,
        last_update_words=last_words,
        insertion_step=last,
        insertion_words=last_words,
        step_count=step_count,
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )
    learner = PrototypeMemoryLearner(config)
    learner._require_state_contract(migrated)
    if not bool(jax.device_get(learner.state_is_valid(migrated))):
        raise ValueError("legacy prototype-memory state violates the v2 contract")
    return migrated


def save_prototype_memory_checkpoint(
    learner: PrototypeMemoryLearner,
    state: PrototypeMemoryState,
    path: str | Path,
) -> None:
    """Persist one authenticated v2 fixed-capacity memory state."""
    learner._require_state_contract(state)
    if not bool(jax.device_get(learner.state_is_valid(state))):
        raise ValueError("prototype-memory checkpoint state is invalid")
    budget = learner.resource_budget(state)
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA,
            "learner_config": learner.to_config(),
            "memory_accounting": budget.to_dict(),
        },
    )


def load_prototype_memory_checkpoint(
    path: str | Path,
) -> tuple[PrototypeMemoryLearner, PrototypeMemoryState]:
    """Restore only an authenticated exact-clock v2 checkpoint."""
    metadata = load_checkpoint_metadata(path)
    expected = {"schema", "learner_config", "memory_accounting"}
    if set(metadata) != expected:
        raise ValueError("prototype-memory checkpoint metadata fields are invalid")
    schema = metadata.get("schema")
    if schema == _LEGACY_PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy prototype-memory checkpoint v1 lacks exact identities; "
            "migrate its state and resave it"
        )
    if schema != PROTOTYPE_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError("prototype-memory checkpoint schema is unsupported")
    raw_config = metadata.get("learner_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("prototype-memory checkpoint learner_config is invalid")
    learner = PrototypeMemoryLearner.from_config(dict(raw_config))
    restored, restored_metadata = load_checkpoint(learner.init(), path)
    if restored_metadata != metadata:
        raise ValueError("prototype-memory checkpoint metadata changed between reads")
    state = cast(PrototypeMemoryState, restored)
    learner._require_state_contract(state)
    if not bool(jax.device_get(learner.state_is_valid(state))):
        raise ValueError("restored prototype-memory state is invalid")
    if learner.resource_budget(state).to_dict() != metadata.get("memory_accounting"):
        raise ValueError("prototype-memory checkpoint resource contract does not match")
    return learner, state
