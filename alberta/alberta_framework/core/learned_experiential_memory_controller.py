# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Bounded causal learning for experiential-memory admission and retention.

``ExperientialMemory`` deliberately implements a fixed retrieval rule and a
fixed utility/recency eviction rule.  This module is a separate v1 owner that
learns two narrower decisions from explicitly supplied, same-decision
counterfactual usefulness:

* a seven-feature linear admission model may reject (but never relax) a valid
  fixed-store retrieval; and
* the fixed store's non-negative utility channel is owned as a learned
  per-exemplar retention estimate, so it affects which row is evicted when the
  configured utility weight is positive.

The ordering is causal.  :meth:`step` queries the pre-write store, applies the
current detached admission model, records only an admitted access, and then
writes the current exemplar with a fixed retention prior.  An admitted query
creates one pending receipt.  :meth:`settle` is the only learning path and must
match that exact receipt.  It updates admission weights and still-live neighbor
rows only when the caller says the retrieval was used and supplies a bounded
counterfactual utility delta.  Slot numbers are not identities: retention
updates additionally match insertion clock, provenance, and source.

Counterfactual feedback is caller asserted and integrity-bound, not
authenticated.  This mechanism owns no action, learner, environment, safety,
evidence, or promotion authority and establishes no memory benefit.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryRetrieval,
    ExperientialMemoryState,
)

LEARNED_EXPERIENTIAL_MEMORY_CONFIG_SCHEMA = (
    "alberta.learned-experiential-memory-controller.config.v1"
)
LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA = (
    "alberta.learned-experiential-memory-controller.checkpoint.v1"
)
LEARNED_EXPERIENTIAL_MEMORY_STATE_SCHEMA = (
    "alberta.learned-experiential-memory-controller.state.v1"
)
LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS = "l0-mechanism-only-not-assessed"
LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT = 7
LEARNED_EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False

_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1
_CONFIG_DIGEST_WORDS = 8


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    converted = float(value)
    represented = float(np.float32(converted))
    if (
        not math.isfinite(converted)
        or not math.isfinite(represented)
        or converted < minimum
        or converted > maximum
    ):
        raise ValueError(f"{name} must remain in [{minimum}, {maximum}] in float32")
    return converted


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _config_digest_words(value: object) -> Array:
    raw = hashlib.sha256(_canonical_json_bytes(value)).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(raw[offset : offset + 4], "little")
            for offset in range(0, len(raw), 4)
        ),
        dtype=jnp.uint32,
    )


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(available, proposed, words), available


def _saturating_increment(value: Array, condition: Array | bool = True) -> Array:
    enabled = jnp.asarray(condition, dtype=jnp.bool_)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(enabled & (value < maximum), value + 1, value)


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class LearnedExperientialMemoryControllerConfig:
    """Fixed learning, clipping, and nested-store contract.

    The controller owns the nested store's utility channel.  Caller-supplied
    entry utility is replaced by ``retention_prior`` and marked available.
    ``ExperientialMemoryConfig.eviction_utility_weight`` must therefore be
    positive.  Its recency term remains active exactly as configured.
    """

    memory: ExperientialMemoryConfig
    admission_step_size: float = 0.05
    retention_step_size: float = 0.1
    admission_threshold: float = 0.0
    initial_admission_bias: float = 0.0
    max_abs_admission_weight: float = 8.0
    max_abs_counterfactual_delta: float = 1.0
    retention_prior: float = 0.5

    SCHEMA_VERSION: ClassVar[str] = LEARNED_EXPERIENTIAL_MEMORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.memory) is not ExperientialMemoryConfig:
            raise TypeError("memory must be an exact ExperientialMemoryConfig")
        ExperientialMemory(self.memory)
        if self.memory.eviction_utility_weight <= 0.0:
            raise ValueError("learned retention requires positive eviction_utility_weight")
        _finite_float32(
            self.admission_step_size,
            name="admission_step_size",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_float32(
            self.retention_step_size,
            name="retention_step_size",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_float32(
            self.max_abs_admission_weight,
            name="max_abs_admission_weight",
            minimum=float(np.finfo(np.float32).tiny),
            maximum=1.0e6,
        )
        bound = float(self.max_abs_admission_weight)
        _finite_float32(
            self.admission_threshold,
            name="admission_threshold",
            minimum=-bound,
            maximum=bound,
        )
        _finite_float32(
            self.initial_admission_bias,
            name="initial_admission_bias",
            minimum=-bound,
            maximum=bound,
        )
        _finite_float32(
            self.max_abs_counterfactual_delta,
            name="max_abs_counterfactual_delta",
            minimum=float(np.finfo(np.float32).tiny),
            maximum=1.0e12,
        )
        _finite_float32(
            self.retention_prior,
            name="retention_prior",
            minimum=0.0,
            maximum=1.0,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "type": type(self).__name__,
            "mechanism_status": LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS,
            "memory": self.memory.to_config(),
            "admission_step_size": self.admission_step_size,
            "retention_step_size": self.retention_step_size,
            "admission_threshold": self.admission_threshold,
            "initial_admission_bias": self.initial_admission_bias,
            "max_abs_admission_weight": self.max_abs_admission_weight,
            "max_abs_counterfactual_delta": self.max_abs_counterfactual_delta,
            "retention_prior": self.retention_prior,
            "fixed_store_gate_can_only_reject": True,
            "controller_owns_entry_utility_channel": True,
            "counterfactual_feedback_authenticated": False,
            "action_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        value: object,
    ) -> LearnedExperientialMemoryControllerConfig:
        if type(value) is not dict:
            raise ValueError("learned memory config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "memory",
            "admission_step_size",
            "retention_step_size",
            "admission_threshold",
            "initial_admission_bias",
            "max_abs_admission_weight",
            "max_abs_counterfactual_delta",
            "retention_prior",
            "fixed_store_gate_can_only_reject",
            "controller_owns_entry_utility_channel",
            "counterfactual_feedback_authenticated",
            "action_dispatch_authority",
            "safety_authority",
            "evidence_authority",
            "promotion_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("learned memory config fields differ from v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "type": cls.__name__,
            "mechanism_status": LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS,
            "fixed_store_gate_can_only_reject": True,
            "controller_owns_entry_utility_channel": True,
            "counterfactual_feedback_authenticated": False,
            "action_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }
        if any(
            type(raw.get(name)) is not type(item) or raw.get(name) != item
            for name, item in fixed.items()
        ):
            raise ValueError("learned memory fixed config fields are invalid")
        memory_raw = raw["memory"]
        if type(memory_raw) is not dict:
            raise ValueError("learned memory nested config must be an exact dict")
        for name in (
            "admission_step_size",
            "retention_step_size",
            "admission_threshold",
            "initial_admission_bias",
            "max_abs_admission_weight",
            "max_abs_counterfactual_delta",
            "retention_prior",
        ):
            if type(raw[name]) is not float:
                raise ValueError(f"{name} must be a JSON float")
        result = cls(
            memory=ExperientialMemoryConfig.from_config(cast(dict[str, Any], memory_raw)),
            admission_step_size=cast(float, raw["admission_step_size"]),
            retention_step_size=cast(float, raw["retention_step_size"]),
            admission_threshold=cast(float, raw["admission_threshold"]),
            initial_admission_bias=cast(float, raw["initial_admission_bias"]),
            max_abs_admission_weight=cast(float, raw["max_abs_admission_weight"]),
            max_abs_counterfactual_delta=cast(
                float, raw["max_abs_counterfactual_delta"]
            ),
            retention_prior=cast(float, raw["retention_prior"]),
        )
        if result.to_config() != raw:
            raise ValueError("learned memory config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryPendingState:
    """One admitted retrieval awaiting exact realized-use feedback."""

    available: Bool[Array, ""]
    transaction_words: UInt[Array, " 2"]
    admission_features: Float[Array, " 7"]
    admission_score: Float[Array, ""]
    neighbor_indices: Int[Array, " top_k"]
    neighbor_mask: Bool[Array, " top_k"]
    neighbor_weights: Float[Array, " top_k"]
    neighbor_insertion_step_words: UInt[Array, "top_k 2"]
    neighbor_provenance_ids: Int[Array, " top_k"]
    neighbor_source_ids: Int[Array, " top_k"]


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryControllerState:
    """Nested store, bounded learner, exact clock, and one pending owner."""

    memory: ExperientialMemoryState
    admission_weights: Float[Array, " 7"]
    transaction_words: UInt[Array, " 2"]
    feedback_count: Int[Array, ""]
    learned_feedback_count: Int[Array, ""]
    positive_feedback_count: Int[Array, ""]
    nonpositive_feedback_count: Int[Array, ""]
    pending: LearnedExperientialMemoryPendingState
    config_digest_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryFeedback:
    """Caller-asserted usefulness for one exact admitted retrieval.

    ``counterfactual_delta`` means realized utility with the retrieval minus
    utility under a same-decision fallback.  The core cannot authenticate that
    causal interpretation.  Learning requires both ``retrieval_used`` and
    ``counterfactual_available``; a matching receipt without either simply
    clears the pending transaction.
    """

    transaction_words: UInt[Array, " 2"]
    retrieval_used: Bool[Array, ""]
    counterfactual_available: Bool[Array, ""]
    counterfactual_delta: Float[Array, ""]


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryStepDiagnostics:
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    pending_blocked: Bool[Array, ""]
    fixed_store_retrieval_accepted: Bool[Array, ""]
    learned_admission_score: Float[Array, ""]
    learned_retrieval_admitted: Bool[Array, ""]
    write_succeeded: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    pending_created: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryStepResult:
    state: LearnedExperientialMemoryControllerState
    retrieval: ExperientialMemoryRetrieval
    fixed_store_retrieval: ExperientialMemoryRetrieval
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]
    diagnostics: LearnedExperientialMemoryStepDiagnostics


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryFeedbackDiagnostics:
    source_state_valid: Bool[Array, ""]
    pending_available: Bool[Array, ""]
    receipt_matches: Bool[Array, ""]
    feedback_valid: Bool[Array, ""]
    learning_eligible: Bool[Array, ""]
    admission_updated: Bool[Array, ""]
    retention_rows_updated: Int[Array, ""]
    transaction_applied: Bool[Array, ""]
    counterfactual_feedback_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LearnedExperientialMemoryFeedbackResult:
    state: LearnedExperientialMemoryControllerState
    diagnostics: LearnedExperientialMemoryFeedbackDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class LearnedExperientialMemoryResourceBudget:
    memory_capacity: int
    top_k: int
    admission_feature_count: int
    admission_trainable_float32_scalars: int
    owned_persistent_state_bytes: int
    nested_memory_persistent_state_bytes: int
    maximum_memory_queries_per_step: int
    maximum_memory_writes_per_step: int
    maximum_admission_updates_per_feedback: int
    maximum_retention_updates_per_feedback: int
    random_draws_per_step: int
    caller_counterfactual_feedback_authenticated: bool
    action_dispatch_authority: bool
    safety_authority: bool
    evidence_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class LearnedExperientialMemoryController:
    """One atomic owner for fixed-store state and learned memory decisions."""

    def __init__(self, config: LearnedExperientialMemoryControllerConfig):
        if type(config) is not LearnedExperientialMemoryControllerConfig:
            raise TypeError(
                "config must be an exact LearnedExperientialMemoryControllerConfig"
            )
        self._config = config
        self._memory = ExperientialMemory(config.memory)
        self._config_digest = _config_digest_words(config.to_config())

    @property
    def config(self) -> LearnedExperientialMemoryControllerConfig:
        return self._config

    @property
    def memory(self) -> ExperientialMemory:
        return self._memory

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> LearnedExperientialMemoryController:
        return cls(LearnedExperientialMemoryControllerConfig.from_config(value))

    def _empty_pending(self) -> LearnedExperientialMemoryPendingState:
        top_k = self._config.memory.top_k
        return LearnedExperientialMemoryPendingState(
            available=jnp.asarray(False, dtype=jnp.bool_),
            transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            admission_features=jnp.zeros(
                (LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,), dtype=jnp.float32
            ),
            admission_score=jnp.asarray(0.0, dtype=jnp.float32),
            neighbor_indices=jnp.full((top_k,), -1, dtype=jnp.int32),
            neighbor_mask=jnp.zeros((top_k,), dtype=jnp.bool_),
            neighbor_weights=jnp.zeros((top_k,), dtype=jnp.float32),
            neighbor_insertion_step_words=jnp.zeros((top_k, 2), dtype=jnp.uint32),
            neighbor_provenance_ids=jnp.full((top_k,), -1, dtype=jnp.int32),
            neighbor_source_ids=jnp.full((top_k,), -1, dtype=jnp.int32),
        )

    def init(self) -> LearnedExperientialMemoryControllerState:
        weights = jnp.zeros(
            (LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,), dtype=jnp.float32
        ).at[0].set(
            jnp.asarray(self._config.initial_admission_bias, dtype=jnp.float32)
        )
        state = LearnedExperientialMemoryControllerState(
            memory=self._memory.init(),
            admission_weights=weights,
            transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            feedback_count=jnp.asarray(0, dtype=jnp.int32),
            learned_feedback_count=jnp.asarray(0, dtype=jnp.int32),
            positive_feedback_count=jnp.asarray(0, dtype=jnp.int32),
            nonpositive_feedback_count=jnp.asarray(0, dtype=jnp.int32),
            pending=self._empty_pending(),
            config_digest_words=self._config_digest,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise RuntimeError("initial learned memory state is invalid")
        return state

    def _validate_pending_static(
        self, pending: LearnedExperientialMemoryPendingState
    ) -> None:
        if not isinstance(pending, LearnedExperientialMemoryPendingState):
            raise TypeError("state.pending must be LearnedExperientialMemoryPendingState")
        top_k = self._config.memory.top_k
        for name, shape, dtype in (
            ("available", (), jnp.bool_),
            ("transaction_words", (2,), jnp.uint32),
            (
                "admission_features",
                (LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,),
                jnp.float32,
            ),
            ("admission_score", (), jnp.float32),
            ("neighbor_indices", (top_k,), jnp.int32),
            ("neighbor_mask", (top_k,), jnp.bool_),
            ("neighbor_weights", (top_k,), jnp.float32),
            ("neighbor_insertion_step_words", (top_k, 2), jnp.uint32),
            ("neighbor_provenance_ids", (top_k,), jnp.int32),
            ("neighbor_source_ids", (top_k,), jnp.int32),
        ):
            _require_array(
                getattr(pending, name),
                name=f"state.pending.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _validate_state_static(
        self, state: LearnedExperientialMemoryControllerState
    ) -> None:
        if not isinstance(state, LearnedExperientialMemoryControllerState):
            raise TypeError("state must be LearnedExperientialMemoryControllerState")
        self._memory._validate_state_static_contract(state.memory)
        self._validate_pending_static(state.pending)
        _require_array(
            state.admission_weights,
            name="state.admission_weights",
            shape=(LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,),
            dtype=jnp.float32,
        )
        _require_array(
            state.transaction_words,
            name="state.transaction_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        for name in (
            "feedback_count",
            "learned_feedback_count",
            "positive_feedback_count",
            "nonpositive_feedback_count",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            state.config_digest_words,
            name="state.config_digest_words",
            shape=(_CONFIG_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )

    def _pending_valid(self, pending: LearnedExperientialMemoryPendingState) -> Array:
        cfg = self._config.memory
        finite = jnp.all(jnp.isfinite(pending.admission_features)) & jnp.isfinite(
            pending.admission_score
        ) & jnp.all(jnp.isfinite(pending.neighbor_weights))
        indices_valid = jnp.all(
            (~pending.neighbor_mask)
            | ((pending.neighbor_indices >= 0) & (pending.neighbor_indices < cfg.capacity))
        )
        weights_valid = (
            jnp.all(pending.neighbor_weights >= 0.0)
            & jnp.all(
                jnp.where(
                    pending.neighbor_mask,
                    jnp.asarray(True),
                    pending.neighbor_weights == 0.0,
                )
            )
            & jnp.isclose(
                jnp.sum(pending.neighbor_weights),
                jnp.where(pending.available, 1.0, 0.0),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )
        active_metadata = jnp.all(
            (~pending.neighbor_mask)
            | (
                (pending.neighbor_provenance_ids >= 0)
                & (pending.neighbor_source_ids >= 0)
            )
        )
        inactive_exact = (
            ~pending.available
            & jnp.all(pending.transaction_words == 0)
            & jnp.all(pending.admission_features == 0.0)
            & (pending.admission_score == 0.0)
            & jnp.all(pending.neighbor_indices == -1)
            & ~jnp.any(pending.neighbor_mask)
            & jnp.all(pending.neighbor_weights == 0.0)
            & jnp.all(pending.neighbor_insertion_step_words == 0)
            & jnp.all(pending.neighbor_provenance_ids == -1)
            & jnp.all(pending.neighbor_source_ids == -1)
        )
        active_valid = (
            pending.available
            & jnp.any(pending.transaction_words != 0)
            & (
                jnp.sum(pending.neighbor_mask.astype(jnp.int32))
                >= cfg.min_neighbors
            )
            & jnp.all(
                jnp.where(
                    pending.neighbor_mask,
                    jnp.asarray(True),
                    pending.neighbor_indices == -1,
                )
            )
            & jnp.all(
                jnp.where(
                    pending.neighbor_mask[:, None],
                    jnp.asarray(True),
                    pending.neighbor_insertion_step_words == 0,
                )
            )
            & jnp.all(
                jnp.where(
                    pending.neighbor_mask,
                    jnp.asarray(True),
                    pending.neighbor_provenance_ids == -1,
                )
            )
            & jnp.all(
                jnp.where(
                    pending.neighbor_mask,
                    jnp.asarray(True),
                    pending.neighbor_source_ids == -1,
                )
            )
        )
        return finite & indices_valid & weights_valid & active_metadata & (
            inactive_exact | active_valid
        )

    def _state_is_valid(
        self, state: LearnedExperientialMemoryControllerState
    ) -> Array:
        weights_valid = (
            jnp.all(jnp.isfinite(state.admission_weights))
            & jnp.all(
                jnp.abs(state.admission_weights)
                <= jnp.asarray(
                    self._config.max_abs_admission_weight, dtype=jnp.float32
                )
            )
        )
        counters_valid = (
            (state.feedback_count >= 0)
            & (state.learned_feedback_count >= 0)
            & (state.positive_feedback_count >= 0)
            & (state.nonpositive_feedback_count >= 0)
            & (state.learned_feedback_count <= state.feedback_count)
            & (state.positive_feedback_count <= state.learned_feedback_count)
            & (state.nonpositive_feedback_count <= state.learned_feedback_count)
            & (
                state.positive_feedback_count
                == state.learned_feedback_count - state.nonpositive_feedback_count
            )
        )
        pending_clock_valid = (~state.pending.available) | jnp.array_equal(
            state.pending.transaction_words, state.transaction_words
        )
        return (
            self._memory._state_is_valid(state.memory)
            & weights_valid
            & counters_valid
            & self._pending_valid(state.pending)
            & pending_clock_valid
            & jnp.array_equal(state.transaction_words, state.memory.step_words)
            & jnp.array_equal(state.config_digest_words, self._config_digest)
        )

    def state_valid(
        self, state: LearnedExperientialMemoryControllerState
    ) -> Bool[Array, ""]:
        self._validate_state_static(state)
        return cast(Bool[Array, ""], self._state_valid_jit(state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_valid_jit(
        self, state: LearnedExperientialMemoryControllerState
    ) -> Array:
        return self._state_is_valid(state)

    def _blank_retrieval(self) -> ExperientialMemoryRetrieval:
        cfg = self._config.memory
        return ExperientialMemoryRetrieval(
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            observation=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
            action=jnp.zeros((cfg.action_dim,), dtype=jnp.float32),
            outcome=jnp.zeros((cfg.outcome_dim,), dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            effective_reliability=jnp.asarray(0.0, dtype=jnp.float32),
            neighbor_indices=jnp.zeros((cfg.top_k,), dtype=jnp.int32),
            neighbor_mask=jnp.zeros((cfg.top_k,), dtype=jnp.bool_),
            neighbor_weights=jnp.zeros((cfg.top_k,), dtype=jnp.float32),
            neighbor_similarities=jnp.zeros((cfg.top_k,), dtype=jnp.float32),
            neighbor_reliabilities=jnp.zeros((cfg.top_k,), dtype=jnp.float32),
            neighbor_ages=jnp.zeros((cfg.top_k,), dtype=jnp.int32),
            neighbor_provenance_ids=jnp.full((cfg.top_k,), -1, dtype=jnp.int32),
            state_valid=jnp.asarray(False, dtype=jnp.bool_),
            query_valid=jnp.asarray(False, dtype=jnp.bool_),
            version_compatible=jnp.asarray(False, dtype=jnp.bool_),
            freshness_ok=jnp.asarray(False, dtype=jnp.bool_),
            uncertainty_available=jnp.asarray(False, dtype=jnp.bool_),
            safety_cost_available=jnp.asarray(False, dtype=jnp.bool_),
            uncertainty_ok=jnp.asarray(False, dtype=jnp.bool_),
            safety_ok=jnp.asarray(False, dtype=jnp.bool_),
            has_neighbors=jnp.asarray(False, dtype=jnp.bool_),
        )

    @staticmethod
    def _gate_retrieval(
        retrieval: ExperientialMemoryRetrieval, admitted: Array
    ) -> ExperientialMemoryRetrieval:
        def gated(value: Array) -> Array:
            return jnp.where(admitted, value, jnp.zeros_like(value))

        return dataclasses.replace(
            retrieval,
            accepted=admitted,
            observation=gated(retrieval.observation),
            action=gated(retrieval.action),
            outcome=gated(retrieval.outcome),
            reward=gated(retrieval.reward),
            uncertainty=gated(retrieval.uncertainty),
            safety_cost=gated(retrieval.safety_cost),
            effective_reliability=gated(retrieval.effective_reliability),
        )

    def _admission_features(
        self, retrieval: ExperientialMemoryRetrieval
    ) -> Array:
        cfg = self._config.memory
        weights = retrieval.neighbor_weights
        mean_similarity = jnp.sum(weights * retrieval.neighbor_similarities)
        reliability = retrieval.effective_reliability
        uncertainty_quality = jnp.clip(
            1.0
            - retrieval.uncertainty
            / jnp.maximum(
                jnp.asarray(cfg.max_uncertainty, dtype=jnp.float32),
                jnp.asarray(1.0e-6, dtype=jnp.float32),
            ),
            0.0,
            1.0,
        )
        safety_quality = jnp.clip(
            1.0
            - retrieval.safety_cost
            / jnp.maximum(
                jnp.asarray(cfg.max_safety_cost, dtype=jnp.float32),
                jnp.asarray(1.0e-6, dtype=jnp.float32),
            ),
            0.0,
            1.0,
        )
        support = jnp.sum(retrieval.neighbor_mask.astype(jnp.float32)) / float(
            cfg.top_k
        )
        ages = jnp.maximum(retrieval.neighbor_ages, 0).astype(jnp.float32)
        recency = jnp.sum(
            weights
            / (
                1.0
                + ages
                / jnp.asarray(cfg.recency_scale, dtype=jnp.float32)
            )
        )
        features = jnp.stack(
            (
                jnp.asarray(1.0, dtype=jnp.float32),
                mean_similarity,
                reliability,
                uncertainty_quality,
                safety_quality,
                support,
                recency,
            )
        ).astype(jnp.float32)
        return jax.lax.stop_gradient(
            jnp.where(retrieval.accepted, features, jnp.zeros_like(features))
        )

    def _controlled_entry(self, entry: ExperientialMemoryEntry) -> ExperientialMemoryEntry:
        return dataclasses.replace(
            entry,
            utility=jnp.asarray(self._config.retention_prior, dtype=jnp.float32),
            utility_available=jnp.asarray(True, dtype=jnp.bool_),
        )

    def _make_pending(
        self,
        source_memory: ExperientialMemoryState,
        retrieval: ExperientialMemoryRetrieval,
        transaction_words: Array,
        features: Array,
        score: Array,
    ) -> LearnedExperientialMemoryPendingState:
        indices = retrieval.neighbor_indices
        mask = retrieval.neighbor_mask
        safe_indices = jnp.where(mask, indices, 0)
        entries = source_memory.entries
        return LearnedExperientialMemoryPendingState(
            available=jnp.asarray(True, dtype=jnp.bool_),
            transaction_words=transaction_words,
            admission_features=features,
            admission_score=score,
            neighbor_indices=jnp.where(mask, indices, -1).astype(jnp.int32),
            neighbor_mask=mask,
            neighbor_weights=jnp.where(mask, retrieval.neighbor_weights, 0.0),
            neighbor_insertion_step_words=jnp.where(
                mask[:, None],
                entries.insertion_step_words[safe_indices],
                jnp.zeros((self._config.memory.top_k, 2), dtype=jnp.uint32),
            ),
            neighbor_provenance_ids=jnp.where(
                mask, entries.provenance_ids[safe_indices], -1
            ).astype(jnp.int32),
            neighbor_source_ids=jnp.where(
                mask, entries.source_ids[safe_indices], -1
            ).astype(jnp.int32),
        )

    def _validate_step_inputs(
        self,
        state: LearnedExperientialMemoryControllerState,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        entry: ExperientialMemoryEntry,
    ) -> None:
        self._validate_state_static(state)
        self._memory._validate_entry_static_contract(entry)
        cfg = self._config.memory
        _require_array(
            query_key,
            name="query_key",
            shape=(cfg.key_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            representation_version,
            name="representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )

    def step(
        self,
        state: LearnedExperientialMemoryControllerState,
        query_key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
        entry: ExperientialMemoryEntry,
    ) -> LearnedExperientialMemoryStepResult:
        """Query, learned-gate, record admitted access, and write atomically."""

        self._validate_step_inputs(
            state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
            entry,
        )
        return cast(
            LearnedExperientialMemoryStepResult,
            self._step_jit(
                state,
                query_key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
                entry,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: LearnedExperientialMemoryControllerState,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        entry: ExperientialMemoryEntry,
    ) -> LearnedExperientialMemoryStepResult:
        source_valid = self._state_is_valid(state)
        # Query validity controls only retrieval, exactly as in the fixed
        # store.  A missing/non-finite query estimate may abstain while a valid
        # current exemplar is still written; learned admission must not gain a
        # second write-veto authority.
        input_valid = self._memory._entry_is_valid(self._controlled_entry(entry))
        blocked = state.pending.available
        next_words, clock_available = _checked_words_increment(state.transaction_words)

        base_retrieval = self._memory._query_jit(
            state.memory,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
        )
        features = self._admission_features(base_retrieval)
        score = jnp.dot(state.admission_weights, features)
        admitted = (
            base_retrieval.accepted
            & jnp.isfinite(score)
            & (
                score
                >= jnp.asarray(self._config.admission_threshold, dtype=jnp.float32)
            )
        )
        gated_retrieval = self._gate_retrieval(base_retrieval, admitted)

        def apply(_: None) -> LearnedExperientialMemoryStepResult:
            advanced = self._memory._advance(state.memory)
            accessed = self._memory._record_query(advanced, gated_retrieval)
            write = self._memory._write_advanced(
                accessed, self._controlled_entry(entry)
            )
            pending = jax.lax.cond(
                admitted,
                lambda: self._make_pending(
                    state.memory,
                    gated_retrieval,
                    next_words,
                    features,
                    score,
                ),
                self._empty_pending,
            )
            candidate = LearnedExperientialMemoryControllerState(
                memory=write.state,
                admission_weights=state.admission_weights,
                transaction_words=next_words,
                feedback_count=state.feedback_count,
                learned_feedback_count=state.learned_feedback_count,
                positive_feedback_count=state.positive_feedback_count,
                nonpositive_feedback_count=state.nonpositive_feedback_count,
                pending=pending,
                config_digest_words=state.config_digest_words,
            )
            accepted = self._state_is_valid(candidate)
            final_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(accepted, new, old), candidate, state
            )
            exposed_retrieval = jax.tree_util.tree_map(
                lambda new, old: jnp.where(accepted, new, old),
                gated_retrieval,
                self._blank_retrieval(),
            )
            exposed_base = jax.tree_util.tree_map(
                lambda new, old: jnp.where(accepted, new, old),
                base_retrieval,
                self._blank_retrieval(),
            )
            return LearnedExperientialMemoryStepResult(
                state=final_state,
                retrieval=exposed_retrieval,
                fixed_store_retrieval=exposed_base,
                wrote=write.wrote & accepted,
                slot=jnp.where(accepted, write.slot, -1).astype(jnp.int32),
                evicted=write.evicted & accepted,
                evicted_provenance_id=jnp.where(
                    accepted, write.evicted_provenance_id, -1
                ).astype(jnp.int32),
                diagnostics=LearnedExperientialMemoryStepDiagnostics(
                    source_state_valid=source_valid,
                    input_valid=input_valid,
                    pending_blocked=blocked,
                    fixed_store_retrieval_accepted=base_retrieval.accepted & accepted,
                    learned_admission_score=jnp.where(accepted, score, 0.0),
                    learned_retrieval_admitted=admitted & accepted,
                    write_succeeded=write.wrote & accepted,
                    transaction_applied=accepted,
                    pending_created=admitted & accepted,
                ),
            )

        def reject(_: None) -> LearnedExperientialMemoryStepResult:
            blank = self._blank_retrieval()
            return LearnedExperientialMemoryStepResult(
                state=state,
                retrieval=blank,
                fixed_store_retrieval=blank,
                wrote=jnp.asarray(False, dtype=jnp.bool_),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False, dtype=jnp.bool_),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
                diagnostics=LearnedExperientialMemoryStepDiagnostics(
                    source_state_valid=source_valid,
                    input_valid=input_valid,
                    pending_blocked=blocked,
                    fixed_store_retrieval_accepted=jnp.asarray(False),
                    learned_admission_score=jnp.asarray(0.0, dtype=jnp.float32),
                    learned_retrieval_admitted=jnp.asarray(False),
                    write_succeeded=jnp.asarray(False),
                    transaction_applied=jnp.asarray(False),
                    pending_created=jnp.asarray(False),
                ),
            )

        return cast(
            LearnedExperientialMemoryStepResult,
            jax.lax.cond(
                source_valid & input_valid & (~blocked) & clock_available,
                apply,
                reject,
                operand=None,
            ),
        )

    def _validate_feedback_static(
        self,
        state: LearnedExperientialMemoryControllerState,
        feedback: LearnedExperientialMemoryFeedback,
    ) -> None:
        self._validate_state_static(state)
        if not isinstance(feedback, LearnedExperientialMemoryFeedback):
            raise TypeError("feedback must be LearnedExperientialMemoryFeedback")
        for name, shape, dtype in (
            ("transaction_words", (2,), jnp.uint32),
            ("retrieval_used", (), jnp.bool_),
            ("counterfactual_available", (), jnp.bool_),
            ("counterfactual_delta", (), jnp.float32),
        ):
            _require_array(
                getattr(feedback, name),
                name=f"feedback.{name}",
                shape=shape,
                dtype=dtype,
            )

    def settle(
        self,
        state: LearnedExperientialMemoryControllerState,
        feedback: LearnedExperientialMemoryFeedback,
    ) -> LearnedExperientialMemoryFeedbackResult:
        """Settle one exact pending receipt and optionally learn from it."""

        self._validate_feedback_static(state, feedback)
        return cast(
            LearnedExperientialMemoryFeedbackResult,
            self._settle_jit(state, feedback),
        )

    def _updated_memory_utilities(
        self,
        memory_state: ExperientialMemoryState,
        pending: LearnedExperientialMemoryPendingState,
        retention_target: Array,
        learning_eligible: Array,
    ) -> tuple[ExperientialMemoryState, Array]:
        entries = memory_state.entries
        utilities = entries.utilities
        availability = entries.utility_available
        update_count = jnp.asarray(0, dtype=jnp.int32)
        step_size = jnp.asarray(self._config.retention_step_size, dtype=jnp.float32)
        for position in range(self._config.memory.top_k):
            slot = jnp.maximum(pending.neighbor_indices[position], 0)
            identity_matches = (
                learning_eligible
                & pending.neighbor_mask[position]
                & entries.valid[slot]
                & jnp.array_equal(
                    entries.insertion_step_words[slot],
                    pending.neighbor_insertion_step_words[position],
                )
                & (
                    entries.provenance_ids[slot]
                    == pending.neighbor_provenance_ids[position]
                )
                & (entries.source_ids[slot] == pending.neighbor_source_ids[position])
            )
            old = utilities[slot]
            proposed = jnp.clip(
                old
                + step_size
                * pending.neighbor_weights[position]
                * (retention_target - old),
                0.0,
                1.0,
            )
            utilities = utilities.at[slot].set(
                jnp.where(identity_matches, proposed, old)
            )
            availability = availability.at[slot].set(
                jnp.where(identity_matches, True, availability[slot])
            )
            update_count = _saturating_increment(update_count, identity_matches)
        updated_entries = dataclasses.replace(
            entries,
            utilities=utilities,
            utility_available=availability,
        )
        return dataclasses.replace(memory_state, entries=updated_entries), update_count

    @functools.partial(jax.jit, static_argnums=(0,))
    def _settle_jit(
        self,
        state: LearnedExperientialMemoryControllerState,
        feedback: LearnedExperientialMemoryFeedback,
    ) -> LearnedExperientialMemoryFeedbackResult:
        source_valid = self._state_is_valid(state)
        receipt_matches = state.pending.available & jnp.array_equal(
            feedback.transaction_words, state.pending.transaction_words
        )
        feedback_valid = (
            jnp.isfinite(feedback.counterfactual_delta)
            & (
                jnp.abs(feedback.counterfactual_delta)
                <= jnp.asarray(
                    self._config.max_abs_counterfactual_delta,
                    dtype=jnp.float32,
                )
            )
        )
        applies = source_valid & receipt_matches & feedback_valid
        learning_eligible = (
            applies & feedback.retrieval_used & feedback.counterfactual_available
        )
        normalized_target = jnp.clip(
            feedback.counterfactual_delta
            / jnp.asarray(
                self._config.max_abs_counterfactual_delta, dtype=jnp.float32
            ),
            -1.0,
            1.0,
        )
        prediction = jnp.tanh(state.pending.admission_score)
        derivative = 1.0 - prediction * prediction
        gradient = (
            (normalized_target - prediction)
            * derivative
            * state.pending.admission_features
        )
        proposed_weights = jnp.clip(
            state.admission_weights
            + jnp.asarray(self._config.admission_step_size, dtype=jnp.float32)
            * gradient,
            -jnp.asarray(
                self._config.max_abs_admission_weight, dtype=jnp.float32
            ),
            jnp.asarray(self._config.max_abs_admission_weight, dtype=jnp.float32),
        )
        admission_weights = jnp.where(
            learning_eligible, proposed_weights, state.admission_weights
        )
        retention_target = jnp.clip(0.5 + 0.5 * normalized_target, 0.0, 1.0)
        memory, retention_updates = self._updated_memory_utilities(
            state.memory,
            state.pending,
            retention_target,
            learning_eligible,
        )
        candidate = LearnedExperientialMemoryControllerState(
            memory=memory,
            admission_weights=admission_weights,
            transaction_words=state.transaction_words,
            feedback_count=_saturating_increment(state.feedback_count, applies),
            learned_feedback_count=_saturating_increment(
                state.learned_feedback_count, learning_eligible
            ),
            positive_feedback_count=_saturating_increment(
                state.positive_feedback_count,
                learning_eligible & (feedback.counterfactual_delta > 0.0),
            ),
            nonpositive_feedback_count=_saturating_increment(
                state.nonpositive_feedback_count,
                learning_eligible & (feedback.counterfactual_delta <= 0.0),
            ),
            pending=self._empty_pending(),
            config_digest_words=state.config_digest_words,
        )
        candidate_valid = self._state_is_valid(candidate)
        committed = applies & candidate_valid
        final_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(committed, new, old), candidate, state
        )
        return LearnedExperientialMemoryFeedbackResult(
            state=final_state,
            diagnostics=LearnedExperientialMemoryFeedbackDiagnostics(
                source_state_valid=source_valid,
                pending_available=state.pending.available,
                receipt_matches=receipt_matches,
                feedback_valid=feedback_valid,
                learning_eligible=learning_eligible & committed,
                admission_updated=learning_eligible & committed,
                retention_rows_updated=jnp.where(
                    committed, retention_updates, 0
                ).astype(jnp.int32),
                transaction_applied=committed,
                counterfactual_feedback_authenticated=jnp.asarray(
                    False, dtype=jnp.bool_
                ),
            ),
        )

    def resource_budget(
        self,
        state: LearnedExperientialMemoryControllerState | None = None,
    ) -> LearnedExperientialMemoryResourceBudget:
        checked = self.init() if state is None else state
        self._validate_state_static(checked)
        if not bool(jax.device_get(self._state_is_valid(checked))):
            raise ValueError("resource measurement requires a valid controller state")
        total = _tree_nbytes(checked)
        nested = self._memory.resource_budget(checked.memory).persistent_state_bytes
        return LearnedExperientialMemoryResourceBudget(
            memory_capacity=self._config.memory.capacity,
            top_k=self._config.memory.top_k,
            admission_feature_count=LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,
            admission_trainable_float32_scalars=(
                LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT
            ),
            owned_persistent_state_bytes=total,
            nested_memory_persistent_state_bytes=nested,
            maximum_memory_queries_per_step=1,
            maximum_memory_writes_per_step=1,
            maximum_admission_updates_per_feedback=1,
            maximum_retention_updates_per_feedback=self._config.memory.top_k,
            random_draws_per_step=0,
            caller_counterfactual_feedback_authenticated=False,
            action_dispatch_authority=False,
            safety_authority=False,
            evidence_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
        )


def save_learned_experiential_memory_checkpoint(
    controller: LearnedExperientialMemoryController,
    state: LearnedExperientialMemoryControllerState,
    path: str | Path,
) -> None:
    """Persist one exact v1 owner state with strict nested construction."""

    if type(controller) is not LearnedExperientialMemoryController:
        raise TypeError("controller must be LearnedExperientialMemoryController")
    controller._validate_state_static(state)
    if not bool(jax.device_get(controller._state_is_valid(state))):
        raise ValueError("learned experiential-memory state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA,
            "state_schema": LEARNED_EXPERIENTIAL_MEMORY_STATE_SCHEMA,
            "mechanism_status": LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "controller": controller.to_config(),
            "resource_budget": controller.resource_budget(state).to_config(),
        },
    )


def load_learned_experiential_memory_checkpoint(
    path: str | Path,
) -> tuple[
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerState,
]:
    """Restore only an exact config/resource-bound v1 owner state."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "state_schema",
        "mechanism_status",
        "scientific_promotion_allowed",
        "controller",
        "resource_budget",
    }
    if set(metadata) != expected:
        raise ValueError("learned memory checkpoint fields differ from v1")
    if metadata["schema"] != LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError("learned memory checkpoint schema is unsupported")
    if metadata["state_schema"] != LEARNED_EXPERIENTIAL_MEMORY_STATE_SCHEMA:
        raise ValueError("learned memory state schema is unsupported")
    if metadata["mechanism_status"] != LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS:
        raise ValueError("learned memory checkpoint mechanism status differs")
    if metadata["scientific_promotion_allowed"] is not False:
        raise ValueError("learned memory checkpoint cannot claim promotion")
    controller = LearnedExperientialMemoryController.from_config(
        metadata["controller"]
    )
    restored, second_metadata = load_checkpoint(controller.init(), path)
    if second_metadata != metadata:
        raise ValueError("learned memory checkpoint metadata changed between reads")
    state = cast(LearnedExperientialMemoryControllerState, restored)
    controller._validate_state_static(state)
    if not bool(jax.device_get(controller._state_is_valid(state))):
        raise ValueError("learned memory checkpoint state is invalid")
    if metadata["resource_budget"] != controller.resource_budget(state).to_config():
        raise ValueError("learned memory checkpoint resource contract changed")
    return controller, state


__all__ = [
    "LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA",
    "LEARNED_EXPERIENTIAL_MEMORY_CONFIG_SCHEMA",
    "LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT",
    "LEARNED_EXPERIENTIAL_MEMORY_MECHANISM_STATUS",
    "LEARNED_EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "LEARNED_EXPERIENTIAL_MEMORY_STATE_SCHEMA",
    "LearnedExperientialMemoryController",
    "LearnedExperientialMemoryControllerConfig",
    "LearnedExperientialMemoryControllerState",
    "LearnedExperientialMemoryFeedback",
    "LearnedExperientialMemoryFeedbackDiagnostics",
    "LearnedExperientialMemoryFeedbackResult",
    "LearnedExperientialMemoryPendingState",
    "LearnedExperientialMemoryResourceBudget",
    "LearnedExperientialMemoryStepDiagnostics",
    "LearnedExperientialMemoryStepResult",
    "load_learned_experiential_memory_checkpoint",
    "save_learned_experiential_memory_checkpoint",
]
