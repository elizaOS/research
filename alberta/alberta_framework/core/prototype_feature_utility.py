# mypy: disable-error-code="attr-defined,call-arg"
"""Diagnostic-only causal utility audit for Prototype pair features.

The auditor measures two one-step, frozen-consumer counterfactuals.  For an
active feature it asks how much squared prediction loss would change if the
feature's current linear contribution were deleted.  For a candidate it asks
how much loss would change if a separately learned shadow contribution were
inserted.  Scores are tracked per consumer task and aggregated with one half
of the mass assigned to control and one half shared across the ordered Horde
demons.

This is bounded L0 instrumentation.  Its private shadow probe is audit state;
it performs no downstream consumer update, router call, curation decision,
threshold, evidence claim, or promotion.  In particular, the returned scores
must not be used to mutate a live feature bank without a separately reviewed
causal routing transaction.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA = (
    "alberta.prototype-feature-utility.config.v2"
)
PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA = (
    "alberta.prototype-feature-utility.state.v2"
)
_LEGACY_PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA = (
    "alberta.prototype-feature-utility.config.v1"
)
PROTOTYPE_FEATURE_UTILITY_MECHANISM_STATUS = (
    "L0_DIAGNOSTIC_ONLY_NO_CURATION_OR_EVIDENCE_AUTHORITY"
)
PROTOTYPE_FEATURE_UTILITY_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY = False

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
PROTOTYPE_FEATURE_UTILITY_TELEMETRY_COUNTER_NBYTES = 4
PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_NBYTES = 12
PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_DELTA_NBYTES = 8
PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES = 24
PROTOTYPE_FEATURE_UTILITY_COUNTER_DELTA_NBYTES = 16
_MAX_TOTAL_FEATURE_DIM = 4_096
_MAX_PAIR_SLOTS = 262_144
_MAX_TASKS = 4_096
_MAX_BOUNDED_CELLS = 4_194_304


def _strict_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact Python int")
    return value


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact Python int")
    return value


def _strict_unit_interval(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be a finite exact float in [0, 1)")
    return value


def _strict_positive_float32(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive exact Python float")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented) or represented <= 0.0:
        raise ValueError(f"{name} must be finite and positive in float32")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureUtilityConfig:
    """Static dimensions and update rates for the bounded utility auditor."""

    base_feature_dim: int
    active_pair_slots: int
    candidate_pair_slots: int
    managed_horde_demons: int
    utility_decay: float = 0.99
    shadow_step_size: float = 0.01
    second_moment_decay: float = 0.99
    scale_epsilon: float = 1.0e-6
    # Optional logical budget. The default spans the complete two-word
    # lifetime; the all-ones value is terminal and is never wrapped.
    max_observations: int = _UINT64_MAX

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        base = _strict_positive_int(self.base_feature_dim, name="base_feature_dim")
        if base < 2 or base > _MAX_TOTAL_FEATURE_DIM:
            raise ValueError("base_feature_dim must be in [2, 4096]")
        active = _strict_positive_int(
            self.active_pair_slots,
            name="active_pair_slots",
        )
        candidates = _strict_nonnegative_int(
            self.candidate_pair_slots,
            name="candidate_pair_slots",
        )
        demons = _strict_positive_int(
            self.managed_horde_demons,
            name="managed_horde_demons",
        )
        if active > _MAX_PAIR_SLOTS or candidates > _MAX_PAIR_SLOTS:
            raise ValueError("pair slots exceed the static 262144-slot ceiling")
        tasks = 1 + demons
        if tasks > _MAX_TASKS:
            raise ValueError("n_tasks exceeds the static 4096-task ceiling")
        if base + active > _MAX_TOTAL_FEATURE_DIM:
            raise ValueError("total_feature_dim exceeds the static 4096-width ceiling")
        pair_space = base * (base - 1) // 2
        if active > pair_space or candidates > pair_space:
            raise ValueError("pair slots exceed the canonical pair space")
        _strict_unit_interval(self.utility_decay, name="utility_decay")
        shadow_step_size = _strict_positive_float32(
            self.shadow_step_size,
            name="shadow_step_size",
        )
        if shadow_step_size > 1.0:
            raise ValueError("shadow_step_size must be at most 1.0")
        _strict_unit_interval(
            self.second_moment_decay,
            name="second_moment_decay",
        )
        _strict_positive_float32(self.scale_epsilon, name="scale_epsilon")
        maximum = _strict_positive_int(
            self.max_observations,
            name="max_observations",
        )
        if maximum > _UINT64_MAX:
            raise ValueError("max_observations must fit an unsigned 64-bit identity")

        task_feature_cells = tasks * (active + candidates)
        if task_feature_cells > _MAX_BOUNDED_CELLS:
            raise ValueError("task-feature cells exceed the static safety ceiling")
        persistent_logical_scalars = (
            6
            + 2 * active
            + 2 * tasks * active
            + 3 * candidates
            + 3 * tasks * candidates
            + tasks
        )
        if persistent_logical_scalars > _MAX_BOUNDED_CELLS:
            raise ValueError("persistent state exceeds the static safety ceiling")
        descriptor_cells = 3 * (active * active + candidates * candidates) + (
            active * candidates
        )
        if descriptor_cells > _MAX_BOUNDED_CELLS:
            raise ValueError("descriptor work exceeds the static safety ceiling")

    @property
    def n_tasks(self) -> int:
        """Control followed by the declared ordered Horde demons."""

        return 1 + self.managed_horde_demons

    @property
    def total_feature_dim(self) -> int:
        """Base representation plus deployed pair-feature slots."""

        return self.base_feature_dim + self.active_pair_slots

    @property
    def task_utility_weights(self) -> tuple[float, ...]:
        """Give control and the aggregate Horde group equal score mass."""

        demon_weight = 0.5 / float(self.managed_horde_demons)
        return (0.5, *(demon_weight for _ in range(self.managed_horde_demons)))

    def to_config(self) -> dict[str, object]:
        """Return a strict versioned JSON-compatible configuration."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA,
            "base_feature_dim": self.base_feature_dim,
            "active_pair_slots": self.active_pair_slots,
            "candidate_pair_slots": self.candidate_pair_slots,
            "managed_horde_demons": self.managed_horde_demons,
            "utility_decay": self.utility_decay,
            "shadow_step_size": self.shadow_step_size,
            "second_moment_decay": self.second_moment_decay,
            "scale_epsilon": self.scale_epsilon,
            "max_observations": self.max_observations,
        }

    @classmethod
    def from_config(cls, value: object) -> PrototypeFeatureUtilityConfig:
        """Reconstruct only the exact current schema and key set."""

        if type(value) is not dict:
            raise ValueError("feature utility config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {
            "schema_version",
            "state_schema",
            "base_feature_dim",
            "active_pair_slots",
            "candidate_pair_slots",
            "managed_horde_demons",
            "utility_decay",
            "shadow_step_size",
            "second_moment_decay",
            "scale_epsilon",
            "max_observations",
        }
        if set(raw) != expected:
            raise ValueError("feature utility config keys differ from schema v2")
        if raw["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("feature utility config schema_version differs")
        if raw["state_schema"] != PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA:
            raise ValueError("feature utility config state_schema differs")
        return cls(
            base_feature_dim=cast(int, raw["base_feature_dim"]),
            active_pair_slots=cast(int, raw["active_pair_slots"]),
            candidate_pair_slots=cast(int, raw["candidate_pair_slots"]),
            managed_horde_demons=cast(int, raw["managed_horde_demons"]),
            utility_decay=cast(float, raw["utility_decay"]),
            shadow_step_size=cast(float, raw["shadow_step_size"]),
            second_moment_decay=cast(float, raw["second_moment_decay"]),
            scale_epsilon=cast(float, raw["scale_epsilon"]),
            max_observations=cast(int, raw["max_observations"]),
        )


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityState:
    """All persistent fixed-budget state owned by the auditor."""

    semantic_generation: Int[Array, ""]
    semantic_generation_words: UInt[Array, " 2"]
    observation_count: Int[Array, ""]
    observation_words: UInt[Array, " 2"]
    active_descriptors: Int[Array, "active_pair_slots 2"]
    candidate_descriptors: Int[Array, "candidate_pair_slots 2"]
    active_task_utilities: Float[Array, "n_tasks active_pair_slots"]
    active_task_evidence_counts: Int[Array, "n_tasks active_pair_slots"]
    candidate_shadow_weights: Float[Array, "n_tasks candidate_pair_slots"]
    candidate_task_utilities: Float[Array, "n_tasks candidate_pair_slots"]
    candidate_task_evidence_counts: Int[Array, "n_tasks candidate_pair_slots"]
    candidate_second_moments: Float[Array, " candidate_pair_slots"]
    target_second_moments: Float[Array, " n_tasks"]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityEvent:
    """One frozen-consumer observation supplied by the Prototype adapter."""

    base_observation: Float[Array, " base_feature_dim"]
    augmented_observation: Float[Array, " total_feature_dim"]
    targets: Float[Array, " n_tasks"]
    predictions: Float[Array, " n_tasks"]
    target_available: Bool[Array, " n_tasks"]
    active_consumer_tail_weights: Float[
        Array,
        "n_tasks active_pair_slots",
    ]
    semantic_generation: Int[Array, ""]
    semantic_generation_words: UInt[Array, " 2"]
    active_descriptors: Int[Array, "active_pair_slots 2"]
    candidate_descriptors: Int[Array, "candidate_pair_slots 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityDiagnostics:
    """Finite primitive facts for one applied or rejected audit transaction."""

    available: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    capacity_capped: Bool[Array, ""]
    binding_rebound: Bool[Array, ""]
    stale_generation: Bool[Array, ""]
    skipped_generation: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    descriptor_contract_valid: Bool[Array, ""]
    observation_binding_valid: Bool[Array, ""]
    state_values_valid: Bool[Array, ""]
    event_values_valid: Bool[Array, ""]
    state_descriptors_valid: Bool[Array, ""]
    event_descriptors_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    same_generation_descriptor_mismatch: Bool[Array, ""]
    observation_matches_source: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    generation_capacity_available: Bool[Array, ""]
    numerical_update_valid: Bool[Array, ""]
    any_task_available: Bool[Array, ""]
    semantic_generation_before: Int[Array, ""]
    semantic_generation_after: Int[Array, ""]
    semantic_generation_words_before: UInt[Array, " 2"]
    semantic_generation_words_after: UInt[Array, " 2"]
    observation_count_before: Int[Array, ""]
    observation_count_after: Int[Array, ""]
    observation_words_before: UInt[Array, " 2"]
    observation_words_after: UInt[Array, " 2"]
    targets: Float[Array, " n_tasks"]
    predictions: Float[Array, " n_tasks"]
    target_available: Bool[Array, " n_tasks"]
    task_weights: Float[Array, " n_tasks"]
    source_active_descriptors: Int[Array, "active_pair_slots 2"]
    source_candidate_descriptors: Int[Array, "candidate_pair_slots 2"]
    active_live_mask: Bool[Array, " active_pair_slots"]
    candidate_eligible_mask: Bool[Array, " candidate_pair_slots"]
    active_values: Float[Array, " active_pair_slots"]
    candidate_values: Float[Array, " candidate_pair_slots"]
    active_survivor_mask: Bool[Array, " active_pair_slots"]
    candidate_survivor_mask: Bool[Array, " candidate_pair_slots"]
    candidate_collision_mask: Bool[Array, " candidate_pair_slots"]
    target_scale_second_moments: Float[Array, " n_tasks"]
    normalized_errors: Float[Array, " n_tasks"]
    active_normalized_contributions: Float[
        Array,
        "n_tasks active_pair_slots",
    ]
    active_loss_changes: Float[Array, "n_tasks active_pair_slots"]
    active_bounded_gains: Float[Array, "n_tasks active_pair_slots"]
    active_signed_scores: Float[Array, "n_tasks active_pair_slots"]
    active_aggregate_signal: Float[Array, " active_pair_slots"]
    candidate_normalized_contributions: Float[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_loss_changes: Float[Array, "n_tasks candidate_pair_slots"]
    candidate_bounded_gains: Float[Array, "n_tasks candidate_pair_slots"]
    candidate_signed_scores: Float[Array, "n_tasks candidate_pair_slots"]
    candidate_aggregate_signal: Float[Array, " candidate_pair_slots"]
    active_task_utilities_before: Float[
        Array,
        "n_tasks active_pair_slots",
    ]
    active_task_utilities_after: Float[
        Array,
        "n_tasks active_pair_slots",
    ]
    active_aggregate_utilities_before: Float[Array, " active_pair_slots"]
    active_aggregate_utilities_after: Float[Array, " active_pair_slots"]
    active_task_evidence_counts_before: Int[
        Array,
        "n_tasks active_pair_slots",
    ]
    active_task_evidence_counts_after: Int[
        Array,
        "n_tasks active_pair_slots",
    ]
    candidate_shadow_weights_before: Float[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_shadow_weights_after: Float[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_task_utilities_before: Float[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_task_utilities_after: Float[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_aggregate_utilities_before: Float[Array, " candidate_pair_slots"]
    candidate_aggregate_utilities_after: Float[Array, " candidate_pair_slots"]
    candidate_task_evidence_counts_before: Int[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_task_evidence_counts_after: Int[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    candidate_second_moments_before: Float[Array, " candidate_pair_slots"]
    candidate_second_moments_after: Float[Array, " candidate_pair_slots"]
    target_second_moments_before: Float[Array, " n_tasks"]
    target_second_moments_after: Float[Array, " n_tasks"]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityResult:
    """State and primitive diagnostic record from one audit transaction."""

    state: PrototypeFeatureUtilityState
    diagnostics: PrototypeFeatureUtilityDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureUtilityResourceBudget:
    """Exact persistent-state and per-observation logical work declaration."""

    persistent_logical_scalars: int
    persistent_state_nbytes: int
    telemetry_counter_nbytes: int
    exact_counter_nbytes: int
    counter_delta_nbytes: int
    counter_nbytes: int
    pair_products_per_observe: int
    task_feature_score_cells_per_observe: int
    shadow_update_cells_per_observe: int
    state_descriptor_validation_cells_per_observe: int
    event_descriptor_validation_cells_per_observe: int
    identity_rebind_cells_per_observe: int
    candidate_active_collision_cells_per_observe: int
    descriptor_comparison_cells_per_observe: int
    rng_draws_per_observe: int
    backward_passes_per_observe: int
    consumer_updates_per_observe: int
    router_calls_per_observe: int
    curation_decisions_per_observe: int
    mechanism_status: str
    scientific_promotion_allowed: bool
    curation_authority: bool
    max_observations: int


def _is_exact_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    name: str,
) -> Array:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a concrete JAX array")
    if value.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    if value.dtype != dtype:
        raise TypeError(f"{name} has the wrong dtype")
    return value


def _descriptor_rows_valid(descriptors: Array, base_feature_dim: int) -> Array:
    return jnp.all(
        (descriptors[:, 0] >= 0)
        & (descriptors[:, 0] < descriptors[:, 1])
        & (descriptors[:, 1] < base_feature_dim)
    )


def _descriptor_rows_unique(descriptors: Array) -> Array:
    count = descriptors.shape[0]
    if count < 2:
        return jnp.asarray(True, dtype=jnp.bool_)
    equal = jnp.all(descriptors[:, None, :] == descriptors[None, :, :], axis=2)
    return ~jnp.any(jnp.triu(equal, k=1))


def _candidate_collision_mask(
    active_descriptors: Array,
    candidate_descriptors: Array,
) -> Array:
    """Return candidates whose identity is already deployed as active."""

    return jnp.any(
        jnp.all(
            candidate_descriptors[:, None, :] == active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )


def _float_bits_equal(left: Array, right: Array) -> Array:
    left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
    right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
    return jnp.array_equal(left_bits, right_bits)


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact increment without wrapping the all-ones value."""

    _is_exact_array(words, shape=(2,), dtype=jnp.uint32, name="lifetime_words")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity, proposed, words), capacity


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Authenticate one exact identity against saturating int32 telemetry."""

    _is_exact_array(words, shape=(2,), dtype=jnp.uint32, name="lifetime_words")
    _is_exact_array(telemetry, shape=(), dtype=jnp.int32, name="lifetime_telemetry")
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _words_to_int32_telemetry(words: Array) -> Int[Array, ""]:
    """Project an exact word identity to non-authoritative int32 telemetry."""

    _is_exact_array(words, shape=(2,), dtype=jnp.uint32, name="lifetime_words")
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_words_less(left: Array, right: Array) -> Bool[Array, ""]:
    """Return exact lexicographic ``left < right`` for big-endian words."""

    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] < right[1])
    )


def _lifetime_words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Return exact lexicographic ``left <= right`` for big-endian words."""

    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] <= right[1])
    )


def _python_uint64_words(value: int) -> UInt[Array, " 2"]:
    """Encode one already-validated Python uint64 as big-endian words."""

    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _saturating_count_increment(value: Array, mask: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    incremented = jnp.minimum(jnp.maximum(value, 0), maximum - 1) + 1
    return jnp.where(mask, incremented, value)


def _finite_or_zero(value: Array) -> Array:
    """Return finite diagnostics even when an input/update is rejected."""

    return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class PrototypeFeatureUtilityAuditor:
    """Fixed-shape, JIT-compatible causal utility instrumentation."""

    def __init__(self, config: PrototypeFeatureUtilityConfig) -> None:
        if type(config) is not PrototypeFeatureUtilityConfig:
            raise TypeError("config must be a PrototypeFeatureUtilityConfig")
        self._config = config

    @property
    def config(self) -> PrototypeFeatureUtilityConfig:
        """Return the immutable static configuration."""

        return self._config

    def _validate_descriptors(self, value: object, *, candidate: bool) -> Array:
        slots = (
            self._config.candidate_pair_slots
            if candidate
            else self._config.active_pair_slots
        )
        name = "candidate_descriptors" if candidate else "active_descriptors"
        return _is_exact_array(
            value,
            shape=(slots, 2),
            dtype=jnp.int32,
            name=name,
        )

    def _validate_state_static(self, state: object) -> PrototypeFeatureUtilityState:
        if type(state) is not PrototypeFeatureUtilityState:
            raise TypeError("state must be an exact PrototypeFeatureUtilityState")
        checked = state
        tasks = self._config.n_tasks
        active = self._config.active_pair_slots
        candidates = self._config.candidate_pair_slots
        _is_exact_array(
            checked.semantic_generation,
            shape=(),
            dtype=jnp.int32,
            name="state.semantic_generation",
        )
        _is_exact_array(
            checked.semantic_generation_words,
            shape=(2,),
            dtype=jnp.uint32,
            name="state.semantic_generation_words",
        )
        _is_exact_array(
            checked.observation_count,
            shape=(),
            dtype=jnp.int32,
            name="state.observation_count",
        )
        _is_exact_array(
            checked.observation_words,
            shape=(2,),
            dtype=jnp.uint32,
            name="state.observation_words",
        )
        self._validate_descriptors(checked.active_descriptors, candidate=False)
        self._validate_descriptors(checked.candidate_descriptors, candidate=True)
        specs = (
            ("active_task_utilities", checked.active_task_utilities, (tasks, active), jnp.float32),
            (
                "active_task_evidence_counts",
                checked.active_task_evidence_counts,
                (tasks, active),
                jnp.int32,
            ),
            (
                "candidate_shadow_weights",
                checked.candidate_shadow_weights,
                (tasks, candidates),
                jnp.float32,
            ),
            (
                "candidate_task_utilities",
                checked.candidate_task_utilities,
                (tasks, candidates),
                jnp.float32,
            ),
            (
                "candidate_task_evidence_counts",
                checked.candidate_task_evidence_counts,
                (tasks, candidates),
                jnp.int32,
            ),
            (
                "candidate_second_moments",
                checked.candidate_second_moments,
                (candidates,),
                jnp.float32,
            ),
            (
                "target_second_moments",
                checked.target_second_moments,
                (tasks,),
                jnp.float32,
            ),
        )
        for name, value, shape, dtype in specs:
            _is_exact_array(
                value,
                shape=shape,
                dtype=dtype,
                name=f"state.{name}",
            )
        return checked

    def _validate_event_static(self, event: object) -> PrototypeFeatureUtilityEvent:
        if type(event) is not PrototypeFeatureUtilityEvent:
            raise TypeError("event must be an exact PrototypeFeatureUtilityEvent")
        checked = event
        tasks = self._config.n_tasks
        active = self._config.active_pair_slots
        specs = (
            (
                "base_observation",
                checked.base_observation,
                (self._config.base_feature_dim,),
                jnp.float32,
            ),
            (
                "augmented_observation",
                checked.augmented_observation,
                (self._config.total_feature_dim,),
                jnp.float32,
            ),
            ("targets", checked.targets, (tasks,), jnp.float32),
            ("predictions", checked.predictions, (tasks,), jnp.float32),
            ("target_available", checked.target_available, (tasks,), jnp.bool_),
            (
                "active_consumer_tail_weights",
                checked.active_consumer_tail_weights,
                (tasks, active),
                jnp.float32,
            ),
            ("semantic_generation", checked.semantic_generation, (), jnp.int32),
            (
                "semantic_generation_words",
                checked.semantic_generation_words,
                (2,),
                jnp.uint32,
            ),
        )
        for name, value, shape, dtype in specs:
            _is_exact_array(value, shape=shape, dtype=dtype, name=name)
        self._validate_descriptors(checked.active_descriptors, candidate=False)
        self._validate_descriptors(checked.candidate_descriptors, candidate=True)
        return checked

    def init(
        self,
        *,
        active_descriptors: Array,
        candidate_descriptors: Array,
        semantic_generation: Array,
        semantic_generation_words: Array,
    ) -> PrototypeFeatureUtilityState:
        """Initialize neutral state bound to one exact descriptor generation."""

        active = self._validate_descriptors(active_descriptors, candidate=False)
        candidates = self._validate_descriptors(candidate_descriptors, candidate=True)
        generation = _is_exact_array(
            semantic_generation,
            shape=(),
            dtype=jnp.int32,
            name="semantic_generation",
        )
        generation_words = _is_exact_array(
            semantic_generation_words,
            shape=(2,),
            dtype=jnp.uint32,
            name="semantic_generation_words",
        )
        descriptor_valid = (
            _descriptor_rows_valid(active, self._config.base_feature_dim)
            & _descriptor_rows_unique(active)
            & _descriptor_rows_valid(candidates, self._config.base_feature_dim)
            & _descriptor_rows_unique(candidates)
            & _lifetime_counter_valid(generation_words, generation)
        )
        if not bool(descriptor_valid):
            raise ValueError("initial feature utility descriptors are invalid")
        tasks = self._config.n_tasks
        active_slots = self._config.active_pair_slots
        candidate_slots = self._config.candidate_pair_slots
        return PrototypeFeatureUtilityState(
            semantic_generation=generation,
            semantic_generation_words=generation_words,
            observation_count=jnp.asarray(0, dtype=jnp.int32),
            observation_words=jnp.zeros((2,), dtype=jnp.uint32),
            active_descriptors=active,
            candidate_descriptors=candidates,
            active_task_utilities=jnp.zeros((tasks, active_slots), dtype=jnp.float32),
            active_task_evidence_counts=jnp.zeros(
                (tasks, active_slots),
                dtype=jnp.int32,
            ),
            candidate_shadow_weights=jnp.zeros(
                (tasks, candidate_slots),
                dtype=jnp.float32,
            ),
            candidate_task_utilities=jnp.zeros(
                (tasks, candidate_slots),
                dtype=jnp.float32,
            ),
            candidate_task_evidence_counts=jnp.zeros(
                (tasks, candidate_slots),
                dtype=jnp.int32,
            ),
            candidate_second_moments=jnp.zeros(
                (candidate_slots,),
                dtype=jnp.float32,
            ),
            target_second_moments=jnp.zeros((tasks,), dtype=jnp.float32),
        )

    def state_valid(self, state: object) -> Bool[Array, ""]:
        """Return the complete dynamic invariant for persistent audit state."""

        checked = self._validate_state_static(state)
        collision = _candidate_collision_mask(
            checked.active_descriptors,
            checked.candidate_descriptors,
        )
        collision_rows = collision[None, :]
        active_utilities_valid = (
            jnp.all(jnp.isfinite(checked.active_task_utilities))
            & jnp.all(checked.active_task_utilities >= 0.0)
            & jnp.all(checked.active_task_utilities <= 1.0)
        )
        candidate_utilities_valid = (
            jnp.all(jnp.isfinite(checked.candidate_task_utilities))
            & jnp.all(checked.candidate_task_utilities >= 0.0)
            & jnp.all(checked.candidate_task_utilities <= 1.0)
        )
        counts_valid = (
            jnp.all(checked.active_task_evidence_counts >= 0)
            & jnp.all(
                checked.active_task_evidence_counts <= checked.observation_count
            )
            & jnp.all(checked.candidate_task_evidence_counts >= 0)
            & jnp.all(
                checked.candidate_task_evidence_counts <= checked.observation_count
            )
        )
        lifetime_counters_valid = (
            _lifetime_counter_valid(
                checked.semantic_generation_words,
                checked.semantic_generation,
            )
            & _lifetime_counter_valid(
                checked.observation_words,
                checked.observation_count,
            )
            & _lifetime_words_less_equal(
                checked.observation_words,
                _python_uint64_words(self._config.max_observations),
            )
        )
        collided_candidates_zero = (
            jnp.all(
                jnp.where(
                    collision_rows,
                    checked.candidate_shadow_weights == 0.0,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    collision_rows,
                    checked.candidate_task_utilities == 0.0,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    collision_rows,
                    checked.candidate_task_evidence_counts == 0,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    collision,
                    checked.candidate_second_moments == 0.0,
                    True,
                )
            )
        )
        return (
            lifetime_counters_valid
            & _descriptor_rows_valid(
                checked.active_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(checked.active_descriptors)
            & _descriptor_rows_valid(
                checked.candidate_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(checked.candidate_descriptors)
            & active_utilities_valid
            & candidate_utilities_valid
            & counts_valid
            & jnp.all(jnp.isfinite(checked.candidate_shadow_weights))
            & jnp.all(jnp.isfinite(checked.candidate_second_moments))
            & jnp.all(checked.candidate_second_moments >= 0.0)
            & jnp.all(jnp.isfinite(checked.target_second_moments))
            & jnp.all(checked.target_second_moments >= 0.0)
            & collided_candidates_zero
        )

    def resource_budget(self) -> PrototypeFeatureUtilityResourceBudget:
        """Return exact state bytes and declared logical update work."""

        tasks = self._config.n_tasks
        active = self._config.active_pair_slots
        candidates = self._config.candidate_pair_slots
        logical_scalars = (
            6
            + 2 * active
            + 2 * tasks * active
            + 3 * candidates
            + 3 * tasks * candidates
            + tasks
        )
        descriptor_validation = active * active + candidates * candidates
        identity_rebind = active * active + candidates * candidates
        collision = active * candidates
        return PrototypeFeatureUtilityResourceBudget(
            persistent_logical_scalars=logical_scalars,
            persistent_state_nbytes=4 * logical_scalars,
            telemetry_counter_nbytes=(
                2 * PROTOTYPE_FEATURE_UTILITY_TELEMETRY_COUNTER_NBYTES
            ),
            exact_counter_nbytes=(
                2 * PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_DELTA_NBYTES
            ),
            counter_delta_nbytes=PROTOTYPE_FEATURE_UTILITY_COUNTER_DELTA_NBYTES,
            counter_nbytes=PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES,
            pair_products_per_observe=active + candidates,
            task_feature_score_cells_per_observe=tasks * (active + candidates),
            shadow_update_cells_per_observe=tasks * candidates,
            state_descriptor_validation_cells_per_observe=descriptor_validation,
            event_descriptor_validation_cells_per_observe=descriptor_validation,
            identity_rebind_cells_per_observe=identity_rebind,
            candidate_active_collision_cells_per_observe=collision,
            descriptor_comparison_cells_per_observe=(
                3 * descriptor_validation + collision
            ),
            rng_draws_per_observe=0,
            backward_passes_per_observe=0,
            consumer_updates_per_observe=0,
            router_calls_per_observe=0,
            curation_decisions_per_observe=0,
            mechanism_status=PROTOTYPE_FEATURE_UTILITY_MECHANISM_STATUS,
            scientific_promotion_allowed=False,
            curation_authority=False,
            max_observations=self._config.max_observations,
        )

    def _neutral_diagnostics(
        self,
        state: PrototypeFeatureUtilityState,
    ) -> PrototypeFeatureUtilityDiagnostics:
        tasks = self._config.n_tasks
        active = self._config.active_pair_slots
        candidates = self._config.candidate_pair_slots
        false = jnp.asarray(False, dtype=jnp.bool_)
        zeros_t = jnp.zeros((tasks,), dtype=jnp.float32)
        zeros_a = jnp.zeros((active,), dtype=jnp.float32)
        zeros_c = jnp.zeros((candidates,), dtype=jnp.float32)
        zeros_ta = jnp.zeros((tasks, active), dtype=jnp.float32)
        zeros_tc = jnp.zeros((tasks, candidates), dtype=jnp.float32)
        false_a = jnp.zeros((active,), dtype=jnp.bool_)
        false_c = jnp.zeros((candidates,), dtype=jnp.bool_)
        task_weights = jnp.asarray(
            self._config.task_utility_weights,
            dtype=jnp.float32,
        )
        safe_active_utilities = _finite_or_zero(state.active_task_utilities)
        safe_candidate_shadow = _finite_or_zero(state.candidate_shadow_weights)
        safe_candidate_utilities = _finite_or_zero(state.candidate_task_utilities)
        safe_candidate_moments = _finite_or_zero(state.candidate_second_moments)
        safe_target_moments = _finite_or_zero(state.target_second_moments)
        safe_active_counts = jnp.where(
            state.active_task_evidence_counts >= 0,
            state.active_task_evidence_counts,
            0,
        )
        safe_candidate_counts = jnp.where(
            state.candidate_task_evidence_counts >= 0,
            state.candidate_task_evidence_counts,
            0,
        )
        state_descriptors_valid = (
            _descriptor_rows_valid(
                state.active_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(state.active_descriptors)
            & _descriptor_rows_valid(
                state.candidate_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(state.candidate_descriptors)
        )
        state_values_valid = self.state_valid(state)
        observation_capacity_available = _lifetime_words_less(
            state.observation_words,
            _python_uint64_words(self._config.max_observations),
        )
        return PrototypeFeatureUtilityDiagnostics(
            available=false,
            transaction_applied=false,
            capacity_capped=false,
            binding_rebound=false,
            stale_generation=false,
            skipped_generation=false,
            inputs_finite=false,
            descriptor_contract_valid=false,
            observation_binding_valid=false,
            state_values_valid=state_values_valid,
            event_values_valid=false,
            state_descriptors_valid=state_descriptors_valid,
            event_descriptors_valid=false,
            binding_valid=false,
            same_generation_descriptor_mismatch=false,
            observation_matches_source=false,
            capacity_available=observation_capacity_available,
            generation_capacity_available=~jnp.all(
                state.semantic_generation_words
                == jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
            ),
            numerical_update_valid=false,
            any_task_available=false,
            semantic_generation_before=state.semantic_generation,
            semantic_generation_after=state.semantic_generation,
            semantic_generation_words_before=state.semantic_generation_words,
            semantic_generation_words_after=state.semantic_generation_words,
            observation_count_before=state.observation_count,
            observation_count_after=state.observation_count,
            observation_words_before=state.observation_words,
            observation_words_after=state.observation_words,
            targets=zeros_t,
            predictions=zeros_t,
            target_available=jnp.zeros((tasks,), dtype=jnp.bool_),
            task_weights=task_weights,
            source_active_descriptors=state.active_descriptors,
            source_candidate_descriptors=state.candidate_descriptors,
            active_live_mask=false_a,
            candidate_eligible_mask=false_c,
            active_values=zeros_a,
            candidate_values=zeros_c,
            active_survivor_mask=false_a,
            candidate_survivor_mask=false_c,
            candidate_collision_mask=false_c,
            target_scale_second_moments=zeros_t,
            normalized_errors=zeros_t,
            active_normalized_contributions=zeros_ta,
            active_loss_changes=zeros_ta,
            active_bounded_gains=zeros_ta,
            active_signed_scores=zeros_ta,
            active_aggregate_signal=zeros_a,
            candidate_normalized_contributions=zeros_tc,
            candidate_loss_changes=zeros_tc,
            candidate_bounded_gains=zeros_tc,
            candidate_signed_scores=zeros_tc,
            candidate_aggregate_signal=zeros_c,
            active_task_utilities_before=safe_active_utilities,
            active_task_utilities_after=safe_active_utilities,
            active_aggregate_utilities_before=jnp.sum(
                task_weights[:, None] * safe_active_utilities,
                axis=0,
            ),
            active_aggregate_utilities_after=jnp.sum(
                task_weights[:, None] * safe_active_utilities,
                axis=0,
            ),
            active_task_evidence_counts_before=(
                safe_active_counts
            ),
            active_task_evidence_counts_after=(
                safe_active_counts
            ),
            candidate_shadow_weights_before=safe_candidate_shadow,
            candidate_shadow_weights_after=safe_candidate_shadow,
            candidate_task_utilities_before=safe_candidate_utilities,
            candidate_task_utilities_after=safe_candidate_utilities,
            candidate_aggregate_utilities_before=jnp.sum(
                task_weights[:, None] * safe_candidate_utilities,
                axis=0,
            ),
            candidate_aggregate_utilities_after=jnp.sum(
                task_weights[:, None] * safe_candidate_utilities,
                axis=0,
            ),
            candidate_task_evidence_counts_before=(
                safe_candidate_counts
            ),
            candidate_task_evidence_counts_after=(
                safe_candidate_counts
            ),
            candidate_second_moments_before=safe_candidate_moments,
            candidate_second_moments_after=safe_candidate_moments,
            target_second_moments_before=safe_target_moments,
            target_second_moments_after=safe_target_moments,
        )

    def unavailable_diagnostics(
        self,
        state: PrototypeFeatureUtilityState,
    ) -> PrototypeFeatureUtilityDiagnostics:
        """Return finite neutral diagnostics without changing any state."""

        return self._neutral_diagnostics(self._validate_state_static(state))

    @staticmethod
    def _pair_values(base: Array, descriptors: Array) -> Array:
        left = jnp.clip(descriptors[:, 0], 0, base.shape[0] - 1)
        right = jnp.clip(descriptors[:, 1], 0, base.shape[0] - 1)
        live = (
            (descriptors[:, 0] >= 0)
            & (descriptors[:, 0] < descriptors[:, 1])
            & (descriptors[:, 1] < base.shape[0])
        )
        return base[left] * base[right] * live.astype(jnp.float32)

    def _route_generation(
        self,
        state: PrototypeFeatureUtilityState,
        active_descriptors: Array,
        candidate_descriptors: Array,
        semantic_generation: Array,
        semantic_generation_words: Array,
    ) -> tuple[PrototypeFeatureUtilityState, Array, Array, Array]:
        active_matches = jnp.all(
            active_descriptors[:, None, :] == state.active_descriptors[None, :, :],
            axis=2,
        )
        active_survivors = jnp.any(active_matches, axis=1)
        active_sources = jnp.argmax(active_matches, axis=1)
        routed_active_utilities = jnp.where(
            active_survivors[None, :],
            state.active_task_utilities[:, active_sources],
            jnp.zeros_like(state.active_task_utilities),
        )
        routed_active_counts = jnp.where(
            active_survivors[None, :],
            state.active_task_evidence_counts[:, active_sources],
            jnp.zeros_like(state.active_task_evidence_counts),
        )

        collision = _candidate_collision_mask(
            active_descriptors,
            candidate_descriptors,
        )
        if self._config.candidate_pair_slots:
            candidate_matches = jnp.all(
                candidate_descriptors[:, None, :]
                == state.candidate_descriptors[None, :, :],
                axis=2,
            )
            candidate_survivors = jnp.any(candidate_matches, axis=1) & ~collision
            candidate_sources = jnp.argmax(candidate_matches, axis=1)
            routed_shadow = jnp.where(
                candidate_survivors[None, :],
                state.candidate_shadow_weights[:, candidate_sources],
                jnp.zeros_like(state.candidate_shadow_weights),
            )
            routed_candidate_utilities = jnp.where(
                candidate_survivors[None, :],
                state.candidate_task_utilities[:, candidate_sources],
                jnp.zeros_like(state.candidate_task_utilities),
            )
            routed_candidate_counts = jnp.where(
                candidate_survivors[None, :],
                state.candidate_task_evidence_counts[:, candidate_sources],
                jnp.zeros_like(state.candidate_task_evidence_counts),
            )
            routed_candidate_moments = jnp.where(
                candidate_survivors,
                state.candidate_second_moments[candidate_sources],
                jnp.zeros_like(state.candidate_second_moments),
            )
        else:
            candidate_survivors = jnp.zeros((0,), dtype=jnp.bool_)
            routed_shadow = state.candidate_shadow_weights
            routed_candidate_utilities = state.candidate_task_utilities
            routed_candidate_counts = state.candidate_task_evidence_counts
            routed_candidate_moments = state.candidate_second_moments
        routed = PrototypeFeatureUtilityState(
            semantic_generation=semantic_generation,
            semantic_generation_words=semantic_generation_words,
            observation_count=state.observation_count,
            observation_words=state.observation_words,
            active_descriptors=active_descriptors,
            candidate_descriptors=candidate_descriptors,
            active_task_utilities=routed_active_utilities,
            active_task_evidence_counts=routed_active_counts,
            candidate_shadow_weights=routed_shadow,
            candidate_task_utilities=routed_candidate_utilities,
            candidate_task_evidence_counts=routed_candidate_counts,
            candidate_second_moments=routed_candidate_moments,
            target_second_moments=state.target_second_moments,
        )
        return routed, active_survivors, candidate_survivors, collision

    def rebind(
        self,
        state: PrototypeFeatureUtilityState,
        *,
        active_descriptors: Array,
        candidate_descriptors: Array,
        semantic_generation: Array,
        semantic_generation_words: Array,
    ) -> PrototypeFeatureUtilityResult:
        """Route identity-local audit state across exactly one generation."""

        checked = self._validate_state_static(state)
        active = self._validate_descriptors(active_descriptors, candidate=False)
        candidates = self._validate_descriptors(candidate_descriptors, candidate=True)
        generation = _is_exact_array(
            semantic_generation,
            shape=(),
            dtype=jnp.int32,
            name="semantic_generation",
        )
        generation_words = _is_exact_array(
            semantic_generation_words,
            shape=(2,),
            dtype=jnp.uint32,
            name="semantic_generation_words",
        )
        next_generation_words, can_advance = _checked_lifetime_words_increment(
            checked.semantic_generation_words
        )
        next_generation = _words_to_int32_telemetry(next_generation_words)
        generation_counter_valid = _lifetime_counter_valid(
            generation_words,
            generation,
        )
        generation_valid = (
            can_advance
            & generation_counter_valid
            & jnp.all(generation_words == next_generation_words)
            & (generation == next_generation)
        )
        stale = _lifetime_words_less_equal(
            generation_words,
            checked.semantic_generation_words,
        )
        skipped = _lifetime_words_less(
            checked.semantic_generation_words,
            generation_words,
        ) & ~generation_valid
        descriptor_valid = (
            _descriptor_rows_valid(active, self._config.base_feature_dim)
            & _descriptor_rows_unique(active)
            & _descriptor_rows_valid(candidates, self._config.base_feature_dim)
            & _descriptor_rows_unique(candidates)
        )
        state_values_valid = self.state_valid(checked)
        routed, active_survivors, candidate_survivors, collision = (
            self._route_generation(
                checked,
                active,
                candidates,
                generation,
                generation_words,
            )
        )
        routed_values_valid = self.state_valid(routed)
        allowed = (
            state_values_valid
            & descriptor_valid
            & generation_valid
            & routed_values_valid
        )
        task_weights = jnp.asarray(
            self._config.task_utility_weights,
            dtype=jnp.float32,
        )
        active_aggregate = jnp.sum(
            task_weights[:, None] * routed.active_task_utilities,
            axis=0,
        )
        candidate_aggregate = jnp.sum(
            task_weights[:, None] * routed.candidate_task_utilities,
            axis=0,
        )
        diagnostics = cast(
            PrototypeFeatureUtilityDiagnostics,
            self._neutral_diagnostics(checked).replace(
                available=allowed,
                transaction_applied=allowed,
                binding_rebound=allowed,
                stale_generation=stale,
                skipped_generation=skipped,
                inputs_finite=jnp.asarray(True, dtype=jnp.bool_),
                descriptor_contract_valid=descriptor_valid,
                observation_binding_valid=jnp.asarray(True, dtype=jnp.bool_),
                state_values_valid=state_values_valid,
                event_values_valid=descriptor_valid & generation_counter_valid,
                event_descriptors_valid=descriptor_valid,
                binding_valid=generation_valid,
                generation_capacity_available=can_advance,
                observation_matches_source=jnp.asarray(True, dtype=jnp.bool_),
                numerical_update_valid=routed_values_valid,
                source_active_descriptors=active,
                source_candidate_descriptors=candidates,
                active_live_mask=jnp.where(
                    allowed,
                    jnp.ones_like(active_survivors),
                    jnp.zeros_like(active_survivors),
                ),
                candidate_eligible_mask=jnp.where(
                    allowed,
                    ~collision,
                    jnp.zeros_like(collision),
                ),
                semantic_generation_after=jnp.where(
                    allowed,
                    generation,
                    checked.semantic_generation,
                ),
                semantic_generation_words_after=jnp.where(
                    allowed,
                    generation_words,
                    checked.semantic_generation_words,
                ),
                active_survivor_mask=jnp.where(
                    allowed,
                    active_survivors,
                    jnp.zeros_like(active_survivors),
                ),
                candidate_survivor_mask=jnp.where(
                    allowed,
                    candidate_survivors,
                    jnp.zeros_like(candidate_survivors),
                ),
                candidate_collision_mask=jnp.where(
                    allowed,
                    collision,
                    jnp.zeros_like(collision),
                ),
                active_task_utilities_before=jnp.where(
                    allowed,
                    routed.active_task_utilities,
                    0.0,
                ),
                active_task_utilities_after=jnp.where(
                    allowed,
                    routed.active_task_utilities,
                    0.0,
                ),
                active_aggregate_utilities_before=jnp.where(
                    allowed,
                    active_aggregate,
                    0.0,
                ),
                active_aggregate_utilities_after=jnp.where(
                    allowed,
                    active_aggregate,
                    0.0,
                ),
                active_task_evidence_counts_before=(
                    jnp.where(allowed, routed.active_task_evidence_counts, 0)
                ),
                active_task_evidence_counts_after=(
                    jnp.where(allowed, routed.active_task_evidence_counts, 0)
                ),
                candidate_shadow_weights_before=jnp.where(
                    allowed,
                    routed.candidate_shadow_weights,
                    0.0,
                ),
                candidate_shadow_weights_after=jnp.where(
                    allowed,
                    routed.candidate_shadow_weights,
                    0.0,
                ),
                candidate_task_utilities_before=jnp.where(
                    allowed,
                    routed.candidate_task_utilities,
                    0.0,
                ),
                candidate_task_utilities_after=jnp.where(
                    allowed,
                    routed.candidate_task_utilities,
                    0.0,
                ),
                candidate_aggregate_utilities_before=jnp.where(
                    allowed,
                    candidate_aggregate,
                    0.0,
                ),
                candidate_aggregate_utilities_after=jnp.where(
                    allowed,
                    candidate_aggregate,
                    0.0,
                ),
                candidate_task_evidence_counts_before=(
                    jnp.where(allowed, routed.candidate_task_evidence_counts, 0)
                ),
                candidate_task_evidence_counts_after=(
                    jnp.where(allowed, routed.candidate_task_evidence_counts, 0)
                ),
                candidate_second_moments_before=(
                    jnp.where(allowed, routed.candidate_second_moments, 0.0)
                ),
                candidate_second_moments_after=jnp.where(
                    allowed,
                    routed.candidate_second_moments,
                    0.0,
                ),
                target_second_moments_before=jnp.where(
                    allowed,
                    routed.target_second_moments,
                    0.0,
                ),
                target_second_moments_after=jnp.where(
                    allowed,
                    routed.target_second_moments,
                    0.0,
                ),
            ),
        )
        final_state = cast(
            PrototypeFeatureUtilityState,
            jax.lax.cond(allowed, lambda: routed, lambda: checked),
        )
        return PrototypeFeatureUtilityResult(
            state=final_state,
            diagnostics=diagnostics,
        )

    def observe(
        self,
        state: PrototypeFeatureUtilityState,
        event: PrototypeFeatureUtilityEvent,
    ) -> PrototypeFeatureUtilityResult:
        """Score frozen consumers before updating EMAs, probes, or moments.

        Observation count advances for every valid accepted event.  Per-task
        evidence advances only for available tasks, while unavailable rows
        still receive a zero-gain utility-EMA update (and therefore decay).
        Missing task weight is never renormalized onto the available tasks.
        """

        checked = self._validate_state_static(state)
        item = self._validate_event_static(event)
        state_values_valid = self.state_valid(checked)
        state_descriptors_valid = (
            _descriptor_rows_valid(
                checked.active_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(checked.active_descriptors)
            & _descriptor_rows_valid(
                checked.candidate_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(checked.candidate_descriptors)
        )
        event_descriptors_valid = (
            _descriptor_rows_valid(
                item.active_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(item.active_descriptors)
            & _descriptor_rows_valid(
                item.candidate_descriptors,
                self._config.base_feature_dim,
            )
            & _descriptor_rows_unique(item.candidate_descriptors)
        )
        event_generation_counter_valid = _lifetime_counter_valid(
            item.semantic_generation_words,
            item.semantic_generation,
        )
        same_generation = (
            event_generation_counter_valid
            & (item.semantic_generation == checked.semantic_generation)
            & jnp.all(
                item.semantic_generation_words
                == checked.semantic_generation_words
            )
        )
        stale = _lifetime_words_less(
            item.semantic_generation_words,
            checked.semantic_generation_words,
        )
        skipped = _lifetime_words_less(
            checked.semantic_generation_words,
            item.semantic_generation_words,
        )
        same_binding = (
            jnp.array_equal(item.active_descriptors, checked.active_descriptors)
            & jnp.array_equal(item.candidate_descriptors, checked.candidate_descriptors)
        )
        same_generation_mismatch = same_generation & ~same_binding
        binding_valid = same_generation & same_binding
        working = checked
        collision = _candidate_collision_mask(
            item.active_descriptors,
            item.candidate_descriptors,
        )
        candidate_eligible = ~collision

        active_values = item.augmented_observation[self._config.base_feature_dim :]
        expected_active_values = self._pair_values(
            item.base_observation,
            item.active_descriptors,
        )
        candidate_values = self._pair_values(
            item.base_observation,
            item.candidate_descriptors,
        )
        observation_binding = (
            _float_bits_equal(
                item.base_observation,
                item.augmented_observation[: self._config.base_feature_dim],
            )
            & _float_bits_equal(active_values, expected_active_values)
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(item.base_observation))
            & jnp.all(jnp.isfinite(item.augmented_observation))
            & jnp.all(jnp.isfinite(item.targets))
            & jnp.all(jnp.isfinite(item.predictions))
            & jnp.all(jnp.isfinite(item.active_consumer_tail_weights))
        )
        proposed_observation_words, exact_lifetime_capacity_available = (
            _checked_lifetime_words_increment(checked.observation_words)
        )
        configured_capacity_available = _lifetime_words_less_equal(
            proposed_observation_words,
            _python_uint64_words(self._config.max_observations),
        )
        capacity_available = (
            exact_lifetime_capacity_available & configured_capacity_available
        )
        preconditions_valid = (
            state_values_valid
            & state_descriptors_valid
            & event_descriptors_valid
            & binding_valid
            & observation_binding
            & inputs_finite
            & event_generation_counter_valid
            & capacity_available
            & ~stale
            & ~skipped
        )

        epsilon = jnp.asarray(self._config.scale_epsilon, dtype=jnp.float32)
        scale2 = jnp.maximum(
            jnp.maximum(working.target_second_moments, jnp.square(item.targets)),
            jnp.maximum(jnp.square(item.predictions), epsilon),
        )
        target_scale = jnp.sqrt(scale2)
        error = (item.targets - item.predictions) / target_scale
        active_contribution = (
            item.active_consumer_tail_weights
            * active_values[None, :]
            / target_scale[:, None]
        )
        available = item.target_available[:, None]
        active_change = 0.5 * (
            jnp.square(error[:, None] + active_contribution)
            - jnp.square(error[:, None])
        )
        active_change = jnp.where(available, active_change, 0.0)
        active_positive = jnp.maximum(active_change, 0.0)
        active_gain = active_positive / (1.0 + active_positive)
        active_signed = active_change / (1.0 + jnp.abs(active_change))

        candidate_mask = available & candidate_eligible[None, :]
        candidate_contribution = jnp.where(
            candidate_eligible[None, :],
            working.candidate_shadow_weights * candidate_values[None, :],
            0.0,
        )
        candidate_change = 0.5 * (
            jnp.square(error[:, None])
            - jnp.square(error[:, None] - candidate_contribution)
        )
        candidate_change = jnp.where(candidate_mask, candidate_change, 0.0)
        candidate_positive = jnp.maximum(candidate_change, 0.0)
        candidate_gain = candidate_positive / (1.0 + candidate_positive)
        candidate_signed = candidate_change / (1.0 + jnp.abs(candidate_change))
        task_weights = jnp.asarray(
            self._config.task_utility_weights,
            dtype=jnp.float32,
        )
        active_aggregate = jnp.sum(task_weights[:, None] * active_gain, axis=0)
        candidate_aggregate = jnp.sum(
            task_weights[:, None] * candidate_gain,
            axis=0,
        )

        utility_decay = jnp.asarray(self._config.utility_decay, dtype=jnp.float32)
        one_minus_utility = 1.0 - utility_decay
        active_utilities = (
            utility_decay * working.active_task_utilities
            + one_minus_utility * active_gain
        )
        candidate_utilities = jnp.where(
            candidate_eligible[None, :],
            utility_decay * working.candidate_task_utilities
            + one_minus_utility * candidate_gain,
            0.0,
        )
        active_counts = _saturating_count_increment(
            working.active_task_evidence_counts,
            available,
        )
        candidate_counts = _saturating_count_increment(
            working.candidate_task_evidence_counts,
            candidate_mask,
        )

        moment_decay = jnp.asarray(
            self._config.second_moment_decay,
            dtype=jnp.float32,
        )
        one_minus_moment = 1.0 - moment_decay
        candidate_square = jnp.square(candidate_values)
        if self._config.candidate_pair_slots > 0:
            eligible_float = candidate_eligible.astype(jnp.float32)
            eligible_count = jnp.maximum(
                jnp.sum(eligible_float),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            candidate_energy = jnp.sum(
                jnp.where(candidate_eligible, candidate_square, 0.0)
            ) / eligible_count
            shadow_normalizer = jnp.maximum(
                jnp.maximum(
                    jnp.maximum(
                        working.candidate_second_moments,
                        candidate_square,
                    ),
                    candidate_energy,
                ),
                epsilon,
            )
            lipschitz = 1.0 + candidate_square / shadow_normalizer
            shadow_delta = (
                jnp.asarray(self._config.shadow_step_size, dtype=jnp.float32)
                * (error[:, None] - candidate_contribution)
                * candidate_values[None, :]
                / (shadow_normalizer[None, :] * lipschitz[None, :])
            )
            shadow_weights = jnp.where(
                candidate_eligible[None, :],
                working.candidate_shadow_weights
                + jnp.where(candidate_mask, shadow_delta, 0.0),
                0.0,
            )
            candidate_moments = jnp.where(
                candidate_eligible,
                moment_decay * working.candidate_second_moments
                + one_minus_moment * candidate_square,
                0.0,
            )
        else:
            shadow_weights = working.candidate_shadow_weights
            candidate_moments = working.candidate_second_moments
        target_moments = jnp.where(
            item.target_available,
            moment_decay * working.target_second_moments
            + one_minus_moment * jnp.square(item.targets),
            working.target_second_moments,
        )
        proposed = PrototypeFeatureUtilityState(
            semantic_generation=working.semantic_generation,
            semantic_generation_words=working.semantic_generation_words,
            observation_count=_words_to_int32_telemetry(
                proposed_observation_words
            ),
            observation_words=proposed_observation_words,
            active_descriptors=working.active_descriptors,
            candidate_descriptors=working.candidate_descriptors,
            active_task_utilities=active_utilities,
            active_task_evidence_counts=active_counts,
            candidate_shadow_weights=shadow_weights,
            candidate_task_utilities=candidate_utilities,
            candidate_task_evidence_counts=candidate_counts,
            candidate_second_moments=candidate_moments,
            target_second_moments=target_moments,
        )
        numerical_update_valid = (
            jnp.all(jnp.isfinite(scale2))
            & jnp.all(jnp.isfinite(error))
            & jnp.all(jnp.isfinite(active_contribution))
            & jnp.all(jnp.isfinite(active_change))
            & jnp.all(jnp.isfinite(active_gain))
            & jnp.all(jnp.isfinite(active_signed))
            & jnp.all(jnp.isfinite(candidate_values))
            & jnp.all(jnp.isfinite(candidate_contribution))
            & jnp.all(jnp.isfinite(candidate_change))
            & jnp.all(jnp.isfinite(candidate_gain))
            & jnp.all(jnp.isfinite(candidate_signed))
            & self.state_valid(proposed)
        )
        apply = preconditions_valid & numerical_update_valid
        final_state = cast(
            PrototypeFeatureUtilityState,
            jax.lax.cond(apply, lambda: proposed, lambda: checked),
        )

        def selected(value: Array) -> Array:
            return jnp.where(apply, value, jnp.zeros_like(value))

        active_utility_before_aggregate = jnp.sum(
            task_weights[:, None] * working.active_task_utilities,
            axis=0,
        )
        active_utility_after_aggregate = jnp.sum(
            task_weights[:, None] * active_utilities,
            axis=0,
        )
        candidate_utility_before_aggregate = jnp.sum(
            task_weights[:, None] * working.candidate_task_utilities,
            axis=0,
        )
        candidate_utility_after_aggregate = jnp.sum(
            task_weights[:, None] * candidate_utilities,
            axis=0,
        )
        diagnostics = cast(
            PrototypeFeatureUtilityDiagnostics,
            self._neutral_diagnostics(checked).replace(
                available=apply,
                transaction_applied=apply,
                capacity_capped=~capacity_available,
                binding_rebound=jnp.asarray(False, dtype=jnp.bool_),
                stale_generation=stale,
                skipped_generation=skipped,
                inputs_finite=inputs_finite,
                descriptor_contract_valid=event_descriptors_valid,
                observation_binding_valid=observation_binding,
                state_values_valid=state_values_valid,
                event_values_valid=(
                    inputs_finite & event_generation_counter_valid
                ),
                state_descriptors_valid=state_descriptors_valid,
                event_descriptors_valid=event_descriptors_valid,
                binding_valid=binding_valid,
                same_generation_descriptor_mismatch=(
                    same_generation_mismatch
                ),
                observation_matches_source=observation_binding,
                capacity_available=capacity_available,
                numerical_update_valid=numerical_update_valid,
                any_task_available=jnp.any(item.target_available),
                semantic_generation_after=jnp.where(
                    apply,
                    proposed.semantic_generation,
                    checked.semantic_generation,
                ),
                semantic_generation_words_after=jnp.where(
                    apply,
                    proposed.semantic_generation_words,
                    checked.semantic_generation_words,
                ),
                observation_count_after=jnp.where(
                    apply,
                    proposed.observation_count,
                    checked.observation_count,
                ),
                observation_words_after=jnp.where(
                    apply,
                    proposed.observation_words,
                    checked.observation_words,
                ),
                targets=_finite_or_zero(item.targets),
                predictions=_finite_or_zero(item.predictions),
                target_available=item.target_available,
                task_weights=task_weights,
                source_active_descriptors=item.active_descriptors,
                source_candidate_descriptors=item.candidate_descriptors,
                active_live_mask=jnp.where(
                    apply,
                    jnp.ones((self._config.active_pair_slots,), dtype=jnp.bool_),
                    jnp.zeros((self._config.active_pair_slots,), dtype=jnp.bool_),
                ),
                candidate_eligible_mask=jnp.where(
                    apply,
                    candidate_eligible,
                    jnp.zeros_like(candidate_eligible),
                ),
                active_values=selected(active_values),
                candidate_values=selected(candidate_values),
                active_survivor_mask=jnp.where(
                    False,
                    jnp.ones((self._config.active_pair_slots,), dtype=jnp.bool_),
                    jnp.zeros((self._config.active_pair_slots,), dtype=jnp.bool_),
                ),
                candidate_survivor_mask=jnp.where(
                    False,
                    jnp.ones((self._config.candidate_pair_slots,), dtype=jnp.bool_),
                    jnp.zeros((self._config.candidate_pair_slots,), dtype=jnp.bool_),
                ),
                candidate_collision_mask=jnp.where(
                    apply,
                    collision,
                    jnp.zeros_like(collision),
                ),
                target_scale_second_moments=selected(scale2),
                normalized_errors=selected(error),
                active_normalized_contributions=selected(active_contribution),
                active_loss_changes=selected(active_change),
                active_bounded_gains=selected(active_gain),
                active_signed_scores=selected(active_signed),
                active_aggregate_signal=selected(active_aggregate),
                candidate_normalized_contributions=selected(
                    candidate_contribution
                ),
                candidate_loss_changes=selected(candidate_change),
                candidate_bounded_gains=selected(candidate_gain),
                candidate_signed_scores=selected(candidate_signed),
                candidate_aggregate_signal=selected(candidate_aggregate),
                active_task_utilities_before=selected(
                    working.active_task_utilities
                ),
                active_task_utilities_after=selected(active_utilities),
                active_aggregate_utilities_before=selected(
                    active_utility_before_aggregate
                ),
                active_aggregate_utilities_after=selected(
                    active_utility_after_aggregate
                ),
                active_task_evidence_counts_before=selected(
                    working.active_task_evidence_counts
                ),
                active_task_evidence_counts_after=selected(active_counts),
                candidate_shadow_weights_before=selected(
                    working.candidate_shadow_weights
                ),
                candidate_shadow_weights_after=selected(shadow_weights),
                candidate_task_utilities_before=selected(
                    working.candidate_task_utilities
                ),
                candidate_task_utilities_after=selected(candidate_utilities),
                candidate_aggregate_utilities_before=selected(
                    candidate_utility_before_aggregate
                ),
                candidate_aggregate_utilities_after=selected(
                    candidate_utility_after_aggregate
                ),
                candidate_task_evidence_counts_before=selected(
                    working.candidate_task_evidence_counts
                ),
                candidate_task_evidence_counts_after=selected(candidate_counts),
                candidate_second_moments_before=selected(
                    working.candidate_second_moments
                ),
                candidate_second_moments_after=selected(candidate_moments),
                target_second_moments_before=selected(
                    working.target_second_moments
                ),
                target_second_moments_after=selected(target_moments),
            ),
        )
        return PrototypeFeatureUtilityResult(state=final_state, diagnostics=diagnostics)


def prototype_feature_utility_lifetime_counter_nbytes() -> int:
    """Return bytes for one telemetry scalar plus one exact word identity."""

    return PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_NBYTES


def prototype_feature_utility_counter_nbytes() -> int:
    """Return bytes for both utility-owned exact lifetime identities."""

    return PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES


def measure_prototype_feature_utility_state_nbytes(
    state: PrototypeFeatureUtilityState,
) -> int:
    """Measure every persistent array leaf in one exact utility state."""

    if type(state) is not PrototypeFeatureUtilityState:
        raise TypeError("state must be a PrototypeFeatureUtilityState")
    return sum(int(leaf.nbytes) for leaf in jax.tree.leaves(state))


def migrate_legacy_prototype_feature_utility_config(
    legacy_config: object,
) -> PrototypeFeatureUtilityConfig:
    """Explicitly relabel an exact v1 mechanism configuration as v2."""

    if type(legacy_config) is not dict:
        raise ValueError("legacy feature utility config must be an exact dict")
    raw = dict(cast(dict[object, object], legacy_config))
    expected = {
        "schema_version",
        "base_feature_dim",
        "active_pair_slots",
        "candidate_pair_slots",
        "managed_horde_demons",
        "utility_decay",
        "shadow_step_size",
        "second_moment_decay",
        "scale_epsilon",
        "max_observations",
    }
    if set(raw) != expected:
        raise ValueError("legacy feature utility config keys differ from schema v1")
    if raw["schema_version"] != _LEGACY_PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA:
        raise ValueError("legacy feature utility config schema_version differs")
    legacy_maximum = raw["max_observations"]
    if (
        type(legacy_maximum) is not int
        or not 1 <= legacy_maximum <= _INT32_MAX - 1
    ):
        raise ValueError(
            "legacy feature utility max_observations exceeds its int32-safe schema"
        )
    raw["schema_version"] = PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA
    raw["state_schema"] = PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA
    return PrototypeFeatureUtilityConfig.from_config(raw)


def migrate_legacy_prototype_feature_utility_state(
    auditor: PrototypeFeatureUtilityAuditor,
    legacy_state: Any,
) -> PrototypeFeatureUtilityState:
    """Migrate v1 clocks only when both int32 histories are unambiguous."""

    if type(auditor) is not PrototypeFeatureUtilityAuditor:
        raise TypeError("auditor must be a PrototypeFeatureUtilityAuditor")
    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy feature utility state must be a mapping or dataclass")
    exact_word_fields = {"semantic_generation_words", "observation_words"}
    current_names = {
        field.name
        for field in dataclasses.fields(PrototypeFeatureUtilityState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - exact_word_fields
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy feature utility field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    for telemetry_name, words_name in (
        ("semantic_generation", "semantic_generation_words"),
        ("observation_count", "observation_words"),
    ):
        telemetry = jnp.asarray(fields[telemetry_name])
        if telemetry.shape != () or telemetry.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy feature utility {telemetry_name} must be scalar int32")
        count = int(telemetry)
        if count < 0:
            raise ValueError(
                f"negative legacy feature utility {telemetry_name} indicates wrap"
            )
        if count >= _INT32_MAX:
            raise ValueError(
                f"saturated legacy feature utility {telemetry_name} is ambiguous"
            )
        fields[words_name] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = PrototypeFeatureUtilityState(**fields)
    if not bool(auditor.state_valid(migrated)):
        raise ValueError("legacy feature utility state violates the exact v2 contract")
    return migrated


__all__ = [
    "PROTOTYPE_FEATURE_UTILITY_COUNTER_DELTA_NBYTES",
    "PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES",
    "PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY",
    "PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_DELTA_NBYTES",
    "PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_NBYTES",
    "PROTOTYPE_FEATURE_UTILITY_MECHANISM_STATUS",
    "PROTOTYPE_FEATURE_UTILITY_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA",
    "PROTOTYPE_FEATURE_UTILITY_TELEMETRY_COUNTER_NBYTES",
    "PrototypeFeatureUtilityAuditor",
    "PrototypeFeatureUtilityConfig",
    "PrototypeFeatureUtilityDiagnostics",
    "PrototypeFeatureUtilityEvent",
    "PrototypeFeatureUtilityResourceBudget",
    "PrototypeFeatureUtilityResult",
    "PrototypeFeatureUtilityState",
    "measure_prototype_feature_utility_state_nbytes",
    "migrate_legacy_prototype_feature_utility_config",
    "migrate_legacy_prototype_feature_utility_state",
    "prototype_feature_utility_counter_nbytes",
    "prototype_feature_utility_lifetime_counter_nbytes",
]
