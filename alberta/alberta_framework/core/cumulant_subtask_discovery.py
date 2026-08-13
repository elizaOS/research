# mypy: disable-error-code="attr-defined,call-arg"
"""Bounded proposal-only cumulant and subtask discovery.

Version 1 owns a fixed candidate universe and three matched-budget proposal
cohorts: discovered candidates, projections sampled once at initialization,
and caller-identity-bound hand-authored cumulants.  It deliberately has no
promotion, option, consumer, router, Horde, or go/no-go authority.

The update boundary is two phase.  :meth:`CumulantSubtaskDiscovery.arm` is
called after action selection and freezes every predict-before-update value.
``observe`` accepts the outcome only when its uint32 transition identity,
semantic generation, source digest, state revision, and canonical universe
digest exactly match the arm.  Runtime-invalid observations are atomic
no-ops.  Individual finite-but-unavailable evidence cells only freeze those
cells; availability flags never license a declared NaN or infinity.

Descriptors have four signed-int32 columns::

    [source_family, source_index, polarity, caller_tag]

``polarity`` is exactly ``-1`` or ``1``.  A reward-relevant descriptor refers
to a typed transition atom; environment reward is never a candidate source.
Reward/model targets are used only by a frozen pre-update insertion audit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

CUMULANT_SUBTASK_DISCOVERY_CONFIG_SCHEMA = "alberta.cumulant-subtask-discovery.config.v1"
CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA = "alberta.cumulant-subtask-discovery.state.v1"
CUMULANT_SUBTASK_DISCOVERY_RANKING_SEMANTICS = (
    "v1-fixed-family-order-fixed-quota-score-desc-descriptor-lexicographic-index"
)
CUMULANT_SUBTASK_DISCOVERY_AUTHORITY = False
CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY = False
CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY = False
CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED = False

CUMULANT_SOURCE_CONTROLLABLE_EVENT = 0
CUMULANT_SOURCE_FEATURE_CHANGE = 1
CUMULANT_SOURCE_REWARD_TRANSITION_ATOM = 2
CUMULANT_SOURCE_PREDICTION_BOTTLENECK = 3
CUMULANT_SOURCE_RANDOM_PROJECTION = 4
CUMULANT_SOURCE_HAND_AUTHORED = 5

_DISCOVERED_FAMILIES = (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
)
_DESCRIPTOR_WIDTH = 4
_INT32_MAX = 2**31 - 1
_MAX_CANDIDATES = 4_096
_MAX_OPTION_BUDGET = 1_024
_MAX_CELLS = 8_388_608
_MAX_DIMENSION = 4_096

Descriptor = tuple[int, int, int, int]


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact Python int")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact Python int")
    return value


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented) or (positive and represented <= 0.0):
        raise ValueError(f"{name} must remain valid in float32")
    return value


def _descriptor_tuple(value: object, *, name: str) -> tuple[Descriptor, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be a tuple or JSON list of descriptors")
    rows: list[Descriptor] = []
    for row_index, row in enumerate(cast(tuple[object, ...] | list[object], value)):
        if type(row) not in (tuple, list) or len(cast(Any, row)) != _DESCRIPTOR_WIDTH:
            raise ValueError(f"{name}[{row_index}] must contain four exact integers")
        raw = cast(tuple[object, ...] | list[object], row)
        if any(type(cell) is not int for cell in raw):
            raise ValueError(f"{name}[{row_index}] must contain four exact integers")
        rows.append(cast(Descriptor, tuple(cast(int, cell) for cell in raw)))
    return tuple(rows)


def _int_tuple(value: object, *, name: str, length: int) -> tuple[int, ...]:
    if type(value) not in (tuple, list) or len(cast(Any, value)) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    raw = cast(tuple[object, ...] | list[object], value)
    if any(type(cell) is not int for cell in raw):
        raise ValueError(f"{name} must contain exact Python integers")
    return tuple(cast(int, cell) for cell in raw)


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be a tuple or JSON list")
    raw = cast(tuple[object, ...] | list[object], value)
    if any(type(cell) is not float for cell in raw):
        raise ValueError(f"{name} must contain exact Python floats")
    return tuple(cast(float, cell) for cell in raw)


def _require_array_contract(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Apply trace-time shape and effective-dtype checks."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        array = jnp.asarray(value, dtype=jnp.int32)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.int32:
        raise TypeError(f"{name} must have dtype int32, got {array.dtype}")
    return array


def _float32_scalar(value: float | Array, *, name: str) -> Array:
    if type(value) is float:
        array = jnp.asarray(value, dtype=jnp.float32)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    return array


def _bool_scalar(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        array = jnp.asarray(value, dtype=jnp.bool_)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.bool_:
        raise TypeError(f"{name} must have dtype bool, got {array.dtype}")
    return array


def _uint32_words_to_bytes(words: Array) -> Array:
    shifts = jnp.asarray((0, 8, 16, 24), dtype=jnp.uint32)
    return cast(
        Array,
        ((words.reshape((-1, 1)) >> shifts) & jnp.uint32(0xFF))
        .astype(jnp.uint8)
        .reshape((-1,)),
    )


def _checksum_arrays(arrays: tuple[Array, ...], *, seed: Array) -> Array:
    """Return a deterministic two-word checksum usable inside JIT traces."""

    acc0 = seed[0] ^ jnp.uint32(0x9E3779B9)
    acc1 = seed[1] ^ jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.int32:
            words = array.astype(jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        mixed = words ^ (indices * jnp.uint32(0x165667B1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(mixed)
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), axis=0).astype(jnp.uint32)


def _persistent_resource_counts(
    *,
    raw_features: int,
    probe_features: int,
    actions: int,
    option_budget: int,
    candidates: int,
    incumbents: int,
    reward_tasks: int,
    model_tasks: int,
) -> tuple[int, int]:
    tasks = reward_tasks + model_tasks
    floats = (
        option_budget * raw_features
        + raw_features
        + candidates
        + candidates * (probe_features + actions)
        + candidates * 3
        + candidates * actions * 2
        + candidates * incumbents
        + candidates * candidates
        + candidates * (reward_tasks + model_tasks)
        + candidates * tasks
        + candidates * 3
    )
    four_byte_integers = (
        1
        + 2
        + 1
        + 2
        + 2
        + 2
        + candidates
        + candidates
        + candidates
        + candidates * actions
        + candidates * incumbents
        + candidates * candidates
        + candidates * tasks
        + candidates
    )
    uint8_digest = 32
    bools = 1 + raw_features + candidates
    logical = floats + four_byte_integers + uint8_digest + bools
    nbytes = 4 * floats + 4 * four_byte_integers + uint8_digest + bools
    return logical, nbytes


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantSubtaskDiscoveryConfig:
    """Static candidate universe, matched budgets, and frozen gate semantics."""

    raw_feature_dim: int
    probe_feature_dim: int
    n_actions: int
    controllable_event_dim: int
    transition_atom_dim: int
    prediction_bottleneck_dim: int
    option_budget: int
    family_quotas: tuple[int, int, int, int]
    controllable_event_descriptors: tuple[Descriptor, ...]
    feature_change_descriptors: tuple[Descriptor, ...]
    reward_transition_descriptors: tuple[Descriptor, ...]
    prediction_bottleneck_descriptors: tuple[Descriptor, ...]
    incumbent_descriptors: tuple[Descriptor, ...]
    hand_comparator_descriptors: tuple[Descriptor, ...]
    hand_comparator_identity: tuple[int, int]
    reward_task_weights: tuple[float, ...]
    model_task_weights: tuple[float, ...]
    probe_step_size: float = 0.05
    shadow_step_size: float = 0.05
    learnability_evidence_floor: int = 4
    controllability_evidence_floor_per_action: int = 2
    novelty_evidence_floor: int = 2
    contribution_evidence_floor: int = 2
    bottleneck_evidence_floor: int = 2
    learnability_threshold: float = 0.05
    baseline_variance_floor: float = 1.0e-6
    controllability_threshold: float = 0.05
    novelty_threshold: float = 1.0e-3
    contribution_threshold: float = 0.0
    bottleneck_epistemic_floor: float = 0.0
    bottleneck_progress_floor: float = 0.0
    bottleneck_aleatoric_ceiling: float = 1.0
    random_projection_scale: float = 1.0
    statistic_epsilon: float = 1.0e-6
    max_observations: int = 100_000

    SCHEMA_VERSION: ClassVar[str] = CUMULANT_SUBTASK_DISCOVERY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.family_quotas) is not tuple:
            raise ValueError("family_quotas must be an exact tuple")
        if type(self.hand_comparator_identity) is not tuple:
            raise ValueError("hand_comparator_identity must be an exact tuple")
        if type(self.reward_task_weights) is not tuple:
            raise ValueError("reward_task_weights must be an exact tuple")
        if type(self.model_task_weights) is not tuple:
            raise ValueError("model_task_weights must be an exact tuple")
        for name in (
            "controllable_event_descriptors",
            "feature_change_descriptors",
            "reward_transition_descriptors",
            "prediction_bottleneck_descriptors",
            "incumbent_descriptors",
            "hand_comparator_descriptors",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(row) is not tuple for row in value):
                raise ValueError(f"{name} must be a tuple of descriptor tuples")
            if _descriptor_tuple(value, name=name) != value:
                raise ValueError(f"{name} differs from the canonical descriptor encoding")
        dimensions = {
            "raw_feature_dim": self.raw_feature_dim,
            "probe_feature_dim": self.probe_feature_dim,
            "n_actions": self.n_actions,
            "controllable_event_dim": self.controllable_event_dim,
            "transition_atom_dim": self.transition_atom_dim,
            "prediction_bottleneck_dim": self.prediction_bottleneck_dim,
            "option_budget": self.option_budget,
        }
        for name, value in dimensions.items():
            _positive_int(value, name=name)
            if name != "option_budget" and value > _MAX_DIMENSION:
                raise ValueError(f"{name} exceeds the static 4096-cell dimension ceiling")
        if self.option_budget > _MAX_OPTION_BUDGET:
            raise ValueError("option_budget exceeds the static 1024-slot ceiling")
        quotas = _int_tuple(self.family_quotas, name="family_quotas", length=4)
        if any(quota < 1 for quota in quotas):
            raise ValueError("all four family_quotas must be positive")
        if sum(quotas) != self.option_budget:
            raise ValueError("family_quotas must sum exactly to option_budget")

        families = self.family_descriptors
        source_dims = (
            self.controllable_event_dim,
            self.raw_feature_dim,
            self.transition_atom_dim,
            self.prediction_bottleneck_dim,
        )
        for family, (rows, quota, source_dim) in enumerate(
            zip(families, quotas, source_dims, strict=True)
        ):
            if not rows:
                raise ValueError("all four discovered source families must be nonempty")
            if len(rows) < quota:
                raise ValueError("each source family must contain at least its fixed quota")
            for descriptor in rows:
                if any(not -(2**31) <= cell <= _INT32_MAX for cell in descriptor):
                    raise ValueError("descriptor cells must be signed-int32 compatible")
                if descriptor[0] != family:
                    raise ValueError("descriptor family column differs from its source family")
                if not 0 <= descriptor[1] < source_dim:
                    raise ValueError("descriptor source_index is out of range")
                if descriptor[2] not in (-1, 1):
                    raise ValueError("descriptor polarity must be exactly -1 or 1")
                if descriptor[3] < 0:
                    raise ValueError("descriptor caller_tag must be non-negative")

        candidate_count = sum(len(rows) for rows in families)
        if candidate_count > _MAX_CANDIDATES:
            raise ValueError("candidate universe exceeds the static 4096-candidate ceiling")
        for name, rows in (
            ("incumbent_descriptors", self.incumbent_descriptors),
            ("hand_comparator_descriptors", self.hand_comparator_descriptors),
        ):
            for descriptor in rows:
                if any(type(cell) is not int for cell in descriptor):
                    raise ValueError(f"{name} must contain exact integer descriptors")
                if any(not -(2**31) <= cell <= _INT32_MAX for cell in descriptor):
                    raise ValueError(f"{name} cells must be signed-int32 compatible")
                if descriptor[1] < 0:
                    raise ValueError(f"{name} source_index must be non-negative")
                if descriptor[2] not in (-1, 1):
                    raise ValueError(f"{name} polarity must be exactly -1 or 1")
                if descriptor[3] < 0:
                    raise ValueError(f"{name} caller_tag must be non-negative")
        if not self.incumbent_descriptors:
            raise ValueError("incumbent_descriptors must be nonempty for the novelty gate")
        if len(self.hand_comparator_descriptors) != self.option_budget:
            raise ValueError("hand comparator cohort must have exactly option_budget entries")
        discovered = self.candidate_descriptors
        if len(set(discovered)) != len(discovered):
            raise ValueError("discovered candidate descriptors must be globally unique")
        discovered_semantics = {descriptor[1:] for descriptor in discovered}
        incumbent_semantics = {descriptor[1:] for descriptor in self.incumbent_descriptors}
        hand_semantics = {descriptor[1:] for descriptor in self.hand_comparator_descriptors}
        if discovered_semantics & incumbent_semantics:
            raise ValueError("discovered descriptors collide with incumbent semantics")
        if len(discovered_semantics) != len(discovered):
            raise ValueError("discovered candidates contain duplicate canonical semantics")
        if discovered_semantics & hand_semantics:
            raise ValueError("discovered descriptors collide with hand-authored semantics")
        if len(hand_semantics) != len(self.hand_comparator_descriptors):
            raise ValueError("hand-authored comparator semantics must be unique")
        if any(row[0] != CUMULANT_SOURCE_HAND_AUTHORED for row in self.hand_comparator_descriptors):
            raise ValueError("hand comparator descriptors must use the hand-authored family id")
        identity = _int_tuple(
            self.hand_comparator_identity,
            name="hand_comparator_identity",
            length=2,
        )
        if any(not 0 <= cell <= 0xFFFFFFFF for cell in identity):
            raise ValueError("hand_comparator_identity cells must be uint32-compatible")

        weights = (*self.reward_task_weights, *self.model_task_weights)
        if not weights:
            raise ValueError("at least one frozen reward/model task channel is required")
        for weight in weights:
            _finite_float(weight, name="task weight")
            if weight < 0.0:
                raise ValueError("task weights must be non-negative")
        represented_mass = np.sum(
            np.asarray(weights, dtype=np.float32), dtype=np.float32
        )
        if represented_mass != np.float32(1.0):
            raise ValueError(
                "reward/model task weights must have float32 total mass exactly 1"
            )

        for name in (
            "learnability_evidence_floor",
            "controllability_evidence_floor_per_action",
            "novelty_evidence_floor",
            "contribution_evidence_floor",
            "bottleneck_evidence_floor",
            "max_observations",
        ):
            value = _positive_int(getattr(self, name), name=name)
            if value > _INT32_MAX - 1:
                raise ValueError(f"{name} exceeds the signed-int32 counter ceiling")
        if max(
            self.learnability_evidence_floor,
            self.controllability_evidence_floor_per_action,
            self.novelty_evidence_floor,
            self.contribution_evidence_floor,
            self.bottleneck_evidence_floor,
        ) > self.max_observations:
            raise ValueError("gate evidence floors must not exceed max_observations")

        for name in (
            "probe_step_size",
            "shadow_step_size",
            "baseline_variance_floor",
            "novelty_threshold",
            "random_projection_scale",
            "statistic_epsilon",
        ):
            _finite_float(getattr(self, name), name=name, positive=True)
        if self.probe_step_size > 1.0 or self.shadow_step_size > 1.0:
            raise ValueError("probe_step_size and shadow_step_size must be at most 1")
        for name in (
            "learnability_threshold",
            "controllability_threshold",
            "contribution_threshold",
            "bottleneck_epistemic_floor",
            "bottleneck_progress_floor",
            "bottleneck_aleatoric_ceiling",
        ):
            _finite_float(getattr(self, name), name=name)
        if self.learnability_threshold < 0.0 or self.learnability_threshold > 1.0:
            raise ValueError("learnability_threshold must be in [0, 1]")
        if self.controllability_threshold < 0.0:
            raise ValueError("controllability_threshold must be non-negative")
        if self.contribution_threshold < 0.0:
            raise ValueError("contribution_threshold must be non-negative")
        if self.bottleneck_epistemic_floor < 0.0 or self.bottleneck_progress_floor < 0.0:
            raise ValueError("bottleneck evidence thresholds must be non-negative")
        if self.bottleneck_aleatoric_ceiling < 0.0:
            raise ValueError("bottleneck_aleatoric_ceiling must be non-negative")

        incumbent_count = len(self.incumbent_descriptors)
        logical_cells, _ = _persistent_resource_counts(
            raw_features=self.raw_feature_dim,
            probe_features=self.probe_feature_dim,
            actions=self.n_actions,
            option_budget=self.option_budget,
            candidates=candidate_count,
            incumbents=incumbent_count,
            reward_tasks=len(self.reward_task_weights),
            model_tasks=len(self.model_task_weights),
        )
        if logical_cells > _MAX_CELLS:
            raise ValueError("configured discovery state exceeds the fixed cell ceiling")

    @property
    def family_descriptors(self) -> tuple[tuple[Descriptor, ...], ...]:
        return (
            self.controllable_event_descriptors,
            self.feature_change_descriptors,
            self.reward_transition_descriptors,
            self.prediction_bottleneck_descriptors,
        )

    @property
    def candidate_descriptors(self) -> tuple[Descriptor, ...]:
        return tuple(descriptor for family in self.family_descriptors for descriptor in family)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_descriptors)

    @property
    def task_count(self) -> int:
        return len(self.reward_task_weights) + len(self.model_task_weights)

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible version-1 configuration."""

        payload = dataclasses.asdict(self)
        payload["schema_version"] = self.SCHEMA_VERSION
        return payload

    @classmethod
    def from_config(cls, value: object) -> CumulantSubtaskDiscoveryConfig:
        """Reconstruct only an exact schema-v1 configuration."""

        if type(value) is not dict:
            raise ValueError("cumulant discovery config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {field.name for field in dataclasses.fields(cls)} | {"schema_version"}
        if set(raw) != expected:
            raise ValueError("cumulant discovery config keys differ from schema v1")
        if raw["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("cumulant discovery config schema_version differs")
        kwargs = {key: raw[key] for key in expected if key != "schema_version"}
        for name in (
            "controllable_event_descriptors",
            "feature_change_descriptors",
            "reward_transition_descriptors",
            "prediction_bottleneck_descriptors",
            "incumbent_descriptors",
            "hand_comparator_descriptors",
        ):
            kwargs[name] = _descriptor_tuple(kwargs[name], name=name)
        kwargs["family_quotas"] = cast(
            tuple[int, int, int, int],
            _int_tuple(kwargs["family_quotas"], name="family_quotas", length=4),
        )
        kwargs["hand_comparator_identity"] = cast(
            tuple[int, int],
            _int_tuple(
                kwargs["hand_comparator_identity"],
                name="hand_comparator_identity",
                length=2,
            ),
        )
        kwargs["reward_task_weights"] = _float_tuple(
            kwargs["reward_task_weights"], name="reward_task_weights"
        )
        kwargs["model_task_weights"] = _float_tuple(
            kwargs["model_task_weights"], name="model_task_weights"
        )
        return cls(**cast(Any, kwargs))


@chex.dataclass(frozen=True)
class CumulantSubtaskDiscoveryState:
    """Fixed-shape persistent state for one bound candidate universe."""

    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 32"]
    observation_count: Int[Array, ""]
    has_last_transition: Bool[Array, ""]
    last_transition_id: UInt[Array, " 2"]
    random_projection_key: UInt[Array, " 2"]
    random_projection_digest: UInt[Array, " 2"]
    random_projections: Float[Array, "option_budget raw_feature_dim"]
    last_raw_features: Float[Array, " raw_feature_dim"]
    last_raw_available: Bool[Array, " raw_feature_dim"]
    reward_birth_observations: Int[Array, " candidate_count"]
    last_candidate_values: Float[Array, " candidate_count"]
    last_candidate_available: Bool[Array, " candidate_count"]
    probe_weights: Float[Array, "candidate_count probe_design_dim"]
    candidate_means: Float[Array, " candidate_count"]
    candidate_value_counts: Int[Array, " candidate_count"]
    probe_squared_error_sums: Float[Array, " candidate_count"]
    baseline_squared_error_sums: Float[Array, " candidate_count"]
    learnability_counts: Int[Array, " candidate_count"]
    action_outcome_weighted_sums: Float[Array, "candidate_count n_actions"]
    action_importance_masses: Float[Array, "candidate_count n_actions"]
    action_evidence_counts: Int[Array, "candidate_count n_actions"]
    incumbent_novelty_sums: Float[Array, "candidate_count incumbent_count"]
    incumbent_novelty_counts: Int[Array, "candidate_count incumbent_count"]
    pair_novelty_sums: Float[Array, "candidate_count candidate_count"]
    pair_novelty_counts: Int[Array, "candidate_count candidate_count"]
    reward_shadow_weights: Float[Array, "candidate_count reward_channel_count"]
    model_shadow_weights: Float[Array, "candidate_count model_channel_count"]
    task_contribution_sums: Float[Array, "candidate_count task_count"]
    task_contribution_counts: Int[Array, "candidate_count task_count"]
    bottleneck_epistemic_sums: Float[Array, " candidate_count"]
    bottleneck_progress_sums: Float[Array, " candidate_count"]
    bottleneck_aleatoric_sums: Float[Array, " candidate_count"]
    bottleneck_evidence_counts: Int[Array, " candidate_count"]


@chex.dataclass(frozen=True)
class CumulantSubtaskDiscoveryArm:
    """Frozen pre-outcome cache for exactly one transition identity."""

    available: Bool[Array, ""]
    transition_id: UInt[Array, " 2"]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 32"]
    cache_digest: UInt[Array, " 2"]
    state_observation_count: Int[Array, ""]
    action: Int[Array, ""]
    behavior_propensity: Float[Array, ""]
    randomized: Bool[Array, ""]
    current_raw_features: Float[Array, " raw_feature_dim"]
    current_raw_available: Bool[Array, " raw_feature_dim"]
    current_controllable_events: Float[Array, " controllable_event_dim"]
    current_controllable_events_available: Bool[Array, " controllable_event_dim"]
    current_transition_atoms: Float[Array, " transition_atom_dim"]
    current_transition_atoms_available: Bool[Array, " transition_atom_dim"]
    current_bottleneck_values: Float[Array, " prediction_bottleneck_dim"]
    current_bottleneck_available: Bool[Array, " prediction_bottleneck_dim"]
    probe_features: Float[Array, " probe_feature_dim"]
    probe_design: Float[Array, " probe_design_dim"]
    current_candidate_values: Float[Array, " candidate_count"]
    current_candidate_available: Bool[Array, " candidate_count"]
    probe_predictions: Float[Array, " candidate_count"]
    baseline_predictions: Float[Array, " candidate_count"]
    current_random_values: Float[Array, " option_budget"]
    current_hand_values: Float[Array, " option_budget"]
    current_hand_available: Bool[Array, " option_budget"]
    current_incumbent_values: Float[Array, " incumbent_count"]
    current_incumbent_available: Bool[Array, " incumbent_count"]
    frozen_reward_base_predictions: Float[Array, " reward_channel_count"]
    frozen_reward_inserted_predictions: Float[Array, "candidate_count reward_channel_count"]
    frozen_model_base_predictions: Float[Array, " model_channel_count"]
    frozen_model_inserted_predictions: Float[Array, "candidate_count model_channel_count"]
    reward_birth_observations: Int[Array, " candidate_count"]
    hand_comparator_identity: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantSubtaskProposalBundle:
    """One exact-budget proposal cohort; candidate IDs never become tail slots."""

    ready: Bool[Array, ""]
    cohort_id: Int[Array, ""]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 32"]
    transition_id: UInt[Array, " 2"]
    state_observation_count: Int[Array, ""]
    binding_digest: UInt[Array, " 2"]
    selected_candidate_indices: Int[Array, " option_budget"]
    selected_family_ids: Int[Array, " option_budget"]
    selected_descriptors: Int[Array, "option_budget 4"]
    selected_scores: Float[Array, " option_budget"]
    selected_cumulants: Float[Array, " option_budget"]
    tail_slot_indices: Int[Array, " option_budget"]


@chex.dataclass(frozen=True)
class CumulantSubtaskDiscoveryDiagnostics:
    """Primitive validity, gate, quota, and matched-cohort facts."""

    transaction_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    capacity_capped: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    arm_valid: Bool[Array, ""]
    arm_cache_valid: Bool[Array, ""]
    transition_identity_matches: Bool[Array, ""]
    source_binding_matches: Bool[Array, ""]
    hand_identity_matches: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    reward_births_this_transition: Bool[Array, " candidate_count"]
    semantic_available: Bool[Array, " candidate_count"]
    learnability_ready: Bool[Array, " candidate_count"]
    controllability_ready: Bool[Array, " candidate_count"]
    novelty_against_incumbents_ready: Bool[Array, " candidate_count"]
    contribution_ready: Bool[Array, " candidate_count"]
    bottleneck_ready: Bool[Array, " candidate_count"]
    all_local_gates_ready: Bool[Array, " candidate_count"]
    selected_mask: Bool[Array, " candidate_count"]
    family_selected_counts: Int[Array, " 4"]
    family_quotas: Int[Array, " 4"]
    bundle_ready: Bool[Array, ""]
    random_comparator_ready: Bool[Array, ""]
    hand_comparator_ready: Bool[Array, ""]
    candidate_scores: Float[Array, " candidate_count"]
    learnability_scores: Float[Array, " candidate_count"]
    controllability_scores: Float[Array, " candidate_count"]
    novelty_scores: Float[Array, " candidate_count"]
    contribution_scores: Float[Array, " candidate_count"]


@chex.dataclass(frozen=True)
class CumulantSubtaskDiscoveryResult:
    """Atomic next state and three matched-budget proposal cohorts."""

    state: CumulantSubtaskDiscoveryState
    discovered: CumulantSubtaskProposalBundle
    random_comparator: CumulantSubtaskProposalBundle
    hand_comparator: CumulantSubtaskProposalBundle
    diagnostics: CumulantSubtaskDiscoveryDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantSubtaskDiscoveryResourceBudget:
    """Exact logical resource and no-authority declaration for version 1."""

    persistent_logical_scalars: int
    persistent_state_nbytes: int
    candidate_count: int
    option_budget: int
    task_count: int
    pair_novelty_cells: int
    incumbent_novelty_cells: int
    per_action_moment_cells: int
    task_shadow_cells: int
    random_draws_at_init: int
    random_generator_calls_at_init: int
    random_generator_calls_per_arm: int
    random_generator_calls_per_observe: int
    projection_checksum_cells_per_state_validation: int
    state_validation_calls_per_arm: int
    state_validation_calls_per_observe: int
    probe_forward_evaluations_per_arm: int
    shadow_forward_evaluations_per_arm: int
    probe_updates_per_observe: int
    shadow_updates_per_observe: int
    backward_passes_per_observe: int
    consumer_updates_per_observe: int
    router_calls_per_observe: int
    horde_updates_per_observe: int
    option_updates_per_observe: int
    promotion_decisions_per_observe: int
    curation_authority: bool
    promotion_authority: bool
    go_no_go_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str
    ranking_semantics: str


class CumulantSubtaskDiscovery:
    """Fixed-budget JAX discovery mechanism; method bodies follow below."""

    def __init__(self, config: CumulantSubtaskDiscoveryConfig) -> None:
        if type(config) is not CumulantSubtaskDiscoveryConfig:
            raise TypeError("config must be an exact CumulantSubtaskDiscoveryConfig")
        self._config = config
        canonical = json.dumps(config.to_config(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        self._config_digest_bytes = tuple(digest)
        self._descriptors = jnp.asarray(config.candidate_descriptors, dtype=jnp.int32)
        self._families = self._descriptors[:, 0]
        self._source_indices = self._descriptors[:, 1]
        self._polarities = self._descriptors[:, 2].astype(jnp.float32)
        self._incumbent_descriptors = jnp.asarray(
            config.incumbent_descriptors, dtype=jnp.int32
        ).reshape((-1, _DESCRIPTOR_WIDTH))
        self._hand_descriptors = jnp.asarray(
            config.hand_comparator_descriptors, dtype=jnp.int32
        )

    @property
    def config(self) -> CumulantSubtaskDiscoveryConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> CumulantSubtaskDiscovery:
        return cls(CumulantSubtaskDiscoveryConfig.from_config(value))

    @property
    def random_comparator_descriptors(self) -> Array:
        indices = jnp.arange(self._config.option_budget, dtype=jnp.int32)
        return jnp.stack(
            (
                jnp.full_like(indices, CUMULANT_SOURCE_RANDOM_PROJECTION),
                indices,
                jnp.ones_like(indices),
                jnp.zeros_like(indices),
            ),
            axis=1,
        )

    @property
    def hand_comparator_descriptors(self) -> Array:
        return self._hand_descriptors

    @property
    def resource_budget(self) -> CumulantSubtaskDiscoveryResourceBudget:
        """Return the exact fixed logical state and authority declaration."""

        cfg = self._config
        candidates = cfg.candidate_count
        actions = cfg.n_actions
        incumbents = len(cfg.incumbent_descriptors)
        rewards = len(cfg.reward_task_weights)
        models = len(cfg.model_task_weights)
        tasks = cfg.task_count
        logical, nbytes = _persistent_resource_counts(
            raw_features=cfg.raw_feature_dim,
            probe_features=cfg.probe_feature_dim,
            actions=actions,
            option_budget=cfg.option_budget,
            candidates=candidates,
            incumbents=incumbents,
            reward_tasks=rewards,
            model_tasks=models,
        )
        return CumulantSubtaskDiscoveryResourceBudget(
            persistent_logical_scalars=logical,
            persistent_state_nbytes=nbytes,
            candidate_count=candidates,
            option_budget=cfg.option_budget,
            task_count=tasks,
            pair_novelty_cells=candidates * candidates,
            incumbent_novelty_cells=candidates * incumbents,
            per_action_moment_cells=3 * candidates * actions,
            task_shadow_cells=candidates * (rewards + models + 2 * tasks),
            random_draws_at_init=cfg.option_budget * cfg.raw_feature_dim,
            random_generator_calls_at_init=1,
            random_generator_calls_per_arm=0,
            random_generator_calls_per_observe=0,
            projection_checksum_cells_per_state_validation=(
                cfg.option_budget * cfg.raw_feature_dim
            ),
            state_validation_calls_per_arm=1,
            state_validation_calls_per_observe=2,
            probe_forward_evaluations_per_arm=candidates,
            shadow_forward_evaluations_per_arm=candidates * tasks,
            probe_updates_per_observe=candidates,
            shadow_updates_per_observe=candidates * tasks,
            backward_passes_per_observe=0,
            consumer_updates_per_observe=0,
            router_calls_per_observe=0,
            horde_updates_per_observe=0,
            option_updates_per_observe=0,
            promotion_decisions_per_observe=0,
            curation_authority=False,
            promotion_authority=False,
            go_no_go_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA,
            ranking_semantics=CUMULANT_SUBTASK_DISCOVERY_RANKING_SEMANTICS,
        )

    def _expected_random_projections(self, key_data: Array) -> Array:
        key = jr.wrap_key_data(key_data)
        return (
            jr.normal(
                key,
                (self._config.option_budget, self._config.raw_feature_dim),
                dtype=jnp.float32,
            )
            * jnp.asarray(self._config.random_projection_scale, dtype=jnp.float32)
        )

    def _canonical_digest(
        self,
        projection_key: Array,
        projection_digest: Array,
        semantic_generation: Array,
        source_digest: Array,
    ) -> Array:
        config_bytes = jnp.asarray(
            self._config_digest_bytes,
            dtype=jnp.uint8,
        )
        generation_bytes = _uint32_words_to_bytes(
            semantic_generation.astype(jnp.uint32).reshape((1,))
        )
        key_bytes = _uint32_words_to_bytes(projection_key)
        projection_bytes = _uint32_words_to_bytes(projection_digest)
        source_bytes = _uint32_words_to_bytes(source_digest)
        binding = jnp.concatenate(
            (generation_bytes, key_bytes, projection_bytes, source_bytes), axis=0
        )
        tiled = jnp.tile(binding, (32 + binding.shape[0] - 1) // binding.shape[0])[:32]
        return jnp.bitwise_xor(config_bytes, tiled)

    def init(
        self,
        key: Array,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> CumulantSubtaskDiscoveryState:
        """Sample the frozen comparator once and initialize a bound state."""

        key_data = _require_array_contract(
            jr.key_data(key), name="key data", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(semantic_generation, name="semantic_generation")
        source = _require_array_contract(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        cfg = self._config
        candidates = cfg.candidate_count
        actions = cfg.n_actions
        incumbents = len(cfg.incumbent_descriptors)
        rewards = len(cfg.reward_task_weights)
        models = len(cfg.model_task_weights)
        tasks = cfg.task_count
        projections = self._expected_random_projections(key_data)
        projection_digest = _checksum_arrays((projections,), seed=key_data)
        reward_family = self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM
        births = jnp.where(reward_family, -1, 0).astype(jnp.int32)
        return CumulantSubtaskDiscoveryState(
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=self._canonical_digest(
                key_data, projection_digest, generation, source
            ),
            observation_count=jnp.asarray(0, dtype=jnp.int32),
            has_last_transition=jnp.asarray(False, dtype=jnp.bool_),
            last_transition_id=jnp.zeros((2,), dtype=jnp.uint32),
            random_projection_key=key_data,
            random_projection_digest=projection_digest,
            random_projections=projections,
            last_raw_features=jnp.zeros((cfg.raw_feature_dim,), dtype=jnp.float32),
            last_raw_available=jnp.zeros((cfg.raw_feature_dim,), dtype=jnp.bool_),
            reward_birth_observations=births,
            last_candidate_values=jnp.zeros((candidates,), dtype=jnp.float32),
            last_candidate_available=jnp.zeros((candidates,), dtype=jnp.bool_),
            probe_weights=jnp.zeros(
                (candidates, cfg.probe_feature_dim + actions), dtype=jnp.float32
            ),
            candidate_means=jnp.zeros((candidates,), dtype=jnp.float32),
            candidate_value_counts=jnp.zeros((candidates,), dtype=jnp.int32),
            probe_squared_error_sums=jnp.zeros((candidates,), dtype=jnp.float32),
            baseline_squared_error_sums=jnp.zeros((candidates,), dtype=jnp.float32),
            learnability_counts=jnp.zeros((candidates,), dtype=jnp.int32),
            action_outcome_weighted_sums=jnp.zeros(
                (candidates, actions), dtype=jnp.float32
            ),
            action_importance_masses=jnp.zeros(
                (candidates, actions), dtype=jnp.float32
            ),
            action_evidence_counts=jnp.zeros((candidates, actions), dtype=jnp.int32),
            incumbent_novelty_sums=jnp.zeros(
                (candidates, incumbents), dtype=jnp.float32
            ),
            incumbent_novelty_counts=jnp.zeros(
                (candidates, incumbents), dtype=jnp.int32
            ),
            pair_novelty_sums=jnp.zeros(
                (candidates, candidates), dtype=jnp.float32
            ),
            pair_novelty_counts=jnp.zeros(
                (candidates, candidates), dtype=jnp.int32
            ),
            reward_shadow_weights=jnp.zeros((candidates, rewards), dtype=jnp.float32),
            model_shadow_weights=jnp.zeros((candidates, models), dtype=jnp.float32),
            task_contribution_sums=jnp.zeros((candidates, tasks), dtype=jnp.float32),
            task_contribution_counts=jnp.zeros((candidates, tasks), dtype=jnp.int32),
            bottleneck_epistemic_sums=jnp.zeros((candidates,), dtype=jnp.float32),
            bottleneck_progress_sums=jnp.zeros((candidates,), dtype=jnp.float32),
            bottleneck_aleatoric_sums=jnp.zeros((candidates,), dtype=jnp.float32),
            bottleneck_evidence_counts=jnp.zeros((candidates,), dtype=jnp.int32),
        )

    def _check_state_contract(self, state: CumulantSubtaskDiscoveryState) -> None:
        if type(state) is not CumulantSubtaskDiscoveryState:
            raise TypeError("state must be an exact CumulantSubtaskDiscoveryState")
        cfg = self._config
        c = cfg.candidate_count
        a = cfg.n_actions
        i = len(cfg.incumbent_descriptors)
        r = len(cfg.reward_task_weights)
        m = len(cfg.model_task_weights)
        contracts = (
            (state.semantic_generation, "semantic_generation", (), jnp.int32),
            (state.source_digest, "source_digest", (2,), jnp.uint32),
            (state.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (state.observation_count, "observation_count", (), jnp.int32),
            (state.has_last_transition, "has_last_transition", (), jnp.bool_),
            (state.last_transition_id, "last_transition_id", (2,), jnp.uint32),
            (state.random_projection_key, "random_projection_key", (2,), jnp.uint32),
            (
                state.random_projection_digest,
                "random_projection_digest",
                (2,),
                jnp.uint32,
            ),
            (
                state.random_projections,
                "random_projections",
                (cfg.option_budget, cfg.raw_feature_dim),
                jnp.float32,
            ),
            (state.last_raw_features, "last_raw_features", (cfg.raw_feature_dim,), jnp.float32),
            (state.last_raw_available, "last_raw_available", (cfg.raw_feature_dim,), jnp.bool_),
            (state.reward_birth_observations, "reward_birth_observations", (c,), jnp.int32),
            (state.last_candidate_values, "last_candidate_values", (c,), jnp.float32),
            (state.last_candidate_available, "last_candidate_available", (c,), jnp.bool_),
            (
                state.probe_weights,
                "probe_weights",
                (c, cfg.probe_feature_dim + a),
                jnp.float32,
            ),
            (state.candidate_means, "candidate_means", (c,), jnp.float32),
            (state.candidate_value_counts, "candidate_value_counts", (c,), jnp.int32),
            (state.probe_squared_error_sums, "probe_squared_error_sums", (c,), jnp.float32),
            (
                state.baseline_squared_error_sums,
                "baseline_squared_error_sums",
                (c,),
                jnp.float32,
            ),
            (state.learnability_counts, "learnability_counts", (c,), jnp.int32),
            (
                state.action_outcome_weighted_sums,
                "action_outcome_weighted_sums",
                (c, a),
                jnp.float32,
            ),
            (state.action_importance_masses, "action_importance_masses", (c, a), jnp.float32),
            (state.action_evidence_counts, "action_evidence_counts", (c, a), jnp.int32),
            (state.incumbent_novelty_sums, "incumbent_novelty_sums", (c, i), jnp.float32),
            (
                state.incumbent_novelty_counts,
                "incumbent_novelty_counts",
                (c, i),
                jnp.int32,
            ),
            (state.pair_novelty_sums, "pair_novelty_sums", (c, c), jnp.float32),
            (state.pair_novelty_counts, "pair_novelty_counts", (c, c), jnp.int32),
            (state.reward_shadow_weights, "reward_shadow_weights", (c, r), jnp.float32),
            (state.model_shadow_weights, "model_shadow_weights", (c, m), jnp.float32),
            (
                state.task_contribution_sums,
                "task_contribution_sums",
                (c, cfg.task_count),
                jnp.float32,
            ),
            (
                state.task_contribution_counts,
                "task_contribution_counts",
                (c, cfg.task_count),
                jnp.int32,
            ),
            (state.bottleneck_epistemic_sums, "bottleneck_epistemic_sums", (c,), jnp.float32),
            (state.bottleneck_progress_sums, "bottleneck_progress_sums", (c,), jnp.float32),
            (state.bottleneck_aleatoric_sums, "bottleneck_aleatoric_sums", (c,), jnp.float32),
            (state.bottleneck_evidence_counts, "bottleneck_evidence_counts", (c,), jnp.int32),
        )
        for value, name, shape, dtype in contracts:
            _require_array_contract(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def validate_state(
        self,
        state: CumulantSubtaskDiscoveryState,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> Array:
        """Return a scalar dynamic validity result for a structurally exact state."""

        self._check_state_contract(state)
        generation = _int32_scalar(semantic_generation, name="semantic_generation")
        source = _require_array_contract(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        cfg = self._config
        float_leaves = (
            state.random_projections,
            state.last_raw_features,
            state.last_candidate_values,
            state.probe_weights,
            state.candidate_means,
            state.probe_squared_error_sums,
            state.baseline_squared_error_sums,
            state.action_outcome_weighted_sums,
            state.action_importance_masses,
            state.incumbent_novelty_sums,
            state.pair_novelty_sums,
            state.reward_shadow_weights,
            state.model_shadow_weights,
            state.task_contribution_sums,
            state.bottleneck_epistemic_sums,
            state.bottleneck_progress_sums,
            state.bottleneck_aleatoric_sums,
        )
        finite = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in float_leaves:
            finite = finite & jnp.all(jnp.isfinite(leaf))
        count_leaves = (
            state.candidate_value_counts,
            state.learnability_counts,
            state.action_evidence_counts,
            state.incumbent_novelty_counts,
            state.pair_novelty_counts,
            state.task_contribution_counts,
            state.bottleneck_evidence_counts,
        )
        counts_valid = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in count_leaves:
            counts_valid = counts_valid & jnp.all(
                (leaf >= 0) & (leaf <= state.observation_count)
            )
        reward_family = self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM
        births_valid = jnp.all(
            jnp.where(
                reward_family,
                (state.reward_birth_observations == -1)
                | (
                    (state.reward_birth_observations >= 1)
                    & (state.reward_birth_observations <= state.observation_count)
                ),
                state.reward_birth_observations == 0,
            )
        )
        initial_float_leaves = (
            state.last_raw_features,
            state.last_candidate_values,
            state.probe_weights,
            state.candidate_means,
            state.probe_squared_error_sums,
            state.baseline_squared_error_sums,
            state.action_outcome_weighted_sums,
            state.action_importance_masses,
            state.incumbent_novelty_sums,
            state.pair_novelty_sums,
            state.reward_shadow_weights,
            state.model_shadow_weights,
            state.task_contribution_sums,
            state.bottleneck_epistemic_sums,
            state.bottleneck_progress_sums,
            state.bottleneck_aleatoric_sums,
        )
        initial_values_zero = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in initial_float_leaves:
            initial_values_zero = initial_values_zero & jnp.all(leaf == 0.0)
        expected_projection_digest = _checksum_arrays(
            (state.random_projections,), seed=state.random_projection_key
        )
        projection_valid = jnp.array_equal(
            state.random_projection_digest, expected_projection_digest
        )
        expected_digest = self._canonical_digest(
            state.random_projection_key,
            state.random_projection_digest,
            state.semantic_generation,
            state.source_digest,
        )
        return (
            finite
            & counts_valid
            & births_valid
            & (state.semantic_generation >= 0)
            & (state.semantic_generation == generation)
            & jnp.array_equal(state.source_digest, source)
            & jnp.array_equal(state.canonical_digest, expected_digest)
            & (state.observation_count >= 0)
            & (state.observation_count <= cfg.max_observations)
            & jnp.all(state.action_importance_masses >= 0.0)
            & jnp.all(state.probe_squared_error_sums >= 0.0)
            & jnp.all(state.baseline_squared_error_sums >= 0.0)
            & jnp.all(state.incumbent_novelty_sums >= 0.0)
            & jnp.all(state.pair_novelty_sums >= 0.0)
            & jnp.all(state.bottleneck_epistemic_sums >= 0.0)
            & jnp.all(state.bottleneck_progress_sums >= 0.0)
            & jnp.all(state.bottleneck_aleatoric_sums >= 0.0)
            & (state.has_last_transition == (state.observation_count > 0))
            & jnp.array_equal(state.candidate_value_counts, state.learnability_counts)
            & jnp.all(
                jnp.sum(state.action_evidence_counts, axis=1)
                <= state.observation_count
            )
            & jnp.array_equal(
                state.pair_novelty_counts, state.pair_novelty_counts.T
            )
            & jnp.array_equal(state.pair_novelty_sums, state.pair_novelty_sums.T)
            & jnp.all(jnp.diag(state.pair_novelty_counts) == 0)
            & jnp.all(jnp.diag(state.pair_novelty_sums) == 0.0)
            & jnp.where(
                state.observation_count == 0,
                ~jnp.any(state.last_raw_available)
                & ~jnp.any(state.last_candidate_available)
                & jnp.all(state.last_transition_id == 0)
                & initial_values_zero
                & jnp.all(
                    jnp.where(
                        reward_family,
                        state.reward_birth_observations == -1,
                        state.reward_birth_observations == 0,
                    )
                ),
                True,
            )
            & projection_valid
        )

    def _direct_semantics(
        self,
        *,
        raw_features: Array,
        raw_available: Array,
        controllable_events: Array,
        controllable_events_available: Array,
        transition_atoms: Array,
        transition_atoms_available: Array,
        bottleneck_values: Array,
        bottleneck_available: Array,
    ) -> tuple[Array, Array]:
        indices = self._source_indices
        event_values = controllable_events[
            jnp.clip(indices, 0, self._config.controllable_event_dim - 1)
        ]
        event_available = controllable_events_available[
            jnp.clip(indices, 0, self._config.controllable_event_dim - 1)
        ]
        raw_values = raw_features[jnp.clip(indices, 0, self._config.raw_feature_dim - 1)]
        raw_cells_available = raw_available[
            jnp.clip(indices, 0, self._config.raw_feature_dim - 1)
        ]
        atom_values = transition_atoms[jnp.clip(indices, 0, self._config.transition_atom_dim - 1)]
        atom_available = transition_atoms_available[
            jnp.clip(indices, 0, self._config.transition_atom_dim - 1)
        ]
        bottleneck_source_values = bottleneck_values[
            jnp.clip(indices, 0, self._config.prediction_bottleneck_dim - 1)
        ]
        bottleneck_cells_available = bottleneck_available[
            jnp.clip(indices, 0, self._config.prediction_bottleneck_dim - 1)
        ]
        values = jnp.where(
            self._families == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            event_values,
            jnp.where(
                self._families == CUMULANT_SOURCE_FEATURE_CHANGE,
                raw_values,
                jnp.where(
                    self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    atom_values,
                    bottleneck_source_values,
                ),
            ),
        ) * self._polarities
        available = jnp.where(
            self._families == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            event_available,
            jnp.where(
                self._families == CUMULANT_SOURCE_FEATURE_CHANGE,
                raw_cells_available,
                jnp.where(
                    self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    atom_available,
                    bottleneck_cells_available,
                ),
            ),
        )
        return values, available

    def _empty_bundle(self, cohort_id: int) -> CumulantSubtaskProposalBundle:
        budget = self._config.option_budget
        return CumulantSubtaskProposalBundle(
            ready=jnp.asarray(False, dtype=jnp.bool_),
            cohort_id=jnp.asarray(cohort_id, dtype=jnp.int32),
            semantic_generation=jnp.asarray(-1, dtype=jnp.int32),
            source_digest=jnp.zeros((2,), dtype=jnp.uint32),
            canonical_digest=jnp.zeros((32,), dtype=jnp.uint8),
            transition_id=jnp.zeros((2,), dtype=jnp.uint32),
            state_observation_count=jnp.asarray(-1, dtype=jnp.int32),
            binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
            selected_candidate_indices=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_family_ids=jnp.full((budget,), cohort_id, dtype=jnp.int32),
            selected_descriptors=jnp.zeros((budget, _DESCRIPTOR_WIDTH), dtype=jnp.int32),
            selected_scores=jnp.zeros((budget,), dtype=jnp.float32),
            selected_cumulants=jnp.zeros((budget,), dtype=jnp.float32),
            tail_slot_indices=jnp.arange(
                self._config.raw_feature_dim,
                self._config.raw_feature_dim + budget,
                dtype=jnp.int32,
            ),
        )

    def empty_proposal_bundle(
        self,
        cohort_id: int = -1,
    ) -> CumulantSubtaskProposalBundle:
        """Return the canonical unavailable bundle for one declared cohort."""

        if type(cohort_id) is not int or cohort_id not in (
            -1,
            CUMULANT_SOURCE_RANDOM_PROJECTION,
            CUMULANT_SOURCE_HAND_AUTHORED,
        ):
            raise ValueError("cohort_id must identify discovered, random, or hand proposals")
        return self._empty_bundle(cohort_id)

    def check_proposal_bundle_contract(
        self,
        bundle: CumulantSubtaskProposalBundle,
    ) -> None:
        """Validate only the public fixed-shape proposal type contract."""

        self._check_bundle_contract(bundle)

    def arm(
        self,
        state: CumulantSubtaskDiscoveryState,
        *,
        current_raw_features: Array,
        current_raw_available: Array,
        current_controllable_events: Array,
        current_controllable_events_available: Array,
        current_transition_atoms: Array,
        current_transition_atoms_available: Array,
        current_bottleneck_values: Array,
        current_bottleneck_available: Array,
        probe_features: Array,
        current_incumbent_values: Array,
        current_incumbent_available: Array,
        current_hand_values: Array,
        current_hand_available: Array,
        hand_comparator_identity: Array,
        reward_base_predictions: Array,
        model_base_predictions: Array,
        action: int | Array,
        behavior_propensity: float | Array,
        randomized: bool | Array,
        transition_id: Array,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> CumulantSubtaskDiscoveryArm:
        """Freeze all pre-outcome values after the behavior action is selected."""

        self._check_state_contract(state)
        cfg = self._config
        i = len(cfg.incumbent_descriptors)
        r = len(cfg.reward_task_weights)
        m = len(cfg.model_task_weights)
        raw = _require_array_contract(
            current_raw_features,
            name="current_raw_features",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.float32,
        )
        raw_available = _require_array_contract(
            current_raw_available,
            name="current_raw_available",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.bool_,
        )
        events = _require_array_contract(
            current_controllable_events,
            name="current_controllable_events",
            shape=(cfg.controllable_event_dim,),
            dtype=jnp.float32,
        )
        events_available = _require_array_contract(
            current_controllable_events_available,
            name="current_controllable_events_available",
            shape=(cfg.controllable_event_dim,),
            dtype=jnp.bool_,
        )
        atoms = _require_array_contract(
            current_transition_atoms,
            name="current_transition_atoms",
            shape=(cfg.transition_atom_dim,),
            dtype=jnp.float32,
        )
        atoms_available = _require_array_contract(
            current_transition_atoms_available,
            name="current_transition_atoms_available",
            shape=(cfg.transition_atom_dim,),
            dtype=jnp.bool_,
        )
        bottlenecks = _require_array_contract(
            current_bottleneck_values,
            name="current_bottleneck_values",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.float32,
        )
        bottlenecks_available = _require_array_contract(
            current_bottleneck_available,
            name="current_bottleneck_available",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.bool_,
        )
        probe = _require_array_contract(
            probe_features,
            name="probe_features",
            shape=(cfg.probe_feature_dim,),
            dtype=jnp.float32,
        )
        incumbent_values = _require_array_contract(
            current_incumbent_values,
            name="current_incumbent_values",
            shape=(i,),
            dtype=jnp.float32,
        )
        incumbent_available = _require_array_contract(
            current_incumbent_available,
            name="current_incumbent_available",
            shape=(i,),
            dtype=jnp.bool_,
        )
        hand_values = _require_array_contract(
            current_hand_values,
            name="current_hand_values",
            shape=(cfg.option_budget,),
            dtype=jnp.float32,
        )
        hand_available = _require_array_contract(
            current_hand_available,
            name="current_hand_available",
            shape=(cfg.option_budget,),
            dtype=jnp.bool_,
        )
        hand_identity = _require_array_contract(
            hand_comparator_identity,
            name="hand_comparator_identity",
            shape=(2,),
            dtype=jnp.uint32,
        )
        reward_predictions = _require_array_contract(
            reward_base_predictions,
            name="reward_base_predictions",
            shape=(r,),
            dtype=jnp.float32,
        )
        model_predictions = _require_array_contract(
            model_base_predictions,
            name="model_base_predictions",
            shape=(m,),
            dtype=jnp.float32,
        )
        selected_action = _int32_scalar(action, name="action")
        propensity = _float32_scalar(behavior_propensity, name="behavior_propensity")
        randomized_flag = _bool_scalar(randomized, name="randomized")
        identity = _require_array_contract(
            transition_id, name="transition_id", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(semantic_generation, name="semantic_generation")
        source = _require_array_contract(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )

        direct_values, direct_available = self._direct_semantics(
            raw_features=raw,
            raw_available=raw_available,
            controllable_events=events,
            controllable_events_available=events_available,
            transition_atoms=atoms,
            transition_atoms_available=atoms_available,
            bottleneck_values=bottlenecks,
            bottleneck_available=bottlenecks_available,
        )
        non_change = self._families != CUMULANT_SOURCE_FEATURE_CHANGE
        direct_cache_match = jnp.all(
            jnp.where(
                non_change,
                (direct_available == state.last_candidate_available)
                & jnp.where(
                    direct_available,
                    direct_values == state.last_candidate_values,
                    True,
                ),
                True,
            )
        )
        raw_cache_match = jnp.array_equal(raw_available, state.last_raw_available) & jnp.all(
            jnp.where(raw_available, raw == state.last_raw_features, True)
        )
        current_cache_matches = (~state.has_last_transition) | (
            direct_cache_match & raw_cache_match
        )
        action_valid = (selected_action >= 0) & (selected_action < cfg.n_actions)
        propensity_valid = (
            jnp.isfinite(propensity)
            & (propensity >= 0.0)
            & (propensity <= 1.0)
            & (~randomized_flag | (propensity > 0.0))
        )
        hand_identity_matches = jnp.array_equal(
            hand_identity,
            jnp.asarray(cfg.hand_comparator_identity, dtype=jnp.uint32),
        )
        replayed = state.has_last_transition & jnp.array_equal(
            identity, state.last_transition_id
        )
        state_valid = self.validate_state(
            state,
            semantic_generation=generation,
            source_digest=source,
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(events))
            & jnp.all(jnp.isfinite(atoms))
            & jnp.all(jnp.isfinite(bottlenecks))
            & jnp.all(jnp.isfinite(probe))
            & jnp.all(jnp.isfinite(incumbent_values))
            & jnp.all(jnp.isfinite(hand_values))
            & jnp.all(jnp.isfinite(reward_predictions))
            & jnp.all(jnp.isfinite(model_predictions))
            & jnp.isfinite(propensity)
        )
        one_hot = jax.nn.one_hot(
            jnp.clip(selected_action, 0, cfg.n_actions - 1),
            cfg.n_actions,
            dtype=jnp.float32,
        )
        design = jnp.concatenate((probe, one_hot), axis=0)
        candidate_values = state.last_candidate_values
        candidate_available = state.last_candidate_available
        probe_predictions = state.probe_weights @ design
        baseline_predictions = state.candidate_means
        random_values = jnp.tanh(
            state.random_projections @ raw
            / jnp.sqrt(jnp.asarray(cfg.raw_feature_dim, dtype=jnp.float32))
        )
        reward_inserted = reward_predictions[None, :] + (
            state.reward_shadow_weights * candidate_values[:, None]
        )
        model_inserted = model_predictions[None, :] + (
            state.model_shadow_weights * candidate_values[:, None]
        )
        available = (
            state_valid
            & inputs_finite
            & action_valid
            & propensity_valid
            & hand_identity_matches
            & ~replayed
            & current_cache_matches
        )
        armed = CumulantSubtaskDiscoveryArm(
            available=available,
            transition_id=identity,
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=state.canonical_digest,
            cache_digest=jnp.zeros((2,), dtype=jnp.uint32),
            state_observation_count=state.observation_count,
            action=selected_action,
            behavior_propensity=propensity,
            randomized=randomized_flag,
            current_raw_features=raw,
            current_raw_available=raw_available,
            current_controllable_events=events,
            current_controllable_events_available=events_available,
            current_transition_atoms=atoms,
            current_transition_atoms_available=atoms_available,
            current_bottleneck_values=bottlenecks,
            current_bottleneck_available=bottlenecks_available,
            probe_features=probe,
            probe_design=design,
            current_candidate_values=candidate_values,
            current_candidate_available=candidate_available,
            probe_predictions=probe_predictions,
            baseline_predictions=baseline_predictions,
            current_random_values=random_values,
            current_hand_values=hand_values,
            current_hand_available=hand_available,
            current_incumbent_values=incumbent_values,
            current_incumbent_available=incumbent_available,
            frozen_reward_base_predictions=reward_predictions,
            frozen_reward_inserted_predictions=reward_inserted,
            frozen_model_base_predictions=model_predictions,
            frozen_model_inserted_predictions=model_inserted,
            reward_birth_observations=state.reward_birth_observations,
            hand_comparator_identity=hand_identity,
        )
        return dataclasses.replace(  # type: ignore[type-var]
            armed, cache_digest=self._arm_checksum(armed, state)
        )

    def _arm_checksum(
        self,
        arm: CumulantSubtaskDiscoveryArm,
        state: CumulantSubtaskDiscoveryState,
    ) -> Array:
        values = (
            arm.available,
            arm.transition_id,
            arm.semantic_generation,
            arm.source_digest,
            arm.canonical_digest,
            arm.state_observation_count,
            arm.action,
            arm.behavior_propensity,
            arm.randomized,
            arm.current_raw_features,
            arm.current_raw_available,
            arm.current_controllable_events,
            arm.current_controllable_events_available,
            arm.current_transition_atoms,
            arm.current_transition_atoms_available,
            arm.current_bottleneck_values,
            arm.current_bottleneck_available,
            arm.probe_features,
            arm.probe_design,
            arm.current_candidate_values,
            arm.current_candidate_available,
            arm.probe_predictions,
            arm.baseline_predictions,
            arm.current_random_values,
            arm.current_hand_values,
            arm.current_hand_available,
            arm.current_incumbent_values,
            arm.current_incumbent_available,
            arm.frozen_reward_base_predictions,
            arm.frozen_reward_inserted_predictions,
            arm.frozen_model_base_predictions,
            arm.frozen_model_inserted_predictions,
            arm.reward_birth_observations,
            arm.hand_comparator_identity,
        )
        return _checksum_arrays(values, seed=state.random_projection_key)

    def _check_arm_contract(self, arm: CumulantSubtaskDiscoveryArm) -> None:
        if type(arm) is not CumulantSubtaskDiscoveryArm:
            raise TypeError("arm must be an exact CumulantSubtaskDiscoveryArm")
        cfg = self._config
        c = cfg.candidate_count
        i = len(cfg.incumbent_descriptors)
        r = len(cfg.reward_task_weights)
        m = len(cfg.model_task_weights)
        contracts = (
            (arm.available, "available", (), jnp.bool_),
            (arm.transition_id, "transition_id", (2,), jnp.uint32),
            (arm.semantic_generation, "semantic_generation", (), jnp.int32),
            (arm.source_digest, "source_digest", (2,), jnp.uint32),
            (arm.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (arm.cache_digest, "cache_digest", (2,), jnp.uint32),
            (arm.state_observation_count, "state_observation_count", (), jnp.int32),
            (arm.action, "action", (), jnp.int32),
            (arm.behavior_propensity, "behavior_propensity", (), jnp.float32),
            (arm.randomized, "randomized", (), jnp.bool_),
            (arm.current_raw_features, "current_raw_features", (cfg.raw_feature_dim,), jnp.float32),
            (arm.current_raw_available, "current_raw_available", (cfg.raw_feature_dim,), jnp.bool_),
            (
                arm.current_controllable_events,
                "current_controllable_events",
                (cfg.controllable_event_dim,),
                jnp.float32,
            ),
            (
                arm.current_controllable_events_available,
                "current_controllable_events_available",
                (cfg.controllable_event_dim,),
                jnp.bool_,
            ),
            (
                arm.current_transition_atoms,
                "current_transition_atoms",
                (cfg.transition_atom_dim,),
                jnp.float32,
            ),
            (
                arm.current_transition_atoms_available,
                "current_transition_atoms_available",
                (cfg.transition_atom_dim,),
                jnp.bool_,
            ),
            (
                arm.current_bottleneck_values,
                "current_bottleneck_values",
                (cfg.prediction_bottleneck_dim,),
                jnp.float32,
            ),
            (
                arm.current_bottleneck_available,
                "current_bottleneck_available",
                (cfg.prediction_bottleneck_dim,),
                jnp.bool_,
            ),
            (arm.probe_features, "probe_features", (cfg.probe_feature_dim,), jnp.float32),
            (
                arm.probe_design,
                "probe_design",
                (cfg.probe_feature_dim + cfg.n_actions,),
                jnp.float32,
            ),
            (arm.current_candidate_values, "current_candidate_values", (c,), jnp.float32),
            (arm.current_candidate_available, "current_candidate_available", (c,), jnp.bool_),
            (arm.probe_predictions, "probe_predictions", (c,), jnp.float32),
            (arm.baseline_predictions, "baseline_predictions", (c,), jnp.float32),
            (arm.current_random_values, "current_random_values", (cfg.option_budget,), jnp.float32),
            (arm.current_hand_values, "current_hand_values", (cfg.option_budget,), jnp.float32),
            (arm.current_hand_available, "current_hand_available", (cfg.option_budget,), jnp.bool_),
            (arm.current_incumbent_values, "current_incumbent_values", (i,), jnp.float32),
            (arm.current_incumbent_available, "current_incumbent_available", (i,), jnp.bool_),
            (
                arm.frozen_reward_base_predictions,
                "frozen_reward_base_predictions",
                (r,),
                jnp.float32,
            ),
            (
                arm.frozen_reward_inserted_predictions,
                "frozen_reward_inserted_predictions",
                (c, r),
                jnp.float32,
            ),
            (arm.frozen_model_base_predictions, "frozen_model_base_predictions", (m,), jnp.float32),
            (
                arm.frozen_model_inserted_predictions,
                "frozen_model_inserted_predictions",
                (c, m),
                jnp.float32,
            ),
            (arm.reward_birth_observations, "reward_birth_observations", (c,), jnp.int32),
            (arm.hand_comparator_identity, "hand_comparator_identity", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array_contract(value, name=f"arm.{name}", shape=shape, dtype=dtype)

    def _arm_cache_valid(
        self,
        state: CumulantSubtaskDiscoveryState,
        arm: CumulantSubtaskDiscoveryArm,
    ) -> Array:
        cfg = self._config
        direct_values, direct_available = self._direct_semantics(
            raw_features=arm.current_raw_features,
            raw_available=arm.current_raw_available,
            controllable_events=arm.current_controllable_events,
            controllable_events_available=arm.current_controllable_events_available,
            transition_atoms=arm.current_transition_atoms,
            transition_atoms_available=arm.current_transition_atoms_available,
            bottleneck_values=arm.current_bottleneck_values,
            bottleneck_available=arm.current_bottleneck_available,
        )
        non_change = self._families != CUMULANT_SOURCE_FEATURE_CHANGE
        direct_cache_match = jnp.all(
            jnp.where(
                non_change,
                (direct_available == state.last_candidate_available)
                & jnp.where(
                    direct_available,
                    direct_values == state.last_candidate_values,
                    True,
                ),
                True,
            )
        )

        raw_cache_match = jnp.array_equal(
            arm.current_raw_available, state.last_raw_available
        ) & jnp.all(
            jnp.where(
                arm.current_raw_available,
                arm.current_raw_features == state.last_raw_features,
                True,
            )
        )
        current_cache_matches = (~state.has_last_transition) | (
            direct_cache_match & raw_cache_match
        )
        one_hot = jax.nn.one_hot(
            jnp.clip(arm.action, 0, cfg.n_actions - 1),
            cfg.n_actions,
            dtype=jnp.float32,
        )
        expected_design = jnp.concatenate((arm.probe_features, one_hot), axis=0)
        expected_probe = state.probe_weights @ expected_design
        expected_random = jnp.tanh(
            state.random_projections @ arm.current_raw_features
            / jnp.sqrt(jnp.asarray(cfg.raw_feature_dim, dtype=jnp.float32))
        )
        expected_reward_inserted = arm.frozen_reward_base_predictions[None, :] + (
            state.reward_shadow_weights * state.last_candidate_values[:, None]
        )
        expected_model_inserted = arm.frozen_model_base_predictions[None, :] + (
            state.model_shadow_weights * state.last_candidate_values[:, None]
        )
        float_fields = (
            arm.current_raw_features,
            arm.current_controllable_events,
            arm.current_transition_atoms,
            arm.current_bottleneck_values,
            arm.probe_features,
            arm.probe_design,
            arm.current_candidate_values,
            arm.probe_predictions,
            arm.baseline_predictions,
            arm.current_random_values,
            arm.current_hand_values,
            arm.current_incumbent_values,
            arm.frozen_reward_base_predictions,
            arm.frozen_reward_inserted_predictions,
            arm.frozen_model_base_predictions,
            arm.frozen_model_inserted_predictions,
        )
        finite = jnp.asarray(True, dtype=jnp.bool_)
        for value in float_fields:
            finite = finite & jnp.all(jnp.isfinite(value))
        action_valid = (arm.action >= 0) & (arm.action < cfg.n_actions)
        propensity_valid = (
            jnp.isfinite(arm.behavior_propensity)
            & (arm.behavior_propensity >= 0.0)
            & (arm.behavior_propensity <= 1.0)
            & (~arm.randomized | (arm.behavior_propensity > 0.0))
        )
        return (
            arm.available
            & finite
            & action_valid
            & propensity_valid
            & current_cache_matches
            & jnp.array_equal(arm.cache_digest, self._arm_checksum(arm, state))
            & jnp.array_equal(arm.canonical_digest, state.canonical_digest)
            & (arm.state_observation_count == state.observation_count)
            & jnp.array_equal(arm.current_candidate_values, state.last_candidate_values)
            & jnp.array_equal(arm.current_candidate_available, state.last_candidate_available)
            & jnp.array_equal(arm.probe_design, expected_design)
            & jnp.array_equal(arm.probe_predictions, expected_probe)
            & jnp.array_equal(arm.baseline_predictions, state.candidate_means)
            & jnp.array_equal(arm.current_random_values, expected_random)
            & jnp.array_equal(arm.frozen_reward_inserted_predictions, expected_reward_inserted)
            & jnp.array_equal(arm.frozen_model_inserted_predictions, expected_model_inserted)
            & jnp.array_equal(arm.reward_birth_observations, state.reward_birth_observations)
            & jnp.array_equal(
                arm.hand_comparator_identity,
                jnp.asarray(cfg.hand_comparator_identity, dtype=jnp.uint32),
            )
        )

    def _lexicographic_best(self, eligible: Array, scores: Array) -> tuple[Array, Array]:
        any_eligible = jnp.any(eligible)
        best_score = jnp.max(jnp.where(eligible, scores, -jnp.inf))
        tied = eligible & (scores == best_score)
        for column in range(_DESCRIPTOR_WIDTH):
            values = self._descriptors[:, column]
            minimum = jnp.min(jnp.where(tied, values, _INT32_MAX))
            tied = tied & (values == minimum)
        indices = jnp.arange(self._config.candidate_count, dtype=jnp.int32)
        selected = jnp.min(
            jnp.where(tied, indices, jnp.asarray(self._config.candidate_count, jnp.int32))
        )
        return jnp.where(any_eligible, selected, -1), any_eligible

    def _select_discovered(
        self,
        local_ready: Array,
        pair_novelty_ready: Array,
        candidate_scores: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Fill each fixed family quota without reassignment or partial readiness."""

        cfg = self._config
        selected_indices = jnp.full((cfg.option_budget,), -1, dtype=jnp.int32)
        selected_mask = jnp.zeros((cfg.candidate_count,), dtype=jnp.bool_)
        cursor = 0
        for family, quota in zip(_DISCOVERED_FAMILIES, cfg.family_quotas, strict=True):
            for _ in range(quota):
                safe_prior = jnp.clip(selected_indices, 0, cfg.candidate_count - 1)
                prior_valid = selected_indices >= 0
                pair_ready_for_prior = pair_novelty_ready[:, safe_prior]
                novel_against_prior = jnp.all(
                    (~prior_valid[None, :]) | pair_ready_for_prior,
                    axis=1,
                )
                eligible = (
                    local_ready
                    & (self._families == family)
                    & ~selected_mask
                    & novel_against_prior
                )
                selected, found = self._lexicographic_best(eligible, candidate_scores)
                selected_indices = selected_indices.at[cursor].set(
                    jnp.where(found, selected, -1)
                )
                safe_selected = jnp.clip(selected, 0, cfg.candidate_count - 1)
                selected_mask = selected_mask | (
                    jax.nn.one_hot(
                        safe_selected, cfg.candidate_count, dtype=jnp.bool_
                    )
                    & found
                )
                cursor += 1
        safe = jnp.clip(selected_indices, 0, cfg.candidate_count - 1)
        selected_families = self._families[safe]
        valid = selected_indices >= 0
        counts = jnp.stack(
            tuple(
                jnp.sum(valid & (selected_families == family), dtype=jnp.int32)
                for family in _DISCOVERED_FAMILIES
            ),
            axis=0,
        )
        ready = jnp.all(counts == jnp.asarray(cfg.family_quotas, dtype=jnp.int32))
        return selected_indices, selected_mask, counts, ready

    def _make_bundle(
        self,
        *,
        ready: Array,
        cohort_id: int,
        indices: Array,
        family_ids: Array,
        descriptors: Array,
        scores: Array,
        cumulants: Array,
        semantic_generation: Array,
        source_digest: Array,
        canonical_digest: Array,
        transition_id: Array,
        state_observation_count: Array,
    ) -> CumulantSubtaskProposalBundle:
        budget = self._config.option_budget
        bundle = CumulantSubtaskProposalBundle(
            ready=jnp.asarray(ready, dtype=jnp.bool_),
            cohort_id=jnp.asarray(cohort_id, dtype=jnp.int32),
            semantic_generation=semantic_generation,
            source_digest=source_digest,
            canonical_digest=canonical_digest,
            transition_id=transition_id,
            state_observation_count=state_observation_count,
            binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
            selected_candidate_indices=jnp.where(
                ready, indices, jnp.full((budget,), -1, dtype=jnp.int32)
            ),
            selected_family_ids=jnp.where(
                ready,
                family_ids,
                jnp.full((budget,), cohort_id, dtype=jnp.int32),
            ),
            selected_descriptors=jnp.where(
                ready, descriptors, jnp.zeros_like(descriptors)
            ),
            selected_scores=jnp.where(ready, scores, jnp.zeros_like(scores)),
            selected_cumulants=jnp.where(
                ready, cumulants, jnp.zeros_like(cumulants)
            ),
            tail_slot_indices=jnp.arange(
                self._config.raw_feature_dim,
                self._config.raw_feature_dim + budget,
                dtype=jnp.int32,
            ),
        )
        return dataclasses.replace(  # type: ignore[type-var]
            bundle, binding_digest=self._bundle_checksum(bundle)
        )

    def _bundle_checksum(self, bundle: CumulantSubtaskProposalBundle) -> Array:
        return _checksum_arrays(
            (
                bundle.ready,
                bundle.cohort_id,
                bundle.semantic_generation,
                bundle.source_digest,
                bundle.canonical_digest,
                bundle.transition_id,
                bundle.state_observation_count,
                bundle.selected_candidate_indices,
                bundle.selected_family_ids,
                bundle.selected_descriptors,
                bundle.selected_scores,
                bundle.selected_cumulants,
                bundle.tail_slot_indices,
            ),
            seed=bundle.source_digest,
        )

    def observe(
        self,
        state: CumulantSubtaskDiscoveryState,
        arm: CumulantSubtaskDiscoveryArm,
        *,
        next_raw_features: Array,
        next_raw_available: Array,
        next_controllable_events: Array,
        next_controllable_events_available: Array,
        next_transition_atoms: Array,
        next_transition_atoms_available: Array,
        next_bottleneck_values: Array,
        next_bottleneck_available: Array,
        bottleneck_epistemic: Array,
        bottleneck_progress: Array,
        bottleneck_aleatoric: Array,
        bottleneck_evidence_available: Array,
        randomized_action_evidence: Array,
        next_incumbent_values: Array,
        next_incumbent_available: Array,
        next_hand_values: Array,
        next_hand_available: Array,
        hand_comparator_identity: Array,
        reward_targets: Array,
        reward_targets_available: Array,
        model_targets: Array,
        model_targets_available: Array,
        transition_id: Array,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> CumulantSubtaskDiscoveryResult:
        """Audit one outcome and atomically emit three matched-budget cohorts."""

        self._check_state_contract(state)
        self._check_arm_contract(arm)
        cfg = self._config
        c = cfg.candidate_count
        i = len(cfg.incumbent_descriptors)
        r = len(cfg.reward_task_weights)
        m = len(cfg.model_task_weights)
        raw = _require_array_contract(
            next_raw_features,
            name="next_raw_features",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.float32,
        )
        raw_available = _require_array_contract(
            next_raw_available,
            name="next_raw_available",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.bool_,
        )
        events = _require_array_contract(
            next_controllable_events,
            name="next_controllable_events",
            shape=(cfg.controllable_event_dim,),
            dtype=jnp.float32,
        )
        events_available = _require_array_contract(
            next_controllable_events_available,
            name="next_controllable_events_available",
            shape=(cfg.controllable_event_dim,),
            dtype=jnp.bool_,
        )
        atoms = _require_array_contract(
            next_transition_atoms,
            name="next_transition_atoms",
            shape=(cfg.transition_atom_dim,),
            dtype=jnp.float32,
        )
        atoms_available = _require_array_contract(
            next_transition_atoms_available,
            name="next_transition_atoms_available",
            shape=(cfg.transition_atom_dim,),
            dtype=jnp.bool_,
        )
        bottlenecks = _require_array_contract(
            next_bottleneck_values,
            name="next_bottleneck_values",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.float32,
        )
        bottlenecks_available = _require_array_contract(
            next_bottleneck_available,
            name="next_bottleneck_available",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.bool_,
        )
        epistemic = _require_array_contract(
            bottleneck_epistemic,
            name="bottleneck_epistemic",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.float32,
        )
        progress = _require_array_contract(
            bottleneck_progress,
            name="bottleneck_progress",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.float32,
        )
        aleatoric = _require_array_contract(
            bottleneck_aleatoric,
            name="bottleneck_aleatoric",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.float32,
        )
        bottleneck_evidence = _require_array_contract(
            bottleneck_evidence_available,
            name="bottleneck_evidence_available",
            shape=(cfg.prediction_bottleneck_dim,),
            dtype=jnp.bool_,
        )
        intervention_evidence = _require_array_contract(
            randomized_action_evidence,
            name="randomized_action_evidence",
            shape=(cfg.n_actions,),
            dtype=jnp.bool_,
        )
        incumbent_values = _require_array_contract(
            next_incumbent_values,
            name="next_incumbent_values",
            shape=(i,),
            dtype=jnp.float32,
        )
        incumbent_available = _require_array_contract(
            next_incumbent_available,
            name="next_incumbent_available",
            shape=(i,),
            dtype=jnp.bool_,
        )
        hand_values = _require_array_contract(
            next_hand_values,
            name="next_hand_values",
            shape=(cfg.option_budget,),
            dtype=jnp.float32,
        )
        hand_available = _require_array_contract(
            next_hand_available,
            name="next_hand_available",
            shape=(cfg.option_budget,),
            dtype=jnp.bool_,
        )
        hand_identity = _require_array_contract(
            hand_comparator_identity,
            name="hand_comparator_identity",
            shape=(2,),
            dtype=jnp.uint32,
        )
        reward_target_values = _require_array_contract(
            reward_targets,
            name="reward_targets",
            shape=(r,),
            dtype=jnp.float32,
        )
        reward_available = _require_array_contract(
            reward_targets_available,
            name="reward_targets_available",
            shape=(r,),
            dtype=jnp.bool_,
        )
        model_target_values = _require_array_contract(
            model_targets,
            name="model_targets",
            shape=(m,),
            dtype=jnp.float32,
        )
        model_available = _require_array_contract(
            model_targets_available,
            name="model_targets_available",
            shape=(m,),
            dtype=jnp.bool_,
        )
        identity = _require_array_contract(
            transition_id, name="transition_id", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(semantic_generation, name="semantic_generation")
        source = _require_array_contract(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )

        state_valid = self.validate_state(
            state,
            semantic_generation=generation,
            source_digest=source,
        )
        transition_matches = jnp.array_equal(identity, arm.transition_id)
        source_binding_matches = (
            (generation == state.semantic_generation)
            & (generation == arm.semantic_generation)
            & jnp.array_equal(source, state.source_digest)
            & jnp.array_equal(source, arm.source_digest)
            & jnp.array_equal(arm.canonical_digest, state.canonical_digest)
        )
        hand_identity_matches = (
            jnp.array_equal(hand_identity, arm.hand_comparator_identity)
            & jnp.array_equal(
                hand_identity,
                jnp.asarray(cfg.hand_comparator_identity, dtype=jnp.uint32),
            )
        )
        arm_cache_valid = self._arm_cache_valid(state, arm)
        float_inputs = (
            raw,
            events,
            atoms,
            bottlenecks,
            epistemic,
            progress,
            aleatoric,
            incumbent_values,
            hand_values,
            reward_target_values,
            model_target_values,
        )
        inputs_finite = jnp.asarray(True, dtype=jnp.bool_)
        for value in float_inputs:
            inputs_finite = inputs_finite & jnp.all(jnp.isfinite(value))
        metrics_valid = (
            jnp.all(epistemic >= 0.0)
            & jnp.all(progress >= 0.0)
            & jnp.all(aleatoric >= 0.0)
        )
        arm_valid = (
            arm_cache_valid
            & transition_matches
            & source_binding_matches
            & hand_identity_matches
        )
        base_transaction_valid = state_valid & arm_valid & inputs_finite & metrics_valid
        capacity_capped = state.observation_count == cfg.max_observations
        capacity_available = state.observation_count < cfg.max_observations

        safe_raw = jnp.nan_to_num(raw)
        safe_events = jnp.nan_to_num(events)
        safe_atoms = jnp.nan_to_num(atoms)
        safe_bottlenecks = jnp.nan_to_num(bottlenecks)
        direct_values, direct_available = self._direct_semantics(
            raw_features=safe_raw,
            raw_available=raw_available,
            controllable_events=safe_events,
            controllable_events_available=events_available,
            transition_atoms=safe_atoms,
            transition_atoms_available=atoms_available,
            bottleneck_values=safe_bottlenecks,
            bottleneck_available=bottlenecks_available,
        )
        raw_indices = jnp.clip(self._source_indices, 0, cfg.raw_feature_dim - 1)
        feature_delta = self._polarities * (
            safe_raw[raw_indices] - jnp.nan_to_num(arm.current_raw_features)[raw_indices]
        )
        feature_available = raw_available[raw_indices] & arm.current_raw_available[raw_indices]
        feature_family = self._families == CUMULANT_SOURCE_FEATURE_CHANGE
        candidate_values = jnp.where(feature_family, feature_delta, direct_values)
        semantic_available = jnp.where(feature_family, feature_available, direct_available)
        reward_family = self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM
        born_before_arm = (~reward_family) | (arm.reward_birth_observations >= 0)
        evidence_available = semantic_available & born_before_arm & capacity_available
        births_this_transition = (
            reward_family
            & (arm.reward_birth_observations < 0)
            & semantic_available
            & capacity_available
        )

        probe_error = candidate_values - arm.probe_predictions
        baseline_error = candidate_values - arm.baseline_predictions
        learn_mask = evidence_available
        next_learnability_counts = state.learnability_counts + learn_mask.astype(jnp.int32)
        next_probe_error_sums = state.probe_squared_error_sums + jnp.where(
            learn_mask, jnp.square(probe_error), 0.0
        )
        next_baseline_error_sums = state.baseline_squared_error_sums + jnp.where(
            learn_mask, jnp.square(baseline_error), 0.0
        )
        next_value_counts = state.candidate_value_counts + learn_mask.astype(jnp.int32)
        safe_value_counts = jnp.maximum(next_value_counts, 1).astype(jnp.float32)
        next_candidate_means = jnp.where(
            learn_mask,
            state.candidate_means
            + (candidate_values - state.candidate_means) / safe_value_counts,
            state.candidate_means,
        )
        design_scale = jnp.sum(jnp.square(arm.probe_design)) + jnp.asarray(
            cfg.statistic_epsilon, dtype=jnp.float32
        )
        probe_delta = (
            jnp.asarray(cfg.probe_step_size, dtype=jnp.float32)
            * probe_error[:, None]
            * arm.probe_design[None, :]
            / design_scale
        )
        next_probe_weights = state.probe_weights + jnp.where(
            learn_mask[:, None], probe_delta, 0.0
        )

        safe_action = jnp.clip(arm.action, 0, cfg.n_actions - 1)
        action_one_hot_bool = jax.nn.one_hot(
            safe_action, cfg.n_actions, dtype=jnp.bool_
        )
        selected_intervention_available = intervention_evidence[safe_action]
        intervention_valid = (
            arm.randomized
            & (arm.behavior_propensity > 0.0)
            & selected_intervention_available
            & capacity_available
        )
        control_mask = evidence_available & intervention_valid
        control_cell_mask = control_mask[:, None] & action_one_hot_bool[None, :]
        inverse_propensity = 1.0 / jnp.maximum(
            arm.behavior_propensity,
            jnp.asarray(cfg.statistic_epsilon, dtype=jnp.float32),
        )
        next_action_sums = state.action_outcome_weighted_sums + jnp.where(
            control_cell_mask,
            candidate_values[:, None] * inverse_propensity,
            0.0,
        )
        next_action_masses = state.action_importance_masses + jnp.where(
            control_cell_mask, inverse_propensity, 0.0
        )
        next_action_counts = state.action_evidence_counts + control_cell_mask.astype(
            jnp.int32
        )

        safe_incumbents = jnp.nan_to_num(incumbent_values)
        incumbent_cells = evidence_available[:, None] & incumbent_available[None, :]
        incumbent_distances = jnp.square(
            candidate_values[:, None] - safe_incumbents[None, :]
        )
        next_incumbent_novelty_sums = state.incumbent_novelty_sums + jnp.where(
            incumbent_cells, incumbent_distances, 0.0
        )
        next_incumbent_novelty_counts = (
            state.incumbent_novelty_counts + incumbent_cells.astype(jnp.int32)
        )
        pair_cells = evidence_available[:, None] & evidence_available[None, :]
        pair_cells = pair_cells & ~jnp.eye(c, dtype=jnp.bool_)
        pair_distances = jnp.square(
            candidate_values[:, None] - candidate_values[None, :]
        )
        next_pair_novelty_sums = state.pair_novelty_sums + jnp.where(
            pair_cells, pair_distances, 0.0
        )
        next_pair_novelty_counts = state.pair_novelty_counts + pair_cells.astype(
            jnp.int32
        )

        current_contribution_available = (
            arm.current_candidate_available & born_before_arm & capacity_available
        )
        safe_reward_targets = jnp.nan_to_num(reward_target_values)
        reward_base_loss = jnp.square(
            safe_reward_targets - arm.frozen_reward_base_predictions
        )
        reward_inserted_loss = jnp.square(
            safe_reward_targets[None, :] - arm.frozen_reward_inserted_predictions
        )
        reward_improvement = reward_base_loss[None, :] - reward_inserted_loss
        reward_cells = current_contribution_available[:, None] & reward_available[None, :]
        safe_model_targets = jnp.nan_to_num(model_target_values)
        model_base_loss = jnp.square(
            safe_model_targets - arm.frozen_model_base_predictions
        )
        model_inserted_loss = jnp.square(
            safe_model_targets[None, :] - arm.frozen_model_inserted_predictions
        )
        model_improvement = model_base_loss[None, :] - model_inserted_loss
        model_cells = current_contribution_available[:, None] & model_available[None, :]
        contribution_delta = jnp.concatenate((reward_improvement, model_improvement), axis=1)
        contribution_cells = jnp.concatenate((reward_cells, model_cells), axis=1)
        next_contribution_sums = state.task_contribution_sums + jnp.where(
            contribution_cells, contribution_delta, 0.0
        )
        next_contribution_counts = (
            state.task_contribution_counts + contribution_cells.astype(jnp.int32)
        )
        shadow_scale = jnp.square(arm.current_candidate_values) + jnp.asarray(
            cfg.statistic_epsilon, dtype=jnp.float32
        )
        reward_shadow_error = (
            safe_reward_targets[None, :] - arm.frozen_reward_inserted_predictions
        )
        reward_shadow_delta = (
            jnp.asarray(cfg.shadow_step_size, dtype=jnp.float32)
            * reward_shadow_error
            * arm.current_candidate_values[:, None]
            / shadow_scale[:, None]
        )
        next_reward_shadow_weights = state.reward_shadow_weights + jnp.where(
            reward_cells, reward_shadow_delta, 0.0
        )
        model_shadow_error = (
            safe_model_targets[None, :] - arm.frozen_model_inserted_predictions
        )
        model_shadow_delta = (
            jnp.asarray(cfg.shadow_step_size, dtype=jnp.float32)
            * model_shadow_error
            * arm.current_candidate_values[:, None]
            / shadow_scale[:, None]
        )
        next_model_shadow_weights = state.model_shadow_weights + jnp.where(
            model_cells, model_shadow_delta, 0.0
        )

        bottleneck_indices = jnp.clip(
            self._source_indices, 0, cfg.prediction_bottleneck_dim - 1
        )
        candidate_epistemic = jnp.nan_to_num(epistemic)[bottleneck_indices]
        candidate_progress = jnp.nan_to_num(progress)[bottleneck_indices]
        candidate_aleatoric = jnp.nan_to_num(aleatoric)[bottleneck_indices]
        bottleneck_family = self._families == CUMULANT_SOURCE_PREDICTION_BOTTLENECK
        bottleneck_cells = (
            bottleneck_family
            & evidence_available
            & bottleneck_evidence[bottleneck_indices]
        )
        next_bottleneck_counts = (
            state.bottleneck_evidence_counts + bottleneck_cells.astype(jnp.int32)
        )
        next_bottleneck_epistemic = state.bottleneck_epistemic_sums + jnp.where(
            bottleneck_cells, candidate_epistemic, 0.0
        )
        next_bottleneck_progress = state.bottleneck_progress_sums + jnp.where(
            bottleneck_cells, candidate_progress, 0.0
        )
        next_bottleneck_aleatoric = state.bottleneck_aleatoric_sums + jnp.where(
            bottleneck_cells, candidate_aleatoric, 0.0
        )

        next_birth_observations = jnp.where(
            births_this_transition,
            state.observation_count + jnp.asarray(1, dtype=jnp.int32),
            state.reward_birth_observations,
        )
        proposed_state = CumulantSubtaskDiscoveryState(
            semantic_generation=state.semantic_generation,
            source_digest=state.source_digest,
            canonical_digest=state.canonical_digest,
            observation_count=state.observation_count
            + capacity_available.astype(jnp.int32),
            has_last_transition=jnp.where(capacity_available, True, state.has_last_transition),
            last_transition_id=jnp.where(
                capacity_available, identity, state.last_transition_id
            ),
            random_projection_key=state.random_projection_key,
            random_projection_digest=state.random_projection_digest,
            random_projections=state.random_projections,
            last_raw_features=jnp.where(
                capacity_available, safe_raw, state.last_raw_features
            ),
            last_raw_available=jnp.where(
                capacity_available, raw_available, state.last_raw_available
            ),
            reward_birth_observations=next_birth_observations,
            last_candidate_values=jnp.where(
                capacity_available & semantic_available,
                candidate_values,
                state.last_candidate_values,
            ),
            last_candidate_available=jnp.where(
                capacity_available,
                semantic_available,
                state.last_candidate_available,
            ),
            probe_weights=next_probe_weights,
            candidate_means=next_candidate_means,
            candidate_value_counts=next_value_counts,
            probe_squared_error_sums=next_probe_error_sums,
            baseline_squared_error_sums=next_baseline_error_sums,
            learnability_counts=next_learnability_counts,
            action_outcome_weighted_sums=next_action_sums,
            action_importance_masses=next_action_masses,
            action_evidence_counts=next_action_counts,
            incumbent_novelty_sums=next_incumbent_novelty_sums,
            incumbent_novelty_counts=next_incumbent_novelty_counts,
            pair_novelty_sums=next_pair_novelty_sums,
            pair_novelty_counts=next_pair_novelty_counts,
            reward_shadow_weights=next_reward_shadow_weights,
            model_shadow_weights=next_model_shadow_weights,
            task_contribution_sums=next_contribution_sums,
            task_contribution_counts=next_contribution_counts,
            bottleneck_epistemic_sums=next_bottleneck_epistemic,
            bottleneck_progress_sums=next_bottleneck_progress,
            bottleneck_aleatoric_sums=next_bottleneck_aleatoric,
            bottleneck_evidence_counts=next_bottleneck_counts,
        )

        proposed_state_valid = self.validate_state(
            proposed_state,
            semantic_generation=generation,
            source_digest=source,
        )
        transaction_valid = base_transaction_valid & proposed_state_valid
        transaction_applied = transaction_valid & capacity_available
        committed_state = jax.tree_util.tree_map(
            lambda proposed, old: jnp.where(transaction_applied, proposed, old),
            proposed_state,
            state,
        )

        epsilon = jnp.asarray(cfg.statistic_epsilon, dtype=jnp.float32)
        learn_counts_f = jnp.maximum(next_learnability_counts, 1).astype(jnp.float32)
        probe_mse = next_probe_error_sums / learn_counts_f
        baseline_mse = next_baseline_error_sums / learn_counts_f
        learnability_scores = jnp.clip(
            1.0 - probe_mse / jnp.maximum(baseline_mse, epsilon),
            0.0,
            1.0,
        )
        learnability_ready = (
            (next_learnability_counts >= cfg.learnability_evidence_floor)
            & (baseline_mse >= cfg.baseline_variance_floor)
            & (learnability_scores >= cfg.learnability_threshold)
        )

        action_means = next_action_sums / jnp.maximum(next_action_masses, epsilon)
        controllability_scores = jnp.max(action_means, axis=1) - jnp.min(
            action_means, axis=1
        )
        controllability_ready = (
            jnp.all(
                next_action_counts >= cfg.controllability_evidence_floor_per_action,
                axis=1,
            )
            & (controllability_scores >= cfg.controllability_threshold)
        )

        incumbent_means = next_incumbent_novelty_sums / jnp.maximum(
            next_incumbent_novelty_counts.astype(jnp.float32), 1.0
        )
        novelty_scores = jnp.min(incumbent_means, axis=1)
        novelty_against_incumbents_ready = (
            jnp.all(
                next_incumbent_novelty_counts >= cfg.novelty_evidence_floor,
                axis=1,
            )
            & jnp.all(incumbent_means >= cfg.novelty_threshold, axis=1)
        )
        pair_means = next_pair_novelty_sums / jnp.maximum(
            next_pair_novelty_counts.astype(jnp.float32), 1.0
        )
        descriptor_duplicates = jnp.all(
            self._descriptors[:, None, :] == self._descriptors[None, :, :], axis=2
        )
        pair_novelty_ready = (
            (next_pair_novelty_counts >= cfg.novelty_evidence_floor)
            & (pair_means >= cfg.novelty_threshold)
            & ~descriptor_duplicates
        )

        task_means = next_contribution_sums / jnp.maximum(
            next_contribution_counts.astype(jnp.float32), 1.0
        )
        task_weights = jnp.asarray(
            (*cfg.reward_task_weights, *cfg.model_task_weights), dtype=jnp.float32
        )
        contribution_scores = jnp.sum(task_means * task_weights[None, :], axis=1)
        contribution_ready = (
            jnp.all(
                next_contribution_counts >= cfg.contribution_evidence_floor,
                axis=1,
            )
            & (contribution_scores >= cfg.contribution_threshold)
        )

        bottleneck_count_f = jnp.maximum(next_bottleneck_counts, 1).astype(jnp.float32)
        mean_epistemic = next_bottleneck_epistemic / bottleneck_count_f
        mean_progress = next_bottleneck_progress / bottleneck_count_f
        mean_aleatoric = next_bottleneck_aleatoric / bottleneck_count_f
        bottleneck_specific_ready = (
            (next_bottleneck_counts >= cfg.bottleneck_evidence_floor)
            & (mean_epistemic >= cfg.bottleneck_epistemic_floor)
            & (mean_progress >= cfg.bottleneck_progress_floor)
            & (mean_aleatoric <= cfg.bottleneck_aleatoric_ceiling)
        )
        bottleneck_ready = (~bottleneck_family) | bottleneck_specific_ready
        all_local_gates = (
            learnability_ready
            & controllability_ready
            & novelty_against_incumbents_ready
            & contribution_ready
            & bottleneck_ready
            & semantic_available
            & born_before_arm
        )
        bottleneck_score = jnp.where(
            bottleneck_family,
            mean_epistemic + mean_progress - mean_aleatoric,
            0.0,
        )
        candidate_scores = jnp.nan_to_num(
            learnability_scores
            + controllability_scores
            + novelty_scores
            + contribution_scores
            + bottleneck_score,
            nan=0.0,
            posinf=jnp.asarray(3.4028235e38, dtype=jnp.float32),
            neginf=jnp.asarray(-3.4028235e38, dtype=jnp.float32),
        )
        selected_indices, selected_mask, family_counts, quota_ready = (
            self._select_discovered(
                all_local_gates,
                pair_novelty_ready,
                candidate_scores,
            )
        )
        safe_selected = jnp.clip(selected_indices, 0, c - 1)
        matched_ready = (
            transaction_applied
            & quota_ready
            & jnp.all(raw_available)
            & hand_identity_matches
            & jnp.all(hand_available)
        )
        discovered_ready = matched_ready
        discovered = self._make_bundle(
            ready=discovered_ready,
            cohort_id=-1,
            indices=selected_indices,
            family_ids=self._families[safe_selected],
            descriptors=self._descriptors[safe_selected],
            scores=candidate_scores[safe_selected],
            cumulants=candidate_values[safe_selected],
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=state.canonical_digest,
            transition_id=identity,
            state_observation_count=proposed_state.observation_count,
        )

        comparator_indices = jnp.arange(cfg.option_budget, dtype=jnp.int32)
        random_values = jnp.tanh(
            state.random_projections @ safe_raw
            / jnp.sqrt(jnp.asarray(cfg.raw_feature_dim, dtype=jnp.float32))
        )
        random_ready = matched_ready
        random_comparator = self._make_bundle(
            ready=random_ready,
            cohort_id=CUMULANT_SOURCE_RANDOM_PROJECTION,
            indices=comparator_indices,
            family_ids=jnp.full(
                (cfg.option_budget,),
                CUMULANT_SOURCE_RANDOM_PROJECTION,
                dtype=jnp.int32,
            ),
            descriptors=self.random_comparator_descriptors,
            scores=jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            cumulants=random_values,
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=state.canonical_digest,
            transition_id=identity,
            state_observation_count=proposed_state.observation_count,
        )
        hand_ready = matched_ready
        hand_comparator = self._make_bundle(
            ready=hand_ready,
            cohort_id=CUMULANT_SOURCE_HAND_AUTHORED,
            indices=comparator_indices,
            family_ids=jnp.full(
                (cfg.option_budget,), CUMULANT_SOURCE_HAND_AUTHORED, dtype=jnp.int32
            ),
            descriptors=self._hand_descriptors,
            scores=jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            cumulants=jnp.nan_to_num(hand_values),
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=state.canonical_digest,
            transition_id=identity,
            state_observation_count=proposed_state.observation_count,
        )
        diagnostics = CumulantSubtaskDiscoveryDiagnostics(
            transaction_valid=transaction_valid,
            transaction_applied=transaction_applied,
            capacity_capped=capacity_capped,
            state_valid=state_valid,
            arm_valid=arm_valid,
            arm_cache_valid=arm_cache_valid,
            transition_identity_matches=transition_matches,
            source_binding_matches=source_binding_matches,
            hand_identity_matches=hand_identity_matches,
            inputs_finite=inputs_finite,
            reward_births_this_transition=births_this_transition & transaction_applied,
            semantic_available=semantic_available,
            learnability_ready=learnability_ready,
            controllability_ready=controllability_ready,
            novelty_against_incumbents_ready=novelty_against_incumbents_ready,
            contribution_ready=contribution_ready,
            bottleneck_ready=bottleneck_ready,
            all_local_gates_ready=all_local_gates,
            selected_mask=selected_mask & transaction_applied,
            family_selected_counts=jnp.where(
                transaction_applied,
                family_counts,
                jnp.zeros((4,), dtype=jnp.int32),
            ),
            family_quotas=jnp.asarray(cfg.family_quotas, dtype=jnp.int32),
            bundle_ready=discovered_ready,
            random_comparator_ready=random_ready,
            hand_comparator_ready=hand_ready,
            candidate_scores=candidate_scores,
            learnability_scores=learnability_scores,
            controllability_scores=controllability_scores,
            novelty_scores=novelty_scores,
            contribution_scores=contribution_scores,
        )
        return CumulantSubtaskDiscoveryResult(
            state=committed_state,
            discovered=discovered,
            random_comparator=random_comparator,
            hand_comparator=hand_comparator,
            diagnostics=diagnostics,
        )

    def _check_bundle_contract(self, bundle: CumulantSubtaskProposalBundle) -> None:
        if type(bundle) is not CumulantSubtaskProposalBundle:
            raise TypeError("proposals must be an exact CumulantSubtaskProposalBundle")
        budget = self._config.option_budget
        contracts = (
            (bundle.ready, "ready", (), jnp.bool_),
            (bundle.cohort_id, "cohort_id", (), jnp.int32),
            (bundle.semantic_generation, "semantic_generation", (), jnp.int32),
            (bundle.source_digest, "source_digest", (2,), jnp.uint32),
            (bundle.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (bundle.transition_id, "transition_id", (2,), jnp.uint32),
            (bundle.state_observation_count, "state_observation_count", (), jnp.int32),
            (bundle.binding_digest, "binding_digest", (2,), jnp.uint32),
            (bundle.selected_candidate_indices, "selected_candidate_indices", (budget,), jnp.int32),
            (bundle.selected_family_ids, "selected_family_ids", (budget,), jnp.int32),
            (bundle.selected_descriptors, "selected_descriptors", (budget, 4), jnp.int32),
            (bundle.selected_scores, "selected_scores", (budget,), jnp.float32),
            (bundle.selected_cumulants, "selected_cumulants", (budget,), jnp.float32),
            (bundle.tail_slot_indices, "tail_slot_indices", (budget,), jnp.int32),
        )
        for value, name, shape, dtype in contracts:
            _require_array_contract(value, name=f"proposals.{name}", shape=shape, dtype=dtype)

    def validate_proposal_bundle(
        self,
        bundle: CumulantSubtaskProposalBundle,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
        canonical_digest: Array,
        transition_id: Array,
        state_observation_count: int | Array,
    ) -> Array:
        """Validate internal payload integrity and an exact current transaction binding."""

        self._check_bundle_contract(bundle)
        generation = _int32_scalar(semantic_generation, name="semantic_generation")
        source = _require_array_contract(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        canonical = _require_array_contract(
            canonical_digest, name="canonical_digest", shape=(32,), dtype=jnp.uint8
        )
        identity = _require_array_contract(
            transition_id, name="transition_id", shape=(2,), dtype=jnp.uint32
        )
        revision = _int32_scalar(state_observation_count, name="state_observation_count")
        budget = self._config.option_budget
        expected_tail = jnp.arange(
            self._config.raw_feature_dim,
            self._config.raw_feature_dim + budget,
            dtype=jnp.int32,
        )
        indices = bundle.selected_candidate_indices
        safe_discovered = jnp.clip(indices, 0, self._config.candidate_count - 1)
        discovered_payload_valid = (
            jnp.all((indices >= 0) & (indices < self._config.candidate_count))
            & jnp.array_equal(bundle.selected_family_ids, self._families[safe_discovered])
            & jnp.array_equal(bundle.selected_descriptors, self._descriptors[safe_discovered])
        )
        comparator_indices = jnp.arange(budget, dtype=jnp.int32)
        random_payload_valid = (
            jnp.array_equal(indices, comparator_indices)
            & jnp.all(bundle.selected_family_ids == CUMULANT_SOURCE_RANDOM_PROJECTION)
            & jnp.array_equal(
                bundle.selected_descriptors, self.random_comparator_descriptors
            )
        )
        hand_payload_valid = (
            jnp.array_equal(indices, comparator_indices)
            & jnp.all(bundle.selected_family_ids == CUMULANT_SOURCE_HAND_AUTHORED)
            & jnp.array_equal(bundle.selected_descriptors, self._hand_descriptors)
        )
        cohort_payload_valid = jnp.where(
            bundle.cohort_id == -1,
            discovered_payload_valid,
            jnp.where(
                bundle.cohort_id == CUMULANT_SOURCE_RANDOM_PROJECTION,
                random_payload_valid,
                jnp.where(
                    bundle.cohort_id == CUMULANT_SOURCE_HAND_AUTHORED,
                    hand_payload_valid,
                    False,
                ),
            ),
        )
        return (
            bundle.ready
            & (bundle.semantic_generation == generation)
            & jnp.array_equal(bundle.source_digest, source)
            & jnp.array_equal(bundle.canonical_digest, canonical)
            & jnp.array_equal(bundle.transition_id, identity)
            & (bundle.state_observation_count == revision)
            & jnp.array_equal(bundle.binding_digest, self._bundle_checksum(bundle))
            & jnp.array_equal(bundle.tail_slot_indices, expected_tail)
            & jnp.all(jnp.isfinite(bundle.selected_scores))
            & jnp.all(jnp.isfinite(bundle.selected_cumulants))
            & cohort_payload_valid
        )

    def materialize(
        self,
        raw_features: Array,
        proposals: CumulantSubtaskProposalBundle,
        *,
        semantic_generation: int | Array | None = None,
        source_digest: Array | None = None,
        canonical_digest: Array | None = None,
        transition_id: Array | None = None,
        state_observation_count: int | Array | None = None,
    ) -> Array:
        """Append a compact tail, zeroing tampering or an explicitly stale binding.

        Two-argument use validates the bundle's internal checksum.  A live
        adapter must also pass its expected binding keywords to reject a
        structurally intact bundle from an older transaction.
        """

        raw = _require_array_contract(
            raw_features,
            name="raw_features",
            shape=(self._config.raw_feature_dim,),
            dtype=jnp.float32,
        )
        self._check_bundle_contract(proposals)
        generation = (
            proposals.semantic_generation
            if semantic_generation is None
            else _int32_scalar(semantic_generation, name="semantic_generation")
        )
        source = proposals.source_digest if source_digest is None else source_digest
        canonical = (
            proposals.canonical_digest if canonical_digest is None else canonical_digest
        )
        identity = proposals.transition_id if transition_id is None else transition_id
        revision = (
            proposals.state_observation_count
            if state_observation_count is None
            else _int32_scalar(state_observation_count, name="state_observation_count")
        )
        binding_valid = self.validate_proposal_bundle(
            proposals,
            semantic_generation=generation,
            source_digest=source,
            canonical_digest=canonical,
            transition_id=identity,
            state_observation_count=revision,
        )
        raw_finite = jnp.all(jnp.isfinite(raw))
        tail = jnp.where(
            binding_valid & raw_finite,
            proposals.selected_cumulants,
            jnp.zeros((self._config.option_budget,), dtype=jnp.float32),
        )
        return jnp.concatenate((jnp.nan_to_num(raw), tail), axis=0)

    def checkpoint_payload(
        self,
        state: CumulantSubtaskDiscoveryState,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> dict[str, object]:
        """Return a strict schema/config/state payload for a generic PyTree serializer."""

        valid = self.validate_state(
            state,
            semantic_generation=semantic_generation,
            source_digest=source_digest,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("cannot checkpoint an invalid cumulant discovery state")
        return {
            "schema_version": CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_digest": self._state_checksum(state),
        }

    def _state_checksum(self, state: CumulantSubtaskDiscoveryState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            host = np.asarray(jax.device_get(leaf))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def restore_checkpoint(
        self,
        payload: object,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
    ) -> CumulantSubtaskDiscoveryState:
        """Restore only an exact v1 payload with the same config and live binding."""

        if type(payload) is not dict:
            raise ValueError("cumulant discovery checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema_version", "config", "state", "state_digest"}:
            raise ValueError("cumulant discovery checkpoint keys differ from schema v1")
        if raw["schema_version"] != CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA:
            raise ValueError("cumulant discovery checkpoint schema_version differs")
        restored_config = CumulantSubtaskDiscoveryConfig.from_config(raw["config"])
        if restored_config != self._config:
            raise ValueError("cumulant discovery checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not CumulantSubtaskDiscoveryState:
            raise ValueError("cumulant discovery checkpoint state type differs")
        state = restored
        persisted_digest = _require_array_contract(
            cast(Array, raw["state_digest"]),
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        if not bool(
            jax.device_get(jnp.array_equal(persisted_digest, self._state_checksum(state)))
        ):
            raise ValueError("cumulant discovery checkpoint state digest differs")
        valid = self.validate_state(
            state,
            semantic_generation=semantic_generation,
            source_digest=source_digest,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("cumulant discovery checkpoint state is invalid or stale")
        return state


__all__ = [
    "CUMULANT_SOURCE_CONTROLLABLE_EVENT",
    "CUMULANT_SOURCE_FEATURE_CHANGE",
    "CUMULANT_SOURCE_HAND_AUTHORED",
    "CUMULANT_SOURCE_PREDICTION_BOTTLENECK",
    "CUMULANT_SOURCE_RANDOM_PROJECTION",
    "CUMULANT_SOURCE_REWARD_TRANSITION_ATOM",
    "CUMULANT_SUBTASK_DISCOVERY_AUTHORITY",
    "CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA",
    "CUMULANT_SUBTASK_DISCOVERY_CONFIG_SCHEMA",
    "CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY",
    "CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY",
    "CUMULANT_SUBTASK_DISCOVERY_RANKING_SEMANTICS",
    "CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED",
    "CumulantSubtaskDiscovery",
    "CumulantSubtaskDiscoveryArm",
    "CumulantSubtaskDiscoveryConfig",
    "CumulantSubtaskDiscoveryDiagnostics",
    "CumulantSubtaskDiscoveryResourceBudget",
    "CumulantSubtaskDiscoveryResult",
    "CumulantSubtaskDiscoveryState",
    "CumulantSubtaskProposalBundle",
]
