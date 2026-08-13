# mypy: disable-error-code="attr-defined,call-arg"
"""Stateless audit-ranking policy for Prototype pair-feature curation.

This adapter turns mature per-task causal-utility EMAs into the transient rank
arrays accepted by :class:`InteractionCurationPriorityOverride`. It does not
observe data, update the utility auditor, route feature identities, select a
promotion, or decide whether curation is safe to run. A caller must separately
gate the returned override on the current auditor observation transaction and
the lifecycle's own safety/capacity conditions.

The policy is deliberately fail-closed. It requires an exact source-generation
and descriptor binding plus a valid utility state. Valid but immature slots get
reserved finite sentinels, and ``curation_ready`` remains false until at least
one active and one non-collided candidate have evidence from every configured
task. Fixed task mass is never redistributed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, cast

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.interaction_features import (
    CURATION_ACTIVE_INELIGIBLE_RANK,
    CURATION_CANDIDATE_INELIGIBLE_RANK,
    InteractionCurationPriorityOverride,
)
from alberta_framework.core.prototype_feature_utility import (
    PrototypeFeatureUtilityAuditor,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityState,
)

PROTOTYPE_FEATURE_UTILITY_CURATION_MECHANISM_STATUS = (
    "L0_AUDIT_RANKING_INFLUENCE_ONLY_NO_PROMOTION_GO_NO_GO_OR_SCIENTIFIC_AUTHORITY"
)
PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA = (
    "alberta.prototype-feature-utility-curation.config.v2"
)
_LEGACY_PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA = (
    "alberta.prototype-feature-utility-curation.config.v1"
)
PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE = True
PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY = False
PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY = False
PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY = False
PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureUtilityCurationConfig:
    """Strict evidence maturity required before a slot can receive a rank."""

    minimum_task_evidence: int

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.minimum_task_evidence) is not int:
            raise TypeError("minimum_task_evidence must be an exact Python int")
        if self.minimum_task_evidence < 1:
            raise ValueError("minimum_task_evidence must be positive")
        if self.minimum_task_evidence > _INT32_MAX:
            raise ValueError("minimum_task_evidence must fit saturating int32 evidence")

    def to_config(self) -> dict[str, object]:
        """Return the exact current JSON-compatible schema."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "minimum_task_evidence": self.minimum_task_evidence,
        }

    @classmethod
    def from_config(cls, value: object) -> PrototypeFeatureUtilityCurationConfig:
        """Reconstruct only an exact schema-v2 configuration."""

        if type(value) is not dict:
            raise ValueError("feature utility curation config must be an exact dict")
        raw = cast(dict[object, object], value)
        if set(raw) != {"schema_version", "minimum_task_evidence"}:
            raise ValueError("feature utility curation config keys differ from schema v2")
        if raw["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("feature utility curation config schema_version differs")
        return cls(
            minimum_task_evidence=cast(int, raw["minimum_task_evidence"]),
        )


def _require_array_contract(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Return an array after trace-time shape and effective-dtype checks."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Authenticate exact words against their saturating int32 projection."""

    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_words_less(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] < right[1])
    )


def _lifetime_words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] <= right[1])
    )


def _python_uint64_words(value: int) -> UInt[Array, " 2"]:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _words_to_int32_telemetry(words: Array) -> Int[Array, ""]:
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


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
    return jnp.any(
        jnp.all(
            candidate_descriptors[:, None, :] == active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityCurationDiagnostics:
    """Primitive facts behind one enabled, neutral, or rejected proposal."""

    available: Bool[Array, ""]
    transaction_valid: Bool[Array, ""]
    override_enabled: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    source_descriptors_valid: Bool[Array, ""]
    source_generation_matches: Bool[Array, ""]
    source_generation_counter_valid: Bool[Array, ""]
    source_generation_words_match: Bool[Array, ""]
    stale_state_generation: Bool[Array, ""]
    stale_source_generation: Bool[Array, ""]
    active_descriptors_match: Bool[Array, ""]
    candidate_descriptors_match: Bool[Array, ""]
    same_generation_descriptor_fork: Bool[Array, ""]
    source_binding_valid: Bool[Array, ""]
    observation_capacity_valid: Bool[Array, ""]
    observation_capacity_available: Bool[Array, ""]
    observation_capacity_capped: Bool[Array, ""]
    any_active_rank_ready: Bool[Array, ""]
    any_candidate_rank_ready: Bool[Array, ""]
    curation_ready: Bool[Array, ""]
    rank_values_finite: Bool[Array, ""]
    observation_count: Int[Array, ""]
    maximum_observations: Int[Array, ""]
    state_semantic_generation_words: UInt[Array, " 2"]
    source_semantic_generation_words: UInt[Array, " 2"]
    observation_words: UInt[Array, " 2"]
    maximum_observation_words: UInt[Array, " 2"]
    minimum_task_evidence: Int[Array, ""]
    task_weights: Float[Array, " n_tasks"]
    active_task_evidence_ready: Bool[Array, "n_tasks active_pair_slots"]
    candidate_task_evidence_ready: Bool[
        Array,
        "n_tasks candidate_pair_slots",
    ]
    active_all_tasks_evidence_ready: Bool[Array, " active_pair_slots"]
    candidate_all_tasks_evidence_ready: Bool[Array, " candidate_pair_slots"]
    candidate_collision_mask: Bool[Array, " candidate_pair_slots"]
    candidate_rank_ready_mask: Bool[Array, " candidate_pair_slots"]
    raw_active_fixed_mass_utilities: Float[Array, " active_pair_slots"]
    raw_candidate_fixed_mass_utilities: Float[Array, " candidate_pair_slots"]
    emitted_active_ranks: Float[Array, " active_pair_slots"]
    emitted_candidate_ranks: Float[Array, " candidate_pair_slots"]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityCurationResult:
    """Transient override and its complete audit-ranking diagnostics."""

    override: InteractionCurationPriorityOverride
    diagnostics: PrototypeFeatureUtilityCurationDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureUtilityCurationResourceBudget:
    """Fixed logical work and zero-persistent-state declaration."""

    persistent_logical_scalars: int
    persistent_state_nbytes: int
    state_validation_calls_per_rank: int
    task_evidence_cells_per_rank: int
    task_aggregate_cells_per_rank: int
    source_descriptor_validation_cells_per_rank: int
    binding_descriptor_cells_per_rank: int
    candidate_active_collision_cells_per_rank: int
    rank_output_cells_per_rank: int
    rng_draws_per_rank: int
    backward_passes_per_rank: int
    consumer_updates_per_rank: int
    router_calls_per_rank: int
    curation_decisions_per_rank: int
    ranking_influence: bool
    curation_authority: bool
    promotion_authority: bool
    go_no_go_authority: bool
    scientific_promotion_allowed: bool
    mechanism_status: str
    minimum_task_evidence: int


class PrototypeFeatureUtilityCurationPolicy:
    """Pure fixed-shape conversion from mature utility EMAs to rank arrays."""

    def __init__(
        self,
        utility_config: PrototypeFeatureUtilityConfig,
        curation_config: PrototypeFeatureUtilityCurationConfig,
    ) -> None:
        if type(utility_config) is not PrototypeFeatureUtilityConfig:
            raise TypeError("utility_config must be a PrototypeFeatureUtilityConfig")
        if type(curation_config) is not PrototypeFeatureUtilityCurationConfig:
            raise TypeError(
                "curation_config must be a PrototypeFeatureUtilityCurationConfig"
            )
        if curation_config.minimum_task_evidence > utility_config.max_observations:
            raise ValueError(
                "minimum_task_evidence must not exceed utility_config.max_observations"
            )
        self._utility_config = utility_config
        self._curation_config = curation_config
        self._auditor = PrototypeFeatureUtilityAuditor(utility_config)

    @property
    def utility_config(self) -> PrototypeFeatureUtilityConfig:
        """Return the immutable utility schema whose task weights are used."""

        return self._utility_config

    @property
    def curation_config(self) -> PrototypeFeatureUtilityCurationConfig:
        """Return the immutable serializable ranking-policy configuration."""

        return self._curation_config

    @property
    def minimum_task_evidence(self) -> int:
        """Return the strict evidence floor applied to every task/slot cell."""

        return self._curation_config.minimum_task_evidence

    def resource_budget(self) -> PrototypeFeatureUtilityCurationResourceBudget:
        """Declare fixed logical ranking work and the absence of policy state."""

        tasks = self._utility_config.n_tasks
        active = self._utility_config.active_pair_slots
        candidates = self._utility_config.candidate_pair_slots
        task_feature_cells = tasks * (active + candidates)
        descriptor_validation = active * active + candidates * candidates
        return PrototypeFeatureUtilityCurationResourceBudget(
            persistent_logical_scalars=0,
            persistent_state_nbytes=0,
            state_validation_calls_per_rank=1,
            task_evidence_cells_per_rank=task_feature_cells,
            task_aggregate_cells_per_rank=task_feature_cells,
            source_descriptor_validation_cells_per_rank=descriptor_validation,
            binding_descriptor_cells_per_rank=2 * (active + candidates),
            candidate_active_collision_cells_per_rank=active * candidates,
            rank_output_cells_per_rank=active + candidates,
            rng_draws_per_rank=0,
            backward_passes_per_rank=0,
            consumer_updates_per_rank=0,
            router_calls_per_rank=0,
            curation_decisions_per_rank=0,
            ranking_influence=True,
            curation_authority=False,
            promotion_authority=False,
            go_no_go_authority=False,
            scientific_promotion_allowed=False,
            mechanism_status=PROTOTYPE_FEATURE_UTILITY_CURATION_MECHANISM_STATUS,
            minimum_task_evidence=self._curation_config.minimum_task_evidence,
        )

    def rank(
        self,
        state: PrototypeFeatureUtilityState,
        *,
        source_semantic_generation: Array,
        source_semantic_generation_words: Array,
        source_active_descriptors: Array,
        source_candidate_descriptors: Array,
    ) -> PrototypeFeatureUtilityCurationResult:
        """Return finite ranks only for an exact, mature source-bank binding.

        Shape and dtype drift raises at trace time. Dynamic state corruption,
        non-finite values, stale generations, same-generation descriptor forks,
        and non-finite ranks return ``override.enabled=False`` with zero rank
        payloads. A valid evidence-unready policy remains enabled with reserved
        sentinels. Capacity exhaustion is a valid but neutral disabled result.
        """

        if type(state) is not PrototypeFeatureUtilityState:
            raise TypeError("state must be an exact PrototypeFeatureUtilityState")
        checked_state = state
        active_slots = self._utility_config.active_pair_slots
        candidate_slots = self._utility_config.candidate_pair_slots
        source_generation = _require_array_contract(
            source_semantic_generation,
            name="source_semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        source_generation_words = _require_array_contract(
            source_semantic_generation_words,
            name="source_semantic_generation_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        source_active = _require_array_contract(
            source_active_descriptors,
            name="source_active_descriptors",
            shape=(active_slots, 2),
            dtype=jnp.int32,
        )
        source_candidates = _require_array_contract(
            source_candidate_descriptors,
            name="source_candidate_descriptors",
            shape=(candidate_slots, 2),
            dtype=jnp.int32,
        )

        state_valid = self._auditor.state_valid(checked_state)
        source_descriptors_valid = (
            _descriptor_rows_valid(
                source_active,
                self._utility_config.base_feature_dim,
            )
            & _descriptor_rows_unique(source_active)
            & _descriptor_rows_valid(
                source_candidates,
                self._utility_config.base_feature_dim,
            )
            & _descriptor_rows_unique(source_candidates)
        )
        source_generation_counter_valid = _lifetime_counter_valid(
            source_generation_words,
            source_generation,
        )
        generation_words_match = jnp.all(
            source_generation_words == checked_state.semantic_generation_words
        )
        generation_matches = (
            source_generation_counter_valid
            & (source_generation == checked_state.semantic_generation)
            & generation_words_match
        )
        stale_state_generation = _lifetime_words_less(
            checked_state.semantic_generation_words,
            source_generation_words,
        )
        stale_source_generation = _lifetime_words_less(
            source_generation_words,
            checked_state.semantic_generation_words,
        )
        active_descriptors_match = jnp.array_equal(
            source_active,
            checked_state.active_descriptors,
        )
        candidate_descriptors_match = jnp.array_equal(
            source_candidates,
            checked_state.candidate_descriptors,
        )
        descriptor_fork = generation_matches & ~(
            active_descriptors_match & candidate_descriptors_match
        )
        source_binding_valid = (
            source_descriptors_valid
            & generation_matches
            & active_descriptors_match
            & candidate_descriptors_match
        )
        policy_input_valid = state_valid & source_binding_valid

        floor = jnp.asarray(
            self._curation_config.minimum_task_evidence,
            dtype=jnp.int32,
        )
        maximum_observation_words = _python_uint64_words(
            self._utility_config.max_observations
        )
        maximum_observations = _words_to_int32_telemetry(
            maximum_observation_words
        )
        observation_capacity_valid = (
            _lifetime_counter_valid(
                checked_state.observation_words,
                checked_state.observation_count,
            )
            & _lifetime_words_less_equal(
                checked_state.observation_words,
                maximum_observation_words,
            )
        )
        observation_capacity_capped = (
            jnp.all(
                checked_state.observation_words == maximum_observation_words
            )
        ) & observation_capacity_valid
        observation_capacity_available = (
            _lifetime_words_less(
                checked_state.observation_words,
                maximum_observation_words,
            )
        ) & observation_capacity_valid
        active_task_ready = (
            checked_state.active_task_evidence_counts >= floor
        ) & policy_input_valid
        candidate_task_ready = (
            checked_state.candidate_task_evidence_counts >= floor
        ) & policy_input_valid
        active_all_tasks_ready = jnp.all(active_task_ready, axis=0)
        candidate_all_tasks_ready = jnp.all(candidate_task_ready, axis=0)
        collision = _candidate_collision_mask(source_active, source_candidates)
        candidate_rank_ready = candidate_all_tasks_ready & ~collision

        task_weights = jnp.asarray(
            self._utility_config.task_utility_weights,
            dtype=jnp.float32,
        )
        raw_active_fixed_mass = jnp.sum(
            task_weights[:, None] * checked_state.active_task_utilities,
            axis=0,
        )
        raw_candidate_fixed_mass = jnp.sum(
            task_weights[:, None] * checked_state.candidate_task_utilities,
            axis=0,
        )
        active_ineligible_rank = jnp.asarray(
            CURATION_ACTIVE_INELIGIBLE_RANK,
            dtype=jnp.float32,
        )
        candidate_ineligible_rank = jnp.asarray(
            CURATION_CANDIDATE_INELIGIBLE_RANK,
            dtype=jnp.float32,
        )
        proposed_active_ranks = jnp.where(
            active_all_tasks_ready,
            raw_active_fixed_mass,
            active_ineligible_rank,
        )
        proposed_candidate_ranks = jnp.where(
            candidate_rank_ready,
            raw_candidate_fixed_mass,
            candidate_ineligible_rank,
        )
        rank_values_finite = (
            jnp.all(jnp.isfinite(proposed_active_ranks))
            & jnp.all(jnp.isfinite(proposed_candidate_ranks))
            & jnp.all(jnp.isfinite(raw_active_fixed_mass))
            & jnp.all(jnp.isfinite(raw_candidate_fixed_mass))
        )
        any_active_ready = jnp.any(active_all_tasks_ready)
        any_candidate_ready = jnp.any(candidate_rank_ready)
        transaction_valid = (
            policy_input_valid
            & observation_capacity_valid
            & rank_values_finite
        )
        enabled = transaction_valid & observation_capacity_available
        curation_ready = enabled & any_active_ready & any_candidate_ready
        active_ranks = jnp.where(
            enabled,
            proposed_active_ranks,
            jnp.zeros((active_slots,), dtype=jnp.float32),
        )
        candidate_ranks = jnp.where(
            enabled,
            proposed_candidate_ranks,
            jnp.zeros((candidate_slots,), dtype=jnp.float32),
        )
        override = InteractionCurationPriorityOverride(
            enabled=jnp.asarray(enabled, dtype=jnp.bool_),
            active_ranks=active_ranks,
            candidate_ranks=candidate_ranks,
        )
        diagnostics = PrototypeFeatureUtilityCurationDiagnostics(
            available=transaction_valid,
            transaction_valid=transaction_valid,
            override_enabled=enabled,
            state_valid=state_valid,
            source_descriptors_valid=source_descriptors_valid,
            source_generation_matches=generation_matches,
            source_generation_counter_valid=source_generation_counter_valid,
            source_generation_words_match=generation_words_match,
            stale_state_generation=stale_state_generation,
            stale_source_generation=stale_source_generation,
            active_descriptors_match=active_descriptors_match,
            candidate_descriptors_match=candidate_descriptors_match,
            same_generation_descriptor_fork=descriptor_fork,
            source_binding_valid=source_binding_valid,
            observation_capacity_valid=observation_capacity_valid,
            observation_capacity_available=observation_capacity_available,
            observation_capacity_capped=observation_capacity_capped,
            any_active_rank_ready=any_active_ready,
            any_candidate_rank_ready=any_candidate_ready,
            curation_ready=curation_ready,
            rank_values_finite=rank_values_finite,
            observation_count=checked_state.observation_count,
            maximum_observations=maximum_observations,
            state_semantic_generation_words=(
                checked_state.semantic_generation_words
            ),
            source_semantic_generation_words=source_generation_words,
            observation_words=checked_state.observation_words,
            maximum_observation_words=maximum_observation_words,
            minimum_task_evidence=floor,
            task_weights=task_weights,
            active_task_evidence_ready=active_task_ready,
            candidate_task_evidence_ready=candidate_task_ready,
            active_all_tasks_evidence_ready=active_all_tasks_ready,
            candidate_all_tasks_evidence_ready=candidate_all_tasks_ready,
            candidate_collision_mask=collision,
            candidate_rank_ready_mask=candidate_rank_ready,
            raw_active_fixed_mass_utilities=raw_active_fixed_mass,
            raw_candidate_fixed_mass_utilities=raw_candidate_fixed_mass,
            emitted_active_ranks=active_ranks,
            emitted_candidate_ranks=candidate_ranks,
        )
        return PrototypeFeatureUtilityCurationResult(
            override=override,
            diagnostics=diagnostics,
        )


def migrate_legacy_prototype_feature_utility_curation_config(
    legacy_config: object,
) -> PrototypeFeatureUtilityCurationConfig:
    """Explicitly relabel one exact v1 ranking configuration as v2."""

    if type(legacy_config) is not dict:
        raise ValueError("legacy feature utility curation config must be an exact dict")
    raw = dict(cast(dict[object, object], legacy_config))
    if set(raw) != {"schema_version", "minimum_task_evidence"}:
        raise ValueError("legacy feature utility curation keys differ from schema v1")
    if (
        raw["schema_version"]
        != _LEGACY_PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA
    ):
        raise ValueError("legacy feature utility curation schema_version differs")
    raw["schema_version"] = PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA
    return PrototypeFeatureUtilityCurationConfig.from_config(raw)


__all__ = [
    "PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_MECHANISM_STATUS",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "PrototypeFeatureUtilityCurationConfig",
    "PrototypeFeatureUtilityCurationDiagnostics",
    "PrototypeFeatureUtilityCurationPolicy",
    "PrototypeFeatureUtilityCurationResourceBudget",
    "PrototypeFeatureUtilityCurationResult",
    "migrate_legacy_prototype_feature_utility_curation_config",
]
