# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Prepare a fresh same-family proposal for exactly one cold option slot.

This opt-in version-2 sidecar is intentionally smaller than an installer.  It
reads one accepted discovery result, one incumbent cohort, and one live source
snapshot; it returns a source-bound candidate cohort or an unavailable bundle.
It owns no persistent state, RNG, adoption path, retirement decision, or
authority receipt.  In particular, preparing a cohort never installs it.

The v1 discovery bundle validator intentionally validates descriptor lookup but
does not enforce fixed family quotas.  This sidecar therefore revalidates the
slot-family layout, family counts, and candidate uniqueness independently.  A
checksum-valid cross-family splice cannot become a prepared v2 candidate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallation,
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CumulantSubtaskDiscoveryDiagnostics,
    CumulantSubtaskDiscoveryResult,
    CumulantSubtaskDiscoveryState,
    CumulantSubtaskProposalBundle,
    Descriptor,
)

FRESH_COLD_SLOT_CUMULANT_COHORT_CONFIG_SCHEMA = (
    "alberta.fresh-cold-slot-cumulant-cohort-filter.config.v2"
)
FRESH_COLD_SLOT_CUMULANT_COHORT_UNIVERSE_SCHEMA = (
    "alberta.fresh-cold-slot-cumulant-candidate-universe.v2"
)
FRESH_COLD_SLOT_CUMULANT_COHORT_SELECTION_SEMANTICS = (
    "v2-one-cold-slot-same-family-local-gates-pair-novelty-live-preserving-"
    "score-desc-descriptor-lexicographic-index"
)
FRESH_COLD_SLOT_CUMULANT_COHORT_ASSESSMENT = "not_assessed"
FRESH_COLD_SLOT_CUMULANT_COHORT_OUTPUT_WRITES = False
FRESH_COLD_SLOT_CUMULANT_COHORT_EVIDENCE_AUTHORITY = False
FRESH_COLD_SLOT_CUMULANT_COHORT_PROMOTION_AUTHORITY = False
FRESH_COLD_SLOT_CUMULANT_COHORT_SAFETY_AUTHORITY = False
FRESH_COLD_SLOT_CUMULANT_COHORT_GO_NO_GO_AUTHORITY = False
FRESH_COLD_SLOT_CUMULANT_COHORT_ADOPTION_AUTHORITY = False
FRESH_COLD_SLOT_CUMULANT_COHORT_SCIENTIFIC_PROMOTION_ALLOWED = False

_DISCOVERED_FAMILIES = (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
)
_DIGEST_WORDS = 8
_DESCRIPTOR_WIDTH = 4
_INT32_MAX = 2**31 - 1


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _float_bits_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.uint32),
        jax.lax.bitcast_convert_type(right, jnp.uint32),
    )


def _tree_array_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            valid = valid & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            valid = valid & _float_bits_equal(left_array, right_array)
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _checksum_arrays(
    arrays: tuple[Array, ...],
    *,
    seed: Array | None = None,
) -> Array:
    if seed is None:
        acc0 = jnp.uint32(0x9E3779B9)
        acc1 = jnp.uint32(0x85EBCA6B)
    else:
        acc0 = seed[0] ^ jnp.uint32(0x9E3779B9)
        acc1 = seed[1] ^ jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.int32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _descriptor_rows(value: object, *, name: str) -> tuple[Descriptor, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be a tuple or JSON list")
    rows: list[Descriptor] = []
    for row_index, row in enumerate(cast(tuple[object, ...] | list[object], value)):
        if type(row) not in (tuple, list) or len(cast(Any, row)) != _DESCRIPTOR_WIDTH:
            raise ValueError(f"{name}[{row_index}] must contain four exact integers")
        cells = cast(tuple[object, ...] | list[object], row)
        if any(type(cell) is not int for cell in cells):
            raise ValueError(f"{name}[{row_index}] must contain four exact integers")
        values = tuple(cast(int, cell) for cell in cells)
        if any(not -(2**31) <= cell <= _INT32_MAX for cell in values):
            raise ValueError(f"{name}[{row_index}] must be signed-int32 compatible")
        rows.append(cast(Descriptor, values))
    if not rows or len(set(rows)) != len(rows):
        raise ValueError(f"{name} must be nonempty and globally unique")
    return tuple(rows)


def _quota_tuple(value: object) -> tuple[int, int, int, int]:
    if type(value) not in (tuple, list) or len(cast(Any, value)) != 4:
        raise ValueError("family_quotas must contain exactly four integers")
    cells = cast(tuple[object, ...] | list[object], value)
    if any(type(cell) is not int or cell < 1 for cell in cells):
        raise ValueError("family_quotas must contain positive exact integers")
    return cast(tuple[int, int, int, int], tuple(cast(int, cell) for cell in cells))


@dataclasses.dataclass(frozen=True, slots=True)
class FreshColdSlotCumulantCohortFilterConfig:
    """Explicit v2 candidate-universe manifest for the stateless filter."""

    candidate_universe_schema: str
    candidate_descriptors: tuple[Descriptor, ...]
    family_quotas: tuple[int, int, int, int]
    source_discovery_config_sha256: str
    selection_semantics: str = FRESH_COLD_SLOT_CUMULANT_COHORT_SELECTION_SEMANTICS

    SCHEMA_VERSION: ClassVar[str] = FRESH_COLD_SLOT_CUMULANT_COHORT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.candidate_universe_schema != FRESH_COLD_SLOT_CUMULANT_COHORT_UNIVERSE_SCHEMA:
            raise ValueError("candidate_universe_schema must identify the opt-in v2 universe")
        if type(self.candidate_descriptors) is not tuple or any(
            type(row) is not tuple for row in self.candidate_descriptors
        ):
            raise ValueError("candidate_descriptors must be an exact tuple of tuples")
        if _descriptor_rows(self.candidate_descriptors, name="candidate_descriptors") != (
            self.candidate_descriptors
        ):
            raise ValueError("candidate_descriptors differ from their canonical encoding")
        if type(self.family_quotas) is not tuple:
            raise ValueError("family_quotas must be an exact tuple")
        quotas = _quota_tuple(self.family_quotas)
        if sum(quotas) < 1:
            raise ValueError("family_quotas must declare a nonempty fixed budget")
        counts = tuple(
            sum(descriptor[0] == family for descriptor in self.candidate_descriptors)
            for family in _DISCOVERED_FAMILIES
        )
        if any(count < quota for count, quota in zip(counts, quotas, strict=True)):
            raise ValueError("the v2 universe must contain every fixed family quota")
        if (
            type(self.source_discovery_config_sha256) is not str
            or len(self.source_discovery_config_sha256) != 64
            or any(
                cell not in "0123456789abcdef"
                for cell in self.source_discovery_config_sha256
            )
        ):
            raise ValueError("source_discovery_config_sha256 must be lowercase SHA-256 hex")
        if self.selection_semantics != FRESH_COLD_SLOT_CUMULANT_COHORT_SELECTION_SEMANTICS:
            raise ValueError("selection_semantics differ from the fixed v2 procedure")

    @classmethod
    def from_installation(
        cls,
        installation: CumulantOptionInstallation,
    ) -> FreshColdSlotCumulantCohortFilterConfig:
        if type(installation) is not CumulantOptionInstallation:
            raise TypeError("installation must be an exact CumulantOptionInstallation")
        discovery = installation.discovery
        return cls(
            candidate_universe_schema=FRESH_COLD_SLOT_CUMULANT_COHORT_UNIVERSE_SCHEMA,
            candidate_descriptors=discovery.config.candidate_descriptors,
            family_quotas=discovery.config.family_quotas,
            source_discovery_config_sha256=_canonical_sha256(discovery.to_config()),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "candidate_universe_schema": self.candidate_universe_schema,
            "candidate_descriptors": self.candidate_descriptors,
            "family_quotas": self.family_quotas,
            "source_discovery_config_sha256": self.source_discovery_config_sha256,
            "selection_semantics": self.selection_semantics,
            "assessment": FRESH_COLD_SLOT_CUMULANT_COHORT_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "adoption_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> FreshColdSlotCumulantCohortFilterConfig:
        if type(value) is not dict:
            raise ValueError("fresh cold-slot cohort config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "candidate_universe_schema",
            "candidate_descriptors",
            "family_quotas",
            "source_discovery_config_sha256",
            "selection_semantics",
            "assessment",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "adoption_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("fresh cold-slot cohort config keys differ from schema v2")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("fresh cold-slot cohort config schema differs")
        if raw.pop("assessment") != FRESH_COLD_SLOT_CUMULANT_COHORT_ASSESSMENT:
            raise ValueError("fresh cold-slot cohort assessment must remain not_assessed")
        for name in (
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "adoption_authority",
            "scientific_promotion_allowed",
        ):
            if raw.pop(name) is not False:
                raise ValueError(f"fresh cold-slot cohort cannot claim {name}")
        return cls(
            candidate_universe_schema=cast(str, raw["candidate_universe_schema"]),
            candidate_descriptors=_descriptor_rows(
                raw["candidate_descriptors"], name="candidate_descriptors"
            ),
            family_quotas=_quota_tuple(raw["family_quotas"]),
            source_discovery_config_sha256=cast(
                str, raw["source_discovery_config_sha256"]
            ),
            selection_semantics=cast(str, raw["selection_semantics"]),
        )


@chex.dataclass(frozen=True)
class FreshColdSlotCumulantCohortSource:
    """Read-only source facts for one v2 preparation."""

    discovery_result: CumulantSubtaskDiscoveryResult
    installed_bundle: CumulantSubtaskProposalBundle
    installed_semantic_digests: UInt[Array, "option_budget 8"]
    installed_slot_mask: Bool[Array, " option_budget"]
    previous_raw_features: Float[Array, " raw_feature_dim"]
    previous_raw_available: Bool[Array, " raw_feature_dim"]
    live_inputs: CumulantOptionLiveInputs


@chex.dataclass(frozen=True)
class FreshColdSlotCumulantCohortDiagnostics:
    """Primitive facts; ``candidate_ready`` is their fail-closed conjunction."""

    source_contract_valid: Bool[Array, ""]
    discovery_result_valid: Bool[Array, ""]
    live_binding_valid: Bool[Array, ""]
    incumbent_bundle_valid: Bool[Array, ""]
    incumbent_semantics_valid: Bool[Array, ""]
    exactly_one_cold_slot: Bool[Array, ""]
    family_quota_layout_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    local_gate_ready: Bool[Array, ""]
    pair_novelty_ready: Bool[Array, ""]
    alternate_available: Bool[Array, ""]
    same_family_selected: Bool[Array, ""]
    candidate_family_quota_valid: Bool[Array, ""]
    live_slots_preserved: Bool[Array, ""]
    target_semantic_fresh: Bool[Array, ""]
    exact_target_change: Bool[Array, ""]
    filtered_bundle_valid: Bool[Array, ""]
    candidate_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class FreshColdSlotCumulantCohortPrepared:
    """Transient prepared cohort; there is deliberately no commit method."""

    source: FreshColdSlotCumulantCohortSource
    filtered_bundle: CumulantSubtaskProposalBundle
    candidate_semantic_digests: UInt[Array, "option_budget 8"]
    changed_slots: Bool[Array, " option_budget"]
    target_slot: Int[Array, ""]
    target_mask: Bool[Array, " option_budget"]
    selected_candidate_index: Int[Array, ""]
    diagnostics: FreshColdSlotCumulantCohortDiagnostics
    prepared_checksum: UInt[Array, " 2"]


class FreshColdSlotCumulantCohortFilter:
    """Pure same-family filter over one explicitly versioned candidate universe."""

    def __init__(
        self,
        installation: CumulantOptionInstallation,
        config: FreshColdSlotCumulantCohortFilterConfig,
    ) -> None:
        if type(installation) is not CumulantOptionInstallation:
            raise TypeError("installation must be an exact CumulantOptionInstallation")
        if type(config) is not FreshColdSlotCumulantCohortFilterConfig:
            raise TypeError("config must be an exact FreshColdSlotCumulantCohortFilterConfig")
        expected = FreshColdSlotCumulantCohortFilterConfig.from_installation(installation)
        if config != expected:
            raise ValueError("v2 universe manifest differs from the borrowed discovery")
        self._installation = installation
        self._config = config
        self._descriptors = jnp.asarray(config.candidate_descriptors, dtype=jnp.int32)
        self._families = self._descriptors[:, 0]
        self._source_indices = self._descriptors[:, 1]
        self._polarities = self._descriptors[:, 2].astype(jnp.float32)
        self._slot_families = jnp.asarray(
            tuple(
                family
                for family, quota in zip(
                    _DISCOVERED_FAMILIES, config.family_quotas, strict=True
                )
                for _ in range(quota)
            ),
            dtype=jnp.int32,
        )

    @property
    def installation(self) -> CumulantOptionInstallation:
        return self._installation

    @property
    def config(self) -> FreshColdSlotCumulantCohortFilterConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def _check_live_inputs(self, inputs: CumulantOptionLiveInputs) -> None:
        if type(inputs) is not CumulantOptionLiveInputs:
            raise TypeError("live_inputs must be an exact CumulantOptionLiveInputs")
        cfg = self._installation.discovery.config
        contracts = (
            (inputs.raw_features, "raw_features", (cfg.raw_feature_dim,), jnp.float32),
            (inputs.raw_available, "raw_available", (cfg.raw_feature_dim,), jnp.bool_),
            (
                inputs.controllable_events,
                "controllable_events",
                (cfg.controllable_event_dim,),
                jnp.float32,
            ),
            (
                inputs.controllable_events_available,
                "controllable_events_available",
                (cfg.controllable_event_dim,),
                jnp.bool_,
            ),
            (
                inputs.transition_atoms,
                "transition_atoms",
                (cfg.transition_atom_dim,),
                jnp.float32,
            ),
            (
                inputs.transition_atoms_available,
                "transition_atoms_available",
                (cfg.transition_atom_dim,),
                jnp.bool_,
            ),
            (
                inputs.bottleneck_values,
                "bottleneck_values",
                (cfg.prediction_bottleneck_dim,),
                jnp.float32,
            ),
            (
                inputs.bottleneck_available,
                "bottleneck_available",
                (cfg.prediction_bottleneck_dim,),
                jnp.bool_,
            ),
            (inputs.semantic_generation, "semantic_generation", (), jnp.int32),
            (inputs.source_digest, "source_digest", (2,), jnp.uint32),
            (inputs.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (inputs.transition_id, "transition_id", (2,), jnp.uint32),
            (inputs.state_observation_count, "state_observation_count", (), jnp.int32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"live_inputs.{name}", shape=shape, dtype=dtype)

    def _check_diagnostics_contract(
        self,
        diagnostics: CumulantSubtaskDiscoveryDiagnostics,
    ) -> None:
        if type(diagnostics) is not CumulantSubtaskDiscoveryDiagnostics:
            raise TypeError("diagnostics must be exact CumulantSubtaskDiscoveryDiagnostics")
        candidates = self._installation.discovery.config.candidate_count
        scalar_bools = (
            diagnostics.transaction_valid,
            diagnostics.transaction_applied,
            diagnostics.capacity_capped,
            diagnostics.state_valid,
            diagnostics.arm_valid,
            diagnostics.arm_cache_valid,
            diagnostics.transition_identity_matches,
            diagnostics.source_binding_matches,
            diagnostics.hand_identity_matches,
            diagnostics.inputs_finite,
            diagnostics.bundle_ready,
            diagnostics.random_comparator_ready,
            diagnostics.hand_comparator_ready,
        )
        for index, value in enumerate(scalar_bools):
            _require_array(
                value,
                name=f"diagnostics.scalar_bool[{index}]",
                shape=(),
                dtype=jnp.bool_,
            )
        vector_bools = (
            diagnostics.reward_births_this_transition,
            diagnostics.semantic_available,
            diagnostics.learnability_ready,
            diagnostics.controllability_ready,
            diagnostics.novelty_against_incumbents_ready,
            diagnostics.contribution_ready,
            diagnostics.bottleneck_ready,
            diagnostics.all_local_gates_ready,
            diagnostics.selected_mask,
        )
        for index, value in enumerate(vector_bools):
            _require_array(
                value,
                name=f"diagnostics.vector_bool[{index}]",
                shape=(candidates,),
                dtype=jnp.bool_,
            )
        _require_array(
            diagnostics.family_selected_counts,
            name="diagnostics.family_selected_counts",
            shape=(4,),
            dtype=jnp.int32,
        )
        _require_array(
            diagnostics.family_quotas,
            name="diagnostics.family_quotas",
            shape=(4,),
            dtype=jnp.int32,
        )
        score_vectors = (
            diagnostics.candidate_scores,
            diagnostics.learnability_scores,
            diagnostics.controllability_scores,
            diagnostics.novelty_scores,
            diagnostics.contribution_scores,
        )
        for index, value in enumerate(score_vectors):
            _require_array(
                value,
                name=f"diagnostics.score[{index}]",
                shape=(candidates,),
                dtype=jnp.float32,
            )

    def _check_source_contract(self, source: FreshColdSlotCumulantCohortSource) -> None:
        if type(source) is not FreshColdSlotCumulantCohortSource:
            raise TypeError("source must be an exact FreshColdSlotCumulantCohortSource")
        if type(source.discovery_result) is not CumulantSubtaskDiscoveryResult:
            raise TypeError("discovery_result must be an exact CumulantSubtaskDiscoveryResult")
        if type(source.discovery_result.state) is not CumulantSubtaskDiscoveryState:
            raise TypeError("discovery_result.state must be exact CumulantSubtaskDiscoveryState")
        discovery = self._installation.discovery
        discovery.check_proposal_bundle_contract(source.discovery_result.discovered)
        discovery.check_proposal_bundle_contract(source.discovery_result.random_comparator)
        discovery.check_proposal_bundle_contract(source.discovery_result.hand_comparator)
        discovery.check_proposal_bundle_contract(source.installed_bundle)
        self._check_diagnostics_contract(source.discovery_result.diagnostics)
        self._check_live_inputs(source.live_inputs)
        cfg = discovery.config
        _require_array(
            source.installed_semantic_digests,
            name="installed_semantic_digests",
            shape=(cfg.option_budget, _DIGEST_WORDS),
            dtype=jnp.uint32,
        )
        _require_array(
            source.installed_slot_mask,
            name="installed_slot_mask",
            shape=(cfg.option_budget,),
            dtype=jnp.bool_,
        )
        _require_array(
            source.previous_raw_features,
            name="previous_raw_features",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            source.previous_raw_available,
            name="previous_raw_available",
            shape=(cfg.raw_feature_dim,),
            dtype=jnp.bool_,
        )

    def _candidate_values_and_available(
        self,
        source: FreshColdSlotCumulantCohortSource,
    ) -> tuple[Array, Array]:
        cfg = self._installation.discovery.config
        live = source.live_inputs
        indices = self._source_indices
        event_indices = jnp.clip(indices, 0, cfg.controllable_event_dim - 1)
        raw_indices = jnp.clip(indices, 0, cfg.raw_feature_dim - 1)
        atom_indices = jnp.clip(indices, 0, cfg.transition_atom_dim - 1)
        bottleneck_indices = jnp.clip(indices, 0, cfg.prediction_bottleneck_dim - 1)
        direct_values = jnp.where(
            self._families == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            live.controllable_events[event_indices],
            jnp.where(
                self._families == CUMULANT_SOURCE_FEATURE_CHANGE,
                live.raw_features[raw_indices],
                jnp.where(
                    self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    live.transition_atoms[atom_indices],
                    live.bottleneck_values[bottleneck_indices],
                ),
            ),
        ) * self._polarities
        direct_available = jnp.where(
            self._families == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            live.controllable_events_available[event_indices],
            jnp.where(
                self._families == CUMULANT_SOURCE_FEATURE_CHANGE,
                live.raw_available[raw_indices],
                jnp.where(
                    self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    live.transition_atoms_available[atom_indices],
                    live.bottleneck_available[bottleneck_indices],
                ),
            ),
        )
        feature_values = self._polarities * (
            live.raw_features[raw_indices] - source.previous_raw_features[raw_indices]
        )
        feature_available = (
            live.raw_available[raw_indices] & source.previous_raw_available[raw_indices]
        )
        feature_family = self._families == CUMULANT_SOURCE_FEATURE_CHANGE
        return (
            jnp.where(feature_family, feature_values, direct_values),
            jnp.where(feature_family, feature_available, direct_available),
        )

    def _local_gates_and_scores(self, state: CumulantSubtaskDiscoveryState) -> tuple[Array, Array]:
        cfg = self._installation.discovery.config
        epsilon = jnp.asarray(cfg.statistic_epsilon, dtype=jnp.float32)
        learn_counts = jnp.maximum(state.learnability_counts, 1).astype(jnp.float32)
        probe_mse = state.probe_squared_error_sums / learn_counts
        baseline_mse = state.baseline_squared_error_sums / learn_counts
        learnability_scores = jnp.clip(
            1.0 - probe_mse / jnp.maximum(baseline_mse, epsilon),
            0.0,
            1.0,
        )
        learnability_ready = (
            (state.learnability_counts >= cfg.learnability_evidence_floor)
            & (baseline_mse >= cfg.baseline_variance_floor)
            & (learnability_scores >= cfg.learnability_threshold)
        )
        action_means = state.action_outcome_weighted_sums / jnp.maximum(
            state.action_importance_masses, epsilon
        )
        controllability_scores = jnp.max(action_means, axis=1) - jnp.min(
            action_means, axis=1
        )
        controllability_ready = (
            jnp.all(
                state.action_evidence_counts >= cfg.controllability_evidence_floor_per_action,
                axis=1,
            )
            & (controllability_scores >= cfg.controllability_threshold)
        )
        incumbent_means = state.incumbent_novelty_sums / jnp.maximum(
            state.incumbent_novelty_counts.astype(jnp.float32), 1.0
        )
        novelty_scores = jnp.min(incumbent_means, axis=1)
        novelty_ready = (
            jnp.all(
                state.incumbent_novelty_counts >= cfg.novelty_evidence_floor,
                axis=1,
            )
            & jnp.all(incumbent_means >= cfg.novelty_threshold, axis=1)
        )
        task_means = state.task_contribution_sums / jnp.maximum(
            state.task_contribution_counts.astype(jnp.float32), 1.0
        )
        task_weights = jnp.asarray(
            (*cfg.reward_task_weights, *cfg.model_task_weights), dtype=jnp.float32
        )
        contribution_scores = jnp.sum(task_means * task_weights[None, :], axis=1)
        contribution_ready = (
            jnp.all(
                state.task_contribution_counts >= cfg.contribution_evidence_floor,
                axis=1,
            )
            & (contribution_scores >= cfg.contribution_threshold)
        )
        bottleneck_counts = jnp.maximum(state.bottleneck_evidence_counts, 1).astype(
            jnp.float32
        )
        mean_epistemic = state.bottleneck_epistemic_sums / bottleneck_counts
        mean_progress = state.bottleneck_progress_sums / bottleneck_counts
        mean_aleatoric = state.bottleneck_aleatoric_sums / bottleneck_counts
        bottleneck_specific_ready = (
            (state.bottleneck_evidence_counts >= cfg.bottleneck_evidence_floor)
            & (mean_epistemic >= cfg.bottleneck_epistemic_floor)
            & (mean_progress >= cfg.bottleneck_progress_floor)
            & (mean_aleatoric <= cfg.bottleneck_aleatoric_ceiling)
        )
        bottleneck_family = self._families == CUMULANT_SOURCE_PREDICTION_BOTTLENECK
        bottleneck_ready = (~bottleneck_family) | bottleneck_specific_ready
        reward_family = self._families == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM
        born_before_transition = (~reward_family) | (
            (state.reward_birth_observations >= 0)
            & (state.reward_birth_observations < state.observation_count)
        )
        all_local_gates = (
            learnability_ready
            & controllability_ready
            & novelty_ready
            & contribution_ready
            & bottleneck_ready
            & state.last_candidate_available
            & born_before_transition
        )
        bottleneck_score = jnp.where(
            bottleneck_family,
            mean_epistemic + mean_progress - mean_aleatoric,
            0.0,
        )
        scores = jnp.nan_to_num(
            learnability_scores
            + controllability_scores
            + novelty_scores
            + contribution_scores
            + bottleneck_score,
            nan=0.0,
            posinf=jnp.asarray(3.4028235e38, dtype=jnp.float32),
            neginf=jnp.asarray(-3.4028235e38, dtype=jnp.float32),
        )
        return all_local_gates, scores

    def _family_layout_valid(self, bundle: CumulantSubtaskProposalBundle) -> Array:
        cfg = self._installation.discovery.config
        indices = bundle.selected_candidate_indices
        safe = jnp.clip(indices, 0, cfg.candidate_count - 1)
        pairwise_distinct = (indices[:, None] != indices[None, :]) | jnp.eye(
            cfg.option_budget, dtype=jnp.bool_
        )
        counts = jnp.stack(
            tuple(
                jnp.sum(bundle.selected_family_ids == family, dtype=jnp.int32)
                for family in _DISCOVERED_FAMILIES
            ),
            axis=0,
        )
        return (
            bundle.ready
            & (bundle.cohort_id == -1)
            & jnp.all((indices >= 0) & (indices < cfg.candidate_count))
            & jnp.all(pairwise_distinct)
            & jnp.array_equal(bundle.selected_family_ids, self._slot_families)
            & jnp.array_equal(bundle.selected_family_ids, self._families[safe])
            & jnp.array_equal(bundle.selected_descriptors, self._descriptors[safe])
            & jnp.array_equal(counts, jnp.asarray(cfg.family_quotas, dtype=jnp.int32))
        )

    def _lexicographic_best(self, eligible: Array, scores: Array) -> tuple[Array, Array]:
        candidate_count = self._installation.discovery.config.candidate_count
        any_eligible = jnp.any(eligible)
        best_score = jnp.max(jnp.where(eligible, scores, -jnp.inf))
        tied = eligible & (scores == best_score)
        for column in range(_DESCRIPTOR_WIDTH):
            values = self._descriptors[:, column]
            minimum = jnp.min(jnp.where(tied, values, _INT32_MAX))
            tied = tied & (values == minimum)
        indices = jnp.arange(candidate_count, dtype=jnp.int32)
        selected = jnp.min(jnp.where(tied, indices, candidate_count))
        return jnp.where(any_eligible, selected, -1), any_eligible

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

    def _make_bundle(
        self,
        *,
        ready: Array,
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
        cfg = self._installation.discovery.config
        bundle = CumulantSubtaskProposalBundle(
            ready=jnp.asarray(ready, dtype=jnp.bool_),
            cohort_id=jnp.asarray(-1, dtype=jnp.int32),
            semantic_generation=semantic_generation,
            source_digest=source_digest,
            canonical_digest=canonical_digest,
            transition_id=transition_id,
            state_observation_count=state_observation_count,
            binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
            selected_candidate_indices=jnp.where(
                ready,
                indices,
                jnp.full((cfg.option_budget,), -1, dtype=jnp.int32),
            ),
            selected_family_ids=jnp.where(
                ready,
                family_ids,
                jnp.full((cfg.option_budget,), -1, dtype=jnp.int32),
            ),
            selected_descriptors=jnp.where(
                ready,
                descriptors,
                jnp.zeros((cfg.option_budget, _DESCRIPTOR_WIDTH), dtype=jnp.int32),
            ),
            selected_scores=jnp.where(
                ready,
                scores,
                jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            ),
            selected_cumulants=jnp.where(
                ready,
                cumulants,
                jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            ),
            tail_slot_indices=jnp.arange(
                cfg.raw_feature_dim,
                cfg.raw_feature_dim + cfg.option_budget,
                dtype=jnp.int32,
            ),
        )
        return dataclasses.replace(
            bundle,
            binding_digest=self._bundle_checksum(bundle),
        )

    def _prepared_payload_arrays(
        self,
        prepared: FreshColdSlotCumulantCohortPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(FreshColdSlotCumulantCohortPrepared)
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_prepared_checksum(
        self,
        prepared: FreshColdSlotCumulantCohortPrepared,
    ) -> FreshColdSlotCumulantCohortPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._prepared_payload_arrays(prepared)),
        )

    def prepare(
        self,
        source: FreshColdSlotCumulantCohortSource,
    ) -> FreshColdSlotCumulantCohortPrepared:
        """Prepare one deterministic candidate without mutating or adopting state."""

        self._check_source_contract(source)
        discovery = self._installation.discovery
        cfg = discovery.config
        result = source.discovery_result
        live = source.live_inputs
        finite_inputs = jnp.asarray(True, dtype=jnp.bool_)
        for value in (
            source.previous_raw_features,
            live.raw_features,
            live.controllable_events,
            live.transition_atoms,
            live.bottleneck_values,
        ):
            finite_inputs = finite_inputs & jnp.all(jnp.isfinite(value))
        source_contract_valid = finite_inputs

        state_valid = discovery.validate_state(
            result.state,
            semantic_generation=live.semantic_generation,
            source_digest=live.source_digest,
        )
        binding = {
            "semantic_generation": live.semantic_generation,
            "source_digest": live.source_digest,
            "canonical_digest": live.canonical_digest,
            "transition_id": live.transition_id,
            "state_observation_count": live.state_observation_count,
        }
        discovered_valid = discovery.validate_proposal_bundle(result.discovered, **binding)
        random_valid = discovery.validate_proposal_bundle(result.random_comparator, **binding)
        hand_valid = discovery.validate_proposal_bundle(result.hand_comparator, **binding)
        derived_values, derived_available = self._candidate_values_and_available(source)
        local_gates, candidate_scores = self._local_gates_and_scores(result.state)
        diagnostics_consistent = (
            result.diagnostics.transaction_valid
            & result.diagnostics.transaction_applied
            & result.diagnostics.state_valid
            & result.diagnostics.arm_valid
            & result.diagnostics.arm_cache_valid
            & result.diagnostics.transition_identity_matches
            & result.diagnostics.source_binding_matches
            & result.diagnostics.hand_identity_matches
            & result.diagnostics.inputs_finite
            & result.diagnostics.bundle_ready
            & result.diagnostics.random_comparator_ready
            & result.diagnostics.hand_comparator_ready
            & jnp.array_equal(result.diagnostics.family_quotas, self._slot_quota_counts())
            & jnp.array_equal(result.diagnostics.all_local_gates_ready, local_gates)
            & _tree_array_equal(result.diagnostics.candidate_scores, candidate_scores)
        )
        discovery_result_valid = (
            state_valid
            & discovered_valid
            & random_valid
            & hand_valid
            & diagnostics_consistent
        )
        live_binding_valid = (
            jnp.array_equal(result.state.canonical_digest, live.canonical_digest)
            & (result.state.semantic_generation == live.semantic_generation)
            & jnp.array_equal(result.state.source_digest, live.source_digest)
            & jnp.array_equal(result.state.last_transition_id, live.transition_id)
            & (result.state.observation_count == live.state_observation_count)
            & jnp.array_equal(result.state.last_raw_available, live.raw_available)
            & jnp.all(
                jnp.where(
                    live.raw_available,
                    result.state.last_raw_features == live.raw_features,
                    True,
                )
            )
            & jnp.array_equal(result.state.last_candidate_available, derived_available)
            & jnp.all(
                jnp.where(
                    derived_available,
                    result.state.last_candidate_values == derived_values,
                    True,
                )
            )
        )
        incumbent_bundle_valid = discovery.validate_proposal_bundle(
            source.installed_bundle,
            semantic_generation=source.installed_bundle.semantic_generation,
            source_digest=source.installed_bundle.source_digest,
            canonical_digest=source.installed_bundle.canonical_digest,
            transition_id=source.installed_bundle.transition_id,
            state_observation_count=source.installed_bundle.state_observation_count,
        )
        family_quota_layout_valid = self._family_layout_valid(source.installed_bundle)
        expected_semantics = self._installation.semantic_digests_for_bundle(
            source.installed_bundle
        )
        incumbent_semantics_valid = jnp.array_equal(
            source.installed_semantic_digests, expected_semantics
        )
        cold = ~source.installed_slot_mask
        exactly_one_cold = jnp.sum(cold.astype(jnp.int32)) == 1
        source_valid = (
            source_contract_valid
            & discovery_result_valid
            & live_binding_valid
            & incumbent_bundle_valid
            & incumbent_semantics_valid
            & exactly_one_cold
            & family_quota_layout_valid
        )

        target_slot = jnp.argmax(cold.astype(jnp.int32)).astype(jnp.int32)
        target_mask = jax.nn.one_hot(target_slot, cfg.option_budget, dtype=jnp.bool_)
        safe_installed = jnp.clip(
            source.installed_bundle.selected_candidate_indices,
            0,
            cfg.candidate_count - 1,
        )
        pair_means = result.state.pair_novelty_sums / jnp.maximum(
            result.state.pair_novelty_counts.astype(jnp.float32), 1.0
        )
        pair_ready = (
            (result.state.pair_novelty_counts >= cfg.novelty_evidence_floor)
            & (pair_means >= cfg.novelty_threshold)
            & ~jnp.all(self._descriptors[:, None, :] == self._descriptors[None, :, :], axis=2)
        )
        live_pair_ready = jnp.all(
            (~source.installed_slot_mask[None, :]) | pair_ready[:, safe_installed],
            axis=1,
        )
        not_installed = ~jnp.any(
            jnp.arange(cfg.candidate_count, dtype=jnp.int32)[:, None]
            == source.installed_bundle.selected_candidate_indices[None, :],
            axis=1,
        )
        expected_family = self._slot_families[target_slot]
        eligible = (
            local_gates
            & live_pair_ready
            & not_installed
            & (self._families == expected_family)
        )
        selected_index, found = self._lexicographic_best(eligible, candidate_scores)
        safe_selected = jnp.clip(selected_index, 0, cfg.candidate_count - 1)
        selected_indices = source.installed_bundle.selected_candidate_indices.at[
            target_slot
        ].set(safe_selected)
        safe_indices = jnp.clip(selected_indices, 0, cfg.candidate_count - 1)
        provisional = self._make_bundle(
            ready=jnp.asarray(True, dtype=jnp.bool_),
            indices=selected_indices,
            family_ids=self._families[safe_indices],
            descriptors=self._descriptors[safe_indices],
            scores=candidate_scores[safe_indices],
            cumulants=result.state.last_candidate_values[safe_indices],
            semantic_generation=live.semantic_generation,
            source_digest=live.source_digest,
            canonical_digest=live.canonical_digest,
            transition_id=live.transition_id,
            state_observation_count=live.state_observation_count,
        )
        candidate_semantics = self._installation.semantic_digests_for_bundle(provisional)
        changed = jnp.any(
            candidate_semantics != source.installed_semantic_digests, axis=1
        )
        candidate_digest = candidate_semantics[target_slot]
        target_semantic_fresh = found & ~jnp.any(
            jnp.all(
                source.installed_semantic_digests == candidate_digest[None, :],
                axis=1,
            )
        )
        live_slots_preserved = (
            jnp.all(
                jnp.where(
                    source.installed_slot_mask,
                    provisional.selected_candidate_indices
                    == source.installed_bundle.selected_candidate_indices,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    source.installed_slot_mask[:, None],
                    provisional.selected_descriptors
                    == source.installed_bundle.selected_descriptors,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    source.installed_slot_mask[:, None],
                    candidate_semantics == source.installed_semantic_digests,
                    True,
                )
            )
        )
        same_family_selected = found & (
            provisional.selected_family_ids[target_slot] == expected_family
        )
        candidate_family_quota_valid = self._family_layout_valid(provisional)
        exact_target_change = jnp.array_equal(changed, target_mask)
        filtered_bundle_valid = discovery.validate_proposal_bundle(provisional, **binding)
        local_gate_ready = found & local_gates[safe_selected]
        selected_pair_ready = found & live_pair_ready[safe_selected]
        alternate_available = source_valid & found
        candidate_ready = (
            alternate_available
            & local_gate_ready
            & selected_pair_ready
            & same_family_selected
            & candidate_family_quota_valid
            & live_slots_preserved
            & target_semantic_fresh
            & exact_target_change
            & filtered_bundle_valid
        )
        unavailable = self._make_bundle(
            ready=jnp.asarray(False, dtype=jnp.bool_),
            indices=jnp.full((cfg.option_budget,), -1, dtype=jnp.int32),
            family_ids=jnp.full((cfg.option_budget,), -1, dtype=jnp.int32),
            descriptors=jnp.zeros((cfg.option_budget, _DESCRIPTOR_WIDTH), dtype=jnp.int32),
            scores=jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            cumulants=jnp.zeros((cfg.option_budget,), dtype=jnp.float32),
            semantic_generation=live.semantic_generation,
            source_digest=live.source_digest,
            canonical_digest=live.canonical_digest,
            transition_id=live.transition_id,
            state_observation_count=live.state_observation_count,
        )
        filtered_bundle = jax.tree_util.tree_map(
            lambda candidate, fallback: jnp.where(candidate_ready, candidate, fallback),
            provisional,
            unavailable,
        )
        output_semantics = jnp.where(
            candidate_ready,
            candidate_semantics,
            source.installed_semantic_digests,
        )
        output_changed = jnp.where(
            candidate_ready,
            changed,
            jnp.zeros((cfg.option_budget,), dtype=jnp.bool_),
        )
        diagnostics = FreshColdSlotCumulantCohortDiagnostics(
            source_contract_valid=source_contract_valid,
            discovery_result_valid=discovery_result_valid,
            live_binding_valid=live_binding_valid,
            incumbent_bundle_valid=incumbent_bundle_valid,
            incumbent_semantics_valid=incumbent_semantics_valid,
            exactly_one_cold_slot=exactly_one_cold,
            family_quota_layout_valid=family_quota_layout_valid,
            source_valid=source_valid,
            local_gate_ready=local_gate_ready,
            pair_novelty_ready=selected_pair_ready,
            alternate_available=alternate_available,
            same_family_selected=same_family_selected,
            candidate_family_quota_valid=candidate_family_quota_valid,
            live_slots_preserved=live_slots_preserved,
            target_semantic_fresh=target_semantic_fresh,
            exact_target_change=exact_target_change,
            filtered_bundle_valid=filtered_bundle_valid,
            candidate_ready=candidate_ready,
        )
        prepared = FreshColdSlotCumulantCohortPrepared(
            source=source,
            filtered_bundle=filtered_bundle,
            candidate_semantic_digests=output_semantics,
            changed_slots=output_changed,
            target_slot=target_slot,
            target_mask=target_mask,
            selected_candidate_index=jnp.where(candidate_ready, selected_index, -1).astype(
                jnp.int32
            ),
            diagnostics=diagnostics,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_prepared_checksum(prepared)

    def _slot_quota_counts(self) -> Array:
        return jnp.asarray(self._config.family_quotas, dtype=jnp.int32)

    def validate(self, prepared: FreshColdSlotCumulantCohortPrepared) -> Array:
        """Re-derive the entire preparation and reject tamper or replay edits."""

        if type(prepared) is not FreshColdSlotCumulantCohortPrepared:
            raise TypeError("prepared must be exact FreshColdSlotCumulantCohortPrepared")
        _require_array(
            prepared.prepared_checksum,
            name="prepared_checksum",
            shape=(2,),
            dtype=jnp.uint32,
        )
        checksum_valid = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._prepared_payload_arrays(prepared)),
        )
        recomputed = self.prepare(prepared.source)
        return checksum_valid & _tree_array_equal(prepared, recomputed)


__all__ = [
    "FRESH_COLD_SLOT_CUMULANT_COHORT_ADOPTION_AUTHORITY",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_ASSESSMENT",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_CONFIG_SCHEMA",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_EVIDENCE_AUTHORITY",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_GO_NO_GO_AUTHORITY",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_OUTPUT_WRITES",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_PROMOTION_AUTHORITY",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_SAFETY_AUTHORITY",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_SCIENTIFIC_PROMOTION_ALLOWED",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_SELECTION_SEMANTICS",
    "FRESH_COLD_SLOT_CUMULANT_COHORT_UNIVERSE_SCHEMA",
    "FreshColdSlotCumulantCohortDiagnostics",
    "FreshColdSlotCumulantCohortFilter",
    "FreshColdSlotCumulantCohortFilterConfig",
    "FreshColdSlotCumulantCohortPrepared",
    "FreshColdSlotCumulantCohortSource",
]
