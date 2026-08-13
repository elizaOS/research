# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return"
"""Atomic one-agent seam for latent context and prospective lineage retention.

The seam composes exactly one :class:`ContextInference` owner with exactly one
fixed-H=2 :class:`SequentialLineageCache` owner.  It adds only the semantic
birth ledger needed to bind recyclable context slots to lineage identities and
two integrity tokens for the composite transaction.

The transaction is deliberately split at the causal boundary.  :meth:`prepare`
is called before the reward is available and snapshots an ordinal protection
score derived only from already-confirmed lineage rescue counters.  :meth:`step`
then supplies that frozen vector to the context owner, constructs the complete
post-outcome lineage event, and adopts the context, birth ledger, and lineage
successors together or adopts none of them.  Evidence opened or resolved by
the current reward can affect only future preparations; it cannot change the
eviction that exposed that reward.

There is no phase, reset, replay, random-number, memory, feature, world-model,
planner, or policy owner here.  The composite is host/eager-only because its
SHA256 integrity receipts intentionally bind exact host bytes.  Those receipts
detect stale field mutation; they are not caller authentication or external
provenance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.context_inference import (
    ContextInference,
    ContextInferenceConfig,
    ContextInferencePrioritizedUpdateResult,
    ContextInferenceState,
)
from alberta_framework.core.sequential_lineage_cache import (
    SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON,
    SequentialLineageCache,
    SequentialLineageCacheConfig,
    SequentialLineageCacheEvent,
    SequentialLineageCacheProposal,
    SequentialLineageCacheResourceRecord,
    SequentialLineageCacheState,
    SequentialLineageCacheWorkRecord,
    measure_sequential_lineage_cache_state_nbytes,
)

CONTEXT_LINEAGE_RETENTION_CONFIG_SCHEMA = "alberta.context-lineage-retention.config.v1"
CONTEXT_LINEAGE_RETENTION_STATE_SCHEMA = "alberta.context-lineage-retention.state.v1"
CONTEXT_LINEAGE_RETENTION_PREPARATION_SCHEMA = (
    "alberta.context-lineage-retention.preparation.v1"
)
CONTEXT_LINEAGE_RETENTION_RESOURCE_SCHEMA = "alberta.context-lineage-retention.resource.v1"
CONTEXT_LINEAGE_RETENTION_WORK_SCHEMA = "alberta.context-lineage-retention.work.v1"
CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON = (
    SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON
)

_MAX_CONTEXTS = 3
_N_ACTIONS = 2
_OBSERVATION_DIM = 2
_TOKEN_NBYTES = 32

__all__ = [
    "CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON",
    "CONTEXT_LINEAGE_RETENTION_CONFIG_SCHEMA",
    "CONTEXT_LINEAGE_RETENTION_PREPARATION_SCHEMA",
    "CONTEXT_LINEAGE_RETENTION_RESOURCE_SCHEMA",
    "CONTEXT_LINEAGE_RETENTION_STATE_SCHEMA",
    "CONTEXT_LINEAGE_RETENTION_WORK_SCHEMA",
    "ContextLineageRetentionPreparation",
    "ContextLineageRetentionResourceRecord",
    "ContextLineageRetentionSeam",
    "ContextLineageRetentionSeamConfig",
    "ContextLineageRetentionSeamState",
    "ContextLineageRetentionStepResult",
    "ContextLineageRetentionWorkRecord",
]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_python_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without bool/int or tuple/list aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[object, object], left)
        right_mapping = cast(dict[object, object], right)
        return len(left_mapping) == len(right_mapping) and all(
            key in right_mapping
            and _canonical_python_equal(value, right_mapping[key])
            for key, value in left_mapping.items()
        )
    if type(left) in {list, tuple}:
        left_items = cast(list[object] | tuple[object, ...], left)
        right_items = cast(list[object] | tuple[object, ...], right)
        return len(left_items) == len(right_items) and all(
            _canonical_python_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return jnp.asarray(value)


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _digest_tree(schema: str, value: object) -> UInt[Array, " 32"]:
    """Hash one fixed-structure array tree with exact shape/dtype/byte binding."""

    if _contains_tracer(value):
        raise TypeError("context-lineage retention integrity is host/eager-only")
    leaves, structure = jax.tree.flatten(value)
    digest = hashlib.sha256()
    digest.update(schema.encode("ascii"))
    digest.update(repr(structure).encode("utf-8"))
    for leaf in leaves:
        array = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        descriptor = _canonical_json(
            {"dtype": str(array.dtype), "shape": list(array.shape)}
        )
        digest.update(descriptor.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(object, left_structure) != cast(object, right_structure) or len(
        left_leaves
    ) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.ascontiguousarray(np.asarray(jax.device_get(left_leaf)))
        right_array = np.ascontiguousarray(np.asarray(jax.device_get(right_leaf)))
        if (
            left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape
            or left_array.tobytes(order="C") != right_array.tobytes(order="C")
        ):
            return False
    return True


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


def _words_le(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] <= right[..., 1])
    )


def _words_lt(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] < right[..., 1])
    )


@dataclasses.dataclass(frozen=True)
class ContextLineageRetentionSeamConfig:
    """One exact K=3, D=2, A=2 context owner; lineage geometry is derived."""

    context: ContextInferenceConfig

    def __post_init__(self) -> None:
        if type(self.context) is not ContextInferenceConfig:
            raise TypeError("context must be an exact ContextInferenceConfig")
        for name in (
            "model_step_size",
            "error_decay",
            "switch_threshold",
            "novelty_prior_error",
            "update_error_gate",
            "initial_reward_estimate",
        ):
            if type(getattr(self.context, name)) is not float:
                raise TypeError(f"context.{name} must be an exact float")
        if self.context.max_contexts != _MAX_CONTEXTS:
            raise ValueError("context max_contexts must equal 3")
        if self.context.observation_dim != _OBSERVATION_DIM:
            raise ValueError("context observation_dim must equal 2")
        if self.context.n_actions != _N_ACTIONS:
            raise ValueError("context n_actions must equal 2")

    @property
    def lineage(self) -> SequentialLineageCacheConfig:
        """Derive the only lineage geometry compatible with the context owner."""

        return SequentialLineageCacheConfig(
            max_contexts=self.context.max_contexts,
            n_actions=self.context.n_actions,
            observation_dim=self.context.observation_dim,
            initial_reward_estimate=self.context.initial_reward_estimate,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": CONTEXT_LINEAGE_RETENTION_CONFIG_SCHEMA,
            "state_schema": CONTEXT_LINEAGE_RETENTION_STATE_SCHEMA,
            "preparation_schema": CONTEXT_LINEAGE_RETENTION_PREPARATION_SCHEMA,
            "context": self.context.to_config(),
            "lineage": dataclasses.asdict(self.lineage),
            "context_state_owners": 1,
            "sequential_lineage_state_owners": 1,
            "confirmation_horizon": CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON,
            "composite_jit_supported": False,
            "replay_capacity": 0,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ContextLineageRetentionSeamConfig:
        expected_fields = {
            "type",
            "schema",
            "state_schema",
            "preparation_schema",
            "context",
            "lineage",
            "context_state_owners",
            "sequential_lineage_state_owners",
            "confirmation_horizon",
            "composite_jit_supported",
            "replay_capacity",
        }
        if set(payload) != expected_fields:
            raise ValueError("context-lineage retention config fields do not match")
        context_payload = payload["context"]
        if not isinstance(context_payload, Mapping):
            raise TypeError("context config must be a mapping")
        candidate = cls(context=ContextInferenceConfig.from_config(context_payload))
        if not _canonical_python_equal(dict(payload), candidate.to_config()):
            raise ValueError("context-lineage retention config is not canonical")
        return candidate


@chex.dataclass(frozen=True)
class ContextLineageRetentionSeamState:
    """Exactly one context owner, one birth binding, and one lineage owner."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    context: ContextInferenceState
    slot_birth_words: UInt[Array, "3 2"]
    lineage: SequentialLineageCacheState


@chex.dataclass(frozen=True)
class ContextLineageRetentionPreparation:
    """Integrity-bound state-only eviction protection captured before reward."""

    config_token: UInt[Array, " 32"]
    source_content_token: UInt[Array, " 32"]
    source_step_words: UInt[Array, " 2"]
    observation: Float[Array, " 2"]
    action: Int[Array, ""]
    context_coordinates: Float[Array, " 3"]
    active_context: Int[Array, ""]
    active_birth_words: UInt[Array, " 2"]
    live_lineage_words: UInt[Array, "3 2"]
    live_rescue_words: UInt[Array, "3 2"]
    eviction_protection: Float[Array, " 3"]
    source_state_valid: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@dataclasses.dataclass(frozen=True, slots=True)
class ContextLineageRetentionResourceRecord:
    schema: str
    context_state_owners: int
    sequential_lineage_state_owners: int
    birth_ledger_owners: int
    context_coordinate_dim: int
    confirmation_horizon: int
    context_state_nbytes: int
    birth_ledger_nbytes: int
    lineage_state_nbytes: int
    composite_integrity_nbytes: int
    total_persistent_state_nbytes: int
    measured_total_persistent_state_nbytes: int
    preparation_binding_nbytes: int
    logical_atomic_candidate_nbytes: int
    lineage: SequentialLineageCacheResourceRecord
    replay_capacity: int
    persistent_capacity_growth: int
    random_state_nbytes: int
    composite_jit_supported: bool
    scan_supported: bool
    preoutcome_call_order_authenticated: bool
    outcome_provenance_claimed: bool
    external_state_provenance_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ContextLineageRetentionWorkRecord:
    schema: str
    confirmation_horizon: int
    total_steps: int
    pre_outcome_protection_snapshots: int
    protection_binding_recomputations: int
    ordinal_rescue_word_comparisons: int
    context_coordinate_reads: int
    composite_state_audits: int
    context_state_audits: int
    birth_ledger_audits: int
    outer_lineage_binding_audits: int
    outer_lineage_content_digest_evaluations: int
    context_update_proposals: int
    context_reward_prediction_bank_calls: int
    context_scalar_reward_predictions: int
    context_reward_prediction_coefficient_products: int
    context_reward_prediction_dot_additions: int
    context_active_model_prediction_calls: int
    context_active_model_coefficient_products: int
    context_observation_norm_products: int
    birth_ledger_proposals: int
    sequential_lineage_proposals: int
    outer_commit_decisions: int
    composite_state_integrity_evaluations: int
    preparation_integrity_evaluations: int
    lineage: SequentialLineageCacheWorkRecord
    replay_updates: int
    random_draws: int
    reset_callbacks: int
    exact_named_logical_counts: bool
    exhaustive_primitive_operation_count_claimed: bool
    compiled_flop_count_claimed: bool


@chex.dataclass(frozen=True)
class ContextLineageRetentionStepResult:
    """One all-or-none successor plus explicit causal and H=2 diagnostics."""

    state: ContextLineageRetentionSeamState
    preparation: ContextLineageRetentionPreparation
    context_result: ContextInferencePrioritizedUpdateResult
    lineage_event: SequentialLineageCacheEvent
    lineage_proposal: SequentialLineageCacheProposal
    pre_context_coordinates: Float[Array, " 3"]
    post_context_coordinates: Float[Array, " 3"]
    pre_eviction_protection: Float[Array, " 3"]
    source_state_valid: Bool[Array, ""]
    preparation_integrity_valid: Bool[Array, ""]
    preparation_matches_source: Bool[Array, ""]
    protection_binding_valid: Bool[Array, ""]
    birth_binding_valid: Bool[Array, ""]
    context_update_applied: Bool[Array, ""]
    lineage_update_applied: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    context_owner_committed: Bool[Array, ""]
    lineage_owner_committed: Bool[Array, ""]
    context_allocation_requested: Bool[Array, ""]
    context_full_bank_eviction_requested: Bool[Array, ""]
    context_eviction_protection_used: Bool[Array, ""]
    context_eviction_target_adjusted: Bool[Array, ""]
    context_ordinary_lru_slot: Int[Array, ""]
    context_selected_eviction_slot: Int[Array, ""]
    lineage_full_bank_birth: Bool[Array, ""]
    lineage_archive_current_victim_selected: Bool[Array, ""]
    prospective_cache_tested: Bool[Array, ""]
    prospective_quarantine_opened: Bool[Array, ""]
    prospective_second_evidence: Bool[Array, ""]
    prospective_quarantine_confirmed: Bool[Array, ""]
    prospective_quarantine_rejected: Bool[Array, ""]
    lineage_transferred: Bool[Array, ""]
    rescue_incremented: Bool[Array, ""]
    protection_snapshotted_before_outcome: Bool[Array, ""]
    current_outcome_changed_current_eviction_protection: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class ContextLineageRetentionSeam:
    """Host/eager all-or-none owner for one context bank and one H=2 cache."""

    def __init__(self, config: ContextLineageRetentionSeamConfig):
        if type(config) is not ContextLineageRetentionSeamConfig:
            raise TypeError("config must be an exact ContextLineageRetentionSeamConfig")
        canonical = ContextLineageRetentionSeamConfig.from_config(config.to_config())
        self._config = canonical
        self._context = ContextInference(canonical.context)
        self._lineage = SequentialLineageCache(canonical.lineage)
        self._config_token = jnp.asarray(
            tuple(
                hashlib.sha256(
                    _canonical_json(canonical.to_config()).encode("utf-8")
                ).digest()
            ),
            dtype=jnp.uint8,
        )

    @property
    def config(self) -> ContextLineageRetentionSeamConfig:
        return self._config

    @property
    def context(self) -> ContextInference:
        return self._context

    @property
    def lineage(self) -> SequentialLineageCache:
        return self._lineage

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ContextLineageRetentionSeam:
        return cls(ContextLineageRetentionSeamConfig.from_config(payload))

    def _state_token(
        self,
        state: ContextLineageRetentionSeamState,
    ) -> Array:
        return _digest_tree(
            CONTEXT_LINEAGE_RETENTION_STATE_SCHEMA,
            (
                state.config_token,
                state.context,
                state.slot_birth_words,
                state.lineage,
            ),
        )

    def _seal_state(
        self,
        state: ContextLineageRetentionSeamState,
    ) -> ContextLineageRetentionSeamState:
        return cast(
            ContextLineageRetentionSeamState,
            cast(Any, state).replace(content_token=self._state_token(state)),
        )

    def _preparation_token(
        self,
        preparation: ContextLineageRetentionPreparation,
    ) -> Array:
        return _digest_tree(
            CONTEXT_LINEAGE_RETENTION_PREPARATION_SCHEMA,
            (
                preparation.config_token,
                preparation.source_content_token,
                preparation.source_step_words,
                preparation.observation,
                preparation.action,
                preparation.context_coordinates,
                preparation.active_context,
                preparation.active_birth_words,
                preparation.live_lineage_words,
                preparation.live_rescue_words,
                preparation.eviction_protection,
                preparation.source_state_valid,
            ),
        )

    def _seal_preparation(
        self,
        preparation: ContextLineageRetentionPreparation,
    ) -> ContextLineageRetentionPreparation:
        return cast(
            ContextLineageRetentionPreparation,
            cast(Any, preparation).replace(
                content_token=self._preparation_token(preparation)
            ),
        )

    def _require_state_contract(self, state: ContextLineageRetentionSeamState) -> None:
        _require_array(
            state.config_token,
            name="state.config_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.dtype(jnp.uint8),
        )
        _require_array(
            state.content_token,
            name="state.content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.dtype(jnp.uint8),
        )
        _require_array(
            state.slot_birth_words,
            name="state.slot_birth_words",
            shape=(_MAX_CONTEXTS, 2),
            dtype=jnp.dtype(jnp.uint32),
        )
        self._context._require_state_contract(state.context)
        self._lineage._require_state_contract(state.lineage)

    def _require_preparation_contract(
        self,
        preparation: ContextLineageRetentionPreparation,
    ) -> None:
        for name in ("config_token", "source_content_token", "content_token"):
            _require_array(
                getattr(preparation, name),
                name=f"preparation.{name}",
                shape=(_TOKEN_NBYTES,),
                dtype=jnp.dtype(jnp.uint8),
            )
        _require_array(
            preparation.source_step_words,
            name="preparation.source_step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_array(
            preparation.observation,
            name="preparation.observation",
            shape=(_OBSERVATION_DIM,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            preparation.action,
            name="preparation.action",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            preparation.context_coordinates,
            name="preparation.context_coordinates",
            shape=(_MAX_CONTEXTS,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            preparation.active_context,
            name="preparation.active_context",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            preparation.active_birth_words,
            name="preparation.active_birth_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        for name in ("live_lineage_words", "live_rescue_words"):
            _require_array(
                getattr(preparation, name),
                name=f"preparation.{name}",
                shape=(_MAX_CONTEXTS, 2),
                dtype=jnp.dtype(jnp.uint32),
            )
        _require_array(
            preparation.eviction_protection,
            name="preparation.eviction_protection",
            shape=(_MAX_CONTEXTS,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            preparation.source_state_valid,
            name="preparation.source_state_valid",
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
        )

    def init(self) -> ContextLineageRetentionSeamState:
        context = self._context.init()
        unsigned = ContextLineageRetentionSeamState(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            context=context,
            slot_birth_words=jnp.zeros((_MAX_CONTEXTS, 2), dtype=jnp.uint32),
            lineage=self._lineage.init(),
        )
        sealed = self._seal_state(unsigned)
        if not bool(self.state_is_valid(sealed)):
            raise RuntimeError("context-lineage retention genesis is invalid")
        return sealed

    def _birth_ledger_valid(self, state: ContextLineageRetentionSeamState) -> Array:
        births = state.slot_birth_words
        in_use = state.context.in_use
        zero = jnp.zeros((2,), dtype=jnp.uint32)
        unused_zero = jnp.all(
            jnp.where(in_use[:, None], jnp.asarray(True), births == zero)
        )
        not_future = jnp.all(
            jnp.where(
                in_use,
                _words_le(births, state.context.step_words[None, :]),
                jnp.asarray(True),
            )
        )
        equality = jnp.all(births[:, None, :] == births[None, :, :], axis=2)
        diagonal = jnp.eye(_MAX_CONTEXTS, dtype=jnp.bool_)
        duplicate_live = (in_use[:, None] & in_use[None, :]) & equality & ~diagonal
        return unused_zero & not_future & ~jnp.any(duplicate_live)

    def state_is_valid(
        self,
        state: ContextLineageRetentionSeamState,
    ) -> Bool[Array, ""]:
        """Validate static contracts, integrity, and both child-owner bindings."""

        self._require_state_contract(state)
        if _contains_tracer(state):
            raise TypeError("context-lineage retention validity is host/eager-only")
        config_bound = np.array_equal(
            np.asarray(state.config_token), np.asarray(self._config_token)
        )
        content_bound = np.array_equal(
            np.asarray(state.content_token), np.asarray(self._state_token(state))
        )
        context_valid = self._context.state_is_valid(state.context)
        ledger_valid = self._birth_ledger_valid(state)
        lineage_valid = self._lineage.state_valid(
            state.lineage,
            state.context.step_words,
            state.slot_birth_words,
            state.context.in_use,
        )
        return (
            jnp.asarray(config_bound & content_bound, dtype=jnp.bool_)
            & context_valid
            & ledger_valid
            & lineage_valid
        )

    def context_coordinates(
        self,
        state: ContextLineageRetentionSeamState,
    ) -> Float[Array, " 3"]:
        """Return the current inferred-context one-hot coordinates."""

        self._require_state_contract(state)
        return self._context.context_onehot(state.context)

    def _ordinal_eviction_protection(
        self,
        state: ContextLineageRetentionSeamState,
    ) -> Array:
        """Map exact rescue words to an order-preserving float32 rank in [0, 2]."""

        words = state.lineage.live_rescue_words
        in_use = state.context.in_use
        smaller = _words_lt(words[:, None, :], words[None, :, :])
        ranks = jnp.sum((in_use[:, None] & smaller).astype(jnp.int32), axis=0)
        return jnp.where(in_use, ranks.astype(jnp.float32), jnp.float32(0.0))

    def prepare(
        self,
        state: ContextLineageRetentionSeamState,
        observation: Array,
        action: Array,
    ) -> ContextLineageRetentionPreparation:
        """Seal state-only protection before the corresponding reward exists."""

        self._require_state_contract(state)
        obs = _require_array(
            observation,
            name="observation",
            shape=(_OBSERVATION_DIM,),
            dtype=jnp.dtype(jnp.float32),
        )
        action_value = _require_array(
            action,
            name="action",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        if _contains_tracer((state, obs, action_value)):
            raise TypeError("context-lineage retention preparation is host/eager-only")
        source_valid = self.state_is_valid(state)
        safe_active = jnp.clip(state.context.active_context, 0, _MAX_CONTEXTS - 1)
        protection = jnp.where(
            source_valid,
            self._ordinal_eviction_protection(state),
            jnp.zeros((_MAX_CONTEXTS,), dtype=jnp.float32),
        )
        unsigned = ContextLineageRetentionPreparation(
            config_token=self._config_token,
            source_content_token=state.content_token,
            source_step_words=state.context.step_words,
            observation=obs,
            action=action_value,
            context_coordinates=self._context.context_onehot(state.context),
            active_context=state.context.active_context,
            active_birth_words=state.slot_birth_words[safe_active],
            live_lineage_words=state.lineage.live_lineage_words,
            live_rescue_words=state.lineage.live_rescue_words,
            eviction_protection=protection,
            source_state_valid=source_valid,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return self._seal_preparation(unsigned)

    def _preparation_integrity_valid(
        self,
        preparation: ContextLineageRetentionPreparation,
    ) -> bool:
        return bool(
            np.array_equal(
                np.asarray(preparation.config_token),
                np.asarray(self._config_token),
            )
            and np.array_equal(
                np.asarray(preparation.content_token),
                np.asarray(self._preparation_token(preparation)),
            )
        )

    def step(
        self,
        state: ContextLineageRetentionSeamState,
        preparation: ContextLineageRetentionPreparation,
        reward: Array,
    ) -> ContextLineageRetentionStepResult:
        """Propose the outcome update and adopt both child owners or neither."""

        self._require_state_contract(state)
        self._require_preparation_contract(preparation)
        reward_value = _require_array(
            reward,
            name="reward",
            shape=(),
            dtype=jnp.dtype(jnp.float32),
        )
        if _contains_tracer((state, preparation, reward_value)):
            raise TypeError("context-lineage retention step is host/eager-only")

        source_valid = self.state_is_valid(state)
        preparation_integrity = self._preparation_integrity_valid(preparation)
        expected_preparation = self.prepare(
            state,
            preparation.observation,
            preparation.action,
        )
        preparation_matches = _tree_exact_equal(preparation, expected_preparation)
        protection_binding = bool(
            preparation_matches
            and np.array_equal(
                np.asarray(preparation.eviction_protection),
                np.asarray(expected_preparation.eviction_protection),
            )
        )

        context_result = self._context.update_result_with_eviction_protection(
            state.context,
            preparation.observation,
            preparation.action,
            reward_value,
            preparation.eviction_protection,
        )
        safe_target = jnp.clip(
            context_result.state.active_context,
            0,
            _MAX_CONTEXTS - 1,
        )
        allocated_births = state.slot_birth_words.at[safe_target].set(
            context_result.post_step_words
        )
        post_births = jnp.where(
            context_result.allocation_requested,
            allocated_births,
            state.slot_birth_words,
        )
        event = SequentialLineageCacheEvent(
            source_step_words=state.context.step_words,
            post_step_words=context_result.post_step_words,
            source_birth_words=state.slot_birth_words,
            post_birth_words=post_births,
            source_in_use=state.context.in_use,
            post_in_use=context_result.state.in_use,
            source_reward_weights=state.context.reward_weights,
            observation=preparation.observation,
            action=preparation.action,
            reward=reward_value,
            allocated=context_result.allocation_requested,
            evicted=context_result.full_bank_eviction_requested,
            target_slot=context_result.state.active_context,
            context_update_applied=context_result.update_applied,
        )
        lineage_proposal = self._lineage.propose(state.lineage, event)
        unsigned_candidate = ContextLineageRetentionSeamState(
            config_token=state.config_token,
            content_token=state.content_token,
            context=context_result.state,
            slot_birth_words=post_births,
            lineage=lineage_proposal.state,
        )
        candidate = self._seal_state(unsigned_candidate)
        candidate_valid = self.state_is_valid(candidate)
        birth_binding = bool(
            np.array_equal(
                np.asarray(event.source_birth_words),
                np.asarray(state.slot_birth_words),
            )
            and np.array_equal(
                np.asarray(event.post_birth_words),
                np.asarray(post_births),
            )
        )
        applied = (
            source_valid
            & jnp.asarray(preparation_integrity, dtype=jnp.bool_)
            & jnp.asarray(preparation_matches, dtype=jnp.bool_)
            & preparation.source_state_valid
            & jnp.asarray(protection_binding, dtype=jnp.bool_)
            & context_result.update_applied
            & lineage_proposal.update_applied
            & jnp.asarray(birth_binding, dtype=jnp.bool_)
            & candidate_valid
        )
        committed = candidate if bool(applied) else state
        committed_pre_coordinates = preparation.context_coordinates
        committed_post_coordinates = jnp.where(
            applied,
            context_result.context_onehot,
            committed_pre_coordinates,
        )

        def committed_flag(value: Array) -> Array:
            return jnp.asarray(applied & value, dtype=jnp.bool_)

        return ContextLineageRetentionStepResult(
            state=committed,
            preparation=preparation,
            context_result=context_result,
            lineage_event=event,
            lineage_proposal=lineage_proposal,
            pre_context_coordinates=committed_pre_coordinates,
            post_context_coordinates=committed_post_coordinates,
            pre_eviction_protection=preparation.eviction_protection,
            source_state_valid=source_valid,
            preparation_integrity_valid=jnp.asarray(
                preparation_integrity, dtype=jnp.bool_
            ),
            preparation_matches_source=jnp.asarray(
                preparation_matches, dtype=jnp.bool_
            ),
            protection_binding_valid=jnp.asarray(protection_binding, dtype=jnp.bool_),
            birth_binding_valid=jnp.asarray(birth_binding, dtype=jnp.bool_),
            context_update_applied=context_result.update_applied,
            lineage_update_applied=lineage_proposal.update_applied,
            candidate_state_valid=candidate_valid,
            context_owner_committed=committed_flag(context_result.update_applied),
            lineage_owner_committed=committed_flag(lineage_proposal.update_applied),
            context_allocation_requested=committed_flag(
                context_result.allocation_requested
            ),
            context_full_bank_eviction_requested=committed_flag(
                context_result.full_bank_eviction_requested
            ),
            context_eviction_protection_used=committed_flag(
                context_result.eviction_protection_used
            ),
            context_eviction_target_adjusted=committed_flag(
                context_result.eviction_target_adjusted
            ),
            context_ordinary_lru_slot=jnp.where(
                applied,
                context_result.ordinary_lru_slot,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            context_selected_eviction_slot=jnp.where(
                applied,
                context_result.selected_eviction_slot,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            lineage_full_bank_birth=committed_flag(lineage_proposal.full_bank_birth),
            lineage_archive_current_victim_selected=committed_flag(
                lineage_proposal.archive_current_victim_selected
            ),
            prospective_cache_tested=committed_flag(lineage_proposal.cache_tested),
            prospective_quarantine_opened=committed_flag(
                lineage_proposal.quarantine_opened
            ),
            prospective_second_evidence=committed_flag(
                lineage_proposal.quarantine_second_evidence
            ),
            prospective_quarantine_confirmed=committed_flag(
                lineage_proposal.quarantine_confirmed
            ),
            prospective_quarantine_rejected=committed_flag(
                lineage_proposal.quarantine_rejected
            ),
            lineage_transferred=committed_flag(lineage_proposal.lineage_transferred),
            rescue_incremented=committed_flag(lineage_proposal.rescue_incremented),
            protection_snapshotted_before_outcome=jnp.asarray(
                applied, dtype=jnp.bool_
            ),
            current_outcome_changed_current_eviction_protection=jnp.asarray(
                False, dtype=jnp.bool_
            ),
            update_applied=jnp.asarray(applied, dtype=jnp.bool_),
        )

    def resource_record(
        self,
        state: ContextLineageRetentionSeamState | None = None,
    ) -> ContextLineageRetentionResourceRecord:
        """Return exact persistent bytes for the fixed K=3, D=2, A=2 seam."""

        source = self.init() if state is None else state
        if not bool(self.state_is_valid(source)):
            raise ValueError("resource measurement requires a valid seam state")
        context_bytes = self._context.resource_budget.state_nbytes
        ledger_bytes = _MAX_CONTEXTS * 2 * jnp.dtype(jnp.uint32).itemsize
        integrity_bytes = 2 * _TOKEN_NBYTES
        base_bytes = context_bytes + ledger_bytes + integrity_bytes
        lineage = self._lineage.resource_record(
            n_agents=1,
            base_scan_carry_nbytes=base_bytes,
        )
        lineage_bytes = measure_sequential_lineage_cache_state_nbytes(source.lineage)
        total = base_bytes + lineage_bytes
        measured = _tree_nbytes(source)
        preparation_bytes = _tree_nbytes(
            self.prepare(
                source,
                jnp.zeros((_OBSERVATION_DIM,), dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
            )
        )
        if measured != total or lineage.total_scan_carry_nbytes != total:
            raise AssertionError("context-lineage persistent byte formula disagrees")
        return ContextLineageRetentionResourceRecord(
            schema=CONTEXT_LINEAGE_RETENTION_RESOURCE_SCHEMA,
            context_state_owners=1,
            sequential_lineage_state_owners=1,
            birth_ledger_owners=1,
            context_coordinate_dim=_MAX_CONTEXTS,
            confirmation_horizon=CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON,
            context_state_nbytes=context_bytes,
            birth_ledger_nbytes=ledger_bytes,
            lineage_state_nbytes=lineage_bytes,
            composite_integrity_nbytes=integrity_bytes,
            total_persistent_state_nbytes=total,
            measured_total_persistent_state_nbytes=measured,
            preparation_binding_nbytes=preparation_bytes,
            logical_atomic_candidate_nbytes=total,
            lineage=lineage,
            replay_capacity=0,
            persistent_capacity_growth=0,
            random_state_nbytes=0,
            composite_jit_supported=False,
            scan_supported=False,
            preoutcome_call_order_authenticated=False,
            outcome_provenance_claimed=False,
            external_state_provenance_claimed=False,
        )

    def work_record(self, *, total_steps: int) -> ContextLineageRetentionWorkRecord:
        """Return exact named logical counts for one invocation on every step."""

        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        k = _MAX_CONTEXTS
        d = _OBSERVATION_DIM
        lineage = self._lineage.work_record(total_steps=total_steps, n_agents=1)
        return ContextLineageRetentionWorkRecord(
            schema=CONTEXT_LINEAGE_RETENTION_WORK_SCHEMA,
            confirmation_horizon=CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON,
            total_steps=total_steps,
            pre_outcome_protection_snapshots=total_steps,
            protection_binding_recomputations=total_steps,
            ordinal_rescue_word_comparisons=2 * total_steps * k * k,
            context_coordinate_reads=3 * total_steps,
            composite_state_audits=4 * total_steps,
            context_state_audits=6 * total_steps,
            birth_ledger_audits=4 * total_steps,
            outer_lineage_binding_audits=4 * total_steps,
            outer_lineage_content_digest_evaluations=4 * total_steps,
            context_update_proposals=total_steps,
            context_reward_prediction_bank_calls=total_steps,
            context_scalar_reward_predictions=total_steps * k,
            context_reward_prediction_coefficient_products=total_steps * k * d,
            context_reward_prediction_dot_additions=total_steps * k * max(d - 1, 0),
            context_active_model_prediction_calls=total_steps,
            context_active_model_coefficient_products=total_steps * d,
            context_observation_norm_products=total_steps * d,
            birth_ledger_proposals=total_steps,
            sequential_lineage_proposals=total_steps,
            outer_commit_decisions=total_steps,
            composite_state_integrity_evaluations=5 * total_steps,
            preparation_integrity_evaluations=3 * total_steps,
            lineage=lineage,
            replay_updates=0,
            random_draws=0,
            reset_callbacks=0,
            exact_named_logical_counts=True,
            exhaustive_primitive_operation_count_claimed=False,
            compiled_flop_count_claimed=False,
        )
