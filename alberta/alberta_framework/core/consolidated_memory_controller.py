# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Causal base-control composition for consolidated procedural memory.

``ConsolidatedProceduralMemoryController`` is an optional L0 sidecar around a
caller-owned categorical base action.  A valid decision queries procedural
memory before any write, asks the stateless procedural policy for a hard-safe
proposal, and otherwise returns the caller's exact base action.  Feedback is
the only path that writes outcome evidence, and it must exactly match the
pending decision, actions, semantic provenance, and lifecycle identity.

Memory exhaustion or an optional query/write rejection permanently freezes
the memory sidecar while later valid base actions continue.  Corruption of the
persistent composed state is different: it fails closed and requires recovery
from a valid checkpoint.  The controller never dispatches an action, mutates a
learning agent, creates a skill identity, uses randomness, promotes evidence,
or makes a scientific claim.  The opt-in Prototype procedural mechanism exists;
physical live-environment integration and efficacy remain open.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.consolidated_memory import (
    ConsolidatedMemory,
    ConsolidatedMemoryConfig,
    ConsolidatedMemoryResourceBudget,
    ConsolidatedMemoryState,
    MemoryWriteDiagnostics,
    ProceduralMemoryRecord,
    ProceduralMemoryRequest,
    ProceduralMemoryRetrieval,
    SemanticMemoryRecord,
    SemanticMemoryRequest,
    SemanticMemoryRetrieval,
    SemanticMemoryStepResult,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicy,
    ConsolidatedProceduralMemoryPolicyConfig,
    ConsolidatedProceduralMemoryPolicyResourceBudget,
    ConsolidatedProceduralMemoryProposal,
)

CONSOLIDATED_MEMORY_CONTROLLER_CONFIG_SCHEMA = (
    "alberta.consolidated-procedural-memory-controller.config.v1"
)
CONSOLIDATED_MEMORY_CONTROLLER_CHECKPOINT_SCHEMA = (
    "alberta.consolidated-procedural-memory-controller.state.v1"
)
CONSOLIDATED_MEMORY_CONTROLLER_MECHANISM_STATUS = "l0_controller_integration_only"
CONSOLIDATED_MEMORY_CONTROLLER_ACTION_DISPATCH_AUTHORITY = False
CONSOLIDATED_MEMORY_CONTROLLER_AGENT_MUTATION_AUTHORITY = False
CONSOLIDATED_MEMORY_CONTROLLER_AUTONOMOUS_SKILL_CREATION_AUTHORITY = False
CONSOLIDATED_MEMORY_CONTROLLER_PROMOTION_AUTHORITY = False
CONSOLIDATED_MEMORY_CONTROLLER_SCIENTIFIC_PROMOTION_ALLOWED = False

MEMORY_ERROR_NONE = 0
MEMORY_ERROR_QUERY_REJECTED = 1
MEMORY_ERROR_WRITE_REJECTED = 2
MEMORY_ERROR_CAP_EXHAUSTED = 3
MEMORY_ERROR_POLICY_CONTRACT = 4
MEMORY_ERROR_COMPOSED_STATE_INVALID = 5

DECISION_REASON_AVAILABLE = 0
DECISION_REASON_STATE_INVALID = 1
DECISION_REASON_BASE_UNAVAILABLE = 2
DECISION_REASON_BASE_SCORES = 3
DECISION_REASON_BASE_ACTION = 4
DECISION_REASON_BASE_UNSAFE = 5
DECISION_REASON_IDENTITY = 6
DECISION_REASON_DUPLICATE = 7
DECISION_REASON_PENDING_FALLBACK = 8
DECISION_REASON_MEMORY_UNAVAILABLE_FALLBACK = 9
DECISION_REASON_POLICY_FALLBACK = 10
DECISION_REASON_MEMORY_PROPOSAL = 11
DECISION_REASON_STALE = 12

FEEDBACK_REASON_APPLIED = 0
FEEDBACK_REASON_STATE_INVALID = 1
FEEDBACK_REASON_NO_PENDING = 2
FEEDBACK_REASON_IDENTITY = 3
FEEDBACK_REASON_DUPLICATE_EVENT = 4
FEEDBACK_REASON_ACTION = 5
FEEDBACK_REASON_BINDING = 6
FEEDBACK_REASON_NONFINITE = 7
FEEDBACK_REASON_MEMORY_CHANGED = 8
FEEDBACK_REASON_WRITE_REJECTED = 9
FEEDBACK_REASON_STALE_EVENT = 10

SEMANTIC_REASON_APPLIED = 0
SEMANTIC_REASON_NOT_REQUESTED = 1
SEMANTIC_REASON_STATE_INVALID = 2
SEMANTIC_REASON_PENDING_PROCEDURAL_FEEDBACK = 3
SEMANTIC_REASON_MEMORY_UNAVAILABLE = 4
SEMANTIC_REASON_CAP_EXHAUSTED = 5
SEMANTIC_REASON_INPUT_REJECTED = 6

_DIGEST_BYTES = 32
_DECISION_WORDS = 4
_INT32_MAX = 2**31 - 1


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype metadata")
    array = jnp.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}, got {array.dtype}")
    return array


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        array = jnp.asarray(value, dtype=jnp.int32)
    else:
        array = jnp.asarray(value)
    return _require_array(array, name=name, shape=(), dtype=jnp.int32)


def _float32_scalar(value: float | Array, *, name: str) -> Array:
    if type(value) is float:
        array = jnp.asarray(value, dtype=jnp.float32)
    else:
        array = jnp.asarray(value)
    return _require_array(array, name=name, shape=(), dtype=jnp.float32)


def _bool_scalar(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        array = jnp.asarray(value, dtype=jnp.bool_)
    else:
        array = jnp.asarray(value)
    return _require_array(array, name=name, shape=(), dtype=jnp.bool_)


def _digest_nonzero(value: Array) -> Array:
    return jnp.any(value != jnp.asarray(0, dtype=jnp.uint8))


def _identity_nonzero(value: Array) -> Array:
    return jnp.any(value != jnp.asarray(0, dtype=jnp.uint32))


def _identity_strictly_advances(candidate: Array, previous: Array) -> Array:
    equal_prefix = jnp.asarray(True, dtype=jnp.bool_)
    greater = jnp.asarray(False, dtype=jnp.bool_)
    for index in range(_DECISION_WORDS):
        greater = greater | (equal_prefix & (candidate[index] > previous[index]))
        equal_prefix = equal_prefix & (candidate[index] == previous[index])
    return greater


def _saturating_increment(value: Array, increment: Array | int = 1) -> Array:
    amount = jnp.asarray(increment, dtype=jnp.int32)
    room = jnp.asarray(_INT32_MAX, dtype=jnp.int32) - value
    return value + jnp.minimum(jnp.maximum(amount, 0), room)


def _tree_nbytes(tree: object) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_logical_scalars(tree: object) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedProceduralMemoryControllerConfig:
    """Exact compatible memory and stateless policy configuration."""

    memory: ConsolidatedMemoryConfig
    policy: ConsolidatedProceduralMemoryPolicyConfig

    SCHEMA_VERSION: ClassVar[str] = CONSOLIDATED_MEMORY_CONTROLLER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.memory) is not ConsolidatedMemoryConfig:
            raise TypeError("memory must be an exact ConsolidatedMemoryConfig")
        if type(self.policy) is not ConsolidatedProceduralMemoryPolicyConfig:
            raise TypeError(
                "policy must be an exact ConsolidatedProceduralMemoryPolicyConfig"
            )
        if self.memory.procedural_payload_dim != self.policy.n_actions:
            raise ValueError("procedural payload width must equal policy n_actions")
        if self.memory.procedural_outcome_dim != self.policy.outcome_dim:
            raise ValueError("procedural outcome width must equal policy outcome_dim")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "mechanism_status": CONSOLIDATED_MEMORY_CONTROLLER_MECHANISM_STATUS,
            "memory": self.memory.to_config(),
            "policy": self.policy.to_config(),
            "caller_owns_base_control": True,
            "hard_safety_mask_required": True,
            "query_before_write_required": True,
            "action_dispatch_authority": False,
            "agent_mutation_authority": False,
            "autonomous_skill_creation_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls, value: object
    ) -> ConsolidatedProceduralMemoryControllerConfig:
        if type(value) is not dict:
            raise ValueError("controller config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {
            "schema",
            "mechanism_status",
            "memory",
            "policy",
            "caller_owns_base_control",
            "hard_safety_mask_required",
            "query_before_write_required",
            "action_dispatch_authority",
            "agent_mutation_authority",
            "autonomous_skill_creation_authority",
            "promotion_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("controller config fields differ from schema v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "mechanism_status": CONSOLIDATED_MEMORY_CONTROLLER_MECHANISM_STATUS,
            "caller_owns_base_control": True,
            "hard_safety_mask_required": True,
            "query_before_write_required": True,
            "action_dispatch_authority": False,
            "agent_mutation_authority": False,
            "autonomous_skill_creation_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }
        if any(
            type(raw[name]) is not type(item) or raw[name] != item
            for name, item in fixed.items()
        ):
            raise ValueError("controller config fixed fields differ")
        return cls(
            memory=ConsolidatedMemoryConfig.from_config(raw["memory"]),
            policy=ConsolidatedProceduralMemoryPolicyConfig.from_config(raw["policy"]),
        )


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryControllerState:
    """Persistent memory, causal pending slot, bindings, and audit counters."""

    memory: ConsolidatedMemoryState
    controller_binding_digest: UInt[Array, " 32"]
    policy_identity_digest: UInt[Array, " 32"]
    pending: Bool[Array, ""]
    pending_decision_id: UInt[Array, " 4"]
    pending_base_action: Int[Array, ""]
    pending_effective_action: Int[Array, ""]
    pending_memory_selected: Bool[Array, ""]
    pending_hard_safety_mask: Bool[Array, " n_actions"]
    pending_semantic_digest: UInt[Array, " 32"]
    pending_generation: Int[Array, ""]
    pending_provenance_digest: UInt[Array, " 32"]
    pending_representation_revision: Int[Array, ""]
    pending_source_revision: Int[Array, ""]
    pending_lifecycle_link_available: Bool[Array, ""]
    pending_lifecycle_digest: UInt[Array, " 32"]
    pending_lifecycle_generation: Int[Array, ""]
    pending_lifecycle_revision: Int[Array, ""]
    pending_query_accepted: Bool[Array, ""]
    pending_query_slot: Int[Array, ""]
    pending_query_operation_before: Int[Array, ""]
    pending_query_operation_after: Int[Array, ""]
    pending_procedural_write_count_before: Int[Array, ""]
    has_last_decision: Bool[Array, ""]
    last_decision_id: UInt[Array, " 4"]
    has_last_feedback_event: Bool[Array, ""]
    last_feedback_event_id: UInt[Array, " 4"]
    memory_unavailable: Bool[Array, ""]
    memory_error: Int[Array, ""]
    decision_count: Int[Array, ""]
    tracked_decision_count: Int[Array, ""]
    memory_proposal_count: Int[Array, ""]
    base_fallback_count: Int[Array, ""]
    memory_unavailable_noop_count: Int[Array, ""]
    feedback_count: Int[Array, ""]
    successful_memory_write_count: Int[Array, ""]
    failed_memory_write_count: Int[Array, ""]
    recorded_success_count: Int[Array, ""]
    recorded_failure_count: Int[Array, ""]
    memory_error_count: Int[Array, ""]
    checksum: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryDecisionDiagnostics:
    state_valid: Bool[Array, ""]
    checksum_valid: Bool[Array, ""]
    base_action_available: Bool[Array, ""]
    base_scores_finite: Bool[Array, ""]
    base_scores_nonnegative: Bool[Array, ""]
    base_positive_mass: Bool[Array, ""]
    base_action_index_valid: Bool[Array, ""]
    base_action_positive_mass: Bool[Array, ""]
    base_action_hard_safe: Bool[Array, ""]
    base_valid: Bool[Array, ""]
    decision_identity_valid: Bool[Array, ""]
    decision_id_strictly_advancing: Bool[Array, ""]
    duplicate_decision: Bool[Array, ""]
    stale_decision: Bool[Array, ""]
    pending_conflict: Bool[Array, ""]
    memory_available_before: Bool[Array, ""]
    query_attempted: Bool[Array, ""]
    query_transaction_applied: Bool[Array, ""]
    query_accepted: Bool[Array, ""]
    query_pre_write_verified: Bool[Array, ""]
    policy_contract_valid: Bool[Array, ""]
    policy_proposal_available: Bool[Array, ""]
    memory_selected: Bool[Array, ""]
    used_base_fallback: Bool[Array, ""]
    feedback_trackable: Bool[Array, ""]
    action_available: Bool[Array, ""]
    memory_became_unavailable: Bool[Array, ""]
    memory_error: Int[Array, ""]
    operation_count_before: Int[Array, ""]
    operation_count_after: Int[Array, ""]
    reason: Int[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryDecisionResult:
    state: ConsolidatedProceduralMemoryControllerState
    action_available: Bool[Array, ""]
    action: Int[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    memory_proposed_action: Int[Array, ""]
    retrieval: ProceduralMemoryRetrieval
    proposal: ConsolidatedProceduralMemoryProposal
    diagnostics: ConsolidatedProceduralMemoryDecisionDiagnostics


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryFeedbackDiagnostics:
    state_valid: Bool[Array, ""]
    checksum_valid: Bool[Array, ""]
    pending_available: Bool[Array, ""]
    decision_identity_matches: Bool[Array, ""]
    feedback_event_identity_valid: Bool[Array, ""]
    feedback_event_strictly_advancing: Bool[Array, ""]
    duplicate_feedback_event: Bool[Array, ""]
    stale_feedback_event: Bool[Array, ""]
    base_action_matches: Bool[Array, ""]
    effective_action_matches: Bool[Array, ""]
    semantic_binding_matches: Bool[Array, ""]
    provenance_binding_matches: Bool[Array, ""]
    representation_source_matches: Bool[Array, ""]
    lifecycle_binding_matches: Bool[Array, ""]
    memory_unchanged_since_query: Bool[Array, ""]
    feedback_inputs_finite: Bool[Array, ""]
    feedback_valid: Bool[Array, ""]
    write_attempted: Bool[Array, ""]
    write_applied: Bool[Array, ""]
    success_recorded: Bool[Array, ""]
    failure_recorded: Bool[Array, ""]
    pending_cleared: Bool[Array, ""]
    memory_became_unavailable: Bool[Array, ""]
    memory_error: Int[Array, ""]
    operation_count_before: Int[Array, ""]
    operation_count_after: Int[Array, ""]
    reason: Int[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryFeedbackResult:
    state: ConsolidatedProceduralMemoryControllerState
    write: MemoryWriteDiagnostics
    diagnostics: ConsolidatedProceduralMemoryFeedbackDiagnostics


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryDispatchCancellationDiagnostics:
    """Audit for canceling one recommendation that was never executed."""

    state_valid_before: Bool[Array, ""]
    cancellation_requested: Bool[Array, ""]
    pending_available: Bool[Array, ""]
    decision_identity_matches: Bool[Array, ""]
    effective_action_matches: Bool[Array, ""]
    cancellation_required: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    cancellation_applied: Bool[Array, ""]
    transaction_satisfied: Bool[Array, ""]
    memory_unchanged: Bool[Array, ""]
    counters_unchanged: Bool[Array, ""]
    learning_applied: Bool[Array, ""]
    evidence_written: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedProceduralMemoryDispatchCancellationResult:
    """State and diagnostics for an exact non-learning pending cancellation."""

    state: ConsolidatedProceduralMemoryControllerState
    diagnostics: ConsolidatedProceduralMemoryDispatchCancellationDiagnostics


@chex.dataclass(frozen=True)
class ConsolidatedSemanticMemoryControllerDiagnostics:
    """Audit for one shared-store semantic query-before-write transaction."""

    state_valid: Bool[Array, ""]
    checksum_valid: Bool[Array, ""]
    transaction_allowed: Bool[Array, ""]
    procedural_feedback_pending: Bool[Array, ""]
    memory_available_before: Bool[Array, ""]
    operation_capacity_available: Bool[Array, ""]
    transaction_attempted: Bool[Array, ""]
    query_before_write_verified: Bool[Array, ""]
    retrieval_accepted: Bool[Array, ""]
    write_applied: Bool[Array, ""]
    terminal_capacity_reached: Bool[Array, ""]
    memory_became_unavailable: Bool[Array, ""]
    operation_count_before: Int[Array, ""]
    operation_count_after: Int[Array, ""]
    reason: Int[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedSemanticMemoryControllerResult:
    """Shared controller state and semantic pre-write retrieval result."""

    state: ConsolidatedProceduralMemoryControllerState
    retrieval: SemanticMemoryRetrieval
    write: MemoryWriteDiagnostics
    diagnostics: ConsolidatedSemanticMemoryControllerDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedProceduralMemoryControllerResourceBudget:
    """Exact fixed allocations, work ceilings, and authority declaration."""

    persistent_logical_scalars: int
    persistent_state_bytes: int
    pending_slots: int
    n_actions: int
    outcome_dim: int
    maximum_memory_operations: int
    memory_operations_per_complete_lifecycle: int
    maximum_complete_memory_lifecycles: int
    memory_queries_per_tracked_decision: int
    policy_proposals_per_tracked_decision: int
    memory_writes_per_valid_feedback: int
    pending_cancellation_identity_checks_per_call: int
    memory_writes_per_dispatch_cancellation: int
    counter_advances_per_dispatch_cancellation: int
    learning_updates_per_dispatch_cancellation: int
    categorical_cells_scanned_per_decision: int
    outcome_cells_written_per_feedback: int
    saturating_telemetry_counter_ceiling: int
    persistent_growth_per_event_bytes: int
    random_generator_calls_per_event: int
    action_dispatches_per_event: int
    agent_parameter_mutations_per_event: int
    autonomous_skill_creations_per_event: int
    promotion_decisions_per_event: int
    caller_base_fallback_guaranteed_when_memory_unavailable: bool
    exact_semantic_provenance_binding_required: bool
    exact_lifecycle_binding_required: bool
    action_dispatch_authority: bool
    agent_mutation_authority: bool
    autonomous_skill_creation_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str
    memory: ConsolidatedMemoryResourceBudget
    policy: ConsolidatedProceduralMemoryPolicyResourceBudget


class ConsolidatedProceduralMemoryController:
    """Persistent, optional procedural-memory sidecar for categorical control."""

    def __init__(self, config: ConsolidatedProceduralMemoryControllerConfig) -> None:
        if type(config) is not ConsolidatedProceduralMemoryControllerConfig:
            raise TypeError(
                "config must be an exact ConsolidatedProceduralMemoryControllerConfig"
            )
        self._config = config
        self._memory = ConsolidatedMemory(config.memory)
        self._policy = ConsolidatedProceduralMemoryPolicy(config.policy)
        encoded = json.dumps(
            config.to_config(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).digest()
        self._config_digest = jnp.asarray(tuple(digest), dtype=jnp.uint8)
        self._checksum_seed = jnp.asarray(
            tuple(
                int.from_bytes(digest[index : index + 4], "little")
                for index in range(0, 16, 4)
            ),
            dtype=jnp.uint32,
        )
        self._policy_identity_digest = jnp.asarray(
            tuple(bytes.fromhex(self._policy.identity_sha256)), dtype=jnp.uint8
        )
        dummy = self._initial_state(
            source_digest=jnp.ones((_DIGEST_BYTES,), dtype=jnp.uint8),
            semantic_namespace_digest=jnp.full(
                (_DIGEST_BYTES,), 2, dtype=jnp.uint8
            ),
            representation_revision=jnp.asarray(0, dtype=jnp.int32),
            source_revision=jnp.asarray(0, dtype=jnp.int32),
        )
        self._persistent_state_bytes = _tree_nbytes(dummy)
        self._persistent_logical_scalars = _tree_logical_scalars(dummy)

    @property
    def config(self) -> ConsolidatedProceduralMemoryControllerConfig:
        return self._config

    @property
    def memory(self) -> ConsolidatedMemory:
        return self._memory

    @property
    def policy(self) -> ConsolidatedProceduralMemoryPolicy:
        return self._policy

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls, value: object
    ) -> ConsolidatedProceduralMemoryController:
        return cls(ConsolidatedProceduralMemoryControllerConfig.from_config(value))

    @property
    def resource_budget(self) -> ConsolidatedProceduralMemoryControllerResourceBudget:
        cfg = self._config
        return ConsolidatedProceduralMemoryControllerResourceBudget(
            persistent_logical_scalars=self._persistent_logical_scalars,
            persistent_state_bytes=self._persistent_state_bytes,
            pending_slots=1,
            n_actions=cfg.policy.n_actions,
            outcome_dim=cfg.policy.outcome_dim,
            maximum_memory_operations=cfg.memory.max_operations,
            memory_operations_per_complete_lifecycle=2,
            maximum_complete_memory_lifecycles=cfg.memory.max_operations // 2,
            memory_queries_per_tracked_decision=1,
            policy_proposals_per_tracked_decision=1,
            memory_writes_per_valid_feedback=1,
            pending_cancellation_identity_checks_per_call=2,
            memory_writes_per_dispatch_cancellation=0,
            counter_advances_per_dispatch_cancellation=0,
            learning_updates_per_dispatch_cancellation=0,
            categorical_cells_scanned_per_decision=cfg.policy.n_actions,
            outcome_cells_written_per_feedback=cfg.policy.outcome_dim,
            saturating_telemetry_counter_ceiling=_INT32_MAX,
            persistent_growth_per_event_bytes=0,
            random_generator_calls_per_event=0,
            action_dispatches_per_event=0,
            agent_parameter_mutations_per_event=0,
            autonomous_skill_creations_per_event=0,
            promotion_decisions_per_event=0,
            caller_base_fallback_guaranteed_when_memory_unavailable=True,
            exact_semantic_provenance_binding_required=True,
            exact_lifecycle_binding_required=True,
            action_dispatch_authority=False,
            agent_mutation_authority=False,
            autonomous_skill_creation_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=CONSOLIDATED_MEMORY_CONTROLLER_CHECKPOINT_SCHEMA,
            memory=self._memory.resource_budget,
            policy=self._policy.resource_budget,
        )

    def _binding_digest(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> Array:
        revisions = jnp.stack((representation_revision, source_revision)).astype(
            jnp.int32
        )
        revision_bytes = jax.lax.bitcast_convert_type(revisions, jnp.uint8).reshape(
            (8,)
        )
        repeated_revisions = jnp.tile(revision_bytes, (4,))
        positions = jnp.arange(_DIGEST_BYTES, dtype=jnp.uint8)
        return (
            self._config_digest
            ^ source_digest
            ^ jnp.roll(semantic_namespace_digest, 1)
            ^ jnp.roll(self._policy_identity_digest, 2)
            ^ repeated_revisions
            ^ (positions * jnp.asarray(17, dtype=jnp.uint8))
        ).astype(jnp.uint8)

    def _checksum(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> Array:
        zero_checksum = jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32)
        checksumless = dataclasses.replace(state, checksum=zero_checksum)
        encoded_leaves: list[Array] = []
        for leaf_index, leaf in enumerate(jax.tree_util.tree_leaves(checksumless)):
            if jnp.dtype(leaf.dtype) == jnp.dtype(jnp.float32):
                words = jax.lax.bitcast_convert_type(leaf, jnp.uint32).reshape((-1,))
            else:
                words = leaf.astype(jnp.uint32).reshape((-1,))
            leaf_tag = jnp.asarray(leaf_index + 1, dtype=jnp.uint32)
            local_positions = jnp.arange(words.shape[0], dtype=jnp.uint32) + 1
            encoded_leaves.append(
                words
                ^ (leaf_tag * jnp.asarray(0x9E3779B9, dtype=jnp.uint32))
                ^ (local_positions * jnp.asarray(0x85EBCA6B, dtype=jnp.uint32))
            )
        encoded = jnp.concatenate(encoded_leaves)
        positions = jnp.arange(encoded.shape[0], dtype=jnp.uint32) + 1
        first = jnp.bitwise_xor.reduce(
            (encoded ^ positions) * jnp.asarray(0x01000193, dtype=jnp.uint32)
        )
        second = jnp.sum(
            (encoded + positions * jnp.asarray(0x27D4EB2D, dtype=jnp.uint32))
            * jnp.asarray(0x165667B1, dtype=jnp.uint32),
            dtype=jnp.uint32,
        )
        third = jnp.bitwise_xor.reduce(
            (encoded + jnp.roll(encoded, 1))
            * (positions | jnp.asarray(1, dtype=jnp.uint32))
        )
        fourth = jnp.sum(
            (encoded ^ jnp.roll(encoded, 7))
            * jnp.asarray(0x9E3779B1, dtype=jnp.uint32),
            dtype=jnp.uint32,
        )
        return self._checksum_seed ^ jnp.stack(
            (
                first,
                second,
                third,
                fourth,
            )
        ).astype(jnp.uint32)

    def _with_checksum(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> ConsolidatedProceduralMemoryControllerState:
        zeroed = dataclasses.replace(
            state, checksum=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32)
        )
        return dataclasses.replace(zeroed, checksum=self._checksum(zeroed))

    def _initial_state(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> ConsolidatedProceduralMemoryControllerState:
        memory = self._memory.init(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        minus_one = jnp.asarray(-1, dtype=jnp.int32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        state = ConsolidatedProceduralMemoryControllerState(
            memory=memory,
            controller_binding_digest=self._binding_digest(
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation_revision,
                source_revision=source_revision,
            ),
            policy_identity_digest=self._policy_identity_digest,
            pending=false,
            pending_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            pending_base_action=minus_one,
            pending_effective_action=minus_one,
            pending_memory_selected=false,
            pending_hard_safety_mask=jnp.zeros(
                (self._config.policy.n_actions,), dtype=jnp.bool_
            ),
            pending_semantic_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            pending_generation=minus_one,
            pending_provenance_digest=jnp.zeros(
                (_DIGEST_BYTES,), dtype=jnp.uint8
            ),
            pending_representation_revision=minus_one,
            pending_source_revision=minus_one,
            pending_lifecycle_link_available=false,
            pending_lifecycle_digest=jnp.zeros(
                (_DIGEST_BYTES,), dtype=jnp.uint8
            ),
            pending_lifecycle_generation=minus_one,
            pending_lifecycle_revision=minus_one,
            pending_query_accepted=false,
            pending_query_slot=minus_one,
            pending_query_operation_before=minus_one,
            pending_query_operation_after=minus_one,
            pending_procedural_write_count_before=minus_one,
            has_last_decision=false,
            last_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            has_last_feedback_event=false,
            last_feedback_event_id=jnp.zeros(
                (_DECISION_WORDS,), dtype=jnp.uint32
            ),
            memory_unavailable=false,
            memory_error=zero,
            decision_count=zero,
            tracked_decision_count=zero,
            memory_proposal_count=zero,
            base_fallback_count=zero,
            memory_unavailable_noop_count=zero,
            feedback_count=zero,
            successful_memory_write_count=zero,
            failed_memory_write_count=zero,
            recorded_success_count=zero,
            recorded_failure_count=zero,
            memory_error_count=zero,
            checksum=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def init(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> ConsolidatedProceduralMemoryControllerState:
        """Return an empty controller bound to one exact memory namespace."""

        source_digest = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        semantic_namespace_digest = _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(
            representation_revision, name="representation_revision"
        )
        source = _int32_scalar(source_revision, name="source_revision")
        return self._initial_state(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )

    def _validate_state_static_contract(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> None:
        if not isinstance(state, ConsolidatedProceduralMemoryControllerState):
            raise TypeError(
                "state must be a ConsolidatedProceduralMemoryControllerState"
            )
        self._memory._validate_state_static_contract(state.memory)
        for name in (
            "controller_binding_digest",
            "policy_identity_digest",
            "pending_semantic_digest",
            "pending_provenance_digest",
            "pending_lifecycle_digest",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(_DIGEST_BYTES,),
                dtype=jnp.uint8,
            )
        for name in (
            "pending_decision_id",
            "last_decision_id",
            "last_feedback_event_id",
            "checksum",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(_DECISION_WORDS,),
                dtype=jnp.uint32,
            )
        _require_array(
            state.pending_hard_safety_mask,
            name="state.pending_hard_safety_mask",
            shape=(self._config.policy.n_actions,),
            dtype=jnp.bool_,
        )
        for name in (
            "pending",
            "pending_memory_selected",
            "pending_lifecycle_link_available",
            "pending_query_accepted",
            "has_last_decision",
            "has_last_feedback_event",
            "memory_unavailable",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(),
                dtype=jnp.bool_,
            )
        excluded = {
            "memory",
            "controller_binding_digest",
            "policy_identity_digest",
            "pending_decision_id",
            "pending_hard_safety_mask",
            "pending_semantic_digest",
            "pending_provenance_digest",
            "pending_lifecycle_digest",
            "last_decision_id",
            "last_feedback_event_id",
            "checksum",
            "pending",
            "pending_memory_selected",
            "pending_lifecycle_link_available",
            "pending_query_accepted",
            "has_last_decision",
            "has_last_feedback_event",
            "memory_unavailable",
        }
        for field in dataclasses.fields(ConsolidatedProceduralMemoryControllerState):
            if field.name not in excluded:
                _require_array(
                    getattr(state, field.name),
                    name=f"state.{field.name}",
                    shape=(),
                    dtype=jnp.int32,
                )

    def _validate_request_static(self, request: ProceduralMemoryRequest) -> None:
        if not isinstance(request, ProceduralMemoryRequest):
            raise TypeError("request must be a ProceduralMemoryRequest")
        for name in (
            "semantic_digest",
            "provenance_digest",
            "lifecycle_digest",
        ):
            _require_array(
                getattr(request, name),
                name=f"request.{name}",
                shape=(_DIGEST_BYTES,),
                dtype=jnp.uint8,
            )
        _require_array(
            request.lifecycle_link_available,
            name="request.lifecycle_link_available",
            shape=(),
            dtype=jnp.bool_,
        )
        for name in (
            "generation",
            "representation_revision",
            "source_revision",
            "lifecycle_generation",
            "lifecycle_revision",
        ):
            _require_array(
                getattr(request, name),
                name=f"request.{name}",
                shape=(),
                dtype=jnp.int32,
            )

    def _request_is_valid(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        request: ProceduralMemoryRequest,
    ) -> Array:
        return (
            _digest_nonzero(request.semantic_digest)
            & _digest_nonzero(request.provenance_digest)
            & (request.generation >= 0)
            & (request.representation_revision == state.memory.representation_revision)
            & (request.source_revision == state.memory.source_revision)
            & request.lifecycle_link_available
            & _digest_nonzero(request.lifecycle_digest)
            & (request.lifecycle_generation >= 0)
            & (request.lifecycle_revision >= 0)
        )

    def _pending_is_blank(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> Array:
        return (
            (~_identity_nonzero(state.pending_decision_id))
            & (state.pending_base_action == -1)
            & (state.pending_effective_action == -1)
            & (~state.pending_memory_selected)
            & (~jnp.any(state.pending_hard_safety_mask))
            & (~_digest_nonzero(state.pending_semantic_digest))
            & (state.pending_generation == -1)
            & (~_digest_nonzero(state.pending_provenance_digest))
            & (state.pending_representation_revision == -1)
            & (state.pending_source_revision == -1)
            & (~state.pending_lifecycle_link_available)
            & (~_digest_nonzero(state.pending_lifecycle_digest))
            & (state.pending_lifecycle_generation == -1)
            & (state.pending_lifecycle_revision == -1)
            & (~state.pending_query_accepted)
            & (state.pending_query_slot == -1)
            & (state.pending_query_operation_before == -1)
            & (state.pending_query_operation_after == -1)
            & (state.pending_procedural_write_count_before == -1)
        )

    def _pending_is_valid(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> Array:
        n_actions = self._config.policy.n_actions
        base_index = jnp.clip(state.pending_base_action, 0, n_actions - 1)
        effective_index = jnp.clip(state.pending_effective_action, 0, n_actions - 1)
        query_slot_valid = jnp.where(
            state.pending_query_accepted,
            (state.pending_query_slot >= 0)
            & (state.pending_query_slot < self._config.memory.procedural_capacity),
            state.pending_query_slot == -1,
        )
        return (
            _identity_nonzero(state.pending_decision_id)
            & state.has_last_decision
            & jnp.array_equal(
                state.pending_decision_id, state.last_decision_id
            )
            & (state.pending_base_action >= 0)
            & (state.pending_base_action < n_actions)
            & (state.pending_effective_action >= 0)
            & (state.pending_effective_action < n_actions)
            & state.pending_hard_safety_mask[base_index]
            & state.pending_hard_safety_mask[effective_index]
            & jnp.where(
                state.pending_memory_selected,
                state.pending_query_accepted,
                state.pending_effective_action == state.pending_base_action,
            )
            & _digest_nonzero(state.pending_semantic_digest)
            & (state.pending_generation >= 0)
            & _digest_nonzero(state.pending_provenance_digest)
            & (
                state.pending_representation_revision
                == state.memory.representation_revision
            )
            & (state.pending_source_revision == state.memory.source_revision)
            & state.pending_lifecycle_link_available
            & _digest_nonzero(state.pending_lifecycle_digest)
            & (state.pending_lifecycle_generation >= 0)
            & (state.pending_lifecycle_revision >= 0)
            & query_slot_valid
            & (state.pending_query_operation_before >= 0)
            & (
                state.pending_query_operation_after
                == state.pending_query_operation_before + 1
            )
            & (
                state.memory.operation_count
                == state.pending_query_operation_after
            )
            & (
                state.memory.procedural_write_count
                == state.pending_procedural_write_count_before
            )
        )

    def _state_is_valid(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
        checksum_valid: Array | None = None,
    ) -> Array:
        memory_valid = self._memory.validate_state(
            state.memory,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
        )
        if checksum_valid is None:
            checksum_valid = jnp.array_equal(state.checksum, self._checksum(state))
        binding_valid = jnp.array_equal(
            state.controller_binding_digest,
            self._binding_digest(
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation_revision,
                source_revision=source_revision,
            ),
        ) & jnp.array_equal(
            state.policy_identity_digest, self._policy_identity_digest
        )
        counter_names = (
            "decision_count",
            "tracked_decision_count",
            "memory_proposal_count",
            "base_fallback_count",
            "memory_unavailable_noop_count",
            "feedback_count",
            "successful_memory_write_count",
            "failed_memory_write_count",
            "recorded_success_count",
            "recorded_failure_count",
            "memory_error_count",
        )
        counters_valid = jnp.asarray(True, dtype=jnp.bool_)
        for name in counter_names:
            counters_valid = counters_valid & (getattr(state, name) >= 0)
        counter_relations = (
            (state.tracked_decision_count <= state.decision_count)
            & (state.memory_proposal_count <= state.decision_count)
            & (state.base_fallback_count <= state.decision_count)
            & (state.memory_unavailable_noop_count <= state.base_fallback_count)
            & (state.successful_memory_write_count <= state.feedback_count)
            & (state.failed_memory_write_count <= state.feedback_count)
            & (state.recorded_success_count <= state.successful_memory_write_count)
            & (state.recorded_failure_count <= state.successful_memory_write_count)
        )
        terminal_valid = (
            (state.memory_error >= MEMORY_ERROR_NONE)
            & (state.memory_error <= MEMORY_ERROR_COMPOSED_STATE_INVALID)
            & (
                state.memory_unavailable
                == (state.memory_error != MEMORY_ERROR_NONE)
            )
            & (~(state.memory_unavailable & state.pending))
            & jnp.where(
                state.memory_unavailable,
                state.memory_error_count >= 1,
                state.memory_error_count == 0,
            )
        )
        pending_contract = jnp.where(
            state.pending, self._pending_is_valid(state), self._pending_is_blank(state)
        )
        last_ids_valid = jnp.where(
            state.has_last_decision,
            _identity_nonzero(state.last_decision_id),
            ~_identity_nonzero(state.last_decision_id),
        ) & jnp.where(
            state.has_last_feedback_event,
            _identity_nonzero(state.last_feedback_event_id),
            ~_identity_nonzero(state.last_feedback_event_id),
        )
        return (
            memory_valid
            & checksum_valid
            & binding_valid
            & counters_valid
            & counter_relations
            & terminal_valid
            & pending_contract
            & last_ids_valid
        )

    def validate_state(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> Array:
        """Return whether structure, checksum, bindings, and lifecycle are valid."""

        self._validate_state_static_contract(state)
        source_digest = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        semantic_namespace_digest = _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(
            representation_revision, name="representation_revision"
        )
        source = _int32_scalar(source_revision, name="source_revision")
        return self._state_is_valid(
            state,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )

    def _blank_retrieval(
        self, *, state_valid: Array, request_valid: Array
    ) -> ProceduralMemoryRetrieval:
        false = jnp.asarray(False, dtype=jnp.bool_)
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ProceduralMemoryRetrieval(
            accepted=false,
            transaction_applied=false,
            slot=jnp.asarray(-1, dtype=jnp.int32),
            payload=jnp.zeros(
                (self._config.memory.procedural_payload_dim,), dtype=jnp.float32
            ),
            confidence=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_count=zero,
            evidence_mean=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_m2=jnp.asarray(0.0, dtype=jnp.float32),
            success_count=zero,
            failure_count=zero,
            outcome_mean=jnp.zeros(
                (self._config.memory.procedural_outcome_dim,), dtype=jnp.float32
            ),
            outcome_m2=jnp.zeros(
                (self._config.memory.procedural_outcome_dim,), dtype=jnp.float32
            ),
            lifecycle_link_available=false,
            lifecycle_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            lifecycle_generation=jnp.asarray(-1, dtype=jnp.int32),
            lifecycle_revision=jnp.asarray(-1, dtype=jnp.int32),
            state_valid=state_valid,
            request_valid=request_valid,
            identity_found=false,
            compatible=false,
            fresh=false,
            confidence_ok=false,
        )

    def _blank_semantic_retrieval(
        self, *, state_valid: Array, request_valid: Array
    ) -> SemanticMemoryRetrieval:
        false = jnp.asarray(False, dtype=jnp.bool_)
        zero = jnp.asarray(0, dtype=jnp.int32)
        return SemanticMemoryRetrieval(
            accepted=false,
            transaction_applied=false,
            slot=jnp.asarray(-1, dtype=jnp.int32),
            payload=jnp.zeros(
                (self._config.memory.semantic_payload_dim,), dtype=jnp.float32
            ),
            confidence=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_count=zero,
            evidence_mean=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_m2=jnp.asarray(0.0, dtype=jnp.float32),
            state_valid=state_valid,
            request_valid=request_valid,
            identity_found=false,
            compatible=false,
            fresh=false,
            confidence_ok=false,
        )

    @staticmethod
    def _blank_write(*, state_valid: Array, record_valid: Array) -> MemoryWriteDiagnostics:
        false = jnp.asarray(False, dtype=jnp.bool_)
        return MemoryWriteDiagnostics(
            transaction_applied=false,
            wrote=false,
            merged=false,
            revised=false,
            replaced=false,
            reset_evidence=false,
            slot=jnp.asarray(-1, dtype=jnp.int32),
            state_valid=state_valid,
            record_valid=record_valid,
            identity_collision=false,
            generation_compatible=false,
            metadata_compatible=false,
        )

    def _clear_pending(
        self, state: ConsolidatedProceduralMemoryControllerState
    ) -> ConsolidatedProceduralMemoryControllerState:
        return dataclasses.replace(
            state,
            pending=jnp.asarray(False, dtype=jnp.bool_),
            pending_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            pending_base_action=jnp.asarray(-1, dtype=jnp.int32),
            pending_effective_action=jnp.asarray(-1, dtype=jnp.int32),
            pending_memory_selected=jnp.asarray(False, dtype=jnp.bool_),
            pending_hard_safety_mask=jnp.zeros(
                (self._config.policy.n_actions,), dtype=jnp.bool_
            ),
            pending_semantic_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            pending_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_provenance_digest=jnp.zeros(
                (_DIGEST_BYTES,), dtype=jnp.uint8
            ),
            pending_representation_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_source_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_lifecycle_link_available=jnp.asarray(False, dtype=jnp.bool_),
            pending_lifecycle_digest=jnp.zeros(
                (_DIGEST_BYTES,), dtype=jnp.uint8
            ),
            pending_lifecycle_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_lifecycle_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_query_accepted=jnp.asarray(False, dtype=jnp.bool_),
            pending_query_slot=jnp.asarray(-1, dtype=jnp.int32),
            pending_query_operation_before=jnp.asarray(-1, dtype=jnp.int32),
            pending_query_operation_after=jnp.asarray(-1, dtype=jnp.int32),
            pending_procedural_write_count_before=jnp.asarray(
                -1, dtype=jnp.int32
            ),
        )

    def cancel_pending_dispatch(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        cancellation_requested: bool | Array,
        decision_id: Array,
        effective_action: int | Array,
    ) -> ConsolidatedProceduralMemoryDispatchCancellationResult:
        """Cancel an exactly owned recommendation that did not execute.

        This transaction clears only the matching pending recommendation. It
        writes no memory evidence, advances no counter, applies no learning,
        and consumes no randomness. A missing/nonmatching owner is not a
        cancellation target; malformed dynamic state or a matching owner with
        a mismatched action leaves the complete state unchanged and does not
        satisfy the transaction.
        """

        self._validate_state_static_contract(state)
        requested = _bool_scalar(
            cancellation_requested,
            name="cancellation_requested",
        )
        expected_decision = _require_array(
            decision_id,
            name="decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        expected_action = _int32_scalar(
            effective_action,
            name="effective_action",
        )
        state_valid = self._state_is_valid(
            state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        decision_matches = state.pending & jnp.array_equal(
            expected_decision,
            state.pending_decision_id,
        )
        action_matches = state.pending & (
            expected_action == state.pending_effective_action
        )
        cancellation_required = requested & state.pending
        candidate = self._with_checksum(self._clear_pending(state))
        candidate_valid = self._state_is_valid(
            candidate,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        applied = (
            state_valid
            & cancellation_required
            & decision_matches
            & action_matches
            & candidate_valid
        )
        final_state = cast(
            ConsolidatedProceduralMemoryControllerState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, operand=None),
        )
        transaction_satisfied = state_valid & (
            (~cancellation_required) | applied
        )
        return ConsolidatedProceduralMemoryDispatchCancellationResult(
            state=final_state,
            diagnostics=(
                ConsolidatedProceduralMemoryDispatchCancellationDiagnostics(
                    state_valid_before=state_valid,
                    cancellation_requested=requested,
                    pending_available=state.pending,
                    decision_identity_matches=decision_matches,
                    effective_action_matches=action_matches,
                    cancellation_required=cancellation_required,
                    candidate_state_valid=candidate_valid,
                    cancellation_applied=applied,
                    transaction_satisfied=transaction_satisfied,
                    memory_unchanged=jnp.asarray(True, dtype=jnp.bool_),
                    counters_unchanged=jnp.asarray(True, dtype=jnp.bool_),
                    learning_applied=jnp.asarray(False, dtype=jnp.bool_),
                    evidence_written=jnp.asarray(False, dtype=jnp.bool_),
                )
            ),
        )

    def _request_matches_pending(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        request: ProceduralMemoryRequest,
    ) -> tuple[Array, Array, Array, Array]:
        semantic = jnp.array_equal(
            request.semantic_digest, state.pending_semantic_digest
        ) & (request.generation == state.pending_generation)
        provenance = jnp.array_equal(
            request.provenance_digest, state.pending_provenance_digest
        )
        representation_source = (
            request.representation_revision
            == state.pending_representation_revision
        ) & (request.source_revision == state.pending_source_revision)
        lifecycle = (
            request.lifecycle_link_available
            == state.pending_lifecycle_link_available
        ) & jnp.array_equal(
            request.lifecycle_digest, state.pending_lifecycle_digest
        ) & (
            request.lifecycle_generation == state.pending_lifecycle_generation
        ) & (
            request.lifecycle_revision == state.pending_lifecycle_revision
        )
        return semantic, provenance, representation_source, lifecycle

    def _proposal_contract_valid(
        self,
        proposal: ConsolidatedProceduralMemoryProposal,
        *,
        retrieval: ProceduralMemoryRetrieval,
        hard_safety_mask: Array,
    ) -> Array:
        n_actions = self._config.policy.n_actions
        index = jnp.clip(proposal.action, 0, n_actions - 1)
        score_mass = proposal.categorical_score_mass
        score_mass_valid = jnp.all(jnp.isfinite(score_mass)) & jnp.all(
            score_mass >= 0.0
        )
        expected_scores = jnp.where(
            jnp.all(jnp.isfinite(retrieval.payload)),
            retrieval.payload,
            jnp.zeros_like(retrieval.payload),
        )
        expected_safe_positive = (
            hard_safety_mask & (score_mass > 0.0) & score_mass_valid
        )
        available_contract = jnp.where(
            proposal.available,
            (proposal.action >= 0)
            & (proposal.action < n_actions)
            & hard_safety_mask[index]
            & proposal.safe_positive_mask[index]
            & jnp.isfinite(proposal.selected_mass)
            & (proposal.selected_mass > 0.0),
            (proposal.action == -1) & (proposal.selected_mass == 0.0),
        )
        return (
            jnp.isfinite(proposal.selected_mass)
            & jnp.isfinite(proposal.safe_mass_total)
            & (proposal.safe_mass_total >= 0.0)
            & jnp.array_equal(proposal.hard_safety_mask, hard_safety_mask)
            & jnp.array_equal(score_mass, expected_scores)
            & jnp.array_equal(
                proposal.safe_positive_mask, expected_safe_positive
            )
            & available_contract
        )

    def semantic_step(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        request: SemanticMemoryRequest,
        record: SemanticMemoryRecord,
        transaction_allowed: bool | Array = True,
    ) -> ConsolidatedSemanticMemoryControllerResult:
        """Query then write semantic memory through the shared controller state.

        A procedural feedback owner serializes access to the shared store: the
        semantic transaction is an exact no-op while ``state.pending`` is set.
        The caller should settle that owner first. Invalid semantic metadata is
        also a no-op; exhausting the common operation cap permanently disables
        the optional memory sidecar while leaving caller base control available.
        """

        self._validate_state_static_contract(state)
        self._memory._validate_semantic_request_static(request)
        self._memory._validate_semantic_record_static(record)
        allowed = _bool_scalar(
            transaction_allowed,
            name="transaction_allowed",
        )
        return cast(
            ConsolidatedSemanticMemoryControllerResult,
            self._semantic_step_jit(state, request, record, allowed),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _semantic_step_jit(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        request: SemanticMemoryRequest,
        record: SemanticMemoryRecord,
        transaction_allowed: Array,
    ) -> ConsolidatedSemanticMemoryControllerResult:
        checksum_valid = jnp.array_equal(state.checksum, self._checksum(state))
        state_valid = self._state_is_valid(
            state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
            checksum_valid=checksum_valid,
        )
        request_valid = self._memory._semantic_request_valid(
            state.memory,
            request,
        )
        record_valid = self._memory._semantic_record_valid(
            state.memory,
            record,
        )
        semantic_input_valid = request_valid & record_valid
        memory_available_before = state_valid & (~state.memory_unavailable)
        operation_capacity_available = (
            state.memory.operation_count < self._config.memory.max_operations
        )
        common_gate = (
            transaction_allowed
            & state_valid
            & (~state.pending)
            & (~state.memory_unavailable)
            & semantic_input_valid
        )
        transaction_attempted = common_gate & operation_capacity_available
        blank_retrieval = self._blank_semantic_retrieval(
            state_valid=state_valid,
            request_valid=request_valid,
        )
        blank_write = self._blank_write(
            state_valid=state_valid,
            record_valid=record_valid,
        )
        blank_step = SemanticMemoryStepResult(
            state=state.memory,
            retrieval=blank_retrieval,
            write=blank_write,
        )
        raw_step = cast(
            SemanticMemoryStepResult,
            jax.lax.cond(
                transaction_attempted,
                lambda _: self._memory.semantic_step(
                    state.memory,
                    request,
                    record,
                ),
                lambda _: blank_step,
                operand=None,
            ),
        )
        stepped_memory_valid = self._memory.validate_state(
            raw_step.state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        query_before_write_verified = (
            transaction_attempted
            & raw_step.retrieval.transaction_applied
            & raw_step.write.transaction_applied
            & raw_step.write.wrote
            & stepped_memory_valid
            & (
                raw_step.state.operation_count
                == state.memory.operation_count + 1
            )
            & (
                raw_step.state.semantic_query_count
                == state.memory.semantic_query_count + 1
            )
            & (
                raw_step.state.semantic_write_count
                == state.memory.semantic_write_count + 1
            )
            & (
                raw_step.state.procedural_query_count
                == state.memory.procedural_query_count
            )
            & (
                raw_step.state.procedural_write_count
                == state.memory.procedural_write_count
            )
        )
        step_applied = query_before_write_verified
        capacity_exhausted_before = common_gate & (~operation_capacity_available)
        capacity_exhausted_after = step_applied & (
            raw_step.state.operation_count >= self._config.memory.max_operations
        )
        terminal_capacity_reached = (
            capacity_exhausted_before | capacity_exhausted_after
        )
        next_memory = cast(
            ConsolidatedMemoryState,
            jax.lax.cond(
                step_applied,
                lambda _: raw_step.state,
                lambda _: state.memory,
                operand=None,
            ),
        )
        updated = dataclasses.replace(
            state,
            memory=next_memory,
            memory_unavailable=(
                state.memory_unavailable | terminal_capacity_reached
            ),
            memory_error=jnp.where(
                terminal_capacity_reached,
                jnp.asarray(MEMORY_ERROR_CAP_EXHAUSTED, dtype=jnp.int32),
                state.memory_error,
            ).astype(jnp.int32),
            memory_error_count=_saturating_increment(
                state.memory_error_count,
                terminal_capacity_reached.astype(jnp.int32),
            ),
        )
        updated = self._with_checksum(updated)
        state_changed = step_applied | terminal_capacity_reached
        candidate = cast(
            ConsolidatedProceduralMemoryControllerState,
            jax.lax.cond(
                state_valid & state_changed,
                lambda _: updated,
                lambda _: state,
                operand=None,
            ),
        )
        candidate_valid = self._state_is_valid(
            candidate,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        committed_state = cast(
            ConsolidatedProceduralMemoryControllerState,
            jax.lax.cond(
                state_valid & candidate_valid,
                lambda _: candidate,
                lambda _: state,
                operand=None,
            ),
        )
        exposed_retrieval = cast(
            SemanticMemoryRetrieval,
            jax.lax.cond(
                step_applied & candidate_valid,
                lambda _: raw_step.retrieval,
                lambda _: blank_retrieval,
                operand=None,
            ),
        )
        exposed_write = cast(
            MemoryWriteDiagnostics,
            jax.lax.cond(
                step_applied & candidate_valid,
                lambda _: raw_step.write,
                lambda _: blank_write,
                operand=None,
            ),
        )
        memory_became_unavailable = (
            state_valid & candidate_valid & terminal_capacity_reached
        )
        reason = jnp.asarray(SEMANTIC_REASON_APPLIED, dtype=jnp.int32)
        reason = jnp.where(
            transaction_allowed & semantic_input_valid & (~step_applied),
            jnp.asarray(SEMANTIC_REASON_INPUT_REJECTED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            capacity_exhausted_before | capacity_exhausted_after,
            jnp.asarray(SEMANTIC_REASON_CAP_EXHAUSTED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            state.memory_unavailable,
            jnp.asarray(SEMANTIC_REASON_MEMORY_UNAVAILABLE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            state.pending,
            jnp.asarray(
                SEMANTIC_REASON_PENDING_PROCEDURAL_FEEDBACK,
                dtype=jnp.int32,
            ),
            reason,
        )
        reason = jnp.where(
            transaction_allowed & (~semantic_input_valid),
            jnp.asarray(SEMANTIC_REASON_INPUT_REJECTED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~transaction_allowed,
            jnp.asarray(SEMANTIC_REASON_NOT_REQUESTED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            (~state_valid) | (~candidate_valid),
            jnp.asarray(SEMANTIC_REASON_STATE_INVALID, dtype=jnp.int32),
            reason,
        )
        return ConsolidatedSemanticMemoryControllerResult(
            state=committed_state,
            retrieval=exposed_retrieval,
            write=exposed_write,
            diagnostics=ConsolidatedSemanticMemoryControllerDiagnostics(
                state_valid=state_valid,
                checksum_valid=checksum_valid,
                transaction_allowed=transaction_allowed,
                procedural_feedback_pending=state.pending,
                memory_available_before=memory_available_before,
                operation_capacity_available=operation_capacity_available,
                transaction_attempted=transaction_attempted,
                query_before_write_verified=(
                    step_applied & candidate_valid
                ),
                retrieval_accepted=(
                    exposed_retrieval.accepted & candidate_valid
                ),
                write_applied=exposed_write.wrote & candidate_valid,
                terminal_capacity_reached=terminal_capacity_reached,
                memory_became_unavailable=memory_became_unavailable,
                operation_count_before=state.memory.operation_count,
                operation_count_after=committed_state.memory.operation_count,
                reason=reason,
            ),
        )

    def decide(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        decision_id: Array,
        base_categorical_score_mass: Array,
        base_action: int | Array,
        base_action_available: bool | Array,
        hard_safety_mask: Array,
        request: ProceduralMemoryRequest,
    ) -> ConsolidatedProceduralMemoryDecisionResult:
        """Query memory first, then return a proposal or the exact base action."""

        self._validate_state_static_contract(state)
        decision_id = _require_array(
            decision_id,
            name="decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        base_categorical_score_mass = _require_array(
            base_categorical_score_mass,
            name="base_categorical_score_mass",
            shape=(self._config.policy.n_actions,),
            dtype=jnp.float32,
        )
        action = _int32_scalar(base_action, name="base_action")
        available = _bool_scalar(
            base_action_available, name="base_action_available"
        )
        hard_safety_mask = _require_array(
            hard_safety_mask,
            name="hard_safety_mask",
            shape=(self._config.policy.n_actions,),
            dtype=jnp.bool_,
        )
        self._validate_request_static(request)
        return cast(
            ConsolidatedProceduralMemoryDecisionResult,
            self._decide_jit(
                state,
                decision_id,
                base_categorical_score_mass,
                action,
                available,
                hard_safety_mask,
                request,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decide_jit(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        decision_id: Array,
        base_categorical_score_mass: Array,
        base_action: Array,
        base_action_available: Array,
        hard_safety_mask: Array,
        request: ProceduralMemoryRequest,
    ) -> ConsolidatedProceduralMemoryDecisionResult:
        checksum_valid = jnp.array_equal(state.checksum, self._checksum(state))
        state_valid = self._state_is_valid(
            state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
            checksum_valid=checksum_valid,
        )
        n_actions = self._config.policy.n_actions
        base_index_valid = (base_action >= 0) & (base_action < n_actions)
        base_index = jnp.clip(base_action, 0, n_actions - 1)
        score_total = jnp.sum(base_categorical_score_mass)
        base_scores_finite = jnp.all(
            jnp.isfinite(base_categorical_score_mass)
        ) & jnp.isfinite(score_total)
        base_scores_nonnegative = base_scores_finite & jnp.all(
            base_categorical_score_mass >= 0.0
        )
        base_positive_mass = base_scores_nonnegative & (score_total > 0.0)
        base_action_positive_mass = base_index_valid & (
            base_categorical_score_mass[base_index] > 0.0
        )
        base_action_hard_safe = base_index_valid & hard_safety_mask[base_index]
        base_valid = (
            base_action_available
            & base_positive_mass
            & base_index_valid
            & base_action_positive_mass
            & base_action_hard_safe
        )
        request_valid = self._request_is_valid(state, request)
        duplicate_decision = state.has_last_decision & jnp.array_equal(
            decision_id, state.last_decision_id
        )
        decision_id_strictly_advancing = _identity_nonzero(decision_id) & (
            (~state.has_last_decision)
            | _identity_strictly_advances(decision_id, state.last_decision_id)
        )
        stale_decision = (
            state.has_last_decision
            & (~duplicate_decision)
            & (~decision_id_strictly_advancing)
        )
        replayed_decision = duplicate_decision | stale_decision
        pending_conflict = state.pending
        request_is_non_load_bearing = (
            state.memory_unavailable | pending_conflict | replayed_decision
        )
        decision_identity_valid = _identity_nonzero(decision_id) & (
            request_valid | request_is_non_load_bearing
        )
        memory_available_before = state_valid & (~state.memory_unavailable)
        query_attempted = (
            state_valid
            & base_valid
            & decision_identity_valid
            & decision_id_strictly_advancing
            & (~pending_conflict)
            & (~state.memory_unavailable)
        )
        blank_retrieval = self._blank_retrieval(
            state_valid=state_valid, request_valid=request_valid
        )
        complete_lifecycle_capacity_available = (
            state.memory.operation_count
            <= self._config.memory.max_operations - 2
        )
        queried_memory, queried_retrieval = jax.lax.cond(
            query_attempted & complete_lifecycle_capacity_available,
            lambda: self._memory.query_procedural(state.memory, request),
            lambda: (state.memory, blank_retrieval),
        )
        retrieval = jax.lax.cond(
            query_attempted,
            lambda: queried_retrieval,
            lambda: blank_retrieval,
        )
        queried_memory_valid = self._memory.validate_state(
            queried_memory,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        query_pre_write_verified = (
            retrieval.transaction_applied
            & queried_memory_valid
            & (
                queried_memory.procedural_write_count
                == state.memory.procedural_write_count
            )
            & (queried_memory.operation_count == state.memory.operation_count + 1)
        )
        query_succeeded = (
            query_attempted
            & retrieval.transaction_applied
            & query_pre_write_verified
        )
        proposal = self._policy.propose(
            retrieval,
            hard_safety_mask=hard_safety_mask,
            expected_lifecycle_digest=request.lifecycle_digest,
            expected_lifecycle_generation=request.lifecycle_generation,
            expected_lifecycle_revision=request.lifecycle_revision,
        )
        policy_contract_valid = self._proposal_contract_valid(
            proposal,
            retrieval=retrieval,
            hard_safety_mask=hard_safety_mask,
        ) & proposal.diagnostics.retrieval_contract_valid
        policy_contract_failure = query_succeeded & (~policy_contract_valid)
        query_failure = query_attempted & (~query_succeeded)
        memory_became_unavailable = query_failure | policy_contract_failure
        operation_exhausted = ~complete_lifecycle_capacity_available
        new_memory_error = jnp.where(
            policy_contract_failure,
            jnp.asarray(MEMORY_ERROR_POLICY_CONTRACT, dtype=jnp.int32),
            jnp.where(
                operation_exhausted,
                jnp.asarray(MEMORY_ERROR_CAP_EXHAUSTED, dtype=jnp.int32),
                jnp.asarray(MEMORY_ERROR_QUERY_REJECTED, dtype=jnp.int32),
            ),
        )
        trackable = query_succeeded & policy_contract_valid
        proposed_index = jnp.clip(proposal.action, 0, n_actions - 1)
        memory_selected = (
            trackable
            & proposal.available
            & (proposal.action >= 0)
            & (proposal.action < n_actions)
            & hard_safety_mask[proposed_index]
        )
        effective_action = jnp.where(
            memory_selected, proposal.action, base_action
        ).astype(jnp.int32)
        process_decision = (
            state_valid
            & base_valid
            & decision_identity_valid
            & decision_id_strictly_advancing
            & (~pending_conflict)
        )
        memory_after_query = jax.lax.cond(
            trackable, lambda: queried_memory, lambda: state.memory
        )
        updated = dataclasses.replace(
            state,
            memory=memory_after_query,
            pending=trackable,
            pending_decision_id=jnp.where(
                trackable,
                decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ),
            pending_base_action=jnp.where(trackable, base_action, -1).astype(
                jnp.int32
            ),
            pending_effective_action=jnp.where(
                trackable, effective_action, -1
            ).astype(jnp.int32),
            pending_memory_selected=trackable & memory_selected,
            pending_hard_safety_mask=jnp.where(
                trackable,
                hard_safety_mask,
                jnp.zeros_like(hard_safety_mask),
            ),
            pending_semantic_digest=jnp.where(
                trackable,
                request.semantic_digest,
                jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            ),
            pending_generation=jnp.where(trackable, request.generation, -1).astype(
                jnp.int32
            ),
            pending_provenance_digest=jnp.where(
                trackable,
                request.provenance_digest,
                jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            ),
            pending_representation_revision=jnp.where(
                trackable, request.representation_revision, -1
            ).astype(jnp.int32),
            pending_source_revision=jnp.where(
                trackable, request.source_revision, -1
            ).astype(jnp.int32),
            pending_lifecycle_link_available=trackable
            & request.lifecycle_link_available,
            pending_lifecycle_digest=jnp.where(
                trackable,
                request.lifecycle_digest,
                jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            ),
            pending_lifecycle_generation=jnp.where(
                trackable, request.lifecycle_generation, -1
            ).astype(jnp.int32),
            pending_lifecycle_revision=jnp.where(
                trackable, request.lifecycle_revision, -1
            ).astype(jnp.int32),
            pending_query_accepted=trackable & retrieval.accepted,
            pending_query_slot=jnp.where(trackable, retrieval.slot, -1).astype(
                jnp.int32
            ),
            pending_query_operation_before=jnp.where(
                trackable, state.memory.operation_count, -1
            ).astype(jnp.int32),
            pending_query_operation_after=jnp.where(
                trackable, queried_memory.operation_count, -1
            ).astype(jnp.int32),
            pending_procedural_write_count_before=jnp.where(
                trackable, state.memory.procedural_write_count, -1
            ).astype(jnp.int32),
            has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
            last_decision_id=decision_id,
            memory_unavailable=state.memory_unavailable
            | memory_became_unavailable,
            memory_error=jnp.where(
                memory_became_unavailable, new_memory_error, state.memory_error
            ).astype(jnp.int32),
            decision_count=_saturating_increment(state.decision_count),
            tracked_decision_count=_saturating_increment(
                state.tracked_decision_count, trackable.astype(jnp.int32)
            ),
            memory_proposal_count=_saturating_increment(
                state.memory_proposal_count, memory_selected.astype(jnp.int32)
            ),
            base_fallback_count=_saturating_increment(
                state.base_fallback_count, (~memory_selected).astype(jnp.int32)
            ),
            memory_unavailable_noop_count=_saturating_increment(
                state.memory_unavailable_noop_count,
                state.memory_unavailable.astype(jnp.int32),
            ),
            memory_error_count=_saturating_increment(
                state.memory_error_count,
                memory_became_unavailable.astype(jnp.int32),
            ),
        )
        updated = self._with_checksum(updated)
        proposed_state = jax.lax.cond(
            process_decision, lambda: updated, lambda: state
        )
        proposed_valid = state_valid & jnp.where(
            trackable, queried_memory_valid, jnp.asarray(True, dtype=jnp.bool_)
        )
        action_available = (
            state_valid
            & base_valid
            & decision_identity_valid
            & proposed_valid
        )
        used_base_fallback = action_available & (~memory_selected)
        returned_action = jnp.where(action_available, effective_action, -1).astype(
            jnp.int32
        )
        reason = jnp.asarray(DECISION_REASON_AVAILABLE, dtype=jnp.int32)
        reason = jnp.where(
            memory_selected,
            jnp.asarray(DECISION_REASON_MEMORY_PROPOSAL, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            trackable & (~memory_selected),
            jnp.asarray(DECISION_REASON_POLICY_FALLBACK, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            state.memory_unavailable | memory_became_unavailable,
            jnp.asarray(
                DECISION_REASON_MEMORY_UNAVAILABLE_FALLBACK, dtype=jnp.int32
            ),
            reason,
        )
        reason = jnp.where(
            pending_conflict,
            jnp.asarray(DECISION_REASON_PENDING_FALLBACK, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            stale_decision,
            jnp.asarray(DECISION_REASON_STALE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            duplicate_decision,
            jnp.asarray(DECISION_REASON_DUPLICATE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~decision_identity_valid,
            jnp.asarray(DECISION_REASON_IDENTITY, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            base_index_valid & (~base_action_hard_safe),
            jnp.asarray(DECISION_REASON_BASE_UNSAFE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~base_index_valid,
            jnp.asarray(DECISION_REASON_BASE_ACTION, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            base_action_available
            & ((~base_scores_finite) | (~base_scores_nonnegative) | (~base_positive_mass)),
            jnp.asarray(DECISION_REASON_BASE_SCORES, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~base_action_available,
            jnp.asarray(DECISION_REASON_BASE_UNAVAILABLE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            (~state_valid) | (~proposed_valid),
            jnp.asarray(DECISION_REASON_STATE_INVALID, dtype=jnp.int32),
            reason,
        )
        diagnostics = ConsolidatedProceduralMemoryDecisionDiagnostics(
            state_valid=state_valid,
            checksum_valid=checksum_valid,
            base_action_available=base_action_available,
            base_scores_finite=base_scores_finite,
            base_scores_nonnegative=base_scores_nonnegative,
            base_positive_mass=base_positive_mass,
            base_action_index_valid=base_index_valid,
            base_action_positive_mass=base_action_positive_mass,
            base_action_hard_safe=base_action_hard_safe,
            base_valid=base_valid,
            decision_identity_valid=decision_identity_valid,
            decision_id_strictly_advancing=decision_id_strictly_advancing,
            duplicate_decision=duplicate_decision,
            stale_decision=stale_decision,
            pending_conflict=pending_conflict,
            memory_available_before=memory_available_before,
            query_attempted=query_attempted,
            query_transaction_applied=query_attempted
            & retrieval.transaction_applied,
            query_accepted=query_attempted & retrieval.accepted,
            query_pre_write_verified=query_attempted
            & query_pre_write_verified,
            policy_contract_valid=policy_contract_valid,
            policy_proposal_available=trackable & proposal.available,
            memory_selected=action_available & memory_selected,
            used_base_fallback=used_base_fallback,
            feedback_trackable=process_decision & trackable & proposed_valid,
            action_available=action_available,
            memory_became_unavailable=(~state_valid)
            | memory_became_unavailable,
            memory_error=jnp.where(
                ~state_valid,
                jnp.asarray(
                    MEMORY_ERROR_COMPOSED_STATE_INVALID, dtype=jnp.int32
                ),
                jnp.where(
                    memory_became_unavailable,
                    new_memory_error,
                    state.memory_error,
                ),
            ).astype(jnp.int32),
            operation_count_before=state.memory.operation_count,
            operation_count_after=proposed_state.memory.operation_count,
            reason=reason,
        )
        return ConsolidatedProceduralMemoryDecisionResult(
            state=proposed_state,
            action_available=action_available,
            action=returned_action,
            counterfactual_base_action=jnp.where(
                action_available, base_action, -1
            ).astype(jnp.int32),
            memory_proposed_action=jnp.where(
                trackable & proposal.available, proposal.action, -1
            ).astype(jnp.int32),
            retrieval=retrieval,
            proposal=proposal,
            diagnostics=diagnostics,
        )

    def feedback(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        decision_id: Array,
        feedback_event_id: Array,
        base_action: int | Array,
        effective_action: int | Array,
        request: ProceduralMemoryRequest,
        succeeded: bool | Array,
        outcome: Array,
        confidence: float | Array,
        evidence: float | Array,
    ) -> ConsolidatedProceduralMemoryFeedbackResult:
        """Write evidence only for the exact pending decision and observed result."""

        self._validate_state_static_contract(state)
        decision_id = _require_array(
            decision_id,
            name="decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        feedback_event_id = _require_array(
            feedback_event_id,
            name="feedback_event_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        base = _int32_scalar(base_action, name="base_action")
        effective = _int32_scalar(effective_action, name="effective_action")
        self._validate_request_static(request)
        succeeded_array = _bool_scalar(succeeded, name="succeeded")
        outcome = _require_array(
            outcome,
            name="outcome",
            shape=(self._config.policy.outcome_dim,),
            dtype=jnp.float32,
        )
        confidence_array = _float32_scalar(confidence, name="confidence")
        evidence_array = _float32_scalar(evidence, name="evidence")
        return cast(
            ConsolidatedProceduralMemoryFeedbackResult,
            self._feedback_jit(
                state,
                decision_id,
                feedback_event_id,
                base,
                effective,
                request,
                succeeded_array,
                outcome,
                confidence_array,
                evidence_array,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _feedback_jit(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        decision_id: Array,
        feedback_event_id: Array,
        base_action: Array,
        effective_action: Array,
        request: ProceduralMemoryRequest,
        succeeded: Array,
        outcome: Array,
        confidence: Array,
        evidence: Array,
    ) -> ConsolidatedProceduralMemoryFeedbackResult:
        checksum_valid = jnp.array_equal(state.checksum, self._checksum(state))
        state_valid = self._state_is_valid(
            state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
            checksum_valid=checksum_valid,
        )
        decision_identity_matches = state.pending & jnp.array_equal(
            decision_id, state.pending_decision_id
        )
        event_identity_valid = _identity_nonzero(feedback_event_id)
        duplicate_event = state.has_last_feedback_event & jnp.array_equal(
            feedback_event_id, state.last_feedback_event_id
        )
        event_strictly_advancing = _identity_nonzero(feedback_event_id) & (
            (~state.has_last_feedback_event)
            | _identity_strictly_advances(
                feedback_event_id, state.last_feedback_event_id
            )
        )
        stale_event = (
            state.has_last_feedback_event
            & (~duplicate_event)
            & (~event_strictly_advancing)
        )
        base_matches = state.pending & (base_action == state.pending_base_action)
        effective_matches = state.pending & (
            effective_action == state.pending_effective_action
        )
        (
            semantic_matches,
            provenance_matches,
            representation_source_matches,
            lifecycle_matches,
        ) = self._request_matches_pending(state, request)
        memory_unchanged = (
            state.pending
            & (
                state.memory.operation_count
                == state.pending_query_operation_after
            )
            & (
                state.memory.procedural_write_count
                == state.pending_procedural_write_count_before
            )
        )
        numeric_valid = (
            jnp.all(jnp.isfinite(outcome))
            & jnp.isfinite(confidence)
            & (confidence >= 0.0)
            & (confidence <= 1.0)
            & jnp.isfinite(evidence)
        )
        feedback_valid = (
            state_valid
            & state.pending
            & decision_identity_matches
            & event_identity_valid
            & event_strictly_advancing
            & base_matches
            & effective_matches
            & semantic_matches
            & provenance_matches
            & representation_source_matches
            & lifecycle_matches
            & memory_unchanged
            & numeric_valid
            & (~state.memory_unavailable)
        )
        safe_effective_action = jnp.clip(
            state.pending_effective_action, 0, self._config.policy.n_actions - 1
        )
        record = ProceduralMemoryRecord(
            semantic_digest=state.pending_semantic_digest,
            generation=state.pending_generation,
            payload=jax.nn.one_hot(
                safe_effective_action,
                self._config.policy.n_actions,
                dtype=jnp.float32,
            ),
            confidence=confidence,
            provenance_digest=state.pending_provenance_digest,
            representation_revision=state.pending_representation_revision,
            source_revision=state.pending_source_revision,
            evidence=evidence,
            succeeded=succeeded,
            outcome=outcome,
            lifecycle_link_available=state.pending_lifecycle_link_available,
            lifecycle_digest=state.pending_lifecycle_digest,
            lifecycle_generation=state.pending_lifecycle_generation,
            lifecycle_revision=state.pending_lifecycle_revision,
        )
        operation_available = (
            state.memory.operation_count < self._config.memory.max_operations
        )
        blank_write = self._blank_write(
            state_valid=state_valid,
            record_valid=numeric_valid & state.pending,
        )
        written_memory, raw_write = jax.lax.cond(
            feedback_valid & operation_available,
            lambda: self._memory.write_procedural(state.memory, record),
            lambda: (state.memory, blank_write),
        )
        written_memory_valid = self._memory.validate_state(
            written_memory,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        write_applied = (
            feedback_valid
            & raw_write.transaction_applied
            & raw_write.wrote
            & written_memory_valid
            & (written_memory.operation_count == state.memory.operation_count + 1)
            & (
                written_memory.procedural_write_count
                == state.memory.procedural_write_count + 1
            )
        )
        write_rejected = feedback_valid & (~write_applied)
        exhausted_after_write = write_applied & (
            written_memory.operation_count >= self._config.memory.max_operations
        )
        memory_became_unavailable = write_rejected | exhausted_after_write
        memory_error = jnp.where(
            (~operation_available) | exhausted_after_write,
            jnp.asarray(MEMORY_ERROR_CAP_EXHAUSTED, dtype=jnp.int32),
            jnp.asarray(MEMORY_ERROR_WRITE_REJECTED, dtype=jnp.int32),
        )
        accepted_feedback = write_applied | write_rejected
        next_memory = jax.lax.cond(
            write_applied, lambda: written_memory, lambda: state.memory
        )
        cleared = self._clear_pending(dataclasses.replace(state, memory=next_memory))
        updated = dataclasses.replace(
            cleared,
            has_last_feedback_event=jnp.asarray(True, dtype=jnp.bool_),
            last_feedback_event_id=feedback_event_id,
            memory_unavailable=state.memory_unavailable
            | memory_became_unavailable,
            memory_error=jnp.where(
                memory_became_unavailable, memory_error, state.memory_error
            ).astype(jnp.int32),
            feedback_count=_saturating_increment(state.feedback_count),
            successful_memory_write_count=_saturating_increment(
                state.successful_memory_write_count,
                write_applied.astype(jnp.int32),
            ),
            failed_memory_write_count=_saturating_increment(
                state.failed_memory_write_count,
                write_rejected.astype(jnp.int32),
            ),
            recorded_success_count=_saturating_increment(
                state.recorded_success_count,
                (write_applied & succeeded).astype(jnp.int32),
            ),
            recorded_failure_count=_saturating_increment(
                state.recorded_failure_count,
                (write_applied & (~succeeded)).astype(jnp.int32),
            ),
            memory_error_count=_saturating_increment(
                state.memory_error_count,
                memory_became_unavailable.astype(jnp.int32),
            ),
        )
        updated = self._with_checksum(updated)
        proposed_state = jax.lax.cond(
            accepted_feedback, lambda: updated, lambda: state
        )
        proposed_valid = state_valid & jnp.where(
            write_applied,
            written_memory_valid,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        committed = accepted_feedback & proposed_valid
        final_state = jax.lax.cond(committed, lambda: proposed_state, lambda: state)
        exposed_write = jax.lax.cond(
            feedback_valid
            & ((~raw_write.wrote) | (write_applied & proposed_valid)),
            lambda: raw_write,
            lambda: blank_write,
        )
        reason = jnp.asarray(FEEDBACK_REASON_APPLIED, dtype=jnp.int32)
        reason = jnp.where(
            write_rejected,
            jnp.asarray(FEEDBACK_REASON_WRITE_REJECTED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~numeric_valid,
            jnp.asarray(FEEDBACK_REASON_NONFINITE, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~memory_unchanged,
            jnp.asarray(FEEDBACK_REASON_MEMORY_CHANGED, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~(
                semantic_matches
                & provenance_matches
                & representation_source_matches
                & lifecycle_matches
            ),
            jnp.asarray(FEEDBACK_REASON_BINDING, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~(base_matches & effective_matches),
            jnp.asarray(FEEDBACK_REASON_ACTION, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            (~decision_identity_matches) | (~event_identity_valid),
            jnp.asarray(FEEDBACK_REASON_IDENTITY, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            ~state.pending,
            jnp.asarray(FEEDBACK_REASON_NO_PENDING, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            stale_event,
            jnp.asarray(FEEDBACK_REASON_STALE_EVENT, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            duplicate_event,
            jnp.asarray(FEEDBACK_REASON_DUPLICATE_EVENT, dtype=jnp.int32),
            reason,
        )
        reason = jnp.where(
            (~state_valid) | (accepted_feedback & (~proposed_valid)),
            jnp.asarray(FEEDBACK_REASON_STATE_INVALID, dtype=jnp.int32),
            reason,
        )
        diagnostics = ConsolidatedProceduralMemoryFeedbackDiagnostics(
            state_valid=state_valid,
            checksum_valid=checksum_valid,
            pending_available=state.pending,
            decision_identity_matches=decision_identity_matches,
            feedback_event_identity_valid=event_identity_valid,
            feedback_event_strictly_advancing=event_strictly_advancing,
            duplicate_feedback_event=duplicate_event,
            stale_feedback_event=stale_event,
            base_action_matches=base_matches,
            effective_action_matches=effective_matches,
            semantic_binding_matches=semantic_matches,
            provenance_binding_matches=provenance_matches,
            representation_source_matches=representation_source_matches,
            lifecycle_binding_matches=lifecycle_matches,
            memory_unchanged_since_query=memory_unchanged,
            feedback_inputs_finite=numeric_valid,
            feedback_valid=feedback_valid,
            write_attempted=feedback_valid,
            write_applied=write_applied & proposed_valid,
            success_recorded=write_applied & proposed_valid & succeeded,
            failure_recorded=write_applied & proposed_valid & (~succeeded),
            pending_cleared=committed,
            memory_became_unavailable=(~state_valid)
            | (committed & memory_became_unavailable),
            memory_error=jnp.where(
                ~state_valid,
                jnp.asarray(
                    MEMORY_ERROR_COMPOSED_STATE_INVALID, dtype=jnp.int32
                ),
                jnp.where(
                    committed & memory_became_unavailable,
                    memory_error,
                    state.memory_error,
                ),
            ).astype(jnp.int32),
            operation_count_before=state.memory.operation_count,
            operation_count_after=final_state.memory.operation_count,
            reason=reason,
        )
        return ConsolidatedProceduralMemoryFeedbackResult(
            state=final_state,
            write=exposed_write,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _state_sha256(state: ConsolidatedProceduralMemoryControllerState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            host = np.asarray(jax.device_get(leaf))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def _binding_sha256(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> Array:
        digest = hashlib.sha256()
        digest.update(np.asarray(jax.device_get(self._config_digest)).tobytes())
        digest.update(
            np.asarray(jax.device_get(self._policy_identity_digest)).tobytes()
        )
        digest.update(np.asarray(jax.device_get(source_digest)).tobytes())
        digest.update(
            np.asarray(jax.device_get(semantic_namespace_digest)).tobytes()
        )
        revisions = np.asarray(
            (
                int(jax.device_get(representation_revision)),
                int(jax.device_get(source_revision)),
            ),
            dtype=np.int32,
        )
        digest.update(revisions.tobytes())
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def checkpoint_payload(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> dict[str, object]:
        """Return a strict checkpoint with an unkeyed SHA-256 corruption digest.

        The digest detects accidental or unsophisticated tampering; it is not a
        keyed MAC, a signature, or an authenticity boundary.
        """

        self._validate_state_static_contract(state)
        source_digest = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        semantic_namespace_digest = _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(
            representation_revision, name="representation_revision"
        )
        source = _int32_scalar(source_revision, name="source_revision")
        valid = self._state_is_valid(
            state,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("cannot checkpoint an invalid or stale controller state")
        return {
            "schema_version": CONSOLIDATED_MEMORY_CONTROLLER_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_sha256": self._state_sha256(state),
            "binding_sha256": self._binding_sha256(
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation,
                source_revision=source,
            ),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> ConsolidatedProceduralMemoryControllerState:
        """Restore exact schema, unkeyed SHA, checksum, and live bindings."""

        if type(payload) is not dict:
            raise ValueError("controller checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {
            "schema_version",
            "config",
            "state",
            "state_sha256",
            "binding_sha256",
        }:
            raise ValueError("controller checkpoint keys differ from schema v1")
        if raw["schema_version"] != CONSOLIDATED_MEMORY_CONTROLLER_CHECKPOINT_SCHEMA:
            raise ValueError("controller checkpoint schema differs")
        if ConsolidatedProceduralMemoryControllerConfig.from_config(
            raw["config"]
        ) != self._config:
            raise ValueError("controller checkpoint config differs")
        state = raw["state"]
        if type(state) is not ConsolidatedProceduralMemoryControllerState:
            raise ValueError("controller checkpoint state type differs")
        restored = state
        self._validate_state_static_contract(restored)
        state_sha = _require_array(
            cast(Array, raw["state_sha256"]),
            name="checkpoint.state_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        binding_sha = _require_array(
            cast(Array, raw["binding_sha256"]),
            name="checkpoint.binding_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        if not bool(
            jax.device_get(jnp.array_equal(state_sha, self._state_sha256(restored)))
        ):
            raise ValueError("controller checkpoint state SHA differs")
        source_digest = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        semantic_namespace_digest = _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(
            representation_revision, name="representation_revision"
        )
        source = _int32_scalar(source_revision, name="source_revision")
        expected_binding = self._binding_sha256(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(jax.device_get(jnp.array_equal(binding_sha, expected_binding))):
            raise ValueError("controller checkpoint source or relabel binding differs")
        valid = self._state_is_valid(
            restored,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("controller checkpoint state is invalid or stale")
        return restored

    def rebind_reset(
        self,
        state: ConsolidatedProceduralMemoryControllerState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
        discard_pending: bool = False,
    ) -> ConsolidatedProceduralMemoryControllerState:
        """Explicitly discard compatible state and initialize a new binding."""

        if type(discard_pending) is not bool:
            raise TypeError("discard_pending must be an exact bool")
        self._validate_state_static_contract(state)
        old_valid = self._state_is_valid(
            state,
            source_digest=state.memory.source_digest,
            semantic_namespace_digest=state.memory.semantic_namespace_digest,
            representation_revision=state.memory.representation_revision,
            source_revision=state.memory.source_revision,
        )
        if not bool(jax.device_get(old_valid)):
            raise ValueError("cannot rebind or reset a corrupted controller state")
        if bool(jax.device_get(state.pending)) and not discard_pending:
            raise ValueError("discard_pending=True is required for a pending decision")
        return self.init(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
        )
