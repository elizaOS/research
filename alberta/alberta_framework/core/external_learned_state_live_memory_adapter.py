# mypy: disable-error-code="attr-defined,call-arg,no-any-return,union-attr"
"""Atomic live learned-memory rung over the external-state coordinator.

This v1 adapter owns exactly one
:class:`~alberta_framework.core.learned_experiential_memory_controller.LearnedExperientialMemoryController`
beside one :class:`ExternalLearnedStateRouterAuditCoordinator`.  The inner
Prototype's historical experiential-memory lane must be disabled.  The
learned controller is therefore the sole memory and retention owner; the
coordinator remains the sole learned-state, learning-value-router,
feature-lifecycle/router, Horde, Prototype, and routed-model owner.

Ordering is deliberately causal and host-orchestrated.  An exact prior
learned-memory feedback receipt, when required, settles first.  The coordinator
then evaluates one continuing real transition.  Only its post-transition raw
next-decision observation is queried against the pre-write learned store.  The
completed current exemplar is written afterward with the actually executed
prior primitive, grounded reward, and grounded raw next observation.  Version
1 uses raw observations as fixed keys at representation version zero; it has no
learned embedding or re-encoding path.

Only an admitted exact one-hot retrieval may propose a next primitive, and it
does so exclusively through Prototype's public cached-action replacement under
the caller's hard mask.  A soft retrieval retains one pending receipt but has
no action authority and must later settle as unused.  The adapter copies the
resulting Prototype action into the coordinator's exact cache; it creates no
second action owner.  Every admitted retrieval binds one memory transaction,
Prototype decision, and effective action for the next causal feedback.

The outer receipt is exact-content, source-bound, and integrity-bound but not
authenticated.  A missing/stale feedback, coordinator failure, memory failure,
replacement failure, stale/tampered receipt, or corrupt candidate returns the
complete pre-event source so the exact feedback and transition can be retried.
The composite is explicitly host-only: donor calls retain their own compiled
contracts, while monolithic JIT/scan is rejected before donor work.  This L0
mechanism grants no dispatch, safety, authentication, evidence, or promotion
authority and establishes no memory or control benefit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.experiential_memory import ExperientialMemoryEntry
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorResult,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
    measure_external_learned_state_router_audit_coordinator_state_nbytes,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryControllerState,
    LearnedExperientialMemoryFeedback,
    LearnedExperientialMemoryFeedbackResult,
    LearnedExperientialMemoryStepResult,
)
from alberta_framework.core.prototype_agent import (
    PrototypeCachedPrimitiveActionReplacement,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
)

EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA = (
    "alberta.external-learned-state-live-memory-adapter.config.v1"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_STATE_SCHEMA = (
    "alberta.external-learned-state-live-memory-adapter.state.v1"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_RECEIPT_SCHEMA = (
    "alberta.external-learned-state-live-memory-adapter.receipt.v1"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA = (
    "alberta.external-learned-state-live-memory-adapter.checkpoint.v1"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL = "L0"
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS = "not_assessed"
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED = False
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION = 0

_SCHEMA_DIGEST_NBYTES = 32
_KEY_SEMANTICS = (
    "raw-observation-key-v1;representation-version=0;"
    "learned-embedding=false;reencoding=false"
)
_EVENT_SEMANTICS = (
    "settle-exact-prior-feedback;one-coordinator-event;"
    "query-next-raw-before-write-current-exemplar;"
    "categorical-retrieval-proposal-via-public-prototype-replacement;"
    "bind-base-effective-action-and-hard-mask;atomic-outer-adoption"
)
_EXECUTION_MODE = "host-orchestrated-donor-calls"


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            equal = equal & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.dtype(jnp.float32):
            equal = equal & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        total += int(value.nbytes)
    return total


def _contains_tracer(tree: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalLearnedStateLiveMemoryAdapterConfig:
    """Exact one-coordinator/one-learned-memory static composition."""

    coordinator: ExternalLearnedStateRouterAuditCoordinatorConfig
    learned_memory: LearnedExperientialMemoryControllerConfig

    def __post_init__(self) -> None:
        if type(self.coordinator) is not ExternalLearnedStateRouterAuditCoordinatorConfig:
            raise TypeError("coordinator must be an exact coordinator config")
        if type(self.learned_memory) is not LearnedExperientialMemoryControllerConfig:
            raise TypeError("learned_memory must be an exact learned-memory config")
        prototype = self.coordinator.inner.prototype
        if prototype.experiential_memory is not None:
            raise ValueError(
                "inner Prototype experiential memory must be disabled; "
                "the learned controller is the sole memory owner"
            )
        memory = self.learned_memory.memory
        raw_dim = self.coordinator.builder.observation_dim
        n_actions = self.coordinator.builder.n_actions
        if memory.observation_dim != raw_dim:
            raise ValueError("learned memory observation_dim must equal raw observation width")
        if memory.key_dim != raw_dim:
            raise ValueError("v1 raw learned-memory key_dim must equal raw observation width")
        if memory.action_dim != n_actions:
            raise ValueError("learned memory action_dim must equal primitive-action count")
        if memory.outcome_dim != raw_dim:
            raise ValueError("v1 learned-memory outcome_dim must equal raw observation width")

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA,
            "state_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_STATE_SCHEMA,
            "receipt_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_RECEIPT_SCHEMA,
            "evidence_level": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL,
            "outcome_status": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "coordinator": self.coordinator.to_config(),
            "learned_memory": self.learned_memory.to_config(),
            "key_semantics": _KEY_SEMANTICS,
            "event_semantics": _EVENT_SEMANTICS,
            "raw_schema_version": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION,
            "prototype_historical_memory_enabled": False,
            "learned_memory_owner_count": 1,
            "pending_binding_base_action_recorded": True,
            "pending_binding_hard_action_mask_recorded": True,
            "learned_embedding_enabled": False,
            "reencoding_enabled": False,
            "monolithic_jit_supported": False,
            "scan_supported": False,
            "execution_mode": _EXECUTION_MODE,
            "feedback_authenticated": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ExternalLearnedStateLiveMemoryAdapterConfig:
        expected = {
            "type",
            "schema",
            "state_schema",
            "receipt_schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "coordinator",
            "learned_memory",
            "key_semantics",
            "event_semantics",
            "raw_schema_version",
            "prototype_historical_memory_enabled",
            "learned_memory_owner_count",
            "pending_binding_base_action_recorded",
            "pending_binding_hard_action_mask_recorded",
            "learned_embedding_enabled",
            "reencoding_enabled",
            "monolithic_jit_supported",
            "scan_supported",
            "execution_mode",
            "feedback_authenticated",
            "dispatch_authority",
            "safety_authority",
            "evidence_authority",
            "promotion_authority",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("live-memory adapter config fields are not exact")
        fixed = {
            "type": cls.__name__,
            "schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA,
            "state_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_STATE_SCHEMA,
            "receipt_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_RECEIPT_SCHEMA,
            "evidence_level": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL,
            "outcome_status": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "key_semantics": _KEY_SEMANTICS,
            "event_semantics": _EVENT_SEMANTICS,
            "raw_schema_version": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION,
            "prototype_historical_memory_enabled": False,
            "learned_memory_owner_count": 1,
            "pending_binding_base_action_recorded": True,
            "pending_binding_hard_action_mask_recorded": True,
            "learned_embedding_enabled": False,
            "reencoding_enabled": False,
            "monolithic_jit_supported": False,
            "scan_supported": False,
            "execution_mode": _EXECUTION_MODE,
            "feedback_authenticated": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }
        if any(
            type(config.get(name)) is not type(value) or config.get(name) != value
            for name, value in fixed.items()
        ):
            raise ValueError("live-memory adapter fixed semantics differ")
        if type(config["coordinator"]) is not dict:
            raise ValueError("coordinator config must be an exact dict")
        if type(config["learned_memory"]) is not dict:
            raise ValueError("learned_memory config must be an exact dict")
        restored = cls(
            coordinator=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(
                cast(dict[str, object], config["coordinator"])
            ),
            learned_memory=LearnedExperientialMemoryControllerConfig.from_config(
                config["learned_memory"]
            ),
        )
        if _config_digest(restored.to_config()) != _config_digest(dict(config)):
            raise ValueError("live-memory adapter config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryEventInput:
    """Bounded metadata for one raw-key query and grounded exemplar write."""

    query_uncertainty: Array
    query_uncertainty_available: Array
    entry_uncertainty: Array
    entry_uncertainty_available: Array
    entry_safety_cost: Array
    entry_safety_cost_available: Array
    entry_reliability: Array
    provenance_id: Array
    source_id: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryFeedback:
    """Exact prior decision/action binding plus caller counterfactual usefulness."""

    memory_transaction_words: Array
    prototype_decision_id: Array
    base_action_before_retrieval: Array
    effective_action: Array
    hard_action_mask: Array
    retrieval_used: Array
    counterfactual_available: Array
    counterfactual_delta: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryPendingBinding:
    """Non-authoritative exact binding for one admitted retrieval."""

    available: Array
    memory_transaction_words: Array
    prototype_decision_id: Array
    base_action_before_retrieval: Array
    effective_action: Array
    retrieval_action: Array
    hard_action_mask: Array
    categorical_retrieval: Array
    retrieval_used_expected: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryAdapterState:
    """One coordinator, one learned-memory owner, and one feedback binding."""

    coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState
    learned_memory_state: LearnedExperientialMemoryControllerState
    pending_binding: ExternalLearnedStateLiveMemoryPendingBinding
    schema_digest: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryPreparedTransition:
    """All pure donor results and the proposed complete outer destination."""

    source_state: ExternalLearnedStateLiveMemoryAdapterState
    transition: ExternalLearnedStateTransition
    event_input: ExternalLearnedStateLiveMemoryEventInput
    hard_action_mask: Array
    feedback: ExternalLearnedStateLiveMemoryFeedback
    feedback_supplied: Array
    candidate_evidence: Any
    partner_policy_fusion_input: Any
    partner_policy_fusion_feedback: Any
    extended_action_mask: Any
    feedback_identity_valid: Array
    preflight_valid: Array
    settlement_result: LearnedExperientialMemoryFeedbackResult | None
    coordinator_result: ExternalLearnedStateRouterAuditCoordinatorResult | None
    learned_memory_result: LearnedExperientialMemoryStepResult | None
    cached_action_replacement: PrototypeCachedPrimitiveActionReplacement | None
    query_key: Array
    completed_entry: ExperientialMemoryEntry
    categorical_retrieval: Array
    retrieval_action: Array
    replacement_required: Array
    candidate_state: ExternalLearnedStateLiveMemoryAdapterState
    preparation_valid: Array
    settlement_evaluations: Array
    coordinator_evaluations: Array
    learned_memory_query_evaluations: Array
    learned_memory_write_evaluations: Array
    cached_action_replacement_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryIntegrityReceipt:
    """Unkeyed exact-content binding; integrity-bound, not authenticated."""

    prepared: ExternalLearnedStateLiveMemoryPreparedTransition
    integrity_bound: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryDiagnostics:
    source_state_matches: Array
    receipt_matches_preparation: Array
    receipt_integrity_bound: Array
    source_state_valid: Array
    source_transition_matches: Array
    continuing_boundary_valid: Array
    prior_feedback_required: Array
    prior_feedback_supplied: Array
    prior_feedback_identity_valid: Array
    prior_feedback_settled: Array
    prior_feedback_learning_applied: Array
    coordinator_transaction_applied: Array
    learned_memory_transaction_applied: Array
    query_before_write: Array
    completed_entry_executed_action_exact: Array
    categorical_retrieval: Array
    soft_retrieval_denied_action_authority: Array
    cached_action_replacement_required: Array
    cached_action_replacement_committed: Array
    used_safe_current_action_fallback: Array
    next_action_changed: Array
    pending_feedback_created: Array
    pending_decision_action_bound: Array
    candidate_state_valid: Array
    transaction_applied: Array
    complete_source_returned: Array
    rejected: Array
    settlement_evaluations: Array
    coordinator_evaluations: Array
    learned_memory_query_evaluations: Array
    learned_memory_write_evaluations: Array
    cached_action_replacement_evaluations: Array
    learned_memory_owner_count: Array
    prototype_historical_memory_owner_count: Array
    learned_embedding_evaluations: Array
    reencoding_evaluations: Array
    feedback_authenticated: Array
    dispatch_authority: Array
    safety_authority: Array
    evidence_authority: Array
    promotion_authority: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryResult:
    state: ExternalLearnedStateLiveMemoryAdapterState
    prepared: ExternalLearnedStateLiveMemoryPreparedTransition
    receipt: ExternalLearnedStateLiveMemoryIntegrityReceipt
    diagnostics: ExternalLearnedStateLiveMemoryDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalLearnedStateLiveMemoryResourceBudget:
    persistent_state_bytes: int
    coordinator_state_bytes: int
    learned_memory_state_bytes: int
    pending_binding_and_schema_bytes: int
    persistent_capacity_growth: int
    learned_memory_owner_count: int
    prototype_historical_memory_owner_count: int
    coordinator_owner_count: int
    pending_binding_base_action_fields: int
    pending_binding_hard_action_mask_elements: int
    maximum_settlements_per_event: int
    coordinator_evaluations_per_event: int
    learned_memory_queries_per_event: int
    learned_memory_writes_per_event: int
    maximum_cached_action_replacements_per_event: int
    raw_key_materializations_per_event: int
    learned_embedding_evaluations_per_event: int
    reencoding_evaluations_per_event: int
    monolithic_jit_supported: bool
    scan_supported: bool
    feedback_authenticated: bool
    dispatch_authority: bool
    safety_authority: bool
    evidence_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class ExternalLearnedStateLiveMemoryAdapter:
    """Host coordinator for one external-state owner and one live memory."""

    def __init__(self, config: ExternalLearnedStateLiveMemoryAdapterConfig) -> None:
        if type(config) is not ExternalLearnedStateLiveMemoryAdapterConfig:
            raise TypeError("config must be an exact live-memory adapter config")
        self._config = config
        self._coordinator = ExternalLearnedStateRouterAuditCoordinator(
            config.coordinator
        )
        self._learned_memory = LearnedExperientialMemoryController(
            config.learned_memory
        )
        self._raw_dim = config.coordinator.builder.observation_dim
        self._n_actions = config.coordinator.builder.n_actions
        digest = hashlib.sha256(
            json.dumps(
                config.to_config(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        self._schema_digest = jnp.asarray(tuple(digest), dtype=jnp.uint8)

    @property
    def config(self) -> ExternalLearnedStateLiveMemoryAdapterConfig:
        return self._config

    @property
    def coordinator(self) -> ExternalLearnedStateRouterAuditCoordinator:
        return self._coordinator

    @property
    def learned_memory(self) -> LearnedExperientialMemoryController:
        return self._learned_memory

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ExternalLearnedStateLiveMemoryAdapter:
        return cls(ExternalLearnedStateLiveMemoryAdapterConfig.from_config(config))

    def _blank_pending(self) -> ExternalLearnedStateLiveMemoryPendingBinding:
        return ExternalLearnedStateLiveMemoryPendingBinding(
            available=jnp.asarray(False, dtype=jnp.bool_),
            memory_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            base_action_before_retrieval=jnp.asarray(-1, dtype=jnp.int32),
            effective_action=jnp.asarray(-1, dtype=jnp.int32),
            retrieval_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_action_mask=jnp.zeros((self._n_actions,), dtype=jnp.bool_),
            categorical_retrieval=jnp.asarray(False, dtype=jnp.bool_),
            retrieval_used_expected=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _blank_feedback(self) -> ExternalLearnedStateLiveMemoryFeedback:
        return ExternalLearnedStateLiveMemoryFeedback(
            memory_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            base_action_before_retrieval=jnp.asarray(-1, dtype=jnp.int32),
            effective_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_action_mask=jnp.zeros((self._n_actions,), dtype=jnp.bool_),
            retrieval_used=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_available=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_delta=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _blank_entry(self) -> ExperientialMemoryEntry:
        return ExperientialMemoryEntry(
            observation=jnp.zeros((self._raw_dim,), dtype=jnp.float32),
            key=jnp.zeros((self._raw_dim,), dtype=jnp.float32),
            action=jnp.zeros((self._n_actions,), dtype=jnp.float32),
            outcome=jnp.zeros((self._raw_dim,), dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            uncertainty_available=jnp.asarray(False, dtype=jnp.bool_),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(False, dtype=jnp.bool_),
            reliability=jnp.asarray(0.0, dtype=jnp.float32),
            utility=jnp.asarray(0.0, dtype=jnp.float32),
            utility_available=jnp.asarray(False, dtype=jnp.bool_),
            representation_version=jnp.asarray(
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION,
                dtype=jnp.int32,
            ),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=jnp.asarray(0, dtype=jnp.int32),
            source_id=jnp.asarray(0, dtype=jnp.int32),
        )

    def _validate_pending_static(
        self,
        pending: ExternalLearnedStateLiveMemoryPendingBinding,
    ) -> None:
        if type(pending) is not ExternalLearnedStateLiveMemoryPendingBinding:
            raise TypeError("pending_binding must be an exact pending binding")
        for value, name, shape, dtype in (
            (pending.available, "available", (), jnp.bool_),
            (
                pending.memory_transaction_words,
                "memory_transaction_words",
                (2,),
                jnp.uint32,
            ),
            (pending.prototype_decision_id, "prototype_decision_id", (4,), jnp.uint32),
            (
                pending.base_action_before_retrieval,
                "base_action_before_retrieval",
                (),
                jnp.int32,
            ),
            (pending.effective_action, "effective_action", (), jnp.int32),
            (pending.retrieval_action, "retrieval_action", (), jnp.int32),
            (
                pending.hard_action_mask,
                "hard_action_mask",
                (self._n_actions,),
                jnp.bool_,
            ),
            (
                pending.categorical_retrieval,
                "categorical_retrieval",
                (),
                jnp.bool_,
            ),
            (
                pending.retrieval_used_expected,
                "retrieval_used_expected",
                (),
                jnp.bool_,
            ),
        ):
            _require_array(
                value,
                name=f"pending_binding.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _pending_valid(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
    ) -> Array:
        pending = state.pending_binding
        memory_pending = state.learned_memory_state.pending
        inactive = (
            ~pending.available
            & jnp.all(pending.memory_transaction_words == 0)
            & jnp.all(pending.prototype_decision_id == 0)
            & (pending.base_action_before_retrieval == -1)
            & (pending.effective_action == -1)
            & (pending.retrieval_action == -1)
            & ~jnp.any(pending.hard_action_mask)
            & ~pending.categorical_retrieval
            & ~pending.retrieval_used_expected
            & ~memory_pending.available
        )
        retrieval_action_valid = (
            (pending.retrieval_action >= 0)
            & (pending.retrieval_action < self._n_actions)
        )
        active = (
            pending.available
            & memory_pending.available
            & jnp.array_equal(
                pending.memory_transaction_words,
                memory_pending.transaction_words,
            )
            & jnp.array_equal(
                pending.memory_transaction_words,
                state.learned_memory_state.transaction_words,
            )
            & jnp.array_equal(
                pending.prototype_decision_id,
                state.coordinator_state.current_decision_id,
            )
            & (pending.base_action_before_retrieval >= 0)
            & (pending.base_action_before_retrieval < self._n_actions)
            & (pending.effective_action == state.coordinator_state.current_action)
            & (pending.effective_action >= 0)
            & (pending.effective_action < self._n_actions)
            & (pending.categorical_retrieval == retrieval_action_valid)
            & jnp.where(
                pending.retrieval_used_expected,
                pending.categorical_retrieval
                & (pending.retrieval_action == pending.effective_action),
                pending.base_action_before_retrieval == pending.effective_action,
            )
        )
        return inactive | active

    def state_valid(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
    ) -> Array:
        if type(state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("state must be an exact live-memory adapter state")
        self._validate_pending_static(state.pending_binding)
        _require_array(
            state.schema_digest,
            name="state.schema_digest",
            shape=(_SCHEMA_DIGEST_NBYTES,),
            dtype=jnp.uint8,
        )
        return (
            jnp.array_equal(state.schema_digest, self._schema_digest)
            & self._coordinator.state_valid(state.coordinator_state)
            & self._learned_memory.state_valid(state.learned_memory_state)
            & jnp.array_equal(
                state.learned_memory_state.transaction_words,
                state.coordinator_state.event_words,
            )
            & self._pending_valid(state)
        )

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryAdapterState:
        state = ExternalLearnedStateLiveMemoryAdapterState(
            coordinator_state=self._coordinator.init(
                key,
                lifecycle_id=lifecycle_id,
            ),
            learned_memory_state=self._learned_memory.init(),
            pending_binding=self._blank_pending(),
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise RuntimeError("initial live-memory adapter state is invalid")
        return state

    def start(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        initial_observation: Array,
        *,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryAdapterState:
        if _contains_tracer((state, initial_observation, extended_action_mask)):
            raise RuntimeError("live-memory adapter is host-only")
        coordinator_state = self._coordinator.start(
            state.coordinator_state,
            initial_observation,
            extended_action_mask=extended_action_mask,
        )
        candidate = state.replace(coordinator_state=coordinator_state)
        if bool(jax.device_get(self.state_valid(state) & self.state_valid(candidate))):
            return cast(ExternalLearnedStateLiveMemoryAdapterState, candidate)
        return state

    def _validate_event_input_static(
        self,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
    ) -> None:
        if type(event_input) is not ExternalLearnedStateLiveMemoryEventInput:
            raise TypeError("event_input must be an exact live-memory event input")
        for value, name, dtype in (
            (event_input.query_uncertainty, "query_uncertainty", jnp.float32),
            (
                event_input.query_uncertainty_available,
                "query_uncertainty_available",
                jnp.bool_,
            ),
            (event_input.entry_uncertainty, "entry_uncertainty", jnp.float32),
            (
                event_input.entry_uncertainty_available,
                "entry_uncertainty_available",
                jnp.bool_,
            ),
            (event_input.entry_safety_cost, "entry_safety_cost", jnp.float32),
            (
                event_input.entry_safety_cost_available,
                "entry_safety_cost_available",
                jnp.bool_,
            ),
            (event_input.entry_reliability, "entry_reliability", jnp.float32),
            (event_input.provenance_id, "provenance_id", jnp.int32),
            (event_input.source_id, "source_id", jnp.int32),
        ):
            _require_array(
                value,
                name=f"event_input.{name}",
                shape=(),
                dtype=dtype,
            )

    @staticmethod
    def _event_input_valid(
        event_input: ExternalLearnedStateLiveMemoryEventInput,
    ) -> Array:
        return (
            jnp.isfinite(event_input.query_uncertainty)
            & (event_input.query_uncertainty >= 0.0)
            & (
                event_input.query_uncertainty_available
                | (event_input.query_uncertainty == 0.0)
            )
            & jnp.isfinite(event_input.entry_uncertainty)
            & (event_input.entry_uncertainty >= 0.0)
            & (
                event_input.entry_uncertainty_available
                | (event_input.entry_uncertainty == 0.0)
            )
            & jnp.isfinite(event_input.entry_safety_cost)
            & (event_input.entry_safety_cost >= 0.0)
            & (
                event_input.entry_safety_cost_available
                | (event_input.entry_safety_cost == 0.0)
            )
            & jnp.isfinite(event_input.entry_reliability)
            & (event_input.entry_reliability >= 0.0)
            & (event_input.entry_reliability <= 1.0)
            & (event_input.provenance_id >= 0)
            & (event_input.source_id >= 0)
        )

    def _validate_feedback_static(
        self,
        feedback: ExternalLearnedStateLiveMemoryFeedback,
    ) -> None:
        if type(feedback) is not ExternalLearnedStateLiveMemoryFeedback:
            raise TypeError("feedback must be an exact live-memory feedback")
        for value, name, shape, dtype in (
            (
                feedback.memory_transaction_words,
                "memory_transaction_words",
                (2,),
                jnp.uint32,
            ),
            (feedback.prototype_decision_id, "prototype_decision_id", (4,), jnp.uint32),
            (
                feedback.base_action_before_retrieval,
                "base_action_before_retrieval",
                (),
                jnp.int32,
            ),
            (feedback.effective_action, "effective_action", (), jnp.int32),
            (
                feedback.hard_action_mask,
                "hard_action_mask",
                (self._n_actions,),
                jnp.bool_,
            ),
            (feedback.retrieval_used, "retrieval_used", (), jnp.bool_),
            (
                feedback.counterfactual_available,
                "counterfactual_available",
                (),
                jnp.bool_,
            ),
            (feedback.counterfactual_delta, "counterfactual_delta", (), jnp.float32),
        ):
            _require_array(
                value,
                name=f"feedback.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _feedback_identity_valid(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        feedback: ExternalLearnedStateLiveMemoryFeedback,
        supplied: bool,
    ) -> Array:
        pending = state.pending_binding
        supplied_array = jnp.asarray(supplied, dtype=jnp.bool_)
        exact = (
            supplied_array
            & jnp.array_equal(
                feedback.memory_transaction_words,
                pending.memory_transaction_words,
            )
            & jnp.array_equal(
                feedback.prototype_decision_id,
                pending.prototype_decision_id,
            )
            & (
                feedback.base_action_before_retrieval
                == pending.base_action_before_retrieval
            )
            & (feedback.effective_action == pending.effective_action)
            & jnp.array_equal(feedback.hard_action_mask, pending.hard_action_mask)
            & (feedback.retrieval_used == pending.retrieval_used_expected)
            & jnp.isfinite(feedback.counterfactual_delta)
            & (
                feedback.counterfactual_available
                | (feedback.counterfactual_delta == 0.0)
            )
            & jnp.where(
                pending.retrieval_used_expected,
                jnp.asarray(True, dtype=jnp.bool_),
                ~feedback.counterfactual_available
                & (feedback.counterfactual_delta == 0.0),
            )
        )
        return jnp.where(pending.available, exact, ~supplied_array)

    def _completed_entry(
        self,
        transition: ExternalLearnedStateTransition,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
    ) -> ExperientialMemoryEntry:
        executed_action = jnp.clip(transition.action, 0, self._n_actions - 1)
        return ExperientialMemoryEntry(
            observation=transition.observation,
            key=transition.observation,
            action=jax.nn.one_hot(
                executed_action,
                self._n_actions,
                dtype=jnp.float32,
            ),
            outcome=transition.next_observation,
            reward=transition.reward,
            uncertainty=event_input.entry_uncertainty,
            uncertainty_available=event_input.entry_uncertainty_available,
            safety_cost=event_input.entry_safety_cost,
            safety_cost_available=event_input.entry_safety_cost_available,
            reliability=event_input.entry_reliability,
            utility=jnp.asarray(0.0, dtype=jnp.float32),
            utility_available=jnp.asarray(False, dtype=jnp.bool_),
            representation_version=jnp.asarray(
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION,
                dtype=jnp.int32,
            ),
            valid=jnp.asarray(True, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=event_input.provenance_id,
            source_id=event_input.source_id,
        )

    def _categorical_action(self, action: Array, accepted: Array) -> tuple[Array, Array]:
        proposed = jnp.argmax(action).astype(jnp.int32)
        exact = jax.nn.one_hot(proposed, self._n_actions, dtype=jnp.float32)
        categorical = (
            accepted
            & jnp.all(jnp.isfinite(action))
            & jnp.array_equal(
                jax.lax.bitcast_convert_type(action, jnp.uint32),
                jax.lax.bitcast_convert_type(exact, jnp.uint32),
            )
        )
        return categorical, jnp.where(categorical, proposed, -1).astype(jnp.int32)

    def _replace_coordinator_action(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        replacement: PrototypeCachedPrimitiveActionReplacement,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        inner = state.inner_state.replace(prototype_state=replacement.state)
        return state.replace(
            inner_state=inner,
            current_action=replacement.action,
            current_decision_id=replacement.state.current_decision_id,
            cached_prototype_step_words=replacement.state.step_words,
            cached_feature_generation_words=(
                self._coordinator._feature_generation_words(inner)
            ),
        )

    def prepare_transition(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        transition: ExternalLearnedStateTransition,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
        hard_action_mask: Array,
        prior_feedback: ExternalLearnedStateLiveMemoryFeedback | None = None,
        candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        *,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryPreparedTransition:
        """Settle, evaluate once, query-before-write, and prepare outer adoption."""

        all_inputs = (
            state,
            transition,
            event_input,
            hard_action_mask,
            prior_feedback,
            candidate_evidence,
            partner_policy_fusion_input,
            partner_policy_fusion_feedback,
            extended_action_mask,
        )
        if _contains_tracer(all_inputs):
            raise RuntimeError(
                "live-memory adapter is host-only; monolithic JIT is unsupported"
            )
        if type(state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("state must be an exact live-memory adapter state")
        self._coordinator._validate_transition_static(transition)
        self._validate_event_input_static(event_input)
        mask = _require_array(
            hard_action_mask,
            name="hard_action_mask",
            shape=(self._n_actions,),
            dtype=jnp.bool_,
        )
        supplied = prior_feedback is not None
        feedback = self._blank_feedback() if prior_feedback is None else prior_feedback
        self._validate_feedback_static(feedback)

        query_key = transition.next_decision_observation
        entry = self._completed_entry(transition, event_input)
        feedback_identity = self._feedback_identity_valid(state, feedback, supplied)
        source_valid = self.state_valid(state)
        source_transition_matches = self._coordinator._source_transition_matches(
            state.coordinator_state,
            transition,
        )
        continuing = self._coordinator._continuing_boundary_valid(transition)
        preflight = (
            source_valid
            & source_transition_matches
            & continuing
            & self._event_input_valid(event_input)
            & feedback_identity
        )

        false = jnp.asarray(False, dtype=jnp.bool_)
        zero = jnp.asarray(0, dtype=jnp.int32)

        def prepared_result(
            *,
            settlement_result: LearnedExperientialMemoryFeedbackResult | None,
            coordinator_result: ExternalLearnedStateRouterAuditCoordinatorResult | None,
            memory_result: LearnedExperientialMemoryStepResult | None,
            replacement: PrototypeCachedPrimitiveActionReplacement | None,
            categorical: Array = false,
            retrieval_action: Array | None = None,
            replacement_required: Array = false,
            candidate_state: ExternalLearnedStateLiveMemoryAdapterState = state,
            valid: Array = false,
            settlement_evaluations: Array = zero,
            coordinator_evaluations: Array = zero,
            query_evaluations: Array = zero,
            write_evaluations: Array = zero,
            replacement_evaluations: Array = zero,
        ) -> ExternalLearnedStateLiveMemoryPreparedTransition:
            return ExternalLearnedStateLiveMemoryPreparedTransition(
                source_state=state,
                transition=transition,
                event_input=event_input,
                hard_action_mask=mask,
                feedback=feedback,
                feedback_supplied=jnp.asarray(supplied, dtype=jnp.bool_),
                candidate_evidence=candidate_evidence,
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
                extended_action_mask=extended_action_mask,
                feedback_identity_valid=feedback_identity,
                preflight_valid=preflight,
                settlement_result=settlement_result,
                coordinator_result=coordinator_result,
                learned_memory_result=memory_result,
                cached_action_replacement=replacement,
                query_key=query_key,
                completed_entry=entry,
                categorical_retrieval=categorical,
                retrieval_action=(
                    jnp.asarray(-1, dtype=jnp.int32)
                    if retrieval_action is None
                    else retrieval_action
                ),
                replacement_required=replacement_required,
                candidate_state=candidate_state,
                preparation_valid=valid,
                settlement_evaluations=settlement_evaluations,
                coordinator_evaluations=coordinator_evaluations,
                learned_memory_query_evaluations=query_evaluations,
                learned_memory_write_evaluations=write_evaluations,
                cached_action_replacement_evaluations=replacement_evaluations,
            )

        if not bool(jax.device_get(preflight)):
            return prepared_result(
                settlement_result=None,
                coordinator_result=None,
                memory_result=None,
                replacement=None,
            )

        memory_source = state.learned_memory_state
        settlement_result: LearnedExperientialMemoryFeedbackResult | None = None
        settlement_evaluations = zero
        if bool(jax.device_get(state.pending_binding.available)):
            settlement_evaluations = jnp.asarray(1, dtype=jnp.int32)
            settlement_result = self._learned_memory.settle(
                memory_source,
                LearnedExperientialMemoryFeedback(
                    transaction_words=feedback.memory_transaction_words,
                    retrieval_used=feedback.retrieval_used,
                    counterfactual_available=feedback.counterfactual_available,
                    counterfactual_delta=feedback.counterfactual_delta,
                ),
            )
            settlement_valid = (
                settlement_result.diagnostics.transaction_applied
                & ~settlement_result.state.pending.available
                & self._learned_memory.state_valid(settlement_result.state)
            )
            if not bool(jax.device_get(settlement_valid)):
                return prepared_result(
                    settlement_result=settlement_result,
                    coordinator_result=None,
                    memory_result=None,
                    replacement=None,
                    settlement_evaluations=settlement_evaluations,
                )
            memory_source = settlement_result.state

        coordinator_result = self._coordinator.step(
            state.coordinator_state,
            transition,
            candidate_evidence,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        coordinator_evaluations = jnp.asarray(1, dtype=jnp.int32)
        coordinator_valid = (
            coordinator_result.diagnostics.transaction_applied
            & self._coordinator.state_valid(coordinator_result.state)
        )
        if not bool(jax.device_get(coordinator_valid)):
            return prepared_result(
                settlement_result=settlement_result,
                coordinator_result=coordinator_result,
                memory_result=None,
                replacement=None,
                settlement_evaluations=settlement_evaluations,
                coordinator_evaluations=coordinator_evaluations,
            )

        memory_result = self._learned_memory.step(
            memory_source,
            query_key,
            jnp.asarray(
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION,
                dtype=jnp.int32,
            ),
            event_input.query_uncertainty,
            event_input.query_uncertainty_available,
            entry,
        )
        query_evaluations = jnp.asarray(1, dtype=jnp.int32)
        write_evaluations = jnp.asarray(1, dtype=jnp.int32)
        memory_valid = (
            memory_result.diagnostics.transaction_applied
            & memory_result.wrote
            & memory_result.diagnostics.write_succeeded
            & self._learned_memory.state_valid(memory_result.state)
            & jnp.array_equal(
                memory_result.state.transaction_words,
                coordinator_result.state.event_words,
            )
        )
        if not bool(jax.device_get(memory_valid)):
            return prepared_result(
                settlement_result=settlement_result,
                coordinator_result=coordinator_result,
                memory_result=memory_result,
                replacement=None,
                settlement_evaluations=settlement_evaluations,
                coordinator_evaluations=coordinator_evaluations,
                query_evaluations=query_evaluations,
                write_evaluations=write_evaluations,
            )

        categorical, retrieval_action = self._categorical_action(
            memory_result.retrieval.action,
            memory_result.retrieval.accepted,
        )
        replacement_required = categorical
        replacement: PrototypeCachedPrimitiveActionReplacement | None = None
        replacement_evaluations = zero
        coordinator_candidate = coordinator_result.state
        replacement_valid = jnp.asarray(True, dtype=jnp.bool_)
        if bool(jax.device_get(replacement_required)):
            replacement_evaluations = jnp.asarray(1, dtype=jnp.int32)
            prototype = self._coordinator.inner.prototype
            prototype_state = coordinator_candidate.inner_state.prototype_state
            replacement = prototype.replace_cached_primitive_action(
                prototype_state,
                decision_id=prototype_state.current_decision_id,
                decision_observation=prototype_state.current_representation,
                proposed_action=retrieval_action,
                safety_action_mask=mask,
            )
            replacement_valid = replacement.committed
            if bool(jax.device_get(replacement_valid)):
                coordinator_candidate = self._replace_coordinator_action(
                    coordinator_candidate,
                    replacement,
                )

        memory_pending = memory_result.state.pending.available
        if bool(jax.device_get(memory_pending)):
            base_action_before_retrieval = coordinator_result.state.current_action
            effective_action = coordinator_candidate.current_action
            used_fallback = (
                false
                if replacement is None
                else replacement.dispatch_replacement.used_safe_base_fallback
            )
            retrieval_used_expected = (
                categorical
                & replacement_valid
                & ~used_fallback
                & (retrieval_action == effective_action)
            )
            pending = ExternalLearnedStateLiveMemoryPendingBinding(
                available=jnp.asarray(True, dtype=jnp.bool_),
                memory_transaction_words=memory_result.state.transaction_words,
                prototype_decision_id=coordinator_candidate.current_decision_id,
                base_action_before_retrieval=base_action_before_retrieval,
                effective_action=effective_action,
                retrieval_action=retrieval_action,
                hard_action_mask=mask,
                categorical_retrieval=categorical,
                retrieval_used_expected=retrieval_used_expected,
            )
        else:
            pending = self._blank_pending()

        candidate = ExternalLearnedStateLiveMemoryAdapterState(
            coordinator_state=coordinator_candidate,
            learned_memory_state=memory_result.state,
            pending_binding=pending,
            schema_digest=state.schema_digest,
        )
        candidate_valid = self.state_valid(candidate)
        preparation_valid = (
            preflight
            & coordinator_valid
            & memory_valid
            & replacement_valid
            & candidate_valid
        )
        return prepared_result(
            settlement_result=settlement_result,
            coordinator_result=coordinator_result,
            memory_result=memory_result,
            replacement=replacement,
            categorical=categorical,
            retrieval_action=retrieval_action,
            replacement_required=replacement_required,
            candidate_state=candidate,
            valid=preparation_valid,
            settlement_evaluations=settlement_evaluations,
            coordinator_evaluations=coordinator_evaluations,
            query_evaluations=query_evaluations,
            write_evaluations=write_evaluations,
            replacement_evaluations=replacement_evaluations,
        )

    def integrity_receipt(
        self,
        prepared: ExternalLearnedStateLiveMemoryPreparedTransition,
    ) -> ExternalLearnedStateLiveMemoryIntegrityReceipt:
        if type(prepared) is not ExternalLearnedStateLiveMemoryPreparedTransition:
            raise TypeError("prepared must be an exact live-memory preparation")
        return ExternalLearnedStateLiveMemoryIntegrityReceipt(
            prepared=prepared,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
        )

    def adopt_prepared_transition(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        prepared: ExternalLearnedStateLiveMemoryPreparedTransition,
        receipt: ExternalLearnedStateLiveMemoryIntegrityReceipt,
    ) -> ExternalLearnedStateLiveMemoryResult:
        """Atomically select the complete prepared destination or source."""

        if _contains_tracer((state, prepared, receipt)):
            raise RuntimeError("live-memory adapter adoption is host-only")
        if type(state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("state must be an exact live-memory adapter state")
        if type(prepared) is not ExternalLearnedStateLiveMemoryPreparedTransition:
            raise TypeError("prepared must be an exact live-memory preparation")
        if type(receipt) is not ExternalLearnedStateLiveMemoryIntegrityReceipt:
            raise TypeError("receipt must be an exact live-memory integrity receipt")
        _require_array(
            receipt.integrity_bound,
            name="receipt.integrity_bound",
            shape=(),
            dtype=jnp.bool_,
        )
        source_matches = _tree_equal(state, prepared.source_state)
        receipt_matches = _tree_equal(receipt.prepared, prepared)
        source_valid = self.state_valid(state)
        candidate_valid = self.state_valid(prepared.candidate_state)
        commit = (
            source_valid
            & source_matches
            & receipt_matches
            & receipt.integrity_bound
            & prepared.preparation_valid
            & candidate_valid
        )
        selected = prepared.candidate_state if bool(jax.device_get(commit)) else state

        settlement = prepared.settlement_result
        coordinator = prepared.coordinator_result
        memory = prepared.learned_memory_result
        replacement = prepared.cached_action_replacement
        prior_required = state.pending_binding.available
        prior_settled = (
            jnp.asarray(False, dtype=jnp.bool_)
            if settlement is None
            else settlement.diagnostics.transaction_applied
        )
        prior_learning = (
            jnp.asarray(False, dtype=jnp.bool_)
            if settlement is None
            else settlement.diagnostics.learning_eligible
        )
        coordinator_applied = (
            jnp.asarray(False, dtype=jnp.bool_)
            if coordinator is None
            else coordinator.diagnostics.transaction_applied
        )
        memory_applied = (
            jnp.asarray(False, dtype=jnp.bool_)
            if memory is None
            else memory.diagnostics.transaction_applied
        )
        retrieval_accepted = (
            jnp.asarray(False, dtype=jnp.bool_)
            if memory is None
            else memory.retrieval.accepted
        )
        replacement_committed = (
            jnp.asarray(False, dtype=jnp.bool_)
            if replacement is None
            else replacement.committed
        )
        used_fallback = (
            jnp.asarray(False, dtype=jnp.bool_)
            if replacement is None
            else replacement.dispatch_replacement.used_safe_base_fallback
        )
        coordinator_action = (
            state.coordinator_state.current_action
            if coordinator is None
            else coordinator.state.current_action
        )
        executed_one_hot = jax.nn.one_hot(
            jnp.clip(prepared.transition.action, 0, self._n_actions - 1),
            self._n_actions,
            dtype=jnp.float32,
        )
        entry_action_exact = jnp.array_equal(
            jax.lax.bitcast_convert_type(prepared.completed_entry.action, jnp.uint32),
            jax.lax.bitcast_convert_type(executed_one_hot, jnp.uint32),
        )
        query_before_write = (
            (memory is not None)
            and bool(
                jax.device_get(
                    prepared.learned_memory_query_evaluations == 1
                )
            )
            and bool(
                jax.device_get(
                    prepared.learned_memory_write_evaluations == 1
                )
            )
        )
        query_before_write_array = jnp.asarray(query_before_write, dtype=jnp.bool_)
        pending_bound = (
            prepared.candidate_state.pending_binding.available
            & jnp.array_equal(
                prepared.candidate_state.pending_binding.prototype_decision_id,
                prepared.candidate_state.coordinator_state.current_decision_id,
            )
            & (
                prepared.candidate_state.pending_binding.effective_action
                == prepared.candidate_state.coordinator_state.current_action
            )
        )
        diagnostics = ExternalLearnedStateLiveMemoryDiagnostics(
            source_state_matches=source_matches,
            receipt_matches_preparation=receipt_matches,
            receipt_integrity_bound=receipt.integrity_bound,
            source_state_valid=source_valid,
            source_transition_matches=(
                self._coordinator._source_transition_matches(
                    state.coordinator_state,
                    prepared.transition,
                )
            ),
            continuing_boundary_valid=(
                self._coordinator._continuing_boundary_valid(prepared.transition)
            ),
            prior_feedback_required=prior_required,
            prior_feedback_supplied=prepared.feedback_supplied,
            prior_feedback_identity_valid=prepared.feedback_identity_valid,
            prior_feedback_settled=prior_settled,
            prior_feedback_learning_applied=prior_learning,
            coordinator_transaction_applied=coordinator_applied,
            learned_memory_transaction_applied=memory_applied,
            query_before_write=query_before_write_array,
            completed_entry_executed_action_exact=entry_action_exact,
            categorical_retrieval=prepared.categorical_retrieval,
            soft_retrieval_denied_action_authority=(
                retrieval_accepted
                & ~prepared.categorical_retrieval
                & (prepared.cached_action_replacement_evaluations == 0)
            ),
            cached_action_replacement_required=prepared.replacement_required,
            cached_action_replacement_committed=replacement_committed,
            used_safe_current_action_fallback=used_fallback,
            next_action_changed=(
                prepared.candidate_state.coordinator_state.current_action
                != coordinator_action
            ),
            pending_feedback_created=(
                jnp.asarray(False, dtype=jnp.bool_)
                if memory is None
                else memory.diagnostics.pending_created
            ),
            pending_decision_action_bound=pending_bound,
            candidate_state_valid=candidate_valid,
            transaction_applied=commit,
            complete_source_returned=~commit,
            rejected=~commit,
            settlement_evaluations=prepared.settlement_evaluations,
            coordinator_evaluations=prepared.coordinator_evaluations,
            learned_memory_query_evaluations=(
                prepared.learned_memory_query_evaluations
            ),
            learned_memory_write_evaluations=(
                prepared.learned_memory_write_evaluations
            ),
            cached_action_replacement_evaluations=(
                prepared.cached_action_replacement_evaluations
            ),
            learned_memory_owner_count=jnp.asarray(1, dtype=jnp.int32),
            prototype_historical_memory_owner_count=jnp.asarray(
                0, dtype=jnp.int32
            ),
            learned_embedding_evaluations=jnp.asarray(0, dtype=jnp.int32),
            reencoding_evaluations=jnp.asarray(0, dtype=jnp.int32),
            feedback_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            evidence_authority=jnp.asarray(False, dtype=jnp.bool_),
            promotion_authority=jnp.asarray(False, dtype=jnp.bool_),
        )
        return ExternalLearnedStateLiveMemoryResult(
            state=selected,
            prepared=prepared,
            receipt=receipt,
            diagnostics=diagnostics,
        )

    def step(
        self,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        transition: ExternalLearnedStateTransition,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
        hard_action_mask: Array,
        prior_feedback: ExternalLearnedStateLiveMemoryFeedback | None = None,
        candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        *,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryResult:
        prepared = self.prepare_transition(
            state,
            transition,
            event_input,
            hard_action_mask,
            prior_feedback,
            candidate_evidence,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        receipt = self.integrity_receipt(prepared)
        return self.adopt_prepared_transition(state, prepared, receipt)

    @property
    def resource_budget(self) -> ExternalLearnedStateLiveMemoryResourceBudget:
        state = self.init(jr.key(0))
        total = _tree_nbytes(state)
        coordinator_bytes = (
            measure_external_learned_state_router_audit_coordinator_state_nbytes(
                state.coordinator_state
            )
        )
        memory_bytes = self._learned_memory.resource_budget(
            state.learned_memory_state
        ).owned_persistent_state_bytes
        return ExternalLearnedStateLiveMemoryResourceBudget(
            persistent_state_bytes=total,
            coordinator_state_bytes=coordinator_bytes,
            learned_memory_state_bytes=memory_bytes,
            pending_binding_and_schema_bytes=(
                total - coordinator_bytes - memory_bytes
            ),
            persistent_capacity_growth=0,
            learned_memory_owner_count=1,
            prototype_historical_memory_owner_count=0,
            coordinator_owner_count=1,
            pending_binding_base_action_fields=1,
            pending_binding_hard_action_mask_elements=self._n_actions,
            maximum_settlements_per_event=1,
            coordinator_evaluations_per_event=1,
            learned_memory_queries_per_event=1,
            learned_memory_writes_per_event=1,
            maximum_cached_action_replacements_per_event=1,
            raw_key_materializations_per_event=2,
            learned_embedding_evaluations_per_event=0,
            reencoding_evaluations_per_event=0,
            monolithic_jit_supported=False,
            scan_supported=False,
            feedback_authenticated=False,
            dispatch_authority=False,
            safety_authority=False,
            evidence_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
        )


def measure_external_learned_state_live_memory_adapter_state_nbytes(
    state: ExternalLearnedStateLiveMemoryAdapterState,
) -> int:
    if type(state) is not ExternalLearnedStateLiveMemoryAdapterState:
        raise TypeError("state must be an exact live-memory adapter state")
    return _tree_nbytes(state)


def save_external_learned_state_live_memory_adapter_checkpoint(
    adapter: ExternalLearnedStateLiveMemoryAdapter,
    state: ExternalLearnedStateLiveMemoryAdapterState,
    path: str | Path,
) -> None:
    """Persist only the complete live owner state, never a transient receipt."""

    if type(adapter) is not ExternalLearnedStateLiveMemoryAdapter:
        raise TypeError("adapter must be an exact live-memory adapter")
    if not bool(jax.device_get(adapter.state_valid(state))):
        raise ValueError("refusing to save an invalid live-memory adapter state")
    config = adapter.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA,
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": adapter.resource_budget.to_config(),
            "evidence_level": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL,
            "outcome_status": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "transient_receipt_included": False,
            "learned_memory_owner_count": 1,
            "prototype_historical_memory_owner_count": 0,
            "pending_binding_base_action_recorded": True,
            "pending_binding_hard_action_mask_recorded": True,
            "learned_embedding_enabled": False,
            "reencoding_enabled": False,
            "monolithic_jit_supported": False,
            "scan_supported": False,
            "feedback_authenticated": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        },
    )


def load_external_learned_state_live_memory_adapter_checkpoint(
    path: str | Path,
) -> tuple[
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryAdapterState,
]:
    """Strictly restore the sole current live-memory adapter v1 schema."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "owner_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "outcome_status",
        "scientific_promotion_allowed",
        "transient_receipt_included",
        "learned_memory_owner_count",
        "prototype_historical_memory_owner_count",
        "pending_binding_base_action_recorded",
        "pending_binding_hard_action_mask_recorded",
        "learned_embedding_enabled",
        "reencoding_enabled",
        "monolithic_jit_supported",
        "scan_supported",
        "feedback_authenticated",
        "dispatch_authority",
        "safety_authority",
        "evidence_authority",
        "promotion_authority",
    }
    if set(metadata) != expected:
        raise ValueError("live-memory adapter checkpoint fields are not exact")
    if metadata.get("schema") != (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint is not a live-memory adapter v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("live-memory checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("live-memory checkpoint config digest does not match")
    adapter = ExternalLearnedStateLiveMemoryAdapter.from_config(config)
    resource_budget = metadata.get("resource_budget")
    if type(resource_budget) is not dict or _config_digest(
        resource_budget
    ) != _config_digest(adapter.resource_budget.to_config()):
        raise ValueError("live-memory checkpoint resource budget does not match")
    fixed = {
        "evidence_level": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL,
        "outcome_status": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS,
        "scientific_promotion_allowed": False,
        "transient_receipt_included": False,
        "learned_memory_owner_count": 1,
        "prototype_historical_memory_owner_count": 0,
        "pending_binding_base_action_recorded": True,
        "pending_binding_hard_action_mask_recorded": True,
        "learned_embedding_enabled": False,
        "reencoding_enabled": False,
        "monolithic_jit_supported": False,
        "scan_supported": False,
        "feedback_authenticated": False,
        "dispatch_authority": False,
        "safety_authority": False,
        "evidence_authority": False,
        "promotion_authority": False,
    }
    if any(
        type(metadata.get(name)) is not type(value) or metadata.get(name) != value
        for name, value in fixed.items()
    ):
        raise ValueError("live-memory checkpoint fixed semantics differ")
    template = adapter.init(jr.key(0))
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("live-memory checkpoint metadata changed between reads")
    state = cast(ExternalLearnedStateLiveMemoryAdapterState, restored)
    if not bool(jax.device_get(adapter.state_valid(state))):
        raise ValueError("live-memory checkpoint restored an invalid state")
    if measure_external_learned_state_live_memory_adapter_state_nbytes(
        state
    ) != adapter.resource_budget.persistent_state_bytes:
        raise ValueError("live-memory checkpoint restored a wrong-size state")
    return adapter, state


__all__ = [
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_RAW_SCHEMA_VERSION",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_RECEIPT_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_STATE_SCHEMA",
    "ExternalLearnedStateLiveMemoryAdapter",
    "ExternalLearnedStateLiveMemoryAdapterConfig",
    "ExternalLearnedStateLiveMemoryAdapterState",
    "ExternalLearnedStateLiveMemoryDiagnostics",
    "ExternalLearnedStateLiveMemoryEventInput",
    "ExternalLearnedStateLiveMemoryFeedback",
    "ExternalLearnedStateLiveMemoryIntegrityReceipt",
    "ExternalLearnedStateLiveMemoryPendingBinding",
    "ExternalLearnedStateLiveMemoryPreparedTransition",
    "ExternalLearnedStateLiveMemoryResourceBudget",
    "ExternalLearnedStateLiveMemoryResult",
    "load_external_learned_state_live_memory_adapter_checkpoint",
    "measure_external_learned_state_live_memory_adapter_state_nbytes",
    "save_external_learned_state_live_memory_adapter_checkpoint",
]
