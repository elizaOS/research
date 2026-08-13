# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Live semantic-context consumption through shared consolidated memory.

This opt-in wrapper preserves the complete
:class:`PrototypeConsolidatedMemoryState` as its only persistent payload. It
does not clone or shadow :class:`ConsolidatedMemory`: procedural feedback is
settled first, a semantic pre-write query and current-record write use the same
controller memory, Prototype learns and selects its next action from
``[raw_observation, semantic_payload_or_zero]``, and the procedural next-action
query remains last.

Semantic context may influence the next Prototype policy through its ordinary
representation. It has no direct dispatch authority, no hard-safety override,
and no efficacy, evidence-promotion, or scientific claim. Physical dispatch
and live-environment efficacy remain outside this mechanism.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.consolidated_memory import (
    SemanticMemoryRecord,
    SemanticMemoryRequest,
)
from alberta_framework.core.consolidated_memory_controller import (
    ConsolidatedSemanticMemoryControllerResult,
)
from alberta_framework.core.prototype_agent import (
    PrototypeExperientialMemoryInput,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
)
from alberta_framework.core.prototype_consolidated_memory import (
    PrototypeConsolidatedMemoryAgent,
    PrototypeConsolidatedMemoryConfig,
    PrototypeConsolidatedMemoryDecisionInput,
    PrototypeConsolidatedMemoryDispatchSettlementInput,
    PrototypeConsolidatedMemoryDispatchSettlementResult,
    PrototypeConsolidatedMemoryFeedbackAttempt,
    PrototypeConsolidatedMemoryFeedbackInput,
    PrototypeConsolidatedMemoryResourceBudget,
    PrototypeConsolidatedMemoryStartResult,
    PrototypeConsolidatedMemoryState,
    PrototypeConsolidatedMemoryUpdateResult,
    _increment_prototype_decision_id,
)

PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONFIG_SCHEMA = (
    "alberta.prototype-consolidated-semantic-memory.config.v1"
)
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_SCHEMA = (
    "alberta.prototype-consolidated-semantic-memory.state.v2"
)
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER = (
    "consolidated_procedural_feedback",
    "consolidated_semantic_query_pre_write",
    "consolidated_semantic_current_record_write",
    "prototype_experiential_memory",
    "prototype_partner_policy_fusion",
    "consolidated_procedural_memory",
)
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_MECHANISM_STATUS = (
    "l0_live_context_integration_only"
)
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONTEXT_INFLUENCE_ENABLED = True
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_DIRECT_DISPATCH_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_SAFETY_OVERRIDE_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EFFICACY_CLAIM = False
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EVIDENCE_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_PROMOTION_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_HOST_ONLY = True

_DECISION_WORDS = 4
_DIGEST_BYTES = 32


def _tree_select(predicate: Array, selected: Any, fallback: Any) -> Any:
    return jax.lax.cond(
        predicate,
        lambda _: selected,
        lambda _: fallback,
        operand=None,
    )


def _require_float32_vector(
    value: Array,
    *,
    name: str,
    width: int,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype metadata")
    array = jnp.asarray(value)
    if tuple(array.shape) != (width,):
        raise ValueError(f"{name} must have shape {(width,)}, got {array.shape}")
    if jnp.dtype(array.dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    return array


def _require_array(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype metadata")
    array = jnp.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}, got {array.dtype}")
    return array


def _blank_semantic_request() -> SemanticMemoryRequest:
    negative = jnp.asarray(-1, dtype=jnp.int32)
    return SemanticMemoryRequest(
        semantic_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
        generation=negative,
        kind=negative,
        provenance_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
        representation_revision=negative,
        source_revision=negative,
    )


def _blank_semantic_record(payload_dim: int) -> SemanticMemoryRecord:
    negative = jnp.asarray(-1, dtype=jnp.int32)
    return SemanticMemoryRecord(
        semantic_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
        generation=negative,
        kind=negative,
        payload=jnp.zeros((payload_dim,), dtype=jnp.float32),
        confidence=jnp.asarray(0.0, dtype=jnp.float32),
        provenance_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
        representation_revision=negative,
        source_revision=negative,
        evidence=jnp.asarray(0.0, dtype=jnp.float32),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeConsolidatedSemanticMemoryConfig:
    """Exact direct-context layout around an existing procedural composition."""

    composition: PrototypeConsolidatedMemoryConfig
    raw_observation_dim: int

    SCHEMA_VERSION: ClassVar[str] = (
        PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONFIG_SCHEMA
    )

    def __post_init__(self) -> None:
        if type(self.composition) is not PrototypeConsolidatedMemoryConfig:
            raise TypeError(
                "composition must be an exact PrototypeConsolidatedMemoryConfig"
            )
        if type(self.raw_observation_dim) is not int or self.raw_observation_dim < 1:
            raise ValueError("raw_observation_dim must be a positive exact int")
        prototype = self.composition.prototype
        if prototype.state_builder is not None or prototype.gru_perception is not None:
            raise ValueError(
                "semantic direct context requires state_builder and "
                "gru_perception to be disabled"
            )
        semantic_dim = self.composition.controller.memory.semantic_payload_dim
        expected = self.raw_observation_dim + semantic_dim
        if prototype.oak.observation_dim != expected:
            raise ValueError(
                "Prototype observation_dim must equal raw_observation_dim + "
                f"semantic_payload_dim ({expected})"
            )

    @property
    def semantic_payload_dim(self) -> int:
        return self.composition.controller.memory.semantic_payload_dim

    @property
    def prototype_observation_dim(self) -> int:
        return self.raw_observation_dim + self.semantic_payload_dim

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "mechanism_status": (
                PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_MECHANISM_STATUS
            ),
            "composition_order": list(
                PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER
            ),
            "composition": self.composition.to_config(),
            "raw_observation_dim": self.raw_observation_dim,
            "semantic_payload_dim": self.semantic_payload_dim,
            "prototype_observation_dim": self.prototype_observation_dim,
            "context_layout": "raw_then_semantic_payload_or_zero",
            "query_before_write_required": True,
            "shared_controller_memory_required": True,
            "context_influence_enabled": True,
            "direct_dispatch_authority": False,
            "safety_override_authority": False,
            "efficacy_claim": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: object,
    ) -> PrototypeConsolidatedSemanticMemoryConfig:
        if type(payload) is not dict:
            raise ValueError("semantic adapter config must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema",
            "mechanism_status",
            "composition_order",
            "composition",
            "raw_observation_dim",
            "semantic_payload_dim",
            "prototype_observation_dim",
            "context_layout",
            "query_before_write_required",
            "shared_controller_memory_required",
            "context_influence_enabled",
            "direct_dispatch_authority",
            "safety_override_authority",
            "efficacy_claim",
            "evidence_authority",
            "promotion_authority",
        }
        if set(raw) != expected:
            raise ValueError("semantic adapter config fields differ from schema v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "mechanism_status": (
                PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_MECHANISM_STATUS
            ),
            "composition_order": list(
                PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER
            ),
            "context_layout": "raw_then_semantic_payload_or_zero",
            "query_before_write_required": True,
            "shared_controller_memory_required": True,
            "context_influence_enabled": True,
            "direct_dispatch_authority": False,
            "safety_override_authority": False,
            "efficacy_claim": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }
        if any(
            type(raw[name]) is not type(value) or raw[name] != value
            for name, value in fixed.items()
        ):
            raise ValueError("semantic adapter config fixed fields differ")
        composition = PrototypeConsolidatedMemoryConfig.from_config(
            raw["composition"]
        )
        config = cls(
            composition=composition,
            raw_observation_dim=cast(int, raw["raw_observation_dim"]),
        )
        if (
            type(raw["semantic_payload_dim"]) is not int
            or raw["semantic_payload_dim"] != config.semantic_payload_dim
            or type(raw["prototype_observation_dim"]) is not int
            or raw["prototype_observation_dim"]
            != config.prototype_observation_dim
        ):
            raise ValueError("semantic adapter configured dimensions differ")
        return config


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeConsolidatedSemanticMemoryResourceBudget:
    """Exact incremental allocation, bounded work, and authority declaration."""

    incremental_persistent_state_bytes: int
    incremental_persistent_logical_scalars: int
    shared_controller_memory_states: int
    raw_observation_dim: int
    semantic_payload_dim: int
    prototype_observation_dim: int
    semantic_query_before_write_transactions_per_transition: int
    semantic_memory_operations_per_valid_transition: int
    semantic_payload_cells_consumed_per_transition: int
    additional_controller_feedback_evaluations_per_transition: int
    context_vectors_built_per_transition: int
    context_cells_per_vector: int
    dispatch_settlement_delegations_per_external_action: int
    additional_dispatch_settlement_state_bytes: int
    additional_random_generator_calls_per_transition: int
    direct_dispatches_per_transition: int
    safety_overrides_per_transition: int
    persistent_growth_per_transition_bytes: int
    checkpoint_host_only: bool
    shared_controller_memory_required: bool
    context_influence_enabled: bool
    direct_dispatch_authority: bool
    safety_override_authority: bool
    efficacy_claim: bool
    evidence_authority: bool
    promotion_authority: bool
    composition: PrototypeConsolidatedMemoryResourceBudget


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryState:
    """The settled procedural composition; no second memory state exists."""

    composition: PrototypeConsolidatedMemoryState


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryInput:
    """Exact current/next lifecycle binding for one semantic query and write."""

    available: Bool[Array, ""]
    current_prototype_decision_id: UInt[Array, " 4"]
    next_prototype_decision_id: UInt[Array, " 4"]
    request: SemanticMemoryRequest
    record: SemanticMemoryRecord


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticTransition:
    """Environment transition containing only the explicit raw observation."""

    observation: Float[Array, " raw_observation_dim"]
    action: Int[Array, ""]
    decision_id: UInt[Array, " 4"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    next_observation: Float[Array, " raw_observation_dim"]
    next_decision_observation: Float[Array, " raw_observation_dim"]
    horde_cumulants: Array | None = None
    horde_discounts: Array | None = None


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryDiagnostics:
    composed_state_valid_before: Bool[Array, ""]
    raw_transition_valid_before_feedback: Bool[Array, ""]
    procedural_feedback_candidate_applied: Bool[Array, ""]
    procedural_feedback_cleared_before_semantic: Bool[Array, ""]
    semantic_input_supplied: Bool[Array, ""]
    semantic_input_available: Bool[Array, ""]
    semantic_current_decision_matches: Bool[Array, ""]
    semantic_next_decision_matches: Bool[Array, ""]
    semantic_request_record_binding_matches: Bool[Array, ""]
    semantic_serialized_behind_procedural_feedback: Bool[Array, ""]
    semantic_transaction_allowed: Bool[Array, ""]
    semantic_query_before_write_verified: Bool[Array, ""]
    semantic_candidate_retrieval_accepted: Bool[Array, ""]
    semantic_candidate_write_applied: Bool[Array, ""]
    semantic_zero_tail_used: Bool[Array, ""]
    semantic_context_consumed_by_next_prototype_decision: Bool[Array, ""]
    current_action_unchanged_before_learning: Bool[Array, ""]
    prototype_learning_retained: Bool[Array, ""]
    procedural_order_preserved: Bool[Array, ""]
    direct_dispatch_authority: Bool[Array, ""]
    safety_override_authority: Bool[Array, ""]
    outer_transaction_committed: Bool[Array, ""]
    action_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryStartResult:
    state: PrototypeConsolidatedSemanticMemoryState
    action: Int[Array, ""]
    composition: PrototypeConsolidatedMemoryStartResult
    semantic_payload: Float[Array, " semantic_payload_dim"]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryUpdateResult:
    state: PrototypeConsolidatedSemanticMemoryState
    action: Int[Array, ""]
    composition: PrototypeConsolidatedMemoryUpdateResult
    feedback_attempt: PrototypeConsolidatedMemoryFeedbackAttempt
    semantic_candidate: ConsolidatedSemanticMemoryControllerResult
    semantic_payload: Float[Array, " semantic_payload_dim"]
    diagnostics: PrototypeConsolidatedSemanticMemoryDiagnostics


@chex.dataclass(frozen=True)
class PrototypeConsolidatedSemanticMemoryDispatchSettlementResult:
    """Typed semantic-wrapper view of the delegated atomic settlement."""

    state: PrototypeConsolidatedSemanticMemoryState
    action: Int[Array, ""]
    composition: PrototypeConsolidatedMemoryDispatchSettlementResult


class PrototypeConsolidatedSemanticMemoryAgent:
    """Consume shared semantic memory only through the next Prototype context."""

    def __init__(self, config: PrototypeConsolidatedSemanticMemoryConfig) -> None:
        if type(config) is not PrototypeConsolidatedSemanticMemoryConfig:
            raise TypeError(
                "config must be an exact "
                "PrototypeConsolidatedSemanticMemoryConfig"
            )
        self._config = config
        self._composition = PrototypeConsolidatedMemoryAgent(config.composition)

    @property
    def config(self) -> PrototypeConsolidatedSemanticMemoryConfig:
        return self._config

    @property
    def composition(self) -> PrototypeConsolidatedMemoryAgent:
        return self._composition

    @property
    def composition_order(self) -> tuple[str, ...]:
        return PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER

    @property
    def resource_budget(self) -> PrototypeConsolidatedSemanticMemoryResourceBudget:
        composition = self._composition.resource_budget
        return PrototypeConsolidatedSemanticMemoryResourceBudget(
            incremental_persistent_state_bytes=0,
            incremental_persistent_logical_scalars=0,
            shared_controller_memory_states=1,
            raw_observation_dim=self._config.raw_observation_dim,
            semantic_payload_dim=self._config.semantic_payload_dim,
            prototype_observation_dim=self._config.prototype_observation_dim,
            semantic_query_before_write_transactions_per_transition=1,
            semantic_memory_operations_per_valid_transition=1,
            semantic_payload_cells_consumed_per_transition=(
                self._config.semantic_payload_dim
            ),
            additional_controller_feedback_evaluations_per_transition=2,
            context_vectors_built_per_transition=7,
            context_cells_per_vector=self._config.prototype_observation_dim,
            dispatch_settlement_delegations_per_external_action=1,
            additional_dispatch_settlement_state_bytes=0,
            additional_random_generator_calls_per_transition=0,
            direct_dispatches_per_transition=0,
            safety_overrides_per_transition=0,
            persistent_growth_per_transition_bytes=0,
            checkpoint_host_only=True,
            shared_controller_memory_required=True,
            context_influence_enabled=True,
            direct_dispatch_authority=False,
            safety_override_authority=False,
            efficacy_claim=False,
            evidence_authority=False,
            promotion_authority=False,
            composition=composition,
        )

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: object,
    ) -> PrototypeConsolidatedSemanticMemoryAgent:
        return cls(PrototypeConsolidatedSemanticMemoryConfig.from_config(payload))

    def init(
        self,
        key: Array,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
        lifecycle_id: Array | None = None,
    ) -> PrototypeConsolidatedSemanticMemoryState:
        return PrototypeConsolidatedSemanticMemoryState(
            composition=self._composition.init(
                key,
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation_revision,
                source_revision=source_revision,
                lifecycle_id=lifecycle_id,
            )
        )

    def validate_state(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
    ) -> Bool[Array, ""]:
        if type(state) is not PrototypeConsolidatedSemanticMemoryState:
            raise TypeError(
                "state must be a PrototypeConsolidatedSemanticMemoryState"
            )
        return self._composition.validate_state(state.composition)

    def _normalize_semantic_input(
        self,
        value: PrototypeConsolidatedSemanticMemoryInput | None,
    ) -> tuple[PrototypeConsolidatedSemanticMemoryInput, Array]:
        if value is None:
            return (
                PrototypeConsolidatedSemanticMemoryInput(
                    available=jnp.asarray(False, dtype=jnp.bool_),
                    current_prototype_decision_id=jnp.zeros(
                        (_DECISION_WORDS,), dtype=jnp.uint32
                    ),
                    next_prototype_decision_id=jnp.zeros(
                        (_DECISION_WORDS,), dtype=jnp.uint32
                    ),
                    request=_blank_semantic_request(),
                    record=_blank_semantic_record(
                        self._config.semantic_payload_dim
                    ),
                ),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if type(value) is not PrototypeConsolidatedSemanticMemoryInput:
            raise TypeError(
                "semantic_input must be a "
                "PrototypeConsolidatedSemanticMemoryInput"
            )
        _require_array(
            value.available,
            name="semantic_input.available",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            value.current_prototype_decision_id,
            name="semantic_input.current_prototype_decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            value.next_prototype_decision_id,
            name="semantic_input.next_prototype_decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        return value, jnp.asarray(True, dtype=jnp.bool_)

    def _augment_transition(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        transition: PrototypeConsolidatedSemanticTransition,
        semantic_tail: Array,
    ) -> PrototypeTransition:
        if type(transition) is not PrototypeConsolidatedSemanticTransition:
            raise TypeError(
                "transition must be a PrototypeConsolidatedSemanticTransition"
            )
        raw_dim = self._config.raw_observation_dim
        semantic_dim = self._config.semantic_payload_dim
        observation = _require_float32_vector(
            transition.observation,
            name="transition.observation",
            width=raw_dim,
        )
        next_observation = _require_float32_vector(
            transition.next_observation,
            name="transition.next_observation",
            width=raw_dim,
        )
        next_decision_observation = _require_float32_vector(
            transition.next_decision_observation,
            name="transition.next_decision_observation",
            width=raw_dim,
        )
        tail = _require_float32_vector(
            semantic_tail,
            name="semantic_tail",
            width=semantic_dim,
        )
        prototype_state = state.composition.prototype
        current_tail = prototype_state.current_raw_observation[raw_dim:]
        zero_tail = jnp.zeros((semantic_dim,), dtype=jnp.float32)
        boundary = transition.terminated | transition.truncated
        augmented_observation = jnp.concatenate((observation, current_tail))
        augmented_next_observation = jnp.concatenate(
            (
                next_observation,
                jnp.where(boundary, zero_tail, tail),
            )
        )
        augmented_next_decision = jnp.concatenate(
            (next_decision_observation, tail)
        )
        return PrototypeTransition(
            observation=augmented_observation,
            action=transition.action,
            decision_id=transition.decision_id,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            next_observation=augmented_next_observation,
            next_decision_observation=augmented_next_decision,
            horde_cumulants=transition.horde_cumulants,
            horde_discounts=transition.horde_discounts,
        )

    def start(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        raw_initial_observation: Array,
        *,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
    ) -> PrototypeConsolidatedSemanticMemoryStartResult:
        raw = _require_float32_vector(
            raw_initial_observation,
            name="raw_initial_observation",
            width=self._config.raw_observation_dim,
        )
        semantic_payload = jnp.zeros(
            (self._config.semantic_payload_dim,), dtype=jnp.float32
        )
        composition = self._composition.start(
            state.composition,
            jnp.concatenate((raw, semantic_payload)),
            decision_input=decision_input,
        )
        final_state = PrototypeConsolidatedSemanticMemoryState(
            composition=composition.state
        )
        return PrototypeConsolidatedSemanticMemoryStartResult(
            state=final_state,
            action=composition.action,
            composition=composition,
            semantic_payload=semantic_payload,
        )

    def decide(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        *,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
    ) -> PrototypeConsolidatedSemanticMemoryStartResult:
        composition = self._composition.decide(
            state.composition,
            decision_input=decision_input,
        )
        semantic_payload = composition.state.prototype.current_raw_observation[
            self._config.raw_observation_dim :
        ]
        return PrototypeConsolidatedSemanticMemoryStartResult(
            state=PrototypeConsolidatedSemanticMemoryState(
                composition=composition.state
            ),
            action=composition.action,
            composition=composition,
            semantic_payload=semantic_payload,
        )

    def update_transition(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        transition: PrototypeConsolidatedSemanticTransition,
        *,
        semantic_input: PrototypeConsolidatedSemanticMemoryInput | None = None,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
        feedback_input: PrototypeConsolidatedMemoryFeedbackInput | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
    ) -> PrototypeConsolidatedSemanticMemoryUpdateResult:
        """Settle procedural feedback, query-before-write, then learn/decide."""

        zero_tail = jnp.zeros(
            (self._config.semantic_payload_dim,), dtype=jnp.float32
        )
        preliminary_transition = self._augment_transition(
            state,
            transition,
            zero_tail,
        )
        feedback_attempt = self._composition.attempt_feedback(
            state.composition,
            preliminary_transition,
            feedback_input=feedback_input,
        )
        normalized_semantic, semantic_supplied = self._normalize_semantic_input(
            semantic_input
        )
        current_id_matches = jnp.array_equal(
            normalized_semantic.current_prototype_decision_id,
            state.composition.prototype.current_decision_id,
        ) & jnp.array_equal(
            normalized_semantic.current_prototype_decision_id,
            transition.decision_id,
        )
        expected_next_id = _increment_prototype_decision_id(
            state.composition.prototype.current_decision_id
        )
        next_id_matches = jnp.array_equal(
            normalized_semantic.next_prototype_decision_id,
            expected_next_id,
        )
        semantic_binding_matches = (
            jnp.array_equal(
                normalized_semantic.request.semantic_digest,
                normalized_semantic.record.semantic_digest,
            )
            & (
                normalized_semantic.request.generation
                == normalized_semantic.record.generation
            )
            & (normalized_semantic.request.kind == normalized_semantic.record.kind)
            & jnp.array_equal(
                normalized_semantic.request.provenance_digest,
                normalized_semantic.record.provenance_digest,
            )
            & (
                normalized_semantic.request.representation_revision
                == normalized_semantic.record.representation_revision
            )
            & (
                normalized_semantic.request.source_revision
                == normalized_semantic.record.source_revision
            )
        )
        provisional_controller = feedback_attempt.attempted.state
        serialized_behind_feedback = provisional_controller.pending
        semantic_transaction_allowed = (
            feedback_attempt.composed_state_valid
            & feedback_attempt.transition_valid_before_feedback
            & semantic_supplied
            & normalized_semantic.available
            & current_id_matches
            & next_id_matches
            & semantic_binding_matches
            & (~serialized_behind_feedback)
        )
        semantic_candidate = self._composition.controller.semantic_step(
            provisional_controller,
            request=normalized_semantic.request,
            record=normalized_semantic.record,
            transaction_allowed=semantic_transaction_allowed,
        )
        semantic_payload_available = (
            semantic_candidate.retrieval.accepted
            & semantic_candidate.diagnostics.query_before_write_verified
        )
        semantic_payload = jnp.where(
            semantic_payload_available,
            semantic_candidate.retrieval.payload,
            zero_tail,
        )
        augmented_transition = self._augment_transition(
            state,
            transition,
            semantic_payload,
        )
        preprocessed = dataclasses.replace(
            state.composition,
            controller=semantic_candidate.state
        )
        composition_candidate = self._composition.update_transition(
            preprocessed,
            augmented_transition,
            decision_input=decision_input,
            feedback_input=None,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
        )
        outer_committed = (
            feedback_attempt.composed_state_valid
            & composition_candidate.diagnostics.prototype_learning_retained
            & composition_candidate.diagnostics.transaction_committed
            & composition_candidate.diagnostics.composed_state_valid_after
        )
        final_composition_state = cast(
            PrototypeConsolidatedMemoryState,
            _tree_select(
                outer_committed,
                composition_candidate.state,
                state.composition,
            ),
        )
        action_available = (
            outer_committed & composition_candidate.diagnostics.action_available
        )
        final_action = jnp.where(
            action_available,
            composition_candidate.action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        exposed_feedback = _tree_select(
            outer_committed,
            feedback_attempt.attempted,
            feedback_attempt.rejected,
        )
        final_composition_diagnostics = dataclasses.replace(
            composition_candidate.diagnostics,
            composed_state_valid_before=feedback_attempt.composed_state_valid,
            transition_valid_before_feedback=(
                feedback_attempt.transition_valid_before_feedback
            ),
            feedback_input_supplied=feedback_attempt.input_supplied,
            feedback_input_available=feedback_attempt.normalized_input.available,
            feedback_realized_decision_matches=feedback_attempt.decision_matches,
            feedback_realized_action_matches=feedback_attempt.action_matches,
            feedback_settled_before_prototype_learning=(
                outer_committed
                & feedback_attempt.attempted.diagnostics.pending_cleared
            ),
            prior_upstream_mask_available=(
                feedback_attempt.prior_upstream_mask_available
            ),
            prior_upstream_mask_decision_matches=(
                feedback_attempt.prior_upstream_mask_decision_matches
            ),
            realized_action_allowed_by_prior_upstream_mask=(
                feedback_attempt.realized_action_allowed_by_prior_upstream_mask
            ),
        )
        final_composition_result = dataclasses.replace(
            composition_candidate,
            state=final_composition_state,
            action=final_action,
            memory_feedback=exposed_feedback,
            diagnostics=final_composition_diagnostics,
        )
        next_context_expected = jnp.concatenate(
            (transition.next_decision_observation, semantic_payload)
        )
        context_consumed = (
            outer_committed
            & semantic_payload_available
            & final_composition_state.prototype.started
            & jnp.array_equal(
                final_composition_state.prototype.current_raw_observation,
                next_context_expected,
            )
            & jnp.array_equal(
                final_composition_state.prototype.current_representation,
                next_context_expected,
            )
        )
        final_semantic_payload = jnp.where(
            outer_committed & semantic_payload_available,
            semantic_payload,
            zero_tail,
        )
        final_state = PrototypeConsolidatedSemanticMemoryState(
            composition=final_composition_state
        )
        return PrototypeConsolidatedSemanticMemoryUpdateResult(
            state=final_state,
            action=final_action,
            composition=final_composition_result,
            feedback_attempt=feedback_attempt,
            semantic_candidate=semantic_candidate,
            semantic_payload=final_semantic_payload,
            diagnostics=PrototypeConsolidatedSemanticMemoryDiagnostics(
                composed_state_valid_before=(
                    feedback_attempt.composed_state_valid
                ),
                raw_transition_valid_before_feedback=(
                    feedback_attempt.transition_valid_before_feedback
                ),
                procedural_feedback_candidate_applied=(
                    feedback_attempt.attempted.diagnostics.write_applied
                ),
                procedural_feedback_cleared_before_semantic=(
                    (~state.composition.controller.pending)
                    | feedback_attempt.attempted.diagnostics.pending_cleared
                ),
                semantic_input_supplied=semantic_supplied,
                semantic_input_available=normalized_semantic.available,
                semantic_current_decision_matches=current_id_matches,
                semantic_next_decision_matches=next_id_matches,
                semantic_request_record_binding_matches=(
                    semantic_binding_matches
                ),
                semantic_serialized_behind_procedural_feedback=(
                    serialized_behind_feedback
                ),
                semantic_transaction_allowed=semantic_transaction_allowed,
                semantic_query_before_write_verified=(
                    semantic_candidate.diagnostics.query_before_write_verified
                ),
                semantic_candidate_retrieval_accepted=(
                    semantic_candidate.retrieval.accepted
                ),
                semantic_candidate_write_applied=(
                    semantic_candidate.write.wrote
                ),
                semantic_zero_tail_used=(
                    ~(outer_committed & semantic_payload_available)
                ),
                semantic_context_consumed_by_next_prototype_decision=(
                    context_consumed
                ),
                current_action_unchanged_before_learning=(
                    augmented_transition.action
                    == state.composition.prototype.current_action
                ),
                prototype_learning_retained=(
                    outer_committed
                    & composition_candidate.diagnostics.prototype_learning_retained
                ),
                procedural_order_preserved=jnp.asarray(True, dtype=jnp.bool_),
                direct_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                safety_override_authority=jnp.asarray(False, dtype=jnp.bool_),
                outer_transaction_committed=outer_committed,
                action_available=action_available,
            ),
        )

    def settle_dispatch(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        settlement: PrototypeConsolidatedMemoryDispatchSettlementInput,
    ) -> PrototypeConsolidatedSemanticMemoryDispatchSettlementResult:
        """Delegate the exact post-envelope transaction to the composition."""

        if type(state) is not PrototypeConsolidatedSemanticMemoryState:
            raise TypeError(
                "state must be a PrototypeConsolidatedSemanticMemoryState"
            )
        composition = self._composition.settle_dispatch(
            state.composition,
            settlement,
        )
        return PrototypeConsolidatedSemanticMemoryDispatchSettlementResult(
            state=PrototypeConsolidatedSemanticMemoryState(
                composition=composition.state
            ),
            action=composition.action,
            composition=composition,
        )

    def checkpoint_payload(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
    ) -> dict[str, object]:
        """Return a strict host-only corruption checkpoint, not authentication."""

        if not bool(jax.device_get(self.validate_state(state))):
            raise ValueError("cannot checkpoint an invalid semantic composition")
        return {
            "schema": PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "composition": self._composition.checkpoint_payload(
                state.composition
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
    ) -> PrototypeConsolidatedSemanticMemoryState:
        if type(payload) is not dict:
            raise ValueError("semantic checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema", "config", "composition"}:
            raise ValueError("semantic checkpoint fields differ from schema v2")
        if raw["schema"] != PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_SCHEMA:
            raise ValueError("semantic checkpoint schema differs")
        if PrototypeConsolidatedSemanticMemoryConfig.from_config(
            raw["config"]
        ) != self._config:
            raise ValueError("semantic checkpoint config differs")
        composition = self._composition.restore_checkpoint(
            raw["composition"],
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
        )
        restored = PrototypeConsolidatedSemanticMemoryState(
            composition=composition
        )
        if not bool(jax.device_get(self.validate_state(restored))):
            raise ValueError("semantic checkpoint composition is inconsistent")
        return restored

    def rebind_reset(
        self,
        state: PrototypeConsolidatedSemanticMemoryState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
        discard_pending: bool = False,
    ) -> PrototypeConsolidatedSemanticMemoryState:
        return PrototypeConsolidatedSemanticMemoryState(
            composition=self._composition.rebind_reset(
                state.composition,
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation_revision,
                source_revision=source_revision,
                discard_pending=discard_pending,
            )
        )


__all__ = [
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONFIG_SCHEMA",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONTEXT_INFLUENCE_ENABLED",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_DIRECT_DISPATCH_AUTHORITY",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EFFICACY_CLAIM",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EVIDENCE_AUTHORITY",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_MECHANISM_STATUS",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_PROMOTION_AUTHORITY",
    "PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_SAFETY_OVERRIDE_AUTHORITY",
    "PrototypeConsolidatedSemanticMemoryAgent",
    "PrototypeConsolidatedSemanticMemoryConfig",
    "PrototypeConsolidatedSemanticMemoryDiagnostics",
    "PrototypeConsolidatedSemanticMemoryDispatchSettlementResult",
    "PrototypeConsolidatedSemanticMemoryInput",
    "PrototypeConsolidatedSemanticMemoryResourceBudget",
    "PrototypeConsolidatedSemanticMemoryStartResult",
    "PrototypeConsolidatedSemanticMemoryState",
    "PrototypeConsolidatedSemanticTransition",
    "PrototypeConsolidatedSemanticMemoryUpdateResult",
]
