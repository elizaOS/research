# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Opt-in live Prototype action lifecycle for consolidated procedural memory.

The adapter keeps :class:`PrototypeAgent` configuration, state, and checkpoint
schemas untouched. On a valid real transition it settles exact pending memory
feedback, lets Prototype learn from the primitive action that really ran, and
only then queries memory for the next dispatch. Prototype's built-in dispatch
layers run first, so the fixed composition order is experiential memory,
partner fusion, then consolidated procedural memory. The final hard-safety
mask is the intersection of every applicable upstream mask and the caller's
mask.

This is an L0 integration mechanism. It adds no RNG, autonomous dispatch,
skill creation, evidence promotion, or efficacy claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.consolidated_memory import ProceduralMemoryRequest
from alberta_framework.core.consolidated_memory_controller import (
    ConsolidatedProceduralMemoryController,
    ConsolidatedProceduralMemoryControllerConfig,
    ConsolidatedProceduralMemoryControllerResourceBudget,
    ConsolidatedProceduralMemoryControllerState,
    ConsolidatedProceduralMemoryDecisionResult,
    ConsolidatedProceduralMemoryDispatchCancellationResult,
    ConsolidatedProceduralMemoryFeedbackResult,
)
from alberta_framework.core.partner_policy_fusion import (
    PartnerFusionFeedbackCancellationResult,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeCachedPrimitiveActionReplacement,
    PrototypeExperientialMemoryInput,
    PrototypeInteractionState,
    PrototypeMemoryInteractionState,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeTransitionDiagnostics,
    PrototypeUpdateResult,
)

PROTOTYPE_CONSOLIDATED_MEMORY_CONFIG_SCHEMA = (
    "alberta.prototype-consolidated-procedural-memory.config.v1"
)
PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA = (
    "alberta.prototype-consolidated-procedural-memory.state.v2"
)
PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER = (
    "prototype_experiential_memory",
    "prototype_partner_policy_fusion",
    "consolidated_procedural_memory",
)
PROTOTYPE_CONSOLIDATED_MEMORY_MECHANISM_STATUS = "l0_live_integration_only"
PROTOTYPE_CONSOLIDATED_MEMORY_CACHED_ACTION_REPLACEMENT_ENABLED = True
PROTOTYPE_CONSOLIDATED_MEMORY_DISPATCH_SETTLEMENT_ENABLED = True
PROTOTYPE_CONSOLIDATED_MEMORY_AUTONOMOUS_POLICY_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_MEMORY_PHYSICAL_DISPATCH_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY = False
PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_HOST_ONLY = True

_DIGEST_BYTES = 32
_DECISION_WORDS = 4
_UPSTREAM_MASK_CHECKSUM_SEED = jnp.asarray(
    (0xA341316C, 0xC8013EA4, 0xAD90777D, 0x7E95761E),
    dtype=jnp.uint32,
)
_DISPATCH_OWNER_CHECKSUM_SEED = jnp.asarray(
    (0xD1B54A32, 0xD192ED03, 0x94D049BB, 0x133111EB),
    dtype=jnp.uint32,
)


def _tree_select(predicate: Array, selected: Any, fallback: Any) -> Any:
    """Select one of two identical fixed PyTrees without a Python branch."""

    return jax.lax.cond(predicate, lambda _: selected, lambda _: fallback, operand=None)


def _tree_sha256(tree: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        materialized = (
            jax.random.key_data(leaf)
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        )
        host = np.asarray(jax.device_get(materialized))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _blank_request() -> ProceduralMemoryRequest:
    zero_digest = jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8)
    negative = jnp.asarray(-1, dtype=jnp.int32)
    return ProceduralMemoryRequest(
        semantic_digest=zero_digest,
        generation=negative,
        provenance_digest=zero_digest,
        representation_revision=negative,
        source_revision=negative,
        lifecycle_link_available=jnp.asarray(False, dtype=jnp.bool_),
        lifecycle_digest=zero_digest,
        lifecycle_generation=negative,
        lifecycle_revision=negative,
    )


def _increment_prototype_decision_id(decision_id: Array) -> Array:
    """Advance the exact low two words of one Prototype lifecycle ID."""

    one = jnp.asarray(1, dtype=jnp.uint32)
    low = decision_id[3] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = decision_id[2] + carry
    return jnp.stack((decision_id[0], decision_id[1], high, low))


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeConsolidatedMemoryConfig:
    """Exact Prototype and controller configurations for the opt-in adapter."""

    prototype: PrototypeAgentConfig
    controller: ConsolidatedProceduralMemoryControllerConfig

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_CONSOLIDATED_MEMORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.prototype) is not PrototypeAgentConfig:
            raise TypeError("prototype must be an exact PrototypeAgentConfig")
        if type(self.controller) is not ConsolidatedProceduralMemoryControllerConfig:
            raise TypeError(
                "controller must be an exact "
                "ConsolidatedProceduralMemoryControllerConfig"
            )
        if (
            self.prototype.oak.n_primitive_actions
            != self.controller.policy.n_actions
        ):
            raise ValueError(
                "Prototype primitive-action count must equal controller n_actions"
            )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "mechanism_status": PROTOTYPE_CONSOLIDATED_MEMORY_MECHANISM_STATUS,
            "composition_order": list(PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER),
            "prototype": self.prototype.to_config(),
            "controller": self.controller.to_config(),
            "caller_hard_safety_mask_required": True,
            "query_before_write_required": True,
            "cached_action_replacement_enabled": True,
            "autonomous_policy_authority": False,
            "physical_dispatch_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(cls, payload: object) -> PrototypeConsolidatedMemoryConfig:
        if type(payload) is not dict:
            raise ValueError("adapter config must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {
            "schema",
            "mechanism_status",
            "composition_order",
            "prototype",
            "controller",
            "caller_hard_safety_mask_required",
            "query_before_write_required",
            "cached_action_replacement_enabled",
            "autonomous_policy_authority",
            "physical_dispatch_authority",
            "promotion_authority",
        }:
            raise ValueError("adapter config fields differ from schema v1")
        if (
            raw["schema"] != cls.SCHEMA_VERSION
            or raw["mechanism_status"]
            != PROTOTYPE_CONSOLIDATED_MEMORY_MECHANISM_STATUS
            or raw["composition_order"]
            != list(PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER)
            or raw["caller_hard_safety_mask_required"] is not True
            or raw["query_before_write_required"] is not True
            or raw["cached_action_replacement_enabled"] is not True
            or raw["autonomous_policy_authority"] is not False
            or raw["physical_dispatch_authority"] is not False
            or raw["promotion_authority"] is not False
        ):
            raise ValueError("adapter config fixed fields differ")
        prototype = raw["prototype"]
        if type(prototype) is not dict:
            raise ValueError("prototype config must be an exact dict")
        return cls(
            prototype=PrototypeAgentConfig.from_config(
                cast(dict[str, Any], prototype)
            ),
            controller=(
                ConsolidatedProceduralMemoryControllerConfig.from_config(
                    raw["controller"]
                )
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeConsolidatedMemoryResourceBudget:
    """Fixed incremental storage and per-call work of the opt-in adapter."""

    incremental_persistent_state_bytes: int
    incremental_persistent_logical_scalars: int
    pending_memory_decisions: int
    pending_upstream_mask_records: int
    maximum_memory_operations: int
    memory_feedback_attempts_per_transition: int
    controller_feedback_evaluations_per_transition: int
    memory_queries_per_decision_call: int
    cached_action_replacements_per_decision_call: int
    upstream_mask_input_cells_composed_per_transition: int
    final_mask_input_cells_composed_per_dispatch_composition: int
    realized_action_mask_checks_per_transition: int
    cached_action_mask_checks_per_dispatch_composition: int
    upstream_mask_checksum_words: int
    dispatch_owner_checksum_words: int
    pending_dispatch_settlement_records: int
    dispatch_settlement_identity_checks_per_call: int
    cached_action_replacements_per_changed_dispatch_settlement: int
    procedural_cancellations_per_changed_dispatch_settlement: int
    partner_cancellations_per_changed_dispatch_settlement: int
    memory_writes_per_dispatch_settlement: int
    learner_parameter_updates_per_dispatch_settlement: int
    partner_parameter_updates_per_dispatch_settlement: int
    random_generator_calls_per_dispatch_settlement: int
    physical_commands_per_dispatch_settlement: int
    additional_random_generator_calls_per_event: int
    physical_commands_per_event: int
    additional_agent_parameter_updates_per_event: int
    persistent_growth_per_event_bytes: int
    checkpoint_host_only: bool
    cached_action_replacement_enabled: bool
    dispatch_settlement_enabled: bool
    autonomous_policy_authority: bool
    physical_dispatch_authority: bool
    promotion_authority: bool
    controller: ConsolidatedProceduralMemoryControllerResourceBudget


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryUpstreamMaskState:
    """One corruption-detecting upstream mask bound to an exact next decision.

    The unkeyed checksum detects accidental state drift. It is neither an
    authenticator nor protection against a party that can rewrite the state.
    """

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    hard_safety_action_mask: Bool[Array, " n_actions"]
    checksum: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDispatchOwnerState:
    """Checksum-bound authority for the exact post-envelope dispatch receipt.

    The checksum detects accidental drift; it is not authentication against an
    actor able to rewrite both this record and its checksum.
    """

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    hard_safety_action_mask: Bool[Array, " n_actions"]
    checksum: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryState:
    """Prototype state plus its opt-in consolidated-memory sidecar."""

    prototype: PrototypeAgentState
    controller: ConsolidatedProceduralMemoryControllerState
    upstream_mask: PrototypeConsolidatedMemoryUpstreamMaskState
    dispatch_owner: PrototypeConsolidatedMemoryDispatchOwnerState


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDispatchSettlementInput:
    """Executed envelope outcome bound to one exact cached dispatch."""

    action_available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    executed_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDispatchSettlementDiagnostics:
    """Fail-closed audit for the atomic post-envelope owner settlement."""

    composed_state_valid_before: Bool[Array, ""]
    action_available: Bool[Array, ""]
    no_action: Bool[Array, ""]
    dispatch_owner_available: Bool[Array, ""]
    decision_identity_matches: Bool[Array, ""]
    selected_action_matches_owner: Bool[Array, ""]
    selected_action_matches_cache: Bool[Array, ""]
    executed_action_contract_valid: Bool[Array, ""]
    executed_action_allowed_by_bound_mask: Bool[Array, ""]
    action_changed: Bool[Array, ""]
    prototype_replacement_required: Bool[Array, ""]
    prototype_replacement_committed: Bool[Array, ""]
    procedural_owner_current: Bool[Array, ""]
    procedural_cancellation_required: Bool[Array, ""]
    procedural_cancellation_applied: Bool[Array, ""]
    partner_owner_current: Bool[Array, ""]
    partner_owner_consistent_for_change: Bool[Array, ""]
    partner_armed_identity_current: Bool[Array, ""]
    partner_armed_action_source_bound: Bool[Array, ""]
    partner_cancellation_required: Bool[Array, ""]
    partner_cancellation_applied: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    state_changed: Bool[Array, ""]
    learner_update_applied: Bool[Array, ""]
    memory_evidence_written: Bool[Array, ""]
    partner_learning_applied: Bool[Array, ""]
    random_generator_consumed: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    evidence_promotion_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDispatchSettlementResult:
    state: PrototypeConsolidatedMemoryState
    action: Int[Array, ""]
    prototype_replacement: PrototypeCachedPrimitiveActionReplacement
    procedural_cancellation: ConsolidatedProceduralMemoryDispatchCancellationResult
    partner_cancellation: PartnerFusionFeedbackCancellationResult | None
    diagnostics: PrototypeConsolidatedMemoryDispatchSettlementDiagnostics


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDecisionInput:
    """Exact next-decision identity, compatibility request, and hard mask."""

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    request: ProceduralMemoryRequest
    hard_safety_action_mask: Bool[Array, " n_actions"]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryFeedbackInput:
    """Observed outcome bound to the exact prior dispatch and request."""

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    feedback_event_id: UInt[Array, " 4"]
    base_action: Int[Array, ""]
    effective_action: Int[Array, ""]
    request: ProceduralMemoryRequest
    succeeded: Bool[Array, ""]
    outcome: Array
    confidence: Array
    evidence: Array


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryFeedbackAttempt:
    """Read-only assessment plus candidate exact procedural feedback result."""

    attempted: ConsolidatedProceduralMemoryFeedbackResult
    rejected: ConsolidatedProceduralMemoryFeedbackResult
    normalized_input: PrototypeConsolidatedMemoryFeedbackInput
    input_supplied: Bool[Array, ""]
    decision_matches: Bool[Array, ""]
    action_matches: Bool[Array, ""]
    composed_state_valid: Bool[Array, ""]
    transition: PrototypeTransitionDiagnostics
    transition_valid_before_feedback: Bool[Array, ""]
    prior_upstream_mask_available: Bool[Array, ""]
    prior_upstream_mask_decision_matches: Bool[Array, ""]
    realized_action_allowed_by_prior_upstream_mask: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryDiagnostics:
    """Audit of feedback, upstream ordering, hard masks, and final dispatch."""

    composed_state_valid_before: Bool[Array, ""]
    next_dispatch_allowed: Bool[Array, ""]
    transition_valid_before_feedback: Bool[Array, ""]
    feedback_input_supplied: Bool[Array, ""]
    feedback_input_available: Bool[Array, ""]
    feedback_realized_decision_matches: Bool[Array, ""]
    feedback_realized_action_matches: Bool[Array, ""]
    feedback_settled_before_prototype_learning: Bool[Array, ""]
    prior_upstream_mask_available: Bool[Array, ""]
    prior_upstream_mask_decision_matches: Bool[Array, ""]
    realized_action_allowed_by_prior_upstream_mask: Bool[Array, ""]
    prototype_learning_retained: Bool[Array, ""]
    decision_input_supplied: Bool[Array, ""]
    decision_input_available: Bool[Array, ""]
    decision_prototype_id_matches: Bool[Array, ""]
    experiential_memory_precedes_consolidated: Bool[Array, ""]
    partner_fusion_precedes_consolidated: Bool[Array, ""]
    experiential_mask_applicable: Bool[Array, ""]
    partner_mask_applicable: Bool[Array, ""]
    upstream_mask_available: Bool[Array, ""]
    upstream_mask_decision_matches: Bool[Array, ""]
    cached_action_allowed_by_upstream_mask: Bool[Array, ""]
    upstream_mask_consumed: Bool[Array, ""]
    caller_hard_safety_action_mask: Bool[Array, " n_actions"]
    experiential_hard_safety_action_mask: Bool[Array, " n_actions"]
    partner_hard_safety_action_mask: Bool[Array, " n_actions"]
    upstream_hard_safety_action_mask: Bool[Array, " n_actions"]
    final_hard_safety_action_mask: Bool[Array, " n_actions"]
    final_mask_is_exact_intersection: Bool[Array, ""]
    query_after_prototype_learning: Bool[Array, ""]
    dispatch_replacement_committed: Bool[Array, ""]
    composed_state_valid_after: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    action_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryStartResult:
    state: PrototypeConsolidatedMemoryState
    action: Int[Array, ""]
    memory_decision: ConsolidatedProceduralMemoryDecisionResult
    dispatch_replacement: PrototypeCachedPrimitiveActionReplacement
    diagnostics: PrototypeConsolidatedMemoryDiagnostics


@chex.dataclass(frozen=True)
class PrototypeConsolidatedMemoryUpdateResult:
    state: PrototypeConsolidatedMemoryState
    action: Int[Array, ""]
    prototype: PrototypeUpdateResult
    memory_feedback: ConsolidatedProceduralMemoryFeedbackResult
    memory_decision: ConsolidatedProceduralMemoryDecisionResult
    dispatch_replacement: PrototypeCachedPrimitiveActionReplacement
    diagnostics: PrototypeConsolidatedMemoryDiagnostics


class PrototypeConsolidatedMemoryAgent:
    """Run consolidated procedural memory in Prototype's real action lifecycle.

    An explicit ``start``, ``decide``, or ``update_transition`` call may change
    Prototype's cached action and its learning-credit owner through the public
    atomic replacement helper. The adapter has no autonomous policy loop and
    no physical command transport; the caller still decides when to invoke it
    and whether to send the returned cached action to an environment.
    """

    def __init__(self, config: PrototypeConsolidatedMemoryConfig) -> None:
        if type(config) is not PrototypeConsolidatedMemoryConfig:
            raise TypeError("config must be an exact PrototypeConsolidatedMemoryConfig")
        self._config = config
        self._prototype = PrototypeAgent(config.prototype)
        self._controller = ConsolidatedProceduralMemoryController(config.controller)

    @property
    def config(self) -> PrototypeConsolidatedMemoryConfig:
        return self._config

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def controller(self) -> ConsolidatedProceduralMemoryController:
        return self._controller

    @property
    def composition_order(self) -> tuple[str, str, str]:
        return PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER

    @property
    def resource_budget(self) -> PrototypeConsolidatedMemoryResourceBudget:
        controller = self._controller.resource_budget
        n_actions = self._config.controller.policy.n_actions
        # bool available + uint32[4] identity + bool[n] mask + uint32[4]
        # checksum. JAX stores these as separate leaves, so no padding applies.
        upstream_mask_bytes = 1 + 16 + n_actions + 16
        upstream_mask_scalars = 1 + 4 + n_actions + 4
        # The dispatch owner additionally binds one int32 selected action.
        dispatch_owner_bytes = upstream_mask_bytes + 4
        dispatch_owner_scalars = upstream_mask_scalars + 1
        return PrototypeConsolidatedMemoryResourceBudget(
            incremental_persistent_state_bytes=(
                controller.persistent_state_bytes
                + upstream_mask_bytes
                + dispatch_owner_bytes
            ),
            incremental_persistent_logical_scalars=(
                controller.persistent_logical_scalars
                + upstream_mask_scalars
                + dispatch_owner_scalars
            ),
            pending_memory_decisions=controller.pending_slots,
            pending_upstream_mask_records=1,
            maximum_memory_operations=controller.maximum_memory_operations,
            memory_feedback_attempts_per_transition=1,
            controller_feedback_evaluations_per_transition=2,
            memory_queries_per_decision_call=1,
            cached_action_replacements_per_decision_call=1,
            upstream_mask_input_cells_composed_per_transition=2 * n_actions,
            final_mask_input_cells_composed_per_dispatch_composition=(
                2 * n_actions
            ),
            realized_action_mask_checks_per_transition=1,
            cached_action_mask_checks_per_dispatch_composition=1,
            upstream_mask_checksum_words=_DECISION_WORDS,
            dispatch_owner_checksum_words=_DECISION_WORDS,
            pending_dispatch_settlement_records=1,
            dispatch_settlement_identity_checks_per_call=3,
            cached_action_replacements_per_changed_dispatch_settlement=1,
            procedural_cancellations_per_changed_dispatch_settlement=1,
            partner_cancellations_per_changed_dispatch_settlement=int(
                self._prototype.partner_policy_fusion is not None
            ),
            memory_writes_per_dispatch_settlement=0,
            learner_parameter_updates_per_dispatch_settlement=0,
            partner_parameter_updates_per_dispatch_settlement=0,
            random_generator_calls_per_dispatch_settlement=0,
            physical_commands_per_dispatch_settlement=0,
            additional_random_generator_calls_per_event=0,
            physical_commands_per_event=0,
            additional_agent_parameter_updates_per_event=0,
            persistent_growth_per_event_bytes=0,
            checkpoint_host_only=True,
            cached_action_replacement_enabled=True,
            dispatch_settlement_enabled=True,
            autonomous_policy_authority=False,
            physical_dispatch_authority=False,
            promotion_authority=False,
            controller=controller,
        )

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: object) -> PrototypeConsolidatedMemoryAgent:
        return cls(PrototypeConsolidatedMemoryConfig.from_config(payload))

    def init(
        self,
        key: Array,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
        lifecycle_id: Array | None = None,
    ) -> PrototypeConsolidatedMemoryState:
        return PrototypeConsolidatedMemoryState(
            prototype=self._prototype.init(key, lifecycle_id=lifecycle_id),
            controller=self._controller.init(
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation_revision,
                source_revision=source_revision,
            ),
            upstream_mask=self._blank_upstream_mask(),
            dispatch_owner=self._blank_dispatch_owner(),
        )

    def _upstream_mask_checksum(
        self,
        state: PrototypeConsolidatedMemoryUpstreamMaskState,
    ) -> Array:
        """Return a fixed-shape JAX checksum for the pending mask record."""

        words = jnp.concatenate(
            (
                state.available.astype(jnp.uint32).reshape((1,)),
                state.prototype_decision_id.astype(jnp.uint32).reshape((-1,)),
                state.hard_safety_action_mask.astype(jnp.uint32).reshape((-1,)),
            )
        )
        positions = jnp.arange(words.shape[0], dtype=jnp.uint32) + 1
        encoded = (
            words
            ^ (positions * jnp.asarray(0x9E3779B9, dtype=jnp.uint32))
            ^ jnp.roll(words, 1)
        )
        return _UPSTREAM_MASK_CHECKSUM_SEED ^ jnp.stack(
            (
                jnp.bitwise_xor.reduce(
                    encoded * jnp.asarray(0x01000193, dtype=jnp.uint32)
                ),
                jnp.sum(
                    (encoded + positions)
                    * jnp.asarray(0x85EBCA6B, dtype=jnp.uint32),
                    dtype=jnp.uint32,
                ),
                jnp.bitwise_xor.reduce(
                    (encoded + jnp.roll(encoded, 2))
                    * (positions | jnp.asarray(1, dtype=jnp.uint32))
                ),
                jnp.sum(
                    (encoded ^ jnp.roll(encoded, 3))
                    * jnp.asarray(0xC2B2AE35, dtype=jnp.uint32),
                    dtype=jnp.uint32,
                ),
            )
        ).astype(jnp.uint32)

    def _upstream_mask_record(
        self,
        *,
        available: Array,
        prototype_decision_id: Array,
        hard_safety_action_mask: Array,
    ) -> PrototypeConsolidatedMemoryUpstreamMaskState:
        n_actions = self._config.controller.policy.n_actions
        available_array = jnp.asarray(available, dtype=jnp.bool_)
        canonical = PrototypeConsolidatedMemoryUpstreamMaskState(
            available=available_array,
            prototype_decision_id=jnp.where(
                available_array,
                prototype_decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ).astype(jnp.uint32),
            hard_safety_action_mask=jnp.where(
                available_array,
                hard_safety_action_mask,
                jnp.ones((n_actions,), dtype=jnp.bool_),
            ).astype(jnp.bool_),
            checksum=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
        )
        return dataclasses.replace(
            canonical,
            checksum=self._upstream_mask_checksum(canonical),
        )

    def _blank_upstream_mask(
        self,
    ) -> PrototypeConsolidatedMemoryUpstreamMaskState:
        return self._upstream_mask_record(
            available=jnp.asarray(False, dtype=jnp.bool_),
            prototype_decision_id=jnp.zeros(
                (_DECISION_WORDS,), dtype=jnp.uint32
            ),
            hard_safety_action_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
        )

    def _dispatch_owner_checksum(
        self,
        state: PrototypeConsolidatedMemoryDispatchOwnerState,
    ) -> Array:
        words = jnp.concatenate(
            (
                state.available.astype(jnp.uint32).reshape((1,)),
                state.prototype_decision_id.astype(jnp.uint32).reshape((-1,)),
                state.selected_action.astype(jnp.uint32).reshape((1,)),
                state.hard_safety_action_mask.astype(jnp.uint32).reshape((-1,)),
            )
        )
        positions = jnp.arange(words.shape[0], dtype=jnp.uint32) + 1
        encoded = (
            words
            ^ (positions * jnp.asarray(0x9E3779B9, dtype=jnp.uint32))
            ^ jnp.roll(words, 2)
        )
        return _DISPATCH_OWNER_CHECKSUM_SEED ^ jnp.stack(
            (
                jnp.bitwise_xor.reduce(
                    encoded * jnp.asarray(0x01000193, dtype=jnp.uint32)
                ),
                jnp.sum(
                    (encoded + positions)
                    * jnp.asarray(0x85EBCA6B, dtype=jnp.uint32),
                    dtype=jnp.uint32,
                ),
                jnp.bitwise_xor.reduce(
                    (encoded + jnp.roll(encoded, 3))
                    * (positions | jnp.asarray(1, dtype=jnp.uint32))
                ),
                jnp.sum(
                    (encoded ^ jnp.roll(encoded, 1))
                    * jnp.asarray(0xC2B2AE35, dtype=jnp.uint32),
                    dtype=jnp.uint32,
                ),
            )
        ).astype(jnp.uint32)

    def _dispatch_owner_record(
        self,
        *,
        available: Array,
        prototype_decision_id: Array,
        selected_action: Array,
        hard_safety_action_mask: Array,
    ) -> PrototypeConsolidatedMemoryDispatchOwnerState:
        n_actions = self._config.controller.policy.n_actions
        available_array = jnp.asarray(available, dtype=jnp.bool_)
        canonical = PrototypeConsolidatedMemoryDispatchOwnerState(
            available=available_array,
            prototype_decision_id=jnp.where(
                available_array,
                prototype_decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ).astype(jnp.uint32),
            selected_action=jnp.where(available_array, selected_action, -1).astype(
                jnp.int32
            ),
            hard_safety_action_mask=jnp.where(
                available_array,
                hard_safety_action_mask,
                jnp.ones((n_actions,), dtype=jnp.bool_),
            ).astype(jnp.bool_),
            checksum=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
        )
        return dataclasses.replace(
            canonical,
            checksum=self._dispatch_owner_checksum(canonical),
        )

    def _blank_dispatch_owner(
        self,
    ) -> PrototypeConsolidatedMemoryDispatchOwnerState:
        return self._dispatch_owner_record(
            available=jnp.asarray(False, dtype=jnp.bool_),
            prototype_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            selected_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_safety_action_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
        )

    def _validate_dispatch_owner(
        self,
        state: PrototypeConsolidatedMemoryDispatchOwnerState,
        prototype_state: PrototypeAgentState,
    ) -> Array:
        if type(state) is not PrototypeConsolidatedMemoryDispatchOwnerState:
            raise TypeError(
                "dispatch_owner must be a "
                "PrototypeConsolidatedMemoryDispatchOwnerState"
            )
        n_actions = self._config.controller.policy.n_actions
        contracts = (
            (state.available, (), jnp.bool_, "available"),
            (
                state.prototype_decision_id,
                (_DECISION_WORDS,),
                jnp.uint32,
                "prototype_decision_id",
            ),
            (state.selected_action, (), jnp.int32, "selected_action"),
            (
                state.hard_safety_action_mask,
                (n_actions,),
                jnp.bool_,
                "hard_safety_action_mask",
            ),
            (state.checksum, (_DECISION_WORDS,), jnp.uint32, "checksum"),
        )
        for value, shape, dtype, name in contracts:
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"dispatch owner {name} must expose array metadata")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"dispatch owner {name} must have shape {shape}, "
                    f"got {tuple(value.shape)}"
                )
            if jnp.dtype(value.dtype) != jnp.dtype(dtype):
                raise TypeError(
                    f"dispatch owner {name} must have dtype {jnp.dtype(dtype)}, "
                    f"got {value.dtype}"
                )
        safe_action = jnp.clip(state.selected_action, 0, n_actions - 1)
        absent_is_canonical = (
            (~state.available)
            & jnp.all(state.prototype_decision_id == 0)
            & (state.selected_action == -1)
            & jnp.all(state.hard_safety_action_mask)
        )
        available_is_bound = (
            state.available
            & prototype_state.started
            & jnp.array_equal(
                state.prototype_decision_id,
                prototype_state.current_decision_id,
            )
            & (state.selected_action == prototype_state.current_action)
            & (state.selected_action >= 0)
            & (state.selected_action < n_actions)
            & state.hard_safety_action_mask[safe_action]
        )
        return (
            (absent_is_canonical | available_is_bound)
            & jnp.array_equal(state.checksum, self._dispatch_owner_checksum(state))
        )

    def _validate_upstream_mask(
        self,
        state: PrototypeConsolidatedMemoryUpstreamMaskState,
        prototype_state: PrototypeAgentState,
    ) -> Array:
        if type(state) is not PrototypeConsolidatedMemoryUpstreamMaskState:
            raise TypeError(
                "upstream_mask must be a "
                "PrototypeConsolidatedMemoryUpstreamMaskState"
            )
        n_actions = self._config.controller.policy.n_actions
        contracts = (
            (state.available, (), jnp.bool_, "available"),
            (
                state.prototype_decision_id,
                (_DECISION_WORDS,),
                jnp.uint32,
                "prototype_decision_id",
            ),
            (
                state.hard_safety_action_mask,
                (n_actions,),
                jnp.bool_,
                "hard_safety_action_mask",
            ),
            (state.checksum, (_DECISION_WORDS,), jnp.uint32, "checksum"),
        )
        for value, shape, dtype, name in contracts:
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"upstream mask {name} must expose array metadata")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"upstream mask {name} must have shape {shape}, "
                    f"got {tuple(value.shape)}"
                )
            if jnp.dtype(value.dtype) != jnp.dtype(dtype):
                raise TypeError(
                    f"upstream mask {name} must have dtype {jnp.dtype(dtype)}, "
                    f"got {value.dtype}"
                )
        absent_is_canonical = (
            (~state.available)
            & jnp.all(
                state.prototype_decision_id
                == jnp.asarray(0, dtype=jnp.uint32)
            )
            & jnp.all(state.hard_safety_action_mask)
        )
        available_is_bound = (
            state.available
            & prototype_state.started
            & jnp.array_equal(
                state.prototype_decision_id,
                prototype_state.current_decision_id,
            )
        )
        return (
            (absent_is_canonical | available_is_bound)
            & jnp.array_equal(state.checksum, self._upstream_mask_checksum(state))
        )

    def validate_state(
        self,
        state: PrototypeConsolidatedMemoryState,
    ) -> Bool[Array, ""]:
        if type(state) is not PrototypeConsolidatedMemoryState:
            raise TypeError("state must be a PrototypeConsolidatedMemoryState")
        controller_state = state.controller
        controller_valid = self._controller.validate_state(
            controller_state,
            source_digest=controller_state.memory.source_digest,
            semantic_namespace_digest=(
                controller_state.memory.semantic_namespace_digest
            ),
            representation_revision=(
                controller_state.memory.representation_revision
            ),
            source_revision=controller_state.memory.source_revision,
        )
        prototype_valid = self._prototype.validate_state(state.prototype)
        # A missing optional feedback sidecar may leave one exact historical
        # memory decision pending while Prototype base control keeps moving.
        # The controller integrity-binds that pending owner internally; forcing
        # it to equal Prototype's newest cache would freeze ordinary learning.
        upstream_mask_valid = self._validate_upstream_mask(
            state.upstream_mask,
            state.prototype,
        )
        dispatch_owner_valid = self._validate_dispatch_owner(
            state.dispatch_owner,
            state.prototype,
        )
        return (
            controller_valid
            & prototype_valid
            & upstream_mask_valid
            & dispatch_owner_valid
        )

    def _normalize_decision_input(
        self,
        value: PrototypeConsolidatedMemoryDecisionInput | None,
    ) -> tuple[PrototypeConsolidatedMemoryDecisionInput, Array]:
        if value is None:
            return (
                PrototypeConsolidatedMemoryDecisionInput(
                    available=jnp.asarray(False, dtype=jnp.bool_),
                    prototype_decision_id=jnp.zeros(
                        (_DECISION_WORDS,), dtype=jnp.uint32
                    ),
                    request=_blank_request(),
                    hard_safety_action_mask=jnp.ones(
                        (self._config.controller.policy.n_actions,),
                        dtype=jnp.bool_,
                    ),
                ),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if type(value) is not PrototypeConsolidatedMemoryDecisionInput:
            raise TypeError(
                "decision_input must be a PrototypeConsolidatedMemoryDecisionInput"
            )
        return value, jnp.asarray(True, dtype=jnp.bool_)

    def _normalize_feedback_input(
        self,
        value: PrototypeConsolidatedMemoryFeedbackInput | None,
    ) -> tuple[PrototypeConsolidatedMemoryFeedbackInput, Array]:
        if value is None:
            return (
                PrototypeConsolidatedMemoryFeedbackInput(
                    available=jnp.asarray(False, dtype=jnp.bool_),
                    prototype_decision_id=jnp.zeros(
                        (_DECISION_WORDS,), dtype=jnp.uint32
                    ),
                    feedback_event_id=jnp.zeros(
                        (_DECISION_WORDS,), dtype=jnp.uint32
                    ),
                    base_action=jnp.asarray(-1, dtype=jnp.int32),
                    effective_action=jnp.asarray(-1, dtype=jnp.int32),
                    request=_blank_request(),
                    succeeded=jnp.asarray(False, dtype=jnp.bool_),
                    outcome=jnp.zeros(
                        (self._config.controller.policy.outcome_dim,),
                        dtype=jnp.float32,
                    ),
                    confidence=jnp.asarray(0.0, dtype=jnp.float32),
                    evidence=jnp.asarray(0.0, dtype=jnp.float32),
                ),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if type(value) is not PrototypeConsolidatedMemoryFeedbackInput:
            raise TypeError(
                "feedback_input must be a PrototypeConsolidatedMemoryFeedbackInput"
            )
        return value, jnp.asarray(True, dtype=jnp.bool_)

    def _feedback(
        self,
        state: PrototypeConsolidatedMemoryState,
        transition: PrototypeTransition,
        feedback_input: PrototypeConsolidatedMemoryFeedbackInput,
        *,
        feedback_supplied: Array,
        composed_valid: Array,
        transition_valid: Array,
    ) -> tuple[ConsolidatedProceduralMemoryFeedbackResult, Array, Array]:
        decision_matches = jnp.array_equal(
            feedback_input.prototype_decision_id,
            transition.decision_id,
        )
        action_matches = feedback_input.effective_action == transition.action
        gate = (
            feedback_supplied
            & feedback_input.available
            & composed_valid
            & transition_valid
            & decision_matches
            & action_matches
        )
        result = self._controller.feedback(
            state.controller,
            decision_id=jnp.where(
                gate,
                transition.decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ),
            feedback_event_id=jnp.where(
                gate,
                feedback_input.feedback_event_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ),
            base_action=jnp.where(gate, feedback_input.base_action, -1).astype(
                jnp.int32
            ),
            effective_action=jnp.where(gate, transition.action, -1).astype(
                jnp.int32
            ),
            request=feedback_input.request,
            succeeded=feedback_input.succeeded,
            outcome=feedback_input.outcome,
            confidence=feedback_input.confidence,
            evidence=feedback_input.evidence,
        )
        return result, decision_matches, action_matches

    def attempt_feedback(
        self,
        state: PrototypeConsolidatedMemoryState,
        transition: PrototypeTransition,
        *,
        feedback_input: PrototypeConsolidatedMemoryFeedbackInput | None = None,
    ) -> PrototypeConsolidatedMemoryFeedbackAttempt:
        """Assess and form—but do not independently commit—prior feedback.

        The candidate is computed from the exact pre-Prototype state. A caller
        composing another shared-controller operation may use ``attempted`` as
        a provisional state, but must roll it back unless the complete outer
        Prototype transaction commits. This method does not mutate its inputs.
        """

        composed_valid = self.validate_state(state)
        normalized_feedback, feedback_supplied = self._normalize_feedback_input(
            feedback_input
        )
        transition_diagnostics = self._prototype.assess_transition(
            state.prototype,
            transition,
        )
        prior_upstream_mask_available = state.upstream_mask.available
        prior_upstream_mask_decision_matches = jnp.array_equal(
            state.upstream_mask.prototype_decision_id,
            transition.decision_id,
        )
        transition_action_index_valid = (
            (transition.action >= 0)
            & (transition.action < self._config.controller.policy.n_actions)
        )
        safe_transition_action = jnp.clip(
            transition.action,
            0,
            self._config.controller.policy.n_actions - 1,
        )
        realized_action_allowed_by_prior_upstream_mask = (
            (~prior_upstream_mask_available)
            | (
                prior_upstream_mask_decision_matches
                & transition_action_index_valid
                & state.upstream_mask.hard_safety_action_mask[
                    safe_transition_action
                ]
            )
        )
        transition_valid_before_feedback = (
            transition_diagnostics.valid
            & realized_action_allowed_by_prior_upstream_mask
        )
        attempted, decision_matches, action_matches = self._feedback(
            state,
            transition,
            normalized_feedback,
            feedback_supplied=feedback_supplied,
            composed_valid=composed_valid,
            transition_valid=transition_valid_before_feedback,
        )
        rejected, _, _ = self._feedback(
            state,
            transition,
            normalized_feedback,
            feedback_supplied=feedback_supplied,
            composed_valid=jnp.asarray(False, dtype=jnp.bool_),
            transition_valid=jnp.asarray(False, dtype=jnp.bool_),
        )
        return PrototypeConsolidatedMemoryFeedbackAttempt(
            attempted=attempted,
            rejected=rejected,
            normalized_input=normalized_feedback,
            input_supplied=feedback_supplied,
            decision_matches=decision_matches,
            action_matches=action_matches,
            composed_state_valid=composed_valid,
            transition=transition_diagnostics,
            transition_valid_before_feedback=transition_valid_before_feedback,
            prior_upstream_mask_available=prior_upstream_mask_available,
            prior_upstream_mask_decision_matches=(
                prior_upstream_mask_decision_matches
            ),
            realized_action_allowed_by_prior_upstream_mask=(
                realized_action_allowed_by_prior_upstream_mask
            ),
        )

    def _compose_next_dispatch(
        self,
        prototype_state: PrototypeAgentState,
        controller_state: ConsolidatedProceduralMemoryControllerState,
        upstream_mask_state: PrototypeConsolidatedMemoryUpstreamMaskState,
        dispatch_owner_state: PrototypeConsolidatedMemoryDispatchOwnerState,
        decision_input: PrototypeConsolidatedMemoryDecisionInput,
        *,
        decision_supplied: Array,
        composed_valid_before: Array,
        next_dispatch_allowed: Array,
        transition_valid_before_feedback: Array,
        feedback_supplied: Array,
        feedback_input: PrototypeConsolidatedMemoryFeedbackInput,
        feedback_decision_matches: Array,
        feedback_action_matches: Array,
        feedback_result: ConsolidatedProceduralMemoryFeedbackResult,
        prior_upstream_mask_available: Array,
        prior_upstream_mask_decision_matches: Array,
        realized_action_allowed_by_prior_upstream_mask: Array,
        prototype_learning_retained: Array,
        experiential_mask_applicable: Array,
        experiential_mask: Array,
        partner_mask_applicable: Array,
        partner_mask: Array,
    ) -> tuple[
        PrototypeConsolidatedMemoryState,
        Array,
        ConsolidatedProceduralMemoryDecisionResult,
        PrototypeCachedPrimitiveActionReplacement,
        PrototypeConsolidatedMemoryDiagnostics,
    ]:
        n_actions = self._config.controller.policy.n_actions
        ones = jnp.ones((n_actions,), dtype=jnp.bool_)
        decision_id_matches = jnp.array_equal(
            decision_input.prototype_decision_id,
            prototype_state.current_decision_id,
        )
        decision_gate = (
            decision_supplied
            & decision_input.available
            & decision_id_matches
            & next_dispatch_allowed
            & prototype_state.started
        )
        upstream_mask_decision_matches = jnp.array_equal(
            upstream_mask_state.prototype_decision_id,
            prototype_state.current_decision_id,
        )
        upstream_mask_applicable = (
            upstream_mask_state.available
            & upstream_mask_decision_matches
            & next_dispatch_allowed
            & prototype_state.started
        )
        upstream_mask = jnp.where(
            upstream_mask_applicable,
            upstream_mask_state.hard_safety_action_mask,
            ones,
        )
        current_action_index_valid = (
            (prototype_state.current_action >= 0)
            & (prototype_state.current_action < n_actions)
        )
        safe_current_action = jnp.clip(
            prototype_state.current_action,
            0,
            n_actions - 1,
        )
        cached_action_allowed_by_upstream_mask = (
            (~upstream_mask_applicable)
            | (
                current_action_index_valid
                & upstream_mask[safe_current_action]
            )
        )
        caller_mask = jnp.where(
            decision_gate,
            decision_input.hard_safety_action_mask,
            ones,
        )
        final_mask = caller_mask & upstream_mask
        score_mass = jax.nn.one_hot(
            jnp.clip(prototype_state.current_action, 0, n_actions - 1),
            n_actions,
            dtype=jnp.float32,
        )
        memory_decision = self._controller.decide(
            controller_state,
            decision_id=jnp.where(
                decision_gate,
                prototype_state.current_decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ),
            base_categorical_score_mass=score_mass,
            base_action=prototype_state.current_action,
            base_action_available=decision_gate,
            hard_safety_mask=final_mask,
            request=decision_input.request,
        )
        proposed_action = jnp.where(
            memory_decision.action_available,
            memory_decision.action,
            prototype_state.current_action,
        ).astype(jnp.int32)
        dispatch_replacement = self._prototype.replace_cached_primitive_action(
            prototype_state,
            decision_id=prototype_state.current_decision_id,
            decision_observation=prototype_state.current_representation,
            proposed_action=proposed_action,
            safety_action_mask=jnp.where(decision_gate, final_mask, ones),
        )
        dispatch_committed = decision_gate & dispatch_replacement.committed
        next_prototype = cast(
            PrototypeAgentState,
            _tree_select(
                dispatch_committed,
                dispatch_replacement.state,
                prototype_state,
            ),
        )
        keep_memory_decision = dispatch_committed & memory_decision.action_available
        next_controller = cast(
            ConsolidatedProceduralMemoryControllerState,
            _tree_select(
                keep_memory_decision,
                memory_decision.state,
                controller_state,
            ),
        )
        upstream_mask_consumed = dispatch_committed & upstream_mask_applicable
        next_upstream_mask = cast(
            PrototypeConsolidatedMemoryUpstreamMaskState,
            _tree_select(
                upstream_mask_consumed,
                self._blank_upstream_mask(),
                upstream_mask_state,
            ),
        )
        base_path_available = (
            prototype_state.started
            & (~decision_gate)
            & cached_action_allowed_by_upstream_mask
        )
        candidate_action_available_before_validation = (
            base_path_available | dispatch_committed
        )
        retained_dispatch_owner = cast(
            PrototypeConsolidatedMemoryDispatchOwnerState,
            _tree_select(
                prototype_learning_retained,
                self._blank_dispatch_owner(),
                dispatch_owner_state,
            ),
        )
        retained_owner_matches = (
            retained_dispatch_owner.available
            & jnp.array_equal(
                retained_dispatch_owner.prototype_decision_id,
                next_prototype.current_decision_id,
            )
            & (
                retained_dispatch_owner.selected_action
                == next_prototype.current_action
            )
        )
        bind_dispatch_owner = candidate_action_available_before_validation & (
            dispatch_committed | (~retained_owner_matches)
        )
        bound_dispatch_owner = self._dispatch_owner_record(
            available=candidate_action_available_before_validation,
            prototype_decision_id=next_prototype.current_decision_id,
            selected_action=next_prototype.current_action,
            hard_safety_action_mask=final_mask,
        )
        next_dispatch_owner = cast(
            PrototypeConsolidatedMemoryDispatchOwnerState,
            _tree_select(
                bind_dispatch_owner,
                bound_dispatch_owner,
                retained_dispatch_owner,
            ),
        )
        candidate_state = PrototypeConsolidatedMemoryState(
            prototype=next_prototype,
            controller=next_controller,
            upstream_mask=next_upstream_mask,
            dispatch_owner=next_dispatch_owner,
        )
        composed_valid_after = self.validate_state(candidate_state)
        candidate_action_available = (
            candidate_action_available_before_validation
        ) & composed_valid_after
        transaction_committed = composed_valid_before & composed_valid_after
        final_state = cast(
            PrototypeConsolidatedMemoryState,
            _tree_select(
                transaction_committed,
                candidate_state,
                PrototypeConsolidatedMemoryState(
                    prototype=prototype_state,
                    controller=controller_state,
                    upstream_mask=upstream_mask_state,
                    dispatch_owner=dispatch_owner_state,
                ),
            ),
        )
        action_available = transaction_committed & candidate_action_available
        action = jnp.where(
            action_available,
            final_state.prototype.current_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        expected_intersection = caller_mask & upstream_mask
        diagnostics = PrototypeConsolidatedMemoryDiagnostics(
            composed_state_valid_before=composed_valid_before,
            next_dispatch_allowed=next_dispatch_allowed,
            transition_valid_before_feedback=transition_valid_before_feedback,
            feedback_input_supplied=feedback_supplied,
            feedback_input_available=feedback_input.available,
            feedback_realized_decision_matches=feedback_decision_matches,
            feedback_realized_action_matches=feedback_action_matches,
            feedback_settled_before_prototype_learning=(
                feedback_result.diagnostics.pending_cleared
            ),
            prior_upstream_mask_available=prior_upstream_mask_available,
            prior_upstream_mask_decision_matches=(
                prior_upstream_mask_decision_matches
            ),
            realized_action_allowed_by_prior_upstream_mask=(
                realized_action_allowed_by_prior_upstream_mask
            ),
            prototype_learning_retained=prototype_learning_retained,
            decision_input_supplied=decision_supplied,
            decision_input_available=decision_input.available,
            decision_prototype_id_matches=decision_id_matches,
            experiential_memory_precedes_consolidated=jnp.asarray(
                True, dtype=jnp.bool_
            ),
            partner_fusion_precedes_consolidated=jnp.asarray(
                True, dtype=jnp.bool_
            ),
            experiential_mask_applicable=experiential_mask_applicable,
            partner_mask_applicable=partner_mask_applicable,
            upstream_mask_available=upstream_mask_state.available,
            upstream_mask_decision_matches=upstream_mask_decision_matches,
            cached_action_allowed_by_upstream_mask=(
                cached_action_allowed_by_upstream_mask
            ),
            upstream_mask_consumed=upstream_mask_consumed,
            caller_hard_safety_action_mask=caller_mask,
            experiential_hard_safety_action_mask=experiential_mask,
            partner_hard_safety_action_mask=partner_mask,
            upstream_hard_safety_action_mask=upstream_mask,
            final_hard_safety_action_mask=final_mask,
            final_mask_is_exact_intersection=jnp.array_equal(
                final_mask, expected_intersection
            ),
            query_after_prototype_learning=(
                prototype_learning_retained
                & memory_decision.diagnostics.query_attempted
            ),
            dispatch_replacement_committed=dispatch_committed,
            composed_state_valid_after=composed_valid_after,
            transaction_committed=transaction_committed,
            action_available=action_available,
        )
        return (
            final_state,
            action,
            memory_decision,
            dispatch_replacement,
            diagnostics,
        )

    def start(
        self,
        state: PrototypeConsolidatedMemoryState,
        initial_observation: Array,
        *,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
    ) -> PrototypeConsolidatedMemoryStartResult:
        composed_valid = self.validate_state(state)
        normalized_decision, decision_supplied = self._normalize_decision_input(
            decision_input
        )
        prototype_state = self._prototype.start(
            state.prototype,
            initial_observation,
        )
        blank_feedback, feedback_supplied = self._normalize_feedback_input(None)
        feedback_result, feedback_decision_matches, feedback_action_matches = (
            self._feedback(
                state,
                PrototypeTransition(
                    observation=prototype_state.current_raw_observation,
                    action=jnp.asarray(-1, dtype=jnp.int32),
                    decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
                    reward=jnp.asarray(0.0, dtype=jnp.float32),
                    discount=jnp.asarray(1.0, dtype=jnp.float32),
                    terminated=jnp.asarray(False, dtype=jnp.bool_),
                    truncated=jnp.asarray(False, dtype=jnp.bool_),
                    next_observation=prototype_state.current_raw_observation,
                    next_decision_observation=(
                        prototype_state.current_raw_observation
                    ),
                ),
                blank_feedback,
                feedback_supplied=feedback_supplied,
                composed_valid=composed_valid,
                transition_valid=jnp.asarray(False, dtype=jnp.bool_),
            )
        )
        (
            candidate_state,
            action,
            memory_decision,
            dispatch_replacement,
            diagnostics,
        ) = self._compose_next_dispatch(
            prototype_state,
            feedback_result.state,
            state.upstream_mask,
            state.dispatch_owner,
            normalized_decision,
            decision_supplied=decision_supplied,
            composed_valid_before=composed_valid,
            next_dispatch_allowed=composed_valid,
            transition_valid_before_feedback=jnp.asarray(False, dtype=jnp.bool_),
            feedback_supplied=feedback_supplied,
            feedback_input=blank_feedback,
            feedback_decision_matches=feedback_decision_matches,
            feedback_action_matches=feedback_action_matches,
            feedback_result=feedback_result,
            prior_upstream_mask_available=jnp.asarray(
                False, dtype=jnp.bool_
            ),
            prior_upstream_mask_decision_matches=jnp.asarray(
                False, dtype=jnp.bool_
            ),
            realized_action_allowed_by_prior_upstream_mask=jnp.asarray(
                True, dtype=jnp.bool_
            ),
            prototype_learning_retained=jnp.asarray(False, dtype=jnp.bool_),
            experiential_mask_applicable=jnp.asarray(False, dtype=jnp.bool_),
            experiential_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
            partner_mask_applicable=jnp.asarray(False, dtype=jnp.bool_),
            partner_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
        )
        start_committed = composed_valid & prototype_state.started
        final_state = cast(
            PrototypeConsolidatedMemoryState,
            _tree_select(start_committed, candidate_state, state),
        )
        final_action = jnp.where(start_committed, action, -1).astype(jnp.int32)
        return PrototypeConsolidatedMemoryStartResult(
            state=final_state,
            action=final_action,
            memory_decision=memory_decision,
            dispatch_replacement=dispatch_replacement,
            diagnostics=dataclasses.replace(
                diagnostics,
                transaction_committed=(
                    diagnostics.transaction_committed & start_committed
                ),
                action_available=diagnostics.action_available & start_committed,
            ),
        )

    def decide(
        self,
        state: PrototypeConsolidatedMemoryState,
        *,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
    ) -> PrototypeConsolidatedMemoryStartResult:
        """Compose memory into an already armed, not-yet-dispatched decision."""

        composed_valid = self.validate_state(state)
        normalized_decision, decision_supplied = self._normalize_decision_input(
            decision_input
        )
        blank_feedback, feedback_supplied = self._normalize_feedback_input(None)
        inert_transition = PrototypeTransition(
            observation=state.prototype.current_raw_observation,
            action=state.prototype.current_action,
            decision_id=state.prototype.current_decision_id,
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=state.prototype.current_raw_observation,
            next_decision_observation=state.prototype.current_raw_observation,
        )
        feedback_result, feedback_decision_matches, feedback_action_matches = (
            self._feedback(
                state,
                inert_transition,
                blank_feedback,
                feedback_supplied=feedback_supplied,
                composed_valid=composed_valid,
                transition_valid=jnp.asarray(False, dtype=jnp.bool_),
            )
        )
        (
            final_state,
            action,
            memory_decision,
            dispatch_replacement,
            diagnostics,
        ) = self._compose_next_dispatch(
            state.prototype,
            feedback_result.state,
            state.upstream_mask,
            state.dispatch_owner,
            normalized_decision,
            decision_supplied=decision_supplied,
            composed_valid_before=composed_valid,
            next_dispatch_allowed=composed_valid,
            transition_valid_before_feedback=jnp.asarray(False, dtype=jnp.bool_),
            feedback_supplied=feedback_supplied,
            feedback_input=blank_feedback,
            feedback_decision_matches=feedback_decision_matches,
            feedback_action_matches=feedback_action_matches,
            feedback_result=feedback_result,
            prior_upstream_mask_available=state.upstream_mask.available,
            prior_upstream_mask_decision_matches=jnp.array_equal(
                state.upstream_mask.prototype_decision_id,
                state.prototype.current_decision_id,
            ),
            realized_action_allowed_by_prior_upstream_mask=jnp.asarray(
                True, dtype=jnp.bool_
            ),
            prototype_learning_retained=jnp.asarray(False, dtype=jnp.bool_),
            experiential_mask_applicable=jnp.asarray(False, dtype=jnp.bool_),
            experiential_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
            partner_mask_applicable=jnp.asarray(False, dtype=jnp.bool_),
            partner_mask=jnp.ones(
                (self._config.controller.policy.n_actions,), dtype=jnp.bool_
            ),
        )
        return PrototypeConsolidatedMemoryStartResult(
            state=final_state,
            action=action,
            memory_decision=memory_decision,
            dispatch_replacement=dispatch_replacement,
            diagnostics=diagnostics,
        )

    def update_transition(
        self,
        state: PrototypeConsolidatedMemoryState,
        transition: PrototypeTransition,
        *,
        decision_input: PrototypeConsolidatedMemoryDecisionInput | None = None,
        feedback_input: PrototypeConsolidatedMemoryFeedbackInput | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
    ) -> PrototypeConsolidatedMemoryUpdateResult:
        """Settle memory, learn the realized action, then bind the next action."""

        feedback_attempt = self.attempt_feedback(
            state,
            transition,
            feedback_input=feedback_input,
        )
        composed_valid = feedback_attempt.composed_state_valid
        normalized_decision, decision_supplied = self._normalize_decision_input(
            decision_input
        )
        normalized_feedback = feedback_attempt.normalized_input
        feedback_supplied = feedback_attempt.input_supplied
        transition_diagnostics = feedback_attempt.transition
        prior_upstream_mask_available = (
            feedback_attempt.prior_upstream_mask_available
        )
        prior_upstream_mask_decision_matches = (
            feedback_attempt.prior_upstream_mask_decision_matches
        )
        realized_action_allowed_by_prior_upstream_mask = (
            feedback_attempt.realized_action_allowed_by_prior_upstream_mask
        )
        transition_valid_before_feedback = (
            feedback_attempt.transition_valid_before_feedback
        )
        attempted_feedback = feedback_attempt.attempted
        rejected_feedback = feedback_attempt.rejected
        feedback_decision_matches = feedback_attempt.decision_matches
        feedback_action_matches = feedback_attempt.action_matches
        prototype_result = self._prototype.update_transition(
            state.prototype,
            transition,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
        )
        result_diagnostics = prototype_result.transition_diagnostics
        exact_outer_advance = (
            jnp.array_equal(
                prototype_result.state.step_words,
                transition_diagnostics.proposed_step_words,
            )
            & jnp.array_equal(
                prototype_result.state.observation_event_words,
                transition_diagnostics.proposed_observation_event_words,
            )
            & (~jnp.array_equal(
                prototype_result.state.step_words,
                state.prototype.step_words,
            ))
        )
        expected_next_decision_id = _increment_prototype_decision_id(
            state.prototype.current_decision_id
        )
        next_cache_committed = jnp.where(
            prototype_result.state.started,
            jnp.array_equal(
                prototype_result.state.current_decision_id,
                expected_next_decision_id,
            )
            & (prototype_result.action == prototype_result.state.current_action),
            (prototype_result.action == -1)
            & (~prototype_result.state.started),
        )
        prototype_learning_retained = (
            composed_valid
            & transition_valid_before_feedback
            & result_diagnostics.valid
            & result_diagnostics.post_update_checked
            & result_diagnostics.post_update_finite
            & result_diagnostics.post_update_consistent
            & exact_outer_advance
            & next_cache_committed
            & self._prototype.validate_state(prototype_result.state)
        )
        feedback_result = cast(
            ConsolidatedProceduralMemoryFeedbackResult,
            _tree_select(
                prototype_learning_retained,
                attempted_feedback,
                rejected_feedback,
            ),
        )
        learned_prototype_state = cast(
            PrototypeAgentState,
            _tree_select(
                prototype_learning_retained,
                prototype_result.state,
                state.prototype,
            ),
        )
        learned_controller_state = cast(
            ConsolidatedProceduralMemoryControllerState,
            _tree_select(
                prototype_learning_retained,
                feedback_result.state,
                state.controller,
            ),
        )
        ones = jnp.ones(
            (self._config.controller.policy.n_actions,), dtype=jnp.bool_
        )
        experiential_mask_applicable = jnp.asarray(False, dtype=jnp.bool_)
        experiential_mask = ones
        if experiential_memory_input is not None:
            experiential_diagnostics = (
                prototype_result.experiential_memory_diagnostics
            )
            if experiential_diagnostics is None:
                raise RuntimeError(
                    "configured experiential memory requires diagnostics"
                )
            experiential_mask_applicable = (
                prototype_learning_retained
                & experiential_diagnostics.transaction_required
                & experiential_memory_input.available
                & jnp.array_equal(
                    experiential_memory_input.next_prototype_decision_id,
                    learned_prototype_state.current_decision_id,
                )
            )
            experiential_mask = jnp.where(
                experiential_mask_applicable,
                experiential_memory_input.next_action_safety_mask,
                ones,
            )
        partner_mask_applicable = jnp.asarray(False, dtype=jnp.bool_)
        partner_mask = ones
        if partner_policy_fusion_input is not None:
            partner_diagnostics = prototype_result.partner_policy_fusion_diagnostics
            if partner_diagnostics is None:
                raise RuntimeError("configured partner fusion requires diagnostics")
            partner_mask_applicable = (
                prototype_learning_retained
                & partner_diagnostics.transaction_applied
                & partner_policy_fusion_input.available
                & jnp.array_equal(
                    partner_policy_fusion_input.prototype_decision_id,
                    learned_prototype_state.current_decision_id,
                )
            )
            partner_mask = jnp.where(
                partner_mask_applicable,
                partner_policy_fusion_input.safety_action_mask,
                ones,
            )
        next_upstream_mask_available = (
            learned_prototype_state.started
            & (experiential_mask_applicable | partner_mask_applicable)
        )
        next_upstream_mask = self._upstream_mask_record(
            available=next_upstream_mask_available,
            prototype_decision_id=learned_prototype_state.current_decision_id,
            hard_safety_action_mask=experiential_mask & partner_mask,
        )
        (
            candidate_state,
            action,
            memory_decision,
            dispatch_replacement,
            diagnostics,
        ) = self._compose_next_dispatch(
            learned_prototype_state,
            learned_controller_state,
            next_upstream_mask,
            state.dispatch_owner,
            normalized_decision,
            decision_supplied=decision_supplied,
            composed_valid_before=composed_valid,
            next_dispatch_allowed=prototype_learning_retained,
            transition_valid_before_feedback=transition_valid_before_feedback,
            feedback_supplied=feedback_supplied,
            feedback_input=normalized_feedback,
            feedback_decision_matches=feedback_decision_matches,
            feedback_action_matches=feedback_action_matches,
            feedback_result=feedback_result,
            prior_upstream_mask_available=prior_upstream_mask_available,
            prior_upstream_mask_decision_matches=(
                prior_upstream_mask_decision_matches
            ),
            realized_action_allowed_by_prior_upstream_mask=(
                realized_action_allowed_by_prior_upstream_mask
            ),
            prototype_learning_retained=prototype_learning_retained,
            experiential_mask_applicable=experiential_mask_applicable,
            experiential_mask=experiential_mask,
            partner_mask_applicable=partner_mask_applicable,
            partner_mask=partner_mask,
        )
        outer_committed = (
            prototype_learning_retained
            & diagnostics.transaction_committed
            & diagnostics.composed_state_valid_after
        )
        final_state = cast(
            PrototypeConsolidatedMemoryState,
            _tree_select(outer_committed, candidate_state, state),
        )
        final_action = jnp.where(
            outer_committed & diagnostics.action_available,
            action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        final_prototype_result = dataclasses.replace(
            prototype_result,
            state=final_state.prototype,
            action=final_action,
        )
        return PrototypeConsolidatedMemoryUpdateResult(
            state=final_state,
            action=final_action,
            prototype=final_prototype_result,
            memory_feedback=feedback_result,
            memory_decision=memory_decision,
            dispatch_replacement=dispatch_replacement,
            diagnostics=dataclasses.replace(
                diagnostics,
                transaction_committed=outer_committed,
                action_available=(
                    outer_committed & diagnostics.action_available
                ),
            ),
        )

    def settle_dispatch(
        self,
        state: PrototypeConsolidatedMemoryState,
        settlement: PrototypeConsolidatedMemoryDispatchSettlementInput,
    ) -> PrototypeConsolidatedMemoryDispatchSettlementResult:
        """Atomically settle one exact post-envelope dispatch outcome.

        An accepted unchanged action is an exact state no-op. A changed safe
        action rebinds Prototype's cached credit owner, cancels only matching
        procedural and partner recommendation owners, and rebinds the receipt
        to the admitted action and the same exact mask in one transaction.
        No-action, stale, unsafe,
        corrupt, or partially satisfiable inputs leave the complete state
        unchanged. This method performs no learning, evidence write, RNG use,
        or physical dispatch.
        """

        if type(settlement) is not PrototypeConsolidatedMemoryDispatchSettlementInput:
            raise TypeError(
                "settlement must be a "
                "PrototypeConsolidatedMemoryDispatchSettlementInput"
            )
        contracts = (
            (settlement.action_available, (), jnp.bool_, "action_available"),
            (
                settlement.prototype_decision_id,
                (_DECISION_WORDS,),
                jnp.uint32,
                "prototype_decision_id",
            ),
            (settlement.selected_action, (), jnp.int32, "selected_action"),
            (settlement.executed_action, (), jnp.int32, "executed_action"),
        )
        for value, shape, dtype, name in contracts:
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"settlement {name} must expose array metadata")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"settlement {name} must have shape {shape}, "
                    f"got {tuple(value.shape)}"
                )
            if jnp.dtype(value.dtype) != jnp.dtype(dtype):
                raise TypeError(
                    f"settlement {name} must have dtype {jnp.dtype(dtype)}, "
                    f"got {value.dtype}"
                )

        n_actions = self._config.controller.policy.n_actions
        owner = state.dispatch_owner
        state_valid = self.validate_state(state)
        decision_matches = owner.available & jnp.array_equal(
            settlement.prototype_decision_id,
            owner.prototype_decision_id,
        )
        selected_matches_owner = owner.available & (
            settlement.selected_action == owner.selected_action
        )
        selected_matches_cache = state.prototype.started & (
            settlement.selected_action == state.prototype.current_action
        )
        executed_index_valid = (
            (settlement.executed_action >= 0)
            & (settlement.executed_action < n_actions)
        )
        safe_executed_action = jnp.clip(
            settlement.executed_action,
            0,
            n_actions - 1,
        )
        no_action = ~settlement.action_available
        executed_contract_valid = jnp.where(
            settlement.action_available,
            executed_index_valid,
            settlement.executed_action == -1,
        )
        executed_allowed = jnp.where(
            settlement.action_available,
            executed_index_valid
            & owner.hard_safety_action_mask[safe_executed_action],
            jnp.asarray(True, dtype=jnp.bool_),
        )
        receipt_valid = (
            state_valid
            & owner.available
            & decision_matches
            & selected_matches_owner
            & selected_matches_cache
            & executed_contract_valid
            & executed_allowed
        )
        action_changed = settlement.action_available & (
            settlement.executed_action != settlement.selected_action
        )

        replacement = self._prototype.replace_cached_primitive_action(
            state.prototype,
            decision_id=settlement.prototype_decision_id,
            decision_observation=state.prototype.current_representation,
            proposed_action=jnp.where(
                settlement.action_available,
                safe_executed_action,
                settlement.selected_action,
            ).astype(jnp.int32),
            safety_action_mask=owner.hard_safety_action_mask,
        )
        replacement_satisfied = (~action_changed) | replacement.committed

        procedural_owner_current = state.controller.pending & jnp.array_equal(
            state.controller.pending_decision_id,
            settlement.prototype_decision_id,
        )
        procedural_cancellation_required = (
            action_changed & procedural_owner_current
        )
        procedural_cancellation = self._controller.cancel_pending_dispatch(
            state.controller,
            cancellation_requested=procedural_cancellation_required,
            decision_id=settlement.prototype_decision_id,
            effective_action=settlement.selected_action,
        )

        candidate_prototype = cast(
            PrototypeAgentState,
            _tree_select(action_changed, replacement.state, state.prototype),
        )
        partner_owner_current = jnp.asarray(False, dtype=jnp.bool_)
        partner_owner_consistent_for_change = jnp.asarray(True, dtype=jnp.bool_)
        partner_armed_identity_current = jnp.asarray(True, dtype=jnp.bool_)
        partner_armed_action_source_bound = jnp.asarray(True, dtype=jnp.bool_)
        partner_cancellation_required = jnp.asarray(False, dtype=jnp.bool_)
        partner_cancellation_applied = jnp.asarray(False, dtype=jnp.bool_)
        partner_transaction_satisfied = jnp.asarray(True, dtype=jnp.bool_)
        partner_cancellation: PartnerFusionFeedbackCancellationResult | None = None
        fusion = self._prototype.partner_policy_fusion
        if fusion is not None:
            outer_interaction = candidate_prototype.ia_state
            if self._config.prototype.experiential_memory is not None:
                if type(outer_interaction) is not PrototypeMemoryInteractionState:
                    raise TypeError(
                        "configured experiential memory requires its interaction wrapper"
                    )
                memory_interaction = outer_interaction
                interaction = memory_interaction.interaction_state
            else:
                memory_interaction = None
                interaction = outer_interaction
            if type(interaction) is not PrototypeInteractionState:
                raise TypeError(
                    "configured partner fusion requires its interaction wrapper"
                )
            exact_interaction = interaction
            partner_owner_current = (
                exact_interaction.feedback_prototype_decision_id_available
                & jnp.array_equal(
                    exact_interaction.feedback_prototype_decision_id,
                    settlement.prototype_decision_id,
                )
            )
            partner_owner_consistent_for_change = (
                (~action_changed)
                | (~exact_interaction.feedback_prototype_decision_id_available)
                | partner_owner_current
            )
            partner_cancellation_required = action_changed & partner_owner_current
            partner_state = exact_interaction.partner_policy_fusion_state
            partner_armed_identity_current = (
                (~partner_cancellation_required)
                | (
                    partner_state.feedback_armed
                    & jnp.array_equal(
                        partner_state.armed_decision_words,
                        state.prototype.step_words,
                    )
                    & jnp.array_equal(
                        partner_state.armed_event_words,
                        state.prototype.observation_event_words,
                    )
                )
            )
            expected_partner_action = jnp.where(
                procedural_owner_current,
                state.controller.pending_base_action,
                settlement.selected_action,
            ).astype(jnp.int32)
            partner_armed_action_source_bound = (
                (~partner_cancellation_required)
                | (partner_state.armed_action == expected_partner_action)
            )
            partner_cancellation = fusion.cancel_pending_feedback(
                partner_state,
                cancellation_requested=partner_cancellation_required,
                decision_words=partner_state.armed_decision_words,
                event_words=partner_state.armed_event_words,
                effective_action=partner_state.armed_action,
                partner_id=partner_state.armed_partner_id,
            )
            partner_cancellation_applied = (
                partner_cancellation.cancellation_applied
            )
            partner_transaction_satisfied = (
                partner_cancellation.transaction_satisfied
                & partner_owner_consistent_for_change
                & partner_armed_identity_current
                & partner_armed_action_source_bound
            )
            next_interaction = dataclasses.replace(
                exact_interaction,
                partner_policy_fusion_state=partner_cancellation.state,
                feedback_prototype_decision_id=jnp.where(
                    partner_cancellation_applied,
                    jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
                    exact_interaction.feedback_prototype_decision_id,
                ),
                feedback_prototype_decision_id_available=(
                    exact_interaction.feedback_prototype_decision_id_available
                    & (~partner_cancellation_applied)
                ),
            )
            next_ia_state: Any
            if memory_interaction is not None:
                next_ia_state = dataclasses.replace(
                    memory_interaction,
                    interaction_state=next_interaction
                )
            else:
                next_ia_state = next_interaction
            candidate_prototype = dataclasses.replace(
                candidate_prototype,
                ia_state=next_ia_state,
            )

        candidate_state = PrototypeConsolidatedMemoryState(
            prototype=candidate_prototype,
            controller=procedural_cancellation.state,
            upstream_mask=state.upstream_mask,
            dispatch_owner=self._dispatch_owner_record(
                available=jnp.asarray(True, dtype=jnp.bool_),
                prototype_decision_id=settlement.prototype_decision_id,
                selected_action=settlement.executed_action,
                hard_safety_action_mask=owner.hard_safety_action_mask,
            ),
        )
        candidate_valid = self.validate_state(candidate_state)
        procedural_transaction_satisfied = (
            procedural_cancellation.diagnostics.transaction_satisfied
        )
        transaction_committed = (
            receipt_valid
            & replacement_satisfied
            & procedural_transaction_satisfied
            & partner_transaction_satisfied
            & jnp.where(action_changed, candidate_valid, state_valid)
        )
        state_changed = transaction_committed & action_changed
        final_state = cast(
            PrototypeConsolidatedMemoryState,
            _tree_select(state_changed, candidate_state, state),
        )
        final_action = jnp.where(
            transaction_committed & settlement.action_available,
            settlement.executed_action,
            jnp.asarray(-1, dtype=jnp.int32),
        ).astype(jnp.int32)
        return PrototypeConsolidatedMemoryDispatchSettlementResult(
            state=final_state,
            action=final_action,
            prototype_replacement=replacement,
            procedural_cancellation=procedural_cancellation,
            partner_cancellation=partner_cancellation,
            diagnostics=PrototypeConsolidatedMemoryDispatchSettlementDiagnostics(
                composed_state_valid_before=state_valid,
                action_available=settlement.action_available,
                no_action=no_action,
                dispatch_owner_available=owner.available,
                decision_identity_matches=decision_matches,
                selected_action_matches_owner=selected_matches_owner,
                selected_action_matches_cache=selected_matches_cache,
                executed_action_contract_valid=executed_contract_valid,
                executed_action_allowed_by_bound_mask=executed_allowed,
                action_changed=action_changed,
                prototype_replacement_required=action_changed,
                prototype_replacement_committed=replacement.committed,
                procedural_owner_current=procedural_owner_current,
                procedural_cancellation_required=(
                    procedural_cancellation_required
                ),
                procedural_cancellation_applied=(
                    procedural_cancellation.diagnostics.cancellation_applied
                ),
                partner_owner_current=partner_owner_current,
                partner_owner_consistent_for_change=(
                    partner_owner_consistent_for_change
                ),
                partner_armed_identity_current=partner_armed_identity_current,
                partner_armed_action_source_bound=(
                    partner_armed_action_source_bound
                ),
                partner_cancellation_required=partner_cancellation_required,
                partner_cancellation_applied=partner_cancellation_applied,
                candidate_state_valid=candidate_valid,
                transaction_committed=transaction_committed,
                state_changed=state_changed,
                learner_update_applied=jnp.asarray(False, dtype=jnp.bool_),
                memory_evidence_written=jnp.asarray(False, dtype=jnp.bool_),
                partner_learning_applied=jnp.asarray(False, dtype=jnp.bool_),
                random_generator_consumed=jnp.asarray(False, dtype=jnp.bool_),
                physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                evidence_promotion_authority=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def checkpoint_payload(
        self,
        state: PrototypeConsolidatedMemoryState,
    ) -> dict[str, object]:
        """Return a strict host-only in-memory checkpoint for both states.

        Typed PRNG keys and every other leaf are materialized for an unkeyed
        SHA-256 corruption digest. This boundary is intentionally unavailable
        inside JIT or scan and is not a MAC, signature, or authenticity claim.
        """

        if not bool(jax.device_get(self.validate_state(state))):
            raise ValueError("cannot checkpoint an invalid composed state")
        memory = state.controller.memory
        return {
            "schema": PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "prototype_state": state.prototype,
            "prototype_state_sha256": _tree_sha256(state.prototype),
            "upstream_mask_state": state.upstream_mask,
            "upstream_mask_state_sha256": _tree_sha256(state.upstream_mask),
            "dispatch_owner_state": state.dispatch_owner,
            "dispatch_owner_state_sha256": _tree_sha256(state.dispatch_owner),
            "controller": self._controller.checkpoint_payload(
                state.controller,
                source_digest=memory.source_digest,
                semantic_namespace_digest=memory.semantic_namespace_digest,
                representation_revision=memory.representation_revision,
                source_revision=memory.source_revision,
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
    ) -> PrototypeConsolidatedMemoryState:
        if type(payload) is not dict:
            raise ValueError("adapter checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {
            "schema",
            "config",
            "prototype_state",
            "prototype_state_sha256",
            "upstream_mask_state",
            "upstream_mask_state_sha256",
            "dispatch_owner_state",
            "dispatch_owner_state_sha256",
            "controller",
        }:
            raise ValueError("adapter checkpoint fields differ from schema v2")
        if raw["schema"] != PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA:
            raise ValueError("adapter checkpoint schema differs")
        if PrototypeConsolidatedMemoryConfig.from_config(raw["config"]) != self._config:
            raise ValueError("adapter checkpoint config differs")
        prototype_state = raw["prototype_state"]
        if type(prototype_state) is not PrototypeAgentState:
            raise ValueError("adapter checkpoint Prototype state type differs")
        expected_sha = jnp.asarray(raw["prototype_state_sha256"])
        if expected_sha.shape != (_DIGEST_BYTES,) or expected_sha.dtype != jnp.uint8:
            raise ValueError("adapter checkpoint Prototype SHA contract differs")
        if not bool(
            jax.device_get(
                jnp.array_equal(expected_sha, _tree_sha256(prototype_state))
            )
        ):
            raise ValueError("adapter checkpoint Prototype state SHA differs")
        upstream_mask_state = raw["upstream_mask_state"]
        if type(upstream_mask_state) is not (
            PrototypeConsolidatedMemoryUpstreamMaskState
        ):
            raise ValueError("adapter checkpoint upstream mask state type differs")
        expected_upstream_sha = jnp.asarray(
            raw["upstream_mask_state_sha256"]
        )
        if (
            expected_upstream_sha.shape != (_DIGEST_BYTES,)
            or expected_upstream_sha.dtype != jnp.uint8
        ):
            raise ValueError("adapter checkpoint upstream mask SHA contract differs")
        if not bool(
            jax.device_get(
                jnp.array_equal(
                    expected_upstream_sha,
                    _tree_sha256(upstream_mask_state),
                )
            )
        ):
            raise ValueError("adapter checkpoint upstream mask state SHA differs")
        dispatch_owner_state = raw["dispatch_owner_state"]
        if type(dispatch_owner_state) is not (
            PrototypeConsolidatedMemoryDispatchOwnerState
        ):
            raise ValueError("adapter checkpoint dispatch owner state type differs")
        expected_dispatch_owner_sha = jnp.asarray(
            raw["dispatch_owner_state_sha256"]
        )
        if (
            expected_dispatch_owner_sha.shape != (_DIGEST_BYTES,)
            or expected_dispatch_owner_sha.dtype != jnp.uint8
        ):
            raise ValueError(
                "adapter checkpoint dispatch owner SHA contract differs"
            )
        if not bool(
            jax.device_get(
                jnp.array_equal(
                    expected_dispatch_owner_sha,
                    _tree_sha256(dispatch_owner_state),
                )
            )
        ):
            raise ValueError("adapter checkpoint dispatch owner state SHA differs")
        controller_state = self._controller.restore_checkpoint(
            raw["controller"],
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
        )
        restored = PrototypeConsolidatedMemoryState(
            prototype=prototype_state,
            controller=controller_state,
            upstream_mask=upstream_mask_state,
            dispatch_owner=dispatch_owner_state,
        )
        if not bool(jax.device_get(self.validate_state(restored))):
            raise ValueError("adapter checkpoint composed state is inconsistent")
        return restored

    def rebind_reset(
        self,
        state: PrototypeConsolidatedMemoryState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
        discard_pending: bool = False,
    ) -> PrototypeConsolidatedMemoryState:
        if not bool(jax.device_get(self.validate_state(state))):
            raise ValueError("cannot reset a corrupted composed state")
        reset_controller = self._controller.rebind_reset(
            state.controller,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
            discard_pending=discard_pending,
        )
        return PrototypeConsolidatedMemoryState(
            prototype=state.prototype,
            controller=reset_controller,
            upstream_mask=state.upstream_mask,
            dispatch_owner=state.dispatch_owner,
        )


__all__ = [
    "PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER",
    "PROTOTYPE_CONSOLIDATED_MEMORY_CONFIG_SCHEMA",
    "PROTOTYPE_CONSOLIDATED_MEMORY_AUTONOMOUS_POLICY_AUTHORITY",
    "PROTOTYPE_CONSOLIDATED_MEMORY_CACHED_ACTION_REPLACEMENT_ENABLED",
    "PROTOTYPE_CONSOLIDATED_MEMORY_DISPATCH_SETTLEMENT_ENABLED",
    "PROTOTYPE_CONSOLIDATED_MEMORY_MECHANISM_STATUS",
    "PROTOTYPE_CONSOLIDATED_MEMORY_PHYSICAL_DISPATCH_AUTHORITY",
    "PROTOTYPE_CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY",
    "PrototypeConsolidatedMemoryAgent",
    "PrototypeConsolidatedMemoryConfig",
    "PrototypeConsolidatedMemoryDecisionInput",
    "PrototypeConsolidatedMemoryDispatchOwnerState",
    "PrototypeConsolidatedMemoryDispatchSettlementDiagnostics",
    "PrototypeConsolidatedMemoryDispatchSettlementInput",
    "PrototypeConsolidatedMemoryDispatchSettlementResult",
    "PrototypeConsolidatedMemoryDiagnostics",
    "PrototypeConsolidatedMemoryFeedbackAttempt",
    "PrototypeConsolidatedMemoryFeedbackInput",
    "PrototypeConsolidatedMemoryResourceBudget",
    "PrototypeConsolidatedMemoryStartResult",
    "PrototypeConsolidatedMemoryState",
    "PrototypeConsolidatedMemoryUpdateResult",
    "PrototypeConsolidatedMemoryUpstreamMaskState",
]
