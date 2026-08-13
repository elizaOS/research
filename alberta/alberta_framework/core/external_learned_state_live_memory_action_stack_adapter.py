# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,union-attr"
"""Versioned B/M/P action stack over one external-state live-memory owner.

This L0 host-only v2 sibling leaves the v1 adapter unchanged.  It separates the
base Prototype action (B), the post-memory action (M), and the final action sent
to the environment (P).  The persistent coordinator and Prototype cache always
own P.  Learned-memory feedback is identity-bound to M, while completed real
transitions and stored exemplars are required to carry P.

Preparation first settles exact prior M-bound feedback.  It then uses a v1
adapter only as a pure donor kernel from a temporary state whose pending memory
receipt has already been cleared and whose coordinator action is P.  The donor
therefore evaluates the coordinator, query, write, and memory replacement once
without adopting anything.  Its post-memory M candidate is converted to this
schema with provisional P=M.  ``bind_final_action`` installs an already-computed
planner-selected Prototype state by exact content binding; it performs no
Prototype replacement, learner update, model evaluation, or donor adoption.
Only a valid finalized preparation may receive an integrity receipt or be
atomically adopted.

The planner candidate words and all content tags are unkeyed integrity facts,
not caller authentication or proof that the caller ran a particular planner.
This mechanism has no evidence, safety, dispatch, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.experiential_memory import ExperientialMemoryEntry
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryAdapterConfig,
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryFeedback,
    ExternalLearnedStateLiveMemoryPendingBinding,
    ExternalLearnedStateLiveMemoryPreparedTransition,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorDiagnostics,
    ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition,
    ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt,
    ExternalLearnedStateRouterAuditCoordinatorPreparedTransition,
    ExternalLearnedStateRouterAuditCoordinatorResult,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryControllerState,
    LearnedExperientialMemoryFeedback,
    LearnedExperientialMemoryFeedbackDiagnostics,
    LearnedExperientialMemoryFeedbackResult,
    LearnedExperientialMemoryStepDiagnostics,
    LearnedExperientialMemoryStepResult,
)
from alberta_framework.core.options import DispatchedPrimitiveActionDecision
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeCachedPrimitiveActionReplacement,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
)

EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_CONFIG_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.config.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STATE_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.state.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_MEMORY_PREPARATION_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.memory-preparation.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_FINALIZED_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.finalized.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_RECEIPT_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.receipt.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_FINALIZED_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.started-finalized.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_RECEIPT_SCHEMA = (
    "alberta.external-learned-state-live-memory-action-stack.started-receipt.v2"
)
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_EVIDENCE_LEVEL = "L0"
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_OUTCOME_STATUS = "not_assessed"
EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_SCIENTIFIC_PROMOTION_ALLOWED = False

_DIGEST_WORDS = 8
_SCHEMA_DIGEST_BYTES = 32
_UINT32_MAX = 2**32 - 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _config_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(value))).hexdigest()


def _digest_bytes(payload: bytes) -> Array:
    digest = hashlib.sha256(payload).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, len(digest), 4)
        ),
        dtype=jnp.uint32,
    )


def _tree_digest(*values: object) -> Array:
    digest = hashlib.sha256()
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, tree = jax.tree.flatten(value)
        digest.update(repr(tree).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
                array = jnp.asarray(leaf)
                material = (
                    jr.key_data(array)
                    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
                    else array
                )
                host = np.asarray(jax.device_get(material))
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(host).tobytes())
            else:
                digest.update(type(leaf).__module__.encode("utf-8"))
                digest.update(type(leaf).__qualname__.encode("utf-8"))
                digest.update(repr(leaf).encode("utf-8"))
    return _digest_bytes(digest.digest())


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_has_metadata = hasattr(left_leaf, "shape") and hasattr(left_leaf, "dtype")
        right_has_metadata = hasattr(right_leaf, "shape") and hasattr(right_leaf, "dtype")
        if left_has_metadata != right_has_metadata:
            return jnp.asarray(False, dtype=jnp.bool_)
        if not left_has_metadata:
            if type(left_leaf) is not type(right_leaf):
                return jnp.asarray(False, dtype=jnp.bool_)
            if type(left_leaf) is float:
                leaf_equal = struct.pack("!d", left_leaf) == struct.pack("!d", right_leaf)
            else:
                leaf_equal = bool(left_leaf == right_leaf)
            equal = equal & jnp.asarray(leaf_equal, dtype=jnp.bool_)
            continue
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if tuple(left_array.shape) != tuple(right_array.shape) or str(left_array.dtype) != str(
            right_array.dtype
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_material = jr.key_data(left_array)
            right_material = jr.key_data(right_array)
        else:
            left_material = left_array
            right_material = right_array
        left_host = np.asarray(jax.device_get(left_material))
        right_host = np.asarray(jax.device_get(right_material))
        leaf_equal = (
            left_host.dtype == right_host.dtype
            and left_host.shape == right_host.shape
            and np.array_equal(
                np.ascontiguousarray(left_host).view(np.uint8),
                np.ascontiguousarray(right_host).view(np.uint8),
            )
        )
        equal = equal & jnp.asarray(leaf_equal, dtype=jnp.bool_)
    return equal


def _array_static_matches(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and tuple(cast(Any, value).shape) == shape
        and jnp.dtype(cast(Any, value).dtype) == jnp.dtype(dtype)
    )


def _scalar_bool_or_false(value: object) -> Array:
    return (
        cast(Array, value)
        if _array_static_matches(value, shape=(), dtype=jnp.bool_)
        else jnp.asarray(False, dtype=jnp.bool_)
    )


def _saturating_int32_increment(value: Array, condition: Array) -> Array:
    maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    incremented = jnp.where(value == maximum, maximum, value + jnp.asarray(1, dtype=jnp.int32))
    return jnp.where(condition, incremented, value).astype(jnp.int32)


def _tree_static_signature(tree: object) -> tuple[object, tuple[object, ...]]:
    leaves, structure = jax.tree.flatten(tree)
    signature: list[object] = []
    for leaf in leaves:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            signature.append(
                (
                    "array",
                    tuple(cast(Any, leaf).shape),
                    str(cast(Any, leaf).dtype),
                )
            )
        else:
            signature.append(("leaf", type(leaf)))
    return structure, tuple(signature)


def _tree_static_signature_matches(
    tree: object,
    signature: tuple[object, tuple[object, ...]],
) -> bool:
    try:
        return _tree_static_signature(tree) == signature
    except (AttributeError, TypeError, ValueError):
        return False


def _tree_has_only_array_leaves(tree: object) -> bool:
    try:
        return all(
            hasattr(leaf, "shape") and hasattr(leaf, "dtype")
            for leaf in jax.tree.leaves(tree)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _contains_tracer(tree: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


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


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalLearnedStateLiveMemoryActionStackConfig:
    """One coordinator/memory owner plus an exact final-action binding scope."""

    coordinator: ExternalLearnedStateRouterAuditCoordinatorConfig
    learned_memory: LearnedExperientialMemoryControllerConfig
    final_action_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        # Reuse v1's exact single-memory-owner and dimensional validation without
        # reusing its action semantics or persistent state.
        ExternalLearnedStateLiveMemoryAdapterConfig(
            coordinator=self.coordinator,
            learned_memory=self.learned_memory,
        )
        if self.coordinator.inner.prototype.recurrent_latent_world_model_ensemble is not None:
            raise ValueError(
                "action-stack v2 rejects the recurrent latent world-model cached-action lane"
            )
        owner = self.final_action_owner_digest
        if type(owner) is not tuple or len(owner) != _DIGEST_WORDS:
            raise ValueError("final_action_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(owner):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"final_action_owner_digest[{index}] must be uint32")
        if not any(owner):
            raise ValueError("final_action_owner_digest must be nonzero")

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_CONFIG_SCHEMA,
            "state_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STATE_SCHEMA,
            "memory_preparation_schema": (
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_MEMORY_PREPARATION_SCHEMA
            ),
            "finalized_schema": (EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_FINALIZED_SCHEMA),
            "receipt_schema": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_RECEIPT_SCHEMA,
            "started_finalized_schema": (
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_FINALIZED_SCHEMA
            ),
            "started_receipt_schema": (
                EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_RECEIPT_SCHEMA
            ),
            "evidence_level": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_EVIDENCE_LEVEL,
            "outcome_status": EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "coordinator": self.coordinator.to_config(),
            "learned_memory": self.learned_memory.to_config(),
            "final_action_owner_digest": list(self.final_action_owner_digest),
            "persistent_action_layers": ["base", "memory", "final"],
            "memory_feedback_layer": "memory",
            "completed_transition_action_layer": "final",
            "genesis_action_relation": "B=M=P",
            "finalization_required_before_adoption": True,
            "adoption_final_action_binding_reconstructions": (
                "one for a reconstructible finalized record; otherwise zero"
            ),
            "started_planner_bootstrap_supported": True,
            "started_planner_bootstrap_relation": "genesis B=M=P; install action-only P",
            "started_planner_bootstrap_transition_required": False,
            "selected_prototype_projection": (
                "current_action plus STOMP dispatch and active credit-owner action only"
            ),
            "standard_transition_partner_policy_fusion_sidecars_supported": False,
            "standard_transition_extended_action_mask_supported": False,
            "recurrent_latent_world_model_cached_action_projection_supported": False,
            "planner_candidate_authenticated": False,
            "monolithic_jit_supported": False,
            "scan_supported": False,
            "dispatch_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        if type(payload) is not dict:
            raise ValueError("action-stack config must be an exact dict")
        values = dict(payload)
        expected = set(
            cls(
                coordinator=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(
                    cast(dict[str, object], values.get("coordinator"))
                ),
                learned_memory=LearnedExperientialMemoryControllerConfig.from_config(
                    cast(dict[str, object], values.get("learned_memory"))
                ),
                final_action_owner_digest=tuple(
                    cast(list[int], values.get("final_action_owner_digest"))
                ),
            ).to_config()
        )
        if set(values) != expected:
            raise ValueError("action-stack config fields are not exact")
        coordinator = values.get("coordinator")
        learned_memory = values.get("learned_memory")
        owner = values.get("final_action_owner_digest")
        if type(coordinator) is not dict or type(learned_memory) is not dict:
            raise ValueError("action-stack donor configs must be exact dicts")
        if type(owner) is not list or len(owner) != _DIGEST_WORDS:
            raise ValueError("final_action_owner_digest must be an exact list")
        if any(type(word) is not int for word in owner):
            raise ValueError("final_action_owner_digest words must be exact integers")
        restored = cls(
            coordinator=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(coordinator),
            learned_memory=LearnedExperientialMemoryControllerConfig.from_config(learned_memory),
            final_action_owner_digest=tuple(cast(list[int], owner)),
        )
        if _config_digest(restored.to_config()) != _config_digest(values):
            raise ValueError("action-stack config fixed semantics or canonical form differ")
        return restored


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionBinding:
    """Persistent current-decision B/M/P identity, present after start."""

    available: Array
    memory_feedback_required: Array
    memory_transaction_words: Array
    prototype_decision_id: Array
    base_action: Array
    memory_action_before_mask: Array
    memory_action: Array
    planner_action_before_mask: Array
    final_action: Array
    hard_action_mask: Array
    categorical_retrieval: Array
    retrieval_used_expected: Array
    planner_bound: Array
    planner_consumed: Array
    final_action_owner_words: Array
    memory_candidate_words: Array
    planner_candidate_words: Array
    final_prototype_words: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackState:
    coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState
    learned_memory_state: LearnedExperientialMemoryControllerState
    action_binding: ExternalLearnedStateLiveMemoryActionBinding
    schema_digest: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackFeedback:
    """Layer-local memory credit plus final-action execution context."""

    action_binding_words: Array
    memory_transaction_words: Array
    prototype_decision_id: Array
    base_action: Array
    memory_action: Array
    final_action: Array
    hard_action_mask: Array
    retrieval_used: Array
    counterfactual_available: Array
    counterfactual_delta: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackPrepareWork:
    feedback_settlement_evaluations: Array
    coordinator_update_evaluations: Array
    learned_memory_query_evaluations: Array
    learned_memory_write_evaluations: Array
    memory_action_replacement_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
    source_state: ExternalLearnedStateLiveMemoryActionStackState
    transition: ExternalLearnedStateTransition
    event_input: ExternalLearnedStateLiveMemoryEventInput
    hard_action_mask: Array
    feedback: ExternalLearnedStateLiveMemoryActionStackFeedback
    feedback_supplied: Array
    feedback_identity_valid: Array
    transition_final_action_exact: Array
    preflight_valid: Array
    settlement_result: LearnedExperientialMemoryFeedbackResult | None
    donor_prepared: ExternalLearnedStateLiveMemoryPreparedTransition | None
    memory_candidate_state: ExternalLearnedStateLiveMemoryActionStackState
    preparation_valid: Array
    prepare_work: ExternalLearnedStateLiveMemoryActionStackPrepareWork
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryFinalActionBinding:
    source_memory_preparation_words: Array
    final_action_owner_words: Array
    prototype_decision_id: Array
    memory_action: Array
    planner_action_before_mask: Array
    final_action: Array
    hard_action_mask: Array
    planner_candidate_words: Array
    planner_consumed: Array
    selected_prototype_state: PrototypeAgentState
    final_prototype_words: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackBindWork:
    final_action_binding_evaluations: Array
    prototype_replacement_evaluations: Array
    coordinator_update_evaluations: Array
    planner_model_evaluations: Array
    learned_memory_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
    memory_preparation: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
    final_action_binding: ExternalLearnedStateLiveMemoryFinalActionBinding
    candidate_state: ExternalLearnedStateLiveMemoryActionStackState
    finalization_valid: Array
    bind_work: ExternalLearnedStateLiveMemoryActionStackBindWork
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt:
    source_state_words: Array
    finalized_content_tag_words: Array
    final_action_owner_words: Array
    integrity_bound: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryStartedFinalActionBinding:
    """Exact already-computed planner selection for one started genesis state."""

    source_state_words: Array
    final_action_owner_words: Array
    prototype_decision_id: Array
    base_action: Array
    memory_action: Array
    planner_action_before_mask: Array
    final_action: Array
    hard_action_mask: Array
    planner_candidate_words: Array
    planner_consumed: Array
    selected_prototype_state: PrototypeAgentState
    final_prototype_words: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
    """Dedicated no-transition two-phase finalization for the started decision."""

    source_state: ExternalLearnedStateLiveMemoryActionStackState
    final_action_binding: ExternalLearnedStateLiveMemoryStartedFinalActionBinding
    candidate_state: ExternalLearnedStateLiveMemoryActionStackState
    source_genesis_valid: Array
    finalization_valid: Array
    bind_work: ExternalLearnedStateLiveMemoryActionStackBindWork
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt:
    """Integrity receipt specific to a started-state finalization."""

    source_state_words: Array
    finalized_content_tag_words: Array
    final_action_owner_words: Array
    integrity_bound: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackAdoptionWork:
    integrity_evaluations: Array
    final_action_binding_reconstructions: Array
    donor_evaluations: Array
    coordinator_update_evaluations: Array
    prototype_replacement_evaluations: Array
    planner_model_evaluations: Array
    learned_memory_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackDiagnostics:
    source_state_matches: Array
    source_state_valid: Array
    finalized_content_matches: Array
    receipt_static_contract_valid: Array
    receipt_content_tag_valid: Array
    receipt_matches: Array
    receipt_integrity_bound: Array
    memory_preparation_valid: Array
    final_action_binding_valid: Array
    candidate_state_valid: Array
    transition_final_action_exact: Array
    feedback_memory_action_bound: Array
    completed_entry_final_action_exact: Array
    memory_final_actions_differ: Array
    transaction_applied: Array
    complete_source_returned: Array
    rejected: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackResult:
    state: ExternalLearnedStateLiveMemoryActionStackState
    finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition
    receipt: ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt
    diagnostics: ExternalLearnedStateLiveMemoryActionStackDiagnostics
    adoption_work: ExternalLearnedStateLiveMemoryActionStackAdoptionWork


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackStartedDiagnostics:
    """Fail-closed audit for started-state planner adoption."""

    source_state_matches: Array
    source_state_valid: Array
    source_genesis_valid: Array
    finalized_content_matches: Array
    receipt_static_contract_valid: Array
    receipt_content_tag_valid: Array
    receipt_matches: Array
    receipt_integrity_bound: Array
    final_action_binding_valid: Array
    candidate_state_valid: Array
    coordinator_clocks_preserved: Array
    memory_state_preserved: Array
    source_layers_preserved: Array
    transaction_applied: Array
    complete_source_returned: Array
    rejected: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateLiveMemoryActionStackStartedResult:
    """All-or-source started-state planner transaction result."""

    state: ExternalLearnedStateLiveMemoryActionStackState
    finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization
    receipt: ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt
    diagnostics: ExternalLearnedStateLiveMemoryActionStackStartedDiagnostics
    adoption_work: ExternalLearnedStateLiveMemoryActionStackAdoptionWork


class ExternalLearnedStateLiveMemoryActionStackAdapter:
    """Host-only versioned owner for one persistent B/M/P action stack."""

    def __init__(self, config: ExternalLearnedStateLiveMemoryActionStackConfig) -> None:
        if type(config) is not ExternalLearnedStateLiveMemoryActionStackConfig:
            raise TypeError("config must be an exact action-stack config")
        self._config = config
        self._v1 = ExternalLearnedStateLiveMemoryAdapter(
            ExternalLearnedStateLiveMemoryAdapterConfig(
                coordinator=config.coordinator,
                learned_memory=config.learned_memory,
            )
        )
        self._n_actions = config.coordinator.builder.n_actions
        self._owner_words = jnp.asarray(config.final_action_owner_digest, dtype=jnp.uint32)
        digest = hashlib.sha256(_canonical_json_bytes(config.to_config())).digest()
        self._schema_digest = jnp.asarray(tuple(digest), dtype=jnp.uint8)
        # Only its exact blank binding and v1 schema token are reused.
        self._blank_v1_state = self._v1.init(jr.key(0))
        static_reference = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=self._blank_v1_state.coordinator_state,
            learned_memory_state=self._blank_v1_state.learned_memory_state,
            action_binding=self._blank_binding(),
            schema_digest=self._schema_digest,
        )
        self._state_static_signature = _tree_static_signature(static_reference)
        self._prototype_static_signature = _tree_static_signature(
            static_reference.coordinator_state.inner_state.prototype_state
        )
        self._v1_state_static_signature = _tree_static_signature(self._blank_v1_state)
        self._coordinator_state_static_signature = _tree_static_signature(
            self._blank_v1_state.coordinator_state
        )
        self._learned_memory_state_static_signature = _tree_static_signature(
            self._blank_v1_state.learned_memory_state
        )
        self._memory_retrieval_static_signature = _tree_static_signature(
            self._v1.learned_memory._blank_retrieval()
        )
        self._completed_entry_static_signature = _tree_static_signature(
            self._v1._blank_entry()
        )

    @property
    def config(self) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        return self._config

    @property
    def coordinator(self) -> Any:
        return self._v1.coordinator

    @property
    def learned_memory(self) -> Any:
        return self._v1.learned_memory

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExternalLearnedStateLiveMemoryActionStackAdapter:
        return cls(ExternalLearnedStateLiveMemoryActionStackConfig.from_config(payload))

    def _blank_binding(self) -> ExternalLearnedStateLiveMemoryActionBinding:
        return ExternalLearnedStateLiveMemoryActionBinding(
            available=jnp.asarray(False, dtype=jnp.bool_),
            memory_feedback_required=jnp.asarray(False, dtype=jnp.bool_),
            memory_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            base_action=jnp.asarray(-1, dtype=jnp.int32),
            memory_action_before_mask=jnp.asarray(-1, dtype=jnp.int32),
            memory_action=jnp.asarray(-1, dtype=jnp.int32),
            planner_action_before_mask=jnp.asarray(-1, dtype=jnp.int32),
            final_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_action_mask=jnp.zeros((self._n_actions,), dtype=jnp.bool_),
            categorical_retrieval=jnp.asarray(False, dtype=jnp.bool_),
            retrieval_used_expected=jnp.asarray(False, dtype=jnp.bool_),
            planner_bound=jnp.asarray(False, dtype=jnp.bool_),
            planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
            final_action_owner_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            memory_candidate_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            planner_candidate_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            final_prototype_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )

    def _blank_feedback(self) -> ExternalLearnedStateLiveMemoryActionStackFeedback:
        return ExternalLearnedStateLiveMemoryActionStackFeedback(
            action_binding_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            memory_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            base_action=jnp.asarray(-1, dtype=jnp.int32),
            memory_action=jnp.asarray(-1, dtype=jnp.int32),
            final_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_action_mask=jnp.zeros((self._n_actions,), dtype=jnp.bool_),
            retrieval_used=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_available=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_delta=jnp.asarray(0.0, dtype=jnp.float32),
        )

    @staticmethod
    def _zero_prepare_work() -> ExternalLearnedStateLiveMemoryActionStackPrepareWork:
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ExternalLearnedStateLiveMemoryActionStackPrepareWork(
            feedback_settlement_evaluations=zero,
            coordinator_update_evaluations=zero,
            learned_memory_query_evaluations=zero,
            learned_memory_write_evaluations=zero,
            memory_action_replacement_evaluations=zero,
        )

    def _binding_tag(self, binding: ExternalLearnedStateLiveMemoryActionBinding) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionBinding,
            binding.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STATE_SCHEMA,
            bare,
        )

    def _make_binding(
        self,
        *,
        memory_feedback_required: Array,
        memory_transaction_words: Array,
        prototype_decision_id: Array,
        base_action: Array,
        memory_action_before_mask: Array,
        memory_action: Array,
        planner_action_before_mask: Array,
        final_action: Array,
        hard_action_mask: Array,
        categorical_retrieval: Array,
        retrieval_used_expected: Array,
        planner_bound: Array,
        planner_consumed: Array,
        memory_candidate_words: Array,
        planner_candidate_words: Array,
        final_prototype_words: Array,
    ) -> ExternalLearnedStateLiveMemoryActionBinding:
        bare = ExternalLearnedStateLiveMemoryActionBinding(
            available=jnp.asarray(True, dtype=jnp.bool_),
            memory_feedback_required=memory_feedback_required.astype(jnp.bool_),
            memory_transaction_words=memory_transaction_words.astype(jnp.uint32),
            prototype_decision_id=prototype_decision_id.astype(jnp.uint32),
            base_action=base_action.astype(jnp.int32),
            memory_action_before_mask=memory_action_before_mask.astype(jnp.int32),
            memory_action=memory_action.astype(jnp.int32),
            planner_action_before_mask=planner_action_before_mask.astype(jnp.int32),
            final_action=final_action.astype(jnp.int32),
            hard_action_mask=hard_action_mask.astype(jnp.bool_),
            categorical_retrieval=categorical_retrieval.astype(jnp.bool_),
            retrieval_used_expected=retrieval_used_expected.astype(jnp.bool_),
            planner_bound=planner_bound.astype(jnp.bool_),
            planner_consumed=planner_consumed.astype(jnp.bool_),
            final_action_owner_words=self._owner_words,
            memory_candidate_words=memory_candidate_words.astype(jnp.uint32),
            planner_candidate_words=planner_candidate_words.astype(jnp.uint32),
            final_prototype_words=final_prototype_words.astype(jnp.uint32),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionBinding,
            bare.replace(content_tag_words=self._binding_tag(bare)),
        )

    def _validate_binding_static(
        self,
        binding: ExternalLearnedStateLiveMemoryActionBinding,
    ) -> None:
        if type(binding) is not ExternalLearnedStateLiveMemoryActionBinding:
            raise TypeError("action_binding must be an exact B/M/P binding")
        for name, shape, dtype in (
            ("available", (), jnp.bool_),
            ("memory_feedback_required", (), jnp.bool_),
            ("memory_transaction_words", (2,), jnp.uint32),
            ("prototype_decision_id", (4,), jnp.uint32),
            ("base_action", (), jnp.int32),
            ("memory_action_before_mask", (), jnp.int32),
            ("memory_action", (), jnp.int32),
            ("planner_action_before_mask", (), jnp.int32),
            ("final_action", (), jnp.int32),
            ("hard_action_mask", (self._n_actions,), jnp.bool_),
            ("categorical_retrieval", (), jnp.bool_),
            ("retrieval_used_expected", (), jnp.bool_),
            ("planner_bound", (), jnp.bool_),
            ("planner_consumed", (), jnp.bool_),
            ("final_action_owner_words", (_DIGEST_WORDS,), jnp.uint32),
            ("memory_candidate_words", (_DIGEST_WORDS,), jnp.uint32),
            ("planner_candidate_words", (_DIGEST_WORDS,), jnp.uint32),
            ("final_prototype_words", (_DIGEST_WORDS,), jnp.uint32),
            ("content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
        ):
            _require_array(
                getattr(binding, name),
                name=f"action_binding.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _binding_valid(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
    ) -> Array:
        binding = state.action_binding
        coordinator = state.coordinator_state
        memory = state.learned_memory_state
        blank = self._blank_binding()
        inactive = _tree_equal(binding, blank) & ~coordinator.started
        actions = jnp.stack((binding.base_action, binding.memory_action, binding.final_action))
        in_range = jnp.all((actions >= 0) & (actions < self._n_actions))
        safe = jnp.clip(actions, 0, self._n_actions - 1)
        actions_admitted = jnp.all(binding.hard_action_mask[safe])
        memory_before_valid = (binding.memory_action_before_mask >= 0) & (
            binding.memory_action_before_mask < self._n_actions
        )
        planner_before_valid = (binding.planner_action_before_mask >= 0) & (
            binding.planner_action_before_mask < self._n_actions
        )
        memory_relation = jnp.where(
            binding.categorical_retrieval,
            (binding.memory_action == binding.memory_action_before_mask)
            | (binding.memory_action == binding.base_action),
            (binding.memory_action_before_mask == binding.base_action)
            & (binding.memory_action == binding.base_action)
            & ~binding.retrieval_used_expected,
        ) & jnp.where(
            binding.retrieval_used_expected,
            binding.categorical_retrieval
            & (binding.memory_action == binding.memory_action_before_mask),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        planner_relation = jnp.where(
            binding.planner_bound,
            jnp.any(binding.planner_candidate_words != 0)
            & jnp.where(
                binding.planner_consumed,
                (binding.final_action == binding.planner_action_before_mask)
                | (binding.final_action == binding.memory_action),
                (binding.planner_action_before_mask == binding.memory_action)
                & (binding.final_action == binding.memory_action),
            ),
            ~binding.planner_consumed
            & jnp.all(binding.planner_candidate_words == 0)
            & (binding.planner_action_before_mask == binding.memory_action)
            & (binding.final_action == binding.memory_action),
        )
        feedback_relation = (
            binding.memory_feedback_required == memory.pending.available
        ) & jnp.where(
            binding.memory_feedback_required,
            jnp.array_equal(
                binding.memory_transaction_words,
                memory.pending.transaction_words,
            ),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        prototype = coordinator.inner_state.prototype_state
        active = (
            binding.available
            & coordinator.started
            & feedback_relation
            & jnp.array_equal(
                binding.memory_transaction_words,
                memory.transaction_words,
            )
            & jnp.array_equal(
                binding.prototype_decision_id,
                coordinator.current_decision_id,
            )
            & (binding.final_action == coordinator.current_action)
            & in_range
            & jnp.any(binding.hard_action_mask)
            & actions_admitted
            & memory_before_valid
            & planner_before_valid
            & memory_relation
            & planner_relation
            & jnp.array_equal(binding.final_action_owner_words, self._owner_words)
            & jnp.any(binding.memory_candidate_words != 0)
            & jnp.array_equal(
                binding.final_prototype_words,
                _tree_digest("final-prototype-v2", prototype),
            )
            & jnp.array_equal(binding.content_tag_words, self._binding_tag(binding))
        )
        return inactive | active

    def _state_static_contract_valid(self, state: object) -> bool:
        return type(
            state
        ) is ExternalLearnedStateLiveMemoryActionStackState and _tree_static_signature_matches(
            state, self._state_static_signature
        )

    def _prototype_static_contract_valid(self, state: object) -> bool:
        return type(state) is PrototypeAgentState and _tree_static_signature_matches(
            state,
            self._prototype_static_signature,
        )

    def state_valid(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
    ) -> Array:
        """Validate one exact static/semantic state on the host only."""

        if type(state) is not ExternalLearnedStateLiveMemoryActionStackState:
            raise TypeError("state must be an exact action-stack state")
        if _contains_tracer(state):
            raise RuntimeError("action-stack state validation is host-only")
        if not self._state_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        try:
            self._validate_binding_static(state.action_binding)
            _require_array(
                state.schema_digest,
                name="state.schema_digest",
                shape=(_SCHEMA_DIGEST_BYTES,),
                dtype=jnp.uint8,
            )
            return (
                jnp.array_equal(state.schema_digest, self._schema_digest)
                & self._v1.coordinator.state_valid(state.coordinator_state)
                & self._v1.learned_memory.state_valid(state.learned_memory_state)
                & jnp.array_equal(
                    state.coordinator_state.event_words,
                    state.learned_memory_state.transaction_words,
                )
                & self._binding_valid(state)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryActionStackState:
        donor = self._v1.init(key, lifecycle_id=lifecycle_id)
        state = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=donor.coordinator_state,
            learned_memory_state=donor.learned_memory_state,
            action_binding=self._blank_binding(),
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise RuntimeError("initial action-stack state is invalid")
        return state

    def start(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        initial_observation: Array,
        *,
        hard_action_mask: Array,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryActionStackState:
        if _contains_tracer((state, initial_observation, hard_action_mask, extended_action_mask)):
            raise RuntimeError("action-stack adapter is host-only")
        mask = _require_array(
            hard_action_mask,
            name="hard_action_mask",
            shape=(self._n_actions,),
            dtype=jnp.bool_,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            return state
        coordinator = self._v1.coordinator.start(
            state.coordinator_state,
            initial_observation,
            extended_action_mask=extended_action_mask,
        )
        action = coordinator.current_action
        safe_action = int(jax.device_get(action))
        if not bool(jax.device_get(jnp.any(mask))) or not 0 <= safe_action < self._n_actions:
            return state
        if not bool(jax.device_get(mask[safe_action])):
            return state
        memory_words = _tree_digest(
            "genesis-memory-candidate-v2",
            coordinator,
            state.learned_memory_state,
            mask,
        )
        prototype = coordinator.inner_state.prototype_state
        binding = self._make_binding(
            memory_feedback_required=jnp.asarray(False, dtype=jnp.bool_),
            memory_transaction_words=state.learned_memory_state.transaction_words,
            prototype_decision_id=coordinator.current_decision_id,
            base_action=action,
            memory_action_before_mask=action,
            memory_action=action,
            planner_action_before_mask=action,
            final_action=action,
            hard_action_mask=mask,
            categorical_retrieval=jnp.asarray(False, dtype=jnp.bool_),
            retrieval_used_expected=jnp.asarray(False, dtype=jnp.bool_),
            planner_bound=jnp.asarray(False, dtype=jnp.bool_),
            planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
            memory_candidate_words=memory_words,
            planner_candidate_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            final_prototype_words=_tree_digest("final-prototype-v2", prototype),
        )
        candidate = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=coordinator,
            learned_memory_state=state.learned_memory_state,
            action_binding=binding,
            schema_digest=state.schema_digest,
        )
        return candidate if bool(jax.device_get(self.state_valid(candidate))) else state

    def _validate_feedback_static(
        self,
        feedback: ExternalLearnedStateLiveMemoryActionStackFeedback,
    ) -> None:
        if type(feedback) is not ExternalLearnedStateLiveMemoryActionStackFeedback:
            raise TypeError("feedback must be an exact action-stack feedback")
        for name, shape, dtype in (
            ("action_binding_words", (_DIGEST_WORDS,), jnp.uint32),
            ("memory_transaction_words", (2,), jnp.uint32),
            ("prototype_decision_id", (4,), jnp.uint32),
            ("base_action", (), jnp.int32),
            ("memory_action", (), jnp.int32),
            ("final_action", (), jnp.int32),
            ("hard_action_mask", (self._n_actions,), jnp.bool_),
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

    def _feedback_identity_valid(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        feedback: ExternalLearnedStateLiveMemoryActionStackFeedback,
        supplied: bool,
    ) -> Array:
        binding = state.action_binding
        supplied_array = jnp.asarray(supplied, dtype=jnp.bool_)
        exact = (
            supplied_array
            & jnp.array_equal(feedback.action_binding_words, binding.content_tag_words)
            & jnp.array_equal(
                feedback.memory_transaction_words,
                binding.memory_transaction_words,
            )
            & jnp.array_equal(
                feedback.prototype_decision_id,
                binding.prototype_decision_id,
            )
            & (feedback.base_action == binding.base_action)
            & (feedback.memory_action == binding.memory_action)
            & (feedback.final_action == binding.final_action)
            & jnp.array_equal(feedback.hard_action_mask, binding.hard_action_mask)
            & (feedback.retrieval_used == binding.retrieval_used_expected)
            & jnp.isfinite(feedback.counterfactual_delta)
            & (feedback.counterfactual_available | (feedback.counterfactual_delta == 0.0))
            & jnp.where(
                binding.retrieval_used_expected,
                jnp.asarray(True, dtype=jnp.bool_),
                ~feedback.counterfactual_available & (feedback.counterfactual_delta == 0.0),
            )
        )
        return jnp.where(binding.memory_feedback_required, exact, ~supplied_array)

    def _memory_preparation_tag(
        self,
        prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            prepared.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_MEMORY_PREPARATION_SCHEMA,
            bare,
        )

    def _prepare_work_static_contract_valid(self, work: object) -> bool:
        return type(work) is ExternalLearnedStateLiveMemoryActionStackPrepareWork and all(
            _array_static_matches(getattr(work, name), shape=(), dtype=jnp.int32)
            for name in (
                "feedback_settlement_evaluations",
                "coordinator_update_evaluations",
                "learned_memory_query_evaluations",
                "learned_memory_write_evaluations",
                "memory_action_replacement_evaluations",
            )
        )

    def _memory_preparation_static_contract_valid(self, prepared: object) -> bool:
        if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
            return False
        try:
            if not (
                self._state_static_contract_valid(prepared.source_state)
                and self._state_static_contract_valid(prepared.memory_candidate_state)
                and type(prepared.transition) is ExternalLearnedStateTransition
                and type(prepared.event_input) is ExternalLearnedStateLiveMemoryEventInput
                and type(prepared.feedback) is ExternalLearnedStateLiveMemoryActionStackFeedback
                and _array_static_matches(
                    prepared.hard_action_mask,
                    shape=(self._n_actions,),
                    dtype=jnp.bool_,
                )
                and all(
                    _array_static_matches(getattr(prepared, name), shape=(), dtype=jnp.bool_)
                    for name in (
                        "feedback_supplied",
                        "feedback_identity_valid",
                        "transition_final_action_exact",
                        "preflight_valid",
                        "preparation_valid",
                    )
                )
                and self._prepare_work_static_contract_valid(prepared.prepare_work)
                and _array_static_matches(
                    prepared.content_tag_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and (
                    prepared.settlement_result is None
                    or type(prepared.settlement_result) is LearnedExperientialMemoryFeedbackResult
                )
                and (
                    prepared.donor_prepared is None
                    or type(prepared.donor_prepared)
                    is ExternalLearnedStateLiveMemoryPreparedTransition
                )
            ):
                return False
            self._v1.coordinator._validate_transition_static(prepared.transition)
            self._v1._validate_event_input_static(prepared.event_input)
            self._validate_feedback_static(prepared.feedback)
            donor = prepared.donor_prepared
            if donor is not None and not (
                _tree_static_signature_matches(
                    donor.source_state,
                    self._v1_state_static_signature,
                )
                and _tree_static_signature_matches(
                    donor.candidate_state,
                    self._v1_state_static_signature,
                )
                and type(donor.transition) is ExternalLearnedStateTransition
                and type(donor.event_input) is ExternalLearnedStateLiveMemoryEventInput
                and type(donor.feedback) is ExternalLearnedStateLiveMemoryFeedback
                and (
                    donor.settlement_result is None
                    or self._settlement_result_static_contract_valid(
                        donor.settlement_result
                    )
                )
                and (
                    donor.coordinator_result is None
                    or type(donor.coordinator_result)
                    is ExternalLearnedStateRouterAuditCoordinatorResult
                )
                and (
                    donor.learned_memory_result is None
                    or self._memory_step_result_static_contract_valid(
                        donor.learned_memory_result
                    )
                )
                and (
                    donor.cached_action_replacement is None
                    or self._cached_replacement_static_contract_valid(
                        donor.cached_action_replacement
                    )
                )
                and _array_static_matches(
                    donor.preparation_valid,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and _array_static_matches(
                    donor.completed_entry.action,
                    shape=(self._n_actions,),
                    dtype=jnp.float32,
                )
            ):
                return False
            return True
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _bind_work_static_contract_valid(work: object) -> bool:
        return type(work) is ExternalLearnedStateLiveMemoryActionStackBindWork and all(
            _array_static_matches(getattr(work, name), shape=(), dtype=jnp.int32)
            for name in (
                "final_action_binding_evaluations",
                "prototype_replacement_evaluations",
                "coordinator_update_evaluations",
                "planner_model_evaluations",
                "learned_memory_evaluations",
            )
        )

    def _settlement_result_static_contract_valid(self, result: object) -> bool:
        if type(result) is not LearnedExperientialMemoryFeedbackResult:
            return False
        try:
            diagnostics = result.diagnostics
            return (
                _tree_static_signature_matches(
                    result.state,
                    self._learned_memory_state_static_signature,
                )
                and type(diagnostics) is LearnedExperientialMemoryFeedbackDiagnostics
                and all(
                    _array_static_matches(getattr(diagnostics, name), shape=(), dtype=dtype)
                    for name, dtype in (
                        ("source_state_valid", jnp.bool_),
                        ("pending_available", jnp.bool_),
                        ("receipt_matches", jnp.bool_),
                        ("feedback_valid", jnp.bool_),
                        ("learning_eligible", jnp.bool_),
                        ("admission_updated", jnp.bool_),
                        ("retention_rows_updated", jnp.int32),
                        ("transaction_applied", jnp.bool_),
                        ("counterfactual_feedback_authenticated", jnp.bool_),
                    )
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _memory_step_result_static_contract_valid(self, result: object) -> bool:
        if type(result) is not LearnedExperientialMemoryStepResult:
            return False
        try:
            diagnostics = result.diagnostics
            return (
                _tree_static_signature_matches(
                    result.state,
                    self._learned_memory_state_static_signature,
                )
                and _tree_static_signature_matches(
                    result.retrieval,
                    self._memory_retrieval_static_signature,
                )
                and _tree_static_signature_matches(
                    result.fixed_store_retrieval,
                    self._memory_retrieval_static_signature,
                )
                and _array_static_matches(result.wrote, shape=(), dtype=jnp.bool_)
                and _array_static_matches(result.slot, shape=(), dtype=jnp.int32)
                and _array_static_matches(result.evicted, shape=(), dtype=jnp.bool_)
                and _array_static_matches(
                    result.evicted_provenance_id,
                    shape=(),
                    dtype=jnp.int32,
                )
                and type(diagnostics) is LearnedExperientialMemoryStepDiagnostics
                and all(
                    _array_static_matches(getattr(diagnostics, name), shape=(), dtype=dtype)
                    for name, dtype in (
                        ("source_state_valid", jnp.bool_),
                        ("input_valid", jnp.bool_),
                        ("pending_blocked", jnp.bool_),
                        ("fixed_store_retrieval_accepted", jnp.bool_),
                        ("learned_admission_score", jnp.float32),
                        ("learned_retrieval_admitted", jnp.bool_),
                        ("write_succeeded", jnp.bool_),
                        ("transaction_applied", jnp.bool_),
                        ("pending_created", jnp.bool_),
                    )
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _cached_replacement_static_contract_valid(self, replacement: object) -> bool:
        if type(replacement) is not PrototypeCachedPrimitiveActionReplacement:
            return False
        try:
            decision = replacement.dispatch_replacement
            return (
                self._prototype_static_contract_valid(replacement.state)
                and _array_static_matches(replacement.action, shape=(), dtype=jnp.int32)
                and type(decision) is DispatchedPrimitiveActionDecision
                and all(
                    _array_static_matches(getattr(decision, name), shape=(), dtype=dtype)
                    for name, dtype in (
                        ("owner", jnp.int32),
                        ("state_static_contract_valid", jnp.bool_),
                        ("state_values_finite", jnp.bool_),
                        ("state_counters_valid", jnp.bool_),
                        ("rng_key_valid", jnp.bool_),
                        ("ownership_valid", jnp.bool_),
                        ("state_valid", jnp.bool_),
                        ("observation_static_contract_valid", jnp.bool_),
                        ("observation_valid", jnp.bool_),
                        ("observation_matches", jnp.bool_),
                        ("proposed_action_static_contract_valid", jnp.bool_),
                        ("proposed_action_valid", jnp.bool_),
                        ("safety_action_mask_static_contract_valid", jnp.bool_),
                        ("counterfactual_action_safe", jnp.bool_),
                        ("proposed_action_safe", jnp.bool_),
                        ("counterfactual_action", jnp.int32),
                        ("proposed_action", jnp.int32),
                        ("effective_action", jnp.int32),
                        ("used_safe_base_fallback", jnp.bool_),
                        ("applied", jnp.bool_),
                        ("failed_closed", jnp.bool_),
                    )
                )
                and all(
                    _array_static_matches(getattr(replacement, name), shape=(), dtype=jnp.bool_)
                    for name in (
                        "decision_id_matches",
                        "observation_matches",
                        "state_valid_before",
                        "state_valid_after",
                        "committed",
                    )
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _coordinator_result_static_contract_valid(self, result: object) -> bool:
        if type(result) is not ExternalLearnedStateRouterAuditCoordinatorResult:
            return False
        try:
            evaluated = result.evaluated
            prepared = evaluated.prepared
            receipt = result.receipt
            return (
                type(evaluated)
                is ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
                and type(prepared)
                is ExternalLearnedStateRouterAuditCoordinatorPreparedTransition
                and type(receipt)
                is ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt
                and type(result.diagnostics)
                is ExternalLearnedStateRouterAuditCoordinatorDiagnostics
                and _tree_static_signature_matches(
                    result.state,
                    self._coordinator_state_static_signature,
                )
                and _tree_static_signature_matches(
                    prepared.source_state,
                    self._coordinator_state_static_signature,
                )
                and _tree_static_signature_matches(
                    evaluated.candidate_state,
                    self._coordinator_state_static_signature,
                )
                and type(prepared.transition) is ExternalLearnedStateTransition
                and type(evaluated.candidate_evidence)
                is ExternalBuilderCandidateAuditEvidence
                and type(receipt.evaluated)
                is ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
                and _array_static_matches(
                    receipt.integrity_bound,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and _tree_has_only_array_leaves(evaluated)
                and _tree_has_only_array_leaves(receipt)
                and _tree_has_only_array_leaves(result.diagnostics)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _successful_v1_donor_static_contract_valid(self, donor: object) -> bool:
        if type(donor) is not ExternalLearnedStateLiveMemoryPreparedTransition:
            return False
        try:
            if not (
                _tree_static_signature_matches(
                    donor.source_state,
                    self._v1_state_static_signature,
                )
                and _tree_static_signature_matches(
                    donor.candidate_state,
                    self._v1_state_static_signature,
                )
                and type(donor.transition) is ExternalLearnedStateTransition
                and type(donor.event_input) is ExternalLearnedStateLiveMemoryEventInput
                and type(donor.feedback) is ExternalLearnedStateLiveMemoryFeedback
                and self._coordinator_result_static_contract_valid(
                    donor.coordinator_result
                )
                and self._memory_step_result_static_contract_valid(
                    donor.learned_memory_result
                )
                and _tree_static_signature_matches(
                    donor.completed_entry,
                    self._completed_entry_static_signature,
                )
                and type(donor.completed_entry) is ExperientialMemoryEntry
                and _array_static_matches(
                    donor.hard_action_mask,
                    shape=(self._n_actions,),
                    dtype=jnp.bool_,
                )
                and _array_static_matches(
                    donor.query_key,
                    shape=(self._config.coordinator.builder.observation_dim,),
                    dtype=jnp.float32,
                )
                and all(
                    _array_static_matches(getattr(donor, name), shape=(), dtype=dtype)
                    for name, dtype in (
                        ("feedback_supplied", jnp.bool_),
                        ("feedback_identity_valid", jnp.bool_),
                        ("preflight_valid", jnp.bool_),
                        ("categorical_retrieval", jnp.bool_),
                        ("retrieval_action", jnp.int32),
                        ("replacement_required", jnp.bool_),
                        ("preparation_valid", jnp.bool_),
                        ("settlement_evaluations", jnp.int32),
                        ("coordinator_evaluations", jnp.int32),
                        ("learned_memory_query_evaluations", jnp.int32),
                        ("learned_memory_write_evaluations", jnp.int32),
                        ("cached_action_replacement_evaluations", jnp.int32),
                    )
                )
                and donor.settlement_result is None
                and (
                    donor.cached_action_replacement is None
                    or self._cached_replacement_static_contract_valid(
                        donor.cached_action_replacement
                    )
                )
                and (
                    donor.candidate_evidence is None
                    or type(donor.candidate_evidence)
                    is ExternalBuilderCandidateAuditEvidence
                )
                and (
                    donor.partner_policy_fusion_input is None
                    or type(donor.partner_policy_fusion_input)
                    is PrototypePartnerPolicyFusionInput
                )
                and (
                    donor.partner_policy_fusion_feedback is None
                    or type(donor.partner_policy_fusion_feedback)
                    is PrototypePartnerPolicyFusionFeedback
                )
                and (
                    donor.extended_action_mask is None
                    or (
                        hasattr(donor.extended_action_mask, "shape")
                        and hasattr(donor.extended_action_mask, "dtype")
                        and len(donor.extended_action_mask.shape) == 1
                        and jnp.dtype(donor.extended_action_mask.dtype)
                        == jnp.dtype(jnp.bool_)
                    )
                )
            ):
                return False
            self._v1.coordinator._validate_transition_static(donor.transition)
            self._v1._validate_event_input_static(donor.event_input)
            self._v1._validate_feedback_static(donor.feedback)
            if donor.candidate_evidence is not None:
                self._v1.coordinator._validate_candidate_evidence_static(
                    donor.candidate_evidence
                )
            return True
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def _settlement_projection(
        self,
        prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ) -> tuple[Array, LearnedExperientialMemoryControllerState]:
        source_memory = prepared.source_state.learned_memory_state
        required = bool(
            jax.device_get(
                prepared.source_state.action_binding.memory_feedback_required
            )
        )
        supplied = _scalar_bool_or_false(prepared.feedback_supplied)
        feedback_identity = self._feedback_identity_valid(
            prepared.source_state,
            prepared.feedback,
            required,
        )
        common = (
            supplied == jnp.asarray(required, dtype=jnp.bool_)
        ) & feedback_identity
        if not required:
            return (
                common
                & _tree_equal(prepared.feedback, self._blank_feedback())
                & jnp.asarray(prepared.settlement_result is None, dtype=jnp.bool_),
                source_memory,
            )
        if not self._settlement_result_static_contract_valid(
            prepared.settlement_result
        ):
            return jnp.asarray(False, dtype=jnp.bool_), source_memory
        result = cast(
            LearnedExperientialMemoryFeedbackResult,
            prepared.settlement_result,
        )
        feedback = prepared.feedback
        diagnostics = result.diagnostics
        learning_eligible = feedback.retrieval_used & feedback.counterfactual_available
        delta_valid = jnp.isfinite(feedback.counterfactual_delta) & (
            jnp.abs(feedback.counterfactual_delta)
            <= jnp.asarray(
                self._config.learned_memory.max_abs_counterfactual_delta,
                dtype=jnp.float32,
            )
        )
        candidate = result.state
        source_entries = source_memory.memory.entries
        candidate_entries = candidate.memory.entries
        normalized_entries = cast(Any, candidate_entries).replace(
            utilities=source_entries.utilities,
            utility_available=source_entries.utility_available,
        )
        normalized_memory = cast(Any, candidate.memory).replace(
            entries=normalized_entries,
        )
        mutable_projection = _tree_equal(normalized_memory, source_memory.memory)
        no_learning_projection = jnp.where(
            learning_eligible,
            mutable_projection,
            _tree_equal(candidate.memory, source_memory.memory)
            & _tree_equal(candidate.admission_weights, source_memory.admission_weights),
        )
        expected_feedback_count = _saturating_int32_increment(
            source_memory.feedback_count,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        expected_learned_count = _saturating_int32_increment(
            source_memory.learned_feedback_count,
            learning_eligible,
        )
        expected_positive_count = _saturating_int32_increment(
            source_memory.positive_feedback_count,
            learning_eligible & (feedback.counterfactual_delta > 0.0),
        )
        expected_nonpositive_count = _saturating_int32_increment(
            source_memory.nonpositive_feedback_count,
            learning_eligible & (feedback.counterfactual_delta <= 0.0),
        )
        retention_count_valid = (
            diagnostics.retention_rows_updated >= 0
        ) & (
            diagnostics.retention_rows_updated
            <= self._config.learned_memory.memory.top_k
        ) & jnp.where(
            learning_eligible,
            jnp.asarray(True, dtype=jnp.bool_),
            diagnostics.retention_rows_updated == 0,
        )
        valid = (
            common
            & delta_valid
            & self._v1.learned_memory.state_valid(candidate)
            & diagnostics.source_state_valid
            & diagnostics.pending_available
            & diagnostics.receipt_matches
            & diagnostics.feedback_valid
            & (diagnostics.learning_eligible == learning_eligible)
            & (diagnostics.admission_updated == learning_eligible)
            & retention_count_valid
            & diagnostics.transaction_applied
            & ~diagnostics.counterfactual_feedback_authenticated
            & ~candidate.pending.available
            & _tree_equal(
                candidate.pending,
                self._v1.learned_memory._empty_pending(),
            )
            & jnp.array_equal(
                candidate.transaction_words,
                source_memory.transaction_words,
            )
            & jnp.array_equal(
                candidate.config_digest_words,
                source_memory.config_digest_words,
            )
            & (candidate.feedback_count == expected_feedback_count)
            & (candidate.learned_feedback_count == expected_learned_count)
            & (candidate.positive_feedback_count == expected_positive_count)
            & (candidate.nonpositive_feedback_count == expected_nonpositive_count)
            & no_learning_projection
        )
        return valid, candidate

    @staticmethod
    def _increment_words(words: Array) -> Array:
        maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        low = words[1]
        carry = low == maximum
        return jnp.stack(
            (
                jnp.where(carry, words[0] + jnp.asarray(1, dtype=jnp.uint32), words[0]),
                jnp.where(carry, jnp.asarray(0, dtype=jnp.uint32), low + 1),
            )
        ).astype(jnp.uint32)

    def _memory_preparation_semantic_projection_valid(
        self,
        prepared: object,
    ) -> Array:
        """Validate a stored successful preparation without rerunning a donor.

        The v1 coordinator, memory query/write, cached-action replacement, and
        settlement are learning/evaluation owners.  This audit only validates
        their stored successful results and deterministically projects those
        results into the provisional v2 ``B/M/P`` state where ``P=M``.
        """

        false = jnp.asarray(False, dtype=jnp.bool_)
        if not self._memory_preparation_static_contract_valid(prepared):
            return false
        exact = cast(
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            prepared,
        )
        if not self._successful_v1_donor_static_contract_valid(
            exact.donor_prepared
        ):
            return false
        donor = cast(
            ExternalLearnedStateLiveMemoryPreparedTransition,
            exact.donor_prepared,
        )
        try:
            source = exact.source_state
            source_valid = self.state_valid(source)
            source_matches = self._v1.coordinator._source_transition_matches(
                source.coordinator_state,
                exact.transition,
            )
            continuing = self._v1.coordinator._continuing_boundary_valid(
                exact.transition
            )
            event_valid = self._v1._event_input_valid(exact.event_input)
            required = bool(
                jax.device_get(source.action_binding.memory_feedback_required)
            )
            feedback_identity = self._feedback_identity_valid(
                source,
                exact.feedback,
                required,
            )
            transition_final = (
                exact.transition.action == source.action_binding.final_action
            )
            preflight = (
                source_valid
                & source_matches
                & continuing
                & event_valid
                & feedback_identity
                & transition_final
            )
            settlement_valid, memory_source = self._settlement_projection(exact)
            shadow = cast(
                ExternalLearnedStateLiveMemoryAdapterState,
                self._blank_v1_state.replace(
                    coordinator_state=source.coordinator_state,
                    learned_memory_state=memory_source,
                ),
            )
            shadow_valid = self._v1.state_valid(shadow)

            coordinator = cast(
                ExternalLearnedStateRouterAuditCoordinatorResult,
                donor.coordinator_result,
            )
            expected_coordinator = self._v1.coordinator.adopt_evaluated_transition(
                shadow.coordinator_state,
                coordinator.evaluated,
                coordinator.receipt,
            )
            coordinator_valid = (
                _tree_equal(coordinator, expected_coordinator)
                & coordinator.diagnostics.transaction_applied
                & _tree_equal(
                    coordinator.evaluated.prepared.source_state,
                    shadow.coordinator_state,
                )
                & _tree_equal(
                    coordinator.evaluated.prepared.transition,
                    exact.transition,
                )
            )
            evidence_supplied = donor.candidate_evidence is not None
            evidence_valid = (
                coordinator.evaluated.candidate_evidence_supplied
                == jnp.asarray(evidence_supplied, dtype=jnp.bool_)
            )
            if evidence_supplied:
                evidence_valid = evidence_valid & _tree_equal(
                    donor.candidate_evidence,
                    coordinator.evaluated.candidate_evidence,
                )
            else:
                evidence_valid = evidence_valid & _tree_equal(
                    coordinator.evaluated.candidate_evidence,
                    self._v1.coordinator._missing_candidate_evidence(
                        coordinator.evaluated.prepared
                    ),
                )

            memory_result = cast(
                LearnedExperientialMemoryStepResult,
                donor.learned_memory_result,
            )
            next_words = self._increment_words(memory_source.transaction_words)
            features = self._v1.learned_memory._admission_features(
                memory_result.fixed_store_retrieval
            )
            admission_score = jnp.dot(memory_source.admission_weights, features)
            admitted = (
                memory_result.fixed_store_retrieval.accepted
                & jnp.isfinite(admission_score)
                & (
                    admission_score
                    >= jnp.asarray(
                        self._config.learned_memory.admission_threshold,
                        dtype=jnp.float32,
                    )
                )
            )
            expected_retrieval = self._v1.learned_memory._gate_retrieval(
                memory_result.fixed_store_retrieval,
                admitted,
            )
            expected_pending = (
                self._v1.learned_memory._make_pending(
                    memory_source.memory,
                    expected_retrieval,
                    next_words,
                    features,
                    admission_score,
                )
                if bool(jax.device_get(admitted))
                else self._v1.learned_memory._empty_pending()
            )
            memory_valid = (
                self._v1.learned_memory.state_valid(memory_result.state)
                & memory_result.diagnostics.source_state_valid
                & memory_result.diagnostics.input_valid
                & ~memory_result.diagnostics.pending_blocked
                & memory_result.wrote
                & memory_result.diagnostics.write_succeeded
                & memory_result.diagnostics.transaction_applied
                & (
                    memory_result.diagnostics.fixed_store_retrieval_accepted
                    == memory_result.fixed_store_retrieval.accepted
                )
                & jnp.array_equal(
                    jax.lax.bitcast_convert_type(
                        memory_result.diagnostics.learned_admission_score,
                        jnp.uint32,
                    ),
                    jax.lax.bitcast_convert_type(admission_score, jnp.uint32),
                )
                & (
                    memory_result.diagnostics.pending_created == admitted
                )
                & (
                    memory_result.diagnostics.learned_retrieval_admitted
                    == admitted
                )
                & _tree_equal(memory_result.retrieval, expected_retrieval)
                & _tree_equal(memory_result.state.pending, expected_pending)
                & jnp.array_equal(
                    memory_result.state.transaction_words,
                    next_words,
                )
                & jnp.array_equal(
                    memory_result.state.transaction_words,
                    coordinator.state.event_words,
                )
                & _tree_equal(
                    memory_result.state.admission_weights,
                    memory_source.admission_weights,
                )
                & (memory_result.state.feedback_count == memory_source.feedback_count)
                & (
                    memory_result.state.learned_feedback_count
                    == memory_source.learned_feedback_count
                )
                & (
                    memory_result.state.positive_feedback_count
                    == memory_source.positive_feedback_count
                )
                & (
                    memory_result.state.nonpositive_feedback_count
                    == memory_source.nonpositive_feedback_count
                )
                & jnp.array_equal(
                    memory_result.state.config_digest_words,
                    memory_source.config_digest_words,
                )
                & (memory_result.slot >= 0)
                & (
                    memory_result.slot
                    < self._config.learned_memory.memory.capacity
                )
                & jnp.where(
                    memory_result.evicted,
                    memory_result.evicted_provenance_id >= 0,
                    memory_result.evicted_provenance_id == -1,
                )
            )

            categorical, retrieval_action = self._v1._categorical_action(
                memory_result.retrieval.action,
                memory_result.retrieval.accepted,
            )
            replacement = donor.cached_action_replacement
            replacement_count = jnp.where(categorical, 1, 0).astype(jnp.int32)
            replacement_valid = jnp.asarray(replacement is None, dtype=jnp.bool_)
            coordinator_candidate = coordinator.state
            if bool(jax.device_get(categorical)):
                replacement = cast(
                    PrototypeCachedPrimitiveActionReplacement,
                    replacement,
                )
                coordinator_prototype = (
                    coordinator.state.inner_state.prototype_state
                )
                replacement_valid = (
                    replacement.committed
                    & replacement.decision_id_matches
                    & replacement.observation_matches
                    & replacement.state_valid_before
                    & replacement.state_valid_after
                    & (replacement.action == retrieval_action)
                    & (replacement.dispatch_replacement.proposed_action == retrieval_action)
                    & (
                        replacement.dispatch_replacement.effective_action
                        == replacement.action
                    )
                    & ~replacement.dispatch_replacement.failed_closed
                    & self._selected_prototype_source_matches(
                        coordinator_prototype,
                        replacement.state,
                    )
                )
                coordinator_candidate = self._v1._replace_coordinator_action(
                    coordinator.state,
                    replacement,
                )
            v1_pending = (
                ExternalLearnedStateLiveMemoryPendingBinding(
                    available=jnp.asarray(True, dtype=jnp.bool_),
                    memory_transaction_words=memory_result.state.transaction_words,
                    prototype_decision_id=coordinator_candidate.current_decision_id,
                    base_action_before_retrieval=coordinator.state.current_action,
                    effective_action=coordinator_candidate.current_action,
                    retrieval_action=retrieval_action,
                    hard_action_mask=exact.hard_action_mask,
                    categorical_retrieval=categorical,
                    retrieval_used_expected=(
                        categorical
                        & replacement_valid
                        & ~(
                            false
                            if replacement is None
                            else replacement.dispatch_replacement.used_safe_base_fallback
                        )
                        & (retrieval_action == coordinator_candidate.current_action)
                    ),
                )
                if bool(jax.device_get(memory_result.state.pending.available))
                else self._v1._blank_pending()
            )
            expected_v1_candidate = ExternalLearnedStateLiveMemoryAdapterState(
                coordinator_state=coordinator_candidate,
                learned_memory_state=memory_result.state,
                pending_binding=v1_pending,
                schema_digest=shadow.schema_digest,
            )
            donor_projection_valid = (
                _tree_equal(donor.source_state, shadow)
                & _tree_equal(donor.transition, exact.transition)
                & _tree_equal(donor.event_input, exact.event_input)
                & jnp.array_equal(donor.hard_action_mask, exact.hard_action_mask)
                & _tree_equal(donor.feedback, self._v1._blank_feedback())
                & jnp.asarray(
                    donor.partner_policy_fusion_input is None,
                    dtype=jnp.bool_,
                )
                & jnp.asarray(
                    donor.partner_policy_fusion_feedback is None,
                    dtype=jnp.bool_,
                )
                & jnp.asarray(donor.extended_action_mask is None, dtype=jnp.bool_)
                & ~donor.feedback_supplied
                & donor.feedback_identity_valid
                & donor.preflight_valid
                & jnp.array_equal(
                    donor.query_key,
                    exact.transition.next_decision_observation,
                )
                & _tree_equal(
                    donor.completed_entry,
                    self._v1._completed_entry(exact.transition, exact.event_input),
                )
                & coordinator_valid
                & evidence_valid
                & memory_valid
                & (donor.categorical_retrieval == categorical)
                & (donor.retrieval_action == retrieval_action)
                & (donor.replacement_required == categorical)
                & replacement_valid
                & _tree_equal(donor.candidate_state, expected_v1_candidate)
                & self._v1.state_valid(expected_v1_candidate)
                & donor.preparation_valid
                & (donor.settlement_evaluations == 0)
                & (donor.coordinator_evaluations == 1)
                & (donor.learned_memory_query_evaluations == 1)
                & (donor.learned_memory_write_evaluations == 1)
                & (
                    donor.cached_action_replacement_evaluations
                    == replacement_count
                )
            )

            base_action = coordinator.state.current_action
            memory_before = jnp.where(
                categorical,
                retrieval_action,
                base_action,
            ).astype(jnp.int32)
            memory_action = expected_v1_candidate.coordinator_state.current_action
            feedback_required = memory_result.state.pending.available
            retrieval_used = jnp.where(
                feedback_required,
                v1_pending.retrieval_used_expected,
                false,
            )
            memory_words = _tree_digest(
                "memory-candidate-v2",
                expected_v1_candidate.coordinator_state,
                expected_v1_candidate.learned_memory_state,
                base_action,
                memory_before,
                memory_action,
                exact.hard_action_mask,
            )
            prototype = (
                expected_v1_candidate.coordinator_state.inner_state.prototype_state
            )
            expected_binding = self._make_binding(
                memory_feedback_required=feedback_required,
                memory_transaction_words=memory_result.state.transaction_words,
                prototype_decision_id=(
                    expected_v1_candidate.coordinator_state.current_decision_id
                ),
                base_action=base_action,
                memory_action_before_mask=memory_before,
                memory_action=memory_action,
                planner_action_before_mask=memory_action,
                final_action=memory_action,
                hard_action_mask=exact.hard_action_mask,
                categorical_retrieval=categorical,
                retrieval_used_expected=retrieval_used,
                planner_bound=false,
                planner_consumed=false,
                memory_candidate_words=memory_words,
                planner_candidate_words=jnp.zeros(
                    (_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                ),
                final_prototype_words=_tree_digest(
                    "final-prototype-v2",
                    prototype,
                ),
            )
            expected_candidate = ExternalLearnedStateLiveMemoryActionStackState(
                coordinator_state=expected_v1_candidate.coordinator_state,
                learned_memory_state=expected_v1_candidate.learned_memory_state,
                action_binding=expected_binding,
                schema_digest=source.schema_digest,
            )
            expected_work = ExternalLearnedStateLiveMemoryActionStackPrepareWork(
                feedback_settlement_evaluations=jnp.asarray(
                    int(required), dtype=jnp.int32
                ),
                coordinator_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
                learned_memory_query_evaluations=jnp.asarray(1, dtype=jnp.int32),
                learned_memory_write_evaluations=jnp.asarray(1, dtype=jnp.int32),
                memory_action_replacement_evaluations=replacement_count,
            )
            return (
                preflight
                & settlement_valid
                & shadow_valid
                & donor_projection_valid
                & (exact.feedback_supplied == jnp.asarray(required, dtype=jnp.bool_))
                & (exact.feedback_identity_valid == feedback_identity)
                & (exact.transition_final_action_exact == transition_final)
                & (exact.preflight_valid == preflight)
                & _tree_equal(exact.memory_candidate_state, expected_candidate)
                & self.state_valid(expected_candidate)
                & (exact.preparation_valid == (preflight & donor_projection_valid))
                & exact.preparation_valid
                & _tree_equal(exact.prepare_work, expected_work)
                & jnp.array_equal(
                    exact.content_tag_words,
                    self._memory_preparation_tag(exact),
                )
            )
        except Exception:
            return false

    def _memory_preparation(
        self,
        *,
        source_state: ExternalLearnedStateLiveMemoryActionStackState,
        transition: ExternalLearnedStateTransition,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
        hard_action_mask: Array,
        feedback: ExternalLearnedStateLiveMemoryActionStackFeedback,
        feedback_supplied: Array,
        feedback_identity_valid: Array,
        transition_final_action_exact: Array,
        preflight_valid: Array,
        settlement_result: LearnedExperientialMemoryFeedbackResult | None,
        donor_prepared: ExternalLearnedStateLiveMemoryPreparedTransition | None,
        memory_candidate_state: ExternalLearnedStateLiveMemoryActionStackState,
        preparation_valid: Array,
        prepare_work: ExternalLearnedStateLiveMemoryActionStackPrepareWork,
    ) -> ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
        bare = ExternalLearnedStateLiveMemoryActionStackMemoryPreparation(
            source_state=source_state,
            transition=transition,
            event_input=event_input,
            hard_action_mask=hard_action_mask,
            feedback=feedback,
            feedback_supplied=feedback_supplied,
            feedback_identity_valid=feedback_identity_valid,
            transition_final_action_exact=transition_final_action_exact,
            preflight_valid=preflight_valid,
            settlement_result=settlement_result,
            donor_prepared=donor_prepared,
            memory_candidate_state=memory_candidate_state,
            preparation_valid=preparation_valid,
            prepare_work=prepare_work,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            bare.replace(content_tag_words=self._memory_preparation_tag(bare)),
        )

    def prepare_memory_transition(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        transition: ExternalLearnedStateTransition,
        event_input: ExternalLearnedStateLiveMemoryEventInput,
        hard_action_mask: Array,
        prior_feedback: ExternalLearnedStateLiveMemoryActionStackFeedback | None = None,
        candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        *,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: PrototypePartnerPolicyFusionFeedback | None = None,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
        """Settle M credit, evaluate the real P transition, and prepare next M."""

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
            raise RuntimeError("action-stack adapter is host-only; monolithic JIT is unsupported")
        if type(state) is not ExternalLearnedStateLiveMemoryActionStackState:
            raise TypeError("state must be an exact action-stack state")
        if (
            partner_policy_fusion_input is not None
            or partner_policy_fusion_feedback is not None
            or extended_action_mask is not None
        ):
            raise ValueError(
                "standard action-stack preparation rejects donor sidecars that "
                "lack an exact stored result-semantic binding"
            )
        self._v1.coordinator._validate_transition_static(transition)
        self._v1._validate_event_input_static(event_input)
        mask = _require_array(
            hard_action_mask,
            name="hard_action_mask",
            shape=(self._n_actions,),
            dtype=jnp.bool_,
        )
        supplied = prior_feedback is not None
        feedback = self._blank_feedback() if prior_feedback is None else prior_feedback
        self._validate_feedback_static(feedback)
        supplied_array = jnp.asarray(supplied, dtype=jnp.bool_)
        feedback_identity = self._feedback_identity_valid(state, feedback, supplied)
        transition_final = transition.action == state.action_binding.final_action
        source_valid = self.state_valid(state)
        source_matches = self._v1.coordinator._source_transition_matches(
            state.coordinator_state,
            transition,
        )
        continuing = self._v1.coordinator._continuing_boundary_valid(transition)
        preflight = (
            source_valid
            & source_matches
            & continuing
            & self._v1._event_input_valid(event_input)
            & feedback_identity
            & transition_final
        )
        zero_work = self._zero_prepare_work()
        false = jnp.asarray(False, dtype=jnp.bool_)
        if not bool(jax.device_get(preflight)):
            return self._memory_preparation(
                source_state=state,
                transition=transition,
                event_input=event_input,
                hard_action_mask=mask,
                feedback=feedback,
                feedback_supplied=supplied_array,
                feedback_identity_valid=feedback_identity,
                transition_final_action_exact=transition_final,
                preflight_valid=preflight,
                settlement_result=None,
                donor_prepared=None,
                memory_candidate_state=state,
                preparation_valid=false,
                prepare_work=zero_work,
            )

        memory_source = state.learned_memory_state
        settlement: LearnedExperientialMemoryFeedbackResult | None = None
        settlement_count = jnp.asarray(0, dtype=jnp.int32)
        if bool(jax.device_get(state.action_binding.memory_feedback_required)):
            settlement_count = jnp.asarray(1, dtype=jnp.int32)
            settlement = self._v1.learned_memory.settle(
                memory_source,
                LearnedExperientialMemoryFeedback(
                    transaction_words=feedback.memory_transaction_words,
                    retrieval_used=feedback.retrieval_used,
                    counterfactual_available=feedback.counterfactual_available,
                    counterfactual_delta=feedback.counterfactual_delta,
                ),
            )
            settlement_valid = (
                settlement.diagnostics.transaction_applied
                & ~settlement.state.pending.available
                & self._v1.learned_memory.state_valid(settlement.state)
            )
            if not bool(jax.device_get(settlement_valid)):
                work = zero_work.replace(feedback_settlement_evaluations=settlement_count)
                return self._memory_preparation(
                    source_state=state,
                    transition=transition,
                    event_input=event_input,
                    hard_action_mask=mask,
                    feedback=feedback,
                    feedback_supplied=supplied_array,
                    feedback_identity_valid=feedback_identity,
                    transition_final_action_exact=transition_final,
                    preflight_valid=preflight,
                    settlement_result=settlement,
                    donor_prepared=None,
                    memory_candidate_state=state,
                    preparation_valid=false,
                    prepare_work=work,
                )
            memory_source = settlement.state

        shadow = cast(
            ExternalLearnedStateLiveMemoryAdapterState,
            self._blank_v1_state.replace(
                coordinator_state=state.coordinator_state,
                learned_memory_state=memory_source,
            ),
        )
        if not bool(jax.device_get(self._v1.state_valid(shadow))):
            work = zero_work.replace(feedback_settlement_evaluations=settlement_count)
            return self._memory_preparation(
                source_state=state,
                transition=transition,
                event_input=event_input,
                hard_action_mask=mask,
                feedback=feedback,
                feedback_supplied=supplied_array,
                feedback_identity_valid=feedback_identity,
                transition_final_action_exact=transition_final,
                preflight_valid=preflight,
                settlement_result=settlement,
                donor_prepared=None,
                memory_candidate_state=state,
                preparation_valid=false,
                prepare_work=work,
            )

        donor = self._v1.prepare_transition(
            shadow,
            transition,
            event_input,
            mask,
            None,
            candidate_evidence,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        work = ExternalLearnedStateLiveMemoryActionStackPrepareWork(
            feedback_settlement_evaluations=settlement_count,
            coordinator_update_evaluations=donor.coordinator_evaluations,
            learned_memory_query_evaluations=donor.learned_memory_query_evaluations,
            learned_memory_write_evaluations=donor.learned_memory_write_evaluations,
            memory_action_replacement_evaluations=(donor.cached_action_replacement_evaluations),
        )
        if not bool(jax.device_get(donor.preparation_valid)):
            return self._memory_preparation(
                source_state=state,
                transition=transition,
                event_input=event_input,
                hard_action_mask=mask,
                feedback=feedback,
                feedback_supplied=supplied_array,
                feedback_identity_valid=feedback_identity,
                transition_final_action_exact=transition_final,
                preflight_valid=preflight,
                settlement_result=settlement,
                donor_prepared=donor,
                memory_candidate_state=state,
                preparation_valid=false,
                prepare_work=work,
            )

        coordinator_result = cast(Any, donor.coordinator_result)
        v1_candidate = donor.candidate_state
        base_action = coordinator_result.state.current_action
        categorical = donor.categorical_retrieval
        memory_before = jnp.where(
            categorical,
            donor.retrieval_action,
            base_action,
        ).astype(jnp.int32)
        memory_action = v1_candidate.coordinator_state.current_action
        v1_pending = v1_candidate.pending_binding
        feedback_required = v1_candidate.learned_memory_state.pending.available
        retrieval_used = jnp.where(
            feedback_required,
            v1_pending.retrieval_used_expected,
            jnp.asarray(False, dtype=jnp.bool_),
        )
        memory_words = _tree_digest(
            "memory-candidate-v2",
            v1_candidate.coordinator_state,
            v1_candidate.learned_memory_state,
            base_action,
            memory_before,
            memory_action,
            mask,
        )
        prototype = v1_candidate.coordinator_state.inner_state.prototype_state
        binding = self._make_binding(
            memory_feedback_required=feedback_required,
            memory_transaction_words=v1_candidate.learned_memory_state.transaction_words,
            prototype_decision_id=v1_candidate.coordinator_state.current_decision_id,
            base_action=base_action,
            memory_action_before_mask=memory_before,
            memory_action=memory_action,
            planner_action_before_mask=memory_action,
            final_action=memory_action,
            hard_action_mask=mask,
            categorical_retrieval=categorical,
            retrieval_used_expected=retrieval_used,
            planner_bound=jnp.asarray(False, dtype=jnp.bool_),
            planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
            memory_candidate_words=memory_words,
            planner_candidate_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            final_prototype_words=_tree_digest("final-prototype-v2", prototype),
        )
        candidate = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=v1_candidate.coordinator_state,
            learned_memory_state=v1_candidate.learned_memory_state,
            action_binding=binding,
            schema_digest=state.schema_digest,
        )
        candidate_valid = self.state_valid(candidate)
        preparation_valid = preflight & donor.preparation_valid & candidate_valid
        selected_candidate = candidate if bool(jax.device_get(preparation_valid)) else state
        return self._memory_preparation(
            source_state=state,
            transition=transition,
            event_input=event_input,
            hard_action_mask=mask,
            feedback=feedback,
            feedback_supplied=supplied_array,
            feedback_identity_valid=feedback_identity,
            transition_final_action_exact=transition_final,
            preflight_valid=preflight,
            settlement_result=settlement,
            donor_prepared=donor,
            memory_candidate_state=selected_candidate,
            preparation_valid=preparation_valid,
            prepare_work=work,
        )

    def _final_binding_tag(
        self,
        binding: ExternalLearnedStateLiveMemoryFinalActionBinding,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryFinalActionBinding,
            binding.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_FINALIZED_SCHEMA,
            "final-action-binding",
            bare,
        )

    def _final_action_binding_static_contract_valid(self, binding: object) -> bool:
        if type(binding) is not ExternalLearnedStateLiveMemoryFinalActionBinding:
            return False
        try:
            return (
                _array_static_matches(
                    binding.source_memory_preparation_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.final_action_owner_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.prototype_decision_id,
                    shape=(4,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(binding.memory_action, shape=(), dtype=jnp.int32)
                and _array_static_matches(
                    binding.planner_action_before_mask,
                    shape=(),
                    dtype=jnp.int32,
                )
                and _array_static_matches(binding.final_action, shape=(), dtype=jnp.int32)
                and _array_static_matches(
                    binding.hard_action_mask,
                    shape=(self._n_actions,),
                    dtype=jnp.bool_,
                )
                and _array_static_matches(
                    binding.planner_candidate_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.planner_consumed,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and self._prototype_static_contract_valid(binding.selected_prototype_state)
                and _array_static_matches(
                    binding.final_prototype_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.content_tag_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _finalized_static_contract_valid(self, finalized: object) -> bool:
        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
            return False
        try:
            return (
                self._memory_preparation_static_contract_valid(finalized.memory_preparation)
                and self._final_action_binding_static_contract_valid(finalized.final_action_binding)
                and self._state_static_contract_valid(finalized.candidate_state)
                and _array_static_matches(
                    finalized.finalization_valid,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and self._bind_work_static_contract_valid(finalized.bind_work)
                and _array_static_matches(
                    finalized.content_tag_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _finalized_tag(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
            finalized.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_FINALIZED_SCHEMA,
            bare,
        )

    def _selected_prototype_source_matches(
        self,
        source: PrototypeAgentState,
        selected: PrototypeAgentState,
    ) -> Array:
        """Accept exactly the public cached primitive-action projection.

        Re-running the Prototype replacement here would double model work and
        consume a caller-owned evaluation.  Instead, normalize only the three
        documented dispatch leaves that the public replacement may change:
        ``current_action``, STOMP ``last_primitive_action``, and either the
        base or active-option credit-owner action.  Exact tree equality after
        that normalization rejects every learner, model, lifecycle, Horde,
        OaK, builder, IA, GRU, buffer, world, counter, and RNG mutation.
        """

        prototype = self._v1.coordinator.inner.prototype
        source_oak = prototype._oak_component_state(source.oak_state)
        selected_oak = prototype._oak_component_state(selected.oak_state)
        source_stomp = source_oak.stomp_state
        selected_stomp = selected_oak.stomp_state
        base_owner = source_stomp.executing_option == -1
        normalized_stomp = selected_stomp.replace(
            last_primitive_action=source_stomp.last_primitive_action,
            base_last_action=jnp.where(
                base_owner,
                source_stomp.base_last_action,
                selected_stomp.base_last_action,
            ).astype(jnp.int32),
            option_last_intra_action=jnp.where(
                base_owner,
                selected_stomp.option_last_intra_action,
                source_stomp.option_last_intra_action,
            ).astype(jnp.int32),
        )
        normalized_oak = selected_oak.replace(stomp_state=normalized_stomp)
        if prototype._prototype_feature_lifecycle is None:
            normalized_slot = normalized_oak
        else:
            consumer_binding = prototype._feature_consumer_binding(selected.oak_state)
            horde_state = (
                prototype._horde_component_state(selected)
                if prototype._shared_feature_horde_enabled()
                else None
            )
            feature_utility_state = (
                prototype._feature_utility_component_state(selected.oak_state)
                if prototype._feature_utility_enabled()
                else None
            )
            normalized_slot = prototype._oak_state_slot(
                normalized_oak,
                consumer_binding,
                horde_state,
                feature_utility_state,
            )
        normalized_selected = selected.replace(
            oak_state=normalized_slot,
            current_action=source.current_action,
        )
        return _tree_equal(source, normalized_selected)

    def _install_selected_prototype(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        selected: PrototypeAgentState,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        """Assemble one selected cache without invoking a learner or model."""

        coordinator = state.coordinator_state
        inner = coordinator.inner_state.replace(prototype_state=selected)
        return coordinator.replace(
            inner_state=inner,
            current_action=selected.current_action,
            current_decision_id=selected.current_decision_id,
            cached_prototype_step_words=selected.step_words,
            cached_feature_generation_words=(self._v1.coordinator._feature_generation_words(inner)),
        )

    def bind_final_action(
        self,
        prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
        selected_prototype_state: PrototypeAgentState,
        *,
        planner_action_before_mask: Array,
        planner_candidate_words: Array,
        planner_consumed: Array,
    ) -> ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
        """Install an already-computed P candidate without reevaluating any donor."""

        if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
            raise TypeError("prepared must be an exact memory preparation")
        if type(selected_prototype_state) is not PrototypeAgentState:
            raise TypeError("selected_prototype_state must be an exact Prototype state")
        if _contains_tracer(
            (
                prepared,
                selected_prototype_state,
                planner_action_before_mask,
                planner_candidate_words,
                planner_consumed,
            )
        ):
            raise RuntimeError("final-action binding is host-only")
        if not self._memory_preparation_static_contract_valid(prepared):
            raise ValueError("prepared transition has a malformed static contract")
        if not self._prototype_static_contract_valid(selected_prototype_state):
            raise ValueError("selected Prototype has a malformed static contract")
        planner_before = _require_array(
            planner_action_before_mask,
            name="planner_action_before_mask",
            shape=(),
            dtype=jnp.int32,
        )
        planner_words = _require_array(
            planner_candidate_words,
            name="planner_candidate_words",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        consumed = _require_array(
            planner_consumed,
            name="planner_consumed",
            shape=(),
            dtype=jnp.bool_,
        )
        memory_state = prepared.memory_candidate_state
        memory_binding = memory_state.action_binding
        memory_prototype = memory_state.coordinator_state.inner_state.prototype_state
        preparation_projection_valid = (
            self._memory_preparation_semantic_projection_valid(prepared)
        )
        final_action = selected_prototype_state.current_action
        safe_final = jnp.clip(final_action, 0, self._n_actions - 1)
        final_in_range = (final_action >= 0) & (final_action < self._n_actions)
        planner_before_in_range = (planner_before >= 0) & (planner_before < self._n_actions)
        selected_valid = self._v1.coordinator.inner.prototype.validate_state(
            selected_prototype_state
        )
        source_matches = self._selected_prototype_source_matches(
            memory_prototype,
            selected_prototype_state,
        )
        action_relation = jnp.where(
            consumed,
            (final_action == planner_before) | (final_action == memory_binding.memory_action),
            (planner_before == memory_binding.memory_action)
            & (final_action == memory_binding.memory_action),
        )
        binding_valid = (
            preparation_projection_valid
            & prepared.preparation_valid
            & jnp.array_equal(
                prepared.content_tag_words,
                self._memory_preparation_tag(prepared),
            )
            & selected_valid
            & source_matches
            & final_in_range
            & planner_before_in_range
            & prepared.hard_action_mask[safe_final]
            & action_relation
            & jnp.any(planner_words != 0)
        )
        final_prototype_words = _tree_digest(
            "final-prototype-v2",
            selected_prototype_state,
        )
        final_binding_bare = ExternalLearnedStateLiveMemoryFinalActionBinding(
            source_memory_preparation_words=prepared.content_tag_words,
            final_action_owner_words=self._owner_words,
            prototype_decision_id=memory_binding.prototype_decision_id,
            memory_action=memory_binding.memory_action,
            planner_action_before_mask=planner_before,
            final_action=final_action,
            hard_action_mask=prepared.hard_action_mask,
            planner_candidate_words=planner_words,
            planner_consumed=consumed,
            selected_prototype_state=selected_prototype_state,
            final_prototype_words=final_prototype_words,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        final_binding = cast(
            ExternalLearnedStateLiveMemoryFinalActionBinding,
            final_binding_bare.replace(
                content_tag_words=self._final_binding_tag(final_binding_bare)
            ),
        )
        final_coordinator = self._install_selected_prototype(
            memory_state,
            selected_prototype_state,
        )
        action_binding = self._make_binding(
            memory_feedback_required=memory_binding.memory_feedback_required,
            memory_transaction_words=memory_binding.memory_transaction_words,
            prototype_decision_id=memory_binding.prototype_decision_id,
            base_action=memory_binding.base_action,
            memory_action_before_mask=memory_binding.memory_action_before_mask,
            memory_action=memory_binding.memory_action,
            planner_action_before_mask=planner_before,
            final_action=final_action,
            hard_action_mask=memory_binding.hard_action_mask,
            categorical_retrieval=memory_binding.categorical_retrieval,
            retrieval_used_expected=memory_binding.retrieval_used_expected,
            planner_bound=jnp.asarray(True, dtype=jnp.bool_),
            planner_consumed=consumed,
            memory_candidate_words=memory_binding.memory_candidate_words,
            planner_candidate_words=planner_words,
            final_prototype_words=final_prototype_words,
        )
        candidate = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=final_coordinator,
            learned_memory_state=memory_state.learned_memory_state,
            action_binding=action_binding,
            schema_digest=memory_state.schema_digest,
        )
        candidate_valid = self.state_valid(candidate)
        finalization_valid = binding_valid & candidate_valid
        selected_candidate = (
            candidate if bool(jax.device_get(finalization_valid)) else prepared.source_state
        )
        work = ExternalLearnedStateLiveMemoryActionStackBindWork(
            final_action_binding_evaluations=jnp.asarray(1, dtype=jnp.int32),
            prototype_replacement_evaluations=jnp.asarray(0, dtype=jnp.int32),
            coordinator_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            planner_model_evaluations=jnp.asarray(0, dtype=jnp.int32),
            learned_memory_evaluations=jnp.asarray(0, dtype=jnp.int32),
        )
        bare = ExternalLearnedStateLiveMemoryActionStackFinalizedTransition(
            memory_preparation=prepared,
            final_action_binding=final_binding,
            candidate_state=selected_candidate,
            finalization_valid=finalization_valid,
            bind_work=work,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
            bare.replace(content_tag_words=self._finalized_tag(bare)),
        )

    def _receipt_tag(
        self,
        receipt: ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
            receipt.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_RECEIPT_SCHEMA,
            bare,
        )

    def _receipt_static_contract_valid(
        self,
        receipt: object,
    ) -> Array:
        if type(receipt) is not ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt:
            return jnp.asarray(False, dtype=jnp.bool_)
        try:
            valid = all(
                _array_static_matches(getattr(receipt, name), shape=shape, dtype=dtype)
                for name, shape, dtype in (
                    ("source_state_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("finalized_content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("final_action_owner_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("integrity_bound", (), jnp.bool_),
                    ("content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
                )
            )
            return jnp.asarray(valid, dtype=jnp.bool_)
        except (AttributeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def _receipt_content_tag_valid(
        self,
        receipt: object,
    ) -> Array:
        exact_type = (
            type(receipt)
            is ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt
        )
        content_static = exact_type and _array_static_matches(
            receipt.content_tag_words,
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        if not content_static:
            return jnp.asarray(False, dtype=jnp.bool_)
        try:
            return jnp.array_equal(
                receipt.content_tag_words,
                self._receipt_tag(receipt),
            )
        except Exception:
            return jnp.asarray(False, dtype=jnp.bool_)

    def _make_receipt(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ) -> ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt:
        bare = ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt(
            source_state_words=_tree_digest(
                "action-stack-source-v2",
                finalized.memory_preparation.source_state,
            ),
            finalized_content_tag_words=finalized.content_tag_words,
            final_action_owner_words=self._owner_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
            bare.replace(content_tag_words=self._receipt_tag(bare)),
        )

    def _reconstruct_standard_finalized(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ) -> ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
        binding = finalized.final_action_binding
        return self.bind_final_action(
            finalized.memory_preparation,
            binding.selected_prototype_state,
            planner_action_before_mask=binding.planner_action_before_mask,
            planner_candidate_words=binding.planner_candidate_words,
            planner_consumed=binding.planner_consumed,
        )

    def _recomputed_standard_finalized_valid(
        self,
        finalized: object,
        *,
        expected: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition | None = None,
    ) -> Array:
        if not self._finalized_static_contract_valid(finalized):
            return jnp.asarray(False, dtype=jnp.bool_)
        exact = cast(
            ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
            finalized,
        )
        try:
            if expected is None:
                expected = self._reconstruct_standard_finalized(exact)
            return (
                expected.finalization_valid
                & _tree_equal(exact, expected)
                & self._final_action_binding_valid(exact, expected=expected)
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def integrity_receipt(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ) -> ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt:
        """Host-only receipt after complete deterministic reconstruction."""

        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
            raise TypeError("only an exact finalized transition can be integrity-bound")
        if _contains_tracer(finalized):
            raise RuntimeError("action-stack receipt creation is host-only")
        if not bool(jax.device_get(self._recomputed_standard_finalized_valid(finalized))):
            raise ValueError("refusing a record that fails the recomputed finalized contract")
        return self._make_receipt(finalized)

    def _final_action_binding_valid(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
        *,
        expected: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition | None = None,
    ) -> Array:
        if not self._finalized_static_contract_valid(finalized):
            return jnp.asarray(False, dtype=jnp.bool_)
        if expected is None:
            try:
                expected = self._reconstruct_standard_finalized(finalized)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                return jnp.asarray(False, dtype=jnp.bool_)
        binding = finalized.final_action_binding
        state_binding = finalized.candidate_state.action_binding
        memory_binding = finalized.memory_preparation.memory_candidate_state.action_binding
        consumed = binding.planner_consumed
        action_relation = jnp.where(
            consumed,
            (binding.final_action == binding.planner_action_before_mask)
            | (binding.final_action == binding.memory_action),
            (binding.planner_action_before_mask == binding.memory_action)
            & (binding.final_action == binding.memory_action),
        )
        return (
            _tree_equal(binding, expected.final_action_binding)
            & jnp.array_equal(binding.content_tag_words, self._final_binding_tag(binding))
            & jnp.array_equal(
                binding.source_memory_preparation_words,
                finalized.memory_preparation.content_tag_words,
            )
            & jnp.array_equal(binding.final_action_owner_words, self._owner_words)
            & jnp.array_equal(
                binding.prototype_decision_id,
                state_binding.prototype_decision_id,
            )
            & (binding.memory_action == state_binding.memory_action)
            & (binding.memory_action == memory_binding.memory_action)
            & (binding.planner_action_before_mask == state_binding.planner_action_before_mask)
            & (binding.final_action == state_binding.final_action)
            & jnp.array_equal(
                binding.hard_action_mask, finalized.memory_preparation.hard_action_mask
            )
            & jnp.array_equal(binding.hard_action_mask, state_binding.hard_action_mask)
            & jnp.array_equal(
                binding.planner_candidate_words,
                state_binding.planner_candidate_words,
            )
            & (binding.planner_consumed == state_binding.planner_consumed)
            & action_relation
            & jnp.any(binding.planner_candidate_words != 0)
            & jnp.array_equal(
                binding.final_prototype_words,
                state_binding.final_prototype_words,
            )
            & _tree_equal(
                binding.selected_prototype_state,
                finalized.candidate_state.coordinator_state.inner_state.prototype_state,
            )
        )

    def adopt_finalized_transition(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
        receipt: ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
    ) -> ExternalLearnedStateLiveMemoryActionStackResult:
        """Select the exact finalized destination or the complete source."""

        if type(state) is not ExternalLearnedStateLiveMemoryActionStackState:
            raise TypeError("state must be an exact action-stack state")
        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
            raise TypeError("finalized must be an exact finalized transition")
        if type(receipt) is not ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt:
            raise TypeError("receipt must be an exact action-stack receipt")
        if _contains_tracer((state, finalized, receipt)):
            raise RuntimeError("action-stack adoption is host-only")
        false = jnp.asarray(False, dtype=jnp.bool_)
        finalized_static = self._finalized_static_contract_valid(finalized)
        source_valid = self.state_valid(state)
        source_matches = false
        finalized_content_matches = false
        candidate_valid = false
        memory_preparation_valid = false
        final_binding_valid = false
        recomputed_finalized_valid = false
        expected_finalized = None
        reconstruction_count = 0
        if finalized_static:
            source_matches = _tree_equal(state, finalized.memory_preparation.source_state)
            finalized_content_matches = jnp.array_equal(
                finalized.content_tag_words,
                self._finalized_tag(finalized),
            )
            candidate_valid = self.state_valid(finalized.candidate_state)
            memory_preparation_valid = _scalar_bool_or_false(
                finalized.memory_preparation.preparation_valid
            ) & jnp.array_equal(
                finalized.memory_preparation.content_tag_words,
                self._memory_preparation_tag(finalized.memory_preparation),
            )
            try:
                expected_finalized = self._reconstruct_standard_finalized(finalized)
                reconstruction_count = 1
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                expected_finalized = None
            if expected_finalized is not None:
                final_binding_valid = self._final_action_binding_valid(
                    finalized,
                    expected=expected_finalized,
                )
                recomputed_finalized_valid = self._recomputed_standard_finalized_valid(
                    finalized,
                    expected=expected_finalized,
                )
        receipt_static = self._receipt_static_contract_valid(receipt)
        receipt_tag_valid = self._receipt_content_tag_valid(receipt)
        receipt_matches = false
        if finalized_static and bool(jax.device_get(receipt_static)):
            receipt_matches = receipt_tag_valid & _tree_equal(
                receipt,
                self._make_receipt(finalized),
            )
        receipt_integrity_bound = _scalar_bool_or_false(receipt.integrity_bound)
        commit = (
            source_valid
            & source_matches
            & finalized_content_matches
            & receipt_matches
            & receipt_integrity_bound
            & memory_preparation_valid
            & final_binding_valid
            & candidate_valid
            & recomputed_finalized_valid
        )
        selected = finalized.candidate_state if bool(jax.device_get(commit)) else state
        binding = (
            finalized.candidate_state.action_binding
            if finalized_static
            else state.action_binding
            if self._state_static_contract_valid(state)
            else self._blank_binding()
        )
        donor = finalized.memory_preparation.donor_prepared if finalized_static else None
        completed_entry_exact = jnp.asarray(False, dtype=jnp.bool_)
        if donor is not None:
            executed = jax.nn.one_hot(
                jnp.clip(
                    finalized.memory_preparation.transition.action,
                    0,
                    self._n_actions - 1,
                ),
                self._n_actions,
                dtype=jnp.float32,
            )
            completed_entry_exact = jnp.array_equal(
                jax.lax.bitcast_convert_type(donor.completed_entry.action, jnp.uint32),
                jax.lax.bitcast_convert_type(executed, jnp.uint32),
            )
        diagnostics = ExternalLearnedStateLiveMemoryActionStackDiagnostics(
            source_state_matches=source_matches,
            source_state_valid=source_valid,
            finalized_content_matches=finalized_content_matches,
            receipt_static_contract_valid=receipt_static,
            receipt_content_tag_valid=receipt_tag_valid,
            receipt_matches=receipt_matches,
            receipt_integrity_bound=receipt_integrity_bound,
            memory_preparation_valid=memory_preparation_valid,
            final_action_binding_valid=final_binding_valid,
            candidate_state_valid=candidate_valid,
            transition_final_action_exact=(
                _scalar_bool_or_false(finalized.memory_preparation.transition_final_action_exact)
                if finalized_static
                else false
            ),
            feedback_memory_action_bound=(
                _scalar_bool_or_false(finalized.memory_preparation.feedback_identity_valid)
                if finalized_static
                else false
            ),
            completed_entry_final_action_exact=completed_entry_exact,
            memory_final_actions_differ=(binding.memory_action != binding.final_action),
            transaction_applied=commit,
            complete_source_returned=~commit,
            rejected=~commit,
        )
        work = ExternalLearnedStateLiveMemoryActionStackAdoptionWork(
            integrity_evaluations=jnp.asarray(1, dtype=jnp.int32),
            final_action_binding_reconstructions=jnp.asarray(
                reconstruction_count,
                dtype=jnp.int32,
            ),
            donor_evaluations=jnp.asarray(0, dtype=jnp.int32),
            coordinator_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            prototype_replacement_evaluations=jnp.asarray(0, dtype=jnp.int32),
            planner_model_evaluations=jnp.asarray(0, dtype=jnp.int32),
            learned_memory_evaluations=jnp.asarray(0, dtype=jnp.int32),
        )
        return ExternalLearnedStateLiveMemoryActionStackResult(
            state=selected,
            finalized=finalized,
            receipt=receipt,
            diagnostics=diagnostics,
            adoption_work=work,
        )

    def _started_source_genesis_valid(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
    ) -> Array:
        """Recognize the sole post-start, pre-transition B=M=P lifecycle."""

        binding = state.action_binding
        coordinator = state.coordinator_state
        prototype = coordinator.inner_state.prototype_state
        learned = state.learned_memory_state
        memory = learned.memory
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return (
            self.state_valid(state)
            & coordinator.started
            & binding.available
            & ~binding.memory_feedback_required
            & ~learned.pending.available
            & ~binding.planner_bound
            & ~binding.planner_consumed
            & jnp.all(binding.planner_candidate_words == 0)
            & ~binding.categorical_retrieval
            & ~binding.retrieval_used_expected
            & (binding.base_action == binding.memory_action_before_mask)
            & (binding.base_action == binding.memory_action)
            & (binding.base_action == binding.planner_action_before_mask)
            & (binding.base_action == binding.final_action)
            & jnp.array_equal(coordinator.event_words, zero_words)
            & jnp.array_equal(learned.transaction_words, zero_words)
            & jnp.array_equal(memory.step_words, zero_words)
            & jnp.array_equal(prototype.step_words, zero_words)
            & (memory.active_count == 0)
            & (memory.step_count == 0)
            & (memory.query_count == 0)
            & (memory.accepted_query_count == 0)
            & (memory.write_count == 0)
            & (memory.rejected_write_count == 0)
            & (memory.eviction_count == 0)
            & ~jnp.any(memory.entries.valid)
            & (learned.feedback_count == 0)
            & (learned.learned_feedback_count == 0)
            & (learned.positive_feedback_count == 0)
            & (learned.nonpositive_feedback_count == 0)
        )

    def _started_binding_tag(
        self,
        binding: ExternalLearnedStateLiveMemoryStartedFinalActionBinding,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryStartedFinalActionBinding,
            binding.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_FINALIZED_SCHEMA,
            "started-final-action-binding",
            bare,
        )

    def _started_binding_static_contract_valid(self, binding: object) -> bool:
        if type(binding) is not ExternalLearnedStateLiveMemoryStartedFinalActionBinding:
            return False
        try:
            return (
                _array_static_matches(
                    binding.source_state_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.final_action_owner_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.prototype_decision_id,
                    shape=(4,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(binding.base_action, shape=(), dtype=jnp.int32)
                and _array_static_matches(binding.memory_action, shape=(), dtype=jnp.int32)
                and _array_static_matches(
                    binding.planner_action_before_mask,
                    shape=(),
                    dtype=jnp.int32,
                )
                and _array_static_matches(binding.final_action, shape=(), dtype=jnp.int32)
                and _array_static_matches(
                    binding.hard_action_mask,
                    shape=(self._n_actions,),
                    dtype=jnp.bool_,
                )
                and _array_static_matches(
                    binding.planner_candidate_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.planner_consumed,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and self._prototype_static_contract_valid(binding.selected_prototype_state)
                and _array_static_matches(
                    binding.final_prototype_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
                and _array_static_matches(
                    binding.content_tag_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _started_finalized_tag(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
            finalized.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_FINALIZED_SCHEMA,
            bare,
        )

    def _started_finalized_static_contract_valid(self, finalized: object) -> bool:
        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
            return False
        try:
            return (
                self._state_static_contract_valid(finalized.source_state)
                and self._started_binding_static_contract_valid(finalized.final_action_binding)
                and self._state_static_contract_valid(finalized.candidate_state)
                and _array_static_matches(
                    finalized.source_genesis_valid,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and _array_static_matches(
                    finalized.finalization_valid,
                    shape=(),
                    dtype=jnp.bool_,
                )
                and self._bind_work_static_contract_valid(finalized.bind_work)
                and _array_static_matches(
                    finalized.content_tag_words,
                    shape=(_DIGEST_WORDS,),
                    dtype=jnp.uint32,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _started_source_layers_preserved(
        self,
        source: ExternalLearnedStateLiveMemoryActionStackState,
        candidate: ExternalLearnedStateLiveMemoryActionStackState,
    ) -> Array:
        before = source.action_binding
        after = candidate.action_binding
        return (
            (after.available == before.available)
            & (after.memory_feedback_required == before.memory_feedback_required)
            & jnp.array_equal(after.memory_transaction_words, before.memory_transaction_words)
            & jnp.array_equal(after.prototype_decision_id, before.prototype_decision_id)
            & (after.base_action == before.base_action)
            & (after.memory_action_before_mask == before.memory_action_before_mask)
            & (after.memory_action == before.memory_action)
            & jnp.array_equal(after.hard_action_mask, before.hard_action_mask)
            & (after.categorical_retrieval == before.categorical_retrieval)
            & (after.retrieval_used_expected == before.retrieval_used_expected)
            & jnp.array_equal(after.final_action_owner_words, before.final_action_owner_words)
            & jnp.array_equal(after.memory_candidate_words, before.memory_candidate_words)
        )

    def _started_coordinator_clocks_preserved(
        self,
        source: ExternalLearnedStateLiveMemoryActionStackState,
        candidate: ExternalLearnedStateLiveMemoryActionStackState,
    ) -> Array:
        before = source.coordinator_state
        after = candidate.coordinator_state
        before_prototype = before.inner_state.prototype_state
        after_prototype = after.inner_state.prototype_state
        return (
            jnp.array_equal(after.event_words, before.event_words)
            & (after.event_count == before.event_count)
            & jnp.array_equal(after.current_decision_id, before.current_decision_id)
            & jnp.array_equal(
                after.cached_prototype_step_words,
                before.cached_prototype_step_words,
            )
            & jnp.array_equal(after_prototype.step_words, before_prototype.step_words)
            & (after_prototype.step_count == before_prototype.step_count)
            & jnp.array_equal(
                after_prototype.observation_event_words,
                before_prototype.observation_event_words,
            )
            & (after_prototype.observation_event_count == before_prototype.observation_event_count)
        )

    def prepare_started_final_action(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        selected_prototype_state: PrototypeAgentState,
        *,
        planner_action_before_mask: Array,
        planner_candidate_words: Array,
        planner_consumed: Array,
    ) -> ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
        """Prepare P for the initial decision without fabricating a transition."""

        if type(state) is not ExternalLearnedStateLiveMemoryActionStackState:
            raise TypeError("state must be an exact action-stack state")
        if type(selected_prototype_state) is not PrototypeAgentState:
            raise TypeError("selected_prototype_state must be an exact Prototype state")
        if _contains_tracer(
            (
                state,
                selected_prototype_state,
                planner_action_before_mask,
                planner_candidate_words,
                planner_consumed,
            )
        ):
            raise RuntimeError("started final-action preparation is host-only")
        if not self._state_static_contract_valid(state):
            raise ValueError("started source has a malformed static contract")
        if not self._prototype_static_contract_valid(selected_prototype_state):
            raise ValueError("selected Prototype has a malformed static contract")
        planner_before = _require_array(
            planner_action_before_mask,
            name="planner_action_before_mask",
            shape=(),
            dtype=jnp.int32,
        )
        planner_words = _require_array(
            planner_candidate_words,
            name="planner_candidate_words",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        consumed = _require_array(
            planner_consumed,
            name="planner_consumed",
            shape=(),
            dtype=jnp.bool_,
        )
        source_binding = state.action_binding
        source_prototype = state.coordinator_state.inner_state.prototype_state
        final_action = selected_prototype_state.current_action
        safe_final = jnp.clip(final_action, 0, self._n_actions - 1)
        source_genesis_valid = self._started_source_genesis_valid(state)
        selected_valid = self._v1.coordinator.inner.prototype.validate_state(
            selected_prototype_state
        )
        projection_valid = self._selected_prototype_source_matches(
            source_prototype,
            selected_prototype_state,
        )
        final_in_range = (final_action >= 0) & (final_action < self._n_actions)
        planner_before_in_range = (planner_before >= 0) & (planner_before < self._n_actions)
        action_relation = jnp.where(
            consumed,
            (final_action == planner_before) | (final_action == source_binding.memory_action),
            (planner_before == source_binding.memory_action)
            & (final_action == source_binding.memory_action),
        )
        final_prototype_words = _tree_digest(
            "started-final-prototype-v2",
            selected_prototype_state,
        )
        source_words = _tree_digest("started-action-stack-source-v2", state)
        binding_bare = ExternalLearnedStateLiveMemoryStartedFinalActionBinding(
            source_state_words=source_words,
            final_action_owner_words=self._owner_words,
            prototype_decision_id=source_binding.prototype_decision_id,
            base_action=source_binding.base_action,
            memory_action=source_binding.memory_action,
            planner_action_before_mask=planner_before,
            final_action=final_action,
            hard_action_mask=source_binding.hard_action_mask,
            planner_candidate_words=planner_words,
            planner_consumed=consumed,
            selected_prototype_state=selected_prototype_state,
            final_prototype_words=final_prototype_words,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        final_binding = cast(
            ExternalLearnedStateLiveMemoryStartedFinalActionBinding,
            binding_bare.replace(content_tag_words=self._started_binding_tag(binding_bare)),
        )
        coordinator = self._install_selected_prototype(state, selected_prototype_state)
        action_binding = self._make_binding(
            memory_feedback_required=source_binding.memory_feedback_required,
            memory_transaction_words=source_binding.memory_transaction_words,
            prototype_decision_id=source_binding.prototype_decision_id,
            base_action=source_binding.base_action,
            memory_action_before_mask=source_binding.memory_action_before_mask,
            memory_action=source_binding.memory_action,
            planner_action_before_mask=planner_before,
            final_action=final_action,
            hard_action_mask=source_binding.hard_action_mask,
            categorical_retrieval=source_binding.categorical_retrieval,
            retrieval_used_expected=source_binding.retrieval_used_expected,
            planner_bound=jnp.asarray(True, dtype=jnp.bool_),
            planner_consumed=consumed,
            memory_candidate_words=source_binding.memory_candidate_words,
            planner_candidate_words=planner_words,
            final_prototype_words=_tree_digest("final-prototype-v2", selected_prototype_state),
        )
        candidate = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=coordinator,
            learned_memory_state=state.learned_memory_state,
            action_binding=action_binding,
            schema_digest=state.schema_digest,
        )
        candidate_valid = self.state_valid(candidate)
        memory_preserved = _tree_equal(candidate.learned_memory_state, state.learned_memory_state)
        clocks_preserved = self._started_coordinator_clocks_preserved(state, candidate)
        layers_preserved = self._started_source_layers_preserved(state, candidate)
        finalization_valid = (
            source_genesis_valid
            & selected_valid
            & projection_valid
            & final_in_range
            & planner_before_in_range
            & source_binding.hard_action_mask[safe_final]
            & action_relation
            & jnp.any(planner_words != 0)
            & candidate_valid
            & memory_preserved
            & clocks_preserved
            & layers_preserved
        )
        selected_candidate = candidate if bool(jax.device_get(finalization_valid)) else state
        work = ExternalLearnedStateLiveMemoryActionStackBindWork(
            final_action_binding_evaluations=jnp.asarray(1, dtype=jnp.int32),
            prototype_replacement_evaluations=jnp.asarray(0, dtype=jnp.int32),
            coordinator_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            planner_model_evaluations=jnp.asarray(0, dtype=jnp.int32),
            learned_memory_evaluations=jnp.asarray(0, dtype=jnp.int32),
        )
        bare = ExternalLearnedStateLiveMemoryActionStackStartedFinalization(
            source_state=state,
            final_action_binding=final_binding,
            candidate_state=selected_candidate,
            source_genesis_valid=source_genesis_valid,
            finalization_valid=finalization_valid,
            bind_work=work,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
            bare.replace(content_tag_words=self._started_finalized_tag(bare)),
        )

    def _started_receipt_tag(
        self,
        receipt: ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt,
    ) -> Array:
        bare = cast(
            ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt,
            receipt.replace(content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)),
        )
        return _tree_digest(
            EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_RECEIPT_SCHEMA,
            bare,
        )

    def _make_started_receipt(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ) -> ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt:
        bare = ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt(
            source_state_words=_tree_digest(
                "started-action-stack-source-v2",
                finalized.source_state,
            ),
            finalized_content_tag_words=finalized.content_tag_words,
            final_action_owner_words=self._owner_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt,
            bare.replace(content_tag_words=self._started_receipt_tag(bare)),
        )

    def _started_receipt_static_contract_valid(
        self,
        receipt: object,
    ) -> Array:
        if type(receipt) is not ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt:
            return jnp.asarray(False, dtype=jnp.bool_)
        try:
            valid = all(
                _array_static_matches(getattr(receipt, name), shape=shape, dtype=dtype)
                for name, shape, dtype in (
                    ("source_state_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("finalized_content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("final_action_owner_words", (_DIGEST_WORDS,), jnp.uint32),
                    ("integrity_bound", (), jnp.bool_),
                    ("content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
                )
            )
            return jnp.asarray(valid, dtype=jnp.bool_)
        except (AttributeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def _started_receipt_content_tag_valid(
        self,
        receipt: object,
    ) -> Array:
        exact_type = (
            type(receipt)
            is ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt
        )
        content_static = exact_type and _array_static_matches(
            receipt.content_tag_words,
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        if not content_static:
            return jnp.asarray(False, dtype=jnp.bool_)
        try:
            return jnp.array_equal(
                receipt.content_tag_words,
                self._started_receipt_tag(receipt),
            )
        except Exception:
            return jnp.asarray(False, dtype=jnp.bool_)

    def _reconstruct_started_finalized(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ) -> ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
        binding = finalized.final_action_binding
        return self.prepare_started_final_action(
            finalized.source_state,
            binding.selected_prototype_state,
            planner_action_before_mask=binding.planner_action_before_mask,
            planner_candidate_words=binding.planner_candidate_words,
            planner_consumed=binding.planner_consumed,
        )

    def _recomputed_started_finalized_valid(
        self,
        finalized: object,
        *,
        expected: ExternalLearnedStateLiveMemoryActionStackStartedFinalization | None = None,
    ) -> Array:
        if not self._started_finalized_static_contract_valid(finalized):
            return jnp.asarray(False, dtype=jnp.bool_)
        exact = cast(
            ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
            finalized,
        )
        try:
            if expected is None:
                expected = self._reconstruct_started_finalized(exact)
            return (
                expected.source_genesis_valid
                & expected.finalization_valid
                & _tree_equal(exact, expected)
                & self._started_final_action_binding_valid(exact, expected=expected)
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def started_final_action_integrity_receipt(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ) -> ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt:
        """Bind only one valid, exact started-state finalization."""

        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
            raise TypeError("only an exact started finalization can be integrity-bound")
        if _contains_tracer(finalized):
            raise RuntimeError("action-stack started receipt creation is host-only")
        if not bool(jax.device_get(self._recomputed_started_finalized_valid(finalized))):
            raise ValueError("refusing a record that fails the recomputed started contract")
        return self._make_started_receipt(finalized)

    def _started_final_action_binding_valid(
        self,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
        *,
        expected: ExternalLearnedStateLiveMemoryActionStackStartedFinalization | None = None,
    ) -> Array:
        if not self._started_finalized_static_contract_valid(finalized):
            return jnp.asarray(False, dtype=jnp.bool_)
        if expected is None:
            try:
                expected = self._reconstruct_started_finalized(finalized)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                return jnp.asarray(False, dtype=jnp.bool_)
        source = finalized.source_state
        candidate = finalized.candidate_state
        source_binding = source.action_binding
        candidate_binding = candidate.action_binding
        binding = finalized.final_action_binding
        source_prototype = source.coordinator_state.inner_state.prototype_state
        selected = binding.selected_prototype_state
        action_relation = jnp.where(
            binding.planner_consumed,
            (binding.final_action == binding.planner_action_before_mask)
            | (binding.final_action == binding.memory_action),
            (binding.planner_action_before_mask == binding.memory_action)
            & (binding.final_action == binding.memory_action),
        )
        return (
            _tree_equal(binding, expected.final_action_binding)
            & jnp.array_equal(binding.content_tag_words, self._started_binding_tag(binding))
            & jnp.array_equal(
                binding.source_state_words,
                _tree_digest("started-action-stack-source-v2", source),
            )
            & jnp.array_equal(binding.final_action_owner_words, self._owner_words)
            & jnp.array_equal(binding.prototype_decision_id, source_binding.prototype_decision_id)
            & (binding.base_action == source_binding.base_action)
            & (binding.memory_action == source_binding.memory_action)
            & jnp.array_equal(binding.hard_action_mask, source_binding.hard_action_mask)
            & jnp.array_equal(
                binding.planner_candidate_words,
                candidate_binding.planner_candidate_words,
            )
            & (binding.planner_consumed == candidate_binding.planner_consumed)
            & (binding.planner_action_before_mask == candidate_binding.planner_action_before_mask)
            & (binding.final_action == candidate_binding.final_action)
            & jnp.array_equal(
                binding.final_prototype_words,
                _tree_digest("started-final-prototype-v2", selected),
            )
            & _tree_equal(
                selected,
                candidate.coordinator_state.inner_state.prototype_state,
            )
            & self._selected_prototype_source_matches(source_prototype, selected)
            & candidate_binding.planner_bound
            & action_relation
            & jnp.any(binding.planner_candidate_words != 0)
            & jnp.array_equal(candidate_binding.final_action_owner_words, self._owner_words)
        )

    def adopt_started_final_action(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
        receipt: ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt,
    ) -> ExternalLearnedStateLiveMemoryActionStackStartedResult:
        """Adopt the exact started P destination or return the complete caller state."""

        if type(state) is not ExternalLearnedStateLiveMemoryActionStackState:
            raise TypeError("state must be an exact action-stack state")
        if type(finalized) is not ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
            raise TypeError("finalized must be an exact started finalization")
        if type(receipt) is not ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt:
            raise TypeError("receipt must be an exact started integrity receipt")
        if _contains_tracer((state, finalized, receipt)):
            raise RuntimeError("started final-action adoption is host-only")
        false = jnp.asarray(False, dtype=jnp.bool_)
        finalized_static = self._started_finalized_static_contract_valid(finalized)
        source_valid = self.state_valid(state)
        source_matches = false
        source_genesis_valid = false
        finalized_content_matches = false
        candidate_valid = false
        binding_valid = false
        clocks_preserved = false
        memory_preserved = false
        layers_preserved = false
        recomputed_finalized_valid = false
        expected_finalized = None
        reconstruction_count = 0
        if finalized_static:
            source = finalized.source_state
            source_matches = _tree_equal(state, source)
            source_genesis_valid = self._started_source_genesis_valid(source)
            finalized_content_matches = jnp.array_equal(
                finalized.content_tag_words,
                self._started_finalized_tag(finalized),
            )
            candidate_valid = self.state_valid(finalized.candidate_state)
            clocks_preserved = self._started_coordinator_clocks_preserved(
                source,
                finalized.candidate_state,
            )
            memory_preserved = _tree_equal(
                source.learned_memory_state,
                finalized.candidate_state.learned_memory_state,
            )
            layers_preserved = self._started_source_layers_preserved(
                source,
                finalized.candidate_state,
            )
            try:
                expected_finalized = self._reconstruct_started_finalized(finalized)
                reconstruction_count = 1
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                expected_finalized = None
            if expected_finalized is not None:
                binding_valid = self._started_final_action_binding_valid(
                    finalized,
                    expected=expected_finalized,
                )
                recomputed_finalized_valid = self._recomputed_started_finalized_valid(
                    finalized,
                    expected=expected_finalized,
                )
        receipt_static = self._started_receipt_static_contract_valid(receipt)
        receipt_tag_valid = self._started_receipt_content_tag_valid(receipt)
        receipt_matches = false
        if finalized_static and bool(jax.device_get(receipt_static)):
            receipt_matches = receipt_tag_valid & _tree_equal(
                receipt,
                self._make_started_receipt(finalized),
            )
        receipt_integrity_bound = _scalar_bool_or_false(receipt.integrity_bound)
        commit = (
            source_matches
            & source_valid
            & source_genesis_valid
            & finalized_content_matches
            & receipt_matches
            & receipt_integrity_bound
            & binding_valid
            & candidate_valid
            & clocks_preserved
            & memory_preserved
            & layers_preserved
            & recomputed_finalized_valid
        )
        selected = finalized.candidate_state if bool(jax.device_get(commit)) else state
        diagnostics = ExternalLearnedStateLiveMemoryActionStackStartedDiagnostics(
            source_state_matches=source_matches,
            source_state_valid=source_valid,
            source_genesis_valid=source_genesis_valid,
            finalized_content_matches=finalized_content_matches,
            receipt_static_contract_valid=receipt_static,
            receipt_content_tag_valid=receipt_tag_valid,
            receipt_matches=receipt_matches,
            receipt_integrity_bound=receipt_integrity_bound,
            final_action_binding_valid=binding_valid,
            candidate_state_valid=candidate_valid,
            coordinator_clocks_preserved=clocks_preserved,
            memory_state_preserved=memory_preserved,
            source_layers_preserved=layers_preserved,
            transaction_applied=commit,
            complete_source_returned=~commit,
            rejected=~commit,
        )
        work = ExternalLearnedStateLiveMemoryActionStackAdoptionWork(
            integrity_evaluations=jnp.asarray(1, dtype=jnp.int32),
            final_action_binding_reconstructions=jnp.asarray(
                reconstruction_count,
                dtype=jnp.int32,
            ),
            donor_evaluations=jnp.asarray(0, dtype=jnp.int32),
            coordinator_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            prototype_replacement_evaluations=jnp.asarray(0, dtype=jnp.int32),
            planner_model_evaluations=jnp.asarray(0, dtype=jnp.int32),
            learned_memory_evaluations=jnp.asarray(0, dtype=jnp.int32),
        )
        return ExternalLearnedStateLiveMemoryActionStackStartedResult(
            state=selected,
            finalized=finalized,
            receipt=receipt,
            diagnostics=diagnostics,
            adoption_work=work,
        )

    def upgrade_v1_state(
        self,
        v1_adapter: ExternalLearnedStateLiveMemoryAdapter,
        state: ExternalLearnedStateLiveMemoryAdapterState,
        *,
        hard_action_mask: Array,
    ) -> ExternalLearnedStateLiveMemoryActionStackState:
        """Host-only explicit v1 P=M upgrade; never mutate or auto-load it."""

        if type(v1_adapter) is not ExternalLearnedStateLiveMemoryAdapter:
            raise TypeError("v1_adapter must be an exact v1 adapter")
        if type(state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("state must be an exact v1 adapter state")
        if _contains_tracer((state, hard_action_mask)):
            raise RuntimeError("action-stack v1 upgrade is host-only")
        mask = _require_array(
            hard_action_mask,
            name="hard_action_mask",
            shape=(self._n_actions,),
            dtype=jnp.bool_,
        )
        expected_v1 = ExternalLearnedStateLiveMemoryAdapterConfig(
            coordinator=self._config.coordinator,
            learned_memory=self._config.learned_memory,
        ).to_config()
        if _config_digest(v1_adapter.to_config()) != _config_digest(expected_v1):
            raise ValueError("v1 owner config does not match this v2 adapter")
        if not bool(jax.device_get(v1_adapter.state_valid(state))):
            raise ValueError("cannot upgrade an invalid v1 state")
        if not bool(jax.device_get(state.coordinator_state.started)):
            candidate = ExternalLearnedStateLiveMemoryActionStackState(
                coordinator_state=state.coordinator_state,
                learned_memory_state=state.learned_memory_state,
                action_binding=self._blank_binding(),
                schema_digest=self._schema_digest,
            )
            if not bool(jax.device_get(self.state_valid(candidate))):
                raise ValueError("v1 genesis could not be represented exactly")
            return candidate

        pending = state.pending_binding
        current = state.coordinator_state.current_action
        base = jnp.where(
            pending.available,
            pending.base_action_before_retrieval,
            current,
        ).astype(jnp.int32)
        memory = current.astype(jnp.int32)
        memory_before = jnp.where(
            pending.available & pending.categorical_retrieval,
            pending.retrieval_action,
            base,
        ).astype(jnp.int32)
        categorical = pending.available & pending.categorical_retrieval
        used = pending.available & pending.retrieval_used_expected
        if bool(jax.device_get(pending.available)) and not bool(
            jax.device_get(jnp.array_equal(mask, pending.hard_action_mask))
        ):
            raise ValueError("supplied v1 hard mask differs from its pending binding")
        words = _tree_digest(
            "upgraded-v1-memory-candidate-v2",
            state.coordinator_state,
            state.learned_memory_state,
            base,
            memory_before,
            memory,
            mask,
        )
        prototype = state.coordinator_state.inner_state.prototype_state
        binding = self._make_binding(
            memory_feedback_required=state.learned_memory_state.pending.available,
            memory_transaction_words=state.learned_memory_state.transaction_words,
            prototype_decision_id=state.coordinator_state.current_decision_id,
            base_action=base,
            memory_action_before_mask=memory_before,
            memory_action=memory,
            planner_action_before_mask=memory,
            final_action=memory,
            hard_action_mask=mask,
            categorical_retrieval=categorical,
            retrieval_used_expected=used,
            planner_bound=jnp.asarray(False, dtype=jnp.bool_),
            planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
            memory_candidate_words=words,
            planner_candidate_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            final_prototype_words=_tree_digest("final-prototype-v2", prototype),
        )
        candidate = ExternalLearnedStateLiveMemoryActionStackState(
            coordinator_state=state.coordinator_state,
            learned_memory_state=state.learned_memory_state,
            action_binding=binding,
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(candidate))):
            raise ValueError("valid v1 P=M state could not be represented exactly")
        return candidate


__all__ = [
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_CONFIG_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_EVIDENCE_LEVEL",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_FINALIZED_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_MEMORY_PREPARATION_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_OUTCOME_STATUS",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_RECEIPT_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_SCIENTIFIC_PROMOTION_ALLOWED",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STATE_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_FINALIZED_SCHEMA",
    "EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_STARTED_RECEIPT_SCHEMA",
    "ExternalLearnedStateLiveMemoryActionBinding",
    "ExternalLearnedStateLiveMemoryActionStackAdapter",
    "ExternalLearnedStateLiveMemoryActionStackAdoptionWork",
    "ExternalLearnedStateLiveMemoryActionStackBindWork",
    "ExternalLearnedStateLiveMemoryActionStackConfig",
    "ExternalLearnedStateLiveMemoryActionStackDiagnostics",
    "ExternalLearnedStateLiveMemoryActionStackFeedback",
    "ExternalLearnedStateLiveMemoryActionStackFinalizedTransition",
    "ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt",
    "ExternalLearnedStateLiveMemoryActionStackMemoryPreparation",
    "ExternalLearnedStateLiveMemoryActionStackPrepareWork",
    "ExternalLearnedStateLiveMemoryActionStackResult",
    "ExternalLearnedStateLiveMemoryActionStackState",
    "ExternalLearnedStateLiveMemoryActionStackStartedDiagnostics",
    "ExternalLearnedStateLiveMemoryActionStackStartedFinalization",
    "ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt",
    "ExternalLearnedStateLiveMemoryActionStackStartedResult",
    "ExternalLearnedStateLiveMemoryFinalActionBinding",
    "ExternalLearnedStateLiveMemoryStartedFinalActionBinding",
]
