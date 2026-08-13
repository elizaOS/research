# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,union-attr"
"""Bind Kondo actor work to the exact planner action that became executed P.

This host-only L0 bridge closes one narrow accounting gap.  It samples a
fixed batch from one exact :class:`KondoSparseActorState`, binds every sample
to an exact post-memory action-stack preparation, and admits that row to the
actor only after the public action-stack adoption is reconstructed now or was
reconstructed at compact-receipt issuance and the next real transition names
the same executed final action.  The supported v1
policy is deliberately unmasked: every hard action mask must be all true,
because ``KondoSparseActor`` recomputes an ordinary categorical behavior
probability and does not implement a masked-policy correction.

The nested :class:`KondoSparseActorResult` is the only owner of the executed
backward indicator.  A sample therefore sparks joy exactly when its selected
gradient contribution enters an actor backward that actually executes.  This
bridge adds no broader joy alias and makes no claim about gradient finiteness,
parameter acceptance, physical execution authentication, dispatch, safety,
critic execution, evidence, or promotion.  All SHA-256 words here are unkeyed
integrity bindings; they are not caller authentication.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackAdapter,
    ExternalLearnedStateLiveMemoryActionStackConfig,
    ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ExternalLearnedStateLiveMemoryActionStackResult,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    _tree_digest as _action_stack_tree_digest,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    _tree_equal as _action_stack_tree_equal,
)
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorProtectedInputs,
    KondoSparseActor,
    KondoSparseActorBatch,
    KondoSparseActorConfig,
    KondoSparseActorResult,
    KondoSparseActorState,
)

KONDO_EXECUTED_ACTION_LINEAGE_BRIDGE_SCHEMA = (
    "alberta.kondo-executed-action-lineage-bridge.v1"
)
KONDO_EXECUTED_ACTION_PROPOSAL_SCHEMA = (
    "alberta.kondo-executed-action-lineage-proposal.v1"
)
KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA = (
    "alberta.kondo-executed-action-compact-adoption.v1"
)
KONDO_EXECUTED_ACTION_LINEAGE_EVIDENCE_LEVEL = "L0"
KONDO_EXECUTED_ACTION_LINEAGE_OUTCOME_STATUS = "not_assessed"

_DIGEST_WORDS = 8

__all__ = (
    "KONDO_EXECUTED_ACTION_LINEAGE_BRIDGE_SCHEMA",
    "KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA",
    "KONDO_EXECUTED_ACTION_LINEAGE_EVIDENCE_LEVEL",
    "KONDO_EXECUTED_ACTION_LINEAGE_OUTCOME_STATUS",
    "KondoExecutedActionLineageBridge",
    "KondoExecutedActionLineageBridgeConfig",
    "KondoExecutedActionLineageDiagnostics",
    "KondoExecutedActionLineageResult",
    "KondoExecutedActionLineageWork",
    "KondoExecutedActionCompactAdoptionBatch",
    "KondoExecutedActionProposalBatch",
)


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


def _require_typed_keys(value: object, *, batch_size: int) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError("sampling_keys must be a typed PRNG-key array")
    keys = cast(Array, value)
    if not jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key):
        raise TypeError("sampling_keys must use a typed PRNG-key dtype")
    if tuple(keys.shape) != (batch_size,):
        raise ValueError(
            f"sampling_keys must have shape {(batch_size,)}; got {tuple(keys.shape)}"
        )
    return keys


def _bitwise_float32_equal(left: Array, right: Array) -> Array:
    return jax.lax.bitcast_convert_type(left, jnp.uint32) == jax.lax.bitcast_convert_type(
        right,
        jnp.uint32,
    )


def _actor_logits(state: KondoSparseActorState, actor_features: Array) -> Array:
    hidden = jnp.tanh(
        actor_features @ state.parameters.hidden_weight + state.parameters.hidden_bias
    )
    return hidden @ state.parameters.output_weight + state.parameters.output_bias


@dataclasses.dataclass(frozen=True, slots=True)
class KondoExecutedActionLineageBridgeConfig:
    """Exact actor/action-stack dimensions for one fixed lineage batch."""

    actor: KondoSparseActorConfig
    action_stack: ExternalLearnedStateLiveMemoryActionStackConfig
    action_stack_rows: tuple[ExternalLearnedStateLiveMemoryActionStackConfig, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.actor) is not KondoSparseActorConfig:
            raise TypeError("actor must be an exact Kondo sparse actor config")
        if type(self.action_stack) is not ExternalLearnedStateLiveMemoryActionStackConfig:
            raise TypeError("action_stack must be an exact action-stack config")
        if self.actor.action_count != self.action_stack.coordinator.builder.n_actions:
            raise ValueError("actor and action stack must have the same action count")
        rows = self.action_stack_rows
        if rows is not None:
            if type(rows) is not tuple or len(rows) != self.actor.batch_size:
                raise ValueError("action_stack_rows must be an exact fixed-batch tuple")
            for row, config in enumerate(rows):
                if type(config) is not ExternalLearnedStateLiveMemoryActionStackConfig:
                    raise TypeError(f"action_stack_rows[{row}] must be an exact config")
                if self.actor.action_count != config.coordinator.builder.n_actions:
                    raise ValueError(
                        f"actor and action_stack_rows[{row}] must share action count"
                    )
            if rows[0].to_config() != self.action_stack.to_config():
                raise ValueError("action_stack must equal action_stack_rows[0]")

    def to_config(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": KONDO_EXECUTED_ACTION_LINEAGE_BRIDGE_SCHEMA,
            "type": type(self).__name__,
            "actor": self.actor.to_config(),
            "action_stack": self.action_stack.to_config(),
            "evidence_level": KONDO_EXECUTED_ACTION_LINEAGE_EVIDENCE_LEVEL,
            "outcome_status": KONDO_EXECUTED_ACTION_LINEAGE_OUTCOME_STATUS,
            "hard_action_mask_support": "exact-all-true-only",
            "proposal_integrity": "unkeyed-rederived-source-bound",
            "proposal_source_binding": (
                "exact-source-state-plus-memory-preparation-plus-post-memory-action-binding"
            ),
            "executed_action_relation": (
                "proposal-equals-planner-before-mask-equals-final-P-equals-next-transition-action"
            ),
            "delight_semantics": "advantage-times-selected-action-surprisal",
            "invalid_lineage_rows_sanitized_before_actor": True,
            "protected_channels_full_batch": True,
            "maximum_actor_step_calls_per_batch": 1,
            "compact_pending_adoption_lineage_supported": True,
            "compact_pair_preflight_supported": True,
            "compact_pending_mutable_owner_snapshots": 0,
            "compact_pending_integrity": "issuer-reconstructed-unkeyed-content-bound",
            "compact_historic_tree_reconstruction_at_step": False,
            "lineage_mode_codes": {
                "full-current-reconstruction": 0,
                "compact-issued-carry-forward": 1,
            },
            "caller_authenticated": False,
            "physical_execution_authenticated": False,
            "dispatch_authority": False,
            "safety_execution_claimed": False,
            "critic_execution_claimed": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }
        if self.action_stack_rows is not None:
            payload["action_stack_rows"] = [
                config.to_config() for config in self.action_stack_rows
            ]
            payload["action_stack_row_ownership"] = "exact-per-row-distinct-supported"
        return payload

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> KondoExecutedActionLineageBridgeConfig:
        if type(payload) is not dict:
            raise ValueError("lineage bridge config must be an exact dict")
        actor_payload = payload.get("actor")
        action_stack_payload = payload.get("action_stack")
        if type(actor_payload) is not dict or type(action_stack_payload) is not dict:
            raise ValueError("lineage bridge child configs must be exact dicts")
        rows_payload = payload.get("action_stack_rows")
        if rows_payload is None:
            rows = None
        else:
            if type(rows_payload) is not list:
                raise ValueError("action_stack_rows must serialize as an exact list")
            rows = tuple(
                ExternalLearnedStateLiveMemoryActionStackConfig.from_config(item)
                for item in rows_payload
                if type(item) is dict
            )
            if len(rows) != len(rows_payload):
                raise ValueError("every action_stack_rows entry must be an exact dict")
        result = cls(
            actor=KondoSparseActorConfig.from_config(actor_payload),
            action_stack=ExternalLearnedStateLiveMemoryActionStackConfig.from_config(
                action_stack_payload
            ),
            action_stack_rows=rows,
        )
        if result.to_config() != dict(payload):
            raise ValueError("lineage bridge config fields or fixed semantics differ")
        return result


@chex.dataclass(frozen=True)
class KondoExecutedActionProposalBatch:
    """One immutable actor proposal batch bound to post-memory decisions."""

    actor_features: Array
    sampling_keys: Array
    actor_state_words: Array
    policy_revision: Array
    selected_actions: Array
    behavior_log_probability: Array
    action_stack_source_state_words: Array
    action_stack_memory_preparation_words: Array
    action_stack_memory_candidate_binding_words: Array
    action_stack_decision_identities: Array
    hard_action_masks: Array
    proposal_digest_words: Array


@chex.dataclass(frozen=True)
class KondoExecutedActionCompactAdoptionBatch:
    """Fixed-batch adoption proof without persisting mutable owner snapshots.

    The issuer reconstructs the public action-stack adoption once while the
    full result is transient, then retains only exact digests and route facts.
    At the following event, :meth:`step_compact` binds those words to the
    current action-stack source and its real transition.  These unkeyed words
    provide content integrity, not caller authentication.
    """

    source_state_words: Array
    memory_preparation_words: Array
    memory_candidate_binding_words: Array
    decision_identities: Array
    final_action_owner_words: Array
    finalization_words: Array
    integrity_receipt_words: Array
    adoption_result_words: Array
    destination_state_words: Array
    planner_candidate_words: Array
    planner_actions_before_mask: Array
    final_actions: Array
    hard_action_masks: Array
    planner_consumed: Array
    adoption_applied: Array
    content_tag_words: Array


@chex.dataclass(frozen=True)
class KondoExecutedActionLineageDiagnostics:
    """Per-row fail-closed facts used to form the actor validity mask.

    ``lineage_mode`` is zero when the historic public adoption was reconstructed
    during this call and one when a compact certificate carries a reconstruction
    performed at issuance.  In compact mode the historic-reconstruction fields
    (``memory_preparation_integrity_rederived``,
    ``memory_candidate_binding_exact``, ``adoption_result_exact``, and
    ``historic_adoption_reconstructed``) remain false: the old mutable tree is
    intentionally absent.  ``compact_carry_forward_certificate_valid`` owns the
    corresponding issuer-carried/current-source relation instead.
    """

    proposal_integrity_rederived: Array
    actor_snapshot_exact: Array
    policy_revision_exact: Array
    proposal_sample_exact: Array
    behavior_log_probability_exact: Array
    action_stack_source_exact: Array
    memory_preparation_integrity_rederived: Array
    memory_candidate_binding_exact: Array
    hard_action_mask_exact_all_true: Array
    lineage_mode: Array
    historic_adoption_reconstructed: Array
    compact_carry_forward_certificate_valid: Array
    adoption_result_exact: Array
    adoption_transaction_applied: Array
    actor_path_selected: Array
    planner_candidate_exact: Array
    planner_before_mask_exact: Array
    final_action_exact: Array
    next_preparation_integrity_rederived: Array
    next_preparation_source_exact: Array
    next_transition_action_exact: Array
    actor_eligible: Array


@chex.dataclass(frozen=True)
class KondoExecutedActionLineageWork:
    """Logical evaluations; these are not timing or FLOP measurements."""

    actor_step_calls: Array
    adoption_integrity_reconstructions: Array
    compact_adoption_receipt_validations: Array
    action_stack_learner_evaluations: Array
    planner_model_evaluations: Array


@chex.dataclass(frozen=True)
class KondoExecutedActionLineageResult:
    """Nested actual actor result plus exact lineage diagnostics.

    Deliberately do not add a second joy property here.  That execution fact
    belongs only to ``actor_result``.
    """

    actor_result: KondoSparseActorResult
    diagnostics: KondoExecutedActionLineageDiagnostics
    protected: KondoActorProtectedInputs
    work: KondoExecutedActionLineageWork


class KondoExecutedActionLineageBridge:
    """Host-audited proposal, executed-P lineage, and Kondo actor bridge."""

    def __init__(self, config: KondoExecutedActionLineageBridgeConfig) -> None:
        if type(config) is not KondoExecutedActionLineageBridgeConfig:
            raise TypeError("config must be an exact lineage bridge config")
        self._config = config
        self._actor = KondoSparseActor(config.actor)
        row_configs = (
            (config.action_stack,) * config.actor.batch_size
            if config.action_stack_rows is None
            else config.action_stack_rows
        )
        self._action_stacks = tuple(
            ExternalLearnedStateLiveMemoryActionStackAdapter(item)
            for item in row_configs
        )
        self._action_stack = self._action_stacks[0]

    @property
    def config(self) -> KondoExecutedActionLineageBridgeConfig:
        return self._config

    @property
    def actor(self) -> KondoSparseActor:
        return self._actor

    @property
    def action_stack(self) -> ExternalLearnedStateLiveMemoryActionStackAdapter:
        return self._action_stack

    @property
    def action_stacks(
        self,
    ) -> tuple[ExternalLearnedStateLiveMemoryActionStackAdapter, ...]:
        """Return the exact per-row action owners used by lineage validation."""

        return self._action_stacks

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> KondoExecutedActionLineageBridge:
        return cls(KondoExecutedActionLineageBridgeConfig.from_config(payload))

    def actor_state_digest_words(self, state: KondoSparseActorState) -> Array:
        """Return the unkeyed exact-tree actor snapshot binding."""

        if type(state) is not KondoSparseActorState:
            raise TypeError("state must be an exact Kondo sparse actor state")
        self._actor._validate_state_static(state)
        return _action_stack_tree_digest(
            KONDO_EXECUTED_ACTION_PROPOSAL_SCHEMA,
            "actor-state",
            state,
        )

    def action_stack_source_state_digest_words(
        self,
        prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ) -> Array:
        """Bind the complete pre-preparation action-stack source state."""

        if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
            raise TypeError("prepared must be an exact action-stack memory preparation")
        return _action_stack_tree_digest(
            KONDO_EXECUTED_ACTION_PROPOSAL_SCHEMA,
            "action-stack-source-state",
            prepared.source_state,
        )

    def rederive_next_preparation_digest_words(
        self,
        prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
        *,
        row: int = 0,
    ) -> Array:
        """Expose the action-stack's exact unkeyed preparation rederivation."""

        if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
            raise TypeError("prepared must be an exact action-stack memory preparation")
        if type(row) is not int or not 0 <= row < self._config.actor.batch_size:
            raise ValueError("row must name one fixed actor-batch row")
        return self._action_stacks[row]._memory_preparation_tag(prepared)

    def _validate_memory_preparations(
        self,
        preparations: object,
    ) -> tuple[ExternalLearnedStateLiveMemoryActionStackMemoryPreparation, ...]:
        cfg = self._config.actor
        if type(preparations) is not tuple or len(preparations) != cfg.batch_size:
            raise TypeError(
                "action_stack_memory_preparations must be an exact fixed-batch tuple"
            )
        exact = cast(
            tuple[ExternalLearnedStateLiveMemoryActionStackMemoryPreparation, ...],
            preparations,
        )
        # The all-true failure is deliberately first and explicit.  A caller
        # must never mistake this unmasked behavior probability for a masked
        # policy probability, even if another preparation field is malformed.
        for row, prepared in enumerate(exact):
            if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
                raise TypeError(f"preparation[{row}] must be an exact memory preparation")
            mask = _require_array(
                prepared.hard_action_mask,
                name=f"preparation[{row}].hard_action_mask",
                shape=(cfg.action_count,),
                dtype=jnp.bool_,
            )
            if not bool(np.asarray(jnp.all(mask))):
                raise ValueError("Kondo proposal sampling supports all-true masks only")
        for row, prepared in enumerate(exact):
            adapter = self._action_stacks[row]
            if not adapter._memory_preparation_static_contract_valid(prepared):
                raise ValueError(f"preparation[{row}] has a malformed static contract")
            binding = prepared.memory_candidate_state.action_binding
            preparation_exact = (
                prepared.preparation_valid
                & jnp.array_equal(
                    prepared.content_tag_words,
                    adapter._memory_preparation_tag(prepared),
                )
                & adapter.state_valid(prepared.source_state)
                & adapter.state_valid(prepared.memory_candidate_state)
                & binding.available
                & ~binding.planner_bound
                & ~binding.planner_consumed
                & jnp.all(binding.planner_candidate_words == 0)
                & (binding.final_action == binding.memory_action)
                & jnp.array_equal(binding.hard_action_mask, prepared.hard_action_mask)
            )
            if not bool(np.asarray(preparation_exact)):
                raise ValueError(
                    f"preparation[{row}] is not an exact provisional post-memory decision"
                )
        return exact

    def _sample_actions(
        self,
        state: KondoSparseActorState,
        actor_features: Array,
        sampling_keys: Array,
    ) -> Array:
        logits = _actor_logits(state, actor_features)
        return jnp.stack(
            tuple(
                jr.categorical(sampling_keys[row], logits[row]).astype(jnp.int32)
                for row in range(self._config.actor.batch_size)
            )
        )

    def sample_proposals(
        self,
        state: KondoSparseActorState,
        actor_features: Array,
        sampling_keys: Array,
        *,
        action_stack_memory_preparations: tuple[
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            ...,
        ],
    ) -> KondoExecutedActionProposalBatch:
        """Sample one exact unmasked categorical proposal per prepared decision."""

        if type(state) is not KondoSparseActorState:
            raise TypeError("state must be an exact Kondo sparse actor state")
        self._actor._validate_state_static(state)
        if not bool(np.asarray(self._actor._state_valid(state))):
            raise ValueError("proposal sampling requires a valid actor snapshot")
        cfg = self._config.actor
        features = _require_array(
            actor_features,
            name="actor_features",
            shape=(cfg.batch_size, cfg.feature_dim),
            dtype=jnp.float32,
        )
        if not bool(np.asarray(jnp.all(jnp.isfinite(features)))):
            raise ValueError("actor_features must be finite before proposal sampling")
        keys = _require_typed_keys(sampling_keys, batch_size=cfg.batch_size)
        prepared = self._validate_memory_preparations(action_stack_memory_preparations)

        selected_actions = self._sample_actions(state, features, keys)
        behavior = self._actor.behavior_log_probability(
            state,
            features,
            selected_actions,
        )
        actor_state_words = self.actor_state_digest_words(state)
        policy_revision = jnp.full(
            (cfg.batch_size,),
            state.policy_revision,
            dtype=jnp.int32,
        )
        source_words = jnp.stack(
            tuple(self.action_stack_source_state_digest_words(item) for item in prepared)
        )
        preparation_words = jnp.stack(tuple(item.content_tag_words for item in prepared))
        candidate_binding_words = jnp.stack(
            tuple(
                item.memory_candidate_state.action_binding.content_tag_words
                for item in prepared
            )
        )
        decision_identities = jnp.stack(
            tuple(
                item.memory_candidate_state.action_binding.prototype_decision_id
                for item in prepared
            )
        )
        masks = jnp.stack(tuple(item.hard_action_mask for item in prepared))
        bare = KondoExecutedActionProposalBatch(
            actor_features=features,
            sampling_keys=keys,
            actor_state_words=actor_state_words,
            policy_revision=policy_revision,
            selected_actions=selected_actions,
            behavior_log_probability=behavior,
            action_stack_source_state_words=source_words,
            action_stack_memory_preparation_words=preparation_words,
            action_stack_memory_candidate_binding_words=candidate_binding_words,
            action_stack_decision_identities=decision_identities,
            hard_action_masks=masks,
            proposal_digest_words=jnp.zeros(
                (cfg.batch_size, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
        )
        return cast(
            KondoExecutedActionProposalBatch,
            bare.replace(
                proposal_digest_words=self.rederive_proposal_digest_words(bare)
            ),
        )

    def _validate_proposal_static(self, proposal: object) -> KondoExecutedActionProposalBatch:
        if type(proposal) is not KondoExecutedActionProposalBatch:
            raise TypeError("proposal must be an exact Kondo executed-action proposal")
        exact = proposal
        cfg = self._config.actor
        _require_array(
            exact.actor_features,
            name="proposal.actor_features",
            shape=(cfg.batch_size, cfg.feature_dim),
            dtype=jnp.float32,
        )
        _require_typed_keys(exact.sampling_keys, batch_size=cfg.batch_size)
        for name, shape, dtype in (
            ("actor_state_words", (_DIGEST_WORDS,), jnp.uint32),
            ("policy_revision", (cfg.batch_size,), jnp.int32),
            ("selected_actions", (cfg.batch_size,), jnp.int32),
            ("behavior_log_probability", (cfg.batch_size,), jnp.float32),
            (
                "action_stack_source_state_words",
                (cfg.batch_size, _DIGEST_WORDS),
                jnp.uint32,
            ),
            (
                "action_stack_memory_preparation_words",
                (cfg.batch_size, _DIGEST_WORDS),
                jnp.uint32,
            ),
            (
                "action_stack_memory_candidate_binding_words",
                (cfg.batch_size, _DIGEST_WORDS),
                jnp.uint32,
            ),
            ("action_stack_decision_identities", (cfg.batch_size, 4), jnp.uint32),
            ("hard_action_masks", (cfg.batch_size, cfg.action_count), jnp.bool_),
            ("proposal_digest_words", (cfg.batch_size, _DIGEST_WORDS), jnp.uint32),
        ):
            _require_array(
                getattr(exact, name),
                name=f"proposal.{name}",
                shape=shape,
                dtype=dtype,
            )
        return exact

    def rederive_proposal_digest_words(
        self,
        proposal: KondoExecutedActionProposalBatch,
    ) -> Array:
        """Recompute each row's unkeyed proposal integrity binding."""

        exact = self._validate_proposal_static(proposal)
        return jnp.stack(
            tuple(
                _action_stack_tree_digest(
                    KONDO_EXECUTED_ACTION_PROPOSAL_SCHEMA,
                    "proposal-row",
                    row,
                    exact.actor_features[row],
                    exact.sampling_keys[row],
                    exact.actor_state_words,
                    exact.policy_revision[row],
                    exact.selected_actions[row],
                    exact.behavior_log_probability[row],
                    exact.action_stack_source_state_words[row],
                    exact.action_stack_memory_preparation_words[row],
                    exact.action_stack_memory_candidate_binding_words[row],
                    exact.action_stack_decision_identities[row],
                    exact.hard_action_masks[row],
                )
                for row in range(self._config.actor.batch_size)
            )
        )

    def _validate_compact_adoption_static(
        self,
        receipt: object,
    ) -> KondoExecutedActionCompactAdoptionBatch:
        if type(receipt) is not KondoExecutedActionCompactAdoptionBatch:
            raise TypeError("receipt must be an exact compact adoption batch")
        exact = receipt
        batch_size = self._config.actor.batch_size
        action_count = self._config.actor.action_count
        for name, shape, dtype in (
            ("source_state_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("memory_preparation_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            (
                "memory_candidate_binding_words",
                (batch_size, _DIGEST_WORDS),
                jnp.uint32,
            ),
            ("decision_identities", (batch_size, 4), jnp.uint32),
            ("final_action_owner_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("finalization_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("integrity_receipt_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("adoption_result_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("destination_state_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("planner_candidate_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
            ("planner_actions_before_mask", (batch_size,), jnp.int32),
            ("final_actions", (batch_size,), jnp.int32),
            ("hard_action_masks", (batch_size, action_count), jnp.bool_),
            ("planner_consumed", (batch_size,), jnp.bool_),
            ("adoption_applied", (batch_size,), jnp.bool_),
            ("content_tag_words", (batch_size, _DIGEST_WORDS), jnp.uint32),
        ):
            _require_array(
                getattr(exact, name),
                name=f"compact_adoption.{name}",
                shape=shape,
                dtype=dtype,
            )
        return exact

    def _compact_adoption_tags(
        self,
        receipt: KondoExecutedActionCompactAdoptionBatch,
    ) -> Array:
        exact = self._validate_compact_adoption_static(receipt)
        return jnp.stack(
            tuple(
                _action_stack_tree_digest(
                    KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                    "compact-adoption-row",
                    row,
                    exact.source_state_words[row],
                    exact.memory_preparation_words[row],
                    exact.memory_candidate_binding_words[row],
                    exact.decision_identities[row],
                    exact.final_action_owner_words[row],
                    exact.finalization_words[row],
                    exact.integrity_receipt_words[row],
                    exact.adoption_result_words[row],
                    exact.destination_state_words[row],
                    exact.planner_candidate_words[row],
                    exact.planner_actions_before_mask[row],
                    exact.final_actions[row],
                    exact.hard_action_masks[row],
                    exact.planner_consumed[row],
                    exact.adoption_applied[row],
                )
                for row in range(self._config.actor.batch_size)
            )
        )

    def compact_adoption_receipts(
        self,
        proposal: KondoExecutedActionProposalBatch,
        adopted_results: tuple[ExternalLearnedStateLiveMemoryActionStackResult, ...],
    ) -> KondoExecutedActionCompactAdoptionBatch:
        """Reconstruct public adoptions once and retain only immutable proof words."""

        exact_proposal = self._validate_proposal_static(proposal)
        batch_size = self._config.actor.batch_size
        if type(adopted_results) is not tuple or len(adopted_results) != batch_size:
            raise TypeError("adopted_results must be an exact fixed-batch tuple")
        if not bool(
            np.asarray(
                jnp.all(
                    exact_proposal.proposal_digest_words
                    == self.rederive_proposal_digest_words(exact_proposal)
                )
            )
        ):
            raise ValueError("proposal integrity must rederive before receipt issuance")
        if not bool(np.asarray(jnp.all(exact_proposal.hard_action_masks))):
            raise ValueError("compact receipt issuance supports all-true masks only")

        rows: list[tuple[Array, ...]] = []
        for row, supplied in enumerate(adopted_results):
            if type(supplied) is not ExternalLearnedStateLiveMemoryActionStackResult:
                raise TypeError(f"adopted_results[{row}] must be an exact result")
            adapter = self._action_stacks[row]
            finalized = supplied.finalized
            prepared = finalized.memory_preparation
            if not adapter._memory_preparation_static_contract_valid(prepared):
                raise ValueError(f"adopted_results[{row}] has a malformed preparation")
            memory_binding = prepared.memory_candidate_state.action_binding
            final_binding = finalized.final_action_binding
            try:
                reconstructed_receipt = adapter.integrity_receipt(finalized)
                reconstructed = adapter.adopt_finalized_transition(
                    prepared.source_state,
                    finalized,
                    reconstructed_receipt,
                )
                exact = _action_stack_tree_equal(supplied, reconstructed)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                exact = jnp.asarray(False, dtype=jnp.bool_)
                reconstructed = supplied
            relations = (
                exact
                & reconstructed.diagnostics.transaction_applied
                & prepared.preparation_valid
                & jnp.array_equal(
                    prepared.content_tag_words,
                    adapter._memory_preparation_tag(prepared),
                )
                & adapter.state_valid(prepared.source_state)
                & adapter.state_valid(prepared.memory_candidate_state)
                & adapter.state_valid(supplied.state)
                & jnp.array_equal(
                    exact_proposal.action_stack_source_state_words[row],
                    self.action_stack_source_state_digest_words(prepared),
                )
                & jnp.array_equal(
                    exact_proposal.action_stack_memory_preparation_words[row],
                    prepared.content_tag_words,
                )
                & jnp.array_equal(
                    exact_proposal.action_stack_memory_candidate_binding_words[row],
                    memory_binding.content_tag_words,
                )
                & jnp.array_equal(
                    exact_proposal.action_stack_decision_identities[row],
                    memory_binding.prototype_decision_id,
                )
                & jnp.array_equal(
                    exact_proposal.hard_action_masks[row],
                    final_binding.hard_action_mask,
                )
                & final_binding.planner_consumed
                & jnp.array_equal(
                    final_binding.planner_candidate_words,
                    exact_proposal.proposal_digest_words[row],
                )
                & (final_binding.planner_action_before_mask == exact_proposal.selected_actions[row])
                & (final_binding.final_action == exact_proposal.selected_actions[row])
                & (supplied.state.action_binding.final_action == final_binding.final_action)
                & jnp.array_equal(
                    supplied.state.action_binding.planner_candidate_words,
                    final_binding.planner_candidate_words,
                )
            )
            if not bool(np.asarray(relations)):
                raise ValueError(f"adopted_results[{row}] is not exact actor-routed P")
            rows.append(
                (
                    self.action_stack_source_state_digest_words(prepared),
                    prepared.content_tag_words,
                    memory_binding.content_tag_words,
                    memory_binding.prototype_decision_id,
                    final_binding.final_action_owner_words,
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "finalization",
                        finalized,
                    ),
                    reconstructed_receipt.content_tag_words,
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "adoption-result",
                        supplied,
                    ),
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "destination-state",
                        supplied.state,
                    ),
                    final_binding.planner_candidate_words,
                    final_binding.planner_action_before_mask,
                    final_binding.final_action,
                    final_binding.hard_action_mask,
                    final_binding.planner_consumed,
                    reconstructed.diagnostics.transaction_applied,
                )
            )

        def stack(index: int) -> Array:
            return jnp.stack(tuple(item[index] for item in rows))

        bare = KondoExecutedActionCompactAdoptionBatch(
            source_state_words=stack(0).astype(jnp.uint32),
            memory_preparation_words=stack(1).astype(jnp.uint32),
            memory_candidate_binding_words=stack(2).astype(jnp.uint32),
            decision_identities=stack(3).astype(jnp.uint32),
            final_action_owner_words=stack(4).astype(jnp.uint32),
            finalization_words=stack(5).astype(jnp.uint32),
            integrity_receipt_words=stack(6).astype(jnp.uint32),
            adoption_result_words=stack(7).astype(jnp.uint32),
            destination_state_words=stack(8).astype(jnp.uint32),
            planner_candidate_words=stack(9).astype(jnp.uint32),
            planner_actions_before_mask=stack(10).astype(jnp.int32),
            final_actions=stack(11).astype(jnp.int32),
            hard_action_masks=stack(12).astype(jnp.bool_),
            planner_consumed=stack(13).astype(jnp.bool_),
            adoption_applied=stack(14).astype(jnp.bool_),
            content_tag_words=jnp.zeros(
                (batch_size, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
        )
        return cast(
            KondoExecutedActionCompactAdoptionBatch,
            bare.replace(content_tag_words=self._compact_adoption_tags(bare)),
        )

    def _validate_step_inputs(
        self,
        state: KondoSparseActorState,
        proposal: KondoExecutedActionProposalBatch,
        adopted_results: object,
        next_preparations: object,
        protected: KondoActorProtectedInputs,
    ) -> tuple[
        KondoExecutedActionProposalBatch,
        tuple[ExternalLearnedStateLiveMemoryActionStackResult, ...],
        tuple[ExternalLearnedStateLiveMemoryActionStackMemoryPreparation, ...],
    ]:
        if type(state) is not KondoSparseActorState:
            raise TypeError("state must be an exact Kondo sparse actor state")
        self._actor._validate_state_static(state)
        exact_proposal = self._validate_proposal_static(proposal)
        batch_size = self._config.actor.batch_size
        if type(adopted_results) is not tuple or len(adopted_results) != batch_size:
            raise TypeError("adopted_results must be an exact fixed-batch tuple")
        if type(next_preparations) is not tuple or len(next_preparations) != batch_size:
            raise TypeError("next_preparations must be an exact fixed-batch tuple")
        exact_adopted = cast(
            tuple[ExternalLearnedStateLiveMemoryActionStackResult, ...],
            adopted_results,
        )
        exact_next = cast(
            tuple[ExternalLearnedStateLiveMemoryActionStackMemoryPreparation, ...],
            next_preparations,
        )
        for row, result in enumerate(exact_adopted):
            if type(result) is not ExternalLearnedStateLiveMemoryActionStackResult:
                raise TypeError(f"adopted_results[{row}] must be an exact action-stack result")
        for row, prepared in enumerate(exact_next):
            if type(prepared) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
                raise TypeError(
                    f"next_preparations[{row}] must be an exact memory preparation"
                )
            if not self._action_stacks[row]._memory_preparation_static_contract_valid(
                prepared
            ):
                raise ValueError(f"next_preparations[{row}] has a malformed static contract")
        self._actor._validate_protected_static(protected)
        return exact_proposal, exact_adopted, exact_next

    def _finish_actor_step(
        self,
        state: KondoSparseActorState,
        proposal: KondoExecutedActionProposalBatch,
        protected: KondoActorProtectedInputs,
        diagnostics: KondoExecutedActionLineageDiagnostics,
        *,
        adoption_integrity_reconstructions: int,
        compact_adoption_receipt_validations: int,
    ) -> KondoExecutedActionLineageResult:
        """Execute the one actor call after a full or compact lineage audit."""

        batch_size = self._config.actor.batch_size
        actor_eligible = diagnostics.actor_eligible
        # Invalid lineage rows are made finite and action-valid before entering
        # Kondo. Protected critic/baseline/return/safety arrays are never
        # gathered or sanitized: they remain byte-exact at full batch shape.
        sanitized_features = jnp.where(
            actor_eligible[:, None],
            proposal.actor_features,
            jnp.zeros_like(proposal.actor_features),
        )
        sanitized_actions = jnp.where(
            actor_eligible,
            proposal.selected_actions,
            jnp.zeros((batch_size,), dtype=jnp.int32),
        ).astype(jnp.int32)
        sanitized_behavior = self._actor.behavior_log_probability(
            state,
            sanitized_features,
            sanitized_actions,
        )
        actor_batch = KondoSparseActorBatch(
            actor_features=sanitized_features,
            actions=sanitized_actions,
            action_identity=sanitized_actions,
            policy_revision=jnp.full(
                (batch_size,),
                state.policy_revision,
                dtype=jnp.int32,
            ),
            behavior_log_probability=sanitized_behavior,
            valid_mask=actor_eligible,
            force_keep_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
            protected=protected,
        )
        actor_result = self._actor.step(state, actor_batch)
        zero = jnp.asarray(0, dtype=jnp.int32)
        work = KondoExecutedActionLineageWork(
            actor_step_calls=jnp.asarray(1, dtype=jnp.int32),
            adoption_integrity_reconstructions=jnp.asarray(
                adoption_integrity_reconstructions,
                dtype=jnp.int32,
            ),
            compact_adoption_receipt_validations=jnp.asarray(
                compact_adoption_receipt_validations,
                dtype=jnp.int32,
            ),
            action_stack_learner_evaluations=zero,
            planner_model_evaluations=zero,
        )
        return KondoExecutedActionLineageResult(
            actor_result=actor_result,
            diagnostics=diagnostics,
            protected=protected,
            work=work,
        )

    def step(
        self,
        state: KondoSparseActorState,
        proposal: KondoExecutedActionProposalBatch,
        adopted_results: tuple[ExternalLearnedStateLiveMemoryActionStackResult, ...],
        next_preparations: tuple[
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            ...,
        ],
        protected: KondoActorProtectedInputs,
    ) -> KondoExecutedActionLineageResult:
        """Execute Kondo once using only rows with exact executed-P lineage."""

        proposal, adopted_results, next_preparations = self._validate_step_inputs(
            state,
            proposal,
            adopted_results,
            next_preparations,
            protected,
        )
        cfg = self._config.actor
        batch_size = cfg.batch_size
        rederived = self.rederive_proposal_digest_words(proposal)
        proposal_integrity = jnp.all(
            proposal.proposal_digest_words == rederived,
            axis=1,
        )
        actor_snapshot_scalar = jnp.array_equal(
            proposal.actor_state_words,
            self.actor_state_digest_words(state),
        )
        actor_snapshot_exact = jnp.full(
            (batch_size,),
            actor_snapshot_scalar,
            dtype=jnp.bool_,
        )
        policy_revision_exact = proposal.policy_revision == state.policy_revision
        expected_actions = self._sample_actions(
            state,
            proposal.actor_features,
            proposal.sampling_keys,
        )
        proposal_sample_exact = proposal.selected_actions == expected_actions
        expected_behavior = self._actor.behavior_log_probability(
            state,
            proposal.actor_features,
            expected_actions,
        )
        behavior_exact = _bitwise_float32_equal(
            proposal.behavior_log_probability,
            expected_behavior,
        )
        hard_all_true = jnp.all(proposal.hard_action_masks, axis=1)

        action_stack_source_exact: list[Array] = []
        preparation_integrity: list[Array] = []
        candidate_binding_exact: list[Array] = []
        adoption_result_exact: list[Array] = []
        adoption_applied: list[Array] = []
        actor_path_selected: list[Array] = []
        planner_candidate_exact: list[Array] = []
        planner_before_exact: list[Array] = []
        final_action_exact: list[Array] = []
        next_integrity: list[Array] = []
        next_source_exact: list[Array] = []
        next_action_exact: list[Array] = []

        false = jnp.asarray(False, dtype=jnp.bool_)
        for row in range(batch_size):
            adapter = self._action_stacks[row]
            supplied = adopted_results[row]
            finalized = supplied.finalized
            prepared = finalized.memory_preparation
            binding = prepared.memory_candidate_state.action_binding
            source_exact = (
                jnp.array_equal(
                    proposal.action_stack_source_state_words[row],
                    self.action_stack_source_state_digest_words(prepared),
                )
                & jnp.array_equal(
                    proposal.action_stack_memory_preparation_words[row],
                    prepared.content_tag_words,
                )
                & jnp.array_equal(
                    proposal.action_stack_decision_identities[row],
                    binding.prototype_decision_id,
                )
                & jnp.array_equal(
                    proposal.hard_action_masks[row],
                    prepared.hard_action_mask,
                )
            )
            preparation_exact = (
                prepared.preparation_valid
                & jnp.array_equal(
                    prepared.content_tag_words,
                    adapter._memory_preparation_tag(prepared),
                )
            )
            candidate_exact = (
                jnp.array_equal(
                    proposal.action_stack_memory_candidate_binding_words[row],
                    binding.content_tag_words,
                )
                & adapter.state_valid(prepared.memory_candidate_state)
            )

            supplied_exact = false
            supplied_applied = false
            try:
                reconstructed_receipt = adapter.integrity_receipt(finalized)
                reconstructed = adapter.adopt_finalized_transition(
                    prepared.source_state,
                    finalized,
                    reconstructed_receipt,
                )
                supplied_exact = _action_stack_tree_equal(supplied, reconstructed)
                supplied_applied = reconstructed.diagnostics.transaction_applied
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                supplied_exact = false
                supplied_applied = false

            final_binding = finalized.final_action_binding
            actor_selected = final_binding.planner_consumed
            planner_words_exact = jnp.array_equal(
                final_binding.planner_candidate_words,
                proposal.proposal_digest_words[row],
            )
            planner_action_exact = (
                final_binding.planner_action_before_mask
                == proposal.selected_actions[row]
            )
            executed_exact = (
                final_binding.final_action == proposal.selected_actions[row]
            ) & (
                supplied.state.action_binding.final_action
                == proposal.selected_actions[row]
            )

            following = next_preparations[row]
            following_integrity = (
                following.preparation_valid
                & jnp.array_equal(
                    following.content_tag_words,
                    self.rederive_next_preparation_digest_words(following, row=row),
                )
            )
            following_source = _action_stack_tree_equal(
                following.source_state,
                supplied.state,
            )
            following_action = (
                following.transition.action == proposal.selected_actions[row]
            ) & (
                following.transition.action == final_binding.final_action
            )

            action_stack_source_exact.append(source_exact)
            preparation_integrity.append(preparation_exact)
            candidate_binding_exact.append(candidate_exact)
            adoption_result_exact.append(supplied_exact)
            adoption_applied.append(supplied_applied)
            actor_path_selected.append(actor_selected)
            planner_candidate_exact.append(planner_words_exact)
            planner_before_exact.append(planner_action_exact)
            final_action_exact.append(executed_exact)
            next_integrity.append(following_integrity)
            next_source_exact.append(following_source)
            next_action_exact.append(following_action)

        source_exact_array = jnp.stack(tuple(action_stack_source_exact))
        preparation_array = jnp.stack(tuple(preparation_integrity))
        candidate_array = jnp.stack(tuple(candidate_binding_exact))
        adoption_exact_array = jnp.stack(tuple(adoption_result_exact))
        adoption_applied_array = jnp.stack(tuple(adoption_applied))
        actor_path_array = jnp.stack(tuple(actor_path_selected))
        planner_words_array = jnp.stack(tuple(planner_candidate_exact))
        planner_before_array = jnp.stack(tuple(planner_before_exact))
        final_action_array = jnp.stack(tuple(final_action_exact))
        next_integrity_array = jnp.stack(tuple(next_integrity))
        next_source_array = jnp.stack(tuple(next_source_exact))
        next_action_array = jnp.stack(tuple(next_action_exact))
        actor_eligible = (
            proposal_integrity
            & actor_snapshot_exact
            & policy_revision_exact
            & proposal_sample_exact
            & behavior_exact
            & source_exact_array
            & preparation_array
            & candidate_array
            & hard_all_true
            & adoption_exact_array
            & adoption_applied_array
            & actor_path_array
            & planner_words_array
            & planner_before_array
            & final_action_array
            & next_integrity_array
            & next_source_array
            & next_action_array
        )

        diagnostics = KondoExecutedActionLineageDiagnostics(
            proposal_integrity_rederived=proposal_integrity,
            actor_snapshot_exact=actor_snapshot_exact,
            policy_revision_exact=policy_revision_exact,
            proposal_sample_exact=proposal_sample_exact,
            behavior_log_probability_exact=behavior_exact,
            action_stack_source_exact=source_exact_array,
            memory_preparation_integrity_rederived=preparation_array,
            memory_candidate_binding_exact=candidate_array,
            hard_action_mask_exact_all_true=hard_all_true,
            lineage_mode=jnp.zeros((batch_size,), dtype=jnp.int32),
            historic_adoption_reconstructed=adoption_exact_array,
            compact_carry_forward_certificate_valid=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            adoption_result_exact=adoption_exact_array,
            adoption_transaction_applied=adoption_applied_array,
            actor_path_selected=actor_path_array,
            planner_candidate_exact=planner_words_array,
            planner_before_mask_exact=planner_before_array,
            final_action_exact=final_action_array,
            next_preparation_integrity_rederived=next_integrity_array,
            next_preparation_source_exact=next_source_array,
            next_transition_action_exact=next_action_array,
            actor_eligible=actor_eligible,
        )
        return self._finish_actor_step(
            state,
            proposal,
            protected,
            diagnostics,
            adoption_integrity_reconstructions=batch_size,
            compact_adoption_receipt_validations=0,
        )

    def preflight_compact(
        self,
        state: KondoSparseActorState,
        proposal: KondoExecutedActionProposalBatch,
        compact_adoptions: KondoExecutedActionCompactAdoptionBatch,
        next_preparations: tuple[
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            ...,
        ],
    ) -> KondoExecutedActionLineageDiagnostics:
        """Validate compact pending proof without entering an actor backward.

        Receipt issuance already reconstructed the public child adoption while
        its full transient result existed.  This method rebinds that proof to
        each current action-stack state and following transition without
        persisting a second mutable owner tree.  Pair-atomic outer routes can
        require every row before they call :meth:`step_compact`.
        """

        if type(state) is not KondoSparseActorState:
            raise TypeError("state must be an exact Kondo sparse actor state")
        self._actor._validate_state_static(state)
        exact_proposal = self._validate_proposal_static(proposal)
        compact = self._validate_compact_adoption_static(compact_adoptions)
        batch_size = self._config.actor.batch_size
        if type(next_preparations) is not tuple or len(next_preparations) != batch_size:
            raise TypeError("next_preparations must be an exact fixed-batch tuple")
        for row, following in enumerate(next_preparations):
            if type(following) is not (
                ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
            ):
                raise TypeError(f"next_preparations[{row}] must be an exact preparation")
            if not self._action_stacks[row]._memory_preparation_static_contract_valid(
                following
            ):
                raise ValueError(f"next_preparations[{row}] has a malformed contract")

        rederived = self.rederive_proposal_digest_words(exact_proposal)
        proposal_integrity = jnp.all(
            exact_proposal.proposal_digest_words == rederived,
            axis=1,
        )
        compact_integrity = jnp.all(
            compact.content_tag_words == self._compact_adoption_tags(compact),
            axis=1,
        )
        actor_snapshot_exact = jnp.full(
            (batch_size,),
            jnp.array_equal(
                exact_proposal.actor_state_words,
                self.actor_state_digest_words(state),
            ),
            dtype=jnp.bool_,
        )
        policy_revision_exact = exact_proposal.policy_revision == state.policy_revision
        expected_actions = self._sample_actions(
            state,
            exact_proposal.actor_features,
            exact_proposal.sampling_keys,
        )
        proposal_sample_exact = exact_proposal.selected_actions == expected_actions
        expected_behavior = self._actor.behavior_log_probability(
            state,
            exact_proposal.actor_features,
            expected_actions,
        )
        behavior_exact = _bitwise_float32_equal(
            exact_proposal.behavior_log_probability,
            expected_behavior,
        )
        hard_all_true = jnp.all(exact_proposal.hard_action_masks, axis=1)

        source_checks: list[Array] = []
        preparation_checks: list[Array] = []
        candidate_checks: list[Array] = []
        adoption_checks: list[Array] = []
        adoption_applied: list[Array] = []
        actor_selected: list[Array] = []
        planner_words: list[Array] = []
        planner_actions: list[Array] = []
        final_actions: list[Array] = []
        next_integrity: list[Array] = []
        next_source: list[Array] = []
        next_action: list[Array] = []
        for row, following in enumerate(next_preparations):
            adapter = self._action_stacks[row]
            current = following.source_state
            binding = current.action_binding
            current_valid = adapter.state_valid(current)
            source_checks.append(
                compact_integrity[row]
                & jnp.array_equal(
                    compact.source_state_words[row],
                    exact_proposal.action_stack_source_state_words[row],
                )
            )
            preparation_checks.append(
                compact_integrity[row]
                & jnp.array_equal(
                    compact.memory_preparation_words[row],
                    exact_proposal.action_stack_memory_preparation_words[row],
                )
            )
            candidate_checks.append(
                compact_integrity[row]
                & jnp.array_equal(
                    compact.memory_candidate_binding_words[row],
                    exact_proposal.action_stack_memory_candidate_binding_words[row],
                )
                & jnp.array_equal(
                    compact.decision_identities[row],
                    exact_proposal.action_stack_decision_identities[row],
                )
            )
            destination_exact = jnp.array_equal(
                compact.destination_state_words[row],
                _action_stack_tree_digest(
                    KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                    "destination-state",
                    current,
                ),
            )
            route_exact = (
                current_valid
                & destination_exact
                & binding.planner_bound
                & binding.planner_consumed
                & jnp.array_equal(
                    binding.planner_candidate_words,
                    compact.planner_candidate_words[row],
                )
                & jnp.array_equal(
                    binding.final_action_owner_words,
                    compact.final_action_owner_words[row],
                )
                & jnp.array_equal(
                    binding.prototype_decision_id,
                    compact.decision_identities[row],
                )
                & jnp.array_equal(
                    binding.hard_action_mask,
                    compact.hard_action_masks[row],
                )
                & jnp.array_equal(
                    compact.hard_action_masks[row],
                    exact_proposal.hard_action_masks[row],
                )
                & (binding.planner_consumed == compact.planner_consumed[row])
                & (binding.planner_action_before_mask == compact.planner_actions_before_mask[row])
                & (binding.final_action == compact.final_actions[row])
                & jnp.any(compact.finalization_words[row] != 0)
                & jnp.any(compact.integrity_receipt_words[row] != 0)
                & jnp.any(compact.adoption_result_words[row] != 0)
            )
            adoption_checks.append(compact_integrity[row] & route_exact)
            adoption_applied.append(
                compact_integrity[row] & compact.adoption_applied[row]
            )
            actor_selected.append(
                compact_integrity[row] & compact.planner_consumed[row]
            )
            planner_words.append(
                jnp.array_equal(
                    compact.planner_candidate_words[row],
                    exact_proposal.proposal_digest_words[row],
                )
            )
            planner_actions.append(
                compact.planner_actions_before_mask[row]
                == exact_proposal.selected_actions[row]
            )
            final_actions.append(
                (compact.final_actions[row] == exact_proposal.selected_actions[row])
                & (binding.final_action == exact_proposal.selected_actions[row])
            )
            following_valid = (
                following.preparation_valid
                & jnp.array_equal(
                    following.content_tag_words,
                    self.rederive_next_preparation_digest_words(following, row=row),
                )
            )
            next_integrity.append(following_valid)
            next_source.append(current_valid & destination_exact)
            next_action.append(
                (following.transition.action == exact_proposal.selected_actions[row])
                & (following.transition.action == compact.final_actions[row])
            )

        source_array = jnp.stack(tuple(source_checks))
        preparation_array = jnp.stack(tuple(preparation_checks))
        candidate_array = jnp.stack(tuple(candidate_checks))
        adoption_array = jnp.stack(tuple(adoption_checks))
        adoption_applied_array = jnp.stack(tuple(adoption_applied))
        actor_selected_array = jnp.stack(tuple(actor_selected))
        planner_words_array = jnp.stack(tuple(planner_words))
        planner_actions_array = jnp.stack(tuple(planner_actions))
        final_actions_array = jnp.stack(tuple(final_actions))
        next_integrity_array = jnp.stack(tuple(next_integrity))
        next_source_array = jnp.stack(tuple(next_source))
        next_action_array = jnp.stack(tuple(next_action))
        actor_eligible = (
            proposal_integrity
            & actor_snapshot_exact
            & policy_revision_exact
            & proposal_sample_exact
            & behavior_exact
            & source_array
            & preparation_array
            & candidate_array
            & hard_all_true
            & adoption_array
            & adoption_applied_array
            & actor_selected_array
            & planner_words_array
            & planner_actions_array
            & final_actions_array
            & next_integrity_array
            & next_source_array
            & next_action_array
        )
        diagnostics = KondoExecutedActionLineageDiagnostics(
            proposal_integrity_rederived=proposal_integrity,
            actor_snapshot_exact=actor_snapshot_exact,
            policy_revision_exact=policy_revision_exact,
            proposal_sample_exact=proposal_sample_exact,
            behavior_log_probability_exact=behavior_exact,
            action_stack_source_exact=source_array,
            memory_preparation_integrity_rederived=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            memory_candidate_binding_exact=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            hard_action_mask_exact_all_true=hard_all_true,
            lineage_mode=jnp.ones((batch_size,), dtype=jnp.int32),
            historic_adoption_reconstructed=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            compact_carry_forward_certificate_valid=adoption_array,
            adoption_result_exact=jnp.zeros((batch_size,), dtype=jnp.bool_),
            adoption_transaction_applied=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            actor_path_selected=actor_selected_array,
            planner_candidate_exact=planner_words_array,
            planner_before_mask_exact=planner_actions_array,
            final_action_exact=final_actions_array,
            next_preparation_integrity_rederived=next_integrity_array,
            next_preparation_source_exact=next_source_array,
            next_transition_action_exact=next_action_array,
            actor_eligible=actor_eligible,
        )
        return diagnostics

    def step_compact(
        self,
        state: KondoSparseActorState,
        proposal: KondoExecutedActionProposalBatch,
        compact_adoptions: KondoExecutedActionCompactAdoptionBatch,
        next_preparations: tuple[
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            ...,
        ],
        protected: KondoActorProtectedInputs,
    ) -> KondoExecutedActionLineageResult:
        """Execute Kondo once after the compact lineage audit."""

        self._actor._validate_protected_static(protected)
        diagnostics = self.preflight_compact(
            state,
            proposal,
            compact_adoptions,
            next_preparations,
        )
        return self._finish_actor_step(
            state,
            proposal,
            protected,
            diagnostics,
            adoption_integrity_reconstructions=0,
            compact_adoption_receipt_validations=self._config.actor.batch_size,
        )
