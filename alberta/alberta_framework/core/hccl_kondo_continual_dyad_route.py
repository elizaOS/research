# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var,union-attr"
"""Actor-owned P over the atomic HCCL continual dyad.

This host/eager L0 route keeps the factorized planner as a learning-only
shadow with ``planning_enabled=False``.  The Kondo actor supplies each next P;
the two live action stacks remain the only Prototype owners.  Pending lineage
contains only an immutable proposal and compact adoption certificate.  Every
successor event consumes that pair before the next actor proposal is sampled.

The nested Kondo actor result remains the sole owner of actor-backward and
``sparks_joy`` semantics.  Protected TD owns only its full-batch backward
telemetry, and this route exposes no second joy alias.  Its hashes are unkeyed
integrity bindings, not authentication or dispatch authority.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
    ExternalLearnedStateLiveMemoryActionStackResult,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    _tree_digest as _action_stack_tree_digest,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    _B0M1_SLOT,
    _BB_SLOT,
    _DIGEST_WORDS,
    _M0B1_SLOT,
    _MM_SLOT,
    _N_ACTIONS,
    _N_AGENTS,
    _PP_SLOT,
    _TOKEN_NBYTES,
    HCCLContinualDyadActionBinding,
    HCCLContinualDyadState,
    HCCLContinualDyadThroughMemoryAgent,
    HCCLContinualDyadTransaction,
    HCCLContinualDyadTransactionConfig,
    _contains_tracer,
    _require_array,
    _signals_at,
    _tree_digest,
    _tree_exact_equal,
    _words_token,
)
from alberta_framework.core.hccl_memory_credit_estimands import (
    HCCLMemoryCreditEstimandPanel,
    derive_hccl_memory_credit_estimands,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterResult,
)
from alberta_framework.core.kondo_executed_action_lineage_bridge import (
    KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
    KondoExecutedActionCompactAdoptionBatch,
    KondoExecutedActionLineageBridge,
    KondoExecutedActionLineageBridgeConfig,
    KondoExecutedActionLineageResult,
    KondoExecutedActionProposalBatch,
)
from alberta_framework.core.kondo_protected_td import (
    KondoProtectedTDBatch,
    KondoProtectedTDConfig,
    KondoProtectedTDLearner,
    KondoProtectedTDResult,
    KondoProtectedTDState,
)
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorParameters,
    KondoSparseActorState,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerPlannerState,
    PrototypeFactorizedPartnerTransitionResult,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA = (
    "alberta.hccl-kondo-continual-dyad-route.v3"
)
HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATE_SCHEMA = (
    "alberta.hccl-kondo-continual-dyad-route-state.v3"
)
HCCL_KONDO_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA = (
    "alberta.hccl-kondo-continual-dyad-through-memory.v3"
)
HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS = (
    "l0-development-hccl-kondo-actor-owned-p"
)
HCCL_KONDO_CONTINUAL_DYAD_ROUTE_EVIDENCE_LEVEL = "L0"

__all__ = (
    "HCCL_KONDO_CONTINUAL_DYAD_ROUTE_EVIDENCE_LEVEL",
    "HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA",
    "HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATE_SCHEMA",
    "HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS",
    "HCCL_KONDO_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA",
    "HCCLKondoContinualDyadEventResult",
    "HCCLKondoContinualDyadEventWork",
    "HCCLKondoContinualDyadRoute",
    "HCCLKondoContinualDyadRouteConfig",
    "HCCLKondoContinualDyadState",
)

_ACTOR_FEATURE_DIM = 23


def _bool(value: object) -> bool:
    return bool(np.asarray(value))


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLKondoContinualDyadRouteConfig:
    """Exact dyad, actor-lineage, and route-owned protected TD configuration."""

    dyad: HCCLContinualDyadTransactionConfig
    lineage: KondoExecutedActionLineageBridgeConfig
    protected_td: KondoProtectedTDConfig

    def __post_init__(self) -> None:
        if type(self.dyad) is not HCCLContinualDyadTransactionConfig:
            raise TypeError("dyad must be an exact continual-dyad config")
        if type(self.lineage) is not KondoExecutedActionLineageBridgeConfig:
            raise TypeError("lineage must be an exact Kondo lineage config")
        if type(self.protected_td) is not KondoProtectedTDConfig:
            raise TypeError("protected_td must be an exact protected TD config")
        if self.dyad.planner.planning_enabled is not False:
            raise ValueError("the factorized planner must remain a disabled shadow")
        actor = self.lineage.actor
        if actor.batch_size != _N_AGENTS:
            raise ValueError("the Kondo actor must expose exactly two rows")
        if actor.action_count != _N_ACTIONS:
            raise ValueError("the Kondo actor must expose exactly two actions")
        if actor.feature_dim != _ACTOR_FEATURE_DIM:
            raise ValueError("actor features must be the 23-wide post-memory base")
        protected = self.protected_td
        if protected.batch_size != _N_AGENTS:
            raise ValueError("protected TD must expose exactly two full-batch rows")
        if protected.feature_dim != _ACTOR_FEATURE_DIM:
            raise ValueError("protected TD features must be the 23-wide post-memory base")
        if protected.action_count != _N_ACTIONS:
            raise ValueError("protected TD must expose exactly two actions")
        if actor.critic_dim != protected.feature_dim:
            raise ValueError("actor critic features must equal protected TD features")
        if actor.safety_dim != protected.feature_dim:
            raise ValueError("actor safety features must equal protected TD features")
        required_updates = self.dyad.hccl.world_config.maximum_committed_transitions - 1
        if protected.max_updates < required_updates:
            raise ValueError("protected TD max_updates must cover every recurring event")
        if actor.gate.max_screenings < required_updates:
            raise ValueError("Kondo gate max_screenings must cover every recurring event")
        rows = self.lineage.action_stack_rows
        if rows is None or len(rows) != _N_AGENTS:
            raise ValueError("lineage must declare both exact action-stack rows")
        if rows[0].to_config() != self.dyad.agent_0.to_config():
            raise ValueError("lineage row 0 must equal dyad agent 0")
        if rows[1].to_config() != self.dyad.agent_1.to_config():
            raise ValueError("lineage row 1 must equal dyad agent 1")
        if self.lineage.action_stack.to_config() != self.dyad.agent_0.to_config():
            raise ValueError("lineage primary action stack must equal row 0")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA,
            "type": type(self).__name__,
            "dyad": self.dyad.to_config(),
            "lineage": self.lineage.to_config(),
            "protected_td": self.protected_td.to_config(),
            "mechanism_status": HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS,
            "evidence_level": HCCL_KONDO_CONTINUAL_DYAD_ROUTE_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "shadow_planner_enabled": True,
            "shadow_planner_planning_enabled": False,
            "shadow_planner_completed_transition_calls_per_event": 1,
            "actor_batch_rows": _N_AGENTS,
            "actor_feature_source": "post-memory-Prototype-current-raw-observation-23",
            "caller_memory_event_inputs": False,
            "memory_event_input_authority": "exact-causal-core-event-and-agent",
            "memory_event_input_derivations_per_event": _N_AGENTS,
            "caller_protected_actor_inputs": False,
            "protected_actor_input_authority": (
                "exact-pending-P-and-current-PP-transition"
            ),
            "protected_current_features": "exact-pending-P-actor-features",
            "protected_next_features": (
                "current-PP-post-memory-Prototype-current-raw-observation"
            ),
            "protected_full_batch_rows": _N_AGENTS,
            "protected_reward_heads": 1,
            "protected_cost_heads": 1,
            "protected_updates_per_recurring_event": 1,
            "protected_updates_on_event0": 0,
            "actor_and_protected_clocks_bound_to_route": True,
            "route_checkpoint_supported": False,
            "protected_td_checkpoint_is_route_checkpoint": False,
            "hard_action_mask_support": "exact-all-true-only",
            "pending_lineage": "proposal-plus-compact-adoption-certificate-only",
            "pending_consumed_before_new_sampling": True,
            "recurring_events_supported": True,
            "pair_atomic_compact_preflight": True,
            "public_child_adoptions_per_event": [1, 1],
            "outer_joy_alias": False,
            "caller_authenticated": False,
            "dispatch_authority": False,
            "physical_execution_authenticated": False,
            "safety_authority": False,
            "artifact_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLKondoContinualDyadRouteConfig:
        if type(payload) is not dict:
            raise TypeError("route config must be an exact dict")
        dyad_payload = payload.get("dyad")
        lineage_payload = payload.get("lineage")
        protected_payload = payload.get("protected_td")
        if (
            type(dyad_payload) is not dict
            or type(lineage_payload) is not dict
            or type(protected_payload) is not dict
        ):
            raise ValueError("route child configs must be exact dicts")
        result = cls(
            dyad=HCCLContinualDyadTransactionConfig.from_config(dyad_payload),
            lineage=KondoExecutedActionLineageBridgeConfig.from_config(
                lineage_payload
            ),
            protected_td=KondoProtectedTDConfig.from_config(protected_payload),
        )
        if result.to_config() != dict(payload):
            raise ValueError("route config fields or fixed semantics differ")
        return result


@chex.dataclass(frozen=True)
class HCCLKondoContinualDyadState:
    """One owner tree: dyad, actor, protected TD, and pending lineage."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    components: HCCLContinualDyadState
    actor_state: KondoSparseActorState
    protected_td_state: KondoProtectedTDState
    pending_proposal: KondoExecutedActionProposalBatch | None
    pending_compact_adoptions: KondoExecutedActionCompactAdoptionBatch | None
    event_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLKondoContinualDyadEventWork:
    """Exact route-level calls; not timing or FLOP measurements."""

    pending_compact_lineage_steps: Int[Array, ""]
    causal_core_memory_event_input_derivations: Int[Array, ""]
    actor_proposal_batches: Int[Array, ""]
    protected_td_full_batch_backward_calls: Int[Array, ""]
    protected_td_rows: Int[Array, ""]
    shadow_planner_completed_transition_calls: Int[Array, ""]
    action_stack_final_bindings: Int[Array, " 2"]
    public_child_adoptions: Int[Array, " 2"]
    compact_certificate_issuances: Int[Array, ""]
    pending_consumed_before_new_sampling: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLKondoContinualDyadEventResult:
    """One transient event result with no duplicate backward-execution alias."""

    state: HCCLKondoContinualDyadState
    proposal: KondoExecutedActionProposalBatch | None
    compact_adoptions: KondoExecutedActionCompactAdoptionBatch | None
    lineage_result: KondoExecutedActionLineageResult | None
    protected_td_result: KondoProtectedTDResult | None
    shadow_planner_result: PrototypeFactorizedPartnerTransitionResult
    agent_0_adoption: ExternalLearnedStateLiveMemoryActionStackResult | None
    agent_1_adoption: ExternalLearnedStateLiveMemoryActionStackResult | None
    work: HCCLKondoContinualDyadEventWork
    update_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


@chex.dataclass(frozen=True)
class _HCCLKondoThroughMemory:
    source_state: HCCLKondoContinualDyadState
    event: HCCLCausalCoreEventReceipt
    binding: HCCLContinualDyadActionBinding
    hccl_result: HCCLWorldAttributionAdapterResult
    memory_credit_panel: HCCLMemoryCreditEstimandPanel
    next_hard_action_masks: Bool[Array, "2 2"]
    agent_0: HCCLContinualDyadThroughMemoryAgent
    agent_1: HCCLContinualDyadThroughMemoryAgent
    preparation_valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


class HCCLKondoContinualDyadRoute:
    """Host-only event0/event1 route from HCCL M to Kondo actor-owned P."""

    def __init__(self, config: HCCLKondoContinualDyadRouteConfig) -> None:
        if type(config) is not HCCLKondoContinualDyadRouteConfig:
            raise TypeError("config must be an exact HCCL/Kondo route config")
        self._config = config
        self._dyad = HCCLContinualDyadTransaction(config.dyad)
        self._lineage = KondoExecutedActionLineageBridge(config.lineage)
        self._protected_td = KondoProtectedTDLearner(config.protected_td)
        self._config_words = _tree_digest(
            HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA,
            config.to_config(),
        )
        self._config_token = _words_token(self._config_words)

    @property
    def config(self) -> HCCLKondoContinualDyadRouteConfig:
        return self._config

    @property
    def dyad(self) -> HCCLContinualDyadTransaction:
        return self._dyad

    @property
    def lineage(self) -> KondoExecutedActionLineageBridge:
        return self._lineage

    @property
    def protected_td(self) -> KondoProtectedTDLearner:
        return self._protected_td

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLKondoContinualDyadRoute:
        return cls(HCCLKondoContinualDyadRouteConfig.from_config(payload))

    def _state_token(self, state: HCCLKondoContinualDyadState) -> Array:
        return _words_token(
            _tree_digest(
                HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATE_SCHEMA,
                state.config_token,
                state.components,
                state.actor_state,
                state.protected_td_state,
                state.pending_proposal,
                state.pending_compact_adoptions,
                state.event_count,
            )
        )

    def _seal_state(
        self,
        state: HCCLKondoContinualDyadState,
    ) -> HCCLKondoContinualDyadState:
        return cast(
            HCCLKondoContinualDyadState,
            state.replace(content_token=self._state_token(state)),
        )

    def _state_contract(self, state: HCCLKondoContinualDyadState) -> None:
        if type(state) is not HCCLKondoContinualDyadState:
            raise TypeError("state must be an exact HCCL/Kondo route state")
        _require_array(
            state.config_token,
            name="state.config_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.content_token,
            name="state.content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.event_count,
            name="state.event_count",
            shape=(),
            dtype=jnp.int32,
        )

    def state_valid(self, state: HCCLKondoContinualDyadState) -> Bool[Array, ""]:
        """Validate actor-owned P without imposing the shadow's dispatch bit."""

        self._state_contract(state)
        if _contains_tracer(state):
            raise TypeError("HCCL/Kondo route validation is host/eager-only")
        token_valid = np.array_equal(
            np.asarray(state.config_token),
            np.asarray(self._config_token),
        ) and np.array_equal(
            np.asarray(state.content_token),
            np.asarray(self._state_token(state)),
        )
        if not token_valid or int(state.event_count) < 0:
            return jnp.asarray(False, dtype=jnp.bool_)
        components = state.components
        self._dyad._state_contract(components)
        component_token_valid = np.array_equal(
            np.asarray(components.config_token),
            np.asarray(self._dyad._config_token),
        ) and np.array_equal(
            np.asarray(components.content_token),
            np.asarray(self._dyad._state_token(components)),
        )
        actor_valid = _bool(self._lineage.actor._state_valid(state.actor_state))
        protected_valid = _bool(
            self._protected_td.state_valid(state.protected_td_state)
        )
        expected_protected_updates = max(int(state.event_count) - 1, 0)
        protected_clock_valid = (
            int(state.protected_td_state.update_count) == expected_protected_updates
        )
        actor_clock_valid = (
            int(state.actor_state.policy_revision) == expected_protected_updates
        )
        pending_pair = (
            state.pending_proposal is None
            and state.pending_compact_adoptions is None
        ) or (
            state.pending_proposal is not None
            and state.pending_compact_adoptions is not None
        )
        if not (
            component_token_valid
            and actor_valid
            and actor_clock_valid
            and protected_valid
            and protected_clock_valid
            and pending_pair
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        if int(state.event_count) == 0:
            return jnp.asarray(
                state.pending_proposal is None
                and state.pending_compact_adoptions is None
                and _bool(self._dyad.state_valid(components)),
                dtype=jnp.bool_,
            )
        if state.pending_proposal is None or state.pending_compact_adoptions is None:
            return jnp.asarray(False, dtype=jnp.bool_)

        proposal = state.pending_proposal
        compact = state.pending_compact_adoptions
        try:
            self._lineage._validate_proposal_static(proposal)
            self._lineage._validate_compact_adoption_static(compact)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)
        proposal_valid = bool(
            np.all(
                np.asarray(
                    proposal.proposal_digest_words
                    == self._lineage.rederive_proposal_digest_words(proposal)
                )
            )
        )
        expected_actions = self._lineage._sample_actions(
            state.actor_state,
            proposal.actor_features,
            proposal.sampling_keys,
        )
        expected_behavior = self._lineage.actor.behavior_log_probability(
            state.actor_state,
            proposal.actor_features,
            expected_actions,
        )
        proposal_valid = proposal_valid and all(
            (
                np.array_equal(
                    np.asarray(proposal.selected_actions),
                    np.asarray(expected_actions),
                ),
                np.array_equal(
                    np.asarray(
                        jax.lax.bitcast_convert_type(
                            proposal.behavior_log_probability,
                            jnp.uint32,
                        )
                    ),
                    np.asarray(
                        jax.lax.bitcast_convert_type(
                            expected_behavior,
                            jnp.uint32,
                        )
                    ),
                ),
                bool(np.all(np.asarray(proposal.hard_action_masks))),
                bool(np.all(np.asarray(proposal.selected_actions) >= 0)),
                bool(
                    np.all(
                        np.asarray(proposal.selected_actions)
                        < self._config.lineage.actor.action_count
                    )
                ),
            )
        )
        compact_valid = bool(
            np.all(
                np.asarray(
                    compact.content_tag_words
                    == self._lineage._compact_adoption_tags(compact)
                )
            )
        )
        actor_bound = np.array_equal(
            np.asarray(proposal.actor_state_words),
            np.asarray(self._lineage.actor_state_digest_words(state.actor_state)),
        ) and np.all(
            np.asarray(proposal.policy_revision) == int(state.actor_state.policy_revision)
        )
        compact_proposal_bound = all(
            (
                np.array_equal(
                    np.asarray(compact.source_state_words),
                    np.asarray(proposal.action_stack_source_state_words),
                ),
                np.array_equal(
                    np.asarray(compact.memory_preparation_words),
                    np.asarray(proposal.action_stack_memory_preparation_words),
                ),
                np.array_equal(
                    np.asarray(compact.memory_candidate_binding_words),
                    np.asarray(proposal.action_stack_memory_candidate_binding_words),
                ),
                np.array_equal(
                    np.asarray(compact.decision_identities),
                    np.asarray(proposal.action_stack_decision_identities),
                ),
                np.array_equal(
                    np.asarray(compact.planner_candidate_words),
                    np.asarray(proposal.proposal_digest_words),
                ),
                np.array_equal(
                    np.asarray(compact.planner_actions_before_mask),
                    np.asarray(proposal.selected_actions),
                ),
                np.array_equal(
                    np.asarray(compact.final_actions),
                    np.asarray(proposal.selected_actions),
                ),
                np.array_equal(
                    np.asarray(compact.hard_action_masks),
                    np.asarray(proposal.hard_action_masks),
                ),
                bool(np.all(np.asarray(compact.planner_consumed))),
                bool(np.all(np.asarray(compact.adoption_applied))),
            )
        )
        child_valid = all(
            (
                _bool(self._dyad.hccl.state_valid(components.hccl_state)),
                _bool(self._dyad.agent_0.state_valid(components.agent_0_state)),
                _bool(self._dyad.agent_1.state_valid(components.agent_1_state)),
                _bool(self._dyad.context.state_is_valid(components.context_0_state)),
                _bool(self._dyad.context.state_is_valid(components.context_1_state)),
            )
        )
        if not (
            proposal_valid
            and compact_valid
            and actor_bound
            and compact_proposal_bound
            and child_valid
        ):
            return jnp.asarray(False, dtype=jnp.bool_)

        agents = (components.agent_0_state, components.agent_1_state)
        contexts = (components.context_0_state, components.context_1_state)
        planner_agents = (
            components.planner_state.agent_0,
            components.planner_state.agent_1,
        )
        prototypes = tuple(self._dyad._prototype(item) for item in agents)
        physical = self._dyad.hccl.world.observe(components.hccl_state.world_state)
        route_bound = True
        for row, (agent, context, planner_agent, prototype) in enumerate(
            zip(agents, contexts, planner_agents, prototypes, strict=True)
        ):
            coordinator = agent.coordinator_state
            binding = agent.action_binding
            expected_raw = self._dyad._composed_observation(physical[row], context)
            expected_base = jnp.concatenate(
                (expected_raw, coordinator.builder_state.hidden)
            ).astype(jnp.float32)
            selected = proposal.selected_actions[row]
            cache = planner_agent.cache
            destination_words = _action_stack_tree_digest(
                KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                "destination-state",
                agent,
            )
            route_bound = route_bound and all(
                (
                    np.array_equal(
                        np.asarray(coordinator.current_raw_observation),
                        np.asarray(expected_raw),
                    ),
                    np.array_equal(
                        np.asarray(coordinator.current_representation),
                        np.asarray(expected_base),
                    ),
                    np.array_equal(
                        np.asarray(prototype.current_raw_observation),
                        np.asarray(expected_base),
                    ),
                    np.array_equal(
                        np.asarray(coordinator.event_words),
                        np.asarray(components.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(context.context.step_words),
                        np.asarray(components.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(prototype.step_words),
                        np.asarray(components.hccl_state.world_state.step_words),
                    ),
                    bool(binding.available),
                    bool(binding.planner_bound),
                    bool(binding.planner_consumed),
                    int(binding.planner_action_before_mask) == int(selected),
                    int(binding.final_action) == int(selected),
                    np.array_equal(
                        np.asarray(binding.planner_candidate_words),
                        np.asarray(proposal.proposal_digest_words[row]),
                    ),
                    not bool(cache.planner_consumed),
                    int(cache.base_action) == int(selected),
                    int(cache.base_action_guard)
                    == int(jnp.bitwise_not(selected)),
                    int(cache.effective_action) == int(selected),
                    int(prototype.current_action) == int(selected),
                    np.array_equal(
                        np.asarray(compact.destination_state_words[row]),
                        np.asarray(destination_words),
                    ),
                    np.array_equal(
                        np.asarray(compact.planner_candidate_words[row]),
                        np.asarray(proposal.proposal_digest_words[row]),
                    ),
                    int(compact.final_actions[row]) == int(selected),
                    bool(compact.planner_consumed[row]),
                    bool(compact.adoption_applied[row]),
                    bool(np.all(np.asarray(proposal.hard_action_masks[row]))),
                    np.array_equal(
                        np.asarray(proposal.hard_action_masks[row]),
                        np.asarray(compact.hard_action_masks[row]),
                    ),
                )
            )
        authenticated = self._dyad.planner.authenticate_pair(
            components.planner_state,
            prototypes[0],
            prototypes[1],
        )
        return jnp.asarray(
            route_bound and bool(np.all(np.asarray(authenticated))),
            dtype=jnp.bool_,
        )

    def init(
        self,
        dyad_key: Array,
        actor_parameters: KondoActorParameters,
        gate_key: Array,
    ) -> HCCLKondoContinualDyadState:
        """Initialize the one owner tree before actor P is first installed."""

        components = self._dyad.init(dyad_key)
        actor_state = self._lineage.actor.init(actor_parameters, gate_key)
        protected_td_state = self._protected_td.init()
        bare = HCCLKondoContinualDyadState(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            components=components,
            actor_state=actor_state,
            protected_td_state=protected_td_state,
            pending_proposal=None,
            pending_compact_adoptions=None,
            event_count=jnp.asarray(0, dtype=jnp.int32),
        )
        state = self._seal_state(bare)
        if not _bool(self.state_valid(state)):
            raise RuntimeError("HCCL/Kondo genesis violates its owner contract")
        return state

    def _through_tag(self, prepared: _HCCLKondoThroughMemory) -> Array:
        bare = cast(
            _HCCLKondoThroughMemory,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_KONDO_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA,
            self._config_words,
            bare,
        )

    def _prepare_through_memory(
        self,
        state: HCCLKondoContinualDyadState,
        next_hard_action_masks: Array,
    ) -> _HCCLKondoThroughMemory:
        """Evaluate context, HCCL, and both M donors once under route validity."""

        if not _bool(self.state_valid(state)):
            raise ValueError("route preparation requires a valid source state")
        masks = self._dyad._hard_action_masks(
            next_hard_action_masks,
            name="next_hard_action_masks",
        )
        if not bool(np.all(np.asarray(masks))):
            raise ValueError("the Kondo route supports exact all-true masks only")
        if _contains_tracer((state, masks)):
            raise TypeError("HCCL/Kondo route preparation is host/eager-only")

        components = state.components
        event = self._dyad.hccl.world.prepare_event(
            components.hccl_state.world_state
        )
        inputs = self._dyad.causal_core_memory_event_inputs(components, event)
        binding = self._dyad._make_binding(components, event)
        agents = (components.agent_0_state, components.agent_1_state)
        contexts = (components.context_0_state, components.context_1_state)
        adapters = (self._dyad.agent_0, self._dyad.agent_1)
        context_preparations = tuple(
            self._dyad.context.prepare(
                contexts[row],
                jax.nn.one_hot(
                    binding.final_actions[1 - row],
                    _N_ACTIONS,
                    dtype=jnp.float32,
                ),
                binding.final_actions[row],
            )
            for row in range(_N_AGENTS)
        )
        hccl_result = self._dyad.hccl.stage(
            components.hccl_state,
            event,
            binding.base,
            binding.memory,
            binding.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        mm = _signals_at(hccl_result.world_proposals, _MM_SLOT)
        b0m1 = _signals_at(hccl_result.world_proposals, _B0M1_SLOT)
        m0b1 = _signals_at(hccl_result.world_proposals, _M0B1_SLOT)
        bb = _signals_at(hccl_result.world_proposals, _BB_SLOT)
        pp = _signals_at(hccl_result.world_proposals, _PP_SLOT)
        credit_panel = derive_hccl_memory_credit_estimands(
            mm=mm,
            b0m1=b0m1,
            m0b1=m0b1,
            bb=bb,
        )
        credits = (
            credit_panel.baseline_context_direct_effect.net_reward[0, 0],
            credit_panel.baseline_context_direct_effect.net_reward[1, 1],
        )
        pp_proposal = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl_result.world_proposals),
        )
        horde_cumulants, horde_discounts = self._dyad._horde_targets(
            pp_proposal,
            pp,
            binding.final_actions,
        )
        context_results = tuple(
            self._dyad.context.step(
                contexts[row],
                context_preparations[row],
                pp.task_score,
            )
            for row in range(_N_AGENTS)
        )
        next_physical = hccl_result.world_proposals.next_observation[_PP_SLOT]
        next_raw = tuple(
            self._dyad._composed_observation(
                next_physical[row],
                context_results[row].state,
            )
            for row in range(_N_AGENTS)
        )
        feedback = tuple(
            self._dyad._memory_feedback(agents[row], credits[row])
            for row in range(_N_AGENTS)
        )
        transitions = tuple(
            self._dyad._transition(
                agents[row],
                executed_action=binding.final_actions[row],
                reward=pp.net_reward[row],
                next_observation=next_raw[row],
                horde_cumulants=horde_cumulants[row],
                horde_discounts=horde_discounts[row],
            )
            for row in range(_N_AGENTS)
        )
        memory_preparations = tuple(
            adapters[row].prepare_memory_transition(
                agents[row],
                transitions[row],
                inputs[row],
                masks[row],
                feedback[row],
                None,
            )
            for row in range(_N_AGENTS)
        )
        prepared_agents = tuple(
            self._dyad._seal_through_memory_agent(
                HCCLContinualDyadThroughMemoryAgent(
                    agent_index=jnp.asarray(row, dtype=jnp.int32),
                    context_preparation=context_preparations[row],
                    context_result=context_results[row],
                    memory_credit=credits[row],
                    memory_feedback=feedback[row],
                    transition=transitions[row],
                    memory_preparation=memory_preparations[row],
                    content_tag_words=jnp.zeros(
                        (_DIGEST_WORDS,),
                        dtype=jnp.uint32,
                    ),
                )
            )
            for row in range(_N_AGENTS)
        )
        valid = (
            self._dyad.hccl.world.event_receipt_valid(
                components.hccl_state.world_state,
                event,
            )
            & jnp.asarray(
                np.array_equal(
                    np.asarray(binding.content_tag_words),
                    np.asarray(self._dyad._binding_tag(binding)),
                ),
                dtype=jnp.bool_,
            )
            & hccl_result.update_applied
            & credit_panel.algebra.all_identities_hold
            & jnp.all(
                jnp.asarray(
                    tuple(item.update_applied for item in context_results),
                    dtype=jnp.bool_,
                )
            )
            & jnp.all(
                jnp.asarray(
                    tuple(item.preparation_valid for item in memory_preparations),
                    dtype=jnp.bool_,
                )
            )
        )
        bare = _HCCLKondoThroughMemory(
            source_state=state,
            event=event,
            binding=binding,
            hccl_result=hccl_result,
            memory_credit_panel=credit_panel,
            next_hard_action_masks=masks,
            agent_0=prepared_agents[0],
            agent_1=prepared_agents[1],
            preparation_valid=valid,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            _HCCLKondoThroughMemory,
            bare.replace(content_tag_words=self._through_tag(bare)),
        )

    def _shadow_planner(
        self,
        through: _HCCLKondoThroughMemory,
    ) -> PrototypeFactorizedPartnerTransitionResult:
        components = through.source_state.components
        through_agents = (through.agent_0, through.agent_1)
        source_agents = (components.agent_0_state, components.agent_1_state)
        post_memory = tuple(
            self._dyad._prototype(item.memory_preparation.memory_candidate_state)
            for item in through_agents
        )
        pp = _signals_at(through.hccl_result.world_proposals, _PP_SLOT)
        return self._dyad.planner.completed_transition(
            components.planner_state,
            self._dyad._prototype(source_agents[0]),
            self._dyad._prototype(source_agents[1]),
            post_memory[0],
            post_memory[1],
            through.binding.final_actions,
            pp.net_reward,
            jnp.stack(tuple(item.current_raw_observation for item in post_memory)).astype(
                jnp.float32
            ),
            jnp.asarray(self._dyad.config.discount, dtype=jnp.float32),
            through.next_hard_action_masks,
        )

    def _post_memory_actor_features(
        self,
        through: _HCCLKondoThroughMemory,
    ) -> Array:
        """Read the already-produced post-memory 23-wide Prototype features."""

        return jnp.stack(
            tuple(
                self._dyad._prototype(item.memory_preparation.memory_candidate_state)
                .current_raw_observation
                for item in (through.agent_0, through.agent_1)
            )
        ).astype(jnp.float32)

    def _protected_batch(
        self,
        through: _HCCLKondoThroughMemory,
        proposal: KondoExecutedActionProposalBatch,
        next_actor_features: Array,
    ) -> KondoProtectedTDBatch:
        """Bind one protected batch to the pending P and current PP transition."""

        transitions = (
            through.agent_0.memory_preparation.transition,
            through.agent_1.memory_preparation.transition,
        )
        current_features = jnp.stack(
            tuple(item.representation for item in transitions)
        ).astype(jnp.float32)
        actions = jnp.stack(tuple(item.action for item in transitions)).astype(jnp.int32)
        decision_identities = jnp.stack(
            tuple(item.decision_id for item in transitions)
        ).astype(jnp.uint32)
        rewards = jnp.stack(tuple(item.reward for item in transitions)).astype(jnp.float32)
        discounts = jnp.stack(tuple(item.discount for item in transitions)).astype(
            jnp.float32
        )
        pp = _signals_at(through.hccl_result.world_proposals, _PP_SLOT)
        costs = (pp.safety_cost + pp.message_charge).astype(jnp.float32)
        exact = all(
            (
                _tree_exact_equal(current_features, proposal.actor_features),
                _tree_exact_equal(actions, proposal.selected_actions),
                _tree_exact_equal(
                    decision_identities,
                    proposal.action_stack_decision_identities,
                ),
                _tree_exact_equal(rewards, pp.net_reward),
            )
        )
        if not exact:
            raise RuntimeError(
                "protected TD batch is not exact pending-P/current-PP lineage"
            )
        return KondoProtectedTDBatch(
            current_features=current_features,
            next_features=next_actor_features,
            actions=actions,
            decision_identities=decision_identities,
            rewards=rewards,
            discounts=discounts,
            costs=costs,
        )

    def _project_actor_pair(
        self,
        through: _HCCLKondoThroughMemory,
        shadow: PrototypeFactorizedPartnerTransitionResult,
        proposal: KondoExecutedActionProposalBatch,
    ) -> tuple[PrototypeFactorizedPartnerPlannerState, tuple[Any, Any], bool]:
        through_agents = (through.agent_0, through.agent_1)
        post_memory = tuple(
            self._dyad._prototype(item.memory_preparation.memory_candidate_state)
            for item in through_agents
        )
        replacements = tuple(
            self._dyad.planner._prototype.replace_cached_primitive_action(
                post_memory[row],
                decision_id=post_memory[row].current_decision_id,
                decision_observation=post_memory[row].current_representation,
                proposed_action=proposal.selected_actions[row],
                safety_action_mask=through.next_hard_action_masks[row],
            )
            for row in range(_N_AGENTS)
        )
        selected = (replacements[0].state, replacements[1].state)
        shadow_agents = (shadow.state.agent_0, shadow.state.agent_1)
        projected_agents = tuple(
            shadow_agents[row].replace(
                cache=shadow_agents[row].cache.replace(
                    base_action=proposal.selected_actions[row],
                    base_action_guard=jnp.bitwise_not(
                        proposal.selected_actions[row]
                    ).astype(jnp.int32),
                    effective_action=proposal.selected_actions[row],
                )
            )
            for row in range(_N_AGENTS)
        )
        projected = shadow.state.replace(
            agent_0=projected_agents[0],
            agent_1=projected_agents[1],
        )
        excluded = {"base_action", "base_action_guard", "effective_action"}
        only_projection = all(
            _tree_exact_equal(
                getattr(projected_agents[row].cache, field.name),
                getattr(shadow_agents[row].cache, field.name),
            )
            for row in range(_N_AGENTS)
            for field in dataclasses.fields(type(shadow_agents[row].cache))
            if field.name not in excluded
        )
        authenticated = self._dyad.planner.authenticate_pair(
            projected,
            selected[0],
            selected[1],
        )
        valid = all(
            (
                _bool(shadow.diagnostics.transaction_committed),
                all(_bool(item.committed) for item in replacements),
                all(
                    not _bool(projected_agents[row].cache.planner_consumed)
                    for row in range(_N_AGENTS)
                ),
                only_projection,
                bool(np.all(np.asarray(authenticated))),
            )
        )
        return cast(PrototypeFactorizedPartnerPlannerState, projected), selected, valid

    def _compact_certificate(
        self,
        proposal: KondoExecutedActionProposalBatch,
        finalizations: tuple[
            ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
            ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
        ],
        receipts: tuple[
            ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
            ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
        ],
        adoptions: tuple[
            ExternalLearnedStateLiveMemoryActionStackResult,
            ExternalLearnedStateLiveMemoryActionStackResult,
        ],
    ) -> KondoExecutedActionCompactAdoptionBatch:
        """Issue digest-only lineage from the already-adopted public results."""

        rows: list[tuple[Array, ...]] = []
        for row in range(_N_AGENTS):
            finalized = finalizations[row]
            prepared = finalized.memory_preparation
            memory_binding = prepared.memory_candidate_state.action_binding
            final_binding = finalized.final_action_binding
            adopted = adoptions[row]
            relations = all(
                (
                    _tree_exact_equal(adopted.finalized, finalized),
                    _bool(adopted.diagnostics.transaction_applied),
                    np.array_equal(
                        np.asarray(final_binding.planner_candidate_words),
                        np.asarray(proposal.proposal_digest_words[row]),
                    ),
                    int(final_binding.planner_action_before_mask)
                    == int(proposal.selected_actions[row]),
                    int(final_binding.final_action)
                    == int(proposal.selected_actions[row]),
                    _bool(final_binding.planner_consumed),
                )
            )
            if not relations:
                raise ValueError("compact certificate requires exact actor-routed P")
            rows.append(
                (
                    self._lineage.action_stack_source_state_digest_words(prepared),
                    prepared.content_tag_words,
                    memory_binding.content_tag_words,
                    memory_binding.prototype_decision_id,
                    final_binding.final_action_owner_words,
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "finalization",
                        finalized,
                    ),
                    receipts[row].content_tag_words,
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "adoption-result",
                        adopted,
                    ),
                    _action_stack_tree_digest(
                        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
                        "destination-state",
                        adopted.state,
                    ),
                    final_binding.planner_candidate_words,
                    final_binding.planner_action_before_mask,
                    final_binding.final_action,
                    final_binding.hard_action_mask,
                    final_binding.planner_consumed,
                    adopted.diagnostics.transaction_applied,
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
                (_N_AGENTS, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
        )
        return cast(
            KondoExecutedActionCompactAdoptionBatch,
            bare.replace(
                content_tag_words=self._lineage._compact_adoption_tags(bare)
            ),
        )

    @staticmethod
    def _work(
        *,
        pending_steps: int,
        proposal_batches: int,
        protected_calls: int,
        protected_rows: int,
        final_bindings: tuple[int, int],
        child_adoptions: tuple[int, int],
        certificate_issuances: int,
    ) -> HCCLKondoContinualDyadEventWork:
        return HCCLKondoContinualDyadEventWork(
            pending_compact_lineage_steps=jnp.asarray(
                pending_steps,
                dtype=jnp.int32,
            ),
            causal_core_memory_event_input_derivations=jnp.asarray(
                _N_AGENTS,
                dtype=jnp.int32,
            ),
            actor_proposal_batches=jnp.asarray(
                proposal_batches,
                dtype=jnp.int32,
            ),
            protected_td_full_batch_backward_calls=jnp.asarray(
                protected_calls,
                dtype=jnp.int32,
            ),
            protected_td_rows=jnp.asarray(
                protected_rows,
                dtype=jnp.int32,
            ),
            shadow_planner_completed_transition_calls=jnp.asarray(
                1,
                dtype=jnp.int32,
            ),
            action_stack_final_bindings=jnp.asarray(
                final_bindings,
                dtype=jnp.int32,
            ),
            public_child_adoptions=jnp.asarray(
                child_adoptions,
                dtype=jnp.int32,
            ),
            compact_certificate_issuances=jnp.asarray(
                certificate_issuances,
                dtype=jnp.int32,
            ),
            pending_consumed_before_new_sampling=jnp.asarray(
                True,
                dtype=jnp.bool_,
            ),
        )

    def _event(
        self,
        state: HCCLKondoContinualDyadState,
        next_hard_action_masks: Array,
        proposal_sampling_keys: Array,
    ) -> HCCLKondoContinualDyadEventResult:
        through = self._prepare_through_memory(
            state,
            next_hard_action_masks,
        )
        if not (
            _bool(through.preparation_valid)
            and np.array_equal(
                np.asarray(through.content_tag_words),
                np.asarray(self._through_tag(through)),
            )
        ):
            raise RuntimeError("route through-memory preparation was rejected")

        shadow = self._shadow_planner(through)
        through_agents = (through.agent_0, through.agent_1)
        memory_preparations = tuple(
            item.memory_preparation for item in through_agents
        )
        actor_features = self._post_memory_actor_features(through)
        pending = state.pending_proposal is not None
        lineage_result: KondoExecutedActionLineageResult | None = None
        protected_result: KondoProtectedTDResult | None = None
        actor_for_sampling = state.actor_state
        if pending:
            if state.pending_compact_adoptions is None:
                raise ValueError("pending lineage requires its compact proof")
            pending_proposal = cast(
                KondoExecutedActionProposalBatch,
                state.pending_proposal,
            )
            next_preparations = (
                through.agent_0.memory_preparation,
                through.agent_1.memory_preparation,
            )
            preflight = self._lineage.preflight_compact(
                state.actor_state,
                pending_proposal,
                state.pending_compact_adoptions,
                next_preparations,
            )
            if not bool(np.all(np.asarray(preflight.actor_eligible))):
                return HCCLKondoContinualDyadEventResult(
                    state=state,
                    proposal=None,
                    compact_adoptions=None,
                    lineage_result=None,
                    protected_td_result=None,
                    shadow_planner_result=shadow,
                    agent_0_adoption=None,
                    agent_1_adoption=None,
                    work=self._work(
                        pending_steps=0,
                        proposal_batches=0,
                        protected_calls=0,
                        protected_rows=0,
                        final_bindings=(0, 0),
                        child_adoptions=(0, 0),
                        certificate_issuances=0,
                    ),
                    update_applied=jnp.asarray(False, dtype=jnp.bool_),
                    complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
                )
            protected_batch = self._protected_batch(
                through,
                pending_proposal,
                actor_features,
            )
            protected_result = self._protected_td.step(
                state.protected_td_state,
                protected_batch,
            )
            if not _bool(protected_result.transaction_applied):
                return HCCLKondoContinualDyadEventResult(
                    state=state,
                    proposal=None,
                    compact_adoptions=None,
                    lineage_result=None,
                    protected_td_result=protected_result,
                    shadow_planner_result=shadow,
                    agent_0_adoption=None,
                    agent_1_adoption=None,
                    work=self._work(
                        pending_steps=0,
                        proposal_batches=0,
                        protected_calls=1,
                        protected_rows=_N_AGENTS,
                        final_bindings=(0, 0),
                        child_adoptions=(0, 0),
                        certificate_issuances=0,
                    ),
                    update_applied=jnp.asarray(False, dtype=jnp.bool_),
                    complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
                )
            lineage_result = self._lineage.step_compact(
                state.actor_state,
                pending_proposal,
                state.pending_compact_adoptions,
                next_preparations,
                protected_result.actor_inputs,
            )
            lineage_valid = bool(
                np.all(np.asarray(lineage_result.diagnostics.actor_eligible))
            ) and _bool(lineage_result.actor_result.transaction_applied)
            if not lineage_valid:
                return HCCLKondoContinualDyadEventResult(
                    state=state,
                    proposal=None,
                    compact_adoptions=None,
                    lineage_result=lineage_result,
                    protected_td_result=protected_result,
                    shadow_planner_result=shadow,
                    agent_0_adoption=None,
                    agent_1_adoption=None,
                    work=self._work(
                        pending_steps=1,
                        proposal_batches=0,
                        protected_calls=1,
                        protected_rows=_N_AGENTS,
                        final_bindings=(0, 0),
                        child_adoptions=(0, 0),
                        certificate_issuances=0,
                    ),
                    update_applied=jnp.asarray(False, dtype=jnp.bool_),
                    complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
                )
            actor_for_sampling = lineage_result.actor_result.state
        proposal = self._lineage.sample_proposals(
            actor_for_sampling,
            actor_features,
            proposal_sampling_keys,
            action_stack_memory_preparations=memory_preparations,
        )
        projected_planner, selected_prototypes, projection_valid = (
            self._project_actor_pair(through, shadow, proposal)
        )
        if not projection_valid:
            return HCCLKondoContinualDyadEventResult(
                state=state,
                proposal=proposal,
                compact_adoptions=None,
                lineage_result=lineage_result,
                protected_td_result=protected_result,
                shadow_planner_result=shadow,
                agent_0_adoption=None,
                agent_1_adoption=None,
                work=self._work(
                    pending_steps=int(pending),
                    proposal_batches=1,
                    protected_calls=int(pending),
                    protected_rows=_N_AGENTS * int(pending),
                    final_bindings=(0, 0),
                    child_adoptions=(0, 0),
                    certificate_issuances=0,
                ),
                update_applied=jnp.asarray(False, dtype=jnp.bool_),
                complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
            )

        adapters = (self._dyad.agent_0, self._dyad.agent_1)
        finalizations = tuple(
            adapters[row].bind_final_action(
                memory_preparations[row],
                selected_prototypes[row],
                planner_action_before_mask=proposal.selected_actions[row],
                planner_candidate_words=proposal.proposal_digest_words[row],
                planner_consumed=jnp.asarray(True, dtype=jnp.bool_),
            )
            for row in range(_N_AGENTS)
        )
        finalization_valid = all(
            _bool(item.finalization_valid) for item in finalizations
        )
        if not finalization_valid:
            return HCCLKondoContinualDyadEventResult(
                state=state,
                proposal=proposal,
                compact_adoptions=None,
                lineage_result=lineage_result,
                protected_td_result=protected_result,
                shadow_planner_result=shadow,
                agent_0_adoption=None,
                agent_1_adoption=None,
                work=self._work(
                    pending_steps=int(pending),
                    proposal_batches=1,
                    protected_calls=int(pending),
                    protected_rows=_N_AGENTS * int(pending),
                    final_bindings=(1, 1),
                    child_adoptions=(0, 0),
                    certificate_issuances=0,
                ),
                update_applied=jnp.asarray(False, dtype=jnp.bool_),
                complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
            )

        receipts = tuple(
            adapters[row].integrity_receipt(finalizations[row])
            for row in range(_N_AGENTS)
        )
        adoptions = tuple(
            adapters[row].adopt_finalized_transition(
                memory_preparations[row].source_state,
                finalizations[row],
                receipts[row],
            )
            for row in range(_N_AGENTS)
        )
        adoptions_valid = all(
            _bool(item.diagnostics.transaction_applied) for item in adoptions
        )
        if not adoptions_valid:
            return HCCLKondoContinualDyadEventResult(
                state=state,
                proposal=proposal,
                compact_adoptions=None,
                lineage_result=lineage_result,
                protected_td_result=protected_result,
                shadow_planner_result=shadow,
                agent_0_adoption=adoptions[0],
                agent_1_adoption=adoptions[1],
                work=self._work(
                    pending_steps=int(pending),
                    proposal_batches=1,
                    protected_calls=int(pending),
                    protected_rows=_N_AGENTS * int(pending),
                    final_bindings=(1, 1),
                    child_adoptions=(1, 1),
                    certificate_issuances=0,
                ),
                update_applied=jnp.asarray(False, dtype=jnp.bool_),
                complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
            )

        compact = self._compact_certificate(
            proposal,
            cast(Any, finalizations),
            cast(Any, receipts),
            cast(Any, adoptions),
        )
        unsigned_components = HCCLContinualDyadState(
            config_token=state.components.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=through.hccl_result.state,
            agent_0_state=adoptions[0].state,
            agent_1_state=adoptions[1].state,
            planner_state=projected_planner,
            context_0_state=through.agent_0.context_result.state,
            context_1_state=through.agent_1.context_result.state,
        )
        components = self._dyad._seal_state(unsigned_components)
        bare_candidate = HCCLKondoContinualDyadState(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            components=components,
            actor_state=actor_for_sampling,
            protected_td_state=(
                protected_result.state
                if protected_result is not None
                else state.protected_td_state
            ),
            pending_proposal=proposal,
            pending_compact_adoptions=compact,
            event_count=state.event_count + jnp.asarray(1, dtype=jnp.int32),
        )
        candidate = self._seal_state(bare_candidate)
        candidate_valid = _bool(self.state_valid(candidate))
        final_state = candidate if candidate_valid else state
        return HCCLKondoContinualDyadEventResult(
            state=final_state,
            proposal=proposal,
            compact_adoptions=compact,
            lineage_result=lineage_result,
            protected_td_result=protected_result,
            shadow_planner_result=shadow,
            agent_0_adoption=adoptions[0],
            agent_1_adoption=adoptions[1],
            work=self._work(
                pending_steps=int(pending),
                proposal_batches=1,
                protected_calls=int(pending),
                protected_rows=_N_AGENTS * int(pending),
                final_bindings=(1, 1),
                child_adoptions=(1, 1),
                certificate_issuances=1,
            ),
            update_applied=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not candidate_valid,
                dtype=jnp.bool_,
            ),
        )

    def event0(
        self,
        state: HCCLKondoContinualDyadState,
        next_hard_action_masks: Array,
        proposal_sampling_keys: Array,
    ) -> HCCLKondoContinualDyadEventResult:
        """Install the first actor-owned P and its pending compact certificate."""

        self._state_contract(state)
        if (
            int(state.event_count) != 0
            or state.pending_proposal is not None
            or state.pending_compact_adoptions is not None
        ):
            raise ValueError("event0 requires exact pending-free genesis")
        return self._event(
            state,
            next_hard_action_masks,
            proposal_sampling_keys,
        )

    def event1(
        self,
        state: HCCLKondoContinualDyadState,
        next_hard_action_masks: Array,
        proposal_sampling_keys: Array,
    ) -> HCCLKondoContinualDyadEventResult:
        """Compatibility wrapper for the exact first successor event."""

        self._state_contract(state)
        if (
            int(state.event_count) != 1
            or state.pending_proposal is None
            or state.pending_compact_adoptions is None
        ):
            raise ValueError("event1 requires the exact event0 pending lineage")
        return self.event(
            state,
            next_hard_action_masks,
            proposal_sampling_keys,
        )

    def event(
        self,
        state: HCCLKondoContinualDyadState,
        next_hard_action_masks: Array,
        proposal_sampling_keys: Array,
    ) -> HCCLKondoContinualDyadEventResult:
        """Consume pending lineage and atomically install the next actor P."""

        self._state_contract(state)
        if (
            int(state.event_count) < 1
            or state.pending_proposal is None
            or state.pending_compact_adoptions is None
        ):
            raise ValueError("event requires one exact pending lineage pair")
        if int(state.event_count) == np.iinfo(np.int32).max:
            raise OverflowError("event_count cannot exceed signed int32 capacity")
        return self._event(
            state,
            next_hard_action_masks,
            proposal_sampling_keys,
        )
