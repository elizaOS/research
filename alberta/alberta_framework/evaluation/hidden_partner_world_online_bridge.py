"""Causal online bridge from the noisy hidden world to the integrated agent.

The bridge owns evaluator state but passes the learner an explicit oracle-free
transition.  A decision is cached before the environment transition; the
agent consumes that transition before the evaluator updates its Bayes filter
or reads privileged oracle fields.  Learner/world/counter rejection atomically
preserves the three component states and latches the bridge invalid.  Bayes
filter and privileged-oracle validity can invalidate evaluator trace claims,
but neither can gate or otherwise change the learner/world commit path.
"""

# mypy: disable-error-code="attr-defined,call-arg"

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    BASE_FEATURE_DIM,
    CANDIDATE_PAIR_SLOTS,
    DEPLOYED_FEATURE_DIM,
    N_ACTIONS,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
    IntegratedHiddenPartnerState,
    IntegratedPlannerSelection,
    IntegratedUpdateDiagnostics,
)
from alberta_framework.evaluation.hidden_partner_world_filter import (
    HiddenPartnerWorldBayesFilter,
    HiddenPartnerWorldFilterConfig,
    HiddenPartnerWorldFilterState,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    HiddenPartnerWorldFeedbackState,
    HiddenPartnerWorldFeedbackTransition,
    HiddenPartnerWorldFeedbackWorld,
)

_INT32_MAX = 2**31 - 1
HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA = (
    "alberta.hidden-partner-world-online-bridge.config.v3"
)
_CONFIG_TOKEN_BYTES = 32
_GROUNDED_TARGET_DIM = RAW_OBSERVATION_DIM + 2
_JOINT_ACTION_ROWS = N_ACTIONS * N_ACTIONS

FocalActionPolicy = Literal["agent", "balanced_external"]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("bridge config must contain finite canonical JSON data") from exc


def _tree_array_nbytes(tree: object) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _static_array_contract(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    actual_shape = getattr(value, "shape", None)
    actual_dtype = getattr(value, "dtype", None)
    if actual_shape != shape or actual_dtype != jnp.dtype(dtype):
        raise TypeError(
            f"static state contract for {name} requires shape {shape} and dtype "
            f"{jnp.dtype(dtype)}, got {actual_shape} and {actual_dtype}"
        )


def _tree_static_signature(tree: object) -> tuple[object, tuple[tuple[tuple[int, ...], str], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    signature: list[tuple[tuple[int, ...], str]] = []
    for index, leaf in enumerate(leaves):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError(f"static state contract leaf {index} must be an array")
        signature.append((tuple(shape), str(dtype)))
    return structure, tuple(signature)


@chex.dataclass(frozen=True)
class LearnerHiddenPartnerWorldTransition:
    """The complete and only transition surface visible to the learner."""

    observation: Float[Array, " 8"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]


def strip_hidden_partner_world_oracle(
    transition: HiddenPartnerWorldFeedbackTransition,
) -> LearnerHiddenPartnerWorldTransition:
    """Copy learner-visible fields without retaining an oracle reference."""
    return LearnerHiddenPartnerWorldTransition(
        observation=transition.observation,
        focal_action=transition.focal_action,
        partner_action=transition.partner_action,
        reward=transition.reward,
        outcome=transition.outcome,
        next_observation=transition.next_observation,
        terminated=transition.terminated,
        discount=transition.discount,
    )


@chex.dataclass(frozen=True)
class HiddenPartnerWorldOnlineState:
    world: HiddenPartnerWorldFeedbackState
    agent: IntegratedHiddenPartnerState
    world_filter: HiddenPartnerWorldFilterState
    config_token: UInt[Array, " 32"]
    action: Int[Array, ""]
    valid: Bool[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldMechanismTrace:
    """Compact transient projection of one integrated learner update.

    This trace has no execution authority and does not duplicate candidate
    parameters, utilities, descriptors, or replay. Fixed-width transient
    diagnostics are sufficient to reconstruct the narrow C/D lifecycle chains
    in a development runner. Rejected bridge steps use exact neutral values
    with ``-1`` for identities and descriptor/source slots.
    """

    valid: Bool[Array, ""]
    grounded_enabled: Bool[Array, ""]
    grounded_learning_enabled: Bool[Array, ""]
    grounded_prediction_valid: Bool[Array, ""]
    grounded_target_valid: Bool[Array, ""]
    grounded_gradient_valid: Bool[Array, ""]
    grounded_prediction_matches_decision: Bool[Array, ""]
    grounded_row_update_isolated: Bool[Array, ""]
    grounded_update_applied: Bool[Array, ""]
    grounded_executed_joint_index: Int[Array, ""]
    grounded_feature_contribution: Float[Array, " 10"]
    grounded_row_bias: Float[Array, " 10"]
    grounded_raw_predictions: Float[Array, " 10"]
    grounded_targets: Float[Array, " 10"]
    grounded_errors: Float[Array, " 10"]
    grounded_fit_loss_by_head: Float[Array, " 10"]
    grounded_representation_loss_by_head: Float[Array, " 10"]
    grounded_representation_gradient: Float[Array, " 24"]
    grounded_representation_gradient_by_head: Float[Array, "10 24"]
    grounded_representation_gradient_norm_by_head: Float[Array, " 10"]
    grounded_proposed_weight_row_bit_change_mask: Bool[Array, " 4"]
    grounded_proposed_bias_row_bit_change_mask: Bool[Array, " 4"]
    grounded_executed_weight_row_delta_norm_by_head: Float[Array, " 10"]
    grounded_executed_bias_row_delta_by_head: Float[Array, " 10"]
    lifecycle_enabled: Bool[Array, ""]
    lifecycle_proposed: Bool[Array, ""]
    lifecycle_applied: Bool[Array, ""]
    lifecycle_pre_descriptors: Int[Array, "12 2"]
    lifecycle_proposal_descriptors: Int[Array, "12 2"]
    lifecycle_applied_descriptors: Int[Array, "12 2"]
    lifecycle_proposal_replaced_slot: Int[Array, ""]
    lifecycle_proposal_promoted_candidate: Int[Array, ""]
    lifecycle_proposal_refreshed_candidate: Int[Array, ""]
    lifecycle_proposal_retired_slot: Int[Array, ""]
    lifecycle_proposal_retired_left: Int[Array, ""]
    lifecycle_proposal_retired_right: Int[Array, ""]
    lifecycle_applied_replaced_slot: Int[Array, ""]
    lifecycle_applied_promoted_candidate: Int[Array, ""]
    lifecycle_applied_refreshed_candidate: Int[Array, ""]
    lifecycle_applied_retired_slot: Int[Array, ""]
    lifecycle_applied_retired_left: Int[Array, ""]
    lifecycle_applied_retired_right: Int[Array, ""]
    lifecycle_active_evidence_refreshed: Bool[Array, " 12"]
    lifecycle_retention_evidence_refreshed: Bool[Array, " 12"]
    lifecycle_durable_read_mask: Bool[Array, " 12"]
    lifecycle_relevance_probe_scores: Float[Array, " 12"]
    lifecycle_relevance_probe_errors: Float[Array, "1 12"]
    lifecycle_candidate_promotion_signal: Float[Array, " 66"]
    lifecycle_candidate_promotion_raw_evidence: Bool[Array, " 66"]
    lifecycle_candidate_promotion_confirmed: Bool[Array, " 66"]
    lifecycle_candidate_promotion_evidence_streak_pre: Int[Array, " 66"]
    lifecycle_candidate_promotion_evidence_streak_updated: Int[Array, " 66"]
    lifecycle_candidate_promotion_evidence_streak_proposal_post: Int[Array, " 66"]
    lifecycle_candidate_promotion_evidence_streak_post: Int[Array, " 66"]
    lifecycle_candidate_reacquisition_required_pre: Bool[Array, " 66"]
    lifecycle_candidate_reacquisition_required_proposal_post: Bool[Array, " 66"]
    lifecycle_candidate_reacquisition_required_post: Bool[Array, " 66"]
    lifecycle_candidate_reacquisition_confirmed: Bool[Array, " 66"]
    lifecycle_candidate_reset_mask: Bool[Array, " 66"]
    lifecycle_applied_candidate_reset_mask: Bool[Array, " 66"]
    random_curation_enabled: Bool[Array, ""]
    random_curation_attempted: Bool[Array, ""]
    random_curation_applied: Bool[Array, ""]
    random_curation_selected_active_worst_slot: Int[Array, ""]
    random_curation_selected_promotion_candidate: Int[Array, ""]
    random_curation_selected_refresh_candidate: Int[Array, ""]
    random_curation_active_priorities: Float[Array, " 12"]
    random_curation_candidate_priorities: Float[Array, " 66"]
    consumer_read_acquire_pre: Bool[Array, " 12"]
    consumer_read_acquire_post: Bool[Array, " 12"]
    consumer_confirmed_write_pre: Bool[Array, " 12"]
    consumer_confirmed_write_post: Bool[Array, " 12"]
    consumer_read_mask_pre: Bool[Array, " 12"]
    consumer_read_mask_post: Bool[Array, " 12"]
    consumer_active_mask_pre: Bool[Array, " 12"]
    consumer_active_mask_post: Bool[Array, " 12"]
    router_valid: Bool[Array, ""]
    router_applied: Bool[Array, ""]
    router_carry_survivors: Bool[Array, ""]
    router_descriptors_changed: Bool[Array, ""]
    router_source_slots: Int[Array, " 12"]
    router_survivor_mask: Bool[Array, " 12"]
    router_new_mask: Bool[Array, " 12"]
    router_evicted_mask: Bool[Array, " 12"]
    router_route_count_before: Int[Array, ""]
    router_route_count_after: Int[Array, ""]
    router_generation_count_before: Int[Array, ""]
    router_generation_count_after: Int[Array, ""]
    consumer_route_source_slots_exact: Bool[Array, ""]
    consumer_route_identity_masks_exact: Bool[Array, ""]
    consumer_route_stable_prefix_exact: Bool[Array, ""]
    consumer_route_survivor_values_exact: Bool[Array, ""]
    consumer_route_reset_values_exact: Bool[Array, ""]
    consumer_route_no_carry_reset_exact: Bool[Array, ""]
    consumer_route_behavior_values_exact: Bool[Array, ""]
    consumer_route_q_values_exact: Bool[Array, ""]
    consumer_route_trace_values_exact: Bool[Array, ""]
    consumer_route_last_observation_exact: Bool[Array, ""]
    consumer_route_grounded_values_exact: Bool[Array, ""]
    consumer_route_values_exact: Bool[Array, ""]
    consumer_lifecycle_destination_reset_exact: Bool[Array, ""]
    behavior_prediction_matches_decision: Bool[Array, ""]
    behavior_credit_gradient_chi: Float[Array, " 24"]
    behavior_credit_gradient_phi: Float[Array, " 12"]
    behavior_credit_valid: Bool[Array, ""]
    grounded_credit_gradient_chi: Float[Array, " 24"]
    grounded_credit_valid: Bool[Array, ""]
    mixed_credit_gradient_chi: Float[Array, " 24"]
    mixed_credit_gradient_phi: Float[Array, " 12"]
    mixed_credit_valid: Bool[Array, ""]
    mixed_credit_applied: Bool[Array, ""]
    mixed_credit_conflict: Bool[Array, ""]
    state_learning_enabled: Bool[Array, ""]
    state_learning_proposal_valid: Bool[Array, ""]
    state_learning_component_applied: Bool[Array, ""]
    state_learning_committed: Bool[Array, ""]
    state_learning_valid: Bool[Array, ""]
    state_learning_rejected: Bool[Array, ""]
    state_learning_gradient_norm: Float[Array, ""]
    state_learning_clipped_gradient_norm: Float[Array, ""]
    state_learning_parameter_update_norm: Float[Array, ""]
    state_builder_step_delta: Int[Array, ""]
    state_builder_learning_delta: Int[Array, ""]
    behavior_step_delta: Int[Array, ""]
    interaction_step_delta: Int[Array, ""]
    table_world_step_delta: Int[Array, ""]
    grounded_world_step_delta: Int[Array, ""]
    control_step_delta: Int[Array, ""]
    router_route_delta: Int[Array, ""]
    router_generation_delta: Int[Array, ""]
    integrated_step_delta: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldOnlineTrace:
    """Accepted-only prequential values plus explicit proposal diagnostics.

    The agent-partner-belief-conditioned fields combine the exact Bayes world
    filter with the agent's applied partner distribution. They are not a full
    Bayes partner oracle. Filter-derived fields are evaluator-only and neutral
    whenever the filter trace is invalid; privileged fields are evaluator-only
    and neutral whenever the oracle trace is invalid. Neither diagnostic can
    alter ``accepted`` or the learner/world state.
    """

    active: Bool[Array, ""]
    accepted: Bool[Array, ""]
    step: Int[Array, ""]
    entry_state_contract_valid: Bool[Array, ""]
    config_token_valid: Bool[Array, ""]
    counters_synchronized: Bool[Array, ""]
    action_valid: Bool[Array, ""]
    action_policy_valid: Bool[Array, ""]
    selection_binding_valid: Bool[Array, ""]
    policy_replay_valid: Bool[Array, ""]
    filter_entry_valid: Bool[Array, ""]
    observation_pre: Float[Array, " 8"]
    phi_pre: Float[Array, " 12"]
    chi_pre: Float[Array, " 24"]
    focal_action: Int[Array, ""]
    expected_focal_action: Int[Array, ""]
    focal_action_externally_forced: Bool[Array, ""]
    focal_action_noisy_greedy_action: Int[Array, ""]
    focal_action_random_action: Int[Array, ""]
    focal_action_explored: Bool[Array, ""]
    focal_action_ordinary_policy_action: Int[Array, ""]
    focal_action_policy_rng_before: UInt[Array, " 2"]
    focal_action_policy_rng_after: UInt[Array, " 2"]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    corrected_outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    next_action: Int[Array, ""]
    expected_next_action: Int[Array, ""]
    next_action_policy_valid: Bool[Array, ""]
    next_selection_binding_valid: Bool[Array, ""]
    next_policy_replay_valid: Bool[Array, ""]
    next_selection_diagnostics_valid: Bool[Array, ""]
    next_action_externally_forced: Bool[Array, ""]
    next_action_noisy_greedy_action: Int[Array, ""]
    next_action_random_action: Int[Array, ""]
    next_action_explored: Bool[Array, ""]
    next_action_ordinary_policy_action: Int[Array, ""]
    next_action_policy_rng_before: UInt[Array, " 2"]
    next_action_policy_rng_after: UInt[Array, " 2"]
    filter_mean_pre: Float[Array, ""]
    filter_mean_post: Float[Array, ""]
    agent_partner_belief_conditioned_reward_cells: Float[Array, "2 2"]
    agent_applied_partner_probabilities: Float[Array, " 2"]
    agent_partner_belief_conditioned_expected_rewards: Float[Array, " 2"]
    agent_partner_belief_conditioned_greedy_action: Int[Array, ""]
    agent_partner_belief_conditioned_selected_regret: Float[Array, ""]
    agent_partner_belief_conditioned_action_margin: Float[Array, ""]
    agent_partner_belief_conditioned_tied: Bool[Array, ""]
    oracle_step_count: Int[Array, ""]
    oracle_cycle_index: Int[Array, ""]
    oracle_cycle_step: Int[Array, ""]
    oracle_cycle_length: Int[Array, ""]
    oracle_segment_index: Int[Array, ""]
    oracle_segment_step: Int[Array, ""]
    oracle_segment_length: Int[Array, ""]
    oracle_world_sign: Float[Array, ""]
    oracle_next_world_sign: Float[Array, ""]
    oracle_world_flipped: Bool[Array, ""]
    oracle_outcome_flipped: Bool[Array, ""]
    oracle_regime_id: Int[Array, ""]
    oracle_next_cycle_index: Int[Array, ""]
    oracle_next_segment_index: Int[Array, ""]
    oracle_next_regime_id: Int[Array, ""]
    oracle_schedule_switched: Bool[Array, ""]
    oracle_partner_intended_action: Int[Array, ""]
    oracle_partner_intended_sign: Float[Array, ""]
    oracle_partner_flipped: Bool[Array, ""]
    oracle_focal_action_sign: Float[Array, ""]
    oracle_partner_action_sign: Float[Array, ""]
    oracle_world_cue_flipped: Bool[Array, " 2"]
    oracle_next_world_cue_flipped: Bool[Array, " 2"]
    oracle_full_information_action: Int[Array, ""]
    oracle_full_information_action_margin: Float[Array, ""]
    oracle_full_information_action_tied: Bool[Array, ""]
    oracle_realized_counterfactual_rewards: Float[Array, " 2"]
    proposed_agent_update_valid: Bool[Array, ""]
    proposed_filter_update_valid: Bool[Array, ""]
    proposed_filter_decision_valid: Bool[Array, ""]
    learner_trace_valid: Bool[Array, ""]
    filter_trace_valid: Bool[Array, ""]
    oracle_trace_valid: Bool[Array, ""]
    proposed_world_step_delta: Int[Array, ""]
    proposed_agent_step_delta: Int[Array, ""]
    proposed_filter_step_delta: Int[Array, ""]
    proposed_bridge_step_delta: Int[Array, ""]
    committed_world_step_delta: Int[Array, ""]
    committed_agent_step_delta: Int[Array, ""]
    committed_filter_step_delta: Int[Array, ""]
    committed_bridge_step_delta: Int[Array, ""]
    mechanism: HiddenPartnerWorldMechanismTrace
    all_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldOnlineStep:
    state: HiddenPartnerWorldOnlineState
    trace: HiddenPartnerWorldOnlineTrace


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldOnlineResourceBudget:
    """Exact persistent component and bridge-owned state accounting."""

    world_state_nbytes: int
    agent_state_nbytes: int
    filter_state_nbytes: int
    component_state_nbytes: int
    config_token_nbytes: int
    action_nbytes: int
    valid_nbytes: int
    step_count_nbytes: int
    bridge_metadata_nbytes: int
    total_state_nbytes: int
    world_replay_capacity: int
    agent_replay_capacity: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


class HiddenPartnerWorldOnlineBridge:
    """One uninterrupted online world/agent/filter composition."""

    def __init__(
        self,
        world: HiddenPartnerWorldFeedbackWorld | None = None,
        agent: IntegratedHiddenPartnerAgent | None = None,
        world_filter: HiddenPartnerWorldBayesFilter | None = None,
        *,
        focal_action_policy: FocalActionPolicy = "agent",
        initial_external_action: int = 0,
    ) -> None:
        self._world: HiddenPartnerWorldFeedbackWorld = (
            HiddenPartnerWorldFeedbackWorld() if world is None else world
        )
        self._agent: IntegratedHiddenPartnerAgent = (
            IntegratedHiddenPartnerAgent() if agent is None else agent
        )
        if not isinstance(self._world, HiddenPartnerWorldFeedbackWorld):
            raise TypeError("world must be a HiddenPartnerWorldFeedbackWorld")
        if not isinstance(self._agent, IntegratedHiddenPartnerAgent):
            raise TypeError("agent must be an IntegratedHiddenPartnerAgent")
        if world_filter is not None and not isinstance(
            world_filter, HiddenPartnerWorldBayesFilter
        ):
            raise TypeError("world_filter must be a HiddenPartnerWorldBayesFilter")
        if type(focal_action_policy) is not str or focal_action_policy not in (
            "agent",
            "balanced_external",
        ):
            raise ValueError("focal_action_policy must be 'agent' or 'balanced_external'")
        if type(initial_external_action) is not int or initial_external_action not in (0, 1):
            raise ValueError("initial_external_action must be the exact integer 0 or 1")
        if focal_action_policy == "agent" and initial_external_action != 0:
            raise ValueError(
                "agent focal_action_policy requires canonical initial_external_action 0"
            )
        required_agent_mode = (
            "externally_forced" if focal_action_policy == "balanced_external" else "agent"
        )
        if self._agent.config.action_selection_mode != required_agent_mode:
            raise ValueError(
                "bridge focal_action_policy and agent action_selection_mode must match exactly"
            )
        self._focal_action_policy: FocalActionPolicy = focal_action_policy
        self._initial_external_action: int = initial_external_action
        expected_filter_config = HiddenPartnerWorldFilterConfig.from_world_config(
            self._world.config
        )
        self._filter: HiddenPartnerWorldBayesFilter = (
            HiddenPartnerWorldBayesFilter(expected_filter_config)
            if world_filter is None
            else world_filter
        )
        if self._filter.config != expected_filter_config:
            raise ValueError("world and Bayes-filter probability contracts must match exactly")
        self._config_token_hex: str = hashlib.sha256(
            _canonical_json_bytes(self.to_config())
        ).hexdigest()
        self._config_token: Array = jnp.asarray(
            tuple(bytes.fromhex(self._config_token_hex)),
            dtype=jnp.uint8,
        )
        self._state_static_signature: (
            tuple[object, tuple[tuple[tuple[int, ...], str], ...]] | None
        ) = None

    @property
    def world(self) -> HiddenPartnerWorldFeedbackWorld:
        return self._world

    @property
    def agent(self) -> IntegratedHiddenPartnerAgent:
        return self._agent

    @property
    def world_filter(self) -> HiddenPartnerWorldBayesFilter:
        return self._filter

    @property
    def focal_action_policy(self) -> FocalActionPolicy:
        return self._focal_action_policy

    @property
    def initial_external_action(self) -> int:
        return self._initial_external_action

    @property
    def config_token_hex(self) -> str:
        """SHA-256 of the exact authority-free serialized composition."""

        return self._config_token_hex

    def to_config(self) -> dict[str, object]:
        """Serialize the exact static composition without execution authority."""

        return {
            "type": type(self).__name__,
            "schema": HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
            "development_only": True,
            "execution_authorized": False,
            "evidence_authorized": False,
            "scientific_promotion_allowed": False,
            "focal_action_policy": self._focal_action_policy,
            "initial_external_action": self._initial_external_action,
            "world": self._world.to_config(),
            "agent": self._agent.to_config(),
            "world_filter": self._filter.config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> HiddenPartnerWorldOnlineBridge:
        """Reconstruct only an exact authority-free v3 composition."""

        if type(config) is not dict:
            raise ValueError("bridge config must be a plain object")
        payload = dict(config)
        expected_fields = {
            "type",
            "schema",
            "development_only",
            "execution_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
            "focal_action_policy",
            "initial_external_action",
            "world",
            "agent",
            "world_filter",
        }
        if set(payload) != expected_fields:
            raise ValueError("bridge config fields do not match the v3 schema")
        if payload["type"] != cls.__name__:
            raise ValueError("bridge config type differs")
        if payload["schema"] != HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA:
            raise ValueError("bridge config schema differs")
        if payload["development_only"] is not True:
            raise ValueError("bridge must remain development-only")
        for field in (
            "execution_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        ):
            if payload[field] is not False:
                raise ValueError(f"bridge config {field} must remain false")
        world_payload = payload["world"]
        agent_payload = payload["agent"]
        filter_payload = payload["world_filter"]
        if type(world_payload) is not dict:
            raise ValueError("bridge world config must be a plain object")
        if type(agent_payload) is not dict:
            raise ValueError("bridge agent config must be a plain object")
        if type(filter_payload) is not dict:
            raise ValueError("bridge filter config must be a plain object")
        world = HiddenPartnerWorldFeedbackWorld.from_config(
            cast(Mapping[str, Any], world_payload)
        )
        agent_config_payload = cast(dict[str, object], agent_payload).get("config")
        if type(agent_config_payload) is not dict:
            raise ValueError("bridge agent config must contain a plain config object")
        agent = IntegratedHiddenPartnerAgent(
            IntegratedHiddenPartnerConfig.from_config(
                cast(Mapping[str, Any], agent_config_payload)
            )
        )
        if _canonical_json_bytes(agent.to_config()) != _canonical_json_bytes(agent_payload):
            raise ValueError("bridge agent composition differs from its serialized config")
        world_filter = HiddenPartnerWorldBayesFilter(
            HiddenPartnerWorldFilterConfig.from_config(
                cast(Mapping[str, Any], filter_payload)
            )
        )
        bridge = cls(
            world=world,
            agent=agent,
            world_filter=world_filter,
            focal_action_policy=cast(FocalActionPolicy, payload["focal_action_policy"]),
            initial_external_action=cast(int, payload["initial_external_action"]),
        )
        if _canonical_json_bytes(bridge.to_config()) != _canonical_json_bytes(payload):
            raise ValueError("bridge composition differs from the exact serialized config")
        return bridge

    def _require_static_state_contract(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> None:
        if not isinstance(state, HiddenPartnerWorldOnlineState):
            raise TypeError("state must be a HiddenPartnerWorldOnlineState")
        if not isinstance(state.world, HiddenPartnerWorldFeedbackState):
            raise TypeError("static state contract requires HiddenPartnerWorldFeedbackState")
        if not isinstance(state.agent, IntegratedHiddenPartnerState):
            raise TypeError("static state contract requires IntegratedHiddenPartnerState")
        if not isinstance(state.world_filter, HiddenPartnerWorldFilterState):
            raise TypeError("static state contract requires HiddenPartnerWorldFilterState")
        _static_array_contract(
            state.config_token,
            name="state.config_token",
            shape=(_CONFIG_TOKEN_BYTES,),
            dtype=jnp.uint8,
        )
        _static_array_contract(
            state.action,
            name="state.action",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.valid,
            name="state.valid",
            shape=(),
            dtype=jnp.bool_,
        )
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        signature = _tree_static_signature(state)
        if self._state_static_signature is None:
            raise RuntimeError("bridge must initialize its exact static state contract before step")
        if signature != self._state_static_signature:
            raise ValueError("state differs from the exact initialized static state contract")

    def _capture_static_state_contract(self, state: HiddenPartnerWorldOnlineState) -> None:
        signature = _tree_static_signature(state)
        if self._state_static_signature is not None and signature != self._state_static_signature:
            raise ValueError("initial state differs from the captured static state contract")
        self._state_static_signature = signature

    def resource_budget(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> HiddenPartnerWorldOnlineResourceBudget:
        """Return exact state bytes and verify both component budget contracts."""

        self._require_static_state_contract(state)
        world_bytes = _tree_array_nbytes(state.world)
        agent_bytes = _tree_array_nbytes(state.agent)
        filter_bytes = _tree_array_nbytes(state.world_filter)
        agent_budget = self._agent.resource_budget(state.agent)
        if world_bytes != self._world.resource_budget.state_nbytes:
            raise ValueError("world resource accounting differs from the exact state tree")
        if agent_bytes != agent_budget.total_state_nbytes:
            raise ValueError("agent resource accounting differs from the exact state tree")
        token_bytes = _tree_array_nbytes(state.config_token)
        action_bytes = _tree_array_nbytes(state.action)
        valid_bytes = _tree_array_nbytes(state.valid)
        count_bytes = _tree_array_nbytes(state.step_count)
        metadata_bytes = token_bytes + action_bytes + valid_bytes + count_bytes
        component_bytes = world_bytes + agent_bytes + filter_bytes
        total_bytes = component_bytes + metadata_bytes
        if total_bytes != _tree_array_nbytes(state):
            raise ValueError("bridge resource accounting differs from the exact state tree")
        world_replay = self._world.resource_budget.replay_capacity
        agent_replay = agent_budget.replay_capacity
        return HiddenPartnerWorldOnlineResourceBudget(
            world_state_nbytes=world_bytes,
            agent_state_nbytes=agent_bytes,
            filter_state_nbytes=filter_bytes,
            component_state_nbytes=component_bytes,
            config_token_nbytes=token_bytes,
            action_nbytes=action_bytes,
            valid_nbytes=valid_bytes,
            step_count_nbytes=count_bytes,
            bridge_metadata_nbytes=metadata_bytes,
            total_state_nbytes=total_bytes,
            world_replay_capacity=world_replay,
            agent_replay_capacity=agent_replay,
            replay_capacity=world_replay + agent_replay,
        )

    @staticmethod
    def _ordinary_policy_action(selection: IntegratedPlannerSelection) -> Array:
        return jax.lax.select(
            selection.explored,
            selection.random_action,
            selection.noisy_greedy_action,
        )

    def _expected_policy_action(
        self,
        step_count: Array,
        selection: IntegratedPlannerSelection,
    ) -> Array:
        if self._focal_action_policy == "balanced_external":
            parity = jnp.bitwise_and(step_count, jnp.asarray(1, dtype=jnp.int32))
            return jnp.bitwise_xor(
                jnp.asarray(self._initial_external_action, dtype=jnp.int32),
                parity,
            )
        return self._ordinary_policy_action(selection)

    def _selection_policy_valid(
        self,
        action: Array,
        step_count: Array,
        selection: IntegratedPlannerSelection,
    ) -> Array:
        externally_forced = self._focal_action_policy == "balanced_external"
        expected = self._expected_policy_action(step_count, selection)
        return (
            (action >= 0)
            & (action < 2)
            & (selection.action == action)
            & (action == expected)
            & (
                selection.externally_forced
                == jnp.asarray(externally_forced, dtype=jnp.bool_)
            )
        )

    def _agent_selection_contract(
        self,
        agent_state: IntegratedHiddenPartnerState,
        action: Array,
        step_count: Array,
    ) -> tuple[Array, Array, Array, Array]:
        selection = agent_state.current_selection
        ordinary_policy_action = self._ordinary_policy_action(selection)
        replay_control = agent_state.control.replace(rng_key=selection.rng_key_before)
        replayed = self._agent.select_planner_action(
            replay_control,
            agent_state.current_evaluation.planner_scores,
        )

        def exact(left: Array, right: Array) -> Array:
            return jnp.array_equal(jnp.asarray(left), jnp.asarray(right))

        selection_binding_valid = (
            (action >= 0)
            & (action < 2)
            & (selection.action >= 0)
            & (selection.action < 2)
            & (selection.noisy_greedy_action >= 0)
            & (selection.noisy_greedy_action < 2)
            & (selection.random_action >= 0)
            & (selection.random_action < 2)
            & exact(selection.action, action)
            & exact(selection.action, agent_state.control.last_action)
            & exact(
                jax.random.key_data(selection.rng_key_after),
                jax.random.key_data(agent_state.control.rng_key),
            )
        )
        policy_replay_valid = (
            exact(selection.noisy_greedy_action, replayed.noisy_greedy_action)
            & exact(selection.random_action, replayed.random_action)
            & exact(selection.explored, replayed.explored)
            & exact(ordinary_policy_action, replayed.action)
            & exact(
                jax.random.key_data(selection.rng_key_before),
                jax.random.key_data(replayed.rng_key_before),
            )
            & exact(
                jax.random.key_data(selection.rng_key_after),
                jax.random.key_data(replayed.rng_key_after),
            )
        )
        action_policy_valid = self._selection_policy_valid(
            action,
            step_count,
            selection,
        )
        action_valid = selection_binding_valid
        return (
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
        )

    @staticmethod
    def _planner_selections_equal(
        left: IntegratedPlannerSelection,
        right: IntegratedPlannerSelection,
    ) -> Array:
        """Return exact equality, including the raw words of typed PRNG keys."""

        return (
            jnp.array_equal(left.action, right.action)
            & jnp.array_equal(left.noisy_greedy_action, right.noisy_greedy_action)
            & jnp.array_equal(left.random_action, right.random_action)
            & jnp.array_equal(left.explored, right.explored)
            & jnp.array_equal(left.externally_forced, right.externally_forced)
            & jnp.array_equal(
                jax.random.key_data(left.rng_key_before),
                jax.random.key_data(right.rng_key_before),
            )
            & jnp.array_equal(
                jax.random.key_data(left.rng_key_after),
                jax.random.key_data(right.rng_key_after),
            )
        )

    def initialize(
        self,
        world_key: Array,
        agent_key: Array,
    ) -> HiddenPartnerWorldOnlineState:
        for name, key in (("world_key", world_key), ("agent_key", agent_key)):
            if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
                getattr(key, "dtype", None),
                jax.dtypes.prng_key,
            ):
                raise TypeError(f"{name} must be a scalar typed PRNG key")
            key_data = jax.random.key_data(key)
            if (
                str(jax.random.key_impl(key)) != "threefry2x32"
                or getattr(key_data, "shape", None) != (2,)
                or getattr(key_data, "dtype", None) != jnp.dtype(jnp.uint32)
            ):
                raise TypeError(
                    f"{name} must use the exact threefry2x32 uint32[2] key contract"
                )
        world_state = self._world.init(world_key)
        observation = self._world.observe(world_state)
        if self._focal_action_policy == "balanced_external":
            agent_start = self._agent.start_with_forced_action(
                observation,
                agent_key,
                jnp.asarray(self._initial_external_action, dtype=jnp.int32),
            )
        else:
            agent_start = self._agent.start(observation, agent_key)
        cues = observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
        filter_state = self._filter.initialize(cues)
        (
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
        ) = self._agent_selection_contract(
            agent_start.state,
            agent_start.action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        valid = (
            agent_start.diagnostics.all_finite
            & agent_start.diagnostics.descriptors_valid
            & action_valid
            & action_policy_valid
            & selection_binding_valid
            & policy_replay_valid
        )
        state = HiddenPartnerWorldOnlineState(
            world=world_state,
            agent=agent_start.state,
            world_filter=filter_state,
            config_token=self._config_token,
            action=agent_start.action,
            valid=valid,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        self._capture_static_state_contract(state)
        return state

    def _entry_contract(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        learner_counters = jnp.stack(
            (
                state.step_count,
                state.world.step_count,
                state.agent.step_count,
            )
        )
        counters_synchronized = (
            jnp.all(learner_counters >= 0)
            & jnp.all(learner_counters < maximum)
            & jnp.all(learner_counters == learner_counters[0])
        )
        config_token_valid = jnp.array_equal(state.config_token, self._config_token)
        (
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
        ) = self._agent_selection_contract(
            state.agent,
            state.action,
            state.step_count,
        )
        filter_entry_valid = (
            (state.world_filter.step_count >= 0)
            & (state.world_filter.step_count < maximum)
            & (state.world_filter.step_count == state.step_count)
            & state.world_filter.valid
            & jnp.isfinite(state.world_filter.posterior_mean)
            & (jnp.abs(state.world_filter.posterior_mean) <= 1.0)
        )
        entry_valid = (
            config_token_valid
            & counters_synchronized
            & action_valid
            & action_policy_valid
            & selection_binding_valid
            & policy_replay_valid
        )
        return (
            config_token_valid,
            counters_synchronized,
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
            filter_entry_valid,
            entry_valid,
        )

    @staticmethod
    def _neutral_mechanism_trace() -> HiddenPartnerWorldMechanismTrace:
        """Return the sole deterministic sentinel for absent/rejected mechanisms."""

        false = jnp.asarray(False, dtype=jnp.bool_)
        zero_float = jnp.asarray(0.0, dtype=jnp.float32)
        zero_int = jnp.asarray(0, dtype=jnp.int32)
        absent_int = jnp.asarray(-1, dtype=jnp.int32)
        zero_heads = jnp.zeros((_GROUNDED_TARGET_DIM,), dtype=jnp.float32)
        zero_chi = jnp.zeros((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32)
        zero_phi = jnp.zeros((BASE_FEATURE_DIM,), dtype=jnp.float32)
        false_rows = jnp.zeros((_JOINT_ACTION_ROWS,), dtype=jnp.bool_)
        false_active = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
        false_candidates = jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_)
        zero_active = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32)
        zero_candidates = jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.float32)
        zero_candidate_streaks = jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,),
            dtype=jnp.int32,
        )
        absent_descriptors = jnp.full(
            (ACTIVE_PAIR_SLOTS, 2),
            -1,
            dtype=jnp.int32,
        )
        return HiddenPartnerWorldMechanismTrace(
            valid=false,
            grounded_enabled=false,
            grounded_learning_enabled=false,
            grounded_prediction_valid=false,
            grounded_target_valid=false,
            grounded_gradient_valid=false,
            grounded_prediction_matches_decision=false,
            grounded_row_update_isolated=false,
            grounded_update_applied=false,
            grounded_executed_joint_index=absent_int,
            grounded_feature_contribution=zero_heads,
            grounded_row_bias=zero_heads,
            grounded_raw_predictions=zero_heads,
            grounded_targets=zero_heads,
            grounded_errors=zero_heads,
            grounded_fit_loss_by_head=zero_heads,
            grounded_representation_loss_by_head=zero_heads,
            grounded_representation_gradient=zero_chi,
            grounded_representation_gradient_by_head=jnp.zeros(
                (_GROUNDED_TARGET_DIM, DEPLOYED_FEATURE_DIM),
                dtype=jnp.float32,
            ),
            grounded_representation_gradient_norm_by_head=zero_heads,
            grounded_proposed_weight_row_bit_change_mask=false_rows,
            grounded_proposed_bias_row_bit_change_mask=false_rows,
            grounded_executed_weight_row_delta_norm_by_head=zero_heads,
            grounded_executed_bias_row_delta_by_head=zero_heads,
            lifecycle_enabled=false,
            lifecycle_proposed=false,
            lifecycle_applied=false,
            lifecycle_pre_descriptors=absent_descriptors,
            lifecycle_proposal_descriptors=absent_descriptors,
            lifecycle_applied_descriptors=absent_descriptors,
            lifecycle_proposal_replaced_slot=absent_int,
            lifecycle_proposal_promoted_candidate=absent_int,
            lifecycle_proposal_refreshed_candidate=absent_int,
            lifecycle_proposal_retired_slot=absent_int,
            lifecycle_proposal_retired_left=absent_int,
            lifecycle_proposal_retired_right=absent_int,
            lifecycle_applied_replaced_slot=absent_int,
            lifecycle_applied_promoted_candidate=absent_int,
            lifecycle_applied_refreshed_candidate=absent_int,
            lifecycle_applied_retired_slot=absent_int,
            lifecycle_applied_retired_left=absent_int,
            lifecycle_applied_retired_right=absent_int,
            lifecycle_active_evidence_refreshed=false_active,
            lifecycle_retention_evidence_refreshed=false_active,
            lifecycle_durable_read_mask=false_active,
            lifecycle_relevance_probe_scores=zero_active,
            lifecycle_relevance_probe_errors=jnp.zeros(
                (1, ACTIVE_PAIR_SLOTS),
                dtype=jnp.float32,
            ),
            lifecycle_candidate_promotion_signal=zero_candidates,
            lifecycle_candidate_promotion_raw_evidence=false_candidates,
            lifecycle_candidate_promotion_confirmed=false_candidates,
            lifecycle_candidate_promotion_evidence_streak_pre=(
                zero_candidate_streaks
            ),
            lifecycle_candidate_promotion_evidence_streak_updated=(
                zero_candidate_streaks
            ),
            lifecycle_candidate_promotion_evidence_streak_proposal_post=(
                zero_candidate_streaks
            ),
            lifecycle_candidate_promotion_evidence_streak_post=(
                zero_candidate_streaks
            ),
            lifecycle_candidate_reacquisition_required_pre=false_candidates,
            lifecycle_candidate_reacquisition_required_proposal_post=false_candidates,
            lifecycle_candidate_reacquisition_required_post=false_candidates,
            lifecycle_candidate_reacquisition_confirmed=false_candidates,
            lifecycle_candidate_reset_mask=false_candidates,
            lifecycle_applied_candidate_reset_mask=false_candidates,
            random_curation_enabled=false,
            random_curation_attempted=false,
            random_curation_applied=false,
            random_curation_selected_active_worst_slot=absent_int,
            random_curation_selected_promotion_candidate=absent_int,
            random_curation_selected_refresh_candidate=absent_int,
            random_curation_active_priorities=jnp.zeros(
                (ACTIVE_PAIR_SLOTS,),
                dtype=jnp.float32,
            ),
            random_curation_candidate_priorities=jnp.zeros(
                (CANDIDATE_PAIR_SLOTS,),
                dtype=jnp.float32,
            ),
            consumer_read_acquire_pre=false_active,
            consumer_read_acquire_post=false_active,
            consumer_confirmed_write_pre=false_active,
            consumer_confirmed_write_post=false_active,
            consumer_read_mask_pre=false_active,
            consumer_read_mask_post=false_active,
            consumer_active_mask_pre=false_active,
            consumer_active_mask_post=false_active,
            router_valid=false,
            router_applied=false,
            router_carry_survivors=false,
            router_descriptors_changed=false,
            router_source_slots=jnp.full(
                (ACTIVE_PAIR_SLOTS,),
                -1,
                dtype=jnp.int32,
            ),
            router_survivor_mask=false_active,
            router_new_mask=false_active,
            router_evicted_mask=false_active,
            router_route_count_before=zero_int,
            router_route_count_after=zero_int,
            router_generation_count_before=zero_int,
            router_generation_count_after=zero_int,
            consumer_route_source_slots_exact=false,
            consumer_route_identity_masks_exact=false,
            consumer_route_stable_prefix_exact=false,
            consumer_route_survivor_values_exact=false,
            consumer_route_reset_values_exact=false,
            consumer_route_no_carry_reset_exact=false,
            consumer_route_behavior_values_exact=false,
            consumer_route_q_values_exact=false,
            consumer_route_trace_values_exact=false,
            consumer_route_last_observation_exact=false,
            consumer_route_grounded_values_exact=false,
            consumer_route_values_exact=false,
            consumer_lifecycle_destination_reset_exact=false,
            behavior_prediction_matches_decision=false,
            behavior_credit_gradient_chi=zero_chi,
            behavior_credit_gradient_phi=zero_phi,
            behavior_credit_valid=false,
            grounded_credit_gradient_chi=zero_chi,
            grounded_credit_valid=false,
            mixed_credit_gradient_chi=zero_chi,
            mixed_credit_gradient_phi=zero_phi,
            mixed_credit_valid=false,
            mixed_credit_applied=false,
            mixed_credit_conflict=false,
            state_learning_enabled=false,
            state_learning_proposal_valid=false,
            state_learning_component_applied=false,
            state_learning_committed=false,
            state_learning_valid=false,
            state_learning_rejected=false,
            state_learning_gradient_norm=zero_float,
            state_learning_clipped_gradient_norm=zero_float,
            state_learning_parameter_update_norm=zero_float,
            state_builder_step_delta=zero_int,
            state_builder_learning_delta=zero_int,
            behavior_step_delta=zero_int,
            interaction_step_delta=zero_int,
            table_world_step_delta=zero_int,
            grounded_world_step_delta=zero_int,
            control_step_delta=zero_int,
            router_route_delta=zero_int,
            router_generation_delta=zero_int,
            integrated_step_delta=zero_int,
        )

    def _project_mechanism_trace(
        self,
        state: IntegratedHiddenPartnerState,
        diagnostics: IntegratedUpdateDiagnostics,
        accepted: Array,
    ) -> HiddenPartnerWorldMechanismTrace:
        """Project accepted learner diagnostics without retaining learner state."""

        neutral = self._neutral_mechanism_trace()
        route = diagnostics.route
        state_learning = diagnostics.state_learning
        behavior_credit_valid = (
            jnp.all(jnp.isfinite(diagnostics.behavior_gradient_chi))
            & jnp.all(jnp.isfinite(diagnostics.behavior_gradient_phi))
        )
        proposed = neutral.replace(
            valid=diagnostics.all_finite,
            lifecycle_enabled=jnp.asarray(
                self._agent.config.feature_lifecycle_enabled,
                dtype=jnp.bool_,
            ),
            lifecycle_proposed=diagnostics.interaction_lifecycle_proposed,
            lifecycle_applied=diagnostics.interaction_lifecycle_applied,
            lifecycle_pre_descriptors=state.router.descriptors,
            lifecycle_proposal_descriptors=(
                diagnostics.interaction_proposal_descriptors
            ),
            lifecycle_applied_descriptors=diagnostics.interaction_applied_descriptors,
            lifecycle_proposal_replaced_slot=(
                diagnostics.interaction_proposal_replaced_slot
            ),
            lifecycle_proposal_promoted_candidate=(
                diagnostics.interaction_proposal_promoted_candidate
            ),
            lifecycle_proposal_refreshed_candidate=(
                diagnostics.interaction_proposal_refreshed_candidate
            ),
            lifecycle_proposal_retired_slot=(
                diagnostics.interaction_proposal_retired_slot
            ),
            lifecycle_proposal_retired_left=(
                diagnostics.interaction_proposal_retired_left
            ),
            lifecycle_proposal_retired_right=(
                diagnostics.interaction_proposal_retired_right
            ),
            lifecycle_applied_replaced_slot=(
                diagnostics.interaction_applied_replaced_slot
            ),
            lifecycle_applied_promoted_candidate=(
                diagnostics.interaction_applied_promoted_candidate
            ),
            lifecycle_applied_refreshed_candidate=(
                diagnostics.interaction_applied_refreshed_candidate
            ),
            lifecycle_applied_retired_slot=(
                diagnostics.interaction_applied_retired_slot
            ),
            lifecycle_applied_retired_left=(
                diagnostics.interaction_applied_retired_left
            ),
            lifecycle_applied_retired_right=(
                diagnostics.interaction_applied_retired_right
            ),
            lifecycle_active_evidence_refreshed=(
                diagnostics.interaction_evidence_refreshed
            ),
            lifecycle_retention_evidence_refreshed=(
                diagnostics.interaction_retention_evidence_refreshed
            ),
            lifecycle_durable_read_mask=diagnostics.interaction_durable_read_mask,
            lifecycle_relevance_probe_scores=(
                diagnostics.interaction_relevance_probe_scores
            ),
            lifecycle_relevance_probe_errors=(
                diagnostics.interaction_relevance_probe_errors
            ),
            lifecycle_candidate_promotion_signal=(
                diagnostics.interaction_candidate_promotion_signal
            ),
            lifecycle_candidate_promotion_raw_evidence=(
                diagnostics.interaction_candidate_promotion_raw_evidence
            ),
            lifecycle_candidate_promotion_confirmed=(
                diagnostics.interaction_candidate_promotion_confirmed
            ),
            lifecycle_candidate_promotion_evidence_streak_pre=(
                diagnostics.interaction_candidate_promotion_evidence_streak_pre
            ),
            lifecycle_candidate_promotion_evidence_streak_updated=(
                diagnostics.interaction_candidate_promotion_evidence_streak_updated
            ),
            lifecycle_candidate_promotion_evidence_streak_proposal_post=(
                diagnostics.interaction_candidate_promotion_evidence_streak_proposal_post
            ),
            lifecycle_candidate_promotion_evidence_streak_post=(
                diagnostics.interaction_candidate_promotion_evidence_streak_post
            ),
            lifecycle_candidate_reacquisition_required_pre=(
                diagnostics.interaction_candidate_reacquisition_required_pre
            ),
            lifecycle_candidate_reacquisition_required_proposal_post=(
                diagnostics.interaction_candidate_reacquisition_required_proposal_post
            ),
            lifecycle_candidate_reacquisition_required_post=(
                diagnostics.interaction_candidate_reacquisition_required_post
            ),
            lifecycle_candidate_reacquisition_confirmed=(
                diagnostics.interaction_candidate_reacquisition_confirmed
            ),
            lifecycle_candidate_reset_mask=(
                diagnostics.interaction_matching_candidate_reset_mask
            ),
            lifecycle_applied_candidate_reset_mask=(
                diagnostics.interaction_applied_matching_candidate_reset_mask
            ),
            random_curation_enabled=diagnostics.random_curation_enabled,
            random_curation_attempted=diagnostics.random_curation_attempted,
            random_curation_applied=diagnostics.random_curation_applied,
            random_curation_selected_active_worst_slot=(
                diagnostics.curation_selected_active_worst_slot
            ),
            random_curation_selected_promotion_candidate=(
                diagnostics.curation_selected_promotion_candidate
            ),
            random_curation_selected_refresh_candidate=(
                diagnostics.curation_selected_refresh_candidate
            ),
            random_curation_active_priorities=(
                diagnostics.random_active_priorities
            ),
            random_curation_candidate_priorities=(
                diagnostics.random_candidate_priorities
            ),
            consumer_read_acquire_pre=diagnostics.consumer_read_acquire_pre,
            consumer_read_acquire_post=diagnostics.consumer_read_acquire_post,
            consumer_confirmed_write_pre=diagnostics.consumer_confirmed_write_pre,
            consumer_confirmed_write_post=diagnostics.consumer_confirmed_write_post,
            consumer_read_mask_pre=diagnostics.consumer_read_mask_pre,
            consumer_read_mask_post=diagnostics.consumer_read_mask_post,
            consumer_active_mask_pre=diagnostics.consumer_active_mask_pre,
            consumer_active_mask_post=diagnostics.consumer_active_mask_post,
            router_valid=route.valid,
            router_applied=route.route_applied,
            router_carry_survivors=route.carry_survivors,
            router_descriptors_changed=route.descriptors_changed,
            router_source_slots=route.source_slots,
            router_survivor_mask=route.survivor_mask,
            router_new_mask=route.new_mask,
            router_evicted_mask=route.evicted_mask,
            router_route_count_before=route.route_count_before,
            router_route_count_after=route.route_count_after,
            router_generation_count_before=route.generation_count_before,
            router_generation_count_after=route.generation_count_after,
            consumer_route_source_slots_exact=(
                diagnostics.consumer_route_source_slots_exact
            ),
            consumer_route_identity_masks_exact=(
                diagnostics.consumer_route_identity_masks_exact
            ),
            consumer_route_stable_prefix_exact=(
                diagnostics.consumer_route_stable_prefix_exact
            ),
            consumer_route_survivor_values_exact=(
                diagnostics.consumer_route_survivor_values_exact
            ),
            consumer_route_reset_values_exact=(
                diagnostics.consumer_route_reset_values_exact
            ),
            consumer_route_no_carry_reset_exact=(
                diagnostics.consumer_route_no_carry_reset_exact
            ),
            consumer_route_behavior_values_exact=(
                diagnostics.consumer_route_behavior_values_exact
            ),
            consumer_route_q_values_exact=(
                diagnostics.consumer_route_q_values_exact
            ),
            consumer_route_trace_values_exact=(
                diagnostics.consumer_route_trace_values_exact
            ),
            consumer_route_last_observation_exact=(
                diagnostics.consumer_route_last_observation_exact
            ),
            consumer_route_grounded_values_exact=(
                diagnostics.consumer_route_grounded_values_exact
            ),
            consumer_route_values_exact=diagnostics.consumer_route_values_exact,
            consumer_lifecycle_destination_reset_exact=(
                diagnostics.consumer_lifecycle_destination_reset_exact
            ),
            behavior_prediction_matches_decision=(
                diagnostics.behavior_prediction_matches_decision
            ),
            behavior_credit_gradient_chi=diagnostics.behavior_gradient_chi,
            behavior_credit_gradient_phi=diagnostics.behavior_gradient_phi,
            behavior_credit_valid=behavior_credit_valid,
            state_learning_enabled=jnp.asarray(
                self._agent.config.state_learning_enabled,
                dtype=jnp.bool_,
            ),
            state_learning_proposal_valid=state_learning.proposal_valid,
            state_learning_component_applied=state_learning.applied,
            state_learning_committed=(
                state_learning.applied & self._agent.config.state_learning_enabled
            ),
            state_learning_valid=state_learning.valid,
            state_learning_rejected=state_learning.rejected,
            state_learning_gradient_norm=state_learning.gradient_norm,
            state_learning_clipped_gradient_norm=(
                state_learning.clipped_gradient_norm
            ),
            state_learning_parameter_update_norm=(
                state_learning.parameter_update_norm
            ),
            state_builder_step_delta=diagnostics.state_builder_step_delta,
            state_builder_learning_delta=diagnostics.state_builder_learning_delta,
            behavior_step_delta=diagnostics.behavior_step_delta,
            interaction_step_delta=diagnostics.interaction_step_delta,
            table_world_step_delta=diagnostics.world_step_delta,
            grounded_world_step_delta=diagnostics.grounded_world_step_delta,
            control_step_delta=diagnostics.control_step_delta,
            router_route_delta=diagnostics.router_route_delta,
            router_generation_delta=diagnostics.router_generation_delta,
            integrated_step_delta=diagnostics.integrated_step_delta,
        )
        grounded_update = diagnostics.grounded_world_update
        gradient_mix = diagnostics.gradient_mix
        if grounded_update is None:
            if (
                gradient_mix is not None
                or diagnostics.mixed_gradient_chi is not None
                or diagnostics.mixed_gradient_phi is not None
            ):
                raise ValueError("absent grounded mechanism must not carry mixed credit")
        else:
            if (
                gradient_mix is None
                or diagnostics.mixed_gradient_chi is None
                or diagnostics.mixed_gradient_phi is None
            ):
                raise ValueError("grounded mechanism requires complete mixed credit")
            grounded_diagnostics = grounded_update.diagnostics
            proposed = proposed.replace(
                grounded_enabled=jnp.asarray(True, dtype=jnp.bool_),
                grounded_learning_enabled=diagnostics.grounded_world_learning_enabled,
                grounded_prediction_valid=grounded_update.prediction.valid,
                grounded_target_valid=grounded_diagnostics.target_valid,
                grounded_gradient_valid=grounded_update.gradient_valid,
                grounded_prediction_matches_decision=(
                    diagnostics.grounded_world_prediction_matches_decision
                ),
                grounded_row_update_isolated=(
                    grounded_diagnostics.row_update_isolated
                ),
                grounded_update_applied=grounded_diagnostics.applied,
                grounded_executed_joint_index=(
                    grounded_update.prediction.joint_action_index
                ),
                grounded_feature_contribution=(
                    grounded_update.prediction.feature_contribution
                ),
                grounded_row_bias=grounded_update.prediction.row_bias,
                grounded_raw_predictions=grounded_update.prediction.raw_predictions,
                grounded_targets=grounded_update.targets,
                grounded_errors=grounded_update.errors,
                grounded_fit_loss_by_head=grounded_update.fit_loss_by_head,
                grounded_representation_loss_by_head=(
                    grounded_update.representation_loss_by_head
                ),
                grounded_representation_gradient=(
                    grounded_update.representation_gradient
                ),
                grounded_representation_gradient_by_head=(
                    grounded_update.representation_gradient_by_head
                ),
                grounded_representation_gradient_norm_by_head=(
                    grounded_update.representation_gradient_norm_by_head
                ),
                grounded_proposed_weight_row_bit_change_mask=(
                    grounded_update.proposed_weight_row_bit_change_mask
                ),
                grounded_proposed_bias_row_bit_change_mask=(
                    grounded_update.proposed_bias_row_bit_change_mask
                ),
                grounded_executed_weight_row_delta_norm_by_head=(
                    grounded_update.executed_weight_row_delta_norm_by_head
                ),
                grounded_executed_bias_row_delta_by_head=(
                    grounded_update.executed_bias_row_delta_by_head
                ),
                grounded_credit_gradient_chi=(
                    grounded_update.representation_gradient
                ),
                grounded_credit_valid=grounded_update.gradient_valid,
                mixed_credit_gradient_chi=diagnostics.mixed_gradient_chi,
                mixed_credit_gradient_phi=diagnostics.mixed_gradient_phi,
                mixed_credit_valid=gradient_mix.valid,
                mixed_credit_applied=gradient_mix.applied,
                mixed_credit_conflict=gradient_mix.diagnostics.conflict,
            )
        return cast(
            HiddenPartnerWorldMechanismTrace,
            jax.tree_util.tree_map(
                lambda proposed_leaf, neutral_leaf: jnp.where(
                    accepted,
                    proposed_leaf,
                    neutral_leaf,
                ),
                proposed,
                neutral,
            ),
        )

    def _neutral_trace(
        self,
        state: HiddenPartnerWorldOnlineState,
        *,
        active: Array,
    ) -> HiddenPartnerWorldOnlineTrace:
        (
            config_token_valid,
            counters_synchronized,
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
            filter_entry_valid,
            entry_valid,
        ) = self._entry_contract(state)
        neutral_cells = jnp.full((2, 2), 0.5, dtype=jnp.float32)
        neutral_actions = jnp.full((2,), 0.5, dtype=jnp.float32)
        safe_observation = jnp.where(
            jnp.isfinite(state.agent.raw_observation),
            state.agent.raw_observation,
            0.0,
        )
        safe_phi = jnp.where(jnp.isfinite(state.agent.phi), state.agent.phi, 0.0)
        safe_chi = jnp.where(jnp.isfinite(state.agent.chi), state.agent.chi, 0.0)
        safe_action = jnp.where(action_valid, state.action, jnp.asarray(0, dtype=jnp.int32))
        selection = state.agent.current_selection
        ordinary_policy_action = self._ordinary_policy_action(selection)
        expected_action = self._expected_policy_action(state.step_count, selection)
        safe_expected_action = jnp.where(
            (expected_action >= 0) & (expected_action < 2),
            expected_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        safe_noisy_greedy_action = jnp.where(
            (selection.noisy_greedy_action >= 0)
            & (selection.noisy_greedy_action < 2),
            selection.noisy_greedy_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        safe_random_action = jnp.where(
            (selection.random_action >= 0) & (selection.random_action < 2),
            selection.random_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        safe_ordinary_policy_action = jnp.where(
            (ordinary_policy_action >= 0) & (ordinary_policy_action < 2),
            ordinary_policy_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        policy_rng_before = jax.random.key_data(selection.rng_key_before)
        policy_rng_after = jax.random.key_data(selection.rng_key_after)
        safe_filter_mean = jnp.where(
            jnp.isfinite(state.world_filter.posterior_mean)
            & (jnp.abs(state.world_filter.posterior_mean) <= 1.0),
            state.world_filter.posterior_mean,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        zero_delta = jnp.asarray(0, dtype=jnp.int32)
        return HiddenPartnerWorldOnlineTrace(
            active=active,
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            step=state.step_count,
            entry_state_contract_valid=entry_valid,
            config_token_valid=config_token_valid,
            counters_synchronized=counters_synchronized,
            action_valid=action_valid,
            action_policy_valid=action_policy_valid,
            selection_binding_valid=selection_binding_valid,
            policy_replay_valid=policy_replay_valid,
            filter_entry_valid=filter_entry_valid,
            observation_pre=safe_observation,
            phi_pre=safe_phi,
            chi_pre=safe_chi,
            focal_action=safe_action,
            expected_focal_action=safe_expected_action,
            focal_action_externally_forced=selection.externally_forced,
            focal_action_noisy_greedy_action=safe_noisy_greedy_action,
            focal_action_random_action=safe_random_action,
            focal_action_explored=selection.explored,
            focal_action_ordinary_policy_action=safe_ordinary_policy_action,
            focal_action_policy_rng_before=policy_rng_before,
            focal_action_policy_rng_after=policy_rng_after,
            partner_action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            outcome=jnp.asarray(0.0, dtype=jnp.float32),
            corrected_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            next_observation=safe_observation,
            next_action=safe_action,
            expected_next_action=safe_expected_action,
            next_action_policy_valid=action_policy_valid,
            next_selection_binding_valid=selection_binding_valid,
            next_policy_replay_valid=policy_replay_valid,
            next_selection_diagnostics_valid=jnp.asarray(False, dtype=jnp.bool_),
            next_action_externally_forced=selection.externally_forced,
            next_action_noisy_greedy_action=safe_noisy_greedy_action,
            next_action_random_action=safe_random_action,
            next_action_explored=selection.explored,
            next_action_ordinary_policy_action=safe_ordinary_policy_action,
            next_action_policy_rng_before=policy_rng_before,
            next_action_policy_rng_after=policy_rng_after,
            filter_mean_pre=safe_filter_mean,
            filter_mean_post=safe_filter_mean,
            agent_partner_belief_conditioned_reward_cells=neutral_cells,
            agent_applied_partner_probabilities=neutral_actions,
            agent_partner_belief_conditioned_expected_rewards=neutral_actions,
            agent_partner_belief_conditioned_greedy_action=jnp.asarray(0, dtype=jnp.int32),
            agent_partner_belief_conditioned_selected_regret=jnp.asarray(
                0.0, dtype=jnp.float32
            ),
            agent_partner_belief_conditioned_action_margin=jnp.asarray(
                0.0, dtype=jnp.float32
            ),
            agent_partner_belief_conditioned_tied=jnp.asarray(True, dtype=jnp.bool_),
            oracle_step_count=jnp.asarray(-1, dtype=jnp.int32),
            oracle_cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            oracle_cycle_step=jnp.asarray(-1, dtype=jnp.int32),
            oracle_cycle_length=jnp.asarray(-1, dtype=jnp.int32),
            oracle_segment_index=jnp.asarray(-1, dtype=jnp.int32),
            oracle_segment_step=jnp.asarray(-1, dtype=jnp.int32),
            oracle_segment_length=jnp.asarray(-1, dtype=jnp.int32),
            oracle_world_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_next_world_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_world_flipped=jnp.asarray(False, dtype=jnp.bool_),
            oracle_outcome_flipped=jnp.asarray(False, dtype=jnp.bool_),
            oracle_regime_id=jnp.asarray(-1, dtype=jnp.int32),
            oracle_next_cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            oracle_next_segment_index=jnp.asarray(-1, dtype=jnp.int32),
            oracle_next_regime_id=jnp.asarray(-1, dtype=jnp.int32),
            oracle_schedule_switched=jnp.asarray(False, dtype=jnp.bool_),
            oracle_partner_intended_action=jnp.asarray(-1, dtype=jnp.int32),
            oracle_partner_intended_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_partner_flipped=jnp.asarray(False, dtype=jnp.bool_),
            oracle_focal_action_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_partner_action_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_world_cue_flipped=jnp.zeros((2,), dtype=jnp.bool_),
            oracle_next_world_cue_flipped=jnp.zeros((2,), dtype=jnp.bool_),
            oracle_full_information_action=jnp.asarray(0, dtype=jnp.int32),
            oracle_full_information_action_margin=jnp.asarray(
                0.0, dtype=jnp.float32
            ),
            oracle_full_information_action_tied=jnp.asarray(True, dtype=jnp.bool_),
            oracle_realized_counterfactual_rewards=neutral_actions,
            proposed_agent_update_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_filter_update_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_filter_decision_valid=jnp.asarray(False, dtype=jnp.bool_),
            learner_trace_valid=jnp.asarray(False, dtype=jnp.bool_),
            filter_trace_valid=jnp.asarray(False, dtype=jnp.bool_),
            oracle_trace_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_world_step_delta=zero_delta,
            proposed_agent_step_delta=zero_delta,
            proposed_filter_step_delta=zero_delta,
            proposed_bridge_step_delta=zero_delta,
            committed_world_step_delta=zero_delta,
            committed_agent_step_delta=zero_delta,
            committed_filter_step_delta=zero_delta,
            committed_bridge_step_delta=zero_delta,
            mechanism=self._neutral_mechanism_trace(),
            all_finite=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _blocked_step(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> HiddenPartnerWorldOnlineStep:
        return HiddenPartnerWorldOnlineStep(
            state=state,
            trace=self._neutral_trace(
                state,
                active=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def _entry_rejected_step(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> HiddenPartnerWorldOnlineStep:
        return HiddenPartnerWorldOnlineStep(
            state=state.replace(valid=jnp.asarray(False, dtype=jnp.bool_)),
            trace=self._neutral_trace(
                state,
                active=jnp.asarray(True, dtype=jnp.bool_),
            ),
        )

    def _advance(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> HiddenPartnerWorldOnlineStep:
        filter_cells = self._filter.expected_reward_cells(state.world_filter.posterior_mean)
        partner_probabilities = state.agent.current_evaluation.partner_probabilities
        filter_decision = self._filter.marginalize_partner(
            filter_cells,
            partner_probabilities,
        )

        world_transition, proposed_world = self._world.step(state.world, state.action)
        learner_transition = strip_hidden_partner_world_oracle(world_transition)
        if self._focal_action_policy == "balanced_external":
            next_forced_action = self._expected_policy_action(
                state.step_count + jnp.asarray(1, dtype=jnp.int32),
                state.agent.current_selection,
            )
            agent_result = self._agent.update_with_forced_next_action(
                state.agent,
                learner_transition,
                next_forced_action,
            )
        else:
            agent_result = self._agent.update(state.agent, learner_transition)

        focal_sign = 2.0 * learner_transition.focal_action.astype(jnp.float32) - 1.0
        partner_sign = 2.0 * learner_transition.partner_action.astype(jnp.float32) - 1.0
        corrected_outcome = learner_transition.outcome * focal_sign * partner_sign
        next_cues = learner_transition.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
        filter_result = self._filter.advance(
            state.world_filter,
            corrected_outcome,
            next_cues,
        )

        proposed_world_step_delta = proposed_world.step_count - state.world.step_count
        proposed_agent_step_delta = agent_result.state.step_count - state.agent.step_count
        proposed_filter_step_delta = (
            filter_result.state.step_count - state.world_filter.step_count
        )
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        proposed_count = jnp.minimum(state.step_count, maximum - 1) + 1
        proposed_bridge_step_delta = proposed_count - state.step_count
        agent_update_valid = (
            agent_result.diagnostics.transition_semantics_valid
            & ~agent_result.diagnostics.transition_rejected
        )
        (
            config_token_valid,
            counters_synchronized,
            action_valid,
            action_policy_valid,
            selection_binding_valid,
            policy_replay_valid,
            filter_entry_valid,
            entry_valid,
        ) = self._entry_contract(state)
        proposed_learner_counters = jnp.stack(
            (
                proposed_count,
                proposed_world.step_count,
                agent_result.state.step_count,
            )
        )
        proposed_counters_synchronized = jnp.all(
            proposed_learner_counters == proposed_learner_counters[0]
        )
        (
            next_action_valid,
            next_action_policy_valid,
            next_selection_binding_valid,
            next_policy_replay_valid,
        ) = self._agent_selection_contract(
            agent_result.state,
            agent_result.action,
            proposed_count,
        )
        committed_next_selection = agent_result.state.current_selection
        diagnostic_next_selection = agent_result.diagnostics.next_selection
        next_selection_diagnostics_valid = self._planner_selections_equal(
            committed_next_selection,
            diagnostic_next_selection,
        )
        next_rng_continuity_valid = jnp.array_equal(
            jax.random.key_data(committed_next_selection.rng_key_before),
            jax.random.key_data(state.agent.current_selection.rng_key_after),
        )
        next_selection_binding_valid = (
            next_selection_binding_valid & next_rng_continuity_valid
        )
        oracle = world_transition.oracle
        learner_path_values_finite = (
            jnp.all(jnp.isfinite(learner_transition.observation))
            & jnp.all(jnp.isfinite(learner_transition.next_observation))
            & jnp.isfinite(learner_transition.reward)
            & jnp.isfinite(learner_transition.outcome)
            & jnp.isfinite(corrected_outcome)
            & jnp.all(jnp.isfinite(partner_probabilities))
        )
        filter_values_finite = (
            jnp.isfinite(state.world_filter.posterior_mean)
            & jnp.isfinite(filter_result.state.posterior_mean)
            & jnp.all(jnp.isfinite(filter_cells.rewards))
            & jnp.all(jnp.isfinite(filter_decision.expected_rewards))
            & jnp.all(jnp.isfinite(filter_decision.action_regrets))
            & jnp.isfinite(filter_decision.action_margin)
        )
        filter_trace_valid = (
            filter_entry_valid
            & filter_cells.valid
            & filter_result.valid
            & filter_decision.valid
            & (proposed_filter_step_delta == 1)
            & filter_values_finite
        )
        current_schedule = self._world._schedule_position(  # noqa: SLF001
            state.world,
            state.world.step_count,
        )
        next_schedule = self._world._schedule_position(  # noqa: SLF001
            proposed_world,
            proposed_world.step_count,
        )
        expected_intended_sign = self._world._partner_intended_sign(  # noqa: SLF001
            state.world,
            current_schedule.regime_id,
        )
        expected_intended_action = ((expected_intended_sign + 1.0) / 2.0).astype(
            jnp.int32
        )
        expected_schedule_switched = (
            (current_schedule.cycle_index != next_schedule.cycle_index)
            | (current_schedule.segment_index != next_schedule.segment_index)
        )
        expected_partner_sign = jnp.where(
            oracle.partner_flipped,
            -expected_intended_sign,
            expected_intended_sign,
        )
        expected_next_world_sign = jnp.where(
            oracle.world_flipped,
            -state.world.world_sign,
            state.world.world_sign,
        )
        expected_noiseless_outcome = state.world.world_sign * focal_sign * partner_sign
        expected_contextual_outcome = jnp.where(
            oracle.outcome_flipped,
            -expected_noiseless_outcome,
            expected_noiseless_outcome,
        )
        focal_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        expected_outcome_noise_sign = jnp.where(
            oracle.outcome_flipped,
            jnp.asarray(-1.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        expected_counterfactual_rewards = (
            1.0
            + state.world.world_sign
            * focal_signs
            * partner_sign
            * expected_outcome_noise_sign
        ) / 2.0
        expected_full_information_coefficient = jnp.asarray(
            (1.0 - 2.0 * self._world.config.partner_flip_probability)
            * (1.0 - 2.0 * self._world.config.outcome_flip_probability),
            dtype=jnp.float32,
        ) * (state.world.world_sign * expected_intended_sign)
        expected_optimal_sign = jnp.where(
            expected_full_information_coefficient >= 0.0,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-1.0, dtype=jnp.float32),
        )
        expected_optimal_action = ((expected_optimal_sign + 1.0) / 2.0).astype(
            jnp.int32
        )
        expected_action_margin = jnp.abs(expected_full_information_coefficient)
        expected_action_tied = expected_full_information_coefficient == 0.0
        oracle_trace_valid = (
            (oracle.step_count == state.world.step_count)
            & (oracle.step_count == state.step_count)
            & (oracle.cycle_index == current_schedule.cycle_index)
            & (oracle.cycle_step == current_schedule.cycle_step)
            & (oracle.cycle_length == current_schedule.cycle_length)
            & (oracle.segment_index == current_schedule.segment_index)
            & (oracle.segment_step == current_schedule.segment_step)
            & (oracle.segment_length == current_schedule.segment_length)
            & (oracle.regime_id == current_schedule.regime_id)
            & (oracle.next_cycle_index == next_schedule.cycle_index)
            & (oracle.next_segment_index == next_schedule.segment_index)
            & (oracle.next_regime_id == next_schedule.regime_id)
            & (oracle.schedule_switched == expected_schedule_switched)
            & (oracle.partner_intended_action == expected_intended_action)
            & (oracle.partner_intended_sign == expected_intended_sign)
            & (oracle.partner_action_sign == expected_partner_sign)
            & (oracle.partner_action_sign == partner_sign)
            & (oracle.focal_action_sign == focal_sign)
            & (oracle.world_sign == state.world.world_sign)
            & jnp.array_equal(
                oracle.world_cue_flipped,
                state.world.current_cues != state.world.world_sign,
            )
            & (oracle.next_world_sign == proposed_world.world_sign)
            & (oracle.next_world_sign == expected_next_world_sign)
            & jnp.array_equal(
                oracle.next_world_cue_flipped,
                proposed_world.current_cues != proposed_world.world_sign,
            )
            & (oracle.noiseless_contextual_outcome == expected_noiseless_outcome)
            & (oracle.contextual_outcome == expected_contextual_outcome)
            & (oracle.contextual_outcome == learner_transition.outcome)
            & jnp.isfinite(oracle.world_sign)
            & ((oracle.world_sign == -1.0) | (oracle.world_sign == 1.0))
            & jnp.isfinite(oracle.next_world_sign)
            & ((oracle.next_world_sign == -1.0) | (oracle.next_world_sign == 1.0))
            & ((oracle.partner_intended_sign == -1.0) | (oracle.partner_intended_sign == 1.0))
            & ((oracle.focal_action_sign == -1.0) | (oracle.focal_action_sign == 1.0))
            & ((oracle.partner_action_sign == -1.0) | (oracle.partner_action_sign == 1.0))
            & jnp.array_equal(
                oracle.counterfactual_rewards,
                expected_counterfactual_rewards,
            )
            & (oracle.regime_id >= 0)
            & (oracle.regime_id < 4)
            & (
                oracle.full_information_optimal_focal_action
                == expected_optimal_action
            )
            & (oracle.full_information_action_margin == expected_action_margin)
            & (oracle.full_information_action_tied == expected_action_tied)
        )
        accepted = (
            state.valid
            & entry_valid
            & agent_update_valid
            & agent_result.diagnostics.all_finite
            & (proposed_world_step_delta == 1)
            & (proposed_agent_step_delta == 1)
            & (proposed_bridge_step_delta == 1)
            & proposed_counters_synchronized
            & next_action_valid
            & next_action_policy_valid
            & next_selection_binding_valid
            & next_policy_replay_valid
            & learner_path_values_finite
        )
        proposed_state = HiddenPartnerWorldOnlineState(
            world=proposed_world,
            agent=agent_result.state,
            world_filter=filter_result.state,
            config_token=state.config_token,
            action=agent_result.action,
            valid=jnp.asarray(True, dtype=jnp.bool_),
            step_count=proposed_count,
        )
        rejected_state = HiddenPartnerWorldOnlineState(
            world=state.world,
            agent=state.agent,
            world_filter=state.world_filter,
            config_token=state.config_token,
            action=state.action,
            valid=jnp.asarray(False, dtype=jnp.bool_),
            step_count=state.step_count,
        )
        next_state = jax.lax.cond(
            accepted,
            lambda _: proposed_state,
            lambda _: rejected_state,
            operand=None,
        )
        neutral_cells = jnp.full((2, 2), 0.5, dtype=jnp.float32)
        neutral_actions = jnp.full((2,), 0.5, dtype=jnp.float32)
        safe_action = jnp.where(action_valid, state.action, jnp.asarray(0, dtype=jnp.int32))
        safe_index = jnp.clip(safe_action, 0, 1)
        current_selection = state.agent.current_selection
        next_selection = committed_next_selection
        current_ordinary_policy_action = self._ordinary_policy_action(current_selection)
        next_ordinary_policy_action = self._ordinary_policy_action(next_selection)
        expected_focal_action = self._expected_policy_action(
            state.step_count,
            current_selection,
        )
        expected_next_action = self._expected_policy_action(
            proposed_count,
            next_selection,
        )
        next_trace_action = jnp.where(accepted, agent_result.action, safe_action)
        next_trace_noisy_greedy = jnp.where(
            accepted,
            next_selection.noisy_greedy_action,
            current_selection.noisy_greedy_action,
        )
        next_trace_random_action = jnp.where(
            accepted,
            next_selection.random_action,
            current_selection.random_action,
        )
        next_trace_ordinary_policy_action = jnp.where(
            accepted,
            next_ordinary_policy_action,
            current_ordinary_policy_action,
        )
        next_trace_expected_action = jnp.where(
            accepted,
            expected_next_action,
            expected_focal_action,
        )
        current_rng_before = jax.random.key_data(current_selection.rng_key_before)
        current_rng_after = jax.random.key_data(current_selection.rng_key_after)
        next_rng_before = jax.random.key_data(next_selection.rng_key_before)
        next_rng_after = jax.random.key_data(next_selection.rng_key_after)
        zero_float = jnp.asarray(0.0, dtype=jnp.float32)
        zero_delta = jnp.asarray(0, dtype=jnp.int32)
        filter_trace_accepted = accepted & filter_trace_valid
        oracle_trace_accepted = accepted & oracle_trace_valid
        trace = HiddenPartnerWorldOnlineTrace(
            active=jnp.asarray(True, dtype=jnp.bool_),
            accepted=accepted,
            step=state.step_count,
            entry_state_contract_valid=entry_valid,
            config_token_valid=config_token_valid,
            counters_synchronized=counters_synchronized,
            action_valid=action_valid,
            action_policy_valid=action_policy_valid,
            selection_binding_valid=selection_binding_valid,
            policy_replay_valid=policy_replay_valid,
            filter_entry_valid=filter_entry_valid,
            observation_pre=jnp.where(
                accepted,
                learner_transition.observation,
                state.agent.raw_observation,
            ),
            phi_pre=jnp.where(accepted, state.agent.phi, jnp.zeros_like(state.agent.phi)),
            chi_pre=jnp.where(accepted, state.agent.chi, jnp.zeros_like(state.agent.chi)),
            focal_action=jnp.where(accepted, learner_transition.focal_action, safe_action),
            expected_focal_action=jnp.where(
                accepted,
                expected_focal_action,
                safe_action,
            ),
            focal_action_externally_forced=(
                accepted & current_selection.externally_forced
            ),
            focal_action_noisy_greedy_action=jnp.where(
                accepted,
                current_selection.noisy_greedy_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            focal_action_random_action=jnp.where(
                accepted,
                current_selection.random_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            focal_action_explored=accepted & current_selection.explored,
            focal_action_ordinary_policy_action=jnp.where(
                accepted,
                current_ordinary_policy_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            focal_action_policy_rng_before=jnp.where(
                accepted,
                current_rng_before,
                jnp.zeros_like(current_rng_before),
            ),
            focal_action_policy_rng_after=jnp.where(
                accepted,
                current_rng_after,
                jnp.zeros_like(current_rng_after),
            ),
            partner_action=jnp.where(
                accepted,
                learner_transition.partner_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            reward=jnp.where(accepted, learner_transition.reward, zero_float),
            outcome=jnp.where(accepted, learner_transition.outcome, zero_float),
            corrected_outcome=jnp.where(accepted, corrected_outcome, zero_float),
            next_observation=jnp.where(
                accepted,
                learner_transition.next_observation,
                state.agent.raw_observation,
            ),
            next_action=next_trace_action,
            expected_next_action=next_trace_expected_action,
            next_action_policy_valid=next_action_policy_valid,
            next_selection_binding_valid=next_selection_binding_valid,
            next_policy_replay_valid=next_policy_replay_valid,
            next_selection_diagnostics_valid=next_selection_diagnostics_valid,
            next_action_externally_forced=(
                accepted & next_selection.externally_forced
            ),
            next_action_noisy_greedy_action=next_trace_noisy_greedy,
            next_action_random_action=next_trace_random_action,
            next_action_explored=jnp.where(
                accepted,
                next_selection.explored,
                current_selection.explored,
            ),
            next_action_ordinary_policy_action=next_trace_ordinary_policy_action,
            next_action_policy_rng_before=jnp.where(
                accepted,
                next_rng_before,
                current_rng_before,
            ),
            next_action_policy_rng_after=jnp.where(
                accepted,
                next_rng_after,
                current_rng_after,
            ),
            filter_mean_pre=jnp.where(
                filter_trace_accepted,
                state.world_filter.posterior_mean,
                zero_float,
            ),
            filter_mean_post=jnp.where(
                filter_trace_accepted,
                filter_result.state.posterior_mean,
                zero_float,
            ),
            agent_partner_belief_conditioned_reward_cells=jnp.where(
                filter_trace_accepted, filter_cells.rewards, neutral_cells
            ),
            agent_applied_partner_probabilities=jnp.where(
                accepted, partner_probabilities, neutral_actions
            ),
            agent_partner_belief_conditioned_expected_rewards=jnp.where(
                filter_trace_accepted,
                filter_decision.expected_rewards,
                neutral_actions,
            ),
            agent_partner_belief_conditioned_greedy_action=jnp.where(
                filter_trace_accepted,
                filter_decision.greedy_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            agent_partner_belief_conditioned_selected_regret=jnp.where(
                filter_trace_accepted,
                filter_decision.action_regrets[safe_index],
                zero_float,
            ),
            agent_partner_belief_conditioned_action_margin=jnp.where(
                filter_trace_accepted,
                filter_decision.action_margin,
                zero_float,
            ),
            agent_partner_belief_conditioned_tied=jnp.where(
                filter_trace_accepted,
                filter_decision.tied,
                jnp.asarray(True, dtype=jnp.bool_),
            ),
            oracle_step_count=jnp.where(
                oracle_trace_accepted,
                oracle.step_count,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_cycle_index=jnp.where(
                oracle_trace_accepted,
                oracle.cycle_index,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_cycle_step=jnp.where(
                oracle_trace_accepted,
                oracle.cycle_step,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_cycle_length=jnp.where(
                oracle_trace_accepted,
                oracle.cycle_length,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_segment_index=jnp.where(
                oracle_trace_accepted,
                oracle.segment_index,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_segment_step=jnp.where(
                oracle_trace_accepted,
                oracle.segment_step,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_segment_length=jnp.where(
                oracle_trace_accepted,
                oracle.segment_length,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_world_sign=jnp.where(
                oracle_trace_accepted, oracle.world_sign, zero_float
            ),
            oracle_next_world_sign=jnp.where(
                oracle_trace_accepted,
                oracle.next_world_sign,
                zero_float,
            ),
            oracle_world_flipped=oracle_trace_accepted & oracle.world_flipped,
            oracle_outcome_flipped=oracle_trace_accepted & oracle.outcome_flipped,
            oracle_regime_id=jnp.where(
                oracle_trace_accepted,
                oracle.regime_id,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_next_cycle_index=jnp.where(
                oracle_trace_accepted,
                oracle.next_cycle_index,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_next_segment_index=jnp.where(
                oracle_trace_accepted,
                oracle.next_segment_index,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_next_regime_id=jnp.where(
                oracle_trace_accepted,
                oracle.next_regime_id,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_schedule_switched=(
                oracle_trace_accepted & oracle.schedule_switched
            ),
            oracle_partner_intended_action=jnp.where(
                oracle_trace_accepted,
                oracle.partner_intended_action,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_partner_intended_sign=jnp.where(
                oracle_trace_accepted,
                oracle.partner_intended_sign,
                zero_float,
            ),
            oracle_partner_flipped=oracle_trace_accepted & oracle.partner_flipped,
            oracle_focal_action_sign=jnp.where(
                oracle_trace_accepted,
                oracle.focal_action_sign,
                zero_float,
            ),
            oracle_partner_action_sign=jnp.where(
                oracle_trace_accepted,
                oracle.partner_action_sign,
                zero_float,
            ),
            oracle_world_cue_flipped=jnp.where(
                oracle_trace_accepted,
                oracle.world_cue_flipped,
                jnp.zeros((2,), dtype=jnp.bool_),
            ),
            oracle_next_world_cue_flipped=jnp.where(
                oracle_trace_accepted,
                oracle.next_world_cue_flipped,
                jnp.zeros((2,), dtype=jnp.bool_),
            ),
            oracle_full_information_action=jnp.where(
                oracle_trace_accepted,
                oracle.full_information_optimal_focal_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            oracle_full_information_action_margin=jnp.where(
                oracle_trace_accepted,
                oracle.full_information_action_margin,
                zero_float,
            ),
            oracle_full_information_action_tied=jnp.where(
                oracle_trace_accepted,
                oracle.full_information_action_tied,
                jnp.asarray(True, dtype=jnp.bool_),
            ),
            oracle_realized_counterfactual_rewards=jnp.where(
                oracle_trace_accepted, oracle.counterfactual_rewards, neutral_actions
            ),
            proposed_agent_update_valid=agent_update_valid,
            proposed_filter_update_valid=filter_result.valid,
            proposed_filter_decision_valid=filter_decision.valid,
            learner_trace_valid=(
                accepted
                & learner_path_values_finite
                & agent_result.diagnostics.all_finite
                & next_selection_diagnostics_valid
            ),
            filter_trace_valid=filter_trace_valid,
            oracle_trace_valid=oracle_trace_valid,
            proposed_world_step_delta=proposed_world_step_delta,
            proposed_agent_step_delta=proposed_agent_step_delta,
            proposed_filter_step_delta=proposed_filter_step_delta,
            proposed_bridge_step_delta=proposed_bridge_step_delta,
            committed_world_step_delta=jnp.where(
                accepted, proposed_world_step_delta, zero_delta
            ),
            committed_agent_step_delta=jnp.where(
                accepted, proposed_agent_step_delta, zero_delta
            ),
            committed_filter_step_delta=jnp.where(
                accepted, proposed_filter_step_delta, zero_delta
            ),
            committed_bridge_step_delta=jnp.where(
                accepted, proposed_bridge_step_delta, zero_delta
            ),
            mechanism=self._project_mechanism_trace(
                state.agent,
                agent_result.diagnostics,
                accepted,
            ),
            all_finite=(
                accepted
                & learner_path_values_finite
                & filter_trace_valid
                & oracle_trace_valid
                & agent_result.diagnostics.all_finite
                & next_selection_diagnostics_valid
            ),
        )
        return HiddenPartnerWorldOnlineStep(state=next_state, trace=trace)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: HiddenPartnerWorldOnlineState,
    ) -> HiddenPartnerWorldOnlineStep:
        """Advance once, or return an exact no-op after a latched rejection."""
        self._require_static_state_contract(state)
        return cast(
            HiddenPartnerWorldOnlineStep,
            jax.lax.cond(
                state.valid,
                lambda current: jax.lax.cond(
                    self._entry_contract(current)[-1],
                    lambda valid: self._advance(valid),
                    lambda invalid: self._entry_rejected_step(invalid),
                    current,
                ),
                lambda current: self._blocked_step(current),
                state,
            ),
        )


__all__ = [
    "FocalActionPolicy",
    "HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA",
    "HiddenPartnerWorldOnlineBridge",
    "HiddenPartnerWorldMechanismTrace",
    "HiddenPartnerWorldOnlineResourceBudget",
    "HiddenPartnerWorldOnlineState",
    "HiddenPartnerWorldOnlineStep",
    "HiddenPartnerWorldOnlineTrace",
    "LearnerHiddenPartnerWorldTransition",
    "strip_hidden_partner_world_oracle",
]
