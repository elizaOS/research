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
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
    IntegratedHiddenPartnerState,
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
    "alberta.hidden-partner-world-online-bridge.config.v2"
)
_CONFIG_TOKEN_BYTES = 32


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
    filter_entry_valid: Bool[Array, ""]
    observation_pre: Float[Array, " 8"]
    phi_pre: Float[Array, " 12"]
    chi_pre: Float[Array, " 24"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    corrected_outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    next_action: Int[Array, ""]
    filter_mean_pre: Float[Array, ""]
    filter_mean_post: Float[Array, ""]
    agent_partner_belief_conditioned_reward_cells: Float[Array, "2 2"]
    agent_applied_partner_probabilities: Float[Array, " 2"]
    agent_partner_belief_conditioned_expected_rewards: Float[Array, " 2"]
    agent_partner_belief_conditioned_greedy_action: Int[Array, ""]
    agent_partner_belief_conditioned_selected_regret: Float[Array, ""]
    agent_partner_belief_conditioned_action_margin: Float[Array, ""]
    agent_partner_belief_conditioned_tied: Bool[Array, ""]
    oracle_world_sign: Float[Array, ""]
    oracle_world_flipped: Bool[Array, ""]
    oracle_outcome_flipped: Bool[Array, ""]
    oracle_regime_id: Int[Array, ""]
    oracle_full_information_action: Int[Array, ""]
    oracle_realized_counterfactual_rewards: Float[Array, " 2"]
    proposed_agent_update_valid: Bool[Array, ""]
    proposed_filter_update_valid: Bool[Array, ""]
    proposed_filter_decision_valid: Bool[Array, ""]
    oracle_trace_valid: Bool[Array, ""]
    proposed_world_step_delta: Int[Array, ""]
    proposed_agent_step_delta: Int[Array, ""]
    proposed_filter_step_delta: Int[Array, ""]
    proposed_bridge_step_delta: Int[Array, ""]
    committed_world_step_delta: Int[Array, ""]
    committed_agent_step_delta: Int[Array, ""]
    committed_filter_step_delta: Int[Array, ""]
    committed_bridge_step_delta: Int[Array, ""]
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
    ) -> None:
        self._world = HiddenPartnerWorldFeedbackWorld() if world is None else world
        self._agent = IntegratedHiddenPartnerAgent() if agent is None else agent
        if not isinstance(self._world, HiddenPartnerWorldFeedbackWorld):
            raise TypeError("world must be a HiddenPartnerWorldFeedbackWorld")
        if not isinstance(self._agent, IntegratedHiddenPartnerAgent):
            raise TypeError("agent must be an IntegratedHiddenPartnerAgent")
        if world_filter is not None and not isinstance(
            world_filter, HiddenPartnerWorldBayesFilter
        ):
            raise TypeError("world_filter must be a HiddenPartnerWorldBayesFilter")
        expected_filter_config = HiddenPartnerWorldFilterConfig.from_world_config(
            self._world.config
        )
        self._filter = (
            HiddenPartnerWorldBayesFilter(expected_filter_config)
            if world_filter is None
            else world_filter
        )
        if self._filter.config != expected_filter_config:
            raise ValueError("world and Bayes-filter probability contracts must match exactly")
        self._config_token_hex = hashlib.sha256(
            _canonical_json_bytes(self.to_config())
        ).hexdigest()
        self._config_token = jnp.asarray(
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
            "world": self._world.to_config(),
            "agent": self._agent.to_config(),
            "world_filter": self._filter.config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> HiddenPartnerWorldOnlineBridge:
        """Reconstruct only an exact authority-free v2 composition."""

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
            "world",
            "agent",
            "world_filter",
        }
        if set(payload) != expected_fields:
            raise ValueError("bridge config fields do not match the v2 schema")
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
        bridge = cls(world=world, agent=agent, world_filter=world_filter)
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
        world_state = self._world.init(world_key)
        observation = self._world.observe(world_state)
        agent_start = self._agent.start(observation, agent_key)
        cues = observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
        filter_state = self._filter.initialize(cues)
        action_valid = (agent_start.action >= 0) & (agent_start.action < 2)
        valid = (
            agent_start.diagnostics.all_finite
            & agent_start.diagnostics.descriptors_valid
            & action_valid
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
    ) -> tuple[Array, Array, Array, Array, Array]:
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
        action_valid = (
            (state.action >= 0)
            & (state.action < 2)
            & (state.action == state.agent.control.last_action)
        )
        filter_entry_valid = (
            (state.world_filter.step_count >= 0)
            & (state.world_filter.step_count < maximum)
            & (state.world_filter.step_count == state.step_count)
            & state.world_filter.valid
            & jnp.isfinite(state.world_filter.posterior_mean)
            & (jnp.abs(state.world_filter.posterior_mean) <= 1.0)
        )
        entry_valid = config_token_valid & counters_synchronized & action_valid
        return (
            config_token_valid,
            counters_synchronized,
            action_valid,
            filter_entry_valid,
            entry_valid,
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
            filter_entry_valid=filter_entry_valid,
            observation_pre=safe_observation,
            phi_pre=safe_phi,
            chi_pre=safe_chi,
            focal_action=safe_action,
            partner_action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            outcome=jnp.asarray(0.0, dtype=jnp.float32),
            corrected_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            next_observation=safe_observation,
            next_action=safe_action,
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
            oracle_world_sign=jnp.asarray(0.0, dtype=jnp.float32),
            oracle_world_flipped=jnp.asarray(False, dtype=jnp.bool_),
            oracle_outcome_flipped=jnp.asarray(False, dtype=jnp.bool_),
            oracle_regime_id=jnp.asarray(-1, dtype=jnp.int32),
            oracle_full_information_action=jnp.asarray(0, dtype=jnp.int32),
            oracle_realized_counterfactual_rewards=neutral_actions,
            proposed_agent_update_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_filter_update_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_filter_decision_valid=jnp.asarray(False, dtype=jnp.bool_),
            oracle_trace_valid=jnp.asarray(False, dtype=jnp.bool_),
            proposed_world_step_delta=zero_delta,
            proposed_agent_step_delta=zero_delta,
            proposed_filter_step_delta=zero_delta,
            proposed_bridge_step_delta=zero_delta,
            committed_world_step_delta=zero_delta,
            committed_agent_step_delta=zero_delta,
            committed_filter_step_delta=zero_delta,
            committed_bridge_step_delta=zero_delta,
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
        next_action_valid = (
            (agent_result.action >= 0)
            & (agent_result.action < 2)
            & (agent_result.action == agent_result.state.control.last_action)
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
        oracle_trace_valid = (
            jnp.isfinite(oracle.world_sign)
            & ((oracle.world_sign == -1.0) | (oracle.world_sign == 1.0))
            & jnp.all(jnp.isfinite(oracle.counterfactual_rewards))
            & jnp.all(
                (oracle.counterfactual_rewards >= 0.0)
                & (oracle.counterfactual_rewards <= 1.0)
            )
            & (oracle.regime_id >= 0)
            & (oracle.regime_id < 4)
            & (oracle.full_information_optimal_focal_action >= 0)
            & (oracle.full_information_optimal_focal_action < 2)
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
            filter_entry_valid=filter_entry_valid,
            observation_pre=jnp.where(
                accepted,
                learner_transition.observation,
                state.agent.raw_observation,
            ),
            phi_pre=jnp.where(accepted, state.agent.phi, jnp.zeros_like(state.agent.phi)),
            chi_pre=jnp.where(accepted, state.agent.chi, jnp.zeros_like(state.agent.chi)),
            focal_action=jnp.where(accepted, learner_transition.focal_action, safe_action),
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
            next_action=jnp.where(accepted, agent_result.action, safe_action),
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
            oracle_world_sign=jnp.where(
                oracle_trace_accepted, oracle.world_sign, zero_float
            ),
            oracle_world_flipped=oracle_trace_accepted & oracle.world_flipped,
            oracle_outcome_flipped=oracle_trace_accepted & oracle.outcome_flipped,
            oracle_regime_id=jnp.where(
                oracle_trace_accepted,
                oracle.regime_id,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            oracle_full_information_action=jnp.where(
                oracle_trace_accepted,
                oracle.full_information_optimal_focal_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            oracle_realized_counterfactual_rewards=jnp.where(
                oracle_trace_accepted, oracle.counterfactual_rewards, neutral_actions
            ),
            proposed_agent_update_valid=agent_update_valid,
            proposed_filter_update_valid=filter_result.valid,
            proposed_filter_decision_valid=filter_decision.valid,
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
            all_finite=(
                accepted
                & learner_path_values_finite
                & filter_trace_valid
                & oracle_trace_valid
                & agent_result.diagnostics.all_finite
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
    "HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA",
    "HiddenPartnerWorldOnlineBridge",
    "HiddenPartnerWorldOnlineResourceBudget",
    "HiddenPartnerWorldOnlineState",
    "HiddenPartnerWorldOnlineStep",
    "HiddenPartnerWorldOnlineTrace",
    "LearnerHiddenPartnerWorldTransition",
    "strip_hidden_partner_world_oracle",
]
