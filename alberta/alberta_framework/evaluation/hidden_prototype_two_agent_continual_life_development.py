# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-any-return"
"""Hidden-context A-B-A life with two complete Prototype learners.

This is a deliberately small, development-only composition of two existing
lanes.  It retains the bounded pair-feature/OaK/Horde/world-model/experiential-
memory Prototype configuration from
``prototype_two_learning_agent_recurrence_development`` and inserts the exact
``ContextInferenceConfig`` consumed by ``hidden_context_coadaptation_development``.

The environment's visible meet/avoid coordinates are destroyed before any
learner use.  After both actions have been fixed and the common reward has
arrived, each context bank observes only the completed partner action, its own
action, and reward.  Its post-transition one-hot may then occupy the two
formerly-visible coordinates for the *next* Prototype decision.  The matched
unrouted arm performs the same context updates but supplies zeros there.

Every event stages four environment proposals, two context updates, two
discarded no-memory Prototype previews, and two memory-sidecar Prototype
candidates from one immutable source.  One outer ``lax.cond`` carries all five
learning/environment children or returns the complete source state.  There is
no output writer, threshold, winner selection, evidence path, checkpoint claim,
or context-capacity/Alberta-Plan-completion claim.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.context_inference import (
    ContextInference,
    ContextInferenceState,
    measure_context_inference_state_nbytes,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureRepresentationState,
    PrototypeFeatureWorldModelState,
    PrototypeMemoryInteractionState,
    measure_prototype_agent_state_resources,
)
from alberta_framework.core.prototype_feature_memory import PrototypeFeatureMemoryState
from alberta_framework.core.world_model import ActionConditionedWorldModel
from alberta_framework.evaluation.hidden_context_coadaptation_development import (
    _CONTEXT_CONFIG,
    DEVELOPMENT_SEEDS,
)
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    _horde_spec,
    _memory_input,
    _phase_for_step,
    _transition,
)
from alberta_framework.evaluation.prototype_two_learning_agent_recurrence_development import (
    PrototypeTwoLearningAgentRecurrenceProtocol,
    _agent_config,
)
from alberta_framework.streams.recurring_multiagent import (
    AVOID_CONTEXT_INDEX,
    MEET_CONTEXT_INDEX,
    RecurringTwoAgentState,
    RecurringTwoAgentWorld,
)

HIDDEN_PROTOTYPE_TWO_AGENT_PROTOCOL_SCHEMA: Final = (
    "alberta.hidden-prototype-two-agent-continual-life-development.protocol.v1"
)
HIDDEN_PROTOTYPE_TWO_AGENT_REPORT_SCHEMA: Final = (
    "alberta.hidden-prototype-two-agent-continual-life-development.report.v1"
)
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ACCEPTANCE_STATUS: Final = "not-assessed"

HIDDEN_INFERRED_FULL: Literal["hidden_inferred_full"] = "hidden_inferred_full"
HIDDEN_INFERENCE_UNROUTED: Literal["hidden_inference_unrouted"] = (
    "hidden_inference_unrouted"
)
HiddenPrototypeArmName = Literal[
    "hidden_inferred_full",
    "hidden_inference_unrouted",
]

# This is an alias, not a reconstructed or re-calibrated configuration.
INHERITED_CONTEXT_INFERENCE_CONFIG: Final = _CONTEXT_CONFIG
CONSUMED_DEVELOPMENT_ROOT: Final = DEVELOPMENT_SEEDS[0]

_N_AGENTS: Final = 2
_N_ACTIONS: Final = 2
_N_HORDE_DEMONS: Final = 2
_N_FEATURE_TASKS: Final = 1 + _N_HORDE_DEMONS
_LIFECYCLE_TAG: Final = 0x48325041  # "H2PA"
_JOINT_PROPOSAL_NAMES: Final = (
    "actual_actual",
    "base0_actual1",
    "actual0_base1",
    "base_base",
)
NESTED_CLOCK_NAMES: Final = (
    "prototype",
    "oak",
    "stomp",
    "horde",
    "world_model",
    "feature_observe",
    "memory",
)
OLD_BANK_ROUTING_FLAG_NAMES: Final = (
    "consumer_binding_valid",
    "post_update_consumer_clock_valid",
    "input_route_valid_or_not_attempted",
    "output_route_valid_or_not_attempted",
    "route_states_match_or_not_attempted",
    "routed_values_finite_or_not_attempted",
    "postcondition_valid",
    "feature_memory_destination_valid",
    "feature_memory_outer_commit",
)


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPrototypeTwoAgentArm:
    """One state/work-matched routing intervention."""

    name: HiddenPrototypeArmName
    route_inference: bool
    role: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


HIDDEN_PROTOTYPE_TWO_AGENT_ARMS: Final = (
    HiddenPrototypeTwoAgentArm(
        HIDDEN_INFERRED_FULL,
        route_inference=True,
        role="route past-only inferred context into the next Prototype decision",
    ),
    HiddenPrototypeTwoAgentArm(
        HIDDEN_INFERENCE_UNROUTED,
        route_inference=False,
        role="same context state and work; inferred context is not routed to control",
    ),
)
_ARMS_BY_NAME: Final = {arm.name: arm for arm in HIDDEN_PROTOTYPE_TWO_AGENT_ARMS}


def _default_prototype_protocol() -> PrototypeTwoLearningAgentRecurrenceProtocol:
    """Return the existing full-composition geometry with only its full arm."""

    return PrototypeTwoLearningAgentRecurrenceProtocol(arm_names=("joint_full",))


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPrototypeTwoAgentProtocol:
    """Frozen U0 composition around the existing Prototype A-B-A protocol."""

    prototype_protocol: PrototypeTwoLearningAgentRecurrenceProtocol = dataclasses.field(
        default_factory=_default_prototype_protocol
    )
    arm_names: tuple[HiddenPrototypeArmName, ...] = (
        HIDDEN_INFERRED_FULL,
        HIDDEN_INFERENCE_UNROUTED,
    )
    schema_version: str = HIDDEN_PROTOTYPE_TWO_AGENT_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != HIDDEN_PROTOTYPE_TWO_AGENT_PROTOCOL_SCHEMA:
            raise ValueError("hidden Prototype protocol schema is unsupported")
        if type(self.prototype_protocol) is not PrototypeTwoLearningAgentRecurrenceProtocol:
            raise TypeError("prototype_protocol must be the exact existing protocol type")
        if self.prototype_protocol.arm_names != ("joint_full",):
            raise ValueError("prototype_protocol must retain only canonical joint_full")
        expected = tuple(
            arm.name for arm in HIDDEN_PROTOTYPE_TWO_AGENT_ARMS if arm.name in self.arm_names
        )
        if (
            type(self.arm_names) is not tuple
            or not self.arm_names
            or tuple(self.arm_names) != expected
            or len(set(self.arm_names)) != len(self.arm_names)
        ):
            raise ValueError("arm_names must be a nonempty canonical-order subset")

    @property
    def total_steps(self) -> int:
        return self.prototype_protocol.total_steps

    def to_config(self) -> dict[str, object]:
        root = CONSUMED_DEVELOPMENT_ROOT
        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "output_writes_allowed": False,
            "accepted_scientific_evidence": False,
            "thresholds_or_winner_selection": False,
            "context_capacity_pressure_claimed": False,
            "schedule": ["A1", "B", "A2"],
            "resets_after_initialization": 0,
            "boundary_callbacks_used": False,
            "consumed_development_root": {
                "namespace": root.namespace,
                "index": root.index,
                "environment_seed": root.environment_seed,
                "initialization_seed": root.initialization_seed,
            },
            "context_inference": INHERITED_CONTEXT_INFERENCE_CONFIG.to_config(),
            "prototype_protocol": self.prototype_protocol.to_config(),
            "arm_names": list(self.arm_names),
            "causal_order": [
                "four_same-prestate_environment_proposals",
                "two_completed-experience_context_updates",
                "two_no-memory_Prototype_previews",
                "two_memory-sidecar_Prototype_candidates",
                "one_outer_all-or-none_commit",
            ],
            "learner_pre_action_channels": [
                "physical_state",
                "nuisance",
                "past-only-inferred-context-when-routed",
            ],
            "learner_post_action_context_channels": [
                "own_action",
                "observed_partner_action",
                "common_reward",
            ],
            "forbidden_learner_channels": [
                "current_rule",
                "visible_meet_avoid_cue",
                "phase_index",
                "phase_boundary",
                "steps_to_boundary",
                "environment_step_count",
            ],
            "checkpoint_contract": {
                "inherited": False,
                "checkpoint_resume_claimed": False,
                "reason": "the existing shadows do not include both context banks atomically",
            },
        }


@chex.dataclass(frozen=True)
class HiddenPrototypeTwoAgentState:
    """Complete persistent state for one uninterrupted two-learner life."""

    environment: RecurringTwoAgentState
    agent_0: PrototypeAgentState
    agent_1: PrototypeAgentState
    context_0: ContextInferenceState
    context_1: ContextInferenceState
    counterfactual_base_actions: Array
    prior_memory_retrieval_available: Array
    prior_memory_action_changed: Array


@chex.dataclass(frozen=True)
class HiddenPrototypeTwoAgentTrace:
    """Fixed-shape causal audit for one proposed outer event."""

    environment_pre_words: Array
    environment_post_words: Array
    environment_proposal_pre_words: Array
    environment_proposal_post_words: Array
    environment_proposals_applied: Array
    joint_primitive_actions: Array
    joint_rewards: Array
    mean_agent_rewards: Array
    own_action_effects: Array
    partner_action_effects: Array
    interaction_effects: Array
    joint_mean_effect: Array
    context_pre_words: Array
    context_post_words: Array
    context_pre_slots: Array
    context_post_slots: Array
    context_post_onehot: Array
    context_partner_action_observations: Array
    context_candidate_updates_applied: Array
    world_visible_cues_for_evaluator_only: Array
    learner_context_channels_pre: Array
    learner_context_channels_next: Array
    no_oracle_cue_consumed: Array
    nested_pre_words: Array
    nested_post_words: Array
    actions: Array
    counterfactual_base_actions: Array
    next_preview_actions: Array
    next_committed_actions: Array
    prior_memory_retrieval_available: Array
    prior_memory_action_changed: Array
    current_action_differs_from_base: Array
    current_memory_counterfactual_effects: Array
    horde_predictions: Array
    horde_cumulants: Array
    horde_td_errors: Array
    horde_squared_errors: Array
    world_model_next_predictions: Array
    world_model_reward_predictions: Array
    world_model_discount_predictions: Array
    world_model_errors: Array
    world_model_expected_errors: Array
    world_model_contract_valid: Array
    horde_contract_valid: Array
    feature_task_targets: Array
    feature_task_predictions: Array
    feature_generation_pre_words: Array
    feature_generation_post_words: Array
    feature_lifecycle_flags: Array
    old_bank_routing_flags: Array
    old_bank_routing_valid: Array
    option_executing_pre: Array
    option_executing_post: Array
    memory_query_before_write: Array
    memory_prestate_query_count: Array
    memory_wrote: Array
    memory_slots: Array
    memory_evicted: Array
    memory_evicted_provenance_ids: Array
    memory_retrieval_available: Array
    memory_retrieval_neighbor_mask: Array
    memory_retrieval_neighbor_provenance_ids: Array
    memory_action_changed: Array
    memory_rows_reencoded: Array
    feature_memory_rebind_applied: Array
    memory_contract_valid: Array
    source_state_valid: Array
    source_clocks_aligned: Array
    candidate_clocks_aligned: Array
    preview_updates_valid: Array
    candidate_updates_valid: Array
    forced_outer_rejection: Array
    outer_transaction_committed: Array


@chex.dataclass(frozen=True)
class HiddenPrototypeTwoAgentStepResult:
    """Atomically selected state and the complete proposal trace."""

    state: HiddenPrototypeTwoAgentState
    trace: HiddenPrototypeTwoAgentTrace


def _hidden_learner_observation(
    world_observation: Array,
    inferred_context: Array,
    route_inference: Array,
) -> Array:
    """Destroy oracle cues, then optionally route a past-only inferred one-hot."""

    observation = jnp.asarray(world_observation, dtype=jnp.float32)
    if observation.ndim != 1 or observation.shape[0] < AVOID_CONTEXT_INDEX + 1:
        raise ValueError("world_observation has no two-coordinate context slot")
    context = jnp.asarray(inferred_context, dtype=jnp.float32)
    if context.shape != (_N_ACTIONS,):
        raise ValueError("inferred_context must have shape (2,)")
    route = jnp.asarray(route_inference, dtype=jnp.bool_)
    if route.shape != ():
        raise ValueError("route_inference must be scalar")
    destroyed = observation.at[
        jnp.asarray((MEET_CONTEXT_INDEX, AVOID_CONTEXT_INDEX), dtype=jnp.int32)
    ].set(0.0)
    routed = jnp.where(route, context, jnp.zeros_like(context))
    return destroyed.at[MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1].set(routed)


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        raw = leaf
        dtype = getattr(raw, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            raw = jr.key_data(raw)
        array = np.asarray(jax.device_get(raw))
        total += int(array.size) * int(array.dtype.itemsize)
    return total


def _tree_equivalent(left: object, right: object) -> bool:
    """Exact discrete and close float equality for eager/outer-JIT parity."""

    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):  # type: ignore[operator]
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_dtype = getattr(left_leaf, "dtype", None)
        right_dtype = getattr(right_leaf, "dtype", None)
        if left_dtype is not None and jax.dtypes.issubdtype(
            left_dtype, jax.dtypes.prng_key
        ):
            left_leaf = jr.key_data(left_leaf)
        if right_dtype is not None and jax.dtypes.issubdtype(
            right_dtype, jax.dtypes.prng_key
        ):
            right_leaf = jr.key_data(right_leaf)
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if left_array.dtype.kind in "fc":
            if not np.allclose(left_array, right_array, rtol=1.0e-6, atol=1.0e-7):
                return False
        elif not np.array_equal(left_array, right_array):
            return False
    return True


def _feature_bundle(state: PrototypeAgentState) -> PrototypeFeatureOaKHordeState:
    if type(state.oak_state) is not PrototypeFeatureOaKHordeState:
        raise TypeError("hidden Prototype lane requires the exact OaK/Horde bundle")
    return state.oak_state


def _world_model_state(state: PrototypeAgentState) -> Any:
    wrapper = state.world_model_state
    if type(wrapper) is not PrototypeFeatureWorldModelState:
        raise TypeError("hidden Prototype lane requires the feature/world wrapper")
    return wrapper.model_state


def _feature_lifecycle_state(state: PrototypeAgentState) -> Any:
    wrapper = state.state_builder_state
    if type(wrapper) is not PrototypeFeatureRepresentationState:
        raise TypeError("hidden Prototype lane requires the feature representation wrapper")
    return wrapper.feature_lifecycle_state


def _memory_state(state: PrototypeAgentState) -> Any:
    interaction = state.ia_state
    if type(interaction) is not PrototypeMemoryInteractionState:
        raise TypeError("hidden Prototype lane requires the memory interaction wrapper")
    feature_memory = interaction.experiential_memory_state
    if type(feature_memory) is not PrototypeFeatureMemoryState:
        raise TypeError("hidden Prototype lane requires feature-bound experiential memory")
    return feature_memory.memory_state


def _nested_words(state: PrototypeAgentState) -> Array:
    bundle = _feature_bundle(state)
    return jnp.stack(
        (
            state.step_words,
            bundle.oak_state.step_words,
            bundle.oak_state.stomp_state.step_words,
            bundle.horde_state.step_words,
            _world_model_state(state).step_words,
            _feature_lifecycle_state(state).observe_words,
            _memory_state(state).step_words,
        )
    ).astype(jnp.uint32)


def _continuous_joint_actions(primitive_actions: Array) -> Array:
    actions = jnp.asarray(primitive_actions, dtype=jnp.int32)
    if actions.shape != (_N_AGENTS,):
        raise ValueError("joint primitive action must have shape (2,)")
    return jnp.where(actions == 0, -1.0, 1.0).astype(jnp.float32)


def _joint_reward_effects(rewards: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Return mean rewards and correctly owned action effects.

    Rows are ``actual_actual``, ``base0_actual1``, ``actual0_base1``, and
    ``base_base``; columns are receiving agents 0 and 1.  Own-action effects
    replace the receiving agent's action only, while partner effects replace
    the other agent's action only.
    """

    values = jnp.asarray(rewards, dtype=jnp.float32)
    if values.shape != (len(_JOINT_PROPOSAL_NAMES), _N_AGENTS):
        raise ValueError("proposal rewards must have shape (4, 2)")
    means = jnp.mean(values, axis=1)
    own = jnp.stack(
        (
            values[0, 0] - values[1, 0],
            values[0, 1] - values[2, 1],
        )
    )
    partner = jnp.stack(
        (
            values[0, 0] - values[2, 0],
            values[0, 1] - values[1, 1],
        )
    )
    interaction = values[0] - values[1] - values[2] + values[3]
    joint_mean = means[0] - means[3]
    return means, own, partner, interaction, joint_mean


class HiddenPrototypeTwoAgentEvaluator:
    """Static components and one pure outer U0 transaction."""

    def __init__(self, protocol: HiddenPrototypeTwoAgentProtocol | None = None):
        resolved = HiddenPrototypeTwoAgentProtocol() if protocol is None else protocol
        if type(resolved) is not HiddenPrototypeTwoAgentProtocol:
            raise TypeError("protocol must be an exact HiddenPrototypeTwoAgentProtocol")
        self.protocol = resolved
        base = resolved.prototype_protocol
        self.world = RecurringTwoAgentWorld(
            context_length=base.segment_length,
            nuisance_dim=base.nuisance_dim,
            nuisance_scale=base.nuisance_scale,
        )
        self.context = ContextInference(INHERITED_CONTEXT_INFERENCE_CONFIG)
        self.agent = PrototypeAgent(_agent_config(base, feature_promotion_enabled=True))
        self.horde = HordeLearner(_horde_spec(), hidden_sizes=(), step_size=0.05)
        config = self.agent.config.world_model
        if config is None:
            raise RuntimeError("canonical Prototype composition omitted its world model")
        self.world_model = ActionConditionedWorldModel(config)

    def initialize(self, arm_name: HiddenPrototypeArmName) -> HiddenPrototypeTwoAgentState:
        """Initialize only from the inherited consumed development root."""

        if arm_name not in _ARMS_BY_NAME:
            raise ValueError("unknown hidden Prototype arm")
        route = jnp.asarray(_ARMS_BY_NAME[arm_name].route_inference, dtype=jnp.bool_)
        root = CONSUMED_DEVELOPMENT_ROOT
        environment = self.world.init(jr.key(root.environment_seed))
        contexts = (self.context.init(), self.context.init())
        visible = self.world.observe(environment)
        initialization_key = jr.key(root.initialization_seed)
        agent_states: list[PrototypeAgentState] = []
        for agent_index in range(_N_AGENTS):
            observation = _hidden_learner_observation(
                visible[agent_index],
                self.context.context_onehot(contexts[agent_index]),
                route,
            )
            lifecycle_id = jnp.asarray(
                (_LIFECYCLE_TAG, agent_index + 1), dtype=jnp.uint32
            )
            initialized = self.agent.init(
                jr.fold_in(initialization_key, agent_index),
                lifecycle_id=lifecycle_id,
            )
            agent_states.append(self.agent.start(initialized, observation))
        actions = jnp.stack(
            (agent_states[0].current_action, agent_states[1].current_action)
        ).astype(jnp.int32)
        state = HiddenPrototypeTwoAgentState(
            environment=environment,
            agent_0=agent_states[0],
            agent_1=agent_states[1],
            context_0=contexts[0],
            context_1=contexts[1],
            counterfactual_base_actions=actions,
            prior_memory_retrieval_available=jnp.zeros((_N_AGENTS,), dtype=jnp.bool_),
            prior_memory_action_changed=jnp.zeros((_N_AGENTS,), dtype=jnp.bool_),
        )
        if not bool(self._state_valid(state)):
            raise RuntimeError("initialized hidden Prototype state is invalid")
        return state

    def _state_valid(self, state: HiddenPrototypeTwoAgentState) -> Array:
        agents = (state.agent_0, state.agent_1)
        contexts = (state.context_0, state.context_1)
        current_actions = jnp.stack(
            (agents[0].current_action, agents[1].current_action)
        ).astype(jnp.int32)
        action_changed = current_actions != state.counterfactual_base_actions
        auxiliary_valid = (
            (state.counterfactual_base_actions.shape == (_N_AGENTS,))
            and (state.counterfactual_base_actions.dtype == jnp.dtype(jnp.int32))
            and (state.prior_memory_retrieval_available.shape == (_N_AGENTS,))
            and (state.prior_memory_retrieval_available.dtype == jnp.dtype(jnp.bool_))
            and (state.prior_memory_action_changed.shape == (_N_AGENTS,))
            and (state.prior_memory_action_changed.dtype == jnp.dtype(jnp.bool_))
        )
        valid = (
            self.world.state_is_valid(state.environment)
            & self.context.state_is_valid(contexts[0])
            & self.context.state_is_valid(contexts[1])
            & self.agent.validate_state(agents[0])
            & self.agent.validate_state(agents[1])
            & jnp.asarray(auxiliary_valid, dtype=jnp.bool_)
            & jnp.all(state.counterfactual_base_actions >= 0)
            & jnp.all(state.counterfactual_base_actions < _N_ACTIONS)
            & jnp.all(state.prior_memory_action_changed == action_changed)
            & jnp.all(
                (~state.prior_memory_action_changed)
                | state.prior_memory_retrieval_available
            )
        )
        return valid & self._clocks_aligned(state)

    @staticmethod
    def _clocks_aligned(state: HiddenPrototypeTwoAgentState) -> Array:
        expected = state.environment.step_words
        nested = jnp.stack((_nested_words(state.agent_0), _nested_words(state.agent_1)))
        return (
            jnp.all(state.context_0.step_words == expected)
            & jnp.all(state.context_1.step_words == expected)
            & jnp.all(nested == expected[None, None, :])
        )

    def step(
        self,
        state: HiddenPrototypeTwoAgentState,
        event_index: Array,
        *,
        route_inference: Array,
        force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
    ) -> HiddenPrototypeTwoAgentStepResult:
        """Stage four/two/four proposals and carry one all-or-none successor."""

        index = jnp.asarray(event_index, dtype=jnp.int32)
        route = jnp.asarray(route_inference, dtype=jnp.bool_)
        force_reject = jnp.asarray(force_outer_rejection, dtype=jnp.bool_)
        if index.shape != () or route.shape != () or force_reject.shape != ():
            raise ValueError("event_index and control flags must be scalar")

        agents = (state.agent_0, state.agent_1)
        contexts = (state.context_0, state.context_1)
        source_state_valid = self._state_valid(state)
        source_clocks_aligned = self._clocks_aligned(state)
        current_actions = jnp.stack(
            (agents[0].current_action, agents[1].current_action)
        ).astype(jnp.int32)
        base_actions = state.counterfactual_base_actions
        action_valid = (
            jnp.all(current_actions >= 0)
            & jnp.all(current_actions < _N_ACTIONS)
            & jnp.all(base_actions >= 0)
            & jnp.all(base_actions < _N_ACTIONS)
        )
        primitive_proposals = jnp.stack(
            (
                current_actions,
                jnp.stack((base_actions[0], current_actions[1])),
                jnp.stack((current_actions[0], base_actions[1])),
                base_actions,
            )
        ).astype(jnp.int32)
        environment_proposals = tuple(
            self.world.step_result(
                state.environment,
                _continuous_joint_actions(primitive_proposals[proposal_index]),
            )
            for proposal_index in range(len(_JOINT_PROPOSAL_NAMES))
        )
        environment_applied = jnp.stack(
            tuple(result.update_applied for result in environment_proposals)
        ).astype(jnp.bool_)
        environment_pre = jnp.stack(
            tuple(result.pre_step_words for result in environment_proposals)
        ).astype(jnp.uint32)
        environment_post = jnp.stack(
            tuple(result.post_step_words for result in environment_proposals)
        ).astype(jnp.uint32)
        shared_environment_identity = (
            jnp.all(environment_pre == environment_pre[0])
            & jnp.all(environment_post == environment_post[0])
        )
        actual_environment = environment_proposals[0]
        rewards = jnp.stack(
            tuple(result.transition.reward for result in environment_proposals)
        ).astype(jnp.float32)
        (
            mean_agent_rewards,
            own_action_effects,
            partner_action_effects,
            interaction_effects,
            joint_mean_effect,
        ) = _joint_reward_effects(rewards)

        # Actions are irrevocable before either context bank receives the
        # completed partner action or reward.  No oracle/schedule call occurs.
        partner_observations = jnp.stack(
            (
                jax.nn.one_hot(current_actions[1], _N_ACTIONS, dtype=jnp.float32),
                jax.nn.one_hot(current_actions[0], _N_ACTIONS, dtype=jnp.float32),
            )
        )
        context_results = (
            self.context.update_result(
                contexts[0],
                partner_observations[0],
                current_actions[0],
                actual_environment.transition.reward[0],
            ),
            self.context.update_result(
                contexts[1],
                partner_observations[1],
                current_actions[1],
                actual_environment.transition.reward[1],
            ),
        )
        context_updates_applied = jnp.stack(
            tuple(result.update_applied for result in context_results)
        ).astype(jnp.bool_)
        raw_next_observations = actual_environment.transition.next_observation
        next_observations = (
            _hidden_learner_observation(
                raw_next_observations[0], context_results[0].context_onehot, route
            ),
            _hidden_learner_observation(
                raw_next_observations[1], context_results[1].context_onehot, route
            ),
        )
        transitions = (
            _transition(
                agents[0],
                reward=actual_environment.transition.reward[0],
                discount=actual_environment.transition.discount,
                terminated=actual_environment.transition.terminated,
                next_observation=next_observations[0],
            ),
            _transition(
                agents[1],
                reward=actual_environment.transition.reward[1],
                discount=actual_environment.transition.discount,
                terminated=actual_environment.transition.terminated,
                next_observation=next_observations[1],
            ),
        )

        horde_predictions = jnp.stack(
            tuple(
                self.horde.predict(
                    _feature_bundle(agents[agent_index]).horde_state,
                    agents[agent_index].current_representation,
                )
                for agent_index in range(_N_AGENTS)
            )
        )
        horde_cumulants = jnp.stack(
            tuple(cast(Array, transition.horde_cumulants) for transition in transitions)
        )
        horde_td_errors = horde_cumulants - horde_predictions
        world_predictions = tuple(
            self.world_model.predict(
                _world_model_state(agents[agent_index]),
                agents[agent_index].current_raw_observation,
                agents[agent_index].current_action,
            )
            for agent_index in range(_N_AGENTS)
        )

        # These first two Prototype updates are deliberately discarded.
        previews = tuple(
            self.agent.update_transition(agents[index_agent], transitions[index_agent])
            for index_agent in range(_N_AGENTS)
        )
        preview_actions = jnp.stack(tuple(result.action for result in previews)).astype(
            jnp.int32
        )
        memory_inputs = tuple(
            _memory_input(
                agents[agent_index],
                previews[agent_index].state,
                event_index=index,
                reward=actual_environment.transition.reward[agent_index],
                safe_action=None,
            )
            for agent_index in range(_N_AGENTS)
        )
        candidates = tuple(
            self.agent.update_transition(
                agents[agent_index],
                transitions[agent_index],
                experiential_memory_input=memory_inputs[agent_index],
            )
            for agent_index in range(_N_AGENTS)
        )
        preview_valid = jnp.stack(
            tuple(result.transition_diagnostics.valid for result in previews)
        ).astype(jnp.bool_)
        candidate_valid = jnp.stack(
            tuple(result.transition_diagnostics.valid for result in candidates)
        ).astype(jnp.bool_)

        memory_diagnostics = tuple(
            candidate.experiential_memory_diagnostics for candidate in candidates
        )
        feature_diagnostics = tuple(
            candidate.prototype_feature_lifecycle_diagnostics for candidate in candidates
        )
        feature_memory_diagnostics = tuple(
            candidate.prototype_feature_memory_diagnostics for candidate in candidates
        )
        if any(value is None for value in memory_diagnostics):
            raise RuntimeError("canonical Prototype candidate omitted memory diagnostics")
        if any(value is None for value in feature_diagnostics):
            raise RuntimeError("canonical Prototype candidate omitted feature diagnostics")
        if any(value is None for value in feature_memory_diagnostics):
            raise RuntimeError("canonical Prototype candidate omitted feature-memory diagnostics")
        if any(candidate.world_model_error is None for candidate in candidates):
            raise RuntimeError("canonical Prototype candidate omitted world-model error")
        memory = cast(tuple[Any, Any], memory_diagnostics)
        feature = cast(tuple[Any, Any], feature_diagnostics)
        feature_memory = cast(tuple[Any, Any], feature_memory_diagnostics)

        reported_horde_td_errors = jnp.stack(
            tuple(cast(Array, candidate.horde_td_errors) for candidate in candidates)
        )
        horde_contract_valid = jnp.all(
            jnp.isclose(
                reported_horde_td_errors,
                horde_td_errors,
                rtol=1.0e-6,
                atol=1.0e-7,
            ),
            axis=1,
        )
        predicted_next = jnp.stack(
            tuple(prediction.next_observation for prediction in world_predictions)
        )
        predicted_reward = jnp.stack(
            tuple(prediction.reward for prediction in world_predictions)
        )
        predicted_discount = jnp.stack(
            tuple(prediction.discount for prediction in world_predictions)
        )
        next_targets = jnp.stack(next_observations)
        reward_targets = actual_environment.transition.reward
        discount_targets = jnp.full(
            (_N_AGENTS,),
            actual_environment.transition.discount,
            dtype=jnp.float32,
        )
        world_expected_errors = (
            jnp.mean(jnp.square(predicted_next - next_targets), axis=1)
            + jnp.square(predicted_reward - reward_targets)
            + jnp.square(predicted_discount - discount_targets)
        )
        reported_world_errors = jnp.stack(
            tuple(cast(Array, candidate.world_model_error) for candidate in candidates)
        )
        world_contract_valid = jnp.isfinite(reported_world_errors) & jnp.isclose(
            reported_world_errors,
            world_expected_errors,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        memory_contract_valid = jnp.stack(
            tuple(
                item.query_before_write
                & item.wrote
                & item.transaction_applied
                & item.current_prototype_decision_id_matches
                & item.next_prototype_decision_id_matches
                & item.metadata_valid
                & item.retrieval_matches
                & (item.deterministic_prestate_query_count == 2)
                & (item.counterfactual_base_action == preview_actions[agent_index])
                & (
                    item.dispatch_replacement.applied
                    == (candidates[agent_index].action != preview_actions[agent_index])
                )
                for agent_index, item in enumerate(memory)
            )
        ).astype(jnp.bool_)

        old_bank_flags: list[Array] = []
        lifecycle_flags: list[Array] = []
        for agent_index in range(_N_AGENTS):
            lifecycle = feature[agent_index].lifecycle
            rebind = feature_memory[agent_index]
            input_valid = (~lifecycle.routing_attempted) | lifecycle.input_route_valid
            output_valid = (~lifecycle.routing_attempted) | lifecycle.output_route_valid
            states_match = (~lifecycle.routing_attempted) | lifecycle.route_states_match
            values_finite = (~lifecycle.routing_attempted) | lifecycle.routed_values_finite
            flags = jnp.stack(
                (
                    lifecycle.consumer_binding_valid,
                    lifecycle.post_update_consumer_clock_valid,
                    input_valid,
                    output_valid,
                    states_match,
                    values_finite,
                    lifecycle.postcondition_valid,
                    rebind.post_memory_state_valid,
                    rebind.outer_transaction_committed,
                )
            ).astype(jnp.bool_)
            old_bank_flags.append(flags)
            lifecycle_flags.append(
                jnp.stack(
                    (
                        lifecycle.curation_proposed,
                        lifecycle.safe_curation_boundary,
                        lifecycle.curation_deferred,
                        lifecycle.routing_attempted,
                        lifecycle.curation_committed,
                        lifecycle.curation_rolled_back,
                    )
                ).astype(jnp.bool_)
            )
        old_bank_routing_flags = jnp.stack(tuple(old_bank_flags))
        old_bank_routing_valid = jnp.all(old_bank_routing_flags, axis=1)

        next_base_actions = jnp.stack(
            tuple(item.counterfactual_base_action for item in memory)
        ).astype(jnp.int32)
        next_retrieval_available = jnp.stack(
            tuple(item.proposal.available for item in memory)
        ).astype(jnp.bool_)
        next_action_changed = jnp.stack(
            tuple(item.dispatch_replacement.applied for item in memory)
        ).astype(jnp.bool_)
        candidate_state = HiddenPrototypeTwoAgentState(
            environment=actual_environment.state,
            agent_0=candidates[0].state,
            agent_1=candidates[1].state,
            context_0=context_results[0].state,
            context_1=context_results[1].state,
            counterfactual_base_actions=next_base_actions,
            prior_memory_retrieval_available=next_retrieval_available,
            prior_memory_action_changed=next_action_changed,
        )
        candidate_clocks_aligned = self._clocks_aligned(candidate_state)
        candidate_state_valid = self._state_valid(candidate_state)
        outer_commit = (
            source_state_valid
            & source_clocks_aligned
            & action_valid
            & jnp.all(environment_applied)
            & shared_environment_identity
            & jnp.all(context_updates_applied)
            & jnp.all(preview_valid)
            & jnp.all(candidate_valid)
            & jnp.all(horde_contract_valid)
            & jnp.all(world_contract_valid)
            & jnp.all(memory_contract_valid)
            & jnp.all(old_bank_routing_valid)
            & candidate_clocks_aligned
            & candidate_state_valid
            & (~force_reject)
        )
        committed_state = jax.lax.cond(
            outer_commit,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )

        source_nested = jnp.stack((_nested_words(agents[0]), _nested_words(agents[1])))
        committed_nested = jnp.stack(
            (_nested_words(committed_state.agent_0), _nested_words(committed_state.agent_1))
        )
        source_context_words = jnp.stack(
            (contexts[0].step_words, contexts[1].step_words)
        )
        committed_context_words = jnp.stack(
            (committed_state.context_0.step_words, committed_state.context_1.step_words)
        )
        source_context_slots = jnp.stack(
            (contexts[0].active_context, contexts[1].active_context)
        ).astype(jnp.int32)
        committed_context_slots = jnp.stack(
            (
                committed_state.context_0.active_context,
                committed_state.context_1.active_context,
            )
        ).astype(jnp.int32)
        feature_pre = jnp.stack(
            tuple(
                _feature_bundle(agent).consumer_binding.semantic_generation_words
                for agent in agents
            )
        )
        feature_post = jnp.stack(
            (
                _feature_bundle(committed_state.agent_0)
                .consumer_binding.semantic_generation_words,
                _feature_bundle(committed_state.agent_1)
                .consumer_binding.semantic_generation_words,
            )
        )
        current_memory_changed = current_actions != base_actions
        trace = HiddenPrototypeTwoAgentTrace(
            environment_pre_words=state.environment.step_words,
            environment_post_words=committed_state.environment.step_words,
            environment_proposal_pre_words=environment_pre,
            environment_proposal_post_words=environment_post,
            environment_proposals_applied=environment_applied,
            joint_primitive_actions=primitive_proposals,
            joint_rewards=rewards,
            mean_agent_rewards=mean_agent_rewards,
            own_action_effects=own_action_effects,
            partner_action_effects=partner_action_effects,
            interaction_effects=interaction_effects,
            joint_mean_effect=joint_mean_effect,
            context_pre_words=source_context_words,
            context_post_words=committed_context_words,
            context_pre_slots=source_context_slots,
            context_post_slots=committed_context_slots,
            context_post_onehot=jnp.stack(
                (
                    self.context.context_onehot(committed_state.context_0),
                    self.context.context_onehot(committed_state.context_1),
                )
            ),
            context_partner_action_observations=partner_observations,
            context_candidate_updates_applied=context_updates_applied,
            world_visible_cues_for_evaluator_only=raw_next_observations[
                :, MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1
            ],
            learner_context_channels_pre=jnp.stack(
                tuple(
                    agent.current_raw_observation[
                        MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1
                    ]
                    for agent in agents
                )
            ),
            learner_context_channels_next=jnp.stack(
                (
                    committed_state.agent_0.current_raw_observation[
                        MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1
                    ],
                    committed_state.agent_1.current_raw_observation[
                        MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1
                    ],
                )
            ),
            no_oracle_cue_consumed=jnp.zeros((_N_AGENTS,), dtype=jnp.bool_),
            nested_pre_words=source_nested,
            nested_post_words=committed_nested,
            actions=current_actions,
            counterfactual_base_actions=base_actions,
            next_preview_actions=preview_actions,
            next_committed_actions=jnp.stack(
                (
                    committed_state.agent_0.current_action,
                    committed_state.agent_1.current_action,
                )
            ).astype(jnp.int32),
            prior_memory_retrieval_available=state.prior_memory_retrieval_available,
            prior_memory_action_changed=state.prior_memory_action_changed,
            current_action_differs_from_base=current_memory_changed,
            current_memory_counterfactual_effects=own_action_effects,
            horde_predictions=horde_predictions,
            horde_cumulants=horde_cumulants,
            horde_td_errors=reported_horde_td_errors,
            horde_squared_errors=jnp.square(horde_td_errors),
            world_model_next_predictions=predicted_next,
            world_model_reward_predictions=predicted_reward,
            world_model_discount_predictions=predicted_discount,
            world_model_errors=reported_world_errors,
            world_model_expected_errors=world_expected_errors,
            world_model_contract_valid=world_contract_valid,
            horde_contract_valid=horde_contract_valid,
            feature_task_targets=jnp.stack(
                tuple(item.task_targets for item in feature)
            ).reshape((_N_AGENTS, _N_FEATURE_TASKS)),
            feature_task_predictions=jnp.stack(
                tuple(item.task_predictions for item in feature)
            ).reshape((_N_AGENTS, _N_FEATURE_TASKS)),
            feature_generation_pre_words=feature_pre,
            feature_generation_post_words=feature_post,
            feature_lifecycle_flags=jnp.stack(tuple(lifecycle_flags)),
            old_bank_routing_flags=old_bank_routing_flags,
            old_bank_routing_valid=old_bank_routing_valid,
            option_executing_pre=jnp.stack(
                tuple(
                    _feature_bundle(agent).oak_state.stomp_state.executing_option
                    for agent in agents
                )
            ).astype(jnp.int32),
            option_executing_post=jnp.stack(
                (
                    _feature_bundle(committed_state.agent_0)
                    .oak_state.stomp_state.executing_option,
                    _feature_bundle(committed_state.agent_1)
                    .oak_state.stomp_state.executing_option,
                )
            ).astype(jnp.int32),
            memory_query_before_write=jnp.stack(
                tuple(item.query_before_write for item in memory)
            ).astype(jnp.bool_),
            memory_prestate_query_count=jnp.stack(
                tuple(item.deterministic_prestate_query_count for item in memory)
            ).astype(jnp.int32),
            memory_wrote=jnp.stack(tuple(item.wrote for item in memory)).astype(jnp.bool_),
            memory_slots=jnp.stack(tuple(item.slot for item in memory)).astype(jnp.int32),
            memory_evicted=jnp.stack(tuple(item.evicted for item in memory)).astype(jnp.bool_),
            memory_evicted_provenance_ids=jnp.stack(
                tuple(item.evicted_provenance_id for item in memory)
            ).astype(jnp.int32),
            memory_retrieval_available=next_retrieval_available,
            memory_retrieval_neighbor_mask=jnp.stack(
                tuple(item.proposal.retrieval.neighbor_mask for item in memory)
            ),
            memory_retrieval_neighbor_provenance_ids=jnp.stack(
                tuple(item.proposal.retrieval.neighbor_provenance_ids for item in memory)
            ).astype(jnp.int32),
            memory_action_changed=next_action_changed,
            memory_rows_reencoded=jnp.stack(
                tuple(item.rebind.valid_rows_reencoded for item in feature_memory)
            ).astype(jnp.int32),
            feature_memory_rebind_applied=jnp.stack(
                tuple(item.rebind.transaction_applied for item in feature_memory)
            ).astype(jnp.bool_),
            memory_contract_valid=memory_contract_valid,
            source_state_valid=source_state_valid,
            source_clocks_aligned=source_clocks_aligned,
            candidate_clocks_aligned=candidate_clocks_aligned,
            preview_updates_valid=preview_valid,
            candidate_updates_valid=candidate_valid,
            forced_outer_rejection=force_reject,
            outer_transaction_committed=outer_commit,
        )
        return HiddenPrototypeTwoAgentStepResult(state=committed_state, trace=trace)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compiled_step(
        self,
        state: HiddenPrototypeTwoAgentState,
        event_index: Array,
        *,
        route_inference: Array,
        force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
    ) -> HiddenPrototypeTwoAgentStepResult:
        """Outer-JIT sibling of :meth:`step` for parity and repeated execution."""

        return self.step(
            state,
            event_index,
            route_inference=route_inference,
            force_outer_rejection=force_outer_rejection,
        )


def _array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _words_payload(value: Any) -> list[int]:
    return [int(item) for item in _array(value).reshape(2)]


def _vector(value: Any) -> list[Any]:
    array = _array(value)
    if array.dtype.kind == "b":
        return [bool(item) for item in array.tolist()]
    if array.dtype.kind in "iu":
        return [int(item) for item in array.tolist()]
    return [float(item) for item in array.tolist()]


def _matrix(value: Any) -> list[list[Any]]:
    return [_vector(row) for row in _array(value)]


def _memory_effect_outcome(action_changed: bool, effect: float) -> str:
    if not action_changed or effect == 0.0:
        return "neutral"
    return "benefit" if effect > 0.0 else "harm"


def _event_payload(
    trace: HiddenPrototypeTwoAgentTrace,
    *,
    event_index: int,
    protocol: HiddenPrototypeTwoAgentProtocol,
    arm: HiddenPrototypeTwoAgentArm,
) -> dict[str, object]:
    primitive = _array(trace.joint_primitive_actions)
    reward = _array(trace.joint_rewards)
    mean_rewards = _array(trace.mean_agent_rewards)
    own_effects = _array(trace.own_action_effects)
    partner_effects = _array(trace.partner_action_effects)
    interaction_effects = _array(trace.interaction_effects)
    memory_effects = _array(trace.current_memory_counterfactual_effects)
    current_memory_changed = _array(trace.prior_memory_action_changed)
    nested_pre = _array(trace.nested_pre_words)
    nested_post = _array(trace.nested_post_words)
    old_flags = _array(trace.old_bank_routing_flags)
    lifecycle_flags = _array(trace.feature_lifecycle_flags)
    agents: list[dict[str, object]] = []
    for agent_index in range(_N_AGENTS):
        nested = {
            name: {
                "pre": _words_payload(nested_pre[agent_index, clock_index]),
                "post": _words_payload(nested_post[agent_index, clock_index]),
            }
            for clock_index, name in enumerate(NESTED_CLOCK_NAMES)
        }
        old_bank = {
            name: bool(old_flags[agent_index, flag_index])
            for flag_index, name in enumerate(OLD_BANK_ROUTING_FLAG_NAMES)
        }
        old_bank["valid"] = bool(_array(trace.old_bank_routing_valid)[agent_index])
        effect = float(memory_effects[agent_index])
        changed = bool(current_memory_changed[agent_index])
        retrieval_mask = _array(trace.memory_retrieval_neighbor_mask)[agent_index]
        retrieval_ids = _array(trace.memory_retrieval_neighbor_provenance_ids)[agent_index]
        agents.append(
            {
                "agent_index": agent_index,
                "nested_clocks": nested,
                "action": int(_array(trace.actions)[agent_index]),
                "counterfactual_base_action": int(
                    _array(trace.counterfactual_base_actions)[agent_index]
                ),
                "next_preview_action": int(
                    _array(trace.next_preview_actions)[agent_index]
                ),
                "next_committed_action": int(
                    _array(trace.next_committed_actions)[agent_index]
                ),
                "same_prestate_reward_effects": {
                    "own_action": float(own_effects[agent_index]),
                    "partner_action": float(partner_effects[agent_index]),
                    "interaction": float(interaction_effects[agent_index]),
                },
                "context": {
                    "pre_slot": int(_array(trace.context_pre_slots)[agent_index]),
                    "post_slot": int(_array(trace.context_post_slots)[agent_index]),
                    "post_onehot": _vector(trace.context_post_onehot[agent_index]),
                    "partner_action_observation": _vector(
                        trace.context_partner_action_observations[agent_index]
                    ),
                    "candidate_update_applied": bool(
                        _array(trace.context_candidate_updates_applied)[agent_index]
                    ),
                    "feeds_current_action": False,
                    "feeds_next_action_when_routed": arm.route_inference,
                },
                "horde": {
                    "prequential_prediction": _vector(
                        trace.horde_predictions[agent_index]
                    ),
                    "cumulant": _vector(trace.horde_cumulants[agent_index]),
                    "reported_td_error": _vector(
                        trace.horde_td_errors[agent_index]
                    ),
                    "squared_error": _vector(
                        trace.horde_squared_errors[agent_index]
                    ),
                    "explicit_prediction_matches_managed_update": bool(
                        _array(trace.horde_contract_valid)[agent_index]
                    ),
                },
                "world_model": {
                    "action_scope": "owner_primitive_action_only",
                    "partner_action_observed": False,
                    "prequential_next_observation": _vector(
                        trace.world_model_next_predictions[agent_index]
                    ),
                    "prequential_reward": float(
                        _array(trace.world_model_reward_predictions)[agent_index]
                    ),
                    "prequential_discount": float(
                        _array(trace.world_model_discount_predictions)[agent_index]
                    ),
                    "reported_error": float(
                        _array(trace.world_model_errors)[agent_index]
                    ),
                    "reconstructed_error": float(
                        _array(trace.world_model_expected_errors)[agent_index]
                    ),
                    "prediction_error_contract_valid": bool(
                        _array(trace.world_model_contract_valid)[agent_index]
                    ),
                },
                "oak_horde_feature_consumption": {
                    "task_targets": _vector(trace.feature_task_targets[agent_index]),
                    "task_predictions": _vector(
                        trace.feature_task_predictions[agent_index]
                    ),
                    "option_executing_pre": int(
                        _array(trace.option_executing_pre)[agent_index]
                    ),
                    "option_executing_post": int(
                        _array(trace.option_executing_post)[agent_index]
                    ),
                    "feature_generation_pre_words": _words_payload(
                        trace.feature_generation_pre_words[agent_index]
                    ),
                    "feature_generation_post_words": _words_payload(
                        trace.feature_generation_post_words[agent_index]
                    ),
                    "lifecycle_flags": {
                        name: bool(lifecycle_flags[agent_index, flag_index])
                        for flag_index, name in enumerate(
                            (
                                "curation_proposed",
                                "safe_curation_boundary",
                                "curation_deferred",
                                "routing_attempted",
                                "curation_committed",
                                "curation_rolled_back",
                            )
                        )
                    },
                },
                "old_bank_routing": old_bank,
                "memory": {
                    "query_before_write": bool(
                        _array(trace.memory_query_before_write)[agent_index]
                    ),
                    "deterministic_prestate_queries": int(
                        _array(trace.memory_prestate_query_count)[agent_index]
                    ),
                    "wrote": bool(_array(trace.memory_wrote)[agent_index]),
                    "slot": int(_array(trace.memory_slots)[agent_index]),
                    "evicted": bool(_array(trace.memory_evicted)[agent_index]),
                    "evicted_provenance_id": int(
                        _array(trace.memory_evicted_provenance_ids)[agent_index]
                    ),
                    "retrieval_available_for_next_decision": bool(
                        _array(trace.memory_retrieval_available)[agent_index]
                    ),
                    "retrieval_neighbor_mask": [bool(item) for item in retrieval_mask],
                    "retrieval_neighbor_provenance_ids": [
                        int(item) for item in retrieval_ids
                    ],
                    "changed_next_action": bool(
                        _array(trace.memory_action_changed)[agent_index]
                    ),
                    "changed_current_action": changed,
                    "current_action_differs_from_base": bool(
                        _array(trace.current_action_differs_from_base)[agent_index]
                    ),
                    "current_unilateral_reward_effect": effect,
                    "current_counterfactual_outcome": _memory_effect_outcome(
                        changed, effect
                    ),
                    "rows_reencoded": int(
                        _array(trace.memory_rows_reencoded)[agent_index]
                    ),
                    "feature_rebind_applied": bool(
                        _array(trace.feature_memory_rebind_applied)[agent_index]
                    ),
                    "candidate_contract_valid": bool(
                        _array(trace.memory_contract_valid)[agent_index]
                    ),
                    "carried_by_outer_transaction": bool(
                        trace.outer_transaction_committed
                    ),
                },
            }
        )
    return {
        "event_index": event_index,
        "phase": _phase_for_step(
            event_index, protocol.prototype_protocol.segment_length
        ),
        "phase_step": event_index % protocol.prototype_protocol.segment_length,
        "environment_pre_words": _words_payload(trace.environment_pre_words),
        "environment_post_words": _words_payload(trace.environment_post_words),
        "environment_proposal_words": {
            name: {
                "pre": _words_payload(trace.environment_proposal_pre_words[index]),
                "post": _words_payload(trace.environment_proposal_post_words[index]),
            }
            for index, name in enumerate(_JOINT_PROPOSAL_NAMES)
        },
        "joint_dispatch": {
            "primitive_actions": {
                name: [int(item) for item in primitive[index]]
                for index, name in enumerate(_JOINT_PROPOSAL_NAMES)
            },
            "rewards": {
                name: [float(item) for item in reward[index]]
                for index, name in enumerate(_JOINT_PROPOSAL_NAMES)
            },
            "mean_agent_reward": {
                name: float(mean_rewards[index])
                for index, name in enumerate(_JOINT_PROPOSAL_NAMES)
            },
            "reward_aggregation": "arithmetic mean of the two receiving-agent rewards",
            "effects": {
                "per_agent": [
                    {
                        "agent_index": agent_index,
                        "own_action": float(own_effects[agent_index]),
                        "partner_action": float(partner_effects[agent_index]),
                        "interaction": float(interaction_effects[agent_index]),
                    }
                    for agent_index in range(_N_AGENTS)
                ],
                "joint_mean": float(trace.joint_mean_effect),
            },
            "all_proposals_applied": bool(
                np.all(_array(trace.environment_proposals_applied))
            ),
        },
        "context": {
            "pre_words": _matrix(trace.context_pre_words),
            "post_words": _matrix(trace.context_post_words),
            "post_onehot": _matrix(trace.context_post_onehot),
            "candidate_updates_applied": _vector(
                trace.context_candidate_updates_applied
            ),
        },
        "no_oracle": {
            "current_rule_consumed": False,
            "phase_or_boundary_consumed": False,
            "environment_clock_consumed": False,
            "visible_world_cue_consumed": False,
            "world_visible_cues_for_evaluator_only": _matrix(
                trace.world_visible_cues_for_evaluator_only
            ),
            "learner_context_channels_pre": _matrix(
                trace.learner_context_channels_pre
            ),
            "learner_context_channels_next": _matrix(
                trace.learner_context_channels_next
            ),
            "inference_routed": arm.route_inference,
        },
        "agents": agents,
        "source_state_valid": bool(trace.source_state_valid),
        "source_clocks_aligned": bool(trace.source_clocks_aligned),
        "candidate_clocks_aligned": bool(trace.candidate_clocks_aligned),
        "all_previews_valid": bool(np.all(_array(trace.preview_updates_valid))),
        "all_candidates_valid": bool(np.all(_array(trace.candidate_updates_valid))),
        "forced_outer_rejection": bool(trace.forced_outer_rejection),
        "outer_transaction_committed": bool(trace.outer_transaction_committed),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _metrics(
    trace: Sequence[Mapping[str, object]],
    protocol: HiddenPrototypeTwoAgentProtocol,
) -> dict[str, object]:
    phase_rewards: dict[str, dict[str, float]] = {}
    window = protocol.prototype_protocol.metric_window
    for phase in ("A1", "B", "A2"):
        events = [event for event in trace if event["phase"] == phase]
        rewards = [
            cast(
                float,
                cast(Mapping[str, Any], event["joint_dispatch"])[
                    "mean_agent_reward"
                ]["actual_actual"],
            )
            for event in events
        ]
        phase_rewards[phase] = {
            "mean": _mean(rewards),
            "early_mean": _mean(rewards[:window]),
            "tail_mean": _mean(rewards[-window:]),
        }
    evicted_ids: list[int] = []
    outcomes: list[str] = []
    horde_errors: list[float] = []
    world_errors: list[float] = []
    context_switches = [0, 0]
    contexts_used: list[set[tuple[int, ...]]] = [set(), set()]
    memory_evictions = 0
    memory_retrievals = 0
    memory_action_changes = 0
    curation_commits = 0
    reward_effects: dict[str, list[float]] = {
        "agent0_own_action": [],
        "agent1_own_action": [],
        "agent0_partner_action": [],
        "agent1_partner_action": [],
        "agent0_interaction": [],
        "agent1_interaction": [],
        "joint_mean": [],
    }
    for event in trace:
        context = cast(Mapping[str, Any], event["context"])
        for agent_index in range(_N_AGENTS):
            pre_slot = cast(list[int], cast(list[Any], context["post_onehot"])[agent_index])
            contexts_used[agent_index].add(tuple(pre_slot))
        for agent in cast(list[Mapping[str, Any]], event["agents"]):
            index = cast(int, agent["agent_index"])
            agent_context = cast(Mapping[str, Any], agent["context"])
            context_switches[index] += int(
                agent_context["pre_slot"] != agent_context["post_slot"]
            )
            horde_errors.extend(cast(Mapping[str, list[float]], agent["horde"])["squared_error"])
            world_errors.append(
                cast(float, cast(Mapping[str, Any], agent["world_model"])["reported_error"])
            )
            feature = cast(Mapping[str, Any], agent["oak_horde_feature_consumption"])
            flags = cast(Mapping[str, bool], feature["lifecycle_flags"])
            curation_commits += int(flags["curation_committed"])
            memory = cast(Mapping[str, Any], agent["memory"])
            memory_evictions += int(cast(bool, memory["evicted"]))
            if cast(bool, memory["evicted"]):
                evicted_ids.append(cast(int, memory["evicted_provenance_id"]))
            memory_retrievals += int(
                cast(bool, memory["retrieval_available_for_next_decision"])
            )
            memory_action_changes += int(cast(bool, memory["changed_next_action"]))
            outcomes.append(cast(str, memory["current_counterfactual_outcome"]))
        effects = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], event["joint_dispatch"])["effects"],
        )
        per_agent = cast(list[Mapping[str, float]], effects["per_agent"])
        reward_effects["agent0_own_action"].append(per_agent[0]["own_action"])
        reward_effects["agent1_own_action"].append(per_agent[1]["own_action"])
        reward_effects["agent0_partner_action"].append(per_agent[0]["partner_action"])
        reward_effects["agent1_partner_action"].append(per_agent[1]["partner_action"])
        reward_effects["agent0_interaction"].append(per_agent[0]["interaction"])
        reward_effects["agent1_interaction"].append(per_agent[1]["interaction"])
        reward_effects["joint_mean"].append(cast(float, effects["joint_mean"]))
    expected_evictions = 2 * max(
        protocol.total_steps - protocol.prototype_protocol.memory_capacity,
        0,
    )
    return {
        "phase_mean_agent_reward": phase_rewards,
        "reward_aggregation": "arithmetic mean of the two receiving-agent rewards",
        "recurrence": {
            "A2_early_minus_A1_tail": (
                phase_rewards["A2"]["early_mean"] - phase_rewards["A1"]["tail_mean"]
            ),
            "A2_tail_minus_A1_tail": (
                phase_rewards["A2"]["tail_mean"] - phase_rewards["A1"]["tail_mean"]
            ),
            "A2_reacquisition": (
                phase_rewards["A2"]["tail_mean"] - phase_rewards["A2"]["early_mean"]
            ),
        },
        "context_switches": context_switches,
        "context_onehots_used": [sorted(values) for values in contexts_used],
        "mean_horde_squared_error": _mean(horde_errors),
        "mean_world_model_error": _mean(world_errors),
        "feature_curation_commits": curation_commits,
        "memory_evictions": memory_evictions,
        "memory_eviction_accounting": {
            "expected_after_fixed_capacity_fill": expected_evictions,
            "observed": memory_evictions,
            "exact": memory_evictions == expected_evictions,
        },
        "memory_evicted_provenance_summary": {
            "count": len(evicted_ids),
            "unique_count": len(set(evicted_ids)),
            "minimum": min(evicted_ids) if evicted_ids else None,
            "maximum": max(evicted_ids) if evicted_ids else None,
            "first_eight": evicted_ids[:8],
            "last_eight": evicted_ids[-8:],
            "exact_sequence_location": "trace[*].agents[*].memory.evicted_provenance_id",
        },
        "memory_retrievals": memory_retrievals,
        "memory_next_action_changes": memory_action_changes,
        "memory_counterfactual_outcomes_observed": sorted(set(outcomes)),
        "memory_counterfactual_outcome_counts": {
            name: outcomes.count(name) for name in ("benefit", "harm", "neutral")
        },
        "mean_same_prestate_reward_effects": {
            name: _mean(values) for name, values in reward_effects.items()
        },
    }


def _work(total_steps: int) -> dict[str, int]:
    return {
        "requested_joint_transitions": total_steps,
        "environment_proposal_calls": 4 * total_steps,
        "counterfactual_environment_proposal_calls": 3 * total_steps,
        "committed_environment_transitions": total_steps,
        "context_inference_update_calls": 2 * total_steps,
        "context_inference_carried_updates": 2 * total_steps,
        "discarded_preview_update_calls": 2 * total_steps,
        "committed_candidate_update_calls": 2 * total_steps,
        "prototype_update_calls": 4 * total_steps,
        "world_model_update_calls": 4 * total_steps,
        "world_model_carried_updates": 2 * total_steps,
        "explicit_world_model_prediction_calls": 2 * total_steps,
        "managed_horde_update_calls": 4 * total_steps,
        "explicit_horde_prediction_calls": 2 * total_steps,
        "feature_lifecycle_observe_calls": 4 * total_steps,
        "memory_sidecars_supplied": 2 * total_steps,
        "memory_query_before_write_transactions": 2 * total_steps,
        "outer_source_state_validations": total_steps,
        "outer_candidate_state_validations": total_steps,
        "outer_clock_alignment_checks": 2 * total_steps,
        "per_agent_old_bank_contract_checks": 2 * total_steps,
        "per_agent_memory_contract_checks": 2 * total_steps,
        "per_agent_horde_contract_checks": 2 * total_steps,
        "per_agent_world_model_contract_checks": 2 * total_steps,
        "outer_atomic_decisions": total_steps,
        "checkpoint_save_calls": 0,
        "checkpoint_load_calls": 0,
        "resets_after_initialization": 0,
        "boundary_callbacks": 0,
        "external_partner_policy_calls": 0,
    }


def _resources(
    evaluator: HiddenPrototypeTwoAgentEvaluator,
    initial: HiddenPrototypeTwoAgentState,
    final: HiddenPrototypeTwoAgentState,
    phase_boundary_bytes: Sequence[int],
    peak_persistent_bytes: int,
) -> dict[str, object]:
    initial_agents = (
        measure_prototype_agent_state_resources(initial.agent_0),
        measure_prototype_agent_state_resources(initial.agent_1),
    )
    final_agents = (
        measure_prototype_agent_state_resources(final.agent_0),
        measure_prototype_agent_state_resources(final.agent_1),
    )
    context_bytes = measure_context_inference_state_nbytes(initial.context_0)
    environment_bytes = evaluator.world.resource_budget.state_nbytes
    auxiliary_bytes = (
        int(initial.counterfactual_base_actions.nbytes)
        + int(initial.prior_memory_retrieval_available.nbytes)
        + int(initial.prior_memory_action_changed.nbytes)
    )
    initial_total = _tree_nbytes(initial)
    final_total = _tree_nbytes(final)
    agent_bytes = [item.total_nbytes for item in initial_agents]
    proposal_state_copy_lower_bound = (
        4 * environment_bytes + 2 * context_bytes + 2 * sum(agent_bytes)
    )
    decomposed_initial_total = (
        environment_bytes + 2 * context_bytes + sum(agent_bytes) + auxiliary_bytes
    )
    if decomposed_initial_total != initial_total:
        raise RuntimeError("hidden Prototype persistent-state byte decomposition drifted")
    return {
        "logical_fixed_allocation": True,
        "initial_persistent_state_nbytes": initial_total,
        "final_persistent_state_nbytes": final_total,
        "phase_boundary_persistent_state_nbytes": list(phase_boundary_bytes),
        "peak_persistent_state_nbytes": peak_persistent_bytes,
        "environment_state_nbytes": environment_bytes,
        "context_state_nbytes_per_agent": context_bytes,
        "context_resource_declaration": evaluator.context.resource_budget.to_dict(),
        "prototype_state_nbytes_per_agent_initial": agent_bytes,
        "prototype_state_nbytes_per_agent_final": [
            item.total_nbytes for item in final_agents
        ],
        "prototype_state_measurements_initial": [
            item.to_config() for item in initial_agents
        ],
        "outer_auxiliary_state_nbytes": auxiliary_bytes,
        "initial_persistent_state_decomposition_nbytes": decomposed_initial_total,
        "staged_full-state-copy_lower_bound_nbytes": proposal_state_copy_lower_bound,
        "staged_full-state-copy_lower_bound_semantics": (
            "four environment states + two context candidates + two previews + "
            "two Prototype candidates; diagnostics/compiler workspaces excluded"
        ),
        "compiler_allocator_or_device_residency_claimed": False,
    }


def _run_arm(
    evaluator: HiddenPrototypeTwoAgentEvaluator,
    arm: HiddenPrototypeTwoAgentArm,
    *,
    verify_engine_parity: bool,
) -> dict[str, object]:
    state = evaluator.initialize(arm.name)
    initial = state
    route = jnp.asarray(arm.route_inference, dtype=jnp.bool_)
    parity = {
        "checked": verify_engine_parity,
        "state_and_trace_equivalent": False,
        "float_contract": "rtol=1e-6, atol=1e-7; discrete leaves exact",
        "measurement_events_not_carried_into_life": 2 if verify_engine_parity else 0,
        "measurement_logical_work": _work(2 if verify_engine_parity else 0),
    }
    if verify_engine_parity:
        eager = evaluator.step(
            state,
            jnp.asarray(0, dtype=jnp.int32),
            route_inference=route,
        )
        compiled = evaluator.compiled_step(
            state,
            jnp.asarray(0, dtype=jnp.int32),
            route_inference=route,
        )
        parity["state_and_trace_equivalent"] = _tree_equivalent(eager, compiled)
        if not cast(bool, parity["state_and_trace_equivalent"]):
            raise RuntimeError("eager and outer-JIT first events are not equivalent")

    events: list[dict[str, object]] = []
    phase_boundary_bytes = [_tree_nbytes(state)]
    peak_bytes = phase_boundary_bytes[0]
    for event_index in range(evaluator.protocol.total_steps):
        result = evaluator.compiled_step(
            state,
            jnp.asarray(event_index, dtype=jnp.int32),
            route_inference=route,
        )
        if not bool(result.trace.outer_transaction_committed):
            raise RuntimeError(f"hidden Prototype outer event {event_index} rejected")
        events.append(
            _event_payload(
                result.trace,
                event_index=event_index,
                protocol=evaluator.protocol,
                arm=arm,
            )
        )
        state = result.state
        current_bytes = _tree_nbytes(state)
        peak_bytes = max(peak_bytes, current_bytes)
        if (event_index + 1) % evaluator.protocol.prototype_protocol.segment_length == 0:
            phase_boundary_bytes.append(current_bytes)
    if len(phase_boundary_bytes) != 4:
        raise RuntimeError("A-B-A run did not produce four persistent-state boundaries")
    resources = _resources(
        evaluator,
        initial,
        state,
        phase_boundary_bytes,
        peak_bytes,
    )
    metrics = _metrics(events, evaluator.protocol)
    if not cast(
        bool,
        cast(Mapping[str, Any], metrics["memory_eviction_accounting"])["exact"],
    ):
        raise RuntimeError("fixed-capacity memory eviction accounting drifted")
    return {
        "arm": arm.name,
        "route_inference": arm.route_inference,
        "trace": events,
        "metrics": metrics,
        "resources": resources,
        "work": _work(evaluator.protocol.total_steps),
        "execution_parity": parity,
        "final_context_slots": [
            int(state.context_0.active_context),
            int(state.context_1.active_context),
        ],
        "checkpoint_resume_claimed": False,
    }


LIMITATIONS: Final = (
    "one consumed development root and no statistical inference",
    "bounded pair features are not general compositional feature discovery",
    "two inferred slots for two objectives do not exercise context-capacity eviction",
    "the stable-base world model observes only its owner's primitive action",
    "synthetic zero uncertainty/safety sidecars exercise existing memory gates without calibration",
    "memory benefit/harm is a same-event action counterfactual, not a promotion claim",
    "no atomic composite checkpoint/resume mechanism is inherited",
    "logical bytes/work exclude compiler workspaces, allocator residency, FLOPs, and latency",
)


def run_hidden_prototype_two_agent_continual_life_development(
    protocol: HiddenPrototypeTwoAgentProtocol | None = None,
    *,
    verify_engine_parity: bool = True,
) -> dict[str, object]:
    """Run the two predeclared arms in memory on one consumed root."""

    if type(verify_engine_parity) is not bool:
        raise TypeError("verify_engine_parity must be an exact bool")
    evaluator = HiddenPrototypeTwoAgentEvaluator(protocol)
    runs = [
        _run_arm(
            evaluator,
            _ARMS_BY_NAME[name],
            verify_engine_parity=verify_engine_parity,
        )
        for name in evaluator.protocol.arm_names
    ]
    state_shape_matched = len({
        cast(int, cast(Mapping[str, Any], run["resources"])[
            "initial_persistent_state_nbytes"
        ])
        for run in runs
    }) == 1
    work_matched = all(run["work"] == runs[0]["work"] for run in runs[1:])
    return {
        "schema_version": HIDDEN_PROTOTYPE_TWO_AGENT_REPORT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "accepted_scientific_evidence": False,
        "acceptance_status": ACCEPTANCE_STATUS,
        "writer_available": False,
        "winner_selected": False,
        "context_capacity_pressure_claimed": False,
        "checkpoint_resume_claimed": False,
        "protocol": evaluator.protocol.to_config(),
        "environment_config": evaluator.world.to_config(),
        "agent_config": evaluator.agent.to_config(),
        "arm_definitions": [
            _ARMS_BY_NAME[name].to_config() for name in evaluator.protocol.arm_names
        ],
        "runs": runs,
        "matched_comparison": {
            "persistent_state_shape_matched": state_shape_matched,
            "logical_work_matched": work_matched,
            "same_consumed_root": True,
            "same_context_configuration_and_update_work": True,
            "same_Prototype_configuration": True,
            "only_declared_intervention": (
                "route the same past-only inferred onehot into the two formerly "
                "visible cue coordinates"
            ),
        },
        "claim_assessments": {
            "hidden_context_control_benefit": "not-assessed",
            "memory_retention_benefit": "not-assessed",
            "general_feature_finding": "not-assessed",
            "context_capacity_selective_retention": "not-assessed",
            "Alberta_Plan_completion": "not-assessed",
            "scientific_evidence": "not-assessed",
        },
        "limitations": list(LIMITATIONS),
    }


def validate_static_contract() -> tuple[str, ...]:
    """Return configuration drift errors without executing a life."""

    errors: list[str] = []
    if INHERITED_CONTEXT_INFERENCE_CONFIG is not _CONTEXT_CONFIG:
        errors.append("ContextInferenceConfig was reconstructed instead of inherited")
    config = INHERITED_CONTEXT_INFERENCE_CONFIG
    if (config.n_actions, config.observation_dim, config.max_contexts) != (2, 2, 2):
        errors.append("inherited hidden-context geometry drifted")
    if tuple(arm.name for arm in HIDDEN_PROTOTYPE_TWO_AGENT_ARMS) != (
        HIDDEN_INFERRED_FULL,
        HIDDEN_INFERENCE_UNROUTED,
    ):
        errors.append("hidden Prototype arm order drifted")
    if CONSUMED_DEVELOPMENT_ROOT != DEVELOPMENT_SEEDS[0]:
        errors.append("consumed development root drifted")
    try:
        protocol = HiddenPrototypeTwoAgentProtocol()
        config_payload = protocol.to_config()
        if config_payload["schedule"] != ["A1", "B", "A2"]:
            errors.append("A-B-A schedule drifted")
        if config_payload["output_writes_allowed"] is not False:
            errors.append("output writes became available")
    except (TypeError, ValueError) as error:
        errors.append(f"default protocol is invalid: {error}")
    return tuple(errors)


__all__ = [
    "ACCEPTANCE_STATUS",
    "CONSUMED_DEVELOPMENT_ROOT",
    "DEVELOPMENT_ONLY",
    "HIDDEN_INFERENCE_UNROUTED",
    "HIDDEN_INFERRED_FULL",
    "HIDDEN_PROTOTYPE_TWO_AGENT_ARMS",
    "HIDDEN_PROTOTYPE_TWO_AGENT_PROTOCOL_SCHEMA",
    "HIDDEN_PROTOTYPE_TWO_AGENT_REPORT_SCHEMA",
    "HiddenPrototypeTwoAgentEvaluator",
    "HiddenPrototypeTwoAgentProtocol",
    "HiddenPrototypeTwoAgentState",
    "HiddenPrototypeTwoAgentStepResult",
    "HiddenPrototypeTwoAgentTrace",
    "INHERITED_CONTEXT_INFERENCE_CONFIG",
    "NESTED_CLOCK_NAMES",
    "OUTPUT_WRITES_ALLOWED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "run_hidden_prototype_two_agent_continual_life_development",
    "validate_static_contract",
]
