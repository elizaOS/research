# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-any-return"
"""Development-only U1 factorized planning around the routed hidden U0 life.

The routed :class:`HiddenPrototypeTwoAgentEvaluator` remains the sole owner of
environment, context, Prototype, feature, Horde, world-model, and experiential-
memory updates.  This wrapper gives its already-existing four same-prestate
environment proposals a narrower interpretation: the current Prototype action
is the factorized planner's actual dispatch and U0's auxiliary base action is
normalized to the planner cache's post-memory counterfactual base.  Consequently
the four proposal rows are exactly ``actual/actual``, ``base/actual``,
``actual/base``, and ``base/base`` for the planner intervention.  They are not a
same-event memory-effect panel.

After U0 has produced both post-memory Prototype candidates in this same pure
call, the factorized planner learns from the exact executed joint action, row-0
rewards, routed post-memory raw observations, and continuation 1.0.  Both
planned Prototype successors, both planner sidecars, and the complete U0
candidate are then carried by one outer all-or-none decision.  The all-true
caller mask keeps the synthetic intervention reachable; it is not a physical
safety certification.

The full 1,536-event A-B-A runner is process-local, locked, and at-most-once
across both success and failure.  It returns only a compact in-memory report.
There is no writer, artifact, threshold, search, winner, checkpoint, evidence,
or promotion path.  U1 adds no post-initialization random draw or replay; the
canonical U0 policy's inherited epsilon-driven RNG activity remains present and
is explicitly outside that zero-additional-draw claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import platform
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Final, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.behavior_model as behavior_model_module
import alberta_framework.core.context_inference as context_inference_module
import alberta_framework.core.grounded_joint_world_model as grounded_world_model_module
import alberta_framework.core.horde as horde_module
import alberta_framework.core.prototype_agent as prototype_agent_module
import alberta_framework.core.prototype_factorized_partner_planner as planner_core
import alberta_framework.core.prototype_feature_memory as prototype_feature_memory_module
import alberta_framework.core.world_model as world_model_module
import alberta_framework.streams.recurring_multiagent as recurring_multiagent_module
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerPlanner,
    PrototypeFactorizedPartnerPlannerConfig,
    PrototypeFactorizedPartnerPlannerState,
)
from alberta_framework.evaluation import (
    hidden_context_coadaptation_development as hidden_context_module,
)
from alberta_framework.evaluation import (
    hidden_prototype_two_agent_continual_life_development as u0_module,
)
from alberta_framework.evaluation import (
    prototype_feature_memory_recurrence_development as feature_memory_recurrence_module,
)
from alberta_framework.evaluation import (
    prototype_two_learning_agent_recurrence_development as two_agent_recurrence_module,
)
from alberta_framework.evaluation.hidden_prototype_two_agent_continual_life_development import (
    CONSUMED_DEVELOPMENT_ROOT,
    HIDDEN_INFERRED_FULL,
    HiddenPrototypeTwoAgentEvaluator,
    HiddenPrototypeTwoAgentProtocol,
    HiddenPrototypeTwoAgentState,
)

PROTOCOL_SCHEMA: Final = (
    "alberta.hidden-prototype-factorized-partner-planning-development.protocol.v1"
)
REPORT_SCHEMA: Final = "alberta.hidden-prototype-factorized-partner-planning-development.report.v1"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
CHECKPOINT_RESUME_CLAIMED: Final = False
THRESHOLDS_OR_WINNER_SELECTION: Final = False
ARTIFACT_BYTES_WRITTEN: Final = 0
ACCEPTANCE_STATUS: Final = "not-assessed"

UNDERLYING_POST_MEMORY_TRANSITION_BINDING_CLAIMED: Final = (
    planner_core.POST_MEMORY_TRANSITION_BINDING_CLAIMED
)
UNDERLYING_BASE_FALLBACK_SOURCE_BINDING_CLAIMED: Final = (
    planner_core.BASE_FALLBACK_SOURCE_BINDING_CLAIMED
)
WRAPPER_LOCAL_POST_MEMORY_CONSTRUCTION_BOUND: Final = True
ALL_TRUE_CALLER_MASK_IS_SAFETY_CERTIFICATION: Final = False
EAGER_JIT_FLOAT_RTOL: Final = 1.0e-6
EAGER_JIT_FLOAT_ATOL: Final = 1.0e-7
EAGER_JIT_DISCRETE_LEAVES_EXACT: Final = True
ADDITIONAL_WRAPPER_PLANNER_POST_INIT_RANDOM_DRAWS_PER_EVENT: Final = 0
ADDITIONAL_WRAPPER_PLANNER_REPLAY_UPDATES_PER_EVENT: Final = 0
INHERITED_U0_POLICY_POST_INIT_RNG_PRESENT: Final = True

LEARNED_PLANNING_ENABLED: Literal["learned_planning_enabled"] = "learned_planning_enabled"
UNIFORM_PLANNING_ENABLED: Literal["uniform_planning_enabled"] = "uniform_planning_enabled"
LEARNED_PLANNING_DISABLED: Literal["learned_planning_disabled"] = "learned_planning_disabled"
FactorizedPlanningArmName = Literal[
    "learned_planning_enabled",
    "uniform_planning_enabled",
    "learned_planning_disabled",
]

_N_AGENTS: Final = 2
_N_ACTIONS: Final = 2
_PLANNER_INITIALIZATION_DOMAIN: Final = 0x55314650  # "U1FP"
_FULL_SEGMENT_LENGTH: Final = 512
_FULL_TOTAL_STEPS: Final = 1_536
_FULL_OBSERVATION_DIM: Final = 8
_FULL_REPRESENTATION_DIM: Final = 12
_EXPECTED_FULL_PLANNER_PAIR_NBYTES: Final = 3_758
_EVENT_CHAIN_GENESIS: Final = "0" * 64
_OUTCOME_NAMES: Final = ("harm", "neutral", "benefit")
_SOURCE_MANIFEST_SCOPE: Final = "selected-direct-files-not-transitive-closure"
RUNTIME_IDENTITY_SCOPE: Final = (
    "selected Python, NumPy, JAX, backend, x64, and device fields; not an "
    "environment, accelerator-driver, XLA-flag, or compiler closure"
)
RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent composite JAX-array bytes and named logical U0/planner calls; "
    "excludes transient diagnostics, compiler workspaces, allocator residency, FLOPs, "
    "latency, and inherited U0 RNG draw internals"
)
_SELECTED_SOURCE_MODULES: Final = (
    ("u0_evaluation_module_sha256", u0_module),
    ("factorized_planner_core_sha256", planner_core),
    ("prototype_agent_core_sha256", prototype_agent_module),
    ("behavior_model_core_sha256", behavior_model_module),
    ("grounded_joint_world_model_core_sha256", grounded_world_model_module),
    ("context_inference_core_sha256", context_inference_module),
    ("horde_core_sha256", horde_module),
    ("prototype_feature_memory_core_sha256", prototype_feature_memory_module),
    ("world_model_core_sha256", world_model_module),
    ("hidden_context_evaluation_module_sha256", hidden_context_module),
    (
        "prototype_feature_memory_recurrence_evaluation_module_sha256",
        feature_memory_recurrence_module,
    ),
    (
        "prototype_two_agent_recurrence_evaluation_module_sha256",
        two_agent_recurrence_module,
    ),
    ("recurring_multiagent_stream_sha256", recurring_multiagent_module),
)


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizedPlanningArm:
    """One predeclared partner-belief/planning authority intervention."""

    name: FactorizedPlanningArmName
    planning_enabled: bool
    uniform_partner_belief: bool
    role: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


FACTORIZED_PLANNING_ARMS: Final = (
    FactorizedPlanningArm(
        LEARNED_PLANNING_ENABLED,
        planning_enabled=True,
        uniform_partner_belief=False,
        role="learned partner belief with factorized one-step planning authority",
    ),
    FactorizedPlanningArm(
        UNIFORM_PLANNING_ENABLED,
        planning_enabled=True,
        uniform_partner_belief=True,
        role="uniform partner-belief intervention with planning authority retained",
    ),
    FactorizedPlanningArm(
        LEARNED_PLANNING_DISABLED,
        planning_enabled=False,
        uniform_partner_belief=False,
        role="learned partner model and matched planning work without dispatch authority",
    ),
)
_ARMS_BY_NAME: Final = {arm.name: arm for arm in FACTORIZED_PLANNING_ARMS}


def _canonical_u0_protocol() -> HiddenPrototypeTwoAgentProtocol:
    return HiddenPrototypeTwoAgentProtocol(arm_names=(HIDDEN_INFERRED_FULL,))


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPrototypeFactorizedPartnerPlanningProtocol:
    """Exact, non-customizable consumed-root 1,536-event A-B-A declaration."""

    schema_version: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ValueError("factorized planning protocol schema is unsupported")

    @property
    def u0_protocol(self) -> HiddenPrototypeTwoAgentProtocol:
        return _canonical_u0_protocol()

    @property
    def segment_length(self) -> int:
        return self.u0_protocol.prototype_protocol.segment_length

    @property
    def total_steps(self) -> int:
        return self.u0_protocol.total_steps

    @property
    def metric_window(self) -> int:
        return self.u0_protocol.prototype_protocol.metric_window

    def to_config(self) -> dict[str, object]:
        root = CONSUMED_DEVELOPMENT_ROOT
        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "output_writes_allowed": False,
            "artifact_bytes_written": 0,
            "thresholds_or_winner_selection": False,
            "checkpoint_resume_claimed": False,
            "schedule": ["A1", "B", "A2"],
            "segment_length": self.segment_length,
            "total_steps": self.total_steps,
            "consumed_development_root": {
                "namespace": root.namespace,
                "index": root.index,
                "environment_seed": root.environment_seed,
                "initialization_seed": root.initialization_seed,
            },
            "u0_protocol": self.u0_protocol.to_config(),
            "u0_arm": HIDDEN_INFERRED_FULL,
            "arm_order": [arm.name for arm in FACTORIZED_PLANNING_ARMS],
            "planner_initialization": {
                "source": "consumed U0 initialization root",
                "fold_in_domain": _PLANNER_INITIALIZATION_DOMAIN,
                "same_key_for_every_arm": True,
                "behavior_and_grounded_genesis_bit_identical": True,
                "canonical_observation_dim": _FULL_OBSERVATION_DIM,
                "canonical_prototype_representation_dim": _FULL_REPRESENTATION_DIM,
            },
            "transaction": {
                "environment_proposals_per_event": 4,
                "environment_proposals_not_eight": True,
                "event_index_bound_to_source_clock": True,
                "proposal_rows": [
                    "actual_actual",
                    "base0_actual1",
                    "actual0_base1",
                    "base_base",
                ],
                "row_zero_is_executed": True,
                "post_memory_candidates_constructed_locally": True,
                "complete_composite_all_or_none": True,
            },
            "safety": {
                "caller_mask": "all_true",
                "physical_safety_certification": False,
            },
            "rng_and_replay": {
                "additional_wrapper_planner_post_init_draws_per_event": 0,
                "additional_wrapper_planner_replay_updates_per_event": 0,
                "inherited_u0_policy_post_init_rng_present": True,
                "inherited_u0_rng_draw_count_in_u1_scope": False,
            },
            "execution_parity": {
                "float_rtol": EAGER_JIT_FLOAT_RTOL,
                "float_atol": EAGER_JIT_FLOAT_ATOL,
                "discrete_leaves_exact": EAGER_JIT_DISCRETE_LEAVES_EXACT,
                "floating_leaves_bit_identical_claimed": False,
            },
            "nonclaims": {
                "underlying_post_memory_transition_binding_claimed": False,
                "underlying_base_fallback_source_binding_claimed": False,
                "same_event_memory_reward_effect_reported": False,
                "safety_certification": False,
                "checkpoint_resume": False,
                "scientific_evidence": False,
                "Alberta_Plan_completion": False,
            },
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HiddenPrototypeFactorizedPartnerPlanningProtocol:
        canonical = cls()
        if set(payload) != set(canonical.to_config()):
            raise ValueError("factorized planning protocol fields do not match")
        if dict(payload) != canonical.to_config():
            raise ValueError("factorized planning protocol is not the frozen declaration")
        return canonical


class _ProcessAttemptLatch:
    """Execute one exact-string builder once, sealing success or failure."""

    def __init__(self, builder: Callable[[], str]) -> None:
        if not callable(builder):
            raise TypeError("process-attempt builder must be callable")
        self._builder = builder
        self._lock = threading.Lock()
        self._attempted = False
        self._value: str | None = None
        self._failure: BaseException | None = None

    def get(self) -> str:
        with self._lock:
            if self._attempted:
                if self._failure is not None:
                    raise RuntimeError("the process-local U1 panel is sealed after failure") from (
                        self._failure
                    )
                if self._value is None:
                    raise RuntimeError("the process-local U1 latch is internally invalid")
                return self._value
            self._attempted = True
            try:
                value = self._builder()
                if type(value) is not str:
                    raise TypeError("process-attempt builder must return an exact string")
            except BaseException as error:
                self._failure = error
                raise
            self._value = value
            return value


@chex.dataclass(frozen=True)
class HiddenPrototypeFactorizedPartnerPlanningState:
    """Complete U0 plus both factorized planner sidecars."""

    u0: HiddenPrototypeTwoAgentState
    planner: PrototypeFactorizedPartnerPlannerState


@chex.dataclass(frozen=True)
class HiddenPrototypeFactorizedPartnerPlanningTrace:
    """Fixed-shape causal and atomic audit for one composite event."""

    event_index: Array
    event_index_matches_source_clock: Array
    actual_actions: Array
    base_actions: Array
    joint_primitive_actions: Array
    joint_rewards: Array
    planner_action_changed: Array
    planner_reward_effects: Array
    planner_outcome_codes: Array
    next_base_actions: Array
    next_effective_actions: Array
    next_planner_action_changed: Array
    next_learned_partner_probabilities: Array
    next_applied_partner_probabilities: Array
    planner_executed_actions: Array
    planner_observed_partner_actions: Array
    planner_post_memory_observations: Array
    planner_input_rewards: Array
    planner_input_continuation: Array
    planner_grounded_targets: Array
    planner_grounded_joint_action_indices: Array
    behavior_losses: Array
    behavior_update_applied: Array
    grounded_losses: Array
    grounded_update_applied: Array
    context_pre_slots: Array
    context_post_slots: Array
    memory_query_before_write: Array
    memory_wrote: Array
    memory_evicted: Array
    memory_retrieval_available: Array
    memory_action_changed: Array
    source_u0_valid: Array
    source_planner_cache_valid: Array
    source_aux_cache_binding_valid: Array
    source_composite_valid: Array
    u0_candidate_committed: Array
    planner_candidate_committed: Array
    local_post_memory_binding_valid: Array
    planner_cube_binding_valid: Array
    candidate_u0_valid: Array
    candidate_planner_cache_valid: Array
    candidate_aux_cache_binding_valid: Array
    candidate_composite_valid: Array
    no_oracle_channel_consumed: Array
    all_true_caller_mask_used: Array
    all_true_caller_mask_is_safety_certification: Array
    forced_outer_rejection: Array
    outer_transaction_committed: Array


@chex.dataclass(frozen=True)
class HiddenPrototypeFactorizedPartnerPlanningStepResult:
    """Atomically selected composite state plus proposal diagnostics."""

    state: HiddenPrototypeFactorizedPartnerPlanningState
    trace: HiddenPrototypeFactorizedPartnerPlanningTrace


def _planner_config(
    arm: FactorizedPlanningArm,
    *,
    observation_dim: int,
    representation_dim: int,
) -> PrototypeFactorizedPartnerPlannerConfig:
    return PrototypeFactorizedPartnerPlannerConfig(
        observation_dim=observation_dim,
        prototype_representation_dim=representation_dim,
        n_actions=_N_ACTIONS,
        planning_enabled=arm.planning_enabled,
        uniform_partner_belief=arm.uniform_partner_belief,
    )


def _strict_arm(name: FactorizedPlanningArmName) -> FactorizedPlanningArm:
    if type(name) is not str or name not in _ARMS_BY_NAME:
        raise ValueError("factorized planning arm is unsupported")
    return _ARMS_BY_NAME[name]


class HiddenPrototypeFactorizedPartnerPlanningEvaluator:
    """One routed U0 evaluator and one arm-specific paired U1 planner."""

    def __init__(
        self,
        arm_name: FactorizedPlanningArmName,
        *,
        u0_evaluator: HiddenPrototypeTwoAgentEvaluator | None = None,
    ) -> None:
        self.arm = _strict_arm(arm_name)
        if u0_evaluator is None:
            resolved_u0 = HiddenPrototypeTwoAgentEvaluator(_canonical_u0_protocol())
        else:
            if type(u0_evaluator) is not HiddenPrototypeTwoAgentEvaluator:
                raise TypeError("u0_evaluator must be an exact HiddenPrototypeTwoAgentEvaluator")
            resolved_u0 = u0_evaluator
        if resolved_u0.protocol.arm_names != (HIDDEN_INFERRED_FULL,):
            raise ValueError("U1 requires the singular routed hidden-inference U0 arm")
        self.u0 = resolved_u0
        base = resolved_u0.protocol.prototype_protocol
        self.planner = PrototypeFactorizedPartnerPlanner(
            resolved_u0.agent,
            _planner_config(
                self.arm,
                observation_dim=base.base_observation_dim,
                representation_dim=resolved_u0.agent.config.oak.observation_dim,
            ),
        )
        self._safety_masks = jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)

    @property
    def safety_action_masks(self) -> Array:
        return self._safety_masks

    def _planner_initialization_key(self) -> Array:
        return jr.fold_in(
            jr.key(CONSUMED_DEVELOPMENT_ROOT.initialization_seed),
            _PLANNER_INITIALIZATION_DOMAIN,
        )

    def _initialize_u0_source(self) -> HiddenPrototypeTwoAgentState:
        return self.u0.initialize(HIDDEN_INFERRED_FULL)

    @staticmethod
    def _cache_base_actions(state: PrototypeFactorizedPartnerPlannerState) -> Array:
        return jnp.stack((state.agent_0.cache.base_action, state.agent_1.cache.base_action)).astype(
            jnp.int32
        )

    @staticmethod
    def _cache_effective_actions(state: PrototypeFactorizedPartnerPlannerState) -> Array:
        return jnp.stack(
            (state.agent_0.cache.effective_action, state.agent_1.cache.effective_action)
        ).astype(jnp.int32)

    def _normalize_u0_auxiliary(
        self,
        state: HiddenPrototypeTwoAgentState,
        planner_state: PrototypeFactorizedPartnerPlannerState,
        agent_0: Any,
        agent_1: Any,
    ) -> HiddenPrototypeTwoAgentState:
        base_actions = self._cache_base_actions(planner_state)
        effective_actions = self._cache_effective_actions(planner_state)
        changed = effective_actions != base_actions
        # U0's compatibility availability flag is normalized to the planner
        # authority bit so U0's own state validator remains exact.  It is never
        # reported as a memory-retrieval observation in this wrapper; the actual
        # current-event memory diagnostics remain in the U0 trace.
        return state.replace(
            agent_0=agent_0,
            agent_1=agent_1,
            counterfactual_base_actions=base_actions,
            prior_memory_retrieval_available=changed,
            prior_memory_action_changed=changed,
        )

    def _aux_cache_binding_valid(
        self,
        u0_state: HiddenPrototypeTwoAgentState,
        planner_state: PrototypeFactorizedPartnerPlannerState,
    ) -> Array:
        actions = jnp.stack(
            (u0_state.agent_0.current_action, u0_state.agent_1.current_action)
        ).astype(jnp.int32)
        base = self._cache_base_actions(planner_state)
        effective = self._cache_effective_actions(planner_state)
        changed = effective != base
        return (
            jnp.array_equal(actions, effective)
            & jnp.array_equal(u0_state.counterfactual_base_actions, base)
            & jnp.array_equal(u0_state.prior_memory_action_changed, changed)
            & jnp.array_equal(u0_state.prior_memory_retrieval_available, changed)
        )

    def state_is_valid(
        self,
        state: HiddenPrototypeFactorizedPartnerPlanningState,
    ) -> Array:
        planner_valid = self.planner.authenticate_pair(
            state.planner,
            state.u0.agent_0,
            state.u0.agent_1,
        )
        return (
            self.u0._state_valid(state.u0)
            & jnp.all(planner_valid)
            & self._aux_cache_binding_valid(state.u0, state.planner)
        )

    def _compose_initial(
        self,
        u0_source: HiddenPrototypeTwoAgentState,
    ) -> HiddenPrototypeFactorizedPartnerPlanningState:
        planner_source = self.planner.init(self._planner_initialization_key())
        prepared = self.planner.prepare_pair(
            planner_source,
            u0_source.agent_0,
            u0_source.agent_1,
            self._safety_masks,
        )
        if not bool(prepared.diagnostics.pair_committed):
            raise RuntimeError("initial factorized planner preparation rejected")
        normalized = self._normalize_u0_auxiliary(
            u0_source,
            prepared.state,
            prepared.prototype_agent_0,
            prepared.prototype_agent_1,
        )
        composite = HiddenPrototypeFactorizedPartnerPlanningState(
            u0=normalized,
            planner=prepared.state,
        )
        if not bool(self.state_is_valid(composite)):
            raise RuntimeError("initialized U1 composite state is invalid")
        return composite

    def initialize(self) -> HiddenPrototypeFactorizedPartnerPlanningState:
        """Initialize one arm from the exact consumed U0 root and fixed planner domain."""

        return self._compose_initial(self._initialize_u0_source())

    def step(
        self,
        state: HiddenPrototypeFactorizedPartnerPlanningState,
        event_index: Array,
        *,
        force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
    ) -> HiddenPrototypeFactorizedPartnerPlanningStepResult:
        """Propose U0 and planner children, then carry the complete composite once."""

        if (
            getattr(event_index, "shape", None) != ()
            or getattr(event_index, "dtype", None) != jnp.dtype(jnp.int32)
        ):
            raise TypeError("event_index must be an exact scalar int32 array")
        index = jnp.asarray(event_index)
        force_reject = jnp.asarray(force_outer_rejection, dtype=jnp.bool_)
        if force_reject.shape != ():
            raise ValueError("force_outer_rejection must be scalar")

        # An int32 provenance index can represent only the zero-high-word part
        # of the exact uint64 clock.  Check both words as well as the int32
        # telemetry so a saturated telemetry value can never alias a live step.
        expected_source_step_words = jnp.stack(
            (
                jnp.asarray(0, dtype=jnp.uint32),
                index.astype(jnp.uint32),
            )
        )
        event_index_matches_source_clock = (
            (index >= 0)
            & (state.u0.environment.step_count == index)
            & jnp.array_equal(
                state.u0.environment.step_words,
                expected_source_step_words,
            )
        )

        source_planner_cache_valid = self.planner.authenticate_pair(
            state.planner,
            state.u0.agent_0,
            state.u0.agent_1,
        )
        source_u0_valid = self.u0._state_valid(state.u0)
        source_aux_valid = self._aux_cache_binding_valid(state.u0, state.planner)
        source_composite_valid = (
            source_u0_valid & jnp.all(source_planner_cache_valid) & source_aux_valid
        )

        u0_result = self.u0.step(
            state.u0,
            index,
            route_inference=jnp.asarray(True, dtype=jnp.bool_),
        )
        actual_actions = u0_result.trace.actions.astype(jnp.int32)
        base_actions = u0_result.trace.counterfactual_base_actions.astype(jnp.int32)
        expected_cube = jnp.stack(
            (
                actual_actions,
                jnp.stack((base_actions[0], actual_actions[1])),
                jnp.stack((actual_actions[0], base_actions[1])),
                base_actions,
            )
        ).astype(jnp.int32)
        planner_cube_binding_valid = (
            jnp.array_equal(u0_result.trace.joint_primitive_actions, expected_cube)
            & jnp.array_equal(actual_actions, self._cache_effective_actions(state.planner))
            & jnp.array_equal(base_actions, self._cache_base_actions(state.planner))
        )

        actual_rewards = u0_result.trace.joint_rewards[0].astype(jnp.float32)
        post_memory = (u0_result.state.agent_0, u0_result.state.agent_1)
        post_memory_observations = jnp.stack(
            (
                post_memory[0].current_raw_observation,
                post_memory[1].current_raw_observation,
            )
        ).astype(jnp.float32)
        continuation = jnp.asarray(1.0, dtype=jnp.float32)
        planner_result = self.planner.completed_transition(
            state.planner,
            state.u0.agent_0,
            state.u0.agent_1,
            post_memory[0],
            post_memory[1],
            actual_actions,
            actual_rewards,
            post_memory_observations,
            continuation,
            self._safety_masks,
        )
        expected_grounded_targets = jnp.concatenate(
            (
                post_memory_observations,
                actual_rewards[:, None],
                jnp.full((_N_AGENTS, 1), continuation, dtype=jnp.float32),
            ),
            axis=1,
        )
        expected_partner_actions = jnp.stack((actual_actions[1], actual_actions[0])).astype(
            jnp.int32
        )
        expected_grounded_joint_action_indices = (
            actual_actions * _N_ACTIONS + expected_partner_actions
        ).astype(jnp.int32)
        local_post_memory_binding_valid = (
            jnp.array_equal(planner_result.diagnostics.executed_actions, actual_actions)
            & jnp.array_equal(
                planner_result.diagnostics.observed_partner_actions,
                expected_partner_actions,
            )
            & jnp.array_equal(
                planner_result.diagnostics.grounded_targets,
                expected_grounded_targets,
            )
            & jnp.array_equal(
                planner_result.diagnostics.grounded_joint_action_indices,
                expected_grounded_joint_action_indices,
            )
            & jnp.all(planner_result.diagnostics.source_cache_valid)
            & jnp.all(planner_result.diagnostics.next_observations_match)
            & jnp.all(planner_result.diagnostics.candidate_clock_aligned)
            & jnp.all(planner_result.diagnostics.candidate_generation_aligned)
        )

        candidate_u0 = self._normalize_u0_auxiliary(
            u0_result.state,
            planner_result.state,
            planner_result.prototype_agent_0,
            planner_result.prototype_agent_1,
        )
        candidate = HiddenPrototypeFactorizedPartnerPlanningState(
            u0=candidate_u0,
            planner=planner_result.state,
        )
        candidate_u0_valid = self.u0._state_valid(candidate.u0)
        candidate_planner_cache_valid = self.planner.authenticate_pair(
            candidate.planner,
            candidate.u0.agent_0,
            candidate.u0.agent_1,
        )
        candidate_aux_valid = self._aux_cache_binding_valid(
            candidate.u0,
            candidate.planner,
        )
        candidate_composite_valid = (
            candidate_u0_valid & jnp.all(candidate_planner_cache_valid) & candidate_aux_valid
        )
        no_oracle_consumed = jnp.all(~u0_result.trace.no_oracle_cue_consumed)
        all_true_mask_used = jnp.all(self._safety_masks)
        outer_commit = (
            source_composite_valid
            & u0_result.trace.outer_transaction_committed
            & planner_result.diagnostics.transaction_committed
            & planner_cube_binding_valid
            & local_post_memory_binding_valid
            & candidate_composite_valid
            & no_oracle_consumed
            & all_true_mask_used
            & event_index_matches_source_clock
            & (~force_reject)
        )
        committed = cast(
            HiddenPrototypeFactorizedPartnerPlanningState,
            jax.lax.cond(outer_commit, lambda _: candidate, lambda _: state, operand=None),
        )

        planner_changed = actual_actions != base_actions
        planner_effects = u0_result.trace.own_action_effects.astype(jnp.float32)
        planner_outcomes = jnp.where(
            planner_changed & (planner_effects > 0.0),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.where(
                planner_changed & (planner_effects < 0.0),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        next_base = self._cache_base_actions(planner_result.state)
        next_effective = self._cache_effective_actions(planner_result.state)
        trace = HiddenPrototypeFactorizedPartnerPlanningTrace(
            event_index=index,
            event_index_matches_source_clock=event_index_matches_source_clock,
            actual_actions=actual_actions,
            base_actions=base_actions,
            joint_primitive_actions=u0_result.trace.joint_primitive_actions,
            joint_rewards=u0_result.trace.joint_rewards,
            planner_action_changed=planner_changed,
            planner_reward_effects=planner_effects,
            planner_outcome_codes=planner_outcomes,
            next_base_actions=next_base,
            next_effective_actions=next_effective,
            next_planner_action_changed=next_effective != next_base,
            next_learned_partner_probabilities=(
                planner_result.diagnostics.next_prepare.learned_partner_probabilities
            ),
            next_applied_partner_probabilities=(
                planner_result.diagnostics.next_prepare.applied_partner_probabilities
            ),
            planner_executed_actions=planner_result.diagnostics.executed_actions,
            planner_observed_partner_actions=(
                planner_result.diagnostics.observed_partner_actions
            ),
            planner_post_memory_observations=post_memory_observations,
            planner_input_rewards=actual_rewards,
            planner_input_continuation=continuation,
            planner_grounded_targets=planner_result.diagnostics.grounded_targets,
            planner_grounded_joint_action_indices=(
                planner_result.diagnostics.grounded_joint_action_indices
            ),
            behavior_losses=planner_result.diagnostics.behavior_losses,
            behavior_update_applied=planner_result.diagnostics.behavior_update_applied,
            grounded_losses=planner_result.diagnostics.grounded_losses,
            grounded_update_applied=planner_result.diagnostics.grounded_update_applied,
            context_pre_slots=u0_result.trace.context_pre_slots,
            context_post_slots=u0_result.trace.context_post_slots,
            memory_query_before_write=u0_result.trace.memory_query_before_write,
            memory_wrote=u0_result.trace.memory_wrote,
            memory_evicted=u0_result.trace.memory_evicted,
            memory_retrieval_available=u0_result.trace.memory_retrieval_available,
            memory_action_changed=u0_result.trace.memory_action_changed,
            source_u0_valid=source_u0_valid,
            source_planner_cache_valid=source_planner_cache_valid,
            source_aux_cache_binding_valid=source_aux_valid,
            source_composite_valid=source_composite_valid,
            u0_candidate_committed=u0_result.trace.outer_transaction_committed,
            planner_candidate_committed=planner_result.diagnostics.transaction_committed,
            local_post_memory_binding_valid=local_post_memory_binding_valid,
            planner_cube_binding_valid=planner_cube_binding_valid,
            candidate_u0_valid=candidate_u0_valid,
            candidate_planner_cache_valid=candidate_planner_cache_valid,
            candidate_aux_cache_binding_valid=candidate_aux_valid,
            candidate_composite_valid=candidate_composite_valid,
            no_oracle_channel_consumed=~no_oracle_consumed,
            all_true_caller_mask_used=all_true_mask_used,
            all_true_caller_mask_is_safety_certification=jnp.asarray(False, dtype=jnp.bool_),
            forced_outer_rejection=force_reject,
            outer_transaction_committed=outer_commit,
        )
        return HiddenPrototypeFactorizedPartnerPlanningStepResult(
            state=committed,
            trace=trace,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def compiled_step(
        self,
        state: HiddenPrototypeFactorizedPartnerPlanningState,
        event_index: Array,
        *,
        force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
    ) -> HiddenPrototypeFactorizedPartnerPlanningStepResult:
        return self.step(
            state,
            event_index,
            force_outer_rejection=force_outer_rejection,
        )

    def resource_budget(
        self,
        state: HiddenPrototypeFactorizedPartnerPlanningState,
    ) -> dict[str, object]:
        u0_nbytes = u0_module._tree_nbytes(state.u0)
        planner = self.planner.resource_budget(state.planner)
        composite_nbytes = u0_module._tree_nbytes(state)
        decomposed = u0_nbytes + planner.measured_pair_nbytes
        return {
            "scope": RESOURCE_ACCOUNTING_SCOPE,
            "logical_fixed_allocation": True,
            "u0_persistent_state_nbytes": u0_nbytes,
            "planner_persistent_state_nbytes": planner.measured_pair_nbytes,
            "planner_resource_budget": planner.to_dict(),
            "composite_persistent_state_nbytes": composite_nbytes,
            "decomposed_persistent_state_nbytes": decomposed,
            "exact_decomposition": composite_nbytes == decomposed,
            "wrapper_extra_persistent_state_nbytes": 0,
            "compiler_workspace_bytes_included": False,
            "transient_diagnostics_bytes_included": False,
        }

    def work_budget(self, total_steps: int) -> dict[str, object]:
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be a positive exact integer")
        u0_work = u0_module._work(total_steps)
        initial = self.planner.standalone_prepare_work_budget().to_dict()
        per_event = self.planner.completed_transition_work_budget().to_dict()
        return {
            "scope": RESOURCE_ACCOUNTING_SCOPE,
            "events": total_steps,
            "environment_proposal_calls": 4 * total_steps,
            "environment_proposals_per_event": 4,
            "environment_proposals_not_eight": True,
            "u0": u0_work,
            "u0_initialization_calls": 1,
            "u0_initialization_random_draw_internals_counted_in_u1_scope": False,
            "planner_initialization_calls": 1,
            "planner_initialization_key_split_calls": 1,
            "planner_initial_grounded_uniform_draw_calls": 2,
            "planner_initial_behavior_keys_stored": 2,
            "planner_initial_prepare": initial,
            "planner_completed_transition_per_event": per_event,
            "planner_completed_transition_calls": total_steps,
            "wrapper_event_index_source_clock_bindings": total_steps,
            "wrapper_source_composite_authentications": total_steps,
            "wrapper_candidate_composite_authentications": total_steps,
            "wrapper_outer_atomic_decisions": total_steps,
            "additional_wrapper_planner_environment_proposals": 0,
            "additional_wrapper_planner_post_init_random_draws": 0,
            "additional_wrapper_planner_replay_updates": 0,
            "inherited_u0_policy_post_init_rng_present": True,
            "inherited_u0_rng_draw_count_in_scope": False,
            "threshold_evaluations": 0,
            "winner_selection_calls": 0,
            "writer_calls": 0,
            "checkpoint_save_calls": 0,
            "checkpoint_load_calls": 0,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _attach_report_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "report_sha256" in body:
        raise ValueError("unhashed report body already contains report_sha256")
    return cast(
        dict[str, object],
        _json_clone({**body, "report_sha256": _json_sha256(body)}),
    )


def _report_hash_reconstructs(report: Mapping[str, object]) -> bool:
    value = report.get("report_sha256")
    if type(value) is not str:
        return False
    body = dict(report)
    del body["report_sha256"]
    return value == _json_sha256(body)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_source_path(module: Any) -> Path:
    value = getattr(module, "__file__", None)
    if type(value) is not str:
        raise RuntimeError("a selected U1 source module has no exact file path")
    return Path(value).resolve()


def _selected_source_hashes() -> dict[str, str]:
    files = {"evaluation_module_sha256": _sha256_file(Path(__file__).resolve())}
    for label, module in _SELECTED_SOURCE_MODULES:
        if label in files:
            raise RuntimeError("selected U1 source labels are not unique")
        files[label] = _sha256_file(_module_source_path(module))
    return files


_IMPORT_TIME_SELECTED_SOURCE_HASHES: Final = tuple(sorted(_selected_source_hashes().items()))


def _bound_source_manifest(*, stage: str) -> dict[str, str]:
    if type(stage) is not str or not stage:
        raise ValueError("source-binding stage must be a nonempty exact string")
    current = tuple(sorted(_selected_source_hashes().items()))
    if current != _IMPORT_TIME_SELECTED_SOURCE_HASHES:
        raise RuntimeError(
            f"selected U1 source files differ from their import-time bytes at {stage}"
        )
    files = dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES)
    return {**files, "manifest_sha256": _json_sha256(files)}


def _runtime_identity() -> dict[str, object]:
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_device_count": len(devices),
        "jax_device_kinds": [device.device_kind for device in devices],
    }


def _tree_sha256(tree: object) -> str:
    digest = hashlib.sha256()
    leaves, structure = jax.tree.flatten(tree)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        digest.update(
            _canonical_json({"shape": list(array.shape), "dtype": str(array.dtype)}).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: Any) -> np.ndarray[Any, Any]:
    return np.asarray(jax.device_get(value))


def _outcome_name(code: int) -> str:
    if code == -1:
        return "harm"
    if code == 0:
        return "neutral"
    if code == 1:
        return "benefit"
    raise ValueError("planner outcome code is outside {-1, 0, 1}")


def _phase_for_index(index: int, segment_length: int) -> str:
    if index < segment_length:
        return "A1"
    if index < 2 * segment_length:
        return "B"
    return "A2"


def _selected_event_payload(
    trace: HiddenPrototypeFactorizedPartnerPlanningTrace,
    *,
    event_index: int,
    segment_length: int,
) -> dict[str, object]:
    actual_rewards = _array(trace.joint_rewards)[0].astype(np.float64)
    outcome_codes = _array(trace.planner_outcome_codes).astype(np.int64)
    context_pre = _array(trace.context_pre_slots).astype(np.int64)
    context_post = _array(trace.context_post_slots).astype(np.int64)
    return {
        "event_index": event_index,
        "event_index_matches_source_clock": bool(trace.event_index_matches_source_clock),
        "phase": _phase_for_index(event_index, segment_length),
        "actual_actions": [int(value) for value in _array(trace.actual_actions)],
        "base_actions": [int(value) for value in _array(trace.base_actions)],
        "joint_primitive_actions": [
            [int(value) for value in row] for row in _array(trace.joint_primitive_actions)
        ],
        "joint_rewards": [[float(value) for value in row] for row in _array(trace.joint_rewards)],
        "actual_rewards": [float(value) for value in actual_rewards],
        "actual_mean_reward": float(np.mean(actual_rewards)),
        "planner_action_changed": [bool(value) for value in _array(trace.planner_action_changed)],
        "planner_reward_effects": [float(value) for value in _array(trace.planner_reward_effects)],
        "planner_outcomes": [_outcome_name(int(value)) for value in outcome_codes],
        "planner_inputs": {
            "executed_actions": [
                int(value) for value in _array(trace.planner_executed_actions)
            ],
            "observed_partner_actions": [
                int(value) for value in _array(trace.planner_observed_partner_actions)
            ],
            "post_memory_observations": [
                [float(value) for value in row]
                for row in _array(trace.planner_post_memory_observations)
            ],
            "rewards": [float(value) for value in _array(trace.planner_input_rewards)],
            "continuation": float(trace.planner_input_continuation),
            "grounded_targets": [
                [float(value) for value in row]
                for row in _array(trace.planner_grounded_targets)
            ],
            "grounded_joint_action_indices": [
                int(value) for value in _array(trace.planner_grounded_joint_action_indices)
            ],
        },
        "behavior_losses": [float(value) for value in _array(trace.behavior_losses)],
        "grounded_losses": [float(value) for value in _array(trace.grounded_losses)],
        "context_switches": [
            bool(left != right) for left, right in zip(context_pre, context_post, strict=True)
        ],
        "context_post_slots": [int(value) for value in context_post],
        "memory": {
            "query_before_write": [
                bool(value) for value in _array(trace.memory_query_before_write)
            ],
            "wrote": [bool(value) for value in _array(trace.memory_wrote)],
            "evicted": [bool(value) for value in _array(trace.memory_evicted)],
            "retrieval_available": [
                bool(value) for value in _array(trace.memory_retrieval_available)
            ],
            "changed_post_memory_action": [
                bool(value) for value in _array(trace.memory_action_changed)
            ],
            "same_event_reward_effect_reported": False,
        },
        "commits": {
            "u0_candidate": bool(trace.u0_candidate_committed),
            "planner_candidate": bool(trace.planner_candidate_committed),
            "outer_composite": bool(trace.outer_transaction_committed),
        },
        "bindings": {
            "event_index_source_clock": bool(trace.event_index_matches_source_clock),
            "source_u0": bool(trace.source_u0_valid),
            "source_planner": bool(np.all(_array(trace.source_planner_cache_valid))),
            "source_aux": bool(trace.source_aux_cache_binding_valid),
            "planner_cube": bool(trace.planner_cube_binding_valid),
            "local_post_memory": bool(trace.local_post_memory_binding_valid),
            "candidate_u0": bool(trace.candidate_u0_valid),
            "candidate_planner": bool(np.all(_array(trace.candidate_planner_cache_valid))),
            "candidate_aux": bool(trace.candidate_aux_cache_binding_valid),
            "no_oracle": not bool(trace.no_oracle_channel_consumed),
        },
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _metrics_for_events(
    events: Sequence[Mapping[str, Any]],
    *,
    window: int,
) -> dict[str, object]:
    rewards = [cast(float, event["actual_mean_reward"]) for event in events]
    per_agent_rewards: list[list[float]] = [[], []]
    planner_changes = [0, 0]
    planner_effects: list[list[float]] = [[], []]
    outcome_counts: list[dict[str, int]] = [
        dict.fromkeys(_OUTCOME_NAMES, 0),
        dict.fromkeys(_OUTCOME_NAMES, 0),
    ]
    behavior_losses: list[list[float]] = [[], []]
    grounded_losses: list[list[float]] = [[], []]
    context_switches = [0, 0]
    context_slots: list[set[int]] = [set(), set()]
    memory_counts = {
        name: [0, 0]
        for name in (
            "query_before_write",
            "wrote",
            "evicted",
            "retrieval_available",
            "changed_post_memory_action",
        )
    }
    commits = {"u0_candidate": 0, "planner_candidate": 0, "outer_composite": 0}
    all_bindings_valid = True
    for event in events:
        for agent_index in range(_N_AGENTS):
            per_agent_rewards[agent_index].append(event["actual_rewards"][agent_index])
            planner_changes[agent_index] += int(event["planner_action_changed"][agent_index])
            planner_effects[agent_index].append(event["planner_reward_effects"][agent_index])
            outcome = cast(str, event["planner_outcomes"][agent_index])
            outcome_counts[agent_index][outcome] += 1
            behavior_losses[agent_index].append(event["behavior_losses"][agent_index])
            grounded_losses[agent_index].append(event["grounded_losses"][agent_index])
            context_switches[agent_index] += int(event["context_switches"][agent_index])
            context_slots[agent_index].add(event["context_post_slots"][agent_index])
            memory = cast(Mapping[str, Any], event["memory"])
            for name in memory_counts:
                memory_counts[name][agent_index] += int(memory[name][agent_index])
        for name in commits:
            commits[name] += int(cast(Mapping[str, bool], event["commits"])[name])
        all_bindings_valid &= all(cast(Mapping[str, bool], event["bindings"]).values())
    return {
        "event_count": len(events),
        "actual_mean_reward": _mean(rewards),
        "actual_early_mean_reward": _mean(rewards[:window]),
        "actual_tail_mean_reward": _mean(rewards[-window:]),
        "per_agent_actual_mean_reward": [_mean(values) for values in per_agent_rewards],
        "planner_action_changes": planner_changes,
        "planner_mean_unilateral_reward_effect": [_mean(values) for values in planner_effects],
        "planner_outcome_counts": outcome_counts,
        "mean_behavior_loss": [_mean(values) for values in behavior_losses],
        "mean_grounded_loss": [_mean(values) for values in grounded_losses],
        "context_switches": context_switches,
        "context_slots_used": [sorted(values) for values in context_slots],
        "memory_counts": memory_counts,
        "commit_counts": commits,
        "all_bindings_valid": all_bindings_valid,
        "same_event_memory_reward_effect_reported": False,
    }


def _metrics(
    events: Sequence[Mapping[str, Any]],
    protocol: HiddenPrototypeFactorizedPartnerPlanningProtocol,
) -> dict[str, object]:
    phases = {
        phase: [event for event in events if event["phase"] == phase] for phase in ("A1", "B", "A2")
    }
    return {
        "phase": {
            phase: _metrics_for_events(values, window=protocol.metric_window)
            for phase, values in phases.items()
        },
        "lifetime": _metrics_for_events(events, window=protocol.metric_window),
    }


def _run_arm(
    protocol: HiddenPrototypeFactorizedPartnerPlanningProtocol,
    arm: FactorizedPlanningArm,
) -> dict[str, object]:
    evaluator = HiddenPrototypeFactorizedPartnerPlanningEvaluator(arm.name)
    u0_source = evaluator._initialize_u0_source()
    u0_preplanner_sha256 = _tree_sha256(u0_source)
    state = evaluator._compose_initial(u0_source)
    initial_state_sha256 = _tree_sha256(state)
    planner_model_genesis_sha256 = _tree_sha256(
        (
            state.planner.agent_0.behavior,
            state.planner.agent_0.grounded,
            state.planner.agent_1.behavior,
            state.planner.agent_1.grounded,
        )
    )
    initial_resources = evaluator.resource_budget(state)
    planner_bytes = cast(int, initial_resources["planner_persistent_state_nbytes"])
    if protocol.total_steps == _FULL_TOTAL_STEPS and (
        planner_bytes != _EXPECTED_FULL_PLANNER_PAIR_NBYTES
    ):
        raise RuntimeError("canonical U1 planner pair byte formula drifted")

    events: list[Mapping[str, Any]] = []
    chain = _EVENT_CHAIN_GENESIS
    first_event_hash: str | None = None
    phase_tips: dict[str, str] = {}
    boundary_bytes = [cast(int, initial_resources["composite_persistent_state_nbytes"])]
    for event_index in range(protocol.total_steps):
        result = evaluator.compiled_step(
            state,
            jnp.asarray(event_index, dtype=jnp.int32),
        )
        if not bool(result.trace.outer_transaction_committed):
            raise RuntimeError(f"U1 composite event {event_index} rejected")
        payload = _selected_event_payload(
            result.trace,
            event_index=event_index,
            segment_length=protocol.segment_length,
        )
        chain = hashlib.sha256(
            bytes.fromhex(chain) + _canonical_json(payload).encode("ascii")
        ).hexdigest()
        if first_event_hash is None:
            first_event_hash = chain
        events.append(payload)
        state = result.state
        if (event_index + 1) % protocol.segment_length == 0:
            phase = _phase_for_index(event_index, protocol.segment_length)
            phase_tips[phase] = chain
            boundary_bytes.append(
                cast(
                    int,
                    evaluator.resource_budget(state)["composite_persistent_state_nbytes"],
                )
            )
    final_resources = evaluator.resource_budget(state)
    if boundary_bytes != [boundary_bytes[0]] * 4:
        raise RuntimeError("fixed composite persistent allocation changed at a phase boundary")
    if (
        initial_resources["composite_persistent_state_nbytes"]
        != final_resources["composite_persistent_state_nbytes"]
    ):
        raise RuntimeError("fixed composite persistent allocation changed during the life")
    return {
        "arm": arm.name,
        "arm_definition": arm.to_config(),
        "planner_config": evaluator.planner.config.to_config(),
        "u0_preplanner_genesis_sha256": u0_preplanner_sha256,
        "planner_model_genesis_sha256": planner_model_genesis_sha256,
        "initial_composite_state_sha256": initial_state_sha256,
        "final_composite_state_sha256": _tree_sha256(state),
        "event_hash_chain": {
            "algorithm": "sha256(previous_digest_bytes || canonical_event_json_ascii)",
            "genesis_sha256": _EVENT_CHAIN_GENESIS,
            "event_count": protocol.total_steps,
            "first_event_sha256": first_event_hash,
            "phase_tip_sha256": phase_tips,
            "final_event_sha256": chain,
            "event_payloads_retained_in_report": False,
        },
        "metrics": _metrics(events, protocol),
        "resources": {
            "initial": initial_resources,
            "final": final_resources,
            "phase_boundary_composite_nbytes": boundary_bytes,
        },
        "work": evaluator.work_budget(protocol.total_steps),
        "final_environment_step_words": [
            int(value) for value in _array(state.u0.environment.step_words)
        ],
        "checkpoint_resume_claimed": False,
        "same_event_memory_reward_effect_reported": False,
    }


LIMITATIONS: Final = (
    "one already-consumed development root with no statistical inference",
    "a synthetic two-agent A-B-A world is not broad transfer or physical embodiment",
    "the planner is a one-step expected-immediate-reward adapter, not long-horizon search",
    "the all-true caller mask is not a physical safety certification",
    "U0's inherited epsilon policy uses post-init RNG; only additional U1 draws are zero",
    "the source manifest binds selected direct files, not a transitive source closure",
    "runtime identity binds selected fields, not an environment or compiler closure",
    "base/fallback provenance remains the planner core's explicit nonclaim",
    "logical resource/work accounting excludes compiler workspaces, FLOPs, and latency",
    "no thresholds, search, writer, artifact, checkpoint, evidence, or promotion path",
)


def _build_report() -> dict[str, object]:
    static_contract_errors = validate_static_contract()
    if static_contract_errors:
        raise RuntimeError(
            "U1 static contract validation failed: " + "; ".join(static_contract_errors)
        )
    source_manifest = _bound_source_manifest(stage="pre-run")
    runtime_identity = _runtime_identity()
    protocol = HiddenPrototypeFactorizedPartnerPlanningProtocol()
    runs = [_run_arm(protocol, arm) for arm in FACTORIZED_PLANNING_ARMS]
    if _bound_source_manifest(stage="post-run") != source_manifest:
        raise RuntimeError("selected U1 source files changed during the panel")
    if _runtime_identity() != runtime_identity:
        raise RuntimeError("selected U1 runtime identity changed during the panel")
    if len({run["u0_preplanner_genesis_sha256"] for run in runs}) != 1:
        raise RuntimeError("U1 arms do not share a bit-identical pre-planner U0 genesis")
    if len({run["planner_model_genesis_sha256"] for run in runs}) != 1:
        raise RuntimeError("U1 arms do not share bit-identical planner model initialization")
    if not all(run["work"] == runs[0]["work"] for run in runs[1:]):
        raise RuntimeError("U1 arms do not have identical declared work")
    initial_bytes: set[int] = set()
    for run in runs:
        resources = cast(Mapping[str, Any], run["resources"])
        initial = cast(Mapping[str, Any], resources["initial"])
        initial_bytes.add(cast(int, initial["composite_persistent_state_nbytes"]))
    if len(initial_bytes) != 1:
        raise RuntimeError("U1 arms do not have identical persistent state geometry")

    body: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "acceptance_status": ACCEPTANCE_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "accepted_scientific_evidence": False,
        "output_writes_allowed": False,
        "writer_available": False,
        "artifact_bytes_written": 0,
        "winner_selected": False,
        "search_performed": False,
        "thresholds_evaluated": False,
        "checkpoint_resume_claimed": False,
        "protocol": protocol.to_config(),
        "protocol_sha256": _json_sha256(protocol.to_config()),
        "source_manifest": source_manifest,
        "source_manifest_scope": _SOURCE_MANIFEST_SCOPE,
        "source_hash_binding": {
            "captured_at_module_import": True,
            "pre_run_disk_equality_required": True,
            "post_run_disk_equality_required": True,
        },
        "transitive_source_closure_claimed": False,
        "runtime_identity": runtime_identity,
        "runtime_identity_scope": RUNTIME_IDENTITY_SCOPE,
        "runtime_identity_bound_by_validation": True,
        "runtime_environment_or_compiler_closure_claimed": False,
        "resource_accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "arm_order": [arm.name for arm in FACTORIZED_PLANNING_ARMS],
        "arm_definitions": [arm.to_config() for arm in FACTORIZED_PLANNING_ARMS],
        "runs": runs,
        "matched_comparison": {
            "same_consumed_u0_root": True,
            "bit_identical_preplanner_u0_genesis": True,
            "bit_identical_behavior_and_grounded_model_initialization": True,
            "persistent_state_geometry_matched": True,
            "logical_work_matched": True,
            "environment_proposals_per_event": 4,
            "behavioral_experience_matching_claimed": False,
            "varying_planner_fields": [
                "planning_enabled",
                "uniform_partner_belief",
            ],
        },
        "nonclaims": {
            "underlying_post_memory_transition_binding_claimed": False,
            "wrapper_constructs_post_memory_candidate_locally": True,
            "underlying_base_fallback_source_binding_claimed": False,
            "same_event_memory_reward_effect_reported": False,
            "all_true_mask_is_safety_certification": False,
            "additional_u1_post_init_rng_draws": 0,
            "inherited_u0_post_init_rng_present": True,
            "scientific_evidence": "not-assessed",
            "Alberta_Plan_completion": "not-assessed",
        },
        "limitations": list(LIMITATIONS),
    }
    report = _attach_report_hash(body)
    if not _report_hash_reconstructs(report):
        raise RuntimeError("U1 report hash does not reconstruct from its complete body")
    return report


_FULL_PANEL_ATTEMPT = _ProcessAttemptLatch(lambda: _canonical_json(_build_report()))


def run_hidden_prototype_factorized_partner_planning_development() -> dict[str, object]:
    """Run the sole full panel at most once per process and return it in memory."""

    return cast(dict[str, object], json.loads(_FULL_PANEL_ATTEMPT.get()))


def validate_static_contract() -> tuple[str, ...]:
    """Audit the frozen declaration without executing JAX learner/environment steps."""

    errors: list[str] = []
    protocol = HiddenPrototypeFactorizedPartnerPlanningProtocol()
    if protocol.segment_length != _FULL_SEGMENT_LENGTH:
        errors.append("canonical segment length is not 512")
    if protocol.total_steps != _FULL_TOTAL_STEPS:
        errors.append("canonical A-B-A life is not 1,536 events")
    base = protocol.u0_protocol.prototype_protocol
    if (base.base_observation_dim, base.active_pair_slots) != (
        _FULL_OBSERVATION_DIM,
        4,
    ):
        errors.append("canonical U0 observation/feature geometry drifted")
    if base.base_observation_dim + base.active_pair_slots != _FULL_REPRESENTATION_DIM:
        errors.append("canonical U0 Prototype representation width drifted")
    if tuple(arm.name for arm in FACTORIZED_PLANNING_ARMS) != (
        LEARNED_PLANNING_ENABLED,
        UNIFORM_PLANNING_ENABLED,
        LEARNED_PLANNING_DISABLED,
    ):
        errors.append("factorized planning arm order drifted")
    expected_settings = ((True, False), (True, True), (False, False))
    observed_settings = tuple(
        (arm.planning_enabled, arm.uniform_partner_belief) for arm in FACTORIZED_PLANNING_ARMS
    )
    if observed_settings != expected_settings:
        errors.append("factorized planning intervention fields drifted")
    if UNDERLYING_POST_MEMORY_TRANSITION_BINDING_CLAIMED:
        errors.append("underlying post-memory transition binding became claimed")
    if UNDERLYING_BASE_FALLBACK_SOURCE_BINDING_CLAIMED:
        errors.append("underlying base/fallback source binding became claimed")
    if ALL_TRUE_CALLER_MASK_IS_SAFETY_CERTIFICATION:
        errors.append("all-true caller mask became a safety certification")
    if not WRAPPER_LOCAL_POST_MEMORY_CONSTRUCTION_BOUND:
        errors.append("wrapper-local post-memory construction became unbound")
    if (
        ADDITIONAL_WRAPPER_PLANNER_POST_INIT_RANDOM_DRAWS_PER_EVENT != 0
        or ADDITIONAL_WRAPPER_PLANNER_REPLAY_UPDATES_PER_EVENT != 0
    ):
        errors.append("U1 acquired post-init random draws or replay")
    if not INHERITED_U0_POLICY_POST_INIT_RNG_PRESENT:
        errors.append("inherited U0 policy RNG was incorrectly removed from scope notes")
    if OUTPUT_WRITES_ALLOWED or SCIENTIFIC_PROMOTION_ALLOWED or EVIDENCE_AUTHORIZED:
        errors.append("a writer, evidence, or promotion path became available")
    if ARTIFACT_BYTES_WRITTEN != 0 or CHECKPOINT_RESUME_CLAIMED or THRESHOLDS_OR_WINNER_SELECTION:
        errors.append("an artifact, checkpoint, threshold, or winner path became claimed")
    return tuple(errors)


__all__ = [
    "ACCEPTANCE_STATUS",
    "ADDITIONAL_WRAPPER_PLANNER_POST_INIT_RANDOM_DRAWS_PER_EVENT",
    "ADDITIONAL_WRAPPER_PLANNER_REPLAY_UPDATES_PER_EVENT",
    "ALL_TRUE_CALLER_MASK_IS_SAFETY_CERTIFICATION",
    "ARTIFACT_BYTES_WRITTEN",
    "CHECKPOINT_RESUME_CLAIMED",
    "DEVELOPMENT_ONLY",
    "EAGER_JIT_DISCRETE_LEAVES_EXACT",
    "EAGER_JIT_FLOAT_ATOL",
    "EAGER_JIT_FLOAT_RTOL",
    "EVIDENCE_AUTHORIZED",
    "FACTORIZED_PLANNING_ARMS",
    "HiddenPrototypeFactorizedPartnerPlanningEvaluator",
    "HiddenPrototypeFactorizedPartnerPlanningProtocol",
    "HiddenPrototypeFactorizedPartnerPlanningState",
    "HiddenPrototypeFactorizedPartnerPlanningStepResult",
    "HiddenPrototypeFactorizedPartnerPlanningTrace",
    "INHERITED_U0_POLICY_POST_INIT_RNG_PRESENT",
    "LEARNED_PLANNING_DISABLED",
    "LEARNED_PLANNING_ENABLED",
    "OUTPUT_WRITES_ALLOWED",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_OR_WINNER_SELECTION",
    "UNDERLYING_BASE_FALLBACK_SOURCE_BINDING_CLAIMED",
    "UNDERLYING_POST_MEMORY_TRANSITION_BINDING_CLAIMED",
    "UNIFORM_PLANNING_ENABLED",
    "WRAPPER_LOCAL_POST_MEMORY_CONSTRUCTION_BOUND",
    "run_hidden_prototype_factorized_partner_planning_development",
    "validate_static_contract",
]
