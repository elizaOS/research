# mypy: disable-error-code="call-arg"
"""Development-only two-learner hidden-rule coadaptation feasibility probe.

Two independent differential-SARSA agents play one uninterrupted recurring
convention-game life.  In the primary arm neither learner observes the rule,
phase, boundary, schedule, or time-to-boundary.  Both act first.  Only then may
each learner consume its own action, the observed partner action, and the
common reward as ordinary experience.  A bounded :class:`ContextInference`
bank maps that completed experience to the feature used for the *next*
decision, so the current hidden rule cannot leak into the current action.

Three fixed, paired-initialization/schedule conditions are descriptive
diagnostics.  The environment seed remains a namespaced provenance field, but
this deterministic game does not consume environment randomness:

``hidden_inferred``
    Two independent context banks, each with two fixed slots.
``hidden_inference_unrouted``
    The same two context banks receive the same kind of post-action update,
    but their outputs are never routed into control: SARSA always receives
    ``[1, 0]``.  This matches both allocated state and context-update work
    while removing inference from the decision path.
``oracle_visible_ceiling``
    The true rule one-hot is visible before acting.  This is explicitly a
    diagnostic ceiling, never an autonomous-learning result.

The protocol has no artifact writer, output path, threshold search, held-out
seed surface, or promotion entry point.  Its two namespaced seeds are reserved
for development diagnostics only; each becomes consumed when executed.  A
miss is reported as a development rejection, not tuned into a pass.

The control learners, context learners, and recurring-game schedule all have
exact two-word lifetime clocks.  Signed int32 counts are saturating telemetry
only; the environment derives every phase and rule from its exact clock and
disarms atomically at the uint64 all-ones identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.average_reward import (
    DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES,
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
    measure_differential_sarsa_state_nbytes,
)
from alberta_framework.core.context_inference import (
    ContextInference,
    ContextInferenceConfig,
    ContextInferenceState,
    context_inference_clock_nbytes,
    measure_context_inference_state_nbytes,
)
from alberta_framework.streams.matrix_game import (
    CONVENTION_GAME_EXACT_CLOCK_NBYTES,
    ConventionGameConfig,
    ConventionGameState,
    RecurringConventionGame,
)

DEVELOPMENT_SCHEMA = "alberta.hidden-context-coadaptation-development.v1"
DEVELOPMENT_NAMESPACE = "hidden-rule-two-learner-context-inference-feasibility-v1"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
ENVIRONMENT_RANDOMNESS_CONSUMED = False

HIDDEN_INFERRED: Literal["hidden_inferred"] = "hidden_inferred"
HIDDEN_INFERENCE_UNROUTED: Literal["hidden_inference_unrouted"] = (
    "hidden_inference_unrouted"
)
ORACLE_VISIBLE_CEILING: Literal["oracle_visible_ceiling"] = "oracle_visible_ceiling"
Condition = Literal[
    "hidden_inferred",
    "hidden_inference_unrouted",
    "oracle_visible_ceiling",
]
CONDITIONS: tuple[Condition, ...] = (
    HIDDEN_INFERRED,
    HIDDEN_INFERENCE_UNROUTED,
    ORACLE_VISIBLE_CEILING,
)

LEARNER_PRE_ACTION_CHANNELS = ("context_inferred_from_prior_experience",)
LEARNER_POST_ACTION_CHANNELS = (
    "own_action",
    "observed_partner_action",
    "common_reward",
)
FORBIDDEN_LEARNER_CHANNELS = frozenset(
    {
        "current_rule",
        "rule_offset",
        "phase_index",
        "phase_boundary",
        "steps_to_boundary",
        "environment_step_count",
    }
)

N_ACTIONS = 2
OFFSETS = (0, 1)
PHASE_LENGTH = 400
NUM_PHASES = 6
NUM_STEPS = PHASE_LENGTH * NUM_PHASES
SUMMARY_WINDOW = 64
MAX_CONTEXTS = 2
CHANCE_REWARD = 1.0 / N_ACTIONS


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationSeed:
    """Namespaced provenance plus independent learner-initialization key.

    ``environment_seed`` is retained to bind the deterministic schedule's
    provenance.  The current game stores its key but performs no random draw.
    """

    namespace: str
    index: int
    environment_seed: int
    initialization_seed: int


def derive_development_seeds(
    namespace: str,
    count: int,
) -> tuple[HiddenContextCoadaptationSeed, ...]:
    """Derive stable uint32 key pairs without exposing a seed-search surface."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    records: list[HiddenContextCoadaptationSeed] = []
    for index in range(count):
        values: list[int] = []
        for role in ("environment", "initialization"):
            payload = f"{DEVELOPMENT_SCHEMA}|{namespace}|{index}|{role}".encode()
            values.append(
                int.from_bytes(
                    hashlib.sha256(payload).digest()[:4],
                    byteorder="big",
                    signed=False,
                )
            )
        records.append(
            HiddenContextCoadaptationSeed(
                namespace=namespace,
                index=index,
                environment_seed=values[0],
                initialization_seed=values[1],
            )
        )
    return tuple(records)


DEVELOPMENT_SEEDS = (
    HiddenContextCoadaptationSeed(
        namespace=DEVELOPMENT_NAMESPACE,
        index=0,
        environment_seed=1_889_022_034,
        initialization_seed=1_885_607_695,
    ),
    HiddenContextCoadaptationSeed(
        namespace=DEVELOPMENT_NAMESPACE,
        index=1,
        environment_seed=2_804_593_803,
        initialization_seed=306_485_284,
    ),
)


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationProtocol:
    """Frozen shape and causal-timing contract for this development probe."""

    n_actions: int = N_ACTIONS
    offsets: tuple[int, ...] = OFFSETS
    phase_length: int = PHASE_LENGTH
    num_phases: int = NUM_PHASES
    summary_window: int = SUMMARY_WINDOW
    max_contexts: int = MAX_CONTEXTS
    boundary_callbacks_used: bool = False
    resets_after_initialization: int = 0

    @property
    def num_steps(self) -> int:
        return self.phase_length * self.num_phases

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


PROTOCOL = HiddenContextCoadaptationProtocol()


_CONTROL_CONFIG = DifferentialSARSAConfig(
    n_actions=N_ACTIONS,
    q_step_size=0.15,
    average_reward_step_size=0.01,
    epsilon_start=0.05,
    epsilon_end=0.05,
    epsilon_decay_steps=0,
    use_bias=False,
)
_CONTEXT_CONFIG = ContextInferenceConfig(
    n_actions=N_ACTIONS,
    observation_dim=N_ACTIONS,
    max_contexts=MAX_CONTEXTS,
)
_CONTROL_AGENT = DifferentialSARSAAgent(_CONTROL_CONFIG)
_CONTEXT_INFERENCE = ContextInference(_CONTEXT_CONFIG)
_HIDDEN_GAME = RecurringConventionGame(
    ConventionGameConfig(
        n_actions=N_ACTIONS,
        phase_length=PHASE_LENGTH,
        offsets=OFFSETS,
        feature_mode="plain",
    )
)
_ORACLE_GAME = RecurringConventionGame(
    ConventionGameConfig(
        n_actions=N_ACTIONS,
        phase_length=PHASE_LENGTH,
        offsets=OFFSETS,
        feature_mode="context",
    )
)
_CONSTANT_HIDDEN_FEATURE = jnp.asarray((1.0, 0.0), dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class LearnerExperienceContract:
    """Condition-specific account of what can enter a pre-action decision."""

    pre_action_channels: tuple[str, ...]
    post_action_channels: tuple[str, ...]
    current_rule_visible: bool
    partner_action_available_only_after_acting: bool
    diagnostic_ceiling_only: bool
    inference_routed_to_control: bool
    boundary_callbacks_used: bool = False


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationResourceBudget:
    """Exact fixed-shape persistent JAX-array accounting for one condition."""

    control_feature_dim: int
    environment_nbytes: int
    per_agent_control_nbytes: int
    per_agent_context_nbytes: int
    joint_persistent_jax_array_nbytes: int
    per_agent_control_clock_nbytes: int
    per_agent_context_clock_nbytes: int
    per_agent_context_updates_per_transition: int
    environment_clock_nbytes: int
    joint_clock_nbytes: int
    max_context_slots_per_agent: int
    replay_capacity: int
    fixed_shape: bool
    controller_and_context_clocks_exact: bool
    environment_schedule_clock_exact: bool
    environment_schedule_max_steps: int
    environment_randomness_consumed: bool

    def to_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationRun:
    """One uninterrupted two-learner condition trace and final bounded state."""

    condition: Condition
    seed: HiddenContextCoadaptationSeed
    experience_contract: LearnerExperienceContract
    rewards: Array
    actions: Array
    rule_ids_for_evaluator_only: Array
    pre_action_contexts: Array
    post_action_contexts: Array
    contexts_in_use: Array
    controller_updates_applied: Array
    context_updates_applied: Array
    final_game_state: ConventionGameState
    final_controller_states: tuple[DifferentialSARSAState, DifferentialSARSAState]
    final_context_states: tuple[ContextInferenceState, ContextInferenceState] | None
    resource_budget: HiddenContextCoadaptationResourceBudget


@chex.dataclass(frozen=True)
class HiddenContextCoadaptationDiagnostics:
    """Threshold-free descriptive learning and recurrence readout."""

    phase_early_rewards: Array
    phase_tail_rewards: Array
    tail_context_modes: Array
    pre_action_switch_lags: Array
    distinct_rule_slots: Array
    recurrence_slot_reuse: Array
    recurrent_context_agreement: Array
    max_contexts_in_use: Array
    context_switch_counts: Array
    initial_within_phase_learning_gain: Array
    recurrent_early_reward: Array
    recurrent_tail_reward: Array
    overall_reward: Array
    both_controllers_changed: Array
    both_context_models_changed: Array


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationSmoke:
    """The first fixed seed across all three matched conditions."""

    runs: dict[Condition, HiddenContextCoadaptationRun]
    diagnostics: dict[Condition, HiddenContextCoadaptationDiagnostics]
    inferred_minus_unrouted_recurrent_early: float
    oracle_minus_inferred_recurrent_early: float
    inferred_contexts_distinguished_and_reused: bool
    inferred_control_above_analytic_chance: bool
    conclusion: str


@dataclasses.dataclass(frozen=True)
class HiddenContextCoadaptationPanel:
    """The fixed two-seed development panel; never a scientific artifact."""

    runs: dict[Condition, tuple[HiddenContextCoadaptationRun, ...]]
    diagnostics: dict[Condition, tuple[HiddenContextCoadaptationDiagnostics, ...]]
    inferred_minus_unrouted_recurrent_early_by_seed: tuple[float, ...]
    oracle_minus_inferred_recurrent_early_by_seed: tuple[float, ...]
    all_inferred_contexts_distinguished_and_reused: bool
    all_inferred_control_above_analytic_chance: bool
    conclusion: str


def _tree_array_nbytes(tree: object) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, Array)
    )


def _initial_controller(feature: Array, key: Array) -> DifferentialSARSAState:
    state = _CONTROL_AGENT.init(feature.shape[0], key)
    state = cast(
        DifferentialSARSAState,
        state.replace(birth_timestamp=0.0, uptime_s=0.0),  # type: ignore[attr-defined]
    )
    state, _ = _CONTROL_AGENT.start(state, feature)
    return cast(DifferentialSARSAState, state)


_InferredCarry = tuple[
    ConventionGameState,
    DifferentialSARSAState,
    DifferentialSARSAState,
    ContextInferenceState,
    ContextInferenceState,
]
_ControlCarry = tuple[
    ConventionGameState,
    DifferentialSARSAState,
    DifferentialSARSAState,
]
_Trace = tuple[Array, Array, Array, Array, Array, Array, Array]


def _run_inferred(
    environment_key: Array,
    initialization_key: Array,
) -> tuple[_InferredCarry, _Trace]:
    game_state = _HIDDEN_GAME.init(environment_key)
    context_0 = _CONTEXT_INFERENCE.init()
    context_1 = _CONTEXT_INFERENCE.init()
    state_0 = _initial_controller(
        _CONTEXT_INFERENCE.context_onehot(context_0),
        jr.fold_in(initialization_key, 0),
    )
    state_1 = _initial_controller(
        _CONTEXT_INFERENCE.context_onehot(context_1),
        jr.fold_in(initialization_key, 1),
    )

    def scan_step(carry: _InferredCarry, _: None) -> tuple[_InferredCarry, _Trace]:
        game, learner_0, learner_1, inference_0, inference_1 = carry
        action_0 = learner_0.last_action
        action_1 = learner_1.last_action

        # Both actions are irrevocably chosen before either context learner
        # receives partner action or reward.  No rule/schedule method is
        # called anywhere in this learner loop.
        reward, next_game = _HIDDEN_GAME.step(game, action_0, action_1)
        context_result_0 = _CONTEXT_INFERENCE.update_result(
            inference_0,
            jax.nn.one_hot(action_1, N_ACTIONS, dtype=jnp.float32),
            action_0,
            reward,
        )
        context_result_1 = _CONTEXT_INFERENCE.update_result(
            inference_1,
            jax.nn.one_hot(action_0, N_ACTIONS, dtype=jnp.float32),
            action_1,
            reward,
        )
        update_0 = _CONTROL_AGENT.update(
            learner_0,
            reward,
            context_result_0.context_onehot,
        )
        update_1 = _CONTROL_AGENT.update(
            learner_1,
            reward,
            context_result_1.context_onehot,
        )
        pre_contexts = jnp.stack(
            (inference_0.active_context, inference_1.active_context)
        ).astype(jnp.int32)
        post_contexts = jnp.stack(
            (
                context_result_0.state.active_context,
                context_result_1.state.active_context,
            )
        ).astype(jnp.int32)
        contexts_in_use = jnp.stack(
            (
                _CONTEXT_INFERENCE.num_contexts_in_use(context_result_0.state),
                _CONTEXT_INFERENCE.num_contexts_in_use(context_result_1.state),
            )
        )
        next_carry: _InferredCarry = (
            next_game,
            update_0.state,
            update_1.state,
            context_result_0.state,
            context_result_1.state,
        )
        trace: _Trace = (
            reward,
            jnp.stack((action_0, action_1)).astype(jnp.int32),
            pre_contexts,
            post_contexts,
            contexts_in_use,
            jnp.stack((update_0.update_applied, update_1.update_applied)),
            jnp.stack(
                (context_result_0.update_applied, context_result_1.update_applied)
            ),
        )
        return next_carry, trace

    return jax.lax.scan(
        scan_step,
        (game_state, state_0, state_1, context_0, context_1),
        xs=None,
        length=NUM_STEPS,
    )


def _run_inference_unrouted(
    environment_key: Array,
    initialization_key: Array,
) -> tuple[_InferredCarry, _Trace]:
    game_state = _HIDDEN_GAME.init(environment_key)
    context_0_initial = _CONTEXT_INFERENCE.init()
    context_1_initial = _CONTEXT_INFERENCE.init()
    state_0 = _initial_controller(
        _CONSTANT_HIDDEN_FEATURE,
        jr.fold_in(initialization_key, 0),
    )
    state_1 = _initial_controller(
        _CONSTANT_HIDDEN_FEATURE,
        jr.fold_in(initialization_key, 1),
    )

    def scan_step(carry: _InferredCarry, _: None) -> tuple[_InferredCarry, _Trace]:
        game, learner_0, learner_1, context_0, context_1 = carry
        action_0, action_1 = learner_0.last_action, learner_1.last_action
        reward, next_game = _HIDDEN_GAME.step(game, action_0, action_1)
        context_result_0 = _CONTEXT_INFERENCE.update_result(
            context_0,
            jax.nn.one_hot(action_1, N_ACTIONS, dtype=jnp.float32),
            action_0,
            reward,
        )
        context_result_1 = _CONTEXT_INFERENCE.update_result(
            context_1,
            jax.nn.one_hot(action_0, N_ACTIONS, dtype=jnp.float32),
            action_1,
            reward,
        )
        update_0 = _CONTROL_AGENT.update(
            learner_0,
            reward,
            _CONSTANT_HIDDEN_FEATURE,
        )
        update_1 = _CONTROL_AGENT.update(
            learner_1,
            reward,
            _CONSTANT_HIDDEN_FEATURE,
        )
        unavailable = jnp.full((2,), -1, dtype=jnp.int32)
        trace: _Trace = (
            reward,
            jnp.stack((action_0, action_1)).astype(jnp.int32),
            unavailable,
            unavailable,
            jnp.stack(
                (
                    _CONTEXT_INFERENCE.num_contexts_in_use(
                        context_result_0.state
                    ),
                    _CONTEXT_INFERENCE.num_contexts_in_use(
                        context_result_1.state
                    ),
                )
            ),
            jnp.stack((update_0.update_applied, update_1.update_applied)),
            jnp.stack(
                (context_result_0.update_applied, context_result_1.update_applied)
            ),
        )
        return (
            next_game,
            update_0.state,
            update_1.state,
            context_result_0.state,
            context_result_1.state,
        ), trace

    return jax.lax.scan(
        scan_step,
        (game_state, state_0, state_1, context_0_initial, context_1_initial),
        xs=None,
        length=NUM_STEPS,
    )


def _run_oracle_ceiling(
    environment_key: Array,
    initialization_key: Array,
) -> tuple[_ControlCarry, _Trace]:
    game_state = _ORACLE_GAME.init(environment_key)
    feature = _ORACLE_GAME.observe(game_state)
    state_0 = _initial_controller(feature, jr.fold_in(initialization_key, 0))
    state_1 = _initial_controller(feature, jr.fold_in(initialization_key, 1))

    def scan_step(carry: _ControlCarry, _: None) -> tuple[_ControlCarry, _Trace]:
        game, learner_0, learner_1 = carry
        action_0, action_1 = learner_0.last_action, learner_1.last_action
        reward, next_game = _ORACLE_GAME.step(game, action_0, action_1)
        # This deliberately visible rule feature is confined to the labeled
        # ceiling; it is absent from both hidden-condition functions above.
        next_feature = _ORACLE_GAME.observe(next_game)
        update_0 = _CONTROL_AGENT.update(learner_0, reward, next_feature)
        update_1 = _CONTROL_AGENT.update(learner_1, reward, next_feature)
        unavailable = jnp.full((2,), -1, dtype=jnp.int32)
        trace: _Trace = (
            reward,
            jnp.stack((action_0, action_1)).astype(jnp.int32),
            unavailable,
            unavailable,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.stack((update_0.update_applied, update_1.update_applied)),
            jnp.zeros((2,), dtype=jnp.bool_),
        )
        return (next_game, update_0.state, update_1.state), trace

    return jax.lax.scan(
        scan_step,
        (game_state, state_0, state_1),
        xs=None,
        length=NUM_STEPS,
    )


def _experience_contract(condition: Condition) -> LearnerExperienceContract:
    if condition == ORACLE_VISIBLE_CEILING:
        return LearnerExperienceContract(
            pre_action_channels=("current_rule_one_hot",),
            post_action_channels=LEARNER_POST_ACTION_CHANNELS,
            current_rule_visible=True,
            partner_action_available_only_after_acting=True,
            diagnostic_ceiling_only=True,
            inference_routed_to_control=False,
        )
    return LearnerExperienceContract(
        pre_action_channels=(
            LEARNER_PRE_ACTION_CHANNELS
            if condition == HIDDEN_INFERRED
            else ("constant_hidden_feature",)
        ),
        post_action_channels=LEARNER_POST_ACTION_CHANNELS,
        current_rule_visible=False,
        partner_action_available_only_after_acting=True,
        diagnostic_ceiling_only=False,
        inference_routed_to_control=condition == HIDDEN_INFERRED,
    )


def _resource_budget(
    game_state: ConventionGameState,
    controllers: tuple[DifferentialSARSAState, DifferentialSARSAState],
    contexts: tuple[ContextInferenceState, ContextInferenceState] | None,
) -> HiddenContextCoadaptationResourceBudget:
    environment_nbytes = _tree_array_nbytes(game_state)
    control_sizes = tuple(measure_differential_sarsa_state_nbytes(s) for s in controllers)
    if control_sizes[0] != control_sizes[1]:
        raise ValueError("independent control states disagree on their fixed resource shape")
    if contexts is None:
        context_nbytes = 0
        context_clock_nbytes = 0
        max_slots = 0
    else:
        context_sizes = tuple(measure_context_inference_state_nbytes(s) for s in contexts)
        if context_sizes[0] != context_sizes[1]:
            raise ValueError("independent context states disagree on resource shape")
        context_nbytes = context_sizes[0]
        context_clock_nbytes = context_inference_clock_nbytes(MAX_CONTEXTS)
        max_slots = MAX_CONTEXTS
    joint = environment_nbytes + 2 * control_sizes[0] + 2 * context_nbytes
    joint_clocks = (
        CONVENTION_GAME_EXACT_CLOCK_NBYTES
        + 2 * DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES
        + 2 * context_clock_nbytes
    )
    return HiddenContextCoadaptationResourceBudget(
        control_feature_dim=2,
        environment_nbytes=environment_nbytes,
        per_agent_control_nbytes=control_sizes[0],
        per_agent_context_nbytes=context_nbytes,
        joint_persistent_jax_array_nbytes=joint,
        per_agent_control_clock_nbytes=DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES,
        per_agent_context_clock_nbytes=context_clock_nbytes,
        per_agent_context_updates_per_transition=int(contexts is not None),
        environment_clock_nbytes=CONVENTION_GAME_EXACT_CLOCK_NBYTES,
        joint_clock_nbytes=joint_clocks,
        max_context_slots_per_agent=max_slots,
        replay_capacity=0,
        fixed_shape=True,
        controller_and_context_clocks_exact=True,
        environment_schedule_clock_exact=True,
        environment_schedule_max_steps=2**64 - 1,
        environment_randomness_consumed=ENVIRONMENT_RANDOMNESS_CONSUMED,
    )


def run_hidden_context_coadaptation(
    condition: Condition,
    seed: HiddenContextCoadaptationSeed,
) -> HiddenContextCoadaptationRun:
    """Execute one fixed causal condition without resets or output writes."""

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported hidden-context condition: {condition!r}")
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("only the two fixed development seeds may be executed")
    environment_key = jr.key(seed.environment_seed)
    initialization_key = jr.key(seed.initialization_seed)
    if condition == HIDDEN_INFERRED:
        final, trace = _run_inferred(environment_key, initialization_key)
        game, state_0, state_1, context_0, context_1 = final
        contexts: tuple[ContextInferenceState, ContextInferenceState] | None = (
            context_0,
            context_1,
        )
    elif condition == HIDDEN_INFERENCE_UNROUTED:
        final_unrouted, trace = _run_inference_unrouted(
            environment_key,
            initialization_key,
        )
        game, state_0, state_1, context_0, context_1 = final_unrouted
        contexts = (context_0, context_1)
    else:
        final_control, trace = _run_oracle_ceiling(environment_key, initialization_key)
        game, state_0, state_1 = final_control
        contexts = None
    (
        rewards,
        actions,
        pre_contexts,
        post_contexts,
        contexts_in_use,
        controller_updates,
        context_updates,
    ) = trace

    # Rule ids are constructed only after the learner scan has completed.
    # They are evaluator diagnostics and have no dataflow into any action.
    steps = jnp.arange(NUM_STEPS, dtype=jnp.int32)
    rule_ids = ((steps // PHASE_LENGTH) % len(OFFSETS)).astype(jnp.int32)
    controllers = (state_0, state_1)
    return HiddenContextCoadaptationRun(
        condition=condition,
        seed=seed,
        experience_contract=_experience_contract(condition),
        rewards=rewards,
        actions=actions,
        rule_ids_for_evaluator_only=rule_ids,
        pre_action_contexts=pre_contexts,
        post_action_contexts=post_contexts,
        contexts_in_use=contexts_in_use,
        controller_updates_applied=controller_updates,
        context_updates_applied=context_updates,
        final_game_state=game,
        final_controller_states=controllers,
        final_context_states=contexts,
        resource_budget=_resource_budget(game, controllers, contexts),
    )


def summarize_run(run: HiddenContextCoadaptationRun) -> HiddenContextCoadaptationDiagnostics:
    """Compute descriptive recurrence diagnostics without acceptance tuning."""

    rewards = np.asarray(run.rewards, dtype=np.float32).reshape(NUM_PHASES, PHASE_LENGTH)
    early = rewards[:, :SUMMARY_WINDOW].mean(axis=1)
    tail = rewards[:, -SUMMARY_WINDOW:].mean(axis=1)
    controllers_changed = np.asarray(
        [bool(jnp.any(state.q_weights != 0.0)) for state in run.final_controller_states],
        dtype=np.bool_,
    )
    if run.condition != HIDDEN_INFERRED:
        modes = np.full((NUM_PHASES, 2), -1, dtype=np.int32)
        lags = np.full((NUM_PHASES - 1, 2), -1, dtype=np.int32)
        distinct = np.zeros((2,), dtype=np.bool_)
        reuse = np.zeros((2,), dtype=np.bool_)
        agreement = np.zeros((2,), dtype=np.float32)
        max_in_use = np.asarray(run.contexts_in_use).max(axis=0).astype(np.int32)
        switch_counts = np.zeros((2,), dtype=np.int32)
        context_changed = np.asarray(
            (
                [
                    bool(jnp.any(state.reward_weights != 0.5))
                    for state in run.final_context_states
                ]
                if run.final_context_states is not None
                else [False, False]
            ),
            dtype=np.bool_,
        )
    else:
        final_context_states = run.final_context_states
        assert final_context_states is not None
        contexts = np.asarray(run.pre_action_contexts, dtype=np.int32)
        modes = np.empty((NUM_PHASES, 2), dtype=np.int32)
        for phase in range(NUM_PHASES):
            tail_contexts = contexts[
                (phase + 1) * PHASE_LENGTH - SUMMARY_WINDOW : (phase + 1) * PHASE_LENGTH
            ]
            for agent_id in range(2):
                modes[phase, agent_id] = int(
                    np.bincount(tail_contexts[:, agent_id], minlength=MAX_CONTEXTS).argmax()
                )
        distinct = modes[0] != modes[1]
        reuse = np.asarray(
            [
                all(
                    modes[phase, agent_id] == modes[phase % len(OFFSETS), agent_id]
                    for phase in range(2, NUM_PHASES)
                )
                for agent_id in range(2)
            ],
            dtype=np.bool_,
        )
        lags = np.full((NUM_PHASES - 1, 2), PHASE_LENGTH, dtype=np.int32)
        for phase in range(1, NUM_PHASES):
            start = phase * PHASE_LENGTH
            for agent_id in range(2):
                changed = np.flatnonzero(
                    contexts[start : start + PHASE_LENGTH, agent_id]
                    != modes[phase - 1, agent_id]
                )
                if changed.size:
                    lags[phase - 1, agent_id] = int(changed[0])
        agreement_values: list[float] = []
        recurrent_rules = np.asarray(run.rule_ids_for_evaluator_only)[2 * PHASE_LENGTH :]
        for agent_id in range(2):
            reference = modes[: len(OFFSETS), agent_id]
            predicted = contexts[2 * PHASE_LENGTH :, agent_id]
            agreement_values.append(float(np.mean(predicted == reference[recurrent_rules])))
        agreement = np.asarray(agreement_values, dtype=np.float32)
        max_in_use = np.asarray(run.contexts_in_use).max(axis=0).astype(np.int32)
        switch_counts = np.count_nonzero(np.diff(contexts, axis=0), axis=0).astype(np.int32)
        context_changed = np.asarray(
            [
                bool(jnp.any(state.reward_weights != 0.5))
                for state in final_context_states
            ],
            dtype=np.bool_,
        )
    return HiddenContextCoadaptationDiagnostics(
        phase_early_rewards=jnp.asarray(early),
        phase_tail_rewards=jnp.asarray(tail),
        tail_context_modes=jnp.asarray(modes),
        pre_action_switch_lags=jnp.asarray(lags),
        distinct_rule_slots=jnp.asarray(distinct),
        recurrence_slot_reuse=jnp.asarray(reuse),
        recurrent_context_agreement=jnp.asarray(agreement),
        max_contexts_in_use=jnp.asarray(max_in_use),
        context_switch_counts=jnp.asarray(switch_counts),
        initial_within_phase_learning_gain=jnp.asarray(
            float(np.mean(tail[:2] - early[:2])), dtype=jnp.float32
        ),
        recurrent_early_reward=jnp.asarray(float(np.mean(early[2:])), dtype=jnp.float32),
        recurrent_tail_reward=jnp.asarray(float(np.mean(tail[2:])), dtype=jnp.float32),
        overall_reward=jnp.asarray(float(np.mean(rewards)), dtype=jnp.float32),
        both_controllers_changed=jnp.asarray(bool(np.all(controllers_changed))),
        both_context_models_changed=jnp.asarray(bool(np.all(context_changed))),
    )


def _classify(
    inferred: HiddenContextCoadaptationDiagnostics,
    gap: float,
) -> tuple[bool, bool, str]:
    contexts_reused = bool(
        jnp.all(inferred.distinct_rule_slots)
        & jnp.all(inferred.recurrence_slot_reuse)
        & inferred.both_context_models_changed
    )
    above_chance = bool(inferred.recurrent_tail_reward > CHANCE_REWARD)
    if contexts_reused and above_chance and gap > 0.0:
        conclusion = (
            "development feasibility observed: both hidden learners formed distinct "
            "recurring context slots and the inferred arm exceeded its matched "
            "inference-unrouted hidden control"
        )
    else:
        conclusion = (
            "valid development rejection: distinct recurrent context reuse, control above "
            "analytic chance, or paired uplift over the inference-unrouted hidden "
            "control was absent"
        )
    return contexts_reused, above_chance, conclusion


def run_development_smoke() -> HiddenContextCoadaptationSmoke:
    """Run the first namespaced seed across the three fixed conditions."""

    seed = DEVELOPMENT_SEEDS[0]
    runs = {condition: run_hidden_context_coadaptation(condition, seed) for condition in CONDITIONS}
    diagnostics = {condition: summarize_run(run) for condition, run in runs.items()}
    inferred = diagnostics[HIDDEN_INFERRED]
    unrouted = diagnostics[HIDDEN_INFERENCE_UNROUTED]
    oracle = diagnostics[ORACLE_VISIBLE_CEILING]
    gap = float(inferred.recurrent_early_reward - unrouted.recurrent_early_reward)
    oracle_gap = float(oracle.recurrent_early_reward - inferred.recurrent_early_reward)
    contexts_reused, above_chance, conclusion = _classify(inferred, gap)
    return HiddenContextCoadaptationSmoke(
        runs=runs,
        diagnostics=diagnostics,
        inferred_minus_unrouted_recurrent_early=gap,
        oracle_minus_inferred_recurrent_early=oracle_gap,
        inferred_contexts_distinguished_and_reused=contexts_reused,
        inferred_control_above_analytic_chance=above_chance,
        conclusion=conclusion,
    )


def run_fixed_development_panel() -> HiddenContextCoadaptationPanel:
    """Run exactly the two committed development seeds with no seed selection."""

    runs = {
        condition: tuple(
            run_hidden_context_coadaptation(condition, seed) for seed in DEVELOPMENT_SEEDS
        )
        for condition in CONDITIONS
    }
    diagnostics = {
        condition: tuple(summarize_run(run) for run in condition_runs)
        for condition, condition_runs in runs.items()
    }
    gaps = tuple(
        float(inferred.recurrent_early_reward - hidden.recurrent_early_reward)
        for inferred, hidden in zip(
            diagnostics[HIDDEN_INFERRED],
            diagnostics[HIDDEN_INFERENCE_UNROUTED],
            strict=True,
        )
    )
    oracle_gaps = tuple(
        float(oracle.recurrent_early_reward - inferred.recurrent_early_reward)
        for oracle, inferred in zip(
            diagnostics[ORACLE_VISIBLE_CEILING],
            diagnostics[HIDDEN_INFERRED],
            strict=True,
        )
    )
    context_flags = tuple(
        bool(
            jnp.all(diagnostic.distinct_rule_slots)
            & jnp.all(diagnostic.recurrence_slot_reuse)
            & diagnostic.both_context_models_changed
        )
        for diagnostic in diagnostics[HIDDEN_INFERRED]
    )
    chance_flags = tuple(
        bool(diagnostic.recurrent_tail_reward > CHANCE_REWARD)
        for diagnostic in diagnostics[HIDDEN_INFERRED]
    )
    success = all(context_flags) and all(chance_flags) and all(gap > 0.0 for gap in gaps)
    conclusion = (
        "development feasibility observed on both fixed seeds"
        if success
        else "valid development rejection on at least one fixed seed"
    )
    return HiddenContextCoadaptationPanel(
        runs=runs,
        diagnostics=diagnostics,
        inferred_minus_unrouted_recurrent_early_by_seed=gaps,
        oracle_minus_inferred_recurrent_early_by_seed=oracle_gaps,
        all_inferred_contexts_distinguished_and_reused=all(context_flags),
        all_inferred_control_above_analytic_chance=all(chance_flags),
        conclusion=conclusion,
    )


def validate_static_contract() -> tuple[str, ...]:
    """Fail closed if causality, namespace, shape, or nonpromotion drifts."""

    errors: list[str] = []
    if (DEVELOPMENT_ONLY, SCIENTIFIC_PROMOTION_ALLOWED, OUTPUT_WRITES_ALLOWED) != (
        True,
        False,
        False,
    ):
        errors.append("development-only nonpromotion contract changed")
    if ENVIRONMENT_RANDOMNESS_CONSUMED:
        errors.append("deterministic recurring game must not claim environment randomness")
    if DEVELOPMENT_SEEDS != derive_development_seeds(DEVELOPMENT_NAMESPACE, 2):
        errors.append("fixed seed snapshot differs from its namespace derivation")
    if PROTOCOL.num_steps != NUM_STEPS or NUM_PHASES < 4 or NUM_PHASES % 2 != 0:
        errors.append("uninterrupted recurring-life geometry changed")
    if not 0 < SUMMARY_WINDOW <= PHASE_LENGTH:
        errors.append("summary window is invalid")
    if PROTOCOL.boundary_callbacks_used or PROTOCOL.resets_after_initialization != 0:
        errors.append("boundary callbacks or within-life resets were enabled")
    visible = set(LEARNER_PRE_ACTION_CHANNELS) | set(LEARNER_POST_ACTION_CHANNELS)
    if visible & FORBIDDEN_LEARNER_CHANNELS:
        errors.append("hidden learner-visible channels contain a rule oracle")
    if _HIDDEN_GAME.config.feature_mode != "plain":
        errors.append("hidden game exposes its rule through observe")
    if _ORACLE_GAME.config.feature_mode != "context":
        errors.append("diagnostic oracle ceiling is no longer explicit")
    if _CONTROL_CONFIG.use_bias:
        errors.append("shared control bias would create a cross-context forgetting path")
    if _CONTEXT_CONFIG.max_contexts != len(OFFSETS):
        errors.append("context bank is not bounded to the number of recurring rules")
    if CONDITIONS != (
        HIDDEN_INFERRED,
        HIDDEN_INFERENCE_UNROUTED,
        ORACLE_VISIBLE_CEILING,
    ):
        errors.append("matched condition set changed")
    return tuple(errors)


__all__ = [
    "CHANCE_REWARD",
    "CONDITIONS",
    "DEVELOPMENT_NAMESPACE",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "DEVELOPMENT_SEEDS",
    "FORBIDDEN_LEARNER_CHANNELS",
    "HIDDEN_INFERRED",
    "HIDDEN_INFERENCE_UNROUTED",
    "HiddenContextCoadaptationDiagnostics",
    "HiddenContextCoadaptationPanel",
    "HiddenContextCoadaptationProtocol",
    "HiddenContextCoadaptationResourceBudget",
    "HiddenContextCoadaptationRun",
    "HiddenContextCoadaptationSeed",
    "HiddenContextCoadaptationSmoke",
    "LEARNER_POST_ACTION_CHANNELS",
    "LEARNER_PRE_ACTION_CHANNELS",
    "LearnerExperienceContract",
    "NUM_STEPS",
    "ORACLE_VISIBLE_CEILING",
    "OUTPUT_WRITES_ALLOWED",
    "PHASE_LENGTH",
    "PROTOCOL",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "derive_development_seeds",
    "run_development_smoke",
    "run_fixed_development_panel",
    "run_hidden_context_coadaptation",
    "summarize_run",
    "validate_static_contract",
]
