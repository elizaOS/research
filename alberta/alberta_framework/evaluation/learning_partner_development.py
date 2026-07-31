# mypy: disable-error-code="call-arg"
"""Development-only evaluation for a genuine online learning-partner rung.

This module runs one uninterrupted recurring Lewis signaling life and records
prequential primitives plus read-only phase-boundary probes.  It deliberately
defines no scientific acceptance threshold, artifact writer, CLI, replay, or
promotion path.  Every serialized report is explicitly development-only.

The public context directly selects disjoint rows in both role tables.  Thus
recurrence probes here test whether a learned dyadic convention remains usable
when its row is revisited, but retention is structurally interference-free.
This rung establishes co-learning and causal controls, not general resistance
to catastrophic forgetting.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.signaling_bandit import (
    ROLE_VALUE_SHAPE,
    RoleBanditState,
    SignalingBanditAgent,
    SignalingBanditConfig,
    SignalingBanditState,
    decision_with_action,
    role_resource_budget,
    signaling_bandit_keys,
    signaling_bandit_resource_budget,
)
from alberta_framework.streams.learning_partner import (
    CONSTANT_ONE_CHANNEL,
    CONSTANT_ZERO_CHANNEL,
    DIRECT_CHANNEL,
    SHUFFLED_CHANNEL,
    LearningPartnerChannel,
    LearningPartnerWorld,
    LearningPartnerWorldConfig,
    LearningPartnerWorldState,
    learning_partner_world_keys,
)

LEARNING_PARTNER_DEVELOPMENT_SCHEMA = "learning-partner-development-v0"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
FROZEN_ROLE_SEMANTICS = (
    "a false role write mask preserves the float32 value table bitwise while "
    "advancing that role's independent policy RNG key"
)
RETENTION_SCOPE = (
    "public contexts index disjoint table rows; retention is structurally "
    "interference-free and is not catastrophic-forgetting evidence"
)

JOINT_ADAPTIVE: Literal["joint_adaptive"] = "joint_adaptive"
HELPER_FROZEN: Literal["helper_frozen"] = "helper_frozen"
BENEFICIARY_FROZEN: Literal["beneficiary_frozen"] = "beneficiary_frozen"
BOTH_FROZEN: Literal["both_frozen"] = "both_frozen"
CONSTANT_ZERO_DELIVERY: Literal["constant_0_delivery"] = "constant_0_delivery"
CONSTANT_ONE_DELIVERY: Literal["constant_1_delivery"] = "constant_1_delivery"
SHUFFLED_DELIVERY: Literal["shuffled_delivery"] = "shuffled_delivery"
ORACLE_HELPER: Literal["oracle_helper"] = "oracle_helper"
BENEFICIARY_ALONE: Literal["beneficiary_alone"] = "beneficiary_alone"

type MatchedLearningPartnerCondition = Literal[
    "joint_adaptive",
    "helper_frozen",
    "beneficiary_frozen",
    "both_frozen",
    "constant_0_delivery",
    "constant_1_delivery",
    "shuffled_delivery",
    "oracle_helper",
]

MATCHED_CONDITIONS: tuple[MatchedLearningPartnerCondition, ...] = (
    JOINT_ADAPTIVE,
    HELPER_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_FROZEN,
    CONSTANT_ZERO_DELIVERY,
    CONSTANT_ONE_DELIVERY,
    SHUFFLED_DELIVERY,
    ORACLE_HELPER,
)

_WORLD_ROOT_RNG_TAG = 0x574F524C  # ASCII "WORL"
_LEARNER_ROOT_RNG_TAG = 0x4C524E52  # ASCII "LRNR"


@dataclasses.dataclass(frozen=True)
class LearningPartnerDevelopmentConfig:
    """Unfrozen development configuration; defaults to eight 512-step phases."""

    phase_length: int = 512
    n_phases: int = 8
    learning_rate: float = 0.1
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        if type(self.phase_length) is not int or self.phase_length < 1:
            raise ValueError("phase_length must be a positive integer")
        if type(self.n_phases) is not int or self.n_phases < 4:
            raise ValueError("n_phases must be an integer of at least four so both contexts recur")
        SignalingBanditConfig(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
        )

    @property
    def num_steps(self) -> int:
        """Total continuing transitions in the uninterrupted life."""

        return self.phase_length * self.n_phases

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible config with nonpromotion semantics."""

        return {
            "phase_length": self.phase_length,
            "n_phases": self.n_phases,
            "num_steps": self.num_steps,
            "learning_rate": float(self.learning_rate),
            "epsilon": float(self.epsilon),
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_thresholds_frozen": False,
        }


@dataclasses.dataclass(frozen=True)
class _ConditionSpec:
    channel: LearningPartnerChannel
    helper_write: bool
    beneficiary_write: bool
    oracle_helper: bool


def condition_spec(condition: MatchedLearningPartnerCondition) -> _ConditionSpec:
    """Return the exact intervention for one resource-matched condition."""

    if condition == JOINT_ADAPTIVE:
        return _ConditionSpec(DIRECT_CHANNEL, True, True, False)
    if condition == HELPER_FROZEN:
        return _ConditionSpec(DIRECT_CHANNEL, False, True, False)
    if condition == BENEFICIARY_FROZEN:
        return _ConditionSpec(DIRECT_CHANNEL, True, False, False)
    if condition == BOTH_FROZEN:
        return _ConditionSpec(DIRECT_CHANNEL, False, False, False)
    if condition == CONSTANT_ZERO_DELIVERY:
        return _ConditionSpec(CONSTANT_ZERO_CHANNEL, True, True, False)
    if condition == CONSTANT_ONE_DELIVERY:
        return _ConditionSpec(CONSTANT_ONE_CHANNEL, True, True, False)
    if condition == SHUFFLED_DELIVERY:
        return _ConditionSpec(SHUFFLED_CHANNEL, True, True, False)
    if condition == ORACLE_HELPER:
        return _ConditionSpec(DIRECT_CHANNEL, False, True, True)
    raise ValueError(f"unknown matched learning-partner condition: {condition!r}")


@dataclasses.dataclass(frozen=True)
class LearningPartnerPrimitiveTrace:
    """Packed-friendly prequential primitives retained only in memory."""

    step_index: Array
    phase_index: Array
    context: Array
    cue: Array
    oracle_target: Array
    helper_message: Array
    delivered_message: Array
    beneficiary_action: Array
    reward: Array
    helper_write: Array
    beneficiary_write: Array
    helper_value_pre: Array
    helper_value_post: Array
    beneficiary_value_pre: Array
    beneficiary_value_post: Array

    def to_dict(self) -> dict[str, object]:
        """Serialize primitives without replaying the life."""

        return {
            field.name: np.asarray(getattr(self, field.name)).tolist()
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class PhaseBoundaryProbes:
    """Read-only exhaustive four-case probes captured at phase boundaries."""

    phase_context: Array
    helper_messages: Array
    beneficiary_actions: Array
    joint_accuracy: Array
    helper_cue_dependence: Array
    beneficiary_message_dependence: Array
    message_flip_accuracy: Array
    message_flip_effect: Array
    best_constant_accuracy: Array
    gain_over_best_constant: Array
    context_forgetting: Array
    context_forgetting_valid: Array
    context_recovery: Array
    context_recovery_valid: Array

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible probe arrays."""

        return {
            field.name: np.asarray(getattr(self, field.name)).tolist()
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class LearningPartnerResourceReport:
    """Exact state budget and whether it belongs to the matched comparison."""

    helper_state_scalars: int
    helper_state_bytes: int
    beneficiary_state_scalars: int
    beneficiary_state_bytes: int
    total_state_scalars: int
    total_state_bytes: int
    resource_matched: bool
    comparison_group: str
    resource_mismatch_disclosed: bool

    def to_dict(self) -> dict[str, object]:
        """Return exact scalar/byte counts."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LearningPartnerRunResult:
    """One uninterrupted development run and its non-replayed diagnostics."""

    condition: str
    seed: int
    config: LearningPartnerDevelopmentConfig
    trace: LearningPartnerPrimitiveTrace
    probes: PhaseBoundaryProbes
    final_helper_state: RoleBanditState | None
    final_beneficiary_state: RoleBanditState
    resource: LearningPartnerResourceReport

    def to_dict(self, *, include_trace: bool = True) -> dict[str, object]:
        """Serialize this in-memory result as a development-only payload."""

        payload: dict[str, object] = {
            "schema_version": LEARNING_PARTNER_DEVELOPMENT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_thresholds_frozen": False,
            "acceptance_status": "descriptive_only_no_acceptance_gate",
            "no_replay": True,
            "frozen_role_semantics": FROZEN_ROLE_SEMANTICS,
            "retention_scope": RETENTION_SCOPE,
            "condition": self.condition,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "resource": self.resource.to_dict(),
            "probes": self.probes.to_dict(),
            "final_helper_values": (
                None
                if self.final_helper_state is None
                else np.asarray(self.final_helper_state.values).tolist()
            ),
            "final_beneficiary_values": np.asarray(self.final_beneficiary_state.values).tolist(),
        }
        if include_trace:
            payload["trace"] = self.trace.to_dict()
        return payload


@dataclasses.dataclass(frozen=True)
class LearningPartnerCrossPlay:
    """Full dyad matrix plus a fixed cyclic derangement as the primary split."""

    seed_order: tuple[int, ...]
    score_matrix: np.ndarray[Any, np.dtype[np.float64]]
    within_dyad_scores: np.ndarray[Any, np.dtype[np.float64]]
    derangement_columns: np.ndarray[Any, np.dtype[np.int64]]
    deranged_scores: np.ndarray[Any, np.dtype[np.float64]]
    within_dyad_mean: float
    fixed_derangement_mean: float
    within_minus_deranged: float

    def to_dict(self) -> dict[str, object]:
        """Return the complete descriptive cross-play record."""

        return {
            "seed_order": list(self.seed_order),
            "score_matrix": self.score_matrix.tolist(),
            "within_dyad_scores": self.within_dyad_scores.tolist(),
            "derangement_columns": self.derangement_columns.tolist(),
            "deranged_scores": self.deranged_scores.tolist(),
            "within_dyad_mean": self.within_dyad_mean,
            "fixed_derangement_mean": self.fixed_derangement_mean,
            "within_minus_deranged": self.within_minus_deranged,
            "primary_comparison": "within_dyad_mean_minus_fixed_cyclic_derangement_mean",
        }


@dataclasses.dataclass(frozen=True)
class LearningPartnerDevelopmentReport:
    """All matched controls, separately disclosed baseline, and cross-play."""

    config: LearningPartnerDevelopmentConfig
    seeds: tuple[int, ...]
    matched_runs: tuple[LearningPartnerRunResult, ...]
    beneficiary_alone_runs: tuple[LearningPartnerRunResult, ...]
    cross_play: LearningPartnerCrossPlay

    def to_dict(self, *, include_traces: bool = True) -> dict[str, object]:
        """Serialize the report while preserving fail-closed scope labels."""

        return {
            "schema_version": LEARNING_PARTNER_DEVELOPMENT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_thresholds_frozen": False,
            "acceptance_status": "descriptive_only_no_acceptance_gate",
            "no_replay": True,
            "traces_included": include_traces,
            "frozen_role_semantics": FROZEN_ROLE_SEMANTICS,
            "retention_scope": RETENTION_SCOPE,
            "seed_namespace": "unregistered_development_seeds_consumed_on_run",
            "config": self.config.to_dict(),
            "seeds": list(self.seeds),
            "matched_condition_order": list(MATCHED_CONDITIONS),
            "matched_runs": [
                run.to_dict(include_trace=include_traces) for run in self.matched_runs
            ],
            "beneficiary_alone_runs": [
                run.to_dict(include_trace=include_traces) for run in self.beneficiary_alone_runs
            ],
            "cross_play": self.cross_play.to_dict(),
        }


def _probe_one_pair(helper_values: Array, beneficiary_values: Array) -> tuple[Array, ...]:
    """Exhaust all two-context/two-cue cases without updating either table."""

    helper_messages = jnp.argmax(helper_values, axis=-1).astype(jnp.int8)
    beneficiary_actions = jnp.argmax(beneficiary_values, axis=-1).astype(jnp.int8)
    contexts = jnp.arange(2, dtype=jnp.int32)[:, None]
    cues = jnp.arange(2, dtype=jnp.int32)[None, :]
    targets = jnp.bitwise_xor(contexts, cues)
    row_contexts = jnp.broadcast_to(contexts, (2, 2))
    direct_actions = beneficiary_actions[row_contexts, helper_messages]
    flipped_actions = beneficiary_actions[row_contexts, 1 - helper_messages]
    joint_accuracy = jnp.mean((direct_actions == targets).astype(jnp.float32), axis=1)
    flip_accuracy = jnp.mean((flipped_actions == targets).astype(jnp.float32), axis=1)
    constant_zero = jnp.mean(
        (beneficiary_actions[:, 0, None] == targets).astype(jnp.float32),
        axis=1,
    )
    constant_one = jnp.mean(
        (beneficiary_actions[:, 1, None] == targets).astype(jnp.float32),
        axis=1,
    )
    best_constant = jnp.maximum(constant_zero, constant_one)
    return (
        helper_messages,
        beneficiary_actions,
        joint_accuracy,
        (helper_messages[:, 0] != helper_messages[:, 1]).astype(jnp.float32),
        (beneficiary_actions[:, 0] != beneficiary_actions[:, 1]).astype(jnp.float32),
        flip_accuracy,
        joint_accuracy - flip_accuracy,
        best_constant,
        joint_accuracy - best_constant,
    )


def _forgetting_and_recovery(
    joint_accuracy: Array,
) -> tuple[Array, Array, Array, Array]:
    """Compute finite per-context recurrence diagnostics with validity masks."""

    accuracy = np.asarray(joint_accuracy, dtype=np.float32)
    n_phases = accuracy.shape[0]
    forgetting = np.zeros_like(accuracy)
    forgetting_valid = np.zeros_like(accuracy, dtype=np.bool_)
    recovery = np.zeros_like(accuracy)
    recovery_valid = np.zeros_like(accuracy, dtype=np.bool_)
    last_active = np.zeros((2,), dtype=np.float32)
    has_active = np.zeros((2,), dtype=np.bool_)
    for phase in range(n_phases):
        active = phase % 2
        inactive = 1 - active
        if has_active[inactive]:
            forgetting[phase, inactive] = max(
                0.0,
                float(last_active[inactive] - accuracy[phase, inactive]),
            )
            forgetting_valid[phase, inactive] = True
        if phase >= 2:
            recovery[phase, active] = max(
                0.0,
                float(accuracy[phase, active] - accuracy[phase - 1, active]),
            )
            recovery_valid[phase, active] = True
        last_active[active] = accuracy[phase, active]
        has_active[active] = True
    return (
        jnp.asarray(forgetting),
        jnp.asarray(forgetting_valid),
        jnp.asarray(recovery),
        jnp.asarray(recovery_valid),
    )


def _phase_probes(
    helper_value_history: Array,
    beneficiary_value_history: Array,
    config: LearningPartnerDevelopmentConfig,
) -> PhaseBoundaryProbes:
    boundary_indices = (jnp.arange(config.n_phases, dtype=jnp.int32) + 1) * config.phase_length - 1
    helper_boundaries = helper_value_history[boundary_indices]
    beneficiary_boundaries = beneficiary_value_history[boundary_indices]
    probe_arrays = jax.vmap(_probe_one_pair)(helper_boundaries, beneficiary_boundaries)
    (
        helper_messages,
        beneficiary_actions,
        joint_accuracy,
        helper_dependence,
        beneficiary_dependence,
        flip_accuracy,
        flip_effect,
        best_constant,
        constant_gain,
    ) = probe_arrays
    forgetting, forgetting_valid, recovery, recovery_valid = _forgetting_and_recovery(
        joint_accuracy
    )
    return PhaseBoundaryProbes(
        phase_context=(jnp.arange(config.n_phases, dtype=jnp.int8) % 2),
        helper_messages=helper_messages,
        beneficiary_actions=beneficiary_actions,
        joint_accuracy=joint_accuracy,
        helper_cue_dependence=helper_dependence,
        beneficiary_message_dependence=beneficiary_dependence,
        message_flip_accuracy=flip_accuracy,
        message_flip_effect=flip_effect,
        best_constant_accuracy=best_constant,
        gain_over_best_constant=constant_gain,
        context_forgetting=forgetting,
        context_forgetting_valid=forgetting_valid,
        context_recovery=recovery,
        context_recovery_valid=recovery_valid,
    )


def _matched_resource(state: SignalingBanditState) -> LearningPartnerResourceReport:
    budget = signaling_bandit_resource_budget(state)
    return LearningPartnerResourceReport(
        helper_state_scalars=budget.helper.state_scalars,
        helper_state_bytes=budget.helper.state_bytes,
        beneficiary_state_scalars=budget.beneficiary.state_scalars,
        beneficiary_state_bytes=budget.beneficiary.state_bytes,
        total_state_scalars=budget.state_scalars,
        total_state_bytes=budget.state_bytes,
        resource_matched=True,
        comparison_group="matched_two_role_zero_initialized_tables",
        resource_mismatch_disclosed=False,
    )


def _beneficiary_alone_resource(state: RoleBanditState) -> LearningPartnerResourceReport:
    budget = role_resource_budget(state)
    return LearningPartnerResourceReport(
        helper_state_scalars=0,
        helper_state_bytes=0,
        beneficiary_state_scalars=budget.state_scalars,
        beneficiary_state_bytes=budget.state_bytes,
        total_state_scalars=budget.state_scalars,
        total_state_bytes=budget.state_bytes,
        resource_matched=False,
        comparison_group="separate_resource_unmatched_beneficiary_alone",
        resource_mismatch_disclosed=True,
    )


def _channel_code(channel: LearningPartnerChannel) -> int:
    if channel == DIRECT_CHANNEL:
        return 0
    if channel == CONSTANT_ZERO_CHANNEL:
        return 1
    if channel == CONSTANT_ONE_CHANNEL:
        return 2
    if channel == SHUFFLED_CHANNEL:
        return 3
    raise ValueError(f"unknown learning-partner channel: {channel!r}")


@functools.lru_cache(maxsize=16)
def _matched_scan_runner(config: LearningPartnerDevelopmentConfig) -> Any:
    """Compile one dynamic-intervention scan per development config."""

    world = LearningPartnerWorld(LearningPartnerWorldConfig(config.phase_length))
    learner = SignalingBanditAgent(SignalingBanditConfig(config.learning_rate, config.epsilon))

    def run(
        world_state: LearningPartnerWorldState,
        learner_state: SignalingBanditState,
        channel_code: Array,
        helper_write: Array,
        beneficiary_write: Array,
        oracle_helper: Array,
    ) -> Any:
        def scan_step(
            carry: tuple[LearningPartnerWorldState, SignalingBanditState],
            _: Array,
        ) -> tuple[
            tuple[LearningPartnerWorldState, SignalingBanditState],
            tuple[Array, ...],
        ]:
            old_world_state, old_learner_state = carry
            observation = world.observe(old_world_state)
            helper_decision = learner.select_helper(
                old_learner_state.helper,
                observation.public_context,
                observation.helper_cue,
            )
            oracle_message = jnp.bitwise_xor(
                observation.helper_cue,
                observation.public_context,
            )
            actual_message = jnp.where(
                oracle_helper,
                oracle_message,
                helper_decision.action,
            )
            helper_decision = decision_with_action(helper_decision, actual_message)
            shuffled_delivery = world.deliver(
                old_world_state,
                actual_message,
                SHUFFLED_CHANNEL,
            )
            delivered = jnp.select(
                (
                    channel_code == 0,
                    channel_code == 1,
                    channel_code == 2,
                ),
                (
                    actual_message,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(1, dtype=jnp.int32),
                ),
                default=shuffled_delivery,
            ).astype(jnp.int32)
            beneficiary_decision = learner.select_beneficiary(
                old_learner_state.beneficiary,
                observation.public_context,
                delivered,
            )
            transition, next_world_state = world.step_with_delivery(
                old_world_state,
                actual_message,
                delivered,
                beneficiary_decision.action,
            )
            update = learner.update(
                old_learner_state,
                helper_decision,
                beneficiary_decision,
                transition.reward,
                helper_write=helper_write,
                beneficiary_write=beneficiary_write,
            )
            output = (
                transition.oracle.step_count,
                transition.oracle.phase_index,
                transition.oracle.context,
                transition.observation.helper_cue,
                transition.oracle.target,
                transition.helper_message,
                transition.delivered_message,
                transition.beneficiary_action,
                transition.reward,
                helper_write,
                beneficiary_write,
                update.helper_value_pre,
                update.helper_value_post,
                update.beneficiary_value_pre,
                update.beneficiary_value_post,
                update.state.helper.values,
                update.state.beneficiary.values,
            )
            return (next_world_state, update.state), output

        return jax.lax.scan(
            scan_step,
            (world_state, learner_state),
            jnp.arange(config.num_steps, dtype=jnp.int32),
        )

    return jax.jit(run)


def run_learning_partner(
    condition: MatchedLearningPartnerCondition,
    *,
    seed: int,
    config: LearningPartnerDevelopmentConfig | None = None,
) -> LearningPartnerRunResult:
    """Run one resource-matched condition in a single uninterrupted scan."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config = config or LearningPartnerDevelopmentConfig()
    spec = condition_spec(condition)
    world = LearningPartnerWorld(LearningPartnerWorldConfig(config.phase_length))
    learner = SignalingBanditAgent(SignalingBanditConfig(config.learning_rate, config.epsilon))
    root_key = jr.key(seed)
    world_state = world.init(learning_partner_world_keys(jr.fold_in(root_key, _WORLD_ROOT_RNG_TAG)))
    learner_state = learner.init(signaling_bandit_keys(jr.fold_in(root_key, _LEARNER_ROOT_RNG_TAG)))
    resource = _matched_resource(learner_state)

    (_, final_state), outputs = _matched_scan_runner(config)(
        world_state,
        learner_state,
        jnp.asarray(_channel_code(spec.channel), dtype=jnp.int32),
        jnp.asarray(spec.helper_write),
        jnp.asarray(spec.beneficiary_write),
        jnp.asarray(spec.oracle_helper),
    )
    (
        step_index,
        phase_index,
        context,
        cue,
        target,
        message,
        delivered,
        action,
        reward,
        helper_write,
        beneficiary_write,
        helper_value_pre,
        helper_value_post,
        beneficiary_value_pre,
        beneficiary_value_post,
        helper_value_history,
        beneficiary_value_history,
    ) = outputs
    trace = LearningPartnerPrimitiveTrace(
        step_index=step_index.astype(jnp.int32),
        phase_index=phase_index.astype(jnp.int32),
        context=context.astype(jnp.int8),
        cue=cue.astype(jnp.int8),
        oracle_target=target.astype(jnp.int8),
        helper_message=message.astype(jnp.int8),
        delivered_message=delivered.astype(jnp.int8),
        beneficiary_action=action.astype(jnp.int8),
        reward=reward.astype(jnp.float32),
        helper_write=helper_write.astype(jnp.bool_),
        beneficiary_write=beneficiary_write.astype(jnp.bool_),
        helper_value_pre=helper_value_pre.astype(jnp.float32),
        helper_value_post=helper_value_post.astype(jnp.float32),
        beneficiary_value_pre=beneficiary_value_pre.astype(jnp.float32),
        beneficiary_value_post=beneficiary_value_post.astype(jnp.float32),
    )
    probes = _phase_probes(helper_value_history, beneficiary_value_history, config)
    return LearningPartnerRunResult(
        condition=condition,
        seed=seed,
        config=config,
        trace=trace,
        probes=probes,
        final_helper_state=final_state.helper,
        final_beneficiary_state=final_state.beneficiary,
        resource=resource,
    )


def run_beneficiary_alone(
    *,
    seed: int,
    config: LearningPartnerDevelopmentConfig | None = None,
) -> LearningPartnerRunResult:
    """Run the separately reported, resource-unmatched no-helper baseline."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config = config or LearningPartnerDevelopmentConfig()
    world = LearningPartnerWorld(LearningPartnerWorldConfig(config.phase_length))
    learner = SignalingBanditAgent(SignalingBanditConfig(config.learning_rate, config.epsilon))
    root_key = jr.key(seed)
    world_state = world.init(learning_partner_world_keys(jr.fold_in(root_key, _WORLD_ROOT_RNG_TAG)))
    learner_keys = signaling_bandit_keys(jr.fold_in(root_key, _LEARNER_ROOT_RNG_TAG))
    beneficiary_state = learner.init_beneficiary_only(learner_keys.beneficiary)
    resource = _beneficiary_alone_resource(beneficiary_state)

    def scan_step(
        carry: tuple[Any, RoleBanditState],
        _: Array,
    ) -> tuple[tuple[Any, RoleBanditState], tuple[Array, ...]]:
        old_world_state, old_beneficiary_state = carry
        observation = world.observe(old_world_state)
        message = jnp.asarray(0, dtype=jnp.int32)
        decision = learner.select_beneficiary_only(
            old_beneficiary_state,
            observation.public_context,
            message,
        )
        transition, next_world_state = world.step(
            old_world_state,
            message,
            decision.action,
            DIRECT_CHANNEL,
        )
        next_beneficiary_state = learner.update_beneficiary_only(
            old_beneficiary_state,
            decision,
            transition.reward,
        )
        value_post = next_beneficiary_state.values[
            decision.context,
            decision.private_input,
            decision.action,
        ]
        zeros = jnp.asarray(0.0, dtype=jnp.float32)
        output = (
            transition.oracle.step_count,
            transition.oracle.phase_index,
            transition.oracle.context,
            transition.observation.helper_cue,
            transition.oracle.target,
            message,
            transition.delivered_message,
            transition.beneficiary_action,
            transition.reward,
            jnp.asarray(False),
            jnp.asarray(True),
            zeros,
            zeros,
            decision.selected_value,
            value_post,
            jnp.zeros(ROLE_VALUE_SHAPE, dtype=jnp.float32),
            next_beneficiary_state.values,
        )
        return (next_world_state, next_beneficiary_state), output

    (_, final_beneficiary), outputs = jax.lax.scan(
        scan_step,
        (world_state, beneficiary_state),
        jnp.arange(config.num_steps, dtype=jnp.int32),
    )
    (
        step_index,
        phase_index,
        context,
        cue,
        target,
        message,
        delivered,
        action,
        reward,
        helper_write,
        beneficiary_write,
        helper_value_pre,
        helper_value_post,
        beneficiary_value_pre,
        beneficiary_value_post,
        helper_value_history,
        beneficiary_value_history,
    ) = outputs
    trace = LearningPartnerPrimitiveTrace(
        step_index=step_index.astype(jnp.int32),
        phase_index=phase_index.astype(jnp.int32),
        context=context.astype(jnp.int8),
        cue=cue.astype(jnp.int8),
        oracle_target=target.astype(jnp.int8),
        helper_message=message.astype(jnp.int8),
        delivered_message=delivered.astype(jnp.int8),
        beneficiary_action=action.astype(jnp.int8),
        reward=reward.astype(jnp.float32),
        helper_write=helper_write.astype(jnp.bool_),
        beneficiary_write=beneficiary_write.astype(jnp.bool_),
        helper_value_pre=helper_value_pre.astype(jnp.float32),
        helper_value_post=helper_value_post.astype(jnp.float32),
        beneficiary_value_pre=beneficiary_value_pre.astype(jnp.float32),
        beneficiary_value_post=beneficiary_value_post.astype(jnp.float32),
    )
    return LearningPartnerRunResult(
        condition=BENEFICIARY_ALONE,
        seed=seed,
        config=config,
        trace=trace,
        probes=_phase_probes(helper_value_history, beneficiary_value_history, config),
        final_helper_state=None,
        final_beneficiary_state=final_beneficiary,
        resource=resource,
    )


def _cross_play_score(helper_values: Array, beneficiary_values: Array) -> float:
    return float(np.mean(np.asarray(_probe_one_pair(helper_values, beneficiary_values)[2])))


def final_cross_play(
    joint_runs: Sequence[LearningPartnerRunResult],
) -> LearningPartnerCrossPlay:
    """Cross every final helper with every final beneficiary exactly once.

    The full matrix is descriptive.  The sole primary split is the fixed
    cyclic derangement ``helper i`` with ``beneficiary (i + 1) mod n``; it is
    chosen independently of observed scores.
    """

    if len(joint_runs) < 2:
        raise ValueError("cross-play requires at least two joint-adaptive runs")
    if any(run.condition != JOINT_ADAPTIVE for run in joint_runs):
        raise ValueError("cross-play accepts only joint_adaptive runs")
    if any(run.final_helper_state is None for run in joint_runs):
        raise ValueError("cross-play requires an allocated helper for every run")
    n_runs = len(joint_runs)
    matrix = np.empty((n_runs, n_runs), dtype=np.float64)
    for helper_index, helper_run in enumerate(joint_runs):
        helper_state = cast(RoleBanditState, helper_run.final_helper_state)
        for beneficiary_index, beneficiary_run in enumerate(joint_runs):
            matrix[helper_index, beneficiary_index] = _cross_play_score(
                helper_state.values,
                beneficiary_run.final_beneficiary_state.values,
            )
    indices = np.arange(n_runs, dtype=np.int64)
    derangement = (indices + 1) % n_runs
    within = matrix[indices, indices].copy()
    deranged = matrix[indices, derangement].copy()
    within_mean = float(np.mean(within))
    deranged_mean = float(np.mean(deranged))
    return LearningPartnerCrossPlay(
        seed_order=tuple(run.seed for run in joint_runs),
        score_matrix=matrix,
        within_dyad_scores=within,
        derangement_columns=derangement,
        deranged_scores=deranged,
        within_dyad_mean=within_mean,
        fixed_derangement_mean=deranged_mean,
        within_minus_deranged=within_mean - deranged_mean,
    )


def run_learning_partner_development(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3),
    config: LearningPartnerDevelopmentConfig | None = None,
) -> LearningPartnerDevelopmentReport:
    """Run all descriptive controls without creating scientific evidence."""

    config = config or LearningPartnerDevelopmentConfig()
    seed_tuple = tuple(seeds)
    if len(seed_tuple) < 2:
        raise ValueError("development cross-play requires at least two seeds")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("development seeds must be unique")
    if any(type(seed) is not int or seed < 0 for seed in seed_tuple):
        raise ValueError("development seeds must be non-negative integers")
    matched_runs = tuple(
        run_learning_partner(condition, seed=seed, config=config)
        for seed in seed_tuple
        for condition in MATCHED_CONDITIONS
    )
    beneficiary_alone_runs = tuple(
        run_beneficiary_alone(seed=seed, config=config) for seed in seed_tuple
    )
    joint_runs = tuple(run for run in matched_runs if run.condition == JOINT_ADAPTIVE)
    return LearningPartnerDevelopmentReport(
        config=config,
        seeds=seed_tuple,
        matched_runs=matched_runs,
        beneficiary_alone_runs=beneficiary_alone_runs,
        cross_play=final_cross_play(joint_runs),
    )


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value))


def _payload_array(
    value: object,
    *,
    dtype: Any,
) -> np.ndarray[Any, np.dtype[Any]] | None:
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _float32_scalar_bits(value: np.float32) -> int:
    """Return the exact IEEE-754 binary32 encoding of one scalar."""

    return int(np.asarray(value, dtype=np.float32).reshape(()).view(np.uint32))


def _float32_arrays_bit_equal(
    left: np.ndarray[Any, np.dtype[np.float32]],
    right: np.ndarray[Any, np.dtype[np.float32]],
) -> bool:
    """Compare float32 arrays without treating signed zero as interchangeable."""

    return left.shape == right.shape and np.array_equal(
        left.view(np.uint32),
        right.view(np.uint32),
    )


def _portable_float32_update_candidates(
    pre: np.float32,
    reward: np.float32,
    learning_rate: np.float32,
) -> tuple[np.float32, np.float32]:
    """Return the two permitted binary32 evaluation orders for one update.

    The learner first rounds ``reward - pre`` to binary32.  A backend may then
    either round the multiply before the add or contract that outer
    multiply-add.  The unfused result below makes both binary32 roundings
    explicit.  For the fused result, a binary32 product has at most 48
    significant bits, so binary64's 53-bit significand preserves the exact
    product.  In this bounded convex update (all operands and results lie in
    ``[0, 1]``), the binary64 sum supplies more than the guard precision needed
    for the single final binary32 rounding.  One binary64 expression followed
    by one binary32 cast therefore emulates the permitted fused result without
    relying on the validator host's contraction setting.
    """

    delta = np.float32(reward - pre)
    product = np.float32(learning_rate * delta)
    unfused = np.float32(pre + product)
    fused = np.float32(np.float64(pre) + np.float64(learning_rate) * np.float64(delta))
    return unfused, fused


def _scope_errors(payload: Mapping[str, object], prefix: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != LEARNING_PARTNER_DEVELOPMENT_SCHEMA:
        errors.append(f"{prefix}schema_version mismatch")
    if payload.get("development_only") is not True:
        errors.append(f"{prefix}development_only must be true")
    if payload.get("scientific_promotion_allowed") is not False:
        errors.append(f"{prefix}scientific_promotion_allowed must be false")
    if payload.get("claim_thresholds_frozen") is not False:
        errors.append(f"{prefix}claim_thresholds_frozen must be false")
    if payload.get("acceptance_status") != "descriptive_only_no_acceptance_gate":
        errors.append(f"{prefix}acceptance_status must remain descriptive-only")
    if payload.get("no_replay") is not True:
        errors.append(f"{prefix}no_replay must be true")
    if payload.get("frozen_role_semantics") != FROZEN_ROLE_SEMANTICS:
        errors.append(f"{prefix}frozen_role_semantics mismatch")
    if payload.get("retention_scope") != RETENTION_SCOPE:
        errors.append(f"{prefix}retention_scope mismatch")
    return errors


def _validate_serialized_config(
    value: object,
    *,
    prefix: str,
) -> tuple[dict[str, object] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return None, [f"{prefix}config must be a mapping"]
    config = dict(value)
    phase_length = config.get("phase_length")
    n_phases = config.get("n_phases")
    num_steps = config.get("num_steps")
    if (
        not isinstance(phase_length, Integral)
        or isinstance(phase_length, bool)
        or int(phase_length) < 1
    ):
        errors.append(f"{prefix}config.phase_length must be a positive integer")
    if not isinstance(n_phases, Integral) or isinstance(n_phases, bool) or int(n_phases) < 4:
        errors.append(f"{prefix}config.n_phases must be an integer of at least four")
    if not isinstance(num_steps, Integral) or isinstance(num_steps, bool):
        errors.append(f"{prefix}config.num_steps must be an integer")
    elif isinstance(phase_length, Integral) and isinstance(n_phases, Integral):
        if int(num_steps) != int(phase_length) * int(n_phases):
            errors.append(f"{prefix}config.num_steps mismatch")
    learning_rate = config.get("learning_rate")
    epsilon = config.get("epsilon")
    if not _is_number(learning_rate) or not 0.0 < float(cast(Real, learning_rate)) <= 1.0:
        errors.append(f"{prefix}config.learning_rate must lie in (0, 1]")
    if not _is_number(epsilon) or not 0.0 <= float(cast(Real, epsilon)) <= 1.0:
        errors.append(f"{prefix}config.epsilon must lie in [0, 1]")
    if config.get("development_only") is not True:
        errors.append(f"{prefix}config.development_only must be true")
    if config.get("scientific_promotion_allowed") is not False:
        errors.append(f"{prefix}config.scientific_promotion_allowed must be false")
    if config.get("claim_thresholds_frozen") is not False:
        errors.append(f"{prefix}config.claim_thresholds_frozen must be false")
    return config, errors


def _validate_resource_payload(
    value: object,
    *,
    matched: bool,
    prefix: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}resource must be a mapping"]
    expected: dict[str, object]
    if matched:
        expected = {
            "helper_state_scalars": 10,
            "helper_state_bytes": 40,
            "beneficiary_state_scalars": 10,
            "beneficiary_state_bytes": 40,
            "total_state_scalars": 20,
            "total_state_bytes": 80,
            "resource_matched": True,
            "comparison_group": "matched_two_role_zero_initialized_tables",
            "resource_mismatch_disclosed": False,
        }
    else:
        expected = {
            "helper_state_scalars": 0,
            "helper_state_bytes": 0,
            "beneficiary_state_scalars": 10,
            "beneficiary_state_bytes": 40,
            "total_state_scalars": 10,
            "total_state_bytes": 40,
            "resource_matched": False,
            "comparison_group": "separate_resource_unmatched_beneficiary_alone",
            "resource_mismatch_disclosed": True,
        }
    return [
        f"{prefix}resource.{name} mismatch"
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    ]


def _validate_probe_payload(
    value: object,
    *,
    n_phases: int,
    prefix: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}probes must be a mapping"]
    errors: list[str] = []
    shape_by_name = {
        "phase_context": (n_phases,),
        "helper_messages": (n_phases, 2, 2),
        "beneficiary_actions": (n_phases, 2, 2),
        "joint_accuracy": (n_phases, 2),
        "helper_cue_dependence": (n_phases, 2),
        "beneficiary_message_dependence": (n_phases, 2),
        "message_flip_accuracy": (n_phases, 2),
        "message_flip_effect": (n_phases, 2),
        "best_constant_accuracy": (n_phases, 2),
        "gain_over_best_constant": (n_phases, 2),
        "context_forgetting": (n_phases, 2),
        "context_forgetting_valid": (n_phases, 2),
        "context_recovery": (n_phases, 2),
        "context_recovery_valid": (n_phases, 2),
    }
    arrays: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for name, expected_shape in shape_by_name.items():
        array = _payload_array(value.get(name), dtype=np.float64)
        if array is None or array.shape != expected_shape:
            errors.append(f"{prefix}probes.{name} shape/type mismatch")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{prefix}probes.{name} must be finite")
        else:
            arrays[name] = array
    phase_context = arrays.get("phase_context")
    if phase_context is not None and not np.array_equal(
        phase_context,
        np.arange(n_phases) % 2,
    ):
        errors.append(f"{prefix}probes.phase_context schedule mismatch")
    for name in (
        "helper_messages",
        "beneficiary_actions",
        "helper_cue_dependence",
        "beneficiary_message_dependence",
        "context_forgetting_valid",
        "context_recovery_valid",
    ):
        array = arrays.get(name)
        if array is not None and np.any((array != 0.0) & (array != 1.0)):
            errors.append(f"{prefix}probes.{name} must be binary")
    joint = arrays.get("joint_accuracy")
    flip = arrays.get("message_flip_accuracy")
    flip_effect = arrays.get("message_flip_effect")
    best_constant = arrays.get("best_constant_accuracy")
    gain = arrays.get("gain_over_best_constant")
    if joint is not None and flip is not None and flip_effect is not None:
        if not np.allclose(flip_effect, joint - flip, rtol=0.0, atol=1e-7):
            errors.append(f"{prefix}probes.message_flip_effect reconstruction mismatch")
    if best_constant is not None and not np.array_equal(
        best_constant,
        np.full((n_phases, 2), 0.5),
    ):
        errors.append(f"{prefix}probes.best_constant_accuracy must be exactly 0.5")
    if joint is not None and best_constant is not None and gain is not None:
        if not np.allclose(gain, joint - best_constant, rtol=0.0, atol=1e-7):
            errors.append(f"{prefix}probes.gain_over_best_constant reconstruction mismatch")
    return errors


def _validate_trace_payload(
    value: object,
    *,
    condition: str,
    phase_length: int,
    num_steps: int,
    learning_rate: float,
    probes_value: object,
    final_helper_value: object,
    final_beneficiary_value: object,
    prefix: str,
) -> tuple[
    list[str],
    np.ndarray[Any, np.dtype[np.float32]] | None,
    np.ndarray[Any, np.dtype[np.float32]] | None,
]:
    if not isinstance(value, Mapping):
        return [f"{prefix}trace must be a mapping"], None, None
    errors: list[str] = []
    names = tuple(field.name for field in dataclasses.fields(LearningPartnerPrimitiveTrace))
    arrays: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    float32_arrays: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    float32_names = {
        "helper_value_pre",
        "helper_value_post",
        "beneficiary_value_pre",
        "beneficiary_value_post",
    }
    for name in names:
        array = _payload_array(value.get(name), dtype=np.float64)
        if array is None or array.shape != (num_steps,):
            errors.append(f"{prefix}trace.{name} shape/type mismatch")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{prefix}trace.{name} must be finite")
        else:
            arrays[name] = array
            if name in float32_names:
                with np.errstate(over="ignore", invalid="ignore"):
                    float32_array = array.astype(np.float32)
                if not np.all(np.isfinite(float32_array)) or not np.array_equal(
                    array,
                    float32_array.astype(np.float64),
                ):
                    errors.append(f"{prefix}trace.{name} must contain exact float32 values")
                if np.any((float32_array < 0.0) | (float32_array > 1.0)):
                    errors.append(f"{prefix}trace.{name} must lie in [0, 1]")
                float32_arrays[name] = float32_array
    if len(arrays) != len(names):
        return errors, None, None
    if condition != BENEFICIARY_ALONE and condition not in MATCHED_CONDITIONS:
        errors.append(f"{prefix}trace condition is not recognized")
        return errors, None, None
    steps = np.arange(num_steps)
    if not np.array_equal(arrays["step_index"], steps):
        errors.append(f"{prefix}trace.step_index mismatch")
    expected_phases = steps // phase_length
    if not np.array_equal(arrays["phase_index"], expected_phases):
        errors.append(f"{prefix}trace.phase_index mismatch")
    if not np.array_equal(arrays["context"], expected_phases % 2):
        errors.append(f"{prefix}trace.context schedule mismatch")
    binary_valid = True
    for name in (
        "cue",
        "oracle_target",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "reward",
        "helper_write",
        "beneficiary_write",
    ):
        if np.any((arrays[name] != 0.0) & (arrays[name] != 1.0)):
            errors.append(f"{prefix}trace.{name} must be binary")
            binary_valid = False
    if not binary_valid:
        return errors, None, None
    expected_target = np.bitwise_xor(
        arrays["cue"].astype(np.int8),
        arrays["context"].astype(np.int8),
    )
    if not np.array_equal(arrays["oracle_target"], expected_target):
        errors.append(f"{prefix}trace.oracle_target mismatch")
    expected_reward = (arrays["beneficiary_action"] == expected_target).astype(np.float64)
    if not np.array_equal(arrays["reward"], expected_reward):
        errors.append(f"{prefix}trace.reward mismatch")
    if condition in (
        JOINT_ADAPTIVE,
        HELPER_FROZEN,
        BENEFICIARY_FROZEN,
        BOTH_FROZEN,
        ORACLE_HELPER,
    ) and not np.array_equal(arrays["delivered_message"], arrays["helper_message"]):
        errors.append(f"{prefix}trace direct-channel delivery mismatch")
    if condition == CONSTANT_ZERO_DELIVERY and np.any(arrays["delivered_message"] != 0.0):
        errors.append(f"{prefix}trace constant-zero delivery mismatch")
    if condition == CONSTANT_ONE_DELIVERY and np.any(arrays["delivered_message"] != 1.0):
        errors.append(f"{prefix}trace constant-one delivery mismatch")
    if condition == ORACLE_HELPER and not np.array_equal(
        arrays["helper_message"],
        expected_target,
    ):
        errors.append(f"{prefix}trace oracle-helper message mismatch")
    if condition == BENEFICIARY_ALONE and np.any(arrays["helper_message"] != 0.0):
        errors.append(f"{prefix}trace beneficiary-alone helper message mismatch")

    if condition == BENEFICIARY_ALONE:
        helper_write_expected = False
        beneficiary_write_expected = True
    else:
        spec = condition_spec(condition)
        helper_write_expected = spec.helper_write
        beneficiary_write_expected = spec.beneficiary_write
    if np.any(arrays["helper_write"] != float(helper_write_expected)):
        errors.append(f"{prefix}trace.helper_write mismatch")
    if np.any(arrays["beneficiary_write"] != float(beneficiary_write_expected)):
        errors.append(f"{prefix}trace.beneficiary_write mismatch")

    # Reconstruct both tables from the zero state.  This ties every selected
    # pre/post scalar, the final tables, and all phase probes to the trace
    # rather than trusting independently serialized summaries.  Crucially,
    # each validated trace post is committed.  Recomputing the table with one
    # host arithmetic order would drift from a truthful trace produced by the
    # other permitted float32 order.
    helper_table = np.zeros(ROLE_VALUE_SHAPE, dtype=np.float32)
    beneficiary_table = np.zeros(ROLE_VALUE_SHAPE, dtype=np.float32)
    helper_boundaries: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    beneficiary_boundaries: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    learning_rate_float32 = np.float32(learning_rate)
    replay_complete = True
    for step in range(num_steps):
        context = int(arrays["context"][step])
        cue = int(arrays["cue"][step])
        message = int(arrays["helper_message"][step])
        delivered = int(arrays["delivered_message"][step])
        action = int(arrays["beneficiary_action"][step])
        reward = np.float32(arrays["reward"][step])
        role_steps = (
            (
                "helper",
                helper_table,
                (context, cue, message),
                helper_write_expected,
            ),
            (
                "beneficiary",
                beneficiary_table,
                (context, delivered, action),
                beneficiary_write_expected,
            ),
        )
        step_valid = True
        for role, table, index, write_expected in role_steps:
            table_pre = table[index]
            trace_pre = float32_arrays[f"{role}_value_pre"][step]
            trace_post = float32_arrays[f"{role}_value_post"][step]
            if _float32_scalar_bits(trace_pre) != _float32_scalar_bits(table_pre):
                errors.append(
                    f"{prefix}trace.{role}_value_pre routing mismatch "
                    f"(bit-continuity) at step {step}"
                )
                step_valid = False
                continue
            if write_expected:
                unfused, fused = _portable_float32_update_candidates(
                    trace_pre,
                    reward,
                    learning_rate_float32,
                )
                allowed_bits = {
                    _float32_scalar_bits(unfused),
                    _float32_scalar_bits(fused),
                }
                if _float32_scalar_bits(trace_post) not in allowed_bits:
                    errors.append(
                        f"{prefix}trace.{role}_value_post portable float32 "
                        f"update mismatch at step {step}"
                    )
                    step_valid = False
                    continue
            elif _float32_scalar_bits(trace_post) != _float32_scalar_bits(trace_pre):
                errors.append(
                    f"{prefix}trace.{role}_value_post closed-write bit mismatch at step {step}"
                )
                step_valid = False
                continue
            table[index] = trace_post
        if not step_valid:
            replay_complete = False
            break
        if (step + 1) % phase_length == 0:
            helper_boundaries.append(helper_table.copy())
            beneficiary_boundaries.append(beneficiary_table.copy())

    final_helper_raw = _payload_array(final_helper_value, dtype=np.float64)
    final_helper = _payload_array(final_helper_value, dtype=np.float32)
    helper_final_valid = condition == BENEFICIARY_ALONE
    if condition != BENEFICIARY_ALONE:
        if (
            final_helper_raw is None
            or final_helper is None
            or final_helper.shape != ROLE_VALUE_SHAPE
            or not np.all(np.isfinite(final_helper))
            or not np.array_equal(
                final_helper_raw,
                final_helper.astype(np.float64),
            )
        ):
            errors.append(f"{prefix}final_helper_values must contain exact float32 values")
        elif not _float32_arrays_bit_equal(final_helper, helper_table):
            errors.append(f"{prefix}final_helper_values trace reconstruction mismatch")
        else:
            helper_final_valid = True
    final_beneficiary_raw = _payload_array(final_beneficiary_value, dtype=np.float64)
    final_beneficiary = _payload_array(final_beneficiary_value, dtype=np.float32)
    beneficiary_final_valid = False
    if (
        final_beneficiary_raw is None
        or final_beneficiary is None
        or final_beneficiary.shape != ROLE_VALUE_SHAPE
        or not np.all(np.isfinite(final_beneficiary))
        or not np.array_equal(
            final_beneficiary_raw,
            final_beneficiary.astype(np.float64),
        )
    ):
        errors.append(f"{prefix}final_beneficiary_values must contain exact float32 values")
    elif not _float32_arrays_bit_equal(final_beneficiary, beneficiary_table):
        errors.append(f"{prefix}final_beneficiary_values trace reconstruction mismatch")
    else:
        beneficiary_final_valid = True

    if len(helper_boundaries) == num_steps // phase_length and isinstance(probes_value, Mapping):
        expected_rows: dict[str, list[np.ndarray[Any, np.dtype[Any]]]] = {
            "helper_messages": [],
            "beneficiary_actions": [],
            "joint_accuracy": [],
            "helper_cue_dependence": [],
            "beneficiary_message_dependence": [],
            "message_flip_accuracy": [],
            "message_flip_effect": [],
            "best_constant_accuracy": [],
            "gain_over_best_constant": [],
        }
        contexts = np.arange(2, dtype=np.int8)[:, None]
        cues = np.arange(2, dtype=np.int8)[None, :]
        targets = np.bitwise_xor(contexts, cues)
        row_contexts = np.broadcast_to(contexts, (2, 2))
        for helper_values, beneficiary_values in zip(
            helper_boundaries,
            beneficiary_boundaries,
            strict=True,
        ):
            messages = np.argmax(helper_values, axis=-1).astype(np.int8)
            actions = np.argmax(beneficiary_values, axis=-1).astype(np.int8)
            direct_actions = actions[row_contexts, messages]
            flipped_actions = actions[row_contexts, 1 - messages]
            accuracy = np.mean(direct_actions == targets, axis=1).astype(np.float32)
            flip_accuracy = np.mean(flipped_actions == targets, axis=1).astype(np.float32)
            best_constant = np.maximum(
                np.mean(actions[:, 0, None] == targets, axis=1),
                np.mean(actions[:, 1, None] == targets, axis=1),
            ).astype(np.float32)
            expected_rows["helper_messages"].append(messages)
            expected_rows["beneficiary_actions"].append(actions)
            expected_rows["joint_accuracy"].append(accuracy)
            expected_rows["helper_cue_dependence"].append(
                (messages[:, 0] != messages[:, 1]).astype(np.float32)
            )
            expected_rows["beneficiary_message_dependence"].append(
                (actions[:, 0] != actions[:, 1]).astype(np.float32)
            )
            expected_rows["message_flip_accuracy"].append(flip_accuracy)
            expected_rows["message_flip_effect"].append(accuracy - flip_accuracy)
            expected_rows["best_constant_accuracy"].append(best_constant)
            expected_rows["gain_over_best_constant"].append(accuracy - best_constant)
        expected_probe_arrays = {name: np.stack(rows) for name, rows in expected_rows.items()}
        joint_accuracy = expected_probe_arrays["joint_accuracy"]
        forgetting, forgetting_valid, recovery, recovery_valid = _forgetting_and_recovery(
            jnp.asarray(joint_accuracy)
        )
        expected_probe_arrays.update(
            {
                "phase_context": np.arange(len(helper_boundaries), dtype=np.int8) % 2,
                "context_forgetting": np.asarray(forgetting),
                "context_forgetting_valid": np.asarray(forgetting_valid),
                "context_recovery": np.asarray(recovery),
                "context_recovery_valid": np.asarray(recovery_valid),
            }
        )
        for name, expected in expected_probe_arrays.items():
            actual = _payload_array(probes_value.get(name), dtype=expected.dtype)
            if actual is None or not np.array_equal(actual, expected):
                errors.append(f"{prefix}probes.{name} trace reconstruction mismatch")
    validated_helper = (
        helper_table.copy()
        if replay_complete and helper_final_valid and condition != BENEFICIARY_ALONE
        else None
    )
    validated_beneficiary = (
        beneficiary_table.copy() if replay_complete and beneficiary_final_valid else None
    )
    return errors, validated_helper, validated_beneficiary


def validate_learning_partner_development_payload(
    payload: object,
) -> tuple[str, ...]:
    """Validate serialized development semantics; never promote a claim.

    A compact trace-free summary may be serialized for display, but it cannot
    pass this validator because its condition semantics are not auditable.
    """

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ("payload must be a mapping",)
    errors.extend(_scope_errors(payload, ""))
    traces_included = payload.get("traces_included")
    if not isinstance(traces_included, bool):
        errors.append("traces_included must be boolean")
        traces_included = False
    if traces_included is not True:
        errors.append("full validation requires traces_included=true")
    if payload.get("seed_namespace") != "unregistered_development_seeds_consumed_on_run":
        errors.append("seed_namespace mismatch")
    config, config_errors = _validate_serialized_config(payload.get("config"), prefix="")
    errors.extend(config_errors)
    phase_length = (
        int(cast(Any, config["phase_length"])) if config is not None and not config_errors else 0
    )
    n_phases = int(cast(Any, config["n_phases"])) if config is not None and not config_errors else 0
    num_steps = (
        int(cast(Any, config["num_steps"])) if config is not None and not config_errors else 0
    )
    learning_rate = (
        float(cast(Any, config["learning_rate"]))
        if config is not None and not config_errors
        else 0.0
    )

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        errors.append("seeds must be a list containing at least two entries")
        seeds = []
    elif any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in seeds):
        errors.append("seeds must contain only integers")
    elif len(set(seeds)) != len(seeds):
        errors.append("seeds must be unique")

    condition_order = payload.get("matched_condition_order")
    if condition_order != list(MATCHED_CONDITIONS):
        errors.append("matched_condition_order mismatch")
    matched_runs = payload.get("matched_runs")
    expected_runs = len(seeds) * len(MATCHED_CONDITIONS)
    if not isinstance(matched_runs, list) or len(matched_runs) != expected_runs:
        errors.append("matched_runs count mismatch")
        matched_runs = []
    matched_byte_counts: set[int] = set()
    joint_final_pairs: list[
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]],
        ]
    ] = []
    for index, run in enumerate(matched_runs):
        prefix = f"matched_runs[{index}]."
        if not isinstance(run, Mapping):
            errors.append(f"matched_runs[{index}] must be a mapping")
            continue
        errors.extend(_scope_errors(run, prefix))
        if seeds:
            expected_seed = seeds[index // len(MATCHED_CONDITIONS)]
            expected_condition = MATCHED_CONDITIONS[index % len(MATCHED_CONDITIONS)]
            if run.get("seed") != expected_seed:
                errors.append(f"{prefix}seed/order mismatch")
            if run.get("condition") != expected_condition:
                errors.append(f"{prefix}condition/order mismatch")
        nested_config, nested_config_errors = _validate_serialized_config(
            run.get("config"),
            prefix=prefix,
        )
        errors.extend(nested_config_errors)
        if config is not None and nested_config is not None and nested_config != config:
            errors.append(f"{prefix}config differs from report config")
        errors.extend(_validate_resource_payload(run.get("resource"), matched=True, prefix=prefix))
        resource = run.get("resource")
        if isinstance(resource, Mapping) and isinstance(
            resource.get("total_state_bytes"), Integral
        ):
            matched_byte_counts.add(int(cast(Integral, resource["total_state_bytes"])))
        errors.extend(_validate_probe_payload(run.get("probes"), n_phases=n_phases, prefix=prefix))
        helper_values = _payload_array(run.get("final_helper_values"), dtype=np.float64)
        beneficiary_values = _payload_array(
            run.get("final_beneficiary_values"),
            dtype=np.float64,
        )
        if helper_values is None or helper_values.shape != ROLE_VALUE_SHAPE:
            errors.append(f"{prefix}final_helper_values shape/type mismatch")
        elif not np.all(np.isfinite(helper_values)):
            errors.append(f"{prefix}final_helper_values must be finite")
        if beneficiary_values is None or beneficiary_values.shape != ROLE_VALUE_SHAPE:
            errors.append(f"{prefix}final_beneficiary_values shape/type mismatch")
        elif not np.all(np.isfinite(beneficiary_values)):
            errors.append(f"{prefix}final_beneficiary_values must be finite")
        trace = run.get("trace")
        if traces_included is True and trace is None:
            errors.append(f"{prefix}trace is required when traces_included=true")
        elif traces_included is False and trace is not None:
            errors.append(f"{prefix}trace must be absent when traces_included=false")
        elif trace is not None and phase_length > 0 and num_steps > 0:
            trace_errors, trace_helper, trace_beneficiary = _validate_trace_payload(
                trace,
                condition=str(run.get("condition")),
                phase_length=phase_length,
                num_steps=num_steps,
                learning_rate=learning_rate,
                probes_value=run.get("probes"),
                final_helper_value=run.get("final_helper_values"),
                final_beneficiary_value=run.get("final_beneficiary_values"),
                prefix=prefix,
            )
            errors.extend(trace_errors)
            if (
                run.get("condition") == JOINT_ADAPTIVE
                and trace_helper is not None
                and trace_beneficiary is not None
            ):
                joint_final_pairs.append(
                    (
                        trace_helper.astype(np.float64),
                        trace_beneficiary.astype(np.float64),
                    )
                )
    if len(matched_byte_counts) > 1:
        errors.append("all matched controls must have identical state bytes")

    alone_runs = payload.get("beneficiary_alone_runs")
    if not isinstance(alone_runs, list) or len(alone_runs) != len(seeds):
        errors.append("beneficiary_alone_runs count mismatch")
        alone_runs = []
    for index, run in enumerate(alone_runs):
        prefix = f"beneficiary_alone_runs[{index}]."
        if not isinstance(run, Mapping):
            errors.append(f"beneficiary_alone_runs[{index}] must be a mapping")
            continue
        errors.extend(_scope_errors(run, prefix))
        if run.get("condition") != BENEFICIARY_ALONE:
            errors.append(f"{prefix}condition mismatch")
        if index < len(seeds) and run.get("seed") != seeds[index]:
            errors.append(f"{prefix}seed/order mismatch")
        nested_config, nested_config_errors = _validate_serialized_config(
            run.get("config"),
            prefix=prefix,
        )
        errors.extend(nested_config_errors)
        if config is not None and nested_config is not None and nested_config != config:
            errors.append(f"{prefix}config differs from report config")
        errors.extend(_validate_resource_payload(run.get("resource"), matched=False, prefix=prefix))
        if run.get("final_helper_values") is not None:
            errors.append(f"{prefix}final_helper_values must be null")
        beneficiary_values = _payload_array(
            run.get("final_beneficiary_values"),
            dtype=np.float64,
        )
        if beneficiary_values is None or beneficiary_values.shape != ROLE_VALUE_SHAPE:
            errors.append(f"{prefix}final_beneficiary_values shape/type mismatch")
        elif not np.all(np.isfinite(beneficiary_values)):
            errors.append(f"{prefix}final_beneficiary_values must be finite")
        errors.extend(_validate_probe_payload(run.get("probes"), n_phases=n_phases, prefix=prefix))
        trace = run.get("trace")
        if traces_included is True and trace is None:
            errors.append(f"{prefix}trace is required when traces_included=true")
        elif traces_included is False and trace is not None:
            errors.append(f"{prefix}trace must be absent when traces_included=false")
        elif trace is not None and phase_length > 0 and num_steps > 0:
            trace_errors, _, _ = _validate_trace_payload(
                trace,
                condition=BENEFICIARY_ALONE,
                phase_length=phase_length,
                num_steps=num_steps,
                learning_rate=learning_rate,
                probes_value=run.get("probes"),
                final_helper_value=run.get("final_helper_values"),
                final_beneficiary_value=run.get("final_beneficiary_values"),
                prefix=prefix,
            )
            errors.extend(trace_errors)

    cross_play = payload.get("cross_play")
    if not isinstance(cross_play, Mapping):
        errors.append("cross_play must be a mapping")
    else:
        matrix = _payload_array(cross_play.get("score_matrix"), dtype=np.float64)
        n_seeds = len(seeds)
        if matrix is None or matrix.shape != (n_seeds, n_seeds):
            errors.append("cross_play.score_matrix shape mismatch")
        elif not np.all(np.isfinite(matrix)):
            errors.append("cross_play.score_matrix must be finite")
        elif np.any((matrix < 0.0) | (matrix > 1.0)):
            errors.append("cross_play.score_matrix must lie in [0, 1]")
        raw_columns = _payload_array(cross_play.get("derangement_columns"), dtype=np.float64)
        expected_columns = (np.arange(n_seeds, dtype=np.int64) + 1) % max(n_seeds, 1)
        columns: np.ndarray[Any, np.dtype[np.int64]] | None = None
        if (
            raw_columns is None
            or raw_columns.shape != (n_seeds,)
            or not np.all(np.isfinite(raw_columns))
            or np.any(raw_columns != np.floor(raw_columns))
        ):
            errors.append("cross_play.derangement_columns shape mismatch")
        else:
            columns = raw_columns.astype(np.int64)
        if columns is not None and not np.array_equal(columns, expected_columns):
            if np.any(columns == np.arange(n_seeds)):
                errors.append("cross_play primary assignment must be a derangement")
            else:
                errors.append("cross_play must use the fixed cyclic derangement")
        if cross_play.get("seed_order") != seeds:
            errors.append("cross_play.seed_order mismatch")
        within = _payload_array(cross_play.get("within_dyad_scores"), dtype=np.float64)
        deranged = _payload_array(cross_play.get("deranged_scores"), dtype=np.float64)
        if matrix is not None and matrix.shape == (n_seeds, n_seeds):
            expected_within = np.diag(matrix)
            expected_deranged = matrix[np.arange(n_seeds), expected_columns]
            if within is None or not np.array_equal(within, expected_within):
                errors.append("cross_play.within_dyad_scores reconstruction mismatch")
            if deranged is None or not np.array_equal(deranged, expected_deranged):
                errors.append("cross_play.deranged_scores reconstruction mismatch")
            if len(joint_final_pairs) == n_seeds:
                reconstructed_matrix = np.empty_like(matrix)
                for helper_index, (helper_values, _) in enumerate(joint_final_pairs):
                    for beneficiary_index, (_, beneficiary_values) in enumerate(joint_final_pairs):
                        reconstructed_matrix[helper_index, beneficiary_index] = _cross_play_score(
                            jnp.asarray(helper_values, dtype=jnp.float32),
                            jnp.asarray(beneficiary_values, dtype=jnp.float32),
                        )
                if not np.array_equal(matrix, reconstructed_matrix):
                    errors.append("cross_play.score_matrix final-table reconstruction mismatch")
        if cross_play.get("primary_comparison") != (
            "within_dyad_mean_minus_fixed_cyclic_derangement_mean"
        ):
            errors.append("cross_play.primary_comparison mismatch")
        for name in (
            "within_dyad_mean",
            "fixed_derangement_mean",
            "within_minus_deranged",
        ):
            if not _is_number(cross_play.get(name)):
                errors.append(f"cross_play.{name} must be numeric")
        if within is not None and deranged is not None and within.size and deranged.size:
            within_mean = float(np.mean(within))
            deranged_mean = float(np.mean(deranged))
            if _is_number(cross_play.get("within_dyad_mean")) and not np.isclose(
                float(cast(Real, cross_play["within_dyad_mean"])),
                within_mean,
                rtol=0.0,
                atol=1e-12,
            ):
                errors.append("cross_play.within_dyad_mean reconstruction mismatch")
            if _is_number(cross_play.get("fixed_derangement_mean")) and not np.isclose(
                float(cast(Real, cross_play["fixed_derangement_mean"])),
                deranged_mean,
                rtol=0.0,
                atol=1e-12,
            ):
                errors.append("cross_play.fixed_derangement_mean reconstruction mismatch")
            if _is_number(cross_play.get("within_minus_deranged")) and not np.isclose(
                float(cast(Real, cross_play["within_minus_deranged"])),
                within_mean - deranged_mean,
                rtol=0.0,
                atol=1e-12,
            ):
                errors.append("cross_play.within_minus_deranged reconstruction mismatch")
    return tuple(errors)
