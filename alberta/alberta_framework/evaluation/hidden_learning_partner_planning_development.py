# mypy: disable-error-code="arg-type,attr-defined,call-arg,operator"
"""Development-only hidden co-learning dyad with causal one-step planning.

This module composes the existing binary signaling dyad, online behavior
model, and grounded joint-world model without exposing the world's recurring
context to either role.  Both contextual bandits always use physical row zero;
row one is an audited inactive row, not a task-indexed memory bank.

The helper is the focal planning agent.  Before emitting a message it predicts
the beneficiary's response to both possible delivered symbols, predicts the
grounded outcome of all four joint actions, and marginalizes the reward cells.
An independent Bernoulli gate randomizes whether the planner proposal or the
ordinary helper proposal is executed.  The beneficiary's realized action is
ordinary post-action feedback to the helper; context, phase, target, schedule
position, and oracle objects never cross the learner projection.

This is an executable mechanism study only.  It defines no artifact writer,
threshold, reserved seed campaign, CLI, or scientific-promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from numbers import Real
from typing import Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
)
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    GroundedJointWorldModelState,
)
from alberta_framework.core.signaling_bandit import (
    SignalingBanditAgent,
    SignalingBanditConfig,
    SignalingBanditState,
    decision_with_action,
    signaling_bandit_keys,
    signaling_bandit_resource_budget,
)
from alberta_framework.streams.learning_partner import (
    CONSTANT_ONE_CHANNEL,
    CONSTANT_ZERO_CHANNEL,
    DIRECT_CHANNEL,
    SHUFFLED_CHANNEL,
    LearningPartnerChannel,
    LearningPartnerTransition,
    LearningPartnerWorld,
    LearningPartnerWorldConfig,
    LearningPartnerWorldState,
    learning_partner_world_keys,
)

HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA = (
    "alberta.hidden-learning-partner-planning.development.v1"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
CLAIM_THRESHOLDS_FROZEN = False
ACCEPTANCE_STATUS = "descriptive_only_no_acceptance_gate"

JOINT_ADAPTIVE: Literal["joint_adaptive"] = "joint_adaptive"
HELPER_FROZEN: Literal["helper_frozen"] = "helper_frozen"
BENEFICIARY_FROZEN: Literal["beneficiary_frozen"] = "beneficiary_frozen"
BOTH_ROLES_FROZEN: Literal["both_roles_frozen"] = "both_roles_frozen"
BEHAVIOR_FROZEN: Literal["behavior_frozen"] = "behavior_frozen"
GROUNDED_FROZEN: Literal["grounded_frozen"] = "grounded_frozen"
BOTH_MODELS_FROZEN: Literal["both_models_frozen"] = "both_models_frozen"
PLANNER_NEVER_CONSUMED: Literal["planner_never_consumed"] = (
    "planner_never_consumed"
)
CONSTANT_ZERO_DELIVERY: Literal["constant_zero_delivery"] = (
    "constant_zero_delivery"
)
CONSTANT_ONE_DELIVERY: Literal["constant_one_delivery"] = "constant_one_delivery"
SHUFFLED_DELIVERY: Literal["shuffled_delivery"] = "shuffled_delivery"

type HiddenPlanningCondition = Literal[
    "joint_adaptive",
    "helper_frozen",
    "beneficiary_frozen",
    "both_roles_frozen",
    "behavior_frozen",
    "grounded_frozen",
    "both_models_frozen",
    "planner_never_consumed",
    "constant_zero_delivery",
    "constant_one_delivery",
    "shuffled_delivery",
]

MATCHED_CONDITIONS: tuple[HiddenPlanningCondition, ...] = (
    JOINT_ADAPTIVE,
    HELPER_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_ROLES_FROZEN,
    BEHAVIOR_FROZEN,
    GROUNDED_FROZEN,
    BOTH_MODELS_FROZEN,
    PLANNER_NEVER_CONSUMED,
    CONSTANT_ZERO_DELIVERY,
    CONSTANT_ONE_DELIVERY,
    SHUFFLED_DELIVERY,
)

_WORLD_ROOT_RNG_TAG = 0x4850574C  # "HPWL"
_LEARNER_ROOT_RNG_TAG = 0x48504C52  # "HPLR"
_BEHAVIOR_RNG_TAG = 0x48504248  # "HPBH"
_GROUNDED_RNG_TAG = 0x48504752  # "HPGR"
_PLANNER_RNG_TAG = 0x4850504C  # "HPPL"
_INTERVENTION_RNG_TAG = 0x4850494E  # "HPIN"
_CONFIG_TOKEN_BYTES = 32
_INT32_MAX = 2**31 - 1
_SHARED_ROW = 0
_N_ACTIONS = 2
_BEHAVIOR_FEATURE_DIM = 1
_GROUNDED_REPRESENTATION_DIM = 1
_GROUNDED_OBSERVATION_DIM = 1
_EXPECTED_SIGNALING_BYTES = 80
_EXPECTED_BEHAVIOR_BYTES = 48
_EXPECTED_GROUNDED_BYTES = 108
_EXPECTED_WORLD_BYTES = 32
_EXPECTED_TOTAL_BYTES = 321


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tree_array_nbytes(tree: object) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _tree_all_finite(tree: object) -> Array:
    checks: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            continue
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            checks.append(jnp.all(jnp.isfinite(leaf)))
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def _binary(value: Array) -> Array:
    item = jnp.asarray(value, dtype=jnp.int32)
    return (item == 0) | (item == 1)


def _signed_bit(value: Array) -> Array:
    return 2.0 * jnp.asarray(value, dtype=jnp.float32) - 1.0


def _probabilities_valid(probabilities: Array) -> Array:
    values = jnp.asarray(probabilities, dtype=jnp.float32)
    return (
        jnp.all(jnp.isfinite(values))
        & jnp.all(values >= 0.0)
        & jnp.all(values <= 1.0)
        & (jnp.abs(jnp.sum(values, axis=-1) - 1.0) <= 1e-5).all()
    )


def _exact(left: Array, right: Array) -> Array:
    return jnp.array_equal(jnp.asarray(left), jnp.asarray(right))


def _key_words(key: Array) -> Array:
    return jnp.asarray(jr.key_data(key), dtype=jnp.uint32)


def _host_tree_equal(left: object, right: object) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        if not np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf)):
            return False
    return True


def _host_lifetime_words_equal(value: object, count: int) -> bool:
    words = np.asarray(value)
    return (
        0 <= count < _INT32_MAX
        and words.shape == (2,)
        and words.dtype == np.dtype(np.uint32)
        and np.array_equal(words, np.asarray((0, count), dtype=np.uint32))
    )


@dataclasses.dataclass(frozen=True)
class HiddenLearningPartnerPlanningConfig:
    """Unfrozen configuration for one uninterrupted development life."""

    phase_length: int = 512
    n_phases: int = 6
    learning_rate: float = 0.1
    epsilon: float = 0.1
    behavior_step_size: float = 0.05
    grounded_step_size: float = 0.05

    def __post_init__(self) -> None:
        if type(self.phase_length) is not int or self.phase_length < 1:
            raise ValueError("phase_length must be a positive integer")
        if type(self.n_phases) is not int or self.n_phases < 4 or self.n_phases % 2:
            raise ValueError("n_phases must be an even integer of at least four")
        SignalingBanditConfig(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
        )
        for name, value in (
            ("behavior_step_size", self.behavior_step_size),
            ("grounded_step_size", self.grounded_step_size),
        ):
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def num_steps(self) -> int:
        return self.phase_length * self.n_phases

    @property
    def phase_diagnostic_window_steps(self) -> int:
        """Return the fixed quarter-phase window, bounded to 1--128 steps."""

        return max(1, min(128, self.phase_length // 4))

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_length": self.phase_length,
            "n_phases": self.n_phases,
            "num_steps": self.num_steps,
            "learning_rate": float(self.learning_rate),
            "epsilon": float(self.epsilon),
            "behavior_step_size": float(self.behavior_step_size),
            "grounded_step_size": float(self.grounded_step_size),
            "planner_consumption_probability": 0.5,
            "phase_diagnostic_window_steps": self.phase_diagnostic_window_steps,
            "phase_diagnostic_window_semantics": (
                "max(1,min(128,phase_length//4))"
            ),
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_thresholds_frozen": False,
        }


@dataclasses.dataclass(frozen=True)
class HiddenPlanningConditionSpec:
    """The only static interventions permitted in a matched condition."""

    channel: LearningPartnerChannel
    helper_write: bool
    beneficiary_write: bool
    behavior_write: bool
    grounded_write: bool
    planner_consumption: Literal["randomized", "always", "never"]


def condition_spec(condition: HiddenPlanningCondition | str) -> HiddenPlanningConditionSpec:
    """Resolve one exact resource-shape-neutral intervention."""

    full = HiddenPlanningConditionSpec(
        channel=DIRECT_CHANNEL,
        helper_write=True,
        beneficiary_write=True,
        behavior_write=True,
        grounded_write=True,
        planner_consumption="randomized",
    )
    if condition == JOINT_ADAPTIVE:
        return full
    if condition == HELPER_FROZEN:
        return dataclasses.replace(full, helper_write=False)
    if condition == BENEFICIARY_FROZEN:
        return dataclasses.replace(full, beneficiary_write=False)
    if condition == BOTH_ROLES_FROZEN:
        return dataclasses.replace(full, helper_write=False, beneficiary_write=False)
    if condition == BEHAVIOR_FROZEN:
        return dataclasses.replace(full, behavior_write=False)
    if condition == GROUNDED_FROZEN:
        return dataclasses.replace(full, grounded_write=False)
    if condition == BOTH_MODELS_FROZEN:
        return dataclasses.replace(full, behavior_write=False, grounded_write=False)
    if condition == PLANNER_NEVER_CONSUMED:
        return dataclasses.replace(full, planner_consumption="never")
    if condition == CONSTANT_ZERO_DELIVERY:
        return dataclasses.replace(full, channel=CONSTANT_ZERO_CHANNEL)
    if condition == CONSTANT_ONE_DELIVERY:
        return dataclasses.replace(full, channel=CONSTANT_ONE_CHANNEL)
    if condition == SHUFFLED_DELIVERY:
        return dataclasses.replace(full, channel=SHUFFLED_CHANNEL)
    raise ValueError(f"unknown hidden planning condition: {condition!r}")


@chex.dataclass(frozen=True)
class HiddenDyadPreObservation:
    """The complete learner-visible pre-message observation."""

    helper_cue: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenDyadFeedback:
    """Oracle-free post-action feedback visible to the focal helper."""

    helper_cue: Int[Array, ""]
    helper_message: Int[Array, ""]
    delivered_message: Int[Array, ""]
    beneficiary_action: Int[Array, ""]
    reward: Float[Array, ""]
    next_helper_cue: Int[Array, ""]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]


def strip_hidden_learning_partner_oracle(
    transition: LearningPartnerTransition,
) -> HiddenDyadFeedback:
    """Copy only the learner feedback surface, retaining no oracle reference."""

    return HiddenDyadFeedback(
        helper_cue=transition.observation.helper_cue,
        helper_message=transition.helper_message,
        delivered_message=transition.delivered_message,
        beneficiary_action=transition.beneficiary_action,
        reward=transition.reward,
        next_helper_cue=transition.next_observation.helper_cue,
        terminated=transition.terminated,
        discount=transition.discount,
    )


@chex.dataclass(frozen=True)
class HiddenLearningPartnerPlanningState:
    """Exact 321-byte persistent state of the matched composition."""

    world: LearningPartnerWorldState
    learner: SignalingBanditState
    behavior: BehaviorModelState
    grounded: GroundedJointWorldModelState
    planner_key: PRNGKeyArray
    intervention_key: PRNGKeyArray
    config_token: UInt[Array, " 32"]
    valid: Bool[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenLearningPartnerPlanningTrace:
    """Prequential primitives and explicit proposal/commit diagnostics."""

    active: Bool[Array, ""]
    accepted: Bool[Array, ""]
    step: Int[Array, ""]
    config_token_valid: Bool[Array, ""]
    learner_projection_valid: Bool[Array, ""]
    helper_cue: Int[Array, ""]
    next_helper_cue: Int[Array, ""]
    oracle_phase_index: Int[Array, ""]
    oracle_context: Int[Array, ""]
    oracle_target: Int[Array, ""]
    helper_context: Int[Array, ""]
    beneficiary_context: Int[Array, ""]
    ordinary_message: Int[Array, ""]
    planner_message: Int[Array, ""]
    planner_scores: Float[Array, " 2"]
    planner_gate_draw: Bool[Array, ""]
    planner_consumed: Bool[Array, ""]
    action_changed: Bool[Array, ""]
    helper_message: Int[Array, ""]
    delivered_message: Int[Array, ""]
    beneficiary_action: Int[Array, ""]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    behavior_candidate_probabilities: Float[Array, "2 2"]
    behavior_probabilities_pre: Float[Array, " 2"]
    behavior_probabilities_update: Float[Array, " 2"]
    behavior_action_probability: Float[Array, ""]
    behavior_nll: Float[Array, ""]
    behavior_prediction_bound: Bool[Array, ""]
    behavior_proposal_applied: Bool[Array, ""]
    behavior_committed_write: Bool[Array, ""]
    grounded_reward_cells: Float[Array, "2 2"]
    grounded_raw_prediction_pre: Float[Array, " 3"]
    grounded_raw_prediction_update: Float[Array, " 3"]
    grounded_reward_error: Float[Array, ""]
    grounded_prediction_bound: Bool[Array, ""]
    grounded_proposal_applied: Bool[Array, ""]
    grounded_committed_write: Bool[Array, ""]
    delivered_potential_actions: Int[Array, " 2"]
    delivered_potential_rewards: Float[Array, " 2"]
    potential_outcome_bound: Bool[Array, ""]
    intervention_bound: Bool[Array, ""]
    helper_write: Bool[Array, ""]
    beneficiary_write: Bool[Array, ""]
    helper_value_pre: Float[Array, ""]
    helper_value_post: Float[Array, ""]
    beneficiary_value_pre: Float[Array, ""]
    beneficiary_value_post: Float[Array, ""]
    shared_inactive_rows_unchanged: Bool[Array, ""]
    helper_key_before: UInt[Array, " 2"]
    helper_key_after: UInt[Array, " 2"]
    beneficiary_key_before: UInt[Array, " 2"]
    beneficiary_key_after: UInt[Array, " 2"]
    planner_key_before: UInt[Array, " 2"]
    planner_key_after: UInt[Array, " 2"]
    intervention_key_before: UInt[Array, " 2"]
    intervention_key_after: UInt[Array, " 2"]
    all_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenLearningPartnerPlanningStep:
    state: HiddenLearningPartnerPlanningState
    trace: HiddenLearningPartnerPlanningTrace


@dataclasses.dataclass(frozen=True)
class HiddenLearningPartnerPlanningResourceBudget:
    signaling_state_nbytes: int
    behavior_state_nbytes: int
    grounded_state_nbytes: int
    learner_model_state_nbytes: int
    world_state_nbytes: int
    planner_key_nbytes: int
    intervention_key_nbytes: int
    config_token_nbytes: int
    valid_nbytes: int
    step_count_nbytes: int
    metadata_state_nbytes: int
    total_state_nbytes: int
    replay_capacity: int
    exact_tree_match: bool

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenLearningPartnerPhaseDiagnostics:
    """Threshold-free evaluator summaries for each contiguous hidden phase.

    ``switch_cost`` is previous trailing reward minus current leading reward,
    so positive values denote an immediate reward loss after a switch.
    ``recurrence_savings`` is current leading reward minus the leading reward
    of the latest earlier phase with the same hidden context, so positive
    values denote better recurrence entry. Invalid comparisons use exact zero
    plus an explicit false validity bit and zero comparison count.
    """

    n_phases: int
    window_steps: int
    phase_index: tuple[int, ...]
    hidden_context: tuple[int, ...]
    phase_counts: tuple[int, ...]
    phase_valid: tuple[bool, ...]
    mean_reward: tuple[float, ...]
    leading_reward: tuple[float, ...]
    leading_counts: tuple[int, ...]
    trailing_reward: tuple[float, ...]
    trailing_counts: tuple[int, ...]
    behavior_mean_nll: tuple[float, ...]
    grounded_reward_mse: tuple[float, ...]
    switch_cost: tuple[float, ...]
    switch_cost_valid: tuple[bool, ...]
    switch_cost_counts: tuple[int, ...]
    recurrence_reference_phase: tuple[int, ...]
    recurrence_savings: tuple[float, ...]
    recurrence_savings_valid: tuple[bool, ...]
    recurrence_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenLearningPartnerPlanningMetrics:
    """Threshold-free raw summaries reconstructed from primitive traces."""

    num_steps: int
    mean_reward: float
    behavior_mean_nll: float
    behavior_mean_brier: float
    grounded_reward_mse: float
    grounded_next_observation_mse: float
    planner_consumption_rate: float
    action_change_rate: float
    eligible_steps: int
    treated_eligible_steps: int
    control_eligible_steps: int
    randomized_effect: float
    randomized_effect_valid: bool
    potential_effect: float
    potential_effect_valid: bool
    phase_diagnostics: HiddenLearningPartnerPhaseDiagnostics

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenLearningPartnerPlanningRun:
    """One in-memory development run; not an evidence artifact."""

    condition: HiddenPlanningCondition
    seed: int
    config: HiddenLearningPartnerPlanningConfig
    initial_state: HiddenLearningPartnerPlanningState
    final_state: HiddenLearningPartnerPlanningState
    trace: HiddenLearningPartnerPlanningTrace
    metrics: HiddenLearningPartnerPlanningMetrics
    resource: HiddenLearningPartnerPlanningResourceBudget


class HiddenLearningPartnerPlanningBridge:
    """JIT-safe composition for one static matched intervention."""

    def __init__(
        self,
        config: HiddenLearningPartnerPlanningConfig | None = None,
        condition: HiddenPlanningCondition = JOINT_ADAPTIVE,
    ) -> None:
        self._config = config or HiddenLearningPartnerPlanningConfig()
        if condition not in MATCHED_CONDITIONS:
            raise ValueError(f"unknown hidden planning condition: {condition!r}")
        self._condition = condition
        self._spec = condition_spec(condition)
        self._world = LearningPartnerWorld(
            LearningPartnerWorldConfig(self._config.phase_length)
        )
        self._learner = SignalingBanditAgent(
            SignalingBanditConfig(
                learning_rate=self._config.learning_rate,
                epsilon=self._config.epsilon,
            )
        )
        self._behavior = BehaviorModel(
            BehaviorModelConfig(
                n_actions=_N_ACTIONS,
                step_size=self._config.behavior_step_size,
            )
        )
        self._grounded = GroundedJointWorldModel(
            GroundedJointWorldModelConfig(
                representation_dim=_GROUNDED_REPRESENTATION_DIM,
                target_observation_dim=_GROUNDED_OBSERVATION_DIM,
                n_focal_actions=_N_ACTIONS,
                n_partner_actions=_N_ACTIONS,
                step_size=self._config.grounded_step_size,
                initialization_scale=0.01,
            )
        )
        token_payload = {
            "schema": HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA,
            "config": self._config.to_dict(),
            "condition": self._condition,
            "condition_spec": dataclasses.asdict(self._spec),
        }
        token = hashlib.sha256(_canonical_json_bytes(token_payload)).digest()
        self._config_token = jnp.asarray(tuple(token), dtype=jnp.uint8)

    @property
    def config(self) -> HiddenLearningPartnerPlanningConfig:
        return self._config

    @property
    def condition(self) -> HiddenPlanningCondition:
        return self._condition

    @property
    def spec(self) -> HiddenPlanningConditionSpec:
        return self._spec

    @property
    def config_token(self) -> Array:
        return self._config_token

    @staticmethod
    def _require_root_key(key: Array) -> None:
        if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
            getattr(key, "dtype", None), jax.dtypes.prng_key
        ):
            raise TypeError("root_key must be a scalar typed PRNG key")
        if str(jr.key_impl(key)) != "threefry2x32":
            raise TypeError("root_key must use threefry2x32")
        data = jr.key_data(key)
        if data.shape != (2,) or data.dtype != jnp.uint32:
            raise TypeError("root_key must have an exact uint32[2] backing")

    def initialize(self, root_key: Array) -> HiddenLearningPartnerPlanningState:
        """Initialize every matched component from a stable named stream."""

        self._require_root_key(root_key)
        world = self._world.init(
            learning_partner_world_keys(jr.fold_in(root_key, _WORLD_ROOT_RNG_TAG))
        )
        learner = self._learner.init(
            signaling_bandit_keys(jr.fold_in(root_key, _LEARNER_ROOT_RNG_TAG))
        )
        behavior = self._behavior.init(
            _BEHAVIOR_FEATURE_DIM,
            jr.fold_in(root_key, _BEHAVIOR_RNG_TAG),
        )
        grounded = self._grounded.init(jr.fold_in(root_key, _GROUNDED_RNG_TAG))
        state = HiddenLearningPartnerPlanningState(
            world=world,
            learner=learner,
            behavior=behavior,
            grounded=grounded,
            planner_key=jr.fold_in(root_key, _PLANNER_RNG_TAG),
            intervention_key=jr.fold_in(root_key, _INTERVENTION_RNG_TAG),
            config_token=self._config_token,
            valid=jnp.asarray(True, dtype=jnp.bool_),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        budget = self.resource_budget(state)
        if not budget.exact_tree_match or budget.total_state_nbytes != _EXPECTED_TOTAL_BYTES:
            raise ValueError("initialized state violates the exact 321-byte contract")
        return state

    def resource_budget(
        self,
        state: HiddenLearningPartnerPlanningState,
    ) -> HiddenLearningPartnerPlanningResourceBudget:
        """Measure every persistent array and require declared component sizes."""

        signaling = signaling_bandit_resource_budget(state.learner).state_bytes
        behavior = _tree_array_nbytes(state.behavior)
        grounded = _tree_array_nbytes(state.grounded)
        declared_behavior = self._behavior.resource_budget(
            _BEHAVIOR_FEATURE_DIM
        ).state_nbytes
        declared_grounded = self._grounded.resource_budget.state_nbytes
        world = _tree_array_nbytes(state.world)
        planner = _tree_array_nbytes(state.planner_key)
        intervention = _tree_array_nbytes(state.intervention_key)
        token = _tree_array_nbytes(state.config_token)
        valid = _tree_array_nbytes(state.valid)
        count = _tree_array_nbytes(state.step_count)
        learner_model = signaling + behavior + grounded
        metadata = planner + intervention + token + valid + count
        total = learner_model + world + metadata
        exact = total == _tree_array_nbytes(state)
        exact = exact and (
            signaling == _EXPECTED_SIGNALING_BYTES
            and behavior == _EXPECTED_BEHAVIOR_BYTES
            and behavior == declared_behavior
            and grounded == _EXPECTED_GROUNDED_BYTES
            and grounded == declared_grounded
            and world == _EXPECTED_WORLD_BYTES
            and planner == 8
            and intervention == 8
            and token == _CONFIG_TOKEN_BYTES
            and valid == 1
            and count == 4
            and total == _EXPECTED_TOTAL_BYTES
        )
        return HiddenLearningPartnerPlanningResourceBudget(
            signaling_state_nbytes=signaling,
            behavior_state_nbytes=behavior,
            grounded_state_nbytes=grounded,
            learner_model_state_nbytes=learner_model,
            world_state_nbytes=world,
            planner_key_nbytes=planner,
            intervention_key_nbytes=intervention,
            config_token_nbytes=token,
            valid_nbytes=valid,
            step_count_nbytes=count,
            metadata_state_nbytes=metadata,
            total_state_nbytes=total,
            replay_capacity=0,
            exact_tree_match=exact,
        )

    def _candidate_behavior_probabilities(
        self,
        state: BehaviorModelState,
    ) -> tuple[Array, Array]:
        """Return delivered-symbol predictions and channel-marginal planning beliefs."""

        delivered = jnp.stack(
            (
                self._behavior.predict_probabilities(
                    state, jnp.asarray((-1.0,), dtype=jnp.float32)
                ),
                self._behavior.predict_probabilities(
                    state, jnp.asarray((1.0,), dtype=jnp.float32)
                ),
            )
        )
        # Never peek at the pending shuffled-channel key. Hypothetical shuffled
        # delivery is marginalized under its declared fair distribution.
        if self._spec.channel == DIRECT_CHANNEL:
            planning = delivered
        elif self._spec.channel == CONSTANT_ZERO_CHANNEL:
            planning = jnp.stack((delivered[0], delivered[0]))
        elif self._spec.channel == CONSTANT_ONE_CHANNEL:
            planning = jnp.stack((delivered[1], delivered[1]))
        elif self._spec.channel == SHUFFLED_CHANNEL:
            marginal = 0.5 * (delivered[0] + delivered[1])
            planning = jnp.stack((marginal, marginal))
        else:  # pragma: no cover - condition_spec is closed above.
            raise ValueError(f"unknown channel: {self._spec.channel!r}")
        return delivered, planning

    def _grounded_cells(
        self,
        state: GroundedJointWorldModelState,
        representation: Array,
    ) -> tuple[Array, Array, Array]:
        predictions = (
            self._grounded.predict(
                state,
                representation,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            self._grounded.predict(
                state,
                representation,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
            ),
            self._grounded.predict(
                state,
                representation,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            self._grounded.predict(
                state,
                representation,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
            ),
        )
        rewards = jnp.asarray(
            ((predictions[0].reward, predictions[1].reward),
             (predictions[2].reward, predictions[3].reward)),
            dtype=jnp.float32,
        )
        raw = jnp.stack(tuple(item.raw_predictions for item in predictions)).reshape(
            (2, 2, 3)
        )
        valid = jnp.stack(tuple(item.valid for item in predictions))
        return rewards, raw, valid

    @staticmethod
    def _neutral_trace(
        state: HiddenLearningPartnerPlanningState,
    ) -> HiddenLearningPartnerPlanningTrace:
        """Return the sole fixed sentinel emitted after the latch closes."""

        false = jnp.asarray(False, dtype=jnp.bool_)
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        zero_f = jnp.asarray(0.0, dtype=jnp.float32)
        zero_key = jnp.zeros((2,), dtype=jnp.uint32)
        zero_actions = jnp.zeros((2,), dtype=jnp.int32)
        zero_pair = jnp.zeros((2,), dtype=jnp.float32)
        neutral_probabilities = jnp.full((2,), 0.5, dtype=jnp.float32)
        return HiddenLearningPartnerPlanningTrace(
            active=false,
            accepted=false,
            step=state.step_count,
            config_token_valid=false,
            learner_projection_valid=false,
            helper_cue=zero_i,
            next_helper_cue=zero_i,
            oracle_phase_index=zero_i,
            oracle_context=zero_i,
            oracle_target=zero_i,
            helper_context=zero_i,
            beneficiary_context=zero_i,
            ordinary_message=zero_i,
            planner_message=zero_i,
            planner_scores=zero_pair,
            planner_gate_draw=false,
            planner_consumed=false,
            action_changed=false,
            helper_message=zero_i,
            delivered_message=zero_i,
            beneficiary_action=zero_i,
            reward=zero_f,
            discount=zero_f,
            behavior_candidate_probabilities=jnp.stack(
                (neutral_probabilities, neutral_probabilities)
            ),
            behavior_probabilities_pre=neutral_probabilities,
            behavior_probabilities_update=neutral_probabilities,
            behavior_action_probability=zero_f,
            behavior_nll=zero_f,
            behavior_prediction_bound=false,
            behavior_proposal_applied=false,
            behavior_committed_write=false,
            grounded_reward_cells=jnp.zeros((2, 2), dtype=jnp.float32),
            grounded_raw_prediction_pre=jnp.zeros((3,), dtype=jnp.float32),
            grounded_raw_prediction_update=jnp.zeros((3,), dtype=jnp.float32),
            grounded_reward_error=zero_f,
            grounded_prediction_bound=false,
            grounded_proposal_applied=false,
            grounded_committed_write=false,
            delivered_potential_actions=zero_actions,
            delivered_potential_rewards=zero_pair,
            potential_outcome_bound=false,
            intervention_bound=false,
            helper_write=false,
            beneficiary_write=false,
            helper_value_pre=zero_f,
            helper_value_post=zero_f,
            beneficiary_value_pre=zero_f,
            beneficiary_value_post=zero_f,
            shared_inactive_rows_unchanged=false,
            helper_key_before=zero_key,
            helper_key_after=zero_key,
            beneficiary_key_before=zero_key,
            beneficiary_key_after=zero_key,
            planner_key_before=zero_key,
            planner_key_after=zero_key,
            intervention_key_before=zero_key,
            intervention_key_after=zero_key,
            all_finite=false,
        )

    def _blocked_step(
        self,
        state: HiddenLearningPartnerPlanningState,
    ) -> HiddenLearningPartnerPlanningStep:
        return HiddenLearningPartnerPlanningStep(
            state=state,
            trace=self._neutral_trace(state),
        )

    def step(
        self,
        state: HiddenLearningPartnerPlanningState,
    ) -> HiddenLearningPartnerPlanningStep:
        """Advance an active state or emit the fixed latched-invalid sentinel."""

        return cast(
            HiddenLearningPartnerPlanningStep,
            jax.lax.cond(
                state.valid,
                lambda _: self._active_step(state),
                lambda _: self._blocked_step(state),
                operand=None,
            ),
        )

    def _active_step(
        self,
        state: HiddenLearningPartnerPlanningState,
    ) -> HiddenLearningPartnerPlanningStep:
        """Propose one complete transition and atomically commit or latch invalid."""

        observation = self._world.observe(state.world)
        learner_observation = HiddenDyadPreObservation(helper_cue=observation.helper_cue)
        representation = jnp.reshape(
            _signed_bit(learner_observation.helper_cue),
            (1,),
        )
        helper_decision = self._learner.select_helper(
            state.learner.helper,
            jnp.asarray(_SHARED_ROW, dtype=jnp.int32),
            learner_observation.helper_cue,
        )
        ordinary_message = helper_decision.action

        delivered_behavior, planning_behavior = self._candidate_behavior_probabilities(
            state.behavior
        )
        reward_cells, grounded_raw_cells, grounded_cell_valid = self._grounded_cells(
            state.grounded,
            representation,
        )
        planner_scores = jnp.sum(planning_behavior * reward_cells, axis=1)
        next_planner_key, planner_tie_key = jr.split(state.planner_key)
        tie_message = jr.randint(
            planner_tie_key,
            (),
            0,
            _N_ACTIONS,
            dtype=jnp.int32,
        )
        planner_message = jnp.where(
            planner_scores[0] == planner_scores[1],
            tie_message,
            jnp.argmax(planner_scores).astype(jnp.int32),
        )

        next_intervention_key, gate_key = jr.split(state.intervention_key)
        planner_gate_draw = jr.bernoulli(gate_key, 0.5)
        if self._spec.planner_consumption == "randomized":
            planner_consumed = planner_gate_draw
        elif self._spec.planner_consumption == "always":
            planner_consumed = jnp.asarray(True, dtype=jnp.bool_)
        else:
            planner_consumed = jnp.asarray(False, dtype=jnp.bool_)
        helper_message = jnp.where(
            planner_consumed,
            planner_message,
            ordinary_message,
        ).astype(jnp.int32)
        executed_helper_decision = decision_with_action(helper_decision, helper_message)

        # Actual channel delivery is resolved only after the helper action is
        # fixed. In particular, no pre-message operation reads channel_key.
        delivered_message = self._world.deliver(
            state.world,
            helper_message,
            self._spec.channel,
        )
        behavior_probabilities_pre = delivered_behavior[delivered_message]
        beneficiary_decision = self._learner.select_beneficiary(
            state.learner.beneficiary,
            jnp.asarray(_SHARED_ROW, dtype=jnp.int32),
            delivered_message,
        )

        potential_zero = self._learner.select_beneficiary(
            state.learner.beneficiary,
            jnp.asarray(_SHARED_ROW, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        potential_one = self._learner.select_beneficiary(
            state.learner.beneficiary,
            jnp.asarray(_SHARED_ROW, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )
        delivered_potential_actions = jnp.stack(
            (potential_zero.action, potential_one.action)
        ).astype(jnp.int32)
        grounded_raw_prediction_pre = grounded_raw_cells[
            helper_message,
            beneficiary_decision.action,
        ]

        transition, proposed_world = self._world.step_with_delivery(
            state.world,
            helper_message,
            delivered_message,
            beneficiary_decision.action,
        )
        feedback = strip_hidden_learning_partner_oracle(transition)

        learner_update = self._learner.update(
            state.learner,
            executed_helper_decision,
            beneficiary_decision,
            feedback.reward,
            helper_write=self._spec.helper_write,
            beneficiary_write=self._spec.beneficiary_write,
        )
        behavior_update = self._behavior.update(
            state.behavior,
            jnp.reshape(_signed_bit(feedback.delivered_message), (1,)),
            feedback.beneficiary_action,
        )
        grounded_update = self._grounded.update(
            state.grounded,
            representation,
            feedback.helper_message,
            feedback.beneficiary_action,
            jnp.reshape(_signed_bit(feedback.next_helper_cue), (1,)),
            feedback.reward,
            feedback.discount,
        )
        proposed_behavior = cast(
            BehaviorModelState,
            jax.lax.cond(
                jnp.asarray(self._spec.behavior_write),
                lambda _: behavior_update.state,
                lambda _: state.behavior,
                operand=None,
            ),
        )
        proposed_grounded = cast(
            GroundedJointWorldModelState,
            jax.lax.cond(
                jnp.asarray(self._spec.grounded_write),
                lambda _: grounded_update.state,
                lambda _: state.grounded,
                operand=None,
            ),
        )

        config_token_valid = _exact(state.config_token, self._config_token)
        zero_u32 = jnp.asarray(0, dtype=jnp.uint32)
        zero_i32 = jnp.asarray(0, dtype=jnp.int32)
        behavior_committed_count = jnp.where(
            jnp.asarray(self._spec.behavior_write),
            state.step_count,
            zero_i32,
        ).astype(jnp.int32)
        grounded_committed_count = jnp.where(
            jnp.asarray(self._spec.grounded_write),
            state.step_count,
            zero_i32,
        ).astype(jnp.int32)
        expected_behavior_words = jnp.stack(
            (zero_u32, behavior_committed_count.astype(jnp.uint32))
        )
        expected_grounded_words = jnp.stack(
            (zero_u32, grounded_committed_count.astype(jnp.uint32))
        )
        expected_behavior_proposal_words = jnp.stack(
            (zero_u32, (behavior_committed_count + 1).astype(jnp.uint32))
        )
        expected_grounded_proposal_words = jnp.stack(
            (zero_u32, (grounded_committed_count + 1).astype(jnp.uint32))
        )
        counters_valid = (
            (state.step_count >= 0)
            & (state.step_count < _INT32_MAX)
            & (state.world.step_count == state.step_count)
            & (state.behavior.step_count == behavior_committed_count)
            & _exact(state.behavior.step_words, expected_behavior_words)
            & (state.grounded.update_count == grounded_committed_count)
            & _exact(state.grounded.update_words, expected_grounded_words)
        )
        shared_rows_pre_valid = (
            jnp.all(state.learner.helper.values[1] == 0.0)
            & jnp.all(state.learner.beneficiary.values[1] == 0.0)
        )
        shared_inactive_rows_unchanged = (
            shared_rows_pre_valid
            & _exact(learner_update.state.helper.values[1], state.learner.helper.values[1])
            & _exact(
                learner_update.state.beneficiary.values[1],
                state.learner.beneficiary.values[1],
            )
        )
        learner_projection_valid = (
            _binary(feedback.helper_cue)
            & _binary(feedback.helper_message)
            & _binary(feedback.delivered_message)
            & _binary(feedback.beneficiary_action)
            & _binary(feedback.next_helper_cue)
            & jnp.isfinite(feedback.reward)
            & ((feedback.reward == 0.0) | (feedback.reward == 1.0))
            & ~feedback.terminated
            & (feedback.discount == 1.0)
        )
        behavior_prediction_bound = _exact(
            behavior_probabilities_pre,
            behavior_update.probabilities,
        )
        grounded_prediction_bound = _exact(
            grounded_raw_prediction_pre,
            grounded_update.prediction.raw_predictions,
        )
        behavior_proposal_applied = (
            _probabilities_valid(delivered_behavior)
            & _probabilities_valid(planning_behavior)
            & _probabilities_valid(behavior_update.probabilities)
            & behavior_update.lifetime_counter_valid
            & behavior_update.lifetime_capacity_available
            & behavior_update.update_applied
            & _exact(behavior_update.pre_step_words, expected_behavior_words)
            & _exact(
                behavior_update.post_step_words,
                expected_behavior_proposal_words,
            )
            & _exact(
                behavior_update.state.step_words,
                expected_behavior_proposal_words,
            )
            & (behavior_update.state.step_count == state.behavior.step_count + 1)
            & _tree_all_finite(behavior_update.state)
        )
        grounded_proposal_applied = (
            grounded_update.diagnostics.lifetime_counter_valid
            & grounded_update.diagnostics.capacity_available
            & grounded_update.diagnostics.applied
            & ~grounded_update.diagnostics.rejected
            & grounded_update.update_applied
            & _exact(grounded_update.pre_update_words, expected_grounded_words)
            & _exact(
                grounded_update.post_update_words,
                expected_grounded_proposal_words,
            )
            & _exact(
                grounded_update.state.update_words,
                expected_grounded_proposal_words,
            )
            & (
                grounded_update.state.update_count
                == state.grounded.update_count + 1
            )
        )
        intervention_bound = (
            _binary(ordinary_message)
            & _binary(planner_message)
            & _binary(helper_message)
            & (
                helper_message
                == jnp.where(planner_consumed, planner_message, ordinary_message)
            )
        )
        learner_path_finite = (
            _tree_all_finite(state.learner)
            & _tree_all_finite(learner_update.state)
            & _tree_all_finite(state.behavior)
            & _tree_all_finite(state.grounded)
            & jnp.all(jnp.isfinite(planner_scores))
            & jnp.all(jnp.isfinite(reward_cells))
            & jnp.all(jnp.isfinite(grounded_raw_prediction_pre))
            & jnp.isfinite(feedback.reward)
        )
        proposal_valid = (
            config_token_valid
            & counters_valid
            & shared_inactive_rows_unchanged
            & learner_projection_valid
            & (helper_decision.context == _SHARED_ROW)
            & (beneficiary_decision.context == _SHARED_ROW)
            & _binary(delivered_message)
            & jnp.all(grounded_cell_valid)
            & behavior_prediction_bound
            & grounded_prediction_bound
            & behavior_proposal_applied
            & grounded_proposal_applied
            & intervention_bound
            & learner_path_finite
            & (proposed_world.step_count == state.world.step_count + 1)
        )
        accepted = state.valid & proposal_valid
        proposed_state = HiddenLearningPartnerPlanningState(
            world=proposed_world,
            learner=learner_update.state,
            behavior=proposed_behavior,
            grounded=proposed_grounded,
            planner_key=next_planner_key,
            intervention_key=next_intervention_key,
            config_token=state.config_token,
            valid=jnp.asarray(True, dtype=jnp.bool_),
            step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
        )
        rejected_state = HiddenLearningPartnerPlanningState(
            world=state.world,
            learner=state.learner,
            behavior=state.behavior,
            grounded=state.grounded,
            planner_key=state.planner_key,
            intervention_key=state.intervention_key,
            config_token=state.config_token,
            valid=jnp.asarray(False, dtype=jnp.bool_),
            step_count=state.step_count,
        )
        next_state = cast(
            HiddenLearningPartnerPlanningState,
            jax.lax.cond(
                accepted,
                lambda _: proposed_state,
                lambda _: rejected_state,
                operand=None,
            ),
        )

        # Oracle-derived potential outcomes are diagnostic only and cannot
        # alter the learner/world commit decision above.
        delivered_potential_rewards = (
            delivered_potential_actions == transition.oracle.target
        ).astype(jnp.float32)
        potential_outcome_bound = (
            _binary(transition.oracle.target)
            & (
                feedback.reward
                == delivered_potential_rewards[feedback.delivered_message]
            )
            & (
                feedback.beneficiary_action
                == delivered_potential_actions[feedback.delivered_message]
            )
        )
        behavior_nll = -jnp.log(
            jnp.maximum(behavior_update.action_probability, jnp.float32(1e-6))
        )
        trace = HiddenLearningPartnerPlanningTrace(
            active=state.valid,
            accepted=accepted,
            step=state.step_count,
            config_token_valid=config_token_valid,
            learner_projection_valid=learner_projection_valid,
            helper_cue=feedback.helper_cue,
            next_helper_cue=feedback.next_helper_cue,
            oracle_phase_index=transition.oracle.phase_index,
            oracle_context=transition.oracle.context,
            oracle_target=transition.oracle.target,
            helper_context=helper_decision.context,
            beneficiary_context=beneficiary_decision.context,
            ordinary_message=ordinary_message,
            planner_message=planner_message,
            planner_scores=planner_scores,
            planner_gate_draw=planner_gate_draw,
            planner_consumed=planner_consumed,
            action_changed=planner_message != ordinary_message,
            helper_message=feedback.helper_message,
            delivered_message=feedback.delivered_message,
            beneficiary_action=feedback.beneficiary_action,
            reward=feedback.reward,
            discount=feedback.discount,
            behavior_candidate_probabilities=planning_behavior,
            behavior_probabilities_pre=behavior_probabilities_pre,
            behavior_probabilities_update=behavior_update.probabilities,
            behavior_action_probability=behavior_update.action_probability,
            behavior_nll=behavior_nll,
            behavior_prediction_bound=behavior_prediction_bound,
            behavior_proposal_applied=behavior_proposal_applied,
            behavior_committed_write=(accepted & self._spec.behavior_write),
            grounded_reward_cells=reward_cells,
            grounded_raw_prediction_pre=grounded_raw_prediction_pre,
            grounded_raw_prediction_update=grounded_update.prediction.raw_predictions,
            grounded_reward_error=(
                grounded_raw_prediction_pre[_GROUNDED_OBSERVATION_DIM]
                - feedback.reward
            ),
            grounded_prediction_bound=grounded_prediction_bound,
            grounded_proposal_applied=grounded_proposal_applied,
            grounded_committed_write=(accepted & self._spec.grounded_write),
            delivered_potential_actions=delivered_potential_actions,
            delivered_potential_rewards=delivered_potential_rewards,
            potential_outcome_bound=potential_outcome_bound,
            intervention_bound=intervention_bound,
            helper_write=jnp.asarray(self._spec.helper_write) & accepted,
            beneficiary_write=jnp.asarray(self._spec.beneficiary_write) & accepted,
            helper_value_pre=learner_update.helper_value_pre,
            helper_value_post=learner_update.helper_value_post,
            beneficiary_value_pre=learner_update.beneficiary_value_pre,
            beneficiary_value_post=learner_update.beneficiary_value_post,
            shared_inactive_rows_unchanged=shared_inactive_rows_unchanged,
            helper_key_before=_key_words(state.learner.helper.key),
            helper_key_after=_key_words(learner_update.state.helper.key),
            beneficiary_key_before=_key_words(state.learner.beneficiary.key),
            beneficiary_key_after=_key_words(learner_update.state.beneficiary.key),
            planner_key_before=_key_words(state.planner_key),
            planner_key_after=_key_words(next_planner_key),
            intervention_key_before=_key_words(state.intervention_key),
            intervention_key_after=_key_words(next_intervention_key),
            all_finite=learner_path_finite,
        )
        return HiddenLearningPartnerPlanningStep(state=next_state, trace=trace)


def _phase_diagnostics_from_trace(
    trace: HiddenLearningPartnerPlanningTrace,
    *,
    phase_length: int,
    n_phases: int,
) -> HiddenLearningPartnerPhaseDiagnostics:
    """Reconstruct bounded-window phase diagnostics from primitive arrays."""

    phase_trace = np.asarray(trace.oracle_phase_index, dtype=np.int64)
    context_trace = np.asarray(trace.oracle_context, dtype=np.int64)
    rewards = np.asarray(trace.reward, dtype=np.float64)
    behavior_nll = np.asarray(trace.behavior_nll, dtype=np.float64)
    grounded_squared_error = np.square(
        np.asarray(trace.grounded_reward_error, dtype=np.float64)
    )
    window_steps = max(1, min(128, phase_length // 4))

    phase_index: list[int] = []
    hidden_context: list[int] = []
    phase_counts: list[int] = []
    phase_valid: list[bool] = []
    mean_reward: list[float] = []
    leading_reward: list[float] = []
    leading_counts: list[int] = []
    trailing_reward: list[float] = []
    trailing_counts: list[int] = []
    phase_behavior_nll: list[float] = []
    phase_grounded_mse: list[float] = []

    for phase in range(n_phases):
        positions = np.flatnonzero(phase_trace == phase)
        count = int(positions.size)
        phase_index.append(phase)
        phase_counts.append(count)
        if count:
            contexts = context_trace[positions]
            context = int(contexts[0])
            context_valid = context in (0, 1) and bool(np.all(contexts == context))
            width = min(window_steps, count)
            phase_rewards = rewards[positions]
            phase_nll = behavior_nll[positions]
            phase_grounded = grounded_squared_error[positions]
            finite = bool(
                np.all(np.isfinite(phase_rewards))
                and np.all(np.isfinite(phase_nll))
                and np.all(np.isfinite(phase_grounded))
            )
            hidden_context.append(context)
            leading_counts.append(width)
            trailing_counts.append(width)
            mean_reward.append(float(np.mean(phase_rewards)))
            leading_reward.append(float(np.mean(phase_rewards[:width])))
            trailing_reward.append(float(np.mean(phase_rewards[-width:])))
            phase_behavior_nll.append(float(np.mean(phase_nll)))
            phase_grounded_mse.append(float(np.mean(phase_grounded)))
            phase_valid.append(count == phase_length and context_valid and finite)
        else:
            hidden_context.append(-1)
            leading_counts.append(0)
            trailing_counts.append(0)
            mean_reward.append(0.0)
            leading_reward.append(0.0)
            trailing_reward.append(0.0)
            phase_behavior_nll.append(0.0)
            phase_grounded_mse.append(0.0)
            phase_valid.append(False)

    switch_cost: list[float] = []
    switch_cost_valid: list[bool] = []
    switch_cost_counts: list[int] = []
    recurrence_reference_phase: list[int] = []
    recurrence_savings: list[float] = []
    recurrence_savings_valid: list[bool] = []
    recurrence_counts: list[int] = []
    latest_phase_by_context: dict[int, int] = {}
    for phase in range(n_phases):
        switch_valid = (
            phase > 0
            and phase_valid[phase - 1]
            and phase_valid[phase]
            and hidden_context[phase - 1] != hidden_context[phase]
        )
        switch_cost_valid.append(switch_valid)
        switch_cost_counts.append(
            min(trailing_counts[phase - 1], leading_counts[phase])
            if switch_valid
            else 0
        )
        switch_cost.append(
            trailing_reward[phase - 1] - leading_reward[phase]
            if switch_valid
            else 0.0
        )

        context = hidden_context[phase]
        reference = latest_phase_by_context.get(context, -1)
        recurrence_valid = (
            phase_valid[phase]
            and reference >= 0
            and phase_valid[reference]
        )
        recurrence_reference_phase.append(reference if recurrence_valid else -1)
        recurrence_savings_valid.append(recurrence_valid)
        recurrence_counts.append(
            min(leading_counts[reference], leading_counts[phase])
            if recurrence_valid
            else 0
        )
        recurrence_savings.append(
            leading_reward[phase] - leading_reward[reference]
            if recurrence_valid
            else 0.0
        )
        if phase_valid[phase]:
            latest_phase_by_context[context] = phase

    return HiddenLearningPartnerPhaseDiagnostics(
        n_phases=n_phases,
        window_steps=window_steps,
        phase_index=tuple(phase_index),
        hidden_context=tuple(hidden_context),
        phase_counts=tuple(phase_counts),
        phase_valid=tuple(phase_valid),
        mean_reward=tuple(mean_reward),
        leading_reward=tuple(leading_reward),
        leading_counts=tuple(leading_counts),
        trailing_reward=tuple(trailing_reward),
        trailing_counts=tuple(trailing_counts),
        behavior_mean_nll=tuple(phase_behavior_nll),
        grounded_reward_mse=tuple(phase_grounded_mse),
        switch_cost=tuple(switch_cost),
        switch_cost_valid=tuple(switch_cost_valid),
        switch_cost_counts=tuple(switch_cost_counts),
        recurrence_reference_phase=tuple(recurrence_reference_phase),
        recurrence_savings=tuple(recurrence_savings),
        recurrence_savings_valid=tuple(recurrence_savings_valid),
        recurrence_counts=tuple(recurrence_counts),
    )


def _metrics_from_trace(
    trace: HiddenLearningPartnerPlanningTrace,
    *,
    direct_channel: bool,
    phase_length: int,
    n_phases: int,
) -> HiddenLearningPartnerPlanningMetrics:
    rewards = np.asarray(trace.reward, dtype=np.float64)
    probabilities = np.asarray(trace.behavior_probabilities_pre, dtype=np.float64)
    actions = np.asarray(trace.beneficiary_action, dtype=np.int64)
    one_hot = np.eye(2, dtype=np.float64)[actions]
    brier = np.sum(np.square(probabilities - one_hot), axis=1)
    reward_error = np.asarray(trace.grounded_reward_error, dtype=np.float64)
    next_prediction = np.asarray(trace.grounded_raw_prediction_pre, dtype=np.float64)[:, 0]
    next_target = 2.0 * np.asarray(trace.next_helper_cue, dtype=np.float64) - 1.0
    consumed = np.asarray(trace.planner_consumed, dtype=np.bool_)
    eligible = np.asarray(trace.action_changed, dtype=np.bool_)
    treated = eligible & consumed
    control = eligible & ~consumed
    n_eligible = int(np.count_nonzero(eligible))
    n_treated = int(np.count_nonzero(treated))
    n_control = int(np.count_nonzero(control))
    randomized_valid = n_treated > 0 and n_control > 0
    randomized_effect = (
        float(np.mean(rewards[treated]) - np.mean(rewards[control]))
        if randomized_valid
        else 0.0
    )
    potential = np.asarray(trace.delivered_potential_rewards, dtype=np.float64)
    planner = np.asarray(trace.planner_message, dtype=np.int64)
    ordinary = np.asarray(trace.ordinary_message, dtype=np.int64)
    indices = np.arange(rewards.size)
    potential_delta = potential[indices, planner] - potential[indices, ordinary]
    potential_valid = direct_channel and n_eligible > 0
    potential_effect = (
        float(np.mean(potential_delta[eligible])) if potential_valid else 0.0
    )
    return HiddenLearningPartnerPlanningMetrics(
        num_steps=int(rewards.size),
        mean_reward=float(np.mean(rewards)),
        behavior_mean_nll=float(np.mean(np.asarray(trace.behavior_nll))),
        behavior_mean_brier=float(np.mean(brier)),
        grounded_reward_mse=float(np.mean(np.square(reward_error))),
        grounded_next_observation_mse=float(
            np.mean(np.square(next_prediction - next_target))
        ),
        planner_consumption_rate=float(np.mean(consumed)),
        action_change_rate=float(np.mean(eligible)),
        eligible_steps=n_eligible,
        treated_eligible_steps=n_treated,
        control_eligible_steps=n_control,
        randomized_effect=randomized_effect,
        randomized_effect_valid=randomized_valid,
        potential_effect=potential_effect,
        potential_effect_valid=potential_valid,
        phase_diagnostics=_phase_diagnostics_from_trace(
            trace,
            phase_length=phase_length,
            n_phases=n_phases,
        ),
    )


def run_hidden_learning_partner_planning(
    condition: HiddenPlanningCondition = JOINT_ADAPTIVE,
    *,
    seed: int,
    config: HiddenLearningPartnerPlanningConfig | None = None,
    jit_compile: bool = True,
) -> HiddenLearningPartnerPlanningRun:
    """Run one short or default uninterrupted life entirely in memory."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(jit_compile) is not bool:
        raise ValueError("jit_compile must be a boolean")
    resolved = config or HiddenLearningPartnerPlanningConfig()
    bridge = HiddenLearningPartnerPlanningBridge(resolved, condition)
    initial = bridge.initialize(jr.key(seed))

    def scan(initial_state: HiddenLearningPartnerPlanningState) -> tuple[
        HiddenLearningPartnerPlanningState,
        HiddenLearningPartnerPlanningTrace,
    ]:
        def scan_step(
            carry: HiddenLearningPartnerPlanningState,
            _: None,
        ) -> tuple[HiddenLearningPartnerPlanningState, HiddenLearningPartnerPlanningTrace]:
            result = bridge.step(carry)
            return result.state, result.trace

        return jax.lax.scan(scan_step, initial_state, xs=None, length=resolved.num_steps)

    if jit_compile:
        final, trace = jax.jit(scan)(initial)
    else:
        with jax.disable_jit():
            final, trace = scan(initial)
    initial_resource = bridge.resource_budget(initial)
    final_resource = bridge.resource_budget(final)
    if not initial_resource.exact_tree_match or not final_resource.exact_tree_match:
        raise ValueError("initial/final state violates the exact resource contract")
    if initial_resource != final_resource:
        raise ValueError("initial and final persistent resources differ")
    metrics = _metrics_from_trace(
        trace,
        direct_channel=bridge.spec.channel == DIRECT_CHANNEL,
        phase_length=resolved.phase_length,
        n_phases=resolved.n_phases,
    )
    return HiddenLearningPartnerPlanningRun(
        condition=condition,
        seed=seed,
        config=resolved,
        initial_state=initial,
        final_state=final,
        trace=trace,
        metrics=metrics,
        resource=initial_resource,
    )


def _validate_key_chain(
    errors: list[str],
    trace: HiddenLearningPartnerPlanningTrace,
    *,
    prefix: str,
    initial_key: Array,
    final_key: Array,
    split_count: int,
) -> None:
    before = np.asarray(getattr(trace, f"{prefix}_key_before"), dtype=np.uint32)
    after = np.asarray(getattr(trace, f"{prefix}_key_after"), dtype=np.uint32)
    if not np.array_equal(before[0], np.asarray(jr.key_data(initial_key))):
        errors.append(f"{prefix} RNG initial binding failed")
    if len(before) > 1 and not np.array_equal(before[1:], after[:-1]):
        errors.append(f"{prefix} RNG continuity failed")
    if not np.array_equal(after[-1], np.asarray(jr.key_data(final_key))):
        errors.append(f"{prefix} RNG final binding failed")
    typed = jr.wrap_key_data(jnp.asarray(before, dtype=jnp.uint32), impl="threefry2x32")
    expected_after = np.asarray(
        jax.vmap(lambda key: jr.key_data(jr.split(key, split_count)[0]))(typed),
        dtype=np.uint32,
    )
    if not np.array_equal(after, expected_after):
        errors.append(f"{prefix} RNG split transition failed")


def validate_hidden_learning_partner_planning_run(
    run: HiddenLearningPartnerPlanningRun,
) -> tuple[str, ...]:
    """Fail closed on structural, resource, freeze, and intervention drift."""

    errors: list[str] = []
    if not isinstance(run, HiddenLearningPartnerPlanningRun):
        return ("run must be a HiddenLearningPartnerPlanningRun",)
    try:
        spec = condition_spec(run.condition)
        bridge = HiddenLearningPartnerPlanningBridge(run.config, run.condition)
    except (TypeError, ValueError) as exc:
        return (f"run configuration is invalid: {exc}",)
    expected_token = np.asarray(bridge.config_token, dtype=np.uint8)
    if not np.array_equal(
        np.asarray(run.initial_state.config_token, dtype=np.uint8),
        expected_token,
    ):
        errors.append("initial config token differs from the static composition")
    if not np.array_equal(
        np.asarray(run.final_state.config_token, dtype=np.uint8),
        expected_token,
    ):
        errors.append("final config token differs from the static composition")
    if run.resource.total_state_nbytes != _EXPECTED_TOTAL_BYTES:
        errors.append("resource total is not exactly 321 bytes")
    if not run.resource.exact_tree_match:
        errors.append("resource report does not match the exact state tree")
    for label, state in (("initial", run.initial_state), ("final", run.final_state)):
        budget = bridge.resource_budget(state)
        if budget.total_state_nbytes != _EXPECTED_TOTAL_BYTES or not budget.exact_tree_match:
            errors.append(f"{label} state resource contract failed")
        if budget != run.resource:
            errors.append(f"{label} state resource report mismatch")
    n = run.config.num_steps
    trace = run.trace
    for field in dataclasses.fields(trace):
        value = np.asarray(getattr(trace, field.name))
        if value.shape[0] != n:
            errors.append(f"trace.{field.name} length mismatch")
    if errors:
        return tuple(errors)
    if not bool(np.all(np.asarray(trace.accepted))):
        errors.append("not every uninterrupted transition was accepted")
    if not bool(np.all(np.asarray(trace.config_token_valid))):
        errors.append("trace config-token validation failed")
    if not bool(np.all(np.asarray(trace.learner_projection_valid))):
        errors.append("learner projection validation failed")
    if not bool(np.all(np.asarray(trace.all_finite))):
        errors.append("learner-path finiteness validation failed")
    expected_step = np.arange(n, dtype=np.int32)
    if not np.array_equal(np.asarray(trace.step), expected_step):
        errors.append("step trace is not one uninterrupted cursor")
    expected_phase = expected_step // run.config.phase_length
    expected_context = expected_phase % 2
    if not np.array_equal(np.asarray(trace.oracle_phase_index), expected_phase):
        errors.append("oracle phase trace differs from the configured schedule")
    if not np.array_equal(np.asarray(trace.oracle_context), expected_context):
        errors.append("oracle context trace differs from the configured schedule")
    if np.any(np.asarray(trace.helper_context) != _SHARED_ROW) or np.any(
        np.asarray(trace.beneficiary_context) != _SHARED_ROW
    ):
        errors.append("a learner decision used a public task row")
    if not bool(np.all(np.asarray(trace.shared_inactive_rows_unchanged))):
        errors.append("shared inactive row changed")
    expected_target = np.bitwise_xor(
        np.asarray(trace.helper_cue, dtype=np.int32),
        expected_context,
    )
    if not np.array_equal(np.asarray(trace.oracle_target), expected_target):
        errors.append("oracle target reconstruction failed")
    expected_reward = (
        np.asarray(trace.beneficiary_action, dtype=np.int32) == expected_target
    ).astype(np.float32)
    if not np.array_equal(np.asarray(trace.reward), expected_reward):
        errors.append("reward reconstruction failed")
    if not bool(np.all(np.asarray(trace.behavior_prediction_bound))):
        errors.append("cached behavior prediction binding failed")
    if not np.array_equal(
        np.asarray(trace.behavior_probabilities_pre),
        np.asarray(trace.behavior_probabilities_update),
    ):
        errors.append("cached behavior probabilities differ from update probabilities")
    behavior_probabilities_pre = jnp.asarray(
        trace.behavior_probabilities_pre,
        dtype=jnp.float32,
    )
    beneficiary_actions = jnp.asarray(
        trace.beneficiary_action,
        dtype=jnp.int32,
    )
    expected_behavior_action_probability = jnp.take_along_axis(
        behavior_probabilities_pre,
        beneficiary_actions[:, None],
        axis=1,
    )[:, 0]
    if not np.array_equal(
        np.asarray(trace.behavior_action_probability),
        np.asarray(expected_behavior_action_probability),
    ):
        errors.append("behavior action probability is not bound to the cached prediction")
    expected_behavior_nll = -jnp.log(
        jnp.maximum(expected_behavior_action_probability, jnp.float32(1e-6))
    )
    if not np.array_equal(
        np.asarray(trace.behavior_nll),
        np.asarray(expected_behavior_nll),
    ):
        errors.append("behavior NLL is not bound to the cached action probability")
    if not bool(np.all(np.asarray(trace.grounded_prediction_bound))):
        errors.append("cached grounded prediction binding failed")
    if not np.array_equal(
        np.asarray(trace.grounded_raw_prediction_pre),
        np.asarray(trace.grounded_raw_prediction_update),
    ):
        errors.append("cached grounded prediction differs from update prediction")
    expected_grounded_reward_error = (
        jnp.asarray(trace.grounded_raw_prediction_pre, dtype=jnp.float32)[
            :, _GROUNDED_OBSERVATION_DIM
        ]
        - jnp.asarray(trace.reward, dtype=jnp.float32)
    )
    if not np.array_equal(
        np.asarray(trace.grounded_reward_error),
        np.asarray(expected_grounded_reward_error),
    ):
        errors.append("grounded reward error is not bound to prediction and reward")
    if not bool(np.all(np.asarray(trace.potential_outcome_bound))):
        errors.append("potential-outcome binding failed")
    if not bool(np.all(np.asarray(trace.intervention_bound))):
        errors.append("planner intervention binding failed")
    expected_consumed = (
        np.asarray(trace.planner_gate_draw)
        if spec.planner_consumption == "randomized"
        else np.full(n, spec.planner_consumption == "always", dtype=np.bool_)
    )
    if not np.array_equal(np.asarray(trace.planner_consumed), expected_consumed):
        errors.append("planner consumption differs from its static intervention")
    expected_message = np.where(
        expected_consumed,
        np.asarray(trace.planner_message),
        np.asarray(trace.ordinary_message),
    )
    if not np.array_equal(np.asarray(trace.helper_message), expected_message):
        errors.append("executed helper message differs from the selected proposal")
    if not np.array_equal(
        np.asarray(trace.action_changed),
        np.asarray(trace.planner_message) != np.asarray(trace.ordinary_message),
    ):
        errors.append("action-changing eligibility reconstruction failed")
    if not bool(np.all(np.asarray(trace.behavior_proposal_applied))):
        errors.append("behavior proposal was not validly formed")
    if not bool(np.all(np.asarray(trace.grounded_proposal_applied))):
        errors.append("grounded proposal was not validly formed")
    if np.any(np.asarray(trace.behavior_committed_write) != spec.behavior_write):
        errors.append("behavior outer write mask was not honored")
    if np.any(np.asarray(trace.grounded_committed_write) != spec.grounded_write):
        errors.append("grounded outer write mask was not honored")
    if np.any(np.asarray(trace.helper_write) != spec.helper_write):
        errors.append("helper write mask was not honored")
    if np.any(np.asarray(trace.beneficiary_write) != spec.beneficiary_write):
        errors.append("beneficiary write mask was not honored")

    _validate_key_chain(
        errors,
        trace,
        prefix="helper",
        initial_key=run.initial_state.learner.helper.key,
        final_key=run.final_state.learner.helper.key,
        split_count=4,
    )
    _validate_key_chain(
        errors,
        trace,
        prefix="beneficiary",
        initial_key=run.initial_state.learner.beneficiary.key,
        final_key=run.final_state.learner.beneficiary.key,
        split_count=4,
    )
    _validate_key_chain(
        errors,
        trace,
        prefix="planner",
        initial_key=run.initial_state.planner_key,
        final_key=run.final_state.planner_key,
        split_count=2,
    )
    _validate_key_chain(
        errors,
        trace,
        prefix="intervention",
        initial_key=run.initial_state.intervention_key,
        final_key=run.final_state.intervention_key,
        split_count=2,
    )
    if not spec.helper_write and not np.array_equal(
        np.asarray(run.initial_state.learner.helper.values),
        np.asarray(run.final_state.learner.helper.values),
    ):
        errors.append("frozen helper values changed")
    if not spec.beneficiary_write and not np.array_equal(
        np.asarray(run.initial_state.learner.beneficiary.values),
        np.asarray(run.final_state.learner.beneficiary.values),
    ):
        errors.append("frozen beneficiary values changed")
    if not spec.behavior_write and not _host_tree_equal(
        run.initial_state.behavior,
        run.final_state.behavior,
    ):
        errors.append("frozen behavior state changed")
    if not spec.grounded_write and not _host_tree_equal(
        run.initial_state.grounded,
        run.final_state.grounded,
    ):
        errors.append("frozen grounded state changed")
    expected_behavior_updates = n if spec.behavior_write else 0
    expected_grounded_updates = n if spec.grounded_write else 0
    for label, behavior_state, expected_count in (
        ("initial behavior", run.initial_state.behavior, 0),
        ("final behavior", run.final_state.behavior, expected_behavior_updates),
    ):
        if (
            int(behavior_state.step_count) != expected_count
            or not _host_lifetime_words_equal(
                behavior_state.step_words,
                expected_count,
            )
        ):
            errors.append(f"{label} telemetry/exact words differ from committed writes")
    for label, grounded_state, expected_count in (
        ("initial grounded", run.initial_state.grounded, 0),
        ("final grounded", run.final_state.grounded, expected_grounded_updates),
    ):
        if (
            int(grounded_state.update_count) != expected_count
            or not _host_lifetime_words_equal(
                grounded_state.update_words,
                expected_count,
            )
        ):
            errors.append(f"{label} telemetry/exact words differ from committed writes")
    if int(run.final_state.step_count) != n or int(run.final_state.world.step_count) != n:
        errors.append("final world/bridge counters do not equal the life length")
    if not bool(run.final_state.valid):
        errors.append("final validity latch is false")
    metric_values = dataclasses.asdict(run.metrics)
    for name, value in metric_values.items():
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"metrics.{name} is not finite")
    expected_metrics = _metrics_from_trace(
        trace,
        direct_channel=spec.channel == DIRECT_CHANNEL,
        phase_length=run.config.phase_length,
        n_phases=run.config.n_phases,
    )
    if run.metrics.phase_diagnostics != expected_metrics.phase_diagnostics:
        errors.append("phase diagnostics differ from primitive trace reconstruction")
    if run.metrics != expected_metrics:
        errors.append("raw metrics differ from primitive trace reconstruction")
    return tuple(errors)


__all__ = [
    "ACCEPTANCE_STATUS",
    "BEHAVIOR_FROZEN",
    "BENEFICIARY_FROZEN",
    "BOTH_MODELS_FROZEN",
    "BOTH_ROLES_FROZEN",
    "CLAIM_THRESHOLDS_FROZEN",
    "CONSTANT_ONE_DELIVERY",
    "CONSTANT_ZERO_DELIVERY",
    "DEVELOPMENT_ONLY",
    "GROUNDED_FROZEN",
    "HELPER_FROZEN",
    "HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA",
    "HiddenDyadFeedback",
    "HiddenDyadPreObservation",
    "HiddenLearningPartnerPlanningBridge",
    "HiddenLearningPartnerPlanningConfig",
    "HiddenLearningPartnerPlanningMetrics",
    "HiddenLearningPartnerPhaseDiagnostics",
    "HiddenLearningPartnerPlanningResourceBudget",
    "HiddenLearningPartnerPlanningRun",
    "HiddenLearningPartnerPlanningState",
    "HiddenLearningPartnerPlanningStep",
    "HiddenLearningPartnerPlanningTrace",
    "HiddenPlanningCondition",
    "HiddenPlanningConditionSpec",
    "JOINT_ADAPTIVE",
    "MATCHED_CONDITIONS",
    "PLANNER_NEVER_CONSUMED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SHUFFLED_DELIVERY",
    "condition_spec",
    "run_hidden_learning_partner_planning",
    "strip_hidden_learning_partner_oracle",
    "validate_hidden_learning_partner_planning_run",
]
