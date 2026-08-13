# mypy: disable-error-code="attr-defined,call-arg"
"""Bounded one-step integration kernel for the hidden-partner development life.

This module composes existing online components into one explicit causal
transition:

``learned state -> bounded pair discovery -> partner prediction ->
joint-outcome planning -> differential SARSA``.

It is intentionally an L0 integration mechanism.  The hidden-partner-v0
environment has a scripted stochastic partner, so successful use demonstrates
learning *about* another agent, not co-adaptation, promoted L3 evidence, or
completion of the Alberta Plan.

Causal ordering
---------------
``start`` consumes one initial raw observation, constructs a fixed-width
representation, evaluates the four table-world cells and four optional
grounded-world cells once, selects an action with the SARSA exploration RNG,
and stores the exact decision record.

``update`` accepts the resulting environment transition and performs:

1. score the stored pre-action behavior decision, executed table-world cell,
   and (when configured) the representation-conditioned grounded world model;
2. mix the pre-update behavior and grounded-world representation gradients,
   then chain the named behavior-only and mixed gradients through pair products
   in one batched derivative call;
3. learn the recurrent state parameters exactly once before advancing recurrence;
4. update partner prediction and the shadow feature-discovery learner;
5. advance the state builder exactly once with the next observation and the
   preceding action/reward/discount;
6. update only the executed joint-world cell;
7. atomically route behavior, control, and optional grounded-world feature
   columns by pair identity;
8. construct the next representation under the deployed descriptor bank;
9. predict the partner and evaluate all four table-world cells plus all four
   optional grounded-world cells under a static planner-source mask;
10. select the next external planner action while advancing SARSA's RNG; and
11. update differential SARSA with that explicit next action.

All ablations keep the same fixed shapes.  ``planning_enabled=False`` masks
only the centered additive model term; partner and world predictions are still
computed. ``state_learning_enabled=False`` computes but discards the recurrent
parameter update while recurrence still advances. ``feature_lifecycle_enabled=False``
computes and diagnoses the complete curation proposal, then commits the exact
learned pre-curation snapshot. Both interaction and router descriptor banks
therefore remain coherent and frozen while ordinary online learning, evidence,
moments, ages, counters, and RNG advancement continue.
``uniform_partner_belief=True`` still predicts and learns the partner model but
applies a uniform belief to the joint-world marginalization.
``random_feature_curation=True`` still computes the complete utility-learning
path from untouched learned state and supplies deterministic transient ranks
only to active-worst, candidate-best, and candidate-worst transaction choices.
``action_selection_mode="externally_forced"`` still computes the complete
ordinary epsilon-greedy decision and advances its RNG identically; only the
applied action is replaced. The persistent selection flag records that fact,
while the external action schedule remains the responsibility of the caller
and, for serialized runs, an external trace digest.
The grounded-world and representation-gradient mixer lane is an opt-in L0
mechanism. Its absence preserves the legacy state leaves, resource accounting,
and transition path. When present, its four predictions are evaluated even if
``grounded_world_planning_enabled=False``; that static flag selects only which
reward surface reaches planning.
``grounded_world_learning_enabled=False`` is its matched-compute control: the
complete proposed update and representation gradient are still computed, but
the old grounded parameters are selected before consumer gating and routing.
``carry_survivors=False`` preserves learned columns between descriptor
transactions and zeros the entire discovered tail only when the deployed bank
actually changes. ``memory_masked=True`` zeros the four learned hidden
coordinates and their representation derivative; the recurrent learning call
therefore has zero hidden credit while recurrence advances with fixed
parameters.
``evidence_gated_consumer_memory=True`` separates downstream read leases from
overwrite permission. One consecutive-evidence threshold acquires/reopens a
read lease; a second, no-smaller threshold confirms behavior/Q column writes.
Once acquired, a pair's stored columns remain readable through a bounded
evidence-idle lease, but they are not overwritten on idle steps. The preceding
transition's read mask gates pair products and recurrent credit. Confirmation
streaks, read acquisition, write gates, and read leases are routed by
descriptor identity; new identities start closed with zero streak. Closed
SARSA traces are erased. The mechanism adds one fixed-width boolean vector and
one fixed-width int32 streak, not replay or a dormant weight archive.

``independent_relevance_probe=True`` makes that evidence come from a separate
fixed-width relevance probe with explicit versioned semantics. The default
``conditional_v1`` mode uses the complete readable durable-bank residual:
candidate probes score insertion against it, while active probe ``i`` scores
against its leave-one-out residual. ``target_only_v1`` instead uses a separate
learned probe bias and excludes durable-bank contributions from every isolated
active and candidate probe target. First confirmation copies that pre-update
probe into durable memory. Both modes have identical bounded state and compute.

Candidate promotion can independently require consecutive marginal evidence.
The fixed-width streak is tied to candidate identity and resets on promotion,
refresh, invalid identity, or retirement of a matching active pair. The
default one-step setting preserves historical promotion decisions; larger
values reject isolated bursts without raising the utility floor.

Public array surfaces enforce exact effective JAX shapes and dtypes. Dynamic
transition invalidity is fail-closed under eager execution, ``jit``, and
``scan``: no persistent state or action advances and diagnostics explicitly
mark the rejection. Stateless representation kernels return a documented
all-zero neutral value for nonfinite arrays or invalid descriptor banks.

``candidate_reacquisition_confirmation_steps`` is narrower still. When set
above one, only archive entries matching an actually retired active descriptor
are marked for reacquisition confirmation. Initial acquisition remains on the
generic promotion threshold; marked entries use the larger of the generic and
reacquisition thresholds until promotion or identity refresh clears the mark.

The full condition gives active descriptor utilities a slow recurrent-context
retention floor. Setting ``active_utility_retention_decay=None`` is the matched
normal-decay ablation: shapes, update cadence, and replacement opportunities
are unchanged. This protects useful descriptors only while they remain in the
active bank. The router carries behavior and SARSA columns for surviving
descriptor identities, but this L0 kernel has no dormant downstream-weight
archive; an evicted and later rediscovered identity starts those columns at
zero.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal, Protocol, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouteDiagnostics,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    GroundedJointWorldModelState,
    GroundedJointWorldUpdateResult,
)
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    RELEVANCE_PROBE_MODES,
    FixedBudgetInteractionLearner,
    InteractionCurationPriorityOverride,
    InteractionFeatureState,
)
from alberta_framework.core.joint_partner_world import (
    BoundedJointOutcomeConfig,
    BoundedJointOutcomeModel,
    BoundedJointOutcomeState,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
    RepresentationGradientMixResult,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
    StateBuilderLearningDiagnostics,
)

INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION = "alberta.integrated-hidden-partner.l0.v16"
INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_NBYTES = 12
INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_DELTA_NBYTES = 8
DEVELOPMENT_LEVEL = "L0"

RAW_OBSERVATION_DIM = 8
HIDDEN_STATE_DIM = 4
BASE_FEATURE_DIM = RAW_OBSERVATION_DIM + HIDDEN_STATE_DIM
ACTIVE_PAIR_SLOTS = 12
CANDIDATE_PAIR_SLOTS = BASE_FEATURE_DIM * (BASE_FEATURE_DIM - 1) // 2
DEPLOYED_FEATURE_DIM = BASE_FEATURE_DIM + ACTIVE_PAIR_SLOTS
N_ACTIONS = 2
WORLD_OUTCOME_DIM = 1

INITIAL_ACTIVE_DESCRIPTORS: tuple[tuple[int, int], ...] = (
    (0, 6),
    (0, 8),
    (1, 4),
    (1, 7),
    (2, 5),
    (2, 9),
    (3, 6),
    (3, 10),
    (4, 7),
    (5, 6),
    (7, 11),
    (8, 10),
)
"""Unique canonical distractors excluding critical pairs (0, 2) and (4, 5)."""

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

INTEGRATED_CHILD_CLOCK_ALIGNMENT_ORDER: tuple[str, ...] = (
    "behavior_step",
    "joint_world_step",
    "control_step",
    "router_route",
    "interaction_step",
    "state_builder_step_plus_one",
    "state_builder_update_policy",
    "grounded_world_update_policy",
    "router_generation_order",
)

INTEGRATED_DECISION_CACHE_CHECK_ORDER: tuple[str, ...] = (
    "predicted_partner_probabilities",
    "partner_probabilities",
    "partner_probabilities_valid",
    "probability_violation",
    "expected_rewards",
    "expected_outcomes",
    "centered_expected_rewards",
    "model_term",
    "applied_model_term",
    "planner_scores",
    "greedy_action",
    "cell_evaluations",
    "control_last_observation",
    "current_q_value_delta",
    "control_epsilon",
    "control_q_bias_disabled",
    "control_q_trace_bias_disabled",
    "control_step_count",
    "control_step_words",
    "grounded_evaluation",
    "planner_selection",
    "cached_evaluation_finite",
    "q_value_delta_finite",
    "planner_selection_finite",
)


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose the next exact integrated transition identity without wrap."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("integrated lifetime words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("integrated lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(array == maximum)
    low = array[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((array[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, array).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Validate exact integrated identity against saturating compatibility telemetry."""

    array = jnp.asarray(words)
    counter = jnp.asarray(telemetry)
    if array.shape != (2,):
        raise ValueError("integrated lifetime words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("integrated lifetime words must have dtype uint32")
    if counter.shape != ():
        raise ValueError("integrated step_count must be scalar")
    if counter.dtype != jnp.dtype(jnp.int32):
        raise TypeError("integrated step_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (array[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        array[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (counter >= 0) & jnp.where(
        below_saturation,
        counter == array[1].astype(jnp.int32),
        counter == maximum_i32,
    )


def _lifetime_words_le(left: Array, right: Array) -> Bool[Array, ""]:
    """Return the exact unsigned lexicographic order for two-word counters."""

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != (2,) or right_array.shape != (2,):
        raise ValueError("integrated lifetime word comparisons require shape (2,)")
    if (
        left_array.dtype != jnp.dtype(jnp.uint32)
        or right_array.dtype != jnp.dtype(jnp.uint32)
    ):
        raise TypeError("integrated lifetime word comparisons require dtype uint32")
    return (left_array[0] < right_array[0]) | (
        (left_array[0] == right_array[0]) & (left_array[1] <= right_array[1])
    )


def _require_array_contract(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Return an array after trace-time shape and effective-dtype validation."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected_dtype = jnp.dtype(dtype)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _saturating_int32_increment(value: Array) -> Array:
    """Increment a non-negative int32 counter without wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _numeric_tree_finite(tree: Any) -> Array:
    """Return whether every numeric PyTree leaf is finite, ignoring typed keys."""

    checks: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(value.dtype, jnp.number):
            checks.append(jnp.all(jnp.isfinite(value)))
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True, dtype=jnp.bool_)


class HiddenPartnerTransition(Protocol):
    """Structural transition contract consumed from the v0 stream."""

    observation: Array
    focal_action: Array
    partner_action: Array
    reward: Array
    outcome: Array
    next_observation: Array
    terminated: Array
    discount: Array


@dataclasses.dataclass(frozen=True)
class IntegratedHiddenPartnerConfig:
    """Static dimensions, online rates, and matched development ablations."""

    planning_enabled: bool = True
    state_learning_enabled: bool = True
    feature_lifecycle_enabled: bool = True
    carry_survivors: bool = True
    memory_masked: bool = False
    uniform_partner_belief: bool = False
    random_feature_curation: bool = False
    action_selection_mode: Literal["agent", "externally_forced"] = "agent"
    evidence_gated_feature_memory: bool = False
    feature_evidence_confirmation_steps: int = 1
    independent_relevance_probe: bool = False
    relevance_probe_mode: str = RELEVANCE_PROBE_MODE_CONDITIONAL_V1
    evidence_gated_consumer_memory: bool = False
    consumer_evidence_confirmation_steps: int = 1
    consumer_read_confirmation_steps: int = 1
    consumer_read_lease_steps: int = 32
    initial_active_descriptors: tuple[tuple[int, int], ...] = INITIAL_ACTIVE_DESCRIPTORS
    grounded_world_model: GroundedJointWorldModelConfig | None = None
    representation_gradient_mixer: RepresentationGradientMixerConfig | None = None
    grounded_world_learning_enabled: bool = True
    grounded_world_planning_enabled: bool = False
    # Selected on the explicitly consumed hidden-partner-v0 tuning namespace.
    # These are development defaults, not promoted hyperparameters.
    planner_lambda: float = 2.0
    state_step_size: float = 0.005
    state_gradient_clip: float = 5.0
    interaction_step_size: float = 0.03
    interaction_utility_decay: float = 0.995
    active_utility_retention_decay: float | None = 0.9999
    active_utility_retention_grace_steps: int | None = None
    active_utility_evidence_threshold: float = 0.0
    retire_stale_features: bool = False
    candidate_promotion_floor: float = 0.0
    candidate_promotion_confirmation_steps: int = 1
    candidate_reacquisition_confirmation_steps: int = 1
    replacement_interval: int = 64
    min_feature_age: int = 256
    candidate_min_age: int = 128
    candidate_utility_retention_decay: float = 0.9995
    behavior_step_size: float = 0.05
    world_step_size: float = 0.25
    q_step_size: float = 0.03
    average_reward_step_size: float = 0.003
    trace_decay: float = 0.0
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        """Reject non-static flags, non-finite rates, and unsafe counters."""

        def real_value(name: str) -> float:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number, not boolean")
            return float(value)

        for name in (
            "planning_enabled",
            "state_learning_enabled",
            "feature_lifecycle_enabled",
            "carry_survivors",
            "memory_masked",
            "uniform_partner_belief",
            "random_feature_curation",
            "evidence_gated_feature_memory",
            "independent_relevance_probe",
            "evidence_gated_consumer_memory",
            "retire_stale_features",
            "grounded_world_learning_enabled",
            "grounded_world_planning_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if (
            not isinstance(self.action_selection_mode, str)
            or self.action_selection_mode not in ("agent", "externally_forced")
        ):
            raise ValueError(
                "action_selection_mode must be 'agent' or 'externally_forced'"
            )
        if (
            not isinstance(self.relevance_probe_mode, str)
            or self.relevance_probe_mode not in RELEVANCE_PROBE_MODES
        ):
            raise ValueError("relevance_probe_mode must be 'conditional_v1' or 'target_only_v1'")
        descriptors = self.initial_active_descriptors
        if not isinstance(descriptors, tuple) or len(descriptors) != ACTIVE_PAIR_SLOTS:
            raise ValueError(
                f"initial_active_descriptors must be a tuple of exactly {ACTIVE_PAIR_SLOTS} pairs"
            )
        for pair in descriptors:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("initial_active_descriptors entries must be exact 2-tuples")
            left, right = pair
            if (
                isinstance(left, bool)
                or not isinstance(left, int)
                or isinstance(right, bool)
                or not isinstance(right, int)
            ):
                raise ValueError(
                    "initial_active_descriptors endpoints must be non-boolean integers"
                )
            if not 0 <= left < right < BASE_FEATURE_DIM:
                raise ValueError(
                    "initial_active_descriptors pairs must satisfy "
                    f"0 <= left < right < {BASE_FEATURE_DIM}"
                )
        if len(set(descriptors)) != ACTIVE_PAIR_SLOTS:
            raise ValueError("initial_active_descriptors pairs must be unique")
        grounded_config = self.grounded_world_model
        mixer_config = self.representation_gradient_mixer
        if (grounded_config is None) != (mixer_config is None):
            raise ValueError(
                "grounded_world_model and representation_gradient_mixer must be configured together"
            )
        if grounded_config is not None:
            if not isinstance(grounded_config, GroundedJointWorldModelConfig):
                raise ValueError("grounded_world_model must be a GroundedJointWorldModelConfig")
            if not isinstance(mixer_config, RepresentationGradientMixerConfig):
                raise ValueError(
                    "representation_gradient_mixer must be a RepresentationGradientMixerConfig"
                )
            if grounded_config.representation_dim != DEPLOYED_FEATURE_DIM:
                raise ValueError(
                    f"grounded_world_model representation_dim must be {DEPLOYED_FEATURE_DIM}"
                )
            if grounded_config.target_observation_dim != RAW_OBSERVATION_DIM:
                raise ValueError(
                    f"grounded_world_model target_observation_dim must be {RAW_OBSERVATION_DIM}"
                )
            if (
                grounded_config.n_focal_actions != N_ACTIONS
                or grounded_config.n_partner_actions != N_ACTIONS
            ):
                raise ValueError(f"grounded_world_model action dimensions must both be {N_ACTIONS}")
            if mixer_config.representation_dim != DEPLOYED_FEATURE_DIM:
                raise ValueError(
                    "representation_gradient_mixer representation_dim must be "
                    f"{DEPLOYED_FEATURE_DIM}"
                )
        if self.grounded_world_planning_enabled and grounded_config is None:
            raise ValueError("grounded_world_planning_enabled requires grounded_world_model")
        if not self.grounded_world_learning_enabled and grounded_config is None:
            raise ValueError("disabling grounded-world learning requires grounded_world_model")
        for name in (
            "planner_lambda",
            "state_step_size",
            "state_gradient_clip",
            "interaction_step_size",
            "behavior_step_size",
        ):
            value = real_value(name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        world_step_size = real_value("world_step_size")
        if not math.isfinite(world_step_size) or not 0.0 < world_step_size <= 1.0:
            raise ValueError("world_step_size must be finite and lie in (0, 1]")
        for name in ("q_step_size", "average_reward_step_size"):
            value = real_value(name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        interaction_utility_decay = real_value("interaction_utility_decay")
        if (
            not math.isfinite(interaction_utility_decay)
            or not 0.0 <= interaction_utility_decay < 1.0
        ):
            raise ValueError("interaction_utility_decay must lie in [0, 1)")
        active_retention = self.active_utility_retention_decay
        if active_retention is not None:
            active_retention = real_value("active_utility_retention_decay")
            if (
                not math.isfinite(active_retention)
                or not interaction_utility_decay <= active_retention < 1.0
            ):
                raise ValueError(
                    "active_utility_retention_decay must be None or lie in "
                    "[interaction_utility_decay, 1)"
                )
        grace = self.active_utility_retention_grace_steps
        if grace is not None and (
            isinstance(grace, bool) or not isinstance(grace, int) or not 0 <= grace < _INT32_MAX
        ):
            raise ValueError(
                "active_utility_retention_grace_steps must be None or an "
                "int32-safe non-negative integer"
            )
        evidence_threshold = real_value("active_utility_evidence_threshold")
        if not math.isfinite(evidence_threshold) or evidence_threshold < 0.0:
            raise ValueError("active_utility_evidence_threshold must be finite and non-negative")
        if grace is not None and evidence_threshold <= 0.0:
            raise ValueError(
                "active_utility_evidence_threshold must be positive when a "
                "retention grace period is enabled"
            )
        if self.evidence_gated_feature_memory and grace is None:
            raise ValueError(
                "evidence_gated_feature_memory requires active_utility_retention_grace_steps"
            )
        if self.evidence_gated_feature_memory and evidence_threshold <= 0.0:
            raise ValueError(
                "evidence_gated_feature_memory requires a positive "
                "active_utility_evidence_threshold"
            )
        if self.independent_relevance_probe and not self.evidence_gated_feature_memory:
            raise ValueError("independent_relevance_probe requires evidence_gated_feature_memory")
        if self.independent_relevance_probe and not self.evidence_gated_consumer_memory:
            raise ValueError("independent_relevance_probe requires evidence_gated_consumer_memory")
        if self.evidence_gated_consumer_memory and grace is None:
            raise ValueError(
                "evidence_gated_consumer_memory requires active_utility_retention_grace_steps"
            )
        if self.evidence_gated_consumer_memory and evidence_threshold <= 0.0:
            raise ValueError(
                "evidence_gated_consumer_memory requires a positive "
                "active_utility_evidence_threshold"
            )
        for name in (
            "feature_evidence_confirmation_steps",
            "consumer_evidence_confirmation_steps",
            "consumer_read_confirmation_steps",
            "consumer_read_lease_steps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= _INT32_MAX
            ):
                raise ValueError(f"{name} must be a non-negative int32-safe integer")
            if (
                name == "feature_evidence_confirmation_steps"
                and self.evidence_gated_feature_memory
                and value <= 0
            ):
                raise ValueError(
                    "feature_evidence_confirmation_steps must be positive when "
                    "evidence_gated_feature_memory is enabled"
                )
            if (
                name != "feature_evidence_confirmation_steps"
                and self.evidence_gated_consumer_memory
                and value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive when evidence_gated_consumer_memory is enabled"
                )
        if (
            self.evidence_gated_consumer_memory
            and self.consumer_read_confirmation_steps > self.consumer_evidence_confirmation_steps
        ):
            raise ValueError(
                "consumer_read_confirmation_steps must not exceed "
                "consumer_evidence_confirmation_steps"
            )
        promotion_floor = real_value("candidate_promotion_floor")
        if not math.isfinite(promotion_floor) or promotion_floor < 0.0:
            raise ValueError("candidate_promotion_floor must be finite and non-negative")
        if self.retire_stale_features and grace is None:
            raise ValueError("retire_stale_features requires active_utility_retention_grace_steps")
        if self.retire_stale_features and promotion_floor <= 0.0:
            raise ValueError("retire_stale_features requires a positive candidate_promotion_floor")
        candidate_confirmation = self.candidate_promotion_confirmation_steps
        if (
            isinstance(candidate_confirmation, bool)
            or not isinstance(candidate_confirmation, int)
            or not 1 <= candidate_confirmation < _INT32_MAX
        ):
            raise ValueError(
                "candidate_promotion_confirmation_steps must be a positive int32-safe integer"
            )
        candidate_reacquisition = self.candidate_reacquisition_confirmation_steps
        if (
            isinstance(candidate_reacquisition, bool)
            or not isinstance(candidate_reacquisition, int)
            or not 1 <= candidate_reacquisition < _INT32_MAX
        ):
            raise ValueError(
                "candidate_reacquisition_confirmation_steps must be a positive int32-safe integer"
            )
        # Retirement is the only native path that raises the per-candidate
        # reacquisition-required flag. Keeping this threshold above one in the
        # matched no-retirement arm is therefore inert but preserves the exact
        # static configuration and compute contract.
        if candidate_reacquisition > 1 and not self.independent_relevance_probe:
            raise ValueError(
                "candidate_reacquisition_confirmation_steps greater than one "
                "requires independent_relevance_probe"
            )
        candidate_retention = real_value("candidate_utility_retention_decay")
        if (
            not math.isfinite(candidate_retention)
            or not interaction_utility_decay <= candidate_retention < 1.0
        ):
            raise ValueError(
                "candidate_utility_retention_decay must lie in [interaction_utility_decay, 1)"
            )
        for name in (
            "replacement_interval",
            "min_feature_age",
            "candidate_min_age",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= _INT32_MAX
            ):
                raise ValueError(f"{name} must be a non-negative int32-safe integer")
        if self.retire_stale_features and self.replacement_interval <= 0:
            raise ValueError("retire_stale_features requires a positive replacement_interval")
        for name in ("trace_decay", "epsilon"):
            value = real_value(name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_config(self) -> dict[str, Any]:
        """Return a strict JSON-compatible development configuration."""
        values = dataclasses.asdict(self)
        values["initial_active_descriptors"] = [
            list(pair) for pair in self.initial_active_descriptors
        ]
        values["grounded_world_model"] = (
            None if self.grounded_world_model is None else self.grounded_world_model.to_config()
        )
        values["representation_gradient_mixer"] = (
            None
            if self.representation_gradient_mixer is None
            else self.representation_gradient_mixer.to_config()
        )
        return {
            "type": type(self).__name__,
            "schema_version": INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION,
            "development_level": DEVELOPMENT_LEVEL,
            "accepted_scientific_evidence": False,
            **values,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
    ) -> IntegratedHiddenPartnerConfig:
        """Strictly reconstruct :meth:`to_config` output."""
        values = dict(payload)
        metadata = {
            "type",
            "schema_version",
            "development_level",
            "accepted_scientific_evidence",
        }
        expected = {field.name for field in dataclasses.fields(cls)} | metadata
        if set(values) != expected:
            raise ValueError("integrated config fields do not match the v16 schema")
        if values.pop("type") != cls.__name__:
            raise ValueError("integrated config type is invalid")
        if values.pop("schema_version") != INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION:
            raise ValueError("integrated config schema version is unsupported")
        if values.pop("development_level") != DEVELOPMENT_LEVEL:
            raise ValueError("integrated kernel must remain development level L0")
        if values.pop("accepted_scientific_evidence") is not False:
            raise ValueError("integrated kernel is not accepted scientific evidence")
        descriptor_payload = values["initial_active_descriptors"]
        if (
            not isinstance(descriptor_payload, list)
            or len(descriptor_payload) != ACTIVE_PAIR_SLOTS
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in descriptor_payload)
        ):
            raise ValueError("initial_active_descriptors must use exactly 12 ordered JSON lists")
        values["initial_active_descriptors"] = tuple(tuple(pair) for pair in descriptor_payload)
        grounded_payload = values["grounded_world_model"]
        mixer_payload = values["representation_gradient_mixer"]
        if grounded_payload is not None:
            if not isinstance(grounded_payload, Mapping):
                raise ValueError("grounded_world_model must be a config mapping or null")
            values["grounded_world_model"] = GroundedJointWorldModelConfig.from_config(
                grounded_payload
            )
        if mixer_payload is not None:
            if not isinstance(mixer_payload, Mapping):
                raise ValueError("representation_gradient_mixer must be a config mapping or null")
            values["representation_gradient_mixer"] = RepresentationGradientMixerConfig.from_config(
                mixer_payload
            )
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class IntegratedHiddenPartnerResourceBudget:
    """Exact persistent array bytes for one initialized integrated state."""

    raw_observation_dim: int
    base_feature_dim: int
    active_pair_slots: int
    candidate_pair_slots: int
    deployed_feature_dim: int
    state_builder_nbytes: int
    interaction_nbytes: int
    interaction_evidence_idle_nbytes: int
    interaction_utility_evidence_streak_nbytes: int
    interaction_active_output_memory_committed_nbytes: int
    interaction_relevance_probe_nbytes: int
    interaction_relevance_probe_bias_nbytes: int
    interaction_candidate_promotion_evidence_streak_nbytes: int
    interaction_candidate_reacquisition_required_nbytes: int
    behavior_nbytes: int
    joint_world_nbytes: int
    grounded_world_nbytes: int
    grounded_world_parameter_count: int
    grounded_world_parameters_touched_per_update: int
    grounded_world_update_counter_nbytes: int
    control_nbytes: int
    router_nbytes: int
    consumer_active_mask_nbytes: int
    consumer_evidence_streak_nbytes: int
    consumer_read_idle_steps_nbytes: int
    decision_cache_nbytes: int
    integrated_transition_counter_nbytes: int
    total_state_nbytes: int
    legacy_joint_world_cells_per_decision: int
    grounded_world_joint_cells_per_decision: int
    planner_cell_evaluations_per_decision: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact accounting record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class IntegratedGroundedPlannerEvaluation:
    """Both reward-model surfaces computed for an enabled grounded lane."""

    table_expected_rewards: Float[Array, " 2"]
    grounded_raw_predictions: Float[Array, "4 10"]
    grounded_reward_cells: Float[Array, "2 2"]
    grounded_expected_rewards: Float[Array, " 2"]
    predictions_valid: Bool[Array, ""]
    planner_applied: Bool[Array, ""]
    cell_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedPlannerEvaluation:
    """One complete model/Q evaluation made before action selection."""

    predicted_partner_probabilities: Float[Array, " 2"]
    partner_probabilities: Float[Array, " 2"]
    partner_probabilities_valid: Bool[Array, ""]
    probability_violation: Float[Array, ""]
    expected_rewards: Float[Array, " 2"]
    expected_outcomes: Float[Array, "2 1"]
    q_values: Float[Array, " 2"]
    centered_expected_rewards: Float[Array, " 2"]
    model_term: Float[Array, " 2"]
    applied_model_term: Float[Array, " 2"]
    planner_scores: Float[Array, " 2"]
    greedy_action: Int[Array, ""]
    cell_evaluations: Int[Array, ""]
    grounded_world: IntegratedGroundedPlannerEvaluation | None


@chex.dataclass(frozen=True)
class IntegratedPlannerSelection:
    """External epsilon-greedy selection and explicit RNG accounting."""

    action: Int[Array, ""]
    noisy_greedy_action: Int[Array, ""]
    random_action: Int[Array, ""]
    explored: Bool[Array, ""]
    externally_forced: Bool[Array, ""]
    rng_key_before: Array
    rng_key_after: Array


@chex.dataclass(frozen=True)
class IntegratedHiddenPartnerState:
    """All bounded online state plus the active pre-TD decision record.

    ``current_evaluation`` is the model/Q evaluation that selected or scored
    ``control.last_action``.  After an update it deliberately reflects the
    next decision before that transition's SARSA parameter update.  The exact
    post-update difference is retained in ``current_q_value_delta``, making
    the pre-TD Q cache reproducible from the committed control state.
    """

    state_builder: OnlineGatedStateBuilderState
    interaction: InteractionFeatureState
    behavior: BehaviorModelState
    joint_world: BoundedJointOutcomeState
    grounded_world: GroundedJointWorldModelState | None
    control: DifferentialSARSAState
    router: FeatureBankRouterState
    raw_observation: Float[Array, " 8"]
    phi: Float[Array, " 12"]
    chi: Float[Array, " 24"]
    # Persistent downstream read lease. Stored columns are not deleted when it closes.
    consumer_active_mask: Bool[Array, " 12"]
    consumer_evidence_streak: Int[Array, " 12"]
    consumer_read_idle_steps: Int[Array, " 12"]
    current_evaluation: IntegratedPlannerEvaluation
    current_q_value_delta: Float[Array, " 2"]
    current_selection: IntegratedPlannerSelection
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class IntegratedStartDiagnostics:
    """Mechanism diagnostics for initial observation consumption."""

    evaluation: IntegratedPlannerEvaluation
    selection: IntegratedPlannerSelection
    descriptors: Int[Array, "12 2"]
    descriptors_valid: Bool[Array, ""]
    state_advances: Int[Array, ""]
    integrated_step_words: UInt[Array, " 2"]
    outer_lifetime_counter_valid: Bool[Array, ""]
    child_clock_alignment_vector: Bool[Array, " 9"]
    child_clocks_aligned: Bool[Array, ""]
    all_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedStartResult:
    """Initialized state and first externally planned action."""

    state: IntegratedHiddenPartnerState
    action: Int[Array, ""]
    diagnostics: IntegratedStartDiagnostics


@chex.dataclass(frozen=True)
class IntegratedConsumerRouteAudit:
    """Scalar value-level verdicts for one atomic consumer routing transaction."""

    source_slots_exact: Bool[Array, ""]
    identity_masks_exact: Bool[Array, ""]
    stable_prefix_exact: Bool[Array, ""]
    survivor_values_exact: Bool[Array, ""]
    reset_values_exact: Bool[Array, ""]
    no_carry_reset_exact: Bool[Array, ""]
    behavior_values_exact: Bool[Array, ""]
    q_values_exact: Bool[Array, ""]
    trace_values_exact: Bool[Array, ""]
    last_observation_exact: Bool[Array, ""]
    grounded_values_exact: Bool[Array, ""]
    values_exact: Bool[Array, ""]
    lifecycle_destination_reset_exact: Bool[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedUpdateDiagnostics:
    """Prequential ordering, lifecycle, routing, and counter diagnostics."""

    current_evaluation: IntegratedPlannerEvaluation
    next_evaluation: IntegratedPlannerEvaluation
    next_selection: IntegratedPlannerSelection
    behavior_probabilities_preupdate: Float[Array, " 2"]
    behavior_prediction_matches_decision: Bool[Array, ""]
    behavior_loss_preupdate: Float[Array, ""]
    behavior_correct_preupdate: Float[Array, ""]
    behavior_gradient_chi: Float[Array, " 24"]
    behavior_gradient_phi: Float[Array, " 12"]
    grounded_world_update: GroundedJointWorldUpdateResult | None
    grounded_world_learning_enabled: Bool[Array, ""]
    # Saturating compatibility telemetry; exact capacity below is authoritative.
    grounded_world_counter_saturated: Bool[Array, ""]
    grounded_world_lifetime_counter_valid: Bool[Array, ""]
    grounded_world_lifetime_capacity_available: Bool[Array, ""]
    grounded_world_update_applied: Bool[Array, ""]
    grounded_world_pre_update_words: UInt[Array, " 2"]
    grounded_world_proposed_post_update_words: UInt[Array, " 2"]
    grounded_world_post_update_words: UInt[Array, " 2"]
    grounded_world_prediction_matches_decision: Bool[Array, ""]
    gradient_mix: RepresentationGradientMixResult | None
    mixed_gradient_chi: Float[Array, " 24"] | None
    mixed_gradient_phi: Float[Array, " 12"] | None
    state_learning: StateBuilderLearningDiagnostics
    state_builder_transition_state_valid: Bool[Array, ""]
    state_builder_transition_input_valid: Bool[Array, ""]
    state_builder_step_counter_valid: Bool[Array, ""]
    state_builder_step_capacity_available: Bool[Array, ""]
    state_builder_candidate_state_valid: Bool[Array, ""]
    state_builder_candidate_representation_valid: Bool[Array, ""]
    state_builder_transition_applied: Bool[Array, ""]
    state_builder_pre_step_words: UInt[Array, " 2"]
    state_builder_proposed_post_step_words: UInt[Array, " 2"]
    state_builder_post_step_words: UInt[Array, " 2"]
    interaction_prediction_preupdate: Float[Array, " 1"]
    interaction_error_preupdate: Float[Array, " 1"]
    interaction_metrics: Float[Array, " 7"]
    interaction_lifetime_counter_valid: Bool[Array, ""]
    interaction_lifetime_capacity_available: Bool[Array, ""]
    interaction_state_valid: Bool[Array, ""]
    interaction_candidate_state_valid: Bool[Array, ""]
    interaction_proposal_applied: Bool[Array, ""]
    interaction_update_applied: Bool[Array, ""]
    interaction_pre_step_words: UInt[Array, " 2"]
    interaction_proposed_post_step_words: UInt[Array, " 2"]
    interaction_post_step_words: UInt[Array, " 2"]
    # Compatibility fields below report the full, ungated curation proposal.
    interaction_replaced_slot: Int[Array, ""]
    interaction_promoted_candidate: Int[Array, ""]
    interaction_refreshed_candidate: Int[Array, ""]
    interaction_retired_slot: Int[Array, ""]
    interaction_retired_left: Int[Array, ""]
    interaction_retired_right: Int[Array, ""]
    # Explicit proposal/applied names make the matched freeze observable.
    interaction_proposal_replaced_slot: Int[Array, ""]
    interaction_proposal_promoted_candidate: Int[Array, ""]
    interaction_proposal_refreshed_candidate: Int[Array, ""]
    interaction_proposal_retired_slot: Int[Array, ""]
    interaction_proposal_retired_left: Int[Array, ""]
    interaction_proposal_retired_right: Int[Array, ""]
    interaction_lifecycle_proposed: Bool[Array, ""]
    interaction_lifecycle_applied: Bool[Array, ""]
    interaction_applied_replaced_slot: Int[Array, ""]
    interaction_applied_promoted_candidate: Int[Array, ""]
    interaction_applied_refreshed_candidate: Int[Array, ""]
    interaction_applied_retired_slot: Int[Array, ""]
    interaction_applied_retired_left: Int[Array, ""]
    interaction_applied_retired_right: Int[Array, ""]
    interaction_evidence_refreshed: Bool[Array, " 12"]
    interaction_retention_evidence_refreshed: Bool[Array, " 12"]
    interaction_relevance_probe_scores: Float[Array, " 12"]
    interaction_relevance_probe_errors: Float[Array, "1 12"]
    interaction_durable_read_mask: Bool[Array, " 12"]
    interaction_relevance_probe_weights_pre: Float[Array, "1 12"]
    interaction_relevance_probe_weights_post: Float[Array, "1 12"]
    interaction_relevance_probe_biases_pre: Float[Array, " 1"]
    interaction_relevance_probe_biases_post: Float[Array, " 1"]
    interaction_candidate_promotion_signal: Float[Array, " 66"]
    interaction_candidate_promotion_raw_evidence: Bool[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_pre: Int[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_updated: Int[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_proposal_post: Int[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_post: Int[Array, " 66"]
    interaction_candidate_promotion_confirmed: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_pre: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_proposal_post: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_post: Bool[Array, " 66"]
    interaction_candidate_reacquisition_confirmed: Bool[Array, " 66"]
    consumer_evidence_streak_pre: Int[Array, " 12"]
    consumer_evidence_streak_updated_pre: Int[Array, " 12"]
    consumer_evidence_streak_post: Int[Array, " 12"]
    consumer_read_idle_steps_pre: Int[Array, " 12"]
    consumer_read_idle_steps_updated_pre: Int[Array, " 12"]
    consumer_read_idle_steps_post: Int[Array, " 12"]
    consumer_read_acquire_pre: Bool[Array, " 12"]
    consumer_read_acquire_post: Bool[Array, " 12"]
    consumer_confirmed_write_pre: Bool[Array, " 12"]
    consumer_confirmed_write_post: Bool[Array, " 12"]
    # Compatibility name: this is the confirmed old-bank write gate.
    consumer_write_gate_pre: Bool[Array, " 12"]
    consumer_read_mask_pre: Bool[Array, " 12"]
    consumer_read_mask_post: Bool[Array, " 12"]
    # Compatibility names: these are the persistent read-lease masks.
    consumer_active_mask_pre: Bool[Array, " 12"]
    consumer_active_mask_post: Bool[Array, " 12"]
    interaction_matching_candidate_reset_mask: Bool[Array, " 66"]
    interaction_matching_candidate_reset_count: Int[Array, ""]
    interaction_applied_matching_candidate_reset_mask: Bool[Array, " 66"]
    interaction_applied_matching_candidate_reset_count: Int[Array, ""]
    interaction_live_feature_count: Int[Array, ""]
    interaction_vacancy_count: Int[Array, ""]
    interaction_promoted_into_vacancy: Bool[Array, ""]
    interaction_proposal_live_feature_count: Int[Array, ""]
    interaction_proposal_vacancy_count: Int[Array, ""]
    interaction_proposal_promoted_into_vacancy: Bool[Array, ""]
    interaction_applied_live_feature_count: Int[Array, ""]
    interaction_applied_vacancy_count: Int[Array, ""]
    interaction_applied_promoted_into_vacancy: Bool[Array, ""]
    random_curation_enabled: Bool[Array, ""]
    random_curation_attempted: Bool[Array, ""]
    random_curation_applied: Bool[Array, ""]
    random_active_priorities: Float[Array, " 12"]
    random_candidate_priorities: Float[Array, " 66"]
    curation_selected_active_worst_slot: Int[Array, ""]
    curation_selected_promotion_candidate: Int[Array, ""]
    curation_selected_refresh_candidate: Int[Array, ""]
    shadow_descriptors: Int[Array, "12 2"]
    proposed_descriptors: Int[Array, "12 2"]
    interaction_proposal_descriptors: Int[Array, "12 2"]
    interaction_applied_descriptors: Int[Array, "12 2"]
    shadow_descriptors_changed: Bool[Array, ""]
    route: FeatureBankRouteDiagnostics
    router_proposed_post_route_words: UInt[Array, " 2"]
    router_committed_post_route_words: UInt[Array, " 2"]
    router_proposed_post_generation_words: UInt[Array, " 2"]
    router_committed_post_generation_words: UInt[Array, " 2"]
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
    world_reward_prediction_preupdate: Float[Array, ""]
    world_outcome_prediction_preupdate: Float[Array, " 1"]
    world_reward_error: Float[Array, ""]
    world_outcome_error: Float[Array, " 1"]
    world_target_valid: Bool[Array, ""]
    world_lifetime_counter_valid: Bool[Array, ""]
    world_lifetime_capacity_available: Bool[Array, ""]
    world_update_applied: Bool[Array, ""]
    world_pre_step_words: UInt[Array, " 2"]
    world_proposed_post_step_words: UInt[Array, " 2"]
    world_post_step_words: UInt[Array, " 2"]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    control_lifetime_counter_valid: Bool[Array, ""]
    control_lifetime_capacity_available: Bool[Array, ""]
    control_update_applied: Bool[Array, ""]
    control_pre_step_words: UInt[Array, " 2"]
    control_proposed_post_step_words: UInt[Array, " 2"]
    control_post_step_words: UInt[Array, " 2"]
    transition_input_valid: Bool[Array, ""]
    decision_cache_valid: Bool[Array, ""]
    decision_cache_check_vector: Bool[Array, " 24"]
    behavior_gradient_valid: Bool[Array, ""]
    behavior_lifetime_counter_valid: Bool[Array, ""]
    behavior_lifetime_capacity_available: Bool[Array, ""]
    behavior_update_applied: Bool[Array, ""]
    behavior_pre_step_words: UInt[Array, " 2"]
    behavior_proposed_post_step_words: UInt[Array, " 2"]
    behavior_post_step_words: UInt[Array, " 2"]
    grounded_path_valid: Bool[Array, ""]
    candidate_models_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    outer_pre_step_words: UInt[Array, " 2"]
    outer_proposed_post_step_words: UInt[Array, " 2"]
    outer_committed_post_step_words: UInt[Array, " 2"]
    outer_lifetime_counter_valid: Bool[Array, ""]
    outer_lifetime_capacity_available: Bool[Array, ""]
    pre_child_clock_alignment_vector: Bool[Array, " 9"]
    pre_child_clocks_aligned: Bool[Array, ""]
    proposed_post_child_clock_alignment_vector: Bool[Array, " 9"]
    proposed_post_child_clocks_aligned: Bool[Array, ""]
    committed_post_child_clock_alignment_vector: Bool[Array, " 9"]
    committed_post_child_clocks_aligned: Bool[Array, ""]
    transaction_capacity_available: Bool[Array, ""]
    transition_observation_matches: Bool[Array, ""]
    transition_action_matches: Bool[Array, ""]
    # Compatibility name: this is the full transaction verdict, not just input syntax.
    transition_semantics_valid: Bool[Array, ""]
    transition_applied: Bool[Array, ""]
    transition_rejected: Bool[Array, ""]
    model_valid: Bool[Array, ""]
    state_builder_step_delta: Int[Array, ""]
    state_builder_learning_delta: Int[Array, ""]
    behavior_step_delta: Int[Array, ""]
    interaction_step_delta: Int[Array, ""]
    world_step_delta: Int[Array, ""]
    grounded_world_step_delta: Int[Array, ""]
    control_step_delta: Int[Array, ""]
    router_route_delta: Int[Array, ""]
    router_generation_delta: Int[Array, ""]
    integrated_step_delta: Int[Array, ""]
    all_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedUpdateResult:
    """Updated bounded state and already-selected next action."""

    state: IntegratedHiddenPartnerState
    action: Int[Array, ""]
    diagnostics: IntegratedUpdateDiagnostics


def _tree_array_nbytes(tree: object) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


class IntegratedHiddenPartnerAgent:
    """Fixed-shape L0 integration mechanism for hidden-partner-v0."""

    def __init__(
        self,
        config: IntegratedHiddenPartnerConfig | None = None,
    ) -> None:
        self._config = IntegratedHiddenPartnerConfig() if config is None else config
        if not isinstance(self._config, IntegratedHiddenPartnerConfig):
            raise TypeError("config must be an IntegratedHiddenPartnerConfig")
        cfg = self._config
        self._state_builder = OnlineGatedStateBuilder(
            OnlineGatedStateBuilderConfig(
                observation_dim=RAW_OBSERVATION_DIM,
                n_actions=N_ACTIONS,
                hidden_dim=HIDDEN_STATE_DIM,
                step_size=cfg.state_step_size,
                gradient_clip=cfg.state_gradient_clip,
                include_raw_observation=True,
            )
        )
        self._interaction = FixedBudgetInteractionLearner(
            n_features=ACTIVE_PAIR_SLOTS,
            n_tasks=1,
            step_size_output=cfg.interaction_step_size,
            utility_decay=cfg.interaction_utility_decay,
            replacement_interval=cfg.replacement_interval,
            min_feature_age=cfg.min_feature_age,
            candidate_count=CANDIDATE_PAIR_SLOTS,
            candidate_min_age=cfg.candidate_min_age,
            candidate_strategy="all_pairs",
            utility_retention_decay=cfg.active_utility_retention_decay,
            utility_retention_grace_steps=(cfg.active_utility_retention_grace_steps),
            utility_evidence_threshold=cfg.active_utility_evidence_threshold,
            evidence_gated_active_output_memory=cfg.evidence_gated_feature_memory,
            utility_evidence_confirmation_steps=(cfg.feature_evidence_confirmation_steps),
            independent_relevance_probe=cfg.independent_relevance_probe,
            relevance_probe_mode=cfg.relevance_probe_mode,
            retire_stale_features=cfg.retire_stale_features,
            candidate_promotion_floor=cfg.candidate_promotion_floor,
            candidate_promotion_confirmation_steps=(cfg.candidate_promotion_confirmation_steps),
            candidate_reacquisition_confirmation_steps=(
                cfg.candidate_reacquisition_confirmation_steps
            ),
            candidate_utility_retention_decay=(cfg.candidate_utility_retention_decay),
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            include_squares=False,
            scale_robust=True,
        )
        self._behavior = BehaviorModel(
            BehaviorModelConfig(
                n_actions=N_ACTIONS,
                step_size=cfg.behavior_step_size,
            )
        )
        self._joint_world = BoundedJointOutcomeModel(
            BoundedJointOutcomeConfig(
                n_actions=N_ACTIONS,
                outcome_dim=WORLD_OUTCOME_DIM,
                step_size=cfg.world_step_size,
            )
        )
        self._grounded_world = (
            None
            if cfg.grounded_world_model is None
            else GroundedJointWorldModel(cfg.grounded_world_model)
        )
        self._gradient_mixer_config = cfg.representation_gradient_mixer
        self._control = DifferentialSARSAAgent(
            DifferentialSARSAConfig(
                n_actions=N_ACTIONS,
                q_step_size=cfg.q_step_size,
                average_reward_step_size=cfg.average_reward_step_size,
                trace_decay=cfg.trace_decay,
                epsilon_start=cfg.epsilon,
                epsilon_end=cfg.epsilon,
                epsilon_decay_steps=0,
                use_bias=False,
            )
        )
        self._router = FeatureBankRouter(
            FeatureBankRouterConfig(
                base_dim=BASE_FEATURE_DIM,
                active_slots=ACTIVE_PAIR_SLOTS,
            )
        )
        self._memory_mask = jnp.concatenate(
            (
                jnp.ones((RAW_OBSERVATION_DIM,), dtype=jnp.float32),
                jnp.zeros((HIDDEN_STATE_DIM,), dtype=jnp.float32),
            )
        )
        self._initial_descriptors = jnp.asarray(
            cfg.initial_active_descriptors,
            dtype=jnp.int32,
        )

    @property
    def config(self) -> IntegratedHiddenPartnerConfig:
        """Immutable integrated development configuration."""
        return self._config

    @property
    def state_builder(self) -> OnlineGatedStateBuilder:
        """Causal learned-state component."""
        return self._state_builder

    @property
    def interaction_learner(self) -> FixedBudgetInteractionLearner:
        """Bounded shadow/deployed pair-discovery component."""
        return self._interaction

    @property
    def behavior_model(self) -> BehaviorModel:
        """Online partner-action predictor."""
        return self._behavior

    @property
    def joint_world_model(self) -> BoundedJointOutcomeModel:
        """Externally marginalized joint outcome table."""
        return self._joint_world

    @property
    def grounded_world_model(self) -> GroundedJointWorldModel | None:
        """Optional representation-conditioned grounded world model."""
        return self._grounded_world

    @property
    def control_agent(self) -> DifferentialSARSAAgent:
        """Differential SARSA controller."""
        return self._control

    @property
    def router(self) -> FeatureBankRouter:
        """Atomic dynamic feature-consumer router."""
        return self._router

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact fixed composition and static controls."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
            "state_builder": self._state_builder.to_config(),
            "interaction": self._interaction.to_config(),
            "behavior": self._behavior.to_config(),
            "joint_world": self._joint_world.to_config(),
            "grounded_world": (
                None if self._grounded_world is None else self._grounded_world.to_config()
            ),
            "representation_gradient_mixer": (
                None
                if self._gradient_mixer_config is None
                else self._gradient_mixer_config.to_config()
            ),
            "control": self._control.to_config(),
            "router": self._router.to_config(),
            "initial_active_descriptors": [
                list(pair) for pair in self._config.initial_active_descriptors
            ],
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    def resource_budget(
        self,
        state: IntegratedHiddenPartnerState,
    ) -> IntegratedHiddenPartnerResourceBudget:
        """Return exact persistent array bytes without double-counting consumers."""
        consumers: tuple[Array, ...]
        if self._grounded_world is None:
            if state.grounded_world is not None:
                raise ValueError("disabled grounded lane must not carry grounded state")
            consumers = self._consumer_arrays(state.behavior, state.control)
        else:
            if state.grounded_world is None:
                raise ValueError("enabled grounded lane requires grounded state")
            consumers = self._consumer_arrays_with_grounded(
                state.behavior,
                state.control,
                state.grounded_world,
            )
        router_budget = self._router.resource_budget(
            state.router,
            consumers,
        )
        cache_bytes = (
            _tree_array_nbytes(state.raw_observation)
            + _tree_array_nbytes(state.phi)
            + _tree_array_nbytes(state.chi)
            + _tree_array_nbytes(state.current_evaluation)
            + _tree_array_nbytes(state.current_q_value_delta)
            + _tree_array_nbytes(state.current_selection)
            + _tree_array_nbytes(state.step_count)
            + _tree_array_nbytes(state.step_words)
        )
        consumer_active_mask_bytes = _tree_array_nbytes(state.consumer_active_mask)
        consumer_evidence_streak_bytes = _tree_array_nbytes(state.consumer_evidence_streak)
        consumer_read_idle_steps_bytes = _tree_array_nbytes(state.consumer_read_idle_steps)
        builder_bytes = self._state_builder.resource_budget().state_bytes
        # ``start`` is jitted, so the interaction learner's Python timing
        # fields return as scalar array leaves. Count the actual integrated
        # state tree rather than the component's scientific-array-only budget.
        interaction_bytes = _tree_array_nbytes(state.interaction)
        behavior_bytes = self._behavior.resource_budget(DEPLOYED_FEATURE_DIM).state_nbytes
        world_bytes = self._joint_world.resource_budget.state_nbytes
        grounded_bytes = (
            0 if state.grounded_world is None else _tree_array_nbytes(state.grounded_world)
        )
        grounded_budget = (
            None if self._grounded_world is None else self._grounded_world.resource_budget
        )
        control_bytes = _tree_array_nbytes(state.control)
        total = (
            builder_bytes
            + interaction_bytes
            + behavior_bytes
            + world_bytes
            + grounded_bytes
            + control_bytes
            + router_budget.router_state_nbytes
            + consumer_active_mask_bytes
            + consumer_evidence_streak_bytes
            + consumer_read_idle_steps_bytes
            + cache_bytes
        )
        return IntegratedHiddenPartnerResourceBudget(
            raw_observation_dim=RAW_OBSERVATION_DIM,
            base_feature_dim=BASE_FEATURE_DIM,
            active_pair_slots=ACTIVE_PAIR_SLOTS,
            candidate_pair_slots=CANDIDATE_PAIR_SLOTS,
            deployed_feature_dim=DEPLOYED_FEATURE_DIM,
            state_builder_nbytes=builder_bytes,
            interaction_nbytes=interaction_bytes,
            interaction_evidence_idle_nbytes=int(state.interaction.evidence_idle_steps.nbytes),
            interaction_utility_evidence_streak_nbytes=int(
                state.interaction.utility_evidence_streak.nbytes
            ),
            interaction_active_output_memory_committed_nbytes=int(
                state.interaction.active_output_memory_committed.nbytes
            ),
            interaction_relevance_probe_nbytes=int(
                state.interaction.relevance_probe_weights.nbytes
                + state.interaction.relevance_probe_biases.nbytes
            ),
            interaction_relevance_probe_bias_nbytes=int(
                state.interaction.relevance_probe_biases.nbytes
            ),
            interaction_candidate_promotion_evidence_streak_nbytes=int(
                state.interaction.candidate_promotion_evidence_streak.nbytes
            ),
            interaction_candidate_reacquisition_required_nbytes=int(
                state.interaction.candidate_reacquisition_required.nbytes
            ),
            behavior_nbytes=behavior_bytes,
            joint_world_nbytes=world_bytes,
            grounded_world_nbytes=grounded_bytes,
            grounded_world_parameter_count=(
                0 if grounded_budget is None else grounded_budget.trainable_float32_scalars
            ),
            grounded_world_parameters_touched_per_update=(
                0
                if grounded_budget is None
                else grounded_budget.learned_float32_scalars_touched_per_update
            ),
            grounded_world_update_counter_nbytes=(
                0
                if state.grounded_world is None
                else int(
                    state.grounded_world.update_count.nbytes
                    + state.grounded_world.update_words.nbytes
                )
            ),
            control_nbytes=control_bytes,
            router_nbytes=router_budget.router_state_nbytes,
            consumer_active_mask_nbytes=consumer_active_mask_bytes,
            consumer_evidence_streak_nbytes=consumer_evidence_streak_bytes,
            consumer_read_idle_steps_nbytes=consumer_read_idle_steps_bytes,
            decision_cache_nbytes=cache_bytes,
            integrated_transition_counter_nbytes=int(
                state.step_count.nbytes + state.step_words.nbytes
            ),
            total_state_nbytes=total,
            legacy_joint_world_cells_per_decision=(
                self._joint_world.resource_budget.planner_cell_evaluations_per_decision
            ),
            grounded_world_joint_cells_per_decision=(
                0 if self._grounded_world is None else N_ACTIONS * N_ACTIONS
            ),
            planner_cell_evaluations_per_decision=(
                self._joint_world.resource_budget.planner_cell_evaluations_per_decision
                + (0 if self._grounded_world is None else N_ACTIONS * N_ACTIONS)
            ),
            replay_capacity=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def build_chi(
        self,
        phi: Array,
        descriptors: Array,
        consumer_active_mask: Array | None = None,
    ) -> Float[Array, " 24"]:
        """Append descriptor products under an optional downstream read gate.

        Static shape/dtype violations fail while tracing.  A nonfinite ``phi``
        or dynamically invalid descriptor bank returns the all-zero neutral
        representation; exact ``(-1, -1)`` vacancy descriptors remain valid.
        """
        raw_base = _require_array_contract(
            phi,
            name="phi",
            shape=(BASE_FEATURE_DIM,),
            dtype=jnp.float32,
        )
        pairs = _require_array_contract(
            descriptors,
            name="descriptors",
            shape=(ACTIVE_PAIR_SLOTS, 2),
            dtype=jnp.int32,
        )
        consumer_mask = (
            jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
            if consumer_active_mask is None
            else _require_array_contract(
                consumer_active_mask,
                name="consumer_active_mask",
                shape=(ACTIVE_PAIR_SLOTS,),
                dtype=jnp.bool_,
            )
        )
        descriptor_validation = self._router.validate_descriptors(pairs)
        kernel_valid = jnp.all(jnp.isfinite(raw_base)) & descriptor_validation.valid
        base = jnp.where(jnp.isfinite(raw_base), raw_base, 0.0)
        deployed = self._deployed_phi(base)
        left = pairs[:, 0]
        right = pairs[:, 1]
        live = (
            (left >= 0)
            & (right >= 0)
            & (left < BASE_FEATURE_DIM)
            & (right < BASE_FEATURE_DIM)
            & (left < right)
        )
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        products = (
            deployed[safe_left]
            * deployed[safe_right]
            * live.astype(jnp.float32)
            * consumer_mask.astype(jnp.float32)
        )
        result = jnp.concatenate((deployed, products))
        return jnp.where(kernel_valid, result, jnp.zeros_like(result))

    @functools.partial(jax.jit, static_argnums=(0,))
    def chain_chi_gradient_to_phi(
        self,
        phi: Array,
        descriptors: Array,
        chi_gradient: Array,
        consumer_active_mask: Array | None = None,
    ) -> Float[Array, " 12"]:
        """Apply the gated product-feature chain rule back to base ``phi``.

        Static contracts fail while tracing.  Nonfinite arrays or a dynamically
        invalid descriptor bank produce an all-zero neutral gradient.
        """
        raw_base = _require_array_contract(
            phi,
            name="phi",
            shape=(BASE_FEATURE_DIM,),
            dtype=jnp.float32,
        )
        pairs = _require_array_contract(
            descriptors,
            name="descriptors",
            shape=(ACTIVE_PAIR_SLOTS, 2),
            dtype=jnp.int32,
        )
        raw_gradient = _require_array_contract(
            chi_gradient,
            name="chi_gradient",
            shape=(DEPLOYED_FEATURE_DIM,),
            dtype=jnp.float32,
        )
        consumer_mask = (
            jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
            if consumer_active_mask is None
            else _require_array_contract(
                consumer_active_mask,
                name="consumer_active_mask",
                shape=(ACTIVE_PAIR_SLOTS,),
                dtype=jnp.bool_,
            )
        )
        descriptor_validation = self._router.validate_descriptors(pairs)
        kernel_valid = (
            jnp.all(jnp.isfinite(raw_base))
            & jnp.all(jnp.isfinite(raw_gradient))
            & descriptor_validation.valid
        )
        base = jnp.where(jnp.isfinite(raw_base), raw_base, 0.0)
        deployed = self._deployed_phi(base)
        gradient = jnp.where(jnp.isfinite(raw_gradient), raw_gradient, 0.0)
        left = pairs[:, 0]
        right = pairs[:, 1]
        live = (
            (left >= 0)
            & (right >= 0)
            & (left < BASE_FEATURE_DIM)
            & (right < BASE_FEATURE_DIM)
            & (left < right)
        )
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        pair_gradient = (
            gradient[BASE_FEATURE_DIM:]
            * live.astype(jnp.float32)
            * consumer_mask.astype(jnp.float32)
        )
        base_gradient = gradient[:BASE_FEATURE_DIM]
        base_gradient = base_gradient.at[safe_left].add(pair_gradient * deployed[safe_right])
        base_gradient = base_gradient.at[safe_right].add(pair_gradient * deployed[safe_left])
        result = base_gradient * self._deployment_derivative_mask()
        return jnp.where(kernel_valid, result, jnp.zeros_like(result))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _chain_behavior_and_mixed_gradients_to_phi(
        self,
        phi: Array,
        descriptors: Array,
        behavior_gradient: Array,
        mixed_gradient: Array,
        consumer_active_mask: Array,
    ) -> Float[Array, "2 12"]:
        """Chain both named sources in one fixed two-row product-rule call."""
        raw_base = _require_array_contract(
            phi,
            name="phi",
            shape=(BASE_FEATURE_DIM,),
            dtype=jnp.float32,
        )
        pairs = _require_array_contract(
            descriptors,
            name="descriptors",
            shape=(ACTIVE_PAIR_SLOTS, 2),
            dtype=jnp.int32,
        )
        gradients = jnp.stack(
            (
                _require_array_contract(
                    behavior_gradient,
                    name="behavior_gradient",
                    shape=(DEPLOYED_FEATURE_DIM,),
                    dtype=jnp.float32,
                ),
                _require_array_contract(
                    mixed_gradient,
                    name="mixed_gradient",
                    shape=(DEPLOYED_FEATURE_DIM,),
                    dtype=jnp.float32,
                ),
            )
        )
        consumer_mask = _require_array_contract(
            consumer_active_mask,
            name="consumer_active_mask",
            shape=(ACTIVE_PAIR_SLOTS,),
            dtype=jnp.bool_,
        )
        descriptor_validation = self._router.validate_descriptors(pairs)
        kernel_valid = (
            jnp.all(jnp.isfinite(raw_base))
            & jnp.all(jnp.isfinite(gradients))
            & descriptor_validation.valid
        )
        base = jnp.where(jnp.isfinite(raw_base), raw_base, 0.0)
        deployed = self._deployed_phi(base)
        safe_gradients = jnp.where(jnp.isfinite(gradients), gradients, 0.0)
        left = pairs[:, 0]
        right = pairs[:, 1]
        live = (
            (left >= 0)
            & (right >= 0)
            & (left < BASE_FEATURE_DIM)
            & (right < BASE_FEATURE_DIM)
            & (left < right)
        )
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        pair_gradients = (
            safe_gradients[:, BASE_FEATURE_DIM:]
            * live[None, :].astype(jnp.float32)
            * consumer_mask[None, :].astype(jnp.float32)
        )
        base_gradients = safe_gradients[:, :BASE_FEATURE_DIM]
        base_gradients = base_gradients.at[:, safe_left].add(
            pair_gradients * deployed[safe_right][None, :]
        )
        base_gradients = base_gradients.at[:, safe_right].add(
            pair_gradients * deployed[safe_left][None, :]
        )
        result = base_gradients * self._deployment_derivative_mask()[None, :]
        return jnp.where(kernel_valid, result, jnp.zeros_like(result))

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate_models(
        self,
        behavior_state: BehaviorModelState,
        world_state: BoundedJointOutcomeState,
        control_state: DifferentialSARSAState,
        chi: Array,
        grounded_world_state: GroundedJointWorldModelState | None = None,
    ) -> IntegratedPlannerEvaluation:
        """Evaluate the table and every enabled grounded joint-action cell."""
        features = jnp.asarray(chi, dtype=jnp.float32).reshape((DEPLOYED_FEATURE_DIM,))
        predicted_probabilities = self._behavior.predict_probabilities(
            behavior_state,
            features,
        )
        applied_probabilities = (
            jnp.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=jnp.float32)
            if self._config.uniform_partner_belief
            else predicted_probabilities
        )
        marginal = self._joint_world.marginalize(
            world_state,
            applied_probabilities,
        )
        if self._grounded_world is None:
            if grounded_world_state is not None:
                raise ValueError("disabled grounded lane must not receive grounded state")
            grounded_evaluation = None
            planning_rewards = marginal.expected_rewards
            total_cell_evaluations = marginal.cell_evaluations
        else:
            if grounded_world_state is None:
                raise ValueError("enabled grounded lane requires grounded state")
            grounded_predictions = tuple(
                self._grounded_world.predict(
                    grounded_world_state,
                    features,
                    jnp.asarray(focal_action, dtype=jnp.int32),
                    jnp.asarray(partner_action, dtype=jnp.int32),
                )
                for focal_action in range(N_ACTIONS)
                for partner_action in range(N_ACTIONS)
            )
            grounded_reward_cells = jnp.stack(
                tuple(prediction.reward for prediction in grounded_predictions)
            ).reshape((N_ACTIONS, N_ACTIONS))
            grounded_raw_predictions = jnp.stack(
                tuple(prediction.raw_predictions for prediction in grounded_predictions)
            )
            grounded_predictions_valid = jnp.all(
                jnp.stack(tuple(prediction.valid for prediction in grounded_predictions))
            )
            grounded_expected_rewards = grounded_reward_cells @ applied_probabilities
            grounded_planner_applied = jnp.asarray(
                self._config.grounded_world_planning_enabled and self._config.planning_enabled,
                dtype=jnp.bool_,
            )
            grounded_evaluation = IntegratedGroundedPlannerEvaluation(
                table_expected_rewards=marginal.expected_rewards,
                grounded_raw_predictions=grounded_raw_predictions,
                grounded_reward_cells=grounded_reward_cells,
                grounded_expected_rewards=grounded_expected_rewards,
                predictions_valid=grounded_predictions_valid,
                planner_applied=grounded_planner_applied,
                cell_evaluations=jnp.asarray(
                    N_ACTIONS * N_ACTIONS,
                    dtype=jnp.int32,
                ),
            )
            planning_rewards = (
                grounded_expected_rewards
                if self._config.grounded_world_planning_enabled
                else marginal.expected_rewards
            )
            total_cell_evaluations = (
                marginal.cell_evaluations + grounded_evaluation.cell_evaluations
            )
        q_values = self._control.q_values(control_state, features)
        centered = planning_rewards - jnp.mean(planning_rewards)
        model_term = jnp.asarray(self._config.planner_lambda, dtype=jnp.float32) * centered
        applied_model_term = (
            model_term if self._config.planning_enabled else jnp.zeros_like(model_term)
        )
        scores = q_values + applied_model_term
        return IntegratedPlannerEvaluation(
            predicted_partner_probabilities=predicted_probabilities,
            partner_probabilities=marginal.partner_probabilities,
            partner_probabilities_valid=(marginal.partner_probabilities_valid),
            probability_violation=marginal.probability_violation,
            expected_rewards=planning_rewards,
            expected_outcomes=marginal.expected_outcomes,
            q_values=q_values,
            centered_expected_rewards=centered,
            model_term=model_term,
            applied_model_term=applied_model_term,
            planner_scores=scores,
            greedy_action=jnp.argmax(scores).astype(jnp.int32),
            cell_evaluations=total_cell_evaluations,
            grounded_world=grounded_evaluation,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_planner_action(
        self,
        control_state: DifferentialSARSAState,
        planner_scores: Array,
    ) -> IntegratedPlannerSelection:
        """Use SARSA's RNG and epsilon semantics with external planner scores."""
        scores = jnp.asarray(planner_scores, dtype=jnp.float32).reshape((N_ACTIONS,))
        key, explore_key, noise_key, random_key = jr.split(
            control_state.rng_key,
            4,
        )
        greedy_noise = jr.gumbel(
            noise_key,
            shape=scores.shape,
        ) * jnp.asarray(1e-6, dtype=jnp.float32)
        noisy_greedy = jnp.argmax(scores + greedy_noise).astype(jnp.int32)
        random_action = jr.randint(
            random_key,
            (),
            0,
            N_ACTIONS,
        ).astype(jnp.int32)
        explored = jr.uniform(explore_key) < control_state.epsilon
        action = jax.lax.select(
            explored,
            random_action,
            noisy_greedy,
        )
        return IntegratedPlannerSelection(
            action=action,
            noisy_greedy_action=noisy_greedy,
            random_action=random_action,
            explored=explored,
            externally_forced=jnp.asarray(False, dtype=jnp.bool_),
            rng_key_before=control_state.rng_key,
            rng_key_after=key,
        )

    def _child_clock_alignment_vector(
        self,
        state: IntegratedHiddenPartnerState,
    ) -> Bool[Array, " 9"]:
        """Authenticate every persistent child clock against the outer identity."""

        expected_builder_step, _ = _checked_lifetime_words_increment(
            state.step_words
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        expected_builder_update = (
            state.step_words
            if self._config.state_learning_enabled
            else zero_words
        )
        if self._grounded_world is None:
            if state.grounded_world is not None:
                raise ValueError("disabled grounded lane must not carry grounded state")
            grounded_clock_aligned = jnp.asarray(True, dtype=jnp.bool_)
        else:
            if state.grounded_world is None:
                raise ValueError("enabled grounded lane requires grounded state")
            expected_grounded_update = (
                state.step_words
                if self._config.grounded_world_learning_enabled
                else zero_words
            )
            grounded_clock_aligned = jnp.array_equal(
                state.grounded_world.update_words,
                expected_grounded_update,
            )
        checks = jnp.stack(
            (
                jnp.array_equal(state.behavior.step_words, state.step_words),
                jnp.array_equal(state.joint_world.step_words, state.step_words),
                jnp.array_equal(state.control.step_words, state.step_words),
                jnp.array_equal(state.router.route_words, state.step_words),
                jnp.array_equal(state.interaction.step_words, state.step_words),
                jnp.array_equal(
                    state.state_builder.step_words,
                    expected_builder_step,
                ),
                jnp.array_equal(
                    state.state_builder.update_words,
                    expected_builder_update,
                ),
                grounded_clock_aligned,
                _lifetime_words_le(
                    state.router.generation_words,
                    state.router.route_words,
                ),
            )
        )
        if len(INTEGRATED_CHILD_CLOCK_ALIGNMENT_ORDER) != 9:
            raise RuntimeError("integrated child clock order must remain width 9")
        return checks

    def _current_decision_cache_check_vector(
        self,
        state: IntegratedHiddenPartnerState,
        fresh: IntegratedPlannerEvaluation,
    ) -> Bool[Array, " 24"]:
        """Return every reproducible or internally bound cache check in fixed order.

        ``current_evaluation.q_values`` records the values used to score the
        current action *before* the preceding SARSA update.  The persistent
        ``current_q_value_delta`` binds those historical values exactly to the
        committed controller.  Model-derived leaves and ordinary policy RNG
        primitives are recomputed exactly.  In externally-forced mode only the
        applied action differs from that replayed ordinary policy decision.

        ``probability_violation`` is an advisory reduction over the already
        exact-bound probability vector. XLA may reassociate that reduction
        across call-boundary fusion, so that scalar alone permits one float32
        machine epsilon while its validity bit and every decision input remain
        exact.

        This is an internal algebraic provenance check, not an authenticity
        mechanism.  A coordinated coherent mutation of Q values, scores, and
        their delta still requires an external checkpoint/artifact digest to
        detect.
        """

        cached = state.current_evaluation
        selection = state.current_selection

        def exact(left: Array, right: Array) -> Array:
            return jnp.array_equal(jnp.asarray(left), jnp.asarray(right))

        centered = cached.expected_rewards - jnp.mean(cached.expected_rewards)
        model_term = jnp.asarray(self._config.planner_lambda, dtype=jnp.float32) * centered
        applied_model_term = (
            model_term if self._config.planning_enabled else jnp.zeros_like(model_term)
        )
        planner_scores = cached.q_values + applied_model_term
        expected_q_delta = fresh.q_values - cached.q_values
        base_checks = (
            exact(cached.predicted_partner_probabilities, fresh.predicted_partner_probabilities),
            exact(cached.partner_probabilities, fresh.partner_probabilities),
            exact(cached.partner_probabilities_valid, fresh.partner_probabilities_valid),
            jnp.isclose(
                cached.probability_violation,
                fresh.probability_violation,
                atol=jnp.asarray(2.0**-23, dtype=jnp.float32),
                rtol=0.0,
            ),
            exact(cached.expected_rewards, fresh.expected_rewards),
            exact(cached.expected_outcomes, fresh.expected_outcomes),
            exact(cached.centered_expected_rewards, centered),
            exact(cached.model_term, model_term),
            exact(cached.applied_model_term, applied_model_term),
            exact(cached.planner_scores, planner_scores),
            exact(cached.greedy_action, jnp.argmax(planner_scores).astype(jnp.int32)),
            exact(cached.cell_evaluations, fresh.cell_evaluations),
            exact(state.control.last_observation, state.chi),
            exact(state.current_q_value_delta, expected_q_delta),
            exact(
                state.control.epsilon,
                jnp.asarray(self._config.epsilon, dtype=jnp.float32),
            ),
            exact(state.control.q_bias, jnp.zeros((N_ACTIONS,), dtype=jnp.float32)),
            exact(
                state.control.q_trace_bias,
                jnp.zeros((N_ACTIONS,), dtype=jnp.float32),
            ),
            exact(state.control.step_count, state.step_count),
            exact(state.control.step_words, state.step_words),
        )

        if self._grounded_world is None:
            if cached.grounded_world is not None or fresh.grounded_world is not None:
                raise ValueError("disabled grounded lane must not carry grounded evaluation")
            grounded_coherent = jnp.asarray(True, dtype=jnp.bool_)
        else:
            if cached.grounded_world is None or fresh.grounded_world is None:
                raise ValueError("enabled grounded lane requires grounded evaluation")
            cached_grounded = cached.grounded_world
            fresh_grounded = fresh.grounded_world
            grounded_coherent = jnp.all(
                jnp.stack(
                    (
                        exact(
                            cached_grounded.table_expected_rewards,
                            fresh_grounded.table_expected_rewards,
                        ),
                        exact(
                            cached_grounded.grounded_raw_predictions,
                            fresh_grounded.grounded_raw_predictions,
                        ),
                        exact(
                            cached_grounded.grounded_reward_cells,
                            fresh_grounded.grounded_reward_cells,
                        ),
                        exact(
                            cached_grounded.grounded_expected_rewards,
                            fresh_grounded.grounded_expected_rewards,
                        ),
                        exact(cached_grounded.predictions_valid, fresh_grounded.predictions_valid),
                        exact(cached_grounded.planner_applied, fresh_grounded.planner_applied),
                        exact(cached_grounded.cell_evaluations, fresh_grounded.cell_evaluations),
                    )
                )
            )

        replay_control = state.control.replace(rng_key=selection.rng_key_before)
        replayed_selection = self.select_planner_action(
            replay_control,
            cached.planner_scores,
        )
        ordinary_policy_action = jax.lax.select(
            selection.explored,
            selection.random_action,
            selection.noisy_greedy_action,
        )
        externally_forced = self._config.action_selection_mode == "externally_forced"
        applied_action_coherent = (
            jnp.asarray(True, dtype=jnp.bool_)
            if externally_forced
            else exact(selection.action, ordinary_policy_action)
        )
        selection_coherent = jnp.all(
            jnp.stack(
                (
                    applied_action_coherent,
                    exact(selection.noisy_greedy_action, replayed_selection.noisy_greedy_action),
                    exact(selection.random_action, replayed_selection.random_action),
                    exact(selection.explored, replayed_selection.explored),
                    exact(ordinary_policy_action, replayed_selection.action),
                    exact(
                        selection.externally_forced,
                        jnp.asarray(externally_forced, dtype=jnp.bool_),
                    ),
                    exact(
                        jr.key_data(selection.rng_key_before),
                        jr.key_data(replayed_selection.rng_key_before),
                    ),
                    exact(
                        jr.key_data(selection.rng_key_after),
                        jr.key_data(replayed_selection.rng_key_after),
                    ),
                    exact(selection.action, state.control.last_action),
                    (selection.action >= 0) & (selection.action < N_ACTIONS),
                    exact(jr.key_data(selection.rng_key_after), jr.key_data(state.control.rng_key)),
                )
            )
        )
        checks = jnp.stack(
            (
                *base_checks,
                grounded_coherent,
                selection_coherent,
                _numeric_tree_finite(cached),
                _numeric_tree_finite(state.current_q_value_delta),
                _numeric_tree_finite(selection),
            )
        )
        if len(INTEGRATED_DECISION_CACHE_CHECK_ORDER) != 24:
            raise RuntimeError("integrated decision-cache check order must remain width 24")
        return checks

    def _current_decision_cache_coherent(
        self,
        state: IntegratedHiddenPartnerState,
        fresh: IntegratedPlannerEvaluation,
    ) -> Bool[Array, ""]:
        """Return whether all fixed-order decision-cache checks pass."""

        return jnp.all(self._current_decision_cache_check_vector(state, fresh))

    def start(
        self,
        raw_observation: Array,
        key: Array,
    ) -> IntegratedStartResult:
        """Initialize in ordinary agent-selected action mode."""
        if self._config.action_selection_mode != "agent":
            raise ValueError(
                "start is unavailable when action_selection_mode is externally_forced; "
                "use start_with_forced_action"
            )
        return self._start_impl(raw_observation, key, forced_action=None)

    def start_with_forced_action(
        self,
        raw_observation: Array,
        key: Array,
        action: Array,
    ) -> IntegratedStartResult:
        """Initialize while applying one host-validated external action."""
        if self._config.action_selection_mode != "externally_forced":
            raise ValueError(
                "start_with_forced_action is unavailable when action_selection_mode is agent"
            )
        forced_action = _require_array_contract(
            action,
            name="forced_action",
            shape=(),
            dtype=jnp.int32,
        )
        forced_action_host = int(jax.device_get(forced_action))
        if not 0 <= forced_action_host < N_ACTIONS:
            raise ValueError(f"forced_action must lie in [0, {N_ACTIONS})")
        return self._start_impl(raw_observation, key, forced_action=forced_action)

    def _start_impl(
        self,
        raw_observation: Array,
        key: Array,
        *,
        forced_action: Array | None,
    ) -> IntegratedStartResult:
        """Initialize every bounded component and select or apply the first action.

        This is intentionally a host wrapper because component initializers
        attach lifecycle timestamps. The array-level methods it invokes remain
        jitted, and the continuing :meth:`update` path is fully JIT/scan
        compatible.
        """
        raw = _require_array_contract(
            raw_observation,
            name="raw_observation",
            shape=(RAW_OBSERVATION_DIM,),
            dtype=jnp.float32,
        )
        if not bool(jax.device_get(jnp.all(jnp.isfinite(raw)))):
            raise ValueError("raw_observation must contain only finite values")
        builder_key, interaction_key, behavior_key, control_key = jr.split(
            key,
            4,
        )
        builder_initial = self._state_builder.init(builder_key)
        builder_start = self._state_builder.update_with_status(
            builder_initial,
            raw,
            -1,
            0.0,
            1.0,
        )
        if not bool(jax.device_get(builder_start.transition_applied)):
            raise ValueError("initial state-builder transition was rejected")
        builder_state = builder_start.state
        phi = builder_start.representation
        interaction_initial = self._interaction.init(
            BASE_FEATURE_DIM,
            interaction_key,
        )
        interaction_state = interaction_initial.replace(
            feature_left=self._initial_descriptors[:, 0],
            feature_right=self._initial_descriptors[:, 1],
            birth_timestamp=jnp.asarray(
                interaction_initial.birth_timestamp,
                dtype=jnp.float32,
            ),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        router_state = self._router.init(self._initial_descriptors)
        behavior_state = self._behavior.init(
            DEPLOYED_FEATURE_DIM,
            behavior_key,
        )
        world_state = self._joint_world.init()
        grounded_world_state = (
            None
            if self._grounded_world is None
            else self._grounded_world.init(jr.fold_in(key, jnp.asarray(0x47574D, dtype=jnp.uint32)))
        )
        control_initial = self._control.init(
            DEPLOYED_FEATURE_DIM,
            control_key,
        )
        control_state = control_initial.replace(
            birth_timestamp=jnp.asarray(
                control_initial.birth_timestamp,
                dtype=jnp.float32,
            ),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        descriptor_validation = self._router.validate_descriptors(router_state.descriptors)
        consumer_active_mask = (
            jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
            if self._config.evidence_gated_consumer_memory
            else descriptor_validation.live_mask
        )
        consumer_evidence_streak = jnp.zeros(
            (ACTIVE_PAIR_SLOTS,),
            dtype=jnp.int32,
        )
        consumer_read_idle_steps = jnp.zeros(
            (ACTIVE_PAIR_SLOTS,),
            dtype=jnp.int32,
        )
        chi = self.build_chi(
            phi,
            router_state.descriptors,
            self._effective_pair_read_mask(
                interaction_state,
                consumer_active_mask,
            ),
        )
        evaluation = self.evaluate_models(
            behavior_state,
            world_state,
            control_state,
            chi,
            grounded_world_state,
        )
        grounded_evaluation_valid = (
            jnp.asarray(True, dtype=jnp.bool_)
            if evaluation.grounded_world is None
            else evaluation.grounded_world.predictions_valid
        )
        initial_evaluation_valid = (
            evaluation.partner_probabilities_valid & grounded_evaluation_valid
        )
        if not bool(jax.device_get(initial_evaluation_valid)):
            raise ValueError("initial planner evaluation is invalid")
        selection = self.select_planner_action(
            control_state,
            evaluation.planner_scores,
        )
        if forced_action is not None:
            selection = selection.replace(
                action=forced_action,
                externally_forced=jnp.asarray(True, dtype=jnp.bool_),
            )
        control_with_rng = control_state.replace(rng_key=selection.rng_key_after)
        control_started, action = self._control.start_with_action(
            control_with_rng,
            chi,
            selection.action,
        )
        state = IntegratedHiddenPartnerState(
            state_builder=builder_state,
            interaction=interaction_state,
            behavior=behavior_state,
            joint_world=world_state,
            grounded_world=grounded_world_state,
            control=control_started,
            router=router_state,
            raw_observation=raw,
            phi=phi,
            chi=chi,
            consumer_active_mask=consumer_active_mask,
            consumer_evidence_streak=consumer_evidence_streak,
            consumer_read_idle_steps=consumer_read_idle_steps,
            current_evaluation=evaluation,
            current_q_value_delta=jnp.zeros((N_ACTIONS,), dtype=jnp.float32),
            current_selection=selection,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )
        outer_lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        child_clock_alignment_vector = self._child_clock_alignment_vector(state)
        child_clocks_aligned = jnp.all(child_clock_alignment_vector)
        start_valid = (
            self._start_finite(state)
            & outer_lifetime_counter_valid
            & jnp.all(state.step_words == jnp.asarray(0, dtype=jnp.uint32))
            & (state.step_count == jnp.asarray(0, dtype=jnp.int32))
            & child_clocks_aligned
        )
        if not bool(jax.device_get(start_valid)):
            raise ValueError("initial integrated state is invalid")
        diagnostics = IntegratedStartDiagnostics(
            evaluation=evaluation,
            selection=selection,
            descriptors=router_state.descriptors,
            descriptors_valid=descriptor_validation.valid,
            state_advances=(builder_state.step_count - builder_initial.step_count),
            integrated_step_words=state.step_words,
            outer_lifetime_counter_valid=outer_lifetime_counter_valid,
            child_clock_alignment_vector=child_clock_alignment_vector,
            child_clocks_aligned=child_clocks_aligned,
            all_finite=start_valid,
        )
        return IntegratedStartResult(
            state=state,
            action=action,
            diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: IntegratedHiddenPartnerState,
        transition: HiddenPartnerTransition,
    ) -> IntegratedUpdateResult:
        """Consume one transition in ordinary agent-selected action mode."""
        if self._config.action_selection_mode != "agent":
            raise ValueError(
                "update is unavailable when action_selection_mode is externally_forced; "
                "use update_with_forced_next_action"
            )
        return self._update_impl(
            state,
            transition,
            forced_next_action=None,
            forced_next_action_valid=jnp.asarray(True, dtype=jnp.bool_),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_forced_next_action(
        self,
        state: IntegratedHiddenPartnerState,
        transition: HiddenPartnerTransition,
        next_action: Array,
    ) -> IntegratedUpdateResult:
        """Consume one transition and apply a dynamic external next action."""
        if self._config.action_selection_mode != "externally_forced":
            raise ValueError(
                "update_with_forced_next_action is unavailable when "
                "action_selection_mode is agent"
            )
        raw_forced_next_action = _require_array_contract(
            next_action,
            name="forced_next_action",
            shape=(),
            dtype=jnp.int32,
        )
        forced_next_action_valid = (raw_forced_next_action >= 0) & (
            raw_forced_next_action < N_ACTIONS
        )
        safe_forced_next_action = jnp.where(
            forced_next_action_valid,
            raw_forced_next_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return self._update_impl(
            state,
            transition,
            forced_next_action=safe_forced_next_action,
            forced_next_action_valid=forced_next_action_valid,
        )

    def _update_impl(
        self,
        state: IntegratedHiddenPartnerState,
        transition: HiddenPartnerTransition,
        *,
        forced_next_action: Array | None,
        forced_next_action_valid: Array,
    ) -> IntegratedUpdateResult:
        """Consume one exact hidden-partner transition in causal order.

        Static shape/dtype violations fail while tracing.  Dynamic transition
        violations are an atomic no-op: the exact old state and old action are
        returned, every persistent counter delta is zero, and diagnostics set
        ``transition_rejected``.  This contract is identical under eager JAX,
        :func:`jax.jit`, and :func:`jax.lax.scan`.
        """
        _require_array_contract(
            state.current_q_value_delta,
            name="state.current_q_value_delta",
            shape=(N_ACTIONS,),
            dtype=jnp.float32,
        )
        raw_current = _require_array_contract(
            transition.observation,
            name="transition.observation",
            shape=(RAW_OBSERVATION_DIM,),
            dtype=jnp.float32,
        )
        raw_next = _require_array_contract(
            transition.next_observation,
            name="transition.next_observation",
            shape=(RAW_OBSERVATION_DIM,),
            dtype=jnp.float32,
        )
        raw_focal_action = _require_array_contract(
            transition.focal_action,
            name="transition.focal_action",
            shape=(),
            dtype=jnp.int32,
        )
        raw_partner_action = _require_array_contract(
            transition.partner_action,
            name="transition.partner_action",
            shape=(),
            dtype=jnp.int32,
        )
        raw_reward = _require_array_contract(
            transition.reward,
            name="transition.reward",
            shape=(),
            dtype=jnp.float32,
        )
        raw_outcome = _require_array_contract(
            transition.outcome,
            name="transition.outcome",
            shape=(),
            dtype=jnp.float32,
        )
        raw_discount = _require_array_contract(
            transition.discount,
            name="transition.discount",
            shape=(),
            dtype=jnp.float32,
        )
        terminated = _require_array_contract(
            transition.terminated,
            name="transition.terminated",
            shape=(),
            dtype=jnp.bool_,
        )
        outer_proposed_post_step_words, outer_lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        outer_lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        pre_child_clock_alignment_vector = self._child_clock_alignment_vector(
            state
        )
        pre_child_clocks_aligned = jnp.all(pre_child_clock_alignment_vector)

        observations_finite = jnp.all(jnp.isfinite(raw_current)) & jnp.all(jnp.isfinite(raw_next))
        observation_matches = jnp.all(raw_current == state.raw_observation)
        focal_action_valid = (raw_focal_action >= 0) & (raw_focal_action < N_ACTIONS)
        partner_action_valid = (raw_partner_action >= 0) & (raw_partner_action < N_ACTIONS)
        action_ids_valid = focal_action_valid & partner_action_valid
        action_matches = raw_focal_action == state.control.last_action
        outcome_valid = jnp.isfinite(raw_outcome) & ((raw_outcome == -1.0) | (raw_outcome == 1.0))
        reward_semantics = jnp.isfinite(raw_reward) & jnp.isclose(
            raw_reward,
            (1.0 + raw_outcome) / 2.0,
            atol=1e-6,
            rtol=0.0,
        )
        discount_valid = jnp.isfinite(raw_discount) & (raw_discount == 1.0)
        transition_valid = (
            observations_finite
            & observation_matches
            & action_ids_valid
            & action_matches
            & outcome_valid
            & reward_semantics
            & discount_valid
            & ~terminated
            & forced_next_action_valid
        )
        transition_input_valid = transition_valid
        transition_valid = (
            transition_valid
            & outer_lifetime_counter_valid
            & outer_lifetime_capacity_available
            & pre_child_clocks_aligned
        )

        next_raw = jnp.where(jnp.isfinite(raw_next), raw_next, state.raw_observation)
        focal_action = jnp.where(
            focal_action_valid,
            raw_focal_action,
            state.control.last_action,
        )
        partner_action = jnp.where(
            partner_action_valid,
            raw_partner_action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        reward = jnp.where(jnp.isfinite(raw_reward), raw_reward, 0.0)
        outcome = jnp.where(outcome_valid, raw_outcome, 0.0)
        discount = jnp.where(jnp.isfinite(raw_discount), raw_discount, 1.0)

        current_evaluation = state.current_evaluation
        fresh_current_evaluation = self.evaluate_models(
            state.behavior,
            state.joint_world,
            state.control,
            state.chi,
            state.grounded_world,
        )
        decision_cache_check_vector = self._current_decision_cache_check_vector(
            state,
            fresh_current_evaluation,
        )
        complete_decision_cache_valid = jnp.all(decision_cache_check_vector)
        behavior_gradient = self._behavior.input_loss_gradient(
            state.behavior,
            state.chi,
            partner_action,
        )
        behavior_prediction_matches = jnp.array_equal(
            behavior_gradient.probabilities,
            current_evaluation.predicted_partner_probabilities,
        )
        world_prediction = self._joint_world.predict_joint(
            state.joint_world,
            focal_action,
            partner_action,
        )
        behavior_gradient_valid = (
            jnp.all(jnp.isfinite(behavior_gradient.gradient))
            & jnp.all(jnp.isfinite(behavior_gradient.probabilities))
            & jnp.isfinite(behavior_gradient.loss)
        )
        if self._grounded_world is None:
            if state.grounded_world is not None:
                raise ValueError("disabled grounded lane must not carry grounded state")
            grounded_world_update = None
            gradient_mix = None
            mixed_gradient_chi = behavior_gradient.gradient
            grounded_world_counter_saturated = jnp.asarray(False, dtype=jnp.bool_)
            grounded_world_lifetime_counter_valid = jnp.asarray(True, dtype=jnp.bool_)
            grounded_world_lifetime_capacity_available = jnp.asarray(True, dtype=jnp.bool_)
            grounded_world_update_applied = jnp.asarray(False, dtype=jnp.bool_)
            grounded_world_pre_update_words = jnp.zeros((2,), dtype=jnp.uint32)
            grounded_world_post_update_words = jnp.zeros((2,), dtype=jnp.uint32)
            grounded_world_prediction_matches = jnp.asarray(True, dtype=jnp.bool_)
            grounded_path_valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            if state.grounded_world is None or self._gradient_mixer_config is None:
                raise ValueError("enabled grounded lane requires model and mixer state")
            cached_grounded_evaluation = cast(
                IntegratedGroundedPlannerEvaluation,
                current_evaluation.grounded_world,
            )
            fresh_grounded_evaluation = cast(
                IntegratedGroundedPlannerEvaluation,
                fresh_current_evaluation.grounded_world,
            )
            grounded_world_prediction_matches = jnp.array_equal(
                fresh_grounded_evaluation.grounded_raw_predictions,
                cached_grounded_evaluation.grounded_raw_predictions,
            ) & (
                fresh_grounded_evaluation.predictions_valid
                == cached_grounded_evaluation.predictions_valid
            )
            grounded_world_counter_saturated = state.grounded_world.update_count == jnp.asarray(
                _INT32_MAX, dtype=jnp.int32
            )
            grounded_world_update = self._grounded_world.update(
                state.grounded_world,
                state.chi,
                focal_action,
                partner_action,
                next_raw,
                reward,
                discount,
            )
            grounded_world_lifetime_counter_valid = (
                grounded_world_update.diagnostics.lifetime_counter_valid
            )
            grounded_world_lifetime_capacity_available = (
                grounded_world_update.diagnostics.capacity_available
            )
            grounded_world_update_applied = grounded_world_update.update_applied
            grounded_world_pre_update_words = grounded_world_update.pre_update_words
            grounded_world_post_update_words = grounded_world_update.post_update_words
            executed_joint_index = N_ACTIONS * focal_action + partner_action
            executed_prediction = grounded_world_update.prediction
            grounded_world_prediction_matches = (
                grounded_world_prediction_matches
                & jnp.array_equal(
                    executed_prediction.joint_action_index,
                    executed_joint_index,
                )
                & jnp.array_equal(
                    executed_prediction.raw_predictions,
                    cached_grounded_evaluation.grounded_raw_predictions[
                        executed_joint_index
                    ],
                )
                & jnp.array_equal(
                    executed_prediction.raw_predictions,
                    executed_prediction.feature_contribution + executed_prediction.row_bias,
                )
            )
            gradient_mix = mix_representation_gradients(
                self._gradient_mixer_config,
                behavior_gradient.gradient,
                grounded_world_update.representation_gradient,
                behavior_valid=behavior_gradient_valid,
                grounded_world_valid=grounded_world_update.gradient_valid,
            )
            mixed_gradient_chi = gradient_mix.gradient
            grounded_path_valid = (
                behavior_gradient_valid
                & grounded_world_update.diagnostics.applied
                & gradient_mix.valid
            )
        decision_cache_valid = (
            behavior_prediction_matches
            & grounded_world_prediction_matches
            & complete_decision_cache_valid
        )
        transition_valid = transition_valid & grounded_path_valid & decision_cache_valid
        current_pair_read_mask = self._effective_pair_read_mask(
            state.interaction,
            state.consumer_active_mask,
        )
        if grounded_world_update is None:
            behavior_gradient_phi = self.chain_chi_gradient_to_phi(
                state.phi,
                state.router.descriptors,
                behavior_gradient.gradient,
                current_pair_read_mask,
            )
            representation_gradient = behavior_gradient_phi
        else:
            chained_source_gradients = self._chain_behavior_and_mixed_gradients_to_phi(
                state.phi,
                state.router.descriptors,
                behavior_gradient.gradient,
                mixed_gradient_chi,
                current_pair_read_mask,
            )
            behavior_gradient_phi = chained_source_gradients[0]
            representation_gradient = chained_source_gradients[1]

        learned_builder, state_learning = self._state_builder.learn(
            state.state_builder,
            representation_gradient,
        )
        deployed_builder = (
            learned_builder if self._config.state_learning_enabled else state.state_builder
        )
        behavior_update = self._behavior.update(
            state.behavior,
            state.chi,
            partner_action,
        )
        partner_sign_target = 2.0 * partner_action.astype(jnp.float32) - 1.0
        curation_priority_override = self._interaction_curation_input(
            state.interaction
        )
        random_active_priorities = curation_priority_override.active_ranks
        random_candidate_priorities = curation_priority_override.candidate_ranks
        interaction_update = self._interaction.update(
            state.interaction,
            self._deployed_phi(state.phi),
            jnp.reshape(partner_sign_target, (1,)),
            external_read_mask=state.consumer_active_mask,
            curation_priority_override=curation_priority_override,
        )
        interaction_proposal_applied = interaction_update.update_applied
        committed_interaction = (
            interaction_update.state
            if self._config.feature_lifecycle_enabled
            else interaction_update.pre_curation_state
        )
        (
            consumer_evidence_streak_updated_pre,
            consumer_read_acquire_pre,
            consumer_write_gate_pre,
        ) = self._update_consumer_evidence_streak(
            state.consumer_evidence_streak,
            interaction_update.evidence_refreshed,
        )
        consumer_read_idle_steps_updated_pre = self._update_consumer_read_idle_steps(
            state.consumer_read_idle_steps,
            interaction_update.evidence_refreshed,
            (state.router.descriptors[:, 0] >= 0) & (state.router.descriptors[:, 1] >= 0),
        )
        committed_behavior = self._commit_behavior_consumer_update(
            state.behavior,
            behavior_update.state,
            consumer_write_gate_pre,
        )
        committed_grounded_world = (
            None
            if grounded_world_update is None
            else self._commit_grounded_consumer_update(
                cast(GroundedJointWorldModelState, state.grounded_world),
                (
                    grounded_world_update.state
                    if self._config.grounded_world_learning_enabled
                    else cast(GroundedJointWorldModelState, state.grounded_world)
                ),
                consumer_write_gate_pre,
            )
        )

        builder_transition = self._state_builder.update_with_status(
            deployed_builder,
            next_raw,
            focal_action,
            reward,
            discount,
        )
        advanced_builder = builder_transition.state
        next_phi = builder_transition.representation
        world_update = self._joint_world.update(
            state.joint_world,
            focal_action,
            partner_action,
            reward,
            jnp.reshape(outcome, (WORLD_OUTCOME_DIM,)),
        )

        shadow_descriptors = jnp.stack(
            (
                interaction_update.state.feature_left,
                interaction_update.state.feature_right,
            ),
            axis=1,
        ).astype(jnp.int32)
        applied_descriptors = jnp.stack(
            (
                committed_interaction.feature_left,
                committed_interaction.feature_right,
            ),
            axis=1,
        ).astype(jnp.int32)
        # Compatibility name: these are the descriptors proposed to the
        # router after the lifecycle commit gate, hence the applied bank.
        proposed_descriptors = applied_descriptors
        if committed_grounded_world is None:
            (
                routed_behavior,
                routed_control,
                route_diagnostics,
                router_state,
            ) = self._route_feature_consumers(
                state.router,
                committed_behavior,
                state.control,
                proposed_descriptors,
            )
            routed_grounded_world = None
        else:
            (
                routed_behavior,
                routed_control,
                routed_grounded_world,
                route_diagnostics,
                router_state,
            ) = self._route_feature_consumers_with_grounded(
                state.router,
                committed_behavior,
                state.control,
                committed_grounded_world,
                proposed_descriptors,
            )
        consumer_route_audit = self._audit_consumer_identity_route(
            old_descriptors=state.router.descriptors,
            new_descriptors=proposed_descriptors,
            route=route_diagnostics,
            behavior_before=committed_behavior.weights,
            behavior_after=routed_behavior.weights,
            q_before=state.control.q_weights,
            q_after=routed_control.q_weights,
            trace_before=state.control.q_trace_weights,
            trace_after=routed_control.q_trace_weights,
            last_observation_before=state.control.last_observation,
            last_observation_after=routed_control.last_observation,
            grounded_before=(
                None
                if committed_grounded_world is None
                else committed_grounded_world.weights
            ),
            grounded_after=(
                None if routed_grounded_world is None else routed_grounded_world.weights
            ),
            retired_slot=interaction_update.retired_slot,
            replaced_slot=interaction_update.replaced_slot,
        )
        consumer_evidence_streak_post = self._route_consumer_evidence_streak(
            consumer_evidence_streak_updated_pre,
            route_diagnostics,
        )
        consumer_read_idle_steps_post = self._route_consumer_read_idle_steps(
            consumer_read_idle_steps_updated_pre,
            route_diagnostics,
        )
        consumer_write_gate_post = self._route_consumer_confirmed_write(
            consumer_write_gate_pre,
            route_diagnostics,
        )
        consumer_read_acquire_post = self._route_consumer_read_acquire(
            consumer_read_acquire_pre,
            route_diagnostics,
        )
        consumer_active_mask_post = self._route_consumer_active_mask(
            state.consumer_active_mask,
            consumer_read_acquire_pre,
            route=route_diagnostics,
            evidence_idle_steps_post=consumer_read_idle_steps_post,
        )

        next_chi = self.build_chi(
            next_phi,
            router_state.descriptors,
            self._effective_pair_read_mask(
                committed_interaction,
                consumer_active_mask_post,
            ),
        )
        next_evaluation = self.evaluate_models(
            routed_behavior,
            world_update.state,
            routed_control,
            next_chi,
            routed_grounded_world,
        )
        next_selection = self.select_planner_action(
            routed_control,
            next_evaluation.planner_scores,
        )
        if forced_next_action is not None:
            next_selection = next_selection.replace(
                action=forced_next_action,
                externally_forced=jnp.asarray(True, dtype=jnp.bool_),
            )
        routed_control_with_rng = routed_control.replace(rng_key=next_selection.rng_key_after)
        control_update = self._control.update(
            routed_control_with_rng,
            reward,
            next_chi,
            next_action=next_selection.action,
            discount=discount,
        )
        committed_control = self._commit_control_consumer_update(
            routed_control_with_rng,
            control_update.state,
            consumer_write_gate_post,
        )
        next_q_value_delta = self._control.q_values(
            committed_control,
            next_chi,
        ) - next_evaluation.q_values
        grounded_evaluations_valid = (
            jnp.asarray(True, dtype=jnp.bool_)
            if current_evaluation.grounded_world is None
            else (
                current_evaluation.grounded_world.predictions_valid
                & cast(
                    IntegratedGroundedPlannerEvaluation,
                    next_evaluation.grounded_world,
                ).predictions_valid
            )
        )
        transaction_capacity_available = (
            outer_lifetime_capacity_available
            & behavior_update.lifetime_capacity_available
            & world_update.lifetime_capacity_available
            & grounded_world_lifetime_capacity_available
            & control_update.lifetime_capacity_available
            & state_learning.lifetime_capacity_available
            & builder_transition.step_capacity_available
            & interaction_update.lifetime_capacity_available
            & route_diagnostics.lifetime_capacity_available
        )
        candidate_models_valid = (
            current_evaluation.partner_probabilities_valid
            & next_evaluation.partner_probabilities_valid
            & behavior_update.update_applied
            & world_update.update_applied
            & world_update.target_valid
            & grounded_path_valid
            & grounded_evaluations_valid
            & control_update.update_applied
            & state_learning.valid
            & state_learning.update_applied
            & builder_transition.candidate_state_valid
            & builder_transition.candidate_representation_valid
            & builder_transition.transition_applied
            & interaction_update.state_valid
            & interaction_update.candidate_state_valid
            & interaction_update.update_applied
            & route_diagnostics.route_applied
        )
        transition_valid = (
            transition_valid
            & candidate_models_valid
            & transaction_capacity_available
            & consumer_route_audit.values_exact
            & consumer_route_audit.lifecycle_destination_reset_exact
        )

        proposed_state = IntegratedHiddenPartnerState(
            state_builder=advanced_builder,
            interaction=committed_interaction,
            behavior=routed_behavior,
            joint_world=world_update.state,
            grounded_world=routed_grounded_world,
            control=committed_control,
            router=router_state,
            raw_observation=next_raw,
            phi=next_phi,
            chi=next_chi,
            consumer_active_mask=consumer_active_mask_post,
            consumer_evidence_streak=consumer_evidence_streak_post,
            consumer_read_idle_steps=consumer_read_idle_steps_post,
            current_evaluation=next_evaluation,
            current_q_value_delta=next_q_value_delta,
            current_selection=next_selection,
            step_count=_saturating_int32_increment(state.step_count),
            step_words=outer_proposed_post_step_words,
        )
        proposed_post_child_clock_alignment_vector = (
            self._child_clock_alignment_vector(proposed_state)
        )
        proposed_post_child_clocks_aligned = jnp.all(
            proposed_post_child_clock_alignment_vector
        )
        candidate_state_finite = self._update_finite(
            proposed_state,
            mixed_gradient_chi,
            representation_gradient,
            behavior_update.loss,
            interaction_update.metrics,
            interaction_update.candidate_promotion_signal,
            world_update.reward_error,
            world_update.outcome_error,
            control_update.td_error,
        )
        transition_applied = (
            transition_valid
            & candidate_state_finite
            & proposed_post_child_clocks_aligned
        )
        next_state = jax.lax.cond(
            transition_applied,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        committed_post_child_clock_alignment_vector = (
            self._child_clock_alignment_vector(next_state)
        )
        committed_post_child_clocks_aligned = jnp.all(
            committed_post_child_clock_alignment_vector
        )
        transition_valid = transition_applied
        model_valid = candidate_models_valid
        grounded_world_step_delta = (
            jnp.asarray(0, dtype=jnp.int32)
            if state.grounded_world is None
            else cast(
                GroundedJointWorldModelState,
                next_state.grounded_world,
            ).update_count
            - state.grounded_world.update_count
        )
        false_active = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
        false_candidates = jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_)
        rejected_index = jnp.asarray(-1, dtype=jnp.int32)
        old_descriptor_validation = self._router.validate_descriptors(state.router.descriptors)
        old_live_count = jnp.sum(
            old_descriptor_validation.live_mask,
            dtype=jnp.int32,
        )
        lifecycle_enabled = jnp.asarray(
            self._config.feature_lifecycle_enabled,
            dtype=jnp.bool_,
        )
        lifecycle_proposed = (
            (interaction_update.replaced_slot >= 0)
            | (interaction_update.refreshed_candidate >= 0)
            | (interaction_update.retired_slot >= 0)
        )
        lifecycle_apply_gate = transition_valid & lifecycle_enabled
        applied_live_count = jnp.sum(
            (applied_descriptors[:, 0] >= 0) & (applied_descriptors[:, 1] >= 0),
            dtype=jnp.int32,
        )
        rejected_route = dataclasses.replace(
            route_diagnostics,
            route_applied=jnp.asarray(False, dtype=jnp.bool_),
            descriptors_changed=jnp.asarray(False, dtype=jnp.bool_),
            old_validation=old_descriptor_validation,
            new_validation=old_descriptor_validation,
            source_slots=jnp.arange(ACTIVE_PAIR_SLOTS, dtype=jnp.int32),
            survivor_mask=old_descriptor_validation.live_mask,
            new_mask=false_active,
            evicted_mask=false_active,
            survivor_count=old_live_count,
            new_count=jnp.asarray(0, dtype=jnp.int32),
            evicted_count=jnp.asarray(0, dtype=jnp.int32),
            old_live_count=old_live_count,
            new_live_count=old_live_count,
            route_count_after=state.router.route_count,
            generation_count_after=state.router.generation_count,
            route_words_after=state.router.route_words,
            generation_words_after=state.router.generation_words,
        )
        committed_route_diagnostics = jax.lax.cond(
            transition_valid,
            lambda _: route_diagnostics,
            lambda _: rejected_route,
            operand=None,
        )
        diagnostics = IntegratedUpdateDiagnostics(
            current_evaluation=current_evaluation,
            next_evaluation=next_evaluation,
            next_selection=next_selection,
            behavior_probabilities_preupdate=(behavior_update.probabilities),
            behavior_prediction_matches_decision=(behavior_prediction_matches),
            behavior_loss_preupdate=behavior_update.loss,
            behavior_correct_preupdate=behavior_update.correct,
            behavior_gradient_chi=behavior_gradient.gradient,
            behavior_gradient_phi=behavior_gradient_phi,
            grounded_world_update=grounded_world_update,
            grounded_world_learning_enabled=jnp.asarray(
                self._config.grounded_world_learning_enabled,
                dtype=jnp.bool_,
            ),
            grounded_world_counter_saturated=grounded_world_counter_saturated,
            grounded_world_lifetime_counter_valid=(
                grounded_world_lifetime_counter_valid
            ),
            grounded_world_lifetime_capacity_available=(
                grounded_world_lifetime_capacity_available
            ),
            grounded_world_update_applied=(
                transition_valid & grounded_world_update_applied
            ),
            grounded_world_pre_update_words=grounded_world_pre_update_words,
            grounded_world_proposed_post_update_words=(
                grounded_world_post_update_words
            ),
            grounded_world_post_update_words=jnp.where(
                transition_valid,
                grounded_world_post_update_words,
                grounded_world_pre_update_words,
            ),
            grounded_world_prediction_matches_decision=(grounded_world_prediction_matches),
            gradient_mix=gradient_mix,
            mixed_gradient_chi=(None if grounded_world_update is None else mixed_gradient_chi),
            mixed_gradient_phi=(None if grounded_world_update is None else representation_gradient),
            state_learning=state_learning,
            state_builder_transition_state_valid=builder_transition.state_valid,
            state_builder_transition_input_valid=builder_transition.input_valid,
            state_builder_step_counter_valid=builder_transition.step_counter_valid,
            state_builder_step_capacity_available=(
                builder_transition.step_capacity_available
            ),
            state_builder_candidate_state_valid=(
                builder_transition.candidate_state_valid
            ),
            state_builder_candidate_representation_valid=(
                builder_transition.candidate_representation_valid
            ),
            state_builder_transition_applied=(
                transition_valid & builder_transition.transition_applied
            ),
            state_builder_pre_step_words=builder_transition.pre_step_words,
            state_builder_proposed_post_step_words=(
                builder_transition.post_step_words
            ),
            state_builder_post_step_words=jnp.where(
                transition_valid,
                builder_transition.post_step_words,
                builder_transition.pre_step_words,
            ),
            interaction_prediction_preupdate=(interaction_update.predictions),
            interaction_error_preupdate=interaction_update.errors,
            interaction_metrics=interaction_update.metrics,
            interaction_lifetime_counter_valid=(
                interaction_update.lifetime_counter_valid
            ),
            interaction_lifetime_capacity_available=(
                interaction_update.lifetime_capacity_available
            ),
            interaction_state_valid=interaction_update.state_valid,
            interaction_candidate_state_valid=(
                interaction_update.candidate_state_valid
            ),
            interaction_proposal_applied=interaction_proposal_applied,
            interaction_update_applied=(
                transition_valid & interaction_update.update_applied
            ),
            interaction_pre_step_words=interaction_update.pre_step_words,
            interaction_proposed_post_step_words=(
                interaction_update.post_step_words
            ),
            interaction_post_step_words=jnp.where(
                transition_valid,
                interaction_update.post_step_words,
                interaction_update.pre_step_words,
            ),
            interaction_replaced_slot=jnp.where(
                interaction_proposal_applied,
                interaction_update.replaced_slot,
                rejected_index,
            ),
            interaction_promoted_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.promoted_candidate,
                rejected_index,
            ),
            interaction_refreshed_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.refreshed_candidate,
                rejected_index,
            ),
            interaction_retired_slot=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_slot,
                rejected_index,
            ),
            interaction_retired_left=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_left,
                rejected_index,
            ),
            interaction_retired_right=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_right,
                rejected_index,
            ),
            interaction_proposal_replaced_slot=jnp.where(
                interaction_proposal_applied,
                interaction_update.replaced_slot,
                rejected_index,
            ),
            interaction_proposal_promoted_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.promoted_candidate,
                rejected_index,
            ),
            interaction_proposal_refreshed_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.refreshed_candidate,
                rejected_index,
            ),
            interaction_proposal_retired_slot=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_slot,
                rejected_index,
            ),
            interaction_proposal_retired_left=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_left,
                rejected_index,
            ),
            interaction_proposal_retired_right=jnp.where(
                interaction_proposal_applied,
                interaction_update.retired_right,
                rejected_index,
            ),
            interaction_lifecycle_proposed=(
                interaction_proposal_applied & lifecycle_proposed
            ),
            interaction_lifecycle_applied=(
                lifecycle_apply_gate & lifecycle_proposed
            ),
            interaction_applied_replaced_slot=jnp.where(
                lifecycle_apply_gate,
                interaction_update.replaced_slot,
                rejected_index,
            ),
            interaction_applied_promoted_candidate=jnp.where(
                lifecycle_apply_gate,
                interaction_update.promoted_candidate,
                rejected_index,
            ),
            interaction_applied_refreshed_candidate=jnp.where(
                lifecycle_apply_gate,
                interaction_update.refreshed_candidate,
                rejected_index,
            ),
            interaction_applied_retired_slot=jnp.where(
                lifecycle_apply_gate,
                interaction_update.retired_slot,
                rejected_index,
            ),
            interaction_applied_retired_left=jnp.where(
                lifecycle_apply_gate,
                interaction_update.retired_left,
                rejected_index,
            ),
            interaction_applied_retired_right=jnp.where(
                lifecycle_apply_gate,
                interaction_update.retired_right,
                rejected_index,
            ),
            interaction_evidence_refreshed=jnp.where(
                transition_valid,
                interaction_update.evidence_refreshed,
                false_active,
            ),
            interaction_retention_evidence_refreshed=(
                jnp.where(
                    transition_valid,
                    interaction_update.retention_evidence_refreshed,
                    false_active,
                )
            ),
            interaction_relevance_probe_scores=(
                jnp.where(
                    transition_valid,
                    interaction_update.relevance_probe_scores,
                    jnp.zeros_like(interaction_update.relevance_probe_scores),
                )
            ),
            interaction_relevance_probe_errors=(
                jnp.where(
                    transition_valid,
                    interaction_update.relevance_probe_errors,
                    jnp.zeros_like(interaction_update.relevance_probe_errors),
                )
            ),
            interaction_durable_read_mask=interaction_update.durable_read_mask,
            interaction_relevance_probe_weights_pre=(state.interaction.relevance_probe_weights),
            interaction_relevance_probe_weights_post=(
                next_state.interaction.relevance_probe_weights
            ),
            interaction_relevance_probe_biases_pre=(state.interaction.relevance_probe_biases),
            interaction_relevance_probe_biases_post=(next_state.interaction.relevance_probe_biases),
            interaction_candidate_promotion_signal=(
                jnp.where(
                    transition_valid,
                    interaction_update.candidate_promotion_signal,
                    jnp.zeros_like(interaction_update.candidate_promotion_signal),
                )
            ),
            interaction_candidate_promotion_raw_evidence=(
                jnp.where(
                    transition_valid,
                    interaction_update.candidate_promotion_raw_evidence,
                    false_candidates,
                )
            ),
            interaction_candidate_promotion_evidence_streak_pre=(
                interaction_update.candidate_promotion_evidence_streak_pre
            ),
            interaction_candidate_promotion_evidence_streak_updated=(
                jnp.where(
                    interaction_proposal_applied,
                    interaction_update.candidate_promotion_evidence_streak_updated,
                    state.interaction.candidate_promotion_evidence_streak,
                )
            ),
            interaction_candidate_promotion_evidence_streak_proposal_post=(
                jnp.where(
                    interaction_proposal_applied,
                    interaction_update.state.candidate_promotion_evidence_streak,
                    state.interaction.candidate_promotion_evidence_streak,
                )
            ),
            interaction_candidate_promotion_evidence_streak_post=(
                next_state.interaction.candidate_promotion_evidence_streak
            ),
            interaction_candidate_promotion_confirmed=(
                jnp.where(
                    transition_valid,
                    interaction_update.candidate_promotion_confirmed,
                    false_candidates,
                )
            ),
            interaction_candidate_reacquisition_required_pre=(
                interaction_update.candidate_reacquisition_required_pre
            ),
            interaction_candidate_reacquisition_required_proposal_post=(
                jnp.where(
                    interaction_proposal_applied,
                    interaction_update.state.candidate_reacquisition_required,
                    state.interaction.candidate_reacquisition_required,
                )
            ),
            interaction_candidate_reacquisition_required_post=(
                next_state.interaction.candidate_reacquisition_required
            ),
            interaction_candidate_reacquisition_confirmed=(
                jnp.where(
                    transition_valid,
                    interaction_update.candidate_reacquisition_confirmed,
                    false_candidates,
                )
            ),
            consumer_evidence_streak_pre=state.consumer_evidence_streak,
            consumer_evidence_streak_updated_pre=(
                jnp.where(
                    transition_valid,
                    consumer_evidence_streak_updated_pre,
                    state.consumer_evidence_streak,
                )
            ),
            consumer_evidence_streak_post=next_state.consumer_evidence_streak,
            consumer_read_idle_steps_pre=state.consumer_read_idle_steps,
            consumer_read_idle_steps_updated_pre=(
                jnp.where(
                    transition_valid,
                    consumer_read_idle_steps_updated_pre,
                    state.consumer_read_idle_steps,
                )
            ),
            consumer_read_idle_steps_post=next_state.consumer_read_idle_steps,
            consumer_read_acquire_pre=jnp.where(
                transition_valid,
                consumer_read_acquire_pre,
                false_active,
            ),
            consumer_read_acquire_post=jnp.where(
                transition_valid,
                consumer_read_acquire_post,
                false_active,
            ),
            consumer_confirmed_write_pre=jnp.where(
                transition_valid,
                consumer_write_gate_pre,
                false_active,
            ),
            consumer_confirmed_write_post=jnp.where(
                transition_valid,
                consumer_write_gate_post,
                false_active,
            ),
            consumer_write_gate_pre=jnp.where(
                transition_valid,
                consumer_write_gate_pre,
                false_active,
            ),
            consumer_read_mask_pre=state.consumer_active_mask,
            consumer_read_mask_post=next_state.consumer_active_mask,
            consumer_active_mask_pre=state.consumer_active_mask,
            consumer_active_mask_post=next_state.consumer_active_mask,
            interaction_matching_candidate_reset_mask=(
                jnp.where(
                    interaction_proposal_applied,
                    interaction_update.matching_candidate_reset_mask,
                    false_candidates,
                )
            ),
            interaction_matching_candidate_reset_count=jnp.where(
                interaction_proposal_applied,
                jnp.sum(
                    interaction_update.matching_candidate_reset_mask,
                    dtype=jnp.int32,
                ),
                0,
            ),
            interaction_applied_matching_candidate_reset_mask=jnp.where(
                lifecycle_apply_gate,
                interaction_update.matching_candidate_reset_mask,
                false_candidates,
            ),
            interaction_applied_matching_candidate_reset_count=jnp.where(
                lifecycle_apply_gate,
                jnp.sum(
                    interaction_update.matching_candidate_reset_mask,
                    dtype=jnp.int32,
                ),
                0,
            ),
            interaction_live_feature_count=jnp.where(
                interaction_proposal_applied,
                interaction_update.live_feature_count,
                old_live_count,
            ),
            interaction_vacancy_count=jnp.where(
                interaction_proposal_applied,
                interaction_update.vacancy_count,
                ACTIVE_PAIR_SLOTS - old_live_count,
            ),
            interaction_promoted_into_vacancy=(
                interaction_proposal_applied
                & interaction_update.promoted_into_vacancy
            ),
            interaction_proposal_live_feature_count=jnp.where(
                interaction_proposal_applied,
                interaction_update.live_feature_count,
                old_live_count,
            ),
            interaction_proposal_vacancy_count=jnp.where(
                interaction_proposal_applied,
                interaction_update.vacancy_count,
                ACTIVE_PAIR_SLOTS - old_live_count,
            ),
            interaction_proposal_promoted_into_vacancy=(
                interaction_proposal_applied
                & interaction_update.promoted_into_vacancy
            ),
            interaction_applied_live_feature_count=jnp.where(
                transition_valid,
                applied_live_count,
                old_live_count,
            ),
            interaction_applied_vacancy_count=jnp.where(
                transition_valid,
                ACTIVE_PAIR_SLOTS - applied_live_count,
                ACTIVE_PAIR_SLOTS - old_live_count,
            ),
            interaction_applied_promoted_into_vacancy=(
                lifecycle_apply_gate & interaction_update.promoted_into_vacancy
            ),
            random_curation_enabled=curation_priority_override.enabled,
            random_curation_attempted=(
                interaction_proposal_applied
                & interaction_update.curation_attempted
            ),
            random_curation_applied=(
                transition_valid
                & interaction_update.curation_priority_override_applied
            ),
            random_active_priorities=jnp.where(
                interaction_proposal_applied,
                random_active_priorities,
                jnp.zeros_like(random_active_priorities),
            ),
            random_candidate_priorities=jnp.where(
                interaction_proposal_applied,
                random_candidate_priorities,
                jnp.zeros_like(random_candidate_priorities),
            ),
            curation_selected_active_worst_slot=jnp.where(
                interaction_proposal_applied,
                interaction_update.curation_selected_active_worst_slot,
                rejected_index,
            ),
            curation_selected_promotion_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.curation_selected_promotion_candidate,
                rejected_index,
            ),
            curation_selected_refresh_candidate=jnp.where(
                interaction_proposal_applied,
                interaction_update.curation_selected_refresh_candidate,
                rejected_index,
            ),
            shadow_descriptors=jnp.where(
                interaction_proposal_applied,
                shadow_descriptors,
                state.router.descriptors,
            ),
            proposed_descriptors=jnp.where(
                interaction_proposal_applied,
                proposed_descriptors,
                state.router.descriptors,
            ),
            interaction_proposal_descriptors=jnp.where(
                interaction_proposal_applied,
                shadow_descriptors,
                state.router.descriptors,
            ),
            interaction_applied_descriptors=jnp.where(
                transition_valid,
                applied_descriptors,
                state.router.descriptors,
            ),
            shadow_descriptors_changed=(
                interaction_proposal_applied
                & jnp.any(shadow_descriptors != state.router.descriptors)
            ),
            route=committed_route_diagnostics,
            router_proposed_post_route_words=(
                route_diagnostics.route_words_after
            ),
            router_committed_post_route_words=next_state.router.route_words,
            router_proposed_post_generation_words=(
                route_diagnostics.generation_words_after
            ),
            router_committed_post_generation_words=(
                next_state.router.generation_words
            ),
            consumer_route_source_slots_exact=(
                consumer_route_audit.source_slots_exact
            ),
            consumer_route_identity_masks_exact=(
                consumer_route_audit.identity_masks_exact
            ),
            consumer_route_stable_prefix_exact=(
                consumer_route_audit.stable_prefix_exact
            ),
            consumer_route_survivor_values_exact=(
                consumer_route_audit.survivor_values_exact
            ),
            consumer_route_reset_values_exact=(
                consumer_route_audit.reset_values_exact
            ),
            consumer_route_no_carry_reset_exact=(
                consumer_route_audit.no_carry_reset_exact
            ),
            consumer_route_behavior_values_exact=(
                consumer_route_audit.behavior_values_exact
            ),
            consumer_route_q_values_exact=consumer_route_audit.q_values_exact,
            consumer_route_trace_values_exact=(
                consumer_route_audit.trace_values_exact
            ),
            consumer_route_last_observation_exact=(
                consumer_route_audit.last_observation_exact
            ),
            consumer_route_grounded_values_exact=(
                consumer_route_audit.grounded_values_exact
            ),
            consumer_route_values_exact=consumer_route_audit.values_exact,
            consumer_lifecycle_destination_reset_exact=(
                consumer_route_audit.lifecycle_destination_reset_exact
            ),
            world_reward_prediction_preupdate=world_prediction.reward,
            world_outcome_prediction_preupdate=world_prediction.outcome,
            world_reward_error=world_update.reward_error,
            world_outcome_error=world_update.outcome_error,
            world_target_valid=world_update.target_valid,
            world_lifetime_counter_valid=world_update.lifetime_counter_valid,
            world_lifetime_capacity_available=(
                world_update.lifetime_capacity_available
            ),
            world_update_applied=(transition_valid & world_update.update_applied),
            world_pre_step_words=world_update.pre_step_words,
            world_proposed_post_step_words=world_update.post_step_words,
            world_post_step_words=jnp.where(
                transition_valid,
                world_update.post_step_words,
                world_update.pre_step_words,
            ),
            td_error=control_update.td_error,
            average_reward=control_update.average_reward,
            control_lifetime_counter_valid=(
                control_update.lifetime_counter_valid
            ),
            control_lifetime_capacity_available=(
                control_update.lifetime_capacity_available
            ),
            control_update_applied=(transition_valid & control_update.update_applied),
            control_pre_step_words=control_update.pre_step_words,
            control_proposed_post_step_words=control_update.post_step_words,
            control_post_step_words=jnp.where(
                transition_valid,
                control_update.post_step_words,
                control_update.pre_step_words,
            ),
            transition_input_valid=transition_input_valid,
            decision_cache_valid=decision_cache_valid,
            decision_cache_check_vector=decision_cache_check_vector,
            behavior_gradient_valid=behavior_gradient_valid,
            behavior_lifetime_counter_valid=(
                behavior_update.lifetime_counter_valid
            ),
            behavior_lifetime_capacity_available=(
                behavior_update.lifetime_capacity_available
            ),
            behavior_update_applied=(transition_valid & behavior_update.update_applied),
            behavior_pre_step_words=behavior_update.pre_step_words,
            behavior_proposed_post_step_words=behavior_update.post_step_words,
            behavior_post_step_words=jnp.where(
                transition_valid,
                behavior_update.post_step_words,
                behavior_update.pre_step_words,
            ),
            grounded_path_valid=grounded_path_valid,
            candidate_models_valid=candidate_models_valid,
            candidate_state_finite=candidate_state_finite,
            outer_pre_step_words=state.step_words,
            outer_proposed_post_step_words=outer_proposed_post_step_words,
            outer_committed_post_step_words=next_state.step_words,
            outer_lifetime_counter_valid=outer_lifetime_counter_valid,
            outer_lifetime_capacity_available=(
                outer_lifetime_capacity_available
            ),
            pre_child_clock_alignment_vector=(
                pre_child_clock_alignment_vector
            ),
            pre_child_clocks_aligned=pre_child_clocks_aligned,
            proposed_post_child_clock_alignment_vector=(
                proposed_post_child_clock_alignment_vector
            ),
            proposed_post_child_clocks_aligned=(
                proposed_post_child_clocks_aligned
            ),
            committed_post_child_clock_alignment_vector=(
                committed_post_child_clock_alignment_vector
            ),
            committed_post_child_clocks_aligned=(
                committed_post_child_clocks_aligned
            ),
            transaction_capacity_available=transaction_capacity_available,
            transition_observation_matches=observation_matches,
            transition_action_matches=action_matches,
            transition_semantics_valid=transition_valid,
            transition_applied=transition_applied,
            transition_rejected=~transition_applied,
            model_valid=transition_valid & model_valid,
            state_builder_step_delta=(
                next_state.state_builder.step_count - state.state_builder.step_count
            ),
            state_builder_learning_delta=(
                next_state.state_builder.update_count - state.state_builder.update_count
            ),
            behavior_step_delta=(next_state.behavior.step_count - state.behavior.step_count),
            interaction_step_delta=(
                next_state.interaction.step_count - state.interaction.step_count
            ),
            world_step_delta=(next_state.joint_world.step_count - state.joint_world.step_count),
            grounded_world_step_delta=grounded_world_step_delta,
            control_step_delta=(next_state.control.step_count - state.control.step_count),
            router_route_delta=(next_state.router.route_count - state.router.route_count),
            router_generation_delta=(
                next_state.router.generation_count - state.router.generation_count
            ),
            integrated_step_delta=(next_state.step_count - state.step_count),
            all_finite=(
                transition_valid
                & self._update_finite(
                    next_state,
                    mixed_gradient_chi,
                    representation_gradient,
                    behavior_update.loss,
                    interaction_update.metrics,
                    interaction_update.candidate_promotion_signal,
                    world_update.reward_error,
                    world_update.outcome_error,
                    control_update.td_error,
                )
            ),
        )
        return IntegratedUpdateResult(
            state=next_state,
            action=jnp.where(
                transition_valid,
                next_selection.action,
                state.control.last_action,
            ),
            diagnostics=diagnostics,
        )

    def _deployed_phi(self, phi: Array) -> Array:
        base = jnp.asarray(phi, dtype=jnp.float32)
        if self._config.memory_masked:
            return base * self._memory_mask
        return base

    def _effective_pair_read_mask(
        self,
        interaction: InteractionFeatureState,
        consumer_read_mask: Array,
    ) -> Array:
        """Gate downstream products by lease and durable commitment."""
        consumer = jnp.asarray(consumer_read_mask, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.independent_relevance_probe:
            return consumer
        return consumer & interaction.active_output_memory_committed

    def _deployment_derivative_mask(self) -> Array:
        if self._config.memory_masked:
            return self._memory_mask
        return jnp.ones((BASE_FEATURE_DIM,), dtype=jnp.float32)

    def _interaction_curation_input(
        self,
        state: InteractionFeatureState,
    ) -> InteractionCurationPriorityOverride:
        """Derive transient fixed-shape ranks without advancing learner RNG."""

        active_key = jr.fold_in(state.key, jnp.uint32(0x43555241))
        candidate_key = jr.fold_in(state.key, jnp.uint32(0x43555243))
        return InteractionCurationPriorityOverride(
            enabled=jnp.asarray(
                self._config.random_feature_curation,
                dtype=jnp.bool_,
            ),
            active_ranks=jr.permutation(
                active_key,
                ACTIVE_PAIR_SLOTS,
            ).astype(jnp.float32),
            candidate_ranks=jr.permutation(
                candidate_key,
                CANDIDATE_PAIR_SLOTS,
            ).astype(jnp.float32),
        )

    def _update_consumer_evidence_streak(
        self,
        previous_streak: Array,
        evidence_refreshed: Array,
    ) -> tuple[Array, Array, Array]:
        """Return updated streak, read acquisition, and old-bank write confirmation."""
        previous = jnp.asarray(previous_streak, dtype=jnp.int32).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return (
                jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
                jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
                jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
            )
        evidence = jnp.asarray(evidence_refreshed, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        cap = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        incremented = jnp.minimum(jnp.maximum(previous, 0), cap - 1) + 1
        updated = jnp.where(evidence, incremented, 0)
        read_acquire = evidence & (
            updated
            >= jnp.asarray(
                self._config.consumer_read_confirmation_steps,
                dtype=jnp.int32,
            )
        )
        confirmed_write = evidence & (
            updated
            >= jnp.asarray(
                self._config.consumer_evidence_confirmation_steps,
                dtype=jnp.int32,
            )
        )
        return updated, read_acquire, confirmed_write

    def _consumer_write_gate(
        self,
        evidence_refreshed: Array,
        previous_streak: Array | None = None,
    ) -> Array:
        """Return the confirmed old-bank write gate (compatibility helper)."""
        previous = (
            jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
            if previous_streak is None
            else previous_streak
        )
        return self._update_consumer_evidence_streak(
            previous,
            evidence_refreshed,
        )[2]

    def _update_consumer_read_idle_steps(
        self,
        previous_idle_steps: Array,
        evidence_refreshed: Array,
        live_mask: Array,
    ) -> Array:
        """Advance the read lease from raw evidence independently of feature retention."""
        previous = jnp.asarray(previous_idle_steps, dtype=jnp.int32).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        evidence = jnp.asarray(evidence_refreshed, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        live = jnp.asarray(live_mask, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        cap = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        incremented = jnp.minimum(jnp.maximum(previous, 0), cap - 1) + 1
        return jnp.where(live, jnp.where(evidence, 0, incremented), 0)

    def _commit_behavior_consumer_update(
        self,
        previous: BehaviorModelState,
        proposed: BehaviorModelState,
        write_gate: Array,
    ) -> BehaviorModelState:
        """Commit base behavior learning and only evidence-backed tail writes."""
        if not self._config.evidence_gated_consumer_memory:
            return proposed
        gate = jnp.asarray(write_gate, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        committed_tail = jnp.where(
            gate[None, :],
            proposed.weights[:, BASE_FEATURE_DIM:],
            previous.weights[:, BASE_FEATURE_DIM:],
        )
        committed_weights = jnp.concatenate(
            (proposed.weights[:, :BASE_FEATURE_DIM], committed_tail),
            axis=1,
        )
        return cast(
            BehaviorModelState,
            proposed.replace(weights=committed_weights),
        )

    def _commit_grounded_consumer_update(
        self,
        previous: GroundedJointWorldModelState,
        proposed: GroundedJointWorldModelState,
        write_gate: Array,
    ) -> GroundedJointWorldModelState:
        """Commit grounded base columns and only evidence-backed tail writes."""
        if not self._config.evidence_gated_consumer_memory:
            return proposed
        gate = jnp.asarray(write_gate, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        committed_tail = jnp.where(
            gate[None, None, :],
            proposed.weights[..., BASE_FEATURE_DIM:],
            previous.weights[..., BASE_FEATURE_DIM:],
        )
        committed_weights = jnp.concatenate(
            (proposed.weights[..., :BASE_FEATURE_DIM], committed_tail),
            axis=-1,
        )
        return cast(
            GroundedJointWorldModelState,
            proposed.replace(weights=committed_weights),
        )

    @staticmethod
    def _route_consumer_slot_values(
        values: Array,
        route: FeatureBankRouteDiagnostics,
        *,
        new_value: int | bool,
    ) -> Array:
        """Route one old-bank slot vector by exact descriptor identity."""
        old = jnp.asarray(values).reshape((ACTIVE_PAIR_SLOTS,))
        safe_sources = jnp.clip(
            route.source_slots,
            jnp.int32(0),
            jnp.int32(ACTIVE_PAIR_SLOTS - 1),
        )
        routed = jnp.where(
            route.survivor_mask,
            old[safe_sources],
            jnp.asarray(new_value, dtype=old.dtype),
        )
        return jnp.where(route.valid, routed, old)

    def _route_consumer_evidence_streak(
        self,
        updated_streak: Array,
        route: FeatureBankRouteDiagnostics,
    ) -> Array:
        """Route updated consecutive-evidence state; new identities start at zero."""
        updated = jnp.asarray(updated_streak, dtype=jnp.int32).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        return self._route_consumer_slot_values(
            updated,
            route,
            new_value=0,
        )

    def _route_consumer_read_idle_steps(
        self,
        updated_idle_steps: Array,
        route: FeatureBankRouteDiagnostics,
    ) -> Array:
        """Route raw-evidence read-idle state; new descriptor identities start at zero."""
        updated = jnp.asarray(updated_idle_steps, dtype=jnp.int32).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        return self._route_consumer_slot_values(
            updated,
            route,
            new_value=0,
        )

    def _route_consumer_confirmed_write(
        self,
        confirmed_write: Array,
        route: FeatureBankRouteDiagnostics,
    ) -> Array:
        """Route the current write permission; new identities cannot inherit it."""
        confirmed = jnp.asarray(confirmed_write, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.where(
                route.valid,
                route.new_validation.live_mask,
                confirmed,
            )
        return self._route_consumer_slot_values(
            confirmed,
            route,
            new_value=False,
        )

    def _route_consumer_read_acquire(
        self,
        read_acquire: Array,
        route: FeatureBankRouteDiagnostics,
    ) -> Array:
        """Route current read acquisition; new identities cannot inherit it."""
        acquire = jnp.asarray(read_acquire, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.where(
                route.valid,
                route.new_validation.live_mask,
                acquire,
            )
        return self._route_consumer_slot_values(
            acquire,
            route,
            new_value=False,
        )

    def _route_consumer_active_mask(
        self,
        previous_mask: Array,
        read_acquire: Array,
        route: FeatureBankRouteDiagnostics,
        evidence_idle_steps_post: Array | None = None,
    ) -> Array:
        """Route the persistent read lease and close it only after idle expiry."""
        previous = jnp.asarray(previous_mask, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        if not self._config.evidence_gated_consumer_memory:
            return jnp.where(
                route.valid,
                route.new_validation.live_mask,
                previous,
            )
        acquire = jnp.asarray(read_acquire, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        acquired = previous | acquire
        routed = self._route_consumer_slot_values(
            acquired,
            route,
            new_value=False,
        )
        idle_steps = (
            jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
            if evidence_idle_steps_post is None
            else jnp.asarray(evidence_idle_steps_post, dtype=jnp.int32).reshape(
                (ACTIVE_PAIR_SLOTS,)
            )
        )
        lease_open = idle_steps <= jnp.asarray(
            self._config.consumer_read_lease_steps,
            dtype=jnp.int32,
        )
        live = jnp.where(
            route.valid,
            route.new_validation.live_mask,
            route.old_validation.live_mask,
        )
        return routed & live & lease_open

    def _commit_control_consumer_update(
        self,
        previous: DifferentialSARSAState,
        proposed: DifferentialSARSAState,
        write_gate: Array,
    ) -> DifferentialSARSAState:
        """Commit gated Q columns and erase closed eligibility traces."""
        if not self._config.evidence_gated_consumer_memory:
            return proposed
        gate = jnp.asarray(write_gate, dtype=jnp.bool_).reshape((ACTIVE_PAIR_SLOTS,))
        committed_q_tail = jnp.where(
            gate[None, :],
            proposed.q_weights[:, BASE_FEATURE_DIM:],
            previous.q_weights[:, BASE_FEATURE_DIM:],
        )
        committed_trace_tail = jnp.where(
            gate[None, :],
            proposed.q_trace_weights[:, BASE_FEATURE_DIM:],
            jnp.zeros_like(proposed.q_trace_weights[:, BASE_FEATURE_DIM:]),
        )
        return cast(
            DifferentialSARSAState,
            proposed.replace(
                q_weights=jnp.concatenate(
                    (proposed.q_weights[:, :BASE_FEATURE_DIM], committed_q_tail),
                    axis=1,
                ),
                q_trace_weights=jnp.concatenate(
                    (
                        proposed.q_trace_weights[:, :BASE_FEATURE_DIM],
                        committed_trace_tail,
                    ),
                    axis=1,
                ),
            ),
        )

    @staticmethod
    def _consumer_arrays(
        behavior_state: BehaviorModelState,
        control_state: DifferentialSARSAState,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            behavior_state.weights,
            control_state.q_weights,
            control_state.q_trace_weights,
            control_state.last_observation,
        )

    @staticmethod
    def _consumer_arrays_with_grounded(
        behavior_state: BehaviorModelState,
        control_state: DifferentialSARSAState,
        grounded_state: GroundedJointWorldModelState,
    ) -> tuple[Array, Array, Array, Array, Array]:
        return (
            behavior_state.weights,
            control_state.q_weights,
            control_state.q_trace_weights,
            control_state.last_observation,
            grounded_state.weights,
        )

    @staticmethod
    def _consumer_route_value_checks(
        before: Array,
        after: Array,
        *,
        safe_source_slots: Array,
        survivor_mask: Array,
        carry_survivor_values: Array,
        no_carry_change: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Independently compare one consumer before and after identity routing."""

        before_prefix = before[..., :BASE_FEATURE_DIM]
        after_prefix = after[..., :BASE_FEATURE_DIM]
        before_tail = before[..., BASE_FEATURE_DIM:]
        after_tail = after[..., BASE_FEATURE_DIM:]
        gathered = jnp.take(before_tail, safe_source_slots, axis=-1)
        mask_shape = (1,) * (after_tail.ndim - 1) + (ACTIVE_PAIR_SLOTS,)
        survivor = survivor_mask.reshape(mask_shape)

        prefix_exact = jnp.array_equal(after_prefix, before_prefix)
        survivor_exact = ~carry_survivor_values | jnp.all(
            (~survivor) | (after_tail == gathered)
        )
        reset_exact = jnp.all(survivor | (after_tail == 0.0))
        no_carry_exact = ~no_carry_change | jnp.all(after_tail == 0.0)
        component_exact = prefix_exact & survivor_exact & reset_exact & no_carry_exact
        return (
            prefix_exact,
            survivor_exact,
            reset_exact,
            no_carry_exact,
            component_exact,
        )

    @staticmethod
    def _consumer_destination_zero(
        value: Array,
        slot: Array,
        active: Array,
    ) -> Array:
        """Check one dynamic destination without data-dependent output shapes."""

        safe_slot = jnp.clip(slot, 0, ACTIVE_PAIR_SLOTS - 1)
        destination = jnp.take(value[..., BASE_FEATURE_DIM:], safe_slot, axis=-1)
        return ~active | jnp.all(destination == 0.0)

    def _audit_consumer_identity_route(
        self,
        *,
        old_descriptors: Array,
        new_descriptors: Array,
        route: FeatureBankRouteDiagnostics,
        behavior_before: Array,
        behavior_after: Array,
        q_before: Array,
        q_after: Array,
        trace_before: Array,
        trace_after: Array,
        last_observation_before: Array,
        last_observation_after: Array,
        grounded_before: Array | None,
        grounded_after: Array | None,
        retired_slot: Array,
        replaced_slot: Array,
    ) -> IntegratedConsumerRouteAudit:
        """Recompute descriptor routing and all consumer values independently."""

        old_left = old_descriptors[:, 0]
        old_right = old_descriptors[:, 1]
        new_left = new_descriptors[:, 0]
        new_right = new_descriptors[:, 1]
        old_live = (
            (old_left >= 0)
            & (old_left < old_right)
            & (old_right < BASE_FEATURE_DIM)
        )
        new_live = (
            (new_left >= 0)
            & (new_left < new_right)
            & (new_right < BASE_FEATURE_DIM)
        )
        identity_match = jnp.all(
            new_descriptors[:, None, :] == old_descriptors[None, :, :],
            axis=-1,
        )
        identity_match &= new_live[:, None] & old_live[None, :]
        expected_survivor = new_live & jnp.any(identity_match, axis=1)
        expected_new = new_live & ~expected_survivor
        expected_evicted = old_live & ~jnp.any(identity_match, axis=0)
        raw_source_slots = jnp.argmax(identity_match, axis=1).astype(jnp.int32)
        expected_source_slots = jnp.where(
            expected_survivor,
            raw_source_slots,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        safe_source_slots = jnp.where(
            expected_survivor,
            raw_source_slots,
            jnp.asarray(0, dtype=jnp.int32),
        )
        descriptors_changed = jnp.any(new_descriptors != old_descriptors)
        carry_survivor_values = jnp.asarray(
            self._config.carry_survivors,
            dtype=jnp.bool_,
        ) | ~descriptors_changed
        no_carry_change = (
            ~jnp.asarray(self._config.carry_survivors, dtype=jnp.bool_)
            & descriptors_changed
        )
        source_slots_exact = jnp.array_equal(
            route.source_slots,
            expected_source_slots,
        )
        identity_masks_exact = (
            route.valid
            & route.route_applied
            & jnp.array_equal(route.old_validation.live_mask, old_live)
            & jnp.array_equal(route.new_validation.live_mask, new_live)
            & jnp.array_equal(route.survivor_mask, expected_survivor)
            & jnp.array_equal(route.new_mask, expected_new)
            & jnp.array_equal(route.evicted_mask, expected_evicted)
            & (route.survivor_count == jnp.sum(expected_survivor, dtype=jnp.int32))
            & (route.new_count == jnp.sum(expected_new, dtype=jnp.int32))
            & (route.evicted_count == jnp.sum(expected_evicted, dtype=jnp.int32))
            & (route.descriptors_changed == descriptors_changed)
            & (route.carry_survivors == carry_survivor_values)
        )

        behavior_checks = self._consumer_route_value_checks(
            behavior_before,
            behavior_after,
            safe_source_slots=safe_source_slots,
            survivor_mask=expected_survivor,
            carry_survivor_values=carry_survivor_values,
            no_carry_change=no_carry_change,
        )
        q_checks = self._consumer_route_value_checks(
            q_before,
            q_after,
            safe_source_slots=safe_source_slots,
            survivor_mask=expected_survivor,
            carry_survivor_values=carry_survivor_values,
            no_carry_change=no_carry_change,
        )
        trace_checks = self._consumer_route_value_checks(
            trace_before,
            trace_after,
            safe_source_slots=safe_source_slots,
            survivor_mask=expected_survivor,
            carry_survivor_values=carry_survivor_values,
            no_carry_change=no_carry_change,
        )
        observation_checks = self._consumer_route_value_checks(
            last_observation_before,
            last_observation_after,
            safe_source_slots=safe_source_slots,
            survivor_mask=expected_survivor,
            carry_survivor_values=carry_survivor_values,
            no_carry_change=no_carry_change,
        )
        if grounded_before is None or grounded_after is None:
            if grounded_before is not None or grounded_after is not None:
                raise ValueError("grounded consumer route audit requires both value arrays")
            true = jnp.asarray(True, dtype=jnp.bool_)
            grounded_checks = (true, true, true, true, true)
            grounded_values: tuple[Array, ...] = ()
        else:
            grounded_checks = self._consumer_route_value_checks(
                grounded_before,
                grounded_after,
                safe_source_slots=safe_source_slots,
                survivor_mask=expected_survivor,
                carry_survivor_values=carry_survivor_values,
                no_carry_change=no_carry_change,
            )
            grounded_values = (grounded_after,)

        all_checks = (
            behavior_checks,
            q_checks,
            trace_checks,
            observation_checks,
            grounded_checks,
        )
        stable_prefix_exact = jnp.all(jnp.stack(tuple(check[0] for check in all_checks)))
        survivor_values_exact = jnp.all(jnp.stack(tuple(check[1] for check in all_checks)))
        reset_values_exact = jnp.all(jnp.stack(tuple(check[2] for check in all_checks)))
        no_carry_reset_exact = jnp.all(jnp.stack(tuple(check[3] for check in all_checks)))
        component_values_exact = jnp.all(jnp.stack(tuple(check[4] for check in all_checks)))

        lifecycle_enabled = jnp.asarray(
            self._config.feature_lifecycle_enabled,
            dtype=jnp.bool_,
        )
        retired = lifecycle_enabled & (retired_slot >= 0) & (retired_slot < ACTIVE_PAIR_SLOTS)
        replaced = (
            lifecycle_enabled & (replaced_slot >= 0) & (replaced_slot < ACTIVE_PAIR_SLOTS)
        )
        routed_values = (
            behavior_after,
            q_after,
            trace_after,
            last_observation_after,
        ) + grounded_values

        def lifecycle_destination_exact(slot: Array, active: Array) -> Array:
            safe_slot = jnp.clip(slot, 0, ACTIVE_PAIR_SLOTS - 1)
            identity_reset = ~active | ~expected_survivor[safe_slot]
            consumer_reset = jnp.all(
                jnp.stack(
                    tuple(
                        self._consumer_destination_zero(value, slot, active)
                        for value in routed_values
                    )
                )
            )
            return identity_reset & consumer_reset

        lifecycle_destination_reset_exact = lifecycle_destination_exact(
            retired_slot,
            retired,
        ) & lifecycle_destination_exact(replaced_slot, replaced)
        values_exact = (
            route.valid
            & source_slots_exact
            & identity_masks_exact
            & component_values_exact
        )
        return IntegratedConsumerRouteAudit(
            source_slots_exact=source_slots_exact,
            identity_masks_exact=identity_masks_exact,
            stable_prefix_exact=stable_prefix_exact,
            survivor_values_exact=survivor_values_exact,
            reset_values_exact=reset_values_exact,
            no_carry_reset_exact=no_carry_reset_exact,
            behavior_values_exact=behavior_checks[4],
            q_values_exact=q_checks[4],
            trace_values_exact=trace_checks[4],
            last_observation_exact=observation_checks[4],
            grounded_values_exact=grounded_checks[4],
            values_exact=values_exact,
            lifecycle_destination_reset_exact=lifecycle_destination_reset_exact,
        )

    def _route_feature_consumers(
        self,
        router_state: FeatureBankRouterState,
        behavior_state: BehaviorModelState,
        control_state: DifferentialSARSAState,
        proposed_descriptors: Array,
    ) -> tuple[
        BehaviorModelState,
        DifferentialSARSAState,
        FeatureBankRouteDiagnostics,
        FeatureBankRouterState,
    ]:
        """Route every downstream feature-indexed array in one transaction.

        The matched no-carry ablation resets the dynamic consumer tail only on
        a valid descriptor-bank change.  Unchanged routing calls retain learned
        columns while preserving the same route-counter cadence.
        """
        route = self._router.route(
            router_state,
            self._consumer_arrays(behavior_state, control_state),
            proposed_descriptors,
            carry_survivors=True,
        )
        routed = cast(tuple[Array, Array, Array, Array], route.consumers)
        diagnostics = route.diagnostics
        if not self._config.carry_survivors:
            descriptors_changed = diagnostics.descriptors_changed

            def reset_dynamic_tail(value: Array) -> Array:
                stable_prefix = value[..., :BASE_FEATURE_DIM]
                zero_tail = jnp.zeros_like(value[..., BASE_FEATURE_DIM:])
                reset_value = jnp.concatenate(
                    (stable_prefix, zero_tail),
                    axis=-1,
                )
                return jnp.where(
                    descriptors_changed,
                    reset_value,
                    value,
                )

            routed = cast(
                tuple[Array, Array, Array, Array],
                jax.tree_util.tree_map(reset_dynamic_tail, routed),
            )
            diagnostics = dataclasses.replace(
                diagnostics,
                carry_survivors=~descriptors_changed,
            )
        return (
            behavior_state.replace(weights=routed[0]),
            control_state.replace(
                q_weights=routed[1],
                q_trace_weights=routed[2],
                last_observation=routed[3],
            ),
            diagnostics,
            route.state,
        )

    def _route_feature_consumers_with_grounded(
        self,
        router_state: FeatureBankRouterState,
        behavior_state: BehaviorModelState,
        control_state: DifferentialSARSAState,
        grounded_state: GroundedJointWorldModelState,
        proposed_descriptors: Array,
    ) -> tuple[
        BehaviorModelState,
        DifferentialSARSAState,
        GroundedJointWorldModelState,
        FeatureBankRouteDiagnostics,
        FeatureBankRouterState,
    ]:
        """Route all legacy consumers and grounded weights in one transaction."""
        route = self._router.route(
            router_state,
            self._consumer_arrays_with_grounded(
                behavior_state,
                control_state,
                grounded_state,
            ),
            proposed_descriptors,
            carry_survivors=True,
        )
        routed = cast(tuple[Array, Array, Array, Array, Array], route.consumers)
        diagnostics = route.diagnostics
        if not self._config.carry_survivors:
            descriptors_changed = diagnostics.descriptors_changed

            def reset_dynamic_tail(value: Array) -> Array:
                reset_value = jnp.concatenate(
                    (
                        value[..., :BASE_FEATURE_DIM],
                        jnp.zeros_like(value[..., BASE_FEATURE_DIM:]),
                    ),
                    axis=-1,
                )
                return jnp.where(descriptors_changed, reset_value, value)

            routed = cast(
                tuple[Array, Array, Array, Array, Array],
                jax.tree_util.tree_map(reset_dynamic_tail, routed),
            )
            diagnostics = dataclasses.replace(
                diagnostics,
                carry_survivors=~descriptors_changed,
            )
        return (
            behavior_state.replace(weights=routed[0]),
            control_state.replace(
                q_weights=routed[1],
                q_trace_weights=routed[2],
                last_observation=routed[3],
            ),
            grounded_state.replace(weights=routed[4]),
            diagnostics,
            route.state,
        )

    @staticmethod
    def _start_finite(state: IntegratedHiddenPartnerState) -> Array:
        grounded_valid = (
            jnp.asarray(True, dtype=jnp.bool_)
            if state.current_evaluation.grounded_world is None
            else state.current_evaluation.grounded_world.predictions_valid
        )
        return (
            _numeric_tree_finite(state)
            & state.current_evaluation.partner_probabilities_valid
            & grounded_valid
        )

    @staticmethod
    def _update_finite(
        state: IntegratedHiddenPartnerState,
        chi_gradient: Array,
        phi_gradient: Array,
        behavior_loss: Array,
        interaction_metrics: Array,
        candidate_promotion_signal: Array,
        world_reward_error: Array,
        world_outcome_error: Array,
        td_error: Array,
    ) -> Array:
        transients: tuple[Array, ...] = (
            chi_gradient,
            phi_gradient,
            behavior_loss,
            interaction_metrics,
            candidate_promotion_signal,
            world_reward_error,
            world_outcome_error,
            td_error,
        )
        return _numeric_tree_finite(state) & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in transients])
        )
