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
representation, evaluates all four joint-action cells once, selects an action
with the SARSA exploration RNG, and stores the exact decision record.

``update`` accepts the resulting environment transition and performs:

1. score the stored pre-action behavior/world decision and executed world cell;
2. differentiate pre-update partner cross entropy through pair products;
3. learn the recurrent state parameters before advancing recurrence;
4. update partner prediction and the shadow feature-discovery learner;
5. advance the state builder exactly once with the next observation and the
   preceding action/reward/discount;
6. update only the executed joint-world cell;
7. atomically route behavior and control feature columns by pair identity;
8. construct the next representation under the deployed descriptor bank;
9. predict the partner and marginalize all four joint cells exactly once;
10. select the next external planner action while advancing SARSA's RNG; and
11. update differential SARSA with that explicit next action.

All ablations keep the same fixed shapes.  ``planning_enabled=False`` masks
only the centered additive model term; partner and world predictions are still
computed. ``state_learning_enabled=False`` computes but discards the recurrent
parameter update while recurrence still advances. ``feature_lifecycle_enabled
=False`` lets discovery learn in shadow but keeps the deployed birth bank.
``uniform_partner_belief=True`` still predicts and learns the partner model but
applies a uniform belief to the joint-world marginalization.
``random_feature_curation=True`` still computes the complete utility-learning
path but replaces only its active/candidate rankings with seeded random
priorities before each fixed-cadence curation decision.
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
from typing import Any, Protocol, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int

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
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    RELEVANCE_PROBE_MODES,
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
)
from alberta_framework.core.joint_partner_world import (
    BoundedJointOutcomeConfig,
    BoundedJointOutcomeModel,
    BoundedJointOutcomeState,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
    StateBuilderLearningDiagnostics,
)

INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION = "alberta.integrated-hidden-partner.l0.v10"
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
        raise TypeError(
            f"{name} must have dtype {expected_dtype}, got {array.dtype}"
        )
    return array


def _saturating_int32_increment(value: Array) -> Array:
    """Increment a non-negative int32 counter without wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


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
    evidence_gated_feature_memory: bool = False
    feature_evidence_confirmation_steps: int = 1
    independent_relevance_probe: bool = False
    relevance_probe_mode: str = RELEVANCE_PROBE_MODE_CONDITIONAL_V1
    evidence_gated_consumer_memory: bool = False
    consumer_evidence_confirmation_steps: int = 1
    consumer_read_confirmation_steps: int = 1
    consumer_read_lease_steps: int = 32
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
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if (
            not isinstance(self.relevance_probe_mode, str)
            or self.relevance_probe_mode not in RELEVANCE_PROBE_MODES
        ):
            raise ValueError(
                "relevance_probe_mode must be 'conditional_v1' or "
                "'target_only_v1'"
            )
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
        if self.random_feature_curation and interaction_utility_decay <= 0.0:
            raise ValueError(
                "interaction_utility_decay must be positive for random feature curation"
            )
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
        if self.evidence_gated_consumer_memory and not self.feature_lifecycle_enabled:
            raise ValueError(
                "evidence_gated_consumer_memory requires feature_lifecycle_enabled"
            )
        if self.evidence_gated_feature_memory and not self.feature_lifecycle_enabled:
            raise ValueError(
                "evidence_gated_feature_memory requires feature_lifecycle_enabled"
            )
        if self.evidence_gated_feature_memory and grace is None:
            raise ValueError(
                "evidence_gated_feature_memory requires "
                "active_utility_retention_grace_steps"
            )
        if self.evidence_gated_feature_memory and evidence_threshold <= 0.0:
            raise ValueError(
                "evidence_gated_feature_memory requires a positive "
                "active_utility_evidence_threshold"
            )
        if self.independent_relevance_probe and not self.evidence_gated_feature_memory:
            raise ValueError(
                "independent_relevance_probe requires "
                "evidence_gated_feature_memory"
            )
        if self.independent_relevance_probe and not self.evidence_gated_consumer_memory:
            raise ValueError(
                "independent_relevance_probe requires "
                "evidence_gated_consumer_memory"
            )
        if self.evidence_gated_consumer_memory and grace is None:
            raise ValueError(
                "evidence_gated_consumer_memory requires "
                "active_utility_retention_grace_steps"
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
            and self.consumer_read_confirmation_steps
            > self.consumer_evidence_confirmation_steps
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
                "candidate_promotion_confirmation_steps must be a positive "
                "int32-safe integer"
            )
        candidate_reacquisition = self.candidate_reacquisition_confirmation_steps
        if (
            isinstance(candidate_reacquisition, bool)
            or not isinstance(candidate_reacquisition, int)
            or not 1 <= candidate_reacquisition < _INT32_MAX
        ):
            raise ValueError(
                "candidate_reacquisition_confirmation_steps must be a positive "
                "int32-safe integer"
            )
        if candidate_reacquisition > 1 and (
            not self.independent_relevance_probe or not self.retire_stale_features
        ):
            raise ValueError(
                "candidate_reacquisition_confirmation_steps greater than one "
                "requires independent_relevance_probe and retire_stale_features"
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
        return {
            "type": type(self).__name__,
            "schema_version": INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION,
            "development_level": DEVELOPMENT_LEVEL,
            "accepted_scientific_evidence": False,
            **dataclasses.asdict(self),
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
            raise ValueError("integrated config fields do not match the v10 schema")
        if values.pop("type") != cls.__name__:
            raise ValueError("integrated config type is invalid")
        if values.pop("schema_version") != INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION:
            raise ValueError("integrated config schema version is unsupported")
        if values.pop("development_level") != DEVELOPMENT_LEVEL:
            raise ValueError("integrated kernel must remain development level L0")
        if values.pop("accepted_scientific_evidence") is not False:
            raise ValueError("integrated kernel is not accepted scientific evidence")
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
    control_nbytes: int
    router_nbytes: int
    consumer_active_mask_nbytes: int
    consumer_evidence_streak_nbytes: int
    consumer_read_idle_steps_nbytes: int
    decision_cache_nbytes: int
    total_state_nbytes: int
    planner_cell_evaluations_per_decision: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact accounting record."""
        return dataclasses.asdict(self)


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


@chex.dataclass(frozen=True)
class IntegratedPlannerSelection:
    """External epsilon-greedy selection and explicit RNG accounting."""

    action: Int[Array, ""]
    noisy_greedy_action: Int[Array, ""]
    random_action: Int[Array, ""]
    explored: Bool[Array, ""]
    rng_key_before: Array
    rng_key_after: Array


@chex.dataclass(frozen=True)
class IntegratedHiddenPartnerState:
    """All bounded online state plus the active pre-TD decision record.

    ``current_evaluation`` is the model/Q evaluation that selected
    ``control.last_action``.  After an update it deliberately reflects the
    next decision before that transition's SARSA parameter update, rather than
    recomputing Q with post-update weights.
    """

    state_builder: OnlineGatedStateBuilderState
    interaction: InteractionFeatureState
    behavior: BehaviorModelState
    joint_world: BoundedJointOutcomeState
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
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedStartDiagnostics:
    """Mechanism diagnostics for initial observation consumption."""

    evaluation: IntegratedPlannerEvaluation
    selection: IntegratedPlannerSelection
    descriptors: Int[Array, "12 2"]
    descriptors_valid: Bool[Array, ""]
    state_advances: Int[Array, ""]
    all_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class IntegratedStartResult:
    """Initialized state and first externally planned action."""

    state: IntegratedHiddenPartnerState
    action: Int[Array, ""]
    diagnostics: IntegratedStartDiagnostics


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
    state_learning: StateBuilderLearningDiagnostics
    interaction_prediction_preupdate: Float[Array, " 1"]
    interaction_error_preupdate: Float[Array, " 1"]
    interaction_metrics: Float[Array, " 7"]
    interaction_replaced_slot: Int[Array, ""]
    interaction_promoted_candidate: Int[Array, ""]
    interaction_retired_slot: Int[Array, ""]
    interaction_retired_left: Int[Array, ""]
    interaction_retired_right: Int[Array, ""]
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
    interaction_candidate_promotion_evidence_streak_post: Int[Array, " 66"]
    interaction_candidate_promotion_confirmed: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_pre: Bool[Array, " 66"]
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
    interaction_live_feature_count: Int[Array, ""]
    interaction_vacancy_count: Int[Array, ""]
    interaction_promoted_into_vacancy: Bool[Array, ""]
    random_curation_applied: Bool[Array, ""]
    random_active_priorities: Float[Array, " 12"]
    random_candidate_priorities: Float[Array, " 66"]
    shadow_descriptors: Int[Array, "12 2"]
    proposed_descriptors: Int[Array, "12 2"]
    shadow_descriptors_changed: Bool[Array, ""]
    route: FeatureBankRouteDiagnostics
    world_reward_prediction_preupdate: Float[Array, ""]
    world_outcome_prediction_preupdate: Float[Array, " 1"]
    world_reward_error: Float[Array, ""]
    world_outcome_error: Float[Array, " 1"]
    world_target_valid: Bool[Array, ""]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    transition_observation_matches: Bool[Array, ""]
    transition_action_matches: Bool[Array, ""]
    transition_semantics_valid: Bool[Array, ""]
    transition_rejected: Bool[Array, ""]
    model_valid: Bool[Array, ""]
    state_builder_step_delta: Int[Array, ""]
    state_builder_learning_delta: Int[Array, ""]
    behavior_step_delta: Int[Array, ""]
    interaction_step_delta: Int[Array, ""]
    world_step_delta: Int[Array, ""]
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
            utility_evidence_confirmation_steps=(
                cfg.feature_evidence_confirmation_steps
            ),
            independent_relevance_probe=cfg.independent_relevance_probe,
            relevance_probe_mode=cfg.relevance_probe_mode,
            retire_stale_features=cfg.retire_stale_features,
            candidate_promotion_floor=cfg.candidate_promotion_floor,
            candidate_promotion_confirmation_steps=(
                cfg.candidate_promotion_confirmation_steps
            ),
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
            INITIAL_ACTIVE_DESCRIPTORS,
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
            "control": self._control.to_config(),
            "router": self._router.to_config(),
            "initial_active_descriptors": [list(pair) for pair in INITIAL_ACTIVE_DESCRIPTORS],
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    def resource_budget(
        self,
        state: IntegratedHiddenPartnerState,
    ) -> IntegratedHiddenPartnerResourceBudget:
        """Return exact persistent array bytes without double-counting consumers."""
        consumers = self._consumer_arrays(state.behavior, state.control)
        router_budget = self._router.resource_budget(
            state.router,
            consumers,
        )
        cache_bytes = (
            _tree_array_nbytes(state.raw_observation)
            + _tree_array_nbytes(state.phi)
            + _tree_array_nbytes(state.chi)
            + _tree_array_nbytes(state.current_evaluation)
            + _tree_array_nbytes(state.step_count)
        )
        consumer_active_mask_bytes = _tree_array_nbytes(state.consumer_active_mask)
        consumer_evidence_streak_bytes = _tree_array_nbytes(
            state.consumer_evidence_streak
        )
        consumer_read_idle_steps_bytes = _tree_array_nbytes(
            state.consumer_read_idle_steps
        )
        builder_bytes = self._state_builder.resource_budget().state_bytes
        # ``start`` is jitted, so the interaction learner's Python timing
        # fields return as scalar array leaves. Count the actual integrated
        # state tree rather than the component's scientific-array-only budget.
        interaction_bytes = _tree_array_nbytes(state.interaction)
        behavior_bytes = self._behavior.resource_budget(DEPLOYED_FEATURE_DIM).state_nbytes
        world_bytes = self._joint_world.resource_budget.state_nbytes
        control_bytes = _tree_array_nbytes(state.control)
        total = (
            builder_bytes
            + interaction_bytes
            + behavior_bytes
            + world_bytes
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
            control_nbytes=control_bytes,
            router_nbytes=router_budget.router_state_nbytes,
            consumer_active_mask_nbytes=consumer_active_mask_bytes,
            consumer_evidence_streak_nbytes=consumer_evidence_streak_bytes,
            consumer_read_idle_steps_nbytes=consumer_read_idle_steps_bytes,
            decision_cache_nbytes=cache_bytes,
            total_state_nbytes=total,
            planner_cell_evaluations_per_decision=(
                self._joint_world.resource_budget.planner_cell_evaluations_per_decision
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
    def evaluate_models(
        self,
        behavior_state: BehaviorModelState,
        world_state: BoundedJointOutcomeState,
        control_state: DifferentialSARSAState,
        chi: Array,
    ) -> IntegratedPlannerEvaluation:
        """Evaluate partner, all four world cells, Q, and planner scores."""
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
        q_values = self._control.q_values(control_state, features)
        centered = marginal.expected_rewards - jnp.mean(marginal.expected_rewards)
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
            expected_rewards=marginal.expected_rewards,
            expected_outcomes=marginal.expected_outcomes,
            q_values=q_values,
            centered_expected_rewards=centered,
            model_term=model_term,
            applied_model_term=applied_model_term,
            planner_scores=scores,
            greedy_action=jnp.argmax(scores).astype(jnp.int32),
            cell_evaluations=marginal.cell_evaluations,
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
            rng_key_before=control_state.rng_key,
            rng_key_after=key,
        )

    def start(
        self,
        raw_observation: Array,
        key: Array,
    ) -> IntegratedStartResult:
        """Initialize every bounded component and select the first action.

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
        builder_state, phi = self._state_builder.start(
            builder_initial,
            raw,
        )
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
        )
        selection = self.select_planner_action(
            control_state,
            evaluation.planner_scores,
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
            control=control_started,
            router=router_state,
            raw_observation=raw,
            phi=phi,
            chi=chi,
            consumer_active_mask=consumer_active_mask,
            consumer_evidence_streak=consumer_evidence_streak,
            consumer_read_idle_steps=consumer_read_idle_steps,
            current_evaluation=evaluation,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        diagnostics = IntegratedStartDiagnostics(
            evaluation=evaluation,
            selection=selection,
            descriptors=router_state.descriptors,
            descriptors_valid=descriptor_validation.valid,
            state_advances=(builder_state.step_count - builder_initial.step_count),
            all_finite=self._start_finite(state),
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
        """Consume one exact hidden-partner transition in causal order.

        Static shape/dtype violations fail while tracing.  Dynamic transition
        violations are an atomic no-op: the exact old state and old action are
        returned, every persistent counter delta is zero, and diagnostics set
        ``transition_rejected``.  This contract is identical under eager JAX,
        :func:`jax.jit`, and :func:`jax.lax.scan`.
        """
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

        observations_finite = jnp.all(jnp.isfinite(raw_current)) & jnp.all(
            jnp.isfinite(raw_next)
        )
        observation_matches = jnp.all(raw_current == state.raw_observation)
        focal_action_valid = (raw_focal_action >= 0) & (raw_focal_action < N_ACTIONS)
        partner_action_valid = (raw_partner_action >= 0) & (
            raw_partner_action < N_ACTIONS
        )
        action_ids_valid = focal_action_valid & partner_action_valid
        action_matches = raw_focal_action == state.control.last_action
        outcome_valid = jnp.isfinite(raw_outcome) & (
            (raw_outcome == -1.0) | (raw_outcome == 1.0)
        )
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
        behavior_gradient = self._behavior.input_loss_gradient(
            state.behavior,
            state.chi,
            partner_action,
        )
        world_prediction = self._joint_world.predict_joint(
            state.joint_world,
            focal_action,
            partner_action,
        )
        representation_gradient = self.chain_chi_gradient_to_phi(
            state.phi,
            state.router.descriptors,
            behavior_gradient.gradient,
            self._effective_pair_read_mask(
                state.interaction,
                state.consumer_active_mask,
            ),
        )

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
        (
            interaction_input,
            random_active_priorities,
            random_candidate_priorities,
        ) = self._interaction_curation_input(state.interaction)
        interaction_update = self._interaction.update(
            interaction_input,
            self._deployed_phi(state.phi),
            jnp.reshape(partner_sign_target, (1,)),
            external_read_mask=state.consumer_active_mask,
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
            (state.router.descriptors[:, 0] >= 0)
            & (state.router.descriptors[:, 1] >= 0),
        )
        committed_behavior = self._commit_behavior_consumer_update(
            state.behavior,
            behavior_update.state,
            consumer_write_gate_pre,
        )

        advanced_builder, next_phi = self._state_builder.update(
            deployed_builder,
            next_raw,
            focal_action,
            reward,
            discount,
        )
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
        proposed_descriptors = (
            shadow_descriptors
            if self._config.feature_lifecycle_enabled
            else state.router.descriptors
        )
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
                interaction_update.state,
                consumer_active_mask_post,
            ),
        )
        next_evaluation = self.evaluate_models(
            routed_behavior,
            world_update.state,
            routed_control,
            next_chi,
        )
        next_selection = self.select_planner_action(
            routed_control,
            next_evaluation.planner_scores,
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

        proposed_state = IntegratedHiddenPartnerState(
            state_builder=advanced_builder,
            interaction=interaction_update.state,
            behavior=routed_behavior,
            joint_world=world_update.state,
            control=committed_control,
            router=router_state,
            raw_observation=next_raw,
            phi=next_phi,
            chi=next_chi,
            consumer_active_mask=consumer_active_mask_post,
            consumer_evidence_streak=consumer_evidence_streak_post,
            consumer_read_idle_steps=consumer_read_idle_steps_post,
            current_evaluation=next_evaluation,
            step_count=_saturating_int32_increment(state.step_count),
        )
        next_state = jax.lax.cond(
            transition_valid,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        behavior_prediction_matches = jnp.allclose(
            behavior_gradient.probabilities,
            current_evaluation.predicted_partner_probabilities,
            atol=1e-6,
            rtol=1e-6,
        )
        model_valid = (
            current_evaluation.partner_probabilities_valid
            & next_evaluation.partner_probabilities_valid
            & world_update.target_valid
        )
        false_active = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
        false_candidates = jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_)
        rejected_index = jnp.asarray(-1, dtype=jnp.int32)
        old_descriptor_validation = self._router.validate_descriptors(
            state.router.descriptors
        )
        old_live_count = jnp.sum(
            old_descriptor_validation.live_mask,
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
            behavior_gradient_phi=representation_gradient,
            state_learning=state_learning,
            interaction_prediction_preupdate=(interaction_update.predictions),
            interaction_error_preupdate=interaction_update.errors,
            interaction_metrics=interaction_update.metrics,
            interaction_replaced_slot=jnp.where(
                transition_valid,
                interaction_update.replaced_slot,
                rejected_index,
            ),
            interaction_promoted_candidate=jnp.where(
                transition_valid,
                interaction_update.promoted_candidate,
                rejected_index,
            ),
            interaction_retired_slot=jnp.where(
                transition_valid,
                interaction_update.retired_slot,
                rejected_index,
            ),
            interaction_retired_left=jnp.where(
                transition_valid,
                interaction_update.retired_left,
                rejected_index,
            ),
            interaction_retired_right=jnp.where(
                transition_valid,
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
            interaction_relevance_probe_weights_pre=(
                state.interaction.relevance_probe_weights
            ),
            interaction_relevance_probe_weights_post=(
                next_state.interaction.relevance_probe_weights
            ),
            interaction_relevance_probe_biases_pre=(
                state.interaction.relevance_probe_biases
            ),
            interaction_relevance_probe_biases_post=(
                next_state.interaction.relevance_probe_biases
            ),
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
                    transition_valid,
                    interaction_update.candidate_promotion_evidence_streak_updated,
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
                    transition_valid,
                    interaction_update.matching_candidate_reset_mask,
                    false_candidates,
                )
            ),
            interaction_matching_candidate_reset_count=jnp.where(
                transition_valid,
                jnp.sum(
                    interaction_update.matching_candidate_reset_mask,
                    dtype=jnp.int32,
                ),
                0,
            ),
            interaction_live_feature_count=jnp.where(
                transition_valid,
                interaction_update.live_feature_count,
                old_live_count,
            ),
            interaction_vacancy_count=jnp.where(
                transition_valid,
                interaction_update.vacancy_count,
                ACTIVE_PAIR_SLOTS - old_live_count,
            ),
            interaction_promoted_into_vacancy=(
                transition_valid & interaction_update.promoted_into_vacancy
            ),
            random_curation_applied=(
                transition_valid
                & jnp.asarray(
                    self._config.random_feature_curation,
                    dtype=jnp.bool_,
                )
            ),
            random_active_priorities=jnp.where(
                transition_valid,
                random_active_priorities,
                jnp.zeros_like(random_active_priorities),
            ),
            random_candidate_priorities=jnp.where(
                transition_valid,
                random_candidate_priorities,
                jnp.zeros_like(random_candidate_priorities),
            ),
            shadow_descriptors=jnp.where(
                transition_valid,
                shadow_descriptors,
                state.router.descriptors,
            ),
            proposed_descriptors=jnp.where(
                transition_valid,
                proposed_descriptors,
                state.router.descriptors,
            ),
            shadow_descriptors_changed=(
                transition_valid
                & jnp.any(shadow_descriptors != state.router.descriptors)
            ),
            route=committed_route_diagnostics,
            world_reward_prediction_preupdate=world_prediction.reward,
            world_outcome_prediction_preupdate=world_prediction.outcome,
            world_reward_error=world_update.reward_error,
            world_outcome_error=world_update.outcome_error,
            world_target_valid=world_update.target_valid,
            td_error=control_update.td_error,
            average_reward=control_update.average_reward,
            transition_observation_matches=observation_matches,
            transition_action_matches=action_matches,
            transition_semantics_valid=transition_valid,
            transition_rejected=~transition_valid,
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
                    behavior_gradient.gradient,
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
        consumer = jnp.asarray(consumer_read_mask, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
    ) -> tuple[InteractionFeatureState, Array, Array]:
        """Replace only utility rankings for the matched random-curation arm."""
        if not self._config.random_feature_curation:
            return (
                state,
                jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
                jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.float32),
            )
        active_key = jr.fold_in(state.key, jnp.uint32(0x43555241))
        candidate_key = jr.fold_in(state.key, jnp.uint32(0x43555243))
        active_order = jr.permutation(
            active_key,
            ACTIVE_PAIR_SLOTS,
        ).astype(jnp.float32)
        candidate_order = jr.permutation(
            candidate_key,
            CANDIDATE_PAIR_SLOTS,
        ).astype(jnp.float32)
        decay = jnp.asarray(
            self._config.interaction_utility_decay,
            dtype=jnp.float32,
        )
        gap = (2.0 - decay) / decay
        active_priorities = gap * active_order
        candidate_priorities = (4.0 / decay) + gap * candidate_order
        return (
            state.replace(
                utilities=active_priorities,
                candidate_utilities=candidate_priorities,
            ),
            active_priorities,
            candidate_priorities,
        )

    def _update_consumer_evidence_streak(
        self,
        previous_streak: Array,
        evidence_refreshed: Array,
    ) -> tuple[Array, Array, Array]:
        """Return updated streak, read acquisition, and old-bank write confirmation."""
        previous = jnp.asarray(previous_streak, dtype=jnp.int32).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
        if not self._config.evidence_gated_consumer_memory:
            return (
                jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
                jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
                jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
            )
        evidence = jnp.asarray(evidence_refreshed, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        previous = jnp.asarray(previous_idle_steps, dtype=jnp.int32).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
        if not self._config.evidence_gated_consumer_memory:
            return jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        evidence = jnp.asarray(evidence_refreshed, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
        live = jnp.asarray(live_mask, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        updated = jnp.asarray(updated_streak, dtype=jnp.int32).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        updated = jnp.asarray(updated_idle_steps, dtype=jnp.int32).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        confirmed = jnp.asarray(confirmed_write, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        acquire = jnp.asarray(read_acquire, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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
        previous = jnp.asarray(previous_mask, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
        if not self._config.evidence_gated_consumer_memory:
            return jnp.where(
                route.valid,
                route.new_validation.live_mask,
                previous,
            )
        acquire = jnp.asarray(read_acquire, dtype=jnp.bool_).reshape(
            (ACTIVE_PAIR_SLOTS,)
        )
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

    @staticmethod
    def _start_finite(state: IntegratedHiddenPartnerState) -> Array:
        values = (
            state.raw_observation,
            state.phi,
            state.chi,
            state.interaction.relevance_probe_weights,
            state.interaction.relevance_probe_biases,
            state.interaction.candidate_promotion_evidence_streak,
            state.interaction.candidate_reacquisition_required,
            state.current_evaluation.partner_probabilities,
            state.current_evaluation.predicted_partner_probabilities,
            state.current_evaluation.expected_rewards,
            state.current_evaluation.expected_outcomes,
            state.current_evaluation.q_values,
            state.current_evaluation.planner_scores,
        )
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in values]))

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
        values = (
            state.raw_observation,
            state.phi,
            state.chi,
            state.interaction.output_weights,
            state.interaction.relevance_probe_weights,
            state.interaction.relevance_probe_biases,
            state.interaction.output_biases,
            state.interaction.utilities,
            state.interaction.candidate_output_weights,
            state.interaction.candidate_utilities,
            state.interaction.candidate_promotion_evidence_streak,
            state.interaction.candidate_reacquisition_required,
            state.interaction.feature_second_moments,
            state.interaction.candidate_second_moments,
            state.interaction.target_second_moments,
            state.behavior.weights,
            state.behavior.bias,
            state.joint_world.reward_predictions,
            state.joint_world.outcome_predictions,
            state.control.q_weights,
            state.control.q_trace_weights,
            state.current_evaluation.partner_probabilities,
            state.current_evaluation.predicted_partner_probabilities,
            state.current_evaluation.expected_rewards,
            state.current_evaluation.planner_scores,
            chi_gradient,
            phi_gradient,
            behavior_loss,
            interaction_metrics,
            candidate_promotion_signal,
            world_reward_error,
            world_outcome_error,
            td_error,
        )
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in values]))
