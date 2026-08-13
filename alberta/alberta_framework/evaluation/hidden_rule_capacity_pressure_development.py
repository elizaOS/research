# mypy: disable-error-code="call-arg"
"""Consumed-root calibration of hidden-rule retention under slot pressure.

This development-only lane runs two independent differential-SARSA learners
for one uninterrupted 4,000-transition life.  A learner acts from the context
inferred from *completed prior experience*.  Only after both actions are fixed
does its independent :class:`~alberta_framework.core.context_inference.ContextInference`
bank receive ``(partner_action, own_action, common_reward)``.  Neither learner
receives the rule, phase, boundary, schedule, clock, or evaluator diagnostics.

Four epsilon values are a predeclared calibration grid on the single consumed
root ``0``.  They share the same controller PRNG key streams: action selection
draws exploratory and tie-breaking candidates on every call, independent of
which epsilon branch wins.  This module has no arbitrary-root runner, output
writer, artifact schema, threshold, arm selector, or promotion path.

The context bank has three recyclable slots while the life contains four
hidden conventions.  Slot indices are therefore not semantic identities.  An
evaluator-only birth ledger stamps every allocation/replacement with the
context bank's exact post-update identity.  A context identity is the pair
``(agent_namespace, birth_words)``; equality of slot indices, either over time
or between agents, is never treated as identity equality.  The ledger is not
fed to either learner and its bytes are accounted separately.

The schedule also exposes a no-free-lunch boundary.  At first C admission,
the observed past cannot say whether one-shot B or one-shot D will recur.  An
actual future needs ``{A, B, C}``; a prefix-identical counterfactual needs
``{A, D, C}``.  With capacity three, no deterministic online choice made from
the identical prefix can guarantee zero recurrence loss on both futures
without a prior.  :func:`build_prefix_twin_boundary` binds this statement to
an executable common-prefix digest.  It does not execute either future or
make a claim about stochastic optimality.

Post-audit sibling intervention
===============================

The original four-arm panel remains the baseline of record.  A separately
labeled eight-life replay pairs that exact baseline with a birth-authenticated
controller scrub at every epsilon.  On a new semantic birth into slot ``j``,
the sibling authenticates the source/destination banks and birth ledgers,
requires ``j`` to differ from the controller's currently credited source,
and zeros only ``q_weights[:, j]`` and ``q_trace_weights[:, j]`` before SARSA
computes ``q_next`` or selects its next action.  Both paired conditions stage
the same authentication and scrub alternative, so persistent resources, work,
and raw PRNG key streams remain matched.  This replay is descriptive,
threshold-free, winner-free, root-zero-only, and nonpromoting.

Selective-retention sibling intervention
=========================================

A second, separately labeled eight-life replay keeps the authenticated
controller scrub enabled in both conditions and varies only the signal used
at an otherwise-valid full-bank eviction.  The signal condition protects a
slot by the number of recurrence intervals that its *current semantic birth*
has already completed.  Birth-bound occurrence and interval words are held
in an evaluator companion state and updated only after authenticated context
events.  The no-signal condition computes and audits the same history but
dispatches exact zeros.  No schedule, rule label, phase, boundary, future
event, threshold, tuning parameter, or oracle enters the score.

Cross-birth lineage-cache sibling intervention
===============================================

A third eight-life replay asks a narrower question: can a bounded online
mechanism learn retention value *across* semantic rebirths, rather than merely
counting reuse of the current birth?  Each agent receives one evaluator-side
victim-cache record.  A record contains a frozen evicted reward model, its
authenticated source birth, a stable lineage identity, and an exact rescue
counter.  On a later full-bank allocation, the pre-event cached model must
have strictly smaller absolute error on the just-completed transition than
the fresh prior and every live source model.  Equality, nonfinite values, or
an invalid binding abstains.  A unique strict win transfers that lineage to
the new birth and increments its exact rescue counter; otherwise the birth
starts a new lineage.  The victim is archived by a deterministic
value-then-recency rule, with cache capacity fixed at the minimum nonzero
value of one.

The outcome can update only the successor lineage/cache transaction.  The
eviction-protection vector was snapshotted from exact source rescue counters
before actions and reward, so the current outcome can never choose its own
victim.  Both conditions compute the same match, lineage, cache, scrub, and
controller work; the no-signal condition dispatches zeros while the signal
condition dispatches source rescue counts.  There is no task label, rule,
phase, schedule, distance threshold, configurable similarity, archive search,
winner selection, output writer, or promotion path.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
from typing import Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

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
    ContextInferencePrioritizedUpdateResult,
    ContextInferenceState,
    ContextInferenceUpdateResult,
    context_inference_clock_nbytes,
    measure_context_inference_state_nbytes,
)
from alberta_framework.streams.matrix_game import (
    CONVENTION_GAME_EXACT_CLOCK_NBYTES,
    ConventionGameConfig,
    ConventionGameState,
    RecurringConventionGame,
    measure_convention_game_state_nbytes,
)

DEVELOPMENT_SCHEMA = "alberta.hidden-rule-capacity-pressure-development.v1"
DEVELOPMENT_NAMESPACE = "hidden-rule-capacity-pressure-consumed-calibration-root-0-v1"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
ARBITRARY_ROOT_EXECUTION_ALLOWED = False
CALIBRATION_ROOT_CONSUMED = True
ENVIRONMENT_RANDOMNESS_CONSUMED = False

N_ACTIONS = 4
PHASE_LENGTH = 400
OFFSETS = (0, 1, 0, 3, 0, 2, 0, 1, 2, 0)
PHASE_LABELS = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
LABEL_OFFSETS = (0, 1, 2, 3)  # A, B, C, D
NUM_PHASES = len(OFFSETS)
NUM_STEPS = PHASE_LENGTH * NUM_PHASES
SUMMARY_WINDOW = 64
MAX_CONTEXTS = 3
CALIBRATION_ROOT_INDEX = 0
EPSILON_GRID = (0.05, 0.1, 0.2, 0.4)
SEMANTIC_CONTEXT_IDENTITY = "(agent_namespace, exact_birth_words)"

POST_AUDIT_BASELINE: Literal["post_audit_baseline"] = "post_audit_baseline"
BIRTH_AUTHENTICATED_CONTROLLER_SCRUB: Literal[
    "birth_authenticated_controller_scrub"
] = "birth_authenticated_controller_scrub"
PostAuditCondition = Literal[
    "post_audit_baseline",
    "birth_authenticated_controller_scrub",
]
POST_AUDIT_CONDITIONS: tuple[PostAuditCondition, ...] = (
    POST_AUDIT_BASELINE,
    BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
)
POST_AUDIT_ONLY = True

SELECTIVE_RETENTION_NO_SIGNAL: Literal["selective_retention_no_signal"] = (
    "selective_retention_no_signal"
)
SELECTIVE_RETENTION_PAST_RECURRENCE: Literal[
    "selective_retention_past_recurrence"
] = "selective_retention_past_recurrence"
SelectiveRetentionCondition = Literal[
    "selective_retention_no_signal",
    "selective_retention_past_recurrence",
]
SELECTIVE_RETENTION_CONDITIONS: tuple[SelectiveRetentionCondition, ...] = (
    SELECTIVE_RETENTION_NO_SIGNAL,
    SELECTIVE_RETENTION_PAST_RECURRENCE,
)
SELECTIVE_RETENTION_DEVELOPMENT_ONLY = True

LINEAGE_CACHE_NO_SIGNAL: Literal["lineage_cache_no_signal"] = (
    "lineage_cache_no_signal"
)
LINEAGE_CACHE_PREDICTIVE_RESCUE: Literal[
    "lineage_cache_predictive_rescue"
] = "lineage_cache_predictive_rescue"
LineageCacheCondition = Literal[
    "lineage_cache_no_signal",
    "lineage_cache_predictive_rescue",
]
LINEAGE_CACHE_CONDITIONS: tuple[LineageCacheCondition, ...] = (
    LINEAGE_CACHE_NO_SIGNAL,
    LINEAGE_CACHE_PREDICTIVE_RESCUE,
)
LINEAGE_CACHE_CAPACITY = 1
LINEAGE_CACHE_DEVELOPMENT_ONLY = True

LEARNER_PRE_ACTION_CHANNELS = ("context_inferred_from_completed_prior_experience",)
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
        "environment_step_words",
        "birth_ledger",
        "future_schedule",
    }
)

_UINT32_MAX = 2**32 - 1


@dataclasses.dataclass(frozen=True)
class CapacityPressureProtocol:
    """Exact shape, timing, and nonpromotion contract."""

    schema: str = DEVELOPMENT_SCHEMA
    namespace: str = DEVELOPMENT_NAMESPACE
    n_actions: int = N_ACTIONS
    phase_length: int = PHASE_LENGTH
    offsets: tuple[int, ...] = OFFSETS
    phase_labels: tuple[str, ...] = PHASE_LABELS
    max_contexts: int = MAX_CONTEXTS
    summary_window: int = SUMMARY_WINDOW
    epsilon_grid: tuple[float, ...] = EPSILON_GRID
    calibration_root_index: int = CALIBRATION_ROOT_INDEX
    calibration_root_consumed: bool = CALIBRATION_ROOT_CONSUMED
    boundary_callbacks_used: bool = False
    resets_after_initialization: int = 0
    replay_capacity: int = 0
    current_rule_visible_to_learners: bool = False
    schedule_visible_to_learners: bool = False
    evaluator_birth_ledger_routed_to_learners: bool = False
    semantic_context_identity: str = SEMANTIC_CONTEXT_IDENTITY
    slot_indices_are_semantic_identities: bool = False
    cross_agent_birth_words_are_comparable_identities: bool = False

    @property
    def num_steps(self) -> int:
        return self.phase_length * len(self.offsets)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible protocol manifest."""

        return dataclasses.asdict(self)


PROTOCOL = CapacityPressureProtocol()


@dataclasses.dataclass(frozen=True)
class CalibrationRoot:
    """The only executable calibration root in this module."""

    namespace: str = DEVELOPMENT_NAMESPACE
    index: int = CALIBRATION_ROOT_INDEX
    key_seed: int = CALIBRATION_ROOT_INDEX
    consumed: bool = CALIBRATION_ROOT_CONSUMED
    environment_key_derivation: str = "jax.random.fold_in(jax.random.key(0), 0)"
    agent_0_key_derivation: str = "jax.random.fold_in(jax.random.key(0), 1)"
    agent_1_key_derivation: str = "jax.random.fold_in(jax.random.key(0), 2)"


CALIBRATION_ROOT = CalibrationRoot()


def control_config(epsilon: float) -> DifferentialSARSAConfig:
    """Return one exact predeclared control configuration."""

    if epsilon not in EPSILON_GRID:
        raise ValueError("epsilon is not in the predeclared consumed-root grid")
    return DifferentialSARSAConfig(
        n_actions=N_ACTIONS,
        q_step_size=0.15,
        average_reward_step_size=0.01,
        epsilon_start=epsilon,
        epsilon_end=epsilon,
        epsilon_decay_steps=0,
        use_bias=False,
    )


CONTEXT_CONFIG = ContextInferenceConfig(
    n_actions=N_ACTIONS,
    observation_dim=N_ACTIONS,
    max_contexts=MAX_CONTEXTS,
)
GAME_CONFIG = ConventionGameConfig(
    n_actions=N_ACTIONS,
    phase_length=PHASE_LENGTH,
    offsets=OFFSETS,
    feature_mode="plain",
)


@chex.dataclass(frozen=True)
class ContextBirthLedgerState:
    """Evaluator-only semantic births for recyclable context slots.

    Unused rows have no semantic meaning.  Their validity is always taken
    from the corresponding authenticated ``ContextInferenceState.in_use``.
    """

    slot_birth_words: UInt[Array, "max_contexts 2"]


@chex.dataclass(frozen=True)
class BirthRecurrenceHistoryState:
    """Past-only recurrence authority bound to each live semantic birth.

    Every row is exact two-word unsigned arithmetic.  ``occurrence_words``
    counts authenticated entries into the current birth; its protection score
    is therefore ``occurrences - 1``.  ``last_entry_words`` and
    ``last_interval_words`` retain the corresponding recurrence timing for
    audit, but no future-value target or schedule-derived quantity is stored.
    Unused rows are pinned to exact zero in every field.
    """

    bound_birth_words: UInt[Array, "max_contexts 2"]
    occurrence_words: UInt[Array, "max_contexts 2"]
    last_entry_words: UInt[Array, "max_contexts 2"]
    last_interval_words: UInt[Array, "max_contexts 2"]


@chex.dataclass(frozen=True)
class CapacityPressureState:
    """Complete atomic state of the game, learners, banks, and audit ledger."""

    environment: ConventionGameState
    controller_0: DifferentialSARSAState
    controller_1: DifferentialSARSAState
    context_0: ContextInferenceState
    context_1: ContextInferenceState
    ledger_0: ContextBirthLedgerState
    ledger_1: ContextBirthLedgerState


@chex.dataclass(frozen=True)
class SelectiveRetentionState:
    """Atomic base life plus both evaluator-only recurrence histories."""

    base: CapacityPressureState
    recurrence_0: BirthRecurrenceHistoryState
    recurrence_1: BirthRecurrenceHistoryState


@chex.dataclass(frozen=True)
class ContextLineageCacheState:
    """One agent's fixed-capacity cross-birth predictive-rescue ledger.

    ``bound_birth_words`` authenticates every live row against the context
    birth ledger.  ``live_lineage_words`` is the stable semantic origin carried
    across a strict predictive cache match, while ``live_rescue_words`` counts
    only such matches.  The single cache record is invalid exactly when all of
    its payload fields are zero.  Agent namespace remains an external identity
    component, so two agents never share a lineage merely because their word
    values happen to agree.
    """

    bound_birth_words: UInt[Array, "max_contexts 2"]
    live_lineage_words: UInt[Array, "max_contexts 2"]
    live_rescue_words: UInt[Array, "max_contexts 2"]
    cache_valid: Bool[Array, ""]
    cache_source_birth_words: UInt[Array, " 2"]
    cache_lineage_words: UInt[Array, " 2"]
    cache_rescue_words: UInt[Array, " 2"]
    cache_reward_weights: Float[Array, "n_actions observation_dim"]


@chex.dataclass(frozen=True)
class LineageCacheRetentionState:
    """Atomic base life plus two independent one-record lineage caches."""

    base: CapacityPressureState
    lineage_0: ContextLineageCacheState
    lineage_1: ContextLineageCacheState


@chex.dataclass(frozen=True)
class RecurrenceHistoryProposal:
    """One authenticated recurrence-history transaction."""

    state: BirthRecurrenceHistoryState
    source_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    occurrence_capacity_available: Bool[Array, ""]
    allocation_reset: Bool[Array, ""]
    stored_recurrence_recorded: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LineageCacheProposal:
    """One authenticated post-outcome lineage/cache transaction."""

    state: ContextLineageCacheState
    source_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    rescue_capacity_available: Bool[Array, ""]
    full_bank_birth: Bool[Array, ""]
    cache_tested: Bool[Array, ""]
    strict_predictive_dominance: Bool[Array, ""]
    cache_matched: Bool[Array, ""]
    lineage_transferred: Bool[Array, ""]
    rescue_incremented: Bool[Array, ""]
    victim_archived: Bool[Array, ""]
    old_cache_retained: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    cache_error: Float[Array, ""]
    fresh_error: Float[Array, ""]
    live_errors: Float[Array, " max_contexts"]


@chex.dataclass(frozen=True)
class CapacityPressureStepTrace:
    """One atomic transition's causal and lifecycle diagnostics."""

    reward: Float[Array, ""]
    actions: Int[Array, " 2"]
    pre_context_slots: Int[Array, " 2"]
    post_context_slots: Int[Array, " 2"]
    pre_context_birth_words: UInt[Array, "2 2"]
    post_context_birth_words: UInt[Array, "2 2"]
    switches: Bool[Array, " 2"]
    allocations: Bool[Array, " 2"]
    evictions: Bool[Array, " 2"]
    reuses: Bool[Array, " 2"]
    contexts_in_use: Int[Array, " 2"]
    environment_update_proposed: Bool[Array, ""]
    context_updates_proposed: Bool[Array, " 2"]
    controller_updates_proposed: Bool[Array, " 2"]
    source_clocks_aligned: Bool[Array, ""]
    candidate_clocks_aligned: Bool[Array, ""]
    source_state_finite: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    controller_rng_key_words: UInt[Array, "2 2"]
    controller_next_q_values: Float[Array, "2 n_actions"]
    controller_next_actions: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class CapacityPressureStepResult:
    """Atomic successor plus its evaluator trace."""

    state: CapacityPressureState
    trace: CapacityPressureStepTrace


@dataclasses.dataclass(frozen=True)
class CapacityPressureResourceBudget:
    """Exact persistent-array accounting for one epsilon arm."""

    environment_nbytes: int
    per_agent_controller_nbytes: int
    per_agent_context_nbytes: int
    joint_agent_environment_nbytes: int
    per_agent_evaluator_birth_ledger_nbytes: int
    joint_evaluator_birth_ledger_nbytes: int
    total_scan_carry_nbytes: int
    environment_exact_clock_nbytes: int
    per_agent_controller_clock_nbytes: int
    per_agent_context_clock_nbytes: int
    joint_agent_environment_clock_nbytes: int
    max_context_slots_per_agent: int
    replay_capacity: int
    fixed_shape: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return the exact resource record."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CapacityPressureWorkBudget:
    """Matched operation-count contract, independent of epsilon outcomes."""

    transitions: int
    environment_transition_proposals: int
    context_update_proposals: int
    controller_update_proposals: int
    context_event_audits: int
    atomic_commit_decisions: int
    action_selection_calls: int
    action_selection_calls_per_agent: int
    replay_updates: int
    reset_callbacks: int
    fixed_work_across_epsilon_grid: bool
    exploration_and_greedy_draws_both_generated: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return the fixed work record."""

        return dataclasses.asdict(self)


WORK_BUDGET = CapacityPressureWorkBudget(
    transitions=NUM_STEPS,
    environment_transition_proposals=NUM_STEPS,
    context_update_proposals=2 * NUM_STEPS,
    controller_update_proposals=2 * NUM_STEPS,
    context_event_audits=2 * NUM_STEPS,
    atomic_commit_decisions=NUM_STEPS,
    action_selection_calls=2 * (NUM_STEPS + 1),
    action_selection_calls_per_agent=NUM_STEPS + 1,
    replay_updates=0,
    reset_callbacks=0,
    fixed_work_across_epsilon_grid=True,
    exploration_and_greedy_draws_both_generated=True,
)


@dataclasses.dataclass(frozen=True)
class CapacityPressureRun:
    """One epsilon arm on consumed calibration root zero."""

    epsilon: float
    root: CalibrationRoot
    initial_controller_rng_key_words: Array
    trace: CapacityPressureStepTrace
    final_state: CapacityPressureState
    resource_budget: CapacityPressureResourceBudget
    work_budget: CapacityPressureWorkBudget


@dataclasses.dataclass(frozen=True)
class CapacityPressureSummary:
    """Threshold-free descriptive reward and semantic-lifecycle readout."""

    epsilon: float
    phase_early_reward: Array
    phase_tail_reward: Array
    tail_context_slot_modes: Array
    tail_context_birth_modes: Array
    phase_switch_counts: Array
    phase_allocation_counts: Array
    phase_eviction_counts: Array
    phase_reuse_counts: Array
    phase_distinct_birth_counts: Array
    final_label_birth_modes: Array
    final_abc_births_distinct: Array
    recurrence_birth_reuse: Array
    overall_reward: float


@dataclasses.dataclass(frozen=True)
class CommonRandomNumberAudit:
    """Exact controller-key-stream equality across all epsilon arms."""

    root_index: int
    agent_key_stream_sha256: tuple[str, str]
    key_streams_equal_across_arms: bool
    selection_calls_per_agent: int
    branch_independent_key_advance: bool
    environment_randomness_consumed: bool


@dataclasses.dataclass(frozen=True)
class PrefixTwinBoundary:
    """Executable no-free-lunch record for the first capacity eviction."""

    actual_schedule: tuple[int, ...]
    counterfactual_schedule: tuple[int, ...]
    common_prefix_phases: int
    common_prefix_offsets: tuple[int, ...]
    common_prefix_sha256: str
    actual_prefix_sha256: str
    counterfactual_prefix_sha256: str
    first_divergent_phase: int
    differing_phase_indices: tuple[int, ...]
    actual_divergent_offset: int
    counterfactual_divergent_offset: int
    regimes_seen_by_first_c_admission: tuple[str, ...]
    capacity: int
    actual_zero_recurrence_loss_set: tuple[str, ...]
    counterfactual_zero_recurrence_loss_set: tuple[str, ...]
    correct_actual_eviction: str
    correct_counterfactual_eviction: str
    same_policy_rng_implies_identical_prefix_history: bool
    future_schedule_only_divergence: bool
    deterministic_online_guarantee_possible: bool
    future_schedule_revealed_to_learners: bool
    counterfactual_future_executed: bool
    stochastic_optimality_claimed: bool
    conclusion: str


@dataclasses.dataclass(frozen=True)
class FrozenPanelDesign:
    """Unexecuted next-panel design; calibration root zero is excluded."""

    status: str
    root_zero_excluded: bool
    epsilon_arms: tuple[float, ...]
    primary_endpoints: tuple[str, ...]
    causal_diagnostics: tuple[str, ...]
    required_protocol_actions: tuple[str, ...]
    promotion_claimed: bool


@dataclasses.dataclass(frozen=True)
class CapacityPressurePanel:
    """The complete consumed-root grid with no selected arm."""

    runs: tuple[CapacityPressureRun, ...]
    summaries: tuple[CapacityPressureSummary, ...]
    common_random_numbers: CommonRandomNumberAudit
    prefix_twin_boundary: PrefixTwinBoundary
    resources_matched: bool
    work_matched: bool
    selection_performed: bool
    selected_epsilon: None
    causal_gap: str
    next_frozen_panel: FrozenPanelDesign


@chex.dataclass(frozen=True)
class ControllerScrubPreparation:
    """Authenticated, pre-SARSA controller candidate for one context birth."""

    state: DifferentialSARSAState
    scrub_required: Bool[Array, ""]
    scrub_candidate_applied: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    pre_bank_valid: Bool[Array, ""]
    post_bank_valid: Bool[Array, ""]
    pre_ledger_valid: Bool[Array, ""]
    post_ledger_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    controller_shape_valid: Bool[Array, ""]
    controller_source_finite: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    source_destination_separated: Bool[Array, ""]
    biases_zero_before: Bool[Array, ""]
    biases_untouched: Bool[Array, ""]
    average_reward_untouched: Bool[Array, ""]
    rng_untouched_before_update: Bool[Array, ""]
    clock_untouched_before_update: Bool[Array, ""]
    survivor_rows_untouched: Bool[Array, ""]
    prepared_controller_unchanged: Bool[Array, ""]
    source_slot: Int[Array, ""]
    destination_slot: Int[Array, ""]
    pre_destination_birth_words: UInt[Array, " 2"]
    post_destination_birth_words: UInt[Array, " 2"]
    pre_destination_q_weight_l1: Float[Array, ""]
    prepared_destination_q_weight_l1: Float[Array, ""]
    pre_destination_q_trace_l1: Float[Array, ""]
    prepared_destination_q_trace_l1: Float[Array, ""]
    stale_destination_q_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PostAuditScrubTrace:
    """Per-transition authentication, scrub, and contamination diagnostics."""

    scrub_enabled: Bool[Array, ""]
    scrub_required: Bool[Array, " 2"]
    scrub_applied: Bool[Array, " 2"]
    preparation_valid: Bool[Array, " 2"]
    authentication_failed: Bool[Array, " 2"]
    pre_bank_valid: Bool[Array, " 2"]
    post_bank_valid: Bool[Array, " 2"]
    pre_ledger_valid: Bool[Array, " 2"]
    post_ledger_valid: Bool[Array, " 2"]
    binding_valid: Bool[Array, " 2"]
    controller_shape_valid: Bool[Array, " 2"]
    controller_source_finite: Bool[Array, " 2"]
    candidate_finite: Bool[Array, " 2"]
    source_destination_separated: Bool[Array, " 2"]
    biases_zero_before: Bool[Array, " 2"]
    biases_untouched: Bool[Array, " 2"]
    average_reward_untouched: Bool[Array, " 2"]
    rng_untouched_before_update: Bool[Array, " 2"]
    clock_untouched_before_update: Bool[Array, " 2"]
    survivor_rows_untouched: Bool[Array, " 2"]
    prepared_controller_unchanged: Bool[Array, " 2"]
    source_slots: Int[Array, " 2"]
    destination_slots: Int[Array, " 2"]
    pre_destination_birth_words: UInt[Array, "2 2"]
    post_destination_birth_words: UInt[Array, "2 2"]
    pre_destination_q_weight_l1: Float[Array, " 2"]
    prepared_destination_q_weight_l1: Float[Array, " 2"]
    pre_destination_q_trace_l1: Float[Array, " 2"]
    prepared_destination_q_trace_l1: Float[Array, " 2"]
    cross_birth_contamination_available: Bool[Array, " 2"]
    cross_birth_contamination_consumed: Bool[Array, " 2"]
    cross_birth_contamination_prevented: Bool[Array, " 2"]
    scrubbed_parameter_scalars: Int[Array, " 2"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PostAuditStepResult:
    """Selected paired condition successor and both prepared controllers."""

    state: CapacityPressureState
    trace: CapacityPressureStepTrace
    scrub: PostAuditScrubTrace
    prepared_controllers: tuple[DifferentialSARSAState, DifferentialSARSAState]


@dataclasses.dataclass(frozen=True)
class PostAuditResourceBudget:
    """Persistent bytes for a post-audit condition."""

    base: CapacityPressureResourceBudget
    intervention_persistent_nbytes: int = 0
    logical_transient_scrub_candidate_nbytes: int = 0

    @property
    def total_persistent_nbytes(self) -> int:
        return self.base.total_scan_carry_nbytes + self.intervention_persistent_nbytes

    def to_dict(self) -> dict[str, object]:
        return {
            "base": self.base.to_dict(),
            "intervention_persistent_nbytes": self.intervention_persistent_nbytes,
            "logical_transient_scrub_candidate_nbytes": (
                self.logical_transient_scrub_candidate_nbytes
            ),
            "total_persistent_nbytes": self.total_persistent_nbytes,
        }


@dataclasses.dataclass(frozen=True)
class PostAuditWorkBudget:
    """Matched baseline/scrub work; both alternatives are always staged."""

    transitions: int = NUM_STEPS
    environment_transition_proposals: int = NUM_STEPS
    context_update_proposals: int = 2 * NUM_STEPS
    baseline_controller_updates: int = 2 * NUM_STEPS
    scrub_controller_updates: int = 2 * NUM_STEPS
    context_event_audits: int = 2 * NUM_STEPS
    birth_authentication_audits: int = 2 * NUM_STEPS
    scrub_candidate_constructions: int = 2 * NUM_STEPS
    action_selection_calls: int = 4 * NUM_STEPS + 2
    atomic_commit_decisions: int = 2 * NUM_STEPS
    replay_updates: int = 0
    reset_callbacks: int = 0
    fixed_work_across_conditions: bool = True
    branch_independent_rng_advance: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


POST_AUDIT_WORK_BUDGET = PostAuditWorkBudget()


@dataclasses.dataclass(frozen=True)
class PostAuditRun:
    """One selected baseline or scrub life plus exact scrub diagnostics."""

    condition: PostAuditCondition
    capacity_run: CapacityPressureRun
    scrub: PostAuditScrubTrace
    resource_budget: PostAuditResourceBudget
    work_budget: PostAuditWorkBudget

    @property
    def epsilon(self) -> float:
        return self.capacity_run.epsilon

    @property
    def trace(self) -> CapacityPressureStepTrace:
        return self.capacity_run.trace

    @property
    def final_state(self) -> CapacityPressureState:
        return self.capacity_run.final_state


@dataclasses.dataclass(frozen=True)
class PostAuditPair:
    """Matched baseline/scrub lives for one epsilon."""

    epsilon: float
    baseline: PostAuditRun
    scrub: PostAuditRun


@dataclasses.dataclass(frozen=True)
class PostAuditEffect:
    """Threshold-free paired outcome and lifecycle deltas for one epsilon."""

    epsilon: float
    baseline_overall_reward: float
    scrub_overall_reward: float
    scrub_minus_baseline_overall_reward: float
    baseline_phase_early_reward: Array
    scrub_phase_early_reward: Array
    baseline_phase_tail_reward: Array
    scrub_phase_tail_reward: Array
    baseline_tail_birth_modes: Array
    scrub_tail_birth_modes: Array
    baseline_phase_distinct_birth_counts: Array
    scrub_phase_distinct_birth_counts: Array
    baseline_switch_counts: Array
    scrub_switch_counts: Array
    baseline_allocation_counts: Array
    scrub_allocation_counts: Array
    baseline_eviction_counts: Array
    scrub_eviction_counts: Array
    baseline_reuse_counts: Array
    scrub_reuse_counts: Array
    baseline_scrub_count: int
    scrub_scrub_count: int
    baseline_contamination_count: int
    scrub_contamination_count: int
    scrub_prevented_count: int
    cross_birth_contamination_removed: bool


@dataclasses.dataclass(frozen=True)
class PostAuditCommonRandomNumberAudit:
    """Controller-key equality across every one of the eight paired lives."""

    root_index: int
    key_streams_equal_across_all_eight: bool
    branch_independent_key_advance: bool
    selection_calls_per_agent: int
    environment_randomness_consumed: bool


@dataclasses.dataclass(frozen=True)
class PostAuditPairedPanel:
    """Eight consumed-root lives; post-audit, descriptive, and winner-free."""

    runs: tuple[PostAuditRun, ...]
    pairs: tuple[PostAuditPair, ...]
    effects: tuple[PostAuditEffect, ...]
    baseline_calibration_panel: CapacityPressurePanel
    common_random_numbers: PostAuditCommonRandomNumberAudit
    resources_matched: bool
    work_matched: bool
    post_audit_only: bool
    scientific_promotion_allowed: bool
    thresholds_used: bool
    selection_performed: bool
    selected_epsilon: None
    conclusion: str


@chex.dataclass(frozen=True)
class SelectiveRetentionTrace:
    """Causal trace for one past-only selective-retention transition."""

    capacity: CapacityPressureStepTrace
    scrub: PostAuditScrubTrace
    protection_enabled: Bool[Array, ""]
    raw_completed_recurrence_scores: Float[Array, "2 max_contexts"]
    dispatched_eviction_protection: Float[Array, "2 max_contexts"]
    pre_occurrence_words: UInt[Array, "2 max_contexts 2"]
    post_occurrence_words: UInt[Array, "2 max_contexts 2"]
    pre_last_entry_words: UInt[Array, "2 max_contexts 2"]
    post_last_entry_words: UInt[Array, "2 max_contexts 2"]
    post_last_interval_words: UInt[Array, "2 max_contexts 2"]
    history_source_valid: Bool[Array, " 2"]
    history_candidate_valid: Bool[Array, " 2"]
    history_capacity_available: Bool[Array, " 2"]
    history_allocation_resets: Bool[Array, " 2"]
    history_stored_recurrences: Bool[Array, " 2"]
    history_updates_proposed: Bool[Array, " 2"]
    priority_inputs_valid: Bool[Array, " 2"]
    full_bank_evictions_requested: Bool[Array, " 2"]
    eviction_protection_used: Bool[Array, " 2"]
    eviction_targets_adjusted: Bool[Array, " 2"]
    ordinary_lru_slots: Int[Array, " 2"]
    protected_lru_slots: Int[Array, " 2"]
    selected_eviction_slots: Int[Array, " 2"]
    ordinary_lru_completed_recurrence_scores: Float[Array, " 2"]
    selected_completed_recurrence_scores: Float[Array, " 2"]
    selected_eviction_scores: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class SelectiveRetentionStepResult:
    """Atomic selective-retention successor and causal diagnostics."""

    state: SelectiveRetentionState
    trace: SelectiveRetentionTrace


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionResourceBudget:
    """Exact matched persistent and logical transient storage."""

    base: CapacityPressureResourceBudget
    per_agent_recurrence_history_nbytes: int
    joint_recurrence_history_nbytes: int
    total_scan_carry_nbytes: int
    logical_transient_protection_nbytes: int
    logical_transient_scrub_candidate_nbytes: int
    replay_capacity: int = 0
    fixed_shape: bool = True

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionWorkBudget:
    """Matched work: both conditions compute the same past-only history."""

    transitions: int = NUM_STEPS
    environment_transition_proposals: int = NUM_STEPS
    prioritized_context_update_proposals: int = 2 * NUM_STEPS
    recurrence_score_computations: int = 2 * NUM_STEPS
    recurrence_history_audits: int = 2 * NUM_STEPS
    recurrence_history_proposals: int = 2 * NUM_STEPS
    birth_authentication_audits: int = 2 * NUM_STEPS
    scrub_candidate_constructions: int = 2 * NUM_STEPS
    controller_update_proposals: int = 2 * NUM_STEPS
    action_selection_calls: int = 2 * (NUM_STEPS + 1)
    atomic_commit_decisions: int = NUM_STEPS
    replay_updates: int = 0
    reset_callbacks: int = 0
    fixed_work_across_conditions: bool = True
    no_signal_still_computes_history: bool = True
    branch_independent_rng_advance: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


SELECTIVE_RETENTION_WORK_BUDGET = SelectiveRetentionWorkBudget()


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionRun:
    """One no-signal or past-recurrence consumed-root life."""

    condition: SelectiveRetentionCondition
    capacity_run: CapacityPressureRun
    trace: SelectiveRetentionTrace
    final_state: SelectiveRetentionState
    resource_budget: SelectiveRetentionResourceBudget
    work_budget: SelectiveRetentionWorkBudget

    @property
    def epsilon(self) -> float:
        return self.capacity_run.epsilon


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionPair:
    """Matched no-signal/signal lives for one epsilon."""

    epsilon: float
    no_signal: SelectiveRetentionRun
    past_recurrence: SelectiveRetentionRun


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionEffect:
    """Threshold-free paired reward, lifecycle, and intervention deltas."""

    epsilon: float
    no_signal_overall_reward: float
    past_recurrence_overall_reward: float
    past_recurrence_minus_no_signal_overall_reward: float
    no_signal_phase_early_reward: Array
    past_recurrence_phase_early_reward: Array
    no_signal_phase_tail_reward: Array
    past_recurrence_phase_tail_reward: Array
    no_signal_tail_birth_modes: Array
    past_recurrence_tail_birth_modes: Array
    no_signal_phase_switch_counts: Array
    past_recurrence_phase_switch_counts: Array
    no_signal_phase_allocation_counts: Array
    past_recurrence_phase_allocation_counts: Array
    no_signal_phase_eviction_counts: Array
    past_recurrence_phase_eviction_counts: Array
    no_signal_phase_reuse_counts: Array
    past_recurrence_phase_reuse_counts: Array
    no_signal_full_bank_eviction_count: int
    past_recurrence_full_bank_eviction_count: int
    no_signal_adjusted_target_count: int
    past_recurrence_adjusted_target_count: int
    nonzero_selected_eviction_score_count: int
    avoided_completed_recurrence_intervals: float


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionCommonRandomNumberAudit:
    """Exact key-stream equality across the eight selective-retention lives."""

    root_index: int
    key_streams_equal_across_all_eight: bool
    branch_independent_key_advance: bool
    selection_calls_per_agent: int
    environment_randomness_consumed: bool


@dataclasses.dataclass(frozen=True)
class SelectiveRetentionPairedPanel:
    """Eight development-only consumed-root lives with no selected winner."""

    runs: tuple[SelectiveRetentionRun, ...]
    pairs: tuple[SelectiveRetentionPair, ...]
    effects: tuple[SelectiveRetentionEffect, ...]
    common_random_numbers: SelectiveRetentionCommonRandomNumberAudit
    resources_matched: bool
    work_matched: bool
    no_signal_is_controller_scrub_baseline: bool
    past_only_score: str
    prefix_twin_first_eviction_resolved: bool
    development_only: bool
    scientific_promotion_allowed: bool
    thresholds_used: bool
    selection_performed: bool
    selected_epsilon: None
    conclusion: str


@chex.dataclass(frozen=True)
class LineageCacheRetentionTrace:
    """Causal trace for one post-outcome lineage-cache transition."""

    capacity: CapacityPressureStepTrace
    protection_enabled: Bool[Array, ""]
    source_scores_fixed_before_outcome: Bool[Array, ""]
    outcome_routed_to_current_protection: Bool[Array, ""]
    raw_predictive_rescue_scores: Float[Array, "2 max_contexts"]
    dispatched_eviction_protection: Float[Array, "2 max_contexts"]
    pre_live_lineage_words: UInt[Array, "2 max_contexts 2"]
    post_live_lineage_words: UInt[Array, "2 max_contexts 2"]
    pre_live_rescue_words: UInt[Array, "2 max_contexts 2"]
    post_live_rescue_words: UInt[Array, "2 max_contexts 2"]
    pre_cache_valid: Bool[Array, " 2"]
    post_cache_valid: Bool[Array, " 2"]
    pre_cache_source_birth_words: UInt[Array, "2 2"]
    post_cache_source_birth_words: UInt[Array, "2 2"]
    pre_cache_lineage_words: UInt[Array, "2 2"]
    post_cache_lineage_words: UInt[Array, "2 2"]
    pre_cache_rescue_words: UInt[Array, "2 2"]
    post_cache_rescue_words: UInt[Array, "2 2"]
    cache_tested: Bool[Array, " 2"]
    strict_predictive_dominance: Bool[Array, " 2"]
    cache_matched: Bool[Array, " 2"]
    lineage_transferred: Bool[Array, " 2"]
    rescue_incremented: Bool[Array, " 2"]
    victim_archived: Bool[Array, " 2"]
    old_cache_retained: Bool[Array, " 2"]
    lineage_source_valid: Bool[Array, " 2"]
    lineage_candidate_valid: Bool[Array, " 2"]
    rescue_capacity_available: Bool[Array, " 2"]
    lineage_updates_proposed: Bool[Array, " 2"]
    cache_errors: Float[Array, " 2"]
    fresh_errors: Float[Array, " 2"]
    live_errors: Float[Array, "2 max_contexts"]
    scrub_preparations_valid: Bool[Array, " 2"]
    scrub_applied: Bool[Array, " 2"]
    full_bank_evictions_requested: Bool[Array, " 2"]
    eviction_protection_used: Bool[Array, " 2"]
    eviction_targets_adjusted: Bool[Array, " 2"]
    ordinary_lru_slots: Int[Array, " 2"]
    protected_lru_slots: Int[Array, " 2"]
    selected_eviction_slots: Int[Array, " 2"]
    ordinary_lru_predictive_rescue_scores: Float[Array, " 2"]
    selected_predictive_rescue_scores: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class LineageCacheRetentionStepResult:
    """Atomic successor plus causal lineage-cache diagnostics."""

    state: LineageCacheRetentionState
    trace: LineageCacheRetentionTrace


@dataclasses.dataclass(frozen=True)
class LineageCacheResourceBudget:
    """Exact matched persistent and logical transient storage."""

    base: CapacityPressureResourceBudget
    cache_capacity_per_agent: int
    per_agent_lineage_cache_nbytes: int
    joint_lineage_cache_nbytes: int
    total_scan_carry_nbytes: int
    logical_transient_protection_nbytes: int
    logical_transient_prediction_nbytes: int
    logical_transient_lineage_candidate_nbytes: int
    logical_transient_scrub_candidate_nbytes: int
    replay_capacity: int = 0
    fixed_shape: bool = True

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LineageCacheWorkBudget:
    """Matched work for source scoring and post-outcome cache matching."""

    transitions: int = NUM_STEPS
    environment_transition_proposals: int = NUM_STEPS
    prioritized_context_update_proposals: int = 2 * NUM_STEPS
    source_rescue_score_computations: int = 2 * NUM_STEPS
    cache_match_computations: int = 2 * NUM_STEPS
    live_model_error_comparisons: int = 2 * MAX_CONTEXTS * NUM_STEPS
    lineage_cache_audits: int = 2 * NUM_STEPS
    lineage_cache_proposals: int = 2 * NUM_STEPS
    birth_authentication_audits: int = 2 * NUM_STEPS
    scrub_candidate_constructions: int = 2 * NUM_STEPS
    controller_update_proposals: int = 2 * NUM_STEPS
    action_selection_calls: int = 2 * (NUM_STEPS + 1)
    atomic_commit_decisions: int = NUM_STEPS
    replay_updates: int = 0
    reset_callbacks: int = 0
    fixed_work_across_conditions: bool = True
    no_signal_still_computes_cache_match: bool = True
    branch_independent_rng_advance: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


LINEAGE_CACHE_WORK_BUDGET = LineageCacheWorkBudget()


@dataclasses.dataclass(frozen=True)
class LineageCacheRun:
    """One no-signal or predictive-rescue consumed-root life."""

    condition: LineageCacheCondition
    capacity_run: CapacityPressureRun
    trace: LineageCacheRetentionTrace
    final_state: LineageCacheRetentionState
    resource_budget: LineageCacheResourceBudget
    work_budget: LineageCacheWorkBudget

    @property
    def epsilon(self) -> float:
        return self.capacity_run.epsilon


@dataclasses.dataclass(frozen=True)
class LineageCachePair:
    """Matched no-signal/predictive-rescue lives for one epsilon."""

    epsilon: float
    no_signal: LineageCacheRun
    predictive_rescue: LineageCacheRun


@dataclasses.dataclass(frozen=True)
class PredictiveRescueFailureDecomposition:
    """Exact descriptive reasons the strict one-transition match abstained."""

    full_bank_birth_count: int
    cache_valid_test_count: int
    failed_fresh_prior_count: int
    failed_any_live_model_count: int
    fresh_prior_tie_count: int
    any_live_model_tie_count: int
    any_exact_tie_count: int
    victim_archive_count: int
    old_cache_retained_count: int


@dataclasses.dataclass(frozen=True)
class LineageCacheEffect:
    """Threshold-free reward, lifecycle, match, and eviction deltas."""

    epsilon: float
    no_signal_overall_reward: float
    predictive_rescue_overall_reward: float
    predictive_rescue_minus_no_signal_overall_reward: float
    no_signal_phase_early_reward: Array
    predictive_rescue_phase_early_reward: Array
    no_signal_phase_tail_reward: Array
    predictive_rescue_phase_tail_reward: Array
    no_signal_phase_switch_counts: Array
    predictive_rescue_phase_switch_counts: Array
    no_signal_phase_allocation_counts: Array
    predictive_rescue_phase_allocation_counts: Array
    no_signal_phase_eviction_counts: Array
    predictive_rescue_phase_eviction_counts: Array
    no_signal_phase_reuse_counts: Array
    predictive_rescue_phase_reuse_counts: Array
    no_signal_cache_match_count: int
    predictive_rescue_cache_match_count: int
    no_signal_rescue_increment_count: int
    predictive_rescue_rescue_increment_count: int
    no_signal_adjusted_target_count: int
    predictive_rescue_adjusted_target_count: int
    nonzero_selected_predictive_rescue_count: int
    avoided_predictive_rescues: float
    no_signal_failure_decomposition: PredictiveRescueFailureDecomposition
    predictive_rescue_failure_decomposition: PredictiveRescueFailureDecomposition


@dataclasses.dataclass(frozen=True)
class LineageCacheCommonRandomNumberAudit:
    """Exact key-stream equality across all eight lineage-cache lives."""

    root_index: int
    key_streams_equal_across_all_eight: bool
    branch_independent_key_advance: bool
    selection_calls_per_agent: int
    environment_randomness_consumed: bool


@dataclasses.dataclass(frozen=True)
class LineageCachePairedPanel:
    """Eight consumed-root cache lives with no threshold or selected winner."""

    runs: tuple[LineageCacheRun, ...]
    pairs: tuple[LineageCachePair, ...]
    effects: tuple[LineageCacheEffect, ...]
    common_random_numbers: LineageCacheCommonRandomNumberAudit
    resources_matched: bool
    work_matched: bool
    no_signal_is_controller_scrub_baseline: bool
    cache_capacity_per_agent: int
    task_labels_used: bool
    configurable_match_threshold_used: bool
    score_source: str
    prefix_twin_first_eviction_resolved: bool
    development_only: bool
    scientific_promotion_allowed: bool
    selection_performed: bool
    selected_epsilon: None
    conclusion: str


def _tree_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(value))
    return valid


def _initial_controller(
    agent: DifferentialSARSAAgent,
    observation: Array,
    key: Array,
) -> DifferentialSARSAState:
    state = agent.init(observation.shape[0], key)
    state = cast(
        DifferentialSARSAState,
        state.replace(birth_timestamp=0.0, uptime_s=0.0),  # type: ignore[attr-defined]
    )
    state, _ = agent.start(state, observation)
    return cast(DifferentialSARSAState, state)


def _initial_ledger() -> ContextBirthLedgerState:
    return ContextBirthLedgerState(
        slot_birth_words=jnp.zeros((MAX_CONTEXTS, 2), dtype=jnp.uint32)
    )


def _initial_recurrence_history() -> BirthRecurrenceHistoryState:
    zeros = jnp.zeros((MAX_CONTEXTS, 2), dtype=jnp.uint32)
    occurrence = zeros.at[0, 1].set(jnp.asarray(1, dtype=jnp.uint32))
    return BirthRecurrenceHistoryState(
        bound_birth_words=zeros,
        occurrence_words=occurrence,
        last_entry_words=zeros,
        last_interval_words=zeros,
    )


def _initial_lineage_cache() -> ContextLineageCacheState:
    birth_rows = jnp.zeros((MAX_CONTEXTS, 2), dtype=jnp.uint32)
    return ContextLineageCacheState(
        bound_birth_words=birth_rows,
        live_lineage_words=birth_rows,
        live_rescue_words=birth_rows,
        cache_valid=jnp.asarray(False, dtype=jnp.bool_),
        cache_source_birth_words=jnp.zeros((2,), dtype=jnp.uint32),
        cache_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        cache_rescue_words=jnp.zeros((2,), dtype=jnp.uint32),
        cache_reward_weights=jnp.zeros(
            (N_ACTIONS, CONTEXT_CONFIG.observation_dim),
            dtype=jnp.float32,
        ),
    )


def initialize_capacity_pressure_state(epsilon: float) -> CapacityPressureState:
    """Initialize the only allowed root, without advancing the life."""

    agent = DifferentialSARSAAgent(control_config(epsilon))
    context = ContextInference(CONTEXT_CONFIG)
    game = RecurringConventionGame(GAME_CONFIG)
    root = jr.key(CALIBRATION_ROOT_INDEX)
    context_0 = context.init()
    context_1 = context.init()
    controller_0 = _initial_controller(
        agent,
        context.context_onehot(context_0),
        jr.fold_in(root, 1),
    )
    controller_1 = _initial_controller(
        agent,
        context.context_onehot(context_1),
        jr.fold_in(root, 2),
    )
    return CapacityPressureState(
        environment=game.init(jr.fold_in(root, 0)),
        controller_0=controller_0,
        controller_1=controller_1,
        context_0=context_0,
        context_1=context_1,
        ledger_0=_initial_ledger(),
        ledger_1=_initial_ledger(),
    )


def initialize_selective_retention_state(epsilon: float) -> SelectiveRetentionState:
    """Initialize the fixed consumed-root life and exact past-only histories."""

    return SelectiveRetentionState(
        base=initialize_capacity_pressure_state(epsilon),
        recurrence_0=_initial_recurrence_history(),
        recurrence_1=_initial_recurrence_history(),
    )


def initialize_lineage_cache_retention_state(
    epsilon: float,
) -> LineageCacheRetentionState:
    """Initialize the fixed life and two empty one-record lineage caches."""

    return LineageCacheRetentionState(
        base=initialize_capacity_pressure_state(epsilon),
        lineage_0=_initial_lineage_cache(),
        lineage_1=_initial_lineage_cache(),
    )


def _clocks_aligned(state: CapacityPressureState) -> Bool[Array, ""]:
    words = state.environment.step_words
    return (
        jnp.all(words == state.controller_0.step_words)
        & jnp.all(words == state.controller_1.step_words)
        & jnp.all(words == state.context_0.step_words)
        & jnp.all(words == state.context_1.step_words)
    )


def _context_event(
    context: ContextInference,
    source: ContextInferenceState,
    result: ContextInferenceUpdateResult | ContextInferencePrioritizedUpdateResult,
    observation: Array,
    action: Array,
    reward: Array,
) -> tuple[Array, Array, Array, Array]:
    """Reconstruct switch/allocation/eviction/reuse from authenticated input.

    This is an evaluator-only copy of the allocation *predicate*, not another
    decision-maker.  The core result remains the sole authority for the
    successor and target slot.  Reconstructing the predicate is necessary to
    distinguish stored-slot reuse from full-bank replacement, both of which
    leave ``in_use`` true at the target.
    """

    cfg = context.config
    safe_action = jnp.clip(jnp.asarray(action, dtype=jnp.int32), 0, cfg.n_actions - 1)
    safe_observation = jnp.where(
        jnp.isfinite(observation),
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.zeros_like(observation, dtype=jnp.float32),
    )
    safe_reward = jnp.where(
        jnp.isfinite(reward),
        jnp.asarray(reward, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    predictions = source.reward_weights[:, safe_action, :] @ safe_observation
    errors = jnp.abs(safe_reward - predictions)
    error_ema = jnp.float32(cfg.error_decay) * source.error_ema + (
        1.0 - jnp.float32(cfg.error_decay)
    ) * errors
    error_ema = jnp.where(
        source.in_use,
        error_ema,
        jnp.float32(cfg.novelty_prior_error),
    )
    slots = jnp.arange(cfg.max_contexts, dtype=jnp.int32)
    stored_scores = jnp.where(
        (slots == source.active_context) | ~source.in_use,
        jnp.inf,
        error_ema,
    )
    allocation_predicate = (
        stored_scores[jnp.argmin(stored_scores)] > cfg.novelty_prior_error
    )
    switched = (
        result.update_applied
        & (result.state.active_context != source.active_context)
    )
    allocated = switched & allocation_predicate
    target = jnp.clip(result.state.active_context, 0, cfg.max_contexts - 1)
    evicted = allocated & source.in_use[target]
    reused = switched & ~allocated
    return switched, allocated, evicted, reused


def _propose_ledger(
    ledger: ContextBirthLedgerState,
    result: ContextInferenceUpdateResult | ContextInferencePrioritizedUpdateResult,
    allocated: Array,
) -> ContextBirthLedgerState:
    target = jnp.clip(result.state.active_context, 0, MAX_CONTEXTS - 1)
    proposed = ledger.slot_birth_words.at[target].set(result.post_step_words)
    return ContextBirthLedgerState(
        slot_birth_words=jnp.where(allocated, proposed, ledger.slot_birth_words)
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _step_capacity_pressure(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    state: CapacityPressureState,
) -> CapacityPressureStepResult:
    """Stage and atomically commit one oracle-free joint transition."""

    source_clocks_aligned = _clocks_aligned(state)
    source_state_finite = _tree_finite(state)
    actions = jnp.stack(
        (state.controller_0.last_action, state.controller_1.last_action)
    ).astype(jnp.int32)
    pre_context_slots = jnp.stack(
        (state.context_0.active_context, state.context_1.active_context)
    ).astype(jnp.int32)
    pre_context_birth_words = jnp.stack(
        (
            state.ledger_0.slot_birth_words[state.context_0.active_context],
            state.ledger_1.slot_birth_words[state.context_1.active_context],
        )
    ).astype(jnp.uint32)

    # Both actions are fixed before either bank consumes the partner action or
    # reward.  No rule, phase, observation, schedule, or clock query occurs in
    # this learner transaction.
    environment_result = game.step_result(
        state.environment,
        actions[0],
        actions[1],
    )
    context_result_0 = context.update_result(
        state.context_0,
        jax.nn.one_hot(actions[1], N_ACTIONS, dtype=jnp.float32),
        actions[0],
        environment_result.reward,
    )
    context_result_1 = context.update_result(
        state.context_1,
        jax.nn.one_hot(actions[0], N_ACTIONS, dtype=jnp.float32),
        actions[1],
        environment_result.reward,
    )
    controller_result_0 = agent.update(
        state.controller_0,
        environment_result.reward,
        context_result_0.context_onehot,
    )
    controller_result_1 = agent.update(
        state.controller_1,
        environment_result.reward,
        context_result_1.context_onehot,
    )

    event_0 = _context_event(
        context,
        state.context_0,
        context_result_0,
        jax.nn.one_hot(actions[1], N_ACTIONS, dtype=jnp.float32),
        actions[0],
        environment_result.reward,
    )
    event_1 = _context_event(
        context,
        state.context_1,
        context_result_1,
        jax.nn.one_hot(actions[0], N_ACTIONS, dtype=jnp.float32),
        actions[1],
        environment_result.reward,
    )
    proposed_ledger_0 = _propose_ledger(
        state.ledger_0,
        context_result_0,
        event_0[1],
    )
    proposed_ledger_1 = _propose_ledger(
        state.ledger_1,
        context_result_1,
        event_1[1],
    )
    candidate_state = CapacityPressureState(
        environment=environment_result.state,
        controller_0=controller_result_0.state,
        controller_1=controller_result_1.state,
        context_0=context_result_0.state,
        context_1=context_result_1.state,
        ledger_0=proposed_ledger_0,
        ledger_1=proposed_ledger_1,
    )
    candidate_clocks_aligned = _clocks_aligned(candidate_state)
    candidate_state_finite = _tree_finite(candidate_state)
    context_updates_proposed = jnp.stack(
        (context_result_0.update_applied, context_result_1.update_applied)
    ).astype(jnp.bool_)
    controller_updates_proposed = jnp.stack(
        (controller_result_0.update_applied, controller_result_1.update_applied)
    ).astype(jnp.bool_)
    source_children_valid = (
        environment_result.state_valid
        & context_result_0.source_state_valid
        & context_result_1.source_state_valid
        & controller_result_0.state_valid
        & controller_result_1.state_valid
    )
    candidate_children_valid = (
        context_result_0.candidate_state_valid
        & context_result_1.candidate_state_valid
        & controller_result_0.candidate_state_finite
        & controller_result_1.candidate_state_finite
    )
    update_applied = (
        source_clocks_aligned
        & source_state_finite
        & source_children_valid
        & environment_result.update_applied
        & jnp.all(context_updates_proposed)
        & jnp.all(controller_updates_proposed)
        & candidate_clocks_aligned
        & candidate_state_finite
        & candidate_children_valid
    )
    committed_state = jax.lax.cond(
        update_applied,
        lambda _: candidate_state,
        lambda _: state,
        operand=None,
    )
    switches = jnp.stack((event_0[0], event_1[0])) & update_applied
    allocations = jnp.stack((event_0[1], event_1[1])) & update_applied
    evictions = jnp.stack((event_0[2], event_1[2])) & update_applied
    reuses = jnp.stack((event_0[3], event_1[3])) & update_applied
    post_context_slots = jnp.stack(
        (
            committed_state.context_0.active_context,
            committed_state.context_1.active_context,
        )
    ).astype(jnp.int32)
    post_context_birth_words = jnp.stack(
        (
            committed_state.ledger_0.slot_birth_words[
                committed_state.context_0.active_context
            ],
            committed_state.ledger_1.slot_birth_words[
                committed_state.context_1.active_context
            ],
        )
    ).astype(jnp.uint32)
    trace = CapacityPressureStepTrace(
        reward=jnp.where(
            update_applied,
            environment_result.reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        actions=actions,
        pre_context_slots=pre_context_slots,
        post_context_slots=post_context_slots,
        pre_context_birth_words=pre_context_birth_words,
        post_context_birth_words=post_context_birth_words,
        switches=switches,
        allocations=allocations,
        evictions=evictions,
        reuses=reuses,
        contexts_in_use=jnp.stack(
            (
                context.num_contexts_in_use(committed_state.context_0),
                context.num_contexts_in_use(committed_state.context_1),
            )
        ).astype(jnp.int32),
        environment_update_proposed=environment_result.update_applied,
        context_updates_proposed=context_updates_proposed,
        controller_updates_proposed=controller_updates_proposed,
        source_clocks_aligned=source_clocks_aligned,
        candidate_clocks_aligned=candidate_clocks_aligned,
        source_state_finite=source_state_finite,
        candidate_state_finite=candidate_state_finite,
        update_applied=update_applied,
        pre_step_words=state.environment.step_words,
        post_step_words=committed_state.environment.step_words,
        controller_rng_key_words=jnp.stack(
            (
                jr.key_data(committed_state.controller_0.rng_key),
                jr.key_data(committed_state.controller_1.rng_key),
            )
        ).astype(jnp.uint32),
        controller_next_q_values=jnp.stack(
            (controller_result_0.q_values, controller_result_1.q_values)
        ).astype(jnp.float32),
        controller_next_actions=jnp.stack(
            (controller_result_0.action, controller_result_1.action)
        ).astype(jnp.int32),
    )
    return CapacityPressureStepResult(state=committed_state, trace=trace)


def _words_le(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] <= right[..., 1])
    )


def _words_lt(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] < right[..., 1])
    )


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_subtract(left: Array, right: Array) -> tuple[Array, Array]:
    """Return exact ``left - right`` and whether the subtraction was valid."""

    ordered = _words_le(right, left)
    borrow = (left[1] < right[1]).astype(jnp.uint32)
    difference = jnp.stack(
        (
            left[0] - right[0] - borrow,
            left[1] - right[1],
        )
    ).astype(jnp.uint32)
    return jnp.where(ordered, difference, jnp.zeros((2,), dtype=jnp.uint32)), ordered


def _recurrence_history_static_shape_valid(
    history: BirthRecurrenceHistoryState,
) -> bool:
    return all(
        jnp.asarray(getattr(history, name)).shape == (MAX_CONTEXTS, 2)
        and jnp.asarray(getattr(history, name)).dtype == jnp.dtype(jnp.uint32)
        for name in (
            "bound_birth_words",
            "occurrence_words",
            "last_entry_words",
            "last_interval_words",
        )
    )


def _recurrence_history_valid(
    context: ContextInference,
    context_state: ContextInferenceState,
    ledger: ContextBirthLedgerState,
    history: BirthRecurrenceHistoryState,
) -> Bool[Array, ""]:
    """Authenticate one fixed-life history against its semantic births."""

    if not _recurrence_history_static_shape_valid(history):
        return jnp.asarray(False, dtype=jnp.bool_)
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    zero_rows = jnp.all(history.occurrence_words == zero, axis=1)
    one_rows = jnp.all(
        history.occurrence_words
        == jnp.asarray((0, 1), dtype=jnp.uint32),
        axis=1,
    )
    interval_zero = jnp.all(history.last_interval_words == zero, axis=1)
    bound = jnp.all(history.bound_birth_words == ledger.slot_birth_words, axis=1)
    count_within_fixed_life = (
        history.occurrence_words[:, 0] == jnp.asarray(0, dtype=jnp.uint32)
    ) & (
        history.occurrence_words[:, 1]
        <= jnp.asarray(NUM_STEPS + 1, dtype=jnp.uint32)
    )
    birth_before_entry = _words_le(
        history.bound_birth_words,
        history.last_entry_words,
    )
    entry_not_future = _words_le(
        history.last_entry_words,
        context_state.step_words,
    )
    interval_within_entry = _words_le(
        history.last_interval_words,
        history.last_entry_words,
    )
    live_rows_valid = (
        ~zero_rows
        & bound
        & count_within_fixed_life
        & birth_before_entry
        & entry_not_future
        & interval_within_entry
        & jnp.where(
            one_rows,
            jnp.all(
                history.last_entry_words == history.bound_birth_words,
                axis=1,
            )
            & interval_zero,
            ~interval_zero,
        )
    )
    unused_rows_zero = (
        jnp.all(history.bound_birth_words == zero, axis=1)
        & zero_rows
        & jnp.all(history.last_entry_words == zero, axis=1)
        & interval_zero
    )
    return (
        _birth_ledger_valid(context, context_state, ledger)
        & jnp.all(jnp.where(context_state.in_use, live_rows_valid, unused_rows_zero))
    )


def _completed_recurrence_scores(
    history: BirthRecurrenceHistoryState,
) -> Float[Array, " max_contexts"]:
    """Project exact fixed-life counts to finite ``occurrences - 1`` scores."""

    bounded_count = jnp.where(
        history.occurrence_words[:, 0] == jnp.asarray(0, dtype=jnp.uint32),
        jnp.minimum(
            history.occurrence_words[:, 1],
            jnp.asarray(NUM_STEPS + 1, dtype=jnp.uint32),
        ),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    completed = jnp.where(
        bounded_count > jnp.asarray(0, dtype=jnp.uint32),
        bounded_count - jnp.asarray(1, dtype=jnp.uint32),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    return completed.astype(jnp.float32)


def _propose_recurrence_history(
    context: ContextInference,
    source_context: ContextInferenceState,
    post_context: ContextInferenceState,
    source_ledger: ContextBirthLedgerState,
    post_ledger: ContextBirthLedgerState,
    history: BirthRecurrenceHistoryState,
    allocated: Array,
    reused: Array,
    post_step_words: Array,
    context_update_applied: Array,
) -> RecurrenceHistoryProposal:
    """Reset on semantic birth and increment only authenticated stored reuse."""

    source_valid = _recurrence_history_valid(
        context,
        source_context,
        source_ledger,
        history,
    )
    target = jnp.clip(post_context.active_context, 0, MAX_CONTEXTS - 1)
    incremented_count, occurrence_capacity_available = _checked_words_increment(
        history.occurrence_words[target]
    )
    recurrence_interval, interval_ordered = _words_subtract(
        post_step_words,
        history.last_entry_words[target],
    )
    reset_count = jnp.asarray((0, 1), dtype=jnp.uint32)
    allocation_birth = post_ledger.slot_birth_words[target]
    bound_births = jnp.where(
        allocated,
        history.bound_birth_words.at[target].set(allocation_birth),
        history.bound_birth_words,
    )
    occurrence_words = jnp.where(
        allocated,
        history.occurrence_words.at[target].set(reset_count),
        jnp.where(
            reused,
            history.occurrence_words.at[target].set(incremented_count),
            history.occurrence_words,
        ),
    )
    last_entry_words = jnp.where(
        allocated | reused,
        history.last_entry_words.at[target].set(post_step_words),
        history.last_entry_words,
    )
    last_interval_words = jnp.where(
        allocated,
        history.last_interval_words.at[target].set(
            jnp.zeros((2,), dtype=jnp.uint32)
        ),
        jnp.where(
            reused,
            history.last_interval_words.at[target].set(recurrence_interval),
            history.last_interval_words,
        ),
    )
    candidate = BirthRecurrenceHistoryState(
        bound_birth_words=bound_births,
        occurrence_words=occurrence_words,
        last_entry_words=last_entry_words,
        last_interval_words=last_interval_words,
    )
    capacity_available = ~reused | (occurrence_capacity_available & interval_ordered)
    candidate_valid = _recurrence_history_valid(
        context,
        post_context,
        post_ledger,
        candidate,
    )
    update_applied = (
        source_valid
        & context_update_applied
        & capacity_available
        & candidate_valid
    )
    committed = jax.tree_util.tree_map(
        lambda proposed, current: jnp.where(update_applied, proposed, current),
        candidate,
        history,
    )
    return RecurrenceHistoryProposal(
        state=committed,
        source_valid=source_valid,
        candidate_valid=candidate_valid,
        occurrence_capacity_available=capacity_available,
        allocation_reset=allocated & update_applied,
        stored_recurrence_recorded=reused & update_applied,
        update_applied=update_applied,
    )


def _lineage_cache_static_shape_valid(state: ContextLineageCacheState) -> bool:
    word_matrix_fields = (
        state.bound_birth_words,
        state.live_lineage_words,
        state.live_rescue_words,
    )
    word_vector_fields = (
        state.cache_source_birth_words,
        state.cache_lineage_words,
        state.cache_rescue_words,
    )
    return (
        all(
            jnp.asarray(value).shape == (MAX_CONTEXTS, 2)
            and jnp.asarray(value).dtype == jnp.dtype(jnp.uint32)
            for value in word_matrix_fields
        )
        and jnp.asarray(state.cache_valid).shape == ()
        and jnp.asarray(state.cache_valid).dtype == jnp.dtype(jnp.bool_)
        and all(
            jnp.asarray(value).shape == (2,)
            and jnp.asarray(value).dtype == jnp.dtype(jnp.uint32)
            for value in word_vector_fields
        )
        and jnp.asarray(state.cache_reward_weights).shape
        == (N_ACTIONS, CONTEXT_CONFIG.observation_dim)
        and jnp.asarray(state.cache_reward_weights).dtype == jnp.dtype(jnp.float32)
    )


def _lineage_cache_valid(
    context: ContextInference,
    context_state: ContextInferenceState,
    ledger: ContextBirthLedgerState,
    state: ContextLineageCacheState,
) -> Bool[Array, ""]:
    """Authenticate stable lineages, exact rescue counts, and the lone cache."""

    if not _lineage_cache_static_shape_valid(state):
        return jnp.asarray(False, dtype=jnp.bool_)
    zero_words = jnp.zeros((2,), dtype=jnp.uint32)
    rescue_bounded = (
        state.live_rescue_words[:, 0] == jnp.asarray(0, dtype=jnp.uint32)
    ) & (
        state.live_rescue_words[:, 1]
        <= jnp.asarray(NUM_STEPS, dtype=jnp.uint32)
    )
    live_rows_valid = (
        jnp.all(state.bound_birth_words == ledger.slot_birth_words, axis=1)
        & _words_le(state.live_lineage_words, state.bound_birth_words)
        & rescue_bounded
    )
    unused_rows_zero = (
        jnp.all(state.bound_birth_words == zero_words, axis=1)
        & jnp.all(state.live_lineage_words == zero_words, axis=1)
        & jnp.all(state.live_rescue_words == zero_words, axis=1)
    )
    same_lineage = jnp.all(
        state.live_lineage_words[:, None, :]
        == state.live_lineage_words[None, :, :],
        axis=-1,
    )
    used_pairs = context_state.in_use[:, None] & context_state.in_use[None, :]
    off_diagonal = ~jnp.eye(MAX_CONTEXTS, dtype=jnp.bool_)
    live_lineages_unique = ~jnp.any(same_lineage & used_pairs & off_diagonal)
    cache_rescue_bounded = (
        state.cache_rescue_words[0] == jnp.asarray(0, dtype=jnp.uint32)
    ) & (
        state.cache_rescue_words[1]
        <= jnp.asarray(NUM_STEPS, dtype=jnp.uint32)
    )
    cache_distinct_from_live = ~jnp.any(
        context_state.in_use
        & jnp.all(
            state.live_lineage_words == state.cache_lineage_words[None, :],
            axis=1,
        )
    )
    valid_cache_payload = (
        _words_le(state.cache_lineage_words, state.cache_source_birth_words)
        & _words_le(state.cache_source_birth_words, context_state.step_words)
        & cache_rescue_bounded
        & cache_distinct_from_live
        & jnp.all(jnp.isfinite(state.cache_reward_weights))
    )
    invalid_cache_payload_zero = (
        jnp.all(state.cache_source_birth_words == zero_words)
        & jnp.all(state.cache_lineage_words == zero_words)
        & jnp.all(state.cache_rescue_words == zero_words)
        & jnp.all(state.cache_reward_weights == jnp.float32(0.0))
    )
    return (
        _birth_ledger_valid(context, context_state, ledger)
        & jnp.all(jnp.where(context_state.in_use, live_rows_valid, unused_rows_zero))
        & live_lineages_unique
        & jnp.where(state.cache_valid, valid_cache_payload, invalid_cache_payload_zero)
    )


def _predictive_rescue_scores(
    state: ContextLineageCacheState,
) -> Float[Array, " max_contexts"]:
    """Project fixed-life exact rescue counts to finite float32 priorities."""

    bounded = jnp.where(
        state.live_rescue_words[:, 0] == jnp.asarray(0, dtype=jnp.uint32),
        jnp.minimum(
            state.live_rescue_words[:, 1],
            jnp.asarray(NUM_STEPS, dtype=jnp.uint32),
        ),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    return bounded.astype(jnp.float32)


def _propose_lineage_cache(
    context: ContextInference,
    source_context: ContextInferenceState,
    post_context: ContextInferenceState,
    source_ledger: ContextBirthLedgerState,
    post_ledger: ContextBirthLedgerState,
    state: ContextLineageCacheState,
    observation: Array,
    action: Array,
    reward: Array,
    allocated: Array,
    evicted: Array,
    context_update_applied: Array,
) -> LineageCacheProposal:
    """Apply one strictly post-outcome, one-record lineage-cache transaction."""

    source_valid = _lineage_cache_valid(
        context,
        source_context,
        source_ledger,
        state,
    )
    safe_observation = jnp.where(
        jnp.isfinite(observation),
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.zeros_like(observation, dtype=jnp.float32),
    )
    action_index = jnp.squeeze(jnp.asarray(action, dtype=jnp.int32))
    safe_action = jnp.clip(action_index, 0, N_ACTIONS - 1)
    reward_value = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
    safe_reward = jnp.where(
        jnp.isfinite(reward_value),
        reward_value,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    target = jnp.clip(post_context.active_context, 0, MAX_CONTEXTS - 1)
    full_bank_birth = allocated & evicted
    cache_prediction = state.cache_reward_weights[safe_action] @ safe_observation
    fresh_prediction = (
        jnp.full(
            (CONTEXT_CONFIG.observation_dim,),
            jnp.float32(CONTEXT_CONFIG.initial_reward_estimate),
            dtype=jnp.float32,
        )
        @ safe_observation
    )
    live_predictions = (
        source_context.reward_weights[:, safe_action, :] @ safe_observation
    )
    cache_error = jnp.abs(safe_reward - cache_prediction)
    fresh_error = jnp.abs(safe_reward - fresh_prediction)
    live_errors = jnp.abs(safe_reward - live_predictions)
    predictive_inputs_finite = (
        jnp.all(jnp.isfinite(observation))
        & jnp.isfinite(reward_value)
        & (action_index >= 0)
        & (action_index < N_ACTIONS)
        & jnp.isfinite(cache_error)
        & jnp.isfinite(fresh_error)
        & jnp.all(jnp.isfinite(live_errors))
    )
    cache_tested = full_bank_birth & state.cache_valid
    strict_predictive_dominance = (
        cache_tested
        & source_valid
        & predictive_inputs_finite
        & (cache_error < fresh_error)
        & jnp.all(
            jnp.where(
                source_context.in_use,
                cache_error < live_errors,
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
    )
    cache_matched = strict_predictive_dominance
    incremented_rescue, increment_capacity = _checked_words_increment(
        state.cache_rescue_words
    )
    # A terminal exact counter halts the whole transaction even on an
    # abstaining sample.  This keeps a future increment from silently wrapping
    # after an unrelated step has been allowed to pass the terminal state.
    rescue_capacity_available = increment_capacity

    victim_birth = source_ledger.slot_birth_words[target]
    victim_lineage = state.live_lineage_words[target]
    victim_rescue = state.live_rescue_words[target]
    victim_weights = source_context.reward_weights[target]
    rescue_less = _words_lt(state.cache_rescue_words, victim_rescue)
    rescue_equal = jnp.all(state.cache_rescue_words == victim_rescue)
    victim_newer_or_equal = _words_le(state.cache_source_birth_words, victim_birth)
    victim_preferred = (
        ~state.cache_valid
        | rescue_less
        | (rescue_equal & victim_newer_or_equal)
    )
    write_victim = full_bank_birth & (cache_matched | victim_preferred)
    old_cache_retained = (
        full_bank_birth & ~cache_matched & state.cache_valid & ~victim_preferred
    )

    post_birth = post_ledger.slot_birth_words[target]
    next_lineage = jnp.where(cache_matched, state.cache_lineage_words, post_birth)
    next_rescue = jnp.where(
        cache_matched,
        incremented_rescue,
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    bound_birth_words = jnp.where(
        allocated,
        state.bound_birth_words.at[target].set(post_birth),
        state.bound_birth_words,
    )
    live_lineage_words = jnp.where(
        allocated,
        state.live_lineage_words.at[target].set(next_lineage),
        state.live_lineage_words,
    )
    live_rescue_words = jnp.where(
        allocated,
        state.live_rescue_words.at[target].set(next_rescue),
        state.live_rescue_words,
    )
    candidate = ContextLineageCacheState(
        bound_birth_words=bound_birth_words,
        live_lineage_words=live_lineage_words,
        live_rescue_words=live_rescue_words,
        cache_valid=jnp.where(write_victim, jnp.asarray(True), state.cache_valid),
        cache_source_birth_words=jnp.where(
            write_victim,
            victim_birth,
            state.cache_source_birth_words,
        ),
        cache_lineage_words=jnp.where(
            write_victim,
            victim_lineage,
            state.cache_lineage_words,
        ),
        cache_rescue_words=jnp.where(
            write_victim,
            victim_rescue,
            state.cache_rescue_words,
        ),
        cache_reward_weights=jnp.where(
            write_victim,
            victim_weights,
            state.cache_reward_weights,
        ),
    )
    candidate_valid = _lineage_cache_valid(
        context,
        post_context,
        post_ledger,
        candidate,
    )
    update_applied = (
        source_valid
        & context_update_applied
        & rescue_capacity_available
        & candidate_valid
    )
    committed = jax.tree_util.tree_map(
        lambda proposed, current: jnp.where(update_applied, proposed, current),
        candidate,
        state,
    )
    return LineageCacheProposal(
        state=committed,
        source_valid=source_valid,
        candidate_valid=candidate_valid,
        rescue_capacity_available=rescue_capacity_available,
        full_bank_birth=full_bank_birth,
        cache_tested=cache_tested,
        strict_predictive_dominance=strict_predictive_dominance,
        cache_matched=cache_matched,
        lineage_transferred=cache_matched & update_applied,
        rescue_incremented=cache_matched & update_applied,
        victim_archived=write_victim & update_applied,
        old_cache_retained=old_cache_retained & update_applied,
        update_applied=update_applied,
        cache_error=jnp.where(cache_tested, cache_error, jnp.float32(0.0)),
        fresh_error=jnp.where(cache_tested, fresh_error, jnp.float32(0.0)),
        live_errors=jnp.where(
            cache_tested,
            live_errors,
            jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
        ),
    )


def _tree_exact_equal(left: object, right: object) -> Bool[Array, ""]:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        equal = equal & jnp.all(left_array == right_array)
    return equal


def _birth_ledger_valid(
    context: ContextInference,
    state: ContextInferenceState,
    ledger: ContextBirthLedgerState,
) -> Bool[Array, ""]:
    words = jnp.asarray(ledger.slot_birth_words)
    if words.shape != (MAX_CONTEXTS, 2) or words.dtype != jnp.dtype(jnp.uint32):
        return jnp.asarray(False, dtype=jnp.bool_)
    zero_rows = jnp.all(words == jnp.asarray(0, dtype=jnp.uint32), axis=1)
    unused_rows_zero = jnp.all(jnp.where(state.in_use, True, zero_rows))
    # Only slot zero has a legitimate identity-zero genesis.  Every other
    # allocated slot must have an exact nonzero allocation stamp.
    non_genesis_in_use_has_birth = jnp.all(
        jnp.where(
            state.in_use & (jnp.arange(MAX_CONTEXTS, dtype=jnp.int32) != 0),
            ~zero_rows,
            True,
        )
    )
    births_not_from_future = jnp.all(
        jnp.where(state.in_use, _words_le(words, state.step_words), True)
    )
    pair_equal = jnp.all(words[:, None, :] == words[None, :, :], axis=-1)
    different_slots = ~jnp.eye(MAX_CONTEXTS, dtype=jnp.bool_)
    duplicate_live_birth = jnp.any(
        pair_equal
        & different_slots
        & state.in_use[:, None]
        & state.in_use[None, :]
    )
    return (
        context.state_is_valid(state)
        & unused_rows_zero
        & non_genesis_in_use_has_birth
        & births_not_from_future
        & ~duplicate_live_birth
    )


def _prepare_controller_scrub(
    context: ContextInference,
    controller: DifferentialSARSAState,
    pre_context: ContextInferenceState,
    post_context: ContextInferenceState,
    pre_ledger: ContextBirthLedgerState,
    post_ledger: ContextBirthLedgerState,
    allocated: Array,
    post_step_words: Array,
    audit_destination_override: int,
) -> ControllerScrubPreparation:
    """Authenticate one allocation and zero only its destination Q columns."""

    source_slot = pre_context.active_context.astype(jnp.int32)
    actual_destination = post_context.active_context.astype(jnp.int32)
    destination_slot = jnp.where(
        audit_destination_override >= 0,
        jnp.asarray(audit_destination_override, dtype=jnp.int32),
        actual_destination,
    )
    safe_destination = jnp.clip(destination_slot, 0, MAX_CONTEXTS - 1)
    controller_shape_valid = jnp.asarray(
        controller.q_weights.shape == (N_ACTIONS, MAX_CONTEXTS)
        and controller.q_trace_weights.shape == (N_ACTIONS, MAX_CONTEXTS)
        and controller.last_observation.shape == (MAX_CONTEXTS,)
        and controller.q_bias.shape == (N_ACTIONS,)
        and controller.q_trace_bias.shape == (N_ACTIONS,),
        dtype=jnp.bool_,
    )
    pre_bank_valid = context.state_is_valid(pre_context)
    post_bank_valid = context.state_is_valid(post_context)
    pre_ledger_valid = _birth_ledger_valid(context, pre_context, pre_ledger)
    post_ledger_valid = _birth_ledger_valid(context, post_context, post_ledger)
    controller_source_finite = _tree_finite(controller)
    source_destination_separated = destination_slot != source_slot
    destination_in_range = (destination_slot >= 0) & (
        destination_slot < MAX_CONTEXTS
    )
    expected_source_observation = jax.nn.one_hot(
        source_slot,
        MAX_CONTEXTS,
        dtype=jnp.float32,
    )
    source_credit_binding = jnp.all(
        controller.last_observation == expected_source_observation
    )
    destination_matches_bank = destination_slot == actual_destination
    post_destination_birth = post_ledger.slot_birth_words[safe_destination]
    pre_destination_birth = pre_ledger.slot_birth_words[safe_destination]
    destination_birth_bound = jnp.all(post_destination_birth == post_step_words)
    destination_birth_changed = jnp.any(
        post_destination_birth != pre_destination_birth
    )
    rows = jnp.arange(MAX_CONTEXTS, dtype=jnp.int32)
    other_births_unchanged = jnp.all(
        jnp.where(
            (rows != safe_destination)[:, None],
            post_ledger.slot_birth_words == pre_ledger.slot_birth_words,
            True,
        )
    )
    binding_valid = (
        destination_in_range
        & destination_matches_bank
        & source_credit_binding
        & source_destination_separated
        & post_context.in_use[safe_destination]
        & destination_birth_bound
        & destination_birth_changed
        & other_births_unchanged
        & jnp.all(post_context.step_words == post_step_words)
    )
    biases_zero_before = jnp.all(controller.q_bias == 0.0) & jnp.all(
        controller.q_trace_bias == 0.0
    )
    authentication_valid = (
        controller_shape_valid
        & pre_bank_valid
        & post_bank_valid
        & pre_ledger_valid
        & post_ledger_valid
        & controller_source_finite
        & binding_valid
        & biases_zero_before
    )
    proposed = controller.replace(  # type: ignore[attr-defined]
        q_weights=controller.q_weights.at[:, safe_destination].set(
            jnp.asarray(0.0, dtype=jnp.float32)
        ),
        q_trace_weights=controller.q_trace_weights.at[:, safe_destination].set(
            jnp.asarray(0.0, dtype=jnp.float32)
        ),
    )
    scrub_candidate_applied = allocated & authentication_valid
    prepared = jax.lax.cond(
        scrub_candidate_applied,
        lambda _: proposed,
        lambda _: controller,
        operand=None,
    )
    candidate_finite = _tree_finite(prepared)
    preparation_valid = ~allocated | (authentication_valid & candidate_finite)
    survivor_mask = rows != safe_destination
    survivor_rows_untouched = jnp.all(
        jnp.where(
            survivor_mask[None, :],
            prepared.q_weights == controller.q_weights,
            True,
        )
    ) & jnp.all(
        jnp.where(
            survivor_mask[None, :],
            prepared.q_trace_weights == controller.q_trace_weights,
            True,
        )
    )
    biases_untouched = jnp.all(prepared.q_bias == controller.q_bias) & jnp.all(
        prepared.q_trace_bias == controller.q_trace_bias
    )
    average_reward_untouched = prepared.average_reward == controller.average_reward
    rng_untouched = jnp.all(
        jr.key_data(prepared.rng_key) == jr.key_data(controller.rng_key)
    )
    clock_untouched = (
        jnp.all(prepared.step_words == controller.step_words)
        & (prepared.step_count == controller.step_count)
    )
    pre_q_l1 = jnp.sum(jnp.abs(controller.q_weights[:, safe_destination]))
    prepared_q_l1 = jnp.sum(jnp.abs(prepared.q_weights[:, safe_destination]))
    pre_trace_l1 = jnp.sum(
        jnp.abs(controller.q_trace_weights[:, safe_destination])
    )
    prepared_trace_l1 = jnp.sum(
        jnp.abs(prepared.q_trace_weights[:, safe_destination])
    )
    return ControllerScrubPreparation(
        state=prepared,
        scrub_required=allocated,
        scrub_candidate_applied=scrub_candidate_applied,
        preparation_valid=preparation_valid,
        pre_bank_valid=pre_bank_valid,
        post_bank_valid=post_bank_valid,
        pre_ledger_valid=pre_ledger_valid,
        post_ledger_valid=post_ledger_valid,
        binding_valid=binding_valid,
        controller_shape_valid=controller_shape_valid,
        controller_source_finite=controller_source_finite,
        candidate_finite=candidate_finite,
        source_destination_separated=source_destination_separated,
        biases_zero_before=biases_zero_before,
        biases_untouched=biases_untouched,
        average_reward_untouched=average_reward_untouched,
        rng_untouched_before_update=rng_untouched,
        clock_untouched_before_update=clock_untouched,
        survivor_rows_untouched=survivor_rows_untouched,
        prepared_controller_unchanged=_tree_exact_equal(prepared, controller),
        source_slot=source_slot,
        destination_slot=destination_slot,
        pre_destination_birth_words=pre_destination_birth,
        post_destination_birth_words=post_destination_birth,
        pre_destination_q_weight_l1=pre_q_l1,
        prepared_destination_q_weight_l1=prepared_q_l1,
        pre_destination_q_trace_l1=pre_trace_l1,
        prepared_destination_q_trace_l1=prepared_trace_l1,
        stale_destination_q_available=allocated & (pre_q_l1 > 0.0),
    )


def _controller_scrub_shape_valid(controller: DifferentialSARSAState) -> bool:
    return (
        controller.q_weights.shape == (N_ACTIONS, MAX_CONTEXTS)
        and controller.q_trace_weights.shape == (N_ACTIONS, MAX_CONTEXTS)
        and controller.last_observation.shape == (MAX_CONTEXTS,)
        and controller.q_bias.shape == (N_ACTIONS,)
        and controller.q_trace_bias.shape == (N_ACTIONS,)
    )


def _post_audit_static_shapes_valid(state: CapacityPressureState) -> bool:
    return (
        _controller_scrub_shape_valid(state.controller_0)
        and _controller_scrub_shape_valid(state.controller_1)
        and state.ledger_0.slot_birth_words.shape == (MAX_CONTEXTS, 2)
        and state.ledger_1.slot_birth_words.shape == (MAX_CONTEXTS, 2)
        and state.ledger_0.slot_birth_words.dtype == jnp.dtype(jnp.uint32)
        and state.ledger_1.slot_birth_words.dtype == jnp.dtype(jnp.uint32)
    )


def _shape_rejected_post_audit_result(
    condition: PostAuditCondition,
    state: CapacityPressureState,
) -> PostAuditStepResult:
    """Return a bit-exact rollback before malformed shapes reach child kernels."""

    actions = jnp.stack(
        (state.controller_0.last_action, state.controller_1.last_action)
    ).astype(jnp.int32)
    slots = jnp.stack(
        (state.context_0.active_context, state.context_1.active_context)
    ).astype(jnp.int32)
    births = jnp.zeros((2, 2), dtype=jnp.uint32)
    shape_valid = jnp.asarray(
        (
            _controller_scrub_shape_valid(state.controller_0),
            _controller_scrub_shape_valid(state.controller_1),
        ),
        dtype=jnp.bool_,
    )
    false_two = jnp.zeros((2,), dtype=jnp.bool_)
    true_two = jnp.ones((2,), dtype=jnp.bool_)
    zero_l1 = jnp.zeros((2,), dtype=jnp.float32)
    trace = CapacityPressureStepTrace(
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        actions=actions,
        pre_context_slots=slots,
        post_context_slots=slots,
        pre_context_birth_words=births,
        post_context_birth_words=births,
        switches=false_two,
        allocations=false_two,
        evictions=false_two,
        reuses=false_two,
        contexts_in_use=jnp.stack(
            (jnp.sum(state.context_0.in_use), jnp.sum(state.context_1.in_use))
        ).astype(jnp.int32),
        environment_update_proposed=jnp.asarray(False, dtype=jnp.bool_),
        context_updates_proposed=false_two,
        controller_updates_proposed=false_two,
        source_clocks_aligned=_clocks_aligned(state),
        candidate_clocks_aligned=jnp.asarray(False, dtype=jnp.bool_),
        source_state_finite=_tree_finite(state),
        candidate_state_finite=jnp.asarray(False, dtype=jnp.bool_),
        update_applied=jnp.asarray(False, dtype=jnp.bool_),
        pre_step_words=state.environment.step_words,
        post_step_words=state.environment.step_words,
        controller_rng_key_words=jnp.stack(
            (
                jr.key_data(state.controller_0.rng_key),
                jr.key_data(state.controller_1.rng_key),
            )
        ).astype(jnp.uint32),
        controller_next_q_values=jnp.zeros((2, N_ACTIONS), dtype=jnp.float32),
        controller_next_actions=actions,
    )
    scrub = PostAuditScrubTrace(
        scrub_enabled=jnp.asarray(
            condition == BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
            dtype=jnp.bool_,
        ),
        scrub_required=true_two,
        scrub_applied=false_two,
        preparation_valid=false_two,
        authentication_failed=true_two,
        pre_bank_valid=false_two,
        post_bank_valid=false_two,
        pre_ledger_valid=false_two,
        post_ledger_valid=false_two,
        binding_valid=false_two,
        controller_shape_valid=shape_valid,
        controller_source_finite=jnp.stack(
            (_tree_finite(state.controller_0), _tree_finite(state.controller_1))
        ),
        candidate_finite=false_two,
        source_destination_separated=false_two,
        biases_zero_before=false_two,
        biases_untouched=true_two,
        average_reward_untouched=true_two,
        rng_untouched_before_update=true_two,
        clock_untouched_before_update=true_two,
        survivor_rows_untouched=true_two,
        prepared_controller_unchanged=true_two,
        source_slots=slots,
        destination_slots=slots,
        pre_destination_birth_words=births,
        post_destination_birth_words=births,
        pre_destination_q_weight_l1=zero_l1,
        prepared_destination_q_weight_l1=zero_l1,
        pre_destination_q_trace_l1=zero_l1,
        prepared_destination_q_trace_l1=zero_l1,
        cross_birth_contamination_available=false_two,
        cross_birth_contamination_consumed=false_two,
        cross_birth_contamination_prevented=false_two,
        scrubbed_parameter_scalars=jnp.zeros((2,), dtype=jnp.int32),
        update_applied=jnp.asarray(False, dtype=jnp.bool_),
    )
    return PostAuditStepResult(
        state=state,
        trace=trace,
        scrub=scrub,
        prepared_controllers=(state.controller_0, state.controller_1),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 5))
def _step_post_audit_intervention(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: PostAuditCondition,
    state: CapacityPressureState,
    audit_destination_override: tuple[int, int],
) -> PostAuditStepResult:
    """Stage the exact baseline and an authenticated scrub alternative."""

    if not _post_audit_static_shapes_valid(state):
        return _shape_rejected_post_audit_result(condition, state)
    baseline = _step_capacity_pressure(agent, context, game, state)
    preparation_0 = _prepare_controller_scrub(
        context,
        state.controller_0,
        state.context_0,
        baseline.state.context_0,
        state.ledger_0,
        baseline.state.ledger_0,
        baseline.trace.allocations[0],
        baseline.trace.post_step_words,
        audit_destination_override[0],
    )
    preparation_1 = _prepare_controller_scrub(
        context,
        state.controller_1,
        state.context_1,
        baseline.state.context_1,
        state.ledger_1,
        baseline.state.ledger_1,
        baseline.trace.allocations[1],
        baseline.trace.post_step_words,
        audit_destination_override[1],
    )
    scrub_result_0 = agent.update(
        preparation_0.state,
        baseline.trace.reward,
        context.context_onehot(baseline.state.context_0),
    )
    scrub_result_1 = agent.update(
        preparation_1.state,
        baseline.trace.reward,
        context.context_onehot(baseline.state.context_1),
    )
    scrub_candidate_state = CapacityPressureState(
        environment=baseline.state.environment,
        controller_0=scrub_result_0.state,
        controller_1=scrub_result_1.state,
        context_0=baseline.state.context_0,
        context_1=baseline.state.context_1,
        ledger_0=baseline.state.ledger_0,
        ledger_1=baseline.state.ledger_1,
    )
    scrub_candidate_clocks_aligned = _clocks_aligned(scrub_candidate_state)
    scrub_candidate_finite = _tree_finite(scrub_candidate_state)
    scrub_update_applied = (
        baseline.trace.update_applied
        & preparation_0.preparation_valid
        & preparation_1.preparation_valid
        & scrub_result_0.update_applied
        & scrub_result_1.update_applied
        & scrub_candidate_clocks_aligned
        & scrub_candidate_finite
    )
    scrub_committed_state = jax.lax.cond(
        scrub_update_applied,
        lambda _: scrub_candidate_state,
        lambda _: state,
        operand=None,
    )
    scrub_post_context_slots = jnp.stack(
        (
            scrub_committed_state.context_0.active_context,
            scrub_committed_state.context_1.active_context,
        )
    ).astype(jnp.int32)
    scrub_post_birth_words = jnp.stack(
        (
            scrub_committed_state.ledger_0.slot_birth_words[
                scrub_committed_state.context_0.active_context
            ],
            scrub_committed_state.ledger_1.slot_birth_words[
                scrub_committed_state.context_1.active_context
            ],
        )
    ).astype(jnp.uint32)
    scrub_trace = baseline.trace.replace(
        reward=jnp.where(
            scrub_update_applied,
            baseline.trace.reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        post_context_slots=scrub_post_context_slots,
        post_context_birth_words=scrub_post_birth_words,
        switches=baseline.trace.switches & scrub_update_applied,
        allocations=baseline.trace.allocations & scrub_update_applied,
        evictions=baseline.trace.evictions & scrub_update_applied,
        reuses=baseline.trace.reuses & scrub_update_applied,
        contexts_in_use=jnp.stack(
            (
                context.num_contexts_in_use(scrub_committed_state.context_0),
                context.num_contexts_in_use(scrub_committed_state.context_1),
            )
        ).astype(jnp.int32),
        controller_updates_proposed=jnp.stack(
            (scrub_result_0.update_applied, scrub_result_1.update_applied)
        ).astype(jnp.bool_),
        candidate_clocks_aligned=scrub_candidate_clocks_aligned,
        candidate_state_finite=scrub_candidate_finite,
        update_applied=scrub_update_applied,
        post_step_words=scrub_committed_state.environment.step_words,
        controller_rng_key_words=jnp.stack(
            (
                jr.key_data(scrub_committed_state.controller_0.rng_key),
                jr.key_data(scrub_committed_state.controller_1.rng_key),
            )
        ).astype(jnp.uint32),
        controller_next_q_values=jnp.where(
            scrub_update_applied,
            jnp.stack((scrub_result_0.q_values, scrub_result_1.q_values)),
            jnp.zeros((2, N_ACTIONS), dtype=jnp.float32),
        ),
        controller_next_actions=jnp.where(
            scrub_update_applied,
            jnp.stack((scrub_result_0.action, scrub_result_1.action)),
            jnp.stack(
                (state.controller_0.last_action, state.controller_1.last_action)
            ),
        ).astype(jnp.int32),
    )
    scrub_enabled = condition == BIRTH_AUTHENTICATED_CONTROLLER_SCRUB
    if scrub_enabled:
        selected_state = scrub_committed_state
        selected_trace = scrub_trace
    else:
        selected_state = baseline.state
        selected_trace = baseline.trace

    preparations = (preparation_0, preparation_1)

    def stack_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(preparation, name)) for preparation in preparations)
        )

    scrub_required = stack_field("scrub_required").astype(jnp.bool_)
    preparation_valid = stack_field("preparation_valid").astype(jnp.bool_)
    stale_available = stack_field("stale_destination_q_available").astype(jnp.bool_)
    selected_commit = selected_trace.update_applied
    selected_scrub_applied = jnp.where(
        scrub_enabled,
        stack_field("scrub_candidate_applied").astype(jnp.bool_) & selected_commit,
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    contamination_consumed = jnp.where(
        scrub_enabled,
        jnp.zeros((2,), dtype=jnp.bool_),
        stale_available & baseline.trace.update_applied,
    )
    contamination_prevented = jnp.where(
        scrub_enabled,
        stale_available & selected_scrub_applied,
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    scrub_diagnostics = PostAuditScrubTrace(
        scrub_enabled=jnp.asarray(scrub_enabled, dtype=jnp.bool_),
        scrub_required=scrub_required,
        scrub_applied=selected_scrub_applied,
        preparation_valid=preparation_valid,
        authentication_failed=scrub_required & ~preparation_valid,
        pre_bank_valid=stack_field("pre_bank_valid").astype(jnp.bool_),
        post_bank_valid=stack_field("post_bank_valid").astype(jnp.bool_),
        pre_ledger_valid=stack_field("pre_ledger_valid").astype(jnp.bool_),
        post_ledger_valid=stack_field("post_ledger_valid").astype(jnp.bool_),
        binding_valid=stack_field("binding_valid").astype(jnp.bool_),
        controller_shape_valid=stack_field("controller_shape_valid").astype(jnp.bool_),
        controller_source_finite=stack_field("controller_source_finite").astype(jnp.bool_),
        candidate_finite=stack_field("candidate_finite").astype(jnp.bool_),
        source_destination_separated=stack_field(
            "source_destination_separated"
        ).astype(jnp.bool_),
        biases_zero_before=stack_field("biases_zero_before").astype(jnp.bool_),
        biases_untouched=stack_field("biases_untouched").astype(jnp.bool_),
        average_reward_untouched=stack_field("average_reward_untouched").astype(
            jnp.bool_
        ),
        rng_untouched_before_update=stack_field(
            "rng_untouched_before_update"
        ).astype(jnp.bool_),
        clock_untouched_before_update=stack_field(
            "clock_untouched_before_update"
        ).astype(jnp.bool_),
        survivor_rows_untouched=stack_field("survivor_rows_untouched").astype(
            jnp.bool_
        ),
        prepared_controller_unchanged=stack_field(
            "prepared_controller_unchanged"
        ).astype(jnp.bool_),
        source_slots=stack_field("source_slot").astype(jnp.int32),
        destination_slots=stack_field("destination_slot").astype(jnp.int32),
        pre_destination_birth_words=stack_field(
            "pre_destination_birth_words"
        ).astype(jnp.uint32),
        post_destination_birth_words=stack_field(
            "post_destination_birth_words"
        ).astype(jnp.uint32),
        pre_destination_q_weight_l1=stack_field(
            "pre_destination_q_weight_l1"
        ).astype(jnp.float32),
        prepared_destination_q_weight_l1=stack_field(
            "prepared_destination_q_weight_l1"
        ).astype(jnp.float32),
        pre_destination_q_trace_l1=stack_field(
            "pre_destination_q_trace_l1"
        ).astype(jnp.float32),
        prepared_destination_q_trace_l1=stack_field(
            "prepared_destination_q_trace_l1"
        ).astype(jnp.float32),
        cross_birth_contamination_available=stale_available,
        cross_birth_contamination_consumed=contamination_consumed,
        cross_birth_contamination_prevented=contamination_prevented,
        scrubbed_parameter_scalars=(
            selected_scrub_applied.astype(jnp.int32) * (2 * N_ACTIONS)
        ),
        update_applied=selected_trace.update_applied,
    )
    return PostAuditStepResult(
        state=selected_state,
        trace=selected_trace,
        scrub=scrub_diagnostics,
        prepared_controllers=(preparation_0.state, preparation_1.state),
    )


def step_post_audit_intervention(
    epsilon: float,
    condition: PostAuditCondition,
    state: CapacityPressureState,
    *,
    audit_destination_override: tuple[int, int] | None = None,
) -> PostAuditStepResult:
    """Test one paired step; the override exists only for fail-closed audits."""

    if condition not in POST_AUDIT_CONDITIONS:
        raise ValueError("unknown post-audit condition")
    override = (-1, -1) if audit_destination_override is None else audit_destination_override
    if (
        not isinstance(override, tuple)
        or len(override) != 2
        or any(type(value) is not int for value in override)
    ):
        raise ValueError("audit destination override must be a pair of integers")
    return cast(
        PostAuditStepResult,
        _step_post_audit_intervention(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            condition,
            state,
            override,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _scan_post_audit_life(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: PostAuditCondition,
    initial_state: CapacityPressureState,
) -> tuple[
    CapacityPressureState,
    tuple[CapacityPressureStepTrace, PostAuditScrubTrace],
]:
    def scan_step(
        state: CapacityPressureState,
        _: None,
    ) -> tuple[
        CapacityPressureState,
        tuple[CapacityPressureStepTrace, PostAuditScrubTrace],
    ]:
        result = _step_post_audit_intervention(
            agent,
            context,
            game,
            condition,
            state,
            (-1, -1),
        )
        return result.state, (result.trace, result.scrub)

    return jax.lax.scan(scan_step, initial_state, xs=None, length=NUM_STEPS)


def _selective_static_shapes_valid(state: SelectiveRetentionState) -> bool:
    return (
        _post_audit_static_shapes_valid(state.base)
        and _recurrence_history_static_shape_valid(state.recurrence_0)
        and _recurrence_history_static_shape_valid(state.recurrence_1)
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _step_selective_retention(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: SelectiveRetentionCondition,
    state: SelectiveRetentionState,
) -> SelectiveRetentionStepResult:
    """Apply one atomic, scrubbed transition with a past-only eviction signal."""

    base = state.base
    source_clocks_aligned = _clocks_aligned(base)
    source_state_finite = _tree_finite(state)
    history_source_valid_0 = _recurrence_history_valid(
        context,
        base.context_0,
        base.ledger_0,
        state.recurrence_0,
    )
    history_source_valid_1 = _recurrence_history_valid(
        context,
        base.context_1,
        base.ledger_1,
        state.recurrence_1,
    )
    actions = jnp.stack(
        (base.controller_0.last_action, base.controller_1.last_action)
    ).astype(jnp.int32)
    pre_context_slots = jnp.stack(
        (base.context_0.active_context, base.context_1.active_context)
    ).astype(jnp.int32)
    pre_context_birth_words = jnp.stack(
        (
            base.ledger_0.slot_birth_words[base.context_0.active_context],
            base.ledger_1.slot_birth_words[base.context_1.active_context],
        )
    ).astype(jnp.uint32)
    raw_score_0 = _completed_recurrence_scores(state.recurrence_0)
    raw_score_1 = _completed_recurrence_scores(state.recurrence_1)
    protection_enabled = condition == SELECTIVE_RETENTION_PAST_RECURRENCE
    dispatched_0 = jnp.where(
        protection_enabled,
        raw_score_0,
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )
    dispatched_1 = jnp.where(
        protection_enabled,
        raw_score_1,
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )

    # Actions are irrevocably fixed before either context bank sees the
    # partner action or common reward.  Scores above contain authenticated
    # completed entries only and have no environment/schedule query.
    environment_result = game.step_result(
        base.environment,
        actions[0],
        actions[1],
    )
    observation_0 = jax.nn.one_hot(actions[1], N_ACTIONS, dtype=jnp.float32)
    observation_1 = jax.nn.one_hot(actions[0], N_ACTIONS, dtype=jnp.float32)
    context_result_0 = context.update_result_with_eviction_protection(
        base.context_0,
        observation_0,
        actions[0],
        environment_result.reward,
        dispatched_0,
    )
    context_result_1 = context.update_result_with_eviction_protection(
        base.context_1,
        observation_1,
        actions[1],
        environment_result.reward,
        dispatched_1,
    )
    event_0 = _context_event(
        context,
        base.context_0,
        context_result_0,
        observation_0,
        actions[0],
        environment_result.reward,
    )
    event_1 = _context_event(
        context,
        base.context_1,
        context_result_1,
        observation_1,
        actions[1],
        environment_result.reward,
    )
    proposed_ledger_0 = _propose_ledger(
        base.ledger_0,
        context_result_0,
        event_0[1],
    )
    proposed_ledger_1 = _propose_ledger(
        base.ledger_1,
        context_result_1,
        event_1[1],
    )
    recurrence_result_0 = _propose_recurrence_history(
        context,
        base.context_0,
        context_result_0.state,
        base.ledger_0,
        proposed_ledger_0,
        state.recurrence_0,
        event_0[1],
        event_0[3],
        context_result_0.post_step_words,
        context_result_0.update_applied,
    )
    recurrence_result_1 = _propose_recurrence_history(
        context,
        base.context_1,
        context_result_1.state,
        base.ledger_1,
        proposed_ledger_1,
        state.recurrence_1,
        event_1[1],
        event_1[3],
        context_result_1.post_step_words,
        context_result_1.update_applied,
    )

    preparation_0 = _prepare_controller_scrub(
        context,
        base.controller_0,
        base.context_0,
        context_result_0.state,
        base.ledger_0,
        proposed_ledger_0,
        event_0[1],
        context_result_0.post_step_words,
        -1,
    )
    preparation_1 = _prepare_controller_scrub(
        context,
        base.controller_1,
        base.context_1,
        context_result_1.state,
        base.ledger_1,
        proposed_ledger_1,
        event_1[1],
        context_result_1.post_step_words,
        -1,
    )
    controller_result_0 = agent.update(
        preparation_0.state,
        environment_result.reward,
        context_result_0.context_onehot,
    )
    controller_result_1 = agent.update(
        preparation_1.state,
        environment_result.reward,
        context_result_1.context_onehot,
    )
    candidate_base = CapacityPressureState(
        environment=environment_result.state,
        controller_0=controller_result_0.state,
        controller_1=controller_result_1.state,
        context_0=context_result_0.state,
        context_1=context_result_1.state,
        ledger_0=proposed_ledger_0,
        ledger_1=proposed_ledger_1,
    )
    candidate_state = SelectiveRetentionState(
        base=candidate_base,
        recurrence_0=recurrence_result_0.state,
        recurrence_1=recurrence_result_1.state,
    )
    candidate_clocks_aligned = _clocks_aligned(candidate_base)
    candidate_state_finite = _tree_finite(candidate_state)
    context_updates_proposed = jnp.stack(
        (context_result_0.update_applied, context_result_1.update_applied)
    ).astype(jnp.bool_)
    controller_updates_proposed = jnp.stack(
        (controller_result_0.update_applied, controller_result_1.update_applied)
    ).astype(jnp.bool_)
    history_updates_proposed = jnp.stack(
        (recurrence_result_0.update_applied, recurrence_result_1.update_applied)
    ).astype(jnp.bool_)
    source_children_valid = (
        environment_result.state_valid
        & context_result_0.source_state_valid
        & context_result_1.source_state_valid
        & controller_result_0.state_valid
        & controller_result_1.state_valid
        & history_source_valid_0
        & history_source_valid_1
    )
    candidate_children_valid = (
        context_result_0.candidate_state_valid
        & context_result_1.candidate_state_valid
        & controller_result_0.candidate_state_finite
        & controller_result_1.candidate_state_finite
        & recurrence_result_0.candidate_valid
        & recurrence_result_1.candidate_valid
    )
    update_applied = (
        source_clocks_aligned
        & source_state_finite
        & source_children_valid
        & environment_result.update_applied
        & jnp.all(context_updates_proposed)
        & preparation_0.preparation_valid
        & preparation_1.preparation_valid
        & jnp.all(controller_updates_proposed)
        & jnp.all(history_updates_proposed)
        & candidate_clocks_aligned
        & candidate_state_finite
        & candidate_children_valid
    )
    committed_state = jax.lax.cond(
        update_applied,
        lambda _: candidate_state,
        lambda _: state,
        operand=None,
    )
    committed_base = committed_state.base
    switches = jnp.stack((event_0[0], event_1[0])) & update_applied
    allocations = jnp.stack((event_0[1], event_1[1])) & update_applied
    evictions = jnp.stack((event_0[2], event_1[2])) & update_applied
    reuses = jnp.stack((event_0[3], event_1[3])) & update_applied
    post_context_slots = jnp.stack(
        (
            committed_base.context_0.active_context,
            committed_base.context_1.active_context,
        )
    ).astype(jnp.int32)
    post_context_birth_words = jnp.stack(
        (
            committed_base.ledger_0.slot_birth_words[
                committed_base.context_0.active_context
            ],
            committed_base.ledger_1.slot_birth_words[
                committed_base.context_1.active_context
            ],
        )
    ).astype(jnp.uint32)
    capacity_trace = CapacityPressureStepTrace(
        reward=jnp.where(
            update_applied,
            environment_result.reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        actions=actions,
        pre_context_slots=pre_context_slots,
        post_context_slots=post_context_slots,
        pre_context_birth_words=pre_context_birth_words,
        post_context_birth_words=post_context_birth_words,
        switches=switches,
        allocations=allocations,
        evictions=evictions,
        reuses=reuses,
        contexts_in_use=jnp.stack(
            (
                context.num_contexts_in_use(committed_base.context_0),
                context.num_contexts_in_use(committed_base.context_1),
            )
        ).astype(jnp.int32),
        environment_update_proposed=environment_result.update_applied,
        context_updates_proposed=context_updates_proposed,
        controller_updates_proposed=controller_updates_proposed,
        source_clocks_aligned=source_clocks_aligned,
        candidate_clocks_aligned=candidate_clocks_aligned,
        source_state_finite=source_state_finite,
        candidate_state_finite=candidate_state_finite,
        update_applied=update_applied,
        pre_step_words=base.environment.step_words,
        post_step_words=committed_base.environment.step_words,
        controller_rng_key_words=jnp.stack(
            (
                jr.key_data(committed_base.controller_0.rng_key),
                jr.key_data(committed_base.controller_1.rng_key),
            )
        ).astype(jnp.uint32),
        controller_next_q_values=jnp.where(
            update_applied,
            jnp.stack((controller_result_0.q_values, controller_result_1.q_values)),
            jnp.zeros((2, N_ACTIONS), dtype=jnp.float32),
        ),
        controller_next_actions=jnp.where(
            update_applied,
            jnp.stack((controller_result_0.action, controller_result_1.action)),
            actions,
        ).astype(jnp.int32),
    )
    preparations = (preparation_0, preparation_1)

    def stack_preparation_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(preparation, name)) for preparation in preparations)
        )

    scrub_required = stack_preparation_field("scrub_required").astype(jnp.bool_)
    scrub_applied = (
        stack_preparation_field("scrub_candidate_applied").astype(jnp.bool_)
        & update_applied
    )
    stale_available = stack_preparation_field("stale_destination_q_available").astype(
        jnp.bool_
    )
    scrub_trace = PostAuditScrubTrace(
        scrub_enabled=jnp.asarray(True, dtype=jnp.bool_),
        scrub_required=scrub_required,
        scrub_applied=scrub_applied,
        preparation_valid=stack_preparation_field("preparation_valid").astype(jnp.bool_),
        authentication_failed=scrub_required
        & ~stack_preparation_field("preparation_valid").astype(jnp.bool_),
        pre_bank_valid=stack_preparation_field("pre_bank_valid").astype(jnp.bool_),
        post_bank_valid=stack_preparation_field("post_bank_valid").astype(jnp.bool_),
        pre_ledger_valid=stack_preparation_field("pre_ledger_valid").astype(jnp.bool_),
        post_ledger_valid=stack_preparation_field("post_ledger_valid").astype(jnp.bool_),
        binding_valid=stack_preparation_field("binding_valid").astype(jnp.bool_),
        controller_shape_valid=stack_preparation_field("controller_shape_valid").astype(
            jnp.bool_
        ),
        controller_source_finite=stack_preparation_field(
            "controller_source_finite"
        ).astype(jnp.bool_),
        candidate_finite=stack_preparation_field("candidate_finite").astype(jnp.bool_),
        source_destination_separated=stack_preparation_field(
            "source_destination_separated"
        ).astype(jnp.bool_),
        biases_zero_before=stack_preparation_field("biases_zero_before").astype(jnp.bool_),
        biases_untouched=stack_preparation_field("biases_untouched").astype(jnp.bool_),
        average_reward_untouched=stack_preparation_field(
            "average_reward_untouched"
        ).astype(jnp.bool_),
        rng_untouched_before_update=stack_preparation_field(
            "rng_untouched_before_update"
        ).astype(jnp.bool_),
        clock_untouched_before_update=stack_preparation_field(
            "clock_untouched_before_update"
        ).astype(jnp.bool_),
        survivor_rows_untouched=stack_preparation_field(
            "survivor_rows_untouched"
        ).astype(jnp.bool_),
        prepared_controller_unchanged=stack_preparation_field(
            "prepared_controller_unchanged"
        ).astype(jnp.bool_),
        source_slots=stack_preparation_field("source_slot").astype(jnp.int32),
        destination_slots=stack_preparation_field("destination_slot").astype(jnp.int32),
        pre_destination_birth_words=stack_preparation_field(
            "pre_destination_birth_words"
        ).astype(jnp.uint32),
        post_destination_birth_words=stack_preparation_field(
            "post_destination_birth_words"
        ).astype(jnp.uint32),
        pre_destination_q_weight_l1=stack_preparation_field(
            "pre_destination_q_weight_l1"
        ).astype(jnp.float32),
        prepared_destination_q_weight_l1=stack_preparation_field(
            "prepared_destination_q_weight_l1"
        ).astype(jnp.float32),
        pre_destination_q_trace_l1=stack_preparation_field(
            "pre_destination_q_trace_l1"
        ).astype(jnp.float32),
        prepared_destination_q_trace_l1=stack_preparation_field(
            "prepared_destination_q_trace_l1"
        ).astype(jnp.float32),
        cross_birth_contamination_available=stale_available,
        cross_birth_contamination_consumed=jnp.zeros((2,), dtype=jnp.bool_),
        cross_birth_contamination_prevented=stale_available & scrub_applied,
        scrubbed_parameter_scalars=scrub_applied.astype(jnp.int32) * (2 * N_ACTIONS),
        update_applied=update_applied,
    )
    context_results = (context_result_0, context_result_1)
    recurrence_results = (recurrence_result_0, recurrence_result_1)

    def stack_context_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(result, name)) for result in context_results)
        )

    def stack_history_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(result, name)) for result in recurrence_results)
        )

    raw_scores = jnp.stack((raw_score_0, raw_score_1)).astype(jnp.float32)
    dispatched = jnp.stack((dispatched_0, dispatched_1)).astype(jnp.float32)
    selected_slots = stack_context_field("selected_eviction_slot").astype(jnp.int32)
    safe_selected = jnp.clip(selected_slots, 0, MAX_CONTEXTS - 1)
    ordinary_slots = stack_context_field("ordinary_lru_slot").astype(jnp.int32)
    safe_ordinary = jnp.clip(ordinary_slots, 0, MAX_CONTEXTS - 1)
    applied_priority = (
        stack_context_field("eviction_protection_used").astype(jnp.bool_)
        & update_applied
    )
    selected_scores = jnp.where(
        applied_priority,
        dispatched[jnp.arange(2, dtype=jnp.int32), safe_selected],
        jnp.zeros((2,), dtype=jnp.float32),
    )
    ordinary_raw_scores = jnp.where(
        applied_priority,
        raw_scores[jnp.arange(2, dtype=jnp.int32), safe_ordinary],
        jnp.zeros((2,), dtype=jnp.float32),
    )
    selected_raw_scores = jnp.where(
        applied_priority,
        raw_scores[jnp.arange(2, dtype=jnp.int32), safe_selected],
        jnp.zeros((2,), dtype=jnp.float32),
    )
    trace = SelectiveRetentionTrace(
        capacity=capacity_trace,
        scrub=scrub_trace,
        protection_enabled=jnp.asarray(protection_enabled, dtype=jnp.bool_),
        raw_completed_recurrence_scores=raw_scores,
        dispatched_eviction_protection=dispatched,
        pre_occurrence_words=jnp.stack(
            (state.recurrence_0.occurrence_words, state.recurrence_1.occurrence_words)
        ).astype(jnp.uint32),
        post_occurrence_words=jnp.stack(
            (
                committed_state.recurrence_0.occurrence_words,
                committed_state.recurrence_1.occurrence_words,
            )
        ).astype(jnp.uint32),
        pre_last_entry_words=jnp.stack(
            (state.recurrence_0.last_entry_words, state.recurrence_1.last_entry_words)
        ).astype(jnp.uint32),
        post_last_entry_words=jnp.stack(
            (
                committed_state.recurrence_0.last_entry_words,
                committed_state.recurrence_1.last_entry_words,
            )
        ).astype(jnp.uint32),
        post_last_interval_words=jnp.stack(
            (
                committed_state.recurrence_0.last_interval_words,
                committed_state.recurrence_1.last_interval_words,
            )
        ).astype(jnp.uint32),
        history_source_valid=jnp.stack(
            (history_source_valid_0, history_source_valid_1)
        ).astype(jnp.bool_),
        history_candidate_valid=stack_history_field("candidate_valid").astype(jnp.bool_),
        history_capacity_available=stack_history_field(
            "occurrence_capacity_available"
        ).astype(jnp.bool_),
        history_allocation_resets=stack_history_field("allocation_reset").astype(
            jnp.bool_
        )
        & update_applied,
        history_stored_recurrences=stack_history_field(
            "stored_recurrence_recorded"
        ).astype(jnp.bool_)
        & update_applied,
        history_updates_proposed=history_updates_proposed,
        priority_inputs_valid=stack_context_field(
            "eviction_protection_input_valid"
        ).astype(jnp.bool_),
        full_bank_evictions_requested=stack_context_field(
            "full_bank_eviction_requested"
        ).astype(jnp.bool_),
        eviction_protection_used=applied_priority,
        eviction_targets_adjusted=stack_context_field(
            "eviction_target_adjusted"
        ).astype(jnp.bool_)
        & update_applied,
        ordinary_lru_slots=ordinary_slots,
        protected_lru_slots=stack_context_field("protected_lru_slot").astype(jnp.int32),
        selected_eviction_slots=jnp.where(
            applied_priority,
            selected_slots,
            jnp.full((2,), -1, dtype=jnp.int32),
        ),
        ordinary_lru_completed_recurrence_scores=ordinary_raw_scores,
        selected_completed_recurrence_scores=selected_raw_scores,
        selected_eviction_scores=selected_scores,
    )
    return SelectiveRetentionStepResult(state=committed_state, trace=trace)


def step_selective_retention(
    epsilon: float,
    condition: SelectiveRetentionCondition,
    state: SelectiveRetentionState,
) -> SelectiveRetentionStepResult:
    """Test one exact selective-retention step on the fixed root-zero mechanism."""

    if condition not in SELECTIVE_RETENTION_CONDITIONS:
        raise ValueError("unknown selective-retention condition")
    if not _selective_static_shapes_valid(state):
        raise ValueError("selective-retention state has invalid static shapes or dtypes")
    return cast(
        SelectiveRetentionStepResult,
        _step_selective_retention(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            condition,
            state,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _scan_selective_retention_life(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: SelectiveRetentionCondition,
    initial_state: SelectiveRetentionState,
) -> tuple[SelectiveRetentionState, SelectiveRetentionTrace]:
    def scan_step(
        carry: SelectiveRetentionState,
        _: None,
    ) -> tuple[SelectiveRetentionState, SelectiveRetentionTrace]:
        result = _step_selective_retention(agent, context, game, condition, carry)
        return result.state, result.trace

    return jax.lax.scan(scan_step, initial_state, xs=None, length=NUM_STEPS)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 5))
def _advance_selective_retention_state(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: SelectiveRetentionCondition,
    state: SelectiveRetentionState,
    num_steps: int,
) -> SelectiveRetentionState:
    def advance(_: int, carry: SelectiveRetentionState) -> SelectiveRetentionState:
        return cast(
            SelectiveRetentionState,
            _step_selective_retention(agent, context, game, condition, carry).state,
        )

    return cast(
        SelectiveRetentionState,
        jax.lax.fori_loop(0, num_steps, advance, state),
    )


def advance_consumed_selective_retention_state(
    epsilon: float,
    condition: SelectiveRetentionCondition,
    num_steps: int,
) -> SelectiveRetentionState:
    """Advance one fixed root-zero prefix for causal and rollback tests."""

    if condition not in SELECTIVE_RETENTION_CONDITIONS:
        raise ValueError("unknown selective-retention condition")
    if type(num_steps) is not int or not 0 <= num_steps <= NUM_STEPS:
        raise ValueError("num_steps must be an integer within the consumed life")
    return cast(
        SelectiveRetentionState,
        _advance_selective_retention_state(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            condition,
            initialize_selective_retention_state(epsilon),
            num_steps,
        ),
    )


def _lineage_cache_static_shapes_valid(state: LineageCacheRetentionState) -> bool:
    return (
        _post_audit_static_shapes_valid(state.base)
        and _lineage_cache_static_shape_valid(state.lineage_0)
        and _lineage_cache_static_shape_valid(state.lineage_1)
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _step_lineage_cache_retention(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: LineageCacheCondition,
    state: LineageCacheRetentionState,
) -> LineageCacheRetentionStepResult:
    """Stage one scrubbed step with pre-outcome scores and post-outcome matching."""

    base = state.base
    source_clocks_aligned = _clocks_aligned(base)
    source_state_finite = _tree_finite(state)
    lineage_source_valid_0 = _lineage_cache_valid(
        context,
        base.context_0,
        base.ledger_0,
        state.lineage_0,
    )
    lineage_source_valid_1 = _lineage_cache_valid(
        context,
        base.context_1,
        base.ledger_1,
        state.lineage_1,
    )
    actions = jnp.stack(
        (base.controller_0.last_action, base.controller_1.last_action)
    ).astype(jnp.int32)
    pre_context_slots = jnp.stack(
        (base.context_0.active_context, base.context_1.active_context)
    ).astype(jnp.int32)
    pre_context_birth_words = jnp.stack(
        (
            base.ledger_0.slot_birth_words[base.context_0.active_context],
            base.ledger_1.slot_birth_words[base.context_1.active_context],
        )
    ).astype(jnp.uint32)

    # These exact source-state snapshots precede the environment transition.
    # The later cache match cannot flow backward into this eviction decision.
    raw_score_0 = _predictive_rescue_scores(state.lineage_0)
    raw_score_1 = _predictive_rescue_scores(state.lineage_1)
    protection_enabled = condition == LINEAGE_CACHE_PREDICTIVE_RESCUE
    dispatched_0 = jnp.where(
        protection_enabled,
        raw_score_0,
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )
    dispatched_1 = jnp.where(
        protection_enabled,
        raw_score_1,
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )

    environment_result = game.step_result(
        base.environment,
        actions[0],
        actions[1],
    )
    observation_0 = jax.nn.one_hot(actions[1], N_ACTIONS, dtype=jnp.float32)
    observation_1 = jax.nn.one_hot(actions[0], N_ACTIONS, dtype=jnp.float32)
    context_result_0 = context.update_result_with_eviction_protection(
        base.context_0,
        observation_0,
        actions[0],
        environment_result.reward,
        dispatched_0,
    )
    context_result_1 = context.update_result_with_eviction_protection(
        base.context_1,
        observation_1,
        actions[1],
        environment_result.reward,
        dispatched_1,
    )
    event_0 = _context_event(
        context,
        base.context_0,
        context_result_0,
        observation_0,
        actions[0],
        environment_result.reward,
    )
    event_1 = _context_event(
        context,
        base.context_1,
        context_result_1,
        observation_1,
        actions[1],
        environment_result.reward,
    )
    proposed_ledger_0 = _propose_ledger(
        base.ledger_0,
        context_result_0,
        event_0[1],
    )
    proposed_ledger_1 = _propose_ledger(
        base.ledger_1,
        context_result_1,
        event_1[1],
    )

    # Cache evidence is evaluated only now, after both actions were fixed and
    # after the current target was selected from the source score snapshot.
    lineage_result_0 = _propose_lineage_cache(
        context,
        base.context_0,
        context_result_0.state,
        base.ledger_0,
        proposed_ledger_0,
        state.lineage_0,
        observation_0,
        actions[0],
        environment_result.reward,
        event_0[1],
        event_0[2],
        context_result_0.update_applied,
    )
    lineage_result_1 = _propose_lineage_cache(
        context,
        base.context_1,
        context_result_1.state,
        base.ledger_1,
        proposed_ledger_1,
        state.lineage_1,
        observation_1,
        actions[1],
        environment_result.reward,
        event_1[1],
        event_1[2],
        context_result_1.update_applied,
    )
    preparation_0 = _prepare_controller_scrub(
        context,
        base.controller_0,
        base.context_0,
        context_result_0.state,
        base.ledger_0,
        proposed_ledger_0,
        event_0[1],
        context_result_0.post_step_words,
        -1,
    )
    preparation_1 = _prepare_controller_scrub(
        context,
        base.controller_1,
        base.context_1,
        context_result_1.state,
        base.ledger_1,
        proposed_ledger_1,
        event_1[1],
        context_result_1.post_step_words,
        -1,
    )
    controller_result_0 = agent.update(
        preparation_0.state,
        environment_result.reward,
        context_result_0.context_onehot,
    )
    controller_result_1 = agent.update(
        preparation_1.state,
        environment_result.reward,
        context_result_1.context_onehot,
    )
    candidate_base = CapacityPressureState(
        environment=environment_result.state,
        controller_0=controller_result_0.state,
        controller_1=controller_result_1.state,
        context_0=context_result_0.state,
        context_1=context_result_1.state,
        ledger_0=proposed_ledger_0,
        ledger_1=proposed_ledger_1,
    )
    candidate_state = LineageCacheRetentionState(
        base=candidate_base,
        lineage_0=lineage_result_0.state,
        lineage_1=lineage_result_1.state,
    )
    candidate_clocks_aligned = _clocks_aligned(candidate_base)
    candidate_state_finite = _tree_finite(candidate_state)
    context_updates_proposed = jnp.stack(
        (context_result_0.update_applied, context_result_1.update_applied)
    ).astype(jnp.bool_)
    controller_updates_proposed = jnp.stack(
        (controller_result_0.update_applied, controller_result_1.update_applied)
    ).astype(jnp.bool_)
    lineage_updates_proposed = jnp.stack(
        (lineage_result_0.update_applied, lineage_result_1.update_applied)
    ).astype(jnp.bool_)
    source_children_valid = (
        environment_result.state_valid
        & context_result_0.source_state_valid
        & context_result_1.source_state_valid
        & controller_result_0.state_valid
        & controller_result_1.state_valid
        & lineage_source_valid_0
        & lineage_source_valid_1
    )
    candidate_children_valid = (
        context_result_0.candidate_state_valid
        & context_result_1.candidate_state_valid
        & controller_result_0.candidate_state_finite
        & controller_result_1.candidate_state_finite
        & lineage_result_0.candidate_valid
        & lineage_result_1.candidate_valid
    )
    update_applied = (
        source_clocks_aligned
        & source_state_finite
        & source_children_valid
        & environment_result.update_applied
        & jnp.all(context_updates_proposed)
        & preparation_0.preparation_valid
        & preparation_1.preparation_valid
        & jnp.all(controller_updates_proposed)
        & jnp.all(lineage_updates_proposed)
        & candidate_clocks_aligned
        & candidate_state_finite
        & candidate_children_valid
    )
    committed_state = jax.lax.cond(
        update_applied,
        lambda _: candidate_state,
        lambda _: state,
        operand=None,
    )
    committed_base = committed_state.base
    switches = jnp.stack((event_0[0], event_1[0])) & update_applied
    allocations = jnp.stack((event_0[1], event_1[1])) & update_applied
    evictions = jnp.stack((event_0[2], event_1[2])) & update_applied
    reuses = jnp.stack((event_0[3], event_1[3])) & update_applied
    post_context_slots = jnp.stack(
        (
            committed_base.context_0.active_context,
            committed_base.context_1.active_context,
        )
    ).astype(jnp.int32)
    post_context_birth_words = jnp.stack(
        (
            committed_base.ledger_0.slot_birth_words[
                committed_base.context_0.active_context
            ],
            committed_base.ledger_1.slot_birth_words[
                committed_base.context_1.active_context
            ],
        )
    ).astype(jnp.uint32)
    capacity_trace = CapacityPressureStepTrace(
        reward=jnp.where(
            update_applied,
            environment_result.reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        actions=actions,
        pre_context_slots=pre_context_slots,
        post_context_slots=post_context_slots,
        pre_context_birth_words=pre_context_birth_words,
        post_context_birth_words=post_context_birth_words,
        switches=switches,
        allocations=allocations,
        evictions=evictions,
        reuses=reuses,
        contexts_in_use=jnp.stack(
            (
                context.num_contexts_in_use(committed_base.context_0),
                context.num_contexts_in_use(committed_base.context_1),
            )
        ).astype(jnp.int32),
        environment_update_proposed=environment_result.update_applied,
        context_updates_proposed=context_updates_proposed,
        controller_updates_proposed=controller_updates_proposed,
        source_clocks_aligned=source_clocks_aligned,
        candidate_clocks_aligned=candidate_clocks_aligned,
        source_state_finite=source_state_finite,
        candidate_state_finite=candidate_state_finite,
        update_applied=update_applied,
        pre_step_words=base.environment.step_words,
        post_step_words=committed_base.environment.step_words,
        controller_rng_key_words=jnp.stack(
            (
                jr.key_data(committed_base.controller_0.rng_key),
                jr.key_data(committed_base.controller_1.rng_key),
            )
        ).astype(jnp.uint32),
        controller_next_q_values=jnp.where(
            update_applied,
            jnp.stack((controller_result_0.q_values, controller_result_1.q_values)),
            jnp.zeros((2, N_ACTIONS), dtype=jnp.float32),
        ),
        controller_next_actions=jnp.where(
            update_applied,
            jnp.stack((controller_result_0.action, controller_result_1.action)),
            actions,
        ).astype(jnp.int32),
    )

    context_results = (context_result_0, context_result_1)
    lineage_results = (lineage_result_0, lineage_result_1)
    preparations = (preparation_0, preparation_1)

    def stack_context_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(result, name)) for result in context_results)
        )

    def stack_lineage_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(result, name)) for result in lineage_results)
        )

    def stack_preparation_field(name: str) -> Array:
        return jnp.stack(
            tuple(jnp.asarray(getattr(result, name)) for result in preparations)
        )

    raw_scores = jnp.stack((raw_score_0, raw_score_1)).astype(jnp.float32)
    dispatched = jnp.stack((dispatched_0, dispatched_1)).astype(jnp.float32)
    selected_slots = stack_context_field("selected_eviction_slot").astype(jnp.int32)
    safe_selected = jnp.clip(selected_slots, 0, MAX_CONTEXTS - 1)
    ordinary_slots = stack_context_field("ordinary_lru_slot").astype(jnp.int32)
    safe_ordinary = jnp.clip(ordinary_slots, 0, MAX_CONTEXTS - 1)
    applied_priority = (
        stack_context_field("eviction_protection_used").astype(jnp.bool_)
        & update_applied
    )
    trace = LineageCacheRetentionTrace(
        capacity=capacity_trace,
        protection_enabled=jnp.asarray(protection_enabled, dtype=jnp.bool_),
        source_scores_fixed_before_outcome=jnp.asarray(True, dtype=jnp.bool_),
        outcome_routed_to_current_protection=jnp.asarray(False, dtype=jnp.bool_),
        raw_predictive_rescue_scores=raw_scores,
        dispatched_eviction_protection=dispatched,
        pre_live_lineage_words=jnp.stack(
            (state.lineage_0.live_lineage_words, state.lineage_1.live_lineage_words)
        ).astype(jnp.uint32),
        post_live_lineage_words=jnp.stack(
            (
                committed_state.lineage_0.live_lineage_words,
                committed_state.lineage_1.live_lineage_words,
            )
        ).astype(jnp.uint32),
        pre_live_rescue_words=jnp.stack(
            (state.lineage_0.live_rescue_words, state.lineage_1.live_rescue_words)
        ).astype(jnp.uint32),
        post_live_rescue_words=jnp.stack(
            (
                committed_state.lineage_0.live_rescue_words,
                committed_state.lineage_1.live_rescue_words,
            )
        ).astype(jnp.uint32),
        pre_cache_valid=jnp.stack(
            (state.lineage_0.cache_valid, state.lineage_1.cache_valid)
        ).astype(jnp.bool_),
        post_cache_valid=jnp.stack(
            (committed_state.lineage_0.cache_valid, committed_state.lineage_1.cache_valid)
        ).astype(jnp.bool_),
        pre_cache_source_birth_words=jnp.stack(
            (
                state.lineage_0.cache_source_birth_words,
                state.lineage_1.cache_source_birth_words,
            )
        ).astype(jnp.uint32),
        post_cache_source_birth_words=jnp.stack(
            (
                committed_state.lineage_0.cache_source_birth_words,
                committed_state.lineage_1.cache_source_birth_words,
            )
        ).astype(jnp.uint32),
        pre_cache_lineage_words=jnp.stack(
            (state.lineage_0.cache_lineage_words, state.lineage_1.cache_lineage_words)
        ).astype(jnp.uint32),
        post_cache_lineage_words=jnp.stack(
            (
                committed_state.lineage_0.cache_lineage_words,
                committed_state.lineage_1.cache_lineage_words,
            )
        ).astype(jnp.uint32),
        pre_cache_rescue_words=jnp.stack(
            (state.lineage_0.cache_rescue_words, state.lineage_1.cache_rescue_words)
        ).astype(jnp.uint32),
        post_cache_rescue_words=jnp.stack(
            (
                committed_state.lineage_0.cache_rescue_words,
                committed_state.lineage_1.cache_rescue_words,
            )
        ).astype(jnp.uint32),
        cache_tested=stack_lineage_field("cache_tested").astype(jnp.bool_),
        strict_predictive_dominance=stack_lineage_field(
            "strict_predictive_dominance"
        ).astype(jnp.bool_),
        cache_matched=(
            stack_lineage_field("cache_matched").astype(jnp.bool_) & update_applied
        ),
        lineage_transferred=(
            stack_lineage_field("lineage_transferred").astype(jnp.bool_)
            & update_applied
        ),
        rescue_incremented=(
            stack_lineage_field("rescue_incremented").astype(jnp.bool_)
            & update_applied
        ),
        victim_archived=(
            stack_lineage_field("victim_archived").astype(jnp.bool_)
            & update_applied
        ),
        old_cache_retained=(
            stack_lineage_field("old_cache_retained").astype(jnp.bool_)
            & update_applied
        ),
        lineage_source_valid=jnp.stack(
            (lineage_source_valid_0, lineage_source_valid_1)
        ).astype(jnp.bool_),
        lineage_candidate_valid=stack_lineage_field("candidate_valid").astype(
            jnp.bool_
        ),
        rescue_capacity_available=stack_lineage_field(
            "rescue_capacity_available"
        ).astype(jnp.bool_),
        lineage_updates_proposed=lineage_updates_proposed,
        cache_errors=stack_lineage_field("cache_error").astype(jnp.float32),
        fresh_errors=stack_lineage_field("fresh_error").astype(jnp.float32),
        live_errors=stack_lineage_field("live_errors").astype(jnp.float32),
        scrub_preparations_valid=stack_preparation_field("preparation_valid").astype(
            jnp.bool_
        ),
        scrub_applied=(
            stack_preparation_field("scrub_candidate_applied").astype(jnp.bool_)
            & update_applied
        ),
        full_bank_evictions_requested=stack_context_field(
            "full_bank_eviction_requested"
        ).astype(jnp.bool_),
        eviction_protection_used=applied_priority,
        eviction_targets_adjusted=(
            stack_context_field("eviction_target_adjusted").astype(jnp.bool_)
            & update_applied
        ),
        ordinary_lru_slots=ordinary_slots,
        protected_lru_slots=stack_context_field("protected_lru_slot").astype(jnp.int32),
        selected_eviction_slots=jnp.where(
            applied_priority,
            selected_slots,
            jnp.full((2,), -1, dtype=jnp.int32),
        ),
        ordinary_lru_predictive_rescue_scores=jnp.where(
            applied_priority,
            raw_scores[jnp.arange(2, dtype=jnp.int32), safe_ordinary],
            jnp.zeros((2,), dtype=jnp.float32),
        ),
        selected_predictive_rescue_scores=jnp.where(
            applied_priority,
            raw_scores[jnp.arange(2, dtype=jnp.int32), safe_selected],
            jnp.zeros((2,), dtype=jnp.float32),
        ),
    )
    return LineageCacheRetentionStepResult(state=committed_state, trace=trace)


def step_lineage_cache_retention(
    epsilon: float,
    condition: LineageCacheCondition,
    state: LineageCacheRetentionState,
) -> LineageCacheRetentionStepResult:
    """Test one fixed-root lineage-cache step without exposing a new root."""

    if condition not in LINEAGE_CACHE_CONDITIONS:
        raise ValueError("unknown lineage-cache condition")
    if not _lineage_cache_static_shapes_valid(state):
        raise ValueError("lineage-cache state has invalid static shapes or dtypes")
    return cast(
        LineageCacheRetentionStepResult,
        _step_lineage_cache_retention(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            condition,
            state,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _scan_lineage_cache_life(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: LineageCacheCondition,
    initial_state: LineageCacheRetentionState,
) -> tuple[LineageCacheRetentionState, LineageCacheRetentionTrace]:
    def scan_step(
        carry: LineageCacheRetentionState,
        _: None,
    ) -> tuple[LineageCacheRetentionState, LineageCacheRetentionTrace]:
        result = _step_lineage_cache_retention(agent, context, game, condition, carry)
        return result.state, result.trace

    return jax.lax.scan(scan_step, initial_state, xs=None, length=NUM_STEPS)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 5))
def _advance_lineage_cache_state(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    condition: LineageCacheCondition,
    state: LineageCacheRetentionState,
    num_steps: int,
) -> LineageCacheRetentionState:
    def advance(_: int, carry: LineageCacheRetentionState) -> LineageCacheRetentionState:
        return cast(
            LineageCacheRetentionState,
            _step_lineage_cache_retention(agent, context, game, condition, carry).state,
        )

    return cast(
        LineageCacheRetentionState,
        jax.lax.fori_loop(0, num_steps, advance, state),
    )


def advance_consumed_lineage_cache_retention_state(
    epsilon: float,
    condition: LineageCacheCondition,
    num_steps: int,
) -> LineageCacheRetentionState:
    """Advance one consumed-root prefix for causality and rollback tests."""

    if condition not in LINEAGE_CACHE_CONDITIONS:
        raise ValueError("unknown lineage-cache condition")
    if type(num_steps) is not int or not 0 <= num_steps <= NUM_STEPS:
        raise ValueError("num_steps must be an integer within the consumed life")
    return cast(
        LineageCacheRetentionState,
        _advance_lineage_cache_state(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            condition,
            initialize_lineage_cache_retention_state(epsilon),
            num_steps,
        ),
    )


def step_capacity_pressure(
    epsilon: float,
    state: CapacityPressureState,
) -> CapacityPressureStepResult:
    """Testable one-step surface for the fixed mechanism, with no new root."""

    return cast(
        CapacityPressureStepResult,
        _step_capacity_pressure(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            state,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 4))
def _advance_capacity_pressure_state(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    state: CapacityPressureState,
    num_steps: int,
) -> CapacityPressureState:
    def advance(_: int, carry: CapacityPressureState) -> CapacityPressureState:
        return cast(
            CapacityPressureState,
            _step_capacity_pressure(agent, context, game, carry).state,
        )

    return cast(
        CapacityPressureState,
        jax.lax.fori_loop(0, num_steps, advance, state),
    )


def advance_consumed_capacity_pressure_state(
    epsilon: float,
    num_steps: int,
) -> CapacityPressureState:
    """Advance a root-zero prefix for fail-closed intervention tests."""

    if type(num_steps) is not int or not 0 <= num_steps <= NUM_STEPS:
        raise ValueError("num_steps must be an integer within the consumed life")
    return cast(
        CapacityPressureState,
        _advance_capacity_pressure_state(
            DifferentialSARSAAgent(control_config(epsilon)),
            ContextInference(CONTEXT_CONFIG),
            RecurringConventionGame(GAME_CONFIG),
            initialize_capacity_pressure_state(epsilon),
            num_steps,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _scan_life(
    agent: DifferentialSARSAAgent,
    context: ContextInference,
    game: RecurringConventionGame,
    initial_state: CapacityPressureState,
) -> tuple[CapacityPressureState, CapacityPressureStepTrace]:
    def scan_step(
        state: CapacityPressureState,
        _: None,
    ) -> tuple[CapacityPressureState, CapacityPressureStepTrace]:
        result = _step_capacity_pressure(agent, context, game, state)
        return result.state, result.trace

    return jax.lax.scan(scan_step, initial_state, xs=None, length=NUM_STEPS)


def _resource_budget(state: CapacityPressureState) -> CapacityPressureResourceBudget:
    environment = measure_convention_game_state_nbytes(state.environment)
    controller = measure_differential_sarsa_state_nbytes(state.controller_0)
    context = measure_context_inference_state_nbytes(state.context_0)
    ledger = int(state.ledger_0.slot_birth_words.size) * int(
        state.ledger_0.slot_birth_words.dtype.itemsize
    )
    joint_agent = environment + 2 * controller + 2 * context
    joint_ledger = 2 * ledger
    joint_clock = (
        CONVENTION_GAME_EXACT_CLOCK_NBYTES
        + 2 * DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES
        + 2 * context_inference_clock_nbytes(MAX_CONTEXTS)
    )
    return CapacityPressureResourceBudget(
        environment_nbytes=environment,
        per_agent_controller_nbytes=controller,
        per_agent_context_nbytes=context,
        joint_agent_environment_nbytes=joint_agent,
        per_agent_evaluator_birth_ledger_nbytes=ledger,
        joint_evaluator_birth_ledger_nbytes=joint_ledger,
        total_scan_carry_nbytes=joint_agent + joint_ledger,
        environment_exact_clock_nbytes=CONVENTION_GAME_EXACT_CLOCK_NBYTES,
        per_agent_controller_clock_nbytes=DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES,
        per_agent_context_clock_nbytes=context_inference_clock_nbytes(MAX_CONTEXTS),
        joint_agent_environment_clock_nbytes=joint_clock,
        max_context_slots_per_agent=MAX_CONTEXTS,
        replay_capacity=0,
        fixed_shape=True,
    )


def run_consumed_calibration_arm(epsilon: float) -> CapacityPressureRun:
    """Run one predeclared epsilon arm on already-consumed root zero."""

    agent = DifferentialSARSAAgent(control_config(epsilon))
    context = ContextInference(CONTEXT_CONFIG)
    game = RecurringConventionGame(GAME_CONFIG)
    initial_state = initialize_capacity_pressure_state(epsilon)
    initial_rng_words = jnp.stack(
        (
            jr.key_data(initial_state.controller_0.rng_key),
            jr.key_data(initial_state.controller_1.rng_key),
        )
    ).astype(jnp.uint32)
    final_state, trace = _scan_life(agent, context, game, initial_state)
    return CapacityPressureRun(
        epsilon=epsilon,
        root=CALIBRATION_ROOT,
        initial_controller_rng_key_words=initial_rng_words,
        trace=trace,
        final_state=final_state,
        resource_budget=_resource_budget(initial_state),
        work_budget=WORK_BUDGET,
    )


def _mode_rows(words: np.ndarray) -> np.ndarray:
    candidates = np.asarray(words)
    counts = np.sum(np.all(candidates[:, None, :] == candidates[None, :, :], axis=-1), axis=1)
    return np.asarray(candidates[int(np.argmax(counts))])


def _birth_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            np.asarray(left, dtype=np.uint32),
            np.asarray(right, dtype=np.uint32),
        )
    )


def summarize_capacity_pressure_run(run: CapacityPressureRun) -> CapacityPressureSummary:
    """Summarize rewards and semantic births without performance thresholds."""

    rewards = run.trace.reward.reshape(NUM_PHASES, PHASE_LENGTH)
    early = jnp.mean(rewards[:, :SUMMARY_WINDOW], axis=1)
    tail = jnp.mean(rewards[:, -SUMMARY_WINDOW:], axis=1)
    slots = np.asarray(run.trace.post_context_slots).reshape(
        NUM_PHASES,
        PHASE_LENGTH,
        2,
    )
    births = np.asarray(run.trace.post_context_birth_words).reshape(
        NUM_PHASES,
        PHASE_LENGTH,
        2,
        2,
    )
    slot_modes = np.zeros((NUM_PHASES, 2), dtype=np.int32)
    birth_modes = np.zeros((NUM_PHASES, 2, 2), dtype=np.uint32)
    distinct_births = np.zeros((NUM_PHASES, 2), dtype=np.int32)
    for phase in range(NUM_PHASES):
        for agent_index in range(2):
            tail_slots = slots[phase, -SUMMARY_WINDOW:, agent_index]
            slot_modes[phase, agent_index] = int(
                np.argmax(np.bincount(tail_slots, minlength=MAX_CONTEXTS))
            )
            tail_births = births[phase, -SUMMARY_WINDOW:, agent_index, :]
            birth_modes[phase, agent_index, :] = _mode_rows(tail_births)
            distinct_births[phase, agent_index] = len(
                {tuple(int(value) for value in row) for row in births[phase, :, agent_index, :]}
            )

    # Final occurrences are A=9, B=7, C=8, D=3.  These are evaluator labels,
    # never learner inputs.  Agent namespace is the separate axis; cross-agent
    # equality of the two-word values has no semantic interpretation.
    final_phases = (9, 7, 8, 3)
    final_label_modes = birth_modes[np.asarray(final_phases, dtype=np.int32)]
    final_abc_distinct = np.zeros((2,), dtype=np.bool_)
    recurrence_reuse = np.zeros((3, 2), dtype=np.bool_)
    for agent_index in range(2):
        abc = [tuple(int(value) for value in row) for row in final_label_modes[:3, agent_index]]
        final_abc_distinct[agent_index] = len(set(abc)) == 3
        recurrence_reuse[0, agent_index] = any(
            _birth_equal(birth_modes[9, agent_index], birth_modes[phase, agent_index])
            for phase in (0, 2, 4, 6)
        )
        recurrence_reuse[1, agent_index] = _birth_equal(
            birth_modes[7, agent_index],
            birth_modes[1, agent_index],
        )
        recurrence_reuse[2, agent_index] = _birth_equal(
            birth_modes[8, agent_index],
            birth_modes[5, agent_index],
        )

    def phase_counts(values: Array) -> Array:
        return jnp.sum(values.reshape(NUM_PHASES, PHASE_LENGTH, 2), axis=1).astype(
            jnp.int32
        )

    return CapacityPressureSummary(
        epsilon=run.epsilon,
        phase_early_reward=early,
        phase_tail_reward=tail,
        tail_context_slot_modes=jnp.asarray(slot_modes),
        tail_context_birth_modes=jnp.asarray(birth_modes),
        phase_switch_counts=phase_counts(run.trace.switches),
        phase_allocation_counts=phase_counts(run.trace.allocations),
        phase_eviction_counts=phase_counts(run.trace.evictions),
        phase_reuse_counts=phase_counts(run.trace.reuses),
        phase_distinct_birth_counts=jnp.asarray(distinct_births),
        final_label_birth_modes=jnp.asarray(final_label_modes),
        final_abc_births_distinct=jnp.asarray(final_abc_distinct),
        recurrence_birth_reuse=jnp.asarray(recurrence_reuse),
        overall_reward=float(jnp.mean(run.trace.reward)),
    )


def _rng_stream_digest(run: CapacityPressureRun, agent_index: int) -> str:
    initial = np.asarray(run.initial_controller_rng_key_words[agent_index], dtype=np.uint32)
    rest = np.asarray(
        run.trace.controller_rng_key_words[:, agent_index, :],
        dtype=np.uint32,
    )
    stream = np.concatenate((initial[None, :], rest), axis=0).astype(">u4", copy=False)
    return hashlib.sha256(stream.tobytes()).hexdigest()


def build_prefix_twin_boundary() -> PrefixTwinBoundary:
    """Build, but do not execute, the prefix-identical B-vs-D future twin."""

    counterfactual = (0, 1, 0, 3, 0, 2, 0, 3, 2, 0)
    common_prefix_phases = 7
    actual_prefix = OFFSETS[:common_prefix_phases]
    counterfactual_prefix = counterfactual[:common_prefix_phases]
    if actual_prefix != counterfactual_prefix:
        raise AssertionError("prefix-twin schedules do not share the declared prefix")
    payload = {
        "n_actions": N_ACTIONS,
        "phase_length": PHASE_LENGTH,
        "common_prefix_offsets": list(actual_prefix),
        "learner_observation": "plain; post-action partner one-hot/action/reward only",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PrefixTwinBoundary(
        actual_schedule=OFFSETS,
        counterfactual_schedule=counterfactual,
        common_prefix_phases=common_prefix_phases,
        common_prefix_offsets=actual_prefix,
        common_prefix_sha256=digest,
        actual_prefix_sha256=digest,
        counterfactual_prefix_sha256=digest,
        first_divergent_phase=7,
        differing_phase_indices=tuple(
            index
            for index, (actual, twin) in enumerate(zip(OFFSETS, counterfactual, strict=True))
            if actual != twin
        ),
        actual_divergent_offset=1,
        counterfactual_divergent_offset=3,
        regimes_seen_by_first_c_admission=("A", "B", "D", "C"),
        capacity=MAX_CONTEXTS,
        actual_zero_recurrence_loss_set=("A", "B", "C"),
        counterfactual_zero_recurrence_loss_set=("A", "D", "C"),
        correct_actual_eviction="D",
        correct_counterfactual_eviction="B",
        same_policy_rng_implies_identical_prefix_history=True,
        future_schedule_only_divergence=True,
        deterministic_online_guarantee_possible=False,
        future_schedule_revealed_to_learners=False,
        counterfactual_future_executed=False,
        stochastic_optimality_claimed=False,
        conclusion=(
            "B and D are observationally indistinguishable as future-recurrence "
            "targets at the capacity decision: a deterministic bounded online "
            "policy must make the same choice on both prefix twins, while the "
            "zero-recurrence-loss eviction flips. A prior or later evidence is "
            "required; no first-eviction guarantee follows from the prefix."
        ),
    )


NEXT_FROZEN_PANEL = FrozenPanelDesign(
    status="designed-not-issued-not-executed",
    root_zero_excluded=True,
    epsilon_arms=EPSILON_GRID,
    primary_endpoints=(
        "per-agent semantic-birth reuse for recurrent A/B/C",
        "phase-early reward on recurrent A/B/C occurrences",
        "phase-tail reward and early-to-tail recovery gap",
    ),
    causal_diagnostics=(
        "capacity-4 storage ceiling, labeled diagnostic only",
        "matched-work inference-unrouted control ablation",
        "matched-work birth-authenticated controller-row scrub/rebind ablation",
        "allocation, eviction, stored-reuse, and within-phase birth churn",
    ),
    required_protocol_actions=(
        "freeze untouched namespaced roots before execution",
        "predeclare sample count, estimands, intervals, and multiplicity handling",
        "retain all four calibration arms without selecting epsilon from root zero",
        "write any future artifact only to a new versioned path after registration",
    ),
    promotion_claimed=False,
)


def validate_static_contract() -> tuple[str, ...]:
    """Fail closed on configuration, causal-channel, and execution surfaces."""

    errors: list[str] = []
    if PROTOCOL.num_steps != NUM_STEPS or PROTOCOL.offsets != OFFSETS:
        errors.append("protocol shape drifted")
    if PROTOCOL.max_contexts != 3 or len(set(OFFSETS)) != 4:
        errors.append("the three-slot/four-rule capacity pressure was lost")
    if GAME_CONFIG.feature_mode != "plain" or GAME_CONFIG.observation_dim != 1:
        errors.append("the game exposes a rule-derived observation")
    learner_channels = set(LEARNER_PRE_ACTION_CHANNELS) | set(
        LEARNER_POST_ACTION_CHANNELS
    )
    if learner_channels & FORBIDDEN_LEARNER_CHANNELS:
        errors.append("a forbidden learner channel is declared")
    if (
        not DEVELOPMENT_ONLY
        or SCIENTIFIC_PROMOTION_ALLOWED
        or OUTPUT_WRITES_ALLOWED
        or ARBITRARY_ROOT_EXECUTION_ALLOWED
        or not CALIBRATION_ROOT_CONSUMED
    ):
        errors.append("development-only consumed-root guard drifted")
    source = inspect.getsource(_step_capacity_pressure)
    for forbidden_call in (".rule_of(", ".phase_index_of(", ".observe("):
        if forbidden_call in source:
            errors.append(f"learner transaction contains forbidden call {forbidden_call}")
    if "future" in inspect.signature(run_consumed_calibration_arm).parameters:
        errors.append("runner exposes a future-schedule input")
    if CALIBRATION_ROOT.index != 0 or CALIBRATION_ROOT.key_seed != 0:
        errors.append("calibration root drifted")
    if tuple(control_config(value).epsilon_start for value in EPSILON_GRID) != EPSILON_GRID:
        errors.append("epsilon grid/config mapping drifted")
    if CONTEXT_CONFIG != ContextInferenceConfig(
        n_actions=4,
        observation_dim=4,
        max_contexts=3,
    ):
        errors.append("context-inference configuration drifted")
    if GAME_CONFIG != ConventionGameConfig(
        n_actions=4,
        phase_length=400,
        offsets=OFFSETS,
        feature_mode="plain",
    ):
        errors.append("game configuration drifted")
    if POST_AUDIT_CONDITIONS != (
        POST_AUDIT_BASELINE,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
    ):
        errors.append("post-audit paired conditions drifted")
    if not POST_AUDIT_ONLY or POST_AUDIT_WORK_BUDGET.replay_updates != 0:
        errors.append("post-audit nonpromotion/work guard drifted")
    if SELECTIVE_RETENTION_CONDITIONS != (
        SELECTIVE_RETENTION_NO_SIGNAL,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
    ):
        errors.append("selective-retention paired conditions drifted")
    if (
        not SELECTIVE_RETENTION_DEVELOPMENT_ONLY
        or SELECTIVE_RETENTION_WORK_BUDGET.replay_updates != 0
        or not SELECTIVE_RETENTION_WORK_BUDGET.no_signal_still_computes_history
    ):
        errors.append("selective-retention development/work guard drifted")
    selective_source = inspect.getsource(_step_selective_retention)
    for forbidden_call in (".rule_of(", ".phase_index_of(", ".observe("):
        if forbidden_call in selective_source:
            errors.append(
                f"selective-retention transaction contains forbidden call {forbidden_call}"
            )
    if LINEAGE_CACHE_CONDITIONS != (
        LINEAGE_CACHE_NO_SIGNAL,
        LINEAGE_CACHE_PREDICTIVE_RESCUE,
    ):
        errors.append("lineage-cache paired conditions drifted")
    if (
        LINEAGE_CACHE_CAPACITY != 1
        or not LINEAGE_CACHE_DEVELOPMENT_ONLY
        or LINEAGE_CACHE_WORK_BUDGET.replay_updates != 0
        or not LINEAGE_CACHE_WORK_BUDGET.no_signal_still_computes_cache_match
    ):
        errors.append("lineage-cache bounded development/work guard drifted")
    lineage_source = inspect.getsource(_step_lineage_cache_retention)
    for forbidden_call in (".rule_of(", ".phase_index_of(", ".observe("):
        if forbidden_call in lineage_source:
            errors.append(
                f"lineage-cache transaction contains forbidden call {forbidden_call}"
            )
    return tuple(errors)


def _build_consumed_calibration_panel(
    runs: tuple[CapacityPressureRun, ...],
) -> CapacityPressurePanel:
    summaries = tuple(summarize_capacity_pressure_run(run) for run in runs)
    first_resource = runs[0].resource_budget.to_dict()
    first_work = runs[0].work_budget.to_dict()
    resources_matched = all(run.resource_budget.to_dict() == first_resource for run in runs)
    work_matched = all(run.work_budget.to_dict() == first_work for run in runs)
    stream_digests = tuple(_rng_stream_digest(runs[0], agent) for agent in range(2))
    key_streams_equal = all(
        tuple(_rng_stream_digest(run, agent) for agent in range(2)) == stream_digests
        for run in runs
    )
    crn = CommonRandomNumberAudit(
        root_index=CALIBRATION_ROOT_INDEX,
        agent_key_stream_sha256=cast(tuple[str, str], stream_digests),
        key_streams_equal_across_arms=key_streams_equal,
        selection_calls_per_agent=NUM_STEPS + 1,
        branch_independent_key_advance=True,
        environment_randomness_consumed=ENVIRONMENT_RANDOMNESS_CONSUMED,
    )
    return CapacityPressurePanel(
        runs=runs,
        summaries=summaries,
        common_random_numbers=crn,
        prefix_twin_boundary=build_prefix_twin_boundary(),
        resources_matched=resources_matched,
        work_matched=work_matched,
        selection_performed=False,
        selected_epsilon=None,
        causal_gap=(
            "Root-zero behavior can describe the exploration/retention tradeoff, "
            "but it cannot identify a generally correct eviction rule. The bank "
            "has no prospective recurrence signal, and control is routed by a "
            "recyclable slot rather than an authenticated birth, so replacement "
            "can expose a new context to stale Q parameters. Semantic birth reuse "
            "is not task-label inference, reward recovery is not catastrophic-"
            "forgetting immunity, and the prefix-twin boundary makes the first "
            "B-versus-D eviction unknowable without a prior. Untouched roots and "
            "causal ceilings/ablations remain required."
        ),
        next_frozen_panel=NEXT_FROZEN_PANEL,
    )


def run_consumed_calibration_panel() -> CapacityPressurePanel:
    """Run the full predeclared grid once on consumed root zero, without selection."""

    static_errors = validate_static_contract()
    if static_errors:
        raise ValueError("invalid capacity-pressure contract: " + "; ".join(static_errors))
    runs = tuple(run_consumed_calibration_arm(epsilon) for epsilon in EPSILON_GRID)
    return _build_consumed_calibration_panel(runs)


def _run_post_audit_condition(
    epsilon: float,
    condition: PostAuditCondition,
) -> PostAuditRun:
    agent = DifferentialSARSAAgent(control_config(epsilon))
    context = ContextInference(CONTEXT_CONFIG)
    game = RecurringConventionGame(GAME_CONFIG)
    initial_state = initialize_capacity_pressure_state(epsilon)
    initial_rng_words = jnp.stack(
        (
            jr.key_data(initial_state.controller_0.rng_key),
            jr.key_data(initial_state.controller_1.rng_key),
        )
    ).astype(jnp.uint32)
    final_state, (trace, scrub) = _scan_post_audit_life(
        agent,
        context,
        game,
        condition,
        initial_state,
    )
    base_resource = _resource_budget(initial_state)
    capacity_run = CapacityPressureRun(
        epsilon=epsilon,
        root=CALIBRATION_ROOT,
        initial_controller_rng_key_words=initial_rng_words,
        trace=trace,
        final_state=final_state,
        resource_budget=base_resource,
        work_budget=WORK_BUDGET,
    )
    return PostAuditRun(
        condition=condition,
        capacity_run=capacity_run,
        scrub=scrub,
        resource_budget=PostAuditResourceBudget(
            base=base_resource,
            logical_transient_scrub_candidate_nbytes=(
                2 * base_resource.per_agent_controller_nbytes
            ),
        ),
        work_budget=POST_AUDIT_WORK_BUDGET,
    )


def _post_audit_effect(pair: PostAuditPair) -> PostAuditEffect:
    baseline_summary = summarize_capacity_pressure_run(pair.baseline.capacity_run)
    scrub_summary = summarize_capacity_pressure_run(pair.scrub.capacity_run)
    baseline_scrub_count = int(jnp.sum(pair.baseline.scrub.scrub_applied))
    scrub_scrub_count = int(jnp.sum(pair.scrub.scrub.scrub_applied))
    baseline_contamination = int(
        jnp.sum(pair.baseline.scrub.cross_birth_contamination_consumed)
    )
    scrub_contamination = int(
        jnp.sum(pair.scrub.scrub.cross_birth_contamination_consumed)
    )
    scrub_prevented = int(
        jnp.sum(pair.scrub.scrub.cross_birth_contamination_prevented)
    )
    return PostAuditEffect(
        epsilon=pair.epsilon,
        baseline_overall_reward=baseline_summary.overall_reward,
        scrub_overall_reward=scrub_summary.overall_reward,
        scrub_minus_baseline_overall_reward=(
            scrub_summary.overall_reward - baseline_summary.overall_reward
        ),
        baseline_phase_early_reward=baseline_summary.phase_early_reward,
        scrub_phase_early_reward=scrub_summary.phase_early_reward,
        baseline_phase_tail_reward=baseline_summary.phase_tail_reward,
        scrub_phase_tail_reward=scrub_summary.phase_tail_reward,
        baseline_tail_birth_modes=baseline_summary.tail_context_birth_modes,
        scrub_tail_birth_modes=scrub_summary.tail_context_birth_modes,
        baseline_phase_distinct_birth_counts=(
            baseline_summary.phase_distinct_birth_counts
        ),
        scrub_phase_distinct_birth_counts=scrub_summary.phase_distinct_birth_counts,
        baseline_switch_counts=baseline_summary.phase_switch_counts,
        scrub_switch_counts=scrub_summary.phase_switch_counts,
        baseline_allocation_counts=baseline_summary.phase_allocation_counts,
        scrub_allocation_counts=scrub_summary.phase_allocation_counts,
        baseline_eviction_counts=baseline_summary.phase_eviction_counts,
        scrub_eviction_counts=scrub_summary.phase_eviction_counts,
        baseline_reuse_counts=baseline_summary.phase_reuse_counts,
        scrub_reuse_counts=scrub_summary.phase_reuse_counts,
        baseline_scrub_count=baseline_scrub_count,
        scrub_scrub_count=scrub_scrub_count,
        baseline_contamination_count=baseline_contamination,
        scrub_contamination_count=scrub_contamination,
        scrub_prevented_count=scrub_prevented,
        cross_birth_contamination_removed=(scrub_contamination == 0),
    )


def run_post_audit_paired_intervention() -> PostAuditPairedPanel:
    """Run exactly eight root-zero lives: four epsilons by two paired conditions."""

    static_errors = validate_static_contract()
    if static_errors:
        raise ValueError("invalid capacity-pressure contract: " + "; ".join(static_errors))
    runs = tuple(
        _run_post_audit_condition(epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in POST_AUDIT_CONDITIONS
    )
    pairs = tuple(
        PostAuditPair(
            epsilon=epsilon,
            baseline=runs[2 * index],
            scrub=runs[2 * index + 1],
        )
        for index, epsilon in enumerate(EPSILON_GRID)
    )
    baseline_panel = _build_consumed_calibration_panel(
        tuple(pair.baseline.capacity_run for pair in pairs)
    )
    effects = tuple(_post_audit_effect(pair) for pair in pairs)
    first_resource = runs[0].resource_budget.to_dict()
    first_work = runs[0].work_budget.to_dict()
    resources_matched = all(
        run.resource_budget.to_dict() == first_resource for run in runs
    )
    work_matched = all(run.work_budget.to_dict() == first_work for run in runs)
    first_streams = tuple(
        _rng_stream_digest(runs[0].capacity_run, agent_index)
        for agent_index in range(2)
    )
    key_streams_equal = all(
        tuple(
            _rng_stream_digest(run.capacity_run, agent_index)
            for agent_index in range(2)
        )
        == first_streams
        for run in runs
    )
    return PostAuditPairedPanel(
        runs=runs,
        pairs=pairs,
        effects=effects,
        baseline_calibration_panel=baseline_panel,
        common_random_numbers=PostAuditCommonRandomNumberAudit(
            root_index=CALIBRATION_ROOT_INDEX,
            key_streams_equal_across_all_eight=key_streams_equal,
            branch_independent_key_advance=True,
            selection_calls_per_agent=NUM_STEPS + 1,
            environment_randomness_consumed=ENVIRONMENT_RANDOMNESS_CONSUMED,
        ),
        resources_matched=resources_matched,
        work_matched=work_matched,
        post_audit_only=POST_AUDIT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        thresholds_used=False,
        selection_performed=False,
        selected_epsilon=None,
        conclusion=(
            "Descriptive consumed-root post-audit only: the paired intervention "
            "tests whether authenticated destination-row scrubbing removes stale "
            "Q consumption at semantic births. Reward, identity, and churn effects "
            "are reported for every epsilon; no threshold, winner, promotion, or "
            "new-root inference is authorized."
        ),
    )


def _measure_recurrence_history_nbytes(history: BirthRecurrenceHistoryState) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(history)
        if isinstance(leaf, Array)
    )


def _selective_retention_resource_budget(
    state: SelectiveRetentionState,
) -> SelectiveRetentionResourceBudget:
    base = _resource_budget(state.base)
    per_agent_history = _measure_recurrence_history_nbytes(state.recurrence_0)
    joint_history = per_agent_history + _measure_recurrence_history_nbytes(
        state.recurrence_1
    )
    return SelectiveRetentionResourceBudget(
        base=base,
        per_agent_recurrence_history_nbytes=per_agent_history,
        joint_recurrence_history_nbytes=joint_history,
        total_scan_carry_nbytes=base.total_scan_carry_nbytes + joint_history,
        # Raw and dispatched float32 vectors for both agents.
        logical_transient_protection_nbytes=2 * 2 * MAX_CONTEXTS * 4,
        logical_transient_scrub_candidate_nbytes=(
            2 * base.per_agent_controller_nbytes
        ),
    )


def _run_selective_retention_condition(
    epsilon: float,
    condition: SelectiveRetentionCondition,
) -> SelectiveRetentionRun:
    agent = DifferentialSARSAAgent(control_config(epsilon))
    context = ContextInference(CONTEXT_CONFIG)
    game = RecurringConventionGame(GAME_CONFIG)
    initial_state = initialize_selective_retention_state(epsilon)
    initial_rng_words = jnp.stack(
        (
            jr.key_data(initial_state.base.controller_0.rng_key),
            jr.key_data(initial_state.base.controller_1.rng_key),
        )
    ).astype(jnp.uint32)
    final_state, trace = _scan_selective_retention_life(
        agent,
        context,
        game,
        condition,
        initial_state,
    )
    base_resource = _resource_budget(initial_state.base)
    capacity_run = CapacityPressureRun(
        epsilon=epsilon,
        root=CALIBRATION_ROOT,
        initial_controller_rng_key_words=initial_rng_words,
        trace=trace.capacity,
        final_state=final_state.base,
        resource_budget=base_resource,
        work_budget=WORK_BUDGET,
    )
    return SelectiveRetentionRun(
        condition=condition,
        capacity_run=capacity_run,
        trace=trace,
        final_state=final_state,
        resource_budget=_selective_retention_resource_budget(initial_state),
        work_budget=SELECTIVE_RETENTION_WORK_BUDGET,
    )


def _selective_retention_effect(
    pair: SelectiveRetentionPair,
) -> SelectiveRetentionEffect:
    no_signal = summarize_capacity_pressure_run(pair.no_signal.capacity_run)
    recurrence = summarize_capacity_pressure_run(pair.past_recurrence.capacity_run)
    no_signal_trace = pair.no_signal.trace
    recurrence_trace = pair.past_recurrence.trace
    return SelectiveRetentionEffect(
        epsilon=pair.epsilon,
        no_signal_overall_reward=no_signal.overall_reward,
        past_recurrence_overall_reward=recurrence.overall_reward,
        past_recurrence_minus_no_signal_overall_reward=(
            recurrence.overall_reward - no_signal.overall_reward
        ),
        no_signal_phase_early_reward=no_signal.phase_early_reward,
        past_recurrence_phase_early_reward=recurrence.phase_early_reward,
        no_signal_phase_tail_reward=no_signal.phase_tail_reward,
        past_recurrence_phase_tail_reward=recurrence.phase_tail_reward,
        no_signal_tail_birth_modes=no_signal.tail_context_birth_modes,
        past_recurrence_tail_birth_modes=recurrence.tail_context_birth_modes,
        no_signal_phase_switch_counts=no_signal.phase_switch_counts,
        past_recurrence_phase_switch_counts=recurrence.phase_switch_counts,
        no_signal_phase_allocation_counts=no_signal.phase_allocation_counts,
        past_recurrence_phase_allocation_counts=recurrence.phase_allocation_counts,
        no_signal_phase_eviction_counts=no_signal.phase_eviction_counts,
        past_recurrence_phase_eviction_counts=recurrence.phase_eviction_counts,
        no_signal_phase_reuse_counts=no_signal.phase_reuse_counts,
        past_recurrence_phase_reuse_counts=recurrence.phase_reuse_counts,
        no_signal_full_bank_eviction_count=int(
            jnp.sum(no_signal_trace.full_bank_evictions_requested)
        ),
        past_recurrence_full_bank_eviction_count=int(
            jnp.sum(recurrence_trace.full_bank_evictions_requested)
        ),
        no_signal_adjusted_target_count=int(
            jnp.sum(no_signal_trace.eviction_targets_adjusted)
        ),
        past_recurrence_adjusted_target_count=int(
            jnp.sum(recurrence_trace.eviction_targets_adjusted)
        ),
        nonzero_selected_eviction_score_count=int(
            jnp.sum(recurrence_trace.selected_eviction_scores > 0.0)
        ),
        avoided_completed_recurrence_intervals=float(
            jnp.sum(
                jnp.where(
                    recurrence_trace.eviction_targets_adjusted,
                    recurrence_trace.ordinary_lru_completed_recurrence_scores
                    - recurrence_trace.selected_completed_recurrence_scores,
                    jnp.zeros((NUM_STEPS, 2), dtype=jnp.float32),
                )
            )
        ),
    )


def run_selective_retention_paired_intervention() -> SelectiveRetentionPairedPanel:
    """Run the fixed eight-life no-signal/past-recurrence development replay."""

    static_errors = validate_static_contract()
    if static_errors:
        raise ValueError("invalid capacity-pressure contract: " + "; ".join(static_errors))
    runs = tuple(
        _run_selective_retention_condition(epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in SELECTIVE_RETENTION_CONDITIONS
    )
    pairs = tuple(
        SelectiveRetentionPair(
            epsilon=epsilon,
            no_signal=runs[2 * index],
            past_recurrence=runs[2 * index + 1],
        )
        for index, epsilon in enumerate(EPSILON_GRID)
    )
    effects = tuple(_selective_retention_effect(pair) for pair in pairs)
    first_resource = runs[0].resource_budget.to_dict()
    first_work = runs[0].work_budget.to_dict()
    resources_matched = all(
        run.resource_budget.to_dict() == first_resource for run in runs
    )
    work_matched = all(run.work_budget.to_dict() == first_work for run in runs)
    first_streams = tuple(
        _rng_stream_digest(runs[0].capacity_run, agent_index)
        for agent_index in range(2)
    )
    key_streams_equal = all(
        tuple(
            _rng_stream_digest(run.capacity_run, agent_index)
            for agent_index in range(2)
        )
        == first_streams
        for run in runs
    )
    no_signal_contract = all(
        not bool(jnp.any(pair.no_signal.trace.dispatched_eviction_protection))
        and bool(jnp.all(pair.no_signal.trace.scrub.scrub_enabled))
        for pair in pairs
    )
    return SelectiveRetentionPairedPanel(
        runs=runs,
        pairs=pairs,
        effects=effects,
        common_random_numbers=SelectiveRetentionCommonRandomNumberAudit(
            root_index=CALIBRATION_ROOT_INDEX,
            key_streams_equal_across_all_eight=key_streams_equal,
            branch_independent_key_advance=True,
            selection_calls_per_agent=NUM_STEPS + 1,
            environment_randomness_consumed=ENVIRONMENT_RANDOMNESS_CONSUMED,
        ),
        resources_matched=resources_matched,
        work_matched=work_matched,
        no_signal_is_controller_scrub_baseline=no_signal_contract,
        past_only_score="authenticated_current_birth_occurrences_minus_one",
        # At the prefix twin's first C admission, B and D each have zero
        # completed recurrence intervals, so this parameter-free history has
        # no information with which to resolve their opposite future values.
        prefix_twin_first_eviction_resolved=False,
        development_only=SELECTIVE_RETENTION_DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        thresholds_used=False,
        selection_performed=False,
        selected_epsilon=None,
        conclusion=(
            "Consumed-root causal development replay only: both conditions "
            "use the same authenticated controller scrub and recurrence "
            "history; only the full-bank eviction protection vector differs. "
            "Effects are descriptive for all epsilon values. The first prefix-"
            "twin eviction remains unknowable, and no threshold, tuning, winner, "
            "promotion, or new-root inference is authorized."
        ),
    )


def _measure_lineage_cache_nbytes(state: ContextLineageCacheState) -> int:
    return sum(
        int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree.leaves(state)
    )


def _lineage_cache_resource_budget(
    state: LineageCacheRetentionState,
) -> LineageCacheResourceBudget:
    base = _resource_budget(state.base)
    per_agent = _measure_lineage_cache_nbytes(state.lineage_0)
    joint = per_agent + _measure_lineage_cache_nbytes(state.lineage_1)
    return LineageCacheResourceBudget(
        base=base,
        cache_capacity_per_agent=LINEAGE_CACHE_CAPACITY,
        per_agent_lineage_cache_nbytes=per_agent,
        joint_lineage_cache_nbytes=joint,
        total_scan_carry_nbytes=base.total_scan_carry_nbytes + joint,
        # Raw and dispatched float32 score vectors for both agents.
        logical_transient_protection_nbytes=2 * 2 * MAX_CONTEXTS * 4,
        # Cache/fresh/live predictions and their errors for both agents.
        logical_transient_prediction_nbytes=(
            2 * 2 * (LINEAGE_CACHE_CAPACITY + 1 + MAX_CONTEXTS) * 4
        ),
        logical_transient_lineage_candidate_nbytes=joint,
        logical_transient_scrub_candidate_nbytes=(
            2 * base.per_agent_controller_nbytes
        ),
    )


def _run_lineage_cache_condition(
    epsilon: float,
    condition: LineageCacheCondition,
) -> LineageCacheRun:
    agent = DifferentialSARSAAgent(control_config(epsilon))
    context = ContextInference(CONTEXT_CONFIG)
    game = RecurringConventionGame(GAME_CONFIG)
    initial_state = initialize_lineage_cache_retention_state(epsilon)
    initial_rng_words = jnp.stack(
        (
            jr.key_data(initial_state.base.controller_0.rng_key),
            jr.key_data(initial_state.base.controller_1.rng_key),
        )
    ).astype(jnp.uint32)
    final_state, trace = _scan_lineage_cache_life(
        agent,
        context,
        game,
        condition,
        initial_state,
    )
    base_resource = _resource_budget(initial_state.base)
    capacity_run = CapacityPressureRun(
        epsilon=epsilon,
        root=CALIBRATION_ROOT,
        initial_controller_rng_key_words=initial_rng_words,
        trace=trace.capacity,
        final_state=final_state.base,
        resource_budget=base_resource,
        work_budget=WORK_BUDGET,
    )
    return LineageCacheRun(
        condition=condition,
        capacity_run=capacity_run,
        trace=trace,
        final_state=final_state,
        resource_budget=_lineage_cache_resource_budget(initial_state),
        work_budget=LINEAGE_CACHE_WORK_BUDGET,
    )


def _predictive_rescue_failure_decomposition(
    trace: LineageCacheRetentionTrace,
) -> PredictiveRescueFailureDecomposition:
    tested = trace.cache_tested
    cache_error = trace.cache_errors
    strict_fresh = cache_error < trace.fresh_errors
    strict_all_live = jnp.all(cache_error[..., None] < trace.live_errors, axis=-1)
    fresh_tie = cache_error == trace.fresh_errors
    live_tie = jnp.any(cache_error[..., None] == trace.live_errors, axis=-1)
    return PredictiveRescueFailureDecomposition(
        full_bank_birth_count=int(jnp.sum(trace.full_bank_evictions_requested)),
        cache_valid_test_count=int(jnp.sum(tested)),
        failed_fresh_prior_count=int(jnp.sum(tested & ~strict_fresh)),
        failed_any_live_model_count=int(jnp.sum(tested & ~strict_all_live)),
        fresh_prior_tie_count=int(jnp.sum(tested & fresh_tie)),
        any_live_model_tie_count=int(jnp.sum(tested & live_tie)),
        any_exact_tie_count=int(jnp.sum(tested & (fresh_tie | live_tie))),
        victim_archive_count=int(jnp.sum(trace.victim_archived)),
        old_cache_retained_count=int(jnp.sum(trace.old_cache_retained)),
    )


def _lineage_cache_effect(pair: LineageCachePair) -> LineageCacheEffect:
    no_signal = summarize_capacity_pressure_run(pair.no_signal.capacity_run)
    rescue = summarize_capacity_pressure_run(pair.predictive_rescue.capacity_run)
    no_signal_trace = pair.no_signal.trace
    rescue_trace = pair.predictive_rescue.trace
    return LineageCacheEffect(
        epsilon=pair.epsilon,
        no_signal_overall_reward=no_signal.overall_reward,
        predictive_rescue_overall_reward=rescue.overall_reward,
        predictive_rescue_minus_no_signal_overall_reward=(
            rescue.overall_reward - no_signal.overall_reward
        ),
        no_signal_phase_early_reward=no_signal.phase_early_reward,
        predictive_rescue_phase_early_reward=rescue.phase_early_reward,
        no_signal_phase_tail_reward=no_signal.phase_tail_reward,
        predictive_rescue_phase_tail_reward=rescue.phase_tail_reward,
        no_signal_phase_switch_counts=no_signal.phase_switch_counts,
        predictive_rescue_phase_switch_counts=rescue.phase_switch_counts,
        no_signal_phase_allocation_counts=no_signal.phase_allocation_counts,
        predictive_rescue_phase_allocation_counts=rescue.phase_allocation_counts,
        no_signal_phase_eviction_counts=no_signal.phase_eviction_counts,
        predictive_rescue_phase_eviction_counts=rescue.phase_eviction_counts,
        no_signal_phase_reuse_counts=no_signal.phase_reuse_counts,
        predictive_rescue_phase_reuse_counts=rescue.phase_reuse_counts,
        no_signal_cache_match_count=int(jnp.sum(no_signal_trace.cache_matched)),
        predictive_rescue_cache_match_count=int(jnp.sum(rescue_trace.cache_matched)),
        no_signal_rescue_increment_count=int(
            jnp.sum(no_signal_trace.rescue_incremented)
        ),
        predictive_rescue_rescue_increment_count=int(
            jnp.sum(rescue_trace.rescue_incremented)
        ),
        no_signal_adjusted_target_count=int(
            jnp.sum(no_signal_trace.eviction_targets_adjusted)
        ),
        predictive_rescue_adjusted_target_count=int(
            jnp.sum(rescue_trace.eviction_targets_adjusted)
        ),
        nonzero_selected_predictive_rescue_count=int(
            jnp.sum(rescue_trace.selected_predictive_rescue_scores > 0.0)
        ),
        avoided_predictive_rescues=float(
            jnp.sum(
                jnp.where(
                    rescue_trace.eviction_targets_adjusted,
                    rescue_trace.ordinary_lru_predictive_rescue_scores
                    - rescue_trace.selected_predictive_rescue_scores,
                    jnp.zeros((NUM_STEPS, 2), dtype=jnp.float32),
                )
            )
        ),
        no_signal_failure_decomposition=_predictive_rescue_failure_decomposition(
            no_signal_trace
        ),
        predictive_rescue_failure_decomposition=(
            _predictive_rescue_failure_decomposition(rescue_trace)
        ),
    )


def run_lineage_cache_paired_intervention() -> LineageCachePairedPanel:
    """Run exactly eight fixed-root lives with one predictive victim record."""

    static_errors = validate_static_contract()
    if static_errors:
        raise ValueError("invalid capacity-pressure contract: " + "; ".join(static_errors))
    runs = tuple(
        _run_lineage_cache_condition(epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in LINEAGE_CACHE_CONDITIONS
    )
    pairs = tuple(
        LineageCachePair(
            epsilon=epsilon,
            no_signal=runs[2 * index],
            predictive_rescue=runs[2 * index + 1],
        )
        for index, epsilon in enumerate(EPSILON_GRID)
    )
    effects = tuple(_lineage_cache_effect(pair) for pair in pairs)
    first_resource = runs[0].resource_budget.to_dict()
    first_work = runs[0].work_budget.to_dict()
    resources_matched = all(
        run.resource_budget.to_dict() == first_resource for run in runs
    )
    work_matched = all(run.work_budget.to_dict() == first_work for run in runs)
    first_streams = tuple(
        _rng_stream_digest(runs[0].capacity_run, agent_index)
        for agent_index in range(2)
    )
    key_streams_equal = all(
        tuple(
            _rng_stream_digest(run.capacity_run, agent_index)
            for agent_index in range(2)
        )
        == first_streams
        for run in runs
    )
    no_signal_contract = all(
        not bool(jnp.any(pair.no_signal.trace.dispatched_eviction_protection))
        and bool(jnp.all(pair.no_signal.trace.scrub_preparations_valid))
        for pair in pairs
    )
    return LineageCachePairedPanel(
        runs=runs,
        pairs=pairs,
        effects=effects,
        common_random_numbers=LineageCacheCommonRandomNumberAudit(
            root_index=CALIBRATION_ROOT_INDEX,
            key_streams_equal_across_all_eight=key_streams_equal,
            branch_independent_key_advance=True,
            selection_calls_per_agent=NUM_STEPS + 1,
            environment_randomness_consumed=ENVIRONMENT_RANDOMNESS_CONSUMED,
        ),
        resources_matched=resources_matched,
        work_matched=work_matched,
        no_signal_is_controller_scrub_baseline=no_signal_contract,
        cache_capacity_per_agent=LINEAGE_CACHE_CAPACITY,
        task_labels_used=False,
        configurable_match_threshold_used=False,
        score_source="exact_cross_birth_strict_predictive_rescue_count",
        # Before any strict predictive rescue, B and D both have score zero.
        # The prefix-identical first C admission therefore remains unresolved.
        prefix_twin_first_eviction_resolved=False,
        development_only=LINEAGE_CACHE_DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        selection_performed=False,
        selected_epsilon=None,
        conclusion=(
            "Consumed-root causal development replay only: a one-record victim "
            "cache can transfer exact predictive-rescue value across authenticated "
            "semantic births, but only after a unique strict observed prediction "
            "win. Both conditions perform the same cache and scrub transaction; "
            "only the source-state protection vector differs. The prefix-twin "
            "first eviction remains unknowable. On this dyad, one just-observed "
            "transition is insufficient to uniquely identify a prior convention. "
            "No threshold, task label, tuning, winner, promotion, or new-root "
            "inference is authorized."
        ),
    )


__all__ = [
    "ARBITRARY_ROOT_EXECUTION_ALLOWED",
    "BIRTH_AUTHENTICATED_CONTROLLER_SCRUB",
    "CALIBRATION_ROOT",
    "CALIBRATION_ROOT_CONSUMED",
    "CONTEXT_CONFIG",
    "DEVELOPMENT_NAMESPACE",
    "DEVELOPMENT_ONLY",
    "EPSILON_GRID",
    "FORBIDDEN_LEARNER_CHANNELS",
    "GAME_CONFIG",
    "LEARNER_POST_ACTION_CHANNELS",
    "LEARNER_PRE_ACTION_CHANNELS",
    "LINEAGE_CACHE_CAPACITY",
    "LINEAGE_CACHE_CONDITIONS",
    "LINEAGE_CACHE_DEVELOPMENT_ONLY",
    "LINEAGE_CACHE_NO_SIGNAL",
    "LINEAGE_CACHE_PREDICTIVE_RESCUE",
    "LINEAGE_CACHE_WORK_BUDGET",
    "MAX_CONTEXTS",
    "NEXT_FROZEN_PANEL",
    "NUM_STEPS",
    "OFFSETS",
    "OUTPUT_WRITES_ALLOWED",
    "POST_AUDIT_BASELINE",
    "POST_AUDIT_CONDITIONS",
    "PHASE_LABELS",
    "PHASE_LENGTH",
    "PROTOCOL",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SELECTIVE_RETENTION_CONDITIONS",
    "SELECTIVE_RETENTION_DEVELOPMENT_ONLY",
    "SELECTIVE_RETENTION_NO_SIGNAL",
    "SELECTIVE_RETENTION_PAST_RECURRENCE",
    "SELECTIVE_RETENTION_WORK_BUDGET",
    "SEMANTIC_CONTEXT_IDENTITY",
    "SUMMARY_WINDOW",
    "WORK_BUDGET",
    "CapacityPressurePanel",
    "CapacityPressureRun",
    "CapacityPressureState",
    "CapacityPressureStepResult",
    "CapacityPressureSummary",
    "BirthRecurrenceHistoryState",
    "PostAuditPairedPanel",
    "PostAuditRun",
    "PostAuditStepResult",
    "PredictiveRescueFailureDecomposition",
    "PrefixTwinBoundary",
    "SelectiveRetentionPairedPanel",
    "SelectiveRetentionRun",
    "SelectiveRetentionState",
    "SelectiveRetentionStepResult",
    "ContextLineageCacheState",
    "LineageCachePairedPanel",
    "LineageCacheRetentionState",
    "LineageCacheRetentionStepResult",
    "LineageCacheRun",
    "build_prefix_twin_boundary",
    "advance_consumed_selective_retention_state",
    "advance_consumed_lineage_cache_retention_state",
    "control_config",
    "initialize_capacity_pressure_state",
    "initialize_selective_retention_state",
    "initialize_lineage_cache_retention_state",
    "run_consumed_calibration_arm",
    "run_consumed_calibration_panel",
    "run_post_audit_paired_intervention",
    "run_selective_retention_paired_intervention",
    "run_lineage_cache_paired_intervention",
    "step_capacity_pressure",
    "step_post_audit_intervention",
    "step_selective_retention",
    "step_lineage_cache_retention",
    "summarize_capacity_pressure_run",
    "validate_static_contract",
]
