# mypy: disable-error-code="call-arg"
"""Development-only evaluation for the hidden-regime Lewis lifecycle.

The evaluator runs a helper and beneficiary through one uninterrupted hidden
regime life.  The dyad is a Lewis sender--receiver signaling game (Lewis 1969,
*Convention*; Skyrms 2010, *Signals*): the helper privately observes a ternary
cue and emits a ternary message; the beneficiary sees only the delivered
symbol and acts; both share one binary reward.  Both roles receive only their
ordinary local input, their local action, and the common scalar reward.
Segment identity, schedule position, target permutations, and replacement
provenance remain evaluator-only facts.

This module deliberately defines no acceptance threshold, artifact writer,
CLI, held-out protocol, or promotion path.  Its default schedule is long
enough for development inspection (every non-transient segment is sixteen
times the substrate default while D remains sixteen steps), but the reserved
seed namespace below is intentionally unexecuted.  Callers must supply seed
pairs explicitly.

The primitive trace records every persistent role-state field, both decisions,
the lifecycle diagnostics, and both value banks as raw float32 bits before and
after every transition.  It is intentionally sufficient input for a separate
host state-machine audit; deterministic replay in this module remains only a
same-implementation consistency check.  Consequently durable immutability is
an exact same-generation property: a selective durable table may change only
at an atomically reported retirement/commit transition that changes its
generation.  This is a bounded three-slot mechanism test, not evidence that
finite memory can retain an unbounded number of arbitrary conventions.

Retention metrics bind every recurrence to the complete canonical ledger of
earlier same-regime commit generations.  A commit qualifies before recurrence
only when its committed composed mapping is exactly correct on all three cues;
selection is the latest surviving qualified commit, never the best entry-time
table.  Adjacent equal-regime segments are one uninterrupted exposure episode,
not a recurrence.  Entry windows remain exactly one learner lease and are not
snapped to learner boundaries.  Short segments are explicitly missing.  The
all-dormant best-table probes and configurable metric-window summaries remain
serialized as secondary/legacy descriptors, not primary lineage evidence.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    N_SLOTS,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    SLOT_VACANT,
    DurableWritePolicy,
    ReplacementTargetPolicy,
    SlotSignalingAgent,
    SlotSignalingConfig,
    SlotSignalingState,
    slot_signaling_keys,
    slot_signaling_resource_budget,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CONSTANT_ZERO_TERNARY_CHANNEL,
    DEFAULT_REGIME_PERMUTATIONS,
    DEFAULT_SEGMENT_LENGTHS,
    DEFAULT_SEGMENT_REGIMES,
    DIRECT_TERNARY_CHANNEL,
    SHUFFLED_TERNARY_CHANNEL,
    HiddenRegimeChannel,
    HiddenRegimeSignalingWorld,
    HiddenRegimeWorldConfig,
    HiddenRegimeWorldState,
    hidden_regime_world_keys,
)

HIDDEN_REGIME_DEVELOPMENT_SCHEMA = "alberta.hidden-regime-signaling.development.v5"
HIDDEN_REGIME_TRACE_SCHEMA = "alberta.hidden-regime-signaling.primitive-trace.v3"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
ACCEPTANCE_STATUS = "descriptive_only_no_acceptance_gate"

# This namespace is a reservation, not an invitation to run it.  Tests use
# explicit manual seed pairs under unrelated names and never derive this set.
RESERVED_DEVELOPMENT_SEED_NAMESPACE = "hidden-regime-signaling-v0-reserved-development-a-v1"
RESERVED_DEVELOPMENT_SEED_NAMESPACE_EXECUTED = False

CONSTANT_CHANNEL_SYMBOL = 0
# Each role persists a (4 slots x 3 inputs x 3 actions) float32 value bank
# (36 scalars), 2 x 4 relevance scalars, 23 int32/float32 lifecycle scalars
# (four per-slot counters of 4 plus seven scalar cursors), and a 2-word
# uint32 RNG key: 69 four-byte scalars = 276 bytes per role, 552 per dyad.
EXPECTED_DYAD_STATE_BYTES = 552
DEVELOPMENT_CANDIDATE_PROVENANCE = (
    "learning_rate=0.25, epsilon=0.1, relevance_rate=0.1, lease_length=16, "
    "confirmation_steps=8, durable_retrieval_threshold=0.5, "
    "candidate_confirmation_threshold=0.75, and candidate_confirmation_leases=3 "
    "were selected after descriptive runs on 30 consumed manual development pairs; "
    "scratch_training_leases_before_retest=16 was then selected on those same consumed "
    "pairs and is fixed only as this evaluator's development candidate; all candidate "
    "values require a new preregistered protocol before any held-out claim"
)
DEVELOPMENT_CALIBRATION_LIMITATIONS = (
    "the scratch-residency selection used the same 30 consumed manual development pairs "
    "and is nonpromoting: at scratch_training_leases_before_retest=16, selective reward "
    "had mean 0.845515, standard deviation 0.012124, and minimum 0.808507, with recurrence "
    "entry 0.754796 and 29/30 runs having exactly four commits and one replacement; "
    "writable/LRU reward had mean 0.835431, standard deviation 0.012670, and minimum "
    "0.807297, with recurrence entry 0.672114, only 6/30 runs having exactly four commits "
    "and one replacement, and observed replacement counts spanning 2--7; helper-frozen, "
    "beneficiary-frozen, constant-channel, and shuffled-channel controls remained in the "
    "0.332--0.336 mean-reward range with zero commits. These post-selection descriptive "
    "values are not a held-out comparison, an acceptance gate, or scientific evidence"
)
REPLAY_PORTABILITY_SCOPE = (
    "exact named-RNG replay is scoped to the same JAX/XLA backend and runtime; "
    "portable fused-versus-unfused candidate checks are independent, but this "
    "development payload is not a cross-backend evidence artifact"
)

# Preserve the deliberately short D exposure while making every other default
# segment a useful development-length inspection window.
DEFAULT_DEVELOPMENT_SEGMENT_LENGTHS: tuple[int, ...] = tuple(
    length if regime == 4 else length * 16
    for length, regime in zip(
        DEFAULT_SEGMENT_LENGTHS,
        DEFAULT_SEGMENT_REGIMES,
        strict=True,
    )
)

SELECTIVE_FULL: Literal["selective_full"] = "selective_full"
# Semantic alias for the selective/evidence factorial cell; serialized reports
# use the ``selective_full`` string.
SELECTIVE_EVIDENCE: Literal["selective_full"] = SELECTIVE_FULL
WRITABLE_EVIDENCE: Literal["writable_evidence"] = "writable_evidence"
SELECTIVE_LRU: Literal["selective_lru"] = "selective_lru"
WRITABLE_LRU: Literal["writable_lru"] = "writable_lru"
HELPER_FROZEN: Literal["helper_frozen"] = "helper_frozen"
BENEFICIARY_FROZEN: Literal["beneficiary_frozen"] = "beneficiary_frozen"
CONSTANT_CHANNEL: Literal["constant_channel_0"] = "constant_channel_0"
SHUFFLED_CHANNEL: Literal["shuffled_channel"] = "shuffled_channel"

type HiddenRegimeDevelopmentCondition = Literal[
    "selective_full",
    "writable_evidence",
    "selective_lru",
    "writable_lru",
    "helper_frozen",
    "beneficiary_frozen",
    "constant_channel_0",
    "shuffled_channel",
]

MATCHED_CONDITIONS: tuple[HiddenRegimeDevelopmentCondition, ...] = (
    SELECTIVE_FULL,
    WRITABLE_EVIDENCE,
    SELECTIVE_LRU,
    WRITABLE_LRU,
    HELPER_FROZEN,
    BENEFICIARY_FROZEN,
    CONSTANT_CHANNEL,
    SHUFFLED_CHANNEL,
)


def _validated_condition_order(
    conditions: Sequence[HiddenRegimeDevelopmentCondition],
) -> tuple[HiddenRegimeDevelopmentCondition, ...]:
    """Require one explicit ordered subsequence of the canonical matched order."""

    condition_order = tuple(conditions)
    if not condition_order or condition_order[0] != SELECTIVE_FULL:
        raise ValueError("conditions must start with canonical selective_full")
    if any(type(condition) is not str for condition in condition_order):
        raise ValueError("conditions must contain only canonical string identifiers")
    if len(set(condition_order)) != len(condition_order):
        raise ValueError("conditions must be unique, including semantic aliases")
    if any(condition not in MATCHED_CONDITIONS for condition in condition_order):
        raise ValueError("conditions contain an unknown identifier")
    positions = tuple(MATCHED_CONDITIONS.index(condition) for condition in condition_order)
    if positions != tuple(sorted(positions)):
        raise ValueError("conditions must follow the canonical matched-condition order")
    return condition_order

_CHANNEL_DIRECT = 0
_CHANNEL_CONSTANT_ZERO = 1
_CHANNEL_SHUFFLED = 2


def _default_development_world() -> HiddenRegimeWorldConfig:
    return HiddenRegimeWorldConfig(
        segment_lengths=DEFAULT_DEVELOPMENT_SEGMENT_LENGTHS,
        segment_regimes=DEFAULT_SEGMENT_REGIMES,
        regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
        repeat_schedule=False,
    )


def _default_development_learner() -> SlotSignalingConfig:
    """Return the calibrated candidate fixed for this nonpromoting evaluator."""

    return SlotSignalingConfig(
        learning_rate=0.25,
        epsilon=0.1,
        relevance_rate=0.1,
        lease_length=16,
        confirmation_steps=8,
        durable_retrieval_threshold=0.5,
        candidate_confirmation_threshold=0.75,
        candidate_confirmation_leases=3,
        scratch_training_leases_before_retest=16,
    )


@dataclasses.dataclass(frozen=True)
class HiddenRegimeDevelopmentConfig:
    """Exact nonpromoting world, learner, and descriptive metric window.

    ``metric_window`` is the width, in transitions, of the legacy early/late
    segment-reward descriptors (clipped to the segment length).  It is
    serialized as ``legacy_metric_window`` and is never direct-retention
    evidence; the retention probes use ``learner.lease_length`` windows.
    """

    world: HiddenRegimeWorldConfig = dataclasses.field(default_factory=_default_development_world)
    learner: SlotSignalingConfig = dataclasses.field(default_factory=_default_development_learner)
    metric_window: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.world, HiddenRegimeWorldConfig):
            raise TypeError("world must be a HiddenRegimeWorldConfig")
        if self.world.repeat_schedule:
            raise ValueError("development evaluation requires one finite nonrepeating schedule")
        if not isinstance(self.learner, SlotSignalingConfig):
            raise TypeError("learner must be a SlotSignalingConfig")
        if (
            self.learner.effective_durable_write_policy != DURABLE_WRITE_SELECTIVE
            or self.learner.effective_replacement_target_policy != REPLACEMENT_TARGET_EVIDENCE
        ):
            raise ValueError(
                "base learner must be selective/evidence; factorial policies are "
                "evaluator conditions"
            )
        if type(self.metric_window) is not int or self.metric_window < 1:
            raise ValueError("metric_window must be a positive integer")

    @property
    def num_steps(self) -> int:
        """Number of transitions in the uninterrupted finite development life."""

        return self.world.total_schedule_steps

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible nonpromotion record."""

        return {
            "schema_version": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "acceptance_status": ACCEPTANCE_STATUS,
            "claim_thresholds_frozen": False,
            "world": self.world.to_dict(),
            "learner": self.learner.to_dict(),
            "metric_window": self.metric_window,
            "num_steps": self.num_steps,
            "default_schedule_semantics": (
                "non-D substrate segments scaled by 16; D-transient fixed at 16 steps"
            ),
            "development_candidate_provenance": DEVELOPMENT_CANDIDATE_PROVENANCE,
            "development_calibration_limitations": DEVELOPMENT_CALIBRATION_LIMITATIONS,
            "replay_portability_scope": REPLAY_PORTABILITY_SCOPE,
            "retention_scope": (
                "commit-generation lineage over three durable modules plus scratch; "
                "bounded hidden-regime development only"
            ),
            "retention_window_semantics": (
                "exactly learner.lease_length prequential transitions beginning at each "
                "genuine recurrence entry after a different-regime absence; adjacent "
                "equal-regime segments are one coalesced exposure; not aligned to learner "
                "lease boundaries; latest-qualified acquisition comparisons begin at the "
                "coalesced episode containing that commit; a shorter coalesced episode is "
                "an explicitly missing window"
            ),
            "legacy_metric_window_semantics": (
                "metric_window early/late reward and legacy first-ever-exposure comparisons "
                "are retained for compatibility and are not direct-retention evidence"
            ),
        }


@dataclasses.dataclass(frozen=True)
class HiddenRegimeSeedPair:
    """Separated deterministic world and learner seeds."""

    namespace: str
    index: int
    world_seed: int
    learner_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace must be a non-empty string")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        for name in ("world_seed", "learner_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{name} must be a uint32 integer")

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


def derive_hidden_regime_seed_pairs(
    namespace: str,
    count: int,
) -> tuple[HiddenRegimeSeedPair, ...]:
    """Derive stable SHA-256-separated world/learner uint32 seed pairs.

    Merely calling this function does not execute a protocol.  The reserved
    namespace remains unexecuted by this module and its tests.
    """

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    pairs: list[HiddenRegimeSeedPair] = []
    for index in range(count):
        seeds: list[int] = []
        for owner in ("world", "learner"):
            material = (f"{HIDDEN_REGIME_DEVELOPMENT_SCHEMA}|{namespace}|{index}|{owner}").encode()
            seeds.append(int.from_bytes(hashlib.sha256(material).digest()[:4], "big"))
        pairs.append(HiddenRegimeSeedPair(namespace, index, seeds[0], seeds[1]))
    return tuple(pairs)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeConditionSpec:
    """Exact fixed-resource intervention for one matched condition."""

    channel: HiddenRegimeChannel
    helper_write: bool
    beneficiary_write: bool
    durable_write_policy: DurableWritePolicy
    replacement_target_policy: ReplacementTargetPolicy

    @property
    def durable_writes_enabled(self) -> bool:
        """Whether this condition permits ordinary writes to durable values."""

        return self.durable_write_policy == DURABLE_WRITE_WRITABLE

    @property
    def writable_lru_ablation(self) -> bool:
        """``True`` only for the combined writable+LRU cell, mirroring
        the single-flag ``SlotSignalingConfig.writable_lru_ablation``."""

        return (
            self.durable_write_policy == DURABLE_WRITE_WRITABLE
            and self.replacement_target_policy == REPLACEMENT_TARGET_LRU
        )


def condition_spec(condition: HiddenRegimeDevelopmentCondition) -> HiddenRegimeConditionSpec:
    """Return one condition without adding learner-visible oracle inputs."""

    if condition == SELECTIVE_FULL:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    if condition == WRITABLE_EVIDENCE:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_WRITABLE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    if condition == SELECTIVE_LRU:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_LRU,
        )
    if condition == WRITABLE_LRU:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_WRITABLE,
            REPLACEMENT_TARGET_LRU,
        )
    if condition == HELPER_FROZEN:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            False,
            True,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    if condition == BENEFICIARY_FROZEN:
        return HiddenRegimeConditionSpec(
            DIRECT_TERNARY_CHANNEL,
            True,
            False,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    if condition == CONSTANT_CHANNEL:
        return HiddenRegimeConditionSpec(
            CONSTANT_ZERO_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    if condition == SHUFFLED_CHANNEL:
        return HiddenRegimeConditionSpec(
            SHUFFLED_TERNARY_CHANNEL,
            True,
            True,
            DURABLE_WRITE_SELECTIVE,
            REPLACEMENT_TARGET_EVIDENCE,
        )
    raise ValueError(f"unknown hidden-regime development condition: {condition!r}")


def _channel_code(channel: HiddenRegimeChannel) -> int:
    if channel == DIRECT_TERNARY_CHANNEL:
        return _CHANNEL_DIRECT
    if channel == CONSTANT_ZERO_TERNARY_CHANNEL:
        return _CHANNEL_CONSTANT_ZERO
    if channel == SHUFFLED_TERNARY_CHANNEL:
        return _CHANNEL_SHUFFLED
    raise ValueError(f"unsupported development channel: {channel!r}")


@dataclasses.dataclass(frozen=True)
class HiddenRegimePrimitiveTrace:
    """Complete fixed-shape transition record for a future independent host audit.

    Every persistent :class:`SlotRoleState` field and every world transition
    leaf, including actual termination and discount, is recorded.  Float value
    banks are carried as uint32 bits so signed zero and one-bit mutations cannot
    disappear during serialization.  Policy keys are exported only as their two
    uint32 data words; the learner never receives evaluator fields.
    """

    step_index: Array
    segment_index: Array
    segment_step: Array
    regime_id: Array
    world_cue_pre: Array
    world_cue_post: Array
    world_step_count_pre: Array
    world_step_count_post: Array
    world_schedule_position_pre: Array
    world_schedule_position_post: Array
    world_cue_key_data_pre: Array
    world_cue_key_data_post: Array
    world_channel_key_data_pre: Array
    world_channel_key_data_post: Array
    helper_cue: Array
    oracle_target: Array
    helper_message: Array
    delivered_message: Array
    beneficiary_action: Array
    reward: Array
    world_terminated: Array
    world_discount: Array
    helper_write_enabled: Array
    beneficiary_write_enabled: Array
    helper_decision_slot: Array
    helper_private_input: Array
    helper_decision_action: Array
    helper_selected_value: Array
    beneficiary_decision_slot: Array
    beneficiary_private_input: Array
    beneficiary_decision_action: Array
    beneficiary_selected_value: Array
    helper_relevance_mean_pre: Array
    helper_relevance_mean_post: Array
    beneficiary_relevance_mean_pre: Array
    beneficiary_relevance_mean_post: Array
    helper_relevance_mass_pre: Array
    helper_relevance_mass_post: Array
    beneficiary_relevance_mass_pre: Array
    beneficiary_relevance_mass_post: Array
    helper_failed_leases_pre: Array
    helper_failed_leases_post: Array
    beneficiary_failed_leases_pre: Array
    beneficiary_failed_leases_post: Array
    helper_idle_leases_pre: Array
    helper_idle_leases_post: Array
    beneficiary_idle_leases_pre: Array
    beneficiary_idle_leases_post: Array
    helper_active_slot_pre: Array
    helper_active_slot_post: Array
    beneficiary_active_slot_pre: Array
    beneficiary_active_slot_post: Array
    helper_status_pre: Array
    helper_status_post: Array
    beneficiary_status_pre: Array
    beneficiary_status_post: Array
    helper_generation_pre: Array
    helper_generation_post: Array
    beneficiary_generation_pre: Array
    beneficiary_generation_post: Array
    helper_candidate_confirmations_pre: Array
    helper_candidate_confirmations_post: Array
    beneficiary_candidate_confirmations_pre: Array
    beneficiary_candidate_confirmations_post: Array
    helper_lease_offset_pre: Array
    helper_lease_offset_post: Array
    beneficiary_lease_offset_pre: Array
    beneficiary_lease_offset_post: Array
    helper_lease_reward_sum_pre: Array
    helper_lease_reward_sum_post: Array
    beneficiary_lease_reward_sum_pre: Array
    beneficiary_lease_reward_sum_post: Array
    helper_remaining_durable_tests_pre: Array
    helper_remaining_durable_tests_post: Array
    beneficiary_remaining_durable_tests_pre: Array
    beneficiary_remaining_durable_tests_post: Array
    helper_search_cursor_pre: Array
    helper_search_cursor_post: Array
    beneficiary_search_cursor_pre: Array
    beneficiary_search_cursor_post: Array
    helper_next_generation_pre: Array
    helper_next_generation_post: Array
    beneficiary_next_generation_pre: Array
    beneficiary_next_generation_post: Array
    helper_policy_key_data_pre: Array
    helper_policy_key_data_post: Array
    beneficiary_policy_key_data_pre: Array
    beneficiary_policy_key_data_post: Array
    helper_scratch_failed_leases_pre: Array
    helper_scratch_failed_leases_post: Array
    beneficiary_scratch_failed_leases_pre: Array
    beneficiary_scratch_failed_leases_post: Array
    helper_scratch_retest_started: Array
    beneficiary_scratch_retest_started: Array
    helper_value_write: Array
    beneficiary_value_write: Array
    helper_value_pre: Array
    helper_candidate_value: Array
    helper_value_post: Array
    beneficiary_value_pre: Array
    beneficiary_candidate_value: Array
    beneficiary_value_post: Array
    helper_lease_boundary: Array
    beneficiary_lease_boundary: Array
    helper_lease_reward_mean: Array
    beneficiary_lease_reward_mean: Array
    helper_relevance_ready: Array
    beneficiary_relevance_ready: Array
    helper_durable_relevant: Array
    beneficiary_durable_relevant: Array
    helper_candidate_relevant: Array
    beneficiary_candidate_relevant: Array
    helper_candidate_lease_success: Array
    beneficiary_candidate_lease_success: Array
    helper_generation_exhausted: Array
    beneficiary_generation_exhausted: Array
    helper_committed_slot: Array
    helper_committed_generation: Array
    helper_retired_slot: Array
    helper_retired_generation: Array
    beneficiary_committed_slot: Array
    beneficiary_committed_generation: Array
    beneficiary_retired_slot: Array
    beneficiary_retired_generation: Array
    lifecycle_synchronized: Array
    helper_value_bits_pre: Array
    helper_value_bits_post: Array
    beneficiary_value_bits_pre: Array
    beneficiary_value_bits_post: Array
    helper_selective_mutation_violation: Array
    beneficiary_selective_mutation_violation: Array

    def to_dict(self) -> dict[str, object]:
        """Serialize raw primitives without an evaluator replay."""

        return {
            "schema_version": HIDDEN_REGIME_TRACE_SCHEMA,
            **{
                field.name: np.asarray(getattr(self, field.name)).tolist()
                for field in dataclasses.fields(self)
            },
        }


@dataclasses.dataclass(frozen=True)
class SegmentRewardSummary:
    """Legacy configurable-window reward descriptors for one world segment."""

    segment_index: int
    regime_id: int
    regime_label: str
    occurrence_index: int
    steps: int
    mean_reward: float
    early_reward: float
    late_reward: float
    is_recurrence: bool
    previous_occurrence_late_reward: float | None
    recurrence_entry_change: float | None
    within_segment_recovery: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RegimeRecurrenceSummary:
    """Legacy metric-window recurrence descriptor, not a retention probe."""

    regime_id: int
    regime_label: str
    occurrences: int
    recurrence_count: int
    recurrence_entry_reward_mean: float | None
    previous_late_reward_mean: float | None
    recurrence_entry_change_mean: float | None
    recurrence_recovery_mean: float | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DormantGenerationProbe:
    """Canonical evaluator-only greedy probes for one dormant generation."""

    slot: int
    generation: int
    composed_greedy_accuracy: float
    zero_helper_accuracy: float
    zero_beneficiary_accuracy: float
    role_swapped_accuracy: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CommitGenerationLineage:
    """Evaluator-only identity and exact content of one synchronized commit."""

    lineage_index: int
    commit_step: int
    commit_segment_index: int
    commit_segment_step: int
    regime_id: int
    regime_label: str
    slot: int
    generation: int
    target_mapping: tuple[int, int, int]
    committed_composed_greedy_mapping: tuple[int, int, int]
    committed_composed_greedy_accuracy: float
    committed_composed_greedy_tie_free: bool
    acquisition_qualified: bool
    helper_table_uint32_bits: tuple[int, ...]
    beneficiary_table_uint32_bits: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        for name in (
            "target_mapping",
            "committed_composed_greedy_mapping",
            "helper_table_uint32_bits",
            "beneficiary_table_uint32_bits",
        ):
            payload[name] = list(cast(tuple[int, ...], payload[name]))
        return payload


type LineageEntryActivity = Literal["active", "dormant", "mixed", "unavailable"]
type TransitionEventPhase = Literal["pre", "post"]


@dataclasses.dataclass(frozen=True)
class RecurrenceLineageProbe:
    """Entry-time survival and content audit for one prior commit lineage."""

    lineage_index: int
    commit_step: int
    commit_segment_index: int
    slot: int
    generation: int
    acquisition_qualified: bool
    helper_entry_slot_status: int
    helper_entry_slot_generation: int
    helper_slot_generation_present: bool
    beneficiary_entry_slot_status: int
    beneficiary_entry_slot_generation: int
    beneficiary_slot_generation_present: bool
    synchronized_generation_survives: bool
    helper_active_at_entry: bool
    beneficiary_active_at_entry: bool
    entry_activity_status: LineageEntryActivity
    helper_entry_table_uint32_bits: tuple[int, ...] | None
    beneficiary_entry_table_uint32_bits: tuple[int, ...] | None
    entry_composed_greedy_mapping: tuple[int, int, int] | None
    entry_composed_greedy_accuracy: float | None
    entry_minus_commit_accuracy: float | None
    helper_bit_exact_preserved: bool
    beneficiary_bit_exact_preserved: bool
    joint_bit_exact_preserved: bool
    zero_helper_accuracy: float | None
    zero_beneficiary_accuracy: float | None
    role_swapped_accuracy: float | None

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        for name in (
            "helper_entry_table_uint32_bits",
            "beneficiary_entry_table_uint32_bits",
            "entry_composed_greedy_mapping",
        ):
            value = payload[name]
            payload[name] = None if value is None else list(cast(tuple[int, ...], value))
        return payload


@dataclasses.dataclass(frozen=True, kw_only=True)
class RecurrenceRetentionRecord:
    """Evaluator-only direct-retention probes for one genuine recurrence entry.

    ``first_world_window_*`` starts exactly at the world change point and has
    ``learner.lease_length`` transitions.  It is not aligned to the learner's
    lease phase.  ``occurrence_index`` coalesces adjacent equal-regime segments;
    ``raw_segment_occurrence_index`` discloses the legacy segment count.  Commit
    lineage and secondary dormant-table probes read only the immediate entry
    state and never enter either learner policy or update.  The
    ``latest_qualified_acquisition_*`` window is an episode-level adaptation
    descriptor; exact commit-to-entry content retention is reported separately.
    All ``legacy_*`` fields retain the earlier first-ever-exposure comparison.
    """

    segment_index: int
    regime_id: int
    regime_label: str
    occurrence_index: int
    raw_segment_occurrence_index: int = -1
    legacy_first_exposure_segment_index: int
    world_entry_learner_lease_offset: int
    world_entry_steps_to_first_learner_boundary: int
    first_world_window_length: int
    first_world_window_complete: bool
    first_world_window_reward: float | None
    first_world_window_errors: int | None
    first_world_window_error_rate: float | None
    legacy_first_exposure_world_window_complete: bool
    legacy_first_exposure_world_window_reward: float | None
    legacy_first_exposure_world_window_errors: int | None
    legacy_first_exposure_world_window_error_rate: float | None
    legacy_recurrence_minus_first_exposure_error_rate_delta: float | None
    legacy_recurrence_to_first_exposure_error_rate_ratio: float | None
    legacy_recurrence_to_first_exposure_error_rate_ratio_defined: bool
    latest_qualified_acquisition_segment_index: int | None = None
    latest_qualified_acquisition_episode_length: int | None = None
    latest_qualified_acquisition_world_window_complete: bool | None = None
    latest_qualified_acquisition_world_window_reward: float | None = None
    latest_qualified_acquisition_world_window_errors: int | None = None
    latest_qualified_acquisition_world_window_error_rate: float | None = None
    latest_qualified_acquisition_comparison_available: bool = False
    recurrence_minus_latest_qualified_acquisition_error_rate_delta: float | None = None
    recurrence_to_latest_qualified_acquisition_error_rate_ratio: float | None = None
    recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined: bool | None = None
    prior_same_regime_lineages: tuple[RecurrenceLineageProbe, ...] = ()
    prior_same_regime_lineage_count: int = 0
    prior_qualified_lineage_count: int = 0
    prior_unqualified_lineage_count: int = 0
    lineage_retention_applicable: bool = False
    acquisition_coverage_failure: bool = True
    latest_prior_qualified_lineage_index: int | None = None
    latest_prior_qualified_commit_step: int | None = None
    latest_prior_qualified_survived: bool | None = None
    any_prior_qualified_survived: bool | None = None
    surviving_qualified_lineage_count: int = 0
    selected_lineage_available: bool = False
    selected_lineage_index: int | None = None
    selected_lineage_commit_step: int | None = None
    selected_lineage_slot: int | None = None
    selected_lineage_generation: int | None = None
    selected_lineage_entry_activity_status: LineageEntryActivity | None = None
    selected_lineage_entry_composed_greedy_mapping: tuple[int, int, int] | None = None
    selected_lineage_entry_composed_greedy_accuracy: float | None = None
    selected_lineage_entry_minus_commit_accuracy: float | None = None
    selected_lineage_helper_bit_exact_preserved: bool | None = None
    selected_lineage_beneficiary_bit_exact_preserved: bool | None = None
    selected_lineage_joint_bit_exact_preserved: bool | None = None
    selected_lineage_zero_helper_accuracy: float | None = None
    selected_lineage_zero_beneficiary_accuracy: float | None = None
    selected_lineage_role_swapped_accuracy: float | None = None
    selected_exact_generation_relock_observed: bool | None = None
    selected_first_exact_generation_relock_step: int | None = None
    selected_first_exact_generation_relock_segment_step: int | None = None
    selected_exact_generation_relock_phase: TransitionEventPhase | None = None
    selected_observed_learner_boundaries_until_relock: int | None = None
    selected_first_scratch_entry_step: int | None = None
    selected_first_scratch_entry_segment_step: int | None = None
    selected_first_scratch_entry_phase: TransitionEventPhase | None = None
    selected_scratch_entered_before_relock: bool | None = None
    selected_scratch_entered_before_relock_or_segment_end: bool | None = None
    selected_durable_retrieval_before_scratch: bool | None = None
    dormant_probe_available: bool
    eligible_dormant_generations: tuple[DormantGenerationProbe, ...]
    best_dormant_slot: int | None
    best_dormant_generation: int | None
    best_dormant_composed_greedy_accuracy: float | None
    best_dormant_zero_helper_accuracy: float | None
    best_dormant_zero_beneficiary_accuracy: float | None
    best_dormant_role_swapped_accuracy: float | None
    exact_generation_relock_observed: bool
    first_exact_generation_relock_step: int | None
    first_exact_generation_relock_segment_step: int | None
    observed_learner_boundaries_in_segment: int
    observed_learner_boundaries_until_relock: int | None
    scratch_entered_before_relock: bool | None
    scratch_entered_before_relock_or_segment_end: bool
    durable_retrieval_before_scratch: bool

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["prior_same_regime_lineages"] = [
            item.to_dict() for item in self.prior_same_regime_lineages
        ]
        selected_mapping = self.selected_lineage_entry_composed_greedy_mapping
        payload["selected_lineage_entry_composed_greedy_mapping"] = (
            None if selected_mapping is None else list(selected_mapping)
        )
        payload["eligible_dormant_generations"] = [
            item.to_dict() for item in self.eligible_dormant_generations
        ]
        return payload


@dataclasses.dataclass(frozen=True, kw_only=True)
class RetentionAggregateSummary:
    """Aggregate direct-retention metrics with explicit denominators/missingness."""

    recurrence_count: int
    complete_first_world_window_count: int
    missing_first_world_window_count: int
    first_world_window_reward_mean: float | None
    first_world_window_error_rate_mean: float | None
    lineage_retention_applicable_count: int = 0
    acquisition_coverage_failure_count: int = 0
    qualification_coverage_denominator: int = 0
    qualification_coverage_fraction: float | None = None
    prior_same_regime_lineage_count: int = 0
    prior_qualified_lineage_count: int = 0
    prior_unqualified_lineage_count: int = 0
    surviving_qualified_lineage_count: int = 0
    qualified_lineage_survival_denominator: int = 0
    qualified_lineage_survival_fraction: float | None = None
    latest_qualified_version_survival_count: int = 0
    latest_qualified_version_survival_denominator: int = 0
    latest_qualified_version_survival_missing_count: int = 0
    latest_qualified_version_survival_fraction: float | None = None
    any_qualified_knowledge_survival_count: int = 0
    any_qualified_knowledge_survival_denominator: int = 0
    any_qualified_knowledge_survival_missing_count: int = 0
    any_qualified_knowledge_survival_fraction: float | None = None
    selected_lineage_probe_available_count: int = 0
    selected_lineage_probe_denominator: int = 0
    selected_lineage_survival_failure_count: int = 0
    selected_lineage_not_applicable_count: int = 0
    selected_lineage_survival_fraction_given_qualified_prior: float | None = None
    selected_entry_metric_denominator: int = 0
    selected_entry_active_count: int = 0
    selected_entry_dormant_count: int = 0
    selected_entry_mixed_count: int = 0
    selected_entry_composed_greedy_accuracy_mean: float | None = None
    selected_entry_minus_commit_accuracy_mean: float | None = None
    selected_helper_bit_exact_preservation_count: int = 0
    selected_beneficiary_bit_exact_preservation_count: int = 0
    selected_joint_bit_exact_preservation_count: int = 0
    selected_bit_exact_preservation_conditional_denominator: int = 0
    selected_bit_exact_preservation_all_qualified_denominator: int = 0
    selected_helper_bit_exact_preservation_fraction: float | None = None
    selected_beneficiary_bit_exact_preservation_fraction: float | None = None
    selected_joint_bit_exact_preservation_fraction: float | None = None
    selected_helper_bit_exact_preservation_fraction_all_qualified: float | None = None
    selected_beneficiary_bit_exact_preservation_fraction_all_qualified: float | None = None
    selected_joint_bit_exact_preservation_fraction_all_qualified: float | None = None
    selected_zero_helper_accuracy_mean: float | None = None
    selected_zero_beneficiary_accuracy_mean: float | None = None
    selected_role_swapped_accuracy_mean: float | None = None
    selected_exact_generation_relock_count: int = 0
    selected_exact_generation_relock_conditional_denominator: int = 0
    selected_exact_generation_relock_all_qualified_denominator: int = 0
    selected_exact_generation_relock_fraction_given_selected_lineage: float | None = None
    selected_exact_generation_relock_fraction_all_qualified: float | None = None
    selected_observed_learner_boundaries_to_relock_mean: float | None = None
    selected_observed_learner_boundaries_to_relock_available_count: int = 0
    selected_observed_learner_boundaries_to_relock_unavailable_count: int = 0
    selected_durable_retrieval_before_scratch_count: int = 0
    selected_durable_retrieval_before_scratch_conditional_denominator: int = 0
    selected_durable_retrieval_before_scratch_all_qualified_denominator: int = 0
    selected_durable_retrieval_before_scratch_fraction_given_selected_lineage: float | None = None
    selected_durable_retrieval_before_scratch_fraction_all_qualified: float | None = None
    dormant_probe_available_count: int
    dormant_probe_missing_count: int
    dormant_composed_greedy_accuracy_mean: float | None
    dormant_zero_helper_accuracy_mean: float | None
    dormant_zero_beneficiary_accuracy_mean: float | None
    dormant_role_swapped_accuracy_mean: float | None
    exact_generation_relock_count: int
    exact_generation_relock_missing_count: int
    exact_generation_relock_rate_all_recurrences: float | None
    exact_generation_relock_rate_given_dormant_probe: float | None
    observed_learner_boundaries_to_relock_mean: float | None
    observed_learner_boundaries_to_relock_missing_count: int
    durable_retrieval_before_scratch_count: int
    durable_retrieval_before_scratch_fraction_all_recurrences: float | None
    durable_retrieval_before_scratch_fraction_given_dormant_probe: float | None
    latest_qualified_acquisition_comparison_available_count: int = 0
    latest_qualified_acquisition_comparison_denominator: int = 0
    latest_qualified_acquisition_comparison_missing_count: int = 0
    latest_qualified_acquisition_comparison_not_applicable_count: int = 0
    recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean: float | None = None
    recurrence_to_latest_qualified_acquisition_error_rate_ratio_mean: float | None = None
    recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined_count: int = 0
    recurrence_to_latest_qualified_acquisition_error_rate_ratio_undefined_count: int = 0
    legacy_first_exposure_comparison_available_count: int
    legacy_first_exposure_comparison_missing_count: int
    legacy_recurrence_minus_first_exposure_error_rate_delta_mean: float | None
    legacy_recurrence_to_first_exposure_error_rate_ratio_mean: float | None
    legacy_recurrence_to_first_exposure_error_rate_ratio_defined_count: int
    legacy_recurrence_to_first_exposure_error_rate_ratio_undefined_count: int

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeResourceReport:
    """Exact initial/final dyad budget for a fixed-resource condition."""

    initial_state_scalars: int
    final_state_scalars: int
    initial_state_bytes: int
    final_state_bytes: int
    expected_state_bytes: int
    resource_constant: bool
    resource_matched: bool

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeRunSummary:
    """Reconstructed descriptive outcomes for one uninterrupted life."""

    num_steps: int
    mean_prequential_reward: float
    legacy_metric_window: int
    legacy_recurrence_entry_reward_mean: float | None
    legacy_recurrence_recovery_mean: float | None
    recurrence_entry_reward_mean: float | None
    recurrence_recovery_mean: float | None
    segment_rewards: tuple[SegmentRewardSummary, ...]
    recurrence_by_regime: tuple[RegimeRecurrenceSummary, ...]
    commit_generation_lineages: tuple[CommitGenerationLineage, ...]
    synchronized_commit_lineage_count: int
    acquisition_qualified_commit_lineage_count: int
    acquisition_unqualified_commit_lineage_count: int
    recurrence_retention: tuple[RecurrenceRetentionRecord, ...]
    retention: RetentionAggregateSummary
    helper_value_write_count: int
    beneficiary_value_write_count: int
    helper_effective_learning_update_count: int
    beneficiary_effective_learning_update_count: int
    both_roles_learned: bool
    helper_commit_count: int
    beneficiary_commit_count: int
    helper_replacement_count: int
    beneficiary_replacement_count: int
    candidate_confirmation_events: int
    c_old_to_c_new_replacement_count: int
    c_old_to_c_new_target_slots: tuple[int, ...]
    c_old_to_c_new_generation_pairs: tuple[tuple[int, int], ...]
    c_old_to_c_new_exactly_one_target: bool
    d_short_checked: bool
    d_short_non_displacement: bool
    selective_immutability_applicable: bool
    helper_selective_mutation_violations: int
    beneficiary_selective_mutation_violations: int
    selective_durable_bit_immutable_until_atomic_replacement: bool
    lifecycle_synchronized_every_step: bool

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["segment_rewards"] = [item.to_dict() for item in self.segment_rewards]
        payload["recurrence_by_regime"] = [item.to_dict() for item in self.recurrence_by_regime]
        payload["commit_generation_lineages"] = [
            item.to_dict() for item in self.commit_generation_lineages
        ]
        payload["recurrence_retention"] = [
            item.to_dict() for item in self.recurrence_retention
        ]
        payload["retention"] = self.retention.to_dict()
        payload["c_old_to_c_new_target_slots"] = list(self.c_old_to_c_new_target_slots)
        payload["c_old_to_c_new_generation_pairs"] = [
            list(pair) for pair in self.c_old_to_c_new_generation_pairs
        ]
        return payload


@dataclasses.dataclass(frozen=True)
class HiddenRegimeRunResult:
    """One matched development condition with trace and reconstructed summary."""

    condition: HiddenRegimeDevelopmentCondition
    seed_pair: HiddenRegimeSeedPair
    config: HiddenRegimeDevelopmentConfig
    trace: HiddenRegimePrimitiveTrace
    summary: HiddenRegimeRunSummary
    resource: HiddenRegimeResourceReport
    final_state: SlotSignalingState

    def to_dict(self, *, include_trace: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "acceptance_status": ACCEPTANCE_STATUS,
            "claim_thresholds_frozen": False,
            "artifact_written": False,
            "reserved_namespace_executed": False,
            "oracle_upper_bound_included": False,
            "condition": self.condition,
            "seed_pair": self.seed_pair.to_dict(),
            "config": self.config.to_dict(),
            "resource": self.resource.to_dict(),
            "summary": self.summary.to_dict(),
            "trace_included": include_trace,
        }
        if include_trace:
            payload["trace"] = self.trace.to_dict()
        return payload


@dataclasses.dataclass(frozen=True)
class PairedControlMetric:
    """One same-seed, same-world descriptive comparison to selective full."""

    condition: str
    mean_prequential_reward: float
    delta_vs_selective_full: float
    recurrence_entry_reward_mean: float | None
    recurrence_entry_delta_vs_selective_full: float | None
    helper_value_write_count: int
    beneficiary_value_write_count: int
    resource_bytes: int

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeDevelopmentReport:
    """All fixed-resource controls for one explicitly supplied seed pair."""

    seed_pair: HiddenRegimeSeedPair
    config: HiddenRegimeDevelopmentConfig
    runs: tuple[HiddenRegimeRunResult, ...]
    paired_controls: tuple[PairedControlMetric, ...]

    def to_dict(self, *, include_traces: bool = True) -> dict[str, object]:
        return {
            "schema_version": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "acceptance_status": ACCEPTANCE_STATUS,
            "claim_thresholds_frozen": False,
            "artifact_written": False,
            "reserved_seed_namespace": RESERVED_DEVELOPMENT_SEED_NAMESPACE,
            "reserved_namespace_executed": False,
            "oracle_upper_bound_included": False,
            "seed_pair": self.seed_pair.to_dict(),
            "config": self.config.to_dict(),
            "condition_order": [run.condition for run in self.runs],
            "runs": [run.to_dict(include_trace=include_traces) for run in self.runs],
            "paired_controls": [item.to_dict() for item in self.paired_controls],
        }


def _role_selective_mutation_violation(
    status_pre: Array,
    status_post: Array,
    generation_pre: Array,
    generation_post: Array,
    values_pre: Array,
    values_post: Array,
    committed_slot: Array,
    committed_generation: Array,
    retired_slot: Array,
    retired_generation: Array,
    durable_writes_enabled: bool,
) -> Array:
    """Flag any durable bit mutation not explained by one atomic replacement."""

    slot_ids = jnp.arange(N_SLOTS, dtype=jnp.int32)
    changed = jnp.any(values_pre != values_post, axis=(1, 2))
    was_durable = status_pre == SLOT_DURABLE
    atomic_replacement = jnp.logical_and(
        retired_slot == slot_ids,
        jnp.logical_and(
            committed_slot == slot_ids,
            jnp.logical_and(
                retired_generation == generation_pre,
                jnp.logical_and(
                    committed_generation == generation_post,
                    jnp.logical_and(
                        generation_pre != generation_post,
                        status_post == SLOT_DURABLE,
                    ),
                ),
            ),
        ),
    )
    violation = jnp.logical_and(was_durable, jnp.logical_and(changed, ~atomic_replacement))
    return jnp.where(
        jnp.asarray(durable_writes_enabled),
        jnp.zeros_like(violation),
        violation,
    )


@functools.lru_cache(maxsize=16)
def _scan_runner(
    config: HiddenRegimeDevelopmentConfig,
    durable_write_policy: DurableWritePolicy,
    replacement_target_policy: ReplacementTargetPolicy,
    num_steps: int,
) -> Any:
    """Compile a fixed-shape development execution chunk."""

    world = HiddenRegimeSignalingWorld(config.world)
    learner_config = dataclasses.replace(
        config.learner,
        writable_lru_ablation=False,
        durable_write_policy=durable_write_policy,
        replacement_target_policy=replacement_target_policy,
    )
    learner = SlotSignalingAgent(learner_config)

    def run(
        world_state: HiddenRegimeWorldState,
        learner_state: SlotSignalingState,
        channel_code: Array,
        helper_write: Array,
        beneficiary_write: Array,
    ) -> Any:
        def step(
            carry: tuple[HiddenRegimeWorldState, SlotSignalingState],
            _: Array,
        ) -> tuple[
            tuple[HiddenRegimeWorldState, SlotSignalingState],
            dict[str, Array],
        ]:
            old_world, old_learner = carry
            observation = world.observe(old_world)
            # The helper sees only its private cue.
            helper = learner.select_helper(old_learner.helper, observation.helper_cue)
            shuffled_draw_key, _ = jr.split(old_world.channel_key)
            shuffled = jr.randint(shuffled_draw_key, (), 0, 3, dtype=jnp.int32)
            delivered = jnp.select(
                (
                    channel_code == _CHANNEL_DIRECT,
                    channel_code == _CHANNEL_CONSTANT_ZERO,
                ),
                (helper.action, jnp.asarray(CONSTANT_CHANNEL_SYMBOL, dtype=jnp.int32)),
                default=shuffled,
            ).astype(jnp.int32)
            # The beneficiary sees only the delivered ternary symbol.
            beneficiary = learner.select_beneficiary(old_learner.beneficiary, delivered)
            # Evaluator-only target/regime fields are created after both decisions.
            transition, next_world = world.step_with_delivery(
                old_world,
                helper.action,
                delivered,
                beneficiary.action,
            )
            update = learner.update(
                old_learner,
                helper,
                beneficiary,
                transition.reward,
                helper_write=helper_write,
                beneficiary_write=beneficiary_write,
            )
            old_helper_bits = jax.lax.bitcast_convert_type(
                old_learner.helper.values,
                jnp.uint32,
            )
            new_helper_bits = jax.lax.bitcast_convert_type(
                update.state.helper.values,
                jnp.uint32,
            )
            old_beneficiary_bits = jax.lax.bitcast_convert_type(
                old_learner.beneficiary.values,
                jnp.uint32,
            )
            new_beneficiary_bits = jax.lax.bitcast_convert_type(
                update.state.beneficiary.values,
                jnp.uint32,
            )
            helper_violation = _role_selective_mutation_violation(
                old_learner.helper.status,
                update.state.helper.status,
                old_learner.helper.generation,
                update.state.helper.generation,
                old_helper_bits,
                new_helper_bits,
                update.helper.committed_slot,
                update.helper.committed_generation,
                update.helper.retired_slot,
                update.helper.retired_generation,
                learner_config.durable_writes_enabled,
            )
            beneficiary_violation = _role_selective_mutation_violation(
                old_learner.beneficiary.status,
                update.state.beneficiary.status,
                old_learner.beneficiary.generation,
                update.state.beneficiary.generation,
                old_beneficiary_bits,
                new_beneficiary_bits,
                update.beneficiary.committed_slot,
                update.beneficiary.committed_generation,
                update.beneficiary.retired_slot,
                update.beneficiary.retired_generation,
                learner_config.durable_writes_enabled,
            )
            output = {
                "step_index": transition.oracle.step_count,
                "segment_index": transition.oracle.segment_index,
                "segment_step": transition.oracle.segment_step,
                "regime_id": transition.oracle.regime_id,
                "world_cue_pre": old_world.cue,
                "world_cue_post": transition.next_observation.helper_cue,
                "world_step_count_pre": old_world.step_count,
                "world_step_count_post": next_world.step_count,
                "world_schedule_position_pre": old_world.schedule_position,
                "world_schedule_position_post": next_world.schedule_position,
                "world_cue_key_data_pre": jr.key_data(old_world.cue_key),
                "world_cue_key_data_post": jr.key_data(next_world.cue_key),
                "world_channel_key_data_pre": jr.key_data(old_world.channel_key),
                "world_channel_key_data_post": jr.key_data(next_world.channel_key),
                "helper_cue": transition.observation.helper_cue,
                "oracle_target": transition.oracle.target,
                "helper_message": transition.helper_message,
                "delivered_message": transition.delivered_message,
                "beneficiary_action": transition.beneficiary_action,
                "reward": transition.reward,
                "world_terminated": transition.terminated,
                "world_discount": transition.discount,
                "helper_write_enabled": helper_write,
                "beneficiary_write_enabled": beneficiary_write,
                "helper_decision_slot": helper.slot,
                "helper_private_input": helper.private_input,
                "helper_decision_action": helper.action,
                "helper_selected_value": helper.selected_value,
                "beneficiary_decision_slot": beneficiary.slot,
                "beneficiary_private_input": beneficiary.private_input,
                "beneficiary_decision_action": beneficiary.action,
                "beneficiary_selected_value": beneficiary.selected_value,
                "helper_relevance_mean_pre": old_learner.helper.relevance_mean,
                "helper_relevance_mean_post": update.state.helper.relevance_mean,
                "beneficiary_relevance_mean_pre": old_learner.beneficiary.relevance_mean,
                "beneficiary_relevance_mean_post": update.state.beneficiary.relevance_mean,
                "helper_relevance_mass_pre": old_learner.helper.relevance_mass,
                "helper_relevance_mass_post": update.state.helper.relevance_mass,
                "beneficiary_relevance_mass_pre": old_learner.beneficiary.relevance_mass,
                "beneficiary_relevance_mass_post": update.state.beneficiary.relevance_mass,
                "helper_failed_leases_pre": old_learner.helper.failed_leases,
                "helper_failed_leases_post": update.state.helper.failed_leases,
                "beneficiary_failed_leases_pre": old_learner.beneficiary.failed_leases,
                "beneficiary_failed_leases_post": update.state.beneficiary.failed_leases,
                "helper_idle_leases_pre": old_learner.helper.idle_leases,
                "helper_idle_leases_post": update.state.helper.idle_leases,
                "beneficiary_idle_leases_pre": old_learner.beneficiary.idle_leases,
                "beneficiary_idle_leases_post": update.state.beneficiary.idle_leases,
                "helper_active_slot_pre": old_learner.helper.active_slot,
                "helper_active_slot_post": update.state.helper.active_slot,
                "beneficiary_active_slot_pre": old_learner.beneficiary.active_slot,
                "beneficiary_active_slot_post": update.state.beneficiary.active_slot,
                "helper_status_pre": old_learner.helper.status,
                "helper_status_post": update.state.helper.status,
                "beneficiary_status_pre": old_learner.beneficiary.status,
                "beneficiary_status_post": update.state.beneficiary.status,
                "helper_generation_pre": old_learner.helper.generation,
                "helper_generation_post": update.state.helper.generation,
                "beneficiary_generation_pre": old_learner.beneficiary.generation,
                "beneficiary_generation_post": update.state.beneficiary.generation,
                "helper_candidate_confirmations_pre": (
                    old_learner.helper.candidate_successful_leases
                ),
                "helper_candidate_confirmations_post": (
                    update.state.helper.candidate_successful_leases
                ),
                "beneficiary_candidate_confirmations_pre": (
                    old_learner.beneficiary.candidate_successful_leases
                ),
                "beneficiary_candidate_confirmations_post": (
                    update.state.beneficiary.candidate_successful_leases
                ),
                "helper_lease_offset_pre": old_learner.helper.lease_offset,
                "helper_lease_offset_post": update.state.helper.lease_offset,
                "beneficiary_lease_offset_pre": old_learner.beneficiary.lease_offset,
                "beneficiary_lease_offset_post": update.state.beneficiary.lease_offset,
                "helper_lease_reward_sum_pre": old_learner.helper.lease_reward_sum,
                "helper_lease_reward_sum_post": update.state.helper.lease_reward_sum,
                "beneficiary_lease_reward_sum_pre": old_learner.beneficiary.lease_reward_sum,
                "beneficiary_lease_reward_sum_post": update.state.beneficiary.lease_reward_sum,
                "helper_remaining_durable_tests_pre": (
                    old_learner.helper.remaining_durable_tests
                ),
                "helper_remaining_durable_tests_post": (
                    update.state.helper.remaining_durable_tests
                ),
                "beneficiary_remaining_durable_tests_pre": (
                    old_learner.beneficiary.remaining_durable_tests
                ),
                "beneficiary_remaining_durable_tests_post": (
                    update.state.beneficiary.remaining_durable_tests
                ),
                "helper_search_cursor_pre": old_learner.helper.search_cursor,
                "helper_search_cursor_post": update.state.helper.search_cursor,
                "beneficiary_search_cursor_pre": old_learner.beneficiary.search_cursor,
                "beneficiary_search_cursor_post": update.state.beneficiary.search_cursor,
                "helper_next_generation_pre": old_learner.helper.next_generation,
                "helper_next_generation_post": update.state.helper.next_generation,
                "beneficiary_next_generation_pre": old_learner.beneficiary.next_generation,
                "beneficiary_next_generation_post": update.state.beneficiary.next_generation,
                "helper_policy_key_data_pre": jr.key_data(old_learner.helper.key),
                "helper_policy_key_data_post": jr.key_data(update.state.helper.key),
                "beneficiary_policy_key_data_pre": jr.key_data(
                    old_learner.beneficiary.key
                ),
                "beneficiary_policy_key_data_post": jr.key_data(
                    update.state.beneficiary.key
                ),
                "helper_scratch_failed_leases_pre": update.helper.scratch_failed_leases_pre,
                "helper_scratch_failed_leases_post": update.helper.scratch_failed_leases_post,
                "beneficiary_scratch_failed_leases_pre": (
                    update.beneficiary.scratch_failed_leases_pre
                ),
                "beneficiary_scratch_failed_leases_post": (
                    update.beneficiary.scratch_failed_leases_post
                ),
                "helper_scratch_retest_started": update.helper.scratch_retest_started,
                "beneficiary_scratch_retest_started": (
                    update.beneficiary.scratch_retest_started
                ),
                "helper_value_write": update.helper.value_write,
                "beneficiary_value_write": update.beneficiary.value_write,
                "helper_value_pre": update.helper.value_pre,
                "helper_candidate_value": update.helper.candidate_value,
                "helper_value_post": update.helper.value_post,
                "beneficiary_value_pre": update.beneficiary.value_pre,
                "beneficiary_candidate_value": update.beneficiary.candidate_value,
                "beneficiary_value_post": update.beneficiary.value_post,
                "helper_lease_boundary": update.helper.lease_boundary,
                "beneficiary_lease_boundary": update.beneficiary.lease_boundary,
                "helper_lease_reward_mean": update.helper.lease_reward_mean,
                "beneficiary_lease_reward_mean": update.beneficiary.lease_reward_mean,
                "helper_relevance_ready": update.helper.relevance_ready,
                "beneficiary_relevance_ready": update.beneficiary.relevance_ready,
                "helper_durable_relevant": update.helper.durable_relevant,
                "beneficiary_durable_relevant": update.beneficiary.durable_relevant,
                "helper_candidate_relevant": update.helper.candidate_relevant,
                "beneficiary_candidate_relevant": update.beneficiary.candidate_relevant,
                "helper_candidate_lease_success": update.helper.candidate_lease_success,
                "beneficiary_candidate_lease_success": (
                    update.beneficiary.candidate_lease_success
                ),
                "helper_generation_exhausted": update.helper.generation_exhausted,
                "beneficiary_generation_exhausted": update.beneficiary.generation_exhausted,
                "helper_committed_slot": update.helper.committed_slot,
                "helper_committed_generation": update.helper.committed_generation,
                "helper_retired_slot": update.helper.retired_slot,
                "helper_retired_generation": update.helper.retired_generation,
                "beneficiary_committed_slot": update.beneficiary.committed_slot,
                "beneficiary_committed_generation": update.beneficiary.committed_generation,
                "beneficiary_retired_slot": update.beneficiary.retired_slot,
                "beneficiary_retired_generation": update.beneficiary.retired_generation,
                "lifecycle_synchronized": update.lifecycle_synchronized,
                "helper_value_bits_pre": old_helper_bits,
                "helper_value_bits_post": new_helper_bits,
                "beneficiary_value_bits_pre": old_beneficiary_bits,
                "beneficiary_value_bits_post": new_beneficiary_bits,
                "helper_selective_mutation_violation": helper_violation,
                "beneficiary_selective_mutation_violation": beneficiary_violation,
            }
            return (next_world, update.state), output

        return jax.lax.scan(
            step,
            (world_state, learner_state),
            jnp.arange(num_steps, dtype=jnp.int32),
        )

    return jax.jit(run)


def _make_trace(outputs: Mapping[str, Array]) -> HiddenRegimePrimitiveTrace:
    expected_fields = {field.name for field in dataclasses.fields(HiddenRegimePrimitiveTrace)}
    if set(outputs) != expected_fields:
        raise ValueError("scan outputs do not match the complete primitive trace")
    int32_fields = {
        "step_index",
        "segment_index",
        "segment_step",
        "world_step_count_pre",
        "world_step_count_post",
        "world_schedule_position_pre",
        "world_schedule_position_post",
        "helper_decision_slot",
        "helper_private_input",
        "helper_decision_action",
        "beneficiary_decision_slot",
        "beneficiary_private_input",
        "beneficiary_decision_action",
        "helper_failed_leases_pre",
        "helper_failed_leases_post",
        "beneficiary_failed_leases_pre",
        "beneficiary_failed_leases_post",
        "helper_idle_leases_pre",
        "helper_idle_leases_post",
        "beneficiary_idle_leases_pre",
        "beneficiary_idle_leases_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
        "helper_candidate_confirmations_pre",
        "helper_candidate_confirmations_post",
        "beneficiary_candidate_confirmations_pre",
        "beneficiary_candidate_confirmations_post",
        "helper_scratch_failed_leases_pre",
        "helper_scratch_failed_leases_post",
        "beneficiary_scratch_failed_leases_pre",
        "beneficiary_scratch_failed_leases_post",
        "helper_lease_offset_pre",
        "helper_lease_offset_post",
        "beneficiary_lease_offset_pre",
        "beneficiary_lease_offset_post",
        "helper_remaining_durable_tests_pre",
        "helper_remaining_durable_tests_post",
        "beneficiary_remaining_durable_tests_pre",
        "beneficiary_remaining_durable_tests_post",
        "helper_search_cursor_pre",
        "helper_search_cursor_post",
        "beneficiary_search_cursor_pre",
        "beneficiary_search_cursor_post",
        "helper_next_generation_pre",
        "helper_next_generation_post",
        "beneficiary_next_generation_pre",
        "beneficiary_next_generation_post",
        "helper_committed_slot",
        "helper_committed_generation",
        "helper_retired_slot",
        "helper_retired_generation",
        "beneficiary_committed_slot",
        "beneficiary_committed_generation",
        "beneficiary_retired_slot",
        "beneficiary_retired_generation",
    }
    int8_fields = {
        "regime_id",
        "world_cue_pre",
        "world_cue_post",
        "helper_cue",
        "oracle_target",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "helper_active_slot_pre",
        "helper_active_slot_post",
        "beneficiary_active_slot_pre",
        "beneficiary_active_slot_post",
        "helper_status_pre",
        "helper_status_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
    }
    bool_fields = {
        "world_terminated",
        "helper_write_enabled",
        "beneficiary_write_enabled",
        "helper_value_write",
        "beneficiary_value_write",
        "helper_lease_boundary",
        "beneficiary_lease_boundary",
        "helper_candidate_lease_success",
        "beneficiary_candidate_lease_success",
        "helper_relevance_ready",
        "beneficiary_relevance_ready",
        "helper_durable_relevant",
        "beneficiary_durable_relevant",
        "helper_candidate_relevant",
        "beneficiary_candidate_relevant",
        "helper_generation_exhausted",
        "beneficiary_generation_exhausted",
        "helper_scratch_retest_started",
        "beneficiary_scratch_retest_started",
        "lifecycle_synchronized",
        "helper_selective_mutation_violation",
        "beneficiary_selective_mutation_violation",
    }
    uint_fields = {
        "helper_value_bits_pre",
        "helper_value_bits_post",
        "beneficiary_value_bits_pre",
        "beneficiary_value_bits_post",
        "world_cue_key_data_pre",
        "world_cue_key_data_post",
        "world_channel_key_data_pre",
        "world_channel_key_data_post",
        "helper_policy_key_data_pre",
        "helper_policy_key_data_post",
        "beneficiary_policy_key_data_pre",
        "beneficiary_policy_key_data_post",
    }
    converted: dict[str, Array] = {}
    for field in dataclasses.fields(HiddenRegimePrimitiveTrace):
        value = outputs[field.name]
        if field.name in int32_fields:
            converted[field.name] = value.astype(jnp.int32)
        elif field.name in int8_fields:
            converted[field.name] = value.astype(jnp.int8)
        elif field.name in bool_fields:
            converted[field.name] = value.astype(jnp.bool_)
        elif field.name in uint_fields:
            converted[field.name] = value.astype(jnp.uint32)
        else:
            converted[field.name] = value.astype(jnp.float32)
    return HiddenRegimePrimitiveTrace(**converted)


_REGIME_LABELS = {
    0: "A",
    1: "B",
    2: "C-old",
    3: "C-new",
    4: "D-short",
}


def _label(regime: int) -> str:
    return _REGIME_LABELS.get(regime, f"regime-{regime}")


def _segment_summaries(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[tuple[SegmentRewardSummary, ...], tuple[RegimeRecurrenceSummary, ...]]:
    rewards = np.asarray(trace.reward, dtype=np.float32)
    segments = np.asarray(trace.segment_index, dtype=np.int32)
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    occurrences: dict[int, int] = {}
    previous_late: dict[int, float] = {}
    summaries: list[SegmentRewardSummary] = []
    for segment_index, expected_regime in enumerate(config.world.segment_regimes):
        indices = np.flatnonzero(segments == segment_index)
        if indices.size == 0:
            continue
        regime = int(regimes[indices[0]])
        if regime != expected_regime:
            raise ValueError("trace regime does not match evaluator schedule")
        window = min(config.metric_window, int(indices.size))
        values = rewards[indices]
        early = float(np.mean(values[:window], dtype=np.float64))
        late = float(np.mean(values[-window:], dtype=np.float64))
        occurrence = occurrences.get(regime, 0)
        prior = previous_late.get(regime)
        summaries.append(
            SegmentRewardSummary(
                segment_index=segment_index,
                regime_id=regime,
                regime_label=_label(regime),
                occurrence_index=occurrence,
                steps=int(indices.size),
                mean_reward=float(np.mean(values, dtype=np.float64)),
                early_reward=early,
                late_reward=late,
                is_recurrence=occurrence > 0,
                previous_occurrence_late_reward=prior,
                recurrence_entry_change=None if prior is None else early - prior,
                within_segment_recovery=late - early,
            )
        )
        occurrences[regime] = occurrence + 1
        previous_late[regime] = late

    recurrence: list[RegimeRecurrenceSummary] = []
    for regime in sorted(occurrences):
        items = [item for item in summaries if item.regime_id == regime]
        recurrent = [item for item in items if item.is_recurrence]
        recurrence.append(
            RegimeRecurrenceSummary(
                regime_id=regime,
                regime_label=_label(regime),
                occurrences=len(items),
                recurrence_count=len(recurrent),
                recurrence_entry_reward_mean=(
                    None
                    if not recurrent
                    else float(np.mean([item.early_reward for item in recurrent]))
                ),
                previous_late_reward_mean=(
                    None
                    if not recurrent
                    else float(
                        np.mean(
                            [
                                cast(float, item.previous_occurrence_late_reward)
                                for item in recurrent
                            ]
                        )
                    )
                ),
                recurrence_entry_change_mean=(
                    None
                    if not recurrent
                    else float(
                        np.mean([cast(float, item.recurrence_entry_change) for item in recurrent])
                    )
                ),
                recurrence_recovery_mean=(
                    None
                    if not recurrent
                    else float(np.mean([item.within_segment_recovery for item in recurrent]))
                ),
            )
        )
    return tuple(summaries), tuple(recurrence)


def _float_bank_from_bits(values: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Decode one uint32 value bank without numeric conversion."""

    return np.asarray(values, dtype=np.uint32).view(np.float32)


def _composed_greedy_accuracy(
    helper_values: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary_values: np.ndarray[Any, np.dtype[np.float32]],
    target_permutation: np.ndarray[Any, np.dtype[np.int32]],
) -> float:
    """Evaluate all three cues with deterministic first-index greedy ties."""

    helper_messages = np.argmax(helper_values, axis=1)
    beneficiary_actions = np.argmax(beneficiary_values[helper_messages], axis=1)
    return float(np.mean(beneficiary_actions == target_permutation, dtype=np.float64))


def _composed_greedy_mapping(
    helper_values: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary_values: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[int, int, int]:
    """Return the exact deterministic composed action for each private cue."""

    helper_messages = np.argmax(helper_values, axis=1)
    beneficiary_actions = np.argmax(beneficiary_values[helper_messages], axis=1)
    return cast(tuple[int, int, int], tuple(int(action) for action in beneficiary_actions))


def _composed_greedy_tie_free(
    helper_values: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary_values: np.ndarray[Any, np.dtype[np.float32]],
) -> bool:
    """Return whether every decision on the composed greedy path has one maximum."""

    helper_maxima = np.max(helper_values, axis=1, keepdims=True)
    helper_unique = np.count_nonzero(helper_values == helper_maxima, axis=1) == 1
    helper_messages = np.argmax(helper_values, axis=1)
    selected_beneficiary = beneficiary_values[helper_messages]
    beneficiary_maxima = np.max(selected_beneficiary, axis=1, keepdims=True)
    beneficiary_unique = (
        np.count_nonzero(selected_beneficiary == beneficiary_maxima, axis=1) == 1
    )
    return bool(np.all(helper_unique) and np.all(beneficiary_unique))


def _uint32_table_tuple(values: object) -> tuple[int, ...]:
    """Flatten one 3x3 table into canonical row-major uint32 words."""

    table = np.asarray(values, dtype=np.uint32)
    if table.shape != (3, 3):
        raise ValueError("lineage table must have exact shape (3, 3)")
    return tuple(int(word) for word in table.reshape(-1))


def reconstruct_commit_generation_lineages(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[CommitGenerationLineage, ...]:
    """Reconstruct every synchronized commit from primitive post-state bits.

    Qualification is predeclared solely from commit-time content: the composed
    greedy mapping must exactly equal all three targets of the commit regime,
    and every helper and selected-beneficiary argmax on that path must be
    unique.  No later state, recurrence reward, or post-entry behavior
    participates.
    """

    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        raise TypeError("trace must be a HiddenRegimePrimitiveTrace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        raise TypeError("config must be a HiddenRegimeDevelopmentConfig")
    helper_slots = np.asarray(trace.helper_committed_slot, dtype=np.int32)
    helper_generations = np.asarray(trace.helper_committed_generation, dtype=np.int32)
    beneficiary_slots = np.asarray(trace.beneficiary_committed_slot, dtype=np.int32)
    beneficiary_generations = np.asarray(
        trace.beneficiary_committed_generation,
        dtype=np.int32,
    )
    synchronized = np.asarray(trace.lifecycle_synchronized, dtype=np.bool_)
    helper_bits = np.asarray(trace.helper_value_bits_post, dtype=np.uint32)
    beneficiary_bits = np.asarray(trace.beneficiary_value_bits_post, dtype=np.uint32)
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    segments = np.asarray(trace.segment_index, dtype=np.int32)
    segment_steps = np.asarray(trace.segment_step, dtype=np.int32)
    permutations = np.asarray(config.world.regime_permutations, dtype=np.int32)
    lineages: list[CommitGenerationLineage] = []
    commit_events = np.logical_or(helper_slots >= 0, beneficiary_slots >= 0)
    for step_value in np.flatnonzero(commit_events):
        step = int(step_value)
        if (
            not synchronized[step]
            or helper_slots[step] != beneficiary_slots[step]
            or helper_generations[step] != beneficiary_generations[step]
            or helper_slots[step] < 1
            or helper_slots[step] >= N_SLOTS
            or helper_generations[step] < 1
        ):
            raise ValueError(
                "commit lineage requires one synchronized valid slot/generation event"
            )
        slot = int(helper_slots[step])
        generation = int(helper_generations[step])
        regime = int(regimes[step])
        helper_table_bits = helper_bits[step, slot]
        beneficiary_table_bits = beneficiary_bits[step, slot]
        helper_table = helper_table_bits.view(np.float32)
        beneficiary_table = beneficiary_table_bits.view(np.float32)
        if not np.all(np.isfinite(helper_table)) or not np.all(
            np.isfinite(beneficiary_table)
        ):
            raise ValueError("commit lineage table bits must decode to finite float32")
        target = permutations[regime]
        mapping = _composed_greedy_mapping(helper_table, beneficiary_table)
        accuracy = float(
            np.mean(np.asarray(mapping, dtype=np.int32) == target, dtype=np.float64)
        )
        tie_free = _composed_greedy_tie_free(helper_table, beneficiary_table)
        lineages.append(
            CommitGenerationLineage(
                lineage_index=len(lineages),
                commit_step=step,
                commit_segment_index=int(segments[step]),
                commit_segment_step=int(segment_steps[step]),
                regime_id=regime,
                regime_label=_label(regime),
                slot=slot,
                generation=generation,
                target_mapping=cast(
                    tuple[int, int, int],
                    tuple(int(action) for action in target),
                ),
                committed_composed_greedy_mapping=mapping,
                committed_composed_greedy_accuracy=accuracy,
                committed_composed_greedy_tie_free=tie_free,
                acquisition_qualified=accuracy == 1.0 and tie_free,
                helper_table_uint32_bits=_uint32_table_tuple(helper_table_bits),
                beneficiary_table_uint32_bits=_uint32_table_tuple(
                    beneficiary_table_bits
                ),
            )
        )
    return tuple(lineages)


def _entry_window(
    rewards: np.ndarray[Any, np.dtype[np.float32]],
    start: int,
    segment_steps: int,
    window: int,
) -> tuple[bool, float | None, int | None, float | None]:
    """Return one exact world-entry window or an explicit missing record."""

    if segment_steps < window:
        return False, None, None, None
    values = rewards[start : start + window]
    errors = int(window - int(np.sum(values, dtype=np.int64)))
    return (
        True,
        float(np.mean(values, dtype=np.float64)),
        errors,
        float(errors / window),
    )


def _optional_mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values, dtype=np.float64))


def _best_dormant_retention_summaries(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[tuple[RecurrenceRetentionRecord, ...], RetentionAggregateSummary]:
    """Derive the secondary all-dormant/best-table and adaptation descriptors."""

    rewards = np.asarray(trace.reward, dtype=np.float32)
    boundaries = np.asarray(trace.helper_lease_boundary, dtype=np.bool_)
    helper_active_pre = np.asarray(trace.helper_active_slot_pre, dtype=np.int32)
    helper_active_post = np.asarray(trace.helper_active_slot_post, dtype=np.int32)
    beneficiary_active_pre = np.asarray(trace.beneficiary_active_slot_pre, dtype=np.int32)
    beneficiary_active_post = np.asarray(trace.beneficiary_active_slot_post, dtype=np.int32)
    helper_status_pre = np.asarray(trace.helper_status_pre, dtype=np.int32)
    helper_status_post = np.asarray(trace.helper_status_post, dtype=np.int32)
    beneficiary_status_pre = np.asarray(trace.beneficiary_status_pre, dtype=np.int32)
    beneficiary_status_post = np.asarray(trace.beneficiary_status_post, dtype=np.int32)
    helper_generation_pre = np.asarray(trace.helper_generation_pre, dtype=np.int32)
    helper_generation_post = np.asarray(trace.helper_generation_post, dtype=np.int32)
    beneficiary_generation_pre = np.asarray(
        trace.beneficiary_generation_pre,
        dtype=np.int32,
    )
    beneficiary_generation_post = np.asarray(
        trace.beneficiary_generation_post,
        dtype=np.int32,
    )
    helper_durable_relevant = np.asarray(trace.helper_durable_relevant, dtype=np.bool_)
    beneficiary_durable_relevant = np.asarray(
        trace.beneficiary_durable_relevant,
        dtype=np.bool_,
    )
    helper_values_pre = _float_bank_from_bits(trace.helper_value_bits_pre)
    beneficiary_values_pre = _float_bank_from_bits(trace.beneficiary_value_bits_pre)
    permutations = np.asarray(config.world.regime_permutations, dtype=np.int32)
    lease_length = config.learner.lease_length

    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(np.asarray(config.world.segment_lengths[:-1]), dtype=np.int64),
        )
    )
    first_occurrence_by_regime: dict[int, tuple[int, int, int]] = {}
    occurrence_count: dict[int, int] = {}
    records: list[RecurrenceRetentionRecord] = []
    for segment_index, (start_value, segment_steps, regime) in enumerate(
        zip(
            starts,
            config.world.segment_lengths,
            config.world.segment_regimes,
            strict=True,
        )
    ):
        start = int(start_value)
        end = start + segment_steps
        occurrence = occurrence_count.get(regime, 0)
        current_window = _entry_window(rewards, start, segment_steps, lease_length)
        if occurrence == 0:
            first_occurrence_by_regime[regime] = (segment_index, start, segment_steps)
            occurrence_count[regime] = 1
            continue

        acquisition_segment, acquisition_start, acquisition_steps = first_occurrence_by_regime[
            regime
        ]
        acquisition_window = _entry_window(
            rewards,
            acquisition_start,
            acquisition_steps,
            lease_length,
        )
        current_complete, current_reward, current_errors, current_error_rate = current_window
        (
            acquisition_complete,
            acquisition_reward,
            acquisition_errors,
            acquisition_error_rate,
        ) = acquisition_window
        comparison_available = current_complete and acquisition_complete
        error_delta = (
            None
            if not comparison_available
            else cast(float, current_error_rate) - cast(float, acquisition_error_rate)
        )
        ratio_defined = bool(
            comparison_available and cast(float, acquisition_error_rate) > 0.0
        )
        error_ratio = (
            cast(float, current_error_rate) / cast(float, acquisition_error_rate)
            if ratio_defined
            else None
        )

        helper_entry_status = helper_status_pre[start]
        beneficiary_entry_status = beneficiary_status_pre[start]
        helper_entry_generation = helper_generation_pre[start]
        beneficiary_entry_generation = beneficiary_generation_pre[start]
        helper_entry_active = int(helper_active_pre[start])
        beneficiary_entry_active = int(beneficiary_active_pre[start])
        eligible_dormant_generations: list[DormantGenerationProbe] = []
        for slot in range(1, N_SLOTS):
            generation = int(helper_entry_generation[slot])
            if (
                helper_entry_status[slot] != SLOT_DURABLE
                or beneficiary_entry_status[slot] != SLOT_DURABLE
                or generation <= 0
                or int(beneficiary_entry_generation[slot]) != generation
                or slot == helper_entry_active
                or slot == beneficiary_entry_active
            ):
                continue
            helper_table = helper_values_pre[start, slot]
            beneficiary_table = beneficiary_values_pre[start, slot]
            target = permutations[regime]
            composed = _composed_greedy_accuracy(helper_table, beneficiary_table, target)
            zero_helper = _composed_greedy_accuracy(
                np.zeros_like(helper_table),
                beneficiary_table,
                target,
            )
            zero_beneficiary = _composed_greedy_accuracy(
                helper_table,
                np.zeros_like(beneficiary_table),
                target,
            )
            role_swapped = _composed_greedy_accuracy(
                beneficiary_table,
                helper_table,
                target,
            )
            eligible_dormant_generations.append(
                DormantGenerationProbe(
                    slot=slot,
                    generation=generation,
                    composed_greedy_accuracy=composed,
                    zero_helper_accuracy=zero_helper,
                    zero_beneficiary_accuracy=zero_beneficiary,
                    role_swapped_accuracy=role_swapped,
                )
            )
        best = (
            None
            if not eligible_dormant_generations
            else min(
                eligible_dormant_generations,
                key=lambda item: (-item.composed_greedy_accuracy, item.slot),
            )
        )
        best_slot = None if best is None else best.slot
        best_generation = None if best is None else best.generation

        segment_boundaries = np.flatnonzero(boundaries[start:end]) + start
        relock_step: int | None = None
        if best_slot is not None and best_generation is not None:
            for step_value in segment_boundaries:
                step = int(step_value)
                slot = best_slot
                generation = best_generation
                exact_relock = (
                    helper_active_pre[step] == slot
                    and beneficiary_active_pre[step] == slot
                    and helper_active_post[step] == slot
                    and beneficiary_active_post[step] == slot
                    and helper_status_pre[step, slot] == SLOT_DURABLE
                    and beneficiary_status_pre[step, slot] == SLOT_DURABLE
                    and helper_status_post[step, slot] == SLOT_DURABLE
                    and beneficiary_status_post[step, slot] == SLOT_DURABLE
                    and helper_generation_pre[step, slot] == generation
                    and beneficiary_generation_pre[step, slot] == generation
                    and helper_generation_post[step, slot] == generation
                    and beneficiary_generation_post[step, slot] == generation
                    and helper_durable_relevant[step]
                    and beneficiary_durable_relevant[step]
                )
                if exact_relock:
                    relock_step = step
                    break
        scratch_scan_end = end if relock_step is None else relock_step + 1
        scratch_in_prefix = bool(
            np.any(helper_active_pre[start:scratch_scan_end] == SCRATCH_SLOT)
            or np.any(beneficiary_active_pre[start:scratch_scan_end] == SCRATCH_SLOT)
        )
        boundaries_until_relock = (
            None
            if relock_step is None
            else int(np.count_nonzero(boundaries[start : relock_step + 1]))
        )
        entry_offset = int(np.asarray(trace.helper_lease_offset_pre)[start])
        records.append(
            RecurrenceRetentionRecord(
                segment_index=segment_index,
                regime_id=regime,
                regime_label=_label(regime),
                occurrence_index=occurrence,
                legacy_first_exposure_segment_index=acquisition_segment,
                world_entry_learner_lease_offset=entry_offset,
                world_entry_steps_to_first_learner_boundary=lease_length - entry_offset,
                first_world_window_length=lease_length,
                first_world_window_complete=current_complete,
                first_world_window_reward=current_reward,
                first_world_window_errors=current_errors,
                first_world_window_error_rate=current_error_rate,
                legacy_first_exposure_world_window_complete=acquisition_complete,
                legacy_first_exposure_world_window_reward=acquisition_reward,
                legacy_first_exposure_world_window_errors=acquisition_errors,
                legacy_first_exposure_world_window_error_rate=acquisition_error_rate,
                legacy_recurrence_minus_first_exposure_error_rate_delta=error_delta,
                legacy_recurrence_to_first_exposure_error_rate_ratio=error_ratio,
                legacy_recurrence_to_first_exposure_error_rate_ratio_defined=ratio_defined,
                dormant_probe_available=best is not None,
                eligible_dormant_generations=tuple(eligible_dormant_generations),
                best_dormant_slot=best_slot,
                best_dormant_generation=best_generation,
                best_dormant_composed_greedy_accuracy=(
                    None if best is None else best.composed_greedy_accuracy
                ),
                best_dormant_zero_helper_accuracy=(
                    None if best is None else best.zero_helper_accuracy
                ),
                best_dormant_zero_beneficiary_accuracy=(
                    None if best is None else best.zero_beneficiary_accuracy
                ),
                best_dormant_role_swapped_accuracy=(
                    None if best is None else best.role_swapped_accuracy
                ),
                exact_generation_relock_observed=relock_step is not None,
                first_exact_generation_relock_step=relock_step,
                first_exact_generation_relock_segment_step=(
                    None if relock_step is None else relock_step - start
                ),
                observed_learner_boundaries_in_segment=int(segment_boundaries.size),
                observed_learner_boundaries_until_relock=boundaries_until_relock,
                scratch_entered_before_relock=(
                    scratch_in_prefix if relock_step is not None else None
                ),
                scratch_entered_before_relock_or_segment_end=scratch_in_prefix,
                durable_retrieval_before_scratch=(
                    relock_step is not None and not scratch_in_prefix
                ),
            )
        )
        occurrence_count[regime] = occurrence + 1

    recurrence_count = len(records)
    complete_windows = [
        cast(float, item.first_world_window_reward)
        for item in records
        if item.first_world_window_complete
    ]
    complete_error_rates = [
        cast(float, item.first_world_window_error_rate)
        for item in records
        if item.first_world_window_complete
    ]
    dormant = [item for item in records if item.dormant_probe_available]
    relocked = [item for item in records if item.exact_generation_relock_observed]
    retrieved = [item for item in records if item.durable_retrieval_before_scratch]
    comparisons = [
        item
        for item in records
        if item.legacy_recurrence_minus_first_exposure_error_rate_delta is not None
    ]
    ratio_defined_records = [
        item
        for item in comparisons
        if item.legacy_recurrence_to_first_exposure_error_rate_ratio_defined
    ]

    def fraction(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else float(numerator / denominator)

    aggregate = RetentionAggregateSummary(
        recurrence_count=recurrence_count,
        complete_first_world_window_count=len(complete_windows),
        missing_first_world_window_count=recurrence_count - len(complete_windows),
        first_world_window_reward_mean=_optional_mean(complete_windows),
        first_world_window_error_rate_mean=_optional_mean(complete_error_rates),
        dormant_probe_available_count=len(dormant),
        dormant_probe_missing_count=recurrence_count - len(dormant),
        dormant_composed_greedy_accuracy_mean=_optional_mean(
            [cast(float, item.best_dormant_composed_greedy_accuracy) for item in dormant]
        ),
        dormant_zero_helper_accuracy_mean=_optional_mean(
            [cast(float, item.best_dormant_zero_helper_accuracy) for item in dormant]
        ),
        dormant_zero_beneficiary_accuracy_mean=_optional_mean(
            [cast(float, item.best_dormant_zero_beneficiary_accuracy) for item in dormant]
        ),
        dormant_role_swapped_accuracy_mean=_optional_mean(
            [cast(float, item.best_dormant_role_swapped_accuracy) for item in dormant]
        ),
        exact_generation_relock_count=len(relocked),
        exact_generation_relock_missing_count=recurrence_count - len(relocked),
        exact_generation_relock_rate_all_recurrences=fraction(
            len(relocked),
            recurrence_count,
        ),
        exact_generation_relock_rate_given_dormant_probe=fraction(
            len(relocked),
            len(dormant),
        ),
        observed_learner_boundaries_to_relock_mean=_optional_mean(
            [cast(int, item.observed_learner_boundaries_until_relock) for item in relocked]
        ),
        observed_learner_boundaries_to_relock_missing_count=(
            recurrence_count - len(relocked)
        ),
        durable_retrieval_before_scratch_count=len(retrieved),
        durable_retrieval_before_scratch_fraction_all_recurrences=fraction(
            len(retrieved),
            recurrence_count,
        ),
        durable_retrieval_before_scratch_fraction_given_dormant_probe=fraction(
            len(retrieved),
            len(dormant),
        ),
        legacy_first_exposure_comparison_available_count=len(comparisons),
        legacy_first_exposure_comparison_missing_count=(
            recurrence_count - len(comparisons)
        ),
        legacy_recurrence_minus_first_exposure_error_rate_delta_mean=_optional_mean(
            [
                cast(
                    float,
                    item.legacy_recurrence_minus_first_exposure_error_rate_delta,
                )
                for item in comparisons
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_mean=_optional_mean(
            [
                cast(
                    float,
                    item.legacy_recurrence_to_first_exposure_error_rate_ratio,
                )
                for item in ratio_defined_records
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_defined_count=(
            len(ratio_defined_records)
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_undefined_count=(
            len(comparisons) - len(ratio_defined_records)
        ),
    )
    return tuple(records), aggregate


def _lineage_probe_at_entry(
    lineage: CommitGenerationLineage,
    *,
    start: int,
    trace: HiddenRegimePrimitiveTrace,
) -> RecurrenceLineageProbe:
    """Probe one immutable commit identity at a recurrence entry state."""

    slot = lineage.slot
    helper_status = int(np.asarray(trace.helper_status_pre)[start, slot])
    helper_generation = int(np.asarray(trace.helper_generation_pre)[start, slot])
    beneficiary_status = int(np.asarray(trace.beneficiary_status_pre)[start, slot])
    beneficiary_generation = int(
        np.asarray(trace.beneficiary_generation_pre)[start, slot]
    )
    helper_present = helper_status == SLOT_DURABLE and helper_generation == lineage.generation
    beneficiary_present = (
        beneficiary_status == SLOT_DURABLE
        and beneficiary_generation == lineage.generation
    )
    survives = helper_present and beneficiary_present
    helper_active = helper_present and int(np.asarray(trace.helper_active_slot_pre)[start]) == slot
    beneficiary_active = (
        beneficiary_present
        and int(np.asarray(trace.beneficiary_active_slot_pre)[start]) == slot
    )
    if not survives:
        activity: LineageEntryActivity = "unavailable"
    elif helper_active and beneficiary_active:
        activity = "active"
    elif not helper_active and not beneficiary_active:
        activity = "dormant"
    else:
        activity = "mixed"

    helper_bits_array = np.asarray(trace.helper_value_bits_pre, dtype=np.uint32)[
        start,
        slot,
    ]
    beneficiary_bits_array = np.asarray(
        trace.beneficiary_value_bits_pre,
        dtype=np.uint32,
    )[start, slot]
    helper_entry_bits = _uint32_table_tuple(helper_bits_array) if helper_present else None
    beneficiary_entry_bits = (
        _uint32_table_tuple(beneficiary_bits_array) if beneficiary_present else None
    )
    helper_exact = (
        helper_entry_bits == lineage.helper_table_uint32_bits
        if helper_entry_bits is not None
        else False
    )
    beneficiary_exact = (
        beneficiary_entry_bits == lineage.beneficiary_table_uint32_bits
        if beneficiary_entry_bits is not None
        else False
    )

    mapping: tuple[int, int, int] | None = None
    accuracy: float | None = None
    accuracy_change: float | None = None
    zero_helper: float | None = None
    zero_beneficiary: float | None = None
    role_swapped: float | None = None
    if survives:
        helper_table = helper_bits_array.view(np.float32)
        beneficiary_table = beneficiary_bits_array.view(np.float32)
        if not np.all(np.isfinite(helper_table)) or not np.all(
            np.isfinite(beneficiary_table)
        ):
            raise ValueError("surviving lineage tables must decode to finite float32")
        target = np.asarray(lineage.target_mapping, dtype=np.int32)
        mapping = _composed_greedy_mapping(helper_table, beneficiary_table)
        accuracy = float(
            np.mean(np.asarray(mapping, dtype=np.int32) == target, dtype=np.float64)
        )
        accuracy_change = accuracy - lineage.committed_composed_greedy_accuracy
        zero_helper = _composed_greedy_accuracy(
            np.zeros_like(helper_table),
            beneficiary_table,
            target,
        )
        zero_beneficiary = _composed_greedy_accuracy(
            helper_table,
            np.zeros_like(beneficiary_table),
            target,
        )
        role_swapped = _composed_greedy_accuracy(
            beneficiary_table,
            helper_table,
            target,
        )

    return RecurrenceLineageProbe(
        lineage_index=lineage.lineage_index,
        commit_step=lineage.commit_step,
        commit_segment_index=lineage.commit_segment_index,
        slot=slot,
        generation=lineage.generation,
        acquisition_qualified=lineage.acquisition_qualified,
        helper_entry_slot_status=helper_status,
        helper_entry_slot_generation=helper_generation,
        helper_slot_generation_present=helper_present,
        beneficiary_entry_slot_status=beneficiary_status,
        beneficiary_entry_slot_generation=beneficiary_generation,
        beneficiary_slot_generation_present=beneficiary_present,
        synchronized_generation_survives=survives,
        helper_active_at_entry=helper_active,
        beneficiary_active_at_entry=beneficiary_active,
        entry_activity_status=activity,
        helper_entry_table_uint32_bits=helper_entry_bits,
        beneficiary_entry_table_uint32_bits=beneficiary_entry_bits,
        entry_composed_greedy_mapping=mapping,
        entry_composed_greedy_accuracy=accuracy,
        entry_minus_commit_accuracy=accuracy_change,
        helper_bit_exact_preserved=helper_exact,
        beneficiary_bit_exact_preserved=beneficiary_exact,
        joint_bit_exact_preserved=helper_exact and beneficiary_exact,
        zero_helper_accuracy=zero_helper,
        zero_beneficiary_accuracy=zero_beneficiary,
        role_swapped_accuracy=role_swapped,
    )


def _selected_lineage_relock(
    selected: RecurrenceLineageProbe | None,
    *,
    start: int,
    end: int,
    trace: HiddenRegimePrimitiveTrace,
) -> dict[str, Any]:
    """Scan exact selected slot/generation retrieval and scratch ordering.

    A lineage active in both roles at entry has zero-step retrieval.  A dormant
    lineage relocks only after one full selected, durable-relevant lease whose
    boundary has the same slot/generation in both pre and post state; merely
    switching into the slot is not counted.  Scratch entry is an event at a
    transition if either pre-state is scratch or either post-state switches to
    scratch, with explicit ``pre``/``post`` phase ordering and duplicate
    pre/post occupancy collapsed to the pre event.  Thus an entry retrieval can
    precede a same-transition post switch, and a final switch cannot disappear.
    """

    if selected is None:
        return {
            "selected_exact_generation_relock_observed": None,
            "selected_first_exact_generation_relock_step": None,
            "selected_first_exact_generation_relock_segment_step": None,
            "selected_exact_generation_relock_phase": None,
            "selected_observed_learner_boundaries_until_relock": None,
            "selected_first_scratch_entry_step": None,
            "selected_first_scratch_entry_segment_step": None,
            "selected_first_scratch_entry_phase": None,
            "selected_scratch_entered_before_relock": None,
            "selected_scratch_entered_before_relock_or_segment_end": None,
            "selected_durable_retrieval_before_scratch": None,
        }
    immediate_active = selected.entry_activity_status == "active"
    slot = selected.slot
    generation = selected.generation
    helper_boundary = np.asarray(trace.helper_lease_boundary, dtype=np.bool_)
    beneficiary_boundary = np.asarray(
        trace.beneficiary_lease_boundary,
        dtype=np.bool_,
    )
    boundaries = np.logical_and(helper_boundary, beneficiary_boundary)
    boundary_steps = np.flatnonzero(boundaries[start:end]) + start
    helper_active_pre = np.asarray(trace.helper_active_slot_pre, dtype=np.int32)
    helper_active_post = np.asarray(trace.helper_active_slot_post, dtype=np.int32)
    beneficiary_active_pre = np.asarray(
        trace.beneficiary_active_slot_pre,
        dtype=np.int32,
    )
    beneficiary_active_post = np.asarray(
        trace.beneficiary_active_slot_post,
        dtype=np.int32,
    )
    helper_status_pre = np.asarray(trace.helper_status_pre, dtype=np.int32)
    helper_status_post = np.asarray(trace.helper_status_post, dtype=np.int32)
    beneficiary_status_pre = np.asarray(trace.beneficiary_status_pre, dtype=np.int32)
    beneficiary_status_post = np.asarray(trace.beneficiary_status_post, dtype=np.int32)
    helper_generation_pre = np.asarray(trace.helper_generation_pre, dtype=np.int32)
    helper_generation_post = np.asarray(trace.helper_generation_post, dtype=np.int32)
    beneficiary_generation_pre = np.asarray(
        trace.beneficiary_generation_pre,
        dtype=np.int32,
    )
    beneficiary_generation_post = np.asarray(
        trace.beneficiary_generation_post,
        dtype=np.int32,
    )
    helper_relevant = np.asarray(trace.helper_durable_relevant, dtype=np.bool_)
    beneficiary_relevant = np.asarray(
        trace.beneficiary_durable_relevant,
        dtype=np.bool_,
    )

    relock_step: int | None = start if immediate_active else None
    relock_phase: TransitionEventPhase | None = "pre" if immediate_active else None
    if not immediate_active:
        for step_value in boundary_steps:
            step = int(step_value)
            if (
                helper_active_pre[step] == slot
                and beneficiary_active_pre[step] == slot
                and helper_active_post[step] == slot
                and beneficiary_active_post[step] == slot
                and helper_status_pre[step, slot] == SLOT_DURABLE
                and beneficiary_status_pre[step, slot] == SLOT_DURABLE
                and helper_status_post[step, slot] == SLOT_DURABLE
                and beneficiary_status_post[step, slot] == SLOT_DURABLE
                and helper_generation_pre[step, slot] == generation
                and beneficiary_generation_pre[step, slot] == generation
                and helper_generation_post[step, slot] == generation
                and beneficiary_generation_post[step, slot] == generation
                and helper_relevant[step]
                and beneficiary_relevant[step]
            ):
                relock_step = step
                relock_phase = "post"
                break

    first_scratch_step: int | None = None
    first_scratch_phase: TransitionEventPhase | None = None
    for step in range(start, end):
        pre_scratch = (
            helper_active_pre[step] == SCRATCH_SLOT
            or beneficiary_active_pre[step] == SCRATCH_SLOT
        )
        post_scratch = (
            helper_active_post[step] == SCRATCH_SLOT
            or beneficiary_active_post[step] == SCRATCH_SLOT
        )
        if pre_scratch:
            first_scratch_step = step
            first_scratch_phase = "pre"
            break
        if post_scratch:
            first_scratch_step = step
            first_scratch_phase = "post"
            break
    phase_order = {"pre": 0, "post": 1}
    scratch_event = (
        None
        if first_scratch_step is None or first_scratch_phase is None
        else (first_scratch_step, phase_order[first_scratch_phase])
    )
    relock_event = (
        None
        if relock_step is None or relock_phase is None
        else (relock_step, phase_order[relock_phase])
    )
    scratch_before_relock = (
        scratch_event is not None
        and (relock_event is None or scratch_event < relock_event)
    )
    boundaries_until_relock = (
        None
        if relock_step is None
        else 0
        if immediate_active
        else int(np.count_nonzero(boundaries[start : relock_step + 1]))
    )
    return {
        "selected_exact_generation_relock_observed": relock_step is not None,
        "selected_first_exact_generation_relock_step": relock_step,
        "selected_first_exact_generation_relock_segment_step": (
            None if relock_step is None else relock_step - start
        ),
        "selected_exact_generation_relock_phase": relock_phase,
        "selected_observed_learner_boundaries_until_relock": boundaries_until_relock,
        "selected_first_scratch_entry_step": first_scratch_step,
        "selected_first_scratch_entry_segment_step": (
            None if first_scratch_step is None else first_scratch_step - start
        ),
        "selected_first_scratch_entry_phase": first_scratch_phase,
        "selected_scratch_entered_before_relock": scratch_before_relock,
        "selected_scratch_entered_before_relock_or_segment_end": (
            first_scratch_step is not None
        ),
        "selected_durable_retrieval_before_scratch": (
            relock_event is not None
            and (scratch_event is None or relock_event < scratch_event)
        ),
    }


def hidden_regime_lineage_recurrence_segments(
    world: HiddenRegimeWorldConfig,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(segment, regime, coalesced occurrence)`` recall entries.

    Adjacent equal-regime segments are one uninterrupted exposure episode and
    therefore neither increment the occurrence nor create a retention event.
    """

    if not isinstance(world, HiddenRegimeWorldConfig):
        raise TypeError("world must be a HiddenRegimeWorldConfig")
    episode_counts: dict[int, int] = {}
    recurrences: list[tuple[int, int, int]] = []
    for segment_index, regime in enumerate(world.segment_regimes):
        if segment_index > 0 and world.segment_regimes[segment_index - 1] == regime:
            continue
        occurrence = episode_counts.get(regime, 0)
        if occurrence > 0:
            recurrences.append((segment_index, regime, occurrence))
        episode_counts[regime] = occurrence + 1
    return tuple(recurrences)


def hidden_regime_coalesced_episode_bounds(
    world: HiddenRegimeWorldConfig,
    segment_index: int,
) -> tuple[int, int, int, int]:
    """Return segment/step bounds for one contiguous equal-regime episode.

    The tuple is ``(first_segment, end_segment_exclusive, start_step,
    episode_length)`` and is derived only from the evaluator world config.
    """

    if not isinstance(world, HiddenRegimeWorldConfig):
        raise TypeError("world must be a HiddenRegimeWorldConfig")
    if (
        type(segment_index) is not int
        or segment_index < 0
        or segment_index >= len(world.segment_regimes)
    ):
        raise ValueError("segment_index must identify one configured segment")
    regime = world.segment_regimes[segment_index]
    first = segment_index
    while first > 0 and world.segment_regimes[first - 1] == regime:
        first -= 1
    end = segment_index + 1
    while end < len(world.segment_regimes) and world.segment_regimes[end] == regime:
        end += 1
    start_step = sum(world.segment_lengths[:first])
    episode_length = sum(world.segment_lengths[first:end])
    return first, end, start_step, episode_length


def _recurrence_retention_summaries(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[tuple[RecurrenceRetentionRecord, ...], RetentionAggregateSummary]:
    """Add commit-generation lineage retention to secondary legacy probes."""

    base_records, base_aggregate = _best_dormant_retention_summaries(trace, config)
    recurrence_episode_index = {
        segment_index: occurrence
        for segment_index, _, occurrence in hidden_regime_lineage_recurrence_segments(
            config.world
        )
    }
    base_records = tuple(
        record
        for record in base_records
        if record.segment_index in recurrence_episode_index
    )
    lineages = reconstruct_commit_generation_lineages(trace, config)
    rewards = np.asarray(trace.reward, dtype=np.float32)
    lease_length = config.learner.lease_length
    records: list[RecurrenceRetentionRecord] = []
    for base in base_records:
        _, end_segment, start, episode_length = hidden_regime_coalesced_episode_bounds(
            config.world,
            base.segment_index,
        )
        end = start + episode_length
        if end_segment <= base.segment_index:
            raise ValueError("coalesced recurrence episode bounds are invalid")
        current_window = _entry_window(
            rewards,
            start,
            episode_length,
            lease_length,
        )
        prior = tuple(
            lineage
            for lineage in lineages
            if lineage.regime_id == base.regime_id and lineage.commit_step < start
        )
        probes = tuple(
            _lineage_probe_at_entry(lineage, start=start, trace=trace)
            for lineage in prior
        )
        qualified = tuple(probe for probe in probes if probe.acquisition_qualified)
        surviving = tuple(
            probe for probe in qualified if probe.synchronized_generation_survives
        )
        latest = None if not qualified else qualified[-1]
        selected = None if not surviving else surviving[-1]
        latest_acquisition_segment: int | None = None
        latest_acquisition_episode_length: int | None = None
        latest_acquisition_complete: bool | None = None
        latest_acquisition_reward: float | None = None
        latest_acquisition_errors: int | None = None
        latest_acquisition_error_rate: float | None = None
        latest_comparison_available = False
        latest_error_delta: float | None = None
        latest_error_ratio: float | None = None
        latest_ratio_defined: bool | None = None
        if latest is not None:
            (
                latest_acquisition_segment,
                _,
                latest_acquisition_start,
                latest_acquisition_episode_length,
            ) = hidden_regime_coalesced_episode_bounds(
                config.world,
                latest.commit_segment_index,
            )
            latest_acquisition_window = _entry_window(
                rewards,
                latest_acquisition_start,
                latest_acquisition_episode_length,
                lease_length,
            )
            (
                latest_acquisition_complete,
                latest_acquisition_reward,
                latest_acquisition_errors,
                latest_acquisition_error_rate,
            ) = latest_acquisition_window
            latest_comparison_available = current_window[0] and latest_acquisition_complete
            if latest_comparison_available:
                current_error_rate = cast(float, current_window[3])
                acquisition_error_rate = cast(float, latest_acquisition_error_rate)
                latest_error_delta = current_error_rate - acquisition_error_rate
                latest_ratio_defined = acquisition_error_rate > 0.0
                if latest_ratio_defined:
                    latest_error_ratio = current_error_rate / acquisition_error_rate
        relock = _selected_lineage_relock(selected, start=start, end=end, trace=trace)
        records.append(
            dataclasses.replace(
                base,
                occurrence_index=recurrence_episode_index[base.segment_index],
                raw_segment_occurrence_index=base.occurrence_index,
                first_world_window_complete=current_window[0],
                first_world_window_reward=current_window[1],
                first_world_window_errors=current_window[2],
                first_world_window_error_rate=current_window[3],
                latest_qualified_acquisition_segment_index=(
                    latest_acquisition_segment
                ),
                latest_qualified_acquisition_episode_length=(
                    latest_acquisition_episode_length
                ),
                latest_qualified_acquisition_world_window_complete=(
                    latest_acquisition_complete
                ),
                latest_qualified_acquisition_world_window_reward=(
                    latest_acquisition_reward
                ),
                latest_qualified_acquisition_world_window_errors=(
                    latest_acquisition_errors
                ),
                latest_qualified_acquisition_world_window_error_rate=(
                    latest_acquisition_error_rate
                ),
                latest_qualified_acquisition_comparison_available=(
                    latest_comparison_available
                ),
                recurrence_minus_latest_qualified_acquisition_error_rate_delta=(
                    latest_error_delta
                ),
                recurrence_to_latest_qualified_acquisition_error_rate_ratio=(
                    latest_error_ratio
                ),
                recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined=(
                    latest_ratio_defined
                ),
                prior_same_regime_lineages=probes,
                prior_same_regime_lineage_count=len(probes),
                prior_qualified_lineage_count=len(qualified),
                prior_unqualified_lineage_count=len(probes) - len(qualified),
                lineage_retention_applicable=bool(qualified),
                acquisition_coverage_failure=not qualified,
                latest_prior_qualified_lineage_index=(
                    None if latest is None else latest.lineage_index
                ),
                latest_prior_qualified_commit_step=(
                    None if latest is None else latest.commit_step
                ),
                latest_prior_qualified_survived=(
                    None
                    if latest is None
                    else latest.synchronized_generation_survives
                ),
                any_prior_qualified_survived=(
                    None if not qualified else bool(surviving)
                ),
                surviving_qualified_lineage_count=len(surviving),
                selected_lineage_available=selected is not None,
                selected_lineage_index=(
                    None if selected is None else selected.lineage_index
                ),
                selected_lineage_commit_step=(
                    None if selected is None else selected.commit_step
                ),
                selected_lineage_slot=None if selected is None else selected.slot,
                selected_lineage_generation=(
                    None if selected is None else selected.generation
                ),
                selected_lineage_entry_activity_status=(
                    None if selected is None else selected.entry_activity_status
                ),
                selected_lineage_entry_composed_greedy_mapping=(
                    None if selected is None else selected.entry_composed_greedy_mapping
                ),
                selected_lineage_entry_composed_greedy_accuracy=(
                    None if selected is None else selected.entry_composed_greedy_accuracy
                ),
                selected_lineage_entry_minus_commit_accuracy=(
                    None if selected is None else selected.entry_minus_commit_accuracy
                ),
                selected_lineage_helper_bit_exact_preserved=(
                    None if selected is None else selected.helper_bit_exact_preserved
                ),
                selected_lineage_beneficiary_bit_exact_preserved=(
                    None if selected is None else selected.beneficiary_bit_exact_preserved
                ),
                selected_lineage_joint_bit_exact_preserved=(
                    None if selected is None else selected.joint_bit_exact_preserved
                ),
                selected_lineage_zero_helper_accuracy=(
                    None if selected is None else selected.zero_helper_accuracy
                ),
                selected_lineage_zero_beneficiary_accuracy=(
                    None if selected is None else selected.zero_beneficiary_accuracy
                ),
                selected_lineage_role_swapped_accuracy=(
                    None if selected is None else selected.role_swapped_accuracy
                ),
                **relock,
            )
        )

    recurrence_count = len(records)
    applicable = [record for record in records if record.lineage_retention_applicable]
    latest_acquisition_comparisons = [
        record
        for record in applicable
        if record.latest_qualified_acquisition_comparison_available
    ]
    latest_acquisition_ratio_defined = [
        record
        for record in latest_acquisition_comparisons
        if record.recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined
        is True
    ]
    selected_records = [record for record in records if record.selected_lineage_available]
    selected_relocked = [
        record
        for record in selected_records
        if record.selected_exact_generation_relock_observed is True
    ]
    complete_windows = [
        cast(float, record.first_world_window_reward)
        for record in records
        if record.first_world_window_complete
    ]
    complete_error_rates = [
        cast(float, record.first_world_window_error_rate)
        for record in records
        if record.first_world_window_complete
    ]
    dormant_records = [record for record in records if record.dormant_probe_available]
    dormant_relocked = [
        record for record in records if record.exact_generation_relock_observed
    ]
    dormant_retrieved = [
        record for record in records if record.durable_retrieval_before_scratch
    ]
    comparisons = [
        record
        for record in records
        if record.legacy_recurrence_minus_first_exposure_error_rate_delta is not None
    ]
    ratio_defined_records = [
        record
        for record in comparisons
        if record.legacy_recurrence_to_first_exposure_error_rate_ratio_defined
    ]

    def fraction(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else float(numerator / denominator)

    prior_count = sum(record.prior_same_regime_lineage_count for record in records)
    qualified_count = sum(record.prior_qualified_lineage_count for record in records)
    surviving_count = sum(
        record.surviving_qualified_lineage_count for record in records
    )
    latest_survival_count = sum(
        record.latest_prior_qualified_survived is True for record in applicable
    )
    any_survival_count = sum(
        record.any_prior_qualified_survived is True for record in applicable
    )
    selected_count = len(selected_records)
    selected_helper_exact = sum(
        record.selected_lineage_helper_bit_exact_preserved is True
        for record in selected_records
    )
    selected_beneficiary_exact = sum(
        record.selected_lineage_beneficiary_bit_exact_preserved is True
        for record in selected_records
    )
    selected_joint_exact = sum(
        record.selected_lineage_joint_bit_exact_preserved is True
        for record in selected_records
    )
    selected_retrieved = sum(
        record.selected_durable_retrieval_before_scratch is True
        for record in selected_records
    )
    aggregate = dataclasses.replace(
        base_aggregate,
        recurrence_count=recurrence_count,
        complete_first_world_window_count=len(complete_windows),
        missing_first_world_window_count=recurrence_count - len(complete_windows),
        first_world_window_reward_mean=_optional_mean(complete_windows),
        first_world_window_error_rate_mean=_optional_mean(complete_error_rates),
        lineage_retention_applicable_count=len(applicable),
        acquisition_coverage_failure_count=recurrence_count - len(applicable),
        qualification_coverage_denominator=recurrence_count,
        qualification_coverage_fraction=fraction(len(applicable), recurrence_count),
        latest_qualified_acquisition_comparison_available_count=len(
            latest_acquisition_comparisons
        ),
        latest_qualified_acquisition_comparison_denominator=len(applicable),
        latest_qualified_acquisition_comparison_missing_count=(
            len(applicable) - len(latest_acquisition_comparisons)
        ),
        latest_qualified_acquisition_comparison_not_applicable_count=(
            recurrence_count - len(applicable)
        ),
        recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean=(
            _optional_mean(
                [
                    cast(
                        float,
                        record.recurrence_minus_latest_qualified_acquisition_error_rate_delta,
                    )
                    for record in latest_acquisition_comparisons
                ]
            )
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_mean=(
            _optional_mean(
                [
                    cast(
                        float,
                        record.recurrence_to_latest_qualified_acquisition_error_rate_ratio,
                    )
                    for record in latest_acquisition_ratio_defined
                ]
            )
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined_count=(
            len(latest_acquisition_ratio_defined)
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_undefined_count=(
            len(latest_acquisition_comparisons)
            - len(latest_acquisition_ratio_defined)
        ),
        prior_same_regime_lineage_count=prior_count,
        prior_qualified_lineage_count=qualified_count,
        prior_unqualified_lineage_count=prior_count - qualified_count,
        surviving_qualified_lineage_count=surviving_count,
        qualified_lineage_survival_denominator=qualified_count,
        qualified_lineage_survival_fraction=fraction(
            surviving_count,
            qualified_count,
        ),
        latest_qualified_version_survival_count=latest_survival_count,
        latest_qualified_version_survival_denominator=len(applicable),
        latest_qualified_version_survival_missing_count=(
            recurrence_count - len(applicable)
        ),
        latest_qualified_version_survival_fraction=fraction(
            latest_survival_count,
            len(applicable),
        ),
        any_qualified_knowledge_survival_count=any_survival_count,
        any_qualified_knowledge_survival_denominator=len(applicable),
        any_qualified_knowledge_survival_missing_count=(
            recurrence_count - len(applicable)
        ),
        any_qualified_knowledge_survival_fraction=fraction(
            any_survival_count,
            len(applicable),
        ),
        selected_lineage_probe_available_count=selected_count,
        selected_lineage_probe_denominator=len(applicable),
        selected_lineage_survival_failure_count=len(applicable) - selected_count,
        selected_lineage_not_applicable_count=recurrence_count - len(applicable),
        selected_lineage_survival_fraction_given_qualified_prior=fraction(
            selected_count,
            len(applicable),
        ),
        selected_entry_metric_denominator=selected_count,
        selected_entry_active_count=sum(
            record.selected_lineage_entry_activity_status == "active"
            for record in selected_records
        ),
        selected_entry_dormant_count=sum(
            record.selected_lineage_entry_activity_status == "dormant"
            for record in selected_records
        ),
        selected_entry_mixed_count=sum(
            record.selected_lineage_entry_activity_status == "mixed"
            for record in selected_records
        ),
        selected_entry_composed_greedy_accuracy_mean=_optional_mean(
            [
                cast(float, record.selected_lineage_entry_composed_greedy_accuracy)
                for record in selected_records
            ]
        ),
        selected_entry_minus_commit_accuracy_mean=_optional_mean(
            [
                cast(float, record.selected_lineage_entry_minus_commit_accuracy)
                for record in selected_records
            ]
        ),
        selected_helper_bit_exact_preservation_count=selected_helper_exact,
        selected_beneficiary_bit_exact_preservation_count=selected_beneficiary_exact,
        selected_joint_bit_exact_preservation_count=selected_joint_exact,
        selected_bit_exact_preservation_conditional_denominator=selected_count,
        selected_bit_exact_preservation_all_qualified_denominator=len(applicable),
        selected_helper_bit_exact_preservation_fraction=fraction(
            selected_helper_exact,
            selected_count,
        ),
        selected_beneficiary_bit_exact_preservation_fraction=fraction(
            selected_beneficiary_exact,
            selected_count,
        ),
        selected_joint_bit_exact_preservation_fraction=fraction(
            selected_joint_exact,
            selected_count,
        ),
        selected_helper_bit_exact_preservation_fraction_all_qualified=fraction(
            selected_helper_exact,
            len(applicable),
        ),
        selected_beneficiary_bit_exact_preservation_fraction_all_qualified=fraction(
            selected_beneficiary_exact,
            len(applicable),
        ),
        selected_joint_bit_exact_preservation_fraction_all_qualified=fraction(
            selected_joint_exact,
            len(applicable),
        ),
        selected_zero_helper_accuracy_mean=_optional_mean(
            [
                cast(float, record.selected_lineage_zero_helper_accuracy)
                for record in selected_records
            ]
        ),
        selected_zero_beneficiary_accuracy_mean=_optional_mean(
            [
                cast(float, record.selected_lineage_zero_beneficiary_accuracy)
                for record in selected_records
            ]
        ),
        selected_role_swapped_accuracy_mean=_optional_mean(
            [
                cast(float, record.selected_lineage_role_swapped_accuracy)
                for record in selected_records
            ]
        ),
        selected_exact_generation_relock_count=len(selected_relocked),
        selected_exact_generation_relock_conditional_denominator=selected_count,
        selected_exact_generation_relock_all_qualified_denominator=len(applicable),
        selected_exact_generation_relock_fraction_given_selected_lineage=fraction(
            len(selected_relocked),
            selected_count,
        ),
        selected_exact_generation_relock_fraction_all_qualified=fraction(
            len(selected_relocked),
            len(applicable),
        ),
        selected_observed_learner_boundaries_to_relock_mean=_optional_mean(
            [
                cast(
                    int,
                    record.selected_observed_learner_boundaries_until_relock,
                )
                for record in selected_relocked
            ]
        ),
        selected_observed_learner_boundaries_to_relock_available_count=len(
            selected_relocked
        ),
        selected_observed_learner_boundaries_to_relock_unavailable_count=(
            selected_count - len(selected_relocked)
        ),
        selected_durable_retrieval_before_scratch_count=selected_retrieved,
        selected_durable_retrieval_before_scratch_conditional_denominator=selected_count,
        selected_durable_retrieval_before_scratch_all_qualified_denominator=len(
            applicable
        ),
        selected_durable_retrieval_before_scratch_fraction_given_selected_lineage=fraction(
            selected_retrieved,
            selected_count,
        ),
        selected_durable_retrieval_before_scratch_fraction_all_qualified=fraction(
            selected_retrieved,
            len(applicable),
        ),
        dormant_probe_available_count=len(dormant_records),
        dormant_probe_missing_count=recurrence_count - len(dormant_records),
        dormant_composed_greedy_accuracy_mean=_optional_mean(
            [
                cast(float, record.best_dormant_composed_greedy_accuracy)
                for record in dormant_records
            ]
        ),
        dormant_zero_helper_accuracy_mean=_optional_mean(
            [
                cast(float, record.best_dormant_zero_helper_accuracy)
                for record in dormant_records
            ]
        ),
        dormant_zero_beneficiary_accuracy_mean=_optional_mean(
            [
                cast(float, record.best_dormant_zero_beneficiary_accuracy)
                for record in dormant_records
            ]
        ),
        dormant_role_swapped_accuracy_mean=_optional_mean(
            [
                cast(float, record.best_dormant_role_swapped_accuracy)
                for record in dormant_records
            ]
        ),
        exact_generation_relock_count=len(dormant_relocked),
        exact_generation_relock_missing_count=(
            recurrence_count - len(dormant_relocked)
        ),
        exact_generation_relock_rate_all_recurrences=fraction(
            len(dormant_relocked),
            recurrence_count,
        ),
        exact_generation_relock_rate_given_dormant_probe=fraction(
            len(dormant_relocked),
            len(dormant_records),
        ),
        observed_learner_boundaries_to_relock_mean=_optional_mean(
            [
                cast(int, record.observed_learner_boundaries_until_relock)
                for record in dormant_relocked
            ]
        ),
        observed_learner_boundaries_to_relock_missing_count=(
            recurrence_count - len(dormant_relocked)
        ),
        durable_retrieval_before_scratch_count=len(dormant_retrieved),
        durable_retrieval_before_scratch_fraction_all_recurrences=fraction(
            len(dormant_retrieved),
            recurrence_count,
        ),
        durable_retrieval_before_scratch_fraction_given_dormant_probe=fraction(
            len(dormant_retrieved),
            len(dormant_records),
        ),
        legacy_first_exposure_comparison_available_count=len(comparisons),
        legacy_first_exposure_comparison_missing_count=(
            recurrence_count - len(comparisons)
        ),
        legacy_recurrence_minus_first_exposure_error_rate_delta_mean=_optional_mean(
            [
                cast(
                    float,
                    record.legacy_recurrence_minus_first_exposure_error_rate_delta,
                )
                for record in comparisons
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_mean=_optional_mean(
            [
                cast(
                    float,
                    record.legacy_recurrence_to_first_exposure_error_rate_ratio,
                )
                for record in ratio_defined_records
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_defined_count=(
            len(ratio_defined_records)
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_undefined_count=(
            len(comparisons) - len(ratio_defined_records)
        ),
    )
    return tuple(records), aggregate


def reconstruct_hidden_regime_retention(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[tuple[RecurrenceRetentionRecord, ...], RetentionAggregateSummary]:
    """Return pure evaluator-only recurrence probes from already-recorded arrays.

    This function does not replay either learner and does not certify that the
    supplied arrays are a legal lifecycle.  Use the strict run validator for
    that same-implementation check; a separate host state-machine audit is the
    intended independent legality check.
    """

    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        raise TypeError("trace must be a HiddenRegimePrimitiveTrace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        raise TypeError("config must be a HiddenRegimeDevelopmentConfig")
    return _recurrence_retention_summaries(trace, config)


def _replacement_provenance(
    trace: HiddenRegimePrimitiveTrace,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    committed_slots = np.asarray(trace.helper_committed_slot, dtype=np.int32)
    committed_generations = np.asarray(trace.helper_committed_generation, dtype=np.int32)
    retired_generations = np.asarray(trace.helper_retired_generation, dtype=np.int32)
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    generation_regime: dict[int, int] = {}
    targets: list[int] = []
    generation_pairs: list[tuple[int, int]] = []
    for step in np.flatnonzero(committed_slots >= 0):
        new_generation = int(committed_generations[step])
        retired_generation = int(retired_generations[step])
        regime = int(regimes[step])
        if (
            retired_generation >= 0
            and generation_regime.get(retired_generation) == 2
            and regime == 3
        ):
            targets.append(int(committed_slots[step]))
            generation_pairs.append((retired_generation, new_generation))
        generation_regime[new_generation] = regime
    return tuple(targets), tuple(generation_pairs)


def _d_short_non_displacement(
    trace: HiddenRegimePrimitiveTrace,
) -> tuple[bool, bool]:
    segments = np.asarray(trace.segment_index, dtype=np.int32)
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    d_segments = sorted(set(int(x) for x in segments[regimes == 4]))
    if not d_segments:
        return False, False
    for segment in d_segments:
        indices = np.flatnonzero(segments == segment)
        first = int(indices[0])
        last = int(indices[-1])
        for role in ("helper", "beneficiary"):
            status_pre = np.asarray(getattr(trace, f"{role}_status_pre"))[first]
            status_post = np.asarray(getattr(trace, f"{role}_status_post"))[last]
            generation_pre = np.asarray(getattr(trace, f"{role}_generation_pre"))[first]
            generation_post = np.asarray(getattr(trace, f"{role}_generation_post"))[last]
            if not np.array_equal(status_pre, status_post):
                return True, False
            if not np.array_equal(generation_pre, generation_post):
                return True, False
    return True, True


def reconstruct_hidden_regime_summary(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
    condition: HiddenRegimeDevelopmentCondition,
) -> HiddenRegimeRunSummary:
    """Recompute all descriptive metrics from fixed-shape primitives."""

    segments, recurrence = _segment_summaries(trace, config)
    commit_lineages = reconstruct_commit_generation_lineages(trace, config)
    recurrence_retention, retention = reconstruct_hidden_regime_retention(trace, config)
    helper_writes = int(np.count_nonzero(np.asarray(trace.helper_value_write)))
    beneficiary_writes = int(np.count_nonzero(np.asarray(trace.beneficiary_value_write)))
    helper_effective_updates = int(
        np.count_nonzero(
            np.logical_and(
                np.asarray(trace.helper_value_write),
                _float32_bits(trace.helper_candidate_value)
                != _float32_bits(trace.helper_value_pre),
            )
        )
    )
    beneficiary_effective_updates = int(
        np.count_nonzero(
            np.logical_and(
                np.asarray(trace.beneficiary_value_write),
                _float32_bits(trace.beneficiary_candidate_value)
                != _float32_bits(trace.beneficiary_value_pre),
            )
        )
    )
    helper_commits = int(np.count_nonzero(np.asarray(trace.helper_committed_slot) >= 0))
    beneficiary_commits = int(np.count_nonzero(np.asarray(trace.beneficiary_committed_slot) >= 0))
    helper_replacements = int(np.count_nonzero(np.asarray(trace.helper_retired_slot) >= 0))
    beneficiary_replacements = int(
        np.count_nonzero(np.asarray(trace.beneficiary_retired_slot) >= 0)
    )
    confirmation_events = int(
        np.count_nonzero(
            np.logical_and(
                np.asarray(trace.helper_candidate_lease_success),
                np.asarray(trace.helper_candidate_confirmations_post) == 0,
            )
        )
    )
    targets, generations = _replacement_provenance(trace)
    d_checked, d_non_displacement = _d_short_non_displacement(trace)
    spec = condition_spec(condition)
    helper_violations = int(np.count_nonzero(np.asarray(trace.helper_selective_mutation_violation)))
    beneficiary_violations = int(
        np.count_nonzero(np.asarray(trace.beneficiary_selective_mutation_violation))
    )
    immutability_applicable = not spec.durable_writes_enabled
    recurrent_segments = [
        segment
        for segment in segments
        if segment.is_recurrence and segment.regime_id in (0, 1, 2, 3)
    ]
    recurrence_entry_mean = (
        None
        if not recurrent_segments
        else float(np.mean([segment.early_reward for segment in recurrent_segments]))
    )
    recurrence_recovery_mean = (
        None
        if not recurrent_segments
        else float(np.mean([segment.within_segment_recovery for segment in recurrent_segments]))
    )
    return HiddenRegimeRunSummary(
        num_steps=int(np.asarray(trace.reward).size),
        mean_prequential_reward=float(
            np.mean(np.asarray(trace.reward, dtype=np.float32), dtype=np.float64)
        ),
        legacy_metric_window=config.metric_window,
        legacy_recurrence_entry_reward_mean=recurrence_entry_mean,
        legacy_recurrence_recovery_mean=recurrence_recovery_mean,
        recurrence_entry_reward_mean=recurrence_entry_mean,
        recurrence_recovery_mean=recurrence_recovery_mean,
        segment_rewards=segments,
        recurrence_by_regime=recurrence,
        commit_generation_lineages=commit_lineages,
        synchronized_commit_lineage_count=len(commit_lineages),
        acquisition_qualified_commit_lineage_count=sum(
            lineage.acquisition_qualified for lineage in commit_lineages
        ),
        acquisition_unqualified_commit_lineage_count=sum(
            not lineage.acquisition_qualified for lineage in commit_lineages
        ),
        recurrence_retention=recurrence_retention,
        retention=retention,
        helper_value_write_count=helper_writes,
        beneficiary_value_write_count=beneficiary_writes,
        helper_effective_learning_update_count=helper_effective_updates,
        beneficiary_effective_learning_update_count=beneficiary_effective_updates,
        both_roles_learned=helper_effective_updates > 0 and beneficiary_effective_updates > 0,
        helper_commit_count=helper_commits,
        beneficiary_commit_count=beneficiary_commits,
        helper_replacement_count=helper_replacements,
        beneficiary_replacement_count=beneficiary_replacements,
        candidate_confirmation_events=confirmation_events,
        c_old_to_c_new_replacement_count=len(targets),
        c_old_to_c_new_target_slots=targets,
        c_old_to_c_new_generation_pairs=generations,
        c_old_to_c_new_exactly_one_target=len(targets) == 1,
        d_short_checked=d_checked,
        d_short_non_displacement=d_non_displacement,
        selective_immutability_applicable=immutability_applicable,
        helper_selective_mutation_violations=helper_violations,
        beneficiary_selective_mutation_violations=beneficiary_violations,
        selective_durable_bit_immutable_until_atomic_replacement=(
            immutability_applicable and helper_violations == 0 and beneficiary_violations == 0
        ),
        lifecycle_synchronized_every_step=bool(np.all(np.asarray(trace.lifecycle_synchronized))),
    )


def _resource_report(
    initial_state: SlotSignalingState,
    final_state: SlotSignalingState,
) -> HiddenRegimeResourceReport:
    initial = slot_signaling_resource_budget(initial_state)
    final = slot_signaling_resource_budget(final_state)
    constant = (
        initial.state_scalars == final.state_scalars
        and initial.state_bytes == final.state_bytes
        and initial.state_scalars == 138
        and initial.state_bytes == EXPECTED_DYAD_STATE_BYTES
    )
    matched = all(
        role.state_scalars == 69 and role.state_bytes == 276
        for role in (
            initial.helper,
            initial.beneficiary,
            final.helper,
            final.beneficiary,
        )
    )
    return HiddenRegimeResourceReport(
        initial_state_scalars=initial.state_scalars,
        final_state_scalars=final.state_scalars,
        initial_state_bytes=initial.state_bytes,
        final_state_bytes=final.state_bytes,
        expected_state_bytes=EXPECTED_DYAD_STATE_BYTES,
        resource_constant=constant,
        resource_matched=matched,
    )


def run_hidden_regime_condition(
    condition: HiddenRegimeDevelopmentCondition,
    *,
    seed_pair: HiddenRegimeSeedPair,
    config: HiddenRegimeDevelopmentConfig | None = None,
    execution_authorization: object | None = None,
) -> HiddenRegimeRunResult:
    """Run one explicitly seeded, fixed-resource condition in one JIT scan."""

    if not isinstance(seed_pair, HiddenRegimeSeedPair):
        raise TypeError("seed_pair must be a HiddenRegimeSeedPair")
    if seed_pair.namespace == RESERVED_DEVELOPMENT_SEED_NAMESPACE:
        raise ValueError("the reserved development namespace is intentionally unexecuted")
    config = config or HiddenRegimeDevelopmentConfig()
    # Imported lazily so the ordinary evaluator remains free of a module-init
    # cycle with the governance issuer's exact frozen-config reconstruction.
    from alberta_framework.evaluation.hidden_regime_execution_governance import (
        begin_managed_hidden_regime_execution,
        complete_managed_hidden_regime_execution,
    )

    execution_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=execution_authorization,
    )
    spec = condition_spec(condition)
    world = HiddenRegimeSignalingWorld(config.world)
    learner = SlotSignalingAgent(
        dataclasses.replace(
            config.learner,
            writable_lru_ablation=False,
            durable_write_policy=spec.durable_write_policy,
            replacement_target_policy=spec.replacement_target_policy,
        )
    )
    world_state = world.init(hidden_regime_world_keys(jr.key(seed_pair.world_seed)))
    learner_state = learner.init(slot_signaling_keys(jr.key(seed_pair.learner_seed)))
    (_, final_state), outputs = _scan_runner(
        config,
        spec.durable_write_policy,
        spec.replacement_target_policy,
        config.num_steps,
    )(
        world_state,
        learner_state,
        jnp.asarray(_channel_code(spec.channel), dtype=jnp.int32),
        jnp.asarray(spec.helper_write),
        jnp.asarray(spec.beneficiary_write),
    )
    trace = _make_trace(outputs)
    summary = reconstruct_hidden_regime_summary(trace, config, condition)
    resource = _resource_report(learner_state, final_state)
    result = HiddenRegimeRunResult(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        trace=trace,
        summary=summary,
        resource=resource,
        final_state=final_state,
    )
    complete_managed_hidden_regime_execution(execution_ticket, result)
    return result


def run_hidden_regime_development(
    *,
    seed_pair: HiddenRegimeSeedPair,
    config: HiddenRegimeDevelopmentConfig | None = None,
    conditions: Sequence[HiddenRegimeDevelopmentCondition] = MATCHED_CONDITIONS,
) -> HiddenRegimeDevelopmentReport:
    """Run explicitly requested matched controls without promotion or persistence."""

    config = config or HiddenRegimeDevelopmentConfig()
    condition_order = _validated_condition_order(conditions)
    runs = tuple(
        run_hidden_regime_condition(condition, seed_pair=seed_pair, config=config)
        for condition in condition_order
    )
    selective = next(run for run in runs if run.condition == SELECTIVE_FULL)
    controls = tuple(
        PairedControlMetric(
            condition=run.condition,
            mean_prequential_reward=run.summary.mean_prequential_reward,
            delta_vs_selective_full=(
                run.summary.mean_prequential_reward - selective.summary.mean_prequential_reward
            ),
            recurrence_entry_reward_mean=run.summary.recurrence_entry_reward_mean,
            recurrence_entry_delta_vs_selective_full=(
                None
                if run.summary.recurrence_entry_reward_mean is None
                or selective.summary.recurrence_entry_reward_mean is None
                else run.summary.recurrence_entry_reward_mean
                - selective.summary.recurrence_entry_reward_mean
            ),
            helper_value_write_count=run.summary.helper_value_write_count,
            beneficiary_value_write_count=run.summary.beneficiary_value_write_count,
            resource_bytes=run.resource.final_state_bytes,
        )
        for run in runs
        if run.condition != SELECTIVE_FULL
    )
    return HiddenRegimeDevelopmentReport(seed_pair, config, runs, controls)


def _expected_trace_shapes(num_steps: int) -> dict[str, tuple[int, ...]]:
    vector_fields = {
        "helper_relevance_mean_pre",
        "helper_relevance_mean_post",
        "beneficiary_relevance_mean_pre",
        "beneficiary_relevance_mean_post",
        "helper_relevance_mass_pre",
        "helper_relevance_mass_post",
        "beneficiary_relevance_mass_pre",
        "beneficiary_relevance_mass_post",
        "helper_failed_leases_pre",
        "helper_failed_leases_post",
        "beneficiary_failed_leases_pre",
        "beneficiary_failed_leases_post",
        "helper_idle_leases_pre",
        "helper_idle_leases_post",
        "beneficiary_idle_leases_pre",
        "beneficiary_idle_leases_post",
        "helper_status_pre",
        "helper_status_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
        "helper_selective_mutation_violation",
        "beneficiary_selective_mutation_violation",
    }
    bank_fields = {
        "helper_value_bits_pre",
        "helper_value_bits_post",
        "beneficiary_value_bits_pre",
        "beneficiary_value_bits_post",
    }
    key_fields = {
        "world_cue_key_data_pre",
        "world_cue_key_data_post",
        "world_channel_key_data_pre",
        "world_channel_key_data_post",
        "helper_policy_key_data_pre",
        "helper_policy_key_data_post",
        "beneficiary_policy_key_data_pre",
        "beneficiary_policy_key_data_post",
    }
    return {
        field.name: (
            (num_steps, N_SLOTS, 3, 3)
            if field.name in bank_fields
            else (num_steps, 2)
            if field.name in key_fields
            else (num_steps, N_SLOTS)
            if field.name in vector_fields
            else (num_steps,)
        )
        for field in dataclasses.fields(HiddenRegimePrimitiveTrace)
    }


def _expected_trace_dtypes() -> dict[str, np.dtype[Any]]:
    int32_fields = {
        "step_index",
        "segment_index",
        "segment_step",
        "world_step_count_pre",
        "world_step_count_post",
        "world_schedule_position_pre",
        "world_schedule_position_post",
        "helper_decision_slot",
        "helper_private_input",
        "helper_decision_action",
        "beneficiary_decision_slot",
        "beneficiary_private_input",
        "beneficiary_decision_action",
        "helper_failed_leases_pre",
        "helper_failed_leases_post",
        "beneficiary_failed_leases_pre",
        "beneficiary_failed_leases_post",
        "helper_idle_leases_pre",
        "helper_idle_leases_post",
        "beneficiary_idle_leases_pre",
        "beneficiary_idle_leases_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
        "helper_candidate_confirmations_pre",
        "helper_candidate_confirmations_post",
        "beneficiary_candidate_confirmations_pre",
        "beneficiary_candidate_confirmations_post",
        "helper_scratch_failed_leases_pre",
        "helper_scratch_failed_leases_post",
        "beneficiary_scratch_failed_leases_pre",
        "beneficiary_scratch_failed_leases_post",
        "helper_lease_offset_pre",
        "helper_lease_offset_post",
        "beneficiary_lease_offset_pre",
        "beneficiary_lease_offset_post",
        "helper_remaining_durable_tests_pre",
        "helper_remaining_durable_tests_post",
        "beneficiary_remaining_durable_tests_pre",
        "beneficiary_remaining_durable_tests_post",
        "helper_search_cursor_pre",
        "helper_search_cursor_post",
        "beneficiary_search_cursor_pre",
        "beneficiary_search_cursor_post",
        "helper_next_generation_pre",
        "helper_next_generation_post",
        "beneficiary_next_generation_pre",
        "beneficiary_next_generation_post",
        "helper_committed_slot",
        "helper_committed_generation",
        "helper_retired_slot",
        "helper_retired_generation",
        "beneficiary_committed_slot",
        "beneficiary_committed_generation",
        "beneficiary_retired_slot",
        "beneficiary_retired_generation",
    }
    int8_fields = {
        "regime_id",
        "world_cue_pre",
        "world_cue_post",
        "helper_cue",
        "oracle_target",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "helper_active_slot_pre",
        "helper_active_slot_post",
        "beneficiary_active_slot_pre",
        "beneficiary_active_slot_post",
        "helper_status_pre",
        "helper_status_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
    }
    bool_fields = {
        "world_terminated",
        "helper_write_enabled",
        "beneficiary_write_enabled",
        "helper_value_write",
        "beneficiary_value_write",
        "helper_lease_boundary",
        "beneficiary_lease_boundary",
        "helper_candidate_lease_success",
        "beneficiary_candidate_lease_success",
        "helper_relevance_ready",
        "beneficiary_relevance_ready",
        "helper_durable_relevant",
        "beneficiary_durable_relevant",
        "helper_candidate_relevant",
        "beneficiary_candidate_relevant",
        "helper_generation_exhausted",
        "beneficiary_generation_exhausted",
        "helper_scratch_retest_started",
        "beneficiary_scratch_retest_started",
        "lifecycle_synchronized",
        "helper_selective_mutation_violation",
        "beneficiary_selective_mutation_violation",
    }
    uint_fields = {
        "helper_value_bits_pre",
        "helper_value_bits_post",
        "beneficiary_value_bits_pre",
        "beneficiary_value_bits_post",
        "world_cue_key_data_pre",
        "world_cue_key_data_post",
        "world_channel_key_data_pre",
        "world_channel_key_data_post",
        "helper_policy_key_data_pre",
        "helper_policy_key_data_post",
        "beneficiary_policy_key_data_pre",
        "beneficiary_policy_key_data_post",
    }
    return {
        field.name: (
            np.dtype(np.int32)
            if field.name in int32_fields
            else np.dtype(np.int8)
            if field.name in int8_fields
            else np.dtype(np.bool_)
            if field.name in bool_fields
            else np.dtype(np.uint32)
            if field.name in uint_fields
            else np.dtype(np.float32)
        )
        for field in dataclasses.fields(HiddenRegimePrimitiveTrace)
    }


def _array_equal(left: Array, right: Array) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _float32_bits(values: object) -> np.ndarray[Any, np.dtype[np.uint32]]:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _portable_candidate_bits(
    pre: np.ndarray[Any, np.dtype[np.float32]],
    reward: np.ndarray[Any, np.dtype[np.float32]],
    learning_rate: float,
) -> tuple[np.ndarray[Any, np.dtype[np.uint32]], np.ndarray[Any, np.dtype[np.uint32]]]:
    """Return unfused and fused float32 contraction candidates."""

    rate = np.float32(learning_rate)
    delta = np.asarray(reward - pre, dtype=np.float32)
    unfused = np.asarray(pre + np.asarray(rate * delta, dtype=np.float32), dtype=np.float32)
    fused = np.asarray(
        pre.astype(np.float64)
        + np.float64(rate) * (reward.astype(np.float64) - pre.astype(np.float64)),
        dtype=np.float32,
    )
    return unfused.view(np.uint32), fused.view(np.uint32)


@functools.lru_cache(maxsize=16)
def _shuffled_channel_replay(num_steps: int) -> Any:
    """Compile exact channel-only replay without a long Python dispatch loop."""

    @jax.jit
    def replay(initial_key: Array) -> Array:
        def step(key: Array, _: Array) -> tuple[Array, Array]:
            draw_key, next_key = jr.split(key)
            delivered = jr.randint(draw_key, (), 0, 3, dtype=jnp.int32)
            return next_key, delivered

        _, delivered = jax.lax.scan(
            step,
            initial_key,
            jnp.arange(num_steps, dtype=jnp.int32),
        )
        return delivered

    return replay


def _deterministic_replay(
    run: HiddenRegimeRunResult,
    spec: HiddenRegimeConditionSpec,
) -> tuple[HiddenRegimePrimitiveTrace, SlotSignalingState]:
    """Replay every named RNG stream and learner transition for strict audit."""

    world = HiddenRegimeSignalingWorld(run.config.world)
    learner = SlotSignalingAgent(
        dataclasses.replace(
            run.config.learner,
            writable_lru_ablation=False,
            durable_write_policy=spec.durable_write_policy,
            replacement_target_policy=spec.replacement_target_policy,
        )
    )
    world_state = world.init(hidden_regime_world_keys(jr.key(run.seed_pair.world_seed)))
    learner_state = learner.init(slot_signaling_keys(jr.key(run.seed_pair.learner_seed)))
    (_, final_state), outputs = _scan_runner(
        run.config,
        spec.durable_write_policy,
        spec.replacement_target_policy,
        run.config.num_steps,
    )(
        world_state,
        learner_state,
        jnp.asarray(_channel_code(spec.channel), dtype=jnp.int32),
        jnp.asarray(spec.helper_write),
        jnp.asarray(spec.beneficiary_write),
    )
    return _make_trace(outputs), final_state


def validate_hidden_regime_run_result(run: HiddenRegimeRunResult) -> tuple[str, ...]:
    """Fail closed on trace, resource, intervention, and summary inconsistency."""

    errors: list[str] = []
    if not isinstance(run, HiddenRegimeRunResult):
        return ("run must be a HiddenRegimeRunResult",)
    try:
        spec = condition_spec(run.condition)
    except ValueError:
        return ("run condition is unknown",)
    if run.seed_pair.namespace == RESERVED_DEVELOPMENT_SEED_NAMESPACE:
        errors.append("run uses the intentionally unexecuted reserved namespace")
    trace = run.trace
    # Gate 1 — structure: every trace leaf must have the expected shape,
    # dtype, and finiteness before any check below indexes into it.
    expected_shapes = _expected_trace_shapes(run.config.num_steps)
    expected_dtypes = _expected_trace_dtypes()
    for field in dataclasses.fields(trace):
        value = np.asarray(getattr(trace, field.name))
        if value.shape != expected_shapes[field.name]:
            errors.append(f"trace.{field.name} shape mismatch")
        if value.dtype != expected_dtypes[field.name]:
            errors.append(f"trace.{field.name} dtype mismatch")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            errors.append(f"trace.{field.name} contains non-finite values")
    if errors:
        return tuple(errors)

    # Gate 2 — value domains: ternary symbols, slot ids, lifecycle statuses,
    # non-negative counters, and commit/retire sentinel pairs.
    ternary_fields = (
        "world_cue_pre",
        "world_cue_post",
        "helper_cue",
        "oracle_target",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "helper_private_input",
        "helper_decision_action",
        "beneficiary_private_input",
        "beneficiary_decision_action",
    )
    for name in ternary_fields:
        values = np.asarray(getattr(trace, name), dtype=np.int32)
        if np.any(np.logical_or(values < 0, values > 2)):
            errors.append(f"trace.{name} is outside the ternary domain")
    for role in ("helper", "beneficiary"):
        for suffix in ("active_slot_pre", "active_slot_post", "decision_slot"):
            values = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32)
            if np.any(np.logical_or(values < 0, values >= N_SLOTS)):
                errors.append(f"trace.{role}_{suffix} is outside the slot domain")
        for suffix in ("status_pre", "status_post"):
            values = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32)
            if np.any(np.logical_or(values < SLOT_VACANT, values > SLOT_DURABLE)):
                errors.append(f"trace.{role}_{suffix} has an unknown lifecycle status")
        for suffix in ("generation_pre", "generation_post"):
            if np.any(np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32) < 0):
                errors.append(f"trace.{role}_{suffix} contains a negative generation")
        for suffix in (
            "failed_leases_pre",
            "failed_leases_post",
            "idle_leases_pre",
            "idle_leases_post",
            "candidate_confirmations_pre",
            "candidate_confirmations_post",
            "remaining_durable_tests_pre",
            "remaining_durable_tests_post",
        ):
            if np.any(np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32) < 0):
                errors.append(f"trace.{role}_{suffix} became negative")
        for suffix in ("next_generation_pre", "next_generation_post"):
            if np.any(np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32) < 1):
                errors.append(f"trace.{role}_{suffix} is below the first generation")
        for suffix in ("lease_offset_pre", "lease_offset_post"):
            offsets = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32)
            if np.any(np.logical_or(offsets < 0, offsets >= run.config.learner.lease_length)):
                errors.append(f"trace.{role}_{suffix} is outside the learner lease")
        for suffix in ("search_cursor_pre", "search_cursor_post"):
            cursors = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.int32)
            if np.any(np.logical_or(cursors < 1, cursors > 3)):
                errors.append(f"trace.{role}_{suffix} is outside the durable-slot domain")
        for suffix in (
            "relevance_mean_pre",
            "relevance_mean_post",
            "relevance_mass_pre",
            "relevance_mass_post",
            "lease_reward_sum_pre",
            "lease_reward_sum_post",
        ):
            values = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.float32)
            if np.any(values < 0.0):
                errors.append(f"trace.{role}_{suffix} became negative")
        for suffix in ("relevance_mean_pre", "relevance_mean_post"):
            values = np.asarray(getattr(trace, f"{role}_{suffix}"), dtype=np.float32)
            if np.any(values > 1.0):
                errors.append(f"trace.{role}_{suffix} exceeds binary-reward range")
        for suffix in ("value_bits_pre", "value_bits_post"):
            decoded = _float_bank_from_bits(getattr(trace, f"{role}_{suffix}"))
            if not np.all(np.isfinite(decoded)):
                errors.append(f"trace.{role}_{suffix} decodes to non-finite values")
        for event in ("committed", "retired"):
            slots = np.asarray(getattr(trace, f"{role}_{event}_slot"), dtype=np.int32)
            generations = np.asarray(
                getattr(trace, f"{role}_{event}_generation"),
                dtype=np.int32,
            )
            slot_domain = np.logical_or(slots == -1, np.logical_and(slots >= 1, slots <= 3))
            if not np.all(slot_domain):
                errors.append(f"trace.{role}_{event}_slot has an invalid sentinel or target")
            if not np.array_equal(slots == -1, generations == -1):
                errors.append(f"trace.{role}_{event} slot/generation sentinels disagree")
            if np.any(generations < -1):
                errors.append(f"trace.{role}_{event}_generation has an invalid sentinel")
        committed = np.asarray(getattr(trace, f"{role}_committed_slot"), dtype=np.int32)
        retired = np.asarray(getattr(trace, f"{role}_retired_slot"), dtype=np.int32)
        if np.any(np.logical_and(retired >= 0, committed < 0)):
            errors.append(f"{role} retirement occurred without an atomic commit")
    if errors:
        return tuple(errors)

    # Gate 3 — world reconstruction: cursors, pre/post continuity, the segment
    # schedule, oracle targets, prequential reward, and lease boundaries must
    # all rebuild exactly from the config and the named world seed.
    steps = np.asarray(trace.step_index, dtype=np.int32)
    if not np.array_equal(steps, np.arange(run.config.num_steps, dtype=np.int32)):
        errors.append("step_index is not the uninterrupted zero-based life")
    if not np.array_equal(trace.world_step_count_pre, steps):
        errors.append("world step-count pre-state differs from transition oracle")
    expected_world_step_post = np.arange(1, run.config.num_steps + 1, dtype=np.int64)
    expected_world_step_post = np.minimum(
        expected_world_step_post,
        np.iinfo(np.int32).max,
    ).astype(np.int32)
    if not np.array_equal(trace.world_step_count_post, expected_world_step_post):
        errors.append("world step-count post-state does not advance exactly")
    expected_schedule_pre = np.arange(run.config.num_steps, dtype=np.int32)
    expected_schedule_post = np.minimum(
        np.arange(1, run.config.num_steps + 1, dtype=np.int64),
        run.config.num_steps - 1,
    ).astype(np.int32)
    if not np.array_equal(trace.world_schedule_position_pre, expected_schedule_pre):
        errors.append("world schedule pre-state is not the uninterrupted finite cursor")
    if not np.array_equal(trace.world_schedule_position_post, expected_schedule_post):
        errors.append("world schedule post-state does not advance exactly")
    for suffix in (
        "world_cue",
        "world_cue_key_data",
        "world_channel_key_data",
        "world_step_count",
        "world_schedule_position",
    ):
        pre = np.asarray(getattr(trace, f"{suffix}_pre"))
        post = np.asarray(getattr(trace, f"{suffix}_post"))
        if run.config.num_steps > 1 and not np.array_equal(pre[1:], post[:-1]):
            errors.append(f"{suffix} trace continuity failed")
    if not _array_equal(trace.world_cue_pre, trace.helper_cue):
        errors.append("helper observation differs from world cue pre-state")
    initial_world = HiddenRegimeSignalingWorld(run.config.world).init(
        hidden_regime_world_keys(jr.key(run.seed_pair.world_seed))
    )
    for name, expected in (
        ("world_cue_pre", initial_world.cue),
        ("world_step_count_pre", initial_world.step_count),
        ("world_schedule_position_pre", initial_world.schedule_position),
        ("world_cue_key_data_pre", jr.key_data(initial_world.cue_key)),
        ("world_channel_key_data_pre", jr.key_data(initial_world.channel_key)),
    ):
        if not np.array_equal(np.asarray(getattr(trace, name))[0], np.asarray(expected)):
            errors.append(f"trace.{name} does not match the named initial world state")
    expected_segments = np.repeat(
        np.arange(len(run.config.world.segment_lengths), dtype=np.int32),
        run.config.world.segment_lengths,
    )
    expected_segment_steps = np.concatenate(
        [np.arange(length, dtype=np.int32) for length in run.config.world.segment_lengths]
    )
    expected_regimes = np.repeat(
        np.asarray(run.config.world.segment_regimes, dtype=np.int32),
        run.config.world.segment_lengths,
    )
    if not np.array_equal(trace.segment_index, expected_segments):
        errors.append("segment_index does not reconstruct the configured schedule")
    if not np.array_equal(trace.segment_step, expected_segment_steps):
        errors.append("segment_step does not reconstruct configured boundaries")
    if not np.array_equal(trace.regime_id, expected_regimes):
        errors.append("regime_id does not reconstruct the configured schedule")
    cues = np.asarray(trace.helper_cue, dtype=np.int32)
    targets = np.asarray(trace.oracle_target, dtype=np.int32)
    permutations = np.asarray(run.config.world.regime_permutations, dtype=np.int32)
    expected_targets = permutations[expected_regimes, cues]
    if not np.array_equal(targets, expected_targets):
        errors.append("oracle_target does not match evaluator-only permutation")
    expected_reward = (
        np.asarray(trace.beneficiary_action, dtype=np.int32) == expected_targets
    ).astype(np.float32)
    if not np.array_equal(np.asarray(trace.reward, dtype=np.float32), expected_reward):
        errors.append("reward is not the exact prequential action-target equality")
    if np.any(np.asarray(trace.world_terminated, dtype=np.bool_)):
        errors.append("world termination trace is not the actual continuing-world false leaf")
    if not np.array_equal(
        _float32_bits(trace.world_discount),
        np.full((run.config.num_steps,), np.float32(1.0)).view(np.uint32),
    ):
        errors.append("world discount trace is not the actual continuing-world one leaf")
    expected_boundary = (
        np.arange(run.config.num_steps, dtype=np.int64) % run.config.learner.lease_length
        == run.config.learner.lease_length - 1
    )
    if not np.array_equal(trace.helper_lease_boundary, expected_boundary):
        errors.append("helper lease boundaries do not reconstruct from learner config")
    if not np.array_equal(trace.beneficiary_lease_boundary, expected_boundary):
        errors.append("beneficiary lease boundaries do not reconstruct from learner config")

    # Channel intervention: the delivered symbol must match the condition's
    # channel exactly (identity, constant zero, or isolated-stream shuffle).
    if spec.channel == DIRECT_TERNARY_CHANNEL:
        if not _array_equal(trace.delivered_message, trace.helper_message):
            errors.append("direct channel changed the helper message")
    elif spec.channel == CONSTANT_ZERO_TERNARY_CHANNEL:
        if not np.array_equal(
            np.asarray(trace.delivered_message),
            np.full((run.config.num_steps,), CONSTANT_CHANNEL_SYMBOL, dtype=np.int8),
        ):
            errors.append("constant channel is not exact symbol 0")
    elif spec.channel == SHUFFLED_TERNARY_CHANNEL:
        channel_key = hidden_regime_world_keys(jr.key(run.seed_pair.world_seed)).channel
        delivered = _shuffled_channel_replay(run.config.num_steps)(channel_key)
        if not np.array_equal(trace.delivered_message, np.asarray(delivered, dtype=np.int8)):
            errors.append("shuffled channel does not match its isolated world RNG stream")

    # Per-role audit: write gating, pre/post state continuity, exact value-bank
    # and status/generation reconstruction from ordinary writes plus atomic
    # commits, lease and relevance diagnostics, the candidate-confirmation
    # lifecycle, and the selective durable-mutation audit.
    for role, enabled in (
        ("helper", spec.helper_write),
        ("beneficiary", spec.beneficiary_write),
    ):
        if not np.array_equal(
            np.asarray(getattr(trace, f"{role}_write_enabled")),
            np.full((run.config.num_steps,), enabled, dtype=np.bool_),
        ):
            errors.append(f"{role} write intervention changed during the life")
        if not enabled and np.any(np.asarray(getattr(trace, f"{role}_value_write"))):
            errors.append(f"{role} frozen control contains an ordinary value write")
        for suffix in (
            "value_bits",
            "relevance_mean",
            "relevance_mass",
            "failed_leases",
            "idle_leases",
            "status",
            "generation",
            "active_slot",
            "lease_offset",
            "lease_reward_sum",
            "remaining_durable_tests",
            "search_cursor",
            "candidate_confirmations",
            "next_generation",
            "policy_key_data",
        ):
            pre = np.asarray(getattr(trace, f"{role}_{suffix}_pre"))
            post = np.asarray(getattr(trace, f"{role}_{suffix}_post"))
            if run.config.num_steps > 1 and not np.array_equal(pre[1:], post[:-1]):
                errors.append(f"{role} {suffix} trace continuity failed")
        pre_bits = np.asarray(getattr(trace, f"{role}_value_bits_pre"), dtype=np.uint32)
        post_bits = np.asarray(getattr(trace, f"{role}_value_bits_post"), dtype=np.uint32)
        pre_status = np.asarray(getattr(trace, f"{role}_status_pre"), dtype=np.int32)
        post_status = np.asarray(getattr(trace, f"{role}_status_post"), dtype=np.int32)
        pre_generation = np.asarray(
            getattr(trace, f"{role}_generation_pre"),
            dtype=np.int32,
        )
        post_generation = np.asarray(
            getattr(trace, f"{role}_generation_post"),
            dtype=np.int32,
        )
        committed_slot = np.asarray(getattr(trace, f"{role}_committed_slot"), dtype=np.int32)
        committed_generation = np.asarray(
            getattr(trace, f"{role}_committed_generation"),
            dtype=np.int32,
        )
        retired_slot = np.asarray(getattr(trace, f"{role}_retired_slot"), dtype=np.int32)
        retired_generation = np.asarray(
            getattr(trace, f"{role}_retired_generation"),
            dtype=np.int32,
        )
        active_slot = np.asarray(getattr(trace, f"{role}_active_slot_pre"), dtype=np.int32)
        active_slot_post = np.asarray(
            getattr(trace, f"{role}_active_slot_post"),
            dtype=np.int32,
        )
        row = np.arange(run.config.num_steps, dtype=np.int32)
        if np.any(pre_status[row, active_slot] == SLOT_VACANT):
            errors.append(f"{role} pre-state selected a vacant slot")
        if np.any(post_status[row, active_slot_post] == SLOT_VACANT):
            errors.append(f"{role} post-state selected a vacant slot")
        for status_values, generation_values, suffix in (
            (pre_status, pre_generation, "pre"),
            (post_status, post_generation, "post"),
        ):
            durable_generation = np.logical_and(
                status_values == SLOT_DURABLE,
                generation_values > 0,
            )
            zero_generation = np.logical_and(
                status_values != SLOT_DURABLE,
                generation_values == 0,
            )
            if not np.all(np.logical_or(durable_generation, zero_generation)):
                errors.append(f"{role} {suffix} status/generation identity is inconsistent")
        event_generations = committed_generation[committed_generation >= 0]
        if not np.array_equal(
            event_generations,
            np.arange(1, event_generations.size + 1, dtype=np.int32),
        ):
            errors.append(f"{role} committed generations are not unique and consecutive")
        private_input = (
            np.asarray(trace.helper_cue, dtype=np.int32)
            if role == "helper"
            else np.asarray(trace.delivered_message, dtype=np.int32)
        )
        action = (
            np.asarray(trace.helper_message, dtype=np.int32)
            if role == "helper"
            else np.asarray(trace.beneficiary_action, dtype=np.int32)
        )
        decision_slot = np.asarray(getattr(trace, f"{role}_decision_slot"), dtype=np.int32)
        decision_input = np.asarray(getattr(trace, f"{role}_private_input"), dtype=np.int32)
        decision_action = np.asarray(getattr(trace, f"{role}_decision_action"), dtype=np.int32)
        if not np.array_equal(decision_slot, active_slot):
            errors.append(f"{role} decision slot differs from the pre-state active slot")
        if not np.array_equal(decision_input, private_input):
            errors.append(f"{role} decision private input differs from the ordinary stream")
        if not np.array_equal(decision_action, action):
            errors.append(f"{role} decision action differs from the emitted action")
        selected_pre_bits = pre_bits[row, active_slot, private_input, action]
        selected_post_bits = post_bits[row, active_slot, private_input, action]
        recorded_pre_bits = _float32_bits(getattr(trace, f"{role}_value_pre"))
        candidate_bits = _float32_bits(getattr(trace, f"{role}_candidate_value"))
        recorded_post_bits = _float32_bits(getattr(trace, f"{role}_value_post"))
        if not np.array_equal(recorded_pre_bits, selected_pre_bits):
            errors.append(f"{role} value_pre is not the selected pre-state cell")
        if not np.array_equal(
            _float32_bits(getattr(trace, f"{role}_selected_value")),
            selected_pre_bits,
        ):
            errors.append(f"{role} selected value is not the selected pre-state cell")
        if not np.array_equal(recorded_post_bits, selected_post_bits):
            errors.append(f"{role} value_post is not the selected post-state cell")
        pre_values = np.asarray(getattr(trace, f"{role}_value_pre"), dtype=np.float32)
        candidate_unfused, candidate_fused = _portable_candidate_bits(
            pre_values,
            np.asarray(trace.reward, dtype=np.float32),
            run.config.learner.learning_rate,
        )
        if not np.all(
            np.logical_or(candidate_bits == candidate_unfused, candidate_bits == candidate_fused)
        ):
            errors.append(f"{role} candidate value is not a float32 learner update")
        status_at_active = pre_status[row, active_slot]
        expected_value_write = np.logical_and(
            enabled,
            np.logical_or(
                active_slot == 0,
                np.logical_and(spec.durable_writes_enabled, status_at_active == SLOT_DURABLE),
            ),
        )
        recorded_value_write = np.asarray(getattr(trace, f"{role}_value_write"))
        if not np.array_equal(recorded_value_write, expected_value_write):
            errors.append(f"{role} value_write does not match the structural write gate")

        expected_post_bits = pre_bits.copy()
        for step in range(run.config.num_steps):
            slot = int(active_slot[step])
            input_index = int(private_input[step])
            action_index = int(action[step])
            if expected_value_write[step]:
                expected_post_bits[step, slot, input_index, action_index] = candidate_bits[step]
            target_slot = int(committed_slot[step])
            if target_slot >= 0:
                scratch = expected_post_bits[step, 0].copy()
                expected_post_bits[step, target_slot] = scratch
                expected_post_bits[step, 0] = np.uint32(0)
        if not np.array_equal(post_bits, expected_post_bits):
            errors.append(f"{role} value bank does not reconstruct from writes and commits")

        expected_status_post = pre_status.copy()
        expected_generation_post = pre_generation.copy()
        for step_value in np.flatnonzero(committed_slot >= 0):
            step = int(step_value)
            slot = int(committed_slot[step])
            if slot in (1, 2, 3):
                expected_status_post[step, slot] = SLOT_DURABLE
                expected_status_post[step, 0] = SLOT_SCRATCH
                expected_generation_post[step, slot] = committed_generation[step]
                expected_generation_post[step, 0] = 0
        if not np.array_equal(post_status, expected_status_post):
            errors.append(f"{role} status does not reconstruct from atomic commits")
        if not np.array_equal(post_generation, expected_generation_post):
            errors.append(f"{role} generation does not reconstruct from atomic commits")

        candidate_pre = np.asarray(
            getattr(trace, f"{role}_candidate_confirmations_pre"),
            dtype=np.int32,
        )
        candidate_post = np.asarray(
            getattr(trace, f"{role}_candidate_confirmations_post"),
            dtype=np.int32,
        )
        candidate_success = np.asarray(
            getattr(trace, f"{role}_candidate_lease_success"),
            dtype=np.bool_,
        )
        scratch_failed_pre = np.asarray(
            getattr(trace, f"{role}_scratch_failed_leases_pre"),
            dtype=np.int32,
        )
        scratch_failed_post = np.asarray(
            getattr(trace, f"{role}_scratch_failed_leases_post"),
            dtype=np.int32,
        )
        scratch_retest_started = np.asarray(
            getattr(trace, f"{role}_scratch_retest_started"),
            dtype=np.bool_,
        )
        lease_offset_pre = np.asarray(
            getattr(trace, f"{role}_lease_offset_pre"),
            dtype=np.int32,
        )
        lease_offset_post = np.asarray(
            getattr(trace, f"{role}_lease_offset_post"),
            dtype=np.int32,
        )
        recorded_boundary = np.asarray(
            getattr(trace, f"{role}_lease_boundary"),
            dtype=np.bool_,
        )
        offset_boundary = lease_offset_pre == run.config.learner.lease_length - 1
        if not np.array_equal(recorded_boundary, offset_boundary):
            errors.append(f"{role} boundary does not match the recorded lease offset")
        expected_offset_post = np.where(offset_boundary, 0, lease_offset_pre + 1)
        if not np.array_equal(lease_offset_post, expected_offset_post):
            errors.append(f"{role} lease offset does not reconstruct")
        lease_sum_pre = np.asarray(
            getattr(trace, f"{role}_lease_reward_sum_pre"),
            dtype=np.float32,
        )
        lease_sum_post = np.asarray(
            getattr(trace, f"{role}_lease_reward_sum_post"),
            dtype=np.float32,
        )
        lease_sum = np.asarray(
            lease_sum_pre + np.asarray(trace.reward, dtype=np.float32),
            dtype=np.float32,
        )
        expected_lease_sum_post = np.where(
            offset_boundary,
            np.float32(0.0),
            lease_sum,
        ).astype(np.float32)
        if not np.array_equal(
            _float32_bits(lease_sum_post),
            _float32_bits(expected_lease_sum_post),
        ):
            errors.append(f"{role} lease reward sum does not reconstruct")
        expected_lease_mean = np.asarray(
            lease_sum / np.float32(run.config.learner.lease_length),
            dtype=np.float32,
        )
        if not np.array_equal(
            _float32_bits(getattr(trace, f"{role}_lease_reward_mean")),
            _float32_bits(expected_lease_mean),
        ):
            errors.append(f"{role} lease reward diagnostic does not reconstruct")
        relevance_mass_pre = np.asarray(
            getattr(trace, f"{role}_relevance_mass_pre"),
            dtype=np.float32,
        )
        active_mass = np.minimum(
            relevance_mass_pre[row, active_slot] + np.float32(1.0),
            np.float32(16_777_216.0),
        )
        expected_relevance_ready = active_mass >= np.float32(
            run.config.learner.confirmation_steps
        )
        if not np.array_equal(
            np.asarray(getattr(trace, f"{role}_relevance_ready")),
            expected_relevance_ready,
        ):
            errors.append(f"{role} relevance-ready diagnostic does not reconstruct")
        relevance_mean_pre = np.asarray(
            getattr(trace, f"{role}_relevance_mean_pre"),
            dtype=np.float32,
        )
        relevance_gain = np.maximum(
            np.float32(run.config.learner.relevance_rate),
            np.asarray(np.float32(1.0) / active_mass, dtype=np.float32),
        )
        active_relevance = np.asarray(
            relevance_mean_pre[row, active_slot]
            + relevance_gain
            * (
                np.asarray(trace.reward, dtype=np.float32)
                - relevance_mean_pre[row, active_slot]
            ),
            dtype=np.float32,
        )
        expected_durable_relevant = np.logical_and(
            expected_relevance_ready,
            active_relevance >= np.float32(run.config.learner.durable_retrieval_threshold),
        )
        expected_candidate_relevant = np.logical_and(
            expected_relevance_ready,
            active_relevance
            >= np.float32(run.config.learner.candidate_confirmation_threshold),
        )
        if not np.array_equal(
            np.asarray(getattr(trace, f"{role}_durable_relevant")),
            expected_durable_relevant,
        ):
            errors.append(f"{role} durable-relevance diagnostic does not reconstruct")
        if not np.array_equal(
            np.asarray(getattr(trace, f"{role}_candidate_relevant")),
            expected_candidate_relevant,
        ):
            errors.append(f"{role} candidate-relevance diagnostic does not reconstruct")
        expected_candidate_success = np.logical_and(
            offset_boundary,
            np.logical_and(
                active_slot == SCRATCH_SLOT,
                np.logical_and(
                    expected_candidate_relevant,
                    expected_lease_mean
                    >= np.float32(run.config.learner.candidate_confirmation_threshold),
                ),
            ),
        )
        if not np.array_equal(candidate_success, expected_candidate_success):
            errors.append(f"{role} candidate lease success does not reconstruct")
        saturated_candidate = np.minimum(
            candidate_pre.astype(np.int64) + 1,
            np.iinfo(np.int32).max,
        )
        candidate_confirmed = np.logical_and(
            candidate_success,
            saturated_candidate >= run.config.learner.candidate_confirmation_leases,
        )
        next_generation_pre = np.asarray(
            getattr(trace, f"{role}_next_generation_pre"),
            dtype=np.int32,
        )
        expected_generation_exhausted = np.logical_and(
            candidate_confirmed,
            np.logical_and(
                spec.helper_write and spec.beneficiary_write,
                next_generation_pre == np.iinfo(np.int32).max,
            ),
        )
        if not np.array_equal(
            np.asarray(getattr(trace, f"{role}_generation_exhausted")),
            expected_generation_exhausted,
        ):
            errors.append(f"{role} generation-exhausted diagnostic does not reconstruct")
        if scratch_failed_pre[0] != 0 or (
            run.config.num_steps > 1
            and not np.array_equal(scratch_failed_pre[1:], scratch_failed_post[:-1])
        ):
            errors.append(f"{role} scratch-failure counter continuity failed")
        if not np.array_equal(
            scratch_failed_pre,
            np.asarray(getattr(trace, f"{role}_failed_leases_pre"))[:, SCRATCH_SLOT],
        ):
            errors.append(f"{role} scratch-failure pre diagnostic differs from state")
        if not np.array_equal(
            scratch_failed_post,
            np.asarray(getattr(trace, f"{role}_failed_leases_post"))[:, SCRATCH_SLOT],
        ):
            errors.append(f"{role} scratch-failure post diagnostic differs from state")
        if np.any(scratch_failed_pre < 0) or np.any(scratch_failed_post < 0):
            errors.append(f"{role} scratch-failure counter became negative")
        if np.any(scratch_failed_post >= run.config.learner.scratch_training_leases_before_retest):
            errors.append(f"{role} scratch-failure counter exceeded its residency bound")
        if np.any(
            np.logical_and(
                scratch_retest_started,
                np.logical_or(
                    ~expected_boundary,
                    np.logical_or(active_slot != 0, active_slot_post == 0),
                ),
            )
        ):
            errors.append(f"{role} scratch retest did not start at a valid boundary")
        if np.any(np.logical_and(scratch_retest_started, scratch_failed_post != 0)):
            errors.append(f"{role} scratch retest did not reset the failure counter")
        if np.any(np.logical_and(candidate_success, ~expected_boundary)):
            errors.append(f"{role} candidate success occurred outside a lease boundary")
        if np.any(np.logical_and(candidate_success, active_slot != 0)):
            errors.append(f"{role} durable slot was mislabeled as a scratch candidate")
        for step_value in np.flatnonzero(candidate_success):
            step = int(step_value)
            if committed_slot[step] >= 0:
                if candidate_post[step] != 0:
                    errors.append(f"{role} confirmed commit did not reset candidate count")
                if candidate_pre[step] + 1 < run.config.learner.candidate_confirmation_leases:
                    errors.append(f"{role} commit bypassed candidate confirmation leases")
            elif candidate_post[step] != candidate_pre[step] + 1:
                errors.append(f"{role} successful candidate lease did not increment exactly")
        if np.any(np.logical_and(committed_slot >= 0, ~candidate_success)):
            errors.append(f"{role} commit lacked a successful candidate lease")
        recomputed = np.asarray(
            jax.vmap(
                lambda *args: _role_selective_mutation_violation(
                    *args,
                    spec.durable_writes_enabled,
                )
            )(
                jnp.asarray(pre_status),
                jnp.asarray(post_status),
                jnp.asarray(pre_generation),
                jnp.asarray(post_generation),
                jnp.asarray(pre_bits),
                jnp.asarray(post_bits),
                jnp.asarray(committed_slot),
                jnp.asarray(committed_generation),
                jnp.asarray(retired_slot),
                jnp.asarray(retired_generation),
            )
        )
        recorded = np.asarray(getattr(trace, f"{role}_selective_mutation_violation"))
        if not np.array_equal(recorded, recomputed):
            errors.append(f"{role} selective durable mutation audit mismatch")
        if not spec.durable_writes_enabled and np.any(recomputed):
            errors.append(f"{role} selective durable bits changed without atomic replacement")

        for step_value in np.flatnonzero(committed_slot >= 0):
            step = int(step_value)
            slot = int(committed_slot[step])
            if slot not in (1, 2, 3):
                errors.append(f"{role} commit target is not durable")
                continue
            if post_status[step, slot] != SLOT_DURABLE:
                errors.append(f"{role} committed slot is not durable post-state")
            if post_generation[step, slot] != committed_generation[step]:
                errors.append(f"{role} committed generation does not match post-state")
            replacement = retired_slot[step] >= 0
            if replacement:
                if retired_slot[step] != slot:
                    errors.append(f"{role} retirement and commit are not atomic in one slot")
                if pre_status[step, slot] != SLOT_DURABLE:
                    errors.append(f"{role} replacement did not retire a durable slot")
                if pre_generation[step, slot] != retired_generation[step]:
                    errors.append(f"{role} retired generation does not match pre-state")
                if pre_generation[step, slot] == post_generation[step, slot]:
                    errors.append(f"{role} replacement reused the same generation")
            elif pre_status[step, slot] != SLOT_VACANT:
                errors.append(f"{role} vacancy commit did not target a vacant slot")

    # Cross-role synchronization: the two roles run one joint lifecycle state
    # machine, so freeze controls gate both and lifecycle fields must agree.
    if not spec.helper_write or not spec.beneficiary_write:
        for role in ("helper", "beneficiary"):
            if np.any(np.asarray(getattr(trace, f"{role}_committed_slot")) >= 0):
                errors.append("freeze control bypassed the joint lifecycle commit gate")
            if np.any(np.asarray(getattr(trace, f"{role}_retired_slot")) >= 0):
                errors.append("freeze control bypassed the joint lifecycle replacement gate")
    if not _array_equal(
        trace.helper_candidate_lease_success,
        trace.beneficiary_candidate_lease_success,
    ):
        errors.append("candidate lease success differs between synchronized roles")
    if not _array_equal(trace.helper_lease_boundary, trace.beneficiary_lease_boundary):
        errors.append("lease boundaries differ between synchronized roles")
    if not np.all(np.asarray(trace.lifecycle_synchronized)):
        errors.append("role lifecycle states were not synchronized")
    for name in (
        "relevance_mean_pre",
        "relevance_mean_post",
        "relevance_mass_pre",
        "relevance_mass_post",
        "failed_leases_pre",
        "failed_leases_post",
        "idle_leases_pre",
        "idle_leases_post",
        "active_slot_pre",
        "active_slot_post",
        "status_pre",
        "status_post",
        "generation_pre",
        "generation_post",
        "candidate_confirmations_pre",
        "candidate_confirmations_post",
        "lease_offset_pre",
        "lease_offset_post",
        "lease_reward_sum_pre",
        "lease_reward_sum_post",
        "remaining_durable_tests_pre",
        "remaining_durable_tests_post",
        "search_cursor_pre",
        "search_cursor_post",
        "next_generation_pre",
        "next_generation_post",
        "scratch_failed_leases_pre",
        "scratch_failed_leases_post",
        "scratch_retest_started",
        "committed_slot",
        "committed_generation",
        "retired_slot",
        "retired_generation",
        "lease_boundary",
        "lease_reward_mean",
        "relevance_ready",
        "durable_relevant",
        "candidate_relevant",
        "candidate_lease_success",
        "generation_exhausted",
    ):
        if not _array_equal(
            getattr(trace, f"helper_{name}"),
            getattr(trace, f"beneficiary_{name}"),
        ):
            errors.append(f"synchronized lifecycle field {name} differs by role")

    # Derived report: the summary must reconstruct exactly from the primitive
    # trace, and the resource report must describe the constant matched dyad.
    try:
        recomputed_summary = reconstruct_hidden_regime_summary(
            trace,
            run.config,
            run.condition,
        )
    except (IndexError, ValueError) as error:
        errors.append(f"summary lineage reconstruction failed closed: {error}")
    else:
        if recomputed_summary.to_dict() != run.summary.to_dict():
            errors.append("summary does not reconstruct exactly from primitive trace")
    expected_resource = _resource_report(
        SlotSignalingAgent(
            dataclasses.replace(
                run.config.learner,
                writable_lru_ablation=False,
                durable_write_policy=spec.durable_write_policy,
                replacement_target_policy=spec.replacement_target_policy,
            )
        ).init(slot_signaling_keys(jr.key(run.seed_pair.learner_seed))),
        run.final_state,
    )
    if expected_resource != run.resource:
        errors.append("resource report does not match initial/final persistent state")
    # The literal must stay equal to EXPECTED_DYAD_STATE_BYTES; the byte
    # derivation lives next to that constant at the top of the module.
    if not run.resource.resource_constant or run.resource.final_state_bytes != 552:
        errors.append("condition is not the exact constant 552-byte dyad")
    # Trace endpoints: step 0 pre-state and step N-1 post-state must equal the
    # actual initial and final persistent learner states, field by field.
    initial_state = SlotSignalingAgent(
        dataclasses.replace(
            run.config.learner,
            writable_lru_ablation=False,
            durable_write_policy=spec.durable_write_policy,
            replacement_target_policy=spec.replacement_target_policy,
        )
    ).init(slot_signaling_keys(jr.key(run.seed_pair.learner_seed)))
    state_trace_names = {
        "values": "value_bits",
        "relevance_mean": "relevance_mean",
        "relevance_mass": "relevance_mass",
        "failed_leases": "failed_leases",
        "idle_leases": "idle_leases",
        "status": "status",
        "generation": "generation",
        "active_slot": "active_slot",
        "lease_offset": "lease_offset",
        "lease_reward_sum": "lease_reward_sum",
        "remaining_durable_tests": "remaining_durable_tests",
        "search_cursor": "search_cursor",
        "candidate_successful_leases": "candidate_confirmations",
        "next_generation": "next_generation",
        "key": "policy_key_data",
    }
    for role in ("helper", "beneficiary"):
        initial_role = getattr(initial_state, role)
        final_role = getattr(run.final_state, role)
        for state_name, trace_name in state_trace_names.items():
            initial_value = getattr(initial_role, state_name)
            final_value = getattr(final_role, state_name)
            if state_name == "values":
                initial_expected = np.asarray(initial_value).view(np.uint32)
                final_expected = np.asarray(final_value).view(np.uint32)
            elif state_name == "key":
                initial_expected = np.asarray(jr.key_data(initial_value), dtype=np.uint32)
                final_expected = np.asarray(jr.key_data(final_value), dtype=np.uint32)
            else:
                initial_expected = np.asarray(initial_value)
                final_expected = np.asarray(final_value)
            if not np.array_equal(
                np.asarray(getattr(trace, f"{role}_{trace_name}_pre"))[0],
                initial_expected,
            ):
                errors.append(f"{role} initial {state_name} differs from trace")
            if not np.array_equal(
                np.asarray(getattr(trace, f"{role}_{trace_name}_post"))[-1],
                final_expected,
            ):
                errors.append(f"{role} final {state_name} differs from trace")
    # Same-implementation determinism: a full named-RNG replay must be
    # bit-identical.  This is a consistency check, not cross-backend evidence.
    replay_trace, replay_final_state = _deterministic_replay(run, spec)
    for field in dataclasses.fields(trace):
        if not _array_equal(getattr(trace, field.name), getattr(replay_trace, field.name)):
            errors.append(f"trace.{field.name} differs from deterministic named-RNG replay")
    for role in ("helper", "beneficiary"):
        actual_role = getattr(run.final_state, role)
        replay_role = getattr(replay_final_state, role)
        for field in dataclasses.fields(actual_role):
            actual_value = getattr(actual_role, field.name)
            replay_value = getattr(replay_role, field.name)
            equal = (
                np.array_equal(jr.key_data(actual_value), jr.key_data(replay_value))
                if field.name == "key"
                else _array_equal(actual_value, replay_value)
            )
            if not equal:
                errors.append(f"final_state.{role}.{field.name} differs from replay")
    return tuple(errors)


def _config_from_payload(payload: object) -> HiddenRegimeDevelopmentConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("config must be a mapping")
    expected_config_fields = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "acceptance_status",
        "claim_thresholds_frozen",
        "world",
        "learner",
        "metric_window",
        "num_steps",
        "default_schedule_semantics",
        "development_candidate_provenance",
        "development_calibration_limitations",
        "replay_portability_scope",
        "retention_scope",
        "retention_window_semantics",
        "legacy_metric_window_semantics",
    }
    if set(payload) != expected_config_fields:
        raise ValueError("config fields do not match strict v4")
    world_payload = payload.get("world")
    learner_payload = payload.get("learner")
    if not isinstance(world_payload, Mapping) or not isinstance(learner_payload, Mapping):
        raise ValueError("world and learner configs must be mappings")
    expected_world_fields = {
        "segment_lengths",
        "segment_regimes",
        "regime_permutations",
        "repeat_schedule",
        "total_schedule_steps",
        "development_only",
        "scientific_promotion_allowed",
    }
    expected_learner_fields = {
        "learning_rate",
        "epsilon",
        "relevance_rate",
        "lease_length",
        "confirmation_steps",
        "durable_retrieval_threshold",
        "candidate_confirmation_threshold",
        "candidate_confirmation_leases",
        "scratch_training_leases_before_retest",
        "writable_lru_ablation",
        "requested_durable_write_policy",
        "requested_replacement_target_policy",
        "effective_durable_write_policy",
        "effective_replacement_target_policy",
        "development_only",
        "scientific_promotion_allowed",
    }
    if set(world_payload) != expected_world_fields:
        raise ValueError("world config fields do not match strict v4")
    if set(learner_payload) != expected_learner_fields:
        raise ValueError("learner config fields do not match strict v4")
    lengths = world_payload.get("segment_lengths")
    regimes = world_payload.get("segment_regimes")
    permutations = world_payload.get("regime_permutations")
    if not isinstance(lengths, list) or not isinstance(regimes, list):
        raise ValueError("serialized schedule must use lists")
    if not isinstance(permutations, list) or any(not isinstance(row, list) for row in permutations):
        raise ValueError("serialized permutations must use nested lists")
    world = HiddenRegimeWorldConfig(
        segment_lengths=cast(tuple[int, ...], tuple(lengths)),
        segment_regimes=cast(tuple[int, ...], tuple(regimes)),
        regime_permutations=cast(
            tuple[tuple[int, int, int], ...],
            tuple(tuple(row) for row in permutations),
        ),
        repeat_schedule=cast(bool, world_payload.get("repeat_schedule")),
    )
    learner = SlotSignalingConfig(
        learning_rate=cast(float, learner_payload.get("learning_rate")),
        epsilon=cast(float, learner_payload.get("epsilon")),
        relevance_rate=cast(float, learner_payload.get("relevance_rate")),
        lease_length=cast(int, learner_payload.get("lease_length")),
        confirmation_steps=cast(int, learner_payload.get("confirmation_steps")),
        durable_retrieval_threshold=cast(
            float,
            learner_payload.get("durable_retrieval_threshold"),
        ),
        candidate_confirmation_threshold=cast(
            float,
            learner_payload.get("candidate_confirmation_threshold"),
        ),
        candidate_confirmation_leases=cast(
            int,
            learner_payload.get("candidate_confirmation_leases"),
        ),
        scratch_training_leases_before_retest=cast(
            int,
            learner_payload.get("scratch_training_leases_before_retest"),
        ),
        writable_lru_ablation=cast(
            bool,
            learner_payload.get("writable_lru_ablation"),
        ),
        durable_write_policy=cast(
            DurableWritePolicy | None,
            learner_payload.get("requested_durable_write_policy"),
        ),
        replacement_target_policy=cast(
            ReplacementTargetPolicy | None,
            learner_payload.get("requested_replacement_target_policy"),
        ),
    )
    config = HiddenRegimeDevelopmentConfig(
        world=world,
        learner=learner,
        metric_window=cast(int, payload.get("metric_window")),
    )
    if config.to_dict() != dict(payload):
        raise ValueError("serialized config does not reconstruct exactly")
    return config


def _seed_pair_from_payload(payload: object) -> HiddenRegimeSeedPair:
    if not isinstance(payload, Mapping):
        raise ValueError("seed_pair must be a mapping")
    if set(payload) != {"namespace", "index", "world_seed", "learner_seed"}:
        raise ValueError("seed_pair fields do not match strict v4")
    return HiddenRegimeSeedPair(
        namespace=cast(str, payload.get("namespace")),
        index=cast(int, payload.get("index")),
        world_seed=cast(int, payload.get("world_seed")),
        learner_seed=cast(int, payload.get("learner_seed")),
    )


def validate_hidden_regime_development_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Fail closed on a serialized development report via deterministic replay.

    Replay is only a validator for an already-consumed development seed pair;
    it is not an artifact, an acceptance gate, or a promotion mechanism.
    """

    errors: list[str] = []
    expected = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "acceptance_status",
        "claim_thresholds_frozen",
        "artifact_written",
        "reserved_seed_namespace",
        "reserved_namespace_executed",
        "oracle_upper_bound_included",
        "seed_pair",
        "config",
        "condition_order",
        "runs",
        "paired_controls",
    }
    if set(payload) != expected:
        errors.append("report fields do not match the strict v4 schema")
    if payload.get("schema_version") != HIDDEN_REGIME_DEVELOPMENT_SCHEMA:
        errors.append("schema_version mismatch")
    if payload.get("development_only") is not True:
        errors.append("report must remain development-only")
    if payload.get("scientific_promotion_allowed") is not False:
        errors.append("scientific promotion must remain disabled")
    if payload.get("acceptance_status") != ACCEPTANCE_STATUS:
        errors.append("acceptance_status must remain descriptive-only")
    if payload.get("claim_thresholds_frozen") is not False:
        errors.append("development report cannot claim frozen thresholds")
    if payload.get("artifact_written") is not False:
        errors.append("development evaluator cannot write an artifact")
    if payload.get("reserved_seed_namespace") != RESERVED_DEVELOPMENT_SEED_NAMESPACE:
        errors.append("reserved seed namespace mismatch")
    if payload.get("reserved_namespace_executed") is not False:
        errors.append("reserved seed namespace must remain unexecuted")
    if payload.get("oracle_upper_bound_included") is not False:
        errors.append("oracle upper bound is not a learner condition in v4")

    seed_payload = payload.get("seed_pair")
    parsed_seed_pair: HiddenRegimeSeedPair | None
    try:
        parsed_seed_pair = _seed_pair_from_payload(seed_payload)
    except (TypeError, ValueError):
        parsed_seed_pair = None
        errors.append("seed_pair does not reconstruct under the strict v4 schema")
    else:
        assert parsed_seed_pair is not None
        if parsed_seed_pair.namespace == RESERVED_DEVELOPMENT_SEED_NAMESPACE:
            errors.append("report cannot contain the unexecuted reserved seed namespace")
    parsed_config: HiddenRegimeDevelopmentConfig | None
    try:
        parsed_config = _config_from_payload(payload.get("config"))
    except (TypeError, ValueError):
        parsed_config = None
        errors.append("config does not reconstruct under the strict v4 schema")

    condition_payload = payload.get("condition_order")
    run_payload = payload.get("runs")
    control_payload = payload.get("paired_controls")
    if not isinstance(condition_payload, list) or not condition_payload:
        errors.append("condition_order must be a nonempty list")
        conditions: list[object] = []
    else:
        conditions = condition_payload
    if not isinstance(run_payload, list) or len(run_payload) != len(conditions):
        errors.append("runs must align exactly with condition_order")
        runs: list[object] = []
    else:
        runs = run_payload
    if conditions:
        try:
            _validated_condition_order(
                cast(Sequence[HiddenRegimeDevelopmentCondition], conditions)
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"condition_order is not canonical: {exc}")

    expected_trace_fields = {
        "schema_version",
        *(field.name for field in dataclasses.fields(HiddenRegimePrimitiveTrace)),
    }
    for index, run_object in enumerate(runs):
        if not isinstance(run_object, Mapping):
            errors.append(f"run {index} is not a mapping")
            continue
        run = run_object
        trace_included = run.get("trace_included")
        expected_run_fields = {
            "schema_version",
            "development_only",
            "scientific_promotion_allowed",
            "acceptance_status",
            "claim_thresholds_frozen",
            "artifact_written",
            "reserved_namespace_executed",
            "oracle_upper_bound_included",
            "condition",
            "seed_pair",
            "config",
            "resource",
            "summary",
            "trace_included",
        }
        if trace_included is True:
            expected_run_fields.add("trace")
        elif trace_included is not False:
            errors.append(f"run {index} trace_included must be boolean")
        if set(run) != expected_run_fields:
            errors.append(f"run {index} fields do not match the strict v4 schema")
        if run.get("schema_version") != HIDDEN_REGIME_DEVELOPMENT_SCHEMA:
            errors.append(f"run {index} schema mismatch")
        if run.get("development_only") is not True:
            errors.append(f"run {index} must remain development-only")
        if run.get("scientific_promotion_allowed") is not False:
            errors.append(f"run {index} cannot promote scientific evidence")
        if run.get("acceptance_status") != ACCEPTANCE_STATUS:
            errors.append(f"run {index} acceptance scope mismatch")
        if run.get("claim_thresholds_frozen") is not False:
            errors.append(f"run {index} cannot claim frozen thresholds")
        if run.get("artifact_written") is not False:
            errors.append(f"run {index} cannot claim an artifact")
        if run.get("reserved_namespace_executed") is not False:
            errors.append(f"run {index} cannot execute the reserved namespace")
        if run.get("oracle_upper_bound_included") is not False:
            errors.append(f"run {index} cannot include an oracle learner")
        if index < len(conditions) and run.get("condition") != conditions[index]:
            errors.append(f"run {index} condition order mismatch")
        if run.get("seed_pair") != seed_payload:
            errors.append(f"run {index} seed pair differs from paired report")
        if run.get("config") != payload.get("config"):
            errors.append(f"run {index} config differs from paired report")
        resource = run.get("resource")
        if not isinstance(resource, Mapping):
            errors.append(f"run {index} resource missing")
        elif (
            resource.get("initial_state_bytes") != EXPECTED_DYAD_STATE_BYTES
            or resource.get("final_state_bytes") != EXPECTED_DYAD_STATE_BYTES
            or resource.get("resource_constant") is not True
            or resource.get("resource_matched") is not True
        ):
            errors.append(f"run {index} is not a matched constant 552-byte dyad")
        summary = run.get("summary")
        if not isinstance(summary, Mapping):
            errors.append(f"run {index} summary missing")
        else:
            expected_summary_fields = {
                field.name for field in dataclasses.fields(HiddenRegimeRunSummary)
            }
            if set(summary) != expected_summary_fields:
                errors.append(f"run {index} summary fields do not match strict v4")
            segment_payload = summary.get("segment_rewards")
            expected_segment_fields = {
                field.name for field in dataclasses.fields(SegmentRewardSummary)
            }
            if not isinstance(segment_payload, list) or any(
                not isinstance(item, Mapping) or set(item) != expected_segment_fields
                for item in segment_payload
            ):
                errors.append(f"run {index} legacy segment summaries are malformed")
            recurrence_payload = summary.get("recurrence_by_regime")
            expected_recurrence_fields = {
                field.name for field in dataclasses.fields(RegimeRecurrenceSummary)
            }
            if not isinstance(recurrence_payload, list) or any(
                not isinstance(item, Mapping) or set(item) != expected_recurrence_fields
                for item in recurrence_payload
            ):
                errors.append(f"run {index} legacy recurrence summaries are malformed")
            commit_payload = summary.get("commit_generation_lineages")
            expected_commit_fields = {
                field.name for field in dataclasses.fields(CommitGenerationLineage)
            }
            if not isinstance(commit_payload, list):
                errors.append(f"run {index} commit-generation lineages are malformed")
            else:
                for lineage_index, lineage in enumerate(commit_payload):
                    if not isinstance(lineage, Mapping) or set(lineage) != expected_commit_fields:
                        errors.append(
                            f"run {index} commit lineage {lineage_index} fields are malformed"
                        )
                        continue
                    fixed_lengths = {
                        "target_mapping": 3,
                        "committed_composed_greedy_mapping": 3,
                        "helper_table_uint32_bits": 9,
                        "beneficiary_table_uint32_bits": 9,
                    }
                    if any(
                        not isinstance(lineage.get(name), list)
                        or len(cast(list[object], lineage.get(name))) != length
                        for name, length in fixed_lengths.items()
                    ):
                        errors.append(
                            f"run {index} commit lineage {lineage_index} fixed arrays "
                            "are malformed"
                        )
            retention_payload = summary.get("recurrence_retention")
            expected_retention_record_fields = {
                field.name for field in dataclasses.fields(RecurrenceRetentionRecord)
            }
            expected_dormant_fields = {
                field.name for field in dataclasses.fields(DormantGenerationProbe)
            }
            expected_lineage_probe_fields = {
                field.name for field in dataclasses.fields(RecurrenceLineageProbe)
            }
            if not isinstance(retention_payload, list):
                errors.append(f"run {index} recurrence retention records are malformed")
            else:
                for record_index, record in enumerate(retention_payload):
                    if (
                        not isinstance(record, Mapping)
                        or set(record) != expected_retention_record_fields
                    ):
                        errors.append(
                            f"run {index} retention record {record_index} fields are malformed"
                        )
                        continue
                    eligible = record.get("eligible_dormant_generations")
                    if not isinstance(eligible, list) or any(
                        not isinstance(item, Mapping) or set(item) != expected_dormant_fields
                        for item in eligible
                    ):
                        errors.append(
                            f"run {index} retention record {record_index} dormant probes "
                            "are malformed"
                        )
                    lineage_probes = record.get("prior_same_regime_lineages")
                    if not isinstance(lineage_probes, list):
                        errors.append(
                            f"run {index} retention record {record_index} lineage probes "
                            "are malformed"
                        )
                    else:
                        for probe_index, probe in enumerate(lineage_probes):
                            if (
                                not isinstance(probe, Mapping)
                                or set(probe) != expected_lineage_probe_fields
                            ):
                                errors.append(
                                    f"run {index} retention record {record_index} lineage "
                                    f"probe {probe_index} fields are malformed"
                                )
                                continue
                            for name in (
                                "helper_entry_table_uint32_bits",
                                "beneficiary_entry_table_uint32_bits",
                            ):
                                value = probe.get(name)
                                if value is not None and (
                                    not isinstance(value, list) or len(value) != 9
                                ):
                                    errors.append(
                                        f"run {index} retention record {record_index} "
                                        f"lineage probe {probe_index} table is malformed"
                                    )
                            mapping = probe.get("entry_composed_greedy_mapping")
                            if mapping is not None and (
                                not isinstance(mapping, list) or len(mapping) != 3
                            ):
                                errors.append(
                                    f"run {index} retention record {record_index} lineage "
                                    f"probe {probe_index} mapping is malformed"
                                )
            aggregate_payload = summary.get("retention")
            expected_aggregate_fields = {
                field.name for field in dataclasses.fields(RetentionAggregateSummary)
            }
            if (
                not isinstance(aggregate_payload, Mapping)
                or set(aggregate_payload) != expected_aggregate_fields
            ):
                errors.append(f"run {index} retention aggregate is malformed")
        if trace_included is True:
            trace = run.get("trace")
            if not isinstance(trace, Mapping):
                errors.append(f"run {index} trace missing")
            elif set(trace) != expected_trace_fields:
                errors.append(f"run {index} trace fields do not match primitive v3")
            elif trace.get("schema_version") != HIDDEN_REGIME_TRACE_SCHEMA:
                errors.append(f"run {index} trace schema mismatch")

    expected_control_count = max(0, len(conditions) - 1)
    if not isinstance(control_payload, list) or len(control_payload) != expected_control_count:
        errors.append("paired_controls must contain every non-reference condition")
        controls: list[object] = []
    else:
        controls = control_payload
    if controls:
        expected_names = conditions[1:]
        actual_names = [
            item.get("condition") if isinstance(item, Mapping) else None for item in controls
        ]
        if actual_names != expected_names:
            errors.append("paired control order does not match condition_order")
        if any(
            not isinstance(item, Mapping) or item.get("resource_bytes") != EXPECTED_DYAD_STATE_BYTES
            for item in controls
        ):
            errors.append("paired control resource disclosure mismatch")

    if len(runs) == len(conditions) and runs and len(controls) == len(runs) - 1:
        selective_run = runs[0]
        selective_summary = (
            selective_run.get("summary") if isinstance(selective_run, Mapping) else None
        )
        selective_mean = (
            selective_summary.get("mean_prequential_reward")
            if isinstance(selective_summary, Mapping)
            else None
        )
        selective_recurrence = (
            selective_summary.get("recurrence_entry_reward_mean")
            if isinstance(selective_summary, Mapping)
            else None
        )
        for index, (control_object, run_object) in enumerate(zip(controls, runs[1:], strict=True)):
            if not isinstance(control_object, Mapping) or not isinstance(run_object, Mapping):
                continue
            summary = run_object.get("summary")
            if not isinstance(summary, Mapping):
                continue
            run_mean = summary.get("mean_prequential_reward")
            if control_object.get("mean_prequential_reward") != run_mean:
                errors.append(f"paired control {index} mean does not match run")
            if (
                type(run_mean) in (int, float)
                and type(selective_mean) in (int, float)
                and control_object.get("delta_vs_selective_full")
                != cast(float, run_mean) - cast(float, selective_mean)
            ):
                errors.append(f"paired control {index} reward delta does not reconstruct")
            run_recurrence = summary.get("recurrence_entry_reward_mean")
            if control_object.get("recurrence_entry_reward_mean") != run_recurrence:
                errors.append(f"paired control {index} recurrence mean does not match run")
            expected_recurrence_delta = (
                None
                if type(run_recurrence) not in (int, float)
                or type(selective_recurrence) not in (int, float)
                else cast(float, run_recurrence) - cast(float, selective_recurrence)
            )
            if (
                control_object.get("recurrence_entry_delta_vs_selective_full")
                != expected_recurrence_delta
            ):
                errors.append(f"paired control {index} recurrence delta does not reconstruct")
    if not errors and parsed_seed_pair is not None and parsed_config is not None:
        trace_flags = [
            run.get("trace_included") if isinstance(run, Mapping) else None for run in runs
        ]
        if not trace_flags or any(flag is not trace_flags[0] for flag in trace_flags):
            errors.append("all serialized runs must use one trace-inclusion policy")
        else:
            condition_tuple = cast(tuple[HiddenRegimeDevelopmentCondition, ...], tuple(conditions))
            replayed = run_hidden_regime_development(
                seed_pair=parsed_seed_pair,
                config=parsed_config,
                conditions=condition_tuple,
            ).to_dict(include_traces=cast(bool, trace_flags[0]))
            if replayed != dict(payload):
                errors.append("serialized report differs from exact deterministic replay")
    return tuple(errors)


__all__ = [
    "ACCEPTANCE_STATUS",
    "BENEFICIARY_FROZEN",
    "CONSTANT_CHANNEL",
    "CONSTANT_CHANNEL_SYMBOL",
    "DEFAULT_DEVELOPMENT_SEGMENT_LENGTHS",
    "DEVELOPMENT_CALIBRATION_LIMITATIONS",
    "DEVELOPMENT_CANDIDATE_PROVENANCE",
    "DEVELOPMENT_ONLY",
    "CommitGenerationLineage",
    "DormantGenerationProbe",
    "EXPECTED_DYAD_STATE_BYTES",
    "HELPER_FROZEN",
    "HIDDEN_REGIME_DEVELOPMENT_SCHEMA",
    "HIDDEN_REGIME_TRACE_SCHEMA",
    "HiddenRegimeDevelopmentConfig",
    "HiddenRegimeDevelopmentReport",
    "HiddenRegimePrimitiveTrace",
    "RecurrenceLineageProbe",
    "RecurrenceRetentionRecord",
    "RetentionAggregateSummary",
    "HiddenRegimeRunResult",
    "HiddenRegimeRunSummary",
    "HiddenRegimeSeedPair",
    "MATCHED_CONDITIONS",
    "RESERVED_DEVELOPMENT_SEED_NAMESPACE",
    "RESERVED_DEVELOPMENT_SEED_NAMESPACE_EXECUTED",
    "REPLAY_PORTABILITY_SCOPE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SELECTIVE_EVIDENCE",
    "SELECTIVE_FULL",
    "SELECTIVE_LRU",
    "SHUFFLED_CHANNEL",
    "WRITABLE_EVIDENCE",
    "WRITABLE_LRU",
    "condition_spec",
    "derive_hidden_regime_seed_pairs",
    "hidden_regime_coalesced_episode_bounds",
    "hidden_regime_lineage_recurrence_segments",
    "reconstruct_commit_generation_lineages",
    "reconstruct_hidden_regime_summary",
    "reconstruct_hidden_regime_retention",
    "run_hidden_regime_condition",
    "run_hidden_regime_development",
    "validate_hidden_regime_development_payload",
    "validate_hidden_regime_run_result",
]
