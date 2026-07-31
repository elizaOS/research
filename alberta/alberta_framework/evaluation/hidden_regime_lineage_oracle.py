"""Independent host reconstruction of hidden-regime generation lineage.

This module consumes only primitive trace arrays and the static development
configuration.  It does not call the evaluator's lineage, recurrence, or
retention reconstruction helpers and never replays a learner or world.  The
result is compared field-by-field with the serialized run summary so derived
lineage claims cannot be made true by changing summary bytes or a digest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from alberta_framework.core.slot_signaling_agent import (
    N_SLOTS,
    SCRATCH_SLOT,
    SLOT_DURABLE,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    CommitGenerationLineage,
    DormantGenerationProbe,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeRunSummary,
    RecurrenceLineageProbe,
    RecurrenceRetentionRecord,
    RetentionAggregateSummary,
)
from alberta_framework.streams.hidden_regime_signaling import HiddenRegimeWorldConfig

HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA = "alberta.hidden-regime.lineage-oracle.v2"

type EventPhase = Literal["pre", "post"]
type EntryActivity = Literal["active", "dormant", "mixed", "unavailable"]

_REGIME_LABELS = {
    0: "A",
    1: "B",
    2: "C-old",
    3: "C-new",
    4: "D-short",
}


@dataclasses.dataclass(frozen=True)
class HiddenRegimeLineageExpectation:
    """Complete independently reconstructed lineage-bearing summary fields."""

    commit_generation_lineages: tuple[CommitGenerationLineage, ...]
    synchronized_commit_lineage_count: int
    acquisition_qualified_commit_lineage_count: int
    acquisition_unqualified_commit_lineage_count: int
    recurrence_retention: tuple[RecurrenceRetentionRecord, ...]
    retention: RetentionAggregateSummary


@dataclasses.dataclass(frozen=True)
class HiddenRegimeLineageValidation:
    """Field-addressable result of comparing reconstruction with a run summary."""

    valid: bool
    mismatches: tuple[str, ...]
    commit_lineages_checked: int
    recurrence_records_checked: int
    aggregate_fields_checked: int
    expected: HiddenRegimeLineageExpectation | None
    schema: str = HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA


@dataclasses.dataclass(frozen=True)
class _SelectedRelock:
    selected_exact_generation_relock_observed: bool | None
    selected_first_exact_generation_relock_step: int | None
    selected_first_exact_generation_relock_segment_step: int | None
    selected_exact_generation_relock_phase: EventPhase | None
    selected_observed_learner_boundaries_until_relock: int | None
    selected_first_scratch_entry_step: int | None
    selected_first_scratch_entry_segment_step: int | None
    selected_first_scratch_entry_phase: EventPhase | None
    selected_scratch_entered_before_relock: bool | None
    selected_scratch_entered_before_relock_or_segment_end: bool | None
    selected_durable_retrieval_before_scratch: bool | None


def _label(regime: int) -> str:
    return _REGIME_LABELS.get(regime, f"regime-{regime}")


def _float_bank(bits: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _table_bits(bits: object) -> tuple[int, ...]:
    table = np.asarray(bits, dtype=np.uint32)
    if table.shape != (3, 3):
        raise ValueError("lineage table must have exact shape (3, 3)")
    return tuple(int(word) for word in table.reshape(-1))


def _mapping(
    helper: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[int, int, int]:
    if helper.shape != (3, 3) or beneficiary.shape != (3, 3):
        raise ValueError("lineage value tables must have exact shape (3, 3)")
    messages = np.argmax(helper, axis=1)
    actions = np.argmax(beneficiary[messages], axis=1)
    return cast(tuple[int, int, int], tuple(int(action) for action in actions))


def _accuracy(
    helper: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary: np.ndarray[Any, np.dtype[np.float32]],
    target: np.ndarray[Any, np.dtype[np.int32]],
) -> float:
    actual = np.asarray(_mapping(helper, beneficiary), dtype=np.int32)
    return float(np.mean(actual == target, dtype=np.float64))


def _tie_free(
    helper: np.ndarray[Any, np.dtype[np.float32]],
    beneficiary: np.ndarray[Any, np.dtype[np.float32]],
) -> bool:
    """Independently require a unique maximum at every composed decision."""

    if helper.shape != (3, 3) or beneficiary.shape != (3, 3):
        raise ValueError("lineage value tables must have exact shape (3, 3)")
    helper_maxima = np.max(helper, axis=1, keepdims=True)
    helper_unique = np.count_nonzero(helper == helper_maxima, axis=1) == 1
    messages = np.argmax(helper, axis=1)
    selected_beneficiary = beneficiary[messages]
    beneficiary_maxima = np.max(selected_beneficiary, axis=1, keepdims=True)
    beneficiary_unique = (
        np.count_nonzero(selected_beneficiary == beneficiary_maxima, axis=1) == 1
    )
    return bool(np.all(helper_unique) and np.all(beneficiary_unique))


def independent_commit_generation_lineages(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[CommitGenerationLineage, ...]:
    """Reconstruct every synchronized commit from primitive post-state words."""

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
        composed = _mapping(helper_table, beneficiary_table)
        accuracy = float(np.mean(np.asarray(composed, dtype=np.int32) == target, dtype=np.float64))
        tie_free = _tie_free(helper_table, beneficiary_table)
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
                committed_composed_greedy_mapping=composed,
                committed_composed_greedy_accuracy=accuracy,
                committed_composed_greedy_tie_free=tie_free,
                acquisition_qualified=accuracy == 1.0 and tie_free,
                helper_table_uint32_bits=_table_bits(helper_table_bits),
                beneficiary_table_uint32_bits=_table_bits(beneficiary_table_bits),
            )
        )
    return tuple(lineages)


def independent_lineage_recurrence_segments(
    world: HiddenRegimeWorldConfig,
) -> tuple[tuple[int, int, int], ...]:
    """Return genuine recurrence segment, regime, and coalesced occurrence."""

    episode_counts: dict[int, int] = {}
    result: list[tuple[int, int, int]] = []
    for segment, regime in enumerate(world.segment_regimes):
        if segment and world.segment_regimes[segment - 1] == regime:
            continue
        occurrence = episode_counts.get(regime, 0)
        if occurrence:
            result.append((segment, regime, occurrence))
        episode_counts[regime] = occurrence + 1
    return tuple(result)


def independent_coalesced_episode_bounds(
    world: HiddenRegimeWorldConfig,
    segment_index: int,
) -> tuple[int, int, int, int]:
    """Return containing equal-regime episode bounds without producer helpers."""

    if type(segment_index) is not int or not 0 <= segment_index < len(world.segment_regimes):
        raise ValueError("segment_index is outside the world schedule")
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


def _entry_window(
    rewards: np.ndarray[Any, np.dtype[np.float32]],
    start: int,
    segment_steps: int,
    window: int,
) -> tuple[bool, float | None, int | None, float | None]:
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


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values, dtype=np.float64))


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _base_recurrence_records(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[RecurrenceRetentionRecord, ...]:
    """Reconstruct legacy first-exposure and best-dormant fields."""

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
    helper_relevant = np.asarray(trace.helper_durable_relevant, dtype=np.bool_)
    beneficiary_relevant = np.asarray(
        trace.beneficiary_durable_relevant,
        dtype=np.bool_,
    )
    helper_values = _float_bank(trace.helper_value_bits_pre)
    beneficiary_values = _float_bank(trace.beneficiary_value_bits_pre)
    permutations = np.asarray(config.world.regime_permutations, dtype=np.int32)
    lease_length = config.learner.lease_length
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(np.asarray(config.world.segment_lengths[:-1]), dtype=np.int64),
        )
    )
    first_by_regime: dict[int, tuple[int, int, int]] = {}
    occurrence_count: dict[int, int] = {}
    records: list[RecurrenceRetentionRecord] = []
    for segment, (start_value, length, regime) in enumerate(
        zip(
            starts,
            config.world.segment_lengths,
            config.world.segment_regimes,
            strict=True,
        )
    ):
        start = int(start_value)
        end = start + length
        occurrence = occurrence_count.get(regime, 0)
        current_window = _entry_window(rewards, start, length, lease_length)
        if occurrence == 0:
            first_by_regime[regime] = (segment, start, length)
            occurrence_count[regime] = 1
            continue
        acquisition_segment, acquisition_start, acquisition_length = first_by_regime[regime]
        acquisition_window = _entry_window(
            rewards,
            acquisition_start,
            acquisition_length,
            lease_length,
        )
        current_complete, current_reward, current_errors, current_error_rate = current_window
        (
            acquisition_complete,
            acquisition_reward,
            acquisition_errors,
            acquisition_error_rate,
        ) = acquisition_window
        comparison = current_complete and acquisition_complete
        error_delta = (
            None
            if not comparison
            else cast(float, current_error_rate) - cast(float, acquisition_error_rate)
        )
        ratio_defined = bool(comparison and cast(float, acquisition_error_rate) > 0.0)
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
        dormant: list[DormantGenerationProbe] = []
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
            helper_table = helper_values[start, slot]
            beneficiary_table = beneficiary_values[start, slot]
            target = permutations[regime]
            dormant.append(
                DormantGenerationProbe(
                    slot=slot,
                    generation=generation,
                    composed_greedy_accuracy=_accuracy(
                        helper_table,
                        beneficiary_table,
                        target,
                    ),
                    zero_helper_accuracy=_accuracy(
                        np.zeros_like(helper_table),
                        beneficiary_table,
                        target,
                    ),
                    zero_beneficiary_accuracy=_accuracy(
                        helper_table,
                        np.zeros_like(beneficiary_table),
                        target,
                    ),
                    role_swapped_accuracy=_accuracy(
                        beneficiary_table,
                        helper_table,
                        target,
                    ),
                )
            )
        best = (
            None
            if not dormant
            else min(
                dormant,
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
                    break
        scan_end = end if relock_step is None else relock_step + 1
        scratch_in_prefix = bool(
            np.any(helper_active_pre[start:scan_end] == SCRATCH_SLOT)
            or np.any(beneficiary_active_pre[start:scan_end] == SCRATCH_SLOT)
        )
        boundaries_until_relock = (
            None
            if relock_step is None
            else int(np.count_nonzero(boundaries[start : relock_step + 1]))
        )
        entry_offset = int(np.asarray(trace.helper_lease_offset_pre)[start])
        records.append(
            RecurrenceRetentionRecord(
                segment_index=segment,
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
                eligible_dormant_generations=tuple(dormant),
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
    return tuple(records)


def _probe_lineage_at_entry(
    lineage: CommitGenerationLineage,
    *,
    start: int,
    trace: HiddenRegimePrimitiveTrace,
) -> RecurrenceLineageProbe:
    slot = lineage.slot
    helper_status = int(np.asarray(trace.helper_status_pre)[start, slot])
    helper_generation = int(np.asarray(trace.helper_generation_pre)[start, slot])
    beneficiary_status = int(np.asarray(trace.beneficiary_status_pre)[start, slot])
    beneficiary_generation = int(np.asarray(trace.beneficiary_generation_pre)[start, slot])
    helper_present = helper_status == SLOT_DURABLE and helper_generation == lineage.generation
    beneficiary_present = (
        beneficiary_status == SLOT_DURABLE and beneficiary_generation == lineage.generation
    )
    survives = helper_present and beneficiary_present
    helper_active = helper_present and int(np.asarray(trace.helper_active_slot_pre)[start]) == slot
    beneficiary_active = (
        beneficiary_present and int(np.asarray(trace.beneficiary_active_slot_pre)[start]) == slot
    )
    if not survives:
        activity: EntryActivity = "unavailable"
    elif helper_active and beneficiary_active:
        activity = "active"
    elif not helper_active and not beneficiary_active:
        activity = "dormant"
    else:
        activity = "mixed"

    helper_array = np.asarray(trace.helper_value_bits_pre, dtype=np.uint32)[start, slot]
    beneficiary_array = np.asarray(
        trace.beneficiary_value_bits_pre,
        dtype=np.uint32,
    )[start, slot]
    helper_entry_bits = _table_bits(helper_array) if helper_present else None
    beneficiary_entry_bits = _table_bits(beneficiary_array) if beneficiary_present else None
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
        helper_table = helper_array.view(np.float32)
        beneficiary_table = beneficiary_array.view(np.float32)
        target = np.asarray(lineage.target_mapping, dtype=np.int32)
        mapping = _mapping(helper_table, beneficiary_table)
        accuracy = float(np.mean(np.asarray(mapping, dtype=np.int32) == target, dtype=np.float64))
        accuracy_change = accuracy - lineage.committed_composed_greedy_accuracy
        zero_helper = _accuracy(np.zeros_like(helper_table), beneficiary_table, target)
        zero_beneficiary = _accuracy(helper_table, np.zeros_like(beneficiary_table), target)
        role_swapped = _accuracy(beneficiary_table, helper_table, target)
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


def _selected_relock(
    selected: RecurrenceLineageProbe | None,
    *,
    start: int,
    end: int,
    trace: HiddenRegimePrimitiveTrace,
) -> _SelectedRelock:
    if selected is None:
        return _SelectedRelock(
            selected_exact_generation_relock_observed=None,
            selected_first_exact_generation_relock_step=None,
            selected_first_exact_generation_relock_segment_step=None,
            selected_exact_generation_relock_phase=None,
            selected_observed_learner_boundaries_until_relock=None,
            selected_first_scratch_entry_step=None,
            selected_first_scratch_entry_segment_step=None,
            selected_first_scratch_entry_phase=None,
            selected_scratch_entered_before_relock=None,
            selected_scratch_entered_before_relock_or_segment_end=None,
            selected_durable_retrieval_before_scratch=None,
        )
    immediate = selected.entry_activity_status == "active"
    slot = selected.slot
    generation = selected.generation
    helper_boundary = np.asarray(trace.helper_lease_boundary, dtype=np.bool_)
    beneficiary_boundary = np.asarray(trace.beneficiary_lease_boundary, dtype=np.bool_)
    boundaries = np.logical_and(helper_boundary, beneficiary_boundary)
    boundary_steps = np.flatnonzero(boundaries[start:end]) + start
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
    helper_relevant = np.asarray(trace.helper_durable_relevant, dtype=np.bool_)
    beneficiary_relevant = np.asarray(
        trace.beneficiary_durable_relevant,
        dtype=np.bool_,
    )
    relock_step: int | None = start if immediate else None
    relock_phase: EventPhase | None = "pre" if immediate else None
    if not immediate:
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
    first_scratch_phase: EventPhase | None = None
    for step in range(start, end):
        pre_scratch = (
            helper_active_pre[step] == SCRATCH_SLOT or beneficiary_active_pre[step] == SCRATCH_SLOT
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
    scratch_before = scratch_event is not None and (
        relock_event is None or scratch_event < relock_event
    )
    boundaries_until = (
        None
        if relock_step is None
        else 0
        if immediate
        else int(np.count_nonzero(boundaries[start : relock_step + 1]))
    )
    return _SelectedRelock(
        selected_exact_generation_relock_observed=relock_step is not None,
        selected_first_exact_generation_relock_step=relock_step,
        selected_first_exact_generation_relock_segment_step=(
            None if relock_step is None else relock_step - start
        ),
        selected_exact_generation_relock_phase=relock_phase,
        selected_observed_learner_boundaries_until_relock=boundaries_until,
        selected_first_scratch_entry_step=first_scratch_step,
        selected_first_scratch_entry_segment_step=(
            None if first_scratch_step is None else first_scratch_step - start
        ),
        selected_first_scratch_entry_phase=first_scratch_phase,
        selected_scratch_entered_before_relock=scratch_before,
        selected_scratch_entered_before_relock_or_segment_end=first_scratch_step is not None,
        selected_durable_retrieval_before_scratch=(
            relock_event is not None and (scratch_event is None or relock_event < scratch_event)
        ),
    )


def _enriched_recurrence_records(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
    lineages: tuple[CommitGenerationLineage, ...],
) -> tuple[RecurrenceRetentionRecord, ...]:
    base = {record.segment_index: record for record in _base_recurrence_records(trace, config)}
    recurrences = independent_lineage_recurrence_segments(config.world)
    records: list[RecurrenceRetentionRecord] = []
    for segment, regime, coalesced_occurrence in recurrences:
        if segment not in base:
            raise ValueError("coalesced recurrence lacks its raw recurrence record")
        raw = base[segment]
        _, end_segment, start, episode_length = independent_coalesced_episode_bounds(
            config.world,
            segment,
        )
        if end_segment <= segment:
            raise ValueError("coalesced recurrence episode bounds are invalid")
        end = start + episode_length
        current_window = _entry_window(
            np.asarray(trace.reward, dtype=np.float32),
            start,
            episode_length,
            config.learner.lease_length,
        )
        prior = tuple(
            lineage
            for lineage in lineages
            if lineage.regime_id == regime and lineage.commit_step < start
        )
        probes = tuple(
            _probe_lineage_at_entry(lineage, start=start, trace=trace) for lineage in prior
        )
        qualified = tuple(probe for probe in probes if probe.acquisition_qualified)
        surviving = tuple(probe for probe in qualified if probe.synchronized_generation_survives)
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
            ) = independent_coalesced_episode_bounds(
                config.world,
                latest.commit_segment_index,
            )
            latest_window = _entry_window(
                np.asarray(trace.reward, dtype=np.float32),
                latest_acquisition_start,
                latest_acquisition_episode_length,
                config.learner.lease_length,
            )
            (
                latest_acquisition_complete,
                latest_acquisition_reward,
                latest_acquisition_errors,
                latest_acquisition_error_rate,
            ) = latest_window
            latest_comparison_available = current_window[0] and latest_acquisition_complete
            if latest_comparison_available:
                current_error_rate = cast(float, current_window[3])
                acquisition_error_rate = cast(float, latest_acquisition_error_rate)
                latest_error_delta = current_error_rate - acquisition_error_rate
                latest_ratio_defined = acquisition_error_rate > 0.0
                if latest_ratio_defined:
                    latest_error_ratio = current_error_rate / acquisition_error_rate
        relock = _selected_relock(selected, start=start, end=end, trace=trace)
        records.append(
            dataclasses.replace(
                raw,
                occurrence_index=coalesced_occurrence,
                raw_segment_occurrence_index=raw.occurrence_index,
                first_world_window_complete=current_window[0],
                first_world_window_reward=current_window[1],
                first_world_window_errors=current_window[2],
                first_world_window_error_rate=current_window[3],
                latest_qualified_acquisition_segment_index=(latest_acquisition_segment),
                latest_qualified_acquisition_episode_length=(latest_acquisition_episode_length),
                latest_qualified_acquisition_world_window_complete=(latest_acquisition_complete),
                latest_qualified_acquisition_world_window_reward=(latest_acquisition_reward),
                latest_qualified_acquisition_world_window_errors=(latest_acquisition_errors),
                latest_qualified_acquisition_world_window_error_rate=(
                    latest_acquisition_error_rate
                ),
                latest_qualified_acquisition_comparison_available=(latest_comparison_available),
                recurrence_minus_latest_qualified_acquisition_error_rate_delta=(latest_error_delta),
                recurrence_to_latest_qualified_acquisition_error_rate_ratio=(latest_error_ratio),
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
                latest_prior_qualified_commit_step=(None if latest is None else latest.commit_step),
                latest_prior_qualified_survived=(
                    None if latest is None else latest.synchronized_generation_survives
                ),
                any_prior_qualified_survived=(None if not qualified else bool(surviving)),
                surviving_qualified_lineage_count=len(surviving),
                selected_lineage_available=selected is not None,
                selected_lineage_index=(None if selected is None else selected.lineage_index),
                selected_lineage_commit_step=(None if selected is None else selected.commit_step),
                selected_lineage_slot=None if selected is None else selected.slot,
                selected_lineage_generation=(None if selected is None else selected.generation),
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
                selected_exact_generation_relock_observed=(
                    relock.selected_exact_generation_relock_observed
                ),
                selected_first_exact_generation_relock_step=(
                    relock.selected_first_exact_generation_relock_step
                ),
                selected_first_exact_generation_relock_segment_step=(
                    relock.selected_first_exact_generation_relock_segment_step
                ),
                selected_exact_generation_relock_phase=(
                    relock.selected_exact_generation_relock_phase
                ),
                selected_observed_learner_boundaries_until_relock=(
                    relock.selected_observed_learner_boundaries_until_relock
                ),
                selected_first_scratch_entry_step=(relock.selected_first_scratch_entry_step),
                selected_first_scratch_entry_segment_step=(
                    relock.selected_first_scratch_entry_segment_step
                ),
                selected_first_scratch_entry_phase=(relock.selected_first_scratch_entry_phase),
                selected_scratch_entered_before_relock=(
                    relock.selected_scratch_entered_before_relock
                ),
                selected_scratch_entered_before_relock_or_segment_end=(
                    relock.selected_scratch_entered_before_relock_or_segment_end
                ),
                selected_durable_retrieval_before_scratch=(
                    relock.selected_durable_retrieval_before_scratch
                ),
            )
        )
    return tuple(records)


def _aggregate_retention(
    records: tuple[RecurrenceRetentionRecord, ...],
) -> RetentionAggregateSummary:
    """Aggregate every lineage and legacy retention denominator independently."""

    recurrence_count = len(records)
    complete = tuple(record for record in records if record.first_world_window_complete)
    applicable = tuple(record for record in records if record.lineage_retention_applicable)
    latest_comparisons = tuple(
        record for record in applicable if record.latest_qualified_acquisition_comparison_available
    )
    latest_ratio_defined = tuple(
        record
        for record in latest_comparisons
        if record.recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined is True
    )
    selected = tuple(record for record in records if record.selected_lineage_available)
    selected_relocked = tuple(
        record for record in selected if record.selected_exact_generation_relock_observed is True
    )
    dormant = tuple(record for record in records if record.dormant_probe_available)
    dormant_relocked = tuple(
        record for record in records if record.exact_generation_relock_observed
    )
    dormant_retrieved = tuple(
        record for record in records if record.durable_retrieval_before_scratch
    )
    legacy_comparisons = tuple(
        record
        for record in records
        if record.legacy_recurrence_minus_first_exposure_error_rate_delta is not None
    )
    legacy_ratio_defined = tuple(
        record
        for record in legacy_comparisons
        if record.legacy_recurrence_to_first_exposure_error_rate_ratio_defined
    )

    prior_count = sum(record.prior_same_regime_lineage_count for record in records)
    qualified_count = sum(record.prior_qualified_lineage_count for record in records)
    surviving_count = sum(record.surviving_qualified_lineage_count for record in records)
    latest_survival_count = sum(
        record.latest_prior_qualified_survived is True for record in applicable
    )
    any_survival_count = sum(record.any_prior_qualified_survived is True for record in applicable)
    selected_count = len(selected)
    selected_helper_exact = sum(
        record.selected_lineage_helper_bit_exact_preserved is True for record in selected
    )
    selected_beneficiary_exact = sum(
        record.selected_lineage_beneficiary_bit_exact_preserved is True for record in selected
    )
    selected_joint_exact = sum(
        record.selected_lineage_joint_bit_exact_preserved is True for record in selected
    )
    selected_retrieved = sum(
        record.selected_durable_retrieval_before_scratch is True for record in selected
    )

    return RetentionAggregateSummary(
        recurrence_count=recurrence_count,
        complete_first_world_window_count=len(complete),
        missing_first_world_window_count=recurrence_count - len(complete),
        first_world_window_reward_mean=_mean(
            [cast(float, record.first_world_window_reward) for record in complete]
        ),
        first_world_window_error_rate_mean=_mean(
            [cast(float, record.first_world_window_error_rate) for record in complete]
        ),
        lineage_retention_applicable_count=len(applicable),
        acquisition_coverage_failure_count=recurrence_count - len(applicable),
        qualification_coverage_denominator=recurrence_count,
        qualification_coverage_fraction=_fraction(len(applicable), recurrence_count),
        prior_same_regime_lineage_count=prior_count,
        prior_qualified_lineage_count=qualified_count,
        prior_unqualified_lineage_count=prior_count - qualified_count,
        surviving_qualified_lineage_count=surviving_count,
        qualified_lineage_survival_denominator=qualified_count,
        qualified_lineage_survival_fraction=_fraction(surviving_count, qualified_count),
        latest_qualified_version_survival_count=latest_survival_count,
        latest_qualified_version_survival_denominator=len(applicable),
        latest_qualified_version_survival_missing_count=(recurrence_count - len(applicable)),
        latest_qualified_version_survival_fraction=_fraction(
            latest_survival_count,
            len(applicable),
        ),
        any_qualified_knowledge_survival_count=any_survival_count,
        any_qualified_knowledge_survival_denominator=len(applicable),
        any_qualified_knowledge_survival_missing_count=(recurrence_count - len(applicable)),
        any_qualified_knowledge_survival_fraction=_fraction(
            any_survival_count,
            len(applicable),
        ),
        selected_lineage_probe_available_count=selected_count,
        selected_lineage_probe_denominator=len(applicable),
        selected_lineage_survival_failure_count=len(applicable) - selected_count,
        selected_lineage_not_applicable_count=recurrence_count - len(applicable),
        selected_lineage_survival_fraction_given_qualified_prior=_fraction(
            selected_count,
            len(applicable),
        ),
        selected_entry_metric_denominator=selected_count,
        selected_entry_active_count=sum(
            record.selected_lineage_entry_activity_status == "active" for record in selected
        ),
        selected_entry_dormant_count=sum(
            record.selected_lineage_entry_activity_status == "dormant" for record in selected
        ),
        selected_entry_mixed_count=sum(
            record.selected_lineage_entry_activity_status == "mixed" for record in selected
        ),
        selected_entry_composed_greedy_accuracy_mean=_mean(
            [
                cast(float, record.selected_lineage_entry_composed_greedy_accuracy)
                for record in selected
            ]
        ),
        selected_entry_minus_commit_accuracy_mean=_mean(
            [
                cast(float, record.selected_lineage_entry_minus_commit_accuracy)
                for record in selected
            ]
        ),
        selected_helper_bit_exact_preservation_count=selected_helper_exact,
        selected_beneficiary_bit_exact_preservation_count=selected_beneficiary_exact,
        selected_joint_bit_exact_preservation_count=selected_joint_exact,
        selected_bit_exact_preservation_conditional_denominator=selected_count,
        selected_bit_exact_preservation_all_qualified_denominator=len(applicable),
        selected_helper_bit_exact_preservation_fraction=_fraction(
            selected_helper_exact,
            selected_count,
        ),
        selected_beneficiary_bit_exact_preservation_fraction=_fraction(
            selected_beneficiary_exact,
            selected_count,
        ),
        selected_joint_bit_exact_preservation_fraction=_fraction(
            selected_joint_exact,
            selected_count,
        ),
        selected_helper_bit_exact_preservation_fraction_all_qualified=_fraction(
            selected_helper_exact,
            len(applicable),
        ),
        selected_beneficiary_bit_exact_preservation_fraction_all_qualified=_fraction(
            selected_beneficiary_exact,
            len(applicable),
        ),
        selected_joint_bit_exact_preservation_fraction_all_qualified=_fraction(
            selected_joint_exact,
            len(applicable),
        ),
        selected_zero_helper_accuracy_mean=_mean(
            [cast(float, record.selected_lineage_zero_helper_accuracy) for record in selected]
        ),
        selected_zero_beneficiary_accuracy_mean=_mean(
            [cast(float, record.selected_lineage_zero_beneficiary_accuracy) for record in selected]
        ),
        selected_role_swapped_accuracy_mean=_mean(
            [cast(float, record.selected_lineage_role_swapped_accuracy) for record in selected]
        ),
        selected_exact_generation_relock_count=len(selected_relocked),
        selected_exact_generation_relock_conditional_denominator=selected_count,
        selected_exact_generation_relock_all_qualified_denominator=len(applicable),
        selected_exact_generation_relock_fraction_given_selected_lineage=_fraction(
            len(selected_relocked),
            selected_count,
        ),
        selected_exact_generation_relock_fraction_all_qualified=_fraction(
            len(selected_relocked),
            len(applicable),
        ),
        selected_observed_learner_boundaries_to_relock_mean=_mean(
            [
                float(cast(int, record.selected_observed_learner_boundaries_until_relock))
                for record in selected_relocked
            ]
        ),
        selected_observed_learner_boundaries_to_relock_available_count=len(selected_relocked),
        selected_observed_learner_boundaries_to_relock_unavailable_count=(
            selected_count - len(selected_relocked)
        ),
        selected_durable_retrieval_before_scratch_count=selected_retrieved,
        selected_durable_retrieval_before_scratch_conditional_denominator=selected_count,
        selected_durable_retrieval_before_scratch_all_qualified_denominator=len(applicable),
        selected_durable_retrieval_before_scratch_fraction_given_selected_lineage=(
            _fraction(selected_retrieved, selected_count)
        ),
        selected_durable_retrieval_before_scratch_fraction_all_qualified=_fraction(
            selected_retrieved,
            len(applicable),
        ),
        dormant_probe_available_count=len(dormant),
        dormant_probe_missing_count=recurrence_count - len(dormant),
        dormant_composed_greedy_accuracy_mean=_mean(
            [cast(float, record.best_dormant_composed_greedy_accuracy) for record in dormant]
        ),
        dormant_zero_helper_accuracy_mean=_mean(
            [cast(float, record.best_dormant_zero_helper_accuracy) for record in dormant]
        ),
        dormant_zero_beneficiary_accuracy_mean=_mean(
            [cast(float, record.best_dormant_zero_beneficiary_accuracy) for record in dormant]
        ),
        dormant_role_swapped_accuracy_mean=_mean(
            [cast(float, record.best_dormant_role_swapped_accuracy) for record in dormant]
        ),
        exact_generation_relock_count=len(dormant_relocked),
        exact_generation_relock_missing_count=(recurrence_count - len(dormant_relocked)),
        exact_generation_relock_rate_all_recurrences=_fraction(
            len(dormant_relocked),
            recurrence_count,
        ),
        exact_generation_relock_rate_given_dormant_probe=_fraction(
            len(dormant_relocked),
            len(dormant),
        ),
        observed_learner_boundaries_to_relock_mean=_mean(
            [
                float(cast(int, record.observed_learner_boundaries_until_relock))
                for record in dormant_relocked
            ]
        ),
        observed_learner_boundaries_to_relock_missing_count=(
            recurrence_count - len(dormant_relocked)
        ),
        durable_retrieval_before_scratch_count=len(dormant_retrieved),
        durable_retrieval_before_scratch_fraction_all_recurrences=_fraction(
            len(dormant_retrieved),
            recurrence_count,
        ),
        durable_retrieval_before_scratch_fraction_given_dormant_probe=_fraction(
            len(dormant_retrieved),
            len(dormant),
        ),
        latest_qualified_acquisition_comparison_available_count=len(latest_comparisons),
        latest_qualified_acquisition_comparison_denominator=len(applicable),
        latest_qualified_acquisition_comparison_missing_count=(
            len(applicable) - len(latest_comparisons)
        ),
        latest_qualified_acquisition_comparison_not_applicable_count=(
            recurrence_count - len(applicable)
        ),
        recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean=_mean(
            [
                cast(
                    float,
                    record.recurrence_minus_latest_qualified_acquisition_error_rate_delta,
                )
                for record in latest_comparisons
            ]
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_mean=_mean(
            [
                cast(
                    float,
                    record.recurrence_to_latest_qualified_acquisition_error_rate_ratio,
                )
                for record in latest_ratio_defined
            ]
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined_count=len(
            latest_ratio_defined
        ),
        recurrence_to_latest_qualified_acquisition_error_rate_ratio_undefined_count=(
            len(latest_comparisons) - len(latest_ratio_defined)
        ),
        legacy_first_exposure_comparison_available_count=len(legacy_comparisons),
        legacy_first_exposure_comparison_missing_count=(recurrence_count - len(legacy_comparisons)),
        legacy_recurrence_minus_first_exposure_error_rate_delta_mean=_mean(
            [
                cast(
                    float,
                    record.legacy_recurrence_minus_first_exposure_error_rate_delta,
                )
                for record in legacy_comparisons
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_mean=_mean(
            [
                cast(
                    float,
                    record.legacy_recurrence_to_first_exposure_error_rate_ratio,
                )
                for record in legacy_ratio_defined
            ]
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_defined_count=len(
            legacy_ratio_defined
        ),
        legacy_recurrence_to_first_exposure_error_rate_ratio_undefined_count=(
            len(legacy_comparisons) - len(legacy_ratio_defined)
        ),
    )


def reconstruct_hidden_regime_lineage_expectation(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> HiddenRegimeLineageExpectation:
    """Reconstruct every lineage-bearing run-summary field from primitives."""

    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        raise TypeError("trace must be a HiddenRegimePrimitiveTrace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        raise TypeError("config must be a HiddenRegimeDevelopmentConfig")
    lineages = independent_commit_generation_lineages(trace, config)
    records = _enriched_recurrence_records(trace, config, lineages)
    qualified_count = sum(lineage.acquisition_qualified for lineage in lineages)
    return HiddenRegimeLineageExpectation(
        commit_generation_lineages=lineages,
        synchronized_commit_lineage_count=len(lineages),
        acquisition_qualified_commit_lineage_count=qualified_count,
        acquisition_unqualified_commit_lineage_count=len(lineages) - qualified_count,
        recurrence_retention=records,
        retention=_aggregate_retention(records),
    )


def _float64_bits(value: float) -> int:
    scalar = np.asarray(value, dtype=np.float64)
    return int(scalar.view(np.uint64))


def _compare(
    expected: object,
    actual: object,
    path: str,
    mismatches: list[str],
) -> None:
    """Compare nested dataclass state with exact Python types and float bits."""

    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        if type(actual) is not type(expected):
            mismatches.append(f"{path}.type")
            return
        for field in dataclasses.fields(expected):
            _compare(
                getattr(expected, field.name),
                getattr(actual, field.name),
                f"{path}.{field.name}",
                mismatches,
            )
        return
    if isinstance(expected, tuple):
        if type(actual) is not tuple:
            mismatches.append(f"{path}.type")
            return
        if len(actual) != len(expected):
            mismatches.append(f"{path}.length")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=False)):
            _compare(
                expected_item,
                actual_item,
                f"{path}[{index}]",
                mismatches,
            )
        return
    if expected is None:
        if actual is not None:
            mismatches.append(path)
        return
    if type(actual) is not type(expected):
        mismatches.append(f"{path}.type")
        return
    if isinstance(expected, float):
        actual_float = cast(float, actual)
        if not np.isfinite(expected) or not np.isfinite(actual_float):
            if not (np.isnan(expected) and np.isnan(actual_float)):
                mismatches.append(path)
        elif _float64_bits(actual_float) != _float64_bits(expected):
            mismatches.append(path)
        return
    if actual != expected:
        mismatches.append(path)


def validate_hidden_regime_lineage_summary(
    trace: object,
    config: object,
    summary: object,
) -> HiddenRegimeLineageValidation:
    """Fail closed unless all independently derived lineage fields agree."""

    mismatches: list[str] = []
    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        mismatches.append("input.trace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        mismatches.append("input.config")
    if not isinstance(summary, HiddenRegimeRunSummary):
        mismatches.append("input.summary")
    if mismatches:
        return HiddenRegimeLineageValidation(
            valid=False,
            mismatches=tuple(mismatches),
            commit_lineages_checked=0,
            recurrence_records_checked=0,
            aggregate_fields_checked=0,
            expected=None,
        )

    typed_trace = cast(HiddenRegimePrimitiveTrace, trace)
    typed_config = cast(HiddenRegimeDevelopmentConfig, config)
    typed_summary = cast(HiddenRegimeRunSummary, summary)
    try:
        expected = reconstruct_hidden_regime_lineage_expectation(
            typed_trace,
            typed_config,
        )
    except (IndexError, TypeError, ValueError) as error:
        return HiddenRegimeLineageValidation(
            valid=False,
            mismatches=(f"reconstruction: {error}",),
            commit_lineages_checked=0,
            recurrence_records_checked=0,
            aggregate_fields_checked=0,
            expected=None,
        )

    _compare(
        expected.commit_generation_lineages,
        typed_summary.commit_generation_lineages,
        "summary.commit_generation_lineages",
        mismatches,
    )
    _compare(
        expected.synchronized_commit_lineage_count,
        typed_summary.synchronized_commit_lineage_count,
        "summary.synchronized_commit_lineage_count",
        mismatches,
    )
    _compare(
        expected.acquisition_qualified_commit_lineage_count,
        typed_summary.acquisition_qualified_commit_lineage_count,
        "summary.acquisition_qualified_commit_lineage_count",
        mismatches,
    )
    _compare(
        expected.acquisition_unqualified_commit_lineage_count,
        typed_summary.acquisition_unqualified_commit_lineage_count,
        "summary.acquisition_unqualified_commit_lineage_count",
        mismatches,
    )
    _compare(
        expected.recurrence_retention,
        typed_summary.recurrence_retention,
        "summary.recurrence_retention",
        mismatches,
    )
    _compare(expected.retention, typed_summary.retention, "summary.retention", mismatches)
    return HiddenRegimeLineageValidation(
        valid=not mismatches,
        mismatches=tuple(mismatches),
        commit_lineages_checked=len(expected.commit_generation_lineages),
        recurrence_records_checked=len(expected.recurrence_retention),
        aggregate_fields_checked=len(dataclasses.fields(RetentionAggregateSummary)),
        expected=expected,
    )


__all__ = [
    "HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA",
    "HiddenRegimeLineageExpectation",
    "HiddenRegimeLineageValidation",
    "independent_coalesced_episode_bounds",
    "independent_commit_generation_lineages",
    "independent_lineage_recurrence_segments",
    "reconstruct_hidden_regime_lineage_expectation",
    "validate_hidden_regime_lineage_summary",
]
