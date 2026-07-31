"""Independent reconstruction of every hidden-regime summary and resource field.

The development evaluator reports descriptive summaries in addition to its
primitive trace.  A transition audit alone cannot make those derived fields
trustworthy: summary bytes can be changed without changing a legal trace.  This
module therefore reconstructs the complete run summary and resource report
from primitive arrays, the static configuration, and the audited final learner
state without calling any producer summary or resource helper.

Lineage-bearing fields are supplied by the separate independent lineage oracle.
All remaining fields are reconstructed here with audit-owned condition
bindings, segment logic, replacement provenance, and exact float64 comparison.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import jax.random as jr
import numpy as np

from alberta_framework.core.slot_signaling_agent import SlotRoleState, SlotSignalingState
from alberta_framework.evaluation.hidden_regime_lineage_oracle import (
    reconstruct_hidden_regime_lineage_expectation,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    HiddenRegimeDevelopmentCondition,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeResourceReport,
    HiddenRegimeRunSummary,
    RegimeRecurrenceSummary,
    SegmentRewardSummary,
)

HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA = "alberta.hidden-regime.summary-oracle.v1"

_REGIME_LABELS = {
    0: "A",
    1: "B",
    2: "C-old",
    3: "C-new",
    4: "D-short",
}
_KNOWN_CONDITIONS = frozenset(
    {
        "selective_full",
        "writable_evidence",
        "selective_lru",
        "writable_lru",
        "helper_frozen",
        "beneficiary_frozen",
        "constant_channel_0",
        "shuffled_channel",
    }
)
_WRITABLE_CONDITIONS = frozenset({"writable_evidence", "writable_lru"})

_ROLE_ARRAY_FIELDS = (
    "values",
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
    "candidate_successful_leases",
    "next_generation",
)
_TRACE_ROLE_STATE_FIELDS = (
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
)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeSummaryExpectation:
    """Complete independent expectation for one run's two derived records."""

    summary: HiddenRegimeRunSummary
    resource: HiddenRegimeResourceReport


@dataclasses.dataclass(frozen=True)
class HiddenRegimeSummaryValidation:
    """Field-addressable complete summary/resource comparison."""

    valid: bool
    mismatches: tuple[str, ...]
    summary_fields_checked: int
    resource_fields_checked: int
    expected: HiddenRegimeSummaryExpectation | None
    schema: str = HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA


def _label(regime: int) -> str:
    return _REGIME_LABELS.get(regime, f"regime-{regime}")


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values, dtype=np.float64))


def _segment_expectation(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
) -> tuple[tuple[SegmentRewardSummary, ...], tuple[RegimeRecurrenceSummary, ...]]:
    rewards = np.asarray(trace.reward, dtype=np.float32)
    segment_ids = np.asarray(trace.segment_index, dtype=np.int32)
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    occurrence_counts: dict[int, int] = {}
    previous_late: dict[int, float] = {}
    segments: list[SegmentRewardSummary] = []

    for segment_index, scheduled_regime in enumerate(config.world.segment_regimes):
        indices = np.flatnonzero(segment_ids == segment_index)
        if indices.size == 0:
            continue
        regime = int(regimes[int(indices[0])])
        if regime != scheduled_regime:
            raise ValueError("trace regime does not match summary schedule")
        if not bool(np.all(regimes[indices] == regime)):
            raise ValueError("one summary segment contains multiple regimes")
        window = min(config.metric_window, int(indices.size))
        values = rewards[indices]
        early = float(np.mean(values[:window], dtype=np.float64))
        late = float(np.mean(values[-window:], dtype=np.float64))
        occurrence = occurrence_counts.get(regime, 0)
        prior = previous_late.get(regime)
        segments.append(
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
        occurrence_counts[regime] = occurrence + 1
        previous_late[regime] = late

    recurrence: list[RegimeRecurrenceSummary] = []
    for regime in sorted(occurrence_counts):
        regime_segments = tuple(item for item in segments if item.regime_id == regime)
        recurrent = tuple(item for item in regime_segments if item.is_recurrence)
        recurrence.append(
            RegimeRecurrenceSummary(
                regime_id=regime,
                regime_label=_label(regime),
                occurrences=len(regime_segments),
                recurrence_count=len(recurrent),
                recurrence_entry_reward_mean=_mean(
                    [item.early_reward for item in recurrent]
                ),
                previous_late_reward_mean=_mean(
                    [
                        cast(float, item.previous_occurrence_late_reward)
                        for item in recurrent
                    ]
                ),
                recurrence_entry_change_mean=_mean(
                    [cast(float, item.recurrence_entry_change) for item in recurrent]
                ),
                recurrence_recovery_mean=_mean(
                    [item.within_segment_recovery for item in recurrent]
                ),
            )
        )
    return tuple(segments), tuple(recurrence)


def _float32_words(values: object) -> np.ndarray[Any, np.dtype[np.uint32]]:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


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
    for raw_step in np.flatnonzero(committed_slots >= 0):
        step = int(raw_step)
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
    d_segments = sorted(set(int(value) for value in segments[regimes == 4]))
    if not d_segments:
        return False, False
    for segment in d_segments:
        indices = np.flatnonzero(segments == segment)
        if indices.size == 0:
            raise ValueError("D-short segment has no trace rows")
        first = int(indices[0])
        last = int(indices[-1])
        for role in ("helper", "beneficiary"):
            if not np.array_equal(
                np.asarray(getattr(trace, f"{role}_status_pre"))[first],
                np.asarray(getattr(trace, f"{role}_status_post"))[last],
            ):
                return True, False
            if not np.array_equal(
                np.asarray(getattr(trace, f"{role}_generation_pre"))[first],
                np.asarray(getattr(trace, f"{role}_generation_post"))[last],
            ):
                return True, False
    return True, True


def _trace_role_budget(
    trace: HiddenRegimePrimitiveTrace,
    role: str,
) -> tuple[int, int]:
    scalars = 0
    for suffix in _TRACE_ROLE_STATE_FIELDS:
        array = np.asarray(getattr(trace, f"{role}_{suffix}_pre"))
        if array.shape[0] == 0:
            raise ValueError("resource reconstruction requires a nonempty trace")
        leaf = np.asarray(array[0])
        scalars += int(leaf.size)
    # Primitive traces compact status leaves to int8, whereas the persistent
    # SlotRoleState contract stores every numeric leaf as one 32-bit word.
    return scalars, scalars * 4


def _state_role_budget(role: SlotRoleState) -> tuple[int, int]:
    scalars = 0
    state_bytes = 0
    for name in _ROLE_ARRAY_FIELDS:
        leaf = np.asarray(getattr(role, name))
        scalars += int(leaf.size)
        state_bytes += int(leaf.nbytes)
    key_data = np.asarray(jr.key_data(role.key), dtype=np.uint32)
    scalars += int(key_data.size)
    state_bytes += int(key_data.nbytes)
    return scalars, state_bytes


def _resource_expectation(
    trace: HiddenRegimePrimitiveTrace,
    final_state: SlotSignalingState,
) -> HiddenRegimeResourceReport:
    initial_roles = (
        _trace_role_budget(trace, "helper"),
        _trace_role_budget(trace, "beneficiary"),
    )
    final_roles = (
        _state_role_budget(final_state.helper),
        _state_role_budget(final_state.beneficiary),
    )
    initial_scalars = sum(item[0] for item in initial_roles)
    initial_bytes = sum(item[1] for item in initial_roles)
    final_scalars = sum(item[0] for item in final_roles)
    final_bytes = sum(item[1] for item in final_roles)
    expected_role_scalars = 36 + 4 + 4 + 4 + 4 + 4 + 4 + 7 + 2
    expected_role_bytes = expected_role_scalars * 4
    expected_state_bytes = expected_role_bytes * 2
    return HiddenRegimeResourceReport(
        initial_state_scalars=initial_scalars,
        final_state_scalars=final_scalars,
        initial_state_bytes=initial_bytes,
        final_state_bytes=final_bytes,
        expected_state_bytes=expected_state_bytes,
        resource_constant=(
            initial_scalars == final_scalars
            and initial_bytes == final_bytes
            and initial_scalars == expected_role_scalars * 2
            and initial_bytes == expected_state_bytes
        ),
        resource_matched=all(
            scalars == expected_role_scalars and state_bytes == expected_role_bytes
            for scalars, state_bytes in (*initial_roles, *final_roles)
        ),
    )


def reconstruct_hidden_regime_summary_expectation(
    trace: HiddenRegimePrimitiveTrace,
    config: HiddenRegimeDevelopmentConfig,
    condition: HiddenRegimeDevelopmentCondition,
    final_state: SlotSignalingState,
) -> HiddenRegimeSummaryExpectation:
    """Reconstruct the full summary and resource report without producer helpers."""

    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        raise TypeError("trace must be a HiddenRegimePrimitiveTrace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        raise TypeError("config must be a HiddenRegimeDevelopmentConfig")
    if type(condition) is not str or condition not in _KNOWN_CONDITIONS:
        raise ValueError(f"unknown hidden-regime development condition: {condition!r}")
    if not isinstance(final_state, SlotSignalingState):
        raise TypeError("final_state must be a SlotSignalingState")

    segments, recurrence = _segment_expectation(trace, config)
    lineage = reconstruct_hidden_regime_lineage_expectation(trace, config)
    helper_writes_array = np.asarray(trace.helper_value_write, dtype=np.bool_)
    beneficiary_writes_array = np.asarray(trace.beneficiary_value_write, dtype=np.bool_)
    helper_effective_updates = int(
        np.count_nonzero(
            np.logical_and(
                helper_writes_array,
                _float32_words(trace.helper_candidate_value)
                != _float32_words(trace.helper_value_pre),
            )
        )
    )
    beneficiary_effective_updates = int(
        np.count_nonzero(
            np.logical_and(
                beneficiary_writes_array,
                _float32_words(trace.beneficiary_candidate_value)
                != _float32_words(trace.beneficiary_value_pre),
            )
        )
    )
    targets, generation_pairs = _replacement_provenance(trace)
    d_checked, d_non_displacement = _d_short_non_displacement(trace)
    helper_violations = int(
        np.count_nonzero(np.asarray(trace.helper_selective_mutation_violation))
    )
    beneficiary_violations = int(
        np.count_nonzero(np.asarray(trace.beneficiary_selective_mutation_violation))
    )
    immutability_applicable = condition not in _WRITABLE_CONDITIONS
    recurrent_segments = tuple(
        segment
        for segment in segments
        if segment.is_recurrence and segment.regime_id in (0, 1, 2, 3)
    )
    recurrence_entry_mean = _mean(
        [segment.early_reward for segment in recurrent_segments]
    )
    recurrence_recovery_mean = _mean(
        [segment.within_segment_recovery for segment in recurrent_segments]
    )
    rewards = np.asarray(trace.reward, dtype=np.float32)
    summary = HiddenRegimeRunSummary(
        num_steps=int(rewards.size),
        mean_prequential_reward=float(np.mean(rewards, dtype=np.float64)),
        legacy_metric_window=config.metric_window,
        legacy_recurrence_entry_reward_mean=recurrence_entry_mean,
        legacy_recurrence_recovery_mean=recurrence_recovery_mean,
        recurrence_entry_reward_mean=recurrence_entry_mean,
        recurrence_recovery_mean=recurrence_recovery_mean,
        segment_rewards=segments,
        recurrence_by_regime=recurrence,
        commit_generation_lineages=lineage.commit_generation_lineages,
        synchronized_commit_lineage_count=lineage.synchronized_commit_lineage_count,
        acquisition_qualified_commit_lineage_count=(
            lineage.acquisition_qualified_commit_lineage_count
        ),
        acquisition_unqualified_commit_lineage_count=(
            lineage.acquisition_unqualified_commit_lineage_count
        ),
        recurrence_retention=lineage.recurrence_retention,
        retention=lineage.retention,
        helper_value_write_count=int(np.count_nonzero(helper_writes_array)),
        beneficiary_value_write_count=int(np.count_nonzero(beneficiary_writes_array)),
        helper_effective_learning_update_count=helper_effective_updates,
        beneficiary_effective_learning_update_count=beneficiary_effective_updates,
        both_roles_learned=(
            helper_effective_updates > 0 and beneficiary_effective_updates > 0
        ),
        helper_commit_count=int(
            np.count_nonzero(np.asarray(trace.helper_committed_slot) >= 0)
        ),
        beneficiary_commit_count=int(
            np.count_nonzero(np.asarray(trace.beneficiary_committed_slot) >= 0)
        ),
        helper_replacement_count=int(
            np.count_nonzero(np.asarray(trace.helper_retired_slot) >= 0)
        ),
        beneficiary_replacement_count=int(
            np.count_nonzero(np.asarray(trace.beneficiary_retired_slot) >= 0)
        ),
        candidate_confirmation_events=int(
            np.count_nonzero(
                np.logical_and(
                    np.asarray(trace.helper_candidate_lease_success),
                    np.asarray(trace.helper_candidate_confirmations_post) == 0,
                )
            )
        ),
        c_old_to_c_new_replacement_count=len(targets),
        c_old_to_c_new_target_slots=targets,
        c_old_to_c_new_generation_pairs=generation_pairs,
        c_old_to_c_new_exactly_one_target=len(targets) == 1,
        d_short_checked=d_checked,
        d_short_non_displacement=d_non_displacement,
        selective_immutability_applicable=immutability_applicable,
        helper_selective_mutation_violations=helper_violations,
        beneficiary_selective_mutation_violations=beneficiary_violations,
        selective_durable_bit_immutable_until_atomic_replacement=(
            immutability_applicable
            and helper_violations == 0
            and beneficiary_violations == 0
        ),
        lifecycle_synchronized_every_step=bool(
            np.all(np.asarray(trace.lifecycle_synchronized))
        ),
    )
    return HiddenRegimeSummaryExpectation(
        summary=summary,
        resource=_resource_expectation(trace, final_state),
    )


def _float64_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float64).view(np.uint64))


def _compare(expected: object, actual: object, path: str, mismatches: list[str]) -> None:
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
        actual_tuple = cast(tuple[object, ...], actual)
        if len(actual_tuple) != len(expected):
            mismatches.append(f"{path}.length")
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual_tuple, strict=False)
        ):
            _compare(expected_item, actual_item, f"{path}[{index}]", mismatches)
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


def validate_hidden_regime_summary(
    trace: object,
    config: object,
    condition: object,
    final_state: object,
    summary: object,
    resource: object,
) -> HiddenRegimeSummaryValidation:
    """Fail closed unless every independently derived output field agrees."""

    mismatches: list[str] = []
    if not isinstance(trace, HiddenRegimePrimitiveTrace):
        mismatches.append("input.trace")
    if not isinstance(config, HiddenRegimeDevelopmentConfig):
        mismatches.append("input.config")
    if type(condition) is not str or condition not in _KNOWN_CONDITIONS:
        mismatches.append("input.condition")
    if not isinstance(final_state, SlotSignalingState):
        mismatches.append("input.final_state")
    if not isinstance(summary, HiddenRegimeRunSummary):
        mismatches.append("input.summary")
    if not isinstance(resource, HiddenRegimeResourceReport):
        mismatches.append("input.resource")
    if mismatches:
        return HiddenRegimeSummaryValidation(
            valid=False,
            mismatches=tuple(mismatches),
            summary_fields_checked=0,
            resource_fields_checked=0,
            expected=None,
        )

    try:
        expected = reconstruct_hidden_regime_summary_expectation(
            cast(HiddenRegimePrimitiveTrace, trace),
            cast(HiddenRegimeDevelopmentConfig, config),
            cast(HiddenRegimeDevelopmentCondition, condition),
            cast(SlotSignalingState, final_state),
        )
    except (IndexError, TypeError, ValueError) as error:
        return HiddenRegimeSummaryValidation(
            valid=False,
            mismatches=(f"reconstruction: {error}",),
            summary_fields_checked=0,
            resource_fields_checked=0,
            expected=None,
        )

    _compare(expected.summary, summary, "summary", mismatches)
    _compare(expected.resource, resource, "resource", mismatches)
    return HiddenRegimeSummaryValidation(
        valid=not mismatches,
        mismatches=tuple(mismatches),
        summary_fields_checked=len(dataclasses.fields(HiddenRegimeRunSummary)),
        resource_fields_checked=len(dataclasses.fields(HiddenRegimeResourceReport)),
        expected=expected,
    )


__all__ = [
    "HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA",
    "HiddenRegimeSummaryExpectation",
    "HiddenRegimeSummaryValidation",
    "reconstruct_hidden_regime_summary_expectation",
    "validate_hidden_regime_summary",
]
