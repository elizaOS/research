"""Independent lineage reconstruction and exhaustive summary-tamper tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_regime_signaling_development as development_module
from alberta_framework.core.slot_signaling_agent import (
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_VACANT,
    SlotSignalingConfig,
)
from alberta_framework.evaluation.hidden_regime_lineage_oracle import (
    independent_coalesced_episode_bounds,
    independent_lineage_recurrence_segments,
    validate_hidden_regime_lineage_summary,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    SELECTIVE_FULL,
    CommitGenerationLineage,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeRunResult,
    HiddenRegimeRunSummary,
    HiddenRegimeSeedPair,
    RecurrenceLineageProbe,
    RecurrenceRetentionRecord,
    RetentionAggregateSummary,
    reconstruct_hidden_regime_summary,
    run_hidden_regime_condition,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    audit_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DEFAULT_REGIME_PERMUTATIONS,
    DEFAULT_SEGMENT_REGIMES,
    HiddenRegimeWorldConfig,
)

pytestmark = pytest.mark.development

_MANUAL_SEED = HiddenRegimeSeedPair(
    namespace="hidden-regime-manual-independent-lineage-ci-v1",
    index=0,
    world_seed=1033,
    learner_seed=2033,
)
_IDENTITY_BITS = np.eye(3, dtype=np.float32).view(np.uint32)


@pytest.fixture(scope="module")
def lifecycle_config() -> HiddenRegimeDevelopmentConfig:
    lengths = tuple(4 if regime == 4 else 24 for regime in DEFAULT_SEGMENT_REGIMES)
    return HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=lengths,
            segment_regimes=DEFAULT_SEGMENT_REGIMES,
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.5,
            epsilon=0.1,
            relevance_rate=0.5,
            lease_length=4,
            confirmation_steps=1,
            durable_retrieval_threshold=0.1,
            candidate_confirmation_threshold=0.2,
            candidate_confirmation_leases=1,
        ),
        metric_window=4,
    )


@pytest.fixture(scope="module")
def lifecycle_run(lifecycle_config: HiddenRegimeDevelopmentConfig) -> HiddenRegimeRunResult:
    return run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=_MANUAL_SEED,
        config=lifecycle_config,
    )


def _without_events(trace: HiddenRegimePrimitiveTrace) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        for suffix in (
            "committed_slot",
            "committed_generation",
            "retired_slot",
            "retired_generation",
        ):
            value = jnp.asarray(getattr(trace, f"{role}_{suffix}"))
            replacements[f"{role}_{suffix}"] = jnp.full_like(value, -1)
    return dataclasses.replace(trace, **replacements)


def _with_commit(
    trace: HiddenRegimePrimitiveTrace,
    *,
    step: int,
    slot: int,
    generation: int,
) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        slots = jnp.asarray(getattr(trace, f"{role}_committed_slot"))
        generations = jnp.asarray(getattr(trace, f"{role}_committed_generation"))
        bits = jnp.asarray(getattr(trace, f"{role}_value_bits_post"))
        replacements[f"{role}_committed_slot"] = slots.at[step].set(slot)
        replacements[f"{role}_committed_generation"] = generations.at[step].set(generation)
        replacements[f"{role}_value_bits_post"] = bits.at[step, slot].set(
            jnp.asarray(_IDENTITY_BITS, dtype=jnp.uint32)
        )
    return dataclasses.replace(trace, **replacements)


def _with_entry_generation(
    trace: HiddenRegimePrimitiveTrace,
    *,
    start: int,
    slot: int,
    generation: int,
    helper_active: int = 2,
    beneficiary_active: int = 2,
) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role, active in (
        ("helper", helper_active),
        ("beneficiary", beneficiary_active),
    ):
        status = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        generations = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        bits = jnp.asarray(getattr(trace, f"{role}_value_bits_pre"))
        active_slots = jnp.asarray(getattr(trace, f"{role}_active_slot_pre"))
        replacements[f"{role}_status_pre"] = status.at[start, slot].set(SLOT_DURABLE)
        replacements[f"{role}_generation_pre"] = generations.at[start, slot].set(generation)
        replacements[f"{role}_value_bits_pre"] = bits.at[start, slot].set(
            jnp.asarray(_IDENTITY_BITS, dtype=jnp.uint32)
        )
        replacements[f"{role}_active_slot_pre"] = active_slots.at[start].set(active)
    return dataclasses.replace(trace, **replacements)


def _with_relock(
    trace: HiddenRegimePrimitiveTrace,
    *,
    start: int,
    end: int,
    slot: int,
    generation: int,
) -> tuple[HiddenRegimePrimitiveTrace, int]:
    boundaries = np.logical_and(
        np.asarray(trace.helper_lease_boundary)[start:end],
        np.asarray(trace.beneficiary_lease_boundary)[start:end],
    )
    step = start + int(np.flatnonzero(boundaries)[0])
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_pre = jnp.asarray(getattr(trace, f"{role}_active_slot_pre"))
        active_post = jnp.asarray(getattr(trace, f"{role}_active_slot_post"))
        status_pre = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        status_post = jnp.asarray(getattr(trace, f"{role}_status_post"))
        generation_pre = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        generation_post = jnp.asarray(getattr(trace, f"{role}_generation_post"))
        relevant = jnp.asarray(getattr(trace, f"{role}_durable_relevant"))
        replacements[f"{role}_active_slot_pre"] = (
            active_pre.at[start:step].set(2).at[step].set(slot)
        )
        replacements[f"{role}_active_slot_post"] = (
            active_post.at[start:step].set(2).at[step].set(slot)
        )
        replacements[f"{role}_status_pre"] = status_pre.at[step, slot].set(SLOT_DURABLE)
        replacements[f"{role}_status_post"] = status_post.at[step, slot].set(SLOT_DURABLE)
        replacements[f"{role}_generation_pre"] = generation_pre.at[step, slot].set(generation)
        replacements[f"{role}_generation_post"] = generation_post.at[step, slot].set(generation)
        replacements[f"{role}_durable_relevant"] = relevant.at[step].set(True)
    return dataclasses.replace(trace, **replacements), step


@pytest.fixture(scope="module")
def qualified_case(
    lifecycle_run: HiddenRegimeRunResult,
) -> tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int]:
    segment = 2
    start = sum(lifecycle_run.config.world.segment_lengths[:segment])
    end = start + lifecycle_run.config.world.segment_lengths[segment]
    trace = _with_commit(_without_events(lifecycle_run.trace), step=0, slot=1, generation=7)
    trace = _with_entry_generation(trace, start=start, slot=1, generation=7)
    trace, relock_step = _with_relock(
        trace,
        start=start,
        end=end,
        slot=1,
        generation=7,
    )
    summary = reconstruct_hidden_regime_summary(
        trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )
    return trace, summary, segment, start, relock_step


def _mutate(value: object) -> object:
    if value is None:
        return 0
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.125
    if type(value) is str:
        return value + "-tampered"
    if isinstance(value, tuple):
        if not value:
            return (0,)
        return (_mutate(value[0]), *value[1:])
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        field = dataclasses.fields(value)[0]
        return dataclasses.replace(value, **{field.name: _mutate(getattr(value, field.name))})
    raise AssertionError(f"unsupported tamper value: {type(value)!r}")


def _field_names(dataclass_type: type[object]) -> Iterator[str]:
    return (field.name for field in dataclasses.fields(dataclass_type))


def test_independent_oracle_rejects_a_correct_mapping_with_a_tied_argmax(
    lifecycle_run: HiddenRegimeRunResult,
) -> None:
    trace = _with_commit(
        _without_events(lifecycle_run.trace),
        step=0,
        slot=1,
        generation=7,
    )
    tied_helper = np.eye(3, dtype=np.float32)
    tied_helper[0, 1] = np.float32(1.0)
    trace = dataclasses.replace(
        trace,
        helper_value_bits_post=jnp.asarray(trace.helper_value_bits_post)
        .at[0, 1]
        .set(jnp.asarray(tied_helper.view(np.uint32), dtype=jnp.uint32)),
    )
    summary = reconstruct_hidden_regime_summary(
        trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )

    validation = validate_hidden_regime_lineage_summary(
        trace,
        lifecycle_run.config,
        summary,
    )

    assert validation.valid, validation.mismatches
    assert validation.expected is not None
    (lineage,) = validation.expected.commit_generation_lineages
    assert lineage.committed_composed_greedy_mapping == lineage.target_mapping
    assert lineage.committed_composed_greedy_accuracy == 1.0
    assert not lineage.committed_composed_greedy_tie_free
    assert not lineage.acquisition_qualified


def test_real_manual_run_validates_and_integrated_audit_discloses_counts(
    lifecycle_run: HiddenRegimeRunResult,
) -> None:
    validation = validate_hidden_regime_lineage_summary(
        lifecycle_run.trace,
        lifecycle_run.config,
        lifecycle_run.summary,
    )
    assert validation.valid, validation.mismatches
    assert validation.commit_lineages_checked == 4
    assert validation.recurrence_records_checked == 12
    assert validation.aggregate_fields_checked == len(dataclasses.fields(RetentionAggregateSummary))
    assert lifecycle_run.summary.retention.dormant_probe_available_count == 11
    assert any(
        record.eligible_dormant_generations for record in lifecycle_run.summary.recurrence_retention
    )

    report = audit_hidden_regime_run_result(lifecycle_run)
    assert report.valid, report.mismatches
    assert report.commit_lineages_checked == validation.commit_lineages_checked
    assert report.recurrence_records_checked == validation.recurrence_records_checked
    assert report.retention_aggregate_fields_checked == validation.aggregate_fields_checked


def test_lineage_oracle_does_not_call_producer_reconstruction_helpers(
    lifecycle_run: HiddenRegimeRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("producer reconstruction helper must not be called")

    for name in (
        "reconstruct_commit_generation_lineages",
        "hidden_regime_lineage_recurrence_segments",
        "hidden_regime_coalesced_episode_bounds",
        "reconstruct_hidden_regime_retention",
    ):
        monkeypatch.setattr(development_module, name, forbidden)

    validation = validate_hidden_regime_lineage_summary(
        lifecycle_run.trace,
        lifecycle_run.config,
        lifecycle_run.summary,
    )

    assert validation.valid, validation.mismatches


def test_asymmetric_commit_event_fails_closed_in_independent_reconstruction(
    lifecycle_run: HiddenRegimeRunResult,
) -> None:
    trace = _without_events(lifecycle_run.trace)
    trace = dataclasses.replace(
        trace,
        helper_committed_slot=jnp.asarray(trace.helper_committed_slot).at[0].set(1),
        helper_committed_generation=jnp.asarray(trace.helper_committed_generation).at[0].set(7),
    )

    validation = validate_hidden_regime_lineage_summary(
        trace,
        lifecycle_run.config,
        lifecycle_run.summary,
    )

    assert not validation.valid
    assert validation.mismatches == (
        "reconstruction: commit lineage requires one synchronized valid slot/generation event",
    )


def test_integrated_audit_rejects_lineage_summary_tamper(
    lifecycle_run: HiddenRegimeRunResult,
) -> None:
    hostile_retention = dataclasses.replace(
        lifecycle_run.summary.retention,
        recurrence_count=lifecycle_run.summary.retention.recurrence_count + 1,
    )
    hostile_run = dataclasses.replace(
        lifecycle_run,
        summary=dataclasses.replace(
            lifecycle_run.summary,
            retention=hostile_retention,
        ),
    )

    report = audit_hidden_regime_run_result(hostile_run)

    assert not report.valid
    assert "derived.summary.retention.recurrence_count" in report.mismatches


def test_lineage_collection_lengths_fail_closed(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
) -> None:
    trace, summary, segment, _, _ = qualified_case
    record_index = next(
        index
        for index, record in enumerate(summary.recurrence_retention)
        if record.segment_index == segment
    )
    selected_record = summary.recurrence_retention[record_index]
    without_probe = dataclasses.replace(
        selected_record,
        prior_same_regime_lineages=(),
    )
    records_without_probe = list(summary.recurrence_retention)
    records_without_probe[record_index] = without_probe
    hostile_summaries = (
        dataclasses.replace(summary, commit_generation_lineages=()),
        dataclasses.replace(summary, recurrence_retention=summary.recurrence_retention[:-1]),
        dataclasses.replace(
            summary,
            recurrence_retention=tuple(records_without_probe),
        ),
    )

    validations = tuple(
        validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)
        for hostile in hostile_summaries
    )

    assert all(not validation.valid for validation in validations)
    assert "summary.commit_generation_lineages.length" in validations[0].mismatches
    assert "summary.recurrence_retention.length" in validations[1].mismatches
    assert (
        f"summary.recurrence_retention[{record_index}].prior_same_regime_lineages.length"
    ) in validations[2].mismatches


def test_qualified_selected_lineage_and_post_relock_are_reconstructed(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
) -> None:
    trace, summary, segment, start, relock_step = qualified_case
    validation = validate_hidden_regime_lineage_summary(
        trace,
        lifecycle_run.config,
        summary,
    )
    assert validation.valid, validation.mismatches
    record = next(item for item in summary.recurrence_retention if item.segment_index == segment)
    assert record.latest_prior_qualified_lineage_index == 0
    assert record.selected_lineage_index == 0
    assert record.selected_lineage_entry_activity_status == "dormant"
    assert record.selected_exact_generation_relock_observed is True
    assert record.selected_first_exact_generation_relock_step == relock_step
    assert record.selected_first_exact_generation_relock_segment_step == relock_step - start
    assert record.selected_exact_generation_relock_phase == "post"
    assert record.selected_observed_learner_boundaries_until_relock == 1
    assert record.selected_first_scratch_entry_phase == "post"
    assert record.selected_scratch_entered_before_relock is False
    assert record.selected_durable_retrieval_before_scratch is True


def test_active_mixed_and_final_post_scratch_event_phases(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
) -> None:
    base_trace, _, segment, start, _ = qualified_case
    end = start + lifecycle_run.config.world.segment_lengths[segment]

    active_updates: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_updates[f"{role}_active_slot_pre"] = (
            jnp.asarray(getattr(base_trace, f"{role}_active_slot_pre")).at[start:end].set(1)
        )
        active_updates[f"{role}_active_slot_post"] = (
            jnp.asarray(getattr(base_trace, f"{role}_active_slot_post")).at[start:end].set(1)
        )
    active_trace = dataclasses.replace(base_trace, **active_updates)
    active_summary = reconstruct_hidden_regime_summary(
        active_trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )
    active_validation = validate_hidden_regime_lineage_summary(
        active_trace,
        lifecycle_run.config,
        active_summary,
    )
    assert active_validation.valid, active_validation.mismatches
    active = next(
        item for item in active_summary.recurrence_retention if item.segment_index == segment
    )
    assert active.selected_lineage_entry_activity_status == "active"
    assert active.selected_exact_generation_relock_phase == "pre"
    assert active.selected_observed_learner_boundaries_until_relock == 0

    mixed_trace = dataclasses.replace(
        base_trace,
        helper_active_slot_pre=jnp.asarray(base_trace.helper_active_slot_pre).at[start:end].set(1),
        helper_active_slot_post=jnp.asarray(base_trace.helper_active_slot_post)
        .at[start:end]
        .set(1),
        beneficiary_active_slot_pre=jnp.asarray(base_trace.beneficiary_active_slot_pre)
        .at[start:end]
        .set(2),
        beneficiary_active_slot_post=jnp.asarray(base_trace.beneficiary_active_slot_post)
        .at[start:end]
        .set(2),
    )
    mixed_summary = reconstruct_hidden_regime_summary(
        mixed_trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )
    mixed_validation = validate_hidden_regime_lineage_summary(
        mixed_trace,
        lifecycle_run.config,
        mixed_summary,
    )
    assert mixed_validation.valid, mixed_validation.mismatches
    mixed = next(
        item for item in mixed_summary.recurrence_retention if item.segment_index == segment
    )
    assert mixed.selected_lineage_entry_activity_status == "mixed"
    assert mixed.selected_exact_generation_relock_observed is False

    scratch_updates: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        scratch_updates[f"{role}_active_slot_pre"] = (
            jnp.asarray(getattr(base_trace, f"{role}_active_slot_pre")).at[start:end].set(2)
        )
        scratch_updates[f"{role}_active_slot_post"] = (
            jnp.asarray(getattr(base_trace, f"{role}_active_slot_post"))
            .at[start:end]
            .set(2)
            .at[end - 1]
            .set(SCRATCH_SLOT)
        )
    scratch_trace = dataclasses.replace(base_trace, **scratch_updates)
    scratch_summary = reconstruct_hidden_regime_summary(
        scratch_trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )
    scratch_validation = validate_hidden_regime_lineage_summary(
        scratch_trace,
        lifecycle_run.config,
        scratch_summary,
    )
    assert scratch_validation.valid, scratch_validation.mismatches
    scratch = next(
        item for item in scratch_summary.recurrence_retention if item.segment_index == segment
    )
    assert scratch.selected_first_scratch_entry_step == end - 1
    assert scratch.selected_first_scratch_entry_phase == "post"
    assert scratch.selected_scratch_entered_before_relock is True


def test_latest_qualified_can_be_evicted_while_older_selected_survives(
    lifecycle_run: HiddenRegimeRunResult,
) -> None:
    segment = 2
    start = sum(lifecycle_run.config.world.segment_lengths[:segment])
    trace = _with_commit(_without_events(lifecycle_run.trace), step=0, slot=1, generation=7)
    trace = _with_commit(trace, step=1, slot=2, generation=8)
    trace = _with_entry_generation(trace, start=start, slot=1, generation=7)
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        status = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        generation = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        replacements[f"{role}_status_pre"] = status.at[start, 2].set(SLOT_VACANT)
        replacements[f"{role}_generation_pre"] = generation.at[start, 2].set(0)
    trace = dataclasses.replace(trace, **replacements)
    summary = reconstruct_hidden_regime_summary(
        trace,
        lifecycle_run.config,
        lifecycle_run.condition,
    )

    validation = validate_hidden_regime_lineage_summary(
        trace,
        lifecycle_run.config,
        summary,
    )

    assert validation.valid, validation.mismatches
    record = next(item for item in summary.recurrence_retention if item.segment_index == segment)
    assert record.latest_prior_qualified_lineage_index == 1
    assert record.latest_prior_qualified_survived is False
    assert record.selected_lineage_index == 0
    assert record.latest_qualified_acquisition_segment_index == 0


def test_adjacent_equal_segments_are_one_episode_and_not_a_recurrence() -> None:
    world = HiddenRegimeWorldConfig(
        segment_lengths=(2, 2, 2, 2),
        segment_regimes=(0, 0, 1, 0),
        regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
        repeat_schedule=False,
    )
    config = HiddenRegimeDevelopmentConfig(
        world=world,
        learner=SlotSignalingConfig(
            lease_length=2,
            confirmation_steps=1,
            candidate_confirmation_leases=1,
        ),
        metric_window=2,
    )
    run = run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=HiddenRegimeSeedPair(
            namespace="hidden-regime-manual-coalescing-lineage-ci-v1",
            index=0,
            world_seed=71,
            learner_seed=72,
        ),
        config=config,
    )

    assert independent_lineage_recurrence_segments(world) == ((3, 0, 1),)
    assert independent_coalesced_episode_bounds(world, 1) == (0, 2, 0, 4)
    validation = validate_hidden_regime_lineage_summary(run.trace, config, run.summary)
    assert validation.valid, validation.mismatches
    assert len(run.summary.recurrence_retention) == 1
    record = run.summary.recurrence_retention[0]
    assert record.segment_index == 3
    assert record.occurrence_index == 1
    assert record.raw_segment_occurrence_index == 2


@pytest.mark.parametrize("field_name", tuple(_field_names(CommitGenerationLineage)))
def test_every_commit_lineage_field_is_reconstructed_and_tamper_detected(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
    field_name: str,
) -> None:
    trace, summary, _, _, _ = qualified_case
    original = summary.commit_generation_lineages[0]
    tampered = dataclasses.replace(
        original,
        **{field_name: _mutate(getattr(original, field_name))},
    )
    hostile = dataclasses.replace(
        summary,
        commit_generation_lineages=(tampered, *summary.commit_generation_lineages[1:]),
    )

    validation = validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)

    assert not validation.valid
    expected_path = f"summary.commit_generation_lineages[0].{field_name}"
    assert any(path.startswith(expected_path) for path in validation.mismatches)


@pytest.mark.parametrize("field_name", tuple(_field_names(RecurrenceLineageProbe)))
def test_every_recurrence_lineage_probe_field_tamper_is_detected(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
    field_name: str,
) -> None:
    trace, summary, segment, _, _ = qualified_case
    record_index = next(
        index
        for index, record in enumerate(summary.recurrence_retention)
        if record.segment_index == segment
    )
    record = summary.recurrence_retention[record_index]
    probe = record.prior_same_regime_lineages[0]
    tampered_probe = dataclasses.replace(
        probe,
        **{field_name: _mutate(getattr(probe, field_name))},
    )
    tampered_record = dataclasses.replace(
        record,
        prior_same_regime_lineages=(
            tampered_probe,
            *record.prior_same_regime_lineages[1:],
        ),
    )
    hostile_records = list(summary.recurrence_retention)
    hostile_records[record_index] = tampered_record
    hostile = dataclasses.replace(summary, recurrence_retention=tuple(hostile_records))

    validation = validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)

    assert not validation.valid
    expected_path = (
        f"summary.recurrence_retention[{record_index}].prior_same_regime_lineages[0].{field_name}"
    )
    assert any(path.startswith(expected_path) for path in validation.mismatches)


@pytest.mark.parametrize("field_name", tuple(_field_names(RecurrenceRetentionRecord)))
def test_every_recurrence_record_field_tamper_is_detected(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
    field_name: str,
) -> None:
    trace, summary, segment, _, _ = qualified_case
    record_index = next(
        index
        for index, record in enumerate(summary.recurrence_retention)
        if record.segment_index == segment
    )
    record = summary.recurrence_retention[record_index]
    tampered_record = dataclasses.replace(
        record,
        **{field_name: _mutate(getattr(record, field_name))},
    )
    hostile_records = list(summary.recurrence_retention)
    hostile_records[record_index] = tampered_record
    hostile = dataclasses.replace(summary, recurrence_retention=tuple(hostile_records))

    validation = validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)

    assert not validation.valid
    expected_path = f"summary.recurrence_retention[{record_index}].{field_name}"
    assert any(path.startswith(expected_path) for path in validation.mismatches)


@pytest.mark.parametrize("field_name", tuple(_field_names(RetentionAggregateSummary)))
def test_every_retention_aggregate_field_tamper_is_detected(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
    field_name: str,
) -> None:
    trace, summary, _, _, _ = qualified_case
    hostile_retention = dataclasses.replace(
        summary.retention,
        **{field_name: _mutate(getattr(summary.retention, field_name))},
    )
    hostile = dataclasses.replace(summary, retention=hostile_retention)

    validation = validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)

    assert not validation.valid
    expected_path = f"summary.retention.{field_name}"
    assert any(path.startswith(expected_path) for path in validation.mismatches)


@pytest.mark.parametrize(
    "field_name",
    (
        "synchronized_commit_lineage_count",
        "acquisition_qualified_commit_lineage_count",
        "acquisition_unqualified_commit_lineage_count",
    ),
)
def test_every_lineage_summary_count_tamper_is_detected(
    lifecycle_run: HiddenRegimeRunResult,
    qualified_case: tuple[HiddenRegimePrimitiveTrace, HiddenRegimeRunSummary, int, int, int],
    field_name: str,
) -> None:
    trace, summary, _, _, _ = qualified_case
    hostile = dataclasses.replace(
        summary,
        **{field_name: getattr(summary, field_name) + 1},
    )

    validation = validate_hidden_regime_lineage_summary(trace, lifecycle_run.config, hostile)

    assert not validation.valid
    assert f"summary.{field_name}" in validation.mismatches
