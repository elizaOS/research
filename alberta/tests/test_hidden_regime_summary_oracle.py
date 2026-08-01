"""Complete independent hidden-regime summary and resource tamper tests."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import alberta_framework.evaluation.hidden_regime_signaling_development as development_module
import alberta_framework.evaluation.hidden_regime_summary_oracle as summary_oracle
from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    SELECTIVE_FULL,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimeResourceReport,
    HiddenRegimeRunResult,
    HiddenRegimeRunSummary,
    HiddenRegimeSeedPair,
    RegimeRecurrenceSummary,
    SegmentRewardSummary,
    run_hidden_regime_condition,
)
from alberta_framework.evaluation.hidden_regime_summary_oracle import (
    reconstruct_hidden_regime_summary_expectation,
    validate_hidden_regime_summary,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    audit_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DEFAULT_REGIME_PERMUTATIONS,
    HiddenRegimeWorldConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def summary_run() -> HiddenRegimeRunResult:
    config = HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(5, 7, 4),
            segment_regimes=(0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.25,
            epsilon=0.1,
            relevance_rate=0.1,
            lease_length=4,
            confirmation_steps=2,
            durable_retrieval_threshold=0.5,
            candidate_confirmation_threshold=0.75,
            candidate_confirmation_leases=2,
            scratch_training_leases_before_retest=2,
        ),
        metric_window=2,
    )
    return run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=HiddenRegimeSeedPair(
            namespace="manual-complete-summary-oracle-unit-v1",
            index=0,
            world_seed=613,
            learner_seed=947,
        ),
        config=config,
    )


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
        return dataclasses.replace(
            value,
            **{field.name: _mutate(getattr(value, field.name))},
        )
    raise AssertionError(f"unsupported tamper value: {type(value)!r}")


def _validate(run: HiddenRegimeRunResult, *, summary: object, resource: object):
    return validate_hidden_regime_summary(
        run.trace,
        run.config,
        run.condition,
        run.final_state,
        summary,
        resource,
    )


def _d_short_trace() -> SimpleNamespace:
    rows = 4
    fields: dict[str, object] = {
        "segment_index": np.asarray([0, 1, 1, 2], dtype=np.int32),
        "regime_id": np.asarray([0, 4, 4, 0], dtype=np.int32),
    }
    for role in ("helper", "beneficiary"):
        status = np.tile(np.asarray([1, 2, 2, 2], dtype=np.int32), (rows, 1))
        generation = np.tile(np.asarray([0, 1, 2, 3], dtype=np.int32), (rows, 1))
        fields.update(
            {
                f"{role}_committed_slot": np.full(rows, -1, dtype=np.int32),
                f"{role}_retired_slot": np.full(rows, -1, dtype=np.int32),
                f"{role}_status_pre": status.copy(),
                f"{role}_status_post": status.copy(),
                f"{role}_generation_pre": generation.copy(),
                f"{role}_generation_post": generation.copy(),
                f"{role}_next_generation_pre": np.full(rows, 4, dtype=np.int32),
                f"{role}_next_generation_post": np.full(rows, 4, dtype=np.int32),
            }
        )
    return SimpleNamespace(**fields)


def test_d_short_oracle_rejects_events_and_restored_midsegment_generation() -> None:
    trace = _d_short_trace()
    assert summary_oracle._d_short_non_displacement(trace) == (True, True)

    committed = _d_short_trace()
    committed.helper_committed_slot[1] = 1
    assert summary_oracle._d_short_non_displacement(committed) == (True, False)

    retired = _d_short_trace()
    retired.beneficiary_retired_slot[2] = 2
    assert summary_oracle._d_short_non_displacement(retired) == (True, False)

    restored = _d_short_trace()
    restored.helper_generation_post[1, 2] = 4
    restored.helper_generation_pre[2, 2] = 4
    assert np.array_equal(
        restored.helper_generation_pre[1],
        restored.helper_generation_post[2],
    )
    assert summary_oracle._d_short_non_displacement(restored) == (True, False)


def test_complete_summary_and_resource_reconstruct_exactly(
    summary_run: HiddenRegimeRunResult,
) -> None:
    validation = _validate(
        summary_run,
        summary=summary_run.summary,
        resource=summary_run.resource,
    )

    assert validation.valid, validation.mismatches
    assert validation.mismatches == ()
    assert validation.summary_fields_checked == len(
        dataclasses.fields(HiddenRegimeRunSummary)
    )
    assert validation.resource_fields_checked == len(
        dataclasses.fields(HiddenRegimeResourceReport)
    )
    assert validation.expected is not None
    assert validation.expected.summary == summary_run.summary
    assert validation.expected.resource == summary_run.resource

    report = audit_hidden_regime_run_result(summary_run)
    assert report.valid, report.mismatches
    assert report.summary_fields_checked == validation.summary_fields_checked
    assert report.resource_fields_checked == validation.resource_fields_checked


def test_oracle_does_not_call_any_producer_summary_or_resource_helper(
    summary_run: HiddenRegimeRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("producer reconstruction helper must not be called")

    for name in (
        "condition_spec",
        "_segment_summaries",
        "reconstruct_commit_generation_lineages",
        "reconstruct_hidden_regime_retention",
        "_replacement_provenance",
        "_d_short_non_displacement",
        "reconstruct_hidden_regime_summary",
        "slot_signaling_resource_budget",
        "_resource_report",
    ):
        monkeypatch.setattr(development_module, name, forbidden)

    expectation = reconstruct_hidden_regime_summary_expectation(
        summary_run.trace,
        summary_run.config,
        summary_run.condition,
        summary_run.final_state,
    )
    validation = _validate(
        summary_run,
        summary=summary_run.summary,
        resource=summary_run.resource,
    )

    assert expectation.summary == summary_run.summary
    assert expectation.resource == summary_run.resource
    assert validation.valid, validation.mismatches


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in dataclasses.fields(HiddenRegimeRunSummary)),
)
def test_every_top_level_summary_field_tamper_is_detected(
    summary_run: HiddenRegimeRunResult,
    field_name: str,
) -> None:
    hostile = dataclasses.replace(
        summary_run.summary,
        **{field_name: _mutate(getattr(summary_run.summary, field_name))},
    )

    validation = _validate(
        summary_run,
        summary=hostile,
        resource=summary_run.resource,
    )

    assert not validation.valid
    assert any(
        path.startswith(f"summary.{field_name}") for path in validation.mismatches
    )


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in dataclasses.fields(SegmentRewardSummary)),
)
def test_every_segment_reward_field_tamper_is_detected(
    summary_run: HiddenRegimeRunResult,
    field_name: str,
) -> None:
    first = summary_run.summary.segment_rewards[0]
    hostile_first = dataclasses.replace(
        first,
        **{field_name: _mutate(getattr(first, field_name))},
    )
    hostile = dataclasses.replace(
        summary_run.summary,
        segment_rewards=(hostile_first, *summary_run.summary.segment_rewards[1:]),
    )

    validation = _validate(
        summary_run,
        summary=hostile,
        resource=summary_run.resource,
    )

    assert not validation.valid
    assert any(
        path.startswith(f"summary.segment_rewards[0].{field_name}")
        for path in validation.mismatches
    )


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in dataclasses.fields(RegimeRecurrenceSummary)),
)
def test_every_regime_recurrence_field_tamper_is_detected(
    summary_run: HiddenRegimeRunResult,
    field_name: str,
) -> None:
    first = summary_run.summary.recurrence_by_regime[0]
    hostile_first = dataclasses.replace(
        first,
        **{field_name: _mutate(getattr(first, field_name))},
    )
    hostile = dataclasses.replace(
        summary_run.summary,
        recurrence_by_regime=(hostile_first, *summary_run.summary.recurrence_by_regime[1:]),
    )

    validation = _validate(
        summary_run,
        summary=hostile,
        resource=summary_run.resource,
    )

    assert not validation.valid
    assert any(
        path.startswith(f"summary.recurrence_by_regime[0].{field_name}")
        for path in validation.mismatches
    )


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in dataclasses.fields(HiddenRegimeResourceReport)),
)
def test_every_resource_field_tamper_is_detected(
    summary_run: HiddenRegimeRunResult,
    field_name: str,
) -> None:
    hostile = dataclasses.replace(
        summary_run.resource,
        **{field_name: _mutate(getattr(summary_run.resource, field_name))},
    )

    validation = _validate(
        summary_run,
        summary=summary_run.summary,
        resource=hostile,
    )

    assert not validation.valid
    assert any(
        path.startswith(f"resource.{field_name}") for path in validation.mismatches
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "mean_prequential_reward",
        "both_roles_learned",
        "c_old_to_c_new_replacement_count",
        "d_short_non_displacement",
    ),
)
def test_previous_adversarial_summary_bypass_now_fails_integrated_audit(
    summary_run: HiddenRegimeRunResult,
    field_name: str,
) -> None:
    hostile_summary = dataclasses.replace(
        summary_run.summary,
        **{field_name: _mutate(getattr(summary_run.summary, field_name))},
    )

    report = audit_hidden_regime_run_result(
        dataclasses.replace(summary_run, summary=hostile_summary)
    )

    assert not report.valid
    assert any(
        mismatch.startswith(f"derived.summary.{field_name}")
        for mismatch in report.mismatches
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("trace", object()),
        ("config", object()),
        ("condition", []),
        ("final_state", object()),
        ("summary", object()),
        ("resource", object()),
    ),
)
def test_invalid_inputs_fail_closed(
    summary_run: HiddenRegimeRunResult,
    name: str,
    value: object,
) -> None:
    inputs: dict[str, object] = {
        "trace": summary_run.trace,
        "config": summary_run.config,
        "condition": summary_run.condition,
        "final_state": summary_run.final_state,
        "summary": summary_run.summary,
        "resource": summary_run.resource,
    }
    inputs[name] = value

    validation = validate_hidden_regime_summary(**inputs)

    assert not validation.valid
    assert validation.expected is None
    assert validation.summary_fields_checked == 0
    assert validation.resource_fields_checked == 0
    assert f"input.{name}" in validation.mismatches
