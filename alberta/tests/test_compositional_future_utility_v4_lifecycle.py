"""Focused tests for the pure-stdlib v4 lifecycle accounting helper."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from alberta_framework.evaluation._compositional_future_utility_v4_lifecycle import (
    CASCADE_CAUSE,
    COMPOSITIONAL_FUTURE_UTILITY_V4_LIFECYCLE_SCHEMA,
    DIRECT_CAUSE,
    UNMARKED_CAUSE,
    build_v4_target_lifecycle,
    validate_v4_target_lifecycle,
    validate_v4_target_lifecycle_against_sources,
)

pytestmark = pytest.mark.unit

TARGET = "p4"


def _event(step: int, causes: list[str], *, slot: int = 4) -> dict[str, object]:
    return {
        "post_step": step,
        "acquired_slots": [slot],
        "slot_causes": {str(slot): causes},
    }


def _admissions(
    steps: list[int],
    admitted_steps: set[int],
) -> list[dict[str, object]]:
    return [
        {
            "post_step": step,
            "target_admission_outcomes": {
                TARGET: "admitted" if step in admitted_steps else "candidate_absent"
            },
        }
        for step in steps
    ]


def _structural(
    acquisitions: int,
    losses: int,
    present_at_end: bool,
) -> dict[str, dict[str, object]]:
    return {
        TARGET: {
            "initially_present": False,
            "acquisition_episode_count": acquisitions,
            "loss_episode_count": losses,
            "present_at_end": present_at_end,
            "structural_reacquisition_count": max(0, acquisitions - 1),
        }
    }


def _build(
    *,
    steps: list[int],
    events: list[dict[str, object]],
    admitted_steps: set[int],
    losses: int,
    present_at_end: bool,
) -> dict[str, object]:
    return build_v4_target_lifecycle(
        target_names=(TARGET,),
        expected_post_steps=steps,
        acquisition_events_by_target={TARGET: events},
        structural_lifecycle_by_target=_structural(
            len(events),
            losses,
            present_at_end,
        ),
        admission_outcome_records=_admissions(steps, admitted_steps),
    )


def _target(payload: dict[str, object]) -> dict[str, object]:
    targets = payload["targets"]
    assert type(targets) is dict
    record = targets[TARGET]
    assert type(record) is dict
    return record


def _partition_record(payload: dict[str, object], category: str) -> dict[str, object]:
    partition = _target(payload)["acquisition_episode_cause_partition"]
    assert type(partition) is dict
    record = partition[category]
    assert type(record) is dict
    return record


def _mixed_payload() -> dict[str, object]:
    return _build(
        steps=[10, 20, 30],
        events=[
            _event(10, [DIRECT_CAUSE]),
            _event(20, [CASCADE_CAUSE]),
            _event(30, [DIRECT_CAUSE, CASCADE_CAUSE]),
        ],
        admitted_steps={10, 30},
        losses=2,
        present_at_end=True,
    )


def test_direct_and_structural_episode_equality_is_preserved_exactly() -> None:
    payload = _build(
        steps=[11, 22],
        events=[
            _event(11, [DIRECT_CAUSE]),
            _event(22, [DIRECT_CAUSE, CASCADE_CAUSE]),
        ],
        admitted_steps={11, 22},
        losses=1,
        present_at_end=True,
    )

    assert payload == {
        "schema": COMPOSITIONAL_FUTURE_UTILITY_V4_LIFECYCLE_SCHEMA,
        "target_order": [TARGET],
        "opportunity_post_steps": [11, 22],
        "targets": {
            TARGET: {
                "initially_present": False,
                "direct_candidate_admission_episode_count": 2,
                "direct_candidate_admission_post_steps": [11, 22],
                "structural_acquisition_episode_count": 2,
                "structural_acquisition_post_steps": [11, 22],
                "structural_loss_episode_count": 1,
                "structural_reacquisition_episode_count": 1,
                "present_at_end": True,
                "acquisition_episode_cause_partition": {
                    "direct_only": {"episode_count": 1, "post_steps": [11]},
                    "cascade_only": {"episode_count": 0, "post_steps": []},
                    "direct_and_cascade": {
                        "episode_count": 1,
                        "post_steps": [22],
                    },
                },
            }
        },
    }
    assert validate_v4_target_lifecycle(payload) is payload


def test_cascade_only_acquisition_is_not_a_direct_admission() -> None:
    payload = _build(
        steps=[7],
        events=[_event(7, [CASCADE_CAUSE])],
        admitted_steps=set(),
        losses=0,
        present_at_end=True,
    )
    target = _target(payload)

    assert target["direct_candidate_admission_episode_count"] == 0
    assert target["structural_acquisition_episode_count"] == 1
    assert target["direct_candidate_admission_post_steps"] == []
    assert target["acquisition_episode_cause_partition"] == {
        "direct_only": {"episode_count": 0, "post_steps": []},
        "cascade_only": {"episode_count": 1, "post_steps": [7]},
        "direct_and_cascade": {"episode_count": 0, "post_steps": []},
    }


def test_absent_target_with_zero_events_has_a_closed_zero_lifecycle() -> None:
    payload = _build(
        steps=[7],
        events=[],
        admitted_steps=set(),
        losses=0,
        present_at_end=False,
    )
    target = _target(payload)

    assert target["initially_present"] is False
    assert target["direct_candidate_admission_episode_count"] == 0
    assert target["structural_acquisition_episode_count"] == 0
    assert target["structural_loss_episode_count"] == 0
    assert target["structural_reacquisition_episode_count"] == 0
    assert target["present_at_end"] is False


def test_mixed_episode_causes_form_an_exact_disjoint_partition() -> None:
    target = _target(_mixed_payload())

    assert target["direct_candidate_admission_episode_count"] == 2
    assert target["structural_acquisition_episode_count"] == 3
    assert target["structural_reacquisition_episode_count"] == 2
    assert target["acquisition_episode_cause_partition"] == {
        "direct_only": {"episode_count": 1, "post_steps": [10]},
        "cascade_only": {"episode_count": 1, "post_steps": [20]},
        "direct_and_cascade": {"episode_count": 1, "post_steps": [30]},
    }


def test_impossible_direct_excess_fails_before_a_report_can_be_built() -> None:
    with pytest.raises(ValueError, match="direct admission episodes exceed"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10, 20],
            acquisition_events_by_target={TARGET: [_event(10, [DIRECT_CAUSE])]},
            structural_lifecycle_by_target=_structural(1, 0, True),
            admission_outcome_records=_admissions([10, 20], {10, 20}),
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {"post_step": 10, "target_admission_outcomes": {TARGET: "admitted"}},
                {"post_step": 10, "target_admission_outcomes": {TARGET: "admitted"}},
            ],
            "duplicate step",
        ),
        (
            [{"post_step": 10, "target_admission_outcomes": {TARGET: "admitted"}}],
            "missing steps",
        ),
    ],
)
def test_admission_records_reject_duplicate_or_missing_steps(
    records: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10, 20],
            acquisition_events_by_target={TARGET: [_event(10, [DIRECT_CAUSE])]},
            structural_lifecycle_by_target=_structural(1, 0, True),
            admission_outcome_records=records,
        )


def test_duplicate_acquisition_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate step"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10],
            acquisition_events_by_target={
                TARGET: [
                    _event(10, [DIRECT_CAUSE], slot=4),
                    _event(10, [DIRECT_CAUSE], slot=5),
                ]
            },
            structural_lifecycle_by_target=_structural(2, 1, True),
            admission_outcome_records=_admissions([10], {10}),
        )


@pytest.mark.parametrize(
    ("steps", "message"),
    [([20, 10], "strictly increasing"), ([10, 30], "unexpected step")],
)
def test_out_of_order_or_unexpected_acquisition_steps_are_rejected(
    steps: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10, 20],
            acquisition_events_by_target={
                TARGET: [_event(step, [DIRECT_CAUSE]) for step in steps]
            },
            structural_lifecycle_by_target=_structural(2, 1, True),
            admission_outcome_records=_admissions([10, 20], {10, 20}),
        )


@pytest.mark.parametrize("cause", [UNMARKED_CAUSE, "unknown_mutation"])
def test_unmarked_and_unknown_acquisition_causes_fail_closed(cause: str) -> None:
    with pytest.raises(ValueError, match="unmarked|unknown"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10],
            acquisition_events_by_target={TARGET: [_event(10, [cause])]},
            structural_lifecycle_by_target=_structural(1, 0, True),
            admission_outcome_records=_admissions([10], set()),
        )


def test_declared_structural_counts_must_equal_exact_acquisition_events() -> None:
    with pytest.raises(ValueError, match="count differs from exact events"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10],
            acquisition_events_by_target={TARGET: [_event(10, [DIRECT_CAUSE])]},
            structural_lifecycle_by_target=_structural(2, 1, True),
            admission_outcome_records=_admissions([10], {10}),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("schema", "wrong-schema"),
        lambda payload: _target(payload).__setitem__(
            "direct_candidate_admission_episode_count", True
        ),
        lambda payload: _target(payload).__setitem__(
            "structural_reacquisition_episode_count", 0
        ),
        lambda payload: _target(payload).__setitem__(
            "direct_candidate_admission_post_steps", [10, 20]
        ),
        lambda payload: _partition_record(payload, "cascade_only").__setitem__(
            "post_steps", [10, 20]
        ),
    ],
    ids=("schema", "bool-as-int", "reacquisition", "direct-closure", "overlap"),
)
def test_validator_rejects_tampered_payloads(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    payload = copy.deepcopy(_mixed_payload())
    mutate(payload)

    with pytest.raises((TypeError, ValueError)):
        validate_v4_target_lifecycle(payload)


def test_noncanonical_source_container_types_are_rejected() -> None:
    event = _event(10, [DIRECT_CAUSE])
    event["acquired_slots"] = (4,)

    with pytest.raises(TypeError, match="acquired_slots must be an exact list"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10],
            acquisition_events_by_target={TARGET: [event]},
            structural_lifecycle_by_target=_structural(1, 0, True),
            admission_outcome_records=_admissions([10], {10}),
        )


def test_source_must_prove_targets_are_absent_at_genesis() -> None:
    structural = _structural(1, 0, True)
    structural[TARGET]["initially_present"] = True

    with pytest.raises(ValueError, match="initially_present must be false"):
        build_v4_target_lifecycle(
            target_names=(TARGET,),
            expected_post_steps=[10],
            acquisition_events_by_target={TARGET: [_event(10, [DIRECT_CAUSE])]},
            structural_lifecycle_by_target=structural,
            admission_outcome_records=_admissions([10], {10}),
        )


def test_source_bound_validation_catches_internally_consistent_tampering() -> None:
    steps = [10, 20, 30]
    events = [
        _event(10, [DIRECT_CAUSE]),
        _event(20, [CASCADE_CAUSE]),
        _event(30, [DIRECT_CAUSE, CASCADE_CAUSE]),
    ]
    structural = _structural(3, 2, True)
    admissions = _admissions(steps, {10, 30})
    payload = build_v4_target_lifecycle(
        target_names=(TARGET,),
        expected_post_steps=steps,
        acquisition_events_by_target={TARGET: events},
        structural_lifecycle_by_target=structural,
        admission_outcome_records=admissions,
    )
    tampered = copy.deepcopy(payload)
    target = _target(tampered)
    target["structural_loss_episode_count"] = 3
    target["present_at_end"] = False
    assert validate_v4_target_lifecycle(tampered) is tampered

    with pytest.raises(ValueError, match="differs from exact source-derived"):
        validate_v4_target_lifecycle_against_sources(
            tampered,
            target_names=(TARGET,),
            expected_post_steps=steps,
            acquisition_events_by_target={TARGET: events},
            structural_lifecycle_by_target=structural,
            admission_outcome_records=admissions,
        )
    assert (
        validate_v4_target_lifecycle_against_sources(
            payload,
            target_names=(TARGET,),
            expected_post_steps=steps,
            acquisition_events_by_target={TARGET: events},
            structural_lifecycle_by_target=structural,
            admission_outcome_records=admissions,
        )
        is payload
    )
