"""Strict, pure-stdlib lifecycle accounting for a future v4 evaluator.

The helper consumes already-produced, exact curation audit records.  It does
not execute an agent, import a v3 campaign module, authorize a run, write an
artifact, or make an evidence claim.  Its only job is to keep direct candidate
admissions distinct from structural acquisitions and to preserve the exact
cause partition of every structural acquisition episode.

The v4 campaign contract starts every audited target absent.  The structural
source must therefore carry the exact field ``"initially_present": False``;
this is the provenance premise that makes ``acquisitions - losses == end``
the correct closure rather than an accidental equality.

Input acquisition events use the existing audit shape::

    {
        "post_step": 17,
        "acquired_slots": [4, 6],
        "slot_causes": {
            "4": ["promotion_root_replacement"],
            "6": ["cascade_dependency_refill"],
        },
    }

Admission outcomes are supplied once for every expected opportunity step as
``{"post_step": int, "target_admission_outcomes": {target: outcome}}``.
Only the exact outcome ``"admitted"`` counts as a direct candidate admission.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

COMPOSITIONAL_FUTURE_UTILITY_V4_LIFECYCLE_SCHEMA = (
    "alberta.compositional-future-utility.target-lifecycle.v4"
)

DIRECT_CAUSE = "promotion_root_replacement"
CASCADE_CAUSE = "cascade_dependency_refill"
UNMARKED_CAUSE = "unmarked_signature_dependency_change"

DIRECT_ONLY = "direct_only"
CASCADE_ONLY = "cascade_only"
DIRECT_AND_CASCADE = "direct_and_cascade"
CAUSE_PARTITION_ORDER = (DIRECT_ONLY, CASCADE_ONLY, DIRECT_AND_CASCADE)

ADMISSION_OUTCOMES = (
    "admitted",
    "already_active",
    "candidate_absent",
    "candidate_immature",
    "no_eligible_active_destination",
    "topology_blocked",
    "depth_blocked",
    "headroom_blocked",
    "destination_mask_inconsistent",
    "candidate_selection_competition",
    "promotion_margin_failed",
    "selected_topology_recheck_failed",
    "selected_depth_recheck_failed",
    "promotion_gate_not_requested",
    "promotion_commit_rollback",
    "multiple_candidate_constraints",
)

_ROOT_FIELDS = ("schema", "target_order", "opportunity_post_steps", "targets")
_TARGET_FIELDS = (
    "initially_present",
    "direct_candidate_admission_episode_count",
    "direct_candidate_admission_post_steps",
    "structural_acquisition_episode_count",
    "structural_acquisition_post_steps",
    "structural_loss_episode_count",
    "structural_reacquisition_episode_count",
    "present_at_end",
    "acquisition_episode_cause_partition",
)
_STRUCTURAL_SOURCE_FIELDS = (
    "initially_present",
    "acquisition_episode_count",
    "loss_episode_count",
    "present_at_end",
    "structural_reacquisition_count",
)
_ACQUISITION_EVENT_FIELDS = ("post_step", "acquired_slots", "slot_causes")
_ADMISSION_RECORD_FIELDS = ("post_step", "target_admission_outcomes")
_CAUSE_LABEL_ORDER = (DIRECT_CAUSE, CASCADE_CAUSE)


def _plain_dict(value: object, path: str, fields: Sequence[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dict")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise TypeError(f"{path} keys must be exact strings")
    expected = set(fields)
    actual = cast(set[str], set(result))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} fields differ; missing={missing}, extra={extra}")
    return cast(dict[str, object], result)


def _plain_list(value: object, path: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{path} must be an exact list")
    return cast(list[object], value)


def _exact_nonnegative_int(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an exact int")
    result = value
    if result < 0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _canonical_names(value: Sequence[str], path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of exact strings")
    names = tuple(value)
    if not names:
        raise ValueError(f"{path} must not be empty")
    if any(type(name) is not str or not name for name in names):
        raise TypeError(f"{path} entries must be nonempty exact strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate targets")
    return names


def _canonical_expected_steps(value: Sequence[int], path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of exact ints")
    steps = tuple(value)
    if not steps:
        raise ValueError(f"{path} must not be empty")
    if any(type(step) is not int for step in steps):
        raise TypeError(f"{path} entries must be exact ints")
    if any(step < 0 for step in steps):
        raise ValueError(f"{path} entries must be nonnegative")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{path} contains duplicate steps")
    if tuple(sorted(steps)) != steps:
        raise ValueError(f"{path} must be strictly increasing")
    return steps


def _exact_step_list(
    value: object,
    path: str,
    *,
    allowed_steps: frozenset[int],
) -> list[int]:
    raw = _plain_list(value, path)
    if any(type(step) is not int for step in raw):
        raise TypeError(f"{path} entries must be exact ints")
    steps = cast(list[int], raw)
    if len(set(steps)) != len(steps):
        raise ValueError(f"{path} contains duplicate steps")
    if sorted(steps) != steps:
        raise ValueError(f"{path} must be strictly increasing")
    unexpected = sorted(set(steps) - allowed_steps)
    if unexpected:
        raise ValueError(f"{path} contains unexpected steps: {unexpected}")
    return steps


def _target_mapping(
    value: object,
    path: str,
    target_names: tuple[str, ...],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dict")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise TypeError(f"{path} keys must be exact strings")
    actual = cast(set[str], set(result))
    expected = set(target_names)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} targets differ; missing={missing}, extra={extra}")
    return cast(dict[str, object], result)


def _admitted_steps_by_target(
    value: object,
    *,
    target_names: tuple[str, ...],
    expected_steps: tuple[int, ...],
) -> dict[str, list[int]]:
    records = _plain_list(value, "admission_outcome_records")
    seen_steps: set[int] = set()
    admitted: dict[str, list[int]] = {target: [] for target in target_names}
    actual_steps: list[int] = []
    expected_set = frozenset(expected_steps)

    for index, raw_record in enumerate(records):
        path = f"admission_outcome_records[{index}]"
        record = _plain_dict(raw_record, path, _ADMISSION_RECORD_FIELDS)
        step = _exact_nonnegative_int(record["post_step"], f"{path}.post_step")
        if step in seen_steps:
            raise ValueError(f"admission_outcome_records contains duplicate step {step}")
        if step not in expected_set:
            raise ValueError(f"admission_outcome_records contains unexpected step {step}")
        seen_steps.add(step)
        actual_steps.append(step)

        outcomes = _target_mapping(
            record["target_admission_outcomes"],
            f"{path}.target_admission_outcomes",
            target_names,
        )
        for target in target_names:
            outcome = outcomes[target]
            if type(outcome) is not str:
                raise TypeError(
                    f"{path}.target_admission_outcomes.{target} must be an exact str"
                )
            if outcome not in ADMISSION_OUTCOMES:
                raise ValueError(
                    f"{path}.target_admission_outcomes.{target} has unknown outcome "
                    f"{outcome!r}"
                )
            if outcome == "admitted":
                admitted[target].append(step)

    actual = tuple(actual_steps)
    if actual != expected_steps:
        missing = sorted(set(expected_steps) - seen_steps)
        if missing:
            raise ValueError(f"admission_outcome_records is missing steps: {missing}")
        raise ValueError("admission_outcome_records must follow expected step order")
    return admitted


def _acquisition_partition(
    value: object,
    *,
    target: str,
    expected_steps: tuple[int, ...],
) -> tuple[list[int], dict[str, list[int]]]:
    path = f"acquisition_events_by_target.{target}"
    events = _plain_list(value, path)
    expected_set = frozenset(expected_steps)
    seen_steps: set[int] = set()
    acquisition_steps: list[int] = []
    partition: dict[str, list[int]] = {name: [] for name in CAUSE_PARTITION_ORDER}

    for index, raw_event in enumerate(events):
        event_path = f"{path}[{index}]"
        event = _plain_dict(raw_event, event_path, _ACQUISITION_EVENT_FIELDS)
        step = _exact_nonnegative_int(event["post_step"], f"{event_path}.post_step")
        if step in seen_steps:
            raise ValueError(f"{path} contains duplicate step {step}")
        if step not in expected_set:
            raise ValueError(f"{path} contains unexpected step {step}")
        if acquisition_steps and step <= acquisition_steps[-1]:
            raise ValueError(f"{path} must be strictly increasing by post_step")
        seen_steps.add(step)
        acquisition_steps.append(step)

        raw_slots = _plain_list(event["acquired_slots"], f"{event_path}.acquired_slots")
        if not raw_slots:
            raise ValueError(f"{event_path}.acquired_slots must not be empty")
        if any(type(slot) is not int for slot in raw_slots):
            raise TypeError(f"{event_path}.acquired_slots entries must be exact ints")
        slots = cast(list[int], raw_slots)
        if any(slot < 0 for slot in slots):
            raise ValueError(f"{event_path}.acquired_slots entries must be nonnegative")
        if sorted(set(slots)) != slots:
            raise ValueError(
                f"{event_path}.acquired_slots must be strictly increasing and unique"
            )

        causes = _plain_dict(
            event["slot_causes"],
            f"{event_path}.slot_causes",
            tuple(str(slot) for slot in slots),
        )
        if tuple(causes) != tuple(str(slot) for slot in slots):
            raise ValueError(f"{event_path}.slot_causes must follow acquired slot order")

        episode_causes: set[str] = set()
        for slot in slots:
            labels_path = f"{event_path}.slot_causes.{slot}"
            raw_labels = _plain_list(causes[str(slot)], labels_path)
            if not raw_labels:
                raise ValueError(f"{labels_path} must not be empty")
            if any(type(label) is not str for label in raw_labels):
                raise TypeError(f"{labels_path} entries must be exact strings")
            labels = cast(list[str], raw_labels)
            if len(set(labels)) != len(labels):
                raise ValueError(f"{labels_path} contains duplicate causes")
            if UNMARKED_CAUSE in labels:
                raise ValueError(f"{labels_path} contains an unmarked cause")
            unknown = [label for label in labels if label not in _CAUSE_LABEL_ORDER]
            if unknown:
                raise ValueError(f"{labels_path} contains unknown causes: {unknown}")
            canonical = [cause for cause in _CAUSE_LABEL_ORDER if cause in labels]
            if labels != canonical:
                raise ValueError(f"{labels_path} is not in canonical cause order")
            episode_causes.update(labels)

        direct = DIRECT_CAUSE in episode_causes
        cascade = CASCADE_CAUSE in episode_causes
        if direct and cascade:
            category = DIRECT_AND_CASCADE
        elif direct:
            category = DIRECT_ONLY
        elif cascade:
            category = CASCADE_ONLY
        else:  # Every nonempty label list is restricted to the two known causes.
            raise ValueError(f"{event_path} has no recognized acquisition cause")
        partition[category].append(step)

    return acquisition_steps, partition


def _derive_v4_target_lifecycle(
    *,
    target_names: Sequence[str],
    expected_post_steps: Sequence[int],
    acquisition_events_by_target: object,
    structural_lifecycle_by_target: object,
    admission_outcome_records: object,
) -> dict[str, object]:
    targets = _canonical_names(target_names, "target_names")
    expected_steps = _canonical_expected_steps(expected_post_steps, "expected_post_steps")
    acquisition_sources = _target_mapping(
        acquisition_events_by_target,
        "acquisition_events_by_target",
        targets,
    )
    structural_sources = _target_mapping(
        structural_lifecycle_by_target,
        "structural_lifecycle_by_target",
        targets,
    )
    admitted_steps = _admitted_steps_by_target(
        admission_outcome_records,
        target_names=targets,
        expected_steps=expected_steps,
    )

    target_payloads: dict[str, object] = {}
    for target in targets:
        acquisition_steps, partition = _acquisition_partition(
            acquisition_sources[target],
            target=target,
            expected_steps=expected_steps,
        )
        structural_path = f"structural_lifecycle_by_target.{target}"
        structural = _plain_dict(
            structural_sources[target],
            structural_path,
            _STRUCTURAL_SOURCE_FIELDS,
        )
        initially_present = structural["initially_present"]
        if type(initially_present) is not bool:
            raise TypeError(f"{structural_path}.initially_present must be an exact bool")
        if initially_present is not False:
            raise ValueError(f"{structural_path}.initially_present must be false")
        acquisitions = _exact_nonnegative_int(
            structural["acquisition_episode_count"],
            f"{structural_path}.acquisition_episode_count",
        )
        losses = _exact_nonnegative_int(
            structural["loss_episode_count"],
            f"{structural_path}.loss_episode_count",
        )
        end = structural["present_at_end"]
        if type(end) is not bool:
            raise TypeError(f"{structural_path}.present_at_end must be an exact bool")
        reacquisitions = _exact_nonnegative_int(
            structural["structural_reacquisition_count"],
            f"{structural_path}.structural_reacquisition_count",
        )

        direct_steps = admitted_steps[target]
        direct = len(direct_steps)
        if direct > acquisitions:
            raise ValueError(
                f"{target} direct admission episodes exceed structural acquisitions"
            )
        if acquisitions != len(acquisition_steps):
            raise ValueError(
                f"{target} structural acquisition count differs from exact events"
            )
        if acquisitions - losses != int(end):
            raise ValueError(f"{target} structural acquisition/loss/end lifecycle does not close")
        if reacquisitions != max(0, acquisitions - 1):
            raise ValueError(f"{target} structural reacquisition count does not close")

        direct_cause_steps = sorted(partition[DIRECT_ONLY] + partition[DIRECT_AND_CASCADE])
        if direct_steps != direct_cause_steps:
            raise ValueError(
                f"{target} direct admission steps differ from direct-cause acquisitions"
            )

        target_payloads[target] = {
            "initially_present": False,
            "direct_candidate_admission_episode_count": direct,
            "direct_candidate_admission_post_steps": list(direct_steps),
            "structural_acquisition_episode_count": acquisitions,
            "structural_acquisition_post_steps": list(acquisition_steps),
            "structural_loss_episode_count": losses,
            "structural_reacquisition_episode_count": reacquisitions,
            "present_at_end": end,
            "acquisition_episode_cause_partition": {
                category: {
                    "episode_count": len(partition[category]),
                    "post_steps": list(partition[category]),
                }
                for category in CAUSE_PARTITION_ORDER
            },
        }

    return {
        "schema": COMPOSITIONAL_FUTURE_UTILITY_V4_LIFECYCLE_SCHEMA,
        "target_order": list(targets),
        "opportunity_post_steps": list(expected_steps),
        "targets": target_payloads,
    }


def build_v4_target_lifecycle(
    *,
    target_names: Sequence[str],
    expected_post_steps: Sequence[int],
    acquisition_events_by_target: object,
    structural_lifecycle_by_target: object,
    admission_outcome_records: object,
) -> dict[str, object]:
    """Produce and validate the strict v4 per-target lifecycle payload."""

    payload = _derive_v4_target_lifecycle(
        target_names=target_names,
        expected_post_steps=expected_post_steps,
        acquisition_events_by_target=acquisition_events_by_target,
        structural_lifecycle_by_target=structural_lifecycle_by_target,
        admission_outcome_records=admission_outcome_records,
    )
    return validate_v4_target_lifecycle(payload)


def _validate_partition_record(
    value: object,
    *,
    path: str,
    allowed_steps: frozenset[int],
) -> list[int]:
    record = _plain_dict(value, path, ("episode_count", "post_steps"))
    count = _exact_nonnegative_int(record["episode_count"], f"{path}.episode_count")
    steps = _exact_step_list(
        record["post_steps"],
        f"{path}.post_steps",
        allowed_steps=allowed_steps,
    )
    if count != len(steps):
        raise ValueError(f"{path}.episode_count differs from post_steps")
    return steps


def validate_v4_target_lifecycle(value: object) -> dict[str, object]:
    """Fail closed on any noncanonical or internally inconsistent v4 payload."""

    root = _plain_dict(value, "lifecycle", _ROOT_FIELDS)
    if type(root["schema"]) is not str:
        raise TypeError("lifecycle.schema must be an exact str")
    if root["schema"] != COMPOSITIONAL_FUTURE_UTILITY_V4_LIFECYCLE_SCHEMA:
        raise ValueError("lifecycle.schema is not the exact v4 lifecycle schema")

    raw_target_order = _plain_list(root["target_order"], "lifecycle.target_order")
    target_names = _canonical_names(
        cast(Sequence[str], raw_target_order),
        "lifecycle.target_order",
    )
    raw_expected_steps = _plain_list(
        root["opportunity_post_steps"],
        "lifecycle.opportunity_post_steps",
    )
    expected_steps = _canonical_expected_steps(
        cast(Sequence[int], raw_expected_steps),
        "lifecycle.opportunity_post_steps",
    )
    expected_set = frozenset(expected_steps)
    targets = _target_mapping(root["targets"], "lifecycle.targets", target_names)

    for target in target_names:
        path = f"lifecycle.targets.{target}"
        record = _plain_dict(targets[target], path, _TARGET_FIELDS)
        initially_present = record["initially_present"]
        if type(initially_present) is not bool:
            raise TypeError(f"{path}.initially_present must be an exact bool")
        if initially_present is not False:
            raise ValueError(f"{path}.initially_present must be false")
        direct = _exact_nonnegative_int(
            record["direct_candidate_admission_episode_count"],
            f"{path}.direct_candidate_admission_episode_count",
        )
        direct_steps = _exact_step_list(
            record["direct_candidate_admission_post_steps"],
            f"{path}.direct_candidate_admission_post_steps",
            allowed_steps=expected_set,
        )
        acquisitions = _exact_nonnegative_int(
            record["structural_acquisition_episode_count"],
            f"{path}.structural_acquisition_episode_count",
        )
        acquisition_steps = _exact_step_list(
            record["structural_acquisition_post_steps"],
            f"{path}.structural_acquisition_post_steps",
            allowed_steps=expected_set,
        )
        losses = _exact_nonnegative_int(
            record["structural_loss_episode_count"],
            f"{path}.structural_loss_episode_count",
        )
        reacquisitions = _exact_nonnegative_int(
            record["structural_reacquisition_episode_count"],
            f"{path}.structural_reacquisition_episode_count",
        )
        end = record["present_at_end"]
        if type(end) is not bool:
            raise TypeError(f"{path}.present_at_end must be an exact bool")

        if direct != len(direct_steps):
            raise ValueError(f"{path} direct count differs from direct post_steps")
        if acquisitions != len(acquisition_steps):
            raise ValueError(f"{path} acquisition count differs from acquisition post_steps")
        if direct > acquisitions:
            raise ValueError(f"{path} direct episodes exceed structural acquisitions")
        if acquisitions - losses != int(end):
            raise ValueError(f"{path} structural acquisition/loss/end lifecycle does not close")
        if reacquisitions != max(0, acquisitions - 1):
            raise ValueError(f"{path} structural reacquisition count does not close")

        partition = _plain_dict(
            record["acquisition_episode_cause_partition"],
            f"{path}.acquisition_episode_cause_partition",
            CAUSE_PARTITION_ORDER,
        )
        partition_steps = {
            category: _validate_partition_record(
                partition[category],
                path=f"{path}.acquisition_episode_cause_partition.{category}",
                allowed_steps=expected_set,
            )
            for category in CAUSE_PARTITION_ORDER
        }
        flattened = [
            step
            for category in CAUSE_PARTITION_ORDER
            for step in partition_steps[category]
        ]
        if len(set(flattened)) != len(flattened):
            raise ValueError(f"{path} acquisition cause partition overlaps")
        if sorted(flattened) != acquisition_steps:
            raise ValueError(f"{path} acquisition cause partition does not close")
        expected_direct_steps = sorted(
            partition_steps[DIRECT_ONLY] + partition_steps[DIRECT_AND_CASCADE]
        )
        if direct_steps != expected_direct_steps:
            raise ValueError(f"{path} direct-cause episode partition does not close")

    return root


def validate_v4_target_lifecycle_against_sources(
    value: object,
    *,
    target_names: Sequence[str],
    expected_post_steps: Sequence[int],
    acquisition_events_by_target: object,
    structural_lifecycle_by_target: object,
    admission_outcome_records: object,
) -> dict[str, object]:
    """Validate a payload and require exact equality to freshly derived sources."""

    validated = validate_v4_target_lifecycle(value)
    expected = _derive_v4_target_lifecycle(
        target_names=target_names,
        expected_post_steps=expected_post_steps,
        acquisition_events_by_target=acquisition_events_by_target,
        structural_lifecycle_by_target=structural_lifecycle_by_target,
        admission_outcome_records=admission_outcome_records,
    )
    if validated != expected:
        raise ValueError("lifecycle differs from exact source-derived v4 lifecycle")
    return validated
