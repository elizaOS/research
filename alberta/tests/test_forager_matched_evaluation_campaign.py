"""Contract tests for
:mod:`alberta_framework.benchmarks.forager_matched_evaluation_campaign`.

The module under test is pure, reward-free scheduling for the sealed
evaluation block: given a sealed protocol and a validated open-to-sealed
transition, it builds the seed-major 6-candidate x 30-seed schedule (the
four tuning-selected arms followed by the two fixed descriptive arms) and
the canonical transition descriptor.  Tests pin the exact cell layout and
digest bindings and require self-consistent tampering, bad self-hashes,
non-canonical bytes, wrong stages, and drifted transitions to fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from tests import test_forager_matched_open_protocol as protocol_fixtures

pytestmark = pytest.mark.unit


def _canonical_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def sealed_context() -> tuple[
    protocol.ForagerMatchedProtocol,
    protocol.ForagerMatchedProtocol,
    protocol.SealedProtocolValidation,
]:
    open_value = protocol_fixtures._build()
    result_payload = {
        "schema_version": protocol.FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
        "open_protocol_sha256": open_value.protocol_sha256,
        "selection_plan_sha256": open_value.selection_plan.plan_sha256,
        "tuning_seeds": list(open_value.tuning_seeds),
        "ranked_groups": [
            {
                "selection_group": group.selection_group,
                # Rank each group in *reverse* declaration order so the
                # synthetic winners differ from the panel's declared order —
                # any code that confuses declaration order with the selection
                # ranking fails the schedule/transition assertions below.
                "ranked_candidate_ids": list(reversed(group.candidate_ids)),
                "ranking_evidence_sha256": hashlib.sha256(
                    f"sealed-ranking:{group.selection_group}".encode("ascii")
                ).hexdigest(),
            }
            for group in open_value.selection_plan.groups
        ],
    }
    selection_result = protocol.parse_forager_matched_selection_result(result_payload)
    sealed = protocol.seal_forager_matched_protocol(open_value, selection_result)
    transition = protocol.validate_sealed_protocol_transition(
        open_value,
        sealed,
        selection_result,
        selection_result.selection_result_sha256,
    )
    return open_value, sealed, transition


def test_schedule_is_exact_seed_major_six_by_thirty(
    sealed_context: tuple[
        protocol.ForagerMatchedProtocol,
        protocol.ForagerMatchedProtocol,
        protocol.SealedProtocolValidation,
    ],
) -> None:
    _, sealed, transition = sealed_context
    schedule = evaluation.build_sealed_evaluation_schedule(sealed, transition)
    cells = schedule["cells"]

    assert schedule["stage"] == "sealed_evaluation"
    assert schedule["candidate_order"] == list(transition.evaluation_candidate_ids)
    assert schedule["active_seeds"] == list(sealed.evaluation_seeds)
    assert len(schedule["candidate_order"]) == 6
    assert len(schedule["active_seeds"]) == 30
    assert len(cells) == 180
    assert [cell["ordinal"] for cell in cells] == list(range(180))
    for seed_index, seed in enumerate(sealed.evaluation_seeds):
        block = cells[seed_index * 6 : (seed_index + 1) * 6]
        assert [cell["candidate_id"] for cell in block] == list(
            transition.evaluation_candidate_ids
        )
        assert [cell["seed"] for cell in block] == [seed] * 6
    assert {cell["seed"] for cell in cells}.isdisjoint(sealed.tuning_seeds)

    unsigned = dict(schedule)
    del unsigned["schedule_sha256"]
    assert schedule["schedule_sha256"] == _canonical_sha256(unsigned)
    assert schedule["promotion_authorized"] is False
    assert schedule["external_verification_required"] is True


def test_schedule_preserves_selected_four_then_fixed_two_and_transition_binding(
    sealed_context: tuple[
        protocol.ForagerMatchedProtocol,
        protocol.ForagerMatchedProtocol,
        protocol.SealedProtocolValidation,
    ],
) -> None:
    _, sealed, transition = sealed_context
    schedule = evaluation.build_sealed_evaluation_schedule(sealed, transition)
    selected = transition.evaluation_candidate_ids[:4]
    fixed = transition.evaluation_candidate_ids[4:]

    assert selected == tuple(
        item.candidate_id for item in transition.resolved_slots
    )
    assert fixed == open_protocol.MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS
    assert fixed == sealed.evaluation_panel.fixed_descriptive_candidate_ids
    assert all(sealed.candidate_index[item].pairing.eligible for item in selected)
    assert all(
        not sealed.candidate_index[item].pairing.eligible for item in fixed
    )

    descriptor = evaluation.build_sealed_transition_descriptor(sealed, transition)
    assert descriptor["schema_version"] == (
        evaluation.MATCHED_SEALED_TRANSITION_SCHEMA_VERSION
    )
    assert descriptor["evaluation_candidate_ids"] == list(
        transition.evaluation_candidate_ids
    )
    assert descriptor["sealed_protocol_sha256"] == sealed.protocol_sha256
    descriptor_digest = evaluation.canonical_sealed_transition_descriptor_sha256(
        sealed, transition
    )
    assert descriptor_digest == _canonical_sha256(descriptor)
    assert schedule["sealed_transition_sha256"] == descriptor_digest


@pytest.mark.parametrize(
    "tamper",
    (
        "protocol",
        "transition",
        "candidate_order",
        "active_seeds",
        "cell",
    ),
)
def test_parser_rejects_self_consistent_schedule_tampering(
    sealed_context: tuple[
        protocol.ForagerMatchedProtocol,
        protocol.ForagerMatchedProtocol,
        protocol.SealedProtocolValidation,
    ],
    tamper: str,
) -> None:
    _, sealed, transition = sealed_context
    changed = copy.deepcopy(
        evaluation.build_sealed_evaluation_schedule(sealed, transition)
    )
    if tamper == "protocol":
        changed["protocol_sha256"] = "0" * 64
    elif tamper == "transition":
        changed["sealed_transition_sha256"] = "0" * 64
    elif tamper == "candidate_order":
        changed["candidate_order"][0], changed["candidate_order"][1] = (
            changed["candidate_order"][1],
            changed["candidate_order"][0],
        )
    elif tamper == "active_seeds":
        changed["active_seeds"] = list(reversed(changed["active_seeds"]))
    else:
        changed["cells"][0]["candidate_id"] = changed["candidate_order"][1]
    unsigned = dict(changed)
    del unsigned["schedule_sha256"]
    changed["schedule_sha256"] = _canonical_sha256(unsigned)

    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="replayed sealed evaluation schedule",
    ):
        evaluation.parse_sealed_evaluation_schedule(
            changed,
            sealed_protocol=sealed,
            transition=transition,
        )


def test_parser_rejects_bad_self_hash_and_noncanonical_bytes(
    sealed_context: tuple[
        protocol.ForagerMatchedProtocol,
        protocol.ForagerMatchedProtocol,
        protocol.SealedProtocolValidation,
    ],
) -> None:
    _, sealed, transition = sealed_context
    schedule = evaluation.build_sealed_evaluation_schedule(sealed, transition)
    bad_digest = copy.deepcopy(schedule)
    bad_digest["schedule_sha256"] = "0" * 64
    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="self-hash",
    ):
        evaluation.parse_sealed_evaluation_schedule(
            bad_digest,
            sealed_protocol=sealed,
            transition=transition,
        )

    canonical = evaluation.canonical_sealed_evaluation_schedule_bytes(
        schedule,
        sealed_protocol=sealed,
        transition=transition,
        expected_schedule_sha256=schedule["schedule_sha256"],
    )
    parsed = evaluation.parse_sealed_evaluation_schedule(
        canonical,
        sealed_protocol=sealed,
        transition=transition,
    )
    assert parsed["schedule_sha256"] == schedule["schedule_sha256"]
    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="canonical encoding",
    ):
        evaluation.parse_sealed_evaluation_schedule(
            canonical + b"\n",
            sealed_protocol=sealed,
            transition=transition,
        )
    with pytest.raises(TypeError):
        parsed["stage"] = "open_tuning"  # type: ignore[index]


def test_builder_rejects_wrong_stage_and_wrong_typed_transition(
    sealed_context: tuple[
        protocol.ForagerMatchedProtocol,
        protocol.ForagerMatchedProtocol,
        protocol.SealedProtocolValidation,
    ],
) -> None:
    open_value, sealed, transition = sealed_context
    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="stage sealed_evaluation",
    ):
        evaluation.build_sealed_evaluation_schedule(open_value, transition)

    wrong_order = replace(
        transition,
        evaluation_candidate_ids=tuple(reversed(transition.evaluation_candidate_ids)),
    )
    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="evaluation_candidate_ids",
    ):
        evaluation.build_sealed_evaluation_schedule(sealed, wrong_order)

    wrong_result = replace(transition, selection_result_sha256="0" * 64)
    with pytest.raises(
        evaluation.ForagerMatchedEvaluationCampaignError,
        match="selection_result_sha256",
    ):
        evaluation.build_sealed_transition_descriptor(sealed, wrong_result)
