"""Strict synthetic tests for the Forager v3 named-panel analysis."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction

import pytest

from alberta_framework.benchmarks import forager_matched_v3_statistics as statistics_module
from alberta_framework.benchmarks.forager_matched_v3_statistics import (
    CONTRAST_RANGE_WIDTH,
    HORIZON,
    INPUT_SCHEMA,
    MAX_BLOCKS_PER_CANDIDATE,
    MAX_CANDIDATES,
    MAX_SCORE_CELLS,
    RAW_STEP_REWARDS,
    RESULT_SCHEMA,
    SCORE_MAXIMUM,
    SCORE_MINIMUM,
    InferentialScores,
    NamedPanelAnalysisInput,
    V3StatisticsError,
    analyze_named_panel,
    canonical_payload_sha256,
    load_canonical_input,
    load_canonical_result,
    validate_input_payload,
    validate_result_payload,
)

pytestmark = pytest.mark.unit


def test_analysis_implementation_has_an_independent_literal_pin() -> None:
    assert statistics_module.ANALYSIS_IMPLEMENTATION_SHA256 == (
        "558f21ec06d5b588f3724aa3d384d4be08ad27eb7c779398c556badc6e92aec9"
    )


def _row(
    candidate_id: str,
    scores: tuple[int, ...],
    *,
    blocks: tuple[str, ...] | None = None,
) -> InferentialScores:
    if blocks is None:
        blocks = tuple(f"block_{index}" for index in range(len(scores)))
    return InferentialScores(candidate_id=candidate_id, block_ids=blocks, scores=scores)


def _input(
    *rows: InferentialScores,
    alpha: float = 0.05,
    delta: float = 0.0,
) -> NamedPanelAnalysisInput:
    if not rows:
        rows = (
            _row("candidate_b", (8, 9, 10, 11)),
            _row("candidate_a", (1, 2, 3, 4)),
        )
    return NamedPanelAnalysisInput(inferential_scores=rows, alpha=alpha, delta=delta)


def _with_digest(body: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(body)
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def test_empirical_bernstein_formula_matches_exact_hand_oracle() -> None:
    panel = _input(
        _row("candidate_b", (7, 5, 13, 2)),
        _row("candidate_a", (1, 4, 3, 8)),
        alpha=0.125,
        delta=0.25,
    )
    result = analyze_named_panel(panel)
    contrast = result.contrast("candidate_b", "candidate_a")

    differences = (6, 1, 10, -6)
    n_blocks = len(differences)
    family_size = 2
    q = math.log(2 * family_size / panel.alpha)
    mean = Fraction(sum(differences), n_blocks)
    pair_sum = sum(
        (differences[first] - differences[second]) ** 2
        for first in range(n_blocks)
        for second in range(first + 1, n_blocks)
    )
    variance = Fraction(pair_sum, n_blocks * (n_blocks - 1))
    lower = (
        float(mean)
        - math.sqrt(2.0 * float(variance) * q / n_blocks)
        - 7.0 * CONTRAST_RANGE_WIDTH * q / (3.0 * (n_blocks - 1))
    )

    assert result.family_size == 2
    assert result.q == q
    assert contrast.mean_difference == float(mean)
    assert contrast.sample_variance == float(variance)
    assert contrast.lower_bound == lower


def test_zero_variance_and_maximum_variance_are_computed_exactly() -> None:
    zero = analyze_named_panel(
        _input(
            _row("higher", (100, 101, 102)),
            _row("lower", (1, 2, 3)),
        )
    ).contrast("higher", "lower")
    assert zero.mean_difference == 99.0
    assert zero.sample_variance == 0.0

    maximum = analyze_named_panel(
        _input(
            _row("swing", (SCORE_MAXIMUM, SCORE_MINIMUM)),
            _row("inverse", (SCORE_MINIMUM, SCORE_MAXIMUM)),
        )
    ).contrast("swing", "inverse")
    assert maximum.mean_difference == 0.0
    assert maximum.sample_variance == float(Fraction(CONTRAST_RANGE_WIDTH**2, 2))


def test_integer_cancellation_uses_exact_mean_and_pair_variance() -> None:
    result = analyze_named_panel(
        _input(
            _row("candidate", (SCORE_MAXIMUM, SCORE_MINIMUM, 1)),
            _row("baseline", (SCORE_MINIMUM, SCORE_MAXIMUM, 0)),
        )
    )
    contrast = result.contrast("candidate", "baseline")
    differences = (31 * HORIZON, -31 * HORIZON, 1)
    pair_sum = sum(
        (differences[first] - differences[second]) ** 2
        for first in range(3)
        for second in range(first + 1, 3)
    )

    assert contrast.mean_difference == float(Fraction(1, 3))
    assert contrast.sample_variance == float(Fraction(pair_sum, 6))


def test_full_ordered_family_preserves_declared_candidate_order() -> None:
    panel = _input(
        _row("candidate_z", (5, 6)),
        _row("candidate_a", (2, 3)),
        _row("candidate_m", (0, 1)),
    )
    result = analyze_named_panel(panel)

    assert result.inferential_ids == ("candidate_z", "candidate_a", "candidate_m")
    assert tuple(
        (contrast.intervention_id, contrast.comparator_id)
        for contrast in result.ordered_contrasts
    ) == (
        ("candidate_z", "candidate_a"),
        ("candidate_z", "candidate_m"),
        ("candidate_a", "candidate_z"),
        ("candidate_a", "candidate_m"),
        ("candidate_m", "candidate_z"),
        ("candidate_m", "candidate_a"),
    )


def test_sample_leader_ties_break_by_candidate_id_not_declared_order() -> None:
    result = analyze_named_panel(
        _input(
            _row("candidate_z", (4, 6, 8)),
            _row("candidate_a", (3, 6, 9)),
            _row("candidate_m", (0, 0, 0)),
        )
    )
    assert result.sample_leader_id == "candidate_a"


def test_detached_result_rejects_leader_inconsistent_with_sample_means() -> None:
    result = analyze_named_panel(_input())
    wrong_leader = next(
        candidate_id
        for candidate_id in result.inferential_ids
        if candidate_id != result.sample_leader_id
    )

    with pytest.raises(V3StatisticsError, match="sample means"):
        replace(result, sample_leader_id=wrong_leader)


def test_named_panel_claim_uses_strict_delta_boundary() -> None:
    blocks = tuple(f"block_{index}" for index in range(128))
    initial = _input(
        _row("leader", (10_000_000,) * len(blocks), blocks=blocks),
        _row("other", (0,) * len(blocks), blocks=blocks),
    )
    initial_result = analyze_named_panel(initial)
    boundary = initial_result.contrast("leader", "other").lower_bound
    assert boundary > 0.0

    equal = analyze_named_panel(replace(initial, delta=boundary))
    below = analyze_named_panel(replace(initial, delta=math.nextafter(boundary, -math.inf)))
    assert equal.named_panel_superiority_claim_passed is False
    assert below.named_panel_superiority_claim_passed is True


def test_post_selection_uses_every_ordered_contrast_in_family_size() -> None:
    panel = _input(
        _row("leader", (30, 31, 32)),
        _row("middle", (20, 21, 22)),
        _row("low", (10, 11, 12)),
        _row("lowest", (0, 1, 2)),
        alpha=0.2,
    )
    result = analyze_named_panel(panel)
    expected_family_size = 4 * 3

    assert len(result.ordered_contrasts) == expected_family_size
    assert result.family_size == expected_family_size
    assert result.q == math.log(2 * expected_family_size / panel.alpha)
    assert result.to_body()["family_scope"] == "every_ordered_contrast_in_fixed_named_panel"


def test_alpha_and_panel_size_drift_fail_result_replay() -> None:
    panel = _input()
    payload = analyze_named_panel(panel).to_payload()

    with pytest.raises(V3StatisticsError, match="replay"):
        validate_result_payload(payload, replace(panel, alpha=0.1))
    expanded = _input(
        *panel.inferential_scores,
        _row("candidate_c", (0, 0, 0, 0)),
        alpha=panel.alpha,
        delta=panel.delta,
    )
    with pytest.raises(V3StatisticsError, match="replay"):
        validate_result_payload(payload, expanded)


@pytest.mark.parametrize(
    ("changed_blocks", "match"),
    [
        (("block_0", "block_1", "block_2"), "same length"),
        (
            ("block_0", "block_1", "block_2", "block_3", "block_4"),
            "same length",
        ),
        (("block_0", "block_1", "block_1", "block_3"), "unique"),
        (
            ("block_1", "block_0", "block_2", "block_3"),
            "exact common ordered block IDs",
        ),
    ],
)
def test_missing_extra_duplicate_or_reordered_blocks_fail_closed(
    changed_blocks: tuple[str, ...], match: str
) -> None:
    first = _row("first", (1, 2, 3, 4))
    scores: tuple[int, ...] = (4, 3, 2, 1)
    if len(changed_blocks) == 5:
        scores = (*scores, 0)
    with pytest.raises(V3StatisticsError, match=match):
        _input(first, _row("second", scores, blocks=changed_blocks))


@pytest.mark.parametrize("bad_score", [True, False, 1.0, -1.0])
def test_bool_and_float_score_aliases_are_rejected(bad_score: object) -> None:
    with pytest.raises(V3StatisticsError, match="score.*must be an int"):
        InferentialScores(
            candidate_id="candidate",
            block_ids=("block_0", "block_1"),
            scores=(0, bad_score),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_score", [SCORE_MINIMUM - 1, SCORE_MAXIMUM + 1])
def test_cumulative_score_bounds_are_enforced(bad_score: int) -> None:
    with pytest.raises(V3StatisticsError, match="score bounds"):
        _row("candidate", (0, bad_score))


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, math.inf, -math.inf, math.nan])
def test_alpha_must_be_finite_and_strictly_between_zero_and_one(bad_alpha: float) -> None:
    with pytest.raises(V3StatisticsError, match="alpha"):
        _input(alpha=bad_alpha)


@pytest.mark.parametrize("bad_delta", [-0.1, math.inf, -math.inf, math.nan])
def test_delta_must_be_finite_and_nonnegative(bad_delta: float) -> None:
    with pytest.raises(V3StatisticsError, match="delta"):
        _input(delta=bad_delta)


def test_negative_zero_delta_alias_is_rejected() -> None:
    with pytest.raises(V3StatisticsError, match="negative zero"):
        _input(delta=-0.0)


def test_at_least_two_unique_candidates_and_blocks_are_required() -> None:
    with pytest.raises(V3StatisticsError, match="at least two inferential"):
        _input(_row("only", (1, 2)))
    with pytest.raises(V3StatisticsError, match="at least two blocks"):
        _input(_row("first", (1,)), _row("second", (0,)))
    with pytest.raises(V3StatisticsError, match="unique"):
        _input(_row("same", (1, 2)), _row("same", (0, 1)))


def test_result_is_detached_and_explicitly_disclaims_broader_claims() -> None:
    panel = _input()
    result = analyze_named_panel(panel)
    payload = result.to_payload()
    encoded = result.canonical_json()

    assert payload["schema"] == RESULT_SCHEMA
    assert payload["input_sha256"] == panel.payload_sha256
    assert payload["raw_scores_or_differences_embedded"] is False
    assert b'"scores"' not in encoded
    assert b'"block_ids"' not in encoded
    interpretation = payload["interpretation"]
    assert isinstance(interpretation, dict)
    assert interpretation == {
        "compute_matched_claim_authorized": False,
        "evidence_promotion_authorized": False,
        "literature_best_claim_authorized": False,
        "standalone_claim_authorized": False,
        "universal_sota_claim_authorized": False,
    }
    assumptions = payload["assumptions"]
    assert isinstance(assumptions, dict)
    assert assumptions["independent_blocks"] == "required_not_verified"
    assert assumptions["identically_distributed_blocks"] == "required_not_verified"
    assert assumptions["panel_fixed_before_analysis"] == "required_not_verified"
    assert (
        assumptions["cumulative_scores_aggregate_declared_horizon_rewards"]
        == "required_not_verified"
    )


def test_input_and_result_canonical_round_trip_and_replay() -> None:
    panel = _input()
    input_raw = panel.canonical_json()
    result = analyze_named_panel(panel)

    assert panel.to_payload()["schema"] == INPUT_SCHEMA
    assert load_canonical_input(input_raw) == panel
    assert validate_input_payload(panel.to_payload()) == panel
    assert load_canonical_result(result.canonical_json(), panel) == result
    assert validate_result_payload(result.to_payload(), panel) == result


def test_input_construction_enforces_loader_byte_and_node_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = _input()
    raw_size = len(ordinary.canonical_json())

    with monkeypatch.context() as patcher:
        patcher.setattr(statistics_module, "_MAX_CANONICAL_BYTES", raw_size - 1)
        with pytest.raises(V3StatisticsError, match="loader byte limit"):
            _input()

    with monkeypatch.context() as patcher:
        patcher.setattr(statistics_module, "_MAX_CANONICAL_NODES", 8)
        with pytest.raises(V3StatisticsError, match="too many nodes"):
            _input()


def test_input_cardinality_caps_bound_linear_validation_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_CANDIDATES >= 2
    assert MAX_BLOCKS_PER_CANDIDATE >= 2
    assert MAX_SCORE_CELLS >= 4

    with monkeypatch.context() as patcher:
        patcher.setattr(statistics_module, "MAX_CANDIDATES", 2)
        with pytest.raises(V3StatisticsError, match="candidate count exceeds"):
            _input(
                _row("candidate_a", (1, 2)),
                _row("candidate_b", (1, 2)),
                _row("candidate_c", (1, 2)),
            )

    with monkeypatch.context() as patcher:
        patcher.setattr(statistics_module, "MAX_BLOCKS_PER_CANDIDATE", 1)
        with pytest.raises(V3StatisticsError, match="block count exceeds"):
            _input(_row("candidate_a", (1, 2)), _row("candidate_b", (1, 2)))

    with monkeypatch.context() as patcher:
        patcher.setattr(statistics_module, "MAX_SCORE_CELLS", 3)
        with pytest.raises(V3StatisticsError, match="score matrix exceeds"):
            _input(_row("candidate_a", (1, 2)), _row("candidate_b", (1, 2)))

def test_result_construction_enforces_loader_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _input()
    result_size = len(analyze_named_panel(panel).canonical_json())

    monkeypatch.setattr(statistics_module, "_MAX_CANONICAL_BYTES", result_size - 1)
    with pytest.raises(V3StatisticsError, match="loader byte limit"):
        analyze_named_panel(panel)


def test_exact_key_schemas_reject_legacy_inference_fields_even_with_new_digest() -> None:
    panel = _input()
    input_payload = panel.to_payload()
    input_payload["bootstrap"] = {"confidence": 0.95}
    with pytest.raises(V3StatisticsError, match="exact keys"):
        validate_input_payload(input_payload)

    body = analyze_named_panel(panel).to_body()
    body["sign_flip"] = {"p_value": 0.01}
    payload = _with_digest(body)
    with pytest.raises(V3StatisticsError, match="exact keys"):
        validate_result_payload(payload, panel)


def test_canonical_loaders_reject_duplicate_keys_and_noncanonical_bytes() -> None:
    panel = _input()
    raw = panel.canonical_json()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    noncanonical = json.dumps(parsed, indent=2, sort_keys=True).encode()
    with pytest.raises(V3StatisticsError, match="canonical"):
        load_canonical_input(noncanonical)

    duplicate = raw[:-1] + b',"schema":"' + INPUT_SCHEMA.encode() + b'"}'
    with pytest.raises(V3StatisticsError, match="duplicate JSON key"):
        load_canonical_input(duplicate)


def test_digest_and_replay_reject_detached_result_tampering() -> None:
    panel = _input()
    payload = analyze_named_panel(panel).to_payload()
    payload["sample_leader_id"] = "candidate_bogus"
    with pytest.raises(V3StatisticsError, match="SHA-256"):
        validate_result_payload(payload, panel)

    body = copy.deepcopy(payload)
    body.pop("payload_sha256")
    forged = _with_digest(body)
    with pytest.raises(V3StatisticsError, match="replay"):
        validate_result_payload(forged, panel)


def test_input_payload_digest_is_canonical_sha256() -> None:
    panel = _input()
    body = panel.to_body()
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert panel.payload_sha256 == expected


def test_fixed_scientific_constants_are_pinned() -> None:
    assert HORIZON == 499_712
    assert RAW_STEP_REWARDS == (-1, 0, 1, 30)
    assert SCORE_MINIMUM == -HORIZON
    assert SCORE_MAXIMUM == 30 * HORIZON
    assert CONTRAST_RANGE_WIDTH == 62 * HORIZON
