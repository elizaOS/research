"""Synthetic contract tests for reward-agnostic matched Forager statistics."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_statistics as statistics_module
from alberta_framework.benchmarks.forager_matched_statistics import (
    PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
    BootstrapSpec,
    ComparisonSpec,
    DescriptiveDiagnosticScores,
    EvidenceBinding,
    LearningMethodScores,
    MatchedComparisonContract,
    MatchedStatisticsError,
    PermutationSpec,
    analyze_matched_scores,
    canonical_payload_sha256,
    holm_adjust,
    load_canonical_result,
    paired_differences,
    paired_percentile_bootstrap_lower_bound,
    paired_sign_flip_test,
    primary_bootstrap_implementation_descriptor,
    secondary_sign_flip_holm_implementation_descriptor,
    validate_result_payload,
)

pytestmark = pytest.mark.unit

SEEDS = (101, 202, 303, 404)
EVIDENCE = EvidenceBinding(
    horizon=1_000,
    metric_sha256="1" * 64,
    environment_sha256="2" * 64,
    rng_schedule_sha256="3" * 64,
    runtime_profile_sha256="4" * 64,
    source_evidence_sha256="5" * 64,
    executor_evidence_sha256="6" * 64,
    score_evidence_sha256="7" * 64,
    execution_closure_sha256="8" * 64,
    authenticated_bindings_sha256="9" * 64,
    external_verification_subject_sha256="e" * 64,
    external_verification_receipt_sha256="a" * 64,
    sealed_protocol_sha256="b" * 64,
    selection_result_sha256="c" * 64,
    selection_report_sha256="d" * 64,
)
BOOTSTRAP = BootstrapSpec(resamples=257, seed=7_001, confidence=0.95)
PERMUTATION = PermutationSpec(
    monte_carlo_resamples=512,
    seed=8_001,
    familywise_alpha=0.05,
)


def _method(
    method_id: str,
    scores: tuple[float, ...],
    *,
    seeds: tuple[int, ...] = SEEDS,
    evidence: EvidenceBinding = EVIDENCE,
    preregistered: bool = True,
) -> LearningMethodScores:
    return LearningMethodScores(
        method_id=method_id,
        seeds=seeds,
        scores=scores,
        evidence=evidence,
        preregistered=preregistered,
    )


def _contract(
    *,
    alberta_scores: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
    primary_scores: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    secondary: tuple[LearningMethodScores, ...] = (),
    diagnostics: tuple[DescriptiveDiagnosticScores, ...] = (),
    margin: float = 0.0,
) -> MatchedComparisonContract:
    alberta = _method("alberta_candidate_v1", alberta_scores)
    primary = _method("primary_learning_baseline", primary_scores)
    return MatchedComparisonContract(
        methods=(alberta, primary, *secondary),
        primary_comparison=ComparisonSpec(
            hypothesis_id="primary_superiority",
            intervention_id=alberta.method_id,
            comparator_id=primary.method_id,
        ),
        secondary_comparisons=tuple(
            ComparisonSpec(
                hypothesis_id=method.method_id,
                intervention_id=alberta.method_id,
                comparator_id=method.method_id,
            )
            for method in secondary
        ),
        fixed_descriptive_diagnostics=diagnostics,
        bootstrap=BOOTSTRAP,
        permutation=PERMUTATION,
        primary_margin=margin,
        primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
        secondary_analysis_implementation_sha256=(SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256),
    )


def test_bootstrap_uses_paired_pcg64_and_one_sided_lower_quantile() -> None:
    differences = (0.0, 1.0, 2.0, 10.0)
    result = paired_percentile_bootstrap_lower_bound(differences, BOOTSTRAP)

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP.seed))
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP.resamples, len(differences)),
        dtype=np.int64,
    )
    distribution = np.asarray(differences, dtype=np.float64)[indices].mean(axis=1)
    expected = float(np.quantile(distribution, 0.05, method="linear"))

    assert result.lower_bound == expected
    assert result.alpha == pytest.approx(0.05)
    assert result.estimate == pytest.approx(3.25)
    assert result == paired_percentile_bootstrap_lower_bound(differences, BOOTSTRAP)


def test_analysis_implementation_descriptors_are_hash_bound_and_semantically_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_bootstrap_implementation_descriptor()
    secondary = secondary_sign_flip_holm_implementation_descriptor()

    assert canonical_payload_sha256(primary) == PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
    assert canonical_payload_sha256(secondary) == SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
    assert primary["paired_difference_arithmetic"] == ("binary64_subtraction_then_finite_check")
    assert primary["downstream_arithmetic"] == ("exact_dyadic_fraction_then_binary64")
    assert secondary["paired_difference_arithmetic"] == ("binary64_subtraction_then_finite_check")
    multiplicity = secondary["multiplicity"]
    assert isinstance(multiplicity, dict)
    assert multiplicity["alpha_conversion"] == "Fraction.from_float(binary64_alpha)"

    contract = _contract()
    implementations = contract.to_payload()["analysis_implementations"]
    assert isinstance(implementations, dict)
    assert implementations["primary"] == {
        "descriptor": primary,
        "implementation_sha256": PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    }
    assert implementations["secondary"] == {
        "descriptor": secondary,
        "implementation_sha256": SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
    }
    with pytest.raises(MatchedStatisticsError, match="unknown primary"):
        replace(contract, primary_analysis_implementation_sha256="0" * 64)
    with pytest.raises(MatchedStatisticsError, match="unknown secondary"):
        replace(contract, secondary_analysis_implementation_sha256="0" * 64)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            statistics_module,
            "_BOOTSTRAP_CHUNK_ELEMENTS",
            statistics_module._BOOTSTRAP_CHUNK_ELEMENTS + 1,
        )
        with pytest.raises(MatchedStatisticsError, match="primary.*descriptor drifted"):
            contract.to_payload()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            statistics_module,
            "_SIGN_FLIP_CHUNK_ELEMENTS",
            statistics_module._SIGN_FLIP_CHUNK_ELEMENTS + 1,
        )
        with pytest.raises(MatchedStatisticsError, match="secondary.*descriptor drifted"):
            contract.to_payload()


def test_bootstrap_mean_is_exact_for_finite_float_cancellation() -> None:
    differences = (1e16, 1.0, -1e16)
    result = paired_percentile_bootstrap_lower_bound(differences, BOOTSTRAP)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP.seed))
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP.resamples, len(differences)),
        dtype=np.int64,
    )
    exact_values = tuple(Fraction.from_float(value) for value in differences)
    oracle_distribution = [
        float(
            sum(
                (exact_values[int(index)] for index in row),
                start=Fraction(0, 1),
            )
            / len(differences)
        )
        for row in indices
    ]

    assert result.estimate == float(Fraction(1, 3))
    assert result.lower_bound == float(np.quantile(oracle_distribution, 0.05, method="linear"))


def test_primary_gate_is_strict_and_margin_is_nonnegative() -> None:
    equal_boundary = _contract(
        alberta_scores=(1.25, 2.25, 3.25, 4.25),
        primary_scores=(1.0, 2.0, 3.0, 4.0),
        margin=0.25,
    )
    result = analyze_matched_scores(equal_boundary)
    assert result.primary.bootstrap.lower_bound == 0.25
    assert result.primary.superiority_passed is False

    with pytest.raises(MatchedStatisticsError, match="nonnegative"):
        replace(equal_boundary, primary_margin=-0.01)


def test_exact_seed_order_is_required_for_every_learning_method() -> None:
    alberta = _method("alberta", (1.0, 2.0, 3.0, 4.0))
    primary = _method("primary", (0.0, 1.0, 2.0, 3.0))
    reordered = _method(
        "reordered",
        (2.0, 1.0, 3.0, 4.0),
        seeds=(202, 101, 303, 404),
    )
    with pytest.raises(MatchedStatisticsError, match="exact common seed ordering"):
        MatchedComparisonContract(
            methods=(alberta, primary, reordered),
            primary_comparison=ComparisonSpec("primary_h", "alberta", "primary"),
            secondary_comparisons=(ComparisonSpec("secondary_h", "reordered", "primary"),),
            fixed_descriptive_diagnostics=(),
            bootstrap=BOOTSTRAP,
            permutation=PERMUTATION,
            primary_margin=0.0,
            primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
            secondary_analysis_implementation_sha256=(
                SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
            ),
        )


@pytest.mark.parametrize(
    "changed_evidence",
    [
        replace(EVIDENCE, horizon=1_001),
        replace(EVIDENCE, metric_sha256="a" * 64),
        replace(EVIDENCE, environment_sha256="b" * 64),
        replace(EVIDENCE, rng_schedule_sha256="c" * 64),
        replace(EVIDENCE, runtime_profile_sha256="d" * 64),
        replace(EVIDENCE, source_evidence_sha256="e" * 64),
        replace(EVIDENCE, executor_evidence_sha256="f" * 64),
        replace(EVIDENCE, score_evidence_sha256="e" * 64),
        replace(EVIDENCE, execution_closure_sha256="f" * 64),
        replace(EVIDENCE, authenticated_bindings_sha256="e" * 64),
        replace(EVIDENCE, external_verification_receipt_sha256="f" * 64),
        replace(EVIDENCE, sealed_protocol_sha256="e" * 64),
        replace(EVIDENCE, selection_result_sha256="f" * 64),
        replace(EVIDENCE, selection_report_sha256="e" * 64),
    ],
)
def test_each_evidence_binding_mismatch_fails_closed(
    changed_evidence: EvidenceBinding,
) -> None:
    with pytest.raises(MatchedStatisticsError, match="different evidence binding"):
        MatchedComparisonContract(
            methods=(
                _method("alberta", (2.0, 3.0, 4.0, 5.0)),
                _method(
                    "primary",
                    (1.0, 2.0, 3.0, 4.0),
                    evidence=changed_evidence,
                ),
            ),
            primary_comparison=ComparisonSpec("primary_h", "alberta", "primary"),
            secondary_comparisons=(),
            fixed_descriptive_diagnostics=(),
            bootstrap=BOOTSTRAP,
            permutation=PERMUTATION,
            primary_margin=0.0,
            primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
            secondary_analysis_implementation_sha256=(
                SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
            ),
        )


def test_evidence_binding_uses_digests_not_caller_trust_booleans() -> None:
    payload = EVIDENCE.to_payload()
    assert "source_trusted" not in payload
    assert "executor_trusted" not in payload
    assert payload["source_evidence_sha256"] == "5" * 64
    assert payload["executor_evidence_sha256"] == "6" * 64
    assert payload["score_evidence_sha256"] == "7" * 64
    assert payload["execution_closure_sha256"] == "8" * 64
    assert payload["authenticated_bindings_sha256"] == "9" * 64
    assert payload["external_verification_receipt_sha256"] == "a" * 64
    assert payload["sealed_protocol_sha256"] == "b" * 64
    assert payload["selection_result_sha256"] == "c" * 64
    assert payload["selection_report_sha256"] == "d" * 64

    with pytest.raises(MatchedStatisticsError, match="source_evidence_sha256"):
        replace(EVIDENCE, source_evidence_sha256="trusted")


def test_primary_comparison_and_preregistered_methods_are_required() -> None:
    alberta = _method("alberta", (2.0, 3.0, 4.0, 5.0))
    primary = _method("primary", (1.0, 2.0, 3.0, 4.0))

    def build(
        methods: tuple[LearningMethodScores, ...],
        comparison: ComparisonSpec = ComparisonSpec("primary_h", "alberta", "primary"),
    ) -> MatchedComparisonContract:
        return MatchedComparisonContract(
            methods=methods,
            primary_comparison=comparison,
            secondary_comparisons=(),
            fixed_descriptive_diagnostics=(),
            bootstrap=BOOTSTRAP,
            permutation=PERMUTATION,
            primary_margin=0.0,
            primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
            secondary_analysis_implementation_sha256=(
                SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
            ),
        )

    with pytest.raises(MatchedStatisticsError, match="must not be empty"):
        build(())
    with pytest.raises(MatchedStatisticsError, match="unknown methods"):
        build((alberta, primary), ComparisonSpec("primary_h", "alberta", "missing"))
    with pytest.raises(MatchedStatisticsError, match="not preregistered"):
        build((alberta, replace(primary, preregistered=False)))


def test_comparison_ids_and_unordered_pairs_are_unique() -> None:
    methods = (
        _method("a", (3.0, 3.0, 3.0, 3.0)),
        _method("b", (2.0, 2.0, 2.0, 2.0)),
    )
    with pytest.raises(MatchedStatisticsError, match="regardless of direction"):
        MatchedComparisonContract(
            methods=methods,
            primary_comparison=ComparisonSpec("a_vs_b", "a", "b"),
            secondary_comparisons=(ComparisonSpec("b_vs_a", "b", "a"),),
            fixed_descriptive_diagnostics=(),
            bootstrap=BOOTSTRAP,
            permutation=PERMUTATION,
            primary_margin=0.0,
            primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
            secondary_analysis_implementation_sha256=(
                SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
            ),
        )


def test_fixed_descriptive_candidates_retain_order_and_truthful_exclusion_reasons() -> None:
    shared_rng = DescriptiveDiagnosticScores(
        candidate_id="exact_ppo",
        seeds=SEEDS,
        scores=(1e300,) * len(SEEDS),
        exclusion_reasons=("shared_agent_environment_rng",),
    )
    privileged = DescriptiveDiagnosticScores(
        candidate_id="search_oracle",
        seeds=SEEDS,
        scores=(-1e300,) * len(SEEDS),
        exclusion_reasons=("privileged_observation_access",),
    )
    changed = replace(shared_rng, scores=(-1e300,) * len(SEEDS))
    high_result = analyze_matched_scores(_contract(diagnostics=(shared_rng, privileged)))
    low_result = analyze_matched_scores(_contract(diagnostics=(changed, privileged)))

    assert high_result.primary == low_result.primary
    assert high_result.secondary == low_result.secondary == ()
    assert high_result.payload_sha256 != low_result.payload_sha256
    assert tuple(item.candidate_id for item in high_result.fixed_descriptive_exclusions) == (
        "exact_ppo",
        "search_oracle",
    )
    assert high_result.fixed_descriptive_exclusions[0].exclusion_reasons == (
        "shared_agent_environment_rng",
    )
    assert high_result.fixed_descriptive_exclusions[1].exclusion_reasons == (
        "privileged_observation_access",
    )
    assert "PrivilegedDiagnosticScores" not in statistics_module.__all__
    assert "PrivilegedDiagnosticExclusion" not in statistics_module.__all__
    assert not hasattr(statistics_module, "PrivilegedDiagnosticScores")
    assert not hasattr(statistics_module, "PrivilegedDiagnosticExclusion")

    forged = replace(
        high_result.fixed_descriptive_exclusions[0],
        exclusion_reasons=("privileged_observation_access",),
    )
    with pytest.raises(MatchedStatisticsError, match="reasons do not match"):
        replace(
            high_result,
            fixed_descriptive_exclusions=(
                forged,
                *high_result.fixed_descriptive_exclusions[1:],
            ),
        )


def test_fixed_descriptive_inputs_require_common_seeds_and_nonempty_unique_reasons() -> None:
    diagnostic = DescriptiveDiagnosticScores(
        candidate_id="exact_ppo",
        seeds=SEEDS,
        scores=(1.0,) * len(SEEDS),
        exclusion_reasons=("shared_agent_environment_rng",),
    )
    with pytest.raises(MatchedStatisticsError, match="exact common seed ordering"):
        _contract(diagnostics=(replace(diagnostic, seeds=(1, 2, 3, 4)),))
    with pytest.raises(MatchedStatisticsError, match="must not be empty"):
        replace(diagnostic, exclusion_reasons=())
    with pytest.raises(MatchedStatisticsError, match="unique reasons"):
        replace(
            diagnostic,
            exclusion_reasons=(
                "shared_agent_environment_rng",
                "shared_agent_environment_rng",
            ),
        )


def test_exact_sign_flip_enumerates_all_assignments_and_includes_ties() -> None:
    all_positive = paired_sign_flip_test((1.0, 1.0, 1.0), PERMUTATION)
    assert all_positive.mode == "exact"
    assert (all_positive.extreme_count, all_positive.p_numerator, all_positive.p_denominator) == (
        1,
        1,
        8,
    )

    tied = paired_sign_flip_test((1.0, 0.0, -1.0), PERMUTATION)
    assert tied.nonzero_pairs == 2
    assert (tied.extreme_count, tied.p_numerator, tied.p_denominator) == (6, 6, 8)
    assert tied.p_value == 0.75

    all_zero = paired_sign_flip_test((0.0, 0.0, 0.0), PERMUTATION)
    assert (all_zero.extreme_count, all_zero.p_denominator, all_zero.p_value) == (8, 8, 1.0)
    assert paired_sign_flip_test((-1.0,), PERMUTATION).p_value == 1.0


@pytest.mark.parametrize(
    ("differences", "expected_extreme"),
    [
        ((1e16, 1.0, -1e16), 4),
        ((1e16, 1.0, 1.0, -1e16), 6),
    ],
)
def test_exact_sign_flip_is_dyadically_exact_under_cancellation(
    differences: tuple[float, ...], expected_extreme: int
) -> None:
    result = paired_sign_flip_test(differences, PERMUTATION)
    exact_values = tuple(Fraction.from_float(value) for value in differences)
    oracle_count = 0
    for mask in range(1 << len(differences)):
        subset_sum = sum(
            (value for index, value in enumerate(exact_values) if mask & (1 << index)),
            start=Fraction(0, 1),
        )
        oracle_count += int(subset_sum <= 0)

    assert oracle_count == expected_extreme
    assert result.extreme_count == oracle_count


def test_exact_twenty_pair_boundary_uses_full_enumeration() -> None:
    result = paired_sign_flip_test((1.0,) * 20, PERMUTATION)
    assert result.mode == "exact"
    assert result.p_numerator == 1
    assert result.p_denominator == 1 << 20


def test_monte_carlo_sign_flip_is_deterministic_and_plus_one_corrected() -> None:
    differences = (1.0,) * 21
    first = paired_sign_flip_test(differences, PERMUTATION)
    second = paired_sign_flip_test(differences, PERMUTATION)

    assert first == second
    assert first.mode == "monte_carlo"
    assert first.extreme_count == 0
    assert first.p_numerator == 1
    assert first.p_denominator == PERMUTATION.monte_carlo_resamples + 1
    assert first.p_value == 1.0 / (PERMUTATION.monte_carlo_resamples + 1)


def test_monte_carlo_masks_match_independent_pcg64_exact_arithmetic_oracle() -> None:
    differences = (1e16, 1.0, -1e16, *((0.25, -0.5, 0.75) * 6))
    assert len(differences) == 21
    result = paired_sign_flip_test(differences, PERMUTATION)

    rng = np.random.Generator(np.random.PCG64(PERMUTATION.seed))
    masks = rng.integers(
        0,
        2,
        size=(PERMUTATION.monte_carlo_resamples, len(differences)),
        dtype=np.uint8,
    )
    exact_values = tuple(Fraction.from_float(value) for value in differences)
    oracle_extreme = 0
    for mask in masks:
        subset_sum = sum(
            (value for value, is_negative in zip(exact_values, mask, strict=True) if is_negative),
            start=Fraction(0, 1),
        )
        oracle_extreme += int(subset_sum <= 0)

    assert result.mode == "monte_carlo"
    assert result.extreme_count == oracle_extreme
    assert (result.p_numerator, result.p_denominator) == (
        oracle_extreme + 1,
        PERMUTATION.monte_carlo_resamples + 1,
    )


def test_holm_uses_exact_rationals_stable_order_and_inclusive_boundary() -> None:
    decisions = holm_adjust(
        ("first", "second", "third"),
        ((1, 100), (4, 100), (3, 100)),
        0.05,
    )
    assert [decision.rank for decision in decisions] == [1, 3, 2]
    assert [
        (decision.adjusted_numerator, decision.adjusted_denominator) for decision in decisions
    ] == [(3, 100), (3, 50), (3, 50)]
    assert [decision.reject for decision in decisions] == [True, False, False]

    boundary = holm_adjust(("edge", "other"), ((1, 40), (1, 1)), 0.05)
    assert (boundary[0].adjusted_numerator, boundary[0].adjusted_denominator) == (1, 20)
    assert boundary[0].reject is True

    ties = holm_adjust(("earlier", "later"), ((1, 100), (1, 100)), 0.05)
    assert [decision.rank for decision in ties] == [1, 2]


def test_secondary_order_is_preserved_and_does_not_change_primary_gate() -> None:
    secondaries = (
        _method("secondary_b", (1.0, 2.0, 3.0, 4.0)),
        _method("secondary_a", (1.5, 2.5, 3.5, 4.5)),
    )
    without = analyze_matched_scores(_contract())
    with_secondary = analyze_matched_scores(_contract(secondary=secondaries))

    assert with_secondary.primary == without.primary
    assert tuple(item.comparison.hypothesis_id for item in with_secondary.secondary) == (
        "secondary_b",
        "secondary_a",
    )


def test_secondary_comparison_uses_its_explicit_intervention_and_comparator() -> None:
    methods = (
        _method("reference", (100.0, 100.0, 100.0, 100.0)),
        _method("external", (1.0, 1.0, 1.0, 1.0)),
        _method("rtu", (3.0, 3.0, 3.0, 3.0)),
    )
    contract = MatchedComparisonContract(
        methods=methods,
        primary_comparison=ComparisonSpec("reference_vs_external", "reference", "external"),
        secondary_comparisons=(ComparisonSpec("rtu_vs_external", "rtu", "external"),),
        fixed_descriptive_diagnostics=(),
        bootstrap=BOOTSTRAP,
        permutation=PERMUTATION,
        primary_margin=0.0,
        primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
        secondary_analysis_implementation_sha256=(SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256),
    )

    result = analyze_matched_scores(contract)

    assert result.secondary[0].comparison == ComparisonSpec("rtu_vs_external", "rtu", "external")
    assert result.secondary[0].sign_flip.observed_mean == 2.0
    assert result.secondary[0].holm.hypothesis_id == "rtu_vs_external"


def test_difference_properties_hold_for_synthetic_finite_inputs() -> None:
    rng = np.random.Generator(np.random.PCG64(91))
    for length in range(1, 12):
        candidate = tuple(float(value) for value in rng.normal(size=length))
        baseline = tuple(float(value) for value in rng.normal(size=length))
        forward = paired_differences(candidate, baseline)
        reverse = paired_differences(baseline, candidate)
        assert reverse == tuple(-difference for difference in forward)
        assert paired_differences(candidate, candidate) == (0.0,) * length


@pytest.mark.parametrize(
    "invalid_scores",
    [
        (1.0, float("nan"), 3.0, 4.0),
        (1.0, float("inf"), 3.0, 4.0),
        (1.0, 2, 3.0, 4.0),
        (1.0, np.float64(2.0), 3.0, 4.0),
        (1.0, 2.0 + 0.0j, 3.0, 4.0),
    ],
)
def test_nonfinite_or_non_builtin_float_scores_fail_closed(
    invalid_scores: tuple[object, ...],
) -> None:
    with pytest.raises(MatchedStatisticsError):
        LearningMethodScores(
            method_id="invalid",
            seeds=SEEDS,
            scores=invalid_scores,  # type: ignore[arg-type]
            evidence=EVIDENCE,
        )


def test_strict_types_shapes_and_arithmetic_overflow_fail_closed() -> None:
    with pytest.raises(MatchedStatisticsError, match="tuple"):
        LearningMethodScores(
            method_id="list_scores",
            seeds=SEEDS,
            scores=[1.0, 2.0, 3.0, 4.0],  # type: ignore[arg-type]
            evidence=EVIDENCE,
        )
    with pytest.raises(MatchedStatisticsError, match="unique"):
        LearningMethodScores(
            method_id="duplicate_seeds",
            seeds=(1, 1, 2, 3),
            scores=(1.0, 2.0, 3.0, 4.0),
            evidence=EVIDENCE,
        )
    with pytest.raises(MatchedStatisticsError, match="equal lengths"):
        paired_differences((1.0,), (1.0, 2.0))
    with pytest.raises(MatchedStatisticsError, match="not finite"):
        paired_differences((1e308,), (-1e308,))
    with pytest.raises(MatchedStatisticsError, match="must be a float"):
        replace(_contract(), primary_margin=0)
    with pytest.raises(MatchedStatisticsError, match="maximize"):
        replace(_contract(), metric_direction="minimize")  # type: ignore[arg-type]
    with pytest.raises(MatchedStatisticsError, match="maximize"):
        replace(
            _contract(),
            metric_direction=np.str_("maximize"),  # type: ignore[arg-type]
        )


def test_contract_and_records_are_frozen() -> None:
    contract = _contract()
    with pytest.raises(FrozenInstanceError):
        setattr(contract, "primary_margin", 2.0)
    with pytest.raises(FrozenInstanceError):
        setattr(contract.methods[0], "scores", (0.0, 0.0, 0.0, 0.0))


def test_result_is_canonical_hash_bound_and_contains_no_host_metadata() -> None:
    result = analyze_matched_scores(_contract())
    raw = result.canonical_json()
    payload = json.loads(raw)
    body = dict(payload)
    claimed = body.pop("payload_sha256")
    independently_encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert claimed == hashlib.sha256(independently_encoded).hexdigest()
    assert claimed == result.payload_sha256
    assert load_canonical_result(raw, result.contract) == result
    assert payload["schema"] == "alberta.forager_matched_statistics.result.v3"
    assert payload["contract"]["schema"] == ("alberta.forager_matched_statistics.contract.v3")
    text = raw.decode("utf-8").lower()
    assert "/home/" not in text
    assert "timestamp" not in text
    assert "generated_at" not in text
    assert "scores_hex" not in text
    assert "paired_differences_hex" not in text
    assert '"raw_scores_or_differences_embedded":false' in text

    pending: list[object] = [payload]
    serialized_keys: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            serialized_keys.update(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert "scores" not in serialized_keys
    assert "paired_differences" not in serialized_keys


def test_tampering_fails_with_original_or_recomputed_self_hash() -> None:
    result = analyze_matched_scores(_contract())
    payload = json.loads(result.canonical_json())
    unchanged_hash_tamper = copy.deepcopy(payload)
    unchanged_hash_tamper["primary_superiority_passed"] = False
    with pytest.raises(MatchedStatisticsError, match="SHA-256 mismatch"):
        validate_result_payload(unchanged_hash_tamper, result.contract)

    rehashed_tamper = copy.deepcopy(unchanged_hash_tamper)
    rehashed_body = dict(rehashed_tamper)
    rehashed_body.pop("payload_sha256")
    rehashed_tamper["payload_sha256"] = canonical_payload_sha256(rehashed_body)
    with pytest.raises(MatchedStatisticsError, match="does not replay"):
        validate_result_payload(rehashed_tamper, result.contract)


def test_result_replay_uses_raw_scores_from_the_caller_held_contract() -> None:
    contract = _contract()
    result = analyze_matched_scores(contract)
    changed_method = replace(
        contract.methods[0],
        scores=(2.0, 3.0, 4.0, 5.000000000000001),
    )
    changed_contract = replace(
        contract,
        methods=(changed_method, *contract.methods[1:]),
    )

    with pytest.raises(MatchedStatisticsError, match="does not replay"):
        validate_result_payload(result.to_payload(), changed_contract)


def test_result_construction_replays_statistics_and_rejects_forgery() -> None:
    primary_result = analyze_matched_scores(_contract())
    forged_bootstrap = replace(
        primary_result.primary.bootstrap,
        lower_bound=123.0,
        distribution_sha256="0" * 64,
    )
    forged_primary = replace(
        primary_result.primary,
        bootstrap=forged_bootstrap,
        superiority_passed=True,
    )
    with pytest.raises(MatchedStatisticsError, match="bootstrap result does not replay"):
        replace(primary_result, primary=forged_primary)

    secondary_method = _method("secondary", (1.0, 2.0, 3.0, 4.0))
    secondary_result = analyze_matched_scores(_contract(secondary=(secondary_method,)))
    original_secondary = secondary_result.secondary[0]
    forged_sign_flip = replace(
        original_secondary.sign_flip,
        extreme_count=2,
        p_numerator=2,
        p_value=0.125,
    )
    forged_holm = holm_adjust(("secondary",), ((2, 16),), 0.05)[0]
    forged_secondary = replace(
        original_secondary,
        sign_flip=forged_sign_flip,
        holm=forged_holm,
    )
    with pytest.raises(MatchedStatisticsError, match="sign-flip result does not replay"):
        replace(secondary_result, secondary=(forged_secondary,))


def test_result_replay_distinguishes_signed_zero_in_canonical_floats() -> None:
    seeds = (1,)
    contract = MatchedComparisonContract(
        methods=(
            _method("alberta_zero", (-0.0,), seeds=seeds),
            _method("baseline_zero", (0.0,), seeds=seeds),
        ),
        primary_comparison=ComparisonSpec("zero_comparison", "alberta_zero", "baseline_zero"),
        secondary_comparisons=(),
        fixed_descriptive_diagnostics=(),
        bootstrap=BOOTSTRAP,
        permutation=PERMUTATION,
        primary_margin=-0.0,
        primary_analysis_implementation_sha256=(PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
        secondary_analysis_implementation_sha256=(SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256),
    )
    result = analyze_matched_scores(contract)
    positive_contract = replace(
        contract,
        methods=(
            _method("alberta_zero", (0.0,), seeds=seeds),
            _method("baseline_zero", (0.0,), seeds=seeds),
        ),
    )
    positive_result = analyze_matched_scores(positive_contract)
    assert result.primary.paired_differences_sha256 != (
        positive_result.primary.paired_differences_sha256
    )

    positive_difference = replace(
        result.primary,
        paired_differences_sha256=positive_result.primary.paired_differences_sha256,
    )
    with pytest.raises(MatchedStatisticsError, match="primary difference digest"):
        replace(result, primary=positive_difference)

    positive_margin = replace(result.primary, frozen_margin=0.0)
    with pytest.raises(MatchedStatisticsError, match="primary margin"):
        replace(result, primary=positive_margin)


def test_loader_rejects_noncanonical_duplicate_and_nonfinite_json() -> None:
    contract = _contract()
    result = analyze_matched_scores(contract)
    pretty = json.dumps(result.to_payload(), indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(MatchedStatisticsError, match="canonical"):
        load_canonical_result(pretty, contract)
    with pytest.raises(MatchedStatisticsError, match="duplicate JSON key"):
        load_canonical_result(b'{"schema":"x","schema":"y"}', contract)
    with pytest.raises(MatchedStatisticsError, match="nonfinite JSON constant"):
        load_canonical_result(b'{"value":NaN}', contract)


def test_loader_bounds_raw_bytes_nodes_and_nesting_depth() -> None:
    contract = _contract()
    oversized = b" " * (statistics_module._MAX_CANONICAL_RESULT_BYTES + 1)
    with pytest.raises(MatchedStatisticsError, match="maximum byte length"):
        load_canonical_result(oversized, contract)

    too_many_nodes = (
        b'{"items":['
        + b",".join(b"0" for _ in range(statistics_module._MAX_CANONICAL_RESULT_NODES))
        + b"]}"
    )
    with pytest.raises(MatchedStatisticsError, match="too many nodes"):
        load_canonical_result(too_many_nodes, contract)

    nesting = statistics_module._MAX_CANONICAL_RESULT_DEPTH
    too_deep = b'{"item":' + b"[" * nesting + b"0" + b"]" * nesting + b"}"
    with pytest.raises(MatchedStatisticsError, match="nesting depth"):
        load_canonical_result(too_deep, contract)


@pytest.mark.parametrize(
    "decoder_error",
    [RecursionError("synthetic"), ValueError("synthetic"), OverflowError("synthetic")],
)
def test_loader_normalizes_decoder_exceptions(
    monkeypatch: pytest.MonkeyPatch, decoder_error: Exception
) -> None:
    def fail_decode(*_args: object, **_kwargs: object) -> object:
        raise decoder_error

    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_matched_statistics.json.loads",
        fail_decode,
    )
    with pytest.raises(MatchedStatisticsError, match="not valid JSON"):
        load_canonical_result(b"{}", _contract())


def test_any_input_change_changes_contract_and_result_hashes() -> None:
    original = _contract()
    changed = _contract(alberta_scores=(2.0, 3.0, 4.0, 5.000000000000001))
    assert original.payload_sha256 != changed.payload_sha256
    assert (
        analyze_matched_scores(original).payload_sha256
        != analyze_matched_scores(changed).payload_sha256
    )
