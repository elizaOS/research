"""Reward-agnostic paired statistics for a future matched Forager suite.

This module deliberately knows nothing about Forager rewards, agents, or execution.  It
accepts already-computed scalar scores whose metric is declared to be maximized.  Every
learning arm must carry the same ordered seeds and evidence binding, while privileged
diagnostics are represented separately and can never enter a superiority calculation.

The contract and all result objects are immutable.  Bootstrap and sign-flip randomness is
created explicitly with ``numpy.random.Generator(numpy.random.PCG64(seed))``.  A result is
canonical JSON with an embedded SHA-256 integrity digest; validation also replays the
analysis from the caller-supplied contract, so recomputing the digest after tampering is not
sufficient to make a modified result valid.  Raw score vectors and paired differences never
appear in serialized payloads; domain-separated SHA-256 digests bind the caller-held values.

Evidence bindings must be constructed from artifacts validated by an external protocol
transition and trust resolver.  A digest is an integrity identifier, not a signature, and
this module has no promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, NoReturn, cast

import numpy as np

CONTRACT_SCHEMA = "alberta.forager_matched_statistics.contract.v2"
RESULT_SCHEMA = "alberta.forager_matched_statistics.result.v2"
SCORE_VECTOR_SCHEMA = "alberta.forager_matched_statistics.score_vector.v1"
DIFFERENCE_VECTOR_SCHEMA = "alberta.forager_matched_statistics.difference_vector.v1"
COMPARISON_INPUT_SCHEMA = "alberta.forager_matched_statistics.comparison_input.v1"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan-floats-as-hex"
DIGEST_ALGORITHM = "sha256"
RNG_ALGORITHM = "PCG64"
QUANTILE_METHOD: Literal["linear"] = "linear"
EXACT_SIGN_FLIP_MAX_PAIRS = 20

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_UINT64_MAX = (1 << 64) - 1
_BOOTSTRAP_CHUNK_ELEMENTS = 2_000_000
_SIGN_FLIP_CHUNK_ELEMENTS = 1_000_000


class MatchedStatisticsError(ValueError):
    """Raised when a matched-comparison contract or result fails closed."""


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise MatchedStatisticsError(f"{name} must be a bool")
    return value


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise MatchedStatisticsError(f"{name} must be an int")
    result = value
    if result < minimum:
        raise MatchedStatisticsError(f"{name} must be >= {minimum}")
    return result


def _require_uint64(value: object, name: str) -> int:
    result = _require_exact_int(value, name)
    if result > _UINT64_MAX:
        raise MatchedStatisticsError(f"{name} must fit in an unsigned 64-bit integer")
    return result


def _require_finite_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise MatchedStatisticsError(f"{name} must be a float")
    result = value
    if not math.isfinite(result):
        raise MatchedStatisticsError(f"{name} must be finite")
    return result


def _require_probability(value: object, name: str) -> float:
    result = _require_finite_float(value, name)
    if not 0.0 < result < 1.0:
        raise MatchedStatisticsError(f"{name} must be strictly between 0 and 1")
    return result


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise MatchedStatisticsError(f"{name} must be a portable identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MatchedStatisticsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_seed_tuple(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise MatchedStatisticsError(f"{name} must be a tuple")
    seeds = cast(tuple[object, ...], value)
    if not seeds:
        raise MatchedStatisticsError(f"{name} must not be empty")
    checked = tuple(
        _require_exact_int(seed, f"{name}[{index}]")
        for index, seed in enumerate(seeds)
    )
    if len(set(checked)) != len(checked):
        raise MatchedStatisticsError(f"{name} must contain unique seeds")
    return checked


def _require_score_tuple(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise MatchedStatisticsError(f"{name} must be a tuple")
    scores = cast(tuple[object, ...], value)
    if not scores:
        raise MatchedStatisticsError(f"{name} must not be empty")
    return tuple(
        _require_finite_float(score, f"{name}[{index}]") for index, score in enumerate(scores)
    )


def _require_exact_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise MatchedStatisticsError(f"{name} must be a tuple")
    return cast(tuple[object, ...], value)


def _exact_dyadic_components(values: tuple[float, ...]) -> tuple[tuple[int, ...], int]:
    """Return exact integer numerators under one positive power-of-two denominator."""

    ratios = tuple(value.as_integer_ratio() for value in values)
    common_denominator = max(denominator for _, denominator in ratios)
    scaled = tuple(
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    )
    common_factor = common_denominator
    for value in scaled:
        common_factor = math.gcd(common_factor, abs(value))
    if common_factor > 1:
        return (
            tuple(value // common_factor for value in scaled),
            common_denominator // common_factor,
        )
    return scaled, common_denominator


def _finite_mean(values: tuple[float, ...], name: str) -> float:
    integer_values, common_denominator = _exact_dyadic_components(values)
    result = float(Fraction(sum(integer_values), common_denominator * len(values)))
    if not math.isfinite(result):
        raise MatchedStatisticsError(f"{name} must be finite")
    return result


def _float_hex(value: float) -> str:
    return value.hex()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MatchedStatisticsError("payload is not canonical-JSON encodable") from exc
    return text.encode("utf-8")


def canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 of a mapping under this module's canonical JSON encoding."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _score_vector_sha256(
    *,
    record_id: str,
    seeds: tuple[int, ...],
    scores: tuple[float, ...],
) -> str:
    return canonical_payload_sha256(
        {
            "record_id": record_id,
            "schema": SCORE_VECTOR_SCHEMA,
            "scores_hex": [_float_hex(score) for score in scores],
            "seeds": list(seeds),
        }
    )


def _difference_vector_sha256(
    *,
    comparison_id: str,
    differences: tuple[float, ...],
    seeds: tuple[int, ...],
) -> str:
    return canonical_payload_sha256(
        {
            "comparison_id": comparison_id,
            "differences_hex": [_float_hex(value) for value in differences],
            "schema": DIFFERENCE_VECTOR_SCHEMA,
            "seeds": list(seeds),
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """Digest-bound evidence identity shared by every inferential learning arm.

    The source and executor digests name externally validated evidence manifests; this pure
    module does not convert their presence into a trust assertion.
    """

    horizon: int
    metric_sha256: str
    environment_sha256: str
    rng_schedule_sha256: str
    runtime_profile_sha256: str
    source_evidence_sha256: str
    executor_evidence_sha256: str
    sealed_protocol_sha256: str

    def __post_init__(self) -> None:
        _require_exact_int(self.horizon, "horizon", minimum=1)
        _require_sha256(self.metric_sha256, "metric_sha256")
        _require_sha256(self.environment_sha256, "environment_sha256")
        _require_sha256(self.rng_schedule_sha256, "rng_schedule_sha256")
        _require_sha256(self.runtime_profile_sha256, "runtime_profile_sha256")
        _require_sha256(self.source_evidence_sha256, "source_evidence_sha256")
        _require_sha256(self.executor_evidence_sha256, "executor_evidence_sha256")
        _require_sha256(self.sealed_protocol_sha256, "sealed_protocol_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "environment_sha256": self.environment_sha256,
            "executor_evidence_sha256": self.executor_evidence_sha256,
            "horizon": self.horizon,
            "metric_sha256": self.metric_sha256,
            "rng_schedule_sha256": self.rng_schedule_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "sealed_protocol_sha256": self.sealed_protocol_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class LearningMethodScores:
    """Ordered, paired scores for one preregistered learning method."""

    method_id: str
    seeds: tuple[int, ...]
    scores: tuple[float, ...]
    evidence: EvidenceBinding
    preregistered: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.method_id, "method_id")
        seeds = _require_seed_tuple(self.seeds, "seeds")
        scores = _require_score_tuple(self.scores, "scores")
        if len(seeds) != len(scores):
            raise MatchedStatisticsError("seeds and scores must have the same length")
        if type(self.evidence) is not EvidenceBinding:
            raise MatchedStatisticsError("evidence must be an EvidenceBinding")
        _require_exact_bool(self.preregistered, "preregistered")

    @property
    def scores_sha256(self) -> str:
        return _score_vector_sha256(
            record_id=self.method_id,
            seeds=self.seeds,
            scores=self.scores,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_payload(),
            "method_id": self.method_id,
            "preregistered": self.preregistered,
            "score_count": len(self.scores),
            "scores_sha256": self.scores_sha256,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class PrivilegedDiagnosticScores:
    """Scores that are recorded only to prove exclusion from superiority tests."""

    diagnostic_id: str
    seeds: tuple[int, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.diagnostic_id, "diagnostic_id")
        seeds = _require_seed_tuple(self.seeds, "diagnostic seeds")
        scores = _require_score_tuple(self.scores, "diagnostic scores")
        if len(seeds) != len(scores):
            raise MatchedStatisticsError("diagnostic seeds and scores must have the same length")

    def to_payload(self) -> dict[str, object]:
        return {
            "diagnostic_id": self.diagnostic_id,
            "score_count": len(self.scores),
            "scores_sha256": _score_vector_sha256(
                record_id=self.diagnostic_id,
                seeds=self.seeds,
                scores=self.scores,
            ),
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    """Explicit bootstrap choices, frozen by the enclosing immutable contract."""

    resamples: int
    seed: int
    confidence: float

    def __post_init__(self) -> None:
        _require_exact_int(self.resamples, "bootstrap resamples", minimum=1)
        _require_uint64(self.seed, "bootstrap seed")
        _require_probability(self.confidence, "bootstrap confidence")

    def to_payload(self) -> dict[str, object]:
        return {
            "confidence_hex": _float_hex(self.confidence),
            "quantile_method": QUANTILE_METHOD,
            "resamples": self.resamples,
            "rng_algorithm": RNG_ALGORITHM,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PermutationSpec:
    """Explicit sign-flip and familywise-error choices."""

    monte_carlo_resamples: int
    seed: int
    familywise_alpha: float

    def __post_init__(self) -> None:
        _require_exact_int(
            self.monte_carlo_resamples,
            "sign-flip Monte Carlo resamples",
            minimum=1,
        )
        _require_uint64(self.seed, "sign-flip seed")
        _require_probability(self.familywise_alpha, "familywise_alpha")

    def to_payload(self) -> dict[str, object]:
        return {
            "alternative": "greater",
            "exact_max_pairs": EXACT_SIGN_FLIP_MAX_PAIRS,
            "familywise_alpha_hex": _float_hex(self.familywise_alpha),
            "monte_carlo_resamples": self.monte_carlo_resamples,
            "rng_algorithm": RNG_ALGORITHM,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """One explicit ordered, one-sided paired comparison."""

    hypothesis_id: str
    intervention_id: str
    comparator_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.hypothesis_id, "hypothesis_id")
        _require_identifier(self.intervention_id, "intervention_id")
        _require_identifier(self.comparator_id, "comparator_id")
        if self.intervention_id == self.comparator_id:
            raise MatchedStatisticsError("a comparison cannot use the same method twice")

    @property
    def unordered_pair(self) -> tuple[str, str]:
        first, second = sorted((self.intervention_id, self.comparator_id))
        return first, second

    def to_payload(self) -> dict[str, object]:
        return {
            "alternative": "greater",
            "comparator_id": self.comparator_id,
            "difference_order": "intervention_minus_comparator",
            "hypothesis_id": self.hypothesis_id,
            "intervention_id": self.intervention_id,
        }


@dataclass(frozen=True, slots=True)
class MatchedComparisonContract:
    """Complete immutable preregistration for explicit paired comparisons."""

    methods: tuple[LearningMethodScores, ...]
    primary_comparison: ComparisonSpec
    secondary_comparisons: tuple[ComparisonSpec, ...]
    privileged_diagnostics: tuple[PrivilegedDiagnosticScores, ...]
    bootstrap: BootstrapSpec
    permutation: PermutationSpec
    primary_margin: float
    metric_direction: Literal["maximize"] = "maximize"

    def __post_init__(self) -> None:
        method_objects = _require_exact_tuple(self.methods, "methods")
        if not method_objects:
            raise MatchedStatisticsError("methods must not be empty")
        secondary_objects = _require_exact_tuple(
            self.secondary_comparisons, "secondary_comparisons"
        )
        diagnostic_objects = _require_exact_tuple(
            self.privileged_diagnostics, "privileged_diagnostics"
        )
        for index, method in enumerate(method_objects):
            if type(method) is not LearningMethodScores:
                raise MatchedStatisticsError(f"learning method {index} has the wrong type")
        methods = cast(tuple[LearningMethodScores, ...], method_objects)
        if type(self.primary_comparison) is not ComparisonSpec:
            raise MatchedStatisticsError("primary_comparison must be a ComparisonSpec")
        for index, comparison in enumerate(secondary_objects):
            if type(comparison) is not ComparisonSpec:
                raise MatchedStatisticsError(f"secondary comparison {index} has the wrong type")
        secondary = cast(tuple[ComparisonSpec, ...], secondary_objects)

        for index, diagnostic in enumerate(diagnostic_objects):
            if type(diagnostic) is not PrivilegedDiagnosticScores:
                raise MatchedStatisticsError(f"privileged diagnostic {index} has the wrong type")
        diagnostics = cast(tuple[PrivilegedDiagnosticScores, ...], diagnostic_objects)

        if type(self.bootstrap) is not BootstrapSpec:
            raise MatchedStatisticsError("bootstrap must be a BootstrapSpec")
        if type(self.permutation) is not PermutationSpec:
            raise MatchedStatisticsError("permutation must be a PermutationSpec")
        margin = _require_finite_float(self.primary_margin, "primary_margin")
        if margin < 0.0:
            raise MatchedStatisticsError("primary_margin must be nonnegative")
        if type(self.metric_direction) is not str or self.metric_direction != "maximize":
            raise MatchedStatisticsError("metric_direction must be 'maximize'")

        expected_seeds = methods[0].seeds
        expected_evidence = methods[0].evidence
        for method in methods:
            if not method.preregistered:
                raise MatchedStatisticsError(
                    f"learning method {method.method_id!r} is not preregistered"
                )
            if method.seeds != expected_seeds:
                raise MatchedStatisticsError(
                    f"learning method {method.method_id!r} does not have the exact common "
                    "seed ordering"
                )
            if method.evidence != expected_evidence:
                raise MatchedStatisticsError(
                    f"learning method {method.method_id!r} has a different evidence binding"
                )

        method_ids = tuple(method.method_id for method in methods)
        diagnostic_ids = tuple(diagnostic.diagnostic_id for diagnostic in diagnostics)
        all_ids = (*method_ids, *diagnostic_ids)
        if len(set(all_ids)) != len(all_ids):
            raise MatchedStatisticsError("learning and diagnostic identifiers must be unique")
        method_id_set = set(method_ids)
        comparisons = (self.primary_comparison, *secondary)
        hypothesis_ids = tuple(comparison.hypothesis_id for comparison in comparisons)
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise MatchedStatisticsError("comparison hypothesis identifiers must be unique")
        unordered_pairs = tuple(comparison.unordered_pair for comparison in comparisons)
        if len(set(unordered_pairs)) != len(unordered_pairs):
            raise MatchedStatisticsError(
                "comparison endpoint pairs must be unique regardless of direction"
            )
        for comparison in comparisons:
            unknown = {
                comparison.intervention_id,
                comparison.comparator_id,
            } - method_id_set
            if unknown:
                raise MatchedStatisticsError(
                    f"comparison {comparison.hypothesis_id!r} references unknown methods: "
                    + ", ".join(sorted(unknown))
                )

    @property
    def common_seeds(self) -> tuple[int, ...]:
        return self.methods[0].seeds

    @property
    def evidence(self) -> EvidenceBinding:
        return self.methods[0].evidence

    def method(self, method_id: str) -> LearningMethodScores:
        for method in self.methods:
            if method.method_id == method_id:
                return method
        raise MatchedStatisticsError(f"unknown method identifier {method_id!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "bootstrap": self.bootstrap.to_payload(),
            "canonicalization": CANONICALIZATION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "methods": [method.to_payload() for method in self.methods],
            "metric_direction": self.metric_direction,
            "permutation": self.permutation.to_payload(),
            "primary_comparison": self.primary_comparison.to_payload(),
            "primary_margin_hex": _float_hex(self.primary_margin),
            "privileged_diagnostics": [
                diagnostic.to_payload() for diagnostic in self.privileged_diagnostics
            ],
            "schema": CONTRACT_SCHEMA,
            "secondary_comparisons": [
                comparison.to_payload() for comparison in self.secondary_comparisons
            ],
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_payload_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class BootstrapLowerBound:
    n_pairs: int
    estimate: float
    lower_bound: float
    alpha: float
    confidence: float
    resamples: int
    seed: int
    distribution_sha256: str

    def __post_init__(self) -> None:
        _require_exact_int(self.n_pairs, "n_pairs", minimum=1)
        _require_finite_float(self.estimate, "bootstrap estimate")
        _require_finite_float(self.lower_bound, "bootstrap lower_bound")
        _require_probability(self.alpha, "bootstrap alpha")
        _require_probability(self.confidence, "bootstrap confidence")
        if self.alpha != 1.0 - self.confidence:
            raise MatchedStatisticsError("bootstrap alpha must equal one minus confidence")
        _require_exact_int(self.resamples, "bootstrap resamples", minimum=1)
        _require_uint64(self.seed, "bootstrap seed")
        _require_sha256(self.distribution_sha256, "bootstrap distribution_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "alpha_hex": _float_hex(self.alpha),
            "confidence_hex": _float_hex(self.confidence),
            "distribution_sha256": self.distribution_sha256,
            "estimate_hex": _float_hex(self.estimate),
            "lower_bound_hex": _float_hex(self.lower_bound),
            "n_pairs": self.n_pairs,
            "quantile_method": QUANTILE_METHOD,
            "resamples": self.resamples,
            "rng_algorithm": RNG_ALGORITHM,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    n_pairs: int
    nonzero_pairs: int
    observed_mean: float
    mode: Literal["exact", "monte_carlo"]
    extreme_count: int
    p_numerator: int
    p_denominator: int
    p_value: float
    draws: int
    seed: int
    plus_one_correction: bool

    def __post_init__(self) -> None:
        n_pairs = _require_exact_int(self.n_pairs, "n_pairs", minimum=1)
        nonzero_pairs = _require_exact_int(self.nonzero_pairs, "nonzero_pairs")
        if nonzero_pairs > n_pairs:
            raise MatchedStatisticsError("nonzero_pairs cannot exceed n_pairs")
        _require_finite_float(self.observed_mean, "observed_mean")
        if type(self.mode) is not str or self.mode not in ("exact", "monte_carlo"):
            raise MatchedStatisticsError("invalid sign-flip mode")
        extreme = _require_exact_int(self.extreme_count, "extreme_count")
        numerator = _require_exact_int(self.p_numerator, "p_numerator", minimum=1)
        denominator = _require_exact_int(self.p_denominator, "p_denominator", minimum=1)
        if numerator > denominator:
            raise MatchedStatisticsError("p-value numerator cannot exceed its denominator")
        p_value = _require_probability_or_one(self.p_value, "p_value")
        if p_value != float(Fraction(numerator, denominator)):
            raise MatchedStatisticsError("p_value does not match its exact rational value")
        draws = _require_exact_int(self.draws, "draws", minimum=1)
        _require_uint64(self.seed, "sign-flip seed")
        plus_one = _require_exact_bool(self.plus_one_correction, "plus_one_correction")
        if self.mode == "exact":
            if n_pairs > EXACT_SIGN_FLIP_MAX_PAIRS or plus_one:
                raise MatchedStatisticsError("invalid exact sign-flip metadata")
            if draws != 1 << n_pairs or denominator != draws or numerator != extreme:
                raise MatchedStatisticsError("invalid exact sign-flip counts")
        else:
            if n_pairs <= EXACT_SIGN_FLIP_MAX_PAIRS or not plus_one:
                raise MatchedStatisticsError("invalid Monte Carlo sign-flip metadata")
            if denominator != draws + 1 or numerator != extreme + 1:
                raise MatchedStatisticsError("invalid Monte Carlo sign-flip counts")

    def to_payload(self) -> dict[str, object]:
        return {
            "alternative": "greater",
            "draws": self.draws,
            "extreme_count": self.extreme_count,
            "mode": self.mode,
            "n_pairs": self.n_pairs,
            "nonzero_pairs": self.nonzero_pairs,
            "p_denominator": self.p_denominator,
            "p_numerator": self.p_numerator,
            "p_value_hex": _float_hex(self.p_value),
            "plus_one_correction": self.plus_one_correction,
            "rng_algorithm": RNG_ALGORITHM if self.mode == "monte_carlo" else None,
            "seed": self.seed if self.mode == "monte_carlo" else None,
            "statistic": "mean_paired_difference",
            "ties": "inclusive",
        }


def _require_probability_or_one(value: object, name: str) -> float:
    result = _require_finite_float(value, name)
    if not 0.0 < result <= 1.0:
        raise MatchedStatisticsError(f"{name} must be in (0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class HolmDecision:
    hypothesis_id: str
    original_index: int
    rank: int
    raw_numerator: int
    raw_denominator: int
    adjusted_numerator: int
    adjusted_denominator: int
    adjusted_p_value: float
    reject: bool

    def __post_init__(self) -> None:
        _require_identifier(self.hypothesis_id, "hypothesis_id")
        _require_exact_int(self.original_index, "original_index")
        _require_exact_int(self.rank, "rank", minimum=1)
        raw_num = _require_exact_int(self.raw_numerator, "raw_numerator", minimum=1)
        raw_den = _require_exact_int(self.raw_denominator, "raw_denominator", minimum=1)
        adj_num = _require_exact_int(self.adjusted_numerator, "adjusted_numerator", minimum=1)
        adj_den = _require_exact_int(self.adjusted_denominator, "adjusted_denominator", minimum=1)
        if raw_num > raw_den or adj_num > adj_den:
            raise MatchedStatisticsError("Holm p-values must be <= 1")
        adjusted = _require_probability_or_one(self.adjusted_p_value, "adjusted_p_value")
        if adjusted != float(Fraction(adj_num, adj_den)):
            raise MatchedStatisticsError("adjusted_p_value does not match its rational value")
        _require_exact_bool(self.reject, "reject")

    def to_payload(self) -> dict[str, object]:
        return {
            "adjusted_denominator": self.adjusted_denominator,
            "adjusted_numerator": self.adjusted_numerator,
            "adjusted_p_value_hex": _float_hex(self.adjusted_p_value),
            "hypothesis_id": self.hypothesis_id,
            "original_index": self.original_index,
            "rank": self.rank,
            "raw_denominator": self.raw_denominator,
            "raw_numerator": self.raw_numerator,
            "reject": self.reject,
        }


def _comparison_input_sha256(
    comparison: ComparisonSpec,
    intervention: LearningMethodScores,
    comparator: LearningMethodScores,
) -> str:
    return canonical_payload_sha256(
        {
            "comparator": comparator.to_payload(),
            "comparison": comparison.to_payload(),
            "intervention": intervention.to_payload(),
            "schema": COMPARISON_INPUT_SCHEMA,
        }
    )


@dataclass(frozen=True, slots=True)
class PrimaryComparisonResult:
    comparison: ComparisonSpec
    matched_inputs_sha256: str
    paired_differences_sha256: str
    bootstrap: BootstrapLowerBound
    frozen_margin: float
    superiority_passed: bool

    def __post_init__(self) -> None:
        if type(self.comparison) is not ComparisonSpec:
            raise MatchedStatisticsError("comparison must be a ComparisonSpec")
        _require_sha256(self.matched_inputs_sha256, "matched_inputs_sha256")
        _require_sha256(self.paired_differences_sha256, "paired_differences_sha256")
        if type(self.bootstrap) is not BootstrapLowerBound:
            raise MatchedStatisticsError("bootstrap result has the wrong type")
        margin = _require_finite_float(self.frozen_margin, "frozen_margin")
        if margin < 0.0:
            raise MatchedStatisticsError("frozen_margin must be nonnegative")
        passed = _require_exact_bool(self.superiority_passed, "superiority_passed")
        if passed != (self.bootstrap.lower_bound > margin):
            raise MatchedStatisticsError("primary superiority decision is inconsistent")

    def to_payload(self) -> dict[str, object]:
        return {
            "bootstrap": self.bootstrap.to_payload(),
            "comparison": self.comparison.to_payload(),
            "frozen_margin_hex": _float_hex(self.frozen_margin),
            "gate": "lower_bound_strictly_greater_than_margin",
            "matched_inputs_sha256": self.matched_inputs_sha256,
            "paired_difference_count": self.bootstrap.n_pairs,
            "paired_differences_sha256": self.paired_differences_sha256,
            "superiority_passed": self.superiority_passed,
        }


@dataclass(frozen=True, slots=True)
class SecondaryComparisonResult:
    comparison: ComparisonSpec
    matched_inputs_sha256: str
    paired_differences_sha256: str
    sign_flip: SignFlipResult
    holm: HolmDecision

    def __post_init__(self) -> None:
        if type(self.comparison) is not ComparisonSpec:
            raise MatchedStatisticsError("comparison must be a ComparisonSpec")
        _require_sha256(self.matched_inputs_sha256, "matched_inputs_sha256")
        _require_sha256(self.paired_differences_sha256, "paired_differences_sha256")
        if type(self.sign_flip) is not SignFlipResult:
            raise MatchedStatisticsError("sign_flip result has the wrong type")
        if type(self.holm) is not HolmDecision:
            raise MatchedStatisticsError("holm decision has the wrong type")
        if self.holm.hypothesis_id != self.comparison.hypothesis_id:
            raise MatchedStatisticsError("Holm hypothesis does not match the comparison")
        if (self.holm.raw_numerator, self.holm.raw_denominator) != (
            self.sign_flip.p_numerator,
            self.sign_flip.p_denominator,
        ):
            raise MatchedStatisticsError("Holm raw p-value does not match the sign-flip test")

    def to_payload(self) -> dict[str, object]:
        return {
            "comparison": self.comparison.to_payload(),
            "holm": self.holm.to_payload(),
            "matched_inputs_sha256": self.matched_inputs_sha256,
            "paired_difference_count": self.sign_flip.n_pairs,
            "paired_differences_sha256": self.paired_differences_sha256,
            "sign_flip": self.sign_flip.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PrivilegedDiagnosticExclusion:
    diagnostic_id: str
    input_sha256: str
    exclusion_reason: str = "privileged_diagnostic_not_eligible_for_superiority"

    def __post_init__(self) -> None:
        _require_identifier(self.diagnostic_id, "diagnostic_id")
        _require_sha256(self.input_sha256, "input_sha256")
        if (
            type(self.exclusion_reason) is not str
            or self.exclusion_reason
            != "privileged_diagnostic_not_eligible_for_superiority"
        ):
            raise MatchedStatisticsError("invalid privileged diagnostic exclusion reason")

    def to_payload(self) -> dict[str, object]:
        return {
            "diagnostic_id": self.diagnostic_id,
            "exclusion_reason": self.exclusion_reason,
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True, slots=True)
class MatchedComparisonResult:
    contract: MatchedComparisonContract
    primary: PrimaryComparisonResult
    secondary: tuple[SecondaryComparisonResult, ...]
    diagnostic_exclusions: tuple[PrivilegedDiagnosticExclusion, ...]

    def __post_init__(self) -> None:
        if type(self.contract) is not MatchedComparisonContract:
            raise MatchedStatisticsError("contract has the wrong type")
        if type(self.primary) is not PrimaryComparisonResult:
            raise MatchedStatisticsError("primary result has the wrong type")
        secondary_objects = _require_exact_tuple(self.secondary, "secondary results")
        exclusion_objects = _require_exact_tuple(
            self.diagnostic_exclusions, "diagnostic exclusions"
        )
        for item in secondary_objects:
            if type(item) is not SecondaryComparisonResult:
                raise MatchedStatisticsError("secondary result has the wrong type")
        for item in exclusion_objects:
            if type(item) is not PrivilegedDiagnosticExclusion:
                raise MatchedStatisticsError("diagnostic exclusion has the wrong type")
        secondary = cast(tuple[SecondaryComparisonResult, ...], secondary_objects)
        exclusions = cast(tuple[PrivilegedDiagnosticExclusion, ...], exclusion_objects)
        if self.primary.comparison != self.contract.primary_comparison:
            raise MatchedStatisticsError("primary comparison does not match the contract")
        primary_intervention = self.contract.method(
            self.contract.primary_comparison.intervention_id
        )
        primary_comparator = self.contract.method(
            self.contract.primary_comparison.comparator_id
        )
        expected_primary_differences = paired_differences(
            primary_intervention.scores,
            primary_comparator.scores,
        )
        expected_primary_input_sha256 = _comparison_input_sha256(
            self.contract.primary_comparison,
            primary_intervention,
            primary_comparator,
        )
        if self.primary.matched_inputs_sha256 != expected_primary_input_sha256:
            raise MatchedStatisticsError("primary input digest does not match the contract")
        expected_primary_difference_sha256 = _difference_vector_sha256(
            comparison_id=self.contract.primary_comparison.hypothesis_id,
            differences=expected_primary_differences,
            seeds=self.contract.common_seeds,
        )
        if self.primary.paired_differences_sha256 != expected_primary_difference_sha256:
            raise MatchedStatisticsError("primary difference digest does not match the contract")
        if _float_hex(self.primary.frozen_margin) != _float_hex(self.contract.primary_margin):
            raise MatchedStatisticsError("primary margin does not match the contract")
        if (
            self.primary.bootstrap.resamples != self.contract.bootstrap.resamples
            or self.primary.bootstrap.seed != self.contract.bootstrap.seed
            or self.primary.bootstrap.confidence != self.contract.bootstrap.confidence
            or self.primary.bootstrap.alpha != 1.0 - self.contract.bootstrap.confidence
        ):
            raise MatchedStatisticsError("bootstrap metadata does not match the contract")
        expected_bootstrap = paired_percentile_bootstrap_lower_bound(
            expected_primary_differences,
            self.contract.bootstrap,
        )
        if _canonical_json_bytes(
            self.primary.bootstrap.to_payload()
        ) != _canonical_json_bytes(expected_bootstrap.to_payload()):
            raise MatchedStatisticsError("bootstrap result does not replay from the contract")
        if tuple(item.comparison for item in secondary) != self.contract.secondary_comparisons:
            raise MatchedStatisticsError("secondary result ordering does not match the contract")
        for item, comparison in zip(
            secondary,
            self.contract.secondary_comparisons,
            strict=True,
        ):
            intervention = self.contract.method(comparison.intervention_id)
            comparator = self.contract.method(comparison.comparator_id)
            expected_differences = paired_differences(
                intervention.scores,
                comparator.scores,
            )
            expected_input_sha256 = _comparison_input_sha256(
                comparison,
                intervention,
                comparator,
            )
            if item.matched_inputs_sha256 != expected_input_sha256:
                raise MatchedStatisticsError("secondary input digest does not match the contract")
            expected_difference_sha256 = _difference_vector_sha256(
                comparison_id=comparison.hypothesis_id,
                differences=expected_differences,
                seeds=self.contract.common_seeds,
            )
            if item.paired_differences_sha256 != expected_difference_sha256:
                raise MatchedStatisticsError(
                    "secondary difference digest does not match the contract"
                )
            expected_draws = (
                1 << len(expected_differences)
                if len(expected_differences) <= EXACT_SIGN_FLIP_MAX_PAIRS
                else self.contract.permutation.monte_carlo_resamples
            )
            if (
                item.sign_flip.draws != expected_draws
                or item.sign_flip.seed != self.contract.permutation.seed
            ):
                raise MatchedStatisticsError("sign-flip metadata does not match the contract")
            expected_sign_flip = paired_sign_flip_test(
                expected_differences,
                self.contract.permutation,
            )
            if _canonical_json_bytes(
                item.sign_flip.to_payload()
            ) != _canonical_json_bytes(expected_sign_flip.to_payload()):
                raise MatchedStatisticsError("sign-flip result does not replay from the contract")
        expected_holm = holm_adjust(
            tuple(item.comparison.hypothesis_id for item in secondary),
            tuple(
                (item.sign_flip.p_numerator, item.sign_flip.p_denominator)
                for item in secondary
            ),
            self.contract.permutation.familywise_alpha,
        )
        if tuple(item.holm for item in secondary) != expected_holm:
            raise MatchedStatisticsError("Holm decisions do not match the contract")
        if tuple(item.diagnostic_id for item in exclusions) != tuple(
            diagnostic.diagnostic_id for diagnostic in self.contract.privileged_diagnostics
        ):
            raise MatchedStatisticsError("diagnostic exclusions do not match the contract")
        for exclusion, diagnostic in zip(
            exclusions,
            self.contract.privileged_diagnostics,
            strict=True,
        ):
            if exclusion.input_sha256 != canonical_payload_sha256(diagnostic.to_payload()):
                raise MatchedStatisticsError(
                    "diagnostic exclusion digest does not match the contract"
                )

    def to_body(self) -> dict[str, object]:
        return {
            "classification": "paired_learning_method_comparison",
            "contract": self.contract.to_payload(),
            "contract_sha256": self.contract.payload_sha256,
            "diagnostic_exclusions": [item.to_payload() for item in self.diagnostic_exclusions],
            "no_promotion_authority": True,
            "primary": self.primary.to_payload(),
            "primary_superiority_passed": self.primary.superiority_passed,
            "raw_scores_or_differences_embedded": False,
            "schema": RESULT_SCHEMA,
            "secondary": [item.to_payload() for item in self.secondary],
            "evidence_digests_require_external_validation": True,
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_payload_sha256(self.to_body())

    def to_payload(self) -> dict[str, object]:
        payload = self.to_body()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    def canonical_json(self) -> bytes:
        """Return the complete result as canonical UTF-8 JSON bytes."""

        return _canonical_json_bytes(self.to_payload())


def paired_differences(
    candidate_scores: tuple[float, ...], baseline_scores: tuple[float, ...]
) -> tuple[float, ...]:
    """Return finite candidate-minus-baseline differences without reordering pairs."""

    candidate = _require_score_tuple(candidate_scores, "candidate_scores")
    baseline = _require_score_tuple(baseline_scores, "baseline_scores")
    if len(candidate) != len(baseline):
        raise MatchedStatisticsError("paired score vectors must have equal lengths")
    differences: list[float] = []
    for index, (candidate_value, baseline_value) in enumerate(
        zip(candidate, baseline, strict=True)
    ):
        difference = candidate_value - baseline_value
        if not math.isfinite(difference):
            raise MatchedStatisticsError(f"paired difference {index} is not finite")
        differences.append(difference)
    return tuple(differences)


def paired_percentile_bootstrap_lower_bound(
    differences: tuple[float, ...], spec: BootstrapSpec
) -> BootstrapLowerBound:
    """Compute the frozen one-sided percentile lower bound of the paired mean."""

    checked = _require_score_tuple(differences, "differences")
    if type(spec) is not BootstrapSpec:
        raise MatchedStatisticsError("spec must be a BootstrapSpec")
    n_pairs = len(checked)
    integer_values, common_denominator = _exact_dyadic_components(checked)
    bootstrap_denominator = common_denominator * n_pairs
    distribution = np.empty(spec.resamples, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    chunk_size = max(1, min(spec.resamples, _BOOTSTRAP_CHUNK_ELEMENTS // n_pairs))
    for start in range(0, spec.resamples, chunk_size):
        stop = min(start + chunk_size, spec.resamples)
        indices = rng.integers(0, n_pairs, size=(stop - start, n_pairs), dtype=np.int64)
        for offset, row in enumerate(indices):
            numerator = sum(integer_values[int(index)] for index in row)
            distribution[start + offset] = float(Fraction(numerator, bootstrap_denominator))
    if not bool(np.all(np.isfinite(distribution))):
        raise MatchedStatisticsError("bootstrap arithmetic produced a nonfinite value")
    alpha = 1.0 - spec.confidence
    lower = float(np.quantile(distribution, alpha, method=QUANTILE_METHOD))
    if not math.isfinite(lower):
        raise MatchedStatisticsError("bootstrap lower bound is not finite")
    little_endian = np.ascontiguousarray(distribution.astype("<f8", copy=False))
    distribution_sha256 = hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()
    return BootstrapLowerBound(
        n_pairs=n_pairs,
        estimate=_finite_mean(checked, "paired mean"),
        lower_bound=lower,
        alpha=alpha,
        confidence=spec.confidence,
        resamples=spec.resamples,
        seed=spec.seed,
        distribution_sha256=distribution_sha256,
    )


def paired_sign_flip_test(
    differences: tuple[float, ...], spec: PermutationSpec
) -> SignFlipResult:
    """One-sided paired sign-flip test; exact for n<=20, Monte Carlo otherwise."""

    checked = _require_score_tuple(differences, "differences")
    if type(spec) is not PermutationSpec:
        raise MatchedStatisticsError("spec must be a PermutationSpec")
    nonzero_pairs = sum(value != 0.0 for value in checked)
    observed_mean = _finite_mean(checked, "observed paired mean")
    n_pairs = len(checked)
    integer_values = _common_integer_scale(checked)
    if n_pairs <= EXACT_SIGN_FLIP_MAX_PAIRS:
        draws = 1 << n_pairs
        extreme = _exact_sign_flip_extreme_count(integer_values, draws)
        numerator = extreme
        denominator = draws
        mode: Literal["exact", "monte_carlo"] = "exact"
        plus_one = False
    else:
        draws = spec.monte_carlo_resamples
        extreme = _monte_carlo_sign_flip_extreme_count(
            integer_values,
            draws,
            spec.seed,
        )
        numerator = extreme + 1
        denominator = draws + 1
        mode = "monte_carlo"
        plus_one = True
    return SignFlipResult(
        n_pairs=n_pairs,
        nonzero_pairs=nonzero_pairs,
        observed_mean=observed_mean,
        mode=mode,
        extreme_count=extreme,
        p_numerator=numerator,
        p_denominator=denominator,
        p_value=float(Fraction(numerator, denominator)),
        draws=draws,
        seed=spec.seed,
        plus_one_correction=plus_one,
    )


def _common_integer_scale(values: tuple[float, ...]) -> tuple[int, ...]:
    """Represent binary floats as exact integers under one common power-of-two scale."""

    integer_values, _ = _exact_dyadic_components(values)
    return integer_values


def _exact_sign_flip_extreme_count(values: tuple[int, ...], draws: int) -> int:
    """Count all assignments exactly, using Gray code and arbitrary-precision integers."""

    # Flipping the signs of subset S changes T=sum(d) to T-2*sum(S).  Therefore
    # the one-sided inclusive event T_flipped >= T is exactly sum(S) <= 0.  The
    # common integer scale makes this comparison exact even for cancellation-heavy
    # finite floats such as (1e16, 1.0, -1e16).
    subset_sum = 0
    previous_gray = 0
    extreme = 1  # The empty subset has sum zero and is an inclusive tie.
    for assignment in range(1, draws):
        gray = assignment ^ (assignment >> 1)
        changed = gray ^ previous_gray
        bit_index = changed.bit_length() - 1
        if gray & changed:
            subset_sum += values[bit_index]
        else:
            subset_sum -= values[bit_index]
        extreme += int(subset_sum <= 0)
        previous_gray = gray
    return extreme


def _monte_carlo_sign_flip_extreme_count(
    values: tuple[int, ...], draws: int, seed: int
) -> int:
    n_pairs = len(values)
    rng = np.random.Generator(np.random.PCG64(seed))
    chunk_size = max(1, min(draws, _SIGN_FLIP_CHUNK_ELEMENTS // n_pairs))
    extreme = 0
    int64_safe = sum(abs(value) for value in values) <= np.iinfo(np.int64).max
    int64_values = np.asarray(values, dtype=np.int64) if int64_safe else None
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        negative = rng.integers(
            0,
            2,
            size=(stop - start, n_pairs),
            dtype=np.uint8,
        )
        if int64_values is not None:
            negative_sums = negative.astype(np.int64) @ int64_values
            extreme += int(np.count_nonzero(negative_sums <= 0))
        else:
            for row in negative:
                subset_sum = sum(
                    value
                    for value, is_negative in zip(values, row, strict=True)
                    if is_negative
                )
                extreme += int(subset_sum <= 0)
    return extreme


def holm_adjust(
    hypothesis_ids: tuple[str, ...],
    p_values: tuple[tuple[int, int], ...],
    familywise_alpha: float,
) -> tuple[HolmDecision, ...]:
    """Apply stable Holm correction using exact rational p-values."""

    id_objects = _require_exact_tuple(hypothesis_ids, "hypothesis_ids")
    p_objects = _require_exact_tuple(p_values, "p_values")
    alpha = _require_probability(familywise_alpha, "familywise_alpha")
    if len(id_objects) != len(p_objects):
        raise MatchedStatisticsError("hypothesis_ids and p_values must have the same length")
    identifiers = tuple(
        _require_identifier(identifier, f"hypothesis_ids[{index}]")
        for index, identifier in enumerate(id_objects)
    )
    if len(set(identifiers)) != len(identifiers):
        raise MatchedStatisticsError("hypothesis_ids must be unique")
    fractions: list[Fraction] = []
    raw_counts: list[tuple[int, int]] = []
    for index, item in enumerate(p_objects):
        if type(item) is not tuple or len(item) != 2:
            raise MatchedStatisticsError(f"p_values[{index}] must be a numerator/denominator tuple")
        numerator = _require_exact_int(item[0], f"p_values[{index}][0]", minimum=1)
        denominator = _require_exact_int(item[1], f"p_values[{index}][1]", minimum=1)
        if numerator > denominator:
            raise MatchedStatisticsError(f"p_values[{index}] must be <= 1")
        raw_counts.append((numerator, denominator))
        fractions.append(Fraction(numerator, denominator))

    family_size = len(fractions)
    if family_size == 0:
        return ()
    sorted_indices = sorted(range(family_size), key=lambda index: (fractions[index], index))
    adjusted_by_index: list[Fraction] = [Fraction(1, 1)] * family_size
    rank_by_index = [0] * family_size
    running = Fraction(0, 1)
    for zero_rank, original_index in enumerate(sorted_indices):
        candidate = min(Fraction(1, 1), (family_size - zero_rank) * fractions[original_index])
        running = max(running, candidate)
        adjusted_by_index[original_index] = running
        rank_by_index[original_index] = zero_rank + 1

    alpha_fraction = Fraction.from_float(alpha)
    decisions: list[HolmDecision] = []
    for original_index, identifier in enumerate(identifiers):
        adjusted = adjusted_by_index[original_index]
        raw_numerator, raw_denominator = raw_counts[original_index]
        decisions.append(
            HolmDecision(
                hypothesis_id=identifier,
                original_index=original_index,
                rank=rank_by_index[original_index],
                raw_numerator=raw_numerator,
                raw_denominator=raw_denominator,
                adjusted_numerator=adjusted.numerator,
                adjusted_denominator=adjusted.denominator,
                adjusted_p_value=float(adjusted),
                reject=adjusted <= alpha_fraction,
            )
        )
    return tuple(decisions)


def analyze_matched_scores(contract: MatchedComparisonContract) -> MatchedComparisonResult:
    """Evaluate a validated contract without importing or executing a benchmark."""

    if type(contract) is not MatchedComparisonContract:
        raise MatchedStatisticsError("contract must be a MatchedComparisonContract")
    primary_spec = contract.primary_comparison
    primary_intervention = contract.method(primary_spec.intervention_id)
    primary_comparator = contract.method(primary_spec.comparator_id)
    primary_differences = paired_differences(
        primary_intervention.scores,
        primary_comparator.scores,
    )
    bootstrap = paired_percentile_bootstrap_lower_bound(primary_differences, contract.bootstrap)
    primary = PrimaryComparisonResult(
        comparison=primary_spec,
        matched_inputs_sha256=_comparison_input_sha256(
            primary_spec,
            primary_intervention,
            primary_comparator,
        ),
        paired_differences_sha256=_difference_vector_sha256(
            comparison_id=primary_spec.hypothesis_id,
            differences=primary_differences,
            seeds=contract.common_seeds,
        ),
        bootstrap=bootstrap,
        frozen_margin=contract.primary_margin,
        superiority_passed=bootstrap.lower_bound > contract.primary_margin,
    )

    difference_vectors: list[tuple[float, ...]] = []
    sign_flips: list[SignFlipResult] = []
    for comparison in contract.secondary_comparisons:
        intervention = contract.method(comparison.intervention_id)
        comparator = contract.method(comparison.comparator_id)
        differences = paired_differences(intervention.scores, comparator.scores)
        difference_vectors.append(differences)
        sign_flips.append(paired_sign_flip_test(differences, contract.permutation))
    holm = holm_adjust(
        tuple(comparison.hypothesis_id for comparison in contract.secondary_comparisons),
        tuple((test.p_numerator, test.p_denominator) for test in sign_flips),
        contract.permutation.familywise_alpha,
    )
    secondary = tuple(
        SecondaryComparisonResult(
            comparison=comparison,
            matched_inputs_sha256=_comparison_input_sha256(
                comparison,
                contract.method(comparison.intervention_id),
                contract.method(comparison.comparator_id),
            ),
            paired_differences_sha256=_difference_vector_sha256(
                comparison_id=comparison.hypothesis_id,
                differences=differences,
                seeds=contract.common_seeds,
            ),
            sign_flip=sign_flip,
            holm=decision,
        )
        for comparison, differences, sign_flip, decision in zip(
            contract.secondary_comparisons,
            difference_vectors,
            sign_flips,
            holm,
            strict=True,
        )
    )
    exclusions = tuple(
        PrivilegedDiagnosticExclusion(
            diagnostic_id=diagnostic.diagnostic_id,
            input_sha256=canonical_payload_sha256(diagnostic.to_payload()),
        )
        for diagnostic in contract.privileged_diagnostics
    )
    return MatchedComparisonResult(
        contract=contract,
        primary=primary,
        secondary=secondary,
        diagnostic_exclusions=exclusions,
    )


evaluate_matched_comparison = analyze_matched_scores


def validate_result_payload(
    payload: Mapping[str, object], contract: MatchedComparisonContract
) -> MatchedComparisonResult:
    """Validate integrity and replay-equivalence against an expected contract."""

    if type(payload) is not dict:
        raise MatchedStatisticsError("result payload must be a plain object")
    if type(contract) is not MatchedComparisonContract:
        raise MatchedStatisticsError("contract must be a MatchedComparisonContract")
    payload_dict = dict(payload)
    if "payload_sha256" not in payload_dict:
        raise MatchedStatisticsError("result payload is missing payload_sha256")
    claimed_digest = _require_sha256(payload_dict.pop("payload_sha256"), "payload_sha256")
    actual_digest = canonical_payload_sha256(payload_dict)
    if claimed_digest != actual_digest:
        raise MatchedStatisticsError("result payload SHA-256 mismatch")
    expected = analyze_matched_scores(contract)
    if _canonical_json_bytes(payload) != expected.canonical_json():
        raise MatchedStatisticsError("result payload does not replay from the expected contract")
    return expected


def _reject_json_constant(value: str) -> NoReturn:
    raise MatchedStatisticsError(f"nonfinite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MatchedStatisticsError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical_result(
    raw: bytes, contract: MatchedComparisonContract
) -> MatchedComparisonResult:
    """Parse canonical result bytes and replay them against ``contract``."""

    if type(raw) is not bytes:
        raise MatchedStatisticsError("raw result must be bytes")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MatchedStatisticsError("result is not valid UTF-8") from exc
    try:
        parsed_object = cast(
            object,
            json.loads(
                decoded,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            ),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise MatchedStatisticsError("result is not valid JSON") from exc
    if type(parsed_object) is not dict:
        raise MatchedStatisticsError("result JSON must be an object")
    parsed = cast(dict[str, object], parsed_object)
    if _canonical_json_bytes(parsed) != raw:
        raise MatchedStatisticsError("result JSON is not in canonical form")
    return validate_result_payload(parsed, contract)


__all__ = [
    "BootstrapLowerBound",
    "BootstrapSpec",
    "CANONICALIZATION",
    "COMPARISON_INPUT_SCHEMA",
    "CONTRACT_SCHEMA",
    "ComparisonSpec",
    "DIFFERENCE_VECTOR_SCHEMA",
    "DIGEST_ALGORITHM",
    "EXACT_SIGN_FLIP_MAX_PAIRS",
    "EvidenceBinding",
    "HolmDecision",
    "LearningMethodScores",
    "MatchedComparisonContract",
    "MatchedComparisonResult",
    "MatchedStatisticsError",
    "PermutationSpec",
    "PrimaryComparisonResult",
    "PrivilegedDiagnosticExclusion",
    "PrivilegedDiagnosticScores",
    "QUANTILE_METHOD",
    "RESULT_SCHEMA",
    "RNG_ALGORITHM",
    "SCORE_VECTOR_SCHEMA",
    "SecondaryComparisonResult",
    "SignFlipResult",
    "analyze_matched_scores",
    "canonical_payload_sha256",
    "evaluate_matched_comparison",
    "holm_adjust",
    "load_canonical_result",
    "paired_differences",
    "paired_percentile_bootstrap_lower_bound",
    "paired_sign_flip_test",
    "validate_result_payload",
]
