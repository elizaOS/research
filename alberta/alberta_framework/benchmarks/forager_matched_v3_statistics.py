"""Finite-sample simultaneous analysis for a fixed named Forager panel.

The input is a complete candidate-by-block matrix of exact integer cumulative rewards.
Every candidate must name the same blocks in the same order.  The analysis constructs all
``K * (K - 1)`` ordered paired contrasts and applies the same finite-sample empirical-
Bernstein lower-bound formula to each member of that family.  A sample leader is selected by
exact mean cumulative reward (candidate identifier ascending breaks ties); its named-panel
claim passes only when every leader-versus-other lower bound is strictly above ``delta``.

Results are detached from raw scores and block identifiers.  Their input digest binds those
caller-held values, and validation replays the full analysis against the supplied input.
Neither an integrity digest nor a passing named-panel gate grants evidence-promotion,
universal-SOTA, literature-best, or compute-matching authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, NoReturn, cast

HORIZON: Final = 499_712
RAW_STEP_REWARDS: Final = (-1, 0, 1, 30)
SCORE_MINIMUM: Final = -HORIZON
SCORE_MAXIMUM: Final = 30 * HORIZON
CONTRAST_RANGE_WIDTH: Final = 62 * HORIZON
MAX_CANDIDATES: Final = 64
MAX_BLOCKS_PER_CANDIDATE: Final = 25_000
MAX_SCORE_CELLS: Final = 50_000

INPUT_SCHEMA: Final = "alberta.forager_matched_v3_statistics.input.v1"
INFERENTIAL_SCORES_SCHEMA: Final = (
    "alberta.forager_matched_v3_statistics.inferential_scores.v1"
)
RESULT_SCHEMA: Final = "alberta.forager_matched_v3_statistics.result.v1"
ANALYSIS_IMPLEMENTATION_SCHEMA: Final = (
    "alberta.forager_matched_v3_statistics.empirical_bernstein.v1"
)
CANONICALIZATION: Final = "utf8-json-sort-keys-compact-no-nan-floats-as-hex"
DIGEST_ALGORITHM: Final = "sha256"

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CANONICAL_BYTES = 16 * 1_048_576
_MAX_CANONICAL_NODES = 1_000_000
_MAX_CANONICAL_DEPTH = 32

_INPUT_BODY_KEYS = frozenset(
    {
        "alpha_hex",
        "canonicalization",
        "delta_hex",
        "horizon",
        "inferential_scores",
        "metric",
        "metric_direction",
        "raw_step_rewards",
        "schema",
        "score_maximum",
        "score_minimum",
    }
)
_INPUT_PAYLOAD_KEYS = _INPUT_BODY_KEYS | {"payload_sha256"}
_INFERENTIAL_SCORES_KEYS = frozenset({"block_ids", "candidate_id", "schema", "scores"})
_RESULT_BODY_KEYS = frozenset(
    {
        "alpha_hex",
        "analysis_implementation",
        "assumptions",
        "block_count",
        "candidate_count",
        "contrast_range_width",
        "decision_rule",
        "delta_hex",
        "family_scope",
        "family_size",
        "horizon",
        "inferential_ids",
        "input_sha256",
        "interpretation",
        "metric",
        "metric_direction",
        "named_panel_superiority_claim_passed",
        "ordered_contrasts",
        "q_hex",
        "raw_scores_or_differences_embedded",
        "sample_leader_id",
        "sample_means",
        "schema",
        "score_maximum",
        "score_minimum",
    }
)
_RESULT_PAYLOAD_KEYS = _RESULT_BODY_KEYS | {"payload_sha256"}


class V3StatisticsError(ValueError):
    """Raised when a v3 named-panel input or result fails closed."""


def _require_exact_keys(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise V3StatisticsError(f"{name} must be a plain object")
    mapping = cast(dict[str, object], value)
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise V3StatisticsError(
            f"{name} must have exact keys; missing={missing!r}, extra={extra!r}"
        )
    return mapping


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise V3StatisticsError(f"{name} must be an int")
    return value


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise V3StatisticsError(f"{name} must be a bool")
    return value


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise V3StatisticsError(f"{name} must be a portable identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise V3StatisticsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise V3StatisticsError(f"{name} must be a float")
    if not math.isfinite(value):
        raise V3StatisticsError(f"{name} must be finite")
    return value


def _require_alpha(value: object) -> float:
    alpha = _require_finite_float(value, "alpha")
    if not 0.0 < alpha < 1.0:
        raise V3StatisticsError("alpha must be strictly between 0 and 1")
    return alpha


def _require_delta(value: object) -> float:
    delta = _require_finite_float(value, "delta")
    if delta < 0.0:
        raise V3StatisticsError("delta must be nonnegative")
    if delta == 0.0 and math.copysign(1.0, delta) < 0.0:
        raise V3StatisticsError("delta must not use the negative zero alias")
    return delta


def _float_hex(value: float) -> str:
    return value.hex()


def _parse_float_hex(value: object, name: str) -> float:
    if type(value) is not str:
        raise V3StatisticsError(f"{name} must be a canonical binary64 hex string")
    try:
        parsed = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise V3StatisticsError(f"{name} is not a binary64 hex string") from exc
    if not math.isfinite(parsed):
        raise V3StatisticsError(f"{name} must encode a finite value")
    if parsed.hex() != value:
        raise V3StatisticsError(f"{name} is not in canonical binary64 hex form")
    return parsed


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise V3StatisticsError("payload is not canonical-JSON encodable") from exc
    return encoded.encode("utf-8")


def canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    """Return a SHA-256 digest under this module's canonical JSON encoding."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_emitted_payload_limits(payload: Mapping[str, object], name: str) -> None:
    """Ensure construction cannot emit canonical bytes that its loader rejects.

    The shape walk and serialization are both linear in the already-materialized payload.
    Candidate and score-cell caps below bound the work before a result's complete ordered
    contrast family is constructed.
    """

    _validate_canonical_shape(payload, name)
    if len(_canonical_json_bytes(payload)) > _MAX_CANONICAL_BYTES:
        raise V3StatisticsError(f"{name} canonical payload exceeds the loader byte limit")


def analysis_implementation_descriptor() -> dict[str, object]:
    """Return the frozen mathematical identity of the named-panel calculation."""

    return {
        "claim_gate": "leader_lower_bound_vs_every_other_strictly_greater_than_delta",
        "contrast_difference": "intervention_cumulative_reward_minus_comparator",
        "contrast_family": "all_K_times_K_minus_1_ordered_contrasts",
        "contrast_range_width": CONTRAST_RANGE_WIDTH,
        "family_size": "M=K*(K-1)",
        "horizon": HORIZON,
        "lower_bound": (
            "dbar-sqrt(2*s2*q/N)-7*W*q/(3*(N-1))"
        ),
        "mean_and_variance_accumulation": "exact_integer_then_rational_then_binary64",
        "multiplicity_term": "q=ln(2*M/alpha)",
        "raw_step_rewards": list(RAW_STEP_REWARDS),
        "sample_leader": "highest_exact_mean_then_candidate_id_ascending",
        "schema": ANALYSIS_IMPLEMENTATION_SCHEMA,
        "variance": "s2=sum_b_lt_c((D_b-D_c)^2)/(N*(N-1))",
        "width": "W=62*H",
    }


ANALYSIS_IMPLEMENTATION_SHA256: Final = (
    "558f21ec06d5b588f3724aa3d384d4be08ad27eb7c779398c556badc6e92aec9"
)
if canonical_payload_sha256(analysis_implementation_descriptor()) != (
    ANALYSIS_IMPLEMENTATION_SHA256
):
    raise AssertionError("matched-v3 analysis implementation descriptor drifted")


def _validated_analysis_implementation() -> dict[str, object]:
    descriptor = analysis_implementation_descriptor()
    if canonical_payload_sha256(descriptor) != ANALYSIS_IMPLEMENTATION_SHA256:
        raise V3StatisticsError("analysis implementation descriptor drifted")
    return {
        "descriptor": descriptor,
        "implementation_sha256": ANALYSIS_IMPLEMENTATION_SHA256,
    }


@dataclass(frozen=True, slots=True)
class InferentialScores:
    """Exact cumulative rewards for one inferential candidate on named blocks."""

    candidate_id: str
    block_ids: tuple[str, ...]
    scores: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        if type(self.block_ids) is not tuple:
            raise V3StatisticsError("block_ids must be a tuple")
        if type(self.scores) is not tuple:
            raise V3StatisticsError("scores must be a tuple")
        blocks = tuple(
            _require_identifier(block_id, f"block_ids[{index}]")
            for index, block_id in enumerate(self.block_ids)
        )
        if not blocks:
            raise V3StatisticsError("block_ids must not be empty")
        if len(set(blocks)) != len(blocks):
            raise V3StatisticsError("block_ids must be unique")
        if len(self.scores) != len(blocks):
            raise V3StatisticsError("block_ids and scores must have the same length")
        for index, score_object in enumerate(self.scores):
            score = _require_exact_int(score_object, f"score {index}")
            if score < SCORE_MINIMUM or score > SCORE_MAXIMUM:
                raise V3StatisticsError(
                    f"score {index} is outside cumulative score bounds "
                    f"[{SCORE_MINIMUM}, {SCORE_MAXIMUM}]"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "block_ids": list(self.block_ids),
            "candidate_id": self.candidate_id,
            "schema": INFERENTIAL_SCORES_SCHEMA,
            "scores": list(self.scores),
        }


@dataclass(frozen=True, slots=True)
class NamedPanelAnalysisInput:
    """Complete immutable input for one fixed named inferential panel."""

    inferential_scores: tuple[InferentialScores, ...]
    alpha: float
    delta: float

    def __post_init__(self) -> None:
        if type(self.inferential_scores) is not tuple:
            raise V3StatisticsError("inferential_scores must be a tuple")
        if len(self.inferential_scores) < 2:
            raise V3StatisticsError("at least two inferential candidates are required")
        if len(self.inferential_scores) > MAX_CANDIDATES:
            raise V3StatisticsError(
                f"inferential candidate count exceeds the maximum of {MAX_CANDIDATES}"
            )
        for index, row in enumerate(self.inferential_scores):
            if type(row) is not InferentialScores:
                raise V3StatisticsError(f"inferential_scores[{index}] has the wrong type")
        rows = self.inferential_scores
        candidate_ids = tuple(row.candidate_id for row in rows)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise V3StatisticsError("inferential candidate IDs must be unique")
        common_blocks = rows[0].block_ids
        if len(common_blocks) < 2:
            raise V3StatisticsError("at least two blocks are required")
        if len(common_blocks) > MAX_BLOCKS_PER_CANDIDATE:
            raise V3StatisticsError(
                "block count exceeds the maximum of "
                f"{MAX_BLOCKS_PER_CANDIDATE} per candidate"
            )
        if len(rows) * len(common_blocks) > MAX_SCORE_CELLS:
            raise V3StatisticsError(
                f"complete score matrix exceeds the maximum of {MAX_SCORE_CELLS} cells"
            )
        for row in rows[1:]:
            if len(row.block_ids) != len(common_blocks):
                raise V3StatisticsError("every candidate row must have the same length")
            if row.block_ids != common_blocks:
                raise V3StatisticsError(
                    f"candidate {row.candidate_id!r} does not have the exact common ordered "
                    "block IDs"
                )
        _require_alpha(self.alpha)
        _require_delta(self.delta)
        _validate_emitted_payload_limits(self.to_payload(), "input")

    @property
    def inferential_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self.inferential_scores)

    @property
    def block_ids(self) -> tuple[str, ...]:
        return self.inferential_scores[0].block_ids

    @property
    def candidate_count(self) -> int:
        return len(self.inferential_scores)

    @property
    def block_count(self) -> int:
        return len(self.block_ids)

    def to_body(self) -> dict[str, object]:
        return {
            "alpha_hex": _float_hex(self.alpha),
            "canonicalization": CANONICALIZATION,
            "delta_hex": _float_hex(self.delta),
            "horizon": HORIZON,
            "inferential_scores": [row.to_payload() for row in self.inferential_scores],
            "metric": "cumulative_reward",
            "metric_direction": "maximize",
            "raw_step_rewards": list(RAW_STEP_REWARDS),
            "schema": INPUT_SCHEMA,
            "score_maximum": SCORE_MAXIMUM,
            "score_minimum": SCORE_MINIMUM,
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_payload_sha256(self.to_body())

    def to_payload(self) -> dict[str, object]:
        payload = self.to_body()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    def canonical_json(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())


@dataclass(frozen=True, slots=True)
class CandidateMean:
    """Detached sample mean for one candidate."""

    candidate_id: str
    mean_score: float

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        mean = _require_finite_float(self.mean_score, "mean_score")
        if mean < SCORE_MINIMUM or mean > SCORE_MAXIMUM:
            raise V3StatisticsError("mean_score is outside cumulative score bounds")

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "mean_score_hex": _float_hex(self.mean_score),
        }


@dataclass(frozen=True, slots=True)
class EmpiricalBernsteinContrast:
    """One detached ordered paired contrast and its simultaneous lower bound."""

    intervention_id: str
    comparator_id: str
    mean_difference: float
    sample_variance: float
    lower_bound: float
    strictly_exceeds_delta: bool

    def __post_init__(self) -> None:
        intervention = _require_identifier(self.intervention_id, "intervention_id")
        comparator = _require_identifier(self.comparator_id, "comparator_id")
        if intervention == comparator:
            raise V3StatisticsError("an ordered contrast must use two different candidates")
        mean = _require_finite_float(self.mean_difference, "mean_difference")
        if mean < -31 * HORIZON or mean > 31 * HORIZON:
            raise V3StatisticsError("mean_difference is outside the paired score bounds")
        variance = _require_finite_float(self.sample_variance, "sample_variance")
        if variance < 0.0:
            raise V3StatisticsError("sample_variance must be nonnegative")
        _require_finite_float(self.lower_bound, "lower_bound")
        _require_exact_bool(self.strictly_exceeds_delta, "strictly_exceeds_delta")

    def to_payload(self) -> dict[str, object]:
        return {
            "comparator_id": self.comparator_id,
            "difference_order": "intervention_minus_comparator",
            "intervention_id": self.intervention_id,
            "lower_bound_hex": _float_hex(self.lower_bound),
            "mean_difference_hex": _float_hex(self.mean_difference),
            "sample_variance_hex": _float_hex(self.sample_variance),
            "strictly_exceeds_delta": self.strictly_exceeds_delta,
        }


@dataclass(frozen=True, slots=True)
class NamedPanelAnalysisResult:
    """Detached result for all ordered contrasts in one fixed named panel.

    Construction re-derives the leader from the detached binary64 sample means and checks
    decision consistency.  It cannot establish that those aggregates, contrast moments, or
    bounds came from a particular raw matrix; canonical result validation against the
    caller-supplied :class:`NamedPanelAnalysisInput` is the full authority boundary.
    """

    input_sha256: str
    inferential_ids: tuple[str, ...]
    block_count: int
    alpha: float
    delta: float
    family_size: int
    q: float
    sample_means: tuple[CandidateMean, ...]
    sample_leader_id: str
    ordered_contrasts: tuple[EmpiricalBernsteinContrast, ...]
    named_panel_superiority_claim_passed: bool

    def __post_init__(self) -> None:
        _require_sha256(self.input_sha256, "input_sha256")
        if type(self.inferential_ids) is not tuple or len(self.inferential_ids) < 2:
            raise V3StatisticsError("inferential_ids must be a tuple with at least two IDs")
        ids = tuple(
            _require_identifier(candidate_id, f"inferential_ids[{index}]")
            for index, candidate_id in enumerate(self.inferential_ids)
        )
        if len(set(ids)) != len(ids):
            raise V3StatisticsError("inferential_ids must be unique")
        block_count = _require_exact_int(self.block_count, "block_count")
        if block_count < 2:
            raise V3StatisticsError("block_count must be at least two")
        alpha = _require_alpha(self.alpha)
        delta = _require_delta(self.delta)
        expected_family_size = len(ids) * (len(ids) - 1)
        family_size = _require_exact_int(self.family_size, "family_size")
        if family_size != expected_family_size:
            raise V3StatisticsError("family_size must equal K*(K-1)")
        q = _require_finite_float(self.q, "q")
        if q != math.log((2.0 * family_size) / alpha):
            raise V3StatisticsError("q does not equal ln(2*M/alpha)")
        if type(self.sample_means) is not tuple or len(self.sample_means) != len(ids):
            raise V3StatisticsError("sample_means must match inferential_ids")
        for mean in self.sample_means:
            if type(mean) is not CandidateMean:
                raise V3StatisticsError("sample_means contains the wrong type")
        means = self.sample_means
        if tuple(mean.candidate_id for mean in means) != ids:
            raise V3StatisticsError("sample_means must preserve inferential_ids order")
        leader = _require_identifier(self.sample_leader_id, "sample_leader_id")
        if leader not in ids:
            raise V3StatisticsError("sample_leader_id is not in inferential_ids")
        expected_leader_index = 0
        for index in range(1, len(means)):
            if means[index].mean_score > means[expected_leader_index].mean_score or (
                means[index].mean_score == means[expected_leader_index].mean_score
                and means[index].candidate_id < means[expected_leader_index].candidate_id
            ):
                expected_leader_index = index
        if leader != means[expected_leader_index].candidate_id:
            raise V3StatisticsError("sample_leader_id is inconsistent with the sample means")
        if type(self.ordered_contrasts) is not tuple:
            raise V3StatisticsError("ordered_contrasts must be a tuple")
        if len(self.ordered_contrasts) != family_size:
            raise V3StatisticsError("ordered_contrasts must contain the complete family")
        for contrast in self.ordered_contrasts:
            if type(contrast) is not EmpiricalBernsteinContrast:
                raise V3StatisticsError("ordered_contrasts contains the wrong type")
        contrasts = self.ordered_contrasts
        expected_pairs = tuple(
            (intervention, comparator)
            for intervention in ids
            for comparator in ids
            if intervention != comparator
        )
        actual_pairs = tuple(
            (contrast.intervention_id, contrast.comparator_id) for contrast in contrasts
        )
        if actual_pairs != expected_pairs:
            raise V3StatisticsError("ordered_contrasts does not preserve the complete family order")
        for contrast in contrasts:
            if contrast.strictly_exceeds_delta != (contrast.lower_bound > delta):
                raise V3StatisticsError("contrast delta decision is inconsistent")
        passed = _require_exact_bool(
            self.named_panel_superiority_claim_passed,
            "named_panel_superiority_claim_passed",
        )
        leader_passes = tuple(
            contrast.strictly_exceeds_delta
            for contrast in contrasts
            if contrast.intervention_id == leader
        )
        if len(leader_passes) != len(ids) - 1 or passed != all(leader_passes):
            raise V3StatisticsError("named-panel superiority decision is inconsistent")
        _validate_emitted_payload_limits(self.to_payload(), "result")

    def contrast(self, intervention_id: str, comparator_id: str) -> EmpiricalBernsteinContrast:
        """Return one ordered contrast by its candidate identifiers."""

        intervention = _require_identifier(intervention_id, "intervention_id")
        comparator = _require_identifier(comparator_id, "comparator_id")
        for contrast in self.ordered_contrasts:
            if (
                contrast.intervention_id == intervention
                and contrast.comparator_id == comparator
            ):
                return contrast
        raise V3StatisticsError("unknown ordered contrast")

    def to_body(self) -> dict[str, object]:
        return {
            "alpha_hex": _float_hex(self.alpha),
            "analysis_implementation": _validated_analysis_implementation(),
            "assumptions": {
                "bounded_cumulative_scores": "verified_from_input",
                "cumulative_scores_aggregate_declared_horizon_rewards": (
                    "required_not_verified"
                ),
                "independent_blocks": "required_not_verified",
                "identically_distributed_blocks": "required_not_verified",
                "panel_fixed_before_analysis": "required_not_verified",
                "raw_step_reward_support": list(RAW_STEP_REWARDS),
                "raw_step_reward_support_provenance": "required_not_verified",
                "same_ordered_blocks_for_every_candidate": "verified_from_input",
                "simultaneous_scope": "fixed_named_panel_only",
            },
            "block_count": self.block_count,
            "candidate_count": len(self.inferential_ids),
            "contrast_range_width": CONTRAST_RANGE_WIDTH,
            "decision_rule": (
                "sample_leader_lower_bound_vs_every_other_strictly_greater_than_delta"
            ),
            "delta_hex": _float_hex(self.delta),
            "family_scope": "every_ordered_contrast_in_fixed_named_panel",
            "family_size": self.family_size,
            "horizon": HORIZON,
            "inferential_ids": list(self.inferential_ids),
            "input_sha256": self.input_sha256,
            "interpretation": {
                "compute_matched_claim_authorized": False,
                "evidence_promotion_authorized": False,
                "literature_best_claim_authorized": False,
                "standalone_claim_authorized": False,
                "universal_sota_claim_authorized": False,
            },
            "metric": "cumulative_reward",
            "metric_direction": "maximize",
            "named_panel_superiority_claim_passed": (
                self.named_panel_superiority_claim_passed
            ),
            "ordered_contrasts": [contrast.to_payload() for contrast in self.ordered_contrasts],
            "q_hex": _float_hex(self.q),
            "raw_scores_or_differences_embedded": False,
            "sample_leader_id": self.sample_leader_id,
            "sample_means": [mean.to_payload() for mean in self.sample_means],
            "schema": RESULT_SCHEMA,
            "score_maximum": SCORE_MAXIMUM,
            "score_minimum": SCORE_MINIMUM,
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_payload_sha256(self.to_body())

    def to_payload(self) -> dict[str, object]:
        payload = self.to_body()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    def canonical_json(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())


def _exact_mean(values: tuple[int, ...]) -> Fraction:
    return Fraction(sum(values), len(values))


def _contrast_moments(
    intervention: InferentialScores,
    comparator: InferentialScores,
) -> tuple[Fraction, Fraction]:
    differences = tuple(
        intervention_score - comparator_score
        for intervention_score, comparator_score in zip(
            intervention.scores,
            comparator.scores,
            strict=True,
        )
    )
    count = len(differences)
    difference_sum = sum(differences)
    # Exact identity: sum_{b<c}(D_b-D_c)^2 = N*sum(D_b^2)-sum(D_b)^2.
    pairwise_squared_sum = count * sum(value * value for value in differences) - (
        difference_sum * difference_sum
    )
    if pairwise_squared_sum < 0:
        raise V3StatisticsError("exact pairwise variance arithmetic became negative")
    return (
        Fraction(difference_sum, count),
        Fraction(pairwise_squared_sum, count * (count - 1)),
    )


def analyze_named_panel(panel: NamedPanelAnalysisInput) -> NamedPanelAnalysisResult:
    """Analyze a validated fixed panel without executing a benchmark."""

    if type(panel) is not NamedPanelAnalysisInput:
        raise V3StatisticsError("panel must be a NamedPanelAnalysisInput")
    candidate_count = panel.candidate_count
    block_count = panel.block_count
    family_size = candidate_count * (candidate_count - 1)
    q = math.log((2.0 * family_size) / panel.alpha)
    if not math.isfinite(q) or q <= 0.0:
        raise V3StatisticsError("multiplicity arithmetic produced an invalid q")

    exact_means = tuple(_exact_mean(row.scores) for row in panel.inferential_scores)
    leader_index = 0
    for index in range(1, candidate_count):
        if exact_means[index] > exact_means[leader_index] or (
            exact_means[index] == exact_means[leader_index]
            and panel.inferential_ids[index] < panel.inferential_ids[leader_index]
        ):
            leader_index = index
    leader_id = panel.inferential_ids[leader_index]
    sample_means = tuple(
        CandidateMean(candidate_id=row.candidate_id, mean_score=float(mean))
        for row, mean in zip(panel.inferential_scores, exact_means, strict=True)
    )

    correction = (7.0 * CONTRAST_RANGE_WIDTH * q) / (3.0 * (block_count - 1))
    contrasts: list[EmpiricalBernsteinContrast] = []
    for intervention in panel.inferential_scores:
        for comparator in panel.inferential_scores:
            if intervention.candidate_id == comparator.candidate_id:
                continue
            exact_mean, exact_variance = _contrast_moments(intervention, comparator)
            mean = float(exact_mean)
            variance = float(exact_variance)
            lower_bound = mean - math.sqrt(2.0 * variance * q / block_count) - correction
            if not math.isfinite(lower_bound):
                raise V3StatisticsError("lower-bound arithmetic produced a nonfinite value")
            contrasts.append(
                EmpiricalBernsteinContrast(
                    intervention_id=intervention.candidate_id,
                    comparator_id=comparator.candidate_id,
                    mean_difference=mean,
                    sample_variance=variance,
                    lower_bound=lower_bound,
                    strictly_exceeds_delta=lower_bound > panel.delta,
                )
            )
    leader_passed = all(
        contrast.strictly_exceeds_delta
        for contrast in contrasts
        if contrast.intervention_id == leader_id
    )
    return NamedPanelAnalysisResult(
        input_sha256=panel.payload_sha256,
        inferential_ids=panel.inferential_ids,
        block_count=block_count,
        alpha=panel.alpha,
        delta=panel.delta,
        family_size=family_size,
        q=q,
        sample_means=sample_means,
        sample_leader_id=leader_id,
        ordered_contrasts=tuple(contrasts),
        named_panel_superiority_claim_passed=leader_passed,
    )


def _require_exact_constant(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise V3StatisticsError(f"{name} does not match the fixed v3 contract")


def _parse_inferential_scores(value: object, index: int) -> InferentialScores:
    row = _require_exact_keys(value, _INFERENTIAL_SCORES_KEYS, f"inferential_scores[{index}]")
    _require_exact_constant(row["schema"], INFERENTIAL_SCORES_SCHEMA, "inferential score schema")
    block_objects = row["block_ids"]
    score_objects = row["scores"]
    if type(block_objects) is not list:
        raise V3StatisticsError(f"inferential_scores[{index}].block_ids must be a list")
    if type(score_objects) is not list:
        raise V3StatisticsError(f"inferential_scores[{index}].scores must be a list")
    return InferentialScores(
        candidate_id=_require_identifier(
            row["candidate_id"], f"inferential_scores[{index}].candidate_id"
        ),
        block_ids=tuple(
            _require_identifier(item, f"inferential_scores[{index}].block_ids[{item_index}]")
            for item_index, item in enumerate(cast(list[object], block_objects))
        ),
        scores=tuple(
            _require_exact_int(item, f"inferential_scores[{index}].scores[{item_index}]")
            for item_index, item in enumerate(cast(list[object], score_objects))
        ),
    )


def validate_input_payload(payload: Mapping[str, object]) -> NamedPanelAnalysisInput:
    """Validate an exact-key input payload and its embedded canonical digest."""

    mapping = _require_exact_keys(payload, _INPUT_PAYLOAD_KEYS, "input payload")
    claimed_digest = _require_sha256(mapping["payload_sha256"], "payload_sha256")
    body = {key: value for key, value in mapping.items() if key != "payload_sha256"}
    if canonical_payload_sha256(body) != claimed_digest:
        raise V3StatisticsError("input payload SHA-256 mismatch")
    _require_exact_keys(body, _INPUT_BODY_KEYS, "input body")
    _require_exact_constant(body["schema"], INPUT_SCHEMA, "input schema")
    _require_exact_constant(body["canonicalization"], CANONICALIZATION, "canonicalization")
    _require_exact_constant(body["horizon"], HORIZON, "horizon")
    _require_exact_constant(body["metric"], "cumulative_reward", "metric")
    _require_exact_constant(body["metric_direction"], "maximize", "metric_direction")
    _require_exact_constant(body["score_minimum"], SCORE_MINIMUM, "score_minimum")
    _require_exact_constant(body["score_maximum"], SCORE_MAXIMUM, "score_maximum")
    rewards = body["raw_step_rewards"]
    if type(rewards) is not list or len(rewards) != len(RAW_STEP_REWARDS):
        raise V3StatisticsError("raw_step_rewards does not match the fixed v3 contract")
    for index, (actual, expected) in enumerate(
        zip(cast(list[object], rewards), RAW_STEP_REWARDS, strict=True)
    ):
        _require_exact_constant(actual, expected, f"raw_step_rewards[{index}]")
    row_objects = body["inferential_scores"]
    if type(row_objects) is not list:
        raise V3StatisticsError("inferential_scores must be a list")
    panel = NamedPanelAnalysisInput(
        inferential_scores=tuple(
            _parse_inferential_scores(row, index)
            for index, row in enumerate(cast(list[object], row_objects))
        ),
        alpha=_require_alpha(_parse_float_hex(body["alpha_hex"], "alpha_hex")),
        delta=_require_delta(_parse_float_hex(body["delta_hex"], "delta_hex")),
    )
    if _canonical_json_bytes(panel.to_payload()) != _canonical_json_bytes(mapping):
        raise V3StatisticsError("input payload does not reproduce the canonical v3 input")
    return panel


def validate_result_payload(
    payload: Mapping[str, object], panel: NamedPanelAnalysisInput
) -> NamedPanelAnalysisResult:
    """Validate digest integrity and replay a detached result against ``panel``."""

    if type(panel) is not NamedPanelAnalysisInput:
        raise V3StatisticsError("panel must be a NamedPanelAnalysisInput")
    mapping = _require_exact_keys(payload, _RESULT_PAYLOAD_KEYS, "result payload")
    claimed_digest = _require_sha256(mapping["payload_sha256"], "payload_sha256")
    body = {key: value for key, value in mapping.items() if key != "payload_sha256"}
    if canonical_payload_sha256(body) != claimed_digest:
        raise V3StatisticsError("result payload SHA-256 mismatch")
    _require_exact_keys(body, _RESULT_BODY_KEYS, "result body")
    expected = analyze_named_panel(panel)
    if _canonical_json_bytes(mapping) != expected.canonical_json():
        raise V3StatisticsError("result payload does not replay from the expected input")
    return expected


def _reject_json_constant(value: str) -> NoReturn:
    raise V3StatisticsError(f"nonfinite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V3StatisticsError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_canonical_shape(value: object, name: str) -> None:
    nodes = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        node, depth = pending.pop()
        if depth > _MAX_CANONICAL_DEPTH:
            raise V3StatisticsError(f"{name} JSON exceeds the maximum nesting depth")
        nodes += 1
        if type(node) is dict:
            mapping = cast(dict[str, object], node)
            nodes += len(mapping)
            pending.extend((item, depth + 1) for item in mapping.values())
        elif type(node) is list:
            pending.extend((item, depth + 1) for item in cast(list[object], node))
        if nodes + len(pending) > _MAX_CANONICAL_NODES:
            raise V3StatisticsError(f"{name} JSON contains too many nodes")


def _load_canonical_object(raw: bytes, name: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise V3StatisticsError(f"raw {name} must be bytes")
    if len(raw) > _MAX_CANONICAL_BYTES:
        raise V3StatisticsError(f"raw {name} exceeds the maximum byte length")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise V3StatisticsError(f"{name} is not valid UTF-8") from exc
    try:
        parsed_object = cast(
            object,
            json.loads(
                decoded,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            ),
        )
    except V3StatisticsError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V3StatisticsError(f"{name} is not valid JSON") from exc
    _validate_canonical_shape(parsed_object, name)
    if type(parsed_object) is not dict:
        raise V3StatisticsError(f"{name} JSON must be an object")
    parsed = cast(dict[str, object], parsed_object)
    if _canonical_json_bytes(parsed) != raw:
        raise V3StatisticsError(f"{name} JSON is not in canonical form")
    return parsed


def load_canonical_input(raw: bytes) -> NamedPanelAnalysisInput:
    """Load exact canonical input bytes."""

    return validate_input_payload(_load_canonical_object(raw, "input"))


def load_canonical_result(
    raw: bytes, panel: NamedPanelAnalysisInput
) -> NamedPanelAnalysisResult:
    """Load exact canonical result bytes and replay them against ``panel``."""

    return validate_result_payload(_load_canonical_object(raw, "result"), panel)


__all__ = [
    "ANALYSIS_IMPLEMENTATION_SCHEMA",
    "ANALYSIS_IMPLEMENTATION_SHA256",
    "CANONICALIZATION",
    "CONTRAST_RANGE_WIDTH",
    "CandidateMean",
    "DIGEST_ALGORITHM",
    "EmpiricalBernsteinContrast",
    "HORIZON",
    "INFERENTIAL_SCORES_SCHEMA",
    "INPUT_SCHEMA",
    "InferentialScores",
    "MAX_BLOCKS_PER_CANDIDATE",
    "MAX_CANDIDATES",
    "MAX_SCORE_CELLS",
    "NamedPanelAnalysisInput",
    "NamedPanelAnalysisResult",
    "RAW_STEP_REWARDS",
    "RESULT_SCHEMA",
    "SCORE_MAXIMUM",
    "SCORE_MINIMUM",
    "V3StatisticsError",
    "analysis_implementation_descriptor",
    "analyze_named_panel",
    "canonical_payload_sha256",
    "load_canonical_input",
    "load_canonical_result",
    "validate_input_payload",
    "validate_result_payload",
]
