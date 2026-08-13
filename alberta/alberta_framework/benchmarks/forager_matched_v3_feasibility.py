"""Fail-closed conditional-precision arithmetic for matched Forager v3.

This module performs design-time arithmetic only.  It consumes no score, seed,
root token, result, or filesystem path.  A receipt can show that the frozen
finite-sample correction would fit below an assumed observed gap and asserted
variance ceiling at a proposed interaction cap.  It does not model beta or the
probability of observing that gap and is therefore not a power calculation,
campaign-feasibility gate, preregistration, or execution authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from alberta_framework.benchmarks import forager_matched_v3_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol
from alberta_framework.benchmarks import forager_matched_v3_statistics as statistics

FEASIBILITY_RECEIPT_SCHEMA: Final = (
    "alberta.forager_matched_v3_conditional_precision.v1"
)
CONDITIONAL_PRECISION_IMPLEMENTATION_SCHEMA: Final = (
    "alberta.forager_matched_v3_conditional_precision_implementation.v1"
)
_MAX_RECEIPT_BYTES: Final = 256 * 1024
_MAX_INTERACTIONS: Final = 2**63 - 1
_MAX_CONTRAST_GAP: Final = 31 * statistics.HORIZON
_MAX_SAMPLE_VARIANCE: Final = statistics.CONTRAST_RANGE_WIDTH**2 // 2
_BOUNDED_SAMPLE_VARIANCE_MAXIMUM: Final = (
    statistics.CONTRAST_RANGE_WIDTH**2 // 2
)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

type VarianceBoundSource = Literal[
    "bounded_sample_variance_maximum",
    "development_calibration_upper_bound",
]


class V3FeasibilityError(ValueError):
    """A feasibility design or its canonical receipt is invalid."""


def _exact_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise V3FeasibilityError(
            f"{name} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _alpha(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        raise V3FeasibilityError("alpha must be a finite float strictly between zero and one")
    return value


def _sha256_or_none(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise V3FeasibilityError(f"{name} must be null or a lowercase SHA-256 digest")
    return value


def _canonical_hex(value: float, name: str) -> str:
    if not math.isfinite(value):
        raise V3FeasibilityError(f"{name} is not finite")
    return value.hex()


def _parse_hex(value: object, name: str) -> float:
    if type(value) is not str:
        raise V3FeasibilityError(f"{name} must be a canonical binary64 hex string")
    try:
        parsed = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise V3FeasibilityError(f"{name} is not a binary64 hex string") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise V3FeasibilityError(f"{name} is not a canonical finite binary64 value")
    return parsed


@dataclass(frozen=True, slots=True)
class FeasibilityDesign:
    """Inputs to a preliminary conditional precision/resource calculation."""

    candidate_count: int
    alpha: float
    practical_margin: int
    target_gap: int
    variance_upper_bound: int
    variance_bound_source: VarianceBoundSource
    calibration_artifact_sha256: str | None
    compute_cap_interactions: int

    def __post_init__(self) -> None:
        candidate_count = _exact_int(
            self.candidate_count,
            "candidate_count",
            minimum=2,
            maximum=statistics.MAX_CANDIDATES,
        )
        if statistics.MAX_SCORE_CELLS // candidate_count < 2:
            raise V3FeasibilityError("candidate_count leaves fewer than two complete blocks")
        _alpha(self.alpha)
        _exact_int(
            self.practical_margin,
            "practical_margin",
            minimum=0,
            maximum=_MAX_CONTRAST_GAP,
        )
        _exact_int(
            self.target_gap,
            "target_gap",
            minimum=0,
            maximum=_MAX_CONTRAST_GAP,
        )
        variance = _exact_int(
            self.variance_upper_bound,
            "variance_upper_bound",
            minimum=0,
            maximum=_MAX_SAMPLE_VARIANCE,
        )
        if self.variance_bound_source not in {
            "bounded_sample_variance_maximum",
            "development_calibration_upper_bound",
        }:
            raise V3FeasibilityError("variance_bound_source is not recognized")
        calibration = _sha256_or_none(
            self.calibration_artifact_sha256,
            "calibration_artifact_sha256",
        )
        if self.variance_bound_source == "bounded_sample_variance_maximum":
            if variance != _BOUNDED_SAMPLE_VARIANCE_MAXIMUM:
                raise V3FeasibilityError(
                    "bounded_sample_variance_maximum requires the exact global "
                    "unbiased-sample-variance ceiling"
                )
            if calibration is not None:
                raise V3FeasibilityError(
                    "bounded_sample_variance_maximum must not claim a calibration artifact"
                )
        elif calibration is None:
            raise V3FeasibilityError(
                "development calibration requires calibration_artifact_sha256"
            )
        _exact_int(
            self.compute_cap_interactions,
            "compute_cap_interactions",
            minimum=0,
            maximum=_MAX_INTERACTIONS,
        )

    def to_payload(self) -> dict[str, object]:
        """Return the exact detached design fields serialized by a receipt."""

        return {
            "alpha_hex": self.alpha.hex(),
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "candidate_count": self.candidate_count,
            "compute_cap_interactions": self.compute_cap_interactions,
            "practical_margin": self.practical_margin,
            "target_gap": self.target_gap,
            "variance_bound_source": self.variance_bound_source,
            "variance_upper_bound": self.variance_upper_bound,
        }


def finite_sample_penalty(
    *,
    candidate_count: int,
    alpha: float,
    block_count: int,
    variance_upper_bound: int,
) -> tuple[float, float, float]:
    """Return the variance, range, and total empirical-Bernstein corrections."""

    candidates = _exact_int(
        candidate_count,
        "candidate_count",
        minimum=2,
        maximum=statistics.MAX_CANDIDATES,
    )
    significance = _alpha(alpha)
    maximum_blocks = statistics.MAX_SCORE_CELLS // candidates
    blocks = _exact_int(
        block_count,
        "block_count",
        minimum=2,
        maximum=maximum_blocks,
    )
    variance = _exact_int(
        variance_upper_bound,
        "variance_upper_bound",
        minimum=0,
        maximum=_MAX_SAMPLE_VARIANCE,
    )
    family_size = candidates * (candidates - 1)
    q = math.log((2.0 * family_size) / significance)
    variance_term = math.sqrt((2.0 * variance * q) / blocks)
    range_term = (
        7.0
        * statistics.CONTRAST_RANGE_WIDTH
        * q
        / (3.0 * (blocks - 1))
    )
    total = variance_term + range_term
    checked_values = (q, variance_term, range_term, total)
    if not all(math.isfinite(item) and item >= 0.0 for item in checked_values):
        raise V3FeasibilityError("finite-sample penalty arithmetic is not finite")
    return variance_term, range_term, total


def _clears_statistical_target(design: FeasibilityDesign, block_count: int) -> bool:
    total = finite_sample_penalty(
        candidate_count=design.candidate_count,
        alpha=design.alpha,
        block_count=block_count,
        variance_upper_bound=design.variance_upper_bound,
    )[2]
    return design.target_gap > design.practical_margin + total


def _minimum_statistical_block_count(design: FeasibilityDesign) -> int | None:
    maximum = statistics.MAX_SCORE_CELLS // design.candidate_count
    if not _clears_statistical_target(design, maximum):
        return None
    low = 2
    high = maximum
    while low < high:
        middle = (low + high) // 2
        if _clears_statistical_target(design, middle):
            high = middle
        else:
            low = middle + 1
    return low


def _bindings() -> dict[str, object]:
    return {
        "analysis_implementation_schema": statistics.ANALYSIS_IMPLEMENTATION_SCHEMA,
        "analysis_implementation_sha256": statistics.ANALYSIS_IMPLEMENTATION_SHA256,
        "conditional_precision_implementation_schema": (
            CONDITIONAL_PRECISION_IMPLEMENTATION_SCHEMA
        ),
        "conditional_precision_implementation_sha256": (
            CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256
        ),
        "candidate_universe_schema": (
            universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION
        ),
        "candidate_universe_sha256": universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
        "cumulative_reward_metric_schema": (
            protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
        ),
        "cumulative_reward_metric_sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
        "contrast_range_width": statistics.CONTRAST_RANGE_WIDTH,
        "horizon": statistics.HORIZON,
        "maximum_candidates": statistics.MAX_CANDIDATES,
        "maximum_score_cells": statistics.MAX_SCORE_CELLS,
        "trial_block_generator_plan_schema": (
            protocol.TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION
        ),
        "trial_block_generator_plan_sha256": (
            protocol.TRIAL_BLOCK_GENERATOR_PLAN_SHA256
        ),
    }


def _implementation_descriptor() -> dict[str, object]:
    return {
        "schema": CONDITIONAL_PRECISION_IMPLEMENTATION_SCHEMA,
        "classification": "generic_preliminary_conditional_precision_formula",
        "family_size": "candidate_count * (candidate_count - 1)",
        "multiplicity_term": "log((2 * family_size) / alpha)",
        "variance_correction": "sqrt((2 * variance_upper_bound * q) / block_count)",
        "range_correction": (
            "7 * contrast_range_width * q / (3 * (block_count - 1))"
        ),
        "clearance_rule": (
            "target_gap > practical_margin + variance_correction + range_correction"
        ),
        "minimum_search": "monotone_integer_binary_search_over_block_count_ge_2",
        "power_or_beta_modeled": False,
        "campaign_panel_bound": False,
        "calibration_receipt_validated": False,
        "execution_authority": False,
    }


_IMPLEMENTATION_DESCRIPTOR: Final = _implementation_descriptor()
_IMPLEMENTATION_DESCRIPTOR_BYTES: Final = json.dumps(
    _IMPLEMENTATION_DESCRIPTOR,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
_EXPECTED_CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256: Final = (
    "b8f0d2698d58a4b019adafbf620124dc8880a72c6d8b6ad4211a219d1e30c2cd"
)
_actual_implementation_sha256 = hashlib.sha256(
    _IMPLEMENTATION_DESCRIPTOR_BYTES
).hexdigest()
if _actual_implementation_sha256 != (
    _EXPECTED_CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256
):
    raise AssertionError("conditional-precision implementation descriptor drift")
CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256: Final = (
    _EXPECTED_CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256
)


def _receipt_body(design: FeasibilityDesign) -> dict[str, object]:
    if type(design) is not FeasibilityDesign:
        raise V3FeasibilityError("design must be a FeasibilityDesign")
    maximum_blocks = statistics.MAX_SCORE_CELLS // design.candidate_count
    minimum_blocks = _minimum_statistical_block_count(design)
    evaluated_blocks = maximum_blocks if minimum_blocks is None else minimum_blocks
    variance_term, range_term, total_penalty = finite_sample_penalty(
        candidate_count=design.candidate_count,
        alpha=design.alpha,
        block_count=evaluated_blocks,
        variance_upper_bound=design.variance_upper_bound,
    )
    family_size = design.candidate_count * (design.candidate_count - 1)
    q = math.log((2.0 * family_size) / design.alpha)
    maximum_interactions = (
        design.candidate_count * maximum_blocks * statistics.HORIZON
    )
    required_interactions = (
        None
        if minimum_blocks is None
        else design.candidate_count * minimum_blocks * statistics.HORIZON
    )
    conditional_gap_cleared = minimum_blocks is not None
    inferential_interaction_cap_cleared = (
        required_interactions is not None
        and required_interactions <= design.compute_cap_interactions
    )
    global_variance_ceiling_used = (
        design.variance_bound_source == "bounded_sample_variance_maximum"
    )
    preliminary_conditional_design_cleared = (
        conditional_gap_cleared
        and inferential_interaction_cap_cleared
        and global_variance_ceiling_used
    )
    if not global_variance_ceiling_used:
        status = "unvalidated_calibration_assertion_not_gate_eligible"
    elif preliminary_conditional_design_cleared:
        status = "preliminary_conditional_precision_cleared"
    else:
        status = "preliminary_conditional_precision_not_cleared"
    return {
        "arithmetic": {
            "conditional_gap_cleared": conditional_gap_cleared,
            "evaluated_block_count": evaluated_blocks,
            "family_size": family_size,
            "global_variance_ceiling_used": global_variance_ceiling_used,
            "inferential_interaction_cap_cleared": (
                inferential_interaction_cap_cleared
            ),
            "maximum_blocks_by_score_cell_cap": maximum_blocks,
            "maximum_confirmatory_inferential_interactions_before_retries": (
                maximum_interactions
            ),
            "minimum_statistical_block_count": minimum_blocks,
            "power_or_beta_modeled": False,
            "preliminary_conditional_design_cleared": (
                preliminary_conditional_design_cleared
            ),
            "q_hex": _canonical_hex(q, "q"),
            "range_correction_hex": _canonical_hex(range_term, "range correction"),
            "required_confirmatory_inferential_interactions_before_retries": (
                required_interactions
            ),
            "score_cells_at_evaluated_block_count": (
                design.candidate_count * evaluated_blocks
            ),
            "total_correction_hex": _canonical_hex(total_penalty, "total correction"),
            "variance_correction_hex": _canonical_hex(
                variance_term,
                "variance correction",
            ),
        },
        "authority": {
            "campaign_gate_eligible": False,
            "compute_matched_claim_allowed": False,
            "execution_authorized": False,
            "heldout_tokens_may_be_requested": False,
            "literature_sota_claim_allowed": False,
            "named_panel_superiority_claim_allowed": False,
            "performance_claim_allowed": False,
            "preregistration_artifact": False,
            "resource_matched_claim_allowed": False,
            "scientific_promotion_allowed": False,
            "standalone_result_use_allowed": False,
            "universal_sota_claim_allowed": False,
        },
        "bindings": _bindings(),
        "classification": (
            "generic_preliminary_nonpromoting_conditional_precision_arithmetic"
        ),
        "conditional_assumptions": {
            "hypothetical_observed_gap_equals_target_gap": True,
            "power_or_beta_modeled": False,
            "variance_does_not_exceed_asserted_ceiling": True,
        },
        "design": design.to_payload(),
        "interaction_accounting_scope": {
            "included": "candidate_count * block_count * horizon",
            "excluded": [
                "development_selection",
                "development_calibration",
                "qualification",
                "descriptive_arms",
                "retries",
                "diagnostics",
            ],
            "unit": "confirmatory_inferential_environment_interactions_before_retries",
        },
        "limitations": [
            "Candidate identifiers remain unselected and are not bound by this receipt.",
            "A calibration digest binds an asserted variance ceiling, not its validity.",
            "A fractional calibrated variance ceiling must be rounded upward to an integer.",
            "This conditional calculation is not a power analysis or campaign gate.",
            "A campaign receipt must bind the exact accepted 11-arm panel and source closure.",
            "Conditional precision does not authorize execution or held-out randomness.",
            "Environment interactions are not a compute-, memory-, or wall-time equality claim.",
        ],
        "schema": FEASIBILITY_RECEIPT_SCHEMA,
        "status": status,
    }


def _canonical_bytes(value: object) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise V3FeasibilityError("receipt is not finite canonical JSON") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise V3FeasibilityError("receipt exceeds its canonical byte limit")
    return raw


def canonical_feasibility_receipt_body_bytes(design: FeasibilityDesign) -> bytes:
    """Return canonical receipt bytes before the content digest is attached."""

    return _canonical_bytes(_receipt_body(design))


def conditional_precision_implementation_descriptor() -> dict[str, Any]:
    """Return a detached description of the duplicated design-time formula."""

    return cast(
        dict[str, Any],
        json.loads(_IMPLEMENTATION_DESCRIPTOR_BYTES.decode("utf-8")),
    )


def canonical_conditional_precision_implementation_descriptor_bytes() -> bytes:
    """Return canonical bytes for the formula/claim-boundary descriptor."""

    return _IMPLEMENTATION_DESCRIPTOR_BYTES


def build_feasibility_receipt(design: FeasibilityDesign) -> dict[str, Any]:
    """Build a detached, content-addressed, authority-denying receipt."""

    body = _receipt_body(design)
    result = cast(dict[str, Any], json.loads(_canonical_bytes(body)))
    result["payload_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return cast(dict[str, Any], json.loads(_canonical_bytes(result)))


def canonical_feasibility_receipt_bytes(design: FeasibilityDesign) -> bytes:
    """Return the complete canonical receipt encoding."""

    return _canonical_bytes(build_feasibility_receipt(design))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V3FeasibilityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> object:
    raise V3FeasibilityError(f"non-finite JSON constant {token!r}")


def _exact_keys(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise V3FeasibilityError(f"{name} must be a plain object")
    mapping = cast(dict[str, object], value)
    if frozenset(mapping) != expected:
        raise V3FeasibilityError(f"{name} has missing or extra keys")
    return mapping


def _design_from_payload(value: object) -> FeasibilityDesign:
    payload = _exact_keys(
        value,
        frozenset(
            {
                "alpha_hex",
                "calibration_artifact_sha256",
                "candidate_count",
                "compute_cap_interactions",
                "practical_margin",
                "target_gap",
                "variance_bound_source",
                "variance_upper_bound",
            }
        ),
        "receipt design",
    )
    source = payload["variance_bound_source"]
    if type(source) is not str:
        raise V3FeasibilityError("variance_bound_source must be a string")
    return FeasibilityDesign(
        candidate_count=cast(int, payload["candidate_count"]),
        alpha=_parse_hex(payload["alpha_hex"], "alpha_hex"),
        practical_margin=cast(int, payload["practical_margin"]),
        target_gap=cast(int, payload["target_gap"]),
        variance_upper_bound=cast(int, payload["variance_upper_bound"]),
        variance_bound_source=cast(VarianceBoundSource, source),
        calibration_artifact_sha256=cast(
            str | None,
            payload["calibration_artifact_sha256"],
        ),
        compute_cap_interactions=cast(int, payload["compute_cap_interactions"]),
    )


def parse_feasibility_receipt(raw: bytes) -> dict[str, Any]:
    """Strictly replay a canonical receipt and every arithmetic field."""

    if type(raw) is not bytes:
        raise V3FeasibilityError("receipt must be bytes")
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise V3FeasibilityError("receipt exceeds its byte limit")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except V3FeasibilityError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise V3FeasibilityError("receipt is not strict UTF-8 JSON") from exc
    payload = _exact_keys(
        decoded,
        frozenset(
            {
                "arithmetic",
                "authority",
                "bindings",
                "classification",
                "conditional_assumptions",
                "design",
                "interaction_accounting_scope",
                "limitations",
                "payload_sha256",
                "schema",
                "status",
            }
        ),
        "receipt",
    )
    if _canonical_bytes(payload) != raw:
        raise V3FeasibilityError("receipt is not in exact canonical form")
    digest = payload["payload_sha256"]
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise V3FeasibilityError("payload_sha256 is invalid")
    design = _design_from_payload(payload["design"])
    expected = build_feasibility_receipt(design)
    if payload != expected:
        raise V3FeasibilityError("receipt does not replay from its exact design")
    return expected


__all__ = [
    "CONDITIONAL_PRECISION_IMPLEMENTATION_SCHEMA",
    "CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256",
    "FEASIBILITY_RECEIPT_SCHEMA",
    "FeasibilityDesign",
    "V3FeasibilityError",
    "build_feasibility_receipt",
    "canonical_conditional_precision_implementation_descriptor_bytes",
    "canonical_feasibility_receipt_body_bytes",
    "canonical_feasibility_receipt_bytes",
    "conditional_precision_implementation_descriptor",
    "finite_sample_penalty",
    "parse_feasibility_receipt",
]
