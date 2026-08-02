"""Fail-closed threshold freezing for hidden-regime factorial calibration.

This module is deliberately non-executing.  It accepts only the canonical
aggregate produced by the consumed, development-only calibration protocol and
materializes one atomic decision receipt.  A statistically insufficient but
otherwise valid aggregate produces an immutable valid-rejection receipt;
malformed inputs and provenance mismatches raise without producing a receipt.
The checks here are defense in depth: the authoritative ZIP workflow must
first exact-recompute the aggregate from all shards and the managed ledger.

All continuous calculations are performed in the aggregate's already-oriented
space.  Binary64 inputs remain bound by their canonical ``float.hex`` strings,
while frozen thresholds are serialized as exact decimal strings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NoReturn, cast

from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    N_MATCHED_CASES,
    N_SEED_PAIRS,
    SEED_SNAPSHOT_SHA256,
    THRESHOLD_FREEZE_RECEIPT_SCHEMA,
    GateFamilySpec,
    HiddenRegimeFactorialCalibrationDesign,
    MetricContract,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
    canonical_sha256,
)

# The calibration aggregate schema is kept here rather than importing the
# execution module: the threshold engine must remain a small, non-executing
# consumer and must never acquire a path to the calibration worker entry point.
CALIBRATION_AGGREGATE_SCHEMA = "alberta.hidden-regime-factorial.calibration-aggregate.v4"
CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA = (
    "alberta.hidden-regime-factorial.mandatory-audit-summary.v1"
)
READINESS_EQUIVALENCE_CERTIFICATION_ID = (
    "checkpoint_resume_and_decentralized_role_bit_exact_equivalence"
)

THRESHOLD_FREEZE_DECISION_FROZEN = "thresholds_frozen"
THRESHOLD_FREEZE_DECISION_REJECTION = "valid_calibration_rejection"
MANDATORY_STATISTICAL_ENDPOINT_COUNT = 35
# Hash of the canonical ordered list of full identity objects.  Each object is
# ``{"gate_family_id": ..., "reference": {kind, metric_id, condition|estimand_id}}``.
MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256 = (
    "1644bbba320be78e75491b7652a4d73f5fb8db2361d33426614a1fd22994de45"
)
# Hash of the canonical ordered list of 35 strings obtained by hashing each
# full identity object above individually.  This is intentionally distinct
# from the object-list digest.
MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256 = (
    "769325b8f1f52a0f18f095e67a88296ac63ecc8062be238a3c091b80493c91b1"
)
CONTINUOUS_THRESHOLD_QUANTUM = Decimal("0.0001")
EXPECTED_POOLED_N = 30
EXPECTED_MANIFEST_N = 10

_SHA256_LENGTH = 64
_STATISTICAL_THRESHOLD_STATUS = "unset_pending_consumed_calibration_outcomes"
_STATISTICAL_GATE_DECISION = "not_evaluated_no_thresholds"

_STRATUM_KEYS = {
    "eligible_n",
    "conditional_unobserved_n",
    "conditional_unobserved_seed_indices",
    "structural_missing_n",
    "structural_missing_seed_indices",
    "observed_n",
    "mean_hex",
    "sample_standard_deviation_hex",
    "standard_error_hex",
    "student_t_0_95_critical_hex",
    "one_sided_95_percent_lower_confidence_bound_hex",
    "wins",
    "ties",
    "losses",
}
_STATISTICS_KEYS = {"pooled", "by_manifest", "worst_manifest"}
_WORST_MANIFEST_KEYS = {
    "minimum_manifest_mean_hex",
    "minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex",
    "minimum_manifest_wins",
    "maximum_manifest_structural_missing_n",
    "maximum_manifest_conditional_unobserved_n",
}


class ThresholdFreezeError(RuntimeError):
    """Raised when a threshold decision input or receipt is not exact."""


def _fail(message: str) -> NoReturn:
    raise ThresholdFreezeError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object, label: str) -> str:
    _require(_is_sha256(value), f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _plain_dict(value: object, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a plain object")
    return cast(dict[str, object], value)


def _plain_list(value: object, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a plain array")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys are not exact")


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be a strict integer")
    result = cast(int, value)
    if maximum is not None:
        _require(result <= maximum, f"{label} exceeds {maximum}")
    return result


def _float_from_hex(value: object, label: str) -> float:
    _require(type(value) is str and bool(value), f"{label} must be a canonical float hex string")
    text = cast(str, value)
    try:
        result = float.fromhex(text)
    except ValueError as exc:
        raise ThresholdFreezeError(f"{label} is not a float hex string") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    _require(result.hex() == text, f"{label} is not canonical float hex")
    return result


def _optional_float_hex(value: object, label: str) -> float | None:
    return None if value is None else _float_from_hex(value, label)


def _decimal_from_float(value: float) -> Decimal:
    return Decimal.from_float(value)


def _decimal_text(value: Decimal) -> str:
    _require(value.is_finite(), "decimal output must be finite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _quantized_decimal_text(value: Decimal) -> str:
    _require(value.is_finite(), "quantized decimal output must be finite")
    return format(value, ".4f")


def _payload_with_digest(body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    return {**normalized, "receipt_payload_sha256": canonical_sha256(normalized)}


def _validated_digest_payload(
    payload: Mapping[str, object],
    *,
    digest_field: str,
    label: str,
) -> dict[str, object]:
    try:
        normalized = dict(payload)
        canonical_json_bytes(normalized)
    except (TypeError, ValueError) as exc:
        raise ThresholdFreezeError(f"{label} is not canonical JSON data") from exc
    digest = normalized.pop(digest_field, None)
    _sha256(digest, f"{label}.{digest_field}")
    _require(canonical_sha256(normalized) == digest, f"{label} digest differs")
    return normalized


@dataclass(frozen=True, slots=True)
class _ValidatedStratum:
    payload: dict[str, object]
    observed_n: int
    conditional_unobserved_n: int
    structural_missing_n: int
    mean: float | None
    lower_bound: float | None
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True, slots=True)
class _ValidatedStatistics:
    pooled: _ValidatedStratum
    manifests: tuple[tuple[str, _ValidatedStratum], ...]
    worst_manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class _EndpointSource:
    gate_family_id: str
    reference: dict[str, object]
    orientation: str
    oriented_null_hex: str
    statistics: _ValidatedStatistics


def _strict_seed_indices(
    value: object,
    label: str,
    *,
    allowed: set[int],
) -> tuple[int, ...]:
    raw = _plain_list(value, label)
    indices = tuple(_strict_int(item, f"{label}[]", maximum=N_SEED_PAIRS - 1) for item in raw)
    _require(len(set(indices)) == len(indices), f"{label} contains duplicates")
    _require(tuple(sorted(indices)) == indices, f"{label} is not ordered")
    _require(set(indices).issubset(allowed), f"{label} contains an out-of-stratum seed")
    return indices


def _validate_stratum(
    value: object,
    *,
    label: str,
    expected_eligible_n: int,
    allowed_seed_indices: set[int],
) -> _ValidatedStratum:
    payload = _plain_dict(value, label)
    _exact_keys(payload, _STRATUM_KEYS, label)
    eligible_n = _strict_int(payload["eligible_n"], f"{label}.eligible_n")
    _require(
        eligible_n == expected_eligible_n,
        f"{label}.eligible_n must equal {expected_eligible_n}",
    )
    conditional_n = _strict_int(
        payload["conditional_unobserved_n"],
        f"{label}.conditional_unobserved_n",
        maximum=eligible_n,
    )
    structural_n = _strict_int(
        payload["structural_missing_n"],
        f"{label}.structural_missing_n",
        maximum=eligible_n,
    )
    conditional_indices = _strict_seed_indices(
        payload["conditional_unobserved_seed_indices"],
        f"{label}.conditional_unobserved_seed_indices",
        allowed=allowed_seed_indices,
    )
    structural_indices = _strict_seed_indices(
        payload["structural_missing_seed_indices"],
        f"{label}.structural_missing_seed_indices",
        allowed=allowed_seed_indices,
    )
    _require(len(conditional_indices) == conditional_n, f"{label} conditional count differs")
    _require(len(structural_indices) == structural_n, f"{label} structural count differs")
    _require(
        set(conditional_indices).isdisjoint(structural_indices),
        f"{label} missingness categories overlap",
    )
    observed_n = _strict_int(payload["observed_n"], f"{label}.observed_n", maximum=eligible_n)
    _require(
        observed_n + conditional_n + structural_n == eligible_n,
        f"{label} observed and missing counts do not partition eligible cases",
    )
    mean = _optional_float_hex(payload["mean_hex"], f"{label}.mean_hex")
    sample_sd = _optional_float_hex(
        payload["sample_standard_deviation_hex"],
        f"{label}.sample_standard_deviation_hex",
    )
    standard_error = _optional_float_hex(
        payload["standard_error_hex"],
        f"{label}.standard_error_hex",
    )
    critical = _optional_float_hex(
        payload["student_t_0_95_critical_hex"],
        f"{label}.student_t_0_95_critical_hex",
    )
    lower_bound = _optional_float_hex(
        payload["one_sided_95_percent_lower_confidence_bound_hex"],
        f"{label}.one_sided_95_percent_lower_confidence_bound_hex",
    )
    _require((mean is None) == (observed_n == 0), f"{label}.mean presence differs from n")
    inferential = (sample_sd, standard_error, critical, lower_bound)
    _require(
        all(item is None for item in inferential)
        if observed_n < 2
        else all(item is not None for item in inferential),
        f"{label} inferential statistic presence differs from n",
    )
    if observed_n >= 2:
        assert mean is not None
        assert sample_sd is not None
        assert standard_error is not None
        assert critical is not None
        assert lower_bound is not None
        _require(sample_sd >= 0.0, f"{label}.sample_standard_deviation must be nonnegative")
        _require(standard_error >= 0.0, f"{label}.standard_error must be nonnegative")
        _require(critical > 0.0, f"{label}.student_t critical must be positive")
        _require(lower_bound <= mean, f"{label}.lower confidence bound exceeds its mean")
        expected_standard_error = float(sample_sd / math.sqrt(observed_n))
        _require(
            payload["standard_error_hex"] == expected_standard_error.hex(),
            f"{label}.standard_error is algebraically inconsistent with sample SD and n",
        )
        expected_lower_bound = float(mean - critical * standard_error)
        _require(
            payload["one_sided_95_percent_lower_confidence_bound_hex"]
            == expected_lower_bound.hex(),
            f"{label}.lower confidence bound is algebraically inconsistent",
        )
    wins = _strict_int(payload["wins"], f"{label}.wins", maximum=observed_n)
    ties = _strict_int(payload["ties"], f"{label}.ties", maximum=observed_n)
    losses = _strict_int(payload["losses"], f"{label}.losses", maximum=observed_n)
    _require(wins + ties + losses == observed_n, f"{label} win/tie/loss counts differ from n")
    return _ValidatedStratum(
        payload=dict(payload),
        observed_n=observed_n,
        conditional_unobserved_n=conditional_n,
        structural_missing_n=structural_n,
        mean=mean,
        lower_bound=lower_bound,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def _validate_statistics(value: object, label: str) -> _ValidatedStatistics:
    payload = _plain_dict(value, label)
    _exact_keys(payload, _STATISTICS_KEYS, label)
    pooled = _validate_stratum(
        payload["pooled"],
        label=f"{label}.pooled",
        expected_eligible_n=EXPECTED_POOLED_N,
        allowed_seed_indices=set(range(N_SEED_PAIRS)),
    )
    raw_manifests = _plain_list(payload["by_manifest"], f"{label}.by_manifest")
    _require(
        len(raw_manifests) == len(CALIBRATION_MANIFEST_ORDER),
        f"{label} must contain exactly three manifest strata",
    )
    manifests: list[tuple[str, _ValidatedStratum]] = []
    for manifest_index, manifest_name in enumerate(CALIBRATION_MANIFEST_ORDER):
        raw = _plain_dict(raw_manifests[manifest_index], f"{label}.by_manifest[]")
        _exact_keys(raw, _STRATUM_KEYS | {"manifest_name"}, f"{label}.by_manifest[]")
        _require(raw["manifest_name"] == manifest_name, f"{label} manifest order differs")
        assigned = {
            seed_index
            for seed_index in range(N_SEED_PAIRS)
            if seed_index % len(CALIBRATION_MANIFEST_ORDER) == manifest_index
        }
        stratum = _validate_stratum(
            {key: item for key, item in raw.items() if key != "manifest_name"},
            label=f"{label}.{manifest_name}",
            expected_eligible_n=EXPECTED_MANIFEST_N,
            allowed_seed_indices=assigned,
        )
        manifests.append((manifest_name, stratum))

    for field in (
        "conditional_unobserved_seed_indices",
        "structural_missing_seed_indices",
    ):
        pooled_indices = cast(list[object], pooled.payload[field])
        manifest_indices = sorted(
            cast(int, seed_index)
            for _, stratum in manifests
            for seed_index in cast(list[object], stratum.payload[field])
        )
        _require(pooled_indices == manifest_indices, f"{label} pooled {field} differs from strata")
    _require(
        pooled.observed_n == sum(stratum.observed_n for _, stratum in manifests),
        f"{label} pooled observed_n differs from strata",
    )
    _require(
        (pooled.wins, pooled.ties, pooled.losses)
        == tuple(
            sum(getattr(stratum, field) for _, stratum in manifests)
            for field in ("wins", "ties", "losses")
        ),
        f"{label} pooled win/tie/loss counts differ from strata",
    )

    worst = _plain_dict(payload["worst_manifest"], f"{label}.worst_manifest")
    _exact_keys(worst, _WORST_MANIFEST_KEYS, f"{label}.worst_manifest")
    manifest_means = [stratum.mean for _, stratum in manifests]
    manifest_bounds = [stratum.lower_bound for _, stratum in manifests]
    expected_mean = (
        None
        if any(item is None for item in manifest_means)
        else min(cast(float, item) for item in manifest_means)
    )
    expected_bound = (
        None
        if any(item is None for item in manifest_bounds)
        else min(cast(float, item) for item in manifest_bounds)
    )
    actual_mean = _optional_float_hex(
        worst["minimum_manifest_mean_hex"],
        f"{label}.worst_manifest.minimum_manifest_mean_hex",
    )
    actual_bound = _optional_float_hex(
        worst["minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex"],
        f"{label}.worst_manifest.minimum_manifest_bound_hex",
    )
    _require(actual_mean == expected_mean, f"{label} worst manifest mean differs")
    _require(actual_bound == expected_bound, f"{label} worst manifest bound differs")
    _require(
        worst["minimum_manifest_mean_hex"]
        == (None if expected_mean is None else expected_mean.hex()),
        f"{label} worst manifest mean hex differs",
    )
    _require(
        worst["minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex"]
        == (None if expected_bound is None else expected_bound.hex()),
        f"{label} worst manifest bound hex differs",
    )
    minimum_wins = _strict_int(
        worst["minimum_manifest_wins"],
        f"{label}.worst_manifest.minimum_manifest_wins",
        maximum=EXPECTED_MANIFEST_N,
    )
    maximum_structural = _strict_int(
        worst["maximum_manifest_structural_missing_n"],
        f"{label}.worst_manifest.maximum_manifest_structural_missing_n",
        maximum=EXPECTED_MANIFEST_N,
    )
    maximum_conditional = _strict_int(
        worst["maximum_manifest_conditional_unobserved_n"],
        f"{label}.worst_manifest.maximum_manifest_conditional_unobserved_n",
        maximum=EXPECTED_MANIFEST_N,
    )
    _require(
        minimum_wins == min(stratum.wins for _, stratum in manifests),
        f"{label} worst manifest wins differ",
    )
    _require(
        maximum_structural == max(stratum.structural_missing_n for _, stratum in manifests),
        f"{label} worst structural missingness differs",
    )
    _require(
        maximum_conditional == max(stratum.conditional_unobserved_n for _, stratum in manifests),
        f"{label} worst conditional missingness differs",
    )
    return _ValidatedStatistics(pooled, tuple(manifests), dict(worst))


def _metric_contracts(
    design: HiddenRegimeFactorialCalibrationDesign,
) -> dict[str, MetricContract]:
    return {metric.metric_id: metric for metric in design.metrics}


def _expected_statistical_references(
    family: GateFamilySpec,
    support_metric_ids: set[str],
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    if family.estimand_ids:
        for estimand_id in family.estimand_ids:
            for metric_id in family.metric_ids:
                references.append(
                    {
                        "kind": (
                            "paired_population_support_level"
                            if metric_id in support_metric_ids
                            else "paired_contrast"
                        ),
                        "metric_id": metric_id,
                        "estimand_id": estimand_id,
                    }
                )
    else:
        for condition in family.conditions:
            for metric_id in family.metric_ids:
                references.append(
                    {
                        "kind": "absolute_level",
                        "metric_id": metric_id,
                        "condition": condition,
                    }
                )
    return references


def _frozen_mandatory_statistical_endpoint_identities(
    design: HiddenRegimeFactorialCalibrationDesign,
) -> list[dict[str, object]]:
    support_metric_ids = {item.metric_id for item in design.paired_population_support_metrics}
    identities: list[dict[str, object]] = []
    for family in design.gate_families:
        if not family.mandatory or not family.metric_ids:
            continue
        for reference in _expected_statistical_references(family, support_metric_ids):
            identities.append(
                {"gate_family_id": family.gate_family_id, "reference": reference}
            )
    _require(
        len(identities) == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        "frozen mandatory statistical endpoint cardinality differs",
    )
    _require(
        canonical_sha256(identities) == MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
        "frozen ordered mandatory statistical endpoint identity digest differs",
    )
    endpoint_ids = [canonical_sha256(identity) for identity in identities]
    _require(
        canonical_sha256(endpoint_ids) == MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        "frozen ordered mandatory statistical endpoint-ID-list digest differs",
    )
    return identities


def _reference_identity(reference: Mapping[str, object], label: str) -> dict[str, object]:
    kind = reference.get("kind")
    if kind == "absolute_level":
        keys = {"kind", "metric_id", "condition", "statistics_sha256"}
        identity_keys = ("kind", "metric_id", "condition")
    elif kind in {"paired_contrast", "paired_population_support_level"}:
        keys = {"kind", "metric_id", "estimand_id", "statistics_sha256"}
        identity_keys = ("kind", "metric_id", "estimand_id")
    else:
        _fail(f"{label} has an unknown endpoint kind")
    _exact_keys(reference, keys, label)
    _sha256(reference["statistics_sha256"], f"{label}.statistics_sha256")
    return {key: reference[key] for key in identity_keys}


def _summary_indexes(
    aggregate: Mapping[str, object],
    design: HiddenRegimeFactorialCalibrationDesign,
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
]:
    metric_ids = tuple(metric.metric_id for metric in design.metrics)
    expected_level_order = tuple(
        (condition, metric_id)
        for condition in CANONICAL_CONDITION_ORDER
        for metric_id in metric_ids
    )
    level_positions = {key: index for index, key in enumerate(expected_level_order)}
    level_index: dict[tuple[str, str], dict[str, object]] = {}
    observed_level_positions: list[int] = []
    for raw in _plain_list(aggregate["level_summaries"], "aggregate.level_summaries"):
        item = _plain_dict(raw, "aggregate level summary")
        condition = item.get("condition")
        metric_id = item.get("metric_id")
        _require(type(condition) is str and type(metric_id) is str, "level key is invalid")
        key = (cast(str, condition), cast(str, metric_id))
        _require(key in level_positions, "aggregate contains an unexpected level summary row")
        _require(key not in level_index, "duplicate aggregate level endpoint")
        observed_level_positions.append(level_positions[key])
        level_index[key] = item
    _require(
        observed_level_positions == sorted(observed_level_positions),
        "aggregate level summary rows are not in canonical order",
    )
    _require(
        tuple(level_index) == expected_level_order,
        "aggregate level summary rows are incomplete",
    )

    estimand_specs = design.factorial_estimands + design.control_estimands
    estimand_positions = {
        estimand.estimand_id: index for index, estimand in enumerate(estimand_specs)
    }
    allowed_estimand_metrics = {
        estimand.estimand_id: tuple(estimand.metrics) for estimand in estimand_specs
    }
    estimand_index: dict[tuple[str, str], dict[str, object]] = {}
    seen_estimands: set[str] = set()
    observed_estimand_positions: list[int] = []
    for raw in _plain_list(aggregate["estimand_summaries"], "aggregate.estimand_summaries"):
        estimand = _plain_dict(raw, "aggregate estimand summary")
        estimand_id = estimand.get("estimand_id")
        _require(type(estimand_id) is str, "estimand identifier is invalid")
        typed_estimand_id = cast(str, estimand_id)
        _require(
            typed_estimand_id in estimand_positions,
            "aggregate contains an unexpected estimand summary row",
        )
        _require(typed_estimand_id not in seen_estimands, "duplicate aggregate estimand")
        seen_estimands.add(typed_estimand_id)
        observed_estimand_positions.append(estimand_positions[typed_estimand_id])
        estimand_null_hex = estimand.get("oriented_null_hex")
        _require(type(estimand_null_hex) is str, "estimand oriented null is invalid")
        metric_positions = {
            metric_id: index
            for index, metric_id in enumerate(allowed_estimand_metrics[typed_estimand_id])
        }
        observed_metric_positions: list[int] = []
        for raw_metric in _plain_list(estimand.get("metrics"), "aggregate estimand metrics"):
            metric = _plain_dict(raw_metric, "aggregate estimand metric")
            metric_id = metric.get("metric_id")
            _require(type(metric_id) is str, "estimand metric identifier is invalid")
            key = (typed_estimand_id, cast(str, metric_id))
            _require(
                cast(str, metric_id) in metric_positions,
                "aggregate contains an unexpected estimand metric summary row",
            )
            _require(key not in estimand_index, "duplicate aggregate estimand endpoint")
            observed_metric_positions.append(metric_positions[cast(str, metric_id)])
            estimand_index[key] = {**metric, "oriented_null_hex": estimand_null_hex}
        _require(
            observed_metric_positions == sorted(observed_metric_positions),
            "aggregate estimand metric rows are not in canonical order",
        )
        _require(
            tuple(
                metric_id
                for estimand_id, metric_id in estimand_index
                if estimand_id == typed_estimand_id
            )
            == allowed_estimand_metrics[typed_estimand_id],
            "aggregate estimand metric summary rows are incomplete",
        )
    _require(
        observed_estimand_positions == sorted(observed_estimand_positions),
        "aggregate estimand summary rows are not in canonical order",
    )
    _require(
        observed_estimand_positions == list(range(len(estimand_specs))),
        "aggregate estimand summary rows are incomplete",
    )

    expected_support_order = tuple(
        (support.estimand_id, support.metric_id)
        for support in design.paired_population_support_metrics
    )
    support_positions = {key: index for index, key in enumerate(expected_support_order)}
    support_index: dict[tuple[str, str], dict[str, object]] = {}
    observed_support_positions: list[int] = []
    for raw in _plain_list(
        aggregate["paired_population_support_summaries"],
        "aggregate.paired_population_support_summaries",
    ):
        item = _plain_dict(raw, "aggregate support summary")
        estimand_id = item.get("estimand_id")
        metric_id = item.get("metric_id")
        _require(
            type(estimand_id) is str and type(metric_id) is str,
            "support endpoint key is invalid",
        )
        key = (cast(str, estimand_id), cast(str, metric_id))
        _require(key in support_positions, "aggregate contains an unexpected support summary row")
        _require(key not in support_index, "duplicate aggregate support endpoint")
        observed_support_positions.append(support_positions[key])
        support_index[key] = item
    _require(
        observed_support_positions == sorted(observed_support_positions),
        "aggregate support summary rows are not in canonical order",
    )
    _require(
        tuple(support_index) == expected_support_order,
        "aggregate support summary rows are incomplete",
    )
    return level_index, estimand_index, support_index


def _endpoint_source(
    *,
    gate_family_id: str,
    reference: Mapping[str, object],
    metric_contracts: Mapping[str, MetricContract],
    level_index: Mapping[tuple[str, str], Mapping[str, object]],
    estimand_index: Mapping[tuple[str, str], Mapping[str, object]],
    support_index: Mapping[tuple[str, str], Mapping[str, object]],
) -> _EndpointSource:
    identity = _reference_identity(reference, f"{gate_family_id} reference")
    metric_id = cast(str, identity["metric_id"])
    kind = identity["kind"]
    if kind == "absolute_level":
        contract = metric_contracts.get(metric_id)
        _require(contract is not None, f"{gate_family_id} references an unknown metric")
        assert contract is not None
        key = (cast(str, identity["condition"]), metric_id)
        source = level_index.get(key)
        _require(source is not None, f"missing level source for {key!r}")
        assert source is not None
        _require(source.get("orientation_applied") is True, "level orientation was not applied")
        _require(source.get("orientation") == contract.orientation, "level orientation differs")
        expected_null = float(cast(str, contract.null_value_decimal))
        if contract.orientation == "lower":
            expected_null = -expected_null
        orientation = contract.orientation
    elif kind == "paired_contrast":
        contract = metric_contracts.get(metric_id)
        _require(contract is not None, f"{gate_family_id} references an unknown metric")
        assert contract is not None
        key = (cast(str, identity["estimand_id"]), metric_id)
        source = estimand_index.get(key)
        _require(source is not None, f"missing estimand source for {key!r}")
        assert source is not None
        _require(source.get("orientation") == contract.orientation, "contrast orientation differs")
        expected_null = 0.0
        orientation = contract.orientation
    else:
        key = (cast(str, identity["estimand_id"]), metric_id)
        source = support_index.get(key)
        _require(source is not None, f"missing support source for {key!r}")
        assert source is not None
        _require(source.get("orientation") == "higher", "support orientation differs")
        expected_null = 0.0
        orientation = "higher"
    statistics = _plain_dict(source.get("statistics"), "endpoint statistics")
    _require(
        canonical_sha256(statistics) == reference["statistics_sha256"],
        f"{gate_family_id} statistics digest differs from its source",
    )
    null_hex = source.get("oriented_null_hex")
    _require(type(null_hex) is str, "mandatory endpoint lacks an oriented null")
    _require(
        cast(str, null_hex) == expected_null.hex(),
        f"{gate_family_id} oriented null differs from the frozen metric contract",
    )
    return _EndpointSource(
        gate_family_id=gate_family_id,
        reference=dict(reference),
        orientation=orientation,
        oriented_null_hex=cast(str, null_hex),
        statistics=_validate_statistics(statistics, f"{gate_family_id}.{metric_id}.statistics"),
    )


def _validate_statistical_gate_references(
    aggregate: Mapping[str, object],
    design: HiddenRegimeFactorialCalibrationDesign,
) -> tuple[list[_EndpointSource], dict[str, object]]:
    frozen_identities = _frozen_mandatory_statistical_endpoint_identities(design)
    raw_results = _plain_list(aggregate["mandatory_gate_results"], "mandatory_gate_results")
    mandatory_families = tuple(family for family in design.gate_families if family.mandatory)
    _require(
        len(raw_results) == len(mandatory_families),
        "mandatory gate family cardinality differs",
    )
    level_index, estimand_index, support_index = _summary_indexes(aggregate, design)
    metric_contracts = _metric_contracts(design)
    support_metric_ids = {item.metric_id for item in design.paired_population_support_metrics}
    endpoints: list[_EndpointSource] = []
    audit_result: dict[str, object] | None = None
    seen_endpoint_ids: set[str] = set()
    for index, family in enumerate(mandatory_families):
        result = _plain_dict(raw_results[index], f"mandatory gate result {index}")
        _require(result.get("gate_family_id") == family.gate_family_id, "gate family order differs")
        _require(result.get("mandatory") is True, "mandatory gate is not marked mandatory")
        if not family.metric_ids:
            audit_result = result
            continue
        _require(
            result.get("threshold_status") == _STATISTICAL_THRESHOLD_STATUS,
            f"{family.gate_family_id} threshold status differs",
        )
        _require(
            result.get("decision") == _STATISTICAL_GATE_DECISION,
            f"{family.gate_family_id} was decided before threshold freezing",
        )
        raw_references = _plain_list(
            result.get("references"),
            f"{family.gate_family_id}.references",
        )
        expected = _expected_statistical_references(family, support_metric_ids)
        _require(
            len(raw_references) == len(expected),
            f"{family.gate_family_id} reference cardinality differs",
        )
        for reference_index, raw_reference in enumerate(raw_references):
            reference = _plain_dict(raw_reference, f"{family.gate_family_id} reference")
            identity = _reference_identity(reference, f"{family.gate_family_id} reference")
            _require(
                identity == expected[reference_index],
                f"{family.gate_family_id} reference differs",
            )
            endpoint_id = canonical_sha256(
                {"gate_family_id": family.gate_family_id, "reference": identity}
            )
            _require(
                endpoint_id not in seen_endpoint_ids,
                "duplicate mandatory statistical endpoint",
            )
            seen_endpoint_ids.add(endpoint_id)
            endpoints.append(
                _endpoint_source(
                    gate_family_id=family.gate_family_id,
                    reference=reference,
                    metric_contracts=metric_contracts,
                    level_index=level_index,
                    estimand_index=estimand_index,
                    support_index=support_index,
                )
            )
    _require(audit_result is not None, "mandatory audit gate result is missing")
    _require(
        len(endpoints) == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        "mandatory statistical endpoint cardinality must equal 35",
    )
    observed_identities = [
        {
            "gate_family_id": endpoint.gate_family_id,
            "reference": _reference_identity(endpoint.reference, "mandatory endpoint reference"),
        }
        for endpoint in endpoints
    ]
    _require(
        observed_identities == frozen_identities,
        "mandatory statistical endpoint identities differ from the frozen order",
    )
    _require(
        canonical_sha256(observed_identities)
        == MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
        "observed mandatory statistical endpoint identity digest differs",
    )
    _require(
        canonical_sha256([canonical_sha256(identity) for identity in observed_identities])
        == MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        "observed mandatory statistical endpoint-ID-list digest differs",
    )
    assert audit_result is not None
    audit_summary = _plain_dict(
        aggregate.get("mandatory_audit_summary"),
        "mandatory audit summary",
    )
    requirement_results = [
        _plain_dict(item, "mandatory audit requirement result")
        for item in _plain_list(
            audit_summary.get("requirement_results"),
            "mandatory audit requirement results",
        )
    ]
    expected_audit_references = [
        {
            "kind": "threshold_independent_audit_requirement",
            "requirement_id": item["requirement_id"],
            "decision": item["decision"],
            "requirement_result_sha256": item["requirement_result_sha256"],
            "required_references_sha256": item["required_references_sha256"],
        }
        for item in requirement_results
    ]
    _exact_keys(
        audit_result,
        {"gate_family_id", "mandatory", "threshold_status", "decision", "references"},
        "mandatory audit gate result",
    )
    _require(
        audit_result["threshold_status"] == "not_applicable_nonstatistical",
        "mandatory audit threshold status differs",
    )
    _require(
        audit_result["decision"] == audit_summary["decision"],
        "mandatory audit gate and summary decisions differ",
    )
    _require(
        audit_result["references"] == expected_audit_references,
        "mandatory audit gate references differ from the audit summary",
    )
    return endpoints, audit_result


def _validate_case_ledger(value: object) -> tuple[list[dict[str, object]], bool]:
    raw = _plain_list(value, "aggregate.case_ledger")
    _require(len(raw) == N_MATCHED_CASES, "case ledger must contain exactly 240 cases")
    expected_keys = {
        "case_index",
        "seed_index",
        "condition",
        "manifest_name",
        "case_shard_payload_sha256",
        "case_request_binding_sha256",
        "summary_sha256",
        "resource_sha256",
        "worker_provenance_sha256",
        "execution_record_binding",
        "finalized_record_sha256",
        "shard_canonical_sha256",
        "primitive_trace_sha256",
    }
    records: list[dict[str, object]] = []
    shard_digests: set[str] = set()
    for case_index, raw_record in enumerate(raw):
        record = _plain_dict(raw_record, f"case_ledger[{case_index}]")
        _exact_keys(record, expected_keys, f"case_ledger[{case_index}]")
        _require(record["case_index"] == case_index, "case ledger is not in case order")
        seed_index = case_index // len(CANONICAL_CONDITION_ORDER)
        _require(record["seed_index"] == seed_index, "case ledger seed assignment differs")
        _require(
            record["condition"] == CANONICAL_CONDITION_ORDER[case_index % 8],
            "case ledger condition assignment differs",
        )
        _require(
            record["manifest_name"]
            == CALIBRATION_MANIFEST_ORDER[seed_index % len(CALIBRATION_MANIFEST_ORDER)],
            "case ledger manifest assignment differs",
        )
        for field in expected_keys - {
            "case_index",
            "seed_index",
            "condition",
            "manifest_name",
            "execution_record_binding",
        }:
            _sha256(record[field], f"case_ledger[{case_index}].{field}")
        shard_digest = cast(str, record["case_shard_payload_sha256"])
        _require(shard_digest not in shard_digests, "case ledger repeats a shard digest")
        shard_digests.add(shard_digest)
        _plain_dict(record["execution_record_binding"], "case execution record binding")
        records.append(dict(record))
    consumed = {cast(int, record["seed_index"]) for record in records} == set(range(N_SEED_PAIRS))
    return records, consumed


def _validate_managed_ledger(value: object) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = _plain_dict(value, "aggregate.managed_ledger_snapshot")
    body = _validated_digest_payload(
        snapshot,
        digest_field="inventory_sha256",
        label="managed ledger snapshot",
    )
    expected_indices = list(range(N_MATCHED_CASES))
    for field in ("started_case_indices", "completed_case_indices", "finalized_case_indices"):
        _require(body.get(field) == expected_indices, f"managed ledger {field} is incomplete")
    for field in ("learner_interrupted_case_indices", "post_audit_unfinalized_case_indices"):
        _require(body.get(field) == [], f"managed ledger {field} is not empty")
    _require(body.get("expected_case_count") == N_MATCHED_CASES, "ledger case count differs")
    for field in ("started_record_count", "completed_record_count", "finalized_record_count"):
        _require(body.get(field) == N_MATCHED_CASES, f"ledger {field} differs")
    _require(body.get("pristine") is False, "completed ledger claims to be pristine")
    _require(
        body.get("protected_started_record_count") == 0,
        "ledger contains protected namespace starts",
    )
    _require(
        body.get("protected_completed_record_count") == 0,
        "ledger contains protected outcomes",
    )
    _sha256(body.get("genesis_sha256"), "managed ledger genesis")
    records_by_field: dict[str, list[dict[str, object]]] = {}
    for field in ("started_records", "completed_records", "finalized_records", "attempt_records"):
        records = [
            _plain_dict(item, f"managed ledger {field} record")
            for item in _plain_list(body.get(field), f"managed ledger {field}")
        ]
        _require(len(records) == N_MATCHED_CASES, f"managed ledger {field} cardinality differs")
        for expected_case_index, record in enumerate(records):
            case_index = _strict_int(
                record.get("case_index"),
                f"managed ledger {field} case_index",
                maximum=N_MATCHED_CASES - 1,
            )
            _require(
                case_index == expected_case_index,
                f"managed ledger {field} records are duplicate or misindexed",
            )
        records_by_field[field] = records
    attempt_counts: list[int] = []
    for record in records_by_field["attempt_records"]:
        if "managed_execution_attempt_count" not in record and "attempts" not in record:
            continue
        _require(
            "managed_execution_attempt_count" in record and "attempts" in record,
            "managed attempt record count/list presence differs",
        )
        count = _strict_int(
            record["managed_execution_attempt_count"],
            "managed attempt record count",
            minimum=1,
        )
        attempts = _plain_list(record["attempts"], "managed attempt rows")
        _require(len(attempts) == count, "managed attempt row count differs")
        if "attempt_records_sha256" in record:
            _require(
                record["attempt_records_sha256"] == canonical_sha256(attempts),
                "managed attempt record digest differs from its rows",
            )
        if attempts and all(
            type(item) is dict and "attempt_index" in cast(dict[str, object], item)
            for item in attempts
        ):
            for expected_attempt_index, raw_attempt in enumerate(attempts):
                attempt = _plain_dict(raw_attempt, "managed attempt row")
                attempt_index = _strict_int(
                    attempt.get("attempt_index"),
                    "managed attempt row attempt_index",
                    maximum=count - 1,
                )
                _require(
                    attempt_index == expected_attempt_index,
                    "managed attempt rows are duplicate or misindexed",
                )
        attempt_counts.append(count)
    if attempt_counts:
        _require(
            len(attempt_counts) == N_MATCHED_CASES,
            "managed attempt count is absent from some case records",
        )
        _require(
            sum(attempt_counts) == body.get("managed_execution_attempt_count"),
            "managed aggregate attempt count differs from case records",
        )
    for case_index in range(N_MATCHED_CASES):
        started = records_by_field["started_records"][case_index]
        completed = records_by_field["completed_records"][case_index]
        finalized = records_by_field["finalized_records"][case_index]
        attempt = records_by_field["attempt_records"][case_index]
        if "started_record_sha256" in started and "started_record_sha256" in completed:
            _require(
                completed["started_record_sha256"] == started["started_record_sha256"],
                "completed record does not join its started record",
            )
        if "started_record_sha256" in finalized and "started_record_sha256" in started:
            _require(
                finalized["started_record_sha256"] == started["started_record_sha256"],
                "finalized record does not join its started record",
            )
        if "completed_record_sha256" in finalized and "completed_record_sha256" in completed:
            _require(
                finalized["completed_record_sha256"] == completed["completed_record_sha256"],
                "finalized record does not join its completed record",
            )
        if "managed_execution_attempt_count" in finalized:
            _require(
                finalized["managed_execution_attempt_count"]
                == attempt.get("managed_execution_attempt_count"),
                "finalized record attempt count differs from its attempt binding",
            )
        if "attempt_records_sha256" in finalized:
            _require(
                finalized["attempt_records_sha256"] == attempt.get("attempt_records_sha256"),
                "finalized record attempt digest differs from its attempt binding",
            )
    return dict(snapshot), body


def _validate_readiness_binding(value: object) -> tuple[dict[str, object], dict[str, object]]:
    readiness = _plain_dict(value, "aggregate.readiness_binding")
    _exact_keys(
        readiness,
        {
            "readiness_receipt_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "dependency_locks",
            "scipy_version",
            "execution_governance",
        },
        "aggregate readiness binding",
    )
    for field in (
        "readiness_receipt_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
    ):
        _sha256(readiness.get(field), f"readiness.{field}")
    governance = _plain_dict(readiness.get("execution_governance"), "readiness governance")
    for field, expected in (
        ("protocol_payload_sha256", CALIBRATION_DESIGN_PAYLOAD_SHA256),
        ("seed_snapshot_sha256", SEED_SNAPSHOT_SHA256),
        ("source_archive_sha256", readiness["source_archive_sha256"]),
        ("source_manifest_sha256", readiness["source_manifest_sha256"]),
        ("runtime_identity_sha256", readiness["runtime_identity_sha256"]),
    ):
        _require(governance.get(field) == expected, f"readiness governance {field} differs")
    _sha256(governance.get("genesis_sha256"), "readiness governance genesis")
    _require(
        governance.get("protected_execution_permitted") is False,
        "readiness governance permits protected execution",
    )
    dependencies = _plain_list(readiness.get("dependency_locks"), "readiness dependency locks")
    _require(len(dependencies) == 2, "readiness must bind exactly two dependency locks")
    _require(type(readiness.get("scipy_version")) is str, "readiness SciPy version is invalid")
    return dict(readiness), governance


def _validate_readiness_certification_binding(
    value: object,
    *,
    readiness_receipt_sha256: object,
) -> dict[str, object]:
    binding = _plain_dict(value, "aggregation readiness certification binding")
    _exact_keys(
        binding,
        {
            "readiness_receipt_sha256",
            "certification_ids",
            "certification_specifications_sha256",
            "certification_records_sha256",
            "all_required_certifications_passed",
        },
        "aggregation readiness certification binding",
    )
    _require(
        binding["readiness_receipt_sha256"] == readiness_receipt_sha256,
        "aggregation certification readiness receipt differs",
    )
    for field in (
        "readiness_receipt_sha256",
        "certification_specifications_sha256",
        "certification_records_sha256",
    ):
        _sha256(binding[field], f"aggregation certification {field}")
    identifiers = _plain_list(binding["certification_ids"], "certification identifiers")
    _require(
        bool(identifiers)
        and all(type(item) is str and bool(item) for item in identifiers)
        and len(set(cast(list[str], identifiers))) == len(identifiers),
        "aggregation certification identifiers are invalid",
    )
    _require(
        READINESS_EQUIVALENCE_CERTIFICATION_ID in identifiers,
        "aggregation equivalence certification is absent",
    )
    _require(
        binding["all_required_certifications_passed"] is True,
        "aggregation readiness certifications are incomplete",
    )
    return dict(binding)


def _validate_mandatory_audit_summary(
    value: object,
    *,
    design: HiddenRegimeFactorialCalibrationDesign,
    certification_binding_sha256: str,
) -> dict[str, object]:
    summary = _plain_dict(value, "mandatory audit summary")
    _exact_keys(
        summary,
        {
            "schema",
            "threshold_independent",
            "thresholds_consulted",
            "integrity_status",
            "decision",
            "case_audit_reference_count",
            "case_audit_references",
            "case_audit_references_sha256",
            "selective_immutability_result",
            "selective_immutability_result_sha256",
            "requirement_result_count",
            "requirement_results",
            "requirement_results_sha256",
            "failed_requirement_ids",
            "readiness_certification_binding_sha256",
        },
        "mandatory audit summary",
    )
    _require(
        summary["schema"] == CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "mandatory audit summary schema differs",
    )
    _require(
        summary["threshold_independent"] is True
        and summary["thresholds_consulted"] is False,
        "mandatory audit is not threshold-independent",
    )
    _require(
        summary["integrity_status"] == "passed_before_mechanism_decision",
        "mandatory audit integrity status differs",
    )
    case_references = _plain_list(
        summary["case_audit_references"],
        "mandatory case audit references",
    )
    _require(
        summary["case_audit_reference_count"] == N_MATCHED_CASES
        and len(case_references) == N_MATCHED_CASES,
        "mandatory case audit reference cardinality differs",
    )
    for expected_case_index, raw_reference in enumerate(case_references):
        reference = _plain_dict(raw_reference, "mandatory case audit reference")
        case_index = _strict_int(
            reference.get("case_index"),
            "mandatory case audit reference case_index",
            maximum=N_MATCHED_CASES - 1,
        )
        _require(
            case_index == expected_case_index,
            "mandatory case audit references are duplicate or misindexed",
        )
    _require(
        summary["case_audit_references_sha256"] == canonical_sha256(case_references),
        "mandatory case audit reference digest differs",
    )
    selective = _plain_dict(
        summary["selective_immutability_result"],
        "selective immutability result",
    )
    _require(
        summary["selective_immutability_result_sha256"] == canonical_sha256(selective),
        "selective immutability result digest differs",
    )
    results = [
        _plain_dict(item, "mandatory audit requirement result")
        for item in _plain_list(
            summary["requirement_results"],
            "mandatory audit requirement results",
        )
    ]
    expected_ids = [item.requirement_id for item in design.audits]
    _require(
        summary["requirement_result_count"] == len(expected_ids)
        and [item.get("requirement_id") for item in results] == expected_ids,
        "mandatory audit requirement cardinality or order differs",
    )
    _require(
        summary["requirement_results_sha256"] == canonical_sha256(results),
        "mandatory audit requirement result digest differs",
    )
    for result in results:
        result_body = dict(result)
        result_digest = result_body.pop("requirement_result_sha256", None)
        _require(
            _is_sha256(result_digest) and canonical_sha256(result_body) == result_digest,
            "mandatory audit requirement content digest differs",
        )
        _require(
            result.get("decision") in {"passed_nonstatistical", "invalid_calibration"},
            "mandatory audit requirement decision differs",
        )
        references = _plain_list(
            result.get("required_references"),
            "mandatory audit required references",
        )
        count = _strict_int(
            result.get("required_reference_count"),
            "mandatory audit required reference count",
            minimum=1,
        )
        _require(len(references) == count, "mandatory audit required reference count differs")
        _require(
            result.get("required_references_sha256") == canonical_sha256(references),
            "mandatory audit required reference digest differs",
        )
    failed_ids = [
        cast(str, result["requirement_id"])
        for result in results
        if result["decision"] == "invalid_calibration"
    ]
    _require(summary["failed_requirement_ids"] == failed_ids, "failed audit ids differ")
    expected_decision = "invalid_calibration" if failed_ids else "passed_nonstatistical"
    _require(summary["decision"] == expected_decision, "mandatory audit decision differs")
    _require(
        summary["readiness_certification_binding_sha256"]
        == certification_binding_sha256,
        "mandatory audit readiness certification binding differs",
    )
    return dict(summary)


def _validate_aggregate(
    aggregate: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    body = _validated_digest_payload(
        aggregate,
        digest_field="payload_sha256",
        label="calibration aggregate",
    )
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "claim_accepted",
            "thresholds_frozen",
            "threshold_freeze_receipt",
            "promotion_artifact",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "gate_matrix_sha256",
            "readiness_binding",
            "managed_ledger_snapshot",
            "managed_ledger_snapshot_sha256",
            "managed_ledger_content_address",
            "aggregation_readiness_certification_binding",
            "aggregation_readiness_certification_binding_sha256",
            "aggregation_worker_provenance",
            "aggregation_worker_provenance_sha256",
            "aggregation_zip_provenance_attestation",
            "aggregation_zip_provenance_attestation_sha256",
            "case_count",
            "seed_pair_count",
            "condition_count",
            "case_ledger",
            "case_ledger_sha256",
            "level_summaries",
            "level_summaries_sha256",
            "estimand_summaries",
            "estimand_summaries_sha256",
            "paired_population_support_summaries",
            "paired_population_support_summaries_sha256",
            "mandatory_audit_summary",
            "mandatory_audit_summary_sha256",
            "mandatory_audit_decision",
            "mandatory_gate_results",
            "descriptive_only_results",
            "gate_decision_status",
            "scipy_version_for_student_t_quantiles",
            "float_serialization",
            "comparison_rounding",
            "claim_scope",
        },
        "calibration aggregate",
    )
    _require(body.get("schema") == CALIBRATION_AGGREGATE_SCHEMA, "aggregate schema differs")
    for field, expected in (
        ("development_only", True),
        ("scientific_promotion_allowed", False),
        ("claim_accepted", False),
        ("thresholds_frozen", False),
        ("threshold_freeze_receipt", None),
        ("promotion_artifact", False),
        ("protocol_payload_sha256", CALIBRATION_DESIGN_PAYLOAD_SHA256),
        ("seed_snapshot_sha256", SEED_SNAPSHOT_SHA256),
        ("case_count", N_MATCHED_CASES),
        ("seed_pair_count", N_SEED_PAIRS),
        ("condition_count", len(CANONICAL_CONDITION_ORDER)),
        ("float_serialization", "canonical_python_float_hex_exact_ieee754_binary64"),
        ("comparison_rounding", "none_precomparison_display_rounding_forbidden"),
        (
            "claim_scope",
            "nonpromoting consumed calibration of one finite hidden-regime signaling factorial",
        ),
    ):
        _require(body.get(field) == expected, f"aggregate {field} differs")
    design = build_hidden_regime_factorial_calibration_design()
    expected_gate_digest = canonical_sha256(
        [family.to_payload() for family in design.gate_families]
    )
    _require(body.get("gate_matrix_sha256") == expected_gate_digest, "gate matrix digest differs")
    for value_field, digest_field in (
        ("managed_ledger_snapshot", "managed_ledger_snapshot_sha256"),
        ("aggregation_worker_provenance", "aggregation_worker_provenance_sha256"),
        (
            "aggregation_zip_provenance_attestation",
            "aggregation_zip_provenance_attestation_sha256",
        ),
        ("case_ledger", "case_ledger_sha256"),
        ("level_summaries", "level_summaries_sha256"),
        ("estimand_summaries", "estimand_summaries_sha256"),
        (
            "paired_population_support_summaries",
            "paired_population_support_summaries_sha256",
        ),
    ):
        _require(
            canonical_sha256(body.get(value_field)) == body.get(digest_field),
            f"aggregate {digest_field} differs",
        )
    readiness, governance = _validate_readiness_binding(body.get("readiness_binding"))
    _require(
        body.get("scipy_version_for_student_t_quantiles") == readiness["scipy_version"],
        "aggregate and readiness SciPy versions differ",
    )
    certification = _validate_readiness_certification_binding(
        body.get("aggregation_readiness_certification_binding"),
        readiness_receipt_sha256=readiness["readiness_receipt_sha256"],
    )
    _require(
        body.get("aggregation_readiness_certification_binding_sha256")
        == canonical_sha256(certification),
        "aggregation readiness certification binding digest differs",
    )
    audit_summary = _validate_mandatory_audit_summary(
        body.get("mandatory_audit_summary"),
        design=design,
        certification_binding_sha256=canonical_sha256(certification),
    )
    _require(
        body.get("mandatory_audit_summary_sha256") == canonical_sha256(audit_summary),
        "mandatory audit summary digest differs",
    )
    _require(
        body.get("mandatory_audit_decision") == audit_summary["decision"],
        "aggregate mandatory audit decision differs",
    )
    expected_gate_status = (
        "mandatory_audits_passed_statistical_thresholds_unset"
        if audit_summary["decision"] == "passed_nonstatistical"
        else "invalid_calibration_mandatory_audit_failure"
    )
    _require(body.get("gate_decision_status") == expected_gate_status, "gate status differs")
    ledger, ledger_body = _validate_managed_ledger(body.get("managed_ledger_snapshot"))
    _require(
        body.get("managed_ledger_content_address") == ledger_body["genesis_sha256"],
        "managed ledger content address differs",
    )
    _require(
        ledger_body["genesis_sha256"] == governance["genesis_sha256"],
        "readiness and managed ledger genesis differ",
    )
    case_ledger, all_seeds_consumed = _validate_case_ledger(body.get("case_ledger"))
    _require(all_seeds_consumed, "case ledger does not consume all calibration seeds")
    _require(
        canonical_sha256(case_ledger) == body["case_ledger_sha256"],
        "normalized case ledger digest differs",
    )
    # Preserve only facts derived from the aggregate itself.  No caller-supplied
    # authorization or consumption flag enters the decision receipt.
    derived = {
        "aggregate_payload_sha256": aggregate["payload_sha256"],
        "readiness": readiness,
        "readiness_governance": governance,
        "readiness_certification": certification,
        "mandatory_audit_summary": audit_summary,
        "ledger": ledger,
        "ledger_body": ledger_body,
        "case_ledger": case_ledger,
        "all_calibration_seeds_consumed": all_seeds_consumed,
        "calibration_case_count_consumed": len(case_ledger),
        "calibration_seed_pair_count_consumed": len(
            {cast(int, record["seed_index"]) for record in case_ledger}
        ),
    }
    return body, derived


def _continuous_candidate(
    source: _EndpointSource,
) -> tuple[dict[str, object], list[str]]:
    pooled = source.statistics.pooled
    manifests = source.statistics.manifests
    null = _float_from_hex(source.oriented_null_hex, "endpoint oriented null")
    pooled_bound = pooled.lower_bound
    manifest_bounds = [stratum.lower_bound for _, stratum in manifests]
    inputs = {
        "oriented_null_hex": source.oriented_null_hex,
        "pooled_one_sided_95_percent_lower_confidence_bound_hex": pooled.payload[
            "one_sided_95_percent_lower_confidence_bound_hex"
        ],
        "manifest_one_sided_95_percent_lower_confidence_bounds": [
            {
                "manifest_name": manifest_name,
                "lower_confidence_bound_hex": stratum.payload[
                    "one_sided_95_percent_lower_confidence_bound_hex"
                ],
            }
            for manifest_name, stratum in manifests
        ],
    }
    if pooled_bound is None or any(bound is None for bound in manifest_bounds):
        return (
            {
                "inputs": inputs,
                "conservative_oriented_bound_hex": None,
                "conservative_bound_source": None,
                "favorable_margin_decimal": None,
                "half_margin_decimal": None,
                "rounding_quantum_decimal": "0.0001",
                "margin_quantum_decimal": None,
                "oriented_threshold_decimal": None,
            },
            ["continuous_bound_missing_due_to_incomplete_observations"],
        )
    typed_manifest_bounds = [cast(float, bound) for bound in manifest_bounds]
    conservative = min(pooled_bound, *typed_manifest_bounds)
    if conservative == pooled_bound:
        source_name = "pooled"
    else:
        source_name = next(
            manifest_name
            for (manifest_name, _), bound in zip(manifests, typed_manifest_bounds, strict=True)
            if bound == conservative
        )
    # Derive the floor with rational arithmetic so neither the process-global
    # Decimal context nor an intermediate binary64 subtraction can move a
    # rounding-boundary decision.  (B-N)/2 divided by 0.0001 is (B-N)*5000.
    exact_margin = Fraction.from_float(conservative) - Fraction.from_float(null)
    q_units = math.floor(exact_margin * 5000)
    with localcontext() as context:
        # Every finite binary64 has an exact terminating decimal expansion with
        # fewer than 1,200 significant digits.  This makes all serialized
        # diagnostic decimal inputs exact and independent of caller context.
        context.prec = 1200
        bound_decimal = _decimal_from_float(conservative)
        null_decimal = _decimal_from_float(null)
        margin = bound_decimal - null_decimal
        half_margin = margin / Decimal(2)
        q = Decimal(q_units) * CONTINUOUS_THRESHOLD_QUANTUM
        threshold = null_decimal + q
    reasons: list[str] = []
    if conservative <= null:
        reasons.append("continuous_bound_not_strictly_above_oriented_null")
    if q < CONTINUOUS_THRESHOLD_QUANTUM:
        reasons.append("continuous_half_margin_below_rounding_quantum")
    return (
        {
            "inputs": inputs,
            "conservative_oriented_bound_hex": conservative.hex(),
            "conservative_bound_source": source_name,
            "favorable_margin_decimal": _decimal_text(margin),
            "half_margin_decimal": _decimal_text(half_margin),
            "rounding_quantum_decimal": "0.0001",
            "margin_quantum_decimal": _quantized_decimal_text(q),
            "oriented_threshold_decimal": _decimal_text(threshold),
        },
        reasons,
    )


def _win_candidate(
    *,
    label: str,
    expected_n: int,
    stratum: _ValidatedStratum,
) -> tuple[dict[str, object], list[str]]:
    null_wins = expected_n // 2
    q = math.floor((stratum.wins - null_wins) / 2)
    threshold = null_wins + q
    reasons: list[str] = []
    if stratum.structural_missing_n != 0:
        reasons.append(f"{label}_structural_missing_n_not_zero")
    if q < 1:
        reasons.append(f"{label}_win_half_margin_below_one")
    return (
        {
            "population": label,
            "required_n": expected_n,
            "observed_n": stratum.observed_n,
            "conditional_unobserved_n": stratum.conditional_unobserved_n,
            "structural_missing_n": stratum.structural_missing_n,
            "wins": stratum.wins,
            "ties": stratum.ties,
            "losses": stratum.losses,
            "ties_count_as_wins": False,
            "null_win_count": null_wins,
            "half_excess_win_margin": q,
            "win_threshold": threshold if q >= 1 else None,
        },
        reasons,
    )


def _endpoint_result(source: _EndpointSource) -> dict[str, object]:
    identity = _reference_identity(source.reference, "endpoint reference")
    endpoint_id = canonical_sha256(
        {"gate_family_id": source.gate_family_id, "reference": identity}
    )
    continuous, reasons = _continuous_candidate(source)
    win_inputs: list[dict[str, object]] = []
    pooled_win, pooled_reasons = _win_candidate(
        label="pooled",
        expected_n=EXPECTED_POOLED_N,
        stratum=source.statistics.pooled,
    )
    win_inputs.append(pooled_win)
    reasons.extend(pooled_reasons)
    for manifest_name, stratum in source.statistics.manifests:
        manifest_win, manifest_reasons = _win_candidate(
            label=manifest_name,
            expected_n=EXPECTED_MANIFEST_N,
            stratum=stratum,
        )
        win_inputs.append(manifest_win)
        reasons.extend(manifest_reasons)
    decision = (
        "threshold_candidate_sufficient"
        if not reasons
        else THRESHOLD_FREEZE_DECISION_REJECTION
    )
    return {
        "endpoint_id": endpoint_id,
        "gate_family_id": source.gate_family_id,
        "reference": dict(source.reference),
        "orientation": source.orientation,
        "oriented_space": True,
        "decision_status": decision,
        "rejection_reasons": reasons,
        "continuous_threshold_candidate": continuous,
        "win_threshold_candidates": win_inputs,
        "missingness_threshold": 0,
    }


def _frozen_threshold(endpoint: Mapping[str, object]) -> dict[str, object]:
    continuous = _plain_dict(
        endpoint["continuous_threshold_candidate"],
        "continuous threshold candidate",
    )
    _require(
        continuous.get("oriented_threshold_decimal") is not None,
        "successful endpoint has no continuous threshold",
    )
    wins = [
        _plain_dict(item, "win threshold candidate")
        for item in _plain_list(endpoint["win_threshold_candidates"], "win candidates")
    ]
    _require(all(item.get("win_threshold") is not None for item in wins), "win threshold is absent")
    return {
        "endpoint_id": endpoint["endpoint_id"],
        "gate_family_id": endpoint["gate_family_id"],
        "reference": endpoint["reference"],
        "orientation": endpoint["orientation"],
        "threshold_space": "oriented_higher_is_favorable",
        "oriented_null_hex": _plain_dict(continuous["inputs"], "continuous inputs")[
            "oriented_null_hex"
        ],
        "conservative_oriented_bound_hex": continuous["conservative_oriented_bound_hex"],
        "continuous_margin_quantum_decimal": continuous["margin_quantum_decimal"],
        "oriented_continuous_threshold_decimal": continuous["oriented_threshold_decimal"],
        "pooled_win_threshold": wins[0]["win_threshold"],
        "manifest_win_thresholds": [
            {
                "manifest_name": item["population"],
                "win_threshold": item["win_threshold"],
            }
            for item in wins[1:]
        ],
        "missingness_threshold": 0,
        "ties_count_as_wins": False,
    }


def _audit_gate_result(
    aggregate_result: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Normalize the aggregate's threshold-independent mandatory audit decision."""

    # Aggregate v4 binds a nonempty audit reference set and decides this family
    # independently of statistical threshold calibration.  Exact audit-summary
    # validation is performed by the aggregate validator; this consumer still
    # refuses an absent/indeterminate decision.
    gate_family_id = aggregate_result.get("gate_family_id")
    _require(
        gate_family_id == "mandatory_trace_and_lifecycle_audits",
        "mandatory audit gate identifier differs",
    )
    decision = aggregate_result.get("decision")
    _require(
        decision in {"passed_nonstatistical", "invalid_calibration"},
        "mandatory audit decision is indeterminate",
    )
    references = _plain_list(aggregate_result.get("references"), "mandatory audit references")
    _require(bool(references), "mandatory audit references are empty")
    reasons = (
        []
        if decision == "passed_nonstatistical"
        else ["mandatory_trace_or_lifecycle_audit_failed"]
    )
    return (
        {
            "gate_family_id": gate_family_id,
            "mandatory": True,
            "gate_kind": "threshold_independent_audit",
            "decision_status": (
                "passed" if not reasons else THRESHOLD_FREEZE_DECISION_REJECTION
            ),
            "reference_count": len(references),
            "references_sha256": canonical_sha256(references),
            "rejection_reasons": reasons,
        },
        reasons,
    )


def _descriptive_results(
    aggregate: Mapping[str, object],
    design: HiddenRegimeFactorialCalibrationDesign,
) -> list[dict[str, object]]:
    raw = _plain_list(aggregate["descriptive_only_results"], "descriptive_only_results")
    families = tuple(family for family in design.gate_families if not family.mandatory)
    _require(len(raw) == len(families), "descriptive gate family cardinality differs")
    results: list[dict[str, object]] = []
    for index, family in enumerate(families):
        item = _plain_dict(raw[index], f"descriptive result {index}")
        _exact_keys(
            item,
            {"gate_family_id", "mandatory", "threshold_status", "decision", "references"},
            f"descriptive result {index}",
        )
        _require(item.get("gate_family_id") == family.gate_family_id, "descriptive order differs")
        _require(item.get("mandatory") is False, "descriptive family is marked mandatory")
        _require(
            item.get("threshold_status") == _STATISTICAL_THRESHOLD_STATUS,
            "descriptive threshold status differs",
        )
        _require(
            item.get("decision") == _STATISTICAL_GATE_DECISION,
            "descriptive gate was decided",
        )
        references = _plain_list(item.get("references"), f"{family.gate_family_id}.references")
        results.append(
            {
                "gate_family_id": family.gate_family_id,
                "decision_status": "descriptive_only_not_thresholded",
                "reference_count": len(references),
                "references_sha256": canonical_sha256(references),
            }
        )
    return results


def _rounding_worked_examples() -> list[dict[str, object]]:
    return [
        {
            "example_id": "continuous_oriented_twofold_margin",
            "oriented_null_decimal": "0",
            "conservative_oriented_bound_decimal": "0.00035",
            "half_margin_decimal": "0.000175",
            "floor_quantum_decimal": "0.0001",
            "frozen_oriented_threshold_decimal": "0.0001",
        },
        {
            "example_id": "pooled_directional_wins",
            "n": 30,
            "wins": 19,
            "ties": 1,
            "null_win_count": 15,
            "half_excess_win_margin": 2,
            "frozen_win_threshold": 17,
            "ties_count_as_wins": False,
        },
        {
            "example_id": "manifest_directional_wins",
            "n": 10,
            "wins": 8,
            "ties": 1,
            "null_win_count": 5,
            "half_excess_win_margin": 1,
            "frozen_win_threshold": 6,
            "ties_count_as_wins": False,
        },
    ]


def materialize_hidden_regime_factorial_threshold_freeze_receipt(
    calibration_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Build one canonical success or valid-calibration-rejection receipt.

    The input must already have passed exact shard/ledger recomputation.  This
    function independently rechecks every provenance binding it consumes and
    every mandatory endpoint reference.  It has no authorization flags and no
    protected-seed derivation path.
    """

    aggregate, derived = _validate_aggregate(calibration_aggregate)
    design = build_hidden_regime_factorial_calibration_design()
    sources, raw_audit_result = _validate_statistical_gate_references(aggregate, design)
    endpoint_results = [_endpoint_result(source) for source in sources]
    _require(
        len(endpoint_results) == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        "threshold engine did not materialize exactly 35 endpoints",
    )
    audit_result, audit_reasons = _audit_gate_result(raw_audit_result)
    rejection_reasons = [
        {
            "endpoint_id": endpoint["endpoint_id"],
            "reasons": endpoint["rejection_reasons"],
        }
        for endpoint in endpoint_results
        if endpoint["rejection_reasons"]
    ]
    if audit_reasons:
        rejection_reasons.append(
            {
                "gate_family_id": "mandatory_trace_and_lifecycle_audits",
                "reasons": audit_reasons,
            }
        )
    success = not rejection_reasons
    decision_status = (
        THRESHOLD_FREEZE_DECISION_FROZEN
        if success
        else THRESHOLD_FREEZE_DECISION_REJECTION
    )
    frozen_thresholds = (
        [_frozen_threshold(endpoint) for endpoint in endpoint_results] if success else []
    )
    family_results: list[dict[str, object]] = []
    for family in (item for item in design.gate_families if item.mandatory and item.metric_ids):
        family_endpoints = [
            endpoint
            for endpoint in endpoint_results
            if endpoint["gate_family_id"] == family.gate_family_id
        ]
        family_failed = [endpoint for endpoint in family_endpoints if endpoint["rejection_reasons"]]
        family_results.append(
            {
                "gate_family_id": family.gate_family_id,
                "mandatory": True,
                "gate_kind": "statistical_threshold",
                "decision_status": (
                    "threshold_candidates_sufficient"
                    if not family_failed
                    else THRESHOLD_FREEZE_DECISION_REJECTION
                ),
                "endpoint_count": len(family_endpoints),
                "endpoint_results": family_endpoints,
                "rejection_count": len(family_failed),
            }
        )
    family_results.append(audit_result)

    readiness = cast(dict[str, object], derived["readiness"])
    governance = cast(dict[str, object], derived["readiness_governance"])
    ledger_body = cast(dict[str, object], derived["ledger_body"])
    body = {
        "receipt_schema": THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        "decision_status": decision_status,
        "development_only": True,
        "claim_accepted": False,
        "thresholds_frozen": success,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "readiness_receipt_sha256": readiness["readiness_receipt_sha256"],
        "gate_matrix_sha256": aggregate["gate_matrix_sha256"],
        "calibration_outcomes_payload_sha256": derived["aggregate_payload_sha256"],
        "source_closure_sha256": readiness["source_manifest_sha256"],
        "source_archive_sha256": readiness["source_archive_sha256"],
        "environment_identity_sha256": readiness["runtime_identity_sha256"],
        "managed_ledger_snapshot_sha256": aggregate["managed_ledger_snapshot_sha256"],
        "managed_ledger_content_address": aggregate["managed_ledger_content_address"],
        "execution_governance_genesis_sha256": governance["genesis_sha256"],
        "aggregation_readiness_certification_binding_sha256": aggregate[
            "aggregation_readiness_certification_binding_sha256"
        ],
        "mandatory_audit_summary_sha256": aggregate["mandatory_audit_summary_sha256"],
        "mandatory_audit_decision": aggregate["mandatory_audit_decision"],
        "case_ledger_sha256": aggregate["case_ledger_sha256"],
        "mandatory_statistical_endpoint_count": MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        "mandatory_statistical_endpoint_identities_sha256": (
            MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
        ),
        "mandatory_statistical_endpoint_ids_sha256": (
            MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
        ),
        "frozen_thresholds": frozen_thresholds,
        "mandatory_gate_results": family_results,
        "descriptive_only_results": _descriptive_results(aggregate, design),
        "rejection_reasons": rejection_reasons,
        "rounding_worked_examples": _rounding_worked_examples(),
        "all_calibration_seeds_consumed": derived["all_calibration_seeds_consumed"],
        "calibration_case_count_consumed": derived["calibration_case_count_consumed"],
        "calibration_seed_pair_count_consumed": derived[
            "calibration_seed_pair_count_consumed"
        ],
        "protected_namespace_derived": cast(
            int, ledger_body["protected_started_record_count"]
        )
        > 0,
        "protected_outcomes_observed": cast(
            int, ledger_body["protected_completed_record_count"]
        )
        > 0,
        "scientific_promotion_allowed": False,
        "amendments_allowed": False,
    }
    return _payload_with_digest(body)


def validate_hidden_regime_factorial_threshold_freeze_receipt(
    payload: Mapping[str, object],
    *,
    calibration_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Recompute and byte-compare one exact threshold-freeze decision receipt."""

    expected = materialize_hidden_regime_factorial_threshold_freeze_receipt(
        calibration_aggregate
    )
    try:
        actual_bytes = canonical_json_bytes(dict(payload))
    except (TypeError, ValueError) as exc:
        raise ThresholdFreezeError("threshold-freeze receipt is not canonical JSON data") from exc
    if actual_bytes != canonical_json_bytes(expected):
        _fail("threshold-freeze receipt differs from exact recomputation")
    body = _validated_digest_payload(
        payload,
        digest_field="receipt_payload_sha256",
        label="threshold-freeze receipt",
    )
    _require(
        body.get("receipt_schema") == THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        "threshold-freeze receipt schema differs",
    )
    decision = body.get("decision_status")
    _require(
        decision in {THRESHOLD_FREEZE_DECISION_FROZEN, THRESHOLD_FREEZE_DECISION_REJECTION},
        "threshold-freeze decision status differs",
    )
    if decision == THRESHOLD_FREEZE_DECISION_FROZEN:
        _require(
            body.get("thresholds_frozen") is True,
            "success receipt does not freeze thresholds",
        )
        _require(
            len(_plain_list(body.get("frozen_thresholds"), "frozen thresholds"))
            == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
            "success receipt does not contain 35 frozen thresholds",
        )
        _require(body.get("rejection_reasons") == [], "success receipt contains rejection reasons")
    else:
        _require(body.get("thresholds_frozen") is False, "rejection receipt freezes thresholds")
        _require(
            body.get("frozen_thresholds") == [],
            "rejection receipt partially freezes thresholds",
        )
        _require(bool(body.get("rejection_reasons")), "rejection receipt has no rejection reason")
    return expected


def canonical_threshold_freeze_receipt_bytes(
    payload: Mapping[str, object],
    *,
    calibration_aggregate: Mapping[str, object],
) -> bytes:
    """Return canonical bytes only after exact receipt recomputation succeeds."""

    validated = validate_hidden_regime_factorial_threshold_freeze_receipt(
        payload,
        calibration_aggregate=calibration_aggregate,
    )
    return canonical_json_bytes(validated)


# Compact public aliases.  Currently unconsumed in-repo: the calibration
# runner (``hidden_regime_factorial_calibration``) imports the long names.
freeze_hidden_regime_factorial_thresholds = (
    materialize_hidden_regime_factorial_threshold_freeze_receipt
)
validate_threshold_freeze_receipt = validate_hidden_regime_factorial_threshold_freeze_receipt


__all__ = [
    "CALIBRATION_AGGREGATE_SCHEMA",
    "CONTINUOUS_THRESHOLD_QUANTUM",
    "MANDATORY_STATISTICAL_ENDPOINT_COUNT",
    "MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256",
    "MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256",
    "THRESHOLD_FREEZE_DECISION_FROZEN",
    "THRESHOLD_FREEZE_DECISION_REJECTION",
    "ThresholdFreezeError",
    "canonical_threshold_freeze_receipt_bytes",
    "freeze_hidden_regime_factorial_thresholds",
    "materialize_hidden_regime_factorial_threshold_freeze_receipt",
    "validate_hidden_regime_factorial_threshold_freeze_receipt",
    "validate_threshold_freeze_receipt",
]
