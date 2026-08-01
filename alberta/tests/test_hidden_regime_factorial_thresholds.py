from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterator, Mapping
from decimal import ROUND_UP, Decimal, localcontext
from typing import cast

import pytest

from alberta_framework.evaluation import hidden_regime_factorial_thresholds as thresholds
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    N_MATCHED_CASES,
    N_SEED_PAIRS,
    SEED_SNAPSHOT_SHA256,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
    canonical_sha256,
)

pytestmark = pytest.mark.unit


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _coherent_inference(bound: float, observed_n: int) -> dict[str, str]:
    scale = max(abs(bound), 2**-40)
    sample_sd = float(scale * math.sqrt(observed_n) / 8)
    standard_error = float(sample_sd / math.sqrt(observed_n))
    critical = float("1.7")
    product = critical * standard_error
    initial_mean = float(bound + product)
    candidates = [initial_mean]
    higher = initial_mean
    lower = initial_mean
    for _ in range(64):
        higher = math.nextafter(higher, math.inf)
        lower = math.nextafter(lower, -math.inf)
        candidates.extend((higher, lower))
    mean = next(candidate for candidate in candidates if float(candidate - product) == bound)
    return {
        "mean_hex": mean.hex(),
        "sample_standard_deviation_hex": sample_sd.hex(),
        "standard_error_hex": standard_error.hex(),
        "student_t_0_95_critical_hex": critical.hex(),
        "one_sided_95_percent_lower_confidence_bound_hex": bound.hex(),
    }


def _stratum(
    *,
    eligible_n: int,
    seed_indices: list[int],
    bound: float,
    missing_kind: str | None = None,
) -> dict[str, object]:
    missing = [] if missing_kind is None else [seed_indices[0]]
    observed_n = eligible_n - len(missing)
    return {
        "eligible_n": eligible_n,
        "conditional_unobserved_n": len(missing) if missing_kind == "conditional" else 0,
        "conditional_unobserved_seed_indices": (
            missing if missing_kind == "conditional" else []
        ),
        "structural_missing_n": len(missing) if missing_kind == "structural" else 0,
        "structural_missing_seed_indices": missing if missing_kind == "structural" else [],
        "observed_n": observed_n,
        **_coherent_inference(bound, observed_n),
        "wins": observed_n,
        "ties": 0,
        "losses": 0,
    }


def _statistics(
    null: float,
    *,
    margin: float = 0.4,
    missing_kind: str | None = None,
) -> dict[str, object]:
    bound = null + margin
    manifests: list[dict[str, object]] = []
    pooled_missing: list[int] = []
    for manifest_index, manifest_name in enumerate(CALIBRATION_MANIFEST_ORDER):
        seeds = [seed for seed in range(N_SEED_PAIRS) if seed % 3 == manifest_index]
        manifest_missing = missing_kind if manifest_index == 0 else None
        row = _stratum(
            eligible_n=10,
            seed_indices=seeds,
            bound=bound,
            missing_kind=manifest_missing,
        )
        if manifest_missing is not None:
            pooled_missing.append(seeds[0])
        manifests.append({"manifest_name": manifest_name, **row})
    pooled = _stratum(
        eligible_n=30,
        seed_indices=list(range(30)),
        bound=bound,
        missing_kind=None,
    )
    if missing_kind is not None:
        pooled["observed_n"] = 29
        pooled["wins"] = 29
        pooled.update(_coherent_inference(bound, 29))
        missing_count_field = (
            "conditional_unobserved_n"
            if missing_kind == "conditional"
            else "structural_missing_n"
        )
        pooled[missing_count_field] = 1
        pooled[
            f"{missing_kind}_unobserved_seed_indices"
            if missing_kind == "conditional"
            else "structural_missing_seed_indices"
        ] = pooled_missing
    manifest_means = [
        float.fromhex(cast(str, manifest["mean_hex"])) for manifest in manifests
    ]
    worst = {
        "minimum_manifest_mean_hex": min(manifest_means).hex(),
        "minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex": bound.hex(),
        "minimum_manifest_wins": 9 if missing_kind is not None else 10,
        "maximum_manifest_structural_missing_n": 1 if missing_kind == "structural" else 0,
        "maximum_manifest_conditional_unobserved_n": (
            1 if missing_kind == "conditional" else 0
        ),
    }
    return {"pooled": pooled, "by_manifest": manifests, "worst_manifest": worst}


def _expected_references(family: object, support_ids: set[str]) -> list[dict[str, object]]:
    typed = cast(object, family)
    estimand_ids = cast(tuple[str, ...], getattr(typed, "estimand_ids"))
    metric_ids = cast(tuple[str, ...], getattr(typed, "metric_ids"))
    conditions = cast(tuple[str, ...], getattr(typed, "conditions"))
    if estimand_ids:
        return [
            {
                "kind": (
                    "paired_population_support_level"
                    if metric_id in support_ids
                    else "paired_contrast"
                ),
                "metric_id": metric_id,
                "estimand_id": estimand_id,
            }
            for estimand_id in estimand_ids
            for metric_id in metric_ids
        ]
    return [
        {"kind": "absolute_level", "metric_id": metric_id, "condition": condition}
        for condition in conditions
        for metric_id in metric_ids
    ]


def _fake_audit_summary(
    *,
    certification_binding_sha256: str,
) -> dict[str, object]:
    design = build_hidden_regime_factorial_calibration_design()
    results: list[dict[str, object]] = []
    for requirement in design.audits:
        reference = {
            "kind": "synthetic_exact_reference",
            "requirement_id": requirement.requirement_id,
        }
        result_body = {
            **requirement.to_payload(),
            "evaluation_mode": "synthetic_gate_matrix_unit",
            "threshold_independent": True,
            "thresholds_consulted": False,
            "decision": "passed_nonstatistical",
            "required_reference_count": 1,
            "required_references": [reference],
            "required_references_sha256": canonical_sha256([reference]),
            "descriptive_reference_count": 0,
            "descriptive_references": [],
            "descriptive_references_sha256": canonical_sha256([]),
            "failed_case_indices": [],
        }
        results.append(
            {**result_body, "requirement_result_sha256": canonical_sha256(result_body)}
        )
    case_references = [{"case_index": index} for index in range(N_MATCHED_CASES)]
    selective = {
        "subpredicate_id": "selective_immutability_where_applicable",
        "decision": "passed_nonstatistical",
    }
    return {
        "schema": thresholds.CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "threshold_independent": True,
        "thresholds_consulted": False,
        "integrity_status": "passed_before_mechanism_decision",
        "decision": "passed_nonstatistical",
        "case_audit_reference_count": N_MATCHED_CASES,
        "case_audit_references": case_references,
        "case_audit_references_sha256": canonical_sha256(case_references),
        "selective_immutability_result": selective,
        "selective_immutability_result_sha256": canonical_sha256(selective),
        "requirement_result_count": len(results),
        "requirement_results": results,
        "requirement_results_sha256": canonical_sha256(results),
        "failed_requirement_ids": [],
        "readiness_certification_binding_sha256": certification_binding_sha256,
    }


def _fake_aggregate() -> dict[str, object]:
    design = build_hidden_regime_factorial_calibration_design()
    contracts = {metric.metric_id: metric for metric in design.metrics}
    support_ids = {metric.metric_id for metric in design.paired_population_support_metrics}
    levels: list[dict[str, object]] = []
    estimands_by_id: dict[str, dict[str, object]] = {}
    supports: list[dict[str, object]] = []
    mandatory_results: list[dict[str, object]] = []

    for family in (item for item in design.gate_families if item.mandatory):
        if not family.metric_ids:
            mandatory_results.append(
                {
                    "gate_family_id": family.gate_family_id,
                    "mandatory": True,
                    "threshold_status": "not_applicable_nonstatistical",
                    "decision": "passed_nonstatistical",
                    "references": [],
                }
            )
            continue
        references: list[dict[str, object]] = []
        for identity in _expected_references(family, support_ids):
            metric_id = cast(str, identity["metric_id"])
            if identity["kind"] == "absolute_level":
                contract = contracts[metric_id]
                null = float(cast(str, contract.null_value_decimal))
                if contract.orientation == "lower":
                    null = -null
                statistics = _statistics(null)
                levels.append(
                    {
                        "condition": identity["condition"],
                        "metric_id": metric_id,
                        "orientation": contract.orientation,
                        "orientation_applied": True,
                        "oriented_null_hex": null.hex(),
                        "statistics": statistics,
                    }
                )
            elif identity["kind"] == "paired_contrast":
                contract = contracts[metric_id]
                null = 0.0
                statistics = _statistics(null)
                estimand_id = cast(str, identity["estimand_id"])
                estimand = estimands_by_id.setdefault(
                    estimand_id,
                    {
                        "estimand_id": estimand_id,
                        "oriented_null_hex": null.hex(),
                        "metrics": [],
                    },
                )
                cast(list[object], estimand["metrics"]).append(
                    {
                        "metric_id": metric_id,
                        "orientation": contract.orientation,
                        "statistics": statistics,
                    }
                )
            else:
                null = 0.0
                statistics = _statistics(null)
                supports.append(
                    {
                        "estimand_id": identity["estimand_id"],
                        "metric_id": metric_id,
                        "orientation": "higher",
                        "oriented_null_hex": null.hex(),
                        "statistics": statistics,
                    }
                )
            references.append(
                {**identity, "statistics_sha256": canonical_sha256(statistics)}
            )
        mandatory_results.append(
            {
                "gate_family_id": family.gate_family_id,
                "mandatory": True,
                "threshold_status": "unset_pending_consumed_calibration_outcomes",
                "decision": "not_evaluated_no_thresholds",
                "references": references,
            }
        )

    mandatory_levels = {
        (cast(str, item["condition"]), cast(str, item["metric_id"])): item
        for item in levels
    }
    levels = []
    for condition in CANONICAL_CONDITION_ORDER:
        for contract in design.metrics:
            key = (condition, contract.metric_id)
            existing_level = mandatory_levels.get(key)
            if existing_level is not None:
                levels.append(existing_level)
                continue
            null = (
                None
                if contract.null_value_decimal is None
                else float(contract.null_value_decimal)
                * (1.0 if contract.orientation == "higher" else -1.0)
            )
            levels.append(
                {
                    "condition": condition,
                    "metric_id": contract.metric_id,
                    "orientation": contract.orientation,
                    "orientation_applied": True,
                    "oriented_null_hex": None if null is None else null.hex(),
                    "statistics": _statistics(0.0 if null is None else null),
                }
            )

    estimand_summaries: list[dict[str, object]] = []
    for estimand_spec in design.factorial_estimands + design.control_estimands:
        estimand = estimands_by_id.get(
            estimand_spec.estimand_id,
            {
                "estimand_id": estimand_spec.estimand_id,
                "oriented_null_hex": (0.0).hex(),
                "metrics": [],
            },
        )
        existing_metrics = {
            cast(str, item["metric_id"]): item
            for item in cast(list[dict[str, object]], estimand["metrics"])
        }
        estimand["metrics"] = [
            existing_metrics.get(
                metric_id,
                {
                    "metric_id": metric_id,
                    "orientation": contracts[metric_id].orientation,
                    "statistics": _statistics(0.0),
                },
            )
            for metric_id in estimand_spec.metrics
        ]
        estimand_summaries.append(estimand)

    descriptive = [
        {
            "gate_family_id": family.gate_family_id,
            "mandatory": False,
            "threshold_status": "unset_pending_consumed_calibration_outcomes",
            "decision": "not_evaluated_no_thresholds",
            "references": [],
        }
        for family in design.gate_families
        if not family.mandatory
    ]
    genesis = _digest("genesis")
    governance = {
        "genesis_sha256": genesis,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "source_archive_sha256": _digest("archive"),
        "source_manifest_sha256": _digest("manifest"),
        "runtime_identity_sha256": _digest("runtime"),
        "protected_execution_permitted": False,
    }
    readiness = {
        "readiness_receipt_sha256": _digest("readiness"),
        "source_archive_sha256": governance["source_archive_sha256"],
        "source_manifest_sha256": governance["source_manifest_sha256"],
        "runtime_identity_sha256": governance["runtime_identity_sha256"],
        "dependency_locks": [
            {"locator": "pyproject.toml", "sha256": _digest("pyproject")},
            {"locator": "uv.lock", "sha256": _digest("uv-lock")},
        ],
        "scipy_version": "synthetic-test",
        "execution_governance": governance,
    }
    certification_binding = {
        "readiness_receipt_sha256": readiness["readiness_receipt_sha256"],
        "certification_ids": [thresholds.READINESS_EQUIVALENCE_CERTIFICATION_ID],
        "certification_specifications_sha256": _digest("certification-specifications"),
        "certification_records_sha256": _digest("certification-records"),
        "all_required_certifications_passed": True,
    }
    audit_summary = _fake_audit_summary(
        certification_binding_sha256=canonical_sha256(certification_binding)
    )
    audit_gate = mandatory_results[-1]
    audit_gate["references"] = [
        {
            "kind": "threshold_independent_audit_requirement",
            "requirement_id": result["requirement_id"],
            "decision": result["decision"],
            "requirement_result_sha256": result["requirement_result_sha256"],
            "required_references_sha256": result["required_references_sha256"],
        }
        for result in cast(list[dict[str, object]], audit_summary["requirement_results"])
    ]
    indices = list(range(N_MATCHED_CASES))
    ledger_body = {
        "schema": "synthetic-ledger",
        "genesis_sha256": genesis,
        "expected_case_count": N_MATCHED_CASES,
        "started_case_indices": indices,
        "completed_case_indices": indices,
        "finalized_case_indices": indices,
        "learner_interrupted_case_indices": [],
        "post_audit_unfinalized_case_indices": [],
        "started_record_count": N_MATCHED_CASES,
        "completed_record_count": N_MATCHED_CASES,
        "finalized_record_count": N_MATCHED_CASES,
        "managed_execution_attempt_count": N_MATCHED_CASES,
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": False,
        "started_records": [{"case_index": index} for index in indices],
        "completed_records": [{"case_index": index} for index in indices],
        "finalized_records": [{"case_index": index} for index in indices],
        "attempt_records": [
            {
                "case_index": index,
                "managed_execution_attempt_count": 1,
                "attempt_records_sha256": canonical_sha256([{"attempt_index": 0}]),
                "attempts": [{"attempt_index": 0}],
            }
            for index in indices
        ],
    }
    ledger = {**ledger_body, "inventory_sha256": canonical_sha256(ledger_body)}
    case_ledger = [
        {
            "case_index": case_index,
            "seed_index": case_index // len(CANONICAL_CONDITION_ORDER),
            "condition": CANONICAL_CONDITION_ORDER[case_index % 8],
            "manifest_name": CALIBRATION_MANIFEST_ORDER[(case_index // 8) % 3],
            "case_shard_payload_sha256": _digest(f"shard-{case_index}"),
            "case_request_binding_sha256": _digest(f"request-{case_index}"),
            "summary_sha256": _digest(f"summary-{case_index}"),
            "resource_sha256": _digest(f"resource-{case_index}"),
            "worker_provenance_sha256": _digest(f"worker-{case_index}"),
            "execution_record_binding": {"case_index": case_index},
            "finalized_record_sha256": _digest(f"finalized-{case_index}"),
            "shard_canonical_sha256": _digest(f"canonical-{case_index}"),
            "primitive_trace_sha256": _digest(f"trace-{case_index}"),
        }
        for case_index in indices
    ]
    worker_provenance = {"schema": "synthetic-worker", "sha256": _digest("worker")}
    zip_attestation = {"schema": "synthetic-zip", "sha256": _digest("zip")}
    body: dict[str, object] = {
        "schema": thresholds.CALIBRATION_AGGREGATE_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "thresholds_frozen": False,
        "threshold_freeze_receipt": None,
        "promotion_artifact": False,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "gate_matrix_sha256": canonical_sha256(
            [family.to_payload() for family in design.gate_families]
        ),
        "readiness_binding": readiness,
        "managed_ledger_snapshot": ledger,
        "managed_ledger_snapshot_sha256": canonical_sha256(ledger),
        "managed_ledger_content_address": genesis,
        "aggregation_readiness_certification_binding": certification_binding,
        "aggregation_readiness_certification_binding_sha256": canonical_sha256(
            certification_binding
        ),
        "aggregation_worker_provenance": worker_provenance,
        "aggregation_worker_provenance_sha256": canonical_sha256(worker_provenance),
        "aggregation_zip_provenance_attestation": zip_attestation,
        "aggregation_zip_provenance_attestation_sha256": canonical_sha256(zip_attestation),
        "case_count": N_MATCHED_CASES,
        "seed_pair_count": N_SEED_PAIRS,
        "condition_count": len(CANONICAL_CONDITION_ORDER),
        "case_ledger": case_ledger,
        "case_ledger_sha256": canonical_sha256(case_ledger),
        "level_summaries": levels,
        "level_summaries_sha256": canonical_sha256(levels),
        "estimand_summaries": estimand_summaries,
        "estimand_summaries_sha256": canonical_sha256(estimand_summaries),
        "paired_population_support_summaries": supports,
        "paired_population_support_summaries_sha256": canonical_sha256(supports),
        "mandatory_audit_summary": audit_summary,
        "mandatory_audit_summary_sha256": canonical_sha256(audit_summary),
        "mandatory_audit_decision": "passed_nonstatistical",
        "mandatory_gate_results": mandatory_results,
        "descriptive_only_results": descriptive,
        "gate_decision_status": "mandatory_audits_passed_statistical_thresholds_unset",
        "scipy_version_for_student_t_quantiles": "synthetic-test",
        "float_serialization": "canonical_python_float_hex_exact_ieee754_binary64",
        "comparison_rounding": "none_precomparison_display_rounding_forbidden",
        "claim_scope": (
            "nonpromoting consumed calibration of one finite hidden-regime signaling factorial"
        ),
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _iter_statistical_sources(aggregate: Mapping[str, object]) -> Iterator[dict[str, object]]:
    for raw in cast(list[object], aggregate["level_summaries"]):
        yield cast(dict[str, object], raw)
    for raw_estimand in cast(list[object], aggregate["estimand_summaries"]):
        estimand = cast(dict[str, object], raw_estimand)
        for raw_metric in cast(list[object], estimand["metrics"]):
            yield cast(dict[str, object], raw_metric)
    for raw in cast(list[object], aggregate["paired_population_support_summaries"]):
        yield cast(dict[str, object], raw)


def _source_for_reference(
    aggregate: Mapping[str, object], reference: Mapping[str, object]
) -> dict[str, object]:
    kind = reference["kind"]
    if kind == "absolute_level":
        return next(
            cast(dict[str, object], raw)
            for raw in cast(list[object], aggregate["level_summaries"])
            if cast(dict[str, object], raw)["condition"] == reference["condition"]
            and cast(dict[str, object], raw)["metric_id"] == reference["metric_id"]
        )
    if kind == "paired_population_support_level":
        return next(
            cast(dict[str, object], raw)
            for raw in cast(list[object], aggregate["paired_population_support_summaries"])
            if cast(dict[str, object], raw)["estimand_id"] == reference["estimand_id"]
            and cast(dict[str, object], raw)["metric_id"] == reference["metric_id"]
        )
    estimand = next(
        cast(dict[str, object], raw)
        for raw in cast(list[object], aggregate["estimand_summaries"])
        if cast(dict[str, object], raw)["estimand_id"] == reference["estimand_id"]
    )
    return next(
        cast(dict[str, object], raw)
        for raw in cast(list[object], estimand["metrics"])
        if cast(dict[str, object], raw)["metric_id"] == reference["metric_id"]
    )


def _refresh(aggregate: dict[str, object]) -> None:
    for raw_gate in cast(list[object], aggregate["mandatory_gate_results"]):
        gate = cast(dict[str, object], raw_gate)
        if gate["gate_family_id"] == "mandatory_trace_and_lifecycle_audits":
            continue
        for raw_reference in cast(list[object], gate["references"]):
            reference = cast(dict[str, object], raw_reference)
            source = _source_for_reference(aggregate, reference)
            reference["statistics_sha256"] = canonical_sha256(source["statistics"])
    for field in (
        "managed_ledger_snapshot",
        "aggregation_readiness_certification_binding",
        "aggregation_worker_provenance",
        "aggregation_zip_provenance_attestation",
        "case_ledger",
        "level_summaries",
        "estimand_summaries",
        "paired_population_support_summaries",
    ):
        aggregate[f"{field}_sha256"] = canonical_sha256(aggregate[field])
    aggregate["mandatory_audit_summary_sha256"] = canonical_sha256(
        aggregate["mandatory_audit_summary"]
    )
    aggregate.pop("payload_sha256", None)
    aggregate["payload_sha256"] = canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "payload_sha256"}
    )


def _refresh_outer_payload(aggregate: dict[str, object]) -> None:
    aggregate["level_summaries_sha256"] = canonical_sha256(aggregate["level_summaries"])
    aggregate.pop("payload_sha256", None)
    aggregate["payload_sha256"] = canonical_sha256(aggregate)


def _refresh_one_summary_list(aggregate: dict[str, object], field: str) -> None:
    aggregate[f"{field}_sha256"] = canonical_sha256(aggregate[field])
    aggregate.pop("payload_sha256", None)
    aggregate["payload_sha256"] = canonical_sha256(aggregate)


def _rehash_managed_ledger(aggregate: dict[str, object]) -> None:
    ledger = cast(dict[str, object], aggregate["managed_ledger_snapshot"])
    ledger.pop("inventory_sha256", None)
    ledger["inventory_sha256"] = canonical_sha256(ledger)
    _refresh(aggregate)


def _set_audit_failure(aggregate: dict[str, object]) -> None:
    summary = cast(dict[str, object], aggregate["mandatory_audit_summary"])
    results = cast(list[dict[str, object]], summary["requirement_results"])
    first = results[0]
    first["decision"] = "invalid_calibration"
    first["failed_case_indices"] = [0]
    first["requirement_result_sha256"] = canonical_sha256(
        {key: value for key, value in first.items() if key != "requirement_result_sha256"}
    )
    summary["decision"] = "invalid_calibration"
    summary["failed_requirement_ids"] = [first["requirement_id"]]
    summary["requirement_results_sha256"] = canonical_sha256(results)
    aggregate["mandatory_audit_decision"] = "invalid_calibration"
    aggregate["gate_decision_status"] = "invalid_calibration_mandatory_audit_failure"
    audit_gate = cast(
        dict[str, object], cast(list[object], aggregate["mandatory_gate_results"])[-1]
    )
    audit_gate["decision"] = "invalid_calibration"
    first_reference = cast(list[dict[str, object]], audit_gate["references"])[0]
    first_reference["decision"] = "invalid_calibration"
    first_reference["requirement_result_sha256"] = first["requirement_result_sha256"]
    _refresh(aggregate)


def _set_all_bounds(statistics: dict[str, object], bound: float) -> None:
    pooled = cast(dict[str, object], statistics["pooled"])
    pooled.update(_coherent_inference(bound, cast(int, pooled["observed_n"])))
    manifest_means: list[float] = []
    for raw in cast(list[object], statistics["by_manifest"]):
        row = cast(dict[str, object], raw)
        row.update(_coherent_inference(bound, cast(int, row["observed_n"])))
        manifest_means.append(float.fromhex(cast(str, row["mean_hex"])))
    worst = cast(dict[str, object], statistics["worst_manifest"])
    worst["minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex"] = bound.hex()
    worst["minimum_manifest_mean_hex"] = min(manifest_means).hex()


def _endpoint_results(receipt: Mapping[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for raw_family in cast(list[object], receipt["mandatory_gate_results"]):
        family = cast(dict[str, object], raw_family)
        if family["gate_kind"] == "statistical_threshold":
            results.extend(cast(list[dict[str, object]], family["endpoint_results"]))
    return results


def test_success_freezes_exactly_35_oriented_endpoints_and_validates_canonically() -> None:
    aggregate = _fake_aggregate()
    receipt = thresholds.materialize_hidden_regime_factorial_threshold_freeze_receipt(aggregate)

    assert receipt["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_FROZEN
    assert receipt["thresholds_frozen"] is True
    assert len(cast(list[object], receipt["frozen_thresholds"])) == 35
    assert len(_endpoint_results(receipt)) == 35
    assert receipt["all_calibration_seeds_consumed"] is True
    assert receipt["calibration_case_count_consumed"] == 240
    assert receipt["protected_namespace_derived"] is False
    assert receipt["protected_outcomes_observed"] is False
    assert receipt["calibration_outcomes_payload_sha256"] == aggregate["payload_sha256"]
    assert receipt["mandatory_statistical_endpoint_identities_sha256"] == (
        thresholds.MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
    )
    assert receipt["mandatory_statistical_endpoint_ids_sha256"] == (
        thresholds.MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
    )
    required_fields = set(
        build_hidden_regime_factorial_calibration_design().threshold_rule.receipt_required_fields
    )
    assert required_fields.issubset(receipt)
    assert thresholds.validate_threshold_freeze_receipt(
        receipt,
        calibration_aggregate=aggregate,
    ) == receipt
    assert thresholds.canonical_threshold_freeze_receipt_bytes(
        receipt,
        calibration_aggregate=aggregate,
    ) == canonical_json_bytes(receipt)


def test_valid_rejection_is_atomic_immutable_and_contains_no_partial_freeze() -> None:
    aggregate = _fake_aggregate()
    first_source = next(_iter_statistical_sources(aggregate))
    null = float.fromhex(cast(str, first_source["oriented_null_hex"]))
    _set_all_bounds(cast(dict[str, object], first_source["statistics"]), null + 0.0001)
    _refresh(aggregate)

    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)

    assert receipt["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_REJECTION
    assert receipt["thresholds_frozen"] is False
    assert receipt["frozen_thresholds"] == []
    assert set(receipt) == set(
        thresholds.freeze_hidden_regime_factorial_thresholds(_fake_aggregate())
    )
    reasons = cast(list[dict[str, object]], receipt["rejection_reasons"])
    assert any(
        "continuous_half_margin_below_rounding_quantum"
        in cast(list[str], item["reasons"])
        for item in reasons
    )
    assert thresholds.validate_threshold_freeze_receipt(
        receipt,
        calibration_aggregate=aggregate,
    ) == receipt


def test_win_thresholds_use_full_denominator_and_ties_never_become_wins() -> None:
    success = thresholds.freeze_hidden_regime_factorial_thresholds(_fake_aggregate())
    first_frozen = cast(list[dict[str, object]], success["frozen_thresholds"])[0]
    assert first_frozen["pooled_win_threshold"] == 22
    assert [
        item["win_threshold"]
        for item in cast(list[dict[str, object]], first_frozen["manifest_win_thresholds"])
    ] == [7, 7, 7]
    assert first_frozen["ties_count_as_wins"] is False

    aggregate = _fake_aggregate()
    source = next(_iter_statistical_sources(aggregate))
    statistics = cast(dict[str, object], source["statistics"])
    pooled = cast(dict[str, object], statistics["pooled"])
    pooled.update({"wins": 18, "ties": 12, "losses": 0})
    for raw_manifest in cast(list[object], statistics["by_manifest"]):
        manifest = cast(dict[str, object], raw_manifest)
        manifest.update({"wins": 6, "ties": 4, "losses": 0})
    cast(dict[str, object], statistics["worst_manifest"])["minimum_manifest_wins"] = 6
    _refresh(aggregate)

    rejection = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    assert rejection["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_REJECTION
    endpoint = _endpoint_results(rejection)[0]
    win_inputs = cast(list[dict[str, object]], endpoint["win_threshold_candidates"])
    assert win_inputs[0]["null_win_count"] == 15
    assert win_inputs[0]["wins"] == 18
    assert all(item["ties_count_as_wins"] is False for item in win_inputs)
    assert all(item["win_threshold"] is None for item in win_inputs[1:])


def test_lower_is_better_level_is_frozen_in_oriented_space() -> None:
    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(_fake_aggregate())
    frozen = next(
        item
        for item in cast(list[dict[str, object]], receipt["frozen_thresholds"])
        if cast(dict[str, object], item["reference"])["kind"] == "absolute_level"
        and cast(dict[str, object], item["reference"])["metric_id"]
        == "qualified_first_entry_window_error_rate"
    )
    null = Decimal.from_float(float.fromhex(cast(str, frozen["oriented_null_hex"])))
    oriented_threshold = Decimal(cast(str, frozen["oriented_continuous_threshold_decimal"]))
    assert frozen["orientation"] == "lower"
    assert frozen["threshold_space"] == "oriented_higher_is_favorable"
    assert oriented_threshold > null
    assert -oriented_threshold < -null


def test_floor_rounding_accepts_exact_binary64_boundary_and_rejects_predecessor() -> None:
    aggregate = _fake_aggregate()
    support = cast(
        dict[str, object],
        cast(list[object], aggregate["paired_population_support_summaries"])[0],
    )
    boundary = float("0.0002")
    _set_all_bounds(cast(dict[str, object], support["statistics"]), boundary)
    _refresh(aggregate)
    success = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    endpoint = next(
        item
        for item in _endpoint_results(success)
        if cast(dict[str, object], item["reference"])["metric_id"] == support["metric_id"]
    )
    continuous = cast(dict[str, object], endpoint["continuous_threshold_candidate"])
    assert continuous["margin_quantum_decimal"] == "0.0001"

    below = copy.deepcopy(aggregate)
    below_support = cast(
        dict[str, object],
        cast(list[object], below["paired_population_support_summaries"])[0],
    )
    _set_all_bounds(
        cast(dict[str, object], below_support["statistics"]),
        math.nextafter(boundary, -math.inf),
    )
    _refresh(below)
    rejection = thresholds.freeze_hidden_regime_factorial_thresholds(below)
    assert rejection["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_REJECTION


def test_threshold_receipt_is_independent_of_process_decimal_context() -> None:
    aggregate = _fake_aggregate()
    expected = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_UP
        actual = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    assert actual == expected


def test_missing_or_duplicate_mandatory_endpoint_source_is_invalid() -> None:
    missing = _fake_aggregate()
    cast(list[object], missing["level_summaries"]).pop(0)
    _refresh_outer_payload(missing)
    with pytest.raises(thresholds.ThresholdFreezeError, match="incomplete|missing level source"):
        thresholds.freeze_hidden_regime_factorial_thresholds(missing)

    duplicate = _fake_aggregate()
    levels = cast(list[object], duplicate["level_summaries"])
    levels.append(copy.deepcopy(levels[0]))
    _refresh(duplicate)
    with pytest.raises(thresholds.ThresholdFreezeError, match="duplicate aggregate level endpoint"):
        thresholds.freeze_hidden_regime_factorial_thresholds(duplicate)


def test_reference_cardinality_is_exact_and_duplicate_reference_cannot_substitute() -> None:
    missing = _fake_aggregate()
    first_gate = cast(
        dict[str, object], cast(list[object], missing["mandatory_gate_results"])[0]
    )
    cast(list[object], first_gate["references"]).pop()
    _refresh(missing)
    with pytest.raises(thresholds.ThresholdFreezeError, match="reference cardinality"):
        thresholds.freeze_hidden_regime_factorial_thresholds(missing)

    duplicate = _fake_aggregate()
    first_gate = cast(
        dict[str, object], cast(list[object], duplicate["mandatory_gate_results"])[0]
    )
    references = cast(list[object], first_gate["references"])
    references[-1] = copy.deepcopy(references[0])
    _refresh(duplicate)
    with pytest.raises(thresholds.ThresholdFreezeError, match="reference differs"):
        thresholds.freeze_hidden_regime_factorial_thresholds(duplicate)


def test_frozen_ordered_endpoint_identity_digest_is_literal_and_complete() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    identities = thresholds._frozen_mandatory_statistical_endpoint_identities(design)
    assert len(identities) == 35
    assert canonical_sha256(identities) == (
        "1644bbba320be78e75491b7652a4d73f5fb8db2361d33426614a1fd22994de45"
    )
    assert canonical_sha256(identities) == (
        thresholds.MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
    )
    endpoint_ids = [canonical_sha256(identity) for identity in identities]
    assert canonical_sha256(endpoint_ids) == (
        "769325b8f1f52a0f18f095e67a88296ac63ecc8062be238a3c091b80493c91b1"
    )
    assert canonical_sha256(endpoint_ids) == (
        thresholds.MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
    )


@pytest.mark.parametrize(
    "record_field",
    ["started_records", "completed_records", "finalized_records", "attempt_records"],
)
def test_rehashed_duplicate_or_misindexed_managed_case_record_is_invalid(
    record_field: str,
) -> None:
    aggregate = _fake_aggregate()
    ledger = cast(dict[str, object], aggregate["managed_ledger_snapshot"])
    records = cast(list[dict[str, object]], ledger[record_field])
    records[1]["case_index"] = 0
    _rehash_managed_ledger(aggregate)

    with pytest.raises(thresholds.ThresholdFreezeError, match="duplicate or misindexed"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


def test_rehashed_duplicate_mandatory_case_audit_reference_is_invalid() -> None:
    aggregate = _fake_aggregate()
    summary = cast(dict[str, object], aggregate["mandatory_audit_summary"])
    references = cast(list[dict[str, object]], summary["case_audit_references"])
    references[1]["case_index"] = 0
    summary["case_audit_references_sha256"] = canonical_sha256(references)
    _refresh(aggregate)

    with pytest.raises(thresholds.ThresholdFreezeError, match="duplicate or misindexed"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


def test_rehashed_misindexed_nested_managed_attempt_is_invalid() -> None:
    aggregate = _fake_aggregate()
    ledger = cast(dict[str, object], aggregate["managed_ledger_snapshot"])
    attempts_by_case = cast(list[dict[str, object]], ledger["attempt_records"])
    first_attempts = cast(list[dict[str, object]], attempts_by_case[0]["attempts"])
    first_attempts[0]["attempt_index"] = 1
    attempts_by_case[0]["attempt_records_sha256"] = canonical_sha256(first_attempts)
    _rehash_managed_ledger(aggregate)

    with pytest.raises(
        thresholds.ThresholdFreezeError,
        match="attempt_index|duplicate or misindexed",
    ):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


@pytest.mark.parametrize("summary_kind", ["level", "estimand", "support"])
def test_rehashed_unexpected_statistical_summary_row_is_invalid(summary_kind: str) -> None:
    aggregate = _fake_aggregate()
    if summary_kind == "level":
        levels = cast(list[dict[str, object]], aggregate["level_summaries"])
        extra = copy.deepcopy(levels[-1])
        extra["condition"] = "unexpected_condition"
        levels.append(extra)
    elif summary_kind == "estimand":
        estimands = cast(list[dict[str, object]], aggregate["estimand_summaries"])
        estimands.append(
            {
                "estimand_id": "unexpected_estimand",
                "oriented_null_hex": (0.0).hex(),
                "metrics": [],
            }
        )
    else:
        supports = cast(
            list[dict[str, object]],
            aggregate["paired_population_support_summaries"],
        )
        extra = copy.deepcopy(supports[-1])
        extra["metric_id"] = "unexpected_support_metric"
        supports.append(extra)
    _refresh(aggregate)

    with pytest.raises(thresholds.ThresholdFreezeError, match="unexpected"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


@pytest.mark.parametrize(
    ("summary_field", "remove_index"),
    [
        ("level_summaries", -1),
        ("estimand_summaries", 2),
        ("paired_population_support_summaries", -1),
    ],
)
def test_rehashed_missing_even_nonmandatory_summary_row_is_invalid(
    summary_field: str,
    remove_index: int,
) -> None:
    aggregate = _fake_aggregate()
    cast(list[object], aggregate[summary_field]).pop(remove_index)
    _refresh_one_summary_list(aggregate, summary_field)

    with pytest.raises(thresholds.ThresholdFreezeError, match="incomplete"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


def test_rehashed_algebraically_inconsistent_standard_error_is_invalid() -> None:
    aggregate = _fake_aggregate()
    source = next(_iter_statistical_sources(aggregate))
    statistics = cast(dict[str, object], source["statistics"])
    pooled = cast(dict[str, object], statistics["pooled"])
    standard_error = float.fromhex(cast(str, pooled["standard_error_hex"]))
    pooled["standard_error_hex"] = math.nextafter(standard_error, math.inf).hex()
    _refresh(aggregate)

    with pytest.raises(thresholds.ThresholdFreezeError, match="algebraically inconsistent"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


def test_rehashed_algebraically_inconsistent_lower_bound_is_invalid() -> None:
    aggregate = _fake_aggregate()
    source = next(_iter_statistical_sources(aggregate))
    statistics = cast(dict[str, object], source["statistics"])
    pooled = cast(dict[str, object], statistics["pooled"])
    lower_bound = float.fromhex(
        cast(str, pooled["one_sided_95_percent_lower_confidence_bound_hex"])
    )
    pooled["one_sided_95_percent_lower_confidence_bound_hex"] = math.nextafter(
        lower_bound, -math.inf
    ).hex()
    _refresh(aggregate)

    with pytest.raises(thresholds.ThresholdFreezeError, match="algebraically inconsistent"):
        thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)


def test_exact_eligible_cardinality_is_input_invariant_but_structural_loss_rejects() -> None:
    malformed = _fake_aggregate()
    source = next(_iter_statistical_sources(malformed))
    statistics = cast(dict[str, object], source["statistics"])
    cast(dict[str, object], statistics["pooled"])["eligible_n"] = 29
    _refresh(malformed)
    with pytest.raises(thresholds.ThresholdFreezeError, match="eligible_n must equal 30"):
        thresholds.freeze_hidden_regime_factorial_thresholds(malformed)

    rejected = _fake_aggregate()
    source = next(_iter_statistical_sources(rejected))
    null = float.fromhex(cast(str, source["oriented_null_hex"]))
    source["statistics"] = _statistics(null, missing_kind="structural")
    _refresh(rejected)
    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(rejected)
    assert receipt["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_REJECTION
    assert any(
        "structural_missing_n_not_zero" in reason
        for item in cast(list[dict[str, object]], receipt["rejection_reasons"])
        for reason in cast(list[str], item["reasons"])
    )


def test_conditional_unobserved_rows_are_not_structural_missing_or_automatic_rejection() -> None:
    aggregate = _fake_aggregate()
    source = next(_iter_statistical_sources(aggregate))
    null = float.fromhex(cast(str, source["oriented_null_hex"]))
    source["statistics"] = _statistics(null, missing_kind="conditional")
    _refresh(aggregate)

    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)

    assert receipt["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_FROZEN
    endpoint = _endpoint_results(receipt)[0]
    wins = cast(list[dict[str, object]], endpoint["win_threshold_candidates"])
    assert wins[0]["required_n"] == 30
    assert wins[0]["observed_n"] == 29
    assert wins[0]["conditional_unobserved_n"] == 1
    assert wins[0]["null_win_count"] == 15


def test_receipt_and_aggregate_provenance_tampering_fail_closed() -> None:
    aggregate = _fake_aggregate()
    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    tampered = copy.deepcopy(receipt)
    cast(list[dict[str, object]], tampered["frozen_thresholds"])[0][
        "oriented_continuous_threshold_decimal"
    ] = "999"
    tampered["receipt_payload_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_payload_sha256"}
    )
    with pytest.raises(thresholds.ThresholdFreezeError, match="exact recomputation"):
        thresholds.validate_threshold_freeze_receipt(
            tampered,
            calibration_aggregate=aggregate,
        )

    bad_aggregate = copy.deepcopy(aggregate)
    readiness = cast(dict[str, object], bad_aggregate["readiness_binding"])
    governance = cast(dict[str, object], readiness["execution_governance"])
    governance["runtime_identity_sha256"] = _digest("other-runtime")
    _refresh(bad_aggregate)
    with pytest.raises(thresholds.ThresholdFreezeError, match="runtime_identity_sha256 differs"):
        thresholds.freeze_hidden_regime_factorial_thresholds(bad_aggregate)


def test_structurally_valid_audit_failure_rejects_but_indeterminate_audit_is_invalid() -> None:
    aggregate = _fake_aggregate()
    _set_audit_failure(aggregate)
    receipt = thresholds.freeze_hidden_regime_factorial_thresholds(aggregate)
    assert receipt["decision_status"] == thresholds.THRESHOLD_FREEZE_DECISION_REJECTION
    assert receipt["frozen_thresholds"] == []

    indeterminate = _fake_aggregate()
    audit = cast(
        dict[str, object], cast(list[object], indeterminate["mandatory_gate_results"])[-1]
    )
    audit["decision"] = "not_evaluated"
    _refresh(indeterminate)
    with pytest.raises(thresholds.ThresholdFreezeError, match="decisions differ"):
        thresholds.freeze_hidden_regime_factorial_thresholds(indeterminate)
