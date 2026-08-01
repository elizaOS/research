"""Outcome-free protected factorial plan derived from a successful threshold receipt.

This module contains no world constructor, learner runner, execution worker, or
authorization issuer.  Its public builder first asks the fail-closed threshold
engine to recompute the supplied receipt.  Protected seeds are derived only
after that exact receipt is proven to be a successful, immutable 35-endpoint
threshold freeze.  Importing this module derives no seeds and writes nothing.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CANONICAL_CONDITION_ORDER,
    EXPECTED_DYAD_STATE_BYTES,
    EXPECTED_DYAD_STATE_SCALARS,
    EXPECTED_ROLE_STATE_BYTES,
    EXPECTED_ROLE_STATE_SCALARS,
    FROZEN_SEED_PAIRS,
    N_CONDITIONS,
    N_MATCHED_CASES,
    N_SEED_PAIRS,
    SEED_SNAPSHOT_SHA256,
    THRESHOLD_FREEZE_RECEIPT_SCHEMA,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
    canonical_sha256,
)
from alberta_framework.evaluation.hidden_regime_factorial_thresholds import (
    CALIBRATION_AGGREGATE_SCHEMA,
    MANDATORY_STATISTICAL_ENDPOINT_COUNT,
    MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
    MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
    THRESHOLD_FREEZE_DECISION_FROZEN,
    ThresholdFreezeError,
    validate_hidden_regime_factorial_threshold_freeze_receipt,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    PROTECTED_CANDIDATE_PARTITION,
)

PROTECTED_PLAN_SCHEMA = "alberta.hidden-regime-factorial.protected-plan.v1"
PROTECTED_SEED_DERIVATION_SCHEMA = (
    "alberta.hidden-regime-factorial.protected-seed-pairs.sha256-be32-reject.v1"
)
PROTECTED_SEED_DISJOINTNESS_SCHEMA = (
    "alberta.hidden-regime-factorial.protected-seed-disjointness-proof.v1"
)
PROTECTED_EVALUATION_CONTRACT_SCHEMA = (
    "alberta.hidden-regime-factorial.protected-evaluation-contract.v1"
)
PROTECTED_DECISION_RULE_SCHEMA = "alberta.hidden-regime-factorial.protected-decision-rule.v1"

PROTECTED_MANIFEST_ORDER: tuple[str, ...] = (
    "hidden-regime-structural-a-v1",
    "hidden-regime-structural-b-v1",
    "hidden-regime-structural-c-v1",
)
PROTECTED_MANIFEST_PAYLOAD_SHA256: tuple[str, ...] = (
    "b5165407d87f41dd7d4973ec0dea5d15c2f99362a3477fcf5e66a11d7cd5c8a2",
    "a79a47934381a80efd6094cd5033c4ad11b6ed955633f6e810785f32a94b3821",
    "7aefcc79278a0e4b37c524938d3e0bef783dc30ae4a4bad609b4c6564c9fda6c",
)
PROTECTED_MANIFEST_ORDER_SHA256 = "d690a9a29be7cd16fe8703ae1f0d02939da0475d30ecea532c75f6995055095b"
PROTECTED_MANIFEST_BINDINGS_SHA256 = (
    "9d1cd45f3df0d79c2001c88053e890621a0cabd144e59e1566ebd297a877539f"
)
PROTECTED_RECURRENCE_BINDINGS_SHA256 = (
    "1b0f43a20b22d55294f6c458f99e65d0916d3bcd52d37304d5cae64896a09eb4"
)

PROTECTED_SEED_NAMESPACE_PREFIX = "hidden-regime-factorial-protected-v1-"
PROTECTED_SEED_SELECTION_ORDER = (
    "seed_index ascending from 0 through 29; world lane before learner lane; for each lane "
    "select the lowest uint32 counter whose SHA-256 big-endian candidate is absent from all "
    "calibration seeds and all earlier protected candidates"
)

# Public test vectors disclose only the all-zero dummy receipt namespace.  They
# are not an official or evidence-eligible protected seed snapshot.
DUMMY_THRESHOLD_RECEIPT_SHA256 = "0" * 64
DUMMY_PROTECTED_SEED_PAIRS_SHA256 = (
    "262451b8299ea212049534b298596aa526380d7eaaea2c2e6525929ebc074448"
)
DUMMY_PROTECTED_SEED_SNAPSHOT_SHA256 = (
    "79b6bb4ae72ed30765589ebd5e1ed98cf729bd9e05b510ab9e788743949c10ed"
)

EXPECTED_PROTECTED_STEPS = 16_528
EXPECTED_PROTECTED_RECURRENCES = 12
EXPECTED_PROTECTED_MANIFEST_SEED_PAIRS = 10
EXPECTED_PROTECTED_MANIFEST_CASES = 80

_SHA256_LENGTH = 64
_UINT32_MAX = (1 << 32) - 1
_MAX_PLAN_BYTES = 32 * 1024 * 1024
_MAX_BOUND_INPUT_BYTES = 64 * 1024 * 1024
_AT_EMPTY_PATH = 0x1000

type SeedLane = Literal["world", "learner"]


class ProtectedPlanError(RuntimeError):
    """A protected-plan prerequisite, contract, or publication check failed."""


def _fail(message: str) -> NoReturn:
    raise ProtectedPlanError(message)


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


def _payload_with_digest(body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    return {**normalized, "payload_sha256": canonical_sha256(normalized)}


def _validated_flat_digest(
    payload: Mapping[str, object],
    *,
    digest_field: str,
    label: str,
) -> dict[str, object]:
    _require(type(payload) is dict, f"{label} must be a plain object")
    normalized = dict(payload)
    try:
        canonical_json_bytes(normalized)
    except (TypeError, ValueError) as exc:
        raise ProtectedPlanError(f"{label} is not canonical JSON data") from exc
    digest = normalized.pop(digest_field, None)
    _sha256(digest, f"{label}.{digest_field}")
    _require(canonical_sha256(normalized) == digest, f"{label} digest differs")
    return normalized


@dataclass(frozen=True, slots=True)
class _ValidatedSuccessfulThresholdReceipt:
    payload: dict[str, object]
    receipt_payload_sha256: str


def _validate_successful_threshold_receipt(
    threshold_receipt: Mapping[str, object],
    *,
    calibration_aggregate: Mapping[str, object],
) -> _ValidatedSuccessfulThresholdReceipt:
    """Return a capability-like value only after exact threshold recomputation succeeds."""

    try:
        validated = validate_hidden_regime_factorial_threshold_freeze_receipt(
            threshold_receipt,
            calibration_aggregate=calibration_aggregate,
        )
    except (ThresholdFreezeError, TypeError, ValueError) as exc:
        raise ProtectedPlanError("threshold-freeze receipt validation failed") from exc
    body = _validated_flat_digest(
        validated,
        digest_field="receipt_payload_sha256",
        label="threshold-freeze receipt",
    )
    _require(
        body.get("receipt_schema") == THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        "threshold-freeze receipt schema differs",
    )
    _require(
        body.get("decision_status") == THRESHOLD_FREEZE_DECISION_FROZEN
        and body.get("thresholds_frozen") is True,
        "protected planning requires a successful threshold-freeze receipt",
    )
    frozen = _plain_list(body.get("frozen_thresholds"), "frozen thresholds")
    _require(
        len(frozen) == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        "successful receipt must contain exactly 35 frozen thresholds",
    )
    endpoint_ids = [
        _sha256(_plain_dict(item, "frozen threshold").get("endpoint_id"), "endpoint_id")
        for item in frozen
    ]
    _require(len(endpoint_ids) == len(set(endpoint_ids)), "frozen threshold IDs are not unique")
    _require(
        canonical_sha256(endpoint_ids) == MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        "frozen threshold endpoint-ID order differs",
    )
    for field, expected in (
        ("protocol_payload_sha256", CALIBRATION_DESIGN_PAYLOAD_SHA256),
        ("seed_snapshot_sha256", SEED_SNAPSHOT_SHA256),
        ("mandatory_statistical_endpoint_count", MANDATORY_STATISTICAL_ENDPOINT_COUNT),
        ("all_calibration_seeds_consumed", True),
        ("calibration_case_count_consumed", N_MATCHED_CASES),
        ("calibration_seed_pair_count_consumed", N_SEED_PAIRS),
        ("mandatory_audit_decision", "passed_nonstatistical"),
        (
            "mandatory_statistical_endpoint_identities_sha256",
            MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
        ),
        (
            "mandatory_statistical_endpoint_ids_sha256",
            MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        ),
        ("protected_namespace_derived", False),
        ("protected_outcomes_observed", False),
        ("scientific_promotion_allowed", False),
        ("amendments_allowed", False),
        ("claim_accepted", False),
    ):
        _require(body.get(field) == expected, f"threshold-freeze receipt {field} differs")
    _require(body.get("rejection_reasons") == [], "successful receipt contains a rejection")
    for field in (
        "readiness_receipt_sha256",
        "gate_matrix_sha256",
        "calibration_outcomes_payload_sha256",
        "source_closure_sha256",
        "source_archive_sha256",
        "environment_identity_sha256",
        "managed_ledger_snapshot_sha256",
        "managed_ledger_content_address",
        "execution_governance_genesis_sha256",
        "aggregation_readiness_certification_binding_sha256",
        "mandatory_audit_summary_sha256",
        "case_ledger_sha256",
    ):
        _sha256(body.get(field), f"threshold-freeze receipt {field}")
    aggregate_body = _validated_flat_digest(
        calibration_aggregate,
        digest_field="payload_sha256",
        label="calibration aggregate",
    )
    _require(
        aggregate_body.get("schema") == CALIBRATION_AGGREGATE_SCHEMA,
        "calibration aggregate schema differs",
    )
    _require(
        body["calibration_outcomes_payload_sha256"] == calibration_aggregate["payload_sha256"],
        "threshold receipt is not bound to the supplied aggregate",
    )
    digest = _sha256(validated.get("receipt_payload_sha256"), "threshold receipt payload")
    return _ValidatedSuccessfulThresholdReceipt(copy.deepcopy(validated), digest)


def _protected_seed_candidate(
    receipt_payload_sha256: str,
    index: int,
    lane: SeedLane,
    counter: int,
) -> int:
    _sha256(receipt_payload_sha256, "threshold receipt payload")
    _strict_int(index, "protected seed index", maximum=N_SEED_PAIRS - 1)
    _require(lane in {"world", "learner"}, "protected seed lane differs")
    _strict_int(counter, "protected seed counter", maximum=_UINT32_MAX)
    lane_code = b"\x00" if lane == "world" else b"\x01"
    preimage = (
        PROTECTED_SEED_DERIVATION_SCHEMA.encode("ascii")
        + b"\x00"
        + bytes.fromhex(receipt_payload_sha256)
        + index.to_bytes(4, "big")
        + lane_code
        + counter.to_bytes(4, "big")
    )
    return int.from_bytes(hashlib.sha256(preimage).digest()[:4], "big")


def _derive_protected_seed_snapshot(
    receipt: _ValidatedSuccessfulThresholdReceipt,
) -> dict[str, object]:
    """Derive exactly 30 matched pairs after successful-receipt validation."""

    used = {seed for pair in FROZEN_SEED_PAIRS for seed in (pair.world_seed, pair.learner_seed)}
    lanes: tuple[SeedLane, SeedLane] = ("world", "learner")
    pairs: list[dict[str, object]] = []
    for index in range(N_SEED_PAIRS):
        record: dict[str, object] = {"index": index}
        for lane in lanes:
            counter = 0
            while True:
                candidate = _protected_seed_candidate(
                    receipt.receipt_payload_sha256,
                    index,
                    lane,
                    counter,
                )
                if candidate not in used:
                    break
                _require(counter < _UINT32_MAX, "protected seed counter space is exhausted")
                counter += 1
            used.add(candidate)
            record[f"{lane}_seed"] = candidate
            record[f"{lane}_derivation_counter"] = counter
        pairs.append(record)
    return {
        "schema": PROTECTED_SEED_DERIVATION_SCHEMA,
        "threshold_freeze_receipt_payload_sha256": receipt.receipt_payload_sha256,
        "namespace": PROTECTED_SEED_NAMESPACE_PREFIX + receipt.receipt_payload_sha256,
        "pairs": pairs,
    }


def _seed_disjointness_proof(seed_snapshot: Mapping[str, object]) -> dict[str, object]:
    _require(
        set(seed_snapshot) == {
            "schema",
            "threshold_freeze_receipt_payload_sha256",
            "namespace",
            "pairs",
        },
        "protected seed snapshot fields differ",
    )
    _require(
        seed_snapshot.get("schema") == PROTECTED_SEED_DERIVATION_SCHEMA,
        "protected seed derivation schema differs",
    )
    receipt_digest = _sha256(
        seed_snapshot.get("threshold_freeze_receipt_payload_sha256"),
        "protected seed receipt binding",
    )
    _require(
        seed_snapshot.get("namespace") == PROTECTED_SEED_NAMESPACE_PREFIX + receipt_digest,
        "protected seed namespace differs",
    )
    exactly_derived = _derive_protected_seed_snapshot(
        _ValidatedSuccessfulThresholdReceipt(
            payload={},
            receipt_payload_sha256=receipt_digest,
        )
    )
    try:
        supplied_bytes = canonical_json_bytes(dict(seed_snapshot))
    except (TypeError, ValueError) as exc:
        raise ProtectedPlanError("protected seed snapshot is not canonical JSON data") from exc
    _require(
        supplied_bytes == canonical_json_bytes(exactly_derived),
        "protected seed snapshot differs from exact collision-rejecting derivation",
    )
    raw_pairs = _plain_list(seed_snapshot.get("pairs"), "protected seed pairs")
    _require(len(raw_pairs) == N_SEED_PAIRS, "protected seed-pair count differs")
    protected_values: list[int] = []
    counters: list[dict[str, object]] = []
    for expected_index, raw_pair in enumerate(raw_pairs):
        pair = _plain_dict(raw_pair, "protected seed pair")
        _require(
            set(pair)
            == {
                "index",
                "world_seed",
                "world_derivation_counter",
                "learner_seed",
                "learner_derivation_counter",
            },
            "protected seed-pair fields differ",
        )
        _require(pair.get("index") == expected_index, "protected seed indices differ")
        world = _strict_int(pair.get("world_seed"), "protected world seed", maximum=_UINT32_MAX)
        learner = _strict_int(
            pair.get("learner_seed"),
            "protected learner seed",
            maximum=_UINT32_MAX,
        )
        protected_values.extend((world, learner))
        counters.append(
            {
                "index": expected_index,
                "world_derivation_counter": _strict_int(
                    pair.get("world_derivation_counter"),
                    "world derivation counter",
                    maximum=_UINT32_MAX,
                ),
                "learner_derivation_counter": _strict_int(
                    pair.get("learner_derivation_counter"),
                    "learner derivation counter",
                    maximum=_UINT32_MAX,
                ),
            }
        )
    calibration_values = sorted(
        seed for pair in FROZEN_SEED_PAIRS for seed in (pair.world_seed, pair.learner_seed)
    )
    protected_sorted = sorted(protected_values)
    intersection = sorted(set(calibration_values).intersection(protected_values))
    globally_unique = len(protected_values) == len(set(protected_values)) == 2 * N_SEED_PAIRS
    _require(globally_unique and not intersection, "protected seeds are not unique and disjoint")
    return {
        "schema": PROTECTED_SEED_DISJOINTNESS_SCHEMA,
        "selection_order": PROTECTED_SEED_SELECTION_ORDER,
        "calibration_seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "calibration_seed_value_count": len(calibration_values),
        "calibration_seed_values_sha256": canonical_sha256(calibration_values),
        "protected_seed_value_count": len(protected_values),
        "protected_unique_seed_value_count": len(set(protected_values)),
        "protected_seed_values_sha256": canonical_sha256(protected_sorted),
        "derivation_counters_sha256": canonical_sha256(counters),
        "cross_partition_intersection_count": len(intersection),
        "cross_partition_intersection_values": intersection,
        "cross_partition_intersection_sha256": canonical_sha256(intersection),
        "globally_unique_and_disjoint": globally_unique and not intersection,
    }


def _protected_manifest_bindings() -> list[dict[str, object]]:
    _require(
        tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS) == PROTECTED_MANIFEST_ORDER,
        "protected manifest registry order differs",
    )
    bindings: list[dict[str, object]] = []
    for name, expected_digest in zip(
        PROTECTED_MANIFEST_ORDER,
        PROTECTED_MANIFEST_PAYLOAD_SHA256,
        strict=True,
    ):
        manifest = HIDDEN_REGIME_STRUCTURAL_MANIFESTS[name]
        ledger = HIDDEN_REGIME_MANIFEST_USE_LEDGER[name]
        actual_digest = canonical_sha256(manifest.to_dict())
        _require(actual_digest == expected_digest, f"protected manifest content changed: {name}")
        _require(
            manifest.use_partition == PROTECTED_CANDIDATE_PARTITION
            and ledger.use_partition == PROTECTED_CANDIDATE_PARTITION
            and ledger.learner_outcomes_executed is False
            and ledger.calibration_use_allowed is False
            and ledger.protected_evaluation_candidate is True
            and ledger.scientific_promotion_allowed is False,
            f"protected manifest ledger changed: {name}",
        )
        bindings.append(
            {
                "name": name,
                "use_partition": PROTECTED_CANDIDATE_PARTITION,
                "manifest_payload_sha256": expected_digest,
            }
        )
    _require(
        canonical_sha256(list(PROTECTED_MANIFEST_ORDER)) == PROTECTED_MANIFEST_ORDER_SHA256,
        "protected manifest order digest differs",
    )
    _require(
        canonical_sha256(bindings) == PROTECTED_MANIFEST_BINDINGS_SHA256,
        "protected manifest binding digest differs",
    )
    return bindings


def _protected_recurrence_bindings() -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for name in PROTECTED_MANIFEST_ORDER:
        manifest = HIDDEN_REGIME_STRUCTURAL_MANIFESTS[name]
        _require(
            sum(manifest.segment_lengths) == EXPECTED_PROTECTED_STEPS,
            "protected manifest horizon differs",
        )
        episode_starts = tuple(
            index
            for index, regime in enumerate(manifest.segment_regimes)
            if index == 0 or manifest.segment_regimes[index - 1] != regime
        )
        observed: Counter[int] = Counter()
        recurrence_starts: list[int] = []
        recurrence_identities: list[list[int]] = []
        recurrence_counts: Counter[int] = Counter()
        for index in episode_starts:
            regime = manifest.segment_regimes[index]
            if observed[regime] > 0:
                recurrence_starts.append(index)
                recurrence_identities.append([index, regime, observed[regime]])
                recurrence_counts[regime] += 1
            observed[regime] += 1
        _require(
            len(recurrence_starts) == EXPECTED_PROTECTED_RECURRENCES,
            f"protected manifest does not have 12 recurrences: {name}",
        )
        bindings.append(
            {
                "manifest_name": name,
                "eligibility_rule": (
                    "episode entry for a previously observed exact regime after at least one "
                    "complete intervening transition under a different regime; adjacent "
                    "equal-regime segments are one uninterrupted episode and their boundary "
                    "is never a recurrence"
                ),
                "required_runtime_helper": ("hidden_regime_lineage_recurrence_segments(world)"),
                "required_trace_fields": [
                    "segment_index",
                    "regime_id",
                    "occurrence_index",
                    "raw_segment_occurrence_index",
                ],
                "repeat_schedule": False,
                "expected_total_steps": EXPECTED_PROTECTED_STEPS,
                "execution_horizon": "exactly_one_finite_schedule",
                "coalesced_episode_start_segment_indices": list(episode_starts),
                "eligible_recurrence_start_segment_indices": recurrence_starts,
                "eligible_recurrence_count": len(recurrence_starts),
                "eligible_recurrence_counts_by_regime": [
                    recurrence_counts[regime] for regime in range(5)
                ],
                "eligible_recurrence_identities": recurrence_identities,
            }
        )
    _require(
        canonical_sha256(bindings) == PROTECTED_RECURRENCE_BINDINGS_SHA256,
        "protected recurrence binding digest differs",
    )
    return bindings


def _evaluation_contract() -> dict[str, object]:
    design = build_hidden_regime_factorial_calibration_design()
    return {
        "schema": PROTECTED_EVALUATION_CONTRACT_SCHEMA,
        "calibration_protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "factorial_cells": [item.to_payload() for item in design.factorial_cells],
        "condition_runtime_bindings": [
            item.to_payload() for item in design.condition_runtime_bindings
        ],
        "base_config_binding": design.base_config_binding.to_payload(),
        "metric_contracts": [item.to_payload() for item in design.metrics],
        "paired_population_support_metrics": [
            item.to_payload() for item in design.paired_population_support_metrics
        ],
        "factorial_estimands": [item.to_payload() for item in design.factorial_estimands],
        "causal_control_estimands": [item.to_payload() for item in design.control_estimands],
        "audit_requirements": [item.to_payload() for item in design.audits],
        "gate_families": [item.to_payload() for item in design.gate_families],
        "statistical_plan": design.statistical_plan.to_payload(),
        "resource_contract": {
            "per_role_scalars": EXPECTED_ROLE_STATE_SCALARS,
            "per_role_bytes": EXPECTED_ROLE_STATE_BYTES,
            "dyad_scalars": EXPECTED_DYAD_STATE_SCALARS,
            "dyad_bytes": EXPECTED_DYAD_STATE_BYTES,
            "constant_at_every_step": True,
            "matched_across_all_conditions": True,
        },
    }


def _protected_decision_rule() -> dict[str, object]:
    return {
        "schema": PROTECTED_DECISION_RULE_SCHEMA,
        "all_mandatory_statistical_gates_conjunctive": True,
        "mandatory_missingness_maximum": 0,
        "threshold_comparison_rounding": "none",
        "display_rounding_can_affect_decision": False,
        "calibration_reestimation_permitted": False,
        "protected_threshold_adjustment_permitted": False,
        "descriptive_result_can_override_decision": False,
        "statistical_gate_failure_disposition": (
            "valid protected rejection; no retest, seed replacement, condition replacement, "
            "or threshold change"
        ),
        "audit_failure_disposition": (
            "invalid protected evaluation; no scientific decision, retry, or seed replacement"
        ),
        "automatic_promotion_permitted": False,
    }


def _assignments_and_cases(
    seed_snapshot: Mapping[str, object],
    manifest_bindings: Sequence[Mapping[str, object]],
    recurrence_bindings: Sequence[Mapping[str, object]],
    evaluation_contract: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _seed_disjointness_proof(seed_snapshot)
    pairs = [
        _plain_dict(item, "protected seed pair")
        for item in _plain_list(seed_snapshot.get("pairs"), "protected seed pairs")
    ]
    _require(len(pairs) == N_SEED_PAIRS, "protected seed-pair count differs")
    raw_manifests = [_plain_dict(item, "manifest binding") for item in manifest_bindings]
    _require(
        len(raw_manifests) == len(PROTECTED_MANIFEST_ORDER)
        and [item.get("name") for item in raw_manifests] == list(PROTECTED_MANIFEST_ORDER)
        and canonical_sha256(raw_manifests) == PROTECTED_MANIFEST_BINDINGS_SHA256,
        "protected manifest bindings are incomplete or out of order",
    )
    raw_recurrences = [
        _plain_dict(item, "recurrence binding") for item in recurrence_bindings
    ]
    _require(
        len(raw_recurrences) == len(PROTECTED_MANIFEST_ORDER)
        and [item.get("manifest_name") for item in raw_recurrences]
        == list(PROTECTED_MANIFEST_ORDER)
        and canonical_sha256(raw_recurrences) == PROTECTED_RECURRENCE_BINDINGS_SHA256,
        "protected recurrence bindings are incomplete or out of order",
    )
    try:
        supplied_contract_bytes = canonical_json_bytes(dict(evaluation_contract))
    except (TypeError, ValueError) as exc:
        raise ProtectedPlanError("evaluation contract is not canonical JSON data") from exc
    _require(
        supplied_contract_bytes == canonical_json_bytes(_evaluation_contract()),
        "protected evaluation contract differs from exact recomputation",
    )
    manifests = {cast(str, item["name"]): item for item in raw_manifests}
    recurrences = {cast(str, item["manifest_name"]): item for item in raw_recurrences}
    raw_runtime_bindings = [
        _plain_dict(item, "runtime binding")
        for item in _plain_list(
            evaluation_contract.get("condition_runtime_bindings"),
            "condition runtime bindings",
        )
    ]
    _require(
        len(raw_runtime_bindings) == N_CONDITIONS
        and [item.get("condition") for item in raw_runtime_bindings]
        == list(CANONICAL_CONDITION_ORDER),
        "condition runtime bindings are incomplete or out of order",
    )
    runtime_bindings = {
        cast(str, item["condition"]): item for item in raw_runtime_bindings
    }
    assignments: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for seed_index, pair in enumerate(pairs):
        _require(pair.get("index") == seed_index, "protected seed pair order differs")
        manifest_name = PROTECTED_MANIFEST_ORDER[seed_index % len(PROTECTED_MANIFEST_ORDER)]
        manifest = manifests[manifest_name]
        recurrence = recurrences[manifest_name]
        assignments.append({"seed_index": seed_index, "manifest_name": manifest_name})
        for condition_index, condition in enumerate(CANONICAL_CONDITION_ORDER):
            cases.append(
                {
                    "case_index": seed_index * N_CONDITIONS + condition_index,
                    "seed_index": seed_index,
                    "manifest_name": manifest_name,
                    "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                    "recurrence_binding_sha256": canonical_sha256(recurrence),
                    "world_seed": pair["world_seed"],
                    "learner_seed": pair["learner_seed"],
                    "condition": condition,
                    "condition_runtime_binding_sha256": canonical_sha256(
                        runtime_bindings[condition]
                    ),
                }
            )
    _require(len(cases) == N_MATCHED_CASES, "protected case count differs")
    _require(
        Counter(cast(str, item["manifest_name"]) for item in assignments)
        == Counter(
            {name: EXPECTED_PROTECTED_MANIFEST_SEED_PAIRS for name in PROTECTED_MANIFEST_ORDER}
        ),
        "protected assignment manifest balance differs",
    )
    _require(
        Counter(cast(str, item["manifest_name"]) for item in cases)
        == Counter({name: EXPECTED_PROTECTED_MANIFEST_CASES for name in PROTECTED_MANIFEST_ORDER}),
        "protected case manifest balance differs",
    )
    _require(
        Counter(cast(str, item["condition"]) for item in cases)
        == Counter({condition: N_SEED_PAIRS for condition in CANONICAL_CONDITION_ORDER}),
        "protected condition completeness differs",
    )
    _require(
        {(cast(int, item["seed_index"]), cast(str, item["condition"])) for item in cases}
        == {
            (seed_index, condition)
            for seed_index in range(N_SEED_PAIRS)
            for condition in CANONICAL_CONDITION_ORDER
        },
        "protected matched-case Cartesian coverage differs",
    )
    return assignments, cases


def build_hidden_regime_factorial_protected_plan(
    threshold_receipt: Mapping[str, object],
    *,
    calibration_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Build, but do not publish or authorize, the deterministic protected plan."""

    receipt = _validate_successful_threshold_receipt(
        threshold_receipt,
        calibration_aggregate=calibration_aggregate,
    )
    seed_snapshot = _derive_protected_seed_snapshot(receipt)
    disjointness = _seed_disjointness_proof(seed_snapshot)
    manifest_bindings = _protected_manifest_bindings()
    recurrence_bindings = _protected_recurrence_bindings()
    evaluation_contract = _evaluation_contract()
    assignments, cases = _assignments_and_cases(
        seed_snapshot,
        manifest_bindings,
        recurrence_bindings,
        evaluation_contract,
    )
    receipt_body = dict(receipt.payload)
    receipt_body.pop("receipt_payload_sha256")
    frozen_thresholds = copy.deepcopy(
        _plain_list(receipt_body["frozen_thresholds"], "frozen thresholds")
    )
    decision_rule = _protected_decision_rule()
    body = {
        "schema": PROTECTED_PLAN_SCHEMA,
        "status_time_scope": "facts observed when this protected plan was frozen",
        "plan_status": "preregistered_unexecuted",
        "use_partition": PROTECTED_CANDIDATE_PARTITION,
        "scientific_evidence_eligible_if_validated": True,
        "scientific_promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "amendments_allowed": False,
        "thresholds_frozen": True,
        "threshold_adjustment_permitted": False,
        "protected_namespace_derived": True,
        "protected_outcomes_observed": False,
        "learner_outcomes_executed": False,
        "learner_execution_authorized": False,
        "protected_execution_permitted": False,
        "execution_issuer_available": False,
        "protected_readiness_receipt_sha256": None,
        "protected_execution_ledger_genesis_sha256": None,
        "calibration_binding": {
            "protocol_payload_sha256": receipt_body["protocol_payload_sha256"],
            "calibration_seed_snapshot_sha256": receipt_body["seed_snapshot_sha256"],
            "calibration_aggregate_schema": CALIBRATION_AGGREGATE_SCHEMA,
            "calibration_aggregate_payload_sha256": receipt_body[
                "calibration_outcomes_payload_sha256"
            ],
            "calibration_readiness_receipt_sha256": receipt_body["readiness_receipt_sha256"],
            "calibration_gate_matrix_sha256": receipt_body["gate_matrix_sha256"],
            "calibration_source_closure_sha256": receipt_body["source_closure_sha256"],
            "calibration_source_archive_sha256": receipt_body["source_archive_sha256"],
            "calibration_environment_identity_sha256": receipt_body["environment_identity_sha256"],
            "calibration_managed_ledger_snapshot_sha256": receipt_body[
                "managed_ledger_snapshot_sha256"
            ],
            "calibration_managed_ledger_content_address": receipt_body[
                "managed_ledger_content_address"
            ],
            "calibration_execution_governance_genesis_sha256": receipt_body[
                "execution_governance_genesis_sha256"
            ],
            "calibration_case_ledger_sha256": receipt_body["case_ledger_sha256"],
            "aggregation_readiness_certification_binding_sha256": receipt_body[
                "aggregation_readiness_certification_binding_sha256"
            ],
            "mandatory_audit_summary_sha256": receipt_body["mandatory_audit_summary_sha256"],
        },
        "threshold_freeze_receipt_binding": {
            "receipt_schema": THRESHOLD_FREEZE_RECEIPT_SCHEMA,
            "receipt_payload_sha256": receipt.receipt_payload_sha256,
            "decision_status": THRESHOLD_FREEZE_DECISION_FROZEN,
            "mandatory_statistical_endpoint_count": MANDATORY_STATISTICAL_ENDPOINT_COUNT,
            "mandatory_statistical_endpoint_identities_sha256": (
                MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
            ),
            "mandatory_statistical_endpoint_ids_sha256": (
                MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
            ),
        },
        "frozen_thresholds": frozen_thresholds,
        "frozen_thresholds_sha256": canonical_sha256(frozen_thresholds),
        "protected_seed_snapshot": seed_snapshot,
        "protected_seed_snapshot_sha256": canonical_sha256(seed_snapshot),
        "seed_disjointness_proof": disjointness,
        "seed_disjointness_proof_sha256": canonical_sha256(disjointness),
        "structural_manifest_order": list(PROTECTED_MANIFEST_ORDER),
        "structural_manifest_order_sha256": PROTECTED_MANIFEST_ORDER_SHA256,
        "manifest_bindings": manifest_bindings,
        "manifest_bindings_sha256": PROTECTED_MANIFEST_BINDINGS_SHA256,
        "recurrence_eligibility_bindings": recurrence_bindings,
        "recurrence_eligibility_bindings_sha256": PROTECTED_RECURRENCE_BINDINGS_SHA256,
        "assignment_rule": "manifest_order[seed_index modulo 3]",
        "assignments": assignments,
        "assignments_sha256": canonical_sha256(assignments),
        "condition_order": list(CANONICAL_CONDITION_ORDER),
        "condition_order_sha256": canonical_sha256(list(CANONICAL_CONDITION_ORDER)),
        "cases": cases,
        "cases_sha256": canonical_sha256(cases),
        "seed_pair_count": N_SEED_PAIRS,
        "condition_count": N_CONDITIONS,
        "matched_case_count": N_MATCHED_CASES,
        "manifest_seed_pair_counts": [
            {
                "manifest_name": name,
                "count": sum(item["manifest_name"] == name for item in assignments),
            }
            for name in PROTECTED_MANIFEST_ORDER
        ],
        "manifest_case_counts": [
            {
                "manifest_name": name,
                "count": sum(item["manifest_name"] == name for item in cases),
            }
            for name in PROTECTED_MANIFEST_ORDER
        ],
        "condition_case_counts": [
            {
                "condition": condition,
                "count": sum(item["condition"] == condition for item in cases),
            }
            for condition in CANONICAL_CONDITION_ORDER
        ],
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": canonical_sha256(evaluation_contract),
        "protected_decision_rule": decision_rule,
        "protected_decision_rule_sha256": canonical_sha256(decision_rule),
        "claim_scope": (
            "protected evaluation of finite hidden-regime structural generalization under one "
            "preregistered 30-pair, eight-condition factorial"
        ),
        "limitations": [
            "no automatic evidence promotion",
            "no Alberta Plan completion claim",
            "no unbounded-retention or general continual-learning claim",
            "the plan itself contains no protected outcome or execution authorization",
        ],
    }
    return _payload_with_digest(body)


def validate_hidden_regime_factorial_protected_plan(
    payload: Mapping[str, object],
    *,
    threshold_receipt: Mapping[str, object],
    calibration_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Strictly recompute and byte-compare one unexecuted protected plan."""

    try:
        canonical_json_bytes(dict(payload))
    except (TypeError, ValueError) as exc:
        raise ProtectedPlanError("protected plan is not canonical JSON data") from exc
    _validated_flat_digest(payload, digest_field="payload_sha256", label="protected plan")
    expected = build_hidden_regime_factorial_protected_plan(
        threshold_receipt,
        calibration_aggregate=calibration_aggregate,
    )
    if canonical_json_bytes(dict(payload)) != canonical_json_bytes(expected):
        _fail("protected plan differs from exact recomputation")
    return expected


@dataclass(frozen=True, slots=True)
class PublishedProtectedPlan:
    path: Path
    payload_sha256: str
    payload: dict[str, object]


def protected_plan_path(publication_root: Path, payload_sha256: str) -> Path:
    _sha256(payload_sha256, "protected plan payload digest")
    return publication_root.absolute() / f"{payload_sha256}.json"


def _open_directory_without_symlinks(path: Path, *, label: str) -> tuple[int, Path]:
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ProtectedPlanError(
                        f"{label} traverses a symlink or non-directory component"
                    ) from exc
                if exc.errno == errno.ENOENT:
                    raise ProtectedPlanError(f"{label} is missing: {absolute}") from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        _require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"{label} is not a directory")
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _stat_fingerprint(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_uid,
        status.st_gid,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_exact_descriptor_bytes(
    descriptor: int,
    size: int,
    *,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(size - offset, 1024 * 1024), offset)
        _require(bool(chunk), f"{label} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    _require(os.pread(descriptor, 1, size) == b"", f"{label} grew while being read")
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class _OpenedImmutableBoundFile:
    path: Path
    label: str
    parent_descriptor: int
    descriptor: int
    parent_device: int
    parent_inode: int
    fingerprint: tuple[int, ...]
    raw: bytes

    def require_unchanged(self) -> None:
        """Re-resolve the held name and re-read the same inode immediately before use."""

        reopened_parent, normalized_parent = _open_directory_without_symlinks(
            self.path.parent,
            label=f"{self.label} parent",
        )
        try:
            reopened_parent_status = os.fstat(reopened_parent)
            _require(
                normalized_parent == self.path.parent
                and (reopened_parent_status.st_dev, reopened_parent_status.st_ino)
                == (self.parent_device, self.parent_inode),
                f"{self.label} parent changed after validation",
            )
        finally:
            os.close(reopened_parent)
        try:
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOENT}:
                raise ProtectedPlanError(f"{self.label} changed after validation") from exc
            raise
        opened_before = os.fstat(self.descriptor)
        _require(
            _stat_fingerprint(named) == self.fingerprint
            and _stat_fingerprint(opened_before) == self.fingerprint,
            f"{self.label} changed after validation",
        )
        raw = _read_exact_descriptor_bytes(
            self.descriptor,
            opened_before.st_size,
            label=self.label,
        )
        opened_after = os.fstat(self.descriptor)
        _require(
            _stat_fingerprint(opened_after) == self.fingerprint and raw == self.raw,
            f"{self.label} changed after validation",
        )

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)


def _open_immutable_bound_file(
    path: Path,
    payload: Mapping[str, object],
    *,
    digest_field: str,
    max_bytes: int,
    label: str,
) -> _OpenedImmutableBoundFile:
    digest = _sha256(payload.get(digest_field), f"{label} digest")
    _require(path.name == f"{digest}.json", f"{label} is not at its content address")
    parent_fd, normalized_parent = _open_directory_without_symlinks(
        path.parent,
        label=f"{label} parent",
    )
    parent_status = os.fstat(parent_fd)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOENT}:
                raise ProtectedPlanError(f"{label} is missing or symlinked") from exc
            raise
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and stat.S_IMODE(opened.st_mode) == 0o444
            and opened.st_nlink == 1,
            f"{label} must be an immutable single-link regular file",
        )
        _require(opened.st_size <= max_bytes, f"{label} exceeds the size limit")
        raw = _read_exact_descriptor_bytes(descriptor, opened.st_size, label=label)
        closed_snapshot = os.fstat(descriptor)
        _require(
            _stat_fingerprint(opened) == _stat_fingerprint(closed_snapshot),
            f"{label} changed while being read",
        )
        expected = canonical_json_bytes(dict(payload))
        _require(raw == expected, f"{label} is not byte-identical to the validated payload")
        _validated_flat_digest(payload, digest_field=digest_field, label=label)
        bound = _OpenedImmutableBoundFile(
            path=normalized_parent / path.name,
            label=label,
            parent_descriptor=parent_fd,
            descriptor=descriptor,
            parent_device=parent_status.st_dev,
            parent_inode=parent_status.st_ino,
            fingerprint=_stat_fingerprint(opened),
            raw=raw,
        )
        bound.require_unchanged()
        return bound
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise


def _read_immutable_bound_file(
    path: Path,
    payload: Mapping[str, object],
    *,
    digest_field: str,
    max_bytes: int,
    label: str,
) -> bytes:
    bound = _open_immutable_bound_file(
        path,
        payload,
        digest_field=digest_field,
        max_bytes=max_bytes,
        label=label,
    )
    try:
        return bound.raw
    finally:
        bound.close()


def _paths_overlap(first: Path, second: Path) -> bool:
    first_real = Path(os.path.realpath(first.absolute()))
    second_real = Path(os.path.realpath(second.absolute()))
    common = Path(os.path.commonpath((first_real.as_posix(), second_real.as_posix())))
    return common == first_real or common == second_real


def _require_disjoint_publication_root(
    publication_root: Path,
    *,
    protected_roots: Sequence[Path],
) -> None:
    for index, root in enumerate(protected_roots):
        descriptor, normalized = _open_directory_without_symlinks(
            root,
            label=f"protected input root {index}",
        )
        os.close(descriptor)
        _require(
            not _paths_overlap(publication_root, normalized),
            "protected plan publication root overlaps a bound input root",
        )


def _link_anonymous_file_no_replace(
    descriptor: int,
    directory_fd: int,
    name: str,
) -> None:
    try:
        linkat = cast(Any, ctypes.CDLL(None, use_errno=True).linkat)
    except AttributeError as exc:  # pragma: no cover - Linux exposes linkat
        raise ProtectedPlanError("platform lacks linkat for atomic plan publication") from exc
    ctypes.set_errno(0)
    result = int(linkat(descriptor, b"", directory_fd, os.fsencode(name), _AT_EMPTY_PATH))
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), name)
    raise OSError(error_number, os.strerror(error_number), name)


def _atomic_install_new_immutable(
    directory_fd: int,
    name: str,
    raw: bytes,
    *,
    pre_link_checks: Sequence[Callable[[], None]] = (),
) -> None:
    _require(
        bool(name) and name not in {".", ".."} and "/" not in name and "\x00" not in name,
        "protected plan file name is invalid",
    )
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_PLAN_BYTES, "plan bytes are invalid")
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _fail("protected plan target already exists; freeze-once publication refuses reuse")
    anonymous_flag = getattr(os, "O_TMPFILE", None)
    _require(type(anonymous_flag) is int, "platform lacks O_TMPFILE for atomic publication")
    flags = os.O_WRONLY | cast(int, anonymous_flag) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(".", flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise ProtectedPlanError("destination cannot create an anonymous plan inode") from exc
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and stat.S_IMODE(opened.st_mode) == 0o600
            and opened.st_nlink == 0,
            "anonymous plan staging inode is invalid",
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing protected plan")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        _require(
            stat.S_ISREG(staged.st_mode)
            and stat.S_IMODE(staged.st_mode) == 0o444
            and staged.st_nlink == 0
            and staged.st_size == len(raw),
            "sealed anonymous plan staging inode is invalid",
        )
        for check in pre_link_checks:
            check()
        try:
            _link_anonymous_file_no_replace(descriptor, directory_fd, name)
        except FileExistsError as exc:
            raise ProtectedPlanError(
                "protected plan target appeared concurrently; freeze-once publication refused"
            ) from exc
        installed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(
            stat.S_ISREG(installed.st_mode)
            and stat.S_IMODE(installed.st_mode) == 0o444
            and installed.st_nlink == 1
            and installed.st_size == len(raw)
            and (installed.st_dev, installed.st_ino) == (staged.st_dev, staged.st_ino),
            "atomically installed protected plan is invalid",
        )
        os.fsync(directory_fd)
    finally:
        os.close(descriptor)


def publish_hidden_regime_factorial_protected_plan(
    payload: Mapping[str, object],
    *,
    threshold_receipt: Mapping[str, object],
    calibration_aggregate: Mapping[str, object],
    publication_root: Path,
    threshold_receipt_path: Path,
    calibration_aggregate_path: Path,
    additional_disjoint_roots: Sequence[Path] = (),
) -> PublishedProtectedPlan:
    """Validate and atomically install one content-addressed, nonauthorizing plan."""

    payload_snapshot = copy.deepcopy(dict(payload))
    receipt_snapshot = copy.deepcopy(dict(threshold_receipt))
    aggregate_snapshot = copy.deepcopy(dict(calibration_aggregate))
    receipt_file: _OpenedImmutableBoundFile | None = None
    aggregate_file: _OpenedImmutableBoundFile | None = None
    directory_fd: int | None = None
    try:
        receipt_file = _open_immutable_bound_file(
            threshold_receipt_path,
            receipt_snapshot,
            digest_field="receipt_payload_sha256",
            max_bytes=_MAX_BOUND_INPUT_BYTES,
            label="threshold-freeze receipt",
        )
        aggregate_file = _open_immutable_bound_file(
            calibration_aggregate_path,
            aggregate_snapshot,
            digest_field="payload_sha256",
            max_bytes=_MAX_BOUND_INPUT_BYTES,
            label="calibration aggregate",
        )
        validated = validate_hidden_regime_factorial_protected_plan(
            payload_snapshot,
            threshold_receipt=receipt_snapshot,
            calibration_aggregate=aggregate_snapshot,
        )
        receipt_binding = _plain_dict(
            validated.get("threshold_freeze_receipt_binding"),
            "threshold-freeze receipt binding",
        )
        calibration_binding = _plain_dict(
            validated.get("calibration_binding"),
            "calibration binding",
        )
        _require(
            receipt_snapshot.get("receipt_payload_sha256")
            == receipt_binding.get("receipt_payload_sha256"),
            "protected plan threshold receipt binding differs",
        )
        _require(
            aggregate_snapshot.get("payload_sha256")
            == calibration_binding.get("calibration_aggregate_payload_sha256"),
            "protected plan calibration aggregate binding differs",
        )
        digest = _sha256(validated.get("payload_sha256"), "protected plan payload")
        raw = canonical_json_bytes(validated)
        _require(len(raw) <= _MAX_PLAN_BYTES, "protected plan exceeds the size limit")
        name = f"{digest}.json"
        directory_fd, root = _open_directory_without_symlinks(
            publication_root,
            label="protected plan publication root",
        )
        _require_disjoint_publication_root(
            root,
            protected_roots=(
                receipt_file.path.parent,
                aggregate_file.path.parent,
                *additional_disjoint_roots,
            ),
        )
        _atomic_install_new_immutable(
            directory_fd,
            name,
            raw,
            pre_link_checks=(
                receipt_file.require_unchanged,
                aggregate_file.require_unchanged,
            ),
        )
        path = root / name
        installed_raw = _read_immutable_bound_file(
            path,
            validated,
            digest_field="payload_sha256",
            max_bytes=_MAX_PLAN_BYTES,
            label="installed protected plan",
        )
        _require(installed_raw == raw, "installed protected plan is not byte-identical")
        return PublishedProtectedPlan(path=path, payload_sha256=digest, payload=validated)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if aggregate_file is not None:
            aggregate_file.close()
        if receipt_file is not None:
            receipt_file.close()


__all__ = [
    "DUMMY_PROTECTED_SEED_PAIRS_SHA256",
    "DUMMY_PROTECTED_SEED_SNAPSHOT_SHA256",
    "DUMMY_THRESHOLD_RECEIPT_SHA256",
    "PROTECTED_MANIFEST_BINDINGS_SHA256",
    "PROTECTED_MANIFEST_ORDER",
    "PROTECTED_MANIFEST_ORDER_SHA256",
    "PROTECTED_MANIFEST_PAYLOAD_SHA256",
    "PROTECTED_PLAN_SCHEMA",
    "PROTECTED_RECURRENCE_BINDINGS_SHA256",
    "PROTECTED_SEED_DERIVATION_SCHEMA",
    "ProtectedPlanError",
    "PublishedProtectedPlan",
    "build_hidden_regime_factorial_protected_plan",
    "protected_plan_path",
    "publish_hidden_regime_factorial_protected_plan",
    "validate_hidden_regime_factorial_protected_plan",
]
