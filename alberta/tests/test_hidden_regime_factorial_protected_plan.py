from __future__ import annotations

import ast
import copy
import os
import stat
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation import hidden_regime_factorial_protected_plan as plan
from alberta_framework.evaluation import hidden_regime_factorial_thresholds as threshold_engine
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CANONICAL_CONDITION_ORDER,
    FROZEN_SEED_PAIRS,
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
    THRESHOLD_FREEZE_DECISION_REJECTION,
)

_TEST_ENDPOINT_IDENTITIES = threshold_engine._frozen_mandatory_statistical_endpoint_identities(
    build_hidden_regime_factorial_calibration_design()
)


def _sha(label: str) -> str:
    return canonical_sha256({"test-only": label})


def _with_receipt_digest(body: dict[str, object]) -> dict[str, object]:
    return {**body, "receipt_payload_sha256": canonical_sha256(body)}


def _with_payload_digest(body: dict[str, object]) -> dict[str, object]:
    return {**body, "payload_sha256": canonical_sha256(body)}


def _rehash_receipt(receipt: dict[str, object]) -> None:
    body = dict(receipt)
    body.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = canonical_sha256(body)


def _frozen_threshold(index: int) -> dict[str, object]:
    identity = copy.deepcopy(_TEST_ENDPOINT_IDENTITIES[index])
    return {
        "endpoint_id": canonical_sha256(identity),
        "gate_family_id": identity["gate_family_id"],
        "reference": identity["reference"],
        "orientation": "higher",
        "threshold_space": "oriented_higher_is_favorable",
        "oriented_null_hex": "0x0.0p+0",
        "conservative_oriented_bound_hex": "0x1.0p-1",
        "continuous_margin_quantum_decimal": "0.2500",
        "oriented_continuous_threshold_decimal": "0.2500",
        "pooled_win_threshold": 17,
        "manifest_win_thresholds": [
            {"manifest_name": name, "win_threshold": 6}
            for name in (
                "hidden-regime-calibration-a-v1",
                "hidden-regime-calibration-b-v1",
                "hidden-regime-calibration-c-v1",
            )
        ],
        "missingness_threshold": 0,
        "ties_count_as_wins": False,
    }


def _successful_receipt(
    *,
    calibration_outcomes_payload_sha256: str | None = None,
) -> dict[str, object]:
    frozen = [_frozen_threshold(index) for index in range(MANDATORY_STATISTICAL_ENDPOINT_COUNT)]
    aggregate_digest = calibration_outcomes_payload_sha256 or _sha("aggregate")
    return _with_receipt_digest(
        {
            "receipt_schema": THRESHOLD_FREEZE_RECEIPT_SCHEMA,
            "decision_status": THRESHOLD_FREEZE_DECISION_FROZEN,
            "development_only": True,
            "claim_accepted": False,
            "thresholds_frozen": True,
            "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
            "readiness_receipt_sha256": _sha("readiness"),
            "gate_matrix_sha256": _sha("gate-matrix"),
            "calibration_outcomes_payload_sha256": aggregate_digest,
            "source_closure_sha256": _sha("source-closure"),
            "source_archive_sha256": _sha("source-archive"),
            "environment_identity_sha256": _sha("environment"),
            "managed_ledger_snapshot_sha256": _sha("ledger-snapshot"),
            "managed_ledger_content_address": _sha("ledger-content"),
            "execution_governance_genesis_sha256": _sha("genesis"),
            "case_ledger_sha256": _sha("case-ledger"),
            "aggregation_readiness_certification_binding_sha256": _sha("aggregation-certification"),
            "mandatory_audit_summary_sha256": _sha("mandatory-audit"),
            "mandatory_audit_decision": "passed_nonstatistical",
            "mandatory_statistical_endpoint_count": MANDATORY_STATISTICAL_ENDPOINT_COUNT,
            "mandatory_statistical_endpoint_identities_sha256": (
                MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
            ),
            "mandatory_statistical_endpoint_ids_sha256": (
                MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
            ),
            "frozen_thresholds": frozen,
            "mandatory_gate_results": [],
            "descriptive_only_results": [],
            "rejection_reasons": [],
            "rounding_worked_examples": [],
            "all_calibration_seeds_consumed": True,
            "calibration_case_count_consumed": 240,
            "calibration_seed_pair_count_consumed": 30,
            "protected_namespace_derived": False,
            "protected_outcomes_observed": False,
            "scientific_promotion_allowed": False,
            "amendments_allowed": False,
        }
    )


def _rejection_receipt(
    *,
    calibration_outcomes_payload_sha256: str | None = None,
) -> dict[str, object]:
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=calibration_outcomes_payload_sha256
    )
    body = dict(receipt)
    body.pop("receipt_payload_sha256")
    body.update(
        {
            "decision_status": THRESHOLD_FREEZE_DECISION_REJECTION,
            "thresholds_frozen": False,
            "frozen_thresholds": [],
            "rejection_reasons": [{"reasons": ["test-only-valid-rejection"]}],
        }
    )
    return _with_receipt_digest(body)


def _aggregate() -> dict[str, object]:
    return _with_payload_digest(
        {
            "schema": CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "test_only": True,
        }
    )


@pytest.fixture
def threshold_validator_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(
        payload: object,
        *,
        calibration_aggregate: object,
    ) -> dict[str, object]:
        assert calibration_aggregate is not None
        assert type(payload) is dict
        return copy.deepcopy(cast(dict[str, object], payload))

    monkeypatch.setattr(
        plan,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        validate,
    )


def test_dummy_seed_derivation_fixed_vector_and_global_disjointness() -> None:
    validated = plan._ValidatedSuccessfulThresholdReceipt(
        payload={},
        receipt_payload_sha256=plan.DUMMY_THRESHOLD_RECEIPT_SHA256,
    )
    snapshot = plan._derive_protected_seed_snapshot(validated)
    pairs = cast(list[dict[str, int]], snapshot["pairs"])
    assert pairs[0] == {
        "index": 0,
        "world_seed": 2319904712,
        "world_derivation_counter": 0,
        "learner_seed": 594023118,
        "learner_derivation_counter": 0,
    }
    assert pairs[-1] == {
        "index": 29,
        "world_seed": 332534799,
        "world_derivation_counter": 0,
        "learner_seed": 3853836588,
        "learner_derivation_counter": 0,
    }
    assert canonical_sha256(pairs) == plan.DUMMY_PROTECTED_SEED_PAIRS_SHA256
    assert canonical_sha256(snapshot) == plan.DUMMY_PROTECTED_SEED_SNAPSHOT_SHA256
    protected = [value for pair in pairs for value in (pair["world_seed"], pair["learner_seed"])]
    calibration = {
        value for pair in FROZEN_SEED_PAIRS for value in (pair.world_seed, pair.learner_seed)
    }
    assert len(protected) == len(set(protected)) == 60
    assert set(protected).isdisjoint(calibration)


def test_seed_derivation_uses_deterministic_collision_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = plan._ValidatedSuccessfulThresholdReceipt(
        payload={},
        receipt_payload_sha256=plan.DUMMY_THRESHOLD_RECEIPT_SHA256,
    )
    original = plan._protected_seed_candidate
    excluded = FROZEN_SEED_PAIRS[0].world_seed
    accepted_world = original(
        plan.DUMMY_THRESHOLD_RECEIPT_SHA256,
        0,
        "world",
        1,
    )

    def candidate(receipt_sha256: str, index: int, lane: str, counter: int) -> int:
        if index == 0 and lane == "world" and counter == 0:
            return excluded
        if index == 0 and lane == "learner" and counter == 0:
            return accepted_world
        return original(receipt_sha256, index, cast(plan.SeedLane, lane), counter)

    monkeypatch.setattr(plan, "_protected_seed_candidate", candidate)
    snapshot = plan._derive_protected_seed_snapshot(validated)
    assert plan._derive_protected_seed_snapshot(validated) == snapshot
    first = cast(list[dict[str, int]], snapshot["pairs"])[0]
    assert first["world_derivation_counter"] == 1
    assert first["world_seed"] != excluded
    assert first["learner_derivation_counter"] == 1
    assert first["learner_seed"] != first["world_seed"]
    assert plan._seed_disjointness_proof(snapshot)["globally_unique_and_disjoint"] is True

    tampered = copy.deepcopy(snapshot)
    tampered_pairs = cast(list[dict[str, int]], tampered["pairs"])
    tampered_pairs[1]["world_seed"] = tampered_pairs[0]["learner_seed"]
    with pytest.raises(plan.ProtectedPlanError, match="exact collision-rejecting derivation"):
        plan._seed_disjointness_proof(tampered)


def test_structural_manifests_and_recurrence_bindings_are_exact() -> None:
    bindings = plan._protected_manifest_bindings()
    assert [item["name"] for item in bindings] == list(plan.PROTECTED_MANIFEST_ORDER)
    assert [item["manifest_payload_sha256"] for item in bindings] == list(
        plan.PROTECTED_MANIFEST_PAYLOAD_SHA256
    )
    assert canonical_sha256(bindings) == plan.PROTECTED_MANIFEST_BINDINGS_SHA256

    recurrence = plan._protected_recurrence_bindings()
    assert canonical_sha256(recurrence) == plan.PROTECTED_RECURRENCE_BINDINGS_SHA256
    assert all(item["eligible_recurrence_count"] == 12 for item in recurrence)
    assert all(
        item["eligible_recurrence_counts_by_regime"] == [5, 4, 1, 2, 0] for item in recurrence
    )
    assert recurrence[0]["eligible_recurrence_identities"] == [
        [3, 0, 1],
        [4, 1, 1],
        [5, 2, 1],
        [6, 0, 2],
        [7, 1, 2],
        [9, 0, 3],
        [10, 3, 1],
        [11, 1, 3],
        [12, 0, 4],
        [14, 1, 4],
        [15, 3, 2],
        [16, 0, 5],
    ]


def test_plan_is_exactly_unexecuted_balanced_and_strictly_recomputable(
    threshold_validator_stub: None,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    assert payload["schema"] == plan.PROTECTED_PLAN_SCHEMA
    assert payload["plan_status"] == "preregistered_unexecuted"
    assert payload["protected_namespace_derived"] is True
    for field in (
        "protected_outcomes_observed",
        "learner_outcomes_executed",
        "learner_execution_authorized",
        "protected_execution_permitted",
        "execution_issuer_available",
        "scientific_promotion_allowed",
        "automatic_promotion_allowed",
        "amendments_allowed",
        "threshold_adjustment_permitted",
    ):
        assert payload[field] is False
    assert payload["protected_readiness_receipt_sha256"] is None
    assert payload["protected_execution_ledger_genesis_sha256"] is None
    assert payload["seed_pair_count"] == 30
    assert payload["condition_count"] == 8
    assert payload["matched_case_count"] == 240
    assert payload["manifest_seed_pair_counts"] == [
        {"manifest_name": name, "count": 10} for name in plan.PROTECTED_MANIFEST_ORDER
    ]
    assert payload["manifest_case_counts"] == [
        {"manifest_name": name, "count": 80} for name in plan.PROTECTED_MANIFEST_ORDER
    ]
    assert payload["condition_case_counts"] == [
        {"condition": condition, "count": 30} for condition in CANONICAL_CONDITION_ORDER
    ]
    cases = cast(list[dict[str, object]], payload["cases"])
    assert [case["case_index"] for case in cases] == list(range(240))
    assert all(
        case["case_index"]
        == cast(int, case["seed_index"]) * 8
        + CANONICAL_CONDITION_ORDER.index(cast(str, case["condition"]))
        for case in cases
    )
    assert (
        plan.validate_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
        )
        == payload
    )

    tampered = copy.deepcopy(payload)
    cast(list[dict[str, object]], tampered["cases"])[0]["world_seed"] = 0
    tampered_body = dict(tampered)
    tampered_body.pop("payload_sha256")
    tampered["payload_sha256"] = canonical_sha256(tampered_body)
    with pytest.raises(plan.ProtectedPlanError, match="digest|exact recomputation"):
        plan.validate_hidden_regime_factorial_protected_plan(
            tampered,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
        )


def test_successful_plan_transitively_binds_exact_endpoint_and_calibration_identities(
    threshold_validator_stub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    binding = cast(dict[str, object], payload["threshold_freeze_receipt_binding"])
    assert binding["mandatory_statistical_endpoint_identities_sha256"] == (
        MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
    )
    assert binding["mandatory_statistical_endpoint_ids_sha256"] == (
        MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
    )
    calibration_binding = cast(dict[str, object], payload["calibration_binding"])
    assert calibration_binding["calibration_managed_ledger_snapshot_sha256"] == receipt[
        "managed_ledger_snapshot_sha256"
    ]
    assert calibration_binding["calibration_managed_ledger_content_address"] == receipt[
        "managed_ledger_content_address"
    ]
    assert calibration_binding["calibration_execution_governance_genesis_sha256"] == receipt[
        "execution_governance_genesis_sha256"
    ]

    seed_derivation_called = False

    def forbidden(_: object) -> dict[str, object]:
        nonlocal seed_derivation_called
        seed_derivation_called = True
        raise AssertionError("seed derivation preceded exact endpoint validation")

    monkeypatch.setattr(plan, "_derive_protected_seed_snapshot", forbidden)
    hostile_receipts: list[dict[str, object]] = []
    for field in (
        "mandatory_statistical_endpoint_identities_sha256",
        "mandatory_statistical_endpoint_ids_sha256",
    ):
        hostile = copy.deepcopy(receipt)
        hostile[field] = _sha(f"hostile-{field}")
        _rehash_receipt(hostile)
        hostile_receipts.append(hostile)
    reordered = copy.deepcopy(receipt)
    reordered_thresholds = cast(list[object], reordered["frozen_thresholds"])
    reordered_thresholds[0], reordered_thresholds[1] = (
        reordered_thresholds[1],
        reordered_thresholds[0],
    )
    _rehash_receipt(reordered)
    hostile_receipts.append(reordered)

    for hostile in hostile_receipts:
        with pytest.raises(plan.ProtectedPlanError, match="endpoint|receipt"):
            plan.build_hidden_regime_factorial_protected_plan(
                hostile,
                calibration_aggregate=aggregate,
            )
    assert seed_derivation_called is False


def test_assignment_builder_rejects_incomplete_or_duplicate_inputs() -> None:
    validated = plan._ValidatedSuccessfulThresholdReceipt(
        payload={},
        receipt_payload_sha256=plan.DUMMY_THRESHOLD_RECEIPT_SHA256,
    )
    snapshot = plan._derive_protected_seed_snapshot(validated)
    manifests = plan._protected_manifest_bindings()
    recurrences = plan._protected_recurrence_bindings()
    contract = plan._evaluation_contract()
    assignments, cases = plan._assignments_and_cases(
        snapshot,
        manifests,
        recurrences,
        contract,
    )
    assert len(assignments) == 30
    assert len(cases) == 240
    assert len(
        {(case["seed_index"], case["condition"]) for case in cases}
    ) == 240

    duplicate_manifests = copy.deepcopy(manifests)
    duplicate_manifests[1] = copy.deepcopy(duplicate_manifests[0])
    with pytest.raises(plan.ProtectedPlanError, match="manifest bindings"):
        plan._assignments_and_cases(
            snapshot,
            duplicate_manifests,
            recurrences,
            contract,
        )

    incomplete_contract = copy.deepcopy(contract)
    cast(list[object], incomplete_contract["condition_runtime_bindings"]).pop()
    with pytest.raises(plan.ProtectedPlanError, match="evaluation contract"):
        plan._assignments_and_cases(
            snapshot,
            manifests,
            recurrences,
            incomplete_contract,
        )

    incomplete_snapshot = copy.deepcopy(snapshot)
    cast(list[object], incomplete_snapshot["pairs"]).pop()
    with pytest.raises(plan.ProtectedPlanError, match="exact collision-rejecting derivation"):
        plan._assignments_and_cases(
            incomplete_snapshot,
            manifests,
            recurrences,
            contract,
        )


def test_valid_rejection_never_reaches_protected_seed_derivation(
    threshold_validator_stub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(_: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("protected seeds were derived from a rejection")

    monkeypatch.setattr(plan, "_derive_protected_seed_snapshot", forbidden)
    aggregate = _aggregate()
    with pytest.raises(plan.ProtectedPlanError, match="successful threshold-freeze receipt"):
        plan.build_hidden_regime_factorial_protected_plan(
            _rejection_receipt(
                calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
            ),
            calibration_aggregate=aggregate,
        )
    assert called is False


def _install_input(root: Path, payload: dict[str, object], digest_field: str) -> Path:
    root.mkdir()
    digest = cast(str, payload[digest_field])
    path = root / f"{digest}.json"
    path.write_bytes(canonical_json_bytes(payload))
    path.chmod(0o444)
    return path


def test_content_addressed_publisher_is_immutable_and_strictly_freeze_once(
    tmp_path: Path,
    threshold_validator_stub: None,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    receipt_path = _install_input(tmp_path / "receipts", receipt, "receipt_payload_sha256")
    aggregate_path = _install_input(tmp_path / "aggregates", aggregate, "payload_sha256")
    publication_root = tmp_path / "plans"
    publication_root.mkdir()

    published = plan.publish_hidden_regime_factorial_protected_plan(
        payload,
        threshold_receipt=receipt,
        calibration_aggregate=aggregate,
        publication_root=publication_root,
        threshold_receipt_path=receipt_path,
        calibration_aggregate_path=aggregate_path,
    )
    status = published.path.stat()
    assert stat.S_IMODE(status.st_mode) == 0o444
    assert status.st_nlink == 1
    assert published.path.name == f"{payload['payload_sha256']}.json"
    assert published.path.read_bytes() == canonical_json_bytes(payload)

    with pytest.raises(plan.ProtectedPlanError, match="already exists|freeze-once"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )
    assert published.path.stat().st_ino == status.st_ino

    receipt_hardlink = receipt_path.parent / "unexpected-hardlink.json"
    os.link(receipt_path, receipt_hardlink)
    second_publication_root = tmp_path / "plans-with-linked-input"
    second_publication_root.mkdir()
    with pytest.raises(plan.ProtectedPlanError, match="single-link"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=second_publication_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )


def test_publisher_binds_authoritative_inputs_before_any_seed_derivation(
    tmp_path: Path,
    threshold_validator_stub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    receipt_path = receipt_root / f"{receipt['receipt_payload_sha256']}.json"
    aggregate_path = _install_input(tmp_path / "aggregates", aggregate, "payload_sha256")
    publication_root = tmp_path / "plans"
    publication_root.mkdir()
    derived = False

    def forbidden(_: object) -> dict[str, object]:
        nonlocal derived
        derived = True
        raise AssertionError("protected seeds derived before authoritative inputs were bound")

    monkeypatch.setattr(plan, "_derive_protected_seed_snapshot", forbidden)
    with pytest.raises(plan.ProtectedPlanError, match="missing|symlinked"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )
    assert derived is False

    receipt_path.write_bytes(b"not-the-bound-receipt")
    receipt_path.chmod(0o444)
    with pytest.raises(plan.ProtectedPlanError, match="byte-identical"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )
    assert derived is False

    real_receipt_path = _install_input(
        tmp_path / "real-receipt",
        receipt,
        "receipt_payload_sha256",
    )
    symlink_receipt_root = tmp_path / "symlink-receipt"
    symlink_receipt_root.mkdir()
    symlink_receipt_path = symlink_receipt_root / real_receipt_path.name
    symlink_receipt_path.symlink_to(real_receipt_path)
    with pytest.raises(plan.ProtectedPlanError, match="missing|symlinked"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=symlink_receipt_path,
            calibration_aggregate_path=aggregate_path,
        )
    assert derived is False

    wrong_address = real_receipt_path.parent / f"{'0' * 64}.json"
    wrong_address.write_bytes(real_receipt_path.read_bytes())
    wrong_address.chmod(0o444)
    with pytest.raises(plan.ProtectedPlanError, match="content address"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=wrong_address,
            calibration_aggregate_path=aggregate_path,
        )
    assert derived is False


def test_publisher_holds_and_revalidates_bound_input_inode_before_link(
    tmp_path: Path,
    threshold_validator_stub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    receipt_path = _install_input(tmp_path / "receipts", receipt, "receipt_payload_sha256")
    aggregate_path = _install_input(tmp_path / "aggregates", aggregate, "payload_sha256")
    publication_root = tmp_path / "plans"
    publication_root.mkdir()
    receipt_raw = receipt_path.read_bytes()
    original_install = plan._atomic_install_new_immutable
    attacked = False

    def replace_checked_name_then_install(
        directory_fd: int,
        name: str,
        raw: bytes,
        *,
        pre_link_checks: Sequence[Callable[[], None]] = (),
    ) -> None:
        nonlocal attacked
        attacked = True
        detached = receipt_path.parent / "detached-original.json"
        receipt_path.rename(detached)
        receipt_path.write_bytes(receipt_raw)
        receipt_path.chmod(0o444)
        original_install(
            directory_fd,
            name,
            raw,
            pre_link_checks=pre_link_checks,
        )

    monkeypatch.setattr(plan, "_atomic_install_new_immutable", replace_checked_name_then_install)
    with pytest.raises(plan.ProtectedPlanError, match="changed after validation"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=publication_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )
    assert attacked is True
    assert list(publication_root.iterdir()) == []


def test_publisher_rejects_overlap_symlinks_traversal_and_nonidentical_existing_file(
    tmp_path: Path,
    threshold_validator_stub: None,
) -> None:
    aggregate = _aggregate()
    receipt = _successful_receipt(
        calibration_outcomes_payload_sha256=cast(str, aggregate["payload_sha256"])
    )
    payload = plan.build_hidden_regime_factorial_protected_plan(
        receipt,
        calibration_aggregate=aggregate,
    )
    receipt_root = tmp_path / "receipts"
    aggregate_root = tmp_path / "aggregates"
    receipt_path = _install_input(receipt_root, receipt, "receipt_payload_sha256")
    aggregate_path = _install_input(aggregate_root, aggregate, "payload_sha256")

    with pytest.raises(plan.ProtectedPlanError, match="overlaps"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=receipt_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )

    real_root = tmp_path / "real-plans"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-plans"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(plan.ProtectedPlanError, match="symlink"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=symlink_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )

    with pytest.raises(plan.ProtectedPlanError, match="digest"):
        plan.protected_plan_path(real_root, "../escape")

    target = plan.protected_plan_path(real_root, cast(str, payload["payload_sha256"]))
    target.write_bytes(b"not-the-plan")
    target.chmod(0o444)
    with pytest.raises(plan.ProtectedPlanError, match="already exists|freeze-once"):
        plan.publish_hidden_regime_factorial_protected_plan(
            payload,
            threshold_receipt=receipt,
            calibration_aggregate=aggregate,
            publication_root=real_root,
            threshold_receipt_path=receipt_path,
            calibration_aggregate_path=aggregate_path,
        )


def test_module_has_no_runner_issuer_or_import_time_official_plan_surface() -> None:
    source_path = Path(plan.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_imports = {
        "subprocess",
        "jax",
        "random",
        "secrets",
        "alberta_framework.evaluation.hidden_regime_factorial_calibration",
        "alberta_framework.evaluation.hidden_regime_calibration_readiness",
        "alberta_framework.evaluation.hidden_regime_execution_governance",
        "alberta_framework.evaluation.hidden_regime_signaling_development",
    }
    assert imported_modules.isdisjoint(forbidden_imports)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        token in name
        for name in function_names
        for token in ("run_learner", "run_world", "execute_case", "issue_authorization")
    )
    forbidden_module_calls = {
        "_derive_protected_seed_snapshot",
        "_protected_seed_candidate",
        "build_hidden_regime_factorial_protected_plan",
        "publish_hidden_regime_factorial_protected_plan",
    }
    module_call_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                module_call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                module_call_names.add(node.func.attr)
    assert module_call_names.isdisjoint(forbidden_module_calls)
    assert "if __name__" not in source
    assert not hasattr(plan, "PROTECTED_SEED_PAIRS")
    assert not hasattr(plan, "OFFICIAL_PROTECTED_PLAN")
