"""Focused tests for the inert matched-v3 algorithmic-resource measurement ledger."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_algorithmic_resource_contract as resource_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_algorithmic_resource_measurement as measurement,
)

pytestmark = pytest.mark.unit

_EXPECTED_DESCRIPTOR_SHA256 = "627eba09823a2f914df7957d1fd441907116a5e11a88ebef03bd92de3e3fb950"
_TEST_MEASUREMENT_SOURCE_SHA256 = hashlib.sha256(
    b"externally audited measurement source fixture"
).hexdigest()
_UPSTREAM_CONTRACT_DESCRIPTOR_SHA256 = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)
_UPSTREAM_CONTRACT_SOURCE_SHA256 = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)
_ABSENCE_KIND_BY_SUBJECT = {
    "optimizer_updates": "optimizer_subsystem_absent",
    "gradient_updates": "gradient_update_path_absent",
    "sample_updates": "sample_update_path_absent",
    "trainable_parameters": "trainable_parameter_tree_absent",
    "frozen_parameters": "frozen_parameter_tree_absent",
    "optimizer_state": "optimizer_state_tree_absent",
    "target_copy": "target_copy_tree_absent",
    "replay_storage": "replay_subsystem_absent",
    "rollout_storage": "rollout_storage_absent",
    "recurrent_carry": "recurrent_carry_absent",
    "rtrl_sensitivity": "rtrl_sensitivity_absent",
    "eligibility_trace": "eligibility_trace_absent",
}
_CATEGORY_SUBJECTS = frozenset(_ABSENCE_KIND_BY_SUBJECT) - {
    "optimizer_updates",
    "gradient_updates",
    "sample_updates",
}


def _canonical(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _leaf(
    path: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    category: measurement.ArrayCategory,
    lifecycle: str,
    storage_id: str,
    alias_id: str | None = None,
    dtype: str = "float32",
    owner: str = "agent_state",
    storage_kind: measurement.StorageKind = "owned_array",
) -> measurement.ArrayLeafObservationV1:
    return measurement.ArrayLeafObservationV1(
        leaf_path=path,
        shape=shape,
        dtype=dtype,
        owner=owner,
        category=category,
        lifecycle=lifecycle,
        alias_id=storage_id if alias_id is None else alias_id,
        storage_id=storage_id,
        storage_kind=storage_kind,
    )


def _commit_environment(
    ledger: measurement.AlgorithmicResourceMeasurementLedger,
    count: int = 1,
) -> None:
    for index in range(count):
        ledger.begin_environment_transaction(f"environment_{index:06d}").commit()


def _ledger(ledger_id: str) -> measurement.AlgorithmicResourceMeasurementLedger:
    return measurement.AlgorithmicResourceMeasurementLedger(
        ledger_id,
        measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
    )


def _absence(subject: str) -> measurement.StructuralAbsenceProofV1:
    return measurement.StructuralAbsenceProofV1(
        subject=subject,
        absence_kind=_ABSENCE_KIND_BY_SUBJECT[subject],
        proof_id=f"absence_{subject}",
        evidence_kind="exact_runtime_subsystem_inventory",
        details={
            "implementation_inventory": {
                "subject": subject,
                "observed": True,
            }
        },
    )


def _absence_with_details(
    subject: str,
    details: dict[str, Any],
) -> measurement.StructuralAbsenceProofV1:
    return measurement.StructuralAbsenceProofV1(
        subject=subject,
        absence_kind=_ABSENCE_KIND_BY_SUBJECT[subject],
        proof_id=f"absence_{subject}",
        evidence_kind="exact_runtime_subsystem_inventory",
        details=details,
    )


def _nested_details(depth: int) -> dict[str, Any]:
    value: Any = True
    for _ in range(depth):
        value = {"node": value}
    assert isinstance(value, dict)
    return value


def _aggregate_heavy_absences() -> tuple[measurement.StructuralAbsenceProofV1, ...]:
    return tuple(
        _absence_with_details(subject, {"inventory": list(range(3_400))})
        for subject in _ABSENCE_KIND_BY_SUBJECT
    )


def _finish(
    ledger: measurement.AlgorithmicResourceMeasurementLedger,
    *,
    present_categories: frozenset[str] = frozenset(),
    positive_counters: frozenset[str] = frozenset(),
) -> measurement.AlgorithmicResourceMeasurementBasisV1:
    for subject in _ABSENCE_KIND_BY_SUBJECT:
        if subject in present_categories or subject in positive_counters:
            continue
        ledger.declare_structural_absence(_absence(subject))
    return ledger.finalize()


def _field_map(
    basis: measurement.AlgorithmicResourceMeasurementBasisV1,
) -> dict[str, dict[str, Any]]:
    return {item["field_name"]: item for item in basis.to_dict()["fields"]}


def _reseal_basis_dict(value: dict[str, Any]) -> bytes:
    fields = value["fields"]
    value["field_inventory_sha256"] = hashlib.sha256(
        _canonical({"fields": fields}, newline=False)
    ).hexdigest()
    body = copy.deepcopy(value)
    body.pop("measurement_basis_body_sha256", None)
    value["measurement_basis_body_sha256"] = hashlib.sha256(
        _canonical(body, newline=False)
    ).hexdigest()
    return _canonical(value)


def _positive_trainable_basis(
    ledger_id: str,
) -> measurement.AlgorithmicResourceMeasurementBasisV1:
    ledger = _ledger(ledger_id)
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id=f"{ledger_id}_snapshot",
            category="trainable_parameters",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("state", "params"),
                    (1,),
                    category="trainable_parameters",
                    lifecycle="training_live",
                    storage_id=f"{ledger_id}_storage",
                ),
            ),
        )
    )
    return _finish(ledger, present_categories=frozenset({"trainable_parameters"}))


def test_descriptor_freezes_inert_logical_accounting_without_self_pins() -> None:
    descriptor = measurement.algorithmic_resource_measurement_ledger_descriptor()

    assert frozenset(descriptor) == {
        "algorithmic_resource_fields",
        "authority",
        "byte_policy",
        "capabilities",
        "category_inventory",
        "claims",
        "classification",
        "coupled_field_pairs",
        "dtype_itemsize_bytes",
        "field_policy",
        "forbidden_field_tokens",
        "lifecycle_policy",
        "limits",
        "readiness",
        "replay_capacity_policy",
        "rtu_path_policy",
        "sample_contribution_policy",
        "schema_version",
        "self_identity_policy",
        "status",
        "structural_absence_policy",
        "transaction_policy",
        "tree_role_policy",
        "upstream_algorithmic_resource_contract_identity",
    }
    assert descriptor["schema_version"] == measurement.MEASUREMENT_LEDGER_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor["status"] == measurement.MEASUREMENT_LEDGER_STATUS
    assert descriptor["classification"] == measurement.MEASUREMENT_CLASSIFICATION
    assert measurement.PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256 == _EXPECTED_DESCRIPTOR_SHA256
    assert "pins" not in descriptor
    assert descriptor["self_identity_policy"] == {
        "descriptor_bytes_embed_own_descriptor_sha256": False,
        "descriptor_bytes_embed_own_source_sha256": False,
        "descriptor_repository_literal_external_to_descriptor": True,
        "parser_reauthenticates_externally_supplied_source_sha256": True,
        "source_sha256_supplied_by_upstream_audited_identity": True,
    }
    assert descriptor["upstream_algorithmic_resource_contract_identity"] == {
        "schema_version": (
            "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
        ),
        "descriptor_sha256": _UPSTREAM_CONTRACT_DESCRIPTOR_SHA256,
        "source_sha256": _UPSTREAM_CONTRACT_SOURCE_SHA256,
    }
    assert descriptor["tree_role_policy"] == {
        "target_copy_role": "target_only_not_frozen",
        "stop_gradient_global_trainable_role": "trainable_parameters",
        "deduplication_scope": "within_one_category_snapshot_by_storage_id",
        "cross_category_deduplication": False,
        "aliases_recorded": True,
        "view_policy": "view_requires_owned_base_in_same_category_snapshot",
    }
    assert descriptor["lifecycle_policy"] == {
        "complete_simultaneous_live_snapshot_required": True,
        "element_and_byte_high_water_retained_independently": True,
        "every_allocation_lifecycle_boundary_required": True,
        "one_snapshot_contains_one_semantic_category": True,
        "unobserved_lifecycle_inference_allowed": False,
    }
    assert descriptor["replay_capacity_policy"] == {
        "capacity_quantity": "simultaneously_live_addressable_transitions",
        "multiple_live_subsystems_summed": True,
        "replay_array_storage_measured_separately": True,
    }
    assert descriptor["rtu_path_policy"] == {
        "carry_components": ["real", "imaginary"],
        "carry_root": "hstate",
        "eligibility_components": [
            "bias_eligibility_trace",
            "eligibility_trace",
            "eligibility_traces",
            "eligibility_tree",
        ],
        "eligibility_requires_explicit_tree": True,
        "local_carry_roots": ["actor_rtu_state", "critic_rtu_state"],
        "local_eligibility_roots": ["actor_traces", "critic_traces"],
        "local_sensitivity_roots": [
            "actor_sensitivities",
            "actor_taylor_trace",
            "critic_sensitivities",
            "critic_taylor_trace",
        ],
        "sensitivity_markers": ["memory_grad", "sensitivities", "taylor_trace"],
    }
    assert descriptor["structural_absence_policy"] == {
        "configuration_only_zero_allowed": False,
        "coupled_fields_share_subject_proof": True,
        "exact_runtime_subsystem_inventory_required": True,
        "observed_zero_sized_tree_establishes_absence": False,
    }
    assert descriptor["transaction_policy"] == {
        "aborted_transaction_counted": False,
        "commit_required": True,
        "committed_learning_zero_iff_consumed_samples_zero": True,
        "consumed_samples_at_least_committed_learning_transactions": True,
        "duplicate_transaction_id_allowed": False,
        "environment_transaction_increment": 1,
        "handle_fields_are_non_authoritative": True,
        "ledger_owned_frozen_pending_record_controls_accounting": True,
        "open_transaction_at_finalization_allowed": False,
    }
    assert descriptor["byte_policy"] == {
        "quantity": "logical_owned_array_bytes",
        "array_payload_bytes_included": True,
        "physical_allocator_bytes_included": False,
        "python_object_header_bytes_included": False,
        "device_allocator_fragmentation_included": False,
        "array_view_payload_double_counted": False,
    }
    assert descriptor["sample_contribution_policy"] == {
        "definition": "data_items_that_affect_learning_or_adaptation",
        "redo_replay_evaluation_samples_included": True,
        "pt_inner_loop_samples_included": True,
        "drqn_burn_in_samples_included": True,
        "separate_actor_critic_transactions_count_separately": True,
        "separate_actor_critic_condition": (
            "separate_learning_transactions_each_consume_the_transition"
        ),
        "ppo_joint_loss_sample_count": "once_per_sample_per_epoch",
    }
    assert descriptor["forbidden_field_tokens"] == [
        "reward",
        "score",
        "return",
        "rank",
        "outcome",
    ]
    assert descriptor["capabilities"] == {
        "execution_performed": False,
        "family_producer_available": False,
        "production_receipt_available": False,
        "runner_invoked": False,
        "runtime_qualified": False,
    }
    assert descriptor["authority"] == {
        "execution_authorized": False,
        "issuance_performed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
    }
    assert descriptor["claims"] == {
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
    }
    assert descriptor["readiness"] == {
        "descriptor_audited": False,
        "producer_identity_pinned": False,
        "qualification_ready": False,
        "source_audited": False,
    }
    assert descriptor["algorithmic_resource_fields"] == list(
        resource_contract.ALGORITHMIC_RESOURCE_FIELDS
    )
    assert descriptor["category_inventory"] == list(measurement.ARRAY_CATEGORIES)
    assert descriptor["coupled_field_pairs"] == [
        list(pair) for pair in measurement.COUPLED_FIELD_PAIRS
    ]
    assert descriptor["field_policy"] == [
        item.to_dict() for item in resource_contract.matched_v3_algorithmic_resource_field_policy()
    ]
    assert descriptor["dtype_itemsize_bytes"] == {
        "bfloat16": 2,
        "bool": 1,
        "complex128": 16,
        "complex64": 8,
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "int8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
        "uint8": 1,
    }
    assert descriptor["limits"] == {
        "maximum_array_dimension": 2**31 - 1,
        "maximum_array_rank": measurement.MAX_ARRAY_RANK,
        "maximum_basis_bytes": measurement.MAX_MEASUREMENT_BASIS_BYTES,
        "maximum_integer": 2**63 - 1,
        "maximum_json_depth": measurement.MAX_JSON_DEPTH,
        "maximum_json_nodes": measurement.MAX_JSON_NODES,
        "maximum_leaves_per_snapshot": 16_384,
        "maximum_text_length": measurement.MAX_TEXT_LENGTH,
    }


def test_descriptor_bytes_are_canonical_and_match_the_audit_pin() -> None:
    raw = measurement.canonical_algorithmic_resource_measurement_ledger_descriptor_bytes()
    descriptor = measurement.algorithmic_resource_measurement_ledger_descriptor()

    assert raw == _canonical(descriptor)
    assert measurement.algorithmic_resource_measurement_ledger_descriptor_sha256() == (
        hashlib.sha256(raw).hexdigest()
    )
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256


def test_public_surface_is_exact_complete_and_unique() -> None:
    expected = [
        "ABSENCE_KIND_BY_SUBJECT",
        "ARRAY_CATEGORIES",
        "COUPLED_FIELD_PAIRS",
        "MAX_ARRAY_DIMENSION",
        "MAX_ARRAY_RANK",
        "MAX_INTEGER",
        "MAX_JSON_DEPTH",
        "MAX_JSON_NODES",
        "MAX_LEAVES_PER_SNAPSHOT",
        "MAX_MEASUREMENT_BASIS_BYTES",
        "MAX_TEXT_LENGTH",
        "MEASUREMENT_BASIS_SCHEMA_VERSION",
        "MEASUREMENT_BASIS_STATUS",
        "MEASUREMENT_CLASSIFICATION",
        "MEASUREMENT_LEDGER_DESCRIPTOR_SCHEMA_VERSION",
        "MEASUREMENT_LEDGER_STATUS",
        "PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256",
        "UPSTREAM_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256",
        "ArrayCategory",
        "StorageKind",
        "AlgorithmicResourceMeasurementBasisV1",
        "AlgorithmicResourceMeasurementError",
        "AlgorithmicResourceMeasurementLedger",
        "ArrayLeafObservationV1",
        "CategorySnapshotV1",
        "FieldMeasurementV1",
        "ReplayCapacityObservationV1",
        "StructuralAbsenceProofV1",
        "algorithmic_resource_measurement_ledger_descriptor",
        "algorithmic_resource_measurement_ledger_descriptor_sha256",
        "canonical_algorithmic_resource_measurement_basis_bytes",
        "canonical_algorithmic_resource_measurement_ledger_descriptor_bytes",
        "classify_rtu_leaf_path",
        "parse_algorithmic_resource_measurement_basis",
    ]

    assert measurement.__all__ == expected
    assert len(measurement.__all__) == len(set(measurement.__all__))
    assert all(hasattr(measurement, name) for name in measurement.__all__)


def test_explicit_transactions_count_only_committed_work() -> None:
    ledger = _ledger("transaction_ledger")
    _commit_environment(ledger, 2)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="transaction_trainable",
            category="trainable_parameters",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("state", "params"),
                    (1,),
                    category="trainable_parameters",
                    lifecycle="training_live",
                    storage_id="transaction_parameter",
                ),
            ),
        )
    )

    ledger.begin_learning_transaction(
        "actor_update",
        adaptation_kind="actor",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=1,
    ).commit()
    ledger.begin_learning_transaction(
        "critic_update",
        adaptation_kind="critic",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=1,
    ).commit()
    ledger.begin_learning_transaction(
        "ppo_joint_minibatch",
        adaptation_kind="ppo_joint_loss",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=64,
    ).commit()
    ledger.begin_learning_transaction(
        "redo_replay_evaluation",
        adaptation_kind="redo_replay_evaluation",
        optimizer_applied=False,
        gradient_applied=False,
        sample_contributions=32,
    ).commit()
    ledger.begin_learning_transaction(
        "pt_inner_loop",
        adaptation_kind="pt_inner_loop",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=32,
    ).commit()
    ledger.begin_learning_transaction(
        "drqn_burn_in",
        adaptation_kind="drqn_burn_in",
        optimizer_applied=False,
        gradient_applied=False,
        sample_contributions=16,
    ).commit()

    basis = _finish(
        ledger,
        present_categories=frozenset({"trainable_parameters"}),
        positive_counters=frozenset({"optimizer_updates", "gradient_updates", "sample_updates"}),
    )
    fields = _field_map(basis)
    assert fields["max_environment_interactions"]["observed_value"] == 2
    assert fields["max_optimizer_updates"]["observed_value"] == 4
    assert fields["max_gradient_updates"]["observed_value"] == 4
    assert fields["max_sample_updates"]["observed_value"] == 146


def test_transaction_abort_duplicate_and_open_transaction_fail_closed() -> None:
    ledger = _ledger("abort_transactions")
    transaction = ledger.begin_environment_transaction("environment_0")
    transaction.abort()
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        transaction.commit()
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        ledger.begin_environment_transaction("environment_0")

    committed = ledger.begin_environment_transaction("environment_1")
    committed.commit()
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        committed.commit()

    ledger.begin_learning_transaction(
        "left_open",
        adaptation_kind="actor",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=1,
    )
    for subject in _ABSENCE_KIND_BY_SUBJECT:
        ledger.declare_structural_absence(_absence(subject))
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="open"):
        ledger.finalize()


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("_transaction_id", "mutated_transaction"),
        ("_kind", "environment"),
        ("_adaptation_kind", "mutated_adaptation"),
        ("_optimizer_applied", False),
        ("_gradient_applied", False),
        ("_sample_contributions", 999),
    ],
)
def test_transaction_handle_is_structurally_immutable_after_begin(
    attribute: str,
    replacement: object,
) -> None:
    ledger = _ledger(f"immutable_{attribute.removeprefix('_')}")
    transaction = ledger.begin_learning_transaction(
        "learning_transaction",
        adaptation_kind="actor",
        optimizer_applied=True,
        gradient_applied=True,
        sample_contributions=1,
    )

    with pytest.raises(AttributeError):
        setattr(transaction, attribute, replacement)

    transaction.abort()


def test_ledger_owned_transaction_record_defeats_low_level_handle_mutation() -> None:
    ledger = _ledger("low_level_handle_mutation")
    _commit_environment(ledger)
    transaction = ledger.begin_learning_transaction(
        "learning_transaction",
        adaptation_kind="actor",
        optimizer_applied=False,
        gradient_applied=False,
        sample_contributions=1,
    )
    object.__setattr__(transaction, "_sample_contributions", 999)
    transaction.commit()

    basis = _finish(
        ledger,
        positive_counters=frozenset({"sample_updates"}),
    )
    assert _field_map(basis)["max_sample_updates"]["observed_value"] == 1


def test_whole_ledger_abort_prevents_measurement_artifact() -> None:
    ledger = _ledger("whole_abort")
    _commit_environment(ledger)
    ledger.abort("runner_failure")

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="aborted"):
        ledger.finalize()
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="aborted"):
        ledger.begin_environment_transaction("too_late")


@pytest.mark.parametrize("token", ["reward", "score", "return", "rank", "outcome"])
def test_sensitive_field_names_are_rejected_recursively(token: str) -> None:
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="forbidden"):
        measurement.StructuralAbsenceProofV1(
            subject="optimizer_updates",
            absence_kind="optimizer_subsystem_absent",
            proof_id="sensitive_proof",
            evidence_kind="exact_runtime_subsystem_inventory",
            details={"outer": {f"raw_{token}_value": 0}},
        )

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="forbidden"):
        _leaf(
            ("state", f"{token}_trace"),
            (1,),
            category="trainable_parameters",
            lifecycle="initialization",
            storage_id="sensitive_storage",
        )


def test_leaf_metadata_is_canonical_and_dtype_exact() -> None:
    leaf = _leaf(
        ("state", "actor", "kernel"),
        (2, 3),
        category="trainable_parameters",
        lifecycle="initialization",
        storage_id="actor_kernel_storage",
        alias_id="actor_kernel_alias",
        dtype="float32",
        owner="actor_parameters",
    )

    assert leaf.to_dict() == {
        "alias_id": "actor_kernel_alias",
        "canonical_leaf_path": "/state/actor/kernel",
        "category": "trainable_parameters",
        "dtype": "float32",
        "elements": 6,
        "itemsize_bytes": 4,
        "lifecycle": "initialization",
        "logical_owned_bytes": 24,
        "owner": "actor_parameters",
        "shape": [2, 3],
        "storage_id": "actor_kernel_storage",
        "storage_kind": "owned_array",
    }
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="dtype"):
        _leaf(
            ("state", "bad"),
            (1,),
            category="trainable_parameters",
            lifecycle="initialization",
            storage_id="bad_dtype",
            dtype="object",
        )


def test_aliases_are_recorded_and_storage_is_counted_once_within_category() -> None:
    ledger = _ledger("alias_dedup")
    _commit_environment(ledger)
    snapshot = measurement.CategorySnapshotV1(
        snapshot_id="trainable_initial",
        category="trainable_parameters",
        lifecycle="initialization",
        leaves=(
            _leaf(
                ("state", "params", "kernel"),
                (8,),
                category="trainable_parameters",
                lifecycle="initialization",
                storage_id="shared_kernel",
                alias_id="primary_kernel",
            ),
            _leaf(
                ("state", "params", "kernel_alias"),
                (8,),
                category="trainable_parameters",
                lifecycle="initialization",
                storage_id="shared_kernel",
                alias_id="secondary_kernel",
            ),
            _leaf(
                ("state", "params", "kernel_view"),
                (4,),
                category="trainable_parameters",
                lifecycle="initialization",
                storage_id="shared_kernel",
                alias_id="kernel_view",
                storage_kind="array_view",
            ),
        ),
    )
    ledger.observe_category_snapshot(snapshot)
    basis = _finish(
        ledger,
        present_categories=frozenset({"trainable_parameters"}),
    )

    assert _field_map(basis)["max_trainable_parameters"]["observed_value"] == 8
    retained = basis.to_dict()["category_high_water"]["trainable_parameters"]
    aliases = retained["element_snapshot"]["storage_aliases"]
    assert aliases == {"shared_kernel": ["kernel_view", "primary_kernel", "secondary_kernel"]}


def test_view_without_owned_base_cannot_masquerade_as_owned_storage() -> None:
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="owned base"):
        measurement.CategorySnapshotV1(
            snapshot_id="view_only",
            category="rollout_storage",
            lifecycle="minibatch_view",
            leaves=(
                _leaf(
                    ("rollout", "batch_view"),
                    (16,),
                    category="rollout_storage",
                    lifecycle="minibatch_view",
                    storage_id="missing_base",
                    storage_kind="array_view",
                ),
            ),
        )


def test_element_and_byte_high_water_can_occur_at_different_lifecycles() -> None:
    ledger = _ledger("lifecycle_high_water")
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="carry_many_half",
            category="recurrent_carry",
            lifecycle="rollout_start",
            leaves=(
                _leaf(
                    ("carry", "half"),
                    (10,),
                    category="recurrent_carry",
                    lifecycle="rollout_start",
                    storage_id="carry_half",
                    dtype="float16",
                ),
            ),
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="carry_fewer_double",
            category="recurrent_carry",
            lifecycle="rollout_update",
            leaves=(
                _leaf(
                    ("carry", "double"),
                    (4,),
                    category="recurrent_carry",
                    lifecycle="rollout_update",
                    storage_id="carry_double",
                    dtype="float64",
                ),
            ),
        )
    )
    basis = _finish(
        ledger,
        present_categories=frozenset({"recurrent_carry"}),
    )
    fields = _field_map(basis)
    assert fields["max_recurrent_carry_elements"]["observed_value"] == 10
    assert fields["max_recurrent_carry_bytes"]["observed_value"] == 32
    retained = basis.to_dict()["category_high_water"]["recurrent_carry"]
    assert retained["element_snapshot"]["lifecycle"] == "rollout_start"
    assert retained["byte_snapshot"]["lifecycle"] == "rollout_update"


def test_pt_simultaneous_live_union_and_target_role_are_distinct() -> None:
    ledger = _ledger("pt_union")
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="pt_trainable_union",
            category="trainable_parameters",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("state", "params", "kernel"),
                    (5,),
                    category="trainable_parameters",
                    lifecycle="training_live",
                    storage_id="transient_parameters",
                    owner="globally_trainable_stop_gradient_teacher",
                ),
                _leaf(
                    ("state", "perm_params", "kernel"),
                    (7,),
                    category="trainable_parameters",
                    lifecycle="training_live",
                    storage_id="permanent_parameters",
                    owner="permanent_trainable_parameters",
                ),
            ),
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="pt_target",
            category="target_copy",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("state", "target_params", "kernel"),
                    (5,),
                    category="target_copy",
                    lifecycle="training_live",
                    storage_id="target_parameters",
                    owner="transient_target_copy",
                ),
            ),
        )
    )
    basis = _finish(
        ledger,
        present_categories=frozenset({"trainable_parameters", "target_copy"}),
    )
    fields = _field_map(basis)
    assert fields["max_trainable_parameters"]["observed_value"] == 12
    assert fields["max_target_copy_elements"]["observed_value"] == 5
    assert fields["max_frozen_parameters"]["observed_value"] == 0
    assert fields["max_frozen_parameters"]["structural_absence_kind"] == (
        "frozen_parameter_tree_absent"
    )
    with pytest.raises(
        measurement.AlgorithmicResourceMeasurementError,
        match="target-copy",
    ):
        _leaf(
            ("state", "target_params", "misclassified"),
            (1,),
            category="frozen_parameters",
            lifecycle="training_live",
            storage_id="misclassified_target",
        )
    with pytest.raises(
        measurement.AlgorithmicResourceMeasurementError,
        match="trainable parameters",
    ):
        _leaf(
            ("state", "teacher", "misclassified"),
            (1,),
            category="frozen_parameters",
            lifecycle="training_live",
            storage_id="misclassified_teacher",
            owner="globally_trainable_stop_gradient_teacher",
        )


def test_pt_replay_capacity_is_simultaneous_subsystem_union() -> None:
    ledger = _ledger("pt_replay_union")
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="pt_replay_live",
            category="replay_storage",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("replay", "main", "states"),
                    (1_000, 4),
                    category="replay_storage",
                    lifecycle="training_live",
                    storage_id="main_replay_states",
                    dtype="uint8",
                ),
                _leaf(
                    ("replay", "permanent", "states"),
                    (10_000, 4),
                    category="replay_storage",
                    lifecycle="training_live",
                    storage_id="permanent_replay_states",
                    dtype="uint8",
                ),
            ),
            replay_capacities=(
                measurement.ReplayCapacityObservationV1(
                    subsystem_id="main_replay",
                    capacity_transitions=1_000,
                ),
                measurement.ReplayCapacityObservationV1(
                    subsystem_id="permanent_replay",
                    capacity_transitions=10_000,
                ),
            ),
        )
    )
    basis = _finish(
        ledger,
        present_categories=frozenset({"replay_storage"}),
    )
    fields = _field_map(basis)
    assert fields["max_replay_capacity_transitions"]["observed_value"] == 11_000
    assert fields["max_replay_peak_bytes"]["observed_value"] == 44_000


def test_pt_permanent_and_transient_optimizer_state_are_one_simultaneous_union() -> None:
    ledger = _ledger("pt_optimizer_union")
    _commit_environment(ledger)
    ledger.begin_learning_transaction(
        "pt_optimizer_update",
        adaptation_kind="pt_inner_loop",
        optimizer_applied=True,
        gradient_applied=False,
        sample_contributions=1,
    ).commit()
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="pt_trainable_for_optimizer",
            category="trainable_parameters",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("state", "params"),
                    (3,),
                    category="trainable_parameters",
                    lifecycle="training_live",
                    storage_id="pt_trainable_parameters",
                ),
            ),
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="pt_optimizer_state_union",
            category="optimizer_state",
            lifecycle="training_live",
            leaves=(
                _leaf(
                    ("optimizer", "transient", "moments"),
                    (5,),
                    category="optimizer_state",
                    lifecycle="training_live",
                    storage_id="transient_optimizer_moments",
                ),
                _leaf(
                    ("optimizer", "permanent", "moments"),
                    (7,),
                    category="optimizer_state",
                    lifecycle="training_live",
                    storage_id="permanent_optimizer_moments",
                ),
            ),
        )
    )

    basis = _finish(
        ledger,
        present_categories=frozenset({"trainable_parameters", "optimizer_state"}),
        positive_counters=frozenset({"optimizer_updates", "sample_updates"}),
    )
    fields = _field_map(basis)
    assert fields["max_optimizer_state_elements"]["observed_value"] == 12
    assert fields["max_optimizer_state_bytes"]["observed_value"] == 48


def test_exact_rtu_path_classifier_separates_carry_and_sensitivity() -> None:
    ledger = _ledger("rtu_roles")
    _commit_environment(ledger)
    carry_leaves = tuple(
        _leaf(
            ("hstate", owner, component),
            (1, 512),
            category="recurrent_carry",
            lifecycle="rollout_live",
            storage_id=f"{owner}_{component}_carry",
            owner=f"{owner}_rtu_hidden",
        )
        for owner in ("actor", "critic")
        for component in ("real", "imaginary")
    )
    sensitivity_leaves = tuple(
        [
            _leaf(
                ("hstate", owner, "memory_grad", f"diagonal_{index}"),
                (1, 512),
                category="rtrl_sensitivity",
                lifecycle="rollout_live",
                storage_id=f"{owner}_diagonal_{index}",
                owner=f"{owner}_rtrl_sensitivity",
            )
            for owner in ("actor", "critic")
            for index in range(4)
        ]
        + [
            _leaf(
                ("hstate", owner, "memory_grad", f"input_{index}"),
                (1, 69, 512),
                category="rtrl_sensitivity",
                lifecycle="rollout_live",
                storage_id=f"{owner}_input_{index}",
                owner=f"{owner}_rtrl_sensitivity",
            )
            for owner in ("actor", "critic")
            for index in range(4)
        ]
    )
    assert {measurement.classify_rtu_leaf_path(leaf.leaf_path) for leaf in carry_leaves} == {
        "recurrent_carry"
    }
    assert {measurement.classify_rtu_leaf_path(leaf.leaf_path) for leaf in sensitivity_leaves} == {
        "rtrl_sensitivity"
    }
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="rtu_carry_live",
            category="recurrent_carry",
            lifecycle="rollout_live",
            leaves=carry_leaves,
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="rtu_sensitivity_live",
            category="rtrl_sensitivity",
            lifecycle="rollout_live",
            leaves=sensitivity_leaves,
        )
    )
    basis = _finish(
        ledger,
        present_categories=frozenset({"recurrent_carry", "rtrl_sensitivity"}),
    )
    fields = _field_map(basis)
    assert fields["max_recurrent_carry_elements"]["observed_value"] == 2_048
    assert fields["max_recurrent_carry_bytes"]["observed_value"] == 8_192
    assert fields["max_rtrl_sensitivity_elements"]["observed_value"] == 286_720
    assert fields["max_rtrl_sensitivity_bytes"]["observed_value"] == 1_146_880
    assert fields["max_eligibility_elements"]["observed_value"] == 0
    assert fields["max_eligibility_bytes"]["observed_value"] == 0
    assert (
        fields["max_eligibility_elements"]["absence_proof_id"]
        == (fields["max_eligibility_bytes"]["absence_proof_id"])
    )


def test_local_rtu_sensitivity_taylor_and_eligibility_trees_are_exact_unions() -> None:
    assert measurement.classify_rtu_leaf_path(("actor_rtu_state", "real")) == "recurrent_carry"
    assert (
        measurement.classify_rtu_leaf_path(("critic_rtu_state", "imaginary")) == "recurrent_carry"
    )
    assert (
        measurement.classify_rtu_leaf_path(("actor_sensitivities", "nu_log")) == "rtrl_sensitivity"
    )
    assert (
        measurement.classify_rtu_leaf_path(("critic_taylor_trace", "b_imag")) == "rtrl_sensitivity"
    )
    assert measurement.classify_rtu_leaf_path(("actor_traces", "rtu", "nu_log")) == (
        "eligibility_trace"
    )
    assert measurement.classify_rtu_leaf_path(("critic_traces", "head_bias")) == (
        "eligibility_trace"
    )
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        measurement.classify_rtu_leaf_path(("actor_rtu_state", "nu_log"))

    ledger = _ledger("local_rtu_unions")
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="actor_critic_rtu_carry_union",
            category="recurrent_carry",
            lifecycle="rollout_live",
            leaves=(
                _leaf(
                    ("actor_rtu_state", "real"),
                    (5,),
                    category="recurrent_carry",
                    lifecycle="rollout_live",
                    storage_id="actor_rtu_real",
                ),
                _leaf(
                    ("actor_rtu_state", "imaginary"),
                    (5,),
                    category="recurrent_carry",
                    lifecycle="rollout_live",
                    storage_id="actor_rtu_imaginary",
                ),
                _leaf(
                    ("critic_rtu_state", "real"),
                    (4,),
                    category="recurrent_carry",
                    lifecycle="rollout_live",
                    storage_id="critic_rtu_real",
                ),
                _leaf(
                    ("critic_rtu_state", "imaginary"),
                    (4,),
                    category="recurrent_carry",
                    lifecycle="rollout_live",
                    storage_id="critic_rtu_imaginary",
                ),
            ),
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="actor_critic_sensitivity_taylor_union",
            category="rtrl_sensitivity",
            lifecycle="rollout_live",
            leaves=(
                _leaf(
                    ("actor_sensitivities", "nu_log"),
                    (2, 3),
                    category="rtrl_sensitivity",
                    lifecycle="rollout_live",
                    storage_id="actor_sensitivity_nu_log",
                ),
                _leaf(
                    ("critic_sensitivities", "theta_log"),
                    (2, 4),
                    category="rtrl_sensitivity",
                    lifecycle="rollout_live",
                    storage_id="critic_sensitivity_theta_log",
                ),
                _leaf(
                    ("actor_taylor_trace", "b_real"),
                    (2, 5),
                    category="rtrl_sensitivity",
                    lifecycle="rollout_live",
                    storage_id="actor_taylor_b_real",
                ),
                _leaf(
                    ("critic_taylor_trace", "b_imag"),
                    (2, 6),
                    category="rtrl_sensitivity",
                    lifecycle="rollout_live",
                    storage_id="critic_taylor_b_imag",
                ),
            ),
        )
    )
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="positive_eligibility_tree",
            category="eligibility_trace",
            lifecycle="learning_live",
            leaves=(
                _leaf(
                    ("actor_traces", "rtu", "nu_log"),
                    (7,),
                    category="eligibility_trace",
                    lifecycle="learning_live",
                    storage_id="actor_rtu_eligibility_trace",
                ),
                _leaf(
                    ("critic_traces", "head_bias"),
                    (9,),
                    category="eligibility_trace",
                    lifecycle="learning_live",
                    storage_id="critic_head_eligibility_trace",
                ),
            ),
        )
    )
    basis = _finish(
        ledger,
        present_categories=frozenset({"recurrent_carry", "rtrl_sensitivity", "eligibility_trace"}),
    )
    fields = _field_map(basis)
    assert fields["max_recurrent_carry_elements"]["observed_value"] == 18
    assert fields["max_recurrent_carry_bytes"]["observed_value"] == 72
    assert fields["max_rtrl_sensitivity_elements"]["observed_value"] == 36
    assert fields["max_rtrl_sensitivity_bytes"]["observed_value"] == 144
    assert fields["max_eligibility_elements"]["observed_value"] == 16
    assert fields["max_eligibility_bytes"]["observed_value"] == 64


def test_structural_zero_requires_exact_runtime_proof_not_configuration() -> None:
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="evidence kind"):
        measurement.StructuralAbsenceProofV1(
            subject="optimizer_updates",
            absence_kind="optimizer_subsystem_absent",
            proof_id="configuration_zero",
            evidence_kind="configuration_only",
            details={"configured_zero": True},
        )

    ledger = _ledger("missing_absence")
    _commit_environment(ledger)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="absence proof"):
        ledger.finalize()


def test_zero_sized_observed_tree_does_not_establish_structural_absence() -> None:
    ledger = _ledger("zero_sized_tree")
    _commit_environment(ledger)
    ledger.observe_category_snapshot(
        measurement.CategorySnapshotV1(
            snapshot_id="empty_optimizer_tree",
            category="optimizer_state",
            lifecycle="initialization",
            leaves=(
                _leaf(
                    ("optimizer", "empty"),
                    (0,),
                    category="optimizer_state",
                    lifecycle="initialization",
                    storage_id="empty_optimizer_storage",
                ),
            ),
        )
    )
    for subject in _ABSENCE_KIND_BY_SUBJECT:
        if subject != "optimizer_state":
            ledger.declare_structural_absence(_absence(subject))
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="positive"):
        ledger.finalize()


def test_coupled_zero_pairs_share_one_absence_proof_and_basis() -> None:
    ledger = _ledger("coupled_zero")
    _commit_environment(ledger)
    basis = _finish(ledger)
    fields = _field_map(basis)

    for left, right in measurement.COUPLED_FIELD_PAIRS:
        assert fields[left]["observed_value"] == fields[right]["observed_value"] == 0
        assert fields[left]["absence_proof_id"] == fields[right]["absence_proof_id"]
        assert fields[left]["structural_absence_kind"] == (fields[right]["structural_absence_kind"])
    assert (
        basis.body_sha256
        == hashlib.sha256(_canonical(basis.to_body_dict(), newline=False)).hexdigest()
    )


def test_category_snapshot_rejects_role_and_lifecycle_drift() -> None:
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="category"):
        measurement.CategorySnapshotV1(
            snapshot_id="wrong_category",
            category="target_copy",
            lifecycle="initialization",
            leaves=(
                _leaf(
                    ("state", "target"),
                    (1,),
                    category="frozen_parameters",
                    lifecycle="initialization",
                    storage_id="wrong_category_storage",
                ),
            ),
        )
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="lifecycle"):
        measurement.CategorySnapshotV1(
            snapshot_id="wrong_lifecycle",
            category="target_copy",
            lifecycle="initialization",
            leaves=(
                _leaf(
                    ("state", "target"),
                    (1,),
                    category="target_copy",
                    lifecycle="target_sync",
                    storage_id="wrong_lifecycle_storage",
                ),
            ),
        )


def test_basis_bytes_are_bounded_canonical_and_require_caller_file_pin() -> None:
    ledger = _ledger("canonical_basis")
    _commit_environment(ledger)
    basis = _finish(ledger)
    raw = measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)
    file_sha256 = hashlib.sha256(raw).hexdigest()

    parsed = measurement.parse_algorithmic_resource_measurement_basis(
        raw,
        expected_file_sha256=file_sha256,
        expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
    )
    assert parsed == basis.to_dict()
    assert len(raw) <= measurement.MAX_MEASUREMENT_BASIS_BYTES

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="file"):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256="1" * 64,
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )
    with pytest.raises(TypeError):
        measurement.parse_algorithmic_resource_measurement_basis(raw)  # type: ignore[call-arg]
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="canonical"):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw.rstrip(b"\n"),
            expected_file_sha256=hashlib.sha256(raw.rstrip(b"\n")).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="source"):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=file_sha256,
            expected_measurement_source_sha256="1" * 64,
        )


def test_basis_parser_roundtrips_positive_retained_category_snapshots() -> None:
    basis = _positive_trainable_basis("positive_parser")
    raw = measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)

    parsed = measurement.parse_algorithmic_resource_measurement_basis(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
    )

    retained = parsed["category_high_water"]["trainable_parameters"]
    assert retained["observation_count"] == 1
    assert retained["element_snapshot"]["leaves"][0]["elements"] == 1
    assert retained["byte_snapshot"]["logical_owned_bytes"] == 4


@pytest.mark.parametrize(
    ("location", "replacement"),
    [
        ("leaf_elements", True),
        ("snapshot_logical_bytes", 4.0),
        ("policy_false", 0),
    ],
)
def test_parser_rejects_recursive_json_scalar_type_substitutions(
    location: str,
    replacement: object,
) -> None:
    tampered = _positive_trainable_basis(f"exact_json_{location}").to_dict()
    retained = tampered["category_high_water"]["trainable_parameters"]
    if location == "leaf_elements":
        retained["element_snapshot"]["leaves"][0]["elements"] = replacement
    elif location == "snapshot_logical_bytes":
        retained["byte_snapshot"]["logical_owned_bytes"] = replacement
    else:
        tampered["authority"]["execution_authorized"] = replacement
    raw = _reseal_basis_dict(tampered)

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )


def test_parser_enforces_learning_and_sample_accounting_both_directions() -> None:
    ledger = _ledger("learning_accounting")
    _commit_environment(ledger)
    for index in range(2):
        ledger.begin_learning_transaction(
            f"adaptation_{index}",
            adaptation_kind="replay_evaluation",
            optimizer_applied=False,
            gradient_applied=False,
            sample_contributions=1,
        ).commit()
    basis = _finish(
        ledger,
        positive_counters=frozenset({"sample_updates"}),
    ).to_dict()

    zero_learning = copy.deepcopy(basis)
    zero_learning["transaction_accounting"]["committed_learning_transactions"] = 0
    zero_learning["transaction_accounting"]["aborted_transactions"] = 2
    zero_raw = _reseal_basis_dict(zero_learning)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="learning"):
        measurement.parse_algorithmic_resource_measurement_basis(
            zero_raw,
            expected_file_sha256=hashlib.sha256(zero_raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )

    too_few_samples = copy.deepcopy(basis)
    too_few_samples["transaction_accounting"]["consumed_sample_contributions"] = 1
    for field in too_few_samples["fields"]:
        if field["field_name"] == "max_sample_updates":
            field["observed_value"] = 1
    sample_raw = _reseal_basis_dict(too_few_samples)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="sample"):
        measurement.parse_algorithmic_resource_measurement_basis(
            sample_raw,
            expected_file_sha256=hashlib.sha256(sample_raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )


def test_direct_basis_construction_and_serialization_validate_semantics() -> None:
    basis = _positive_trainable_basis("direct_basis_validation")
    invalid_accounting = dict(basis.transaction_accounting)
    invalid_accounting["committed_environment_transactions"] = 2

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        replace(basis, transaction_accounting=invalid_accounting)

    object.__setattr__(basis, "transaction_accounting", invalid_accounting)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError):
        measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)


def test_parser_switches_to_repository_descriptor_literal_after_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis = _positive_trainable_basis("literal_parser")
    raw = measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    computed = measurement.algorithmic_resource_measurement_ledger_descriptor_sha256()

    monkeypatch.setattr(
        measurement,
        "PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256",
        computed,
    )
    assert (
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=file_sha256,
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )
        == basis.to_dict()
    )

    monkeypatch.setattr(
        measurement,
        "PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256",
        "1" * 64,
    )
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="descriptor"):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=file_sha256,
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )


def test_basis_parser_rejects_body_tampering_even_with_new_file_pin() -> None:
    ledger = _ledger("body_tamper")
    _commit_environment(ledger)
    raw = measurement.canonical_algorithmic_resource_measurement_basis_bytes(_finish(ledger))
    tampered = json.loads(raw)
    tampered["ledger_id"] = "different_ledger"
    tampered_raw = _canonical(tampered)

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="body digest"):
        measurement.parse_algorithmic_resource_measurement_basis(
            tampered_raw,
            expected_file_sha256=hashlib.sha256(tampered_raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )


def test_parsed_basis_rejects_recursively_injected_sensitive_field() -> None:
    ledger = _ledger("sensitive_parse")
    _commit_environment(ledger)
    original = _finish(ledger).to_dict()
    tampered = copy.deepcopy(original)
    tampered["observation_chain"]["nested_reward"] = 0
    body = copy.deepcopy(tampered)
    body.pop("measurement_basis_body_sha256")
    tampered["measurement_basis_body_sha256"] = hashlib.sha256(
        _canonical(body, newline=False)
    ).hexdigest()
    raw = _canonical(tampered)

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="forbidden"):
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )


def test_basis_payload_limits_are_enforced_before_serialization() -> None:
    oversized = "x" * (measurement.MAX_TEXT_LENGTH + 1)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="text"):
        measurement.StructuralAbsenceProofV1(
            subject="optimizer_updates",
            absence_kind="optimizer_subsystem_absent",
            proof_id="oversized_proof",
            evidence_kind="exact_runtime_subsystem_inventory",
            details={"diagnostic": oversized},
        )


def test_complete_basis_depth_accounts_for_structural_proof_wrapper_offsets() -> None:
    boundary = _ledger("complete_depth_boundary")
    _commit_environment(boundary)
    for subject in _ABSENCE_KIND_BY_SUBJECT:
        details = (
            _nested_details(measurement.MAX_JSON_DEPTH - 3)
            if subject == ("optimizer_updates")
            else {"observed": True}
        )
        boundary.declare_structural_absence(_absence_with_details(subject, details))
    basis = boundary.finalize()
    raw = measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)
    assert (
        measurement.parse_algorithmic_resource_measurement_basis(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_measurement_source_sha256=_TEST_MEASUREMENT_SOURCE_SHA256,
        )
        == basis.to_dict()
    )

    overflow = _ledger("complete_depth_overflow")
    _commit_environment(overflow)
    for subject in _ABSENCE_KIND_BY_SUBJECT:
        details = (
            _nested_details(measurement.MAX_JSON_DEPTH - 2)
            if subject == ("optimizer_updates")
            else {"observed": True}
        )
        overflow.declare_structural_absence(_absence_with_details(subject, details))
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="depth"):
        overflow.finalize()


def test_complete_basis_rejects_shallow_aggregate_json_node_overflow() -> None:
    ledger = _ledger("complete_node_overflow")
    _commit_environment(ledger)
    for proof in _aggregate_heavy_absences():
        ledger.declare_structural_absence(proof)

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="JSON nodes"):
        ledger.finalize()


def test_direct_constructor_and_canonical_serializer_enforce_complete_node_limit() -> None:
    ledger = _ledger("complete_node_direct")
    _commit_environment(ledger)
    basis = _finish(ledger)
    heavy_absences = _aggregate_heavy_absences()

    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="JSON nodes"):
        replace(basis, structural_absence_proofs=heavy_absences)

    object.__setattr__(basis, "structural_absence_proofs", heavy_absences)
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="JSON nodes"):
        measurement.canonical_algorithmic_resource_measurement_basis_bytes(basis)


def test_finalization_is_idempotent_and_seals_mutation() -> None:
    ledger = _ledger("seal_once")
    _commit_environment(ledger)
    first = _finish(ledger)
    second = ledger.finalize()

    assert second is first
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="sealed"):
        ledger.begin_environment_transaction("after_seal")
    with pytest.raises(measurement.AlgorithmicResourceMeasurementError, match="sealed"):
        ledger.declare_structural_absence(_absence("optimizer_updates"))
