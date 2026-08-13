"""Focused tests for the source-only matched-v3 storage-boundary contract."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_storage_boundary as contract,
)

EXPECTED_DESCRIPTOR_SHA256 = "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"


class _StringSubclass(str):
    pass


class _TupleSubclass(tuple[Any, ...]):
    pass


class _IntSubclass(int):
    pass


class _EqualityProxy:
    def __eq__(self, other: object) -> bool:
        return True


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _body(value: object) -> str:
    return hashlib.sha256(_canonical(value, newline=False)).hexdigest()


def _rebody(value: dict[str, Any], field: str) -> bytes:
    body = copy.deepcopy(value)
    body.pop(field, None)
    value[field] = _body(body)
    return _canonical(value)


def _artifact(schema: str, label: str) -> contract.ArtifactIdentityV1:
    return contract.ArtifactIdentityV1(
        schema_version=schema,
        file_sha256=_sha(f"{label}-file"),
        body_sha256=_sha(f"{label}-body"),
    )


def _component(schema: str, label: str) -> contract.ComponentIdentityV1:
    return contract.ComponentIdentityV1(
        descriptor_schema_version=schema,
        descriptor_sha256=_sha(f"{label}-descriptor"),
        source_sha256=_sha(f"{label}-source"),
    )


def _runtime() -> contract.RuntimeStorageIdentityV1:
    return contract.RuntimeStorageIdentityV1(
        runtime_candidate=_artifact(contract.RUNTIME_CANDIDATE_SCHEMA_VERSION, "runtime"),
        runtime_qualification_receipt=_artifact(
            contract.RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime-qualification",
        ),
        runtime_name="docker_engine_qualified",
        runtime_binary_sha256=_sha("runtime-binary"),
        runtime_configuration_body_sha256=_sha("runtime-configuration"),
    )


def _container() -> contract.ContainerStorageIdentityV1:
    return contract.ContainerStorageIdentityV1(
        container_name="matched-v3-case-00",
        container_identity_commitment_sha256=_sha("container-commitment"),
        mount_namespace_identity_sha256=_sha("mount-namespace"),
        rootfs_mount_identity_sha256=_sha("rootfs-mount"),
        image_layers_read_only=True,
        container_identity_precommitted=True,
        mount_namespace_mutation_disabled_after_go=True,
        rootfs_copy_up_policy="bound_writable_root",
    )


def _architecture(
    kind: str = "worker_exit_then_isolated_terminal_relay",
) -> contract.StorageSealArchitectureV1:
    return contract.StorageSealArchitectureV1(
        architecture_kind=kind,  # type: ignore[arg-type]
        terminal_relay=_component(
            contract.QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
            "relay",
        ),
        nonstorage_control_channel=_component(
            contract.QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
            "channel",
        ),
        write_seal_producer=_component(
            contract.QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
            "seal-producer",
        ),
        terminal_relay_binary_sha256=_sha("relay-binary"),
        nonstorage_channel_commitment_sha256=_sha("nonstorage-channel"),
        write_seal_policy_body_sha256=_sha("write-seal-policy"),
        container_log_driver="none",
        architecture_committed_before_go=True,
        worker_and_terminal_relay_separated_before_go=True,
        terminal_relay_has_measured_writable_namespace=False,
        terminal_relay_has_measured_writable_fd=False,
        terminal_transport_uses_nonstorage_channel=True,
        container_logging_can_write_measured_storage=False,
        late_remount_used_or_permitted=False,
    )


def _roots(*, temporary_absent: bool = False) -> tuple[contract.StorageRootBindingV1, ...]:
    result = [
        contract.StorageRootBindingV1(
            root_id="disk_root",
            field_name="max_disk_peak_bytes",
            absolute_path="/case/disk",
            mount_identity_sha256=_sha("disk-mount"),
            backing_store_identity_sha256=_sha("disk-store"),
            filesystem_type="ext4_project_quota",
            created_exclusively_for_case=True,
            empty_at_fresh_boundary=True,
            writable=True,
            includes_overlay_copy_up=True,
        )
    ]
    if not temporary_absent:
        result.append(
            contract.StorageRootBindingV1(
                root_id="temporary_root",
                field_name="max_temporary_peak_bytes",
                absolute_path="/case/temporary",
                mount_identity_sha256=_sha("temporary-mount"),
                backing_store_identity_sha256=_sha("temporary-store"),
                filesystem_type="tmpfs",
                created_exclusively_for_case=True,
                empty_at_fresh_boundary=True,
                writable=True,
                includes_overlay_copy_up=False,
            )
        )
    return tuple(result)


def _policies(
    roots: tuple[contract.StorageRootBindingV1, ...],
) -> tuple[contract.StorageFieldPolicyV1, ...]:
    temporary_ids = tuple(
        item.root_id for item in roots if item.field_name == "max_temporary_peak_bytes"
    )
    disk_ids = tuple(item.root_id for item in roots if item.field_name == "max_disk_peak_bytes")
    return (
        contract.StorageFieldPolicyV1(
            field_name="max_temporary_peak_bytes",
            field_position=24,
            measurement_mode=contract.EVENT_COMPLETE_ACCOUNTING_MODE,
            value_semantics=contract.EXACT_OBSERVATION,
            lifetime_scope=contract.STORAGE_LIFETIME_SCOPE,
            root_ids=temporary_ids,
            measurement_policy_body_sha256=_sha("temporary-measurement-policy"),
            event_accounting_policy_body_sha256=_sha("temporary-event-policy"),
            quota_enforcement_policy_body_sha256=None,
            hard_limit_bytes=None,
            committed_before_go=True,
            polling_or_sampling_sufficient=False,
            du_snapshot_sufficient=False,
            container_layer_size_sufficient=False,
            missing_value_defaults_to_zero=False,
        ),
        contract.StorageFieldPolicyV1(
            field_name="max_disk_peak_bytes",
            field_position=25,
            measurement_mode=contract.HARD_QUOTA_ENFORCEMENT_MODE,
            value_semantics=contract.CONSERVATIVE_ENFORCED_UPPER_BOUND,
            lifetime_scope=contract.STORAGE_LIFETIME_SCOPE,
            root_ids=disk_ids,
            measurement_policy_body_sha256=_sha("disk-measurement-policy"),
            event_accounting_policy_body_sha256=None,
            quota_enforcement_policy_body_sha256=_sha("disk-quota-policy"),
            hard_limit_bytes=8192,
            committed_before_go=True,
            polling_or_sampling_sufficient=False,
            du_snapshot_sufficient=False,
            container_layer_size_sufficient=False,
            missing_value_defaults_to_zero=False,
        ),
    )


def _producer() -> contract.ProducerIdentityV1:
    return contract.ProducerIdentityV1(
        descriptor_schema_version=(
            contract.QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
        ),
        descriptor_sha256=_sha("storage-producer-descriptor"),
        source_sha256=_sha("storage-producer-source"),
    )


def _intent(
    case_ordinal: int = 0,
    *,
    temporary_absent: bool = False,
    architecture_kind: str = "worker_exit_then_isolated_terminal_relay",
) -> contract.QualificationStorageBoundaryIntentV1:
    candidate = contract.MATCHED_V3_STORAGE_CANDIDATE_IDS[case_ordinal]
    if candidate in contract.MATCHED_V3_LOCAL_CANDIDATE_IDS:
        family = "local"
    elif candidate in contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        family = "external"
    else:
        family = "adapter"
    container = _container()
    roots = _roots(temporary_absent=temporary_absent)
    policies = _policies(roots)
    return contract.QualificationStorageBoundaryIntentV1(
        schema_version=contract.QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        campaign_spine_sha256=_sha("campaign-spine"),
        case_spine_sha256=_sha(f"case-spine-{case_ordinal}"),
        case_ordinal=case_ordinal,
        candidate_id=candidate,
        candidate_family=family,  # type: ignore[arg-type]
        qualification_case_id=f"qualification_{case_ordinal:02d}_{candidate}",
        resource_requirement_body_sha256=_sha(f"resource-requirement-{case_ordinal}"),
        resource_field_order_sha256=contract.RESOURCE_FIELD_ORDER_SHA256,
        resource_fields=contract.RESOURCE_FIELDS,
        image_id=f"sha256:{_sha('image')}",
        runtime_identity=_runtime(),
        container_identity=container,
        seal_architecture=_architecture(architecture_kind),
        host_provisioning_receipt=_artifact(
            contract.HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host-provisioning",
        ),
        measurement_producer=_producer(),
        writable_mount_policy_body_sha256=_sha("writable-mount-policy"),
        writable_path_policy_body_sha256=_sha("writable-path-policy"),
        storage_root_inventory_sha256=contract.storage_root_inventory_sha256(
            roots,
            container,
        ),
        storage_roots=roots,
        field_policy_inventory_sha256=(
            contract.storage_field_policy_inventory_sha256(policies, roots)
        ),
        field_policy=policies,
        intent_committed_before_go=True,
        go_identity_bound_in_intent=False,
    )


def _intent_identity(
    intent: contract.QualificationStorageBoundaryIntentV1,
) -> contract.ArtifactIdentityV1:
    raw = contract.canonical_matched_v3_qualification_storage_boundary_intent_bytes(intent)
    return contract.ArtifactIdentityV1(
        schema_version=contract.QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=intent.body_sha256,
    )


def _boundary_chain(
    intent: contract.QualificationStorageBoundaryIntentV1 | None = None,
) -> contract.CollectionBoundaryChainV1:
    exact_intent = _intent() if intent is None else intent
    intent_identity = _intent_identity(exact_intent)
    request = _artifact(contract.HOST_CASE_REQUEST_SCHEMA_VERSION, "request")
    host_intent = _artifact(contract.HOST_CASE_INTENT_SCHEMA_VERSION, "host-intent")
    ready = _artifact(contract.HOST_READY_SCHEMA_VERSION, "ready")
    anchor = _artifact(contract.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION, "observer-anchor")
    go = _artifact(contract.HOST_GO_SCHEMA_VERSION, "go")
    projections: dict[str, str] = {
        "host_intent_request_file_sha256": request.file_sha256,
        "host_intent_request_body_sha256": request.body_sha256,
        "ready_host_intent_file_sha256": host_intent.file_sha256,
        "ready_host_intent_body_sha256": host_intent.body_sha256,
        "observer_anchor_ready_file_sha256": ready.file_sha256,
        "observer_anchor_ready_body_sha256": ready.body_sha256,
        "request_storage_intent_file_sha256": intent_identity.file_sha256,
        "request_storage_intent_body_sha256": intent_identity.body_sha256,
        "ready_storage_intent_file_sha256": intent_identity.file_sha256,
        "ready_storage_intent_body_sha256": intent_identity.body_sha256,
        "go_ready_file_sha256": ready.file_sha256,
        "go_ready_body_sha256": ready.body_sha256,
        "go_observer_anchor_file_sha256": anchor.file_sha256,
        "go_observer_anchor_body_sha256": anchor.body_sha256,
    }
    projection_body: dict[str, Any] = {
        "host_case_request": request.to_dict(),
        "host_case_intent": host_intent.to_dict(),
        "host_ready": ready.to_dict(),
        "host_observer_anchor": anchor.to_dict(),
        "host_go": go.to_dict(),
        **projections,
    }
    return contract.CollectionBoundaryChainV1(
        host_case_request=request,
        host_case_intent=host_intent,
        host_ready=ready,
        host_observer_anchor=anchor,
        host_go=go,
        handshake_chain_body_sha256=_body(projection_body),
        **projections,
    )


def _surface(
    intent: contract.QualificationStorageBoundaryIntentV1,
) -> contract.WritableSurfaceEvidenceV1:
    return contract.WritableSurfaceEvidenceV1(
        writable_mount_inventory_body_sha256=_sha("writable-mount-inventory"),
        writable_path_inventory_body_sha256=_sha("writable-path-inventory"),
        bound_root_inventory_sha256=intent.storage_root_inventory_sha256,
        proof_body_sha256=_sha("writable-surface-proof"),
        all_writable_mounts_bound=True,
        all_writable_paths_beneath_bound_roots=True,
        unbound_writable_mount_count=0,
        unbound_writable_path_count=0,
        alternate_writable_mounts_possible=False,
        alternate_writable_paths_possible=False,
        rootfs_copy_up_bound_or_impossible=True,
        deleted_open_files_accounted_or_impossible=True,
        anonymous_files_accounted_or_impossible=True,
        memory_backed_files_accounted_or_impossible=True,
        mount_namespace_mutation_disabled=True,
        descendant_mount_creation_disabled=True,
        host_path_write_escape_disabled=True,
        device_write_escape_disabled=True,
        network_storage_write_escape_disabled=True,
        inherited_writable_fd_escape_disabled=True,
    )


def _seal(
    intent: contract.QualificationStorageBoundaryIntentV1,
    surface: contract.WritableSurfaceEvidenceV1,
) -> contract.WriteQuiescenceSealProofV1:
    worker_exit = (
        intent.seal_architecture.architecture_kind == "worker_exit_then_isolated_terminal_relay"
    )
    seal_identity = _artifact(
        contract.QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
        "write-seal",
    )
    wrapper_identity = _artifact(
        contract.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "publication-commitment",
    )
    reload_observation = _sha("publication-reload-observation")
    reload_validation_body_value = {
        "schema_version": contract.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "publication_commitment_wrapper_file_sha256": wrapper_identity.file_sha256,
        "publication_commitment_wrapper_body_sha256": wrapper_identity.body_sha256,
        "expected_reload_observation_sha256": reload_observation,
        "actual_reload_observation_sha256": reload_observation,
        "reload_performed": True,
        "reload_read_only": True,
    }
    reload_validation_body = _body(reload_validation_body_value)
    reload_validation_file_value = {
        **reload_validation_body_value,
        "reload_validation_body_sha256": reload_validation_body,
    }
    reload_validation_identity = contract.ArtifactIdentityV1(
        schema_version=contract.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(_canonical(reload_validation_file_value)).hexdigest(),
        body_sha256=reload_validation_body,
    )
    return contract.WriteQuiescenceSealProofV1(
        publication_commitment=wrapper_identity,
        write_quiescence_seal=seal_identity,
        terminal_relay_preseal_attestation=_artifact(
            contract.QUALIFICATION_STORAGE_TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "relay-preseal-attestation",
        ),
        nonstorage_channel_readiness_attestation=_artifact(
            contract.QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_READINESS_ATTESTATION_SCHEMA_VERSION,
            "channel-readiness-attestation",
        ),
        publication_reload_validation=reload_validation_identity,
        seal_architecture_body_sha256=intent.seal_architecture.body_sha256,
        publication_commitment_wrapper_file_sha256=wrapper_identity.file_sha256,
        publication_commitment_wrapper_body_sha256=wrapper_identity.body_sha256,
        expected_reload_observation_sha256=reload_observation,
        actual_reload_observation_sha256=reload_observation,
        sealed_mount_inventory_body_sha256=(surface.writable_mount_inventory_body_sha256),
        sealed_path_inventory_body_sha256=surface.writable_path_inventory_body_sha256,
        write_quiescence_seal_body_sha256=seal_identity.body_sha256,
        publication_committed_before_seal=True,
        reload_validated_before_seal=True,
        reload_performed=True,
        reload_read_only=True,
        write_quiescence_irreversible=True,
        container_writes_disabled=True,
        descendant_writes_disabled=True,
        later_allocation_possible=False,
        later_copy_up_possible=False,
        later_peak_increase_possible=False,
        terminal_transport_outside_measured_storage=True,
        terminal_transport_can_allocate_measured_storage=False,
        worker_exit_observed=worker_exit,
        irreversible_seccomp_fd_closure_observed=not worker_exit,
        worker_has_measured_writable_namespace=False,
        worker_has_measured_writable_fd=False,
        only_trusted_terminal_relay_retains_terminal_transport_capability=True,
        terminal_relay_input_policy_restricted_to_receipt_identity=True,
        nonstorage_channel_ready_for_post_receipt_terminal=True,
        container_log_driver_none=True,
        container_logging_write_possible=False,
        no_later_writer_exists=True,
        teardown_deletion_only=True,
        teardown_can_increase_measured_usage=False,
        peak_is_whole_fresh_case_lifetime_peak=True,
    )


def _event(
    intent: contract.QualificationStorageBoundaryIntentV1,
    surface: contract.WritableSurfaceEvidenceV1,
    seal: contract.WriteQuiescenceSealProofV1,
) -> contract.EventCompleteAccountingEvidenceV1:
    policy = intent.field_policy[0]
    return contract.EventCompleteAccountingEvidenceV1(
        event_stream_body_sha256=_sha("event-stream"),
        field_name=policy.field_name,
        field_root_ids=policy.root_ids,
        field_root_inventory_sha256=contract.storage_field_root_inventory_sha256(
            policy.field_name,
            policy.root_ids,
            intent.storage_roots,
        ),
        replayed_peak_bytes=0 if not policy.root_ids else 4096,
        replayed_peak_is_simultaneous_aggregate_union_high_water=True,
        root_event_inventory_body_sha256=intent.storage_root_inventory_sha256,
        writable_mount_inventory_body_sha256=(surface.writable_mount_inventory_body_sha256),
        writable_path_inventory_body_sha256=surface.writable_path_inventory_body_sha256,
        write_quiescence_seal_file_sha256=seal.write_quiescence_seal.file_sha256,
        write_quiescence_seal_body_sha256=seal.write_quiescence_seal.body_sha256,
        fresh_boundary_sequence=0,
        seal_boundary_sequence=999,
        accounting_started_before_go=True,
        accounting_closed_at_irreversible_seal=True,
        all_bound_roots_covered=True,
        allocation_and_deallocation_events_complete=True,
        filesystem_copy_up_events_complete=True,
        deleted_open_file_events_complete=True,
        event_loss_count=0,
        event_overflow_count=0,
        polling_or_sampling_used=False,
        du_snapshot_used=False,
        container_layer_size_used=False,
        exact_high_water_replayed=True,
    )


def _quota(
    intent: contract.QualificationStorageBoundaryIntentV1,
    surface: contract.WritableSurfaceEvidenceV1,
    seal: contract.WriteQuiescenceSealProofV1,
) -> contract.QuotaEnforcementEvidenceV1:
    policy = intent.field_policy[1]
    return contract.QuotaEnforcementEvidenceV1(
        enforcement_kind="kernel_project_hard_quota",
        hard_limit_bytes=8192,
        field_name=policy.field_name,
        field_root_ids=policy.root_ids,
        field_root_inventory_sha256=contract.storage_field_root_inventory_sha256(
            policy.field_name,
            policy.root_ids,
            intent.storage_roots,
        ),
        enforcement_receipt_body_sha256=_sha("quota-enforcement-receipt"),
        enforcement_boundary_body_sha256=_sha("quota-enforcement-boundary"),
        writable_mount_inventory_body_sha256=(surface.writable_mount_inventory_body_sha256),
        writable_path_inventory_body_sha256=surface.writable_path_inventory_body_sha256,
        storage_root_inventory_sha256=intent.storage_root_inventory_sha256,
        write_quiescence_seal_file_sha256=seal.write_quiescence_seal.file_sha256,
        write_quiescence_seal_body_sha256=seal.write_quiescence_seal.body_sha256,
        installed_before_go=True,
        immutable_through_container_removal=True,
        non_bypass_through_container_removal=True,
        all_bound_roots_covered=True,
        hard_limit_applies_to_aggregate_union=True,
        alternate_writable_mount_count=0,
        alternate_writable_path_count=0,
        overlay_copy_up_outside_boundary_possible=False,
        descendant_bypass_possible=False,
        quota_breached=False,
        breach_count=0,
        breach_status_final_at_seal=True,
        polling_or_sampling_used=False,
        du_snapshot_used=False,
        container_layer_size_used=False,
    )


def _fields(
    intent: contract.QualificationStorageBoundaryIntentV1,
    surface: contract.WritableSurfaceEvidenceV1,
    seal: contract.WriteQuiescenceSealProofV1,
    *,
    temporary_absent: bool,
) -> tuple[contract.StoragePeakMeasurementV1, ...]:
    absence = (
        contract.StructuralAbsenceEvidenceV1(
            field_name="max_temporary_peak_bytes",
            absence_kind=contract.TEMPORARY_STORAGE_STRUCTURALLY_ABSENT,
            namespace_inventory_body_sha256=(surface.writable_path_inventory_body_sha256),
            absence_proof_body_sha256=_sha("temporary-absence-proof"),
            no_bound_storage_root_for_field=True,
            no_writable_mount_for_field=True,
            no_writable_path_for_field=True,
            no_overlay_copy_up_target_for_field=True,
        )
        if temporary_absent
        else None
    )
    return (
        contract.StoragePeakMeasurementV1(
            field_name="max_temporary_peak_bytes",
            field_position=24,
            observed_value=0 if temporary_absent else 4096,
            value_semantics=contract.EXACT_OBSERVATION,
            measurement_mode=contract.EVENT_COMPLETE_ACCOUNTING_MODE,
            lifetime_scope=contract.STORAGE_LIFETIME_SCOPE,
            measurement_basis_body_sha256=_sha("temporary-measurement-basis"),
            structural_absence_kind=(
                contract.TEMPORARY_STORAGE_STRUCTURALLY_ABSENT
                if temporary_absent
                else contract.NOT_ABSENT
            ),
            event_complete_evidence=_event(intent, surface, seal),
            quota_enforcement_evidence=None,
            structural_absence_evidence=absence,
        ),
        contract.StoragePeakMeasurementV1(
            field_name="max_disk_peak_bytes",
            field_position=25,
            observed_value=8192,
            value_semantics=contract.CONSERVATIVE_ENFORCED_UPPER_BOUND,
            measurement_mode=contract.HARD_QUOTA_ENFORCEMENT_MODE,
            lifetime_scope=contract.STORAGE_LIFETIME_SCOPE,
            measurement_basis_body_sha256=_sha("disk-measurement-basis"),
            structural_absence_kind=contract.NOT_ABSENT,
            event_complete_evidence=None,
            quota_enforcement_evidence=_quota(intent, surface, seal),
            structural_absence_evidence=None,
        ),
    )


def _receipt(
    case_ordinal: int = 0,
    *,
    intent: contract.QualificationStorageBoundaryIntentV1 | None = None,
    temporary_absent: bool = False,
    architecture_kind: str = "worker_exit_then_isolated_terminal_relay",
) -> contract.QualificationStorageBoundaryReceiptV1:
    exact_intent = (
        _intent(
            case_ordinal,
            temporary_absent=temporary_absent,
            architecture_kind=architecture_kind,
        )
        if intent is None
        else intent
    )
    surface = _surface(exact_intent)
    seal = _seal(exact_intent, surface)
    fields = _fields(exact_intent, surface, seal, temporary_absent=temporary_absent)
    field_digest = contract.storage_measurement_inventory_sha256(
        fields,
        exact_intent.field_policy,
        exact_intent.storage_roots,
        exact_intent.storage_root_inventory_sha256,
        surface,
        seal,
    )
    return contract.QualificationStorageBoundaryReceiptV1(
        schema_version=contract.QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
        campaign_spine_sha256=exact_intent.campaign_spine_sha256,
        case_spine_sha256=exact_intent.case_spine_sha256,
        case_ordinal=exact_intent.case_ordinal,
        candidate_id=exact_intent.candidate_id,
        candidate_family=exact_intent.candidate_family,
        qualification_case_id=exact_intent.qualification_case_id,
        resource_requirement_body_sha256=(exact_intent.resource_requirement_body_sha256),
        resource_field_order_sha256=exact_intent.resource_field_order_sha256,
        resource_fields=exact_intent.resource_fields,
        image_id=exact_intent.image_id,
        runtime_identity=exact_intent.runtime_identity,
        container_identity=exact_intent.container_identity,
        seal_architecture=exact_intent.seal_architecture,
        host_provisioning_receipt=exact_intent.host_provisioning_receipt,
        measurement_producer=exact_intent.measurement_producer,
        measurement_intent=_intent_identity(exact_intent),
        collection_boundary_chain=_boundary_chain(exact_intent),
        writable_mount_policy_body_sha256=(exact_intent.writable_mount_policy_body_sha256),
        writable_path_policy_body_sha256=(exact_intent.writable_path_policy_body_sha256),
        storage_root_inventory_sha256=exact_intent.storage_root_inventory_sha256,
        storage_roots=exact_intent.storage_roots,
        field_policy_inventory_sha256=exact_intent.field_policy_inventory_sha256,
        field_policy=exact_intent.field_policy,
        writable_surface_evidence=surface,
        write_quiescence_seal=seal,
        field_inventory_sha256=field_digest,
        fields=fields,
    )


def _all_false(value: object) -> bool:
    if type(value) is dict:
        return all(_all_false(child) for child in value.values())
    return value is False


def _all_mapping_keys(value: object) -> set[str]:
    if type(value) is list:
        result: set[str] = set()
        for child in value:
            result.update(_all_mapping_keys(child))
        return result
    if type(value) is not dict:
        return set()
    result = set(value)
    for child in value.values():
        result.update(_all_mapping_keys(child))
    return result


def test_frozen_candidate_resource_and_storage_field_order() -> None:
    assert len(contract.MATCHED_V3_STORAGE_CANDIDATE_IDS) == 28
    assert contract.MATCHED_V3_STORAGE_CANDIDATE_IDS == (
        contract.MATCHED_V3_LOCAL_CANDIDATE_IDS
        + contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
        + contract.MATCHED_V3_ADAPTER_CANDIDATE_IDS
        + contract.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
    )
    assert len(contract.RESOURCE_FIELDS) == 28
    assert contract.RESOURCE_FIELDS[23:25] == contract.STORAGE_RESOURCE_FIELDS
    assert contract.STORAGE_RESOURCE_FIELDS == (
        "max_temporary_peak_bytes",
        "max_disk_peak_bytes",
    )
    assert dict(contract.STORAGE_FIELD_POSITIONS) == {
        "max_temporary_peak_bytes": 24,
        "max_disk_peak_bytes": 25,
    }
    assert contract.MATCHED_V3_STORAGE_CANDIDATE_ORDER_SHA256 == (
        "d93aaf66053aaf9a7b1c6d268a47740078dd2c1007f7287bd80908707e40b858"
    )
    assert contract.RESOURCE_FIELD_ORDER_SHA256 == (
        "8048ec1a1402b45d8bb4c67684ee7216b242bfb6d3ed9e196c0cfb262c3b93cc"
    )
    assert _body(list(contract.MATCHED_V3_STORAGE_CANDIDATE_IDS)) == (
        contract.MATCHED_V3_STORAGE_CANDIDATE_ORDER_SHA256
    )
    assert _body(list(contract.RESOURCE_FIELDS)) == contract.RESOURCE_FIELD_ORDER_SHA256


def test_descriptor_is_pinned_source_only_and_non_authorizing() -> None:
    descriptor = contract.matched_v3_qualification_storage_boundary_contract_descriptor()
    assert descriptor["status"] == contract.QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_STATUS
    assert descriptor["storage_fields"] == list(contract.STORAGE_RESOURCE_FIELDS)
    assert descriptor["allowed_measurement_modes"] == [
        contract.EVENT_COMPLETE_ACCOUNTING_MODE,
        contract.HARD_QUOTA_ENFORCEMENT_MODE,
    ]
    assert descriptor["anti_substitution_contract"] == {
        "polling_or_periodic_sampling_sufficient": False,
        "du_snapshots_sufficient": False,
        "container_layer_size_sufficient": False,
        "missing_value_defaults_to_zero": False,
        "zero_requires_exact_typed_structural_absence": True,
    }
    assert descriptor["storage_operations_performed"] is False
    assert descriptor["candidate_values_supplied_or_inferred"] is False
    assert descriptor["ceiling_comparison_performed"] is False
    for section in ("capabilities", "readiness", "authority", "claims"):
        assert _all_false(descriptor[section])
    assert (
        contract.PINNED_QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        contract.matched_v3_qualification_storage_boundary_contract_descriptor_sha256()
        == EXPECTED_DESCRIPTOR_SHA256
    )
    assert contract.parse_matched_v3_qualification_storage_boundary_contract_descriptor(
        contract.canonical_matched_v3_qualification_storage_boundary_contract_descriptor_bytes()
    ) == descriptor


def test_descriptor_freezes_acyclic_preterminal_seal_chronology() -> None:
    descriptor = contract.matched_v3_qualification_storage_boundary_contract_descriptor()
    assert descriptor["artifact_chain"] == [
        "pre_go_storage_boundary_intent",
        "host_request_v2_binds_storage_intent",
        "host_intent_v2_binds_request",
        "host_ready_v2_binds_storage_intent",
        "host_ready_v2_binds_host_intent_v2",
        "host_observer_anchor_v2_binds_ready_v2",
        "host_go_v2",
        "fresh_case_storage_boundary",
        "case_execution_and_native_publication",
        "normalized_wrapper_expected_reload_commitment",
        "actual_publication_reload_validation",
        "irreversible_preterminal_write_quiescence_seal",
        "storage_boundary_receipt",
        "terminal_v2_binds_storage_receipt",
        "lifecycle_and_host_success_v2",
        "full_resource_merger",
    ]
    assert descriptor["intent_precedes_go"] is True
    assert descriptor["receipt_precedes_terminal_v2"] is True
    assert descriptor["terminal_v2_binds_receipt_one_way"] is True
    assert descriptor["reverse_receipt_pins_forbidden"] == [
        "terminal_v2",
        "lifecycle_v2",
        "host_success_v2",
        "full_resource_merger",
    ]


def test_descriptor_freezes_worker_relay_channel_and_logging_seal() -> None:
    descriptor = contract.matched_v3_qualification_storage_boundary_contract_descriptor()
    assert descriptor["publication_reload_validation_schema_version"] == (
        contract.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION
    )
    assert "preseal_reload_validation_schema_version" not in descriptor
    seal = descriptor["preterminal_write_seal_contract"]
    assert seal["allowed_architectures"] == [
        "worker_exit_then_isolated_terminal_relay",
        "irreversible_seccomp_fd_closure_then_isolated_terminal_relay",
    ]
    assert seal["only_trusted_terminal_relay_retains_terminal_transport_capability"] is True
    assert seal["inert_seccomp_closed_worker_may_remain"] is True
    assert seal["terminal_relay_input_policy_restricted_to_receipt_identity"] is True
    assert seal["nonstorage_channel_ready_for_post_receipt_terminal"] is True
    assert seal["receipt_delivery_observed_in_storage_receipt"] is False
    assert seal["terminal_emission_observed_in_storage_receipt"] is False
    assert seal["terminal_v2_proves_post_receipt_delivery_and_emission"] is True
    assert seal["publication_reload_validation_body_replayed"] is True
    assert seal["publication_reload_validation_file_replayed"] is True
    assert seal["reload_performed"] is True
    assert seal["reload_read_only"] is True
    assert seal["reload_validated_before_seal"] is True
    assert seal["container_log_driver"] == "none"
    assert seal["container_logging_can_write_measured_storage"] is False
    assert seal["terminal_transport_inside_measured_storage"] is False
    assert seal["teardown_is_deletion_only"] is True
    assert seal["missing_seal_or_no_later_writer_proof_fails_closed"] is True


def test_source_import_and_api_surface_is_stdlib_only_and_nonoperational() -> None:
    tree = ast.parse(inspect.getsource(contract))
    imported_roots: set[str] = set()
    public_functions: set[str] = set()
    forbidden_calls: set[str] = set()
    forbidden_attributes: set[str] = set()
    operational_prefixes = (
        "attach_",
        "connect_",
        "create_",
        "enforce_",
        "execute_",
        "install_",
        "invoke_",
        "issue_",
        "mount_",
        "open_",
        "publish_",
        "qualify_",
        "read_",
        "run_",
        "sample_",
        "spawn_",
        "write_",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public_functions.add(node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "compile", "__import__"}:
                forbidden_calls.add(node.func.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in {
                "fork",
                "mount",
                "open",
                "remove",
                "rmdir",
                "socket",
                "spawn",
                "system",
                "unlink",
            }:
                forbidden_attributes.add(node.attr)
    assert imported_roots == {
        "__future__",
        "collections",
        "copy",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "types",
        "typing",
    }
    assert imported_roots.isdisjoint(
        {
            "asyncio",
            "docker",
            "jax",
            "numpy",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "tempfile",
            "threading",
            "time",
            "urllib",
        }
    )
    assert forbidden_calls == set()
    assert forbidden_attributes == set()
    assert not {name for name in public_functions if name.startswith(operational_prefixes)}


@pytest.mark.parametrize("case_ordinal", [0, 14, 23, 24, 27])
def test_intent_and_receipt_round_trip_with_caller_file_pins(case_ordinal: int) -> None:
    intent = _intent(case_ordinal)
    intent_raw = contract.canonical_matched_v3_qualification_storage_boundary_intent_bytes(intent)
    parsed_intent = contract.parse_matched_v3_qualification_storage_boundary_intent(
        intent_raw,
        expected_file_sha256=hashlib.sha256(intent_raw).hexdigest(),
    )
    assert parsed_intent == intent
    receipt = _receipt(case_ordinal, intent=intent)
    receipt_raw = contract.canonical_matched_v3_qualification_storage_boundary_receipt_bytes(
        receipt
    )
    parsed_receipt = contract.parse_matched_v3_qualification_storage_boundary_receipt(
        receipt_raw,
        expected_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )
    assert parsed_receipt == receipt
    contract.validate_matched_v3_qualification_storage_boundary_chain(
        parsed_intent,
        parsed_receipt,
    )


def test_intent_and_receipt_are_frozen_dataclasses_with_immutable_tuples() -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.candidate_id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.candidate_id = "other"  # type: ignore[misc]
    assert type(intent.resource_fields) is tuple
    assert type(intent.storage_roots) is tuple
    assert type(intent.field_policy) is tuple
    assert type(receipt.fields) is tuple


def test_receipt_reports_exactly_fields_24_and_25() -> None:
    receipt = _receipt()
    assert tuple((item.field_position, item.field_name) for item in receipt.fields) == (
        (24, "max_temporary_peak_bytes"),
        (25, "max_disk_peak_bytes"),
    )
    assert receipt.fields[0].value_semantics == contract.EXACT_OBSERVATION
    assert receipt.fields[1].value_semantics == (contract.CONSERVATIVE_ENFORCED_UPPER_BOUND)


def test_pre_go_intent_binds_required_spines_and_execution_storage_identity() -> None:
    intent = _intent()
    assert intent.campaign_spine_sha256 == _sha("campaign-spine")
    assert intent.case_spine_sha256 == _sha("case-spine-0")
    assert intent.resource_requirement_body_sha256 == _sha("resource-requirement-0")
    assert intent.resource_fields == contract.RESOURCE_FIELDS
    assert intent.resource_field_order_sha256 == contract.RESOURCE_FIELD_ORDER_SHA256
    assert intent.image_id.startswith("sha256:")
    assert intent.runtime_identity.runtime_candidate.schema_version == (
        contract.RUNTIME_CANDIDATE_SCHEMA_VERSION
    )
    assert intent.container_identity.container_identity_precommitted is True
    assert intent.container_identity.mount_namespace_mutation_disabled_after_go is True
    assert intent.storage_root_inventory_sha256 == contract.storage_root_inventory_sha256(
        intent.storage_roots,
        intent.container_identity,
    )
    assert intent.measurement_producer.descriptor_schema_version == (
        contract.QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
    )
    assert intent.intent_committed_before_go is True
    assert intent.go_identity_bound_in_intent is False


@pytest.mark.parametrize("field", ["resource_fields", "resource_field_order_sha256"])
def test_intent_rejects_nonexact_28_field_identity_or_order(field: str) -> None:
    intent = _intent()
    replacement: object
    if field == "resource_fields":
        values = list(contract.RESOURCE_FIELDS)
        values[23], values[24] = values[24], values[23]
        replacement = tuple(values)
    else:
        replacement = _sha("wrong-field-order")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(intent, **{field: replacement})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("case_spine_sha256", "not-a-hash"),
        ("candidate_id", "search_oracle"),
        ("candidate_family", "external"),
        ("qualification_case_id", "qualification_wrong"),
        ("image_id", "latest"),
        ("intent_committed_before_go", False),
        ("go_identity_bound_in_intent", True),
    ],
)
def test_intent_fails_closed_on_identity_or_chronology_drift(
    field: str,
    replacement: object,
) -> None:
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(_intent(), **{field: replacement})  # type: ignore[arg-type]


def test_intent_rejects_wrong_storage_measurement_producer_schema() -> None:
    intent = _intent()
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            intent.measurement_producer,
            descriptor_schema_version=(
                contract.QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION
            ),
        )


def test_reported_direct_fields_reject_subclasses_and_equality_proxies() -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    event = receipt.fields[0].event_complete_evidence
    quota = receipt.fields[1].quota_enforcement_evidence
    absent_receipt = _receipt(temporary_absent=True)
    absence = absent_receipt.fields[0].structural_absence_evidence
    assert event is not None
    assert quota is not None
    assert absence is not None
    subclassed_resource_fields = tuple(
        _StringSubclass(item) if index == 0 else item
        for index, item in enumerate(intent.resource_fields)
    )
    cases: tuple[tuple[object, str, object], ...] = (
        (intent, "case_ordinal", _IntSubclass(intent.case_ordinal)),
        (intent, "candidate_id", _StringSubclass(intent.candidate_id)),
        (intent, "candidate_family", _StringSubclass(intent.candidate_family)),
        (
            intent,
            "qualification_case_id",
            _StringSubclass(intent.qualification_case_id),
        ),
        (intent, "schema_version", _StringSubclass(intent.schema_version)),
        (
            intent,
            "resource_field_order_sha256",
            _StringSubclass(intent.resource_field_order_sha256),
        ),
        (intent, "resource_fields", _TupleSubclass(intent.resource_fields)),
        (intent, "resource_fields", subclassed_resource_fields),
        (
            intent,
            "storage_root_inventory_sha256",
            _StringSubclass(intent.storage_root_inventory_sha256),
        ),
        (
            intent,
            "field_policy_inventory_sha256",
            _StringSubclass(intent.field_policy_inventory_sha256),
        ),
        (
            intent.measurement_producer,
            "descriptor_schema_version",
            _StringSubclass(intent.measurement_producer.descriptor_schema_version),
        ),
        (
            intent.seal_architecture.terminal_relay,
            "descriptor_schema_version",
            _StringSubclass(intent.seal_architecture.terminal_relay.descriptor_schema_version),
        ),
        (intent.seal_architecture, "architecture_kind", _EqualityProxy()),
        (
            intent.seal_architecture,
            "container_log_driver",
            _StringSubclass(intent.seal_architecture.container_log_driver),
        ),
        (
            intent.container_identity,
            "rootfs_copy_up_policy",
            _StringSubclass(intent.container_identity.rootfs_copy_up_policy),
        ),
        (
            intent.storage_roots[0],
            "field_name",
            _StringSubclass(intent.storage_roots[0].field_name),
        ),
        (
            intent.field_policy[0],
            "field_name",
            _StringSubclass(intent.field_policy[0].field_name),
        ),
        (
            intent.field_policy[0],
            "measurement_mode",
            _StringSubclass(intent.field_policy[0].measurement_mode),
        ),
        (
            intent.field_policy[0],
            "value_semantics",
            _StringSubclass(intent.field_policy[0].value_semantics),
        ),
        (
            intent.field_policy[0],
            "lifetime_scope",
            _StringSubclass(intent.field_policy[0].lifetime_scope),
        ),
        (event, "field_name", _StringSubclass(event.field_name)),
        (quota, "enforcement_kind", _StringSubclass(quota.enforcement_kind)),
        (quota, "field_name", _StringSubclass(quota.field_name)),
        (absence, "field_name", _StringSubclass(absence.field_name)),
        (absence, "absence_kind", _StringSubclass(absence.absence_kind)),
        (
            receipt.fields[0],
            "field_name",
            _StringSubclass(receipt.fields[0].field_name),
        ),
        (
            receipt.fields[0],
            "measurement_mode",
            _StringSubclass(receipt.fields[0].measurement_mode),
        ),
        (
            receipt.fields[0],
            "value_semantics",
            _StringSubclass(receipt.fields[0].value_semantics),
        ),
        (
            receipt.fields[0],
            "lifetime_scope",
            _StringSubclass(receipt.fields[0].lifetime_scope),
        ),
        (
            receipt.fields[0],
            "structural_absence_kind",
            _StringSubclass(receipt.fields[0].structural_absence_kind),
        ),
        (
            receipt,
            "field_inventory_sha256",
            _StringSubclass(receipt.field_inventory_sha256),
        ),
    )
    for target, field, replacement in cases:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(  # type: ignore[type-var]
                target,
                **{field: replacement},
            )


def test_both_allowed_modes_have_exact_frozen_semantics() -> None:
    policies = _intent().field_policy
    assert policies[0].measurement_mode == contract.EVENT_COMPLETE_ACCOUNTING_MODE
    assert policies[0].value_semantics == contract.EXACT_OBSERVATION
    assert policies[0].event_accounting_policy_body_sha256 is not None
    assert policies[0].quota_enforcement_policy_body_sha256 is None
    assert policies[0].hard_limit_bytes is None
    assert policies[1].measurement_mode == contract.HARD_QUOTA_ENFORCEMENT_MODE
    assert policies[1].value_semantics == contract.CONSERVATIVE_ENFORCED_UPPER_BOUND
    assert policies[1].event_accounting_policy_body_sha256 is None
    assert policies[1].quota_enforcement_policy_body_sha256 is not None
    assert policies[1].hard_limit_bytes == 8192


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("polling_or_sampling_sufficient", True),
        ("du_snapshot_sufficient", True),
        ("container_layer_size_sufficient", True),
        ("missing_value_defaults_to_zero", True),
        ("committed_before_go", False),
    ],
)
def test_policy_rejects_sampling_snapshots_layer_size_missing_zero_and_late_commit(
    field: str,
    replacement: object,
) -> None:
    policy = _intent().field_policy[0]
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(policy, **{field: replacement})  # type: ignore[arg-type]


def test_event_mode_rejects_quota_semantics_or_quota_fields() -> None:
    policy = _intent().field_policy[0]
    changes = (
        {"value_semantics": contract.CONSERVATIVE_ENFORCED_UPPER_BOUND},
        {"quota_enforcement_policy_body_sha256": _sha("quota")},
        {"hard_limit_bytes": 1},
        {"event_accounting_policy_body_sha256": None},
    )
    for change in changes:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(policy, **change)


def test_quota_mode_rejects_exact_semantics_missing_limit_or_event_policy() -> None:
    policy = _intent().field_policy[1]
    changes = (
        {"value_semantics": contract.EXACT_OBSERVATION},
        {"quota_enforcement_policy_body_sha256": None},
        {"hard_limit_bytes": 0},
        {"event_accounting_policy_body_sha256": _sha("event")},
    )
    for change in changes:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(policy, **change)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("accounting_started_before_go", False),
        ("accounting_closed_at_irreversible_seal", False),
        ("all_bound_roots_covered", False),
        ("allocation_and_deallocation_events_complete", False),
        ("filesystem_copy_up_events_complete", False),
        ("deleted_open_file_events_complete", False),
        ("event_loss_count", 1),
        ("event_overflow_count", 1),
        ("polling_or_sampling_used", True),
        ("du_snapshot_used", True),
        ("container_layer_size_used", True),
        ("exact_high_water_replayed", False),
    ],
)
def test_event_complete_mode_rejects_incomplete_lossy_or_sampled_evidence(
    field: str,
    replacement: object,
) -> None:
    evidence = _receipt().fields[0].event_complete_evidence
    assert evidence is not None
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(evidence, **{field: replacement})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("installed_before_go", False),
        ("immutable_through_container_removal", False),
        ("non_bypass_through_container_removal", False),
        ("all_bound_roots_covered", False),
        ("hard_limit_applies_to_aggregate_union", False),
        ("alternate_writable_mount_count", 1),
        ("alternate_writable_path_count", 1),
        ("overlay_copy_up_outside_boundary_possible", True),
        ("descendant_bypass_possible", True),
        ("quota_breached", True),
        ("breach_count", 1),
        ("breach_status_final_at_seal", False),
        ("polling_or_sampling_used", True),
        ("du_snapshot_used", True),
        ("container_layer_size_used", True),
    ],
)
def test_quota_mode_rejects_bypass_breach_or_sampling_evidence(
    field: str,
    replacement: object,
) -> None:
    evidence = _receipt().fields[1].quota_enforcement_evidence
    assert evidence is not None
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(evidence, **{field: replacement})  # type: ignore[arg-type]


def test_all_zero_counts_and_sequences_reject_boolean_integer_aliases() -> None:
    receipt = _receipt()
    event = receipt.fields[0].event_complete_evidence
    quota = receipt.fields[1].quota_enforcement_evidence
    assert event is not None
    assert quota is not None
    cases: tuple[tuple[object, str], ...] = (
        (receipt.writable_surface_evidence, "unbound_writable_mount_count"),
        (receipt.writable_surface_evidence, "unbound_writable_path_count"),
        (event, "fresh_boundary_sequence"),
        (event, "event_loss_count"),
        (event, "event_overflow_count"),
        (quota, "alternate_writable_mount_count"),
        (quota, "alternate_writable_path_count"),
        (quota, "breach_count"),
    )
    for value, field in cases:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(value, **{field: False})  # type: ignore[type-var]


def test_positive_integer_fields_reject_boolean_aliases() -> None:
    receipt = _receipt()
    event = receipt.fields[0].event_complete_evidence
    quota = receipt.fields[1].quota_enforcement_evidence
    assert event is not None
    assert quota is not None
    cases: tuple[tuple[object, str], ...] = (
        (receipt.fields[0], "field_position"),
        (receipt.fields[0], "observed_value"),
        (event, "replayed_peak_bytes"),
        (event, "seal_boundary_sequence"),
        (quota, "hard_limit_bytes"),
    )
    for value, field in cases:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(value, **{field: True})  # type: ignore[type-var]


def test_quota_measurement_value_must_equal_pre_go_hard_limit() -> None:
    measurement = _receipt().fields[1]
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(measurement, observed_value=8191)


def test_event_measurement_value_must_equal_replayed_aggregate_peak() -> None:
    receipt = _receipt()
    event = receipt.fields[0].event_complete_evidence
    assert event is not None
    wrong_event = dataclasses.replace(event, replayed_peak_bytes=event.replayed_peak_bytes + 1)
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            receipt.fields[0],
            event_complete_evidence=wrong_event,
        )


def test_event_evidence_rejects_field_or_field_root_union_substitution() -> None:
    receipt = _receipt()
    event = receipt.fields[0].event_complete_evidence
    assert event is not None
    wrong_events = (
        dataclasses.replace(event, field_name="max_disk_peak_bytes"),
        dataclasses.replace(event, field_root_ids=()),
        dataclasses.replace(
            event,
            field_root_inventory_sha256=_sha("wrong-event-field-root-inventory"),
        ),
    )
    for wrong_event in wrong_events:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            fields = (
                dataclasses.replace(
                    receipt.fields[0],
                    event_complete_evidence=wrong_event,
                ),
                receipt.fields[1],
            )
            contract.storage_measurement_inventory_sha256(
                fields,
                receipt.field_policy,
                receipt.storage_roots,
                receipt.storage_root_inventory_sha256,
                receipt.writable_surface_evidence,
                receipt.write_quiescence_seal,
            )


def test_quota_measurement_constructor_rejects_evidence_field_substitution() -> None:
    receipt = _receipt()
    quota = receipt.fields[1].quota_enforcement_evidence
    assert quota is not None
    wrong_quota = dataclasses.replace(
        quota,
        field_name="max_temporary_peak_bytes",
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            receipt.fields[1],
            quota_enforcement_evidence=wrong_quota,
        )


def test_event_replay_requires_simultaneous_aggregate_union_semantics() -> None:
    event = _receipt().fields[0].event_complete_evidence
    assert event is not None
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            event,
            replayed_peak_is_simultaneous_aggregate_union_high_water=False,
        )


def test_exact_zero_requires_typed_structural_absence_and_no_field_root() -> None:
    receipt = _receipt(temporary_absent=True)
    zero = receipt.fields[0]
    assert zero.observed_value == 0
    assert zero.measurement_mode == contract.EVENT_COMPLETE_ACCOUNTING_MODE
    assert zero.structural_absence_kind == (contract.TEMPORARY_STORAGE_STRUCTURALLY_ABSENT)
    assert zero.structural_absence_evidence is not None
    assert not {
        item.root_id
        for item in receipt.storage_roots
        if item.field_name == "max_temporary_peak_bytes"
    }


def test_zero_rejects_missing_wrong_or_untyped_absence() -> None:
    zero = _receipt(temporary_absent=True).fields[0]
    changes = (
        {"structural_absence_evidence": None},
        {"structural_absence_kind": contract.NOT_ABSENT},
        {"measurement_mode": contract.HARD_QUOTA_ENFORCEMENT_MODE},
    )
    for change in changes:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(zero, **change)


def test_zero_rejects_existing_bound_field_root() -> None:
    intent = _intent()
    surface = _surface(intent)
    seal = _seal(intent, surface)
    fields = list(_fields(intent, surface, seal, temporary_absent=False))
    event = fields[0].event_complete_evidence
    assert event is not None
    zero_event = dataclasses.replace(event, replayed_peak_bytes=0)
    fields[0] = dataclasses.replace(
        fields[0],
        observed_value=0,
        event_complete_evidence=zero_event,
        structural_absence_kind=contract.TEMPORARY_STORAGE_STRUCTURALLY_ABSENT,
        structural_absence_evidence=contract.StructuralAbsenceEvidenceV1(
            field_name="max_temporary_peak_bytes",
            absence_kind=contract.TEMPORARY_STORAGE_STRUCTURALLY_ABSENT,
            namespace_inventory_body_sha256=surface.writable_path_inventory_body_sha256,
            absence_proof_body_sha256=_sha("false-absence"),
            no_bound_storage_root_for_field=True,
            no_writable_mount_for_field=True,
            no_writable_path_for_field=True,
            no_overlay_copy_up_target_for_field=True,
        ),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.storage_measurement_inventory_sha256(
            tuple(fields),
            intent.field_policy,
            intent.storage_roots,
            intent.storage_root_inventory_sha256,
            surface,
            seal,
        )


def test_positive_value_rejects_no_bound_field_root() -> None:
    intent = _intent(temporary_absent=True)
    surface = _surface(intent)
    seal = _seal(intent, surface)
    fields = list(_fields(intent, surface, seal, temporary_absent=True))
    event = fields[0].event_complete_evidence
    assert event is not None
    positive_event = dataclasses.replace(event, replayed_peak_bytes=1)
    fields[0] = dataclasses.replace(
        fields[0],
        observed_value=1,
        event_complete_evidence=positive_event,
        structural_absence_kind=contract.NOT_ABSENT,
        structural_absence_evidence=None,
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.storage_measurement_inventory_sha256(
            tuple(fields),
            intent.field_policy,
            intent.storage_roots,
            intent.storage_root_inventory_sha256,
            surface,
            seal,
        )


def test_missing_field_cannot_default_to_zero() -> None:
    value = _receipt().to_dict()
    value["fields"].pop(0)
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("all_writable_mounts_bound", False),
        ("all_writable_paths_beneath_bound_roots", False),
        ("unbound_writable_mount_count", 1),
        ("unbound_writable_path_count", 1),
        ("alternate_writable_mounts_possible", True),
        ("alternate_writable_paths_possible", True),
        ("rootfs_copy_up_bound_or_impossible", False),
        ("deleted_open_files_accounted_or_impossible", False),
        ("anonymous_files_accounted_or_impossible", False),
        ("memory_backed_files_accounted_or_impossible", False),
        ("mount_namespace_mutation_disabled", False),
        ("descendant_mount_creation_disabled", False),
        ("host_path_write_escape_disabled", False),
        ("device_write_escape_disabled", False),
        ("network_storage_write_escape_disabled", False),
        ("inherited_writable_fd_escape_disabled", False),
    ],
)
def test_writable_surface_requires_no_alternate_mount_path_or_escape(
    field: str,
    replacement: object,
) -> None:
    evidence = _surface(_intent())
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(evidence, **{field: replacement})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("publication_committed_before_seal", False),
        ("reload_validated_before_seal", False),
        ("reload_performed", False),
        ("reload_read_only", False),
        ("write_quiescence_irreversible", False),
        ("container_writes_disabled", False),
        ("descendant_writes_disabled", False),
        ("later_allocation_possible", True),
        ("later_copy_up_possible", True),
        ("later_peak_increase_possible", True),
        ("terminal_transport_outside_measured_storage", False),
        ("terminal_transport_can_allocate_measured_storage", True),
        ("worker_has_measured_writable_namespace", True),
        ("worker_has_measured_writable_fd", True),
        (
            "only_trusted_terminal_relay_retains_terminal_transport_capability",
            False,
        ),
        ("terminal_relay_input_policy_restricted_to_receipt_identity", False),
        ("nonstorage_channel_ready_for_post_receipt_terminal", False),
        ("container_log_driver_none", False),
        ("container_logging_write_possible", True),
        ("no_later_writer_exists", False),
        ("teardown_deletion_only", False),
        ("teardown_can_increase_measured_usage", True),
        ("peak_is_whole_fresh_case_lifetime_peak", False),
    ],
)
def test_write_seal_requires_no_later_writer_nonstorage_terminal_and_deletion_only_teardown(
    field: str,
    replacement: object,
) -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(seal, **{field: replacement})  # type: ignore[arg-type]


def test_write_seal_rejects_actual_reload_different_from_wrapper_commitment() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            seal,
            actual_reload_observation_sha256=_sha("different-actual-reload"),
        )


def test_publication_reload_validation_uses_exact_independent_body_and_file_encoding() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    body = {
        "schema_version": contract.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "publication_commitment_wrapper_file_sha256": (
            seal.publication_commitment_wrapper_file_sha256
        ),
        "publication_commitment_wrapper_body_sha256": (
            seal.publication_commitment_wrapper_body_sha256
        ),
        "expected_reload_observation_sha256": seal.expected_reload_observation_sha256,
        "actual_reload_observation_sha256": seal.actual_reload_observation_sha256,
        "reload_performed": seal.reload_performed,
        "reload_read_only": seal.reload_read_only,
    }
    assert set(body) == {
        "schema_version",
        "publication_commitment_wrapper_file_sha256",
        "publication_commitment_wrapper_body_sha256",
        "expected_reload_observation_sha256",
        "actual_reload_observation_sha256",
        "reload_performed",
        "reload_read_only",
    }
    body_sha256 = _body(body)
    full_file = {**body, "reload_validation_body_sha256": body_sha256}
    assert set(full_file) == {*body, "reload_validation_body_sha256"}
    assert seal.publication_reload_validation.body_sha256 == body_sha256
    assert (
        seal.publication_reload_validation.file_sha256
        == hashlib.sha256(_canonical(full_file)).hexdigest()
    )


def test_publication_reload_validation_rejects_field_and_bool_alias_substitution() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    substituted_observation = _sha("substituted-reload-observation")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            seal,
            expected_reload_observation_sha256=substituted_observation,
            actual_reload_observation_sha256=substituted_observation,
        )
    for field in ("reload_performed", "reload_read_only"):
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(seal, **{field: 1})  # type: ignore[arg-type]


def test_write_seal_rejects_wrapper_or_reload_validation_artifact_substitution() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    substituted_wrapper = _artifact(
        contract.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "substituted-wrapper",
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(seal, publication_commitment=substituted_wrapper)
    wrong_schema = dataclasses.replace(
        seal.publication_reload_validation,
        schema_version="alberta.forager_matched_v3.unrelated_reload_validation.v1",
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(seal, publication_reload_validation=wrong_schema)
    wrong_body = dataclasses.replace(
        seal.publication_reload_validation,
        body_sha256=_sha("substituted-reload-validation-body"),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(seal, publication_reload_validation=wrong_body)
    wrong_file = dataclasses.replace(
        seal.publication_reload_validation,
        file_sha256=_sha("substituted-reload-validation-file"),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(seal, publication_reload_validation=wrong_file)


def test_seccomp_seal_allows_inert_worker_but_only_relay_retains_transport() -> None:
    intent = _intent(
        architecture_kind=("irreversible_seccomp_fd_closure_then_isolated_terminal_relay")
    )
    seal = _seal(intent, _surface(intent))
    assert seal.worker_exit_observed is False
    assert seal.irreversible_seccomp_fd_closure_observed is True
    assert seal.worker_has_measured_writable_namespace is False
    assert seal.worker_has_measured_writable_fd is False
    assert seal.only_trusted_terminal_relay_retains_terminal_transport_capability is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("container_log_driver", "json-file"),
        ("architecture_committed_before_go", False),
        ("worker_and_terminal_relay_separated_before_go", False),
        ("terminal_relay_has_measured_writable_namespace", True),
        ("terminal_relay_has_measured_writable_fd", True),
        ("terminal_transport_uses_nonstorage_channel", False),
        ("container_logging_can_write_measured_storage", True),
        ("late_remount_used_or_permitted", True),
    ],
)
def test_pre_go_seal_architecture_rejects_logging_late_remount_or_writable_relay(
    field: str,
    replacement: object,
) -> None:
    architecture = _architecture()
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(architecture, **{field: replacement})  # type: ignore[arg-type]


def test_write_seal_requires_exactly_one_precommitted_worker_quiescence_mechanism() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    for worker_exit, seccomp in ((False, False), (True, True)):
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(
                seal,
                worker_exit_observed=worker_exit,
                irreversible_seccomp_fd_closure_observed=seccomp,
            )


def test_seccomp_fd_closure_architecture_round_trips_and_matches_seal() -> None:
    intent = _intent(
        architecture_kind=("irreversible_seccomp_fd_closure_then_isolated_terminal_relay")
    )
    receipt = _receipt(intent=intent, architecture_kind=intent.seal_architecture.architecture_kind)
    assert receipt.write_quiescence_seal.worker_exit_observed is False
    assert receipt.write_quiescence_seal.irreversible_seccomp_fd_closure_observed is True
    contract.validate_matched_v3_qualification_storage_boundary_chain(intent, receipt)


def test_receipt_rejects_seal_mechanism_different_from_pre_go_architecture() -> None:
    receipt = _receipt()
    wrong_seal = dataclasses.replace(
        receipt.write_quiescence_seal,
        worker_exit_observed=False,
        irreversible_seccomp_fd_closure_observed=True,
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(receipt, write_quiescence_seal=wrong_seal)


def test_receipt_rejects_seal_architecture_identity_drift() -> None:
    receipt = _receipt()
    wrong_seal = dataclasses.replace(
        receipt.write_quiescence_seal,
        seal_architecture_body_sha256=_sha("wrong-architecture"),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(receipt, write_quiescence_seal=wrong_seal)


def test_storage_roots_reject_wrong_order_duplicates_overlap_and_unbound_copy_up() -> None:
    intent = _intent()
    roots = intent.storage_roots
    cases = (
        tuple(reversed(roots)),
        (roots[0], roots[0]),
        (
            roots[0],
            dataclasses.replace(
                roots[1],
                absolute_path=f"{roots[0].absolute_path}/nested",
            ),
        ),
        (
            dataclasses.replace(roots[0], includes_overlay_copy_up=False),
            roots[1],
        ),
    )
    for candidate in cases:
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            contract.storage_root_inventory_sha256(candidate, intent.container_identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("created_exclusively_for_case", False),
        ("empty_at_fresh_boundary", False),
        ("writable", False),
        ("absolute_path", "relative/path"),
        ("absolute_path", "/case/../escape"),
    ],
)
def test_storage_root_requires_fresh_exclusive_canonical_writable_boundary(
    field: str,
    replacement: object,
) -> None:
    root = _intent().storage_roots[0]
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(root, **{field: replacement})  # type: ignore[arg-type]


def test_policy_root_projection_must_equal_exact_root_inventory() -> None:
    intent = _intent()
    policies = list(intent.field_policy)
    policies[0] = dataclasses.replace(policies[0], root_ids=())
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.storage_field_policy_inventory_sha256(
            tuple(policies),
            intent.storage_roots,
        )


def test_multi_root_quota_is_one_aggregate_union_bound_not_per_root_limits() -> None:
    base = _intent()
    disk = next(item for item in base.storage_roots if item.field_name == "max_disk_peak_bytes")
    auxiliary = dataclasses.replace(
        disk,
        root_id="disk_aux_root",
        absolute_path="/case/disk-aux",
        mount_identity_sha256=_sha("disk-aux-mount"),
        backing_store_identity_sha256=_sha("disk-aux-store"),
        includes_overlay_copy_up=False,
    )
    roots = tuple(sorted((*base.storage_roots, auxiliary), key=lambda item: item.root_id))
    policies = _policies(roots)
    intent = dataclasses.replace(
        base,
        storage_root_inventory_sha256=contract.storage_root_inventory_sha256(
            roots,
            base.container_identity,
        ),
        storage_roots=roots,
        field_policy_inventory_sha256=contract.storage_field_policy_inventory_sha256(
            policies,
            roots,
        ),
        field_policy=policies,
    )
    receipt = _receipt(intent=intent)
    quota = receipt.fields[1].quota_enforcement_evidence
    assert quota is not None
    assert quota.field_root_ids == ("disk_aux_root", "disk_root")
    assert quota.hard_limit_applies_to_aggregate_union is True
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(quota, hard_limit_applies_to_aggregate_union=False)
    omitted = dataclasses.replace(
        quota,
        field_root_ids=("disk_root",),
        field_root_inventory_sha256=_sha("incomplete-disk-root-union"),
    )
    fields = (
        receipt.fields[0],
        dataclasses.replace(receipt.fields[1], quota_enforcement_evidence=omitted),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.storage_measurement_inventory_sha256(
            fields,
            receipt.field_policy,
            receipt.storage_roots,
            receipt.storage_root_inventory_sha256,
            receipt.writable_surface_evidence,
            receipt.write_quiescence_seal,
        )


def test_receipt_requires_exact_preexisting_request_intent_ready_go_chain() -> None:
    chain = _boundary_chain()
    assert chain.host_case_request.schema_version == contract.HOST_CASE_REQUEST_SCHEMA_VERSION
    assert chain.host_case_intent.schema_version == contract.HOST_CASE_INTENT_SCHEMA_VERSION
    assert chain.host_ready.schema_version == contract.HOST_READY_SCHEMA_VERSION
    assert chain.host_observer_anchor.schema_version == (
        contract.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION
    )
    assert chain.host_go.schema_version == contract.HOST_GO_SCHEMA_VERSION
    for field in (
        "host_intent_request_file_sha256",
        "host_intent_request_body_sha256",
        "ready_host_intent_file_sha256",
        "ready_host_intent_body_sha256",
        "observer_anchor_ready_file_sha256",
        "observer_anchor_ready_body_sha256",
        "go_ready_file_sha256",
        "go_ready_body_sha256",
        "go_observer_anchor_file_sha256",
        "go_observer_anchor_body_sha256",
    ):
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            dataclasses.replace(
                chain,
                **{field: _sha(f"wrong-{field}")},  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "field",
    [
        "request_storage_intent_file_sha256",
        "request_storage_intent_body_sha256",
        "ready_storage_intent_file_sha256",
        "ready_storage_intent_body_sha256",
    ],
)
def test_collection_boundary_constructor_rejects_request_or_ready_storage_intent_cross_wire(
    field: str,
) -> None:
    chain = _boundary_chain()
    projection = chain.to_dict()
    projection.pop("handshake_chain_body_sha256")
    projection[field] = _sha(f"wrong-{field}")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            chain,
            **{field: _sha(f"wrong-{field}")},  # type: ignore[arg-type]
            handshake_chain_body_sha256=_body(projection),
        )


def test_handshake_artifact_identities_cannot_alias() -> None:
    chain = _boundary_chain()
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(chain, host_go=chain.host_ready)


def test_handshake_chain_body_digest_must_replay_exact_projections() -> None:
    chain = _boundary_chain()
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(chain, handshake_chain_body_sha256=_sha("arbitrary-handshake"))


@pytest.mark.parametrize(
    "field",
    [
        "campaign_spine_sha256",
        "case_spine_sha256",
        "resource_requirement_body_sha256",
        "resource_field_order_sha256",
        "writable_mount_policy_body_sha256",
        "writable_path_policy_body_sha256",
        "storage_root_inventory_sha256",
        "field_policy_inventory_sha256",
    ],
)
def test_chain_rejects_intent_projection_cross_wires(field: str) -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        cross_wired = dataclasses.replace(
            receipt,
            **{field: _sha(f"wrong-{field}")},  # type: ignore[arg-type]
        )
        contract.validate_matched_v3_qualification_storage_boundary_chain(
            intent,
            cross_wired,
        )


def test_chain_rejects_wrong_intent_file_or_body_identity() -> None:
    intent = _intent()
    receipt = _receipt(intent=intent)
    for field in ("file_sha256", "body_sha256"):
        identity = dataclasses.replace(
            receipt.measurement_intent,
            **{field: _sha(f"wrong-intent-{field}")},
        )
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            cross_wired = dataclasses.replace(receipt, measurement_intent=identity)
            contract.validate_matched_v3_qualification_storage_boundary_chain(
                intent,
                cross_wired,
            )


@pytest.mark.parametrize(
    "reverse_key",
    [
        "terminal_metadata",
        "terminal_receipt",
        "lifecycle_record",
        "host_success_receipt",
        "host_execution_receipt",
        "full_resource_merger_receipt",
        "observation_handoff",
        "terminal_v2",
        "lifecycle_v2",
        "host_success_v2",
        "issuer_receipt",
        "evaluator_receipt",
    ],
)
def test_intent_parser_rejects_reverse_terminal_host_success_or_merger_pins(
    reverse_key: str,
) -> None:
    value = _intent().to_dict()
    value[reverse_key] = _sha(reverse_key)
    raw = _rebody(value, "intent_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_intent(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    "reverse_key",
    [
        "terminal_metadata",
        "terminal_receipt",
        "lifecycle_record",
        "host_success_receipt",
        "host_execution_receipt",
        "full_resource_merger_receipt",
        "observation_handoff",
        "terminal_v2",
        "lifecycle_v2",
        "host_success_v2",
        "issuer_receipt",
        "evaluator_receipt",
    ],
)
def test_receipt_parser_rejects_reverse_terminal_host_success_or_merger_pins(
    reverse_key: str,
) -> None:
    value = _receipt().to_dict()
    value[reverse_key] = _sha(reverse_key)
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_receipt_allows_only_preexisting_handshake_publication_and_seal_identities() -> None:
    value = _receipt().to_dict()
    assert "measurement_intent" in value
    assert "collection_boundary_chain" in value
    assert "publication_commitment" in value["write_quiescence_seal"]
    assert "write_quiescence_seal" in value["write_quiescence_seal"]
    assert "terminal_relay_preseal_attestation" in value["write_quiescence_seal"]
    assert "nonstorage_channel_readiness_attestation" in value["write_quiescence_seal"]
    forbidden_keys = {
        "relay_received_receipt_identity_only",
        "receipt_identity_used_precommitted_nonstorage_channel",
        "terminal_emitted_over_nonstorage_channel",
        "receipt_precedes_terminal_v2",
        "terminal_metadata",
        "terminal_v2",
        "lifecycle_record",
        "lifecycle_v2",
        "host_success_receipt",
        "host_success_v2",
        "full_resource_merger_receipt",
    }
    assert forbidden_keys.isdisjoint(_all_mapping_keys(value))


def test_parser_requires_caller_full_file_pins() -> None:
    intent_raw = contract.canonical_matched_v3_qualification_storage_boundary_intent_bytes(
        _intent()
    )
    receipt_raw = contract.canonical_matched_v3_qualification_storage_boundary_receipt_bytes(
        _receipt()
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_intent(
            intent_raw,
            expected_file_sha256=_sha("wrong-intent-file"),
        )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            receipt_raw,
            expected_file_sha256=_sha("wrong-receipt-file"),
        )


@pytest.mark.parametrize("artifact_kind", ["intent", "receipt"])
def test_parser_rejects_body_digest_tampering(artifact_kind: str) -> None:
    parser: Callable[..., object]
    if artifact_kind == "intent":
        value = _intent().to_dict()
        value["image_id"] = f"sha256:{_sha('different-image')}"
        raw = _canonical(value)
        parser = contract.parse_matched_v3_qualification_storage_boundary_intent
    else:
        value = _receipt().to_dict()
        value["image_id"] = f"sha256:{_sha('different-image')}"
        raw = _canonical(value)
        parser = contract.parse_matched_v3_qualification_storage_boundary_receipt
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        parser(raw, expected_file_sha256=hashlib.sha256(raw).hexdigest())


def test_parser_rejects_field_inventory_digest_tampering() -> None:
    value = _receipt().to_dict()
    value["field_inventory_sha256"] = _sha("wrong-field-inventory")
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_rejects_true_capability_readiness_authority_or_claim() -> None:
    for section in ("capabilities", "readiness", "authority", "claims"):
        value = _receipt().to_dict()
        first_key = next(iter(value[section]))
        value[section][first_key] = True
        raw = _rebody(value, "receipt_body_sha256")
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            contract.parse_matched_v3_qualification_storage_boundary_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_parser_envelope_rejects_integer_zero_alias_for_false() -> None:
    value = _receipt().to_dict()
    first_key = next(iter(value["capabilities"]))
    value["capabilities"][first_key] = 0
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_rejects_boolean_alias_for_integer() -> None:
    value = _receipt().to_dict()
    value["fields"][0]["observed_value"] = True
    raw = _rebody(value, "receipt_body_sha256")
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.parse_matched_v3_qualification_storage_boundary_receipt(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_parser_rejects_float_nan_duplicate_noncanonical_and_non_ascii_bytes() -> None:
    receipt = _receipt()
    value = receipt.to_dict()
    value["fields"][0]["observed_value"] = 1.5
    float_raw = _canonical(value)
    canonical_raw = contract.canonical_matched_v3_qualification_storage_boundary_receipt_bytes(
        receipt
    )
    duplicate_raw = b'{"schema_version":"duplicate",' + canonical_raw[1:]
    nan_raw = canonical_raw.replace(b'"observed_value":4096', b'"observed_value":NaN', 1)
    non_ascii_raw = b'{"candidate_id":"\xc3\xa9"}\n'
    for raw in (
        float_raw,
        duplicate_raw,
        nan_raw,
        canonical_raw + b"\n",
        non_ascii_raw,
    ):
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            contract.parse_matched_v3_qualification_storage_boundary_receipt(
                raw,
                expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_artifact_component_and_producer_identities_reject_zero_hashes() -> None:
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.ArtifactIdentityV1(
            schema_version="alberta.example.v1",
            file_sha256="0" * 64,
            body_sha256=_sha("body"),
        )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.ProducerIdentityV1(
            descriptor_schema_version=(
                contract.QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=_sha("descriptor"),
            source_sha256="0" * 64,
        )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        contract.ComponentIdentityV1(
            descriptor_schema_version=(
                contract.QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256="0" * 64,
            source_sha256=_sha("source"),
        )


def test_storage_intent_rejects_zero_image_id() -> None:
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(_intent(), image_id="sha256:" + "0" * 64)


def test_receipt_rejects_event_quota_or_seal_inventory_cross_wires() -> None:
    receipt = _receipt()
    event = receipt.fields[0].event_complete_evidence
    quota = receipt.fields[1].quota_enforcement_evidence
    assert event is not None
    assert quota is not None
    wrong_evidence = (
        dataclasses.replace(
            event,
            writable_path_inventory_body_sha256=_sha("wrong-event-paths"),
        ),
        dataclasses.replace(
            event,
            write_quiescence_seal_file_sha256=_sha("wrong-event-seal-file"),
        ),
        dataclasses.replace(
            event,
            write_quiescence_seal_body_sha256=_sha("wrong-event-seal-body"),
        ),
    )
    for wrong_event in wrong_evidence:
        fields = (
            dataclasses.replace(receipt.fields[0], event_complete_evidence=wrong_event),
            receipt.fields[1],
        )
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            contract.storage_measurement_inventory_sha256(
                fields,
                receipt.field_policy,
                receipt.storage_roots,
                receipt.storage_root_inventory_sha256,
                receipt.writable_surface_evidence,
                receipt.write_quiescence_seal,
            )
    wrong_quota_evidence = (
        dataclasses.replace(
            quota,
            writable_mount_inventory_body_sha256=_sha("wrong-quota-mounts"),
        ),
        dataclasses.replace(
            quota,
            storage_root_inventory_sha256=_sha("wrong-quota-roots"),
        ),
        dataclasses.replace(
            quota,
            write_quiescence_seal_file_sha256=_sha("wrong-quota-seal-file"),
        ),
        dataclasses.replace(
            quota,
            write_quiescence_seal_body_sha256=_sha("wrong-quota-seal-body"),
        ),
    )
    for wrong_quota in wrong_quota_evidence:
        fields = (
            receipt.fields[0],
            dataclasses.replace(receipt.fields[1], quota_enforcement_evidence=wrong_quota),
        )
        with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
            contract.storage_measurement_inventory_sha256(
                fields,
                receipt.field_policy,
                receipt.storage_roots,
                receipt.storage_root_inventory_sha256,
                receipt.writable_surface_evidence,
                receipt.write_quiescence_seal,
            )
    wrong_seal = dataclasses.replace(
        receipt.write_quiescence_seal,
        sealed_path_inventory_body_sha256=_sha("wrong-sealed-paths"),
    )
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(receipt, write_quiescence_seal=wrong_seal)


def test_write_seal_body_projection_cannot_disconnect_from_artifact_identity() -> None:
    intent = _intent()
    seal = _seal(intent, _surface(intent))
    with pytest.raises(contract.ForagerMatchedV3QualificationStorageBoundaryError):
        dataclasses.replace(
            seal,
            write_quiescence_seal_body_sha256=_sha("disconnected-seal-body"),
        )
