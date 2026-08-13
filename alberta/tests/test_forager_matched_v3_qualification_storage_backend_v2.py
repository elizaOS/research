"""Adversarial tests for the pure matched-v3 storage metadata contract."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_storage_backend_v2 as storage,
)

ERROR = storage.ForagerMatchedV3QualificationStorageBackendV2Error
SOURCE = (
    Path(__file__).parents[1]
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_qualification_storage_backend_v2.py"
)
ZERO_SHA256 = "0" * 64

CleanupOutcome = Literal[
    "committed_receipt_cleaned",
    "failed_before_receipt_cleaned",
    "receipt_commit_uncertain_cleaned",
    "committed_receipt_cleanup_failed",
    "failed_before_receipt_cleanup_failed",
    "receipt_commit_uncertain_cleanup_failed",
]


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(schema: str, label: str) -> storage.ArtifactIdentityV1:
    return storage.ArtifactIdentityV1(
        schema_version=schema,
        file_sha256=_hash(f"{label}:file"),
        body_sha256=_hash(f"{label}:body"),
    )


def _replace_untyped(instance: Any, **changes: Any) -> Any:
    return replace(instance, **changes)


def _components() -> tuple[storage.PinnedStorageComponentIdentityV1, ...]:
    return tuple(
        storage.PinnedStorageComponentIdentityV1(
            role=cast(Any, role),
            descriptor_schema_version=storage.STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS[role],
            descriptor_file_sha256=_hash(f"{role}:descriptor:file"),
            descriptor_body_sha256=_hash(f"{role}:descriptor:body"),
            source_sha256=_hash(f"{role}:source"),
        )
        for role in storage.STORAGE_COMPONENT_ROLES
    )


def _other_component(
    component: storage.PinnedStorageComponentIdentityV1,
    label: str,
) -> storage.PinnedStorageComponentIdentityV1:
    return replace(
        component,
        descriptor_file_sha256=_hash(f"{label}:descriptor:file"),
        descriptor_body_sha256=_hash(f"{label}:descriptor:body"),
        source_sha256=_hash(f"{label}:source"),
    )


def _raw_artifacts(prefix: str = "raw") -> tuple[storage.ArtifactIdentityV1, ...]:
    return tuple(
        _artifact(schema, f"{prefix}-{index}")
        for index, schema in enumerate(storage.RAW_ARTIFACT_SCHEMA_INVENTORY)
    )


def _policy() -> storage.StorageBackendPolicyV1:
    return storage.StorageBackendPolicyV1(
        qualification_plan=_artifact(storage.QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "plan"),
        max_temporary_peak_bytes=4096,
        tmpfs_hard_size_limit_bytes=4096,
        components=_components(),
    )


def _intent(
    policy: storage.StorageBackendPolicyV1,
    raw: tuple[storage.ArtifactIdentityV1, ...],
) -> storage.StorageBoundaryRuntimeIntentV1:
    return storage.StorageBoundaryRuntimeIntentV1(
        campaign_id="campaign_2026q3",
        case_ordinal=0,
        candidate_id="causal_e025_q050",
        candidate_family="local",
        qualification_case_id="qualification_00_causal_e025_q050",
        qualification_plan=policy.qualification_plan,
        policy=storage.storage_backend_policy_identity_v1(policy),
        image_id=f"sha256:{_hash('image')}",
        runtime_qualification_receipt=_artifact(
            storage.RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime-qualification",
        ),
        host_provisioning_v3_validated_pre_go_prefix=_artifact(
            storage.HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
            "host-prefix",
        ),
        container_name="alberta-qualified-case-00",
        container_id_commitment_sha256=_hash("container-id"),
        outer_cgroup_identity_sha256=_hash("outer-cgroup"),
        mount_namespace_identity=_artifact(
            storage.MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION,
            "mount-namespace",
        ),
        rootfs_mount_identity=_artifact(
            storage.ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION,
            "rootfs-mount",
        ),
        tmpfs_mount_identity=raw[0],
        tmpfs_backing_identity=_artifact(
            storage.TMPFS_BACKING_IDENTITY_SCHEMA_VERSION,
            "tmpfs-backing",
        ),
        mount_inventory=raw[3],
        path_inventory=raw[4],
        storage_root_inventory=_artifact(
            storage.STORAGE_ROOT_INVENTORY_SCHEMA_VERSION,
            "storage-root-inventory",
        ),
        field_inventory=_artifact(
            storage.STORAGE_FIELD_INVENTORY_SCHEMA_VERSION,
            "field-inventory",
        ),
        raw_schema_inventory=_artifact(
            storage.RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION,
            "raw-schema-inventory",
        ),
        outer_cgroup_memory_swap_max_pre_go=raw[8],
        outer_cgroup_swap_counters_initial=raw[9],
        outer_cgroup_memory_zswap_writeback_pre_go=raw[11],
        docker_implicit_mount_inventory=raw[12],
        docker_create_inspect=raw[13],
        final_oci_spec=raw[14],
        console_stdio_inventory=raw[15],
        rootfs_upperdir_pre_go_baseline=raw[17],
        docker_volume_inventory_pre_go_baseline=_artifact(
            storage.DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
            "docker-volume-inventory-pre-go-baseline",
        ),
        max_temporary_peak_bytes=4096,
        tmpfs_hard_size_limit_bytes=4096,
        components=policy.components,
    )


def _receipt(
    policy: storage.StorageBackendPolicyV1,
    intent: storage.StorageBoundaryRuntimeIntentV1,
    raw: tuple[storage.ArtifactIdentityV1, ...],
) -> storage.StorageBoundaryReceiptV2:
    handshake = storage.HostV3HandshakeProjectionV1(
        request=_artifact(storage.HOST_CASE_REQUEST_V3_SCHEMA_VERSION, "host-request"),
        intent=_artifact(storage.HOST_CASE_INTENT_V3_SCHEMA_VERSION, "host-intent"),
        ready=_artifact(storage.HOST_READY_V3_SCHEMA_VERSION, "host-ready"),
        observer_anchor=_artifact(
            storage.HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
            "host-anchor",
        ),
        go=_artifact(storage.HOST_GO_V3_SCHEMA_VERSION, "host-go"),
        campaign_id=intent.campaign_id,
        case_ordinal=intent.case_ordinal,
        candidate_id=intent.candidate_id,
        qualification_case_id=intent.qualification_case_id,
        image_id=intent.image_id,
        container_name=intent.container_name,
        container_id_commitment_sha256=intent.container_id_commitment_sha256,
    )
    policy_identity = storage.storage_backend_policy_identity_v1(policy)
    intent_identity = storage.storage_boundary_runtime_intent_identity_v1(intent)
    wrapper = _artifact(storage.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION, "wrapper")
    reload_validation = _artifact(
        storage.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "reload-validation",
    )
    relay = _artifact(
        storage.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        "terminal-relay-preseal",
    )
    channel = _artifact(
        storage.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        "nonstorage-channel-preseal",
    )
    seal = _artifact(storage.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION, "write-seal")
    host_go_binding = storage.HostGoStorageIntentBindingV1(
        host_go=handshake.go,
        runtime_intent=intent_identity,
        verifier=policy.components[0],
    )
    tmpfs = storage.TmpfsConservativeBoundEvidenceV1(
        aggregate_root_path=storage.TMPFS_AGGREGATE_ROOT,
        hard_limit_bytes=4096,
        initial_observed_used_bytes=256,
        terminal_observed_used_bytes=3072,
        measurement_interval_start=handshake.go,
        measurement_interval_end=seal,
        raw_mountinfo=raw[0],
        raw_statfs_diagnostic_samples=raw[1],
        raw_hard_limit_mount_mutation_closure=raw[2],
        measurement_producer=policy.components[0],
    )
    disk = storage.DiskStructuralAbsenceEvidenceV1(
        measurement_interval_start=handshake.go,
        measurement_interval_end=seal,
        raw_mountinfo=raw[3],
        raw_absence_scan=raw[4],
        raw_writable_fd_inventory=raw[5],
        measurement_producer=policy.components[0],
    )
    closure = storage.SwapAndImplicitMountClosureEvidenceV1(
        outer_cgroup_identity_sha256=intent.outer_cgroup_identity_sha256,
        measurement_interval_start=handshake.go,
        measurement_interval_end=seal,
        docker_volume_inventory_pre_go_baseline=(intent.docker_volume_inventory_pre_go_baseline),
        raw_memory_swap_max_pre_go=raw[8],
        raw_swap_counters_initial=raw[9],
        raw_swap_counters_terminal=raw[10],
        raw_memory_zswap_writeback_pre_go=raw[11],
        raw_docker_implicit_mount_inventory=raw[12],
        raw_docker_create_inspect=raw[13],
        raw_final_oci_spec=raw[14],
        raw_console_stdio_inventory=raw[15],
        raw_docker_api_operation_journal=raw[16],
        raw_rootfs_upperdir_pre_go_baseline=raw[17],
        raw_rootfs_upperdir_interval_delta=raw[18],
        raw_docker_volume_inventory_delta=raw[19],
        measurement_producer=policy.components[0],
        runtime_storage_escape_gate=policy.components[4],
        initial_observation_monotonic_ns=100,
        go_commit_monotonic_ns=200,
        worker_exit_monotonic_ns=300,
        publication_wrapper_monotonic_ns=400,
        reload_validation_monotonic_ns=500,
        relay_preseal_monotonic_ns=600,
        channel_preseal_monotonic_ns=700,
        write_seal_monotonic_ns=800,
        terminal_observation_monotonic_ns=900,
        receipt_precommit_monotonic_ns=1000,
    )
    reload_projection = storage.RawArtifactProjectionV1(
        kind="publication_reload",
        first_class_artifact=reload_validation,
        raw_artifact=raw[6],
        predecessors=(wrapper,),
        producer=policy.components[1],
    )
    seal_projection = storage.RawArtifactProjectionV1(
        kind="write_seal",
        first_class_artifact=seal,
        raw_artifact=raw[7],
        predecessors=(reload_validation, relay, channel),
        producer=policy.components[3],
    )
    return storage.StorageBoundaryReceiptV2(
        campaign_id=intent.campaign_id,
        case_ordinal=intent.case_ordinal,
        candidate_id=intent.candidate_id,
        candidate_family=intent.candidate_family,
        qualification_case_id=intent.qualification_case_id,
        image_id=intent.image_id,
        handshake=handshake,
        policy=policy_identity,
        runtime_intent=intent_identity,
        host_go_storage_intent_binding=host_go_binding,
        publication_wrapper=wrapper,
        publication_reload_validation=reload_validation,
        terminal_relay_preseal_attestation=relay,
        nonstorage_channel_preseal_attestation=channel,
        irreversible_write_seal=seal,
        tmpfs_conservative_bound_evidence=tmpfs,
        disk_absence_evidence=disk,
        swap_and_implicit_mount_closure_evidence=closure,
        raw_publication_reload=raw[6],
        raw_write_seal=raw[7],
        publication_reload_projection=reload_projection,
        write_seal_projection=seal_projection,
        terminal_relay_preseal_producer=policy.components[1],
        nonstorage_channel_preseal_producer=policy.components[2],
        raw_artifacts=raw,
        max_temporary_peak_bytes=4096,
        max_disk_peak_bytes=0,
        components=policy.components,
    )


def _fixture() -> tuple[
    storage.StorageBackendPolicyV1,
    storage.StorageBoundaryRuntimeIntentV1,
    storage.StorageBoundaryReceiptV2,
    tuple[storage.ArtifactIdentityV1, ...],
]:
    raw = _raw_artifacts()
    policy = _policy()
    intent = _intent(policy, raw)
    return policy, intent, _receipt(policy, intent, raw), raw


def _intent_bindings(
    intent: storage.StorageBoundaryRuntimeIntentV1,
) -> storage.StorageRuntimeIntentExternalBindingsV1:
    return storage.StorageRuntimeIntentExternalBindingsV1(
        campaign_id=intent.campaign_id,
        case_ordinal=intent.case_ordinal,
        candidate_id=intent.candidate_id,
        candidate_family=intent.candidate_family,
        qualification_case_id=intent.qualification_case_id,
        image_id=intent.image_id,
        container_name=intent.container_name,
        container_id_commitment_sha256=intent.container_id_commitment_sha256,
        outer_cgroup_identity_sha256=intent.outer_cgroup_identity_sha256,
        qualification_plan=intent.qualification_plan,
        policy=intent.policy,
        runtime_qualification_receipt=intent.runtime_qualification_receipt,
        host_provisioning_v3_validated_pre_go_prefix=(
            intent.host_provisioning_v3_validated_pre_go_prefix
        ),
        mount_namespace_identity=intent.mount_namespace_identity,
        rootfs_mount_identity=intent.rootfs_mount_identity,
        tmpfs_mount_identity=intent.tmpfs_mount_identity,
        tmpfs_backing_identity=intent.tmpfs_backing_identity,
        mount_inventory=intent.mount_inventory,
        path_inventory=intent.path_inventory,
        storage_root_inventory=intent.storage_root_inventory,
        field_inventory=intent.field_inventory,
        raw_schema_inventory=intent.raw_schema_inventory,
        outer_cgroup_memory_swap_max_pre_go=intent.outer_cgroup_memory_swap_max_pre_go,
        outer_cgroup_swap_counters_initial=intent.outer_cgroup_swap_counters_initial,
        outer_cgroup_memory_zswap_writeback_pre_go=(
            intent.outer_cgroup_memory_zswap_writeback_pre_go
        ),
        docker_implicit_mount_inventory=intent.docker_implicit_mount_inventory,
        docker_create_inspect=intent.docker_create_inspect,
        final_oci_spec=intent.final_oci_spec,
        console_stdio_inventory=intent.console_stdio_inventory,
        rootfs_upperdir_pre_go_baseline=intent.rootfs_upperdir_pre_go_baseline,
        docker_volume_inventory_pre_go_baseline=(intent.docker_volume_inventory_pre_go_baseline),
        max_temporary_peak_bytes=intent.max_temporary_peak_bytes,
        aggregate_root_case_exclusive=intent.aggregate_root_case_exclusive,
        components=intent.components,
    )


def _receipt_bindings(
    receipt: storage.StorageBoundaryReceiptV2,
) -> storage.StorageReceiptExternalBindingsV1:
    return storage.StorageReceiptExternalBindingsV1(
        host_handshake=receipt.handshake,
        host_go_storage_intent_binding=receipt.host_go_storage_intent_binding,
        publication_wrapper=receipt.publication_wrapper,
        publication_reload_validation=receipt.publication_reload_validation,
        terminal_relay_preseal_attestation=receipt.terminal_relay_preseal_attestation,
        nonstorage_channel_preseal_attestation=(receipt.nonstorage_channel_preseal_attestation),
        irreversible_write_seal=receipt.irreversible_write_seal,
        raw_artifacts=receipt.raw_artifacts,
    )


def _cleanup(
    intent: storage.StorageBoundaryRuntimeIntentV1,
    receipt: storage.StorageBoundaryReceiptV2,
    outcome: CleanupOutcome = "committed_receipt_cleaned",
) -> storage.StorageCleanupReconciliationV1:
    committed = outcome.startswith("committed_receipt_")
    uncertain = outcome.startswith("receipt_commit_uncertain_")
    cleaned = outcome.endswith("_cleaned")
    receipt_identity = storage.storage_boundary_receipt_v2_identity(receipt)
    failure = None
    if not committed:
        failure = _artifact(
            storage.STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION,
            f"{outcome}:operational-failure",
        )
    namespace_cleanup = None
    cleanup_failure = None
    if cleaned:
        namespace_cleanup = _artifact(
            storage.STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
            f"{outcome}:namespace-cleanup",
        )
    else:
        cleanup_failure = _artifact(
            storage.STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION,
            f"{outcome}:cleanup-failure",
        )
    return storage.StorageCleanupReconciliationV1(
        campaign_id=intent.campaign_id,
        case_ordinal=intent.case_ordinal,
        candidate_id=intent.candidate_id,
        candidate_family=intent.candidate_family,
        qualification_case_id=intent.qualification_case_id,
        runtime_intent=storage.storage_boundary_runtime_intent_identity_v1(intent),
        outcome=cast(Any, outcome),
        receipt=receipt_identity if committed else None,
        attempted_receipt=receipt_identity if uncertain else None,
        failure_frontier=failure,
        namespace_cleanup_receipt=namespace_cleanup,
        cleanup_failure_frontier=cleanup_failure,
        cleanup_producer=intent.components[5],
        cleanup_complete=cleaned,
        residual_storage_state="absent" if cleaned else "unknown",
        tmpfs_unmounted_before_namespace_release=True if cleaned else None,
        aggregate_mount_id_absent_before_namespace_release=True if cleaned else None,
        underlying_aggregate_path_read_only=True if cleaned else None,
        namespace_process_count=0 if cleaned else None,
        retained_namespace_fd_count_after_release=0 if cleaned else None,
        retained_writable_path_count=0 if cleaned else None,
        receipt_committed=committed,
        receipt_commit_uncertain=uncertain,
    )


def _cleanup_bindings(
    cleanup: storage.StorageCleanupReconciliationV1,
) -> storage.StorageCleanupExternalBindingsV1:
    return storage.StorageCleanupExternalBindingsV1(
        attempted_receipt=cleanup.attempted_receipt,
        operational_failure_frontier=cleanup.failure_frontier,
        namespace_cleanup_receipt=cleanup.namespace_cleanup_receipt,
        cleanup_failure_frontier=cleanup.cleanup_failure_frontier,
        cleanup_producer=cleanup.cleanup_producer,
    )


def _artifact_bytes_and_pins(
    value: Any,
    body_serializer: Any,
    file_serializer: Any,
) -> tuple[bytes, str, str]:
    body = body_serializer(value)
    raw = file_serializer(value)
    return raw, hashlib.sha256(raw).hexdigest(), hashlib.sha256(body).hexdigest()


def _rehashed_file(body: dict[str, Any], digest_field: str) -> tuple[bytes, str, str]:
    body_raw = storage.canonical_storage_backend_json_bytes(body, final_lf=False)
    body_sha256 = hashlib.sha256(body_raw).hexdigest()
    value = dict(body)
    value[digest_field] = body_sha256
    raw = storage.canonical_storage_backend_json_bytes(value)
    return raw, hashlib.sha256(raw).hexdigest(), body_sha256


def _rebind_receipt_intent(
    receipt: storage.StorageBoundaryReceiptV2,
    intent: storage.StorageBoundaryRuntimeIntentV1,
) -> storage.StorageBoundaryReceiptV2:
    identity = storage.storage_boundary_runtime_intent_identity_v1(intent)
    binding = replace(receipt.host_go_storage_intent_binding, runtime_intent=identity)
    return replace(
        receipt,
        runtime_intent=identity,
        host_go_storage_intent_binding=binding,
    )


def test_positive_canonical_dual_pin_roundtrips_and_complete_chain() -> None:
    policy, intent, receipt, _ = _fixture()
    cleanup = _cleanup(intent, receipt)
    cases: tuple[tuple[Any, Any, Any, Any], ...] = (
        (
            policy,
            storage.canonical_storage_backend_policy_v1_body_bytes,
            storage.canonical_storage_backend_policy_v1_file_bytes,
            storage.parse_storage_backend_policy_v1,
        ),
        (
            intent,
            storage.canonical_storage_boundary_runtime_intent_v1_body_bytes,
            storage.canonical_storage_boundary_runtime_intent_v1_file_bytes,
            storage.parse_storage_boundary_runtime_intent_v1,
        ),
        (
            receipt,
            storage.canonical_storage_boundary_receipt_v2_body_bytes,
            storage.canonical_storage_boundary_receipt_v2_file_bytes,
            storage.parse_storage_boundary_receipt_v2,
        ),
        (
            cleanup,
            storage.canonical_storage_cleanup_reconciliation_v1_body_bytes,
            storage.canonical_storage_cleanup_reconciliation_v1_file_bytes,
            storage.parse_storage_cleanup_reconciliation_v1,
        ),
    )
    for expected, body_serializer, file_serializer, parser in cases:
        raw, file_pin, body_pin = _artifact_bytes_and_pins(
            expected,
            body_serializer,
            file_serializer,
        )
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        assert (
            parser(
                raw,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )
            == expected
        )
        with pytest.raises(ERROR, match="caller file pin|caller pin"):
            parser(
                raw,
                expected_file_sha256=_hash("wrong-file-pin"),
                expected_body_sha256=body_pin,
            )
        with pytest.raises(ERROR, match="BODY"):
            parser(
                raw,
                expected_file_sha256=file_pin,
                expected_body_sha256=_hash("wrong-body-pin"),
            )

    storage.validate_storage_backend_chain_v2(
        policy,
        intent,
        receipt,
        cleanup,
        intent_bindings=_intent_bindings(intent),
        receipt_bindings=_receipt_bindings(receipt),
        cleanup_bindings=_cleanup_bindings(cleanup),
    )


def test_body_digest_excludes_digest_field_and_hash_inventories_are_authoritative() -> None:
    policy = _policy()
    raw = storage.canonical_storage_backend_policy_v1_file_bytes(policy)
    decoded = json.loads(raw)
    supplied = decoded.pop("storage_backend_policy_body_sha256")
    assert (
        supplied
        == hashlib.sha256(
            storage.canonical_storage_backend_policy_v1_body_bytes(policy)
        ).hexdigest()
    )
    assert raw == storage.canonical_storage_backend_json_bytes(
        {**decoded, "storage_backend_policy_body_sha256": supplied}
    )
    for values, expected in (
        (storage.RESOURCE_FIELDS, storage.RESOURCE_FIELD_ORDER_SHA256),
        (storage.MATCHED_V3_CANDIDATE_IDS, storage.CANDIDATE_ORDER_SHA256),
    ):
        encoded = json.dumps(
            list(values),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        assert hashlib.sha256(encoded).hexdigest() == expected
    assert len(storage.RESOURCE_FIELDS) == 28
    assert storage.RESOURCE_FIELDS[23:25] == (
        "max_temporary_peak_bytes",
        "max_disk_peak_bytes",
    )


def test_phase_contamination_reverse_binding_and_exact_statuses_rejected() -> None:
    policy, intent, receipt, _ = _fixture()
    body = policy.to_body_dict()
    body["actual_container_id"] = "forbidden"
    raw, file_pin, body_pin = _rehashed_file(body, "storage_backend_policy_body_sha256")
    with pytest.raises(ERROR, match="phase-4 contamination"):
        storage.parse_storage_backend_policy_v1(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )

    receipt_body = receipt.to_body_dict()
    receipt_body["terminal"] = {"file_sha256": _hash("terminal")}
    raw, file_pin, body_pin = _rehashed_file(
        receipt_body,
        "storage_boundary_receipt_v2_body_sha256",
    )
    with pytest.raises(ERROR, match="reverse binding"):
        storage.parse_storage_boundary_receipt_v2(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )

    cleanup = _cleanup(intent, receipt)
    for artifact, mutation in (
        (policy, {"status": "wrong"}),
        (policy, {"schema_version": "wrong"}),
        (intent, {"status": "wrong"}),
        (intent, {"schema_version": "wrong"}),
        (receipt, {"status": "wrong"}),
        (receipt, {"schema_version": "wrong"}),
        (cleanup, {"status": "wrong"}),
        (cleanup, {"schema_version": "wrong"}),
    ):
        with pytest.raises(ERROR):
            _replace_untyped(artifact, **mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        {"max_temporary_peak_bytes": True},
        {"temporary_field_position": True},
        {"disk_field_position": False},
        {"aggregate_root_count": True},
        {"application_bind_mount_count": False},
        {"implicit_readonly_etc_bind_count": True},
        {"image_declared_volume_count": False},
        {"outer_cgroup_memory_swap_max_bytes": False},
        {"phase_number": True},
    ),
)
def test_policy_rejects_bool_as_integer(mutation: dict[str, object]) -> None:
    with pytest.raises(ERROR, match="integer|ceiling"):
        _replace_untyped(_policy(), **mutation)


def test_integer_boolean_and_float_confusion_fails_through_nested_artifacts() -> None:
    policy, intent, receipt, _ = _fixture()
    cleanup = _cleanup(intent, receipt)
    mutations: tuple[tuple[Any, dict[str, object]], ...] = (
        (intent, {"case_ordinal": True}),
        (intent, {"aggregate_root_count": False}),
        (intent, {"writable_persistent_fd_count": True}),
        (receipt, {"publication_wrapper_order": False}),
        (receipt, {"max_disk_peak_bytes": False}),
        (
            receipt.tmpfs_conservative_bound_evidence,
            {"terminal_observed_used_bytes": True},
        ),
        (receipt.disk_absence_evidence, {"published_value_bytes": False}),
        (
            receipt.swap_and_implicit_mount_closure_evidence,
            {"go_commit_monotonic_ns": True},
        ),
        (cleanup, {"namespace_process_count": False}),
    )
    for value, mutation in mutations:
        with pytest.raises(ERROR):
            _replace_untyped(value, **mutation)

    body = policy.to_body_dict()
    body["max_temporary_peak_bytes"] = 4096.0
    raw = (
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    with pytest.raises(ERROR, match="float"):
        storage.decode_canonical_storage_backend_json(raw)


def test_strict_json_lf_alias_cycle_depth_nodes_size_and_recursion_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for raw in (
        b'{"a":1,"a":1}\n',
        b'{"a":1.0}\n',
        b'{"a":NaN}\n',
        b'{"a":true}',
        b'{"a":true}\n\n',
        b'{ "a":true}\n',
        b'{"a":"\xc3\xa9"}\n',
    ):
        with pytest.raises(ERROR):
            storage.decode_canonical_storage_backend_json(raw)

    shared: dict[str, object] = {"value": 1}
    with pytest.raises(ERROR, match="alias or cycle"):
        storage.canonical_storage_backend_json_bytes({"left": shared, "right": shared})
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ERROR, match="alias or cycle"):
        storage.canonical_storage_backend_json_bytes({"cycle": cycle})
    with pytest.raises(ERROR, match="boolean"):
        storage.canonical_storage_backend_json_bytes({}, final_lf=cast(Any, 1))

    deep: object = "x"
    for _ in range(70):
        deep = [deep]
    with pytest.raises(ERROR, match="depth or node"):
        storage.canonical_storage_backend_json_bytes({"deep": deep})
    deep_raw = b'{"deep":' + b"[" * 70 + b"0" + b"]" * 70 + b"}\n"
    with pytest.raises(ERROR, match="depth or node"):
        storage.decode_canonical_storage_backend_json(deep_raw)
    many_nodes = {str(index): index for index in range(50_001)}
    with pytest.raises(ERROR, match="depth or node"):
        storage.canonical_storage_backend_json_bytes(many_nodes)
    with pytest.raises(ERROR, match="bound"):
        storage.decode_canonical_storage_backend_json(b"{" + b" " * (4 * 1024 * 1024))

    def _raise_recursion(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RecursionError

    monkeypatch.setattr(json, "dumps", _raise_recursion)
    with pytest.raises(ERROR, match="not canonical"):
        storage.canonical_storage_backend_json_bytes({"safe": 1})


def test_parser_rejects_wrong_body_even_when_file_and_embedded_digest_are_rehashed() -> None:
    policy = _policy()
    body = policy.to_body_dict()
    body["measurement_failure_policy"] = "fail_closed_no_value_no_retry"
    raw, file_pin, body_pin = _rehashed_file(body, "storage_backend_policy_body_sha256")
    with pytest.raises(ERROR, match="BODY"):
        storage.parse_storage_backend_policy_v1(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=_hash("independent-wrong-body"),
        )
    assert (
        storage.parse_storage_backend_policy_v1(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )
        == policy
    )
    with pytest.raises(ERROR):
        storage.parse_storage_backend_policy_v1(
            cast(Any, bytearray(raw)),
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"case_ordinal": 1},
        {"candidate_id": "causal_e025_q075"},
        {"candidate_family": "external"},
        {"qualification_case_id": "qualification_01_causal_e025_q075"},
    ),
)
def test_candidate_order_family_and_case_projection_are_exact(
    mutation: dict[str, object],
) -> None:
    policy, intent, _, _ = _fixture()
    with pytest.raises(ERROR, match="candidate|case"):
        _replace_untyped(intent, **mutation)
    with pytest.raises(ERROR):
        replace(policy, candidate_order_sha256=_hash("other-candidate-order"))


def test_component_roles_schemas_sources_and_descriptor_identities_are_distinct() -> None:
    policy = _policy()
    with pytest.raises(ERROR, match="descriptor schema"):
        replace(policy.components[0], descriptor_schema_version="alberta.wrong.v1")
    with pytest.raises(ERROR, match="role"):
        _replace_untyped(policy.components[0], role="terminal_relay")
    with pytest.raises(ERROR, match="role order"):
        replace(policy, components=tuple(reversed(policy.components)))

    for field in (
        "source_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
    ):
        components = list(policy.components)
        components[1] = _replace_untyped(
            components[1],
            **{field: getattr(components[0], field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(policy, components=tuple(components))


def test_runtime_boundary_rejects_file_and_body_aliases() -> None:
    policy, intent, _, _ = _fixture()
    for digest_field in ("file_sha256", "body_sha256"):
        rootfs = _replace_untyped(
            intent.rootfs_mount_identity,
            **{digest_field: getattr(intent.mount_namespace_identity, digest_field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(intent, rootfs_mount_identity=rootfs)


@pytest.mark.parametrize(
    "field",
    (
        "policy",
        "runtime_qualification_receipt",
        "host_provisioning_v3_validated_pre_go_prefix",
        "mount_namespace_identity",
        "rootfs_mount_identity",
        "tmpfs_mount_identity",
        "tmpfs_backing_identity",
        "mount_inventory",
        "path_inventory",
        "storage_root_inventory",
        "field_inventory",
        "raw_schema_inventory",
        "outer_cgroup_memory_swap_max_pre_go",
        "outer_cgroup_swap_counters_initial",
        "outer_cgroup_memory_zswap_writeback_pre_go",
        "docker_implicit_mount_inventory",
        "docker_create_inspect",
        "final_oci_spec",
        "console_stdio_inventory",
        "rootfs_upperdir_pre_go_baseline",
        "docker_volume_inventory_pre_go_baseline",
    ),
)
@pytest.mark.parametrize("digest_field", ("file_sha256", "body_sha256"))
def test_every_runtime_identity_rejects_cross_schema_file_or_body_alias(
    field: str,
    digest_field: str,
) -> None:
    _, intent, _, _ = _fixture()
    artifact = cast(storage.ArtifactIdentityV1, getattr(intent, field))
    aliased = _replace_untyped(
        artifact,
        **{digest_field: getattr(intent.qualification_plan, digest_field)},
    )
    with pytest.raises(ERROR, match="alias"):
        _replace_untyped(intent, **{field: aliased})


def test_handshake_receipt_projection_and_external_first_class_aliases_rejected() -> None:
    _, _, receipt, _ = _fixture()
    for digest_field in ("file_sha256", "body_sha256"):
        aliased_ready = _replace_untyped(
            receipt.handshake.ready,
            **{digest_field: getattr(receipt.handshake.request, digest_field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(receipt.handshake, ready=aliased_ready)

        aliased_wrapper = _replace_untyped(
            receipt.publication_wrapper,
            **{digest_field: getattr(receipt.handshake.request, digest_field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(receipt, publication_wrapper=aliased_wrapper)

        projection = receipt.publication_reload_projection
        aliased_predecessor = _replace_untyped(
            projection.predecessors[0],
            **{digest_field: getattr(projection.first_class_artifact, digest_field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(projection, predecessors=(aliased_predecessor,))

        bindings = _receipt_bindings(receipt)
        raw = list(bindings.raw_artifacts)
        raw[0] = _replace_untyped(
            raw[0],
            **{digest_field: getattr(bindings.host_handshake.request, digest_field)},
        )
        with pytest.raises(ERROR, match="first-class and raw"):
            replace(bindings, raw_artifacts=tuple(raw))


@pytest.mark.parametrize("index", range(20))
def test_every_raw_artifact_position_is_tied_to_typed_receipt_evidence(index: int) -> None:
    _, _, receipt, _ = _fixture()
    changed = list(receipt.raw_artifacts)
    changed[index] = _artifact(changed[index].schema_version, f"crosswired-raw-{index}")
    with pytest.raises(ERROR, match="typed storage evidence"):
        replace(receipt, raw_artifacts=tuple(changed))


def test_raw_artifact_order_file_alias_and_body_alias_fail_closed() -> None:
    _, _, receipt, _ = _fixture()
    swapped = list(receipt.raw_artifacts)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ERROR, match="inventory"):
        replace(receipt, raw_artifacts=tuple(swapped))
    for digest_field in ("file_sha256", "body_sha256"):
        aliased = list(receipt.raw_artifacts)
        aliased[1] = _replace_untyped(
            aliased[1],
            **{digest_field: getattr(aliased[0], digest_field)},
        )
        with pytest.raises(ERROR, match="alias"):
            replace(receipt, raw_artifacts=tuple(aliased))


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("tmpfs_mount_identity", 0),
        ("mount_inventory", 3),
        ("path_inventory", 4),
        ("outer_cgroup_memory_swap_max_pre_go", 8),
        ("outer_cgroup_swap_counters_initial", 9),
        ("outer_cgroup_memory_zswap_writeback_pre_go", 11),
        ("docker_implicit_mount_inventory", 12),
        ("docker_create_inspect", 13),
        ("final_oci_spec", 14),
        ("console_stdio_inventory", 15),
        ("rootfs_upperdir_pre_go_baseline", 17),
    ),
)
def test_receipt_crosslinks_every_pre_go_raw_identity(field: str, index: int) -> None:
    policy, intent, receipt, _ = _fixture()
    changed_intent = _replace_untyped(
        intent,
        **{field: _artifact(storage.RAW_ARTIFACT_SCHEMA_INVENTORY[index], f"other-{field}")},
    )
    changed_receipt = _rebind_receipt_intent(receipt, changed_intent)
    with pytest.raises(ERROR, match="differ|crosswire"):
        storage.validate_storage_boundary_receipt_v2(
            policy,
            changed_intent,
            changed_receipt,
        )


@pytest.mark.parametrize(
    "field",
    tuple(storage.StorageRuntimeIntentExternalBindingsV1.__dataclass_fields__),
)
def test_every_runtime_intent_external_field_crosswire_rejects(field: str) -> None:
    _, intent, _, _ = _fixture()
    bindings = _intent_bindings(intent)
    value = getattr(bindings, field)
    if isinstance(value, storage.ArtifactIdentityV1):
        changes: dict[str, object] = {
            field: _artifact(value.schema_version, f"external-other-{field}")
        }
    elif field in {
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
    }:
        changes = {
            "case_ordinal": 1,
            "candidate_id": "causal_e025_q075",
            "candidate_family": "local",
            "qualification_case_id": "qualification_01_causal_e025_q075",
        }
    elif field == "campaign_id":
        changes = {field: "other_campaign"}
    elif field == "image_id":
        changes = {field: f"sha256:{_hash('external-other-image')}"}
    elif field == "container_name":
        changes = {field: "other-container"}
    elif field in {"container_id_commitment_sha256", "outer_cgroup_identity_sha256"}:
        changes = {field: _hash(f"external-other-{field}")}
    elif field == "max_temporary_peak_bytes":
        changes = {field: 8192}
    elif field == "components":
        changes = {
            field: tuple(
                _other_component(component, f"external-other-component-{index}")
                for index, component in enumerate(bindings.components)
            )
        }
    else:
        assert field == "aggregate_root_case_exclusive"
        with pytest.raises(ERROR, match="case-exclusive"):
            replace(bindings, aggregate_root_case_exclusive=False)
        return
    changed = _replace_untyped(bindings, **changes)
    with pytest.raises(ERROR, match="runtime intent external"):
        storage.validate_storage_runtime_intent_artifact_bindings_v1(intent, changed)


def test_raw9_is_a_committed_pre_go_baseline_and_crosslinks_receipt() -> None:
    policy, intent, receipt, _ = _fixture()
    assert intent.outer_cgroup_swap_counters_initial == receipt.raw_artifacts[9]
    changed_intent = replace(
        intent,
        outer_cgroup_swap_counters_initial=_artifact(
            storage.RAW_ARTIFACT_SCHEMA_INVENTORY[9],
            "uncommitted-initial-swap-counter-baseline",
        ),
    )
    with pytest.raises(ERROR, match="initial swap counters"):
        storage.validate_storage_boundary_receipt_v2(
            policy,
            changed_intent,
            _rebind_receipt_intent(receipt, changed_intent),
        )


def test_terminal_raw19_cannot_substitute_for_pre_go_volume_inventory_baseline() -> None:
    policy, intent, receipt, raw = _fixture()
    assert raw[19].schema_version == storage.DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION
    assert (
        intent.docker_volume_inventory_pre_go_baseline.schema_version
        == storage.DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION
    )
    assert raw[19] != intent.docker_volume_inventory_pre_go_baseline
    with pytest.raises(ERROR, match="pre-GO baseline"):
        replace(intent, docker_volume_inventory_pre_go_baseline=raw[19])
    with pytest.raises(ERROR, match="pre-GO baseline"):
        replace(
            receipt.swap_and_implicit_mount_closure_evidence,
            docker_volume_inventory_pre_go_baseline=raw[19],
        )

    changed_intent = replace(
        intent,
        docker_volume_inventory_pre_go_baseline=_artifact(
            storage.DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
            "other-pre-go-volume-baseline",
        ),
    )
    with pytest.raises(ERROR, match="volume inventory pre-GO baseline"):
        storage.validate_storage_boundary_receipt_v2(
            policy,
            changed_intent,
            _rebind_receipt_intent(receipt, changed_intent),
        )


def test_shared_tmpfs_is_rejected_at_policy_intent_external_and_evidence_boundaries() -> None:
    policy, intent, receipt, _ = _fixture()
    with pytest.raises(ERROR, match="aggregate root"):
        replace(policy, aggregate_root_case_exclusive=False)
    with pytest.raises(ERROR, match="aggregate root"):
        replace(intent, aggregate_root_case_exclusive=False)
    with pytest.raises(ERROR, match="case exclusivity"):
        replace(
            receipt.tmpfs_conservative_bound_evidence,
            aggregate_root_case_exclusive=False,
        )
    with pytest.raises(ERROR, match="case-exclusive"):
        replace(_intent_bindings(intent), aggregate_root_case_exclusive=False)


@pytest.mark.parametrize(
    ("evidence_field", "endpoint_field", "schema"),
    (
        (
            "tmpfs_conservative_bound_evidence",
            "measurement_interval_start",
            storage.HOST_GO_V3_SCHEMA_VERSION,
        ),
        (
            "tmpfs_conservative_bound_evidence",
            "measurement_interval_end",
            storage.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
        ),
        (
            "swap_and_implicit_mount_closure_evidence",
            "measurement_interval_start",
            storage.HOST_GO_V3_SCHEMA_VERSION,
        ),
        (
            "swap_and_implicit_mount_closure_evidence",
            "measurement_interval_end",
            storage.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
        ),
    ),
)
def test_tmpfs_and_swap_evidence_endpoints_bind_exact_host_go_and_seal(
    evidence_field: str,
    endpoint_field: str,
    schema: str,
) -> None:
    _, _, receipt, _ = _fixture()
    evidence = getattr(receipt, evidence_field)
    changed_evidence = _replace_untyped(
        evidence,
        **{endpoint_field: _artifact(schema, f"other-{evidence_field}-{endpoint_field}")},
    )
    with pytest.raises(ERROR, match="interval"):
        _replace_untyped(receipt, **{evidence_field: changed_evidence})


def test_host_go_binding_is_exact_prior_intent_and_measurement_producer_bound() -> None:
    _, _, receipt, _ = _fixture()
    binding = receipt.host_go_storage_intent_binding
    for mutation in (
        {"exact_bidirectional_projection": False},
        {"intent_committed_before_go": False},
    ):
        with pytest.raises(ERROR, match="host GO"):
            _replace_untyped(binding, **mutation)
    with pytest.raises(ERROR, match="producer role"):
        replace(binding, verifier=receipt.components[1])
    with pytest.raises(ERROR, match="bind"):
        replace(
            receipt,
            host_go_storage_intent_binding=replace(
                binding,
                host_go=_artifact(storage.HOST_GO_V3_SCHEMA_VERSION, "other-go"),
            ),
        )
    with pytest.raises(ERROR, match="bind"):
        replace(
            receipt,
            host_go_storage_intent_binding=replace(
                binding,
                runtime_intent=_artifact(
                    storage.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
                    "other-runtime-intent",
                ),
            ),
        )


def test_reload_and_seal_raw_projections_are_exact_and_producer_bound() -> None:
    _, _, receipt, _ = _fixture()
    reload_projection = receipt.publication_reload_projection
    seal_projection = receipt.write_seal_projection
    for projection in (reload_projection, seal_projection):
        for mutation in (
            {"exact_projection": False},
            {"predecessor_identity_bound": False},
        ):
            with pytest.raises(ERROR, match="projection"):
                _replace_untyped(projection, **mutation)
    with pytest.raises(ERROR, match="producer role"):
        replace(reload_projection, producer=receipt.components[3])
    with pytest.raises(ERROR, match="producer role"):
        replace(seal_projection, producer=receipt.components[1])
    with pytest.raises(ERROR, match="predecessor inventory"):
        replace(reload_projection, predecessors=())
    with pytest.raises(ERROR, match="crosswires"):
        replace(
            receipt,
            publication_reload_projection=replace(
                reload_projection,
                predecessors=(
                    _artifact(
                        storage.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
                        "other-wrapper-predecessor",
                    ),
                ),
            ),
        )
    with pytest.raises(ERROR, match="crosswires"):
        replace(
            receipt,
            write_seal_projection=replace(
                seal_projection,
                predecessors=(
                    receipt.publication_reload_validation,
                    _artifact(
                        storage.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                        "other-relay-predecessor",
                    ),
                    receipt.nonstorage_channel_preseal_attestation,
                ),
            ),
        )
    with pytest.raises(ERROR, match="crosswires"):
        replace(
            receipt,
            publication_reload_projection=replace(
                reload_projection,
                raw_artifact=_artifact(
                    storage.RAW_ARTIFACT_SCHEMA_INVENTORY[6],
                    "other-reload-raw",
                ),
            ),
        )
    with pytest.raises(ERROR, match="crosswires"):
        replace(
            receipt,
            write_seal_projection=replace(
                seal_projection,
                first_class_artifact=_artifact(
                    storage.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
                    "other-seal",
                ),
            ),
        )


def test_tmpfs_publishes_conservative_immutable_ceiling_not_diagnostic_peak() -> None:
    policy, _, receipt, _ = _fixture()
    evidence = receipt.tmpfs_conservative_bound_evidence
    assert evidence.terminal_observed_used_bytes == 3072
    assert receipt.max_temporary_peak_bytes == evidence.hard_limit_bytes == 4096
    assert receipt.max_temporary_peak_bytes == policy.max_temporary_peak_bytes
    for mutation in (
        {"initial_observed_used_bytes": 4097},
        {"terminal_observed_used_bytes": 4097},
        {"hard_limit_unchanged_through_write_seal": False},
        {"noswap_active_pre_go": False},
        {"noswap_unchanged_through_write_seal": False},
        {"mount_mutation_disabled_before_go": False},
        {"mount_mutation_disabled_through_write_seal": False},
        {"aggregate_root_non_bypassable": False},
        {"statfs_samples_are_diagnostic_only": False},
        {"publication_semantics": "observed_peak"},
    ):
        with pytest.raises(ERROR):
            _replace_untyped(evidence, **mutation)
    with pytest.raises(ERROR, match="conservative"):
        replace(receipt, max_temporary_peak_bytes=3072)


def test_disk_zero_is_scoped_go_to_seal_structural_nonaddressability() -> None:
    _, _, receipt, _ = _fixture()
    evidence = receipt.disk_absence_evidence
    assert evidence.measurement_interval_start == receipt.handshake.go
    assert evidence.measurement_interval_end == receipt.irreversible_write_seal
    assert evidence.published_value_bytes == receipt.max_disk_peak_bytes == 0
    for mutation in (
        {"published_value_bytes": 1},
        {"published_value_bytes": False},
        {"persistent_storage_scope": "host_wide"},
        {"measurement_interval": "point_sample"},
        {"trusted_runtime_bookkeeping_exclusions": ()},
        {"writable_persistent_mount_count": 1},
        {"writable_persistent_fd_count": 1},
        {"rootfs_read_only": False},
        {"copy_up_disabled": False},
        {"no_bind_volume_device_network_log_paths": False},
        {"no_inherited_or_alternate_writable_paths": False},
        {"transient_persistent_open_structurally_impossible": False},
        {"structural_nonaddressability_complete": False},
    ):
        with pytest.raises(ERROR):
            _replace_untyped(evidence, **mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        {"memory_swap_max_bytes_pre_go": 1},
        {"memory_swap_current_initial_bytes": 1},
        {"memory_swap_current_terminal_bytes": 1},
        {"memory_swap_peak_initial_bytes": 1},
        {"memory_swap_peak_terminal_bytes": 1},
        {"memory_zswap_current_initial_bytes": 1},
        {"memory_zswap_current_terminal_bytes": 1},
        {"memory_zswap_writeback_enabled_pre_go": True},
        {"retained_counter_endpoints_same_outer_cgroup": False},
        {"counters_retained_from_pre_go_through_terminal": False},
        {"application_writable_tmpfs_count": 2},
        {"application_writable_tmpfs_is_aggregate_root": False},
        {"application_writable_tmpfs_allocatable": False},
        {"docker_implicit_mount_inventory_complete": False},
        {"docker_implicit_mounts_all_read_only": False},
        {"docker_implicit_mounts_all_nonallocatable": False},
        {"docker_implicit_mounts_have_application_writable_path": True},
        {"implicit_readonly_etc_bind_count": 2},
        {"user_bind_mount_count": 1},
        {"user_volume_mount_count": 1},
        {"image_declared_volume_count": 1},
        {"added_device_count": 1},
        {"writable_persistent_mount_count": 1},
        {"writable_persistent_fd_count": 1},
        {"candidate_stdio_transport_count": 1},
        {"forbidden_docker_api_operation_count": 1},
        {"rootfs_upperdir_interval_delta_bytes": 1},
        {"docker_volume_inventory_delta": 1},
        {"docker_ipc_mode": "private"},
        {"docker_shm_mount_present": True},
        {"docker_tty_enabled": True},
        {"docker_stdin_open": True},
        {"docker_exec_permitted": True},
        {"docker_archive_api_permitted": True},
        {"docker_api_candidate_accessible": True},
        {"container_console_or_fifo_candidate_accessible": True},
        {"default_device_inventory_exact": False},
        {"default_devices_can_allocate_storage": True},
        {"memfd_posix_or_sysv_shm_permitted": True},
        {"post_go_mount_mutation_permitted": True},
        {"candidate_cgroup_mutation_permitted": True},
        {"daemon_runtime_storage_candidate_accessible": True},
        {"host_archival_candidate_accessible": True},
        {"final_oci_spec_exact": False},
        {"custom_runtime_neutralized_stock_writable_implicit_mounts": False},
        {"structural_nonaddressability_complete": False},
        {"docker_api_allowlist_complete_and_lossless": False},
        {"device_and_ipc_confinement_active": False},
        {"rootfs_upperdir_candidate_inaccessible": False},
    ),
)
def test_swap_docker_implicit_mount_api_device_and_upperdir_closure_is_load_bearing(
    mutation: dict[str, object],
) -> None:
    _, _, receipt, _ = _fixture()
    with pytest.raises(ERROR):
        _replace_untyped(receipt.swap_and_implicit_mount_closure_evidence, **mutation)


@pytest.mark.parametrize(
    "field",
    (
        "go_commit_monotonic_ns",
        "worker_exit_monotonic_ns",
        "publication_wrapper_monotonic_ns",
        "reload_validation_monotonic_ns",
        "relay_preseal_monotonic_ns",
        "channel_preseal_monotonic_ns",
        "write_seal_monotonic_ns",
        "terminal_observation_monotonic_ns",
        "receipt_precommit_monotonic_ns",
    ),
)
def test_measurement_chronology_is_strict_at_every_boundary(field: str) -> None:
    _, _, receipt, _ = _fixture()
    closure = receipt.swap_and_implicit_mount_closure_evidence
    with pytest.raises(ERROR, match="chronology"):
        _replace_untyped(closure, **{field: closure.initial_observation_monotonic_ns})


def test_receipt_publication_reload_relay_channel_seal_order_is_exact() -> None:
    _, _, receipt, _ = _fixture()
    for mutation in (
        {"reload_validation_order": 2, "relay_preseal_order": 1},
        {"write_seal_order": 3, "channel_preseal_order": 4},
        {"receipt_commit_order": 4},
        {"publication_wrapper_order": False},
        {"reload_read_only": False},
        {"publication_projection_exact": False},
        {"write_seal_irreversible": False},
        {"later_measured_writes_possible": True},
        {"terminal_bound": True},
        {"lifecycle_bound": True},
        {"merger_bound": True},
    ):
        with pytest.raises(ERROR):
            _replace_untyped(receipt, **mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "request",
        "intent",
        "ready",
        "observer_anchor",
        "go",
        "campaign_id",
        "candidate_projection",
        "image_id",
        "container_name",
        "container_id_commitment_sha256",
    ),
)
def test_external_full_host_handshake_projection_scalar_and_identity_crosswires_reject(
    mutation: str,
) -> None:
    _, _, receipt, _ = _fixture()
    bindings = _receipt_bindings(receipt)
    handshake = bindings.host_handshake
    changes: dict[str, object]
    if mutation in {"request", "intent", "ready", "observer_anchor"}:
        artifact = cast(storage.ArtifactIdentityV1, getattr(handshake, mutation))
        changes = {mutation: _artifact(artifact.schema_version, f"other-host-{mutation}")}
    elif mutation == "go":
        other_go = _artifact(storage.HOST_GO_V3_SCHEMA_VERSION, "other-host-go")
        changed_handshake = replace(handshake, go=other_go)
        changed_binding = replace(bindings.host_go_storage_intent_binding, host_go=other_go)
        changed = replace(
            bindings,
            host_handshake=changed_handshake,
            host_go_storage_intent_binding=changed_binding,
        )
        with pytest.raises(ERROR, match="full host handshake projection"):
            storage.validate_storage_receipt_artifact_bindings_v2(receipt, changed)
        return
    elif mutation == "campaign_id":
        changes = {mutation: "other_campaign"}
    elif mutation == "candidate_projection":
        changes = {
            "case_ordinal": 1,
            "candidate_id": "causal_e025_q075",
            "qualification_case_id": "qualification_01_causal_e025_q075",
        }
    elif mutation == "image_id":
        changes = {mutation: f"sha256:{_hash('other-host-image')}"}
    elif mutation == "container_name":
        changes = {mutation: "other-host-container"}
    else:
        assert mutation == "container_id_commitment_sha256"
        changes = {mutation: _hash("other-host-container-commitment")}
    changed = replace(bindings, host_handshake=_replace_untyped(handshake, **changes))
    with pytest.raises(ERROR, match="full host handshake projection"):
        storage.validate_storage_receipt_artifact_bindings_v2(receipt, changed)


@pytest.mark.parametrize("mutation", ("runtime_intent", "verifier"))
def test_external_full_host_go_projection_scalar_crosswires_reject(mutation: str) -> None:
    _, _, receipt, _ = _fixture()
    bindings = _receipt_bindings(receipt)
    projection = bindings.host_go_storage_intent_binding
    if mutation == "runtime_intent":
        changes: dict[str, object] = {
            mutation: _artifact(
                storage.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
                "other-host-go-runtime-intent",
            )
        }
    else:
        changes = {
            mutation: _other_component(
                projection.verifier,
                "other-host-go-measurement-verifier",
            )
        }
    changed = replace(
        bindings,
        host_go_storage_intent_binding=_replace_untyped(projection, **changes),
    )
    with pytest.raises(ERROR, match="full host-GO storage-intent binding projection"):
        storage.validate_storage_receipt_artifact_bindings_v2(receipt, changed)

    for flag in ("exact_bidirectional_projection", "intent_committed_before_go"):
        with pytest.raises(ERROR, match="host GO"):
            _replace_untyped(projection, **{flag: False})


@pytest.mark.parametrize(
    "field",
    (
        "publication_wrapper",
        "publication_reload_validation",
        "terminal_relay_preseal_attestation",
        "nonstorage_channel_preseal_attestation",
        "irreversible_write_seal",
    ),
)
def test_every_external_first_class_receipt_binding_rejects_crosswire(field: str) -> None:
    _, _, receipt, _ = _fixture()
    bindings = _receipt_bindings(receipt)
    value = cast(storage.ArtifactIdentityV1, getattr(bindings, field))
    changed = _replace_untyped(
        bindings,
        **{field: _artifact(value.schema_version, f"other-{field}")},
    )
    with pytest.raises(ERROR, match="external"):
        storage.validate_storage_receipt_artifact_bindings_v2(receipt, changed)


@pytest.mark.parametrize("index", range(20))
def test_every_external_raw_receipt_binding_rejects_crosswire(index: int) -> None:
    _, _, receipt, _ = _fixture()
    bindings = _receipt_bindings(receipt)
    raw = list(bindings.raw_artifacts)
    raw[index] = _artifact(raw[index].schema_version, f"other-external-raw-{index}")
    changed = replace(bindings, raw_artifacts=tuple(raw))
    with pytest.raises(ERROR, match="external raw"):
        storage.validate_storage_receipt_artifact_bindings_v2(receipt, changed)


def test_full_chain_requires_and_invokes_all_three_external_binding_envelopes() -> None:
    policy, intent, receipt, _ = _fixture()
    cleanup = _cleanup(intent, receipt)
    intent_bindings = _intent_bindings(intent)
    receipt_bindings = _receipt_bindings(receipt)
    cleanup_bindings = _cleanup_bindings(cleanup)
    with pytest.raises(TypeError):
        cast(Any, storage.validate_storage_backend_chain_v2)(policy, intent, receipt, cleanup)
    with pytest.raises(ERROR, match="external runtime binding type"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            receipt,
            cleanup,
            intent_bindings=cast(Any, None),
            receipt_bindings=receipt_bindings,
            cleanup_bindings=cleanup_bindings,
        )
    bad_intent_bindings = replace(intent_bindings, campaign_id="other_campaign")
    with pytest.raises(ERROR, match="external campaign_id"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            receipt,
            cleanup,
            intent_bindings=bad_intent_bindings,
            receipt_bindings=receipt_bindings,
            cleanup_bindings=cleanup_bindings,
        )
    with pytest.raises(ERROR, match="external receipt bindings"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            receipt,
            cleanup,
            intent_bindings=intent_bindings,
            receipt_bindings=None,
            cleanup_bindings=cleanup_bindings,
        )
    other_go = _artifact(storage.HOST_GO_V3_SCHEMA_VERSION, "external-other-go")
    bad_receipt_bindings = replace(
        receipt_bindings,
        host_handshake=replace(receipt_bindings.host_handshake, go=other_go),
        host_go_storage_intent_binding=replace(
            receipt_bindings.host_go_storage_intent_binding,
            host_go=other_go,
        ),
    )
    with pytest.raises(ERROR, match="external identity"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            receipt,
            cleanup,
            intent_bindings=intent_bindings,
            receipt_bindings=bad_receipt_bindings,
            cleanup_bindings=cleanup_bindings,
        )
    bad_cleanup_bindings = replace(
        cleanup_bindings,
        cleanup_producer=_other_component(cleanup.cleanup_producer, "other-cleanup-producer"),
    )
    with pytest.raises(ERROR, match="cleanup producer"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            receipt,
            cleanup,
            intent_bindings=intent_bindings,
            receipt_bindings=receipt_bindings,
            cleanup_bindings=bad_cleanup_bindings,
        )

    failed = _cleanup(intent, receipt, "failed_before_receipt_cleaned")
    with pytest.raises(ERROR, match="must not receive receipt bindings"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            None,
            failed,
            intent_bindings=intent_bindings,
            receipt_bindings=receipt_bindings,
            cleanup_bindings=_cleanup_bindings(failed),
        )
    with pytest.raises(ERROR, match="external cleanup binding type"):
        storage.validate_storage_backend_chain_v2(
            policy,
            intent,
            None,
            failed,
            intent_bindings=intent_bindings,
            receipt_bindings=None,
            cleanup_bindings=cast(Any, None),
        )


@pytest.mark.parametrize(
    "outcome",
    (
        "committed_receipt_cleaned",
        "failed_before_receipt_cleaned",
        "receipt_commit_uncertain_cleaned",
        "committed_receipt_cleanup_failed",
        "failed_before_receipt_cleanup_failed",
        "receipt_commit_uncertain_cleanup_failed",
    ),
)
def test_all_six_cleanup_outcomes_roundtrip_and_validate(outcome: CleanupOutcome) -> None:
    policy, intent, receipt, _ = _fixture()
    cleanup = _cleanup(intent, receipt, outcome)
    committed = outcome.startswith("committed_receipt_")
    storage.validate_storage_backend_chain_v2(
        policy,
        intent,
        receipt if committed else None,
        cleanup,
        intent_bindings=_intent_bindings(intent),
        receipt_bindings=_receipt_bindings(receipt) if committed else None,
        cleanup_bindings=_cleanup_bindings(cleanup),
    )
    raw, file_pin, body_pin = _artifact_bytes_and_pins(
        cleanup,
        storage.canonical_storage_cleanup_reconciliation_v1_body_bytes,
        storage.canonical_storage_cleanup_reconciliation_v1_file_bytes,
    )
    assert (
        storage.parse_storage_cleanup_reconciliation_v1(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )
        == cleanup
    )
    for field in (
        "synthesized_temporary_peak_bytes",
        "synthesized_disk_peak_bytes",
    ):
        with pytest.raises(ERROR, match="synthesize"):
            _replace_untyped(cleanup, **{field: 0})


def test_cleanup_failure_preserves_uncertainty_and_never_synthesizes_teardown() -> None:
    _, intent, receipt, _ = _fixture()
    failed = _cleanup(intent, receipt, "receipt_commit_uncertain_cleanup_failed")
    assert failed.attempted_receipt is not None
    assert failed.failure_frontier is not None
    assert failed.cleanup_failure_frontier is not None
    for mutation in (
        {"cleanup_complete": True},
        {"residual_storage_state": "absent"},
        {"tmpfs_unmounted_before_namespace_release": False},
        {"aggregate_mount_id_absent_before_namespace_release": False},
        {"underlying_aggregate_path_read_only": False},
        {"namespace_process_count": 0},
        {"retained_namespace_fd_count_after_release": 0},
        {"retained_writable_path_count": 0},
        {
            "namespace_cleanup_receipt": _artifact(
                storage.STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
                "fabricated-cleanup",
            )
        },
    ):
        with pytest.raises(ERROR, match="uncertainty|failed cleanup"):
            _replace_untyped(failed, **mutation)
    with pytest.raises(ERROR, match="uncertain"):
        replace(failed, attempted_receipt=None)
    with pytest.raises(ERROR, match="uncertain"):
        replace(failed, failure_frontier=None)


def test_cleaned_cleanup_requires_namespace_unmount_mount_id_path_process_and_fd_proof() -> None:
    _, intent, receipt, _ = _fixture()
    cleaned = _cleanup(intent, receipt)
    for mutation in (
        {"namespace_cleanup_receipt": None},
        {"cleanup_complete": False},
        {"residual_storage_state": "unknown"},
        {"tmpfs_unmounted_before_namespace_release": False},
        {"aggregate_mount_id_absent_before_namespace_release": False},
        {"underlying_aggregate_path_read_only": False},
        {"namespace_process_count": 1},
        {"retained_namespace_fd_count_after_release": 1},
        {"retained_writable_path_count": 1},
    ):
        with pytest.raises(ERROR):
            _replace_untyped(cleaned, **mutation)


def test_cleanup_external_attempt_failure_terminal_and_producer_bindings_are_exact() -> None:
    _, intent, receipt, _ = _fixture()
    uncertain_failed = _cleanup(intent, receipt, "receipt_commit_uncertain_cleanup_failed")
    bindings = _cleanup_bindings(uncertain_failed)
    for field in (
        "attempted_receipt",
        "operational_failure_frontier",
        "cleanup_failure_frontier",
    ):
        value = cast(storage.ArtifactIdentityV1, getattr(bindings, field))
        changed = _replace_untyped(
            bindings,
            **{field: _artifact(value.schema_version, f"other-{field}")},
        )
        with pytest.raises(ERROR, match="cleanup external"):
            storage.validate_storage_cleanup_reconciliation_v1(
                intent,
                uncertain_failed,
                receipt=None,
                external_bindings=changed,
            )

    cleaned = _cleanup(intent, receipt, "failed_before_receipt_cleaned")
    cleaned_bindings = _cleanup_bindings(cleaned)
    assert cleaned_bindings.namespace_cleanup_receipt is not None
    changed_cleaned = replace(
        cleaned_bindings,
        namespace_cleanup_receipt=_artifact(
            storage.STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
            "other-namespace-cleanup",
        ),
    )
    with pytest.raises(ERROR, match="cleanup external"):
        storage.validate_storage_cleanup_reconciliation_v1(
            intent,
            cleaned,
            receipt=None,
            external_bindings=changed_cleaned,
        )


def test_descriptor_is_immutable_source_only_zero_sentinel_and_dual_pin_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = dict(storage.SOURCE_ONLY_CAPABILITIES)
    descriptor = storage.StorageBackendContractDescriptorV2(capabilities=capabilities)
    capabilities["container_control"] = True
    assert descriptor.capabilities["container_control"] is False
    with pytest.raises(TypeError):
        cast(Any, descriptor.capabilities)["container_control"] = True
    assert descriptor.artifact_schemas == (
        storage.STORAGE_BACKEND_POLICY_SCHEMA_VERSION,
        storage.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        storage.STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
        storage.STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
    )
    assert descriptor.component_roles == storage.STORAGE_COMPONENT_ROLES
    assert (
        descriptor.component_descriptor_schemas
        == storage.STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY
    )
    assert len(descriptor.component_descriptor_schemas) == 6
    assert descriptor.phase_split == storage.STORAGE_PHASE_SPLIT
    assert descriptor.chronology == storage.STORAGE_CHRONOLOGY
    assert descriptor.resource_fields == storage.RESOURCE_FIELDS
    assert descriptor.raw_artifact_schemas == storage.RAW_ARTIFACT_SCHEMA_INVENTORY
    assert len(descriptor.raw_artifact_schemas) == 20
    assert (
        descriptor.required_external_artifact_schemas == storage.REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS
    )
    assert len(descriptor.required_external_artifact_schemas) == 23
    assert len(set(descriptor.component_descriptor_schemas)) == len(
        descriptor.component_descriptor_schemas
    )
    assert len(set(descriptor.required_external_artifact_schemas)) == len(
        descriptor.required_external_artifact_schemas
    )
    assert set(descriptor.required_external_artifact_schemas).isdisjoint(
        descriptor.artifact_schemas
    )
    assert set(descriptor.required_external_artifact_schemas).isdisjoint(
        descriptor.component_descriptor_schemas
    )
    assert set(descriptor.required_external_artifact_schemas).isdisjoint(
        descriptor.raw_artifact_schemas
    )
    schema_inventory = (
        descriptor.artifact_schemas
        + descriptor.component_descriptor_schemas
        + descriptor.raw_artifact_schemas
        + descriptor.required_external_artifact_schemas
    )
    assert len(schema_inventory) == 53
    assert len(set(schema_inventory)) == len(schema_inventory)
    assert descriptor.operational_apis == ()
    assert descriptor.descriptor_self_pin_sha256 == ZERO_SHA256
    assert descriptor.source_file_sha256_pin is None
    assert storage.PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_FILE_SHA256 == ZERO_SHA256
    assert storage.PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256 == ZERO_SHA256
    for mapping in (
        descriptor.capabilities,
        descriptor.readiness,
        descriptor.authority,
        descriptor.claims,
    ):
        assert mapping and not any(mapping.values())

    body = storage.canonical_storage_backend_contract_descriptor_v2_body_bytes(descriptor)
    raw = storage.canonical_storage_backend_contract_descriptor_v2_file_bytes(descriptor)
    body_pin = hashlib.sha256(body).hexdigest()
    file_pin = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(storage, "PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256", body_pin)
    monkeypatch.setattr(storage, "PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_FILE_SHA256", file_pin)
    assert (
        storage.parse_storage_backend_contract_descriptor_v2(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )
        == descriptor
    )
    assert storage.storage_backend_v2_descriptor_sha256() == file_pin
    with pytest.raises(ERROR, match="caller identity"):
        storage.parse_storage_backend_contract_descriptor_v2(
            raw,
            expected_file_sha256=_hash("wrong-descriptor-file"),
            expected_body_sha256=body_pin,
        )
    with pytest.raises(ERROR, match="caller identity"):
        storage.parse_storage_backend_contract_descriptor_v2(
            raw,
            expected_file_sha256=file_pin,
            expected_body_sha256=_hash("wrong-descriptor-body"),
        )


def test_descriptor_rejects_mutable_semantic_variants_and_exports_are_complete() -> None:
    descriptor = storage.StorageBackendContractDescriptorV2()
    for mutation in (
        {"status": "operational"},
        {"artifact_schemas": tuple(reversed(descriptor.artifact_schemas))},
        {"component_descriptor_schemas": tuple(reversed(descriptor.component_descriptor_schemas))},
        {"component_roles": tuple(reversed(descriptor.component_roles))},
        {"phase_split": tuple(reversed(descriptor.phase_split))},
        {"chronology": tuple(reversed(descriptor.chronology))},
        {"storage_field_positions": (25, 24)},
        {"raw_artifact_schemas": tuple(reversed(descriptor.raw_artifact_schemas))},
        {
            "required_external_artifact_schemas": tuple(
                reversed(descriptor.required_external_artifact_schemas)
            )
        },
        {"operational_apis": ("execute",)},
        {"descriptor_self_pin_sha256": _hash("nonzero-serialized-self-pin")},
        {"source_file_sha256_pin": _hash("source-pin")},
        {"capabilities": {**descriptor.capabilities, "container_control": True}},
        {"readiness": {**descriptor.readiness, "extra": False}},
    ):
        with pytest.raises(ERROR):
            _replace_untyped(descriptor, **mutation)

    required_exports = {
        "PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256",
        "DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION",
        "STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS",
        "RawArtifactProjectionV1",
        "HostGoStorageIntentBindingV1",
        "StorageRuntimeIntentExternalBindingsV1",
        "StorageReceiptExternalBindingsV1",
        "StorageCleanupExternalBindingsV1",
        "storage_backend_v2_descriptor_sha256",
        "validate_storage_runtime_intent_artifact_bindings_v1",
    }
    assert required_exports <= set(storage.__all__)
    assert len(storage.__all__) == len(set(storage.__all__))
    assert all(hasattr(storage, name) for name in storage.__all__)


def test_source_ast_bans_operational_imports_and_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    banned_import_roots = {
        "ctypes",
        "docker",
        "fcntl",
        "jax",
        "multiprocessing",
        "numpy",
        "os",
        "pathlib",
        "resource",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "threading",
        "time",
    }
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & banned_import_roots
    assert not calls & {
        "Popen",
        "check_call",
        "check_output",
        "connect",
        "exec",
        "eval",
        "fork",
        "mount",
        "open",
        "run",
        "socket",
        "system",
        "unmount",
    }
