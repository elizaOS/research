"""Static contract tests for the additive batch-only observation registry v2."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from typing import Any, Literal

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_observations_v2 as observations,
)

_EXPECTED_DESCRIPTOR_BODY_SHA256 = "0" * 64
_EXPECTED_DESCRIPTOR_FILE_SHA256 = "0" * 64
_EXPECTED_DESCRIPTOR_SHA256 = _EXPECTED_DESCRIPTOR_FILE_SHA256
_FRESH_IMAGE_ID = "sha256:93562b7037a45a69b4ac6bb67c8f7e06d21a13f6e05a5374eb10bce68f30f2b5"
_EXPECTED_CANDIDATE_ORDER = (
    "causal_e025_q050",
    "causal_e025_q075",
    "causal_e025_q090",
    "causal_e050_q050",
    "causal_e050_q075",
    "causal_e050_q090",
    "causal_e100_q050",
    "causal_e100_q075",
    "causal_e100_q090",
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
    "alberta_rtu_h08_taylor",
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
    "random_policy",
    "search_nearest",
    "search_oracle",
)
_EXPECTED_OPERATIONAL_PHASES = (
    "request_validated",
    "authorization_validated",
    "intent_committed",
    "fresh_cgroup_created",
    "retained_counter_fds_opened",
    "initial_cgroup_sample_committed",
    "container_created",
    "container_started",
    "driver_ready",
    "observer_anchored",
    "go_committed",
    "workload_started",
    "workload_exited",
    "algorithmic_resource_receipt_committed",
    "native_publication_committed",
    "publication_commitment_wrapper_committed",
    "publication_reload_validated",
    "storage_write_seal_committed",
    "storage_boundary_receipt_committed",
)
_EXPECTED_RECOVERY_NODE_NAMES = (
    "precleanup_cgroup_sample",
    "cgroup_kill",
    "cgroup_empty",
    "container_absence",
    "post_container_remove_cgroup_sample",
    "cgroup_counter_fds_closed",
    "outer_cgroup_absence",
    "final_cgroup_proof",
)
_EXPECTED_RECOVERY_NODE_DEPENDENCIES = {
    "precleanup_cgroup_sample": (),
    "cgroup_kill": ("precleanup_cgroup_sample",),
    "cgroup_empty": ("precleanup_cgroup_sample",),
    "container_absence": (),
    "post_container_remove_cgroup_sample": ("container_absence",),
    "cgroup_counter_fds_closed": ("post_container_remove_cgroup_sample",),
    "outer_cgroup_absence": ("cgroup_counter_fds_closed",),
    "final_cgroup_proof": _EXPECTED_RECOVERY_NODE_NAMES[:-1],
}
_EXPECTED_TERMINAL_BODY_KEYS = (
    "schema_version",
    "case_spine_sha256",
    "case_ordinal",
    "candidate_id",
    "candidate_family",
    "qualification_case_id",
    "record_kind",
    "operational_frontier",
    "cleanup_reconciliation",
    "driver_terminal",
    "algorithmic_resource_receipt",
    "publication_commitment_wrapper",
    "publication_reload_validation",
    "storage_write_seal",
    "storage_boundary_receipt",
    "returncode",
    "timed_out",
    "error_message_sha256",
    "cleanup_proven",
    "case_consumed",
    "same_case_retry_permitted",
    "authority",
    "claims",
)
_EXPECTED_WRAPPER_BODY_KEYS = (
    "schema_version",
    "status",
    "classification",
    "case_spine_sha256",
    "case_ordinal",
    "candidate_id",
    "candidate_family",
    "qualification_case_id",
    "publisher",
    "publisher_metadata",
    "native_atomic_producer",
    "native_publication_receipt",
    "publication_address_sha256",
    "publication_manifest_file_sha256",
    "publication_manifest_body_sha256",
    "file_inventory_sha256",
    "published_bundle_sha256",
    "expected_reload_observation_sha256",
    "file_count",
    "total_size_bytes",
    "maximum_total_size_bytes",
    "video_slot_mode",
    "files",
    "reload_performed_by_wrapper",
    "reload_digest_equality_validated_by_wrapper",
    "content_values_read_by_wrapper",
    "payload_bytes_transported_by_wrapper",
    "capabilities",
    "readiness",
    "authority",
    "claims",
    "limitations",
)
_EXPECTED_RESOURCE_FIELDS = (
    "max_environment_interactions",
    "max_optimizer_updates",
    "max_gradient_updates",
    "max_sample_updates",
    "max_trainable_parameters",
    "max_frozen_parameters",
    "max_optimizer_state_elements",
    "max_optimizer_state_bytes",
    "max_target_copy_elements",
    "max_target_copy_bytes",
    "max_replay_capacity_transitions",
    "max_replay_peak_bytes",
    "max_rollout_storage_elements",
    "max_rollout_peak_bytes",
    "max_recurrent_carry_elements",
    "max_recurrent_carry_bytes",
    "max_rtrl_sensitivity_elements",
    "max_rtrl_sensitivity_bytes",
    "max_eligibility_elements",
    "max_eligibility_bytes",
    "max_peak_rss_bytes",
    "max_cpu_time_ns",
    "max_wall_time_ns",
    "max_temporary_peak_bytes",
    "max_disk_peak_bytes",
    "max_thread_count",
    "max_attempt_count",
    "max_failure_count",
)


class _StringSubclass(str):
    pass


class _EqualityProxy:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


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


def _ordered_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical(list(values), newline=False)).hexdigest()


def _inventory_digest(key: str, values: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical({key: values}, newline=False)).hexdigest()


def _artifact(schema_version: str, label: str) -> observations.ArtifactIdentityV2:
    return observations.ArtifactIdentityV2(
        schema_version=schema_version,
        file_sha256=_sha(f"{label}-file"),
        body_sha256=_sha(f"{label}-body"),
    )


def _canonical_artifact(
    schema_version: str,
    body: dict[str, Any],
    body_sha256_field: str,
) -> observations.ArtifactIdentityV2:
    assert body["schema_version"] == schema_version
    body_sha256 = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    file_sha256 = hashlib.sha256(_canonical({**body, body_sha256_field: body_sha256})).hexdigest()
    return observations.ArtifactIdentityV2(
        schema_version=schema_version,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
    )


def _host_handoff_artifact(
    *,
    spine: observations.CaseSpineV2,
    record_kind: Literal["success", "terminal_failure"],
    terminal_receipt: observations.ArtifactIdentityV2,
    terminal_metadata: observations.ArtifactIdentityV2,
) -> observations.ArtifactIdentityV2:
    return _canonical_artifact(
        "alberta.forager_matched_v3.host_qualification_observation_handoff.v2",
        {
            "schema_version": (
                "alberta.forager_matched_v3.host_qualification_observation_handoff.v2"
            ),
            "case_spine_sha256": spine.body_sha256,
            "case_ordinal": spine.case_ordinal,
            "candidate_id": spine.candidate_id,
            "qualification_case_id": spine.qualification_case_id,
            "record_kind": record_kind,
            "terminal_receipt_file_sha256": terminal_receipt.file_sha256,
            "terminal_receipt_body_sha256": terminal_receipt.body_sha256,
            "terminal_metadata_file_sha256": terminal_metadata.file_sha256,
            "terminal_metadata_body_sha256": terminal_metadata.body_sha256,
        },
        "handoff_body_sha256",
    )


def _operational_frontier_artifact(
    *,
    spine: observations.CaseSpineV2,
    completed_phases: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    container_create_state: str,
    container_start_state: str,
    workload_start_state: str,
    workload_exit_state: str,
    container_create_count_state: str,
    container_create_count: int | None,
    container_start_count_state: str,
    container_start_count: int | None,
    workload_start_count_state: str,
    workload_start_count: int | None,
    workload_exit_count_state: str,
    workload_exit_count: int | None,
    attempt_count_state: str,
    attempt_count: int | None,
    failure_count: int,
) -> observations.ArtifactIdentityV2:
    return _canonical_artifact(
        "alberta.forager_matched_v3.host_qualification_operational_frontier.v2",
        {
            "schema_version": (
                "alberta.forager_matched_v3.host_qualification_operational_frontier.v2"
            ),
            "status": "operational_frontier_recorded_nonretryable_non_authorizing",
            "case_spine_sha256": spine.body_sha256,
            "completed_phases": list(completed_phases),
            "failure_phase": failure_phase,
            "failure_effect_state": failure_effect_state,
            "container_create_state": container_create_state,
            "container_start_state": container_start_state,
            "workload_start_state": workload_start_state,
            "workload_exit_state": workload_exit_state,
            "container_create_count_state": container_create_count_state,
            "container_create_count": container_create_count,
            "container_start_count_state": container_start_count_state,
            "container_start_count": container_start_count,
            "workload_start_count_state": workload_start_count_state,
            "workload_start_count": workload_start_count,
            "workload_exit_count_state": workload_exit_count_state,
            "workload_exit_count": workload_exit_count,
            "attempt_count_state": attempt_count_state,
            "attempt_count": attempt_count,
            "failure_count": failure_count,
            "case_consumed": True,
            "same_case_retry_permitted": False,
            "claims": {
                "execution_authorized": False,
                "execution_performed": False,
                "qualification_granted": False,
                "resource_matched": False,
                "scientific_evidence_created": False,
            },
        },
        "operational_frontier_body_sha256",
    )


def _recovery_nodes(
    prefix: str,
    states: dict[str, str],
) -> tuple[observations.RecoveryNodeCandidateV2, ...]:
    schemas = {
        "precleanup_cgroup_sample": ("alberta.forager_matched_v3.host_precleanup_cgroup_sample.v2"),
        "cgroup_kill": "alberta.forager_matched_v3.host_cgroup_kill_receipt.v2",
        "cgroup_empty": "alberta.forager_matched_v3.host_cgroup_empty_observation.v2",
        "container_absence": ("alberta.forager_matched_v3.host_container_absence_observation.v2"),
        "post_container_remove_cgroup_sample": (
            "alberta.forager_matched_v3.host_post_container_remove_cgroup_sample.v2"
        ),
        "cgroup_counter_fds_closed": (
            "alberta.forager_matched_v3.host_cgroup_counter_fds_closed_receipt.v2"
        ),
        "outer_cgroup_absence": (
            "alberta.forager_matched_v3.host_outer_cgroup_absence_observation.v2"
        ),
        "final_cgroup_proof": "alberta.forager_matched_v3.host_cgroup_v2_boundary_proof.v1",
    }
    return tuple(
        observations.RecoveryNodeCandidateV2(
            node_name=name,
            state=states[name],
            artifact=(
                _artifact(schemas[name], f"{prefix}-{name}")
                if states[name] == "committed"
                else None
            ),
            dependencies=_EXPECTED_RECOVERY_NODE_DEPENDENCIES[name],
            uncertainty_detail_sha256=(
                _sha(f"{prefix}-{name}-detail")
                if states[name] in {"commit_uncertain", "failed_before_commit"}
                else None
            ),
        )
        for name in _EXPECTED_RECOVERY_NODE_NAMES
    )


def _cleanup_reconciliation_artifact(
    *,
    spine: observations.CaseSpineV2,
    operational_frontier: observations.ArtifactIdentityV2,
    cgroup_may_exist: bool,
    recovery_nodes: tuple[observations.RecoveryNodeCandidateV2, ...],
) -> observations.ArtifactIdentityV2:
    unresolved = tuple(
        item.node_name
        for item in recovery_nodes
        if item.state in {"commit_uncertain", "failed_before_commit"}
    )
    cleanup_proven = not cgroup_may_exist or all(
        item.state == "committed" for item in recovery_nodes
    )
    return _canonical_artifact(
        "alberta.forager_matched_v3.host_cleanup_reconciliation.v2",
        {
            "schema_version": "alberta.forager_matched_v3.host_cleanup_reconciliation.v2",
            "status": "conditional_recovery_dag_reconciled_non_authorizing",
            "case_spine_sha256": spine.body_sha256,
            "operational_frontier": operational_frontier.to_dict(),
            "cgroup_may_exist": cgroup_may_exist,
            "recovery_nodes": [item.to_dict() for item in recovery_nodes],
            "cleanup_proven": cleanup_proven,
            "unresolved_recovery_nodes": list(unresolved),
            "recovery_complete": True,
            "terminalization_permitted": True,
            "workload_resume_permitted": False,
            "same_case_retry_permitted": False,
            "claims": {
                "execution_authorized": False,
                "execution_performed": False,
                "qualification_granted": False,
                "resource_matched": False,
                "scientific_evidence_created": False,
            },
        },
        "cleanup_reconciliation_body_sha256",
    )


def _terminal_metadata_artifact(
    *,
    spine: observations.CaseSpineV2,
    record_kind: Literal["success", "terminal_failure"],
    operational_frontier: observations.ArtifactIdentityV2,
    cleanup_reconciliation: observations.ArtifactIdentityV2,
    driver_terminal: observations.ArtifactIdentityV2 | None,
    algorithmic_resource_receipt: observations.ArtifactIdentityV2 | None,
    publication_commitment_wrapper: observations.ArtifactIdentityV2 | None,
    publication_reload_validation: observations.ArtifactIdentityV2 | None,
    storage_write_seal: observations.ArtifactIdentityV2 | None,
    storage_boundary_receipt: observations.ArtifactIdentityV2 | None,
    returncode: int | None,
    timed_out: bool,
    error_message_sha256: str | None,
    cleanup_proven: bool,
) -> observations.ArtifactIdentityV2:
    body = {
        "schema_version": "alberta.forager_matched_v3.host_terminal_metadata.v2",
        "case_spine_sha256": spine.body_sha256,
        "case_ordinal": spine.case_ordinal,
        "candidate_id": spine.candidate_id,
        "candidate_family": spine.candidate_family,
        "qualification_case_id": spine.qualification_case_id,
        "record_kind": record_kind,
        "operational_frontier": operational_frontier.to_dict(),
        "cleanup_reconciliation": cleanup_reconciliation.to_dict(),
        "driver_terminal": None if driver_terminal is None else driver_terminal.to_dict(),
        "algorithmic_resource_receipt": (
            None if algorithmic_resource_receipt is None else algorithmic_resource_receipt.to_dict()
        ),
        "publication_commitment_wrapper": (
            None
            if publication_commitment_wrapper is None
            else publication_commitment_wrapper.to_dict()
        ),
        "publication_reload_validation": (
            None
            if publication_reload_validation is None
            else publication_reload_validation.to_dict()
        ),
        "storage_write_seal": None if storage_write_seal is None else storage_write_seal.to_dict(),
        "storage_boundary_receipt": (
            None if storage_boundary_receipt is None else storage_boundary_receipt.to_dict()
        ),
        "returncode": returncode,
        "timed_out": timed_out,
        "error_message_sha256": error_message_sha256,
        "cleanup_proven": cleanup_proven,
        "case_consumed": True,
        "same_case_retry_permitted": False,
        "authority": {
            "issuer_available": False,
            "evaluator_available": False,
            "merger_available": False,
            "production_backend_available": False,
        },
        "claims": {
            "execution_authorized": False,
            "execution_performed": False,
            "qualification_granted": False,
            "resource_matched": False,
            "scientific_evidence_created": False,
        },
    }
    assert tuple(body) == _EXPECTED_TERMINAL_BODY_KEYS
    return _canonical_artifact(
        "alberta.forager_matched_v3.host_terminal_metadata.v2",
        body,
        "terminal_metadata_body_sha256",
    )


def _final_algorithmic_contract() -> observations.ProducerIdentityV2:
    return observations.ProducerIdentityV2(
        descriptor_schema_version=(
            "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
        ),
        descriptor_sha256=("9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"),
        source_sha256=("c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"),
    )


def _final_storage_contract() -> observations.ProducerIdentityV2:
    return observations.ProducerIdentityV2(
        descriptor_schema_version=(
            "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1"
        ),
        descriptor_sha256=("d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"),
        source_sha256=("9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"),
    )


def _reload_validation_artifact(
    wrapper: observations.ArtifactIdentityV2,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
) -> observations.ArtifactIdentityV2:
    body = {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
        ),
        "publication_commitment_wrapper_file_sha256": wrapper.file_sha256,
        "publication_commitment_wrapper_body_sha256": wrapper.body_sha256,
        "expected_reload_observation_sha256": expected_reload_observation_sha256,
        "actual_reload_observation_sha256": actual_reload_observation_sha256,
        "reload_performed": True,
        "reload_read_only": True,
    }
    body_sha256 = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    file_bytes = _canonical({**body, "reload_validation_body_sha256": body_sha256})
    return observations.ArtifactIdentityV2(
        schema_version=(
            "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
        ),
        file_sha256=hashlib.sha256(file_bytes).hexdigest(),
        body_sha256=body_sha256,
    )


def _producer(schema_version: str, label: str) -> observations.ProducerIdentityV2:
    return observations.ProducerIdentityV2(
        descriptor_schema_version=schema_version,
        descriptor_sha256=_sha(f"{label}-descriptor"),
        source_sha256=_sha(f"{label}-source"),
    )


def _family(candidate_id: str) -> Literal["local", "external", "adapter"]:
    if candidate_id in observations.MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate_id in observations.MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    return "adapter"


def _campaign() -> observations.CampaignSpineV2:
    local_source = _artifact(
        observations.LOCAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
        "local-source-candidate",
    )
    external_source = _artifact(
        observations.EXTERNAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
        "external-source-candidate",
    )
    adapter_source = _artifact(
        observations.ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION,
        "adapter-source-candidate",
    )
    joint_source_closure = _artifact(
        observations.JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION,
        "joint-source-closure-candidate",
    )
    sealed_staging = _artifact(
        observations.SEALED_STAGING_CANDIDATE_SCHEMA_VERSION,
        "sealed-staging-candidate",
    )
    candidate_order_sha256 = _ordered_digest(_EXPECTED_CANDIDATE_ORDER)
    all_case_sequence_intent = _canonical_artifact(
        "alberta.forager_matched_v3.qualification_all_case_sequence_intent.v1",
        {
            "schema_version": (
                "alberta.forager_matched_v3.qualification_all_case_sequence_intent.v1"
            ),
            "candidate_order_sha256": candidate_order_sha256,
            "candidate_order": list(_EXPECTED_CANDIDATE_ORDER),
            "case_count": 28,
            "claims_completion": False,
        },
        "all_case_sequence_intent_body_sha256",
    )
    return observations.CampaignSpineV2(
        qualification_plan_schema_version=observations.QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        qualification_plan_file_sha256=_sha("plan-file"),
        qualification_plan_body_sha256=_sha("plan-body"),
        observation_registry_schema_version=(
            observations.QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION
        ),
        observation_registry_descriptor_sha256=_EXPECTED_DESCRIPTOR_SHA256,
        observation_registry_source_sha256=_sha("observation-registry-v2-source"),
        plan_issuance_receipt=_artifact(
            observations.PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION,
            "plan-issuance",
        ),
        case_ticket_registry=_artifact(
            observations.CASE_TICKET_REGISTRY_SCHEMA_VERSION,
            "case-ticket-registry",
        ),
        publisher_registry=_artifact(
            observations.PUBLISHER_REGISTRY_SCHEMA_VERSION,
            "publisher-registry",
        ),
        seed_registry=_artifact(
            observations.QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
            "seed-registry",
        ),
        seed_pulse_record=_artifact(
            observations.QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
            "seed-pulse-record",
        ),
        seed_trust_root_receipt=_artifact(
            observations.QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION,
            "seed-trust-root-receipt",
        ),
        quicknet_verifier=_producer(
            observations.QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION,
            "runtime-quicknet-verifier",
        ),
        quicknet_verifier_binary_sha256=_sha("runtime-quicknet-verifier-binary"),
        quicknet_verifier_receipt=_artifact(
            observations.QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION,
            "runtime-quicknet-verifier-receipt",
        ),
        seed_chronology_receipt=_artifact(
            observations.SEED_CHRONOLOGY_RECEIPT_SCHEMA_VERSION,
            "seed-chronology-receipt",
        ),
        local_source_candidate=local_source,
        external_source_candidate=external_source,
        adapter_source_candidate=adapter_source,
        joint_source_closure_candidate=joint_source_closure,
        joint_source_closure_local_file_sha256=local_source.file_sha256,
        joint_source_closure_local_body_sha256=local_source.body_sha256,
        joint_source_closure_external_file_sha256=external_source.file_sha256,
        joint_source_closure_external_body_sha256=external_source.body_sha256,
        joint_source_closure_adapter_file_sha256=adapter_source.file_sha256,
        joint_source_closure_adapter_body_sha256=adapter_source.body_sha256,
        sealed_staging_candidate=sealed_staging,
        sealed_staging_joint_source_closure_file_sha256=(joint_source_closure.file_sha256),
        sealed_staging_joint_source_closure_body_sha256=(joint_source_closure.body_sha256),
        fresh_build_candidate=_artifact(
            observations.FRESH_BUILD_CANDIDATE_SCHEMA_VERSION,
            "fresh-build-candidate",
        ),
        fresh_build_sealed_staging_file_sha256=sealed_staging.file_sha256,
        fresh_build_sealed_staging_body_sha256=sealed_staging.body_sha256,
        fresh_build_image_id=_FRESH_IMAGE_ID,
        image_id=_FRESH_IMAGE_ID,
        runtime_candidate=_artifact(
            observations.RUNTIME_CANDIDATE_SCHEMA_VERSION,
            "runtime-candidate",
        ),
        runtime_qualification_receipt=_artifact(
            observations.RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime-qualification-receipt",
        ),
        host_provisioning_receipt=_artifact(
            observations.HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host-provisioning-receipt",
        ),
        host_executor=_producer(
            observations.PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
            "production-host-executor",
        ),
        full_resource_merger=_producer(
            observations.FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION,
            "full-resource-merger",
        ),
        algorithmic_resource_contract=_final_algorithmic_contract(),
        storage_boundary_contract=_final_storage_contract(),
        all_case_sequence_intent=all_case_sequence_intent,
        all_case_sequence_intent_candidate_order_sha256=candidate_order_sha256,
        all_case_sequence_intent_case_count=28,
        all_case_sequence_intent_claims_completion=False,
        candidate_order_sha256=candidate_order_sha256,
        resource_field_order_sha256=_ordered_digest(observations.RESOURCE_CEILING_FIELDS),
    )


def _case_spine(
    campaign: observations.CampaignSpineV2,
    ordinal: int,
) -> observations.CaseSpineV2:
    candidate_id = observations.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS[ordinal]
    prefix = f"case-{ordinal:02d}-{candidate_id}"
    return observations.CaseSpineV2(
        campaign_spine_sha256=campaign.body_sha256,
        case_ordinal=ordinal,
        candidate_id=candidate_id,
        candidate_family=_family(candidate_id),
        qualification_case_id=f"qualification_{ordinal:02d}_{candidate_id}",
        qualification_case_manifest=_artifact(
            observations.QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION,
            f"{prefix}-manifest",
        ),
        case_execution_ticket=_artifact(
            observations.CASE_EXECUTION_TICKET_SCHEMA_VERSION,
            f"{prefix}-ticket",
        ),
        plan_issuance_receipt_file_sha256=(campaign.plan_issuance_receipt.file_sha256),
        plan_issuance_receipt_body_sha256=(campaign.plan_issuance_receipt.body_sha256),
        publisher_registry_entry=_artifact(
            observations.PUBLISHER_REGISTRY_ENTRY_SCHEMA_VERSION,
            f"{prefix}-publisher-entry",
        ),
        resource_requirement_body_sha256=_sha(f"{prefix}-resource-requirement"),
        seed_case_record_sha256=_sha(f"{prefix}-seed-case-record"),
        seed_derivation_record_sha256=_sha(f"{prefix}-seed-derivation-record"),
        environment_derivation_sha256=_sha(f"{prefix}-environment-derivation"),
        agent_derivation_sha256=_sha(f"{prefix}-agent-derivation"),
        environment_seed_commitment_sha256=_sha(f"{prefix}-environment-seed-commitment"),
        agent_seed_commitment_sha256=_sha(f"{prefix}-agent-seed-commitment"),
        attempt_ordinal=0,
        ticket_single_use=True,
        same_case_retry_permitted=False,
    )


def _publication_profile(
    family: Literal["local", "external", "adapter"],
) -> tuple[tuple[tuple[str, str], ...], str, str]:
    if family == "local":
        return (
            observations.LOCAL_PUBLICATION_ROLE_PATHS,
            observations.LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            observations.LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        )
    if family == "external":
        return (
            observations.EXTERNAL_PUBLICATION_ROLE_PATHS,
            observations.EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            observations.EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        )
    return (
        observations.ADAPTER_PUBLICATION_ROLE_PATHS,
        observations.STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION,
        observations.STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
    )


def _publication_files(
    spine: observations.CaseSpineV2,
) -> tuple[observations.PublicationFileCandidateV2, ...]:
    paths, _, _ = _publication_profile(spine.candidate_family)
    result = []
    for index, (role, name) in enumerate(paths):
        size_bytes = index + 1
        digest = _sha(f"{spine.qualification_case_id}-{role}")
        if role == "upstream_video_slot":
            if spine.candidate_id in observations.MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS:
                size_bytes = 1
            else:
                size_bytes = 0
                digest = observations.EMPTY_FILE_SHA256
        result.append(
            observations.PublicationFileCandidateV2(
                role=role,
                name=name,
                size_bytes=size_bytes,
                sha256=digest,
            )
        )
    return tuple(result)


def _publication_wrapper_artifact(
    *,
    spine: observations.CaseSpineV2,
    publisher: observations.ProducerIdentityV2,
    publisher_metadata: observations.ArtifactIdentityV2,
    native_atomic_producer: observations.ProducerIdentityV2,
    native_publication_receipt: observations.ArtifactIdentityV2 | None,
    publication_address_sha256: str,
    publication_manifest_body_sha256: str,
    file_inventory_sha256: str,
    published_bundle_sha256: str,
    expected_reload_observation_sha256: str,
    total_size_bytes: int,
    video_slot_mode: str,
    files: tuple[observations.PublicationFileCandidateV2, ...],
) -> observations.ArtifactIdentityV2:
    body = {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
        ),
        "status": "implemented_source_only_expected_reload_commitment_non_authorizing",
        "classification": "score_blind_metadata_only_normalized_commitment_non_authorizing",
        "case_spine_sha256": spine.body_sha256,
        "case_ordinal": spine.case_ordinal,
        "candidate_id": spine.candidate_id,
        "candidate_family": spine.candidate_family,
        "qualification_case_id": spine.qualification_case_id,
        "publisher": publisher.to_dict(),
        "publisher_metadata": publisher_metadata.to_dict(),
        "native_atomic_producer": native_atomic_producer.to_dict(),
        "native_publication_receipt": (
            None if native_publication_receipt is None else native_publication_receipt.to_dict()
        ),
        "publication_address_sha256": publication_address_sha256,
        "publication_manifest_file_sha256": files[0].sha256,
        "publication_manifest_body_sha256": publication_manifest_body_sha256,
        "file_inventory_sha256": file_inventory_sha256,
        "published_bundle_sha256": published_bundle_sha256,
        "expected_reload_observation_sha256": expected_reload_observation_sha256,
        "file_count": len(files),
        "total_size_bytes": total_size_bytes,
        "maximum_total_size_bytes": 1_073_741_824,
        "video_slot_mode": video_slot_mode,
        "files": [item.to_dict() for item in files],
        "reload_performed_by_wrapper": False,
        "reload_digest_equality_validated_by_wrapper": False,
        "content_values_read_by_wrapper": False,
        "payload_bytes_transported_by_wrapper": False,
        "capabilities": {
            "acceptance_evaluation": False,
            "case_issuance": False,
            "content_value_decoding": False,
            "execution": False,
            "file_publication": False,
            "host_provisioning": False,
            "payload_byte_transport": False,
            "publication_reload": False,
            "reload_digest_equality_validation": False,
        },
        "readiness": {
            "host_execution_ready": False,
            "observation_ready": False,
            "publication_ready": False,
            "qualification_ready": False,
            "reload_observed": False,
        },
        "authority": {
            "execution_authorized": False,
            "observation_issuance_authorized": False,
            "publication_authority_granted": False,
            "qualification_granted": False,
            "scientific_evidence_created": False,
        },
        "claims": {
            "build_qualified": False,
            "performance_claim_allowed": False,
            "publisher_qualified": False,
            "resource_matched": False,
            "runtime_qualified": False,
            "source_qualified": False,
            "universal_sota_claim_allowed": False,
        },
        "limitations": [
            "The wrapper authenticates metadata commitments and never reads publication files.",
            "The reload digest is expected content; this wrapper performs no reload.",
            "A later phase must observe reload output and validate exact digest equality.",
            "Native receipts remain separate artifacts and are not replaced by this wrapper.",
            (
                "Canonical metadata grants no execution, observation, qualification, "
                "or evidence authority."
            ),
        ],
    }
    return _canonical_artifact(
        "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1",
        body,
        "wrapper_body_sha256",
    )


def _resource_schema(family: Literal["local", "external", "adapter"]) -> str:
    return {
        "local": observations.LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
        "external": observations.EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
        "adapter": observations.ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
    }[family]


def _runner_receipt_schema(candidate_id: str) -> str:
    family = _family(candidate_id)
    if family == "local":
        return observations.LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION
    if family == "external":
        return observations.EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION
    if candidate_id == "adapted_full_rainbow":
        return observations.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
    return observations.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION


def _value_semantics(
    field_name: str,
) -> Literal["exact_observation", "conservative_observed_upper_bound"]:
    if field_name in {"max_peak_rss_bytes", "max_thread_count"}:
        return "conservative_observed_upper_bound"
    return "exact_observation"


def _success_case(
    campaign: observations.CampaignSpineV2,
    ordinal: int,
) -> observations.QualificationCaseSuccessCandidateV2:
    spine = _case_spine(campaign, ordinal)
    prefix = spine.qualification_case_id
    request = _artifact(observations.HOST_CASE_REQUEST_SCHEMA_VERSION, f"{prefix}-request")
    intent = _artifact(observations.HOST_CASE_INTENT_SCHEMA_VERSION, f"{prefix}-intent")
    ready = _artifact(observations.HOST_READY_SCHEMA_VERSION, f"{prefix}-ready")
    observer_anchor = _artifact(
        observations.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
        f"{prefix}-observer-anchor",
    )
    go_commitment = _artifact(
        observations.HOST_GO_SCHEMA_VERSION,
        f"{prefix}-go-commitment",
    )
    initial_cgroup_sample = _artifact(
        observations.HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
        f"{prefix}-initial-cgroup-sample",
    )
    retained_fd_set_sha256 = _sha(f"{prefix}-retained-fd-set")
    cgroup_identity_sha256 = _sha(f"{prefix}-cgroup-identity")
    operational_frontier = _operational_frontier_artifact(
        spine=spine,
        completed_phases=_EXPECTED_OPERATIONAL_PHASES,
        failure_phase=None,
        failure_effect_state=None,
        container_create_state="committed",
        container_start_state="committed",
        workload_start_state="committed",
        workload_exit_state="committed",
        container_create_count_state="exact",
        container_create_count=1,
        container_start_count_state="exact",
        container_start_count=1,
        workload_start_count_state="exact",
        workload_start_count=1,
        workload_exit_count_state="exact",
        workload_exit_count=1,
        attempt_count_state="exact",
        attempt_count=1,
        failure_count=0,
    )
    recovery_nodes = _recovery_nodes(
        prefix,
        {name: "committed" for name in _EXPECTED_RECOVERY_NODE_NAMES},
    )
    cgroup_proof = recovery_nodes[-1].artifact
    assert cgroup_proof is not None
    cleanup_reconciliation = _cleanup_reconciliation_artifact(
        spine=spine,
        operational_frontier=operational_frontier,
        cgroup_may_exist=True,
        recovery_nodes=recovery_nodes,
    )
    driver_terminal = _artifact(
        observations.IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION,
        f"{prefix}-driver-terminal",
    )
    algorithmic_intent = _artifact(
        observations.ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
        f"{prefix}-algorithmic-resource-intent",
    )
    algorithmic_receipt = _artifact(
        _resource_schema(spine.candidate_family),
        f"{prefix}-algorithmic-resource",
    )
    storage_intent = _artifact(
        observations.QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        f"{prefix}-storage-boundary-intent",
    )
    storage_write_seal = _artifact(
        observations.QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
        f"{prefix}-storage-write-seal",
    )
    storage_receipt = _artifact(
        observations.QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
        f"{prefix}-storage-boundary-receipt",
    )
    merger_receipt = _artifact(
        observations.FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION,
        f"{prefix}-full-resource-merger",
    )
    publication_files = _publication_files(spine)
    runner_role = {
        "local": "local_runner_receipt",
        "external": "execution_receipt",
        "adapter": "runner_result_receipt",
    }[spine.candidate_family]
    runner_file = next(item for item in publication_files if item.role == runner_role)
    runner_execution_receipt = observations.ArtifactIdentityV2(
        schema_version=_runner_receipt_schema(spine.candidate_id),
        file_sha256=runner_file.sha256,
        body_sha256=_sha(f"{prefix}-runner-execution-receipt-body"),
    )
    file_dicts = [item.to_dict() for item in publication_files]
    file_inventory_sha256 = _inventory_digest("files", file_dicts)
    _, metadata_schema, publisher_schema = _publication_profile(spine.candidate_family)
    publisher_metadata = _artifact(metadata_schema, f"{prefix}-publisher-metadata")
    publisher = _producer(publisher_schema, f"{prefix}-publisher")
    native_atomic_schema = (
        observations.STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        if spine.candidate_family == "adapter"
        else observations.ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    )
    native_atomic_producer = _producer(
        native_atomic_schema,
        f"{prefix}-native-atomic-producer",
    )
    if spine.candidate_family == "local":
        native_receipt = None
    else:
        native_receipt_schema = (
            observations.EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
            if spine.candidate_family == "external"
            else observations.STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
        )
        native_receipt = _artifact(
            native_receipt_schema,
            f"{prefix}-native-publication-receipt",
        )
    publication_manifest_body_sha256 = _sha(f"{prefix}-publication-manifest-body")
    published_bundle_sha256 = _sha(f"{prefix}-published-bundle")
    reload_observation_sha256 = _sha(f"{prefix}-reload-observation")
    total_size_bytes = sum(item.size_bytes for item in publication_files)
    video_slot_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ] = "not_applicable"
    if spine.candidate_family == "external":
        video_slot_mode = (
            "opaque_ppo_video"
            if spine.candidate_id in observations.MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS
            else "absent_for_continuing_zero_length_slot"
        )
    publication_wrapper = _publication_wrapper_artifact(
        spine=spine,
        publisher=publisher,
        publisher_metadata=publisher_metadata,
        native_atomic_producer=native_atomic_producer,
        native_publication_receipt=native_receipt,
        publication_address_sha256=publication_files[0].sha256,
        publication_manifest_body_sha256=publication_manifest_body_sha256,
        file_inventory_sha256=file_inventory_sha256,
        published_bundle_sha256=published_bundle_sha256,
        expected_reload_observation_sha256=reload_observation_sha256,
        total_size_bytes=total_size_bytes,
        video_slot_mode=video_slot_mode,
        files=publication_files,
    )
    publication_reload_validation = _reload_validation_artifact(
        publication_wrapper,
        reload_observation_sha256,
        reload_observation_sha256,
    )
    terminal = _terminal_metadata_artifact(
        spine=spine,
        record_kind="success",
        operational_frontier=operational_frontier,
        cleanup_reconciliation=cleanup_reconciliation,
        driver_terminal=driver_terminal,
        algorithmic_resource_receipt=algorithmic_receipt,
        publication_commitment_wrapper=publication_wrapper,
        publication_reload_validation=publication_reload_validation,
        storage_write_seal=storage_write_seal,
        storage_boundary_receipt=storage_receipt,
        returncode=0,
        timed_out=False,
        error_message_sha256=None,
        cleanup_proven=True,
    )
    lifecycle = _artifact(
        observations.HOST_LIFECYCLE_SCHEMA_VERSION,
        f"{prefix}-lifecycle",
    )
    execution_receipt = _artifact(
        observations.HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
        f"{prefix}-host-execution",
    )
    observation_handoff = _host_handoff_artifact(
        spine=spine,
        record_kind="success",
        terminal_receipt=execution_receipt,
        terminal_metadata=terminal,
    )
    host_success = observations.HostSuccessCandidateV2(
        case_spine_sha256=spine.body_sha256,
        case_ordinal=ordinal,
        candidate_id=spine.candidate_id,
        qualification_case_id=spine.qualification_case_id,
        qualification_plan_file_sha256=campaign.qualification_plan_file_sha256,
        qualification_plan_body_sha256=campaign.qualification_plan_body_sha256,
        case_execution_ticket_file_sha256=spine.case_execution_ticket.file_sha256,
        case_execution_ticket_body_sha256=spine.case_execution_ticket.body_sha256,
        qualification_case_manifest_file_sha256=(spine.qualification_case_manifest.file_sha256),
        qualification_case_manifest_body_sha256=(spine.qualification_case_manifest.body_sha256),
        publisher_registry_entry_file_sha256=(spine.publisher_registry_entry.file_sha256),
        publisher_registry_entry_body_sha256=(spine.publisher_registry_entry.body_sha256),
        resource_requirement_body_sha256=spine.resource_requirement_body_sha256,
        image_id=campaign.image_id,
        host_executor=campaign.host_executor,
        host_provisioning_receipt=campaign.host_provisioning_receipt,
        request=request,
        authorization_request_file_sha256=request.file_sha256,
        authorization_request_body_sha256=request.body_sha256,
        intent=intent,
        initial_cgroup_sample=initial_cgroup_sample,
        initial_sample_intent_file_sha256=intent.file_sha256,
        initial_sample_intent_body_sha256=intent.body_sha256,
        initial_sample_retained_fd_set_sha256=retained_fd_set_sha256,
        initial_sample_cgroup_identity_sha256=cgroup_identity_sha256,
        ready=ready,
        ready_initial_cgroup_sample_file_sha256=initial_cgroup_sample.file_sha256,
        ready_initial_cgroup_sample_body_sha256=initial_cgroup_sample.body_sha256,
        ready_retained_fd_set_sha256=retained_fd_set_sha256,
        ready_cgroup_identity_sha256=cgroup_identity_sha256,
        observer_anchor=observer_anchor,
        observer_initial_cgroup_sample_file_sha256=initial_cgroup_sample.file_sha256,
        observer_initial_cgroup_sample_body_sha256=initial_cgroup_sample.body_sha256,
        observer_retained_fd_set_sha256=retained_fd_set_sha256,
        observer_cgroup_identity_sha256=cgroup_identity_sha256,
        go_commitment=go_commitment,
        go_retained_fd_set_sha256=retained_fd_set_sha256,
        go_cgroup_identity_sha256=cgroup_identity_sha256,
        operational_frontier=operational_frontier,
        driver_terminal=driver_terminal,
        recovery_nodes=recovery_nodes,
        cleanup_cgroup_may_exist=True,
        cleanup_proven=True,
        cleanup_unresolved_recovery_nodes=(),
        cleanup_recovery_complete=True,
        cleanup_terminalization_permitted=True,
        cleanup_workload_resume_permitted=False,
        cleanup_same_case_retry_permitted=False,
        cleanup_reconciliation=cleanup_reconciliation,
        lifecycle=lifecycle,
        lifecycle_operational_frontier_file_sha256=operational_frontier.file_sha256,
        lifecycle_operational_frontier_body_sha256=operational_frontier.body_sha256,
        lifecycle_cleanup_reconciliation_file_sha256=cleanup_reconciliation.file_sha256,
        lifecycle_cleanup_reconciliation_body_sha256=cleanup_reconciliation.body_sha256,
        lifecycle_terminal_metadata_file_sha256=terminal.file_sha256,
        lifecycle_terminal_metadata_body_sha256=terminal.body_sha256,
        completed_phases=_EXPECTED_OPERATIONAL_PHASES,
        cgroup_proof=cgroup_proof,
        terminal_metadata=terminal,
        execution_receipt=execution_receipt,
        receipt_lifecycle_file_sha256=lifecycle.file_sha256,
        receipt_lifecycle_body_sha256=lifecycle.body_sha256,
        receipt_terminal_metadata_file_sha256=terminal.file_sha256,
        receipt_terminal_metadata_body_sha256=terminal.body_sha256,
        observation_handoff=observation_handoff,
        handoff_execution_receipt_file_sha256=execution_receipt.file_sha256,
        handoff_execution_receipt_body_sha256=execution_receipt.body_sha256,
        handoff_terminal_metadata_file_sha256=terminal.file_sha256,
        handoff_terminal_metadata_body_sha256=terminal.body_sha256,
        endpoint_observer_request_file_sha256=None,
        endpoint_observer_request_body_sha256=None,
        endpoint_observer_receipt_file_sha256=None,
        endpoint_observer_receipt_body_sha256=None,
        go_ready_file_sha256=ready.file_sha256,
        go_ready_body_sha256=ready.body_sha256,
        go_observer_anchor_file_sha256=observer_anchor.file_sha256,
        go_observer_anchor_body_sha256=observer_anchor.body_sha256,
        request_algorithmic_measurement_intent_file_sha256=(algorithmic_intent.file_sha256),
        request_algorithmic_measurement_intent_body_sha256=(algorithmic_intent.body_sha256),
        ready_algorithmic_measurement_intent_file_sha256=(algorithmic_intent.file_sha256),
        ready_algorithmic_measurement_intent_body_sha256=(algorithmic_intent.body_sha256),
        terminal_algorithmic_resource_receipt_file_sha256=(algorithmic_receipt.file_sha256),
        terminal_algorithmic_resource_receipt_body_sha256=(algorithmic_receipt.body_sha256),
        request_storage_boundary_intent_file_sha256=storage_intent.file_sha256,
        request_storage_boundary_intent_body_sha256=storage_intent.body_sha256,
        ready_storage_boundary_intent_file_sha256=storage_intent.file_sha256,
        ready_storage_boundary_intent_body_sha256=storage_intent.body_sha256,
        terminal_storage_write_seal=storage_write_seal,
        terminal_storage_boundary_receipt_file_sha256=storage_receipt.file_sha256,
        terminal_storage_boundary_receipt_body_sha256=storage_receipt.body_sha256,
        publication_address_sha256=publication_files[0].sha256,
        publication_commitment_wrapper_file_sha256=publication_wrapper.file_sha256,
        publication_commitment_wrapper_body_sha256=publication_wrapper.body_sha256,
        publisher_descriptor_sha256=publisher.descriptor_sha256,
        publisher_source_sha256=publisher.source_sha256,
        terminal_publication_manifest_file_sha256=publication_files[0].sha256,
        terminal_publication_manifest_body_sha256=publication_manifest_body_sha256,
        terminal_published_bundle_sha256=published_bundle_sha256,
        terminal_reload_observation_sha256=reload_observation_sha256,
        terminal_publication_reload_validation=publication_reload_validation,
        storage_write_seal_reload_validation_file_sha256=(
            publication_reload_validation.file_sha256
        ),
        storage_write_seal_reload_validation_body_sha256=(
            publication_reload_validation.body_sha256
        ),
        terminal_file_inventory_sha256=file_inventory_sha256,
        terminal_file_count=len(publication_files),
        terminal_total_size_bytes=total_size_bytes,
        terminal_family_metadata=publisher_metadata,
        container_create_count=1,
        container_start_count=1,
        go_commit_count=1,
        workload_start_count=1,
        workload_exit_count=1,
        attempt_count=1,
        failure_count=0,
        returncode=0,
        timed_out=False,
        execution_state="metadata_complete_non_authorizing",
        publication_state="committed",
        cleanup_state="proven_empty",
        case_consumed=True,
        same_case_retry_permitted=False,
    )
    execution_receipt_identity = observations.ArtifactIdentityV2(
        schema_version=observations.HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
        file_sha256=execution_receipt.file_sha256,
        body_sha256=execution_receipt.body_sha256,
    )
    resource_fields = []
    for index, (field_name, provenance_kind) in enumerate(
        zip(
            observations.RESOURCE_CEILING_FIELDS,
            observations.RESOURCE_PROVENANCE_KINDS,
            strict=True,
        )
    ):
        declared_ceiling = 10_000 + index
        observed_value = index + 1
        if field_name == "max_environment_interactions":
            declared_ceiling = observations.MATCHED_V3_HORIZON
            observed_value = observations.MATCHED_V3_HORIZON
        elif field_name == "max_attempt_count":
            declared_ceiling = observed_value = 1
        elif field_name == "max_failure_count":
            declared_ceiling = observed_value = 0
        provenance_receipt = (
            storage_receipt
            if provenance_kind == "host_storage_boundary_receipt"
            else algorithmic_receipt
            if provenance_kind == "algorithmic_resource_receipt"
            else execution_receipt_identity
            if provenance_kind == "host_execution_lifecycle"
            else cgroup_proof
        )
        resource_fields.append(
            observations.ResourceFieldCandidateV2(
                field_name=field_name,
                declared_ceiling=declared_ceiling,
                observed_value=observed_value,
                value_semantics=_value_semantics(field_name),
                provenance_kind=provenance_kind,
                provenance_receipt=provenance_receipt,
            )
        )
    fields = tuple(resource_fields)
    resource_merger = observations.ResourceMergerCandidateV2(
        schema_version=(observations.QUALIFICATION_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION),
        candidate_id=spine.candidate_id,
        candidate_family=spine.candidate_family,
        case_spine_sha256=spine.body_sha256,
        host_execution_receipt_file_sha256=execution_receipt.file_sha256,
        host_execution_receipt_body_sha256=execution_receipt.body_sha256,
        host_provisioning_receipt=campaign.host_provisioning_receipt,
        host_cgroup_proof=cgroup_proof,
        host_terminal_metadata=terminal,
        host_observation_handoff=observation_handoff,
        endpoint_corroboration_mode="absent_unavailable_nonblocking",
        endpoint_observer_request=None,
        endpoint_observer_receipt=None,
        algorithmic_resource_contract=campaign.algorithmic_resource_contract,
        algorithmic_measurement_intent=algorithmic_intent,
        algorithmic_resource_receipt=algorithmic_receipt,
        runner_execution_receipt=runner_execution_receipt,
        storage_boundary_contract=campaign.storage_boundary_contract,
        storage_boundary_intent=storage_intent,
        storage_write_seal=storage_write_seal,
        storage_boundary_receipt=storage_receipt,
        merger_receipt=merger_receipt,
        merger=campaign.full_resource_merger,
        resource_requirement_body_sha256=spine.resource_requirement_body_sha256,
        field_inventory_sha256=_inventory_digest(
            "fields",
            [item.to_dict() for item in fields],
        ),
        fields=fields,
    )
    publication = observations.PublicationCandidateV2(
        schema_version=observations.QUALIFICATION_PUBLICATION_CANDIDATE_V2_SCHEMA_VERSION,
        candidate_id=spine.candidate_id,
        candidate_family=spine.candidate_family,
        case_ordinal=spine.case_ordinal,
        qualification_case_id=spine.qualification_case_id,
        case_spine_sha256=spine.body_sha256,
        host_execution_receipt_file_sha256=execution_receipt.file_sha256,
        host_execution_receipt_body_sha256=execution_receipt.body_sha256,
        host_terminal_metadata_file_sha256=terminal.file_sha256,
        host_terminal_metadata_body_sha256=terminal.body_sha256,
        host_observation_handoff=observation_handoff,
        publisher_registry_entry_file_sha256=(spine.publisher_registry_entry.file_sha256),
        publisher_registry_entry_body_sha256=(spine.publisher_registry_entry.body_sha256),
        publisher_metadata=publisher_metadata,
        runner_execution_receipt=runner_execution_receipt,
        publisher=publisher,
        native_atomic_producer=native_atomic_producer,
        native_publication_receipt=native_receipt,
        publication_commitment_contract_descriptor_sha256=(
            "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
        ),
        publication_commitment_contract_source_sha256=(
            "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
        ),
        publication_commitment_wrapper=publication_wrapper,
        wrapper_case_spine_sha256=spine.body_sha256,
        wrapper_candidate_id=spine.candidate_id,
        wrapper_candidate_family=spine.candidate_family,
        wrapper_publisher_descriptor_sha256=publisher.descriptor_sha256,
        wrapper_publisher_source_sha256=publisher.source_sha256,
        wrapper_publisher_metadata_file_sha256=publisher_metadata.file_sha256,
        wrapper_publisher_metadata_body_sha256=publisher_metadata.body_sha256,
        wrapper_native_atomic_producer_descriptor_sha256=(native_atomic_producer.descriptor_sha256),
        wrapper_native_atomic_producer_source_sha256=(native_atomic_producer.source_sha256),
        wrapper_native_publication_receipt_file_sha256=(
            None if native_receipt is None else native_receipt.file_sha256
        ),
        wrapper_native_publication_receipt_body_sha256=(
            None if native_receipt is None else native_receipt.body_sha256
        ),
        wrapper_publication_address_sha256=publication_files[0].sha256,
        wrapper_file_inventory_sha256=file_inventory_sha256,
        wrapper_published_bundle_sha256=published_bundle_sha256,
        wrapper_expected_reload_observation_sha256=reload_observation_sha256,
        wrapper_video_slot_mode=video_slot_mode,
        publication_address_sha256=publication_files[0].sha256,
        publication_manifest_file_sha256=publication_files[0].sha256,
        publication_manifest_body_sha256=publication_manifest_body_sha256,
        published_bundle_sha256=published_bundle_sha256,
        reload_observation_sha256=reload_observation_sha256,
        publication_reload_validation=publication_reload_validation,
        reload_validation_wrapper_file_sha256=publication_wrapper.file_sha256,
        reload_validation_wrapper_body_sha256=publication_wrapper.body_sha256,
        reload_validation_expected_reload_observation_sha256=(reload_observation_sha256),
        reload_validation_actual_reload_observation_sha256=(reload_observation_sha256),
        reload_validation_reload_performed=True,
        reload_validation_read_only=True,
        file_inventory_sha256=file_inventory_sha256,
        file_count=len(publication_files),
        total_size_bytes=total_size_bytes,
        maximum_total_size_bytes=observations.MAX_PUBLICATION_TOTAL_BYTES,
        video_slot_mode=video_slot_mode,
        files=publication_files,
        publication_committed=True,
        value_decoding_performed=False,
        byte_transport_performed=False,
        retry_count=0,
    )
    probes = tuple(
        observations.ProbeCandidateV2(
            probe_kind=kind,
            schema_version=observations.PROBE_SCHEMA_BY_KIND[kind],
            file_sha256=_sha(f"{prefix}-{kind}-file"),
            body_sha256=_sha(f"{prefix}-{kind}-body"),
            case_spine_sha256=spine.body_sha256,
            host_execution_receipt_file_sha256=execution_receipt.file_sha256,
            host_execution_receipt_body_sha256=execution_receipt.body_sha256,
        )
        for kind in observations.PROBE_KINDS
    )
    return observations.QualificationCaseSuccessCandidateV2(
        case_spine=spine,
        host_success=host_success,
        probes=probes,
        resource_merger=resource_merger,
        publication=publication,
    )


def _failure_case(
    campaign: observations.CampaignSpineV2,
    ordinal: int,
    *,
    completed_count: int = 2,
    failure_effect_state: Literal["failed_before_commit", "commit_uncertain"] = (
        "commit_uncertain"
    ),
    recovery_states: dict[str, str] | None = None,
    recovery_only: bool = False,
) -> observations.QualificationCaseTerminalFailureCandidateV2:
    """Build one terminal failure from independent operational and cleanup facts."""

    success = _success_case(campaign, ordinal)
    spine = success.case_spine
    source_host = success.host_success
    publication = success.publication
    resources = success.resource_merger
    prefix = spine.qualification_case_id
    operational = _EXPECTED_OPERATIONAL_PHASES
    assert 0 <= completed_count <= len(operational)
    if recovery_only:
        completed_phases = operational
        failure_phase = None
        operational_effect = None
    else:
        assert completed_count < len(operational)
        completed_phases = operational[:completed_count]
        failure_phase = operational[completed_count]
        operational_effect = failure_effect_state
    completed = set(completed_phases)

    def phase_state(
        phase: str,
    ) -> Literal["not_started", "failed_before_commit", "commit_uncertain", "committed"]:
        if phase in completed:
            return "committed"
        if phase == failure_phase:
            assert operational_effect is not None
            return operational_effect
        return "not_started"

    def boundary(
        phase: str,
    ) -> tuple[str, Literal["exact", "uncertain"], int | None]:
        state = phase_state(phase)
        if state == "committed":
            return state, "exact", 1
        if state == "commit_uncertain":
            return state, "uncertain", None
        return "not_started", "exact", 0

    container_create_state, container_create_count_state, container_create_count = boundary(
        "container_created"
    )
    container_start_state, container_start_count_state, container_start_count = boundary(
        "container_started"
    )
    workload_start_state, workload_start_count_state, workload_start_count = boundary(
        "workload_started"
    )
    workload_exit_state, workload_exit_count_state, workload_exit_count = boundary(
        "workload_exited"
    )
    attempt_count_state = workload_start_count_state
    attempt_count = workload_start_count
    failure_count = 0 if failure_phase is None else 1
    operational_frontier = _operational_frontier_artifact(
        spine=spine,
        completed_phases=completed_phases,
        failure_phase=failure_phase,
        failure_effect_state=operational_effect,
        container_create_state=container_create_state,
        container_start_state=container_start_state,
        workload_start_state=workload_start_state,
        workload_exit_state=workload_exit_state,
        container_create_count_state=container_create_count_state,
        container_create_count=container_create_count,
        container_start_count_state=container_start_count_state,
        container_start_count=container_start_count,
        workload_start_count_state=workload_start_count_state,
        workload_start_count=workload_start_count,
        workload_exit_count_state=workload_exit_count_state,
        workload_exit_count=workload_exit_count,
        attempt_count_state=attempt_count_state,
        attempt_count=attempt_count,
        failure_count=failure_count,
    )

    intent_state = phase_state("intent_committed")
    intent = source_host.intent if intent_state == "committed" else None
    initial_cgroup_sample_state = phase_state("initial_cgroup_sample_committed")
    initial_cgroup_sample = (
        source_host.initial_cgroup_sample if initial_cgroup_sample_state == "committed" else None
    )
    initial_sample_retained_fd_set_sha256 = (
        source_host.initial_sample_retained_fd_set_sha256
        if initial_cgroup_sample is not None
        else None
    )
    initial_sample_cgroup_identity_sha256 = (
        source_host.initial_sample_cgroup_identity_sha256
        if initial_cgroup_sample is not None
        else None
    )
    ready = source_host.ready if "driver_ready" in completed else None
    observer_anchor = source_host.observer_anchor if "observer_anchored" in completed else None
    go_commitment = source_host.go_commitment if "go_committed" in completed else None
    driver_terminal = source_host.driver_terminal if "workload_exited" in completed else None

    algorithmic_state = phase_state("algorithmic_resource_receipt_committed")
    algorithmic_resource_receipt = (
        resources.algorithmic_resource_receipt if algorithmic_state == "committed" else None
    )
    native_state = phase_state("native_publication_committed")
    native_atomic_producer = (
        publication.native_atomic_producer if native_state == "committed" else None
    )
    native_publication_receipt = (
        publication.native_publication_receipt if native_state == "committed" else None
    )
    expected_publication_address_sha256 = (
        publication.publication_address_sha256 if native_state == "committed" else None
    )
    publication_reconciliation_key_sha256 = (
        _sha(f"{prefix}-failure-publication-reconciliation-key")
        if native_state == "committed"
        else None
    )
    publication_reconciliation_reference = (
        _artifact(
            observations.QUALIFICATION_PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION,
            f"{prefix}-failure-publication-reconciliation-reference",
        )
        if native_state == "committed"
        else None
    )

    wrapper_state = phase_state("publication_commitment_wrapper_committed")
    publication_commitment_wrapper = (
        publication.publication_commitment_wrapper if wrapper_state == "committed" else None
    )
    failure_publication_projection = None
    if wrapper_state == "committed":
        assert algorithmic_resource_receipt is not None
        assert native_atomic_producer is not None
        assert publication_reconciliation_key_sha256 is not None
        assert publication_reconciliation_reference is not None
        assert publication_commitment_wrapper is not None
        failure_publication_projection = observations.FailurePublicationProjectionV2(
            schema_version=(
                observations.QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION
            ),
            case_spine_sha256=spine.body_sha256,
            case_ordinal=spine.case_ordinal,
            candidate_id=spine.candidate_id,
            candidate_family=spine.candidate_family,
            qualification_case_id=spine.qualification_case_id,
            publisher_registry_entry_file_sha256=(spine.publisher_registry_entry.file_sha256),
            publisher_registry_entry_body_sha256=(spine.publisher_registry_entry.body_sha256),
            algorithmic_resource_receipt=algorithmic_resource_receipt,
            runner_execution_receipt=publication.runner_execution_receipt,
            algorithmic_receipt_runner_file_sha256=(
                publication.runner_execution_receipt.file_sha256
            ),
            algorithmic_receipt_runner_body_sha256=(
                publication.runner_execution_receipt.body_sha256
            ),
            publisher=publication.publisher,
            publisher_metadata=publication.publisher_metadata,
            native_atomic_producer=native_atomic_producer,
            native_publication_receipt=native_publication_receipt,
            publication_reconciliation_key_sha256=(publication_reconciliation_key_sha256),
            publication_reconciliation_reference=(publication_reconciliation_reference),
            publication_commitment_contract_descriptor_sha256=(
                publication.publication_commitment_contract_descriptor_sha256
            ),
            publication_commitment_contract_source_sha256=(
                publication.publication_commitment_contract_source_sha256
            ),
            publication_commitment_wrapper=publication_commitment_wrapper,
            publication_address_sha256=publication.publication_address_sha256,
            publication_manifest_file_sha256=(publication.publication_manifest_file_sha256),
            publication_manifest_body_sha256=(publication.publication_manifest_body_sha256),
            file_inventory_sha256=publication.file_inventory_sha256,
            published_bundle_sha256=publication.published_bundle_sha256,
            expected_reload_observation_sha256=(publication.reload_observation_sha256),
            file_count=publication.file_count,
            total_size_bytes=publication.total_size_bytes,
            maximum_total_size_bytes=publication.maximum_total_size_bytes,
            video_slot_mode=publication.video_slot_mode,
            files=publication.files,
        )

    reload_state = phase_state("publication_reload_validated")
    publication_reload_validation = (
        publication.publication_reload_validation if reload_state == "committed" else None
    )
    reload_observation_sha256 = (
        publication.reload_observation_sha256 if reload_state == "committed" else None
    )
    storage_write_seal_state = phase_state("storage_write_seal_committed")
    storage_write_seal = (
        resources.storage_write_seal if storage_write_seal_state == "committed" else None
    )
    storage_boundary_receipt_state = phase_state("storage_boundary_receipt_committed")
    storage_boundary_receipt = (
        resources.storage_boundary_receipt
        if storage_boundary_receipt_state == "committed"
        else None
    )

    cleanup_cgroup_may_exist = "fresh_cgroup_created" in completed or (
        failure_phase == "fresh_cgroup_created" and operational_effect == "commit_uncertain"
    )
    if recovery_states is None:
        if not cleanup_cgroup_may_exist:
            recovery_states = {name: "not_applicable" for name in _EXPECTED_RECOVERY_NODE_NAMES}
        elif recovery_only:
            recovery_states = {
                "precleanup_cgroup_sample": "committed",
                "cgroup_kill": "failed_before_commit",
                "cgroup_empty": "committed",
                "container_absence": "committed",
                "post_container_remove_cgroup_sample": "committed",
                "cgroup_counter_fds_closed": "committed",
                "outer_cgroup_absence": "committed",
                "final_cgroup_proof": "failed_before_commit",
            }
        elif initial_cgroup_sample is not None:
            recovery_states = {name: "committed" for name in _EXPECTED_RECOVERY_NODE_NAMES}
        else:
            recovery_states = {
                name: "failed_before_commit" for name in _EXPECTED_RECOVERY_NODE_NAMES
            }
    assert tuple(recovery_states) == _EXPECTED_RECOVERY_NODE_NAMES
    recovery_nodes = _recovery_nodes(prefix, recovery_states)
    recovery_by_name = {item.node_name: item for item in recovery_nodes}
    unresolved_recovery_nodes = tuple(
        name
        for name in _EXPECTED_RECOVERY_NODE_NAMES
        if recovery_states[name] in {"commit_uncertain", "failed_before_commit"}
    )
    cleanup_proven = not cleanup_cgroup_may_exist or all(
        recovery_states[name] == "committed" for name in _EXPECTED_RECOVERY_NODE_NAMES
    )
    cleanup_reconciliation = _cleanup_reconciliation_artifact(
        spine=spine,
        operational_frontier=operational_frontier,
        cgroup_may_exist=cleanup_cgroup_may_exist,
        recovery_nodes=recovery_nodes,
    )

    def recovery_artifact(name: str) -> observations.ArtifactIdentityV2 | None:
        return recovery_by_name[name].artifact

    precleanup_cgroup_sample = recovery_artifact("precleanup_cgroup_sample")
    cgroup_kill_receipt = recovery_artifact("cgroup_kill")
    cgroup_empty_observation = recovery_artifact("cgroup_empty")
    container_absence_observation = recovery_artifact("container_absence")
    post_container_remove_cgroup_sample = recovery_artifact("post_container_remove_cgroup_sample")
    cgroup_counter_fds_closed_receipt = recovery_artifact("cgroup_counter_fds_closed")
    outer_cgroup_absence_observation = recovery_artifact("outer_cgroup_absence")
    cgroup_proof = recovery_artifact("final_cgroup_proof")
    post_committed = post_container_remove_cgroup_sample is not None
    close_committed = cgroup_counter_fds_closed_receipt is not None
    outer_committed = outer_cgroup_absence_observation is not None
    post_retained_fd_set_sha256 = (
        initial_sample_retained_fd_set_sha256
        if post_committed and initial_sample_retained_fd_set_sha256 is not None
        else _sha(f"{prefix}-failure-recovered-retained-fd-set")
        if post_committed
        else None
    )
    post_cgroup_identity_sha256 = (
        initial_sample_cgroup_identity_sha256
        if post_committed and initial_sample_cgroup_identity_sha256 is not None
        else _sha(f"{prefix}-failure-recovered-cgroup-identity")
        if post_committed
        else None
    )
    post_container_identity_sha256 = (
        _sha(f"{prefix}-failure-container-identity") if post_committed else None
    )
    close_post_file_sha256 = (
        post_container_remove_cgroup_sample.file_sha256
        if close_committed and post_container_remove_cgroup_sample is not None
        else None
    )
    close_post_body_sha256 = (
        post_container_remove_cgroup_sample.body_sha256
        if close_committed and post_container_remove_cgroup_sample is not None
        else None
    )
    close_retained_fd_set_sha256 = post_retained_fd_set_sha256 if close_committed else None
    close_cgroup_identity_sha256 = post_cgroup_identity_sha256 if close_committed else None
    close_container_identity_sha256 = post_container_identity_sha256 if close_committed else None
    outer_fd_close_file_sha256 = (
        cgroup_counter_fds_closed_receipt.file_sha256
        if outer_committed and cgroup_counter_fds_closed_receipt is not None
        else None
    )
    outer_fd_close_body_sha256 = (
        cgroup_counter_fds_closed_receipt.body_sha256
        if outer_committed and cgroup_counter_fds_closed_receipt is not None
        else None
    )
    outer_cgroup_identity_sha256 = close_cgroup_identity_sha256 if outer_committed else None
    recovery_failure_count = sum(item.state == "failed_before_commit" for item in recovery_nodes)
    recovery_uncertainty_count = sum(item.state == "commit_uncertain" for item in recovery_nodes)

    publication_states = (native_state, wrapper_state, reload_state)
    storage_states = (storage_write_seal_state, storage_boundary_receipt_state)
    uncertainty_flags = {
        "operational_state": operational_effect == "commit_uncertain",
        "publication_state": "commit_uncertain" in publication_states,
        "storage_state": "commit_uncertain" in storage_states,
        "cleanup_state": recovery_uncertainty_count > 0,
        "terminalization_state": False,
    }
    uncertainty_dimensions = tuple(
        name for name in observations.HOST_UNCERTAINTY_DIMENSIONS if uncertainty_flags[name]
    )
    returncode = 0 if driver_terminal is not None else None
    timed_out = False
    error_message_sha256 = _sha(f"{prefix}-failure-message")
    terminal_metadata = _terminal_metadata_artifact(
        spine=spine,
        record_kind="terminal_failure",
        operational_frontier=operational_frontier,
        cleanup_reconciliation=cleanup_reconciliation,
        driver_terminal=driver_terminal,
        algorithmic_resource_receipt=algorithmic_resource_receipt,
        publication_commitment_wrapper=publication_commitment_wrapper,
        publication_reload_validation=publication_reload_validation,
        storage_write_seal=storage_write_seal,
        storage_boundary_receipt=storage_boundary_receipt,
        returncode=returncode,
        timed_out=timed_out,
        error_message_sha256=error_message_sha256,
        cleanup_proven=cleanup_proven,
    )
    lifecycle = _artifact(
        observations.HOST_LIFECYCLE_SCHEMA_VERSION,
        f"{prefix}-failure-lifecycle",
    )
    failure_receipt = _artifact(
        observations.HOST_FAILURE_RECEIPT_SCHEMA_VERSION,
        f"{prefix}-failure-receipt",
    )
    observation_handoff = _host_handoff_artifact(
        spine=spine,
        record_kind="terminal_failure",
        terminal_receipt=failure_receipt,
        terminal_metadata=terminal_metadata,
    )
    classification = (
        "recovery_failure_after_complete_operational_frontier_nonretryable"
        if failure_phase is None
        else "operational_failed_before_commit_ticket_quarantined_nonretryable"
        if operational_effect == "failed_before_commit"
        else "operational_commit_uncertain_ticket_quarantined_nonretryable"
    )

    host_failure = observations.HostTerminalFailureCandidateV2(
        case_spine_sha256=spine.body_sha256,
        case_ordinal=spine.case_ordinal,
        candidate_id=spine.candidate_id,
        qualification_case_id=spine.qualification_case_id,
        qualification_plan_file_sha256=campaign.qualification_plan_file_sha256,
        qualification_plan_body_sha256=campaign.qualification_plan_body_sha256,
        case_execution_ticket_file_sha256=spine.case_execution_ticket.file_sha256,
        case_execution_ticket_body_sha256=spine.case_execution_ticket.body_sha256,
        qualification_case_manifest_file_sha256=(spine.qualification_case_manifest.file_sha256),
        qualification_case_manifest_body_sha256=(spine.qualification_case_manifest.body_sha256),
        publisher_registry_entry_file_sha256=(spine.publisher_registry_entry.file_sha256),
        publisher_registry_entry_body_sha256=(spine.publisher_registry_entry.body_sha256),
        resource_requirement_body_sha256=spine.resource_requirement_body_sha256,
        image_id=campaign.image_id,
        host_executor=campaign.host_executor,
        host_provisioning_receipt=campaign.host_provisioning_receipt,
        algorithmic_resource_contract=campaign.algorithmic_resource_contract,
        algorithmic_measurement_intent=resources.algorithmic_measurement_intent,
        algorithmic_measurement_intent_case_spine_sha256=spine.body_sha256,
        storage_boundary_contract=campaign.storage_boundary_contract,
        storage_boundary_intent=resources.storage_boundary_intent,
        storage_boundary_intent_case_spine_sha256=spine.body_sha256,
        request=source_host.request,
        authorization_request_file_sha256=source_host.request.file_sha256,
        authorization_request_body_sha256=source_host.request.body_sha256,
        intent_state=intent_state,
        intent=intent,
        initial_cgroup_sample_state=initial_cgroup_sample_state,
        initial_cgroup_sample=initial_cgroup_sample,
        initial_sample_intent_file_sha256=(
            intent.file_sha256 if initial_cgroup_sample is not None and intent is not None else None
        ),
        initial_sample_intent_body_sha256=(
            intent.body_sha256 if initial_cgroup_sample is not None and intent is not None else None
        ),
        initial_sample_retained_fd_set_sha256=initial_sample_retained_fd_set_sha256,
        initial_sample_cgroup_identity_sha256=initial_sample_cgroup_identity_sha256,
        ready=ready,
        ready_initial_cgroup_sample_file_sha256=(
            initial_cgroup_sample.file_sha256
            if ready is not None and initial_cgroup_sample is not None
            else None
        ),
        ready_initial_cgroup_sample_body_sha256=(
            initial_cgroup_sample.body_sha256
            if ready is not None and initial_cgroup_sample is not None
            else None
        ),
        ready_retained_fd_set_sha256=(
            initial_sample_retained_fd_set_sha256 if ready is not None else None
        ),
        ready_cgroup_identity_sha256=(
            initial_sample_cgroup_identity_sha256 if ready is not None else None
        ),
        observer_anchor=observer_anchor,
        observer_initial_cgroup_sample_file_sha256=(
            initial_cgroup_sample.file_sha256
            if observer_anchor is not None and initial_cgroup_sample is not None
            else None
        ),
        observer_initial_cgroup_sample_body_sha256=(
            initial_cgroup_sample.body_sha256
            if observer_anchor is not None and initial_cgroup_sample is not None
            else None
        ),
        observer_retained_fd_set_sha256=(
            initial_sample_retained_fd_set_sha256 if observer_anchor is not None else None
        ),
        observer_cgroup_identity_sha256=(
            initial_sample_cgroup_identity_sha256 if observer_anchor is not None else None
        ),
        go_commitment=go_commitment,
        go_ready_file_sha256=(
            ready.file_sha256 if go_commitment is not None and ready is not None else None
        ),
        go_ready_body_sha256=(
            ready.body_sha256 if go_commitment is not None and ready is not None else None
        ),
        go_observer_anchor_file_sha256=(
            observer_anchor.file_sha256
            if go_commitment is not None and observer_anchor is not None
            else None
        ),
        go_observer_anchor_body_sha256=(
            observer_anchor.body_sha256
            if go_commitment is not None and observer_anchor is not None
            else None
        ),
        go_retained_fd_set_sha256=(
            initial_sample_retained_fd_set_sha256 if go_commitment is not None else None
        ),
        go_cgroup_identity_sha256=(
            initial_sample_cgroup_identity_sha256 if go_commitment is not None else None
        ),
        operational_frontier=operational_frontier,
        completed_phases=completed_phases,
        failure_phase=failure_phase,
        failure_effect_state=operational_effect,
        container_create_state=container_create_state,
        container_start_state=container_start_state,
        workload_start_state=workload_start_state,
        workload_exit_state=workload_exit_state,
        container_create_count_state=container_create_count_state,
        container_create_count=container_create_count,
        container_start_count_state=container_start_count_state,
        container_start_count=container_start_count,
        workload_start_count_state=workload_start_count_state,
        workload_start_count=workload_start_count,
        workload_exit_count_state=workload_exit_count_state,
        workload_exit_count=workload_exit_count,
        attempt_count_state=attempt_count_state,
        attempt_count=attempt_count,
        failure_count_state="exact",
        failure_count=failure_count,
        case_consumed=True,
        same_case_retry_permitted=False,
        driver_terminal=driver_terminal,
        algorithmic_resource_receipt_state=algorithmic_state,
        algorithmic_resource_receipt=algorithmic_resource_receipt,
        algorithmic_resource_receipt_case_spine_sha256=(
            spine.body_sha256 if algorithmic_resource_receipt is not None else None
        ),
        native_publication_state=native_state,
        native_atomic_producer=native_atomic_producer,
        native_publication_receipt=native_publication_receipt,
        expected_publication_address_sha256=expected_publication_address_sha256,
        publication_reconciliation_key_sha256=(publication_reconciliation_key_sha256),
        publication_reconciliation_reference=publication_reconciliation_reference,
        failure_publication_projection=failure_publication_projection,
        publication_commitment_wrapper_state=wrapper_state,
        publication_commitment_wrapper=publication_commitment_wrapper,
        publication_reload_state=reload_state,
        publication_reload_validation=publication_reload_validation,
        reload_observation_sha256=reload_observation_sha256,
        storage_write_seal_state=storage_write_seal_state,
        storage_write_seal=storage_write_seal,
        storage_write_seal_case_spine_sha256=(
            spine.body_sha256 if storage_write_seal is not None else None
        ),
        storage_write_seal_reload_validation_file_sha256=(
            publication_reload_validation.file_sha256
            if storage_write_seal is not None and publication_reload_validation is not None
            else None
        ),
        storage_write_seal_reload_validation_body_sha256=(
            publication_reload_validation.body_sha256
            if storage_write_seal is not None and publication_reload_validation is not None
            else None
        ),
        storage_boundary_receipt_state=storage_boundary_receipt_state,
        storage_boundary_receipt=storage_boundary_receipt,
        storage_boundary_receipt_case_spine_sha256=(
            spine.body_sha256 if storage_boundary_receipt is not None else None
        ),
        storage_boundary_receipt_write_seal_file_sha256=(
            storage_write_seal.file_sha256
            if storage_boundary_receipt is not None and storage_write_seal is not None
            else None
        ),
        storage_boundary_receipt_write_seal_body_sha256=(
            storage_write_seal.body_sha256
            if storage_boundary_receipt is not None and storage_write_seal is not None
            else None
        ),
        recovery_nodes=recovery_nodes,
        cleanup_cgroup_may_exist=cleanup_cgroup_may_exist,
        cleanup_proven=cleanup_proven,
        cleanup_unresolved_recovery_nodes=unresolved_recovery_nodes,
        cleanup_recovery_complete=True,
        cleanup_terminalization_permitted=True,
        cleanup_workload_resume_permitted=False,
        cleanup_same_case_retry_permitted=False,
        cleanup_reconciliation=cleanup_reconciliation,
        precleanup_cgroup_sample=precleanup_cgroup_sample,
        cgroup_kill_receipt=cgroup_kill_receipt,
        cgroup_empty_observation=cgroup_empty_observation,
        container_absence_observation=container_absence_observation,
        post_container_remove_cgroup_sample=post_container_remove_cgroup_sample,
        cgroup_counter_fds_closed_receipt=cgroup_counter_fds_closed_receipt,
        outer_cgroup_absence_observation=outer_cgroup_absence_observation,
        cgroup_proof=cgroup_proof,
        post_container_remove_retained_fd_set_sha256=(post_retained_fd_set_sha256),
        post_container_remove_cgroup_identity_sha256=(post_cgroup_identity_sha256),
        post_container_remove_container_identity_sha256=(post_container_identity_sha256),
        cgroup_counter_fds_closed_post_sample_file_sha256=(close_post_file_sha256),
        cgroup_counter_fds_closed_post_sample_body_sha256=(close_post_body_sha256),
        cgroup_counter_fds_closed_retained_fd_set_sha256=(close_retained_fd_set_sha256),
        cgroup_counter_fds_closed_cgroup_identity_sha256=(close_cgroup_identity_sha256),
        cgroup_counter_fds_closed_container_identity_sha256=(close_container_identity_sha256),
        outer_cgroup_absence_fd_close_file_sha256=outer_fd_close_file_sha256,
        outer_cgroup_absence_fd_close_body_sha256=outer_fd_close_body_sha256,
        outer_cgroup_absence_cgroup_identity_sha256=(outer_cgroup_identity_sha256),
        recovery_failure_count_state="exact",
        recovery_failure_count=recovery_failure_count,
        recovery_uncertainty_count_state="exact",
        recovery_uncertainty_count=recovery_uncertainty_count,
        terminal_metadata_state="committed",
        terminal_metadata=terminal_metadata,
        returncode=returncode,
        timed_out=timed_out,
        exception_type="QualificationHostError",
        error_message_sha256=error_message_sha256,
        uncertainty_dimensions=uncertainty_dimensions,
        lifecycle=lifecycle,
        lifecycle_operational_frontier_file_sha256=operational_frontier.file_sha256,
        lifecycle_operational_frontier_body_sha256=operational_frontier.body_sha256,
        lifecycle_cleanup_reconciliation_file_sha256=(cleanup_reconciliation.file_sha256),
        lifecycle_cleanup_reconciliation_body_sha256=(cleanup_reconciliation.body_sha256),
        lifecycle_terminal_metadata_file_sha256=terminal_metadata.file_sha256,
        lifecycle_terminal_metadata_body_sha256=terminal_metadata.body_sha256,
        prior_host_execution_receipt_state="absent",
        prior_host_execution_receipt=None,
        failure_receipt_state="committed",
        failure_receipt=failure_receipt,
        receipt_lifecycle_file_sha256=lifecycle.file_sha256,
        receipt_lifecycle_body_sha256=lifecycle.body_sha256,
        receipt_terminal_metadata_file_sha256=terminal_metadata.file_sha256,
        receipt_terminal_metadata_body_sha256=terminal_metadata.body_sha256,
        receipt_operational_failure_count=failure_count,
        receipt_recovery_failure_count=recovery_failure_count,
        handoff_state="committed",
        observation_handoff=observation_handoff,
        classification=classification,
        ticket_quarantined=True,
        reconciliation_only=True,
        clean_rejection_recorded=False,
    )
    return observations.QualificationCaseTerminalFailureCandidateV2(
        case_spine=spine,
        terminal_failure=host_failure,
    )


def _case_sequence_inventory_digest(
    cases: tuple[observations.QualificationCaseCandidateV2, ...],
) -> str:
    records: list[dict[str, Any]] = []
    for item in cases:
        host: observations.HostSuccessCandidateV2 | observations.HostTerminalFailureCandidateV2
        if isinstance(item, observations.QualificationCaseSuccessCandidateV2):
            host = item.host_success
            terminal_receipt = item.host_success.execution_receipt
        else:
            assert isinstance(item, observations.QualificationCaseTerminalFailureCandidateV2)
            host = item.terminal_failure
            terminal_receipt = item.terminal_failure.failure_receipt
        terminal_metadata = host.terminal_metadata
        handoff = host.observation_handoff
        assert terminal_receipt is not None
        assert terminal_metadata is not None
        assert handoff is not None
        records.append(
            {
                "case_ordinal": item.case_spine.case_ordinal,
                "candidate_id": item.case_spine.candidate_id,
                "qualification_case_id": item.case_spine.qualification_case_id,
                "record_kind": item.record_kind,
                "case_spine_body_sha256": item.case_spine.body_sha256,
                "terminal_receipt_file_sha256": terminal_receipt.file_sha256,
                "terminal_receipt_body_sha256": terminal_receipt.body_sha256,
                "terminal_metadata_file_sha256": terminal_metadata.file_sha256,
                "terminal_metadata_body_sha256": terminal_metadata.body_sha256,
                "observation_handoff_file_sha256": handoff.file_sha256,
                "observation_handoff_body_sha256": handoff.body_sha256,
            }
        )
    return hashlib.sha256(_canonical({"cases": records}, newline=False)).hexdigest()


def _batch(
    success_ordinals: set[int] | frozenset[int],
) -> observations.MatchedV3QualificationObservationCandidateBatchV2:
    campaign = _campaign()
    cases = tuple(
        _success_case(campaign, ordinal)
        if ordinal in success_ordinals
        else _failure_case(campaign, ordinal)
        for ordinal in range(len(observations.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS))
    )
    inventory_sha256 = _case_sequence_inventory_digest(cases)
    receipt_body = {
        "schema_version": ("alberta.forager_matched_v3.qualification_all_case_sequence_receipt.v1"),
        "all_case_sequence_intent_file_sha256": (campaign.all_case_sequence_intent.file_sha256),
        "all_case_sequence_intent_body_sha256": (campaign.all_case_sequence_intent.body_sha256),
        "campaign_spine_body_sha256": campaign.body_sha256,
        "ordered_terminal_handoff_inventory_sha256": inventory_sha256,
        "case_count": 28,
        "terminal_coverage_complete": True,
    }
    sequence_receipt = _canonical_artifact(
        "alberta.forager_matched_v3.qualification_all_case_sequence_receipt.v1",
        receipt_body,
        "all_case_sequence_receipt_body_sha256",
    )
    return observations.MatchedV3QualificationObservationCandidateBatchV2(
        campaign_spine=campaign,
        cases=cases,
        all_case_sequence_receipt=sequence_receipt,
        all_case_sequence_receipt_intent_file_sha256=(
            campaign.all_case_sequence_intent.file_sha256
        ),
        all_case_sequence_receipt_intent_body_sha256=(
            campaign.all_case_sequence_intent.body_sha256
        ),
        all_case_sequence_receipt_campaign_spine_sha256=campaign.body_sha256,
        all_case_sequence_receipt_cases_inventory_sha256=inventory_sha256,
        all_case_sequence_receipt_case_count=28,
        all_case_sequence_receipt_terminal_coverage_complete=True,
    )


def _batch_with_cases(
    batch: observations.MatchedV3QualificationObservationCandidateBatchV2,
    cases: tuple[observations.QualificationCaseCandidateV2, ...],
) -> observations.MatchedV3QualificationObservationCandidateBatchV2:
    inventory_sha256 = _case_sequence_inventory_digest(cases)
    receipt = _canonical_artifact(
        "alberta.forager_matched_v3.qualification_all_case_sequence_receipt.v1",
        {
            "schema_version": (
                "alberta.forager_matched_v3.qualification_all_case_sequence_receipt.v1"
            ),
            "all_case_sequence_intent_file_sha256": (
                batch.campaign_spine.all_case_sequence_intent.file_sha256
            ),
            "all_case_sequence_intent_body_sha256": (
                batch.campaign_spine.all_case_sequence_intent.body_sha256
            ),
            "campaign_spine_body_sha256": batch.campaign_spine.body_sha256,
            "ordered_terminal_handoff_inventory_sha256": inventory_sha256,
            "case_count": 28,
            "terminal_coverage_complete": True,
        },
        "all_case_sequence_receipt_body_sha256",
    )
    return dataclasses.replace(
        batch,
        cases=cases,
        all_case_sequence_receipt=receipt,
        all_case_sequence_receipt_cases_inventory_sha256=inventory_sha256,
    )


def _pins(
    batch: observations.MatchedV3QualificationObservationCandidateBatchV2,
    raw: bytes,
) -> observations.QualificationObservationCandidateReplayPinsV2:
    spine = batch.campaign_spine
    return observations.QualificationObservationCandidateReplayPinsV2(
        batch_file_sha256=hashlib.sha256(raw).hexdigest(),
        qualification_plan_file_sha256=spine.qualification_plan_file_sha256,
        qualification_plan_body_sha256=spine.qualification_plan_body_sha256,
        observation_registry_source_sha256=spine.observation_registry_source_sha256,
        plan_issuance_receipt_file_sha256=(spine.plan_issuance_receipt.file_sha256),
        plan_issuance_receipt_body_sha256=(spine.plan_issuance_receipt.body_sha256),
        case_ticket_registry_file_sha256=spine.case_ticket_registry.file_sha256,
        case_ticket_registry_body_sha256=spine.case_ticket_registry.body_sha256,
        publisher_registry_file_sha256=spine.publisher_registry.file_sha256,
        publisher_registry_body_sha256=spine.publisher_registry.body_sha256,
        seed_registry_file_sha256=spine.seed_registry.file_sha256,
        seed_registry_body_sha256=spine.seed_registry.body_sha256,
        seed_pulse_record_file_sha256=spine.seed_pulse_record.file_sha256,
        seed_pulse_record_body_sha256=spine.seed_pulse_record.body_sha256,
        seed_trust_root_receipt_file_sha256=(spine.seed_trust_root_receipt.file_sha256),
        seed_trust_root_receipt_body_sha256=(spine.seed_trust_root_receipt.body_sha256),
        quicknet_verifier_descriptor_sha256=(spine.quicknet_verifier.descriptor_sha256),
        quicknet_verifier_source_sha256=spine.quicknet_verifier.source_sha256,
        quicknet_verifier_binary_sha256=spine.quicknet_verifier_binary_sha256,
        quicknet_verifier_receipt_file_sha256=(spine.quicknet_verifier_receipt.file_sha256),
        quicknet_verifier_receipt_body_sha256=(spine.quicknet_verifier_receipt.body_sha256),
        seed_chronology_receipt_file_sha256=(spine.seed_chronology_receipt.file_sha256),
        seed_chronology_receipt_body_sha256=(spine.seed_chronology_receipt.body_sha256),
        local_source_candidate_file_sha256=spine.local_source_candidate.file_sha256,
        local_source_candidate_body_sha256=spine.local_source_candidate.body_sha256,
        external_source_candidate_file_sha256=(spine.external_source_candidate.file_sha256),
        external_source_candidate_body_sha256=(spine.external_source_candidate.body_sha256),
        adapter_source_candidate_file_sha256=(spine.adapter_source_candidate.file_sha256),
        adapter_source_candidate_body_sha256=(spine.adapter_source_candidate.body_sha256),
        joint_source_closure_candidate_file_sha256=(
            spine.joint_source_closure_candidate.file_sha256
        ),
        joint_source_closure_candidate_body_sha256=(
            spine.joint_source_closure_candidate.body_sha256
        ),
        sealed_staging_candidate_file_sha256=(spine.sealed_staging_candidate.file_sha256),
        sealed_staging_candidate_body_sha256=(spine.sealed_staging_candidate.body_sha256),
        fresh_build_candidate_file_sha256=spine.fresh_build_candidate.file_sha256,
        fresh_build_candidate_body_sha256=spine.fresh_build_candidate.body_sha256,
        runtime_candidate_file_sha256=spine.runtime_candidate.file_sha256,
        runtime_candidate_body_sha256=spine.runtime_candidate.body_sha256,
        runtime_qualification_receipt_file_sha256=(spine.runtime_qualification_receipt.file_sha256),
        runtime_qualification_receipt_body_sha256=(spine.runtime_qualification_receipt.body_sha256),
        host_provisioning_receipt_file_sha256=(spine.host_provisioning_receipt.file_sha256),
        host_provisioning_receipt_body_sha256=(spine.host_provisioning_receipt.body_sha256),
        host_executor_descriptor_sha256=spine.host_executor.descriptor_sha256,
        host_executor_source_sha256=spine.host_executor.source_sha256,
        full_resource_merger_descriptor_sha256=(spine.full_resource_merger.descriptor_sha256),
        full_resource_merger_source_sha256=spine.full_resource_merger.source_sha256,
        algorithmic_resource_contract_descriptor_sha256=(
            spine.algorithmic_resource_contract.descriptor_sha256
        ),
        algorithmic_resource_contract_source_sha256=(
            spine.algorithmic_resource_contract.source_sha256
        ),
        storage_boundary_contract_descriptor_sha256=(
            spine.storage_boundary_contract.descriptor_sha256
        ),
        storage_boundary_contract_source_sha256=spine.storage_boundary_contract.source_sha256,
        normalized_publication_contract_descriptor_sha256=(
            "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
        ),
        normalized_publication_contract_source_sha256=(
            "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
        ),
        all_case_sequence_intent_file_sha256=(spine.all_case_sequence_intent.file_sha256),
        all_case_sequence_intent_body_sha256=(spine.all_case_sequence_intent.body_sha256),
        all_case_sequence_receipt_file_sha256=(batch.all_case_sequence_receipt.file_sha256),
        all_case_sequence_receipt_body_sha256=(batch.all_case_sequence_receipt.body_sha256),
        all_case_sequence_cases_inventory_sha256=(
            batch.all_case_sequence_receipt_cases_inventory_sha256
        ),
        candidate_order_sha256=spine.candidate_order_sha256,
        resource_field_order_sha256=spine.resource_field_order_sha256,
        image_id=spine.image_id,
    )


def _encoded(
    success_ordinals: set[int] | frozenset[int],
) -> tuple[
    observations.MatchedV3QualificationObservationCandidateBatchV2,
    bytes,
    observations.QualificationObservationCandidateReplayPinsV2,
]:
    batch = _batch(success_ordinals)
    raw = observations.canonical_matched_v3_qualification_observation_candidate_batch_v2_bytes(
        batch
    )
    return batch, raw, _pins(batch, raw)


def test_descriptor_body_without_lf_has_independent_zero_sentinel_until_pin() -> None:
    descriptor = observations.matched_v3_qualification_observation_registry_v2_descriptor()
    file_bytes = (
        observations.canonical_matched_v3_qualification_observation_registry_v2_descriptor_bytes()
    )
    body_bytes = _canonical(descriptor, newline=False)

    assert not body_bytes.endswith(b"\n")
    assert file_bytes == body_bytes + b"\n"
    assert hashlib.sha256(body_bytes).hexdigest() == _EXPECTED_DESCRIPTOR_BODY_SHA256


def test_descriptor_file_with_one_lf_has_independent_zero_sentinel_until_pin() -> None:
    descriptor = observations.matched_v3_qualification_observation_registry_v2_descriptor()
    file_bytes = (
        observations.canonical_matched_v3_qualification_observation_registry_v2_descriptor_bytes()
    )

    assert file_bytes == _canonical(descriptor)
    assert file_bytes.endswith(b"\n")
    assert not file_bytes.endswith(b"\n\n")
    assert hashlib.sha256(file_bytes).hexdigest() == _EXPECTED_DESCRIPTOR_FILE_SHA256


def test_descriptor_freezes_full_28_order_families_resources_and_no_authority() -> None:
    descriptor = observations.matched_v3_qualification_observation_registry_v2_descriptor()
    raw = observations.canonical_matched_v3_qualification_observation_registry_v2_descriptor_bytes()
    binding = descriptor["plan_v3_literal_binding"]

    assert len(observations.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS) == 28
    assert observations.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS == _EXPECTED_CANDIDATE_ORDER
    assert binding["candidate_order"] == list(_EXPECTED_CANDIDATE_ORDER)
    assert tuple(binding["candidate_order"][:14]) == (observations.MATCHED_V3_LOCAL_CANDIDATE_IDS)
    assert (
        tuple(binding["candidate_order"][14:23])
        == (observations.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9])
    )
    assert tuple(binding["candidate_order"][23:25]) == (
        observations.MATCHED_V3_ADAPTER_CANDIDATE_IDS
    )
    assert (
        tuple(binding["candidate_order"][25:])
        == (observations.MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:])
    )
    assert {key: len(value) for key, value in binding["families"].items()} == {
        "local": 14,
        "external": 12,
        "adapter": 2,
    }
    assert observations.RESOURCE_CEILING_FIELDS == _EXPECTED_RESOURCE_FIELDS
    assert binding["resource_fields"] == list(_EXPECTED_RESOURCE_FIELDS)
    assert len(binding["resource_fields"]) == 28
    assert descriptor["resource_merger_contract"]["within_ceiling_decision_made"] is False
    assert descriptor["resource_merger_contract"]["production_receipt_available"] is False
    assert descriptor["resource_merger_contract"]["endpoint_corroboration_required"] is False
    batch_contract = descriptor["batch_contract"]
    assert batch_contract["pre_execution_sequence_intent_claims_completion"] is False
    assert batch_contract["post_case_sequence_receipt_binds_ordered_terminal_inventory"] is True
    assert batch_contract["post_case_sequence_receipt_binds_batch_file_or_body"] is False
    assert (
        descriptor["host_contract"]["production_executor_descriptor_schema_version"]
        == observations.PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION
    )
    resource_contract = descriptor["resource_merger_contract"]
    assert resource_contract["producer_descriptor_schema_version"] == (
        observations.FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION
    )
    assert resource_contract["endpoint_request_schema_version"] == (
        observations.ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION
    )
    assert resource_contract["endpoint_receipt_schema_version"] == (
        observations.ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION
    )
    assert resource_contract["algorithmic_intent_bound_in_request_and_ready"] is True
    assert resource_contract["algorithmic_receipt_bound_at_terminal"] is True
    assert resource_contract["algorithmic_contract_descriptor_sha256"] == (
        "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
    )
    assert resource_contract["algorithmic_contract_source_sha256"] == (
        "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
    )
    assert resource_contract["storage_intent_bound_at_ready"] is True
    assert resource_contract["storage_write_seal_precedes_storage_receipt"] is True
    assert resource_contract["storage_receipt_bound_at_terminal"] is True
    assert resource_contract["storage_boundary_contract_descriptor_sha256"] == (
        "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
    )
    assert resource_contract["storage_boundary_contract_source_sha256"] == (
        "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"
    )
    assert resource_contract["merger_is_external_and_opaque_to_this_registry"] is True
    assert resource_contract["merger_internal_authentication_performed_by_this_registry"] is False
    assert (
        resource_contract["terminal_metadata_is_sole_post_receipt_delivery_emission_evidence"]
        is True
    )
    assert resource_contract["runner_execution_receipt_crosslinked_to_publication"] is True
    assert resource_contract["cpu_delta_envelope"] == (
        "fresh_cgroup_initial_empty_to_post_container_remove"
    )
    assert resource_contract["wall_delta_envelope"] == (
        "fresh_cgroup_initial_empty_to_post_container_remove"
    )
    seed_contract = descriptor["seed_identity_contract"]
    assert seed_contract["case_derivation_identities_unique_per_role"] is True
    assert seed_contract["cross_role_derivation_digest_inequality_required"] is False
    host_contract = descriptor["host_contract"]
    assert host_contract["lifecycle_phases"] == list(observations.HOST_LIFECYCLE_PHASES)
    assert host_contract["failure_operational_phases"] == list(_EXPECTED_OPERATIONAL_PHASES)
    assert host_contract["failure_recovery_nodes"] == list(_EXPECTED_RECOVERY_NODE_NAMES)
    assert host_contract["failure_recovery_node_states"] == [
        "not_applicable",
        "committed",
        "commit_uncertain",
        "failed_before_commit",
    ]
    assert host_contract["failure_recovery_node_dependencies"] == {
        name: list(dependencies)
        for name, dependencies in _EXPECTED_RECOVERY_NODE_DEPENDENCIES.items()
    }
    assert host_contract["phase_failure_side_effect_states"] == [
        "failed_before_commit",
        "commit_uncertain",
    ]
    assert host_contract["intent_states"] == [
        "not_started",
        "failed_before_commit",
        "commit_uncertain",
        "committed",
    ]
    assert host_contract["authorization_ack_carried_by_request"] is True
    assert host_contract["separate_authorization_artifact"] is False
    assert host_contract["initial_cgroup_sample_carries_retained_fd_inventory"] is True
    assert host_contract["separate_retained_fd_inventory_artifact"] is False
    assert (
        host_contract["container_create_start_and_workload_boundaries_live_in_operational_frontier"]
        is True
    )
    assert host_contract["terminal_metadata_body_keys"] == list(_EXPECTED_TERMINAL_BODY_KEYS)
    assert host_contract["terminal_timed_out_is_exact_boolean_for_both_record_kinds"] is True
    assert host_contract["terminal_failure_error_message_sha256_required"] is True
    assert host_contract["terminal_metadata_does_not_reverse_pin_lifecycle"] is True
    assert host_contract["acyclic_terminalization_order"] == (
        "operational_frontier_to_cleanup_reconciliation_to_terminal_metadata_to_"
        "lifecycle_to_receipt_to_handoff"
    )
    assert host_contract["cleanup_container_branch_has_no_hidden_initial_sample_dependency"] is True
    assert host_contract["terminal_metadata_envelope_location"] == "outer_host"
    assert host_contract["terminal_metadata_is_in_container_record"] is False
    assert host_contract["observer_anchor_schema_version"] == (
        observations.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION
    )
    assert host_contract["observation_handoff_schema_version"] == (
        observations.HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION
    )
    assert host_contract["ready_contains_host_sample"] is False
    assert host_contract["current_v1_containment_qualified"] is False
    assert host_contract["uncertainty_dimensions"] == list(observations.HOST_UNCERTAINTY_DIMENSIONS)
    supply_chain = descriptor["source_build_supply_chain_contract"]
    assert supply_chain["adapter_source_schema_version"] == (
        observations.ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION
    )
    assert supply_chain["joint_source_closure_schema_version"] == (
        observations.JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION
    )
    assert supply_chain["sealed_staging_schema_version"] == (
        observations.SEALED_STAGING_CANDIDATE_SCHEMA_VERSION
    )
    assert supply_chain["historical_build_or_image_can_substitute"] is False
    publication_contracts = {item["family"]: item for item in descriptor["publication_contracts"]}
    assert (
        publication_contracts["local"]["native_atomic_producer_descriptor_schema_version"]
        == observations.ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    )
    assert (
        publication_contracts["external"]["native_atomic_producer_descriptor_schema_version"]
        == observations.ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    )
    assert (
        publication_contracts["adapter"]["native_atomic_producer_descriptor_schema_version"]
        == observations.STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    )
    normalization = descriptor["publication_normalization"]
    assert normalization["wrapper_canonical_builder_implemented"] is True
    assert normalization["wrapper_production_available"] is False
    assert normalization["wrapper_contract_descriptor_sha256"] == (
        "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
    )
    assert normalization["wrapper_contract_source_sha256"] == (
        "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
    )
    assert normalization["wrapper_precommits_expected_reload_observation_digest"] is True
    assert normalization["wrapper_does_not_claim_reload_performed"] is True
    assert normalization["reload_validation_binds_expected_and_actual_reload_digests"] is True
    assert normalization["reload_validation_requires_expected_equals_actual"] is True
    reload_encoding = normalization["reload_validation_canonical_encoding"]
    assert reload_encoding == {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
        ),
        "body_projection_keys": [
            "schema_version",
            "publication_commitment_wrapper_file_sha256",
            "publication_commitment_wrapper_body_sha256",
            "expected_reload_observation_sha256",
            "actual_reload_observation_sha256",
            "reload_performed",
            "reload_read_only",
        ],
        "body_json_encoding": "compact_sorted_ascii_without_lf",
        "body_sha256_field": "reload_validation_body_sha256",
        "file_projection_is_body_plus_body_sha256_field": True,
        "file_json_encoding": "compact_sorted_ascii_with_exactly_one_trailing_lf",
        "file_and_body_sha256_independently_derived": True,
    }
    assert normalization["reload_validation_precedes_storage_write_seal"] is True
    assert normalization["wrapper_binds_case_spine"] is True
    assert normalization["wrapper_binds_outer_publisher_descriptor_and_source"] is True
    incompatibilities = descriptor["explicit_incompatibilities"]
    assert (
        incompatibilities["source_materialization_quicknet_descriptor_sha256"]
        == observations.SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256
    )
    assert incompatibilities["source_materialization_quicknet_source_sha256"] == (
        observations.SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256
    )
    assert incompatibilities["source_only_quicknet_build_descriptor_sha256"] == (
        observations.SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256
    )
    assert incompatibilities["source_only_quicknet_build_source_sha256"] == (
        observations.SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256
    )
    assert incompatibilities["historical_algorithmic_validator_descriptor_sha256s"] == [
        "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733"
    ]
    assert incompatibilities["historical_algorithmic_validator_source_sha256s"] == [
        "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8"
    ]
    assert incompatibilities["algorithmic_validator_cross_kind_merger_substitution_rejected"]
    for public_name in (
        "SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256",
        "SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256",
        "FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256",
        "FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256",
        "FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256",
        "FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256",
        "FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256",
        "FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256",
        "HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION",
        "HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION",
        "IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION",
        "RecoveryNodeCandidateV2",
        "host_operational_frontier_v2_body_projection",
        "host_cleanup_reconciliation_v2_body_projection",
        "host_terminal_metadata_v2_body_projection",
        "canonical_host_operational_frontier_v2_body_bytes",
        "canonical_host_cleanup_reconciliation_v2_file_bytes",
        "canonical_host_terminal_metadata_v2_file_bytes",
    ):
        assert public_name in observations.__all__
    assert incompatibilities["adapter_descriptor_sha256s"] == list(
        observations.INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S
    )
    assert incompatibilities["adapter_source_sha256s"] == list(
        observations.INCOMPATIBLE_ADAPTER_SOURCE_SHA256S
    )
    assert all(value is False for value in descriptor["claims"].values())
    assert all(value is False for value in descriptor["readiness"].values())
    assert all(value is False for value in descriptor["capabilities"].values())
    assert observations.QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256
    assert observations.matched_v3_qualification_observation_registry_v2_descriptor_sha256() == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        observations.parse_matched_v3_qualification_observation_registry_v2_descriptor(raw)
        == descriptor
    )


@pytest.mark.parametrize(
    "success_ordinals",
    (
        frozenset(range(28)),
        frozenset({0, 14, 23, 27}),
        frozenset(),
    ),
)
def test_full_batch_roundtrips_with_exactly_one_terminal_tag_per_ordinal(
    success_ordinals: frozenset[int],
) -> None:
    batch, raw, pins = _encoded(success_ordinals)

    parsed = observations.parse_matched_v3_qualification_observation_candidate_batch_v2(
        raw,
        pins=pins,
    )
    replayed = observations.replay_matched_v3_qualification_observation_candidate_batch_v2(
        raw,
        pins=pins,
    )

    assert parsed == batch
    assert replayed == batch
    assert len(parsed.cases) == 28
    assert [item.case_spine.case_ordinal for item in parsed.cases] == list(range(28))
    assert [item.case_spine.candidate_id for item in parsed.cases] == list(
        observations.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS
    )
    assert {
        index for index, item in enumerate(parsed.cases) if item.record_kind == "success"
    } == set(success_ordinals)
    value = parsed.to_dict()
    assert all(item is False for item in value["claims"].values())
    assert all(item is False for item in value["readiness"].values())
    assert value["terminal_coverage"] == {
        "required_case_count": 28,
        "same_case_retry_permitted": False,
        "selective_omission_permitted": False,
        "metadata_based_ordering_permitted": False,
        "terminal_failure_is_clean_rejection": False,
    }


def test_batch_rejects_case_omission_reordering_and_duplication() -> None:
    batch = _batch(frozenset())
    omitted = batch.cases[:-1]
    reordered = (batch.cases[1], batch.cases[0], *batch.cases[2:])
    duplicated = (batch.cases[0], batch.cases[0], *batch.cases[2:])

    for cases in (omitted, reordered, duplicated):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            _batch_with_cases(batch, cases)


@pytest.mark.parametrize(
    "missing_field",
    ("failure_receipt", "terminal_metadata", "observation_handoff"),
)
def test_batch_parser_rejects_missing_failure_receipt_terminal_or_handoff(
    missing_field: str,
) -> None:
    _, raw, pins = _encoded(frozenset())
    value = json.loads(raw)
    terminal_failure = value["cases"][0]["terminal_failure"]
    del terminal_failure[missing_field]
    body = {key: item for key, item in value.items() if key != "batch_body_sha256"}
    value["batch_body_sha256"] = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    mutated = _canonical(value)
    mutated_pins = dataclasses.replace(
        pins,
        batch_file_sha256=hashlib.sha256(mutated).hexdigest(),
    )

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.parse_matched_v3_qualification_observation_candidate_batch_v2(
            mutated,
            pins=mutated_pins,
        )


def test_sequence_intent_and_post_case_receipt_are_acyclic_and_exact() -> None:
    batch = _batch({0, 14, 23, 27})
    campaign = batch.campaign_spine

    assert campaign.all_case_sequence_intent_claims_completion is False
    assert "all_case_sequence_receipt" not in {
        field.name for field in dataclasses.fields(observations.CampaignSpineV2)
    }
    assert campaign.all_case_sequence_intent_candidate_order_sha256 == (
        campaign.candidate_order_sha256
    )
    assert batch.all_case_sequence_receipt_campaign_spine_sha256 == campaign.body_sha256
    assert batch.all_case_sequence_receipt_case_count == 28
    assert batch.all_case_sequence_receipt_terminal_coverage_complete is True
    assert "batch_body_sha256" not in {
        field.name
        for field in dataclasses.fields(
            observations.MatchedV3QualificationObservationCandidateBatchV2
        )
        if field.name.startswith("all_case_sequence_receipt_")
    }

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            all_case_sequence_intent_claims_completion=True,
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            all_case_sequence_intent=_artifact(
                observations.ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION,
                "noncanonical-all-case-intent",
            ),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            batch,
            all_case_sequence_receipt_campaign_spine_sha256=_sha("wrong-campaign"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            batch,
            all_case_sequence_receipt_cases_inventory_sha256=_sha("wrong-inventory"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            batch,
            all_case_sequence_receipt=_artifact(
                observations.ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION,
                "noncanonical-all-case-receipt",
            ),
        )


@pytest.mark.parametrize("identity_field", ("file_sha256", "body_sha256"))
def test_file_and_body_identity_mutations_fail_independently_across_outer_artifacts(
    identity_field: str,
) -> None:
    batch = _batch({0})
    success = batch.cases[0]
    failure = batch.cases[1]
    assert isinstance(success, observations.QualificationCaseSuccessCandidateV2)
    assert isinstance(failure, observations.QualificationCaseTerminalFailureCandidateV2)

    campaign_intent = dataclasses.replace(
        batch.campaign_spine.all_case_sequence_intent,
        **{identity_field: _sha("all-case-intent-" + identity_field)},
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            batch.campaign_spine,
            all_case_sequence_intent=campaign_intent,
        )

    sequence_receipt = dataclasses.replace(
        batch.all_case_sequence_receipt,
        **{identity_field: _sha("all-case-receipt-" + identity_field)},
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(batch, all_case_sequence_receipt=sequence_receipt)

    artifacts_and_owners = (
        (
            success.host_success,
            "operational_frontier",
            success.host_success.operational_frontier,
        ),
        (
            failure.terminal_failure,
            "operational_frontier",
            failure.terminal_failure.operational_frontier,
        ),
        (
            success.host_success,
            "cleanup_reconciliation",
            success.host_success.cleanup_reconciliation,
        ),
        (
            failure.terminal_failure,
            "cleanup_reconciliation",
            failure.terminal_failure.cleanup_reconciliation,
        ),
        (
            success.host_success,
            "terminal_metadata",
            success.host_success.terminal_metadata,
        ),
        (
            failure.terminal_failure,
            "terminal_metadata",
            failure.terminal_failure.terminal_metadata,
        ),
        (
            success.host_success,
            "observation_handoff",
            success.host_success.observation_handoff,
        ),
        (
            failure.terminal_failure,
            "observation_handoff",
            failure.terminal_failure.observation_handoff,
        ),
        (
            success.publication,
            "publication_commitment_wrapper",
            success.publication.publication_commitment_wrapper,
        ),
    )
    for owner, field_name, artifact in artifacts_and_owners:
        relabelled = dataclasses.replace(
            artifact,
            **{identity_field: _sha(field_name + "-" + identity_field)},
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(owner, **{field_name: relabelled})


def test_reload_validation_v1_canonical_body_and_file_encodings_are_exact() -> None:
    wrapper = _artifact(
        observations.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "canonical-reload-wrapper",
    )
    reload_observation_sha256 = _sha("canonical-reload-observation")
    body = {
        "schema_version": (
            "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
        ),
        "publication_commitment_wrapper_file_sha256": wrapper.file_sha256,
        "publication_commitment_wrapper_body_sha256": wrapper.body_sha256,
        "expected_reload_observation_sha256": reload_observation_sha256,
        "actual_reload_observation_sha256": reload_observation_sha256,
        "reload_performed": True,
        "reload_read_only": True,
    }
    body_bytes = _canonical(body, newline=False)
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()
    file_bytes = _canonical({**body, "reload_validation_body_sha256": body_sha256})

    assert observations.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_KEYS == tuple(body)
    assert (
        observations.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_SHA256_FIELD
        == "reload_validation_body_sha256"
    )
    assert (
        observations.qualification_publication_reload_validation_v1_body_projection(
            publication_commitment_wrapper_file_sha256=wrapper.file_sha256,
            publication_commitment_wrapper_body_sha256=wrapper.body_sha256,
            expected_reload_observation_sha256=reload_observation_sha256,
            actual_reload_observation_sha256=reload_observation_sha256,
            reload_performed=True,
            reload_read_only=True,
        )
        == body
    )
    assert (
        observations.canonical_qualification_publication_reload_validation_v1_body_bytes(
            publication_commitment_wrapper_file_sha256=wrapper.file_sha256,
            publication_commitment_wrapper_body_sha256=wrapper.body_sha256,
            expected_reload_observation_sha256=reload_observation_sha256,
            actual_reload_observation_sha256=reload_observation_sha256,
            reload_performed=True,
            reload_read_only=True,
        )
        == body_bytes
    )
    assert (
        observations.canonical_qualification_publication_reload_validation_v1_file_bytes(
            publication_commitment_wrapper_file_sha256=wrapper.file_sha256,
            publication_commitment_wrapper_body_sha256=wrapper.body_sha256,
            expected_reload_observation_sha256=reload_observation_sha256,
            actual_reload_observation_sha256=reload_observation_sha256,
            reload_performed=True,
            reload_read_only=True,
        )
        == file_bytes
    )
    assert file_bytes.endswith(b"\n")
    assert not file_bytes.endswith(b"\n\n")
    artifact = _reload_validation_artifact(
        wrapper,
        reload_observation_sha256,
        reload_observation_sha256,
    )
    assert artifact.body_sha256 == body_sha256
    assert artifact.file_sha256 == hashlib.sha256(file_bytes).hexdigest()


@pytest.mark.parametrize(
    ("expected", "actual", "reload_performed", "reload_read_only"),
    [
        (_sha("expected-a"), _sha("actual-b"), True, True),
        (_sha("same-c"), _sha("same-c"), False, True),
        (_sha("same-d"), _sha("same-d"), True, False),
        (_sha("same-e"), _sha("same-e"), 1, True),
        (_sha("same-f"), _sha("same-f"), True, 1),
    ],
)
def test_reload_validation_v1_projection_rejects_mismatch_false_and_bool_aliases(
    expected: str,
    actual: str,
    reload_performed: Any,
    reload_read_only: Any,
) -> None:
    wrapper = _artifact(
        observations.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "invalid-reload-wrapper",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.qualification_publication_reload_validation_v1_body_projection(
            publication_commitment_wrapper_file_sha256=wrapper.file_sha256,
            publication_commitment_wrapper_body_sha256=wrapper.body_sha256,
            expected_reload_observation_sha256=expected,
            actual_reload_observation_sha256=actual,
            reload_performed=reload_performed,
            reload_read_only=reload_read_only,
        )


def test_normalized_wrapper_body_and_file_identity_are_exact_and_source_only() -> None:
    publication = _success_case(_campaign(), 0).publication
    facts = {
        "case_spine_sha256": publication.case_spine_sha256,
        "case_ordinal": publication.case_ordinal,
        "candidate_id": publication.candidate_id,
        "candidate_family": publication.candidate_family,
        "qualification_case_id": publication.qualification_case_id,
        "publisher": publication.publisher,
        "publisher_metadata": publication.publisher_metadata,
        "native_atomic_producer": publication.native_atomic_producer,
        "native_publication_receipt": publication.native_publication_receipt,
        "publication_address_sha256": publication.publication_address_sha256,
        "publication_manifest_file_sha256": publication.publication_manifest_file_sha256,
        "publication_manifest_body_sha256": publication.publication_manifest_body_sha256,
        "file_inventory_sha256": publication.file_inventory_sha256,
        "published_bundle_sha256": publication.published_bundle_sha256,
        "expected_reload_observation_sha256": (
            publication.wrapper_expected_reload_observation_sha256
        ),
        "file_count": publication.file_count,
        "total_size_bytes": publication.total_size_bytes,
        "maximum_total_size_bytes": publication.maximum_total_size_bytes,
        "video_slot_mode": publication.video_slot_mode,
        "files": publication.files,
    }
    body = observations.normalized_publication_commitment_wrapper_v1_body_projection(**facts)
    body_bytes = observations.canonical_normalized_publication_commitment_wrapper_v1_body_bytes(
        **facts
    )
    file_bytes = observations.canonical_normalized_publication_commitment_wrapper_v1_file_bytes(
        **facts
    )
    assert observations.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_KEYS == (
        _EXPECTED_WRAPPER_BODY_KEYS
    )
    assert tuple(body) == _EXPECTED_WRAPPER_BODY_KEYS
    assert body_bytes == _canonical(body, newline=False)
    assert file_bytes.endswith(b"\n") and not file_bytes.endswith(b"\n\n")
    assert hashlib.sha256(body_bytes).hexdigest() == (
        publication.publication_commitment_wrapper.body_sha256
    )
    assert hashlib.sha256(file_bytes).hexdigest() == (
        publication.publication_commitment_wrapper.file_sha256
    )
    for mapping_name in ("capabilities", "readiness", "authority", "claims"):
        assert all(value is False for value in body[mapping_name].values())
    assert body["reload_performed_by_wrapper"] is False
    assert body["reload_digest_equality_validated_by_wrapper"] is False
    assert body["content_values_read_by_wrapper"] is False
    assert body["payload_bytes_transported_by_wrapper"] is False


def test_publication_family_inventories_native_forms_and_video_sentinels_are_exact() -> None:
    campaign = _campaign()
    local = _success_case(campaign, 0).publication
    continuing = _success_case(campaign, 14).publication
    ppo = _success_case(campaign, 21).publication
    adapter = _success_case(campaign, 23).publication

    assert tuple((item.role, item.name) for item in local.files) == (
        observations.LOCAL_PUBLICATION_ROLE_PATHS
    )
    assert tuple((item.role, item.name) for item in continuing.files) == (
        observations.EXTERNAL_PUBLICATION_ROLE_PATHS
    )
    assert tuple((item.role, item.name) for item in adapter.files) == (
        observations.ADAPTER_PUBLICATION_ROLE_PATHS
    )
    assert (local.file_count, continuing.file_count, adapter.file_count) == (9, 10, 5)
    assert local.native_publication_receipt is None
    assert continuing.native_publication_receipt is not None
    assert adapter.native_publication_receipt is not None
    assert local.runner_execution_receipt.schema_version == (
        observations.LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION
    )
    assert continuing.runner_execution_receipt.schema_version == (
        observations.EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION
    )
    assert adapter.runner_execution_receipt.schema_version == (
        observations.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
    )
    ppo_adapter = _success_case(campaign, 24).publication
    assert ppo_adapter.runner_execution_receipt.schema_version == (
        observations.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    )
    assert continuing.video_slot_mode == "absent_for_continuing_zero_length_slot"
    assert continuing.files[6].size_bytes == 0
    assert continuing.files[6].sha256 == observations.EMPTY_FILE_SHA256
    assert ppo.video_slot_mode == "opaque_ppo_video"
    assert ppo.files[6].size_bytes > 0
    for publication in (local, continuing, ppo, adapter):
        runner_role = {
            "local": "local_runner_receipt",
            "external": "execution_receipt",
            "adapter": "runner_result_receipt",
        }[publication.candidate_family]
        runner_file = next(item for item in publication.files if item.role == runner_role)
        assert runner_file.sha256 == publication.runner_execution_receipt.file_sha256
        assert publication.wrapper_case_spine_sha256 == publication.case_spine_sha256
        assert publication.wrapper_candidate_id == publication.candidate_id
        assert publication.wrapper_candidate_family == publication.candidate_family
        assert publication.wrapper_expected_reload_observation_sha256 == (
            publication.reload_observation_sha256
        )
        assert publication.reload_validation_wrapper_file_sha256 == (
            publication.publication_commitment_wrapper.file_sha256
        )
        assert publication.reload_validation_reload_performed is True
        assert publication.reload_validation_read_only is True
    assert observations.MAX_PUBLICATION_FILE_BYTES == 1024 * 1024 * 1024
    assert observations.MAX_PUBLICATION_TOTAL_BYTES == 1024 * 1024 * 1024


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("reload_validation_wrapper_file_sha256", _sha("wrong-reload-wrapper-file")),
        ("reload_validation_wrapper_body_sha256", _sha("wrong-reload-wrapper-body")),
        (
            "reload_validation_expected_reload_observation_sha256",
            _sha("wrong-expected-reload"),
        ),
        (
            "reload_validation_actual_reload_observation_sha256",
            _sha("wrong-actual-reload"),
        ),
        ("reload_validation_reload_performed", False),
        ("reload_validation_read_only", False),
        ("reload_validation_reload_performed", 1),
        ("reload_validation_read_only", 1),
    ],
)
def test_publication_rejects_every_reload_projection_mutation_and_bool_alias(
    field_name: str,
    invalid_value: Any,
) -> None:
    publication = _success_case(_campaign(), 0).publication
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(publication, **{field_name: invalid_value})


@pytest.mark.parametrize("identity_field", ["file_sha256", "body_sha256"])
def test_publication_rejects_relabelled_reload_validation_identity(
    identity_field: str,
) -> None:
    publication = _success_case(_campaign(), 0).publication
    relabelled = dataclasses.replace(
        publication.publication_reload_validation,
        **{identity_field: _sha("relabelled-reload-validation-" + identity_field)},
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            publication,
            publication_reload_validation=relabelled,
        )


@pytest.mark.parametrize("identity_field", ["file_sha256", "body_sha256"])
def test_failure_rejects_relabelled_committed_reload_validation_identity(
    identity_field: str,
) -> None:
    completed_count = _EXPECTED_OPERATIONAL_PHASES.index("storage_boundary_receipt_committed")
    failure = _failure_case(
        _campaign(),
        0,
        completed_count=completed_count,
    ).terminal_failure
    assert failure.publication_reload_validation is not None
    relabelled = dataclasses.replace(
        failure.publication_reload_validation,
        **{identity_field: _sha("relabelled-failure-reload-" + identity_field)},
    )
    replacements: dict[str, Any] = {
        "publication_reload_validation": relabelled,
    }
    if identity_field == "file_sha256":
        replacements["storage_write_seal_reload_validation_file_sha256"] = relabelled.file_sha256
    else:
        replacements["storage_write_seal_reload_validation_body_sha256"] = relabelled.body_sha256
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(failure, **replacements)


def test_failure_reload_expected_and_actual_must_both_equal_projection_digest() -> None:
    failure = _failure_case(_campaign(), 0, completed_count=17).terminal_failure
    projection = failure.failure_publication_projection
    wrapper = failure.publication_commitment_wrapper
    assert projection is not None
    assert wrapper is not None
    expected = projection.expected_reload_observation_sha256
    mismatch = _sha("failure-reload-projection-mismatch")

    for validation_expected, validation_actual, observed in (
        (mismatch, expected, expected),
        (expected, mismatch, expected),
        (mismatch, mismatch, mismatch),
    ):
        validation = _reload_validation_artifact(
            wrapper,
            validation_expected,
            validation_actual,
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                failure,
                publication_reload_validation=validation,
                reload_observation_sha256=observed,
                storage_write_seal_reload_validation_file_sha256=(validation.file_sha256),
                storage_write_seal_reload_validation_body_sha256=(validation.body_sha256),
            )


def test_resource_merger_freezes_provenance_storage_semantics_and_no_decision() -> None:
    success = _success_case(_campaign(), 0)
    merger = success.resource_merger
    by_name = {item.field_name: item for item in merger.fields}

    assert tuple(item.field_name for item in merger.fields) == _EXPECTED_RESOURCE_FIELDS
    assert tuple(item.provenance_kind for item in merger.fields) == (
        observations.RESOURCE_PROVENANCE_KINDS
    )
    assert merger.endpoint_corroboration_mode == "absent_unavailable_nonblocking"
    assert merger.endpoint_observer_request is None
    assert merger.endpoint_observer_receipt is None
    assert success.host_success.terminal_algorithmic_resource_receipt_file_sha256 == (
        merger.algorithmic_resource_receipt.file_sha256
    )
    assert success.host_success.terminal_algorithmic_resource_receipt_body_sha256 == (
        merger.algorithmic_resource_receipt.body_sha256
    )
    assert merger.storage_write_seal == success.host_success.terminal_storage_write_seal
    assert merger.runner_execution_receipt == success.publication.runner_execution_receipt
    assert merger.host_observation_handoff == success.host_success.observation_handoff
    assert success.publication.host_observation_handoff == (
        success.host_success.observation_handoff
    )
    assert success.host_success.attempt_count == by_name["max_attempt_count"].observed_value
    assert success.host_success.failure_count == by_name["max_failure_count"].observed_value
    assert by_name["max_peak_rss_bytes"].value_semantics == ("conservative_observed_upper_bound")
    assert by_name["max_thread_count"].value_semantics == ("conservative_observed_upper_bound")
    for name in ("max_temporary_peak_bytes", "max_disk_peak_bytes"):
        assert by_name[name].provenance_kind == "host_storage_boundary_receipt"
        assert by_name[name].provenance_receipt == merger.storage_boundary_receipt
    assert by_name["max_environment_interactions"].observed_value == 499_712
    assert by_name["max_attempt_count"].observed_value == 1
    assert by_name["max_failure_count"].observed_value == 0

    field = by_name["max_trainable_parameters"]
    replacement = dataclasses.replace(
        field,
        observed_value=field.declared_ceiling + 1,
    )
    fields = tuple(replacement if item is field else item for item in merger.fields)
    over_ceiling = dataclasses.replace(
        merger,
        fields=fields,
        field_inventory_sha256=_inventory_digest(
            "fields",
            [item.to_dict() for item in fields],
        ),
    )
    assert over_ceiling.fields[4].observed_value > over_ceiling.fields[4].declared_ceiling

    temporary = by_name["max_temporary_peak_bytes"]
    enforced_temporary = dataclasses.replace(
        temporary,
        value_semantics="conservative_enforced_upper_bound",
    )
    storage_fields = tuple(
        enforced_temporary if item is temporary else item for item in merger.fields
    )
    enforced_storage = dataclasses.replace(
        merger,
        fields=storage_fields,
        field_inventory_sha256=_inventory_digest(
            "fields",
            [item.to_dict() for item in storage_fields],
        ),
    )
    assert enforced_storage.fields[23].value_semantics == ("conservative_enforced_upper_bound")
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            by_name["max_peak_rss_bytes"],
            value_semantics="exact_observation",
        )


def test_host_success_projects_exact_lifecycle_anchor_go_and_counts() -> None:
    success = _success_case(_campaign(), 0)
    host = success.host_success

    assert host.completed_phases == _EXPECTED_OPERATIONAL_PHASES
    assert host.operational_frontier.schema_version == (
        observations.HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION
    )
    assert host.cleanup_reconciliation.schema_version == (
        observations.HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION
    )
    assert tuple(item.node_name for item in host.recovery_nodes) == (_EXPECTED_RECOVERY_NODE_NAMES)
    assert all(item.state == "committed" for item in host.recovery_nodes)
    assert host.ready.file_sha256 != host.observer_anchor.file_sha256
    assert host.ready.body_sha256 != host.observer_anchor.body_sha256
    assert host.go_ready_file_sha256 == host.ready.file_sha256
    assert host.go_ready_body_sha256 == host.ready.body_sha256
    assert host.go_observer_anchor_file_sha256 == host.observer_anchor.file_sha256
    assert host.go_observer_anchor_body_sha256 == host.observer_anchor.body_sha256
    assert host.handoff_execution_receipt_file_sha256 == (host.execution_receipt.file_sha256)
    assert host.handoff_terminal_metadata_body_sha256 == (host.terminal_metadata.body_sha256)
    assert (
        host.container_create_count,
        host.container_start_count,
        host.go_commit_count,
        host.workload_start_count,
        host.workload_exit_count,
        host.attempt_count,
        host.failure_count,
    ) == (1, 1, 1, 1, 1, 1, 0)

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(host, observer_anchor=host.ready)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(host, go_ready_body_sha256=_sha("wrong-ready-body"))
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(host, attempt_count=0)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            host,
            observation_handoff=_artifact(
                observations.HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
                "noncanonical-success-handoff",
            ),
        )


def test_request_authorization_and_initial_sample_fd_chains_fail_closed_both_ways() -> None:
    campaign = _campaign()
    success = _success_case(campaign, 0).host_success
    failure = _failure_case(campaign, 0, completed_count=11).terminal_failure

    for host in (success, failure):
        for field_name in (
            "authorization_request_file_sha256",
            "authorization_request_body_sha256",
            "initial_sample_intent_file_sha256",
            "initial_sample_intent_body_sha256",
            "initial_sample_retained_fd_set_sha256",
            "initial_sample_cgroup_identity_sha256",
            "ready_initial_cgroup_sample_file_sha256",
            "ready_initial_cgroup_sample_body_sha256",
            "ready_retained_fd_set_sha256",
            "ready_cgroup_identity_sha256",
            "observer_initial_cgroup_sample_file_sha256",
            "observer_initial_cgroup_sample_body_sha256",
            "observer_retained_fd_set_sha256",
            "observer_cgroup_identity_sha256",
            "go_retained_fd_set_sha256",
            "go_cgroup_identity_sha256",
        ):
            with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
                dataclasses.replace(
                    host,
                    **{field_name: _sha("crosslink-drift-" + field_name)},
                )

    assert failure.ready is not None
    assert failure.observer_anchor is not None
    aliased_observer = observations.ArtifactIdentityV2(
        schema_version=observations.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
        file_sha256=failure.ready.file_sha256,
        body_sha256=failure.ready.body_sha256,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            observer_anchor=aliased_observer,
            go_observer_anchor_file_sha256=aliased_observer.file_sha256,
            go_observer_anchor_body_sha256=aliased_observer.body_sha256,
        )


@pytest.mark.parametrize("record_kind", ("success", "terminal_failure"))
def test_common_outer_terminal_has_exact_body_and_file_encoding(
    record_kind: Literal["success", "terminal_failure"],
) -> None:
    campaign = _campaign()
    success = _success_case(campaign, 0)
    if record_kind == "success":
        host = success.host_success
        algorithmic_receipt = observations.ArtifactIdentityV2(
            schema_version=_resource_schema(success.case_spine.candidate_family),
            file_sha256=host.terminal_algorithmic_resource_receipt_file_sha256,
            body_sha256=host.terminal_algorithmic_resource_receipt_body_sha256,
        )
        wrapper = observations.ArtifactIdentityV2(
            schema_version=(observations.NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION),
            file_sha256=host.publication_commitment_wrapper_file_sha256,
            body_sha256=host.publication_commitment_wrapper_body_sha256,
        )
        storage_receipt = observations.ArtifactIdentityV2(
            schema_version=observations.QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
            file_sha256=host.terminal_storage_boundary_receipt_file_sha256,
            body_sha256=host.terminal_storage_boundary_receipt_body_sha256,
        )
        facts: dict[str, Any] = {
            "case_spine_sha256": host.case_spine_sha256,
            "case_ordinal": host.case_ordinal,
            "candidate_id": host.candidate_id,
            "candidate_family": success.case_spine.candidate_family,
            "qualification_case_id": host.qualification_case_id,
            "record_kind": record_kind,
            "operational_frontier": host.operational_frontier,
            "cleanup_reconciliation": host.cleanup_reconciliation,
            "driver_terminal": host.driver_terminal,
            "algorithmic_resource_receipt": algorithmic_receipt,
            "publication_commitment_wrapper": wrapper,
            "publication_reload_validation": host.terminal_publication_reload_validation,
            "storage_write_seal": host.terminal_storage_write_seal,
            "storage_boundary_receipt": storage_receipt,
            "returncode": host.returncode,
            "timed_out": host.timed_out,
            "error_message_sha256": None,
            "cleanup_proven": host.cleanup_proven,
            "case_consumed": host.case_consumed,
            "same_case_retry_permitted": host.same_case_retry_permitted,
        }
        terminal = host.terminal_metadata
    else:
        failure = _failure_case(campaign, 0, completed_count=17).terminal_failure
        facts = {
            "case_spine_sha256": failure.case_spine_sha256,
            "case_ordinal": failure.case_ordinal,
            "candidate_id": failure.candidate_id,
            "candidate_family": _family(failure.candidate_id),
            "qualification_case_id": failure.qualification_case_id,
            "record_kind": record_kind,
            "operational_frontier": failure.operational_frontier,
            "cleanup_reconciliation": failure.cleanup_reconciliation,
            "driver_terminal": failure.driver_terminal,
            "algorithmic_resource_receipt": failure.algorithmic_resource_receipt,
            "publication_commitment_wrapper": failure.publication_commitment_wrapper,
            "publication_reload_validation": failure.publication_reload_validation,
            "storage_write_seal": failure.storage_write_seal,
            "storage_boundary_receipt": failure.storage_boundary_receipt,
            "returncode": failure.returncode,
            "timed_out": failure.timed_out,
            "error_message_sha256": failure.error_message_sha256,
            "cleanup_proven": failure.cleanup_proven,
            "case_consumed": failure.case_consumed,
            "same_case_retry_permitted": failure.same_case_retry_permitted,
        }
        terminal = failure.terminal_metadata

    body = observations.host_terminal_metadata_v2_body_projection(**facts)
    body_bytes = observations.canonical_host_terminal_metadata_v2_body_bytes(**facts)
    file_bytes = observations.canonical_host_terminal_metadata_v2_file_bytes(**facts)
    assert tuple(body) == _EXPECTED_TERMINAL_BODY_KEYS
    assert "status" not in body
    assert not body_bytes.endswith(b"\n")
    assert file_bytes.endswith(b"\n")
    assert not file_bytes.endswith(b"\n\n")
    file_projection = json.loads(file_bytes)
    assert set(file_projection) == {
        *_EXPECTED_TERMINAL_BODY_KEYS,
        "terminal_metadata_body_sha256",
    }
    assert (
        file_projection["terminal_metadata_body_sha256"] == hashlib.sha256(body_bytes).hexdigest()
    )
    assert hashlib.sha256(body_bytes).hexdigest() == terminal.body_sha256
    assert hashlib.sha256(file_bytes).hexdigest() == terminal.file_sha256
    for invalid_timeout in (None, 0, 1):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            observations.host_terminal_metadata_v2_body_projection(
                **{**facts, "timed_out": invalid_timeout}
            )
    if record_kind == "terminal_failure":
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            observations.host_terminal_metadata_v2_body_projection(
                **{**facts, "error_message_sha256": None}
            )


def test_failure_outer_terminal_handoff_and_publication_projection_fail_closed() -> None:
    campaign = _campaign()
    failure_case = _failure_case(campaign, 0, completed_count=17)
    other_case = _failure_case(campaign, 1, completed_count=17)
    failure = failure_case.terminal_failure
    other = other_case.terminal_failure
    projection = failure.failure_publication_projection
    assert projection is not None
    assert other.failure_publication_projection is not None
    assert failure.failure_receipt_state == "committed"
    assert failure.terminal_metadata_state == "committed"
    assert failure.handoff_state == "committed"

    for artifact_field, identity_field in (
        ("terminal_metadata", "file_sha256"),
        ("terminal_metadata", "body_sha256"),
        ("failure_receipt", "file_sha256"),
        ("failure_receipt", "body_sha256"),
        ("observation_handoff", "file_sha256"),
        ("observation_handoff", "body_sha256"),
        ("publication_commitment_wrapper", "file_sha256"),
        ("publication_commitment_wrapper", "body_sha256"),
    ):
        artifact = getattr(failure, artifact_field)
        assert artifact is not None
        relabelled = dataclasses.replace(
            artifact,
            **{identity_field: _sha(f"{artifact_field}-{identity_field}-mutation")},
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(failure, **{artifact_field: relabelled})

    for field_name in ("terminal_metadata", "failure_receipt", "observation_handoff"):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(failure, **{field_name: None})

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            failure_publication_projection=other.failure_publication_projection,
        )
    for projection_field in (
        "publisher_registry_entry_file_sha256",
        "publisher_registry_entry_body_sha256",
        "publication_reconciliation_key_sha256",
    ):
        altered_projection = dataclasses.replace(
            projection,
            **{projection_field: _sha("same-case-" + projection_field)},
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                failure,
                failure_publication_projection=altered_projection,
            )
    assert failure.algorithmic_resource_receipt is not None
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            algorithmic_resource_receipt=_artifact(
                observations.LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
                "swapped-failure-algorithmic-receipt",
            ),
        )

    altered_expected = _sha("coordinated-wrapper-expected-reload")
    altered_wrapper = _publication_wrapper_artifact(
        spine=failure_case.case_spine,
        publisher=projection.publisher,
        publisher_metadata=projection.publisher_metadata,
        native_atomic_producer=projection.native_atomic_producer,
        native_publication_receipt=projection.native_publication_receipt,
        publication_address_sha256=projection.publication_address_sha256,
        publication_manifest_body_sha256=(projection.publication_manifest_body_sha256),
        file_inventory_sha256=projection.file_inventory_sha256,
        published_bundle_sha256=projection.published_bundle_sha256,
        expected_reload_observation_sha256=altered_expected,
        total_size_bytes=projection.total_size_bytes,
        video_slot_mode=projection.video_slot_mode,
        files=projection.files,
    )
    altered_projection = dataclasses.replace(
        projection,
        publication_commitment_wrapper=altered_wrapper,
        expected_reload_observation_sha256=altered_expected,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            publication_commitment_wrapper=altered_wrapper,
            failure_publication_projection=altered_projection,
        )


def test_failure_at_wrapper_validates_committed_native_family_identities_directly() -> None:
    campaign = _campaign()
    local = _failure_case(campaign, 0, completed_count=15).terminal_failure
    external = _failure_case(campaign, 14, completed_count=15).terminal_failure
    assert local.native_publication_state == "committed"
    assert local.failure_publication_projection is None
    assert external.native_publication_state == "committed"
    assert external.failure_publication_projection is None

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            local,
            native_atomic_producer=_producer(
                observations.LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
                "wrong-native-producer-family",
            ),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            local,
            native_publication_receipt=_artifact(
                observations.EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION,
                "local-native-receipt-forbidden",
            ),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(external, native_publication_receipt=None)


def test_failure_recovery_retained_fd_chain_is_exact_and_non_aliasable() -> None:
    failure = _failure_case(_campaign(), 0, completed_count=6).terminal_failure
    assert failure.post_container_remove_cgroup_sample is not None
    assert failure.cgroup_counter_fds_closed_receipt is not None
    assert failure.outer_cgroup_absence_observation is not None
    assert failure.cgroup_counter_fds_closed_post_sample_file_sha256 == (
        failure.post_container_remove_cgroup_sample.file_sha256
    )
    assert failure.outer_cgroup_absence_fd_close_body_sha256 == (
        failure.cgroup_counter_fds_closed_receipt.body_sha256
    )
    assert failure.outer_cgroup_absence_cgroup_identity_sha256 == (
        failure.cgroup_counter_fds_closed_cgroup_identity_sha256
    )

    for field_name in (
        "cgroup_counter_fds_closed_post_sample_file_sha256",
        "cgroup_counter_fds_closed_retained_fd_set_sha256",
        "cgroup_counter_fds_closed_container_identity_sha256",
        "outer_cgroup_absence_fd_close_body_sha256",
        "outer_cgroup_absence_cgroup_identity_sha256",
    ):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                failure,
                **{field_name: _sha("wrong-retained-fd-" + field_name)},
            )


def test_integer_and_identifier_fields_reject_bool_and_numeric_aliases() -> None:
    campaign = _campaign()
    spine = _case_spine(campaign, 0)
    success = _success_case(campaign, 0)
    failure = _failure_case(campaign, 0).terminal_failure
    for value in (False, True):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(spine, attempt_ordinal=value)
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(success.publication, retry_count=value)
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(failure, case_ordinal=value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(spine, candidate_id=0)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success.publication, qualification_case_id=1)


@pytest.mark.parametrize(
    "spoof_kind",
    ("string_subclass", "equality_proxy"),
)
def test_exact_text_vocabularies_reject_subclasses_and_equality_proxies(
    spoof_kind: str,
) -> None:
    campaign = _campaign()
    success = _success_case(campaign, 0)
    failure = _failure_case(campaign, 0, completed_count=16).terminal_failure
    projection = failure.failure_publication_projection
    assert projection is not None

    mutations = (
        (campaign, "qualification_plan_schema_version"),
        (success.probes[0], "probe_kind"),
        (success.publication, "schema_version"),
        (projection, "schema_version"),
        (success.resource_merger.fields[0], "field_name"),
        (success.resource_merger.fields[0], "value_semantics"),
        (success.resource_merger, "schema_version"),
        (success.host_success, "execution_state"),
        (failure, "failure_effect_state"),
        (failure.recovery_nodes[0], "state"),
    )
    for candidate, field_name in mutations:
        current = getattr(candidate, field_name)
        spoof: object = (
            _StringSubclass(current) if spoof_kind == "string_subclass" else _EqualityProxy()
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(candidate, **{field_name: spoof})

    completed_spoof: object = (
        _StringSubclass(failure.completed_phases[0])
        if spoof_kind == "string_subclass"
        else _EqualityProxy()
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            completed_phases=(completed_spoof, *failure.completed_phases[1:]),
        )
    dependency_node = failure.recovery_nodes[1]
    dependency_spoof: object = (
        _StringSubclass(dependency_node.dependencies[0])
        if spoof_kind == "string_subclass"
        else _EqualityProxy()
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            dependency_node,
            dependencies=(dependency_spoof,),
        )


@pytest.mark.parametrize("completed_count", range(len(_EXPECTED_OPERATIONAL_PHASES)))
@pytest.mark.parametrize(
    "failure_effect_state",
    ("failed_before_commit", "commit_uncertain"),
)
def test_every_operational_failure_transition_has_exact_prefix_and_effect(
    completed_count: int,
    failure_effect_state: Literal["failed_before_commit", "commit_uncertain"],
) -> None:
    failure = _failure_case(
        _campaign(),
        0,
        completed_count=completed_count,
        failure_effect_state=failure_effect_state,
    ).terminal_failure
    phase = _EXPECTED_OPERATIONAL_PHASES[completed_count]
    completed = set(failure.completed_phases)

    assert observations.HOST_OPERATIONAL_PHASES == _EXPECTED_OPERATIONAL_PHASES
    assert failure.completed_phases == _EXPECTED_OPERATIONAL_PHASES[:completed_count]
    assert failure.failure_phase == phase
    assert failure.failure_effect_state == failure_effect_state
    assert phase not in completed
    assert failure.ticket_quarantined is True
    assert failure.reconciliation_only is True
    assert failure.case_consumed is True
    assert failure.same_case_retry_permitted is False
    assert failure.clean_rejection_recorded is False
    assert failure.failure_count_state == "exact"
    assert failure.failure_count == 1
    assert failure.receipt_operational_failure_count == 1

    phase_fields = (
        ("intent_committed", "intent_state", "intent"),
        (
            "initial_cgroup_sample_committed",
            "initial_cgroup_sample_state",
            "initial_cgroup_sample",
        ),
        (
            "algorithmic_resource_receipt_committed",
            "algorithmic_resource_receipt_state",
            "algorithmic_resource_receipt",
        ),
        (
            "publication_commitment_wrapper_committed",
            "publication_commitment_wrapper_state",
            "publication_commitment_wrapper",
        ),
        (
            "publication_reload_validated",
            "publication_reload_state",
            "publication_reload_validation",
        ),
        ("storage_write_seal_committed", "storage_write_seal_state", "storage_write_seal"),
        (
            "storage_boundary_receipt_committed",
            "storage_boundary_receipt_state",
            "storage_boundary_receipt",
        ),
    )
    for committed_phase, state_name, artifact_name in phase_fields:
        expected_state = (
            "committed"
            if committed_phase in completed
            else failure_effect_state
            if phase == committed_phase
            else "not_started"
        )
        artifact = getattr(failure, artifact_name)
        assert getattr(failure, state_name) == expected_state
        assert (artifact is not None) is (expected_state == "committed")

    native_expected = (
        "committed"
        if "native_publication_committed" in completed
        else failure_effect_state
        if phase == "native_publication_committed"
        else "not_started"
    )
    assert failure.native_publication_state == native_expected
    assert (failure.publication_reconciliation_reference is not None) is (
        native_expected == "committed"
    )
    assert (failure.ready is not None) is ("driver_ready" in completed)
    assert (failure.observer_anchor is not None) is ("observer_anchored" in completed)
    assert (failure.go_commitment is not None) is ("go_committed" in completed)
    assert (failure.driver_terminal is not None) is ("workload_exited" in completed)

    boundary_fields = (
        (
            "container_created",
            "container_create_state",
            "container_create_count_state",
            "container_create_count",
        ),
        (
            "container_started",
            "container_start_state",
            "container_start_count_state",
            "container_start_count",
        ),
        (
            "workload_started",
            "workload_start_state",
            "workload_start_count_state",
            "workload_start_count",
        ),
        (
            "workload_exited",
            "workload_exit_state",
            "workload_exit_count_state",
            "workload_exit_count",
        ),
    )
    for boundary_phase, state_field, count_state_field, count_field in boundary_fields:
        if boundary_phase in completed:
            expected = ("committed", "exact", 1)
        elif phase == boundary_phase and failure_effect_state == "commit_uncertain":
            expected = ("commit_uncertain", "uncertain", None)
        else:
            expected = ("not_started", "exact", 0)
        assert (
            getattr(failure, state_field),
            getattr(failure, count_state_field),
            getattr(failure, count_field),
        ) == expected

    assert failure.attempt_count_state == failure.workload_start_count_state
    assert failure.attempt_count == failure.workload_start_count
    assert failure.terminal_metadata_state == "committed"
    assert failure.failure_receipt_state == "committed"
    assert failure.handoff_state == "committed"
    assert tuple(failure.to_dict()["uncertainty_dimensions"]) == (failure.uncertainty_dimensions)


def test_conditional_recovery_dag_preserves_independent_branches_and_exact_order() -> None:
    failure = _failure_case(
        _campaign(),
        0,
        recovery_only=True,
    ).terminal_failure
    states = {item.node_name: item.state for item in failure.recovery_nodes}

    assert observations.HOST_RECOVERY_NODE_NAMES == _EXPECTED_RECOVERY_NODE_NAMES
    assert observations.HOST_RECOVERY_NODE_STATES == (
        "not_applicable",
        "committed",
        "commit_uncertain",
        "failed_before_commit",
    )
    assert tuple(item.node_name for item in failure.recovery_nodes) == (
        _EXPECTED_RECOVERY_NODE_NAMES
    )
    assert {
        item.node_name: item.dependencies for item in failure.recovery_nodes
    } == _EXPECTED_RECOVERY_NODE_DEPENDENCIES
    assert states["cgroup_kill"] == "failed_before_commit"
    assert states["cgroup_empty"] == "committed"
    assert states["container_absence"] == "committed"
    assert states["outer_cgroup_absence"] == "committed"
    assert states["final_cgroup_proof"] == "failed_before_commit"
    assert failure.cleanup_unresolved_recovery_nodes == (
        "cgroup_kill",
        "final_cgroup_proof",
    )
    assert failure.cleanup_proven is False
    assert failure.cleanup_recovery_complete is True
    assert failure.cleanup_terminalization_permitted is True
    assert failure.cleanup_workload_resume_permitted is False
    assert failure.cleanup_same_case_retry_permitted is False
    assert failure.failure_phase is None
    assert failure.failure_effect_state is None
    assert failure.failure_count == 0
    assert failure.recovery_failure_count == 2
    assert failure.receipt_operational_failure_count == 0
    assert failure.receipt_recovery_failure_count == 2
    assert failure.classification == (
        "recovery_failure_after_complete_operational_frontier_nonretryable"
    )
    assert failure.terminal_metadata_state == "committed"
    assert failure.failure_receipt_state == "committed"
    assert failure.handoff_state == "committed"


@pytest.mark.parametrize("root_state", ("failed_before_commit", "commit_uncertain"))
def test_recovery_root_outcomes_do_not_suppress_unrelated_container_branch(
    root_state: Literal["failed_before_commit", "commit_uncertain"],
) -> None:
    states = {
        "precleanup_cgroup_sample": root_state,
        "cgroup_kill": "failed_before_commit",
        "cgroup_empty": "failed_before_commit",
        "container_absence": "committed",
        "post_container_remove_cgroup_sample": "committed",
        "cgroup_counter_fds_closed": "committed",
        "outer_cgroup_absence": "committed",
        "final_cgroup_proof": "failed_before_commit",
    }
    failure = _failure_case(
        _campaign(),
        0,
        completed_count=6,
        recovery_states=states,
    ).terminal_failure
    by_name = {item.node_name: item for item in failure.recovery_nodes}
    assert by_name["container_absence"].artifact is not None
    assert by_name["outer_cgroup_absence"].artifact is not None
    assert failure.cleanup_proven is False
    assert failure.terminal_metadata_state == "committed"
    assert failure.failure_receipt_state == "committed"
    assert failure.handoff_state == "committed"
    assert ("cleanup_state" in failure.uncertainty_dimensions) is (root_state == "commit_uncertain")


def test_container_recovery_branch_can_commit_before_initial_sample() -> None:
    states = {
        "precleanup_cgroup_sample": "failed_before_commit",
        "cgroup_kill": "failed_before_commit",
        "cgroup_empty": "failed_before_commit",
        "container_absence": "committed",
        "post_container_remove_cgroup_sample": "committed",
        "cgroup_counter_fds_closed": "committed",
        "outer_cgroup_absence": "committed",
        "final_cgroup_proof": "failed_before_commit",
    }
    failure = _failure_case(
        _campaign(),
        0,
        completed_count=4,
        recovery_states=states,
    ).terminal_failure

    assert failure.initial_cgroup_sample is None
    assert failure.container_absence_observation is not None
    assert failure.post_container_remove_cgroup_sample is not None
    assert failure.cgroup_counter_fds_closed_receipt is not None
    assert failure.outer_cgroup_absence_observation is not None
    assert failure.post_container_remove_retained_fd_set_sha256 is not None
    assert failure.post_container_remove_cgroup_identity_sha256 is not None
    assert failure.cleanup_proven is False
    assert failure.failure_receipt_state == "committed"
    assert failure.handoff_state == "committed"


def test_recovery_dag_rejects_committed_child_without_dependency_and_bad_applicability() -> None:
    dependency_violation = {name: "committed" for name in _EXPECTED_RECOVERY_NODE_NAMES}
    dependency_violation["precleanup_cgroup_sample"] = "failed_before_commit"
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        _failure_case(
            _campaign(),
            0,
            completed_count=6,
            recovery_states=dependency_violation,
        )

    inapplicable = {name: "not_applicable" for name in _EXPECTED_RECOVERY_NODE_NAMES}
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        _failure_case(
            _campaign(),
            0,
            completed_count=6,
            recovery_states=inapplicable,
        )

    no_cgroup = _failure_case(_campaign(), 0, completed_count=2).terminal_failure
    assert no_cgroup.cleanup_cgroup_may_exist is False
    assert all(item.state == "not_applicable" for item in no_cgroup.recovery_nodes)
    assert no_cgroup.cleanup_proven is True


def test_terminal_failure_counts_uncertainty_and_acyclic_receipt_semantics() -> None:
    campaign = _campaign()
    intent_uncertain = _failure_case(
        campaign,
        0,
        completed_count=2,
    ).terminal_failure
    intent_failed = _failure_case(
        campaign,
        0,
        completed_count=2,
        failure_effect_state="failed_before_commit",
    ).terminal_failure
    storage_uncertain = _failure_case(
        campaign,
        0,
        completed_count=17,
    ).terminal_failure
    recovery_uncertain_states = {
        "precleanup_cgroup_sample": "committed",
        "cgroup_kill": "commit_uncertain",
        "cgroup_empty": "committed",
        "container_absence": "committed",
        "post_container_remove_cgroup_sample": "committed",
        "cgroup_counter_fds_closed": "committed",
        "outer_cgroup_absence": "committed",
        "final_cgroup_proof": "commit_uncertain",
    }
    recovery_uncertain = _failure_case(
        campaign,
        0,
        recovery_only=True,
        recovery_states=recovery_uncertain_states,
    ).terminal_failure

    assert intent_uncertain.intent_state == "commit_uncertain"
    assert intent_uncertain.intent is None
    assert intent_uncertain.uncertainty_dimensions == ("operational_state",)
    assert intent_failed.intent_state == "failed_before_commit"
    assert intent_failed.uncertainty_dimensions == ()
    assert storage_uncertain.uncertainty_dimensions == (
        "operational_state",
        "storage_state",
    )
    assert recovery_uncertain.failure_count == 0
    assert recovery_uncertain.recovery_failure_count == 0
    assert recovery_uncertain.recovery_uncertainty_count == 2
    assert recovery_uncertain.uncertainty_dimensions == ("cleanup_state",)
    assert recovery_uncertain.cleanup_proven is False
    assert recovery_uncertain.failure_receipt_state == "committed"
    assert recovery_uncertain.handoff_state == "committed"
    assert "lifecycle" not in _EXPECTED_TERMINAL_BODY_KEYS

    for field_name in (
        "lifecycle_operational_frontier_file_sha256",
        "lifecycle_cleanup_reconciliation_body_sha256",
        "lifecycle_terminal_metadata_file_sha256",
        "receipt_lifecycle_body_sha256",
        "receipt_terminal_metadata_body_sha256",
    ):
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                recovery_uncertain,
                **{field_name: _sha("acyclic-link-" + field_name)},
            )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(recovery_uncertain, same_case_retry_permitted=True)


def test_failure_storage_seal_and_receipt_cannot_be_swapped() -> None:
    failure = _failure_case(_campaign(), 0, recovery_only=True).terminal_failure
    assert failure.storage_write_seal is not None
    assert failure.storage_boundary_receipt is not None

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            storage_write_seal=failure.storage_boundary_receipt,
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            failure,
            storage_boundary_receipt=failure.storage_write_seal,
        )


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "RawSeed",
        "resultBytes",
        "payload-digest",
        "rank_value",
        "nestedScoreField",
        "accepted.value",
    ),
)
def test_recursive_normalized_forbidden_key_policy_runs_before_reconstruction(
    forbidden_key: str,
) -> None:
    _, raw, pins = _encoded(frozenset())
    value = json.loads(raw)
    value["cases"][0]["case_spine"][forbidden_key] = _sha(forbidden_key)
    mutated = _canonical(value)
    mutated_pins = dataclasses.replace(
        pins,
        batch_file_sha256=hashlib.sha256(mutated).hexdigest(),
    )

    with pytest.raises(
        observations.ForagerMatchedV3QualificationObservationsV2Error,
        match="forbidden key",
    ):
        observations.parse_matched_v3_qualification_observation_candidate_batch_v2(
            mutated,
            pins=mutated_pins,
        )


def test_recursive_forbidden_key_policy_also_runs_during_canonical_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch(frozenset())
    original_to_dict = observations.MatchedV3QualificationObservationCandidateBatchV2.to_dict

    def contaminated_to_dict(
        self: observations.MatchedV3QualificationObservationCandidateBatchV2,
    ) -> dict[str, Any]:
        value = original_to_dict(self)
        value["safe_envelope"] = {"RawSeed": _sha("forbidden-raw-seed")}
        return value

    monkeypatch.setattr(
        observations.MatchedV3QualificationObservationCandidateBatchV2,
        "to_dict",
        contaminated_to_dict,
    )
    with pytest.raises(
        observations.ForagerMatchedV3QualificationObservationsV2Error,
        match="forbidden key",
    ):
        observations.canonical_matched_v3_qualification_observation_candidate_batch_v2_bytes(batch)


@pytest.mark.parametrize(
    ("shape", "message"),
    (
        ("alias", "alias or cycle"),
        ("cycle", "alias or cycle"),
        ("depth", "exceeds its bound"),
    ),
)
def test_public_serialization_rejects_alias_cycles_and_excess_depth_before_walk(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    message: str,
) -> None:
    batch = _batch(frozenset())
    original_to_dict = observations.MatchedV3QualificationObservationCandidateBatchV2.to_dict

    def hostile_to_dict(
        self: observations.MatchedV3QualificationObservationCandidateBatchV2,
    ) -> dict[str, Any]:
        value = original_to_dict(self)
        if shape == "alias":
            shared: list[object] = []
            value["safe_alias_a"] = shared
            value["safe_alias_b"] = shared
        elif shape == "cycle":
            cycle: dict[str, object] = {}
            cycle["safe_cycle"] = cycle
            value["safe_cycle"] = cycle
        else:
            root: dict[str, object] = {}
            cursor = root
            for _ in range(66):
                child: dict[str, object] = {}
                cursor["safe_child"] = child
                cursor = child
            value["safe_depth"] = root
        return value

    monkeypatch.setattr(
        observations.MatchedV3QualificationObservationCandidateBatchV2,
        "to_dict",
        hostile_to_dict,
    )
    with pytest.raises(
        observations.ForagerMatchedV3QualificationObservationsV2Error,
        match=message,
    ):
        observations.canonical_matched_v3_qualification_observation_candidate_batch_v2_bytes(batch)


def test_strict_json_rejects_duplicates_floats_noncanonical_bytes_and_unknown_keys() -> None:
    _, raw, pins = _encoded(frozenset())
    duplicate = raw.replace(
        b'{"batch_body_sha256":',
        b'{"batch_body_sha256":"' + _sha("duplicate").encode("ascii") + b'","batch_body_sha256":',
        1,
    )
    noncanonical = raw.replace(b"{", b"{ ", 1)
    value = json.loads(raw)
    value["safe_extra"] = 1.5
    floating = _canonical(value)
    value = json.loads(raw)
    value["safe_extra"] = 1
    unknown = _canonical(value)

    for invalid in (duplicate, noncanonical, floating, unknown):
        invalid_pins = dataclasses.replace(
            pins,
            batch_file_sha256=hashlib.sha256(invalid).hexdigest(),
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            observations.parse_matched_v3_qualification_observation_candidate_batch_v2(
                invalid,
                pins=invalid_pins,
            )


def test_publication_paths_and_empty_file_coherence_fail_closed() -> None:
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.PublicationFileCandidateV2(
            role="manifest",
            name="nested/../publication.json",
            size_bytes=1,
            sha256=_sha("unsafe-path"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.PublicationFileCandidateV2(
            role="manifest",
            name="publication.json",
            size_bytes=0,
            sha256=_sha("false-nonempty-digest"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.PublicationFileCandidateV2(
            role="manifest",
            name="publication.json",
            size_bytes=1,
            sha256=observations.EMPTY_FILE_SHA256,
        )

    publication = _success_case(_campaign(), 0).publication
    empty_manifest = dataclasses.replace(
        publication.files[0],
        size_bytes=0,
        sha256=observations.EMPTY_FILE_SHA256,
    )
    files = (empty_manifest, *publication.files[1:])
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            publication,
            publication_address_sha256=observations.EMPTY_FILE_SHA256,
            publication_manifest_file_sha256=observations.EMPTY_FILE_SHA256,
            files=files,
            file_inventory_sha256=_inventory_digest(
                "files",
                [item.to_dict() for item in files],
            ),
            total_size_bytes=sum(item.size_bytes for item in files),
        )


def test_production_slots_reject_nonexecuting_or_family_substitutions() -> None:
    campaign = _campaign()
    current_host = observations.ProducerIdentityV2(
        descriptor_schema_version=(observations.PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION),
        descriptor_sha256=(observations.NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256),
        source_sha256=_sha("different-host-source"),
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(campaign, host_executor=current_host)

    source_only_verifier = observations.ProducerIdentityV2(
        descriptor_schema_version=(
            observations.QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
        ),
        descriptor_sha256=(observations.SOURCE_ONLY_QUICKNET_VERIFIER_DESCRIPTOR_SHA256),
        source_sha256=_sha("different-quicknet-source"),
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(campaign, quicknet_verifier=source_only_verifier)

    materialization_descriptor_verifier = observations.ProducerIdentityV2(
        descriptor_schema_version=(
            observations.QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
        ),
        descriptor_sha256=(observations.SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256),
        source_sha256=_sha("different-materialization-quicknet-source"),
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            quicknet_verifier=materialization_descriptor_verifier,
        )
    materialization_source_verifier = observations.ProducerIdentityV2(
        descriptor_schema_version=(
            observations.QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
        ),
        descriptor_sha256=_sha("different-materialization-quicknet-descriptor"),
        source_sha256=observations.SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            quicknet_verifier=materialization_source_verifier,
        )

    for descriptor_sha256, source_sha256 in (
        (
            observations.SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256,
            _sha("different-build-only-quicknet-source"),
        ),
        (
            _sha("different-build-only-quicknet-descriptor"),
            observations.SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256,
        ),
    ):
        source_only_build_verifier = observations.ProducerIdentityV2(
            descriptor_schema_version=(
                observations.QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=descriptor_sha256,
            source_sha256=source_sha256,
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                campaign,
                quicknet_verifier=source_only_build_verifier,
            )

    endpoint_as_merger = observations.ProducerIdentityV2(
        descriptor_schema_version=observations.FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_sha256=(observations.PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256),
        source_sha256=_sha("different-merger-source"),
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(campaign, full_resource_merger=endpoint_as_merger)

    success = _success_case(campaign, 0)
    for index, incompatible_digest in enumerate(
        (
            "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d",
            "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3",
            "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733",
            "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8",
        )
    ):
        for digest_slot in ("descriptor_sha256", "source_sha256"):
            producer_values = {
                "descriptor_schema_version": (
                    observations.FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION
                ),
                "descriptor_sha256": _sha(f"different-merger-descriptor-{index}-{digest_slot}"),
                "source_sha256": _sha(f"different-merger-source-{index}-{digest_slot}"),
            }
            producer_values[digest_slot] = incompatible_digest
            algorithmic_validator_as_merger = observations.ProducerIdentityV2(**producer_values)
            with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
                dataclasses.replace(campaign, full_resource_merger=algorithmic_validator_as_merger)
            with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
                dataclasses.replace(
                    success.resource_merger,
                    merger=algorithmic_validator_as_merger,
                )

    adapter = _success_case(campaign, 23).publication
    local_producer = _producer(
        observations.LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "local-producer-in-adapter-slot",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(adapter, publisher=local_producer)

    for index, incompatible_digest in enumerate(
        observations.INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S
    ):
        unqualified_publisher = observations.ProducerIdentityV2(
            descriptor_schema_version=(
                observations.STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=incompatible_digest,
            source_sha256=_sha(f"different-adapter-publisher-source-{index}"),
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(adapter, publisher=unqualified_publisher)
    for index, incompatible_digest in enumerate(observations.INCOMPATIBLE_ADAPTER_SOURCE_SHA256S):
        unqualified_publisher = observations.ProducerIdentityV2(
            descriptor_schema_version=(
                observations.STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=_sha(f"different-adapter-publisher-descriptor-{index}"),
            source_sha256=incompatible_digest,
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(adapter, publisher=unqualified_publisher)

    for index, incompatible_digest in enumerate(
        observations.INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S
    ):
        unqualified_atomic_producer = observations.ProducerIdentityV2(
            descriptor_schema_version=(
                observations.STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=incompatible_digest,
            source_sha256=_sha(f"different-adapter-atomic-source-{index}"),
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                adapter,
                native_atomic_producer=unqualified_atomic_producer,
            )
    for index, incompatible_digest in enumerate(observations.INCOMPATIBLE_ADAPTER_SOURCE_SHA256S):
        unqualified_atomic_producer = observations.ProducerIdentityV2(
            descriptor_schema_version=(
                observations.STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            descriptor_sha256=_sha(f"different-adapter-atomic-descriptor-{index}"),
            source_sha256=incompatible_digest,
        )
        with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
            dataclasses.replace(
                adapter,
                native_atomic_producer=unqualified_atomic_producer,
            )

    old_host_receipt = _artifact(
        "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v1",
        "incompatible-host-v1-receipt",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success.host_success, execution_receipt=old_host_receipt)


@pytest.mark.parametrize(
    "incompatible_digest",
    (
        *observations.INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S,
        *observations.INCOMPATIBLE_ADAPTER_SOURCE_SHA256S,
    ),
)
@pytest.mark.parametrize("digest_slot", ("descriptor_sha256", "source_sha256"))
@pytest.mark.parametrize("producer_slot", ("publisher", "native_atomic_producer"))
def test_adapter_denylist_rejects_cross_kind_digest_substitution(
    incompatible_digest: str,
    digest_slot: str,
    producer_slot: str,
) -> None:
    publication = _success_case(_campaign(), 23).publication
    original = getattr(publication, producer_slot)
    poisoned = dataclasses.replace(original, **{digest_slot: incompatible_digest})
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(publication, **{producer_slot: poisoned})


def test_campaign_and_replay_pins_reject_the_zero_image_identity() -> None:
    zero_image_id = "sha256:" + "0" * 64
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(_campaign(), image_id=zero_image_id)

    _, raw, pins = _encoded(frozenset())
    assert raw
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(pins, image_id=zero_image_id)


def test_campaign_supply_chain_binds_three_sources_staging_build_and_image() -> None:
    campaign = _campaign()

    assert campaign.joint_source_closure_adapter_file_sha256 == (
        campaign.adapter_source_candidate.file_sha256
    )
    assert campaign.sealed_staging_joint_source_closure_body_sha256 == (
        campaign.joint_source_closure_candidate.body_sha256
    )
    assert campaign.fresh_build_sealed_staging_file_sha256 == (
        campaign.sealed_staging_candidate.file_sha256
    )
    assert campaign.fresh_build_image_id == campaign.image_id

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            joint_source_closure_adapter_body_sha256=_sha("wrong-adapter-source"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            sealed_staging_joint_source_closure_file_sha256=_sha("wrong-closure"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            fresh_build_sealed_staging_body_sha256=_sha("wrong-staging"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            campaign,
            fresh_build_image_id="sha256:" + _sha("wrong-image"),
        )


def test_endpoint_v1_is_incompatible_but_future_v2_corroboration_is_optional() -> None:
    success = _success_case(_campaign(), 0)
    merger = success.resource_merger
    old_request = _artifact(
        "alberta.forager_matched_v3.qualification_resource_observation_request.v1",
        "old-endpoint-request",
    )
    old_receipt = _artifact(
        "alberta.forager_matched_v3.qualification_resource_observation_receipt.v1",
        "old-endpoint-receipt",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            merger,
            endpoint_corroboration_mode="present_redundant_non_authoritative",
            endpoint_observer_request=old_request,
            endpoint_observer_receipt=old_receipt,
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            merger,
            endpoint_corroboration_mode="present_redundant_non_authoritative",
            endpoint_observer_request=_artifact(
                observations.ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION,
                "future-endpoint-request",
            ),
            endpoint_observer_receipt=None,
        )

    request = _artifact(
        observations.ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION,
        "future-endpoint-request",
    )
    receipt = _artifact(
        observations.ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION,
        "future-endpoint-receipt",
    )
    corroborated_merger = dataclasses.replace(
        merger,
        endpoint_corroboration_mode="present_redundant_non_authoritative",
        endpoint_observer_request=request,
        endpoint_observer_receipt=receipt,
    )
    corroborated_host = dataclasses.replace(
        success.host_success,
        endpoint_observer_request_file_sha256=request.file_sha256,
        endpoint_observer_request_body_sha256=request.body_sha256,
        endpoint_observer_receipt_file_sha256=receipt.file_sha256,
        endpoint_observer_receipt_body_sha256=receipt.body_sha256,
    )
    corroborated = dataclasses.replace(
        success,
        host_success=corroborated_host,
        resource_merger=corroborated_merger,
    )
    assert corroborated.resource_merger.endpoint_observer_receipt == receipt


def test_case_host_terminal_resource_and_publication_crosslinks_fail_closed() -> None:
    campaign = _campaign()
    success = _success_case(campaign, 0)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success.host_success, candidate_id="causal_e025_q075")

    wrong_terminal = _artifact(
        observations.HOST_TERMINAL_METADATA_SCHEMA_VERSION,
        "wrong-terminal",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        host_terminal_metadata=wrong_terminal,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_handoff = _artifact(
        observations.HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
        "wrong-host-observation-handoff",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        host_observation_handoff=wrong_handoff,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_algorithmic_receipt = _artifact(
        observations.LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
        "wrong-terminal-algorithmic-receipt",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.host_success,
            terminal_algorithmic_resource_receipt_file_sha256=(
                wrong_algorithmic_receipt.file_sha256
            ),
            terminal_algorithmic_resource_receipt_body_sha256=(
                wrong_algorithmic_receipt.body_sha256
            ),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.host_success,
            terminal_algorithmic_resource_receipt_file_sha256="0" * 64,
        )

    wrong_algorithmic_intent = _artifact(
        observations.ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
        "wrong-algorithmic-intent",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        algorithmic_measurement_intent=wrong_algorithmic_intent,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_provisioning = _artifact(
        observations.HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
        "wrong-host-provisioning",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        host_provisioning_receipt=wrong_provisioning,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_metadata = _artifact(
        observations.LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION,
        "wrong-local-family-metadata",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.publication,
            publisher_metadata=wrong_metadata,
            wrapper_publisher_metadata_file_sha256=wrong_metadata.file_sha256,
            wrapper_publisher_metadata_body_sha256=wrong_metadata.body_sha256,
        )

    wrong_storage_intent = _artifact(
        observations.QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        "wrong-storage-intent",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        storage_boundary_intent=wrong_storage_intent,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_storage_seal = _artifact(
        observations.QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
        "wrong-storage-seal",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        storage_write_seal=wrong_storage_seal,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    wrong_reload_validation = _artifact(
        observations.QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "wrong-publication-reload-validation",
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.host_success,
            terminal_publication_reload_validation=wrong_reload_validation,
            storage_write_seal_reload_validation_file_sha256=(wrong_reload_validation.file_sha256),
            storage_write_seal_reload_validation_body_sha256=(wrong_reload_validation.body_sha256),
        )

    wrong_runner = _artifact(
        observations.LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION,
        "wrong-runner-execution-receipt",
    )
    rewired_merger = dataclasses.replace(
        success.resource_merger,
        runner_execution_receipt=wrong_runner,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(success, resource_merger=rewired_merger)

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.publication,
            wrapper_publisher_source_sha256=_sha("wrong-wrapper-publisher-source"),
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        dataclasses.replace(
            success.publication,
            wrapper_expected_reload_observation_sha256=_sha("wrong-wrapper-expected-reload"),
        )


def test_batch_rejects_duplicate_case_specific_host_and_probe_artifacts() -> None:
    batch = _batch({0, 1})
    first = batch.cases[0]
    second = batch.cases[1]
    assert isinstance(first, observations.QualificationCaseSuccessCandidateV2)
    assert isinstance(second, observations.QualificationCaseSuccessCandidateV2)

    duplicate_host = dataclasses.replace(
        second.host_success,
        request=first.host_success.request,
        authorization_request_file_sha256=first.host_success.request.file_sha256,
        authorization_request_body_sha256=first.host_success.request.body_sha256,
    )
    duplicate_case = dataclasses.replace(second, host_success=duplicate_host)
    cases = (first, duplicate_case, *batch.cases[2:])
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        _batch_with_cases(batch, cases)

    duplicate_probe = dataclasses.replace(
        second.probes[0],
        file_sha256=first.probes[0].file_sha256,
        body_sha256=first.probes[0].body_sha256,
    )
    probes = (duplicate_probe, *second.probes[1:])
    duplicate_probe_case = dataclasses.replace(second, probes=probes)
    cases = (first, duplicate_probe_case, *batch.cases[2:])
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        _batch_with_cases(batch, cases)


def test_seed_derivation_cross_role_equality_remains_noninferential() -> None:
    campaign = _campaign()
    first = _case_spine(campaign, 0)
    second = _case_spine(campaign, 1)

    cross_role = dataclasses.replace(
        first,
        environment_derivation_sha256=first.seed_derivation_record_sha256,
    )
    assert cross_role.environment_derivation_sha256 == (cross_role.seed_derivation_record_sha256)

    same_role_duplicate = dataclasses.replace(
        first,
        environment_derivation_sha256=second.environment_derivation_sha256,
    )
    assert same_role_duplicate.environment_derivation_sha256 == (
        second.environment_derivation_sha256
    )
    descriptor = observations.matched_v3_qualification_observation_registry_v2_descriptor()
    assert (
        descriptor["seed_identity_contract"]["case_derivation_identities_unique_per_role"] is True
    )
    assert (
        descriptor["seed_identity_contract"]["cross_role_derivation_digest_inequality_required"]
        is False
    )


def test_seed_commitment_equality_does_not_infer_numeric_seed_inequality() -> None:
    spine = _case_spine(_campaign(), 0)
    same_commitment = dataclasses.replace(
        spine,
        agent_seed_commitment_sha256=spine.environment_seed_commitment_sha256,
    )

    assert same_commitment.agent_seed_commitment_sha256 == (
        same_commitment.environment_seed_commitment_sha256
    )
    assert "environment_seed" not in same_commitment.to_dict()
    assert "agent_seed" not in same_commitment.to_dict()


@pytest.mark.parametrize(
    "field_name",
    (
        "qualification_plan_body_sha256",
        "observation_registry_source_sha256",
        "seed_pulse_record_file_sha256",
        "quicknet_verifier_source_sha256",
        "seed_chronology_receipt_body_sha256",
        "local_source_candidate_file_sha256",
        "external_source_candidate_body_sha256",
        "adapter_source_candidate_file_sha256",
        "joint_source_closure_candidate_body_sha256",
        "sealed_staging_candidate_file_sha256",
        "fresh_build_candidate_file_sha256",
        "runtime_candidate_body_sha256",
        "host_executor_descriptor_sha256",
        "full_resource_merger_source_sha256",
        "algorithmic_resource_contract_descriptor_sha256",
        "algorithmic_resource_contract_source_sha256",
        "storage_boundary_contract_descriptor_sha256",
        "storage_boundary_contract_source_sha256",
        "normalized_publication_contract_descriptor_sha256",
        "normalized_publication_contract_source_sha256",
        "all_case_sequence_intent_file_sha256",
        "all_case_sequence_receipt_body_sha256",
        "all_case_sequence_cases_inventory_sha256",
        "candidate_order_sha256",
        "image_id",
    ),
)
def test_independent_replay_pins_fail_closed_across_the_campaign_spine(
    field_name: str,
) -> None:
    _, raw, pins = _encoded(frozenset())
    wrong_value = (
        "sha256:" + _sha(f"wrong-{field_name}")
        if field_name == "image_id"
        else _sha(f"wrong-{field_name}")
    )
    corrupted = dataclasses.replace(pins, **{field_name: wrong_value})

    with pytest.raises(observations.ForagerMatchedV3QualificationObservationsV2Error):
        observations.parse_matched_v3_qualification_observation_candidate_batch_v2(
            raw,
            pins=corrupted,
        )


def test_batch_metadata_contains_commitments_only_and_no_per_case_candidate_byte_api() -> None:
    _, raw, _ = _encoded(frozenset({0}))
    value = json.loads(raw)
    keys: set[str] = set()
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            keys.update(current)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)

    assert "environment_seed" not in keys
    assert "agent_seed" not in keys
    assert "raw_seed" not in keys
    assert "seed_derivation_payload_sha256" not in keys
    assert "environment_seed_commitment_sha256" in keys
    assert "agent_seed_commitment_sha256" in keys
    public_byte_apis = {
        name
        for name in observations.__all__
        if name.startswith(("canonical_", "parse_", "replay_"))
    }
    assert all(
        "candidate_batch" in name
        or "descriptor" in name
        or "publication_reload_validation" in name
        or "host_cleanup_reconciliation" in name
        or "host_observation_handoff" in name
        or "host_operational_frontier" in name
        or "host_terminal_metadata" in name
        or "all_case_sequence" in name
        or "normalized_publication_commitment_wrapper" in name
        for name in public_byte_apis
    )


def test_module_is_in_memory_only_and_has_no_runtime_contract_import_cycle() -> None:
    source = inspect.getsource(observations)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")

    assert imported_modules == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "typing",
    }
    assert "from alberta_framework" not in source
    assert "pathlib" not in imported_modules
    assert "subprocess" not in imported_modules
    assert "socket" not in imported_modules


def test_registry_remains_nonauthorizing_even_for_an_all_success_batch() -> None:
    batch = _batch(set(range(28)))
    value = batch.to_dict()

    assert all(item.record_kind == "success" for item in batch.cases)
    assert all(flag is False for flag in value["claims"].values())
    assert all(flag is False for flag in value["readiness"].values())
    assert value["claims"]["decision_evaluated"] is False
    assert value["claims"]["qualification_granted"] is False
    assert value["claims"]["resource_matched"] is False
    assert value["readiness"]["decision_ready"] is False
    assert value["readiness"]["qualification_ready"] is False
