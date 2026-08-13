"""Batch-only structural observation candidates for matched-v3 qualification.

This additive module validates one canonical metadata-only batch containing all
28 qualification cases in the frozen order.  Every ordinal is represented by
exactly one success or terminal-failure candidate.  The module does not read any
referenced artifact, issue an observation, evaluate a case, execute a workload,
or grant authority.  A structurally valid all-success batch remains
nonauthorizing and unevaluated.

The plan-v3 schema, candidate order, family partitions, and resource-field order
are duplicated as stable literals.  The plan module is deliberately not
imported, and neither a plan descriptor/source digest nor this module's own
source digest is embedded.  A future separate issuer must bind and validate
those identities without creating a source-hash cycle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, cast

QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v2"
)
QUALIFICATION_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_candidate_batch.v2"
)
QUALIFICATION_CASE_SUCCESS_CANDIDATE_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_success_candidate.v2"
)
QUALIFICATION_CASE_TERMINAL_FAILURE_CANDIDATE_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_terminal_failure_candidate.v2"
)
QUALIFICATION_PUBLICATION_CANDIDATE_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_candidate.v2"
)
QUALIFICATION_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_merger_candidate.v2"
)
QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.qualification_plan.v3"
QUALIFICATION_OBSERVATION_REGISTRY_V2_STATUS: Final = (
    "implemented_structural_full_batch_candidate_validator_no_issuer_no_evaluator"
)
QUALIFICATION_OBSERVATION_REGISTRY_V2_CLASSIFICATION: Final = (
    "score_blind_metadata_only_candidate_batch_non_authorizing"
)

PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_issuance_receipt.v1"
)
CASE_TICKET_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_ticket_registry.v1"
)
CASE_EXECUTION_TICKET_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_execution_ticket.v1"
)
PUBLISHER_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publisher_registry.v1"
)
PUBLISHER_REGISTRY_ENTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publisher_registry_entry.v1"
)
QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_manifest.v2"
)
QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_registry.v2"
)
QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_drand_pulse_record.v1"
)
QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_trust_root_receipt.v2"
)
QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_receipt.v1"
)
QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
)
SEED_CHRONOLOGY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.seed_preacceptance_chronology_receipt.v1"
)
ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_all_case_sequence_receipt.v1"
)
ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_all_case_sequence_intent.v1"
)
ALL_CASE_SEQUENCE_INTENT_V1_BODY_KEYS: Final = (
    "schema_version",
    "candidate_order_sha256",
    "candidate_order",
    "case_count",
    "claims_completion",
)
ALL_CASE_SEQUENCE_INTENT_V1_BODY_SHA256_FIELD: Final = "all_case_sequence_intent_body_sha256"
ALL_CASE_SEQUENCE_RECEIPT_V1_BODY_KEYS: Final = (
    "schema_version",
    "all_case_sequence_intent_file_sha256",
    "all_case_sequence_intent_body_sha256",
    "campaign_spine_body_sha256",
    "ordered_terminal_handoff_inventory_sha256",
    "case_count",
    "terminal_coverage_complete",
)
ALL_CASE_SEQUENCE_RECEIPT_V1_BODY_SHA256_FIELD: Final = "all_case_sequence_receipt_body_sha256"
LOCAL_SOURCE_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_observation_candidate.v2"
)
EXTERNAL_SOURCE_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_source_observation_candidate.v2"
)
ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_source_observation_candidate.v1"
)
JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_joint_source_closure_candidate.v1"
)
SEALED_STAGING_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_sealed_staging_candidate.v1"
)
FRESH_BUILD_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.fresh_cpu_oci_build_candidate.v2"
)
RUNTIME_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_observation_candidate.v2"
)
RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_qualification_receipt.v1"
)
HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_receipt.v2"
)

HOST_CASE_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_request.v2"
)
HOST_CASE_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_intent.v2"
)
HOST_READY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_ready.v2"
)
HOST_OBSERVER_ANCHOR_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_observer_anchor.v2"
HOST_GO_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v2"
HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_operational_frontier.v2"
)
HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_initial_cgroup_sample.v2"
)
IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_terminal.v2"
)
HOST_CGROUP_PROOF_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_v2_boundary_proof.v1"
)
HOST_TERMINAL_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_terminal_metadata.v2"
)
HOST_LIFECYCLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_lifecycle_record.v2"
)
HOST_SUCCESS_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v2"
)
HOST_FAILURE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_failure_receipt.v2"
)
HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_observation_handoff.v2"
)
HOST_OBSERVATION_HANDOFF_V2_BODY_KEYS: Final = (
    "schema_version",
    "case_spine_sha256",
    "case_ordinal",
    "candidate_id",
    "qualification_case_id",
    "record_kind",
    "terminal_receipt_file_sha256",
    "terminal_receipt_body_sha256",
    "terminal_metadata_file_sha256",
    "terminal_metadata_body_sha256",
)
HOST_OBSERVATION_HANDOFF_V2_BODY_SHA256_FIELD: Final = "handoff_body_sha256"
INCOMPATIBLE_HOST_HANDOFF_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_observation_handoff.v1"
)
INCOMPATIBLE_HOST_V1_SCHEMA_VERSIONS: Final = (
    "alberta.forager_matched_v3.host_provisioning_receipt.v1",
    "alberta.forager_matched_v3.host_qualification_case_request.v1",
    "alberta.forager_matched_v3.host_qualification_case_intent.v1",
    "alberta.forager_matched_v3.in_container_qualification_driver_ready.v1",
    "alberta.forager_matched_v3.host_qualification_go_commitment.v1",
    "alberta.forager_matched_v3.in_container_qualification_driver_terminal.v1",
    "alberta.forager_matched_v3.host_qualification_lifecycle_record.v1",
    "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v1",
    "alberta.forager_matched_v3.host_qualification_case_failure.v1",
)
ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_request.v2"
)
ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_receipt.v2"
)
FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_resource_merger_receipt.v1"
)
PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_executor_descriptor.v2"
)
FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_resource_merger_descriptor.v1"
)
LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_algorithmic_resource_receipt.v1"
)
EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_algorithmic_resource_receipt.v1"
)
ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_algorithmic_resource_receipt.v1"
)
ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_measurement_intent.v1"
)
ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
)
LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_runner_completion.v1"
)
EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_receipt.v1"
)
FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_result_receipt.v1"
)
PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_result_receipt.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_receipt.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_intent.v1"
)
QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1"
)
QUALIFICATION_PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reconciliation_reference.v1"
)
QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_failure_publication_projection.v2"
)
QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD: Final = (
    "failure_publication_projection_body_sha256"
)
QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)
QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_KEYS: Final = (
    "schema_version",
    "publication_commitment_wrapper_file_sha256",
    "publication_commitment_wrapper_body_sha256",
    "expected_reload_observation_sha256",
    "actual_reload_observation_sha256",
    "reload_performed",
    "reload_read_only",
)
QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_SHA256_FIELD: Final = (
    "reload_validation_body_sha256"
)
HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cleanup_reconciliation.v2"
)
HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_precleanup_cgroup_sample.v2"
)
HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_kill_receipt.v2"
)
HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_empty_observation.v2"
)
HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_container_absence_observation.v2"
)
HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_post_container_remove_cgroup_sample.v2"
)
HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_counter_fds_closed_receipt.v2"
)
HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_outer_cgroup_absence_observation.v2"
)
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_STATUS: Final = (
    "implemented_source_only_expected_reload_commitment_non_authorizing"
)
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_CLASSIFICATION: Final = (
    "score_blind_metadata_only_normalized_commitment_non_authorizing"
)
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_KEYS: Final = (
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
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_SHA256_FIELD: Final = "wrapper_body_sha256"
EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1"
)

LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_metadata.v1"
)
LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_metadata.v1"
)
EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_publication_metadata.v1"
)
STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_publication_descriptor.v1"
)
ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_atomic_publication_descriptor.v1"
)

MATCHED_V3_HORIZON: Final = 499_712
MAX_PUBLICATION_TOTAL_BYTES: Final = 1024 * 1024 * 1024
MAX_PUBLICATION_FILE_BYTES: Final = MAX_PUBLICATION_TOTAL_BYTES
EMPTY_FILE_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

MATCHED_V3_LOCAL_CANDIDATE_IDS: Final = (
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
)
MATCHED_V3_EXTERNAL_CANDIDATE_IDS: Final = (
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "random_policy",
    "search_nearest",
    "search_oracle",
)
MATCHED_V3_ADAPTER_CANDIDATE_IDS: Final = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
)
MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS: Final = (
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
)

RESOURCE_CEILING_FIELDS: Final = (
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
RESOURCE_PROVENANCE_KINDS: Final = (
    *("algorithmic_resource_receipt",) * 20,
    "host_cgroup_memory_high_water_upper_bound",
    "host_cgroup_fresh_lifetime_cpu_delta",
    "host_cgroup_fresh_lifetime_wall_delta",
    "host_storage_boundary_receipt",
    "host_storage_boundary_receipt",
    "host_cgroup_pids_peak_upper_bound",
    "host_execution_lifecycle",
    "host_execution_lifecycle",
)

LOCAL_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("local_bundle_manifest", "local-bundle-manifest.json"),
    ("bootstrap_receipt", "bootstrap-receipt.json"),
    ("bootstrap_child_record", "bootstrap-child-record.json"),
    ("local_runner_receipt", "local-runner-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
EXTERNAL_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("outcome_manifest", "external-outcome-manifest.json"),
    ("execution_receipt", "external-execution-receipt.json"),
    ("conversion_receipt", "external-conversion-receipt.json"),
    ("upstream_reward_npz", "upstream-reward.npz"),
    ("upstream_results_database", "upstream-results.db"),
    ("upstream_video_slot", "upstream-video-slot.bin"),
    ("canonical_reward_npz", "reward-trace.npz"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
ADAPTER_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("adapter_bundle_manifest", "adapter-bundle-manifest.json"),
    ("runner_result_receipt", "runner-result-receipt.json"),
    ("reward_trace", "reward-trace.npz"),
    ("score_receipt", "score-receipt.json"),
)

PROBE_KINDS: Final = (
    "source_candidate",
    "runtime_candidate",
    "seed_candidate",
    "candidate_candidate",
    "fresh_replay_candidate",
)
PROBE_SCHEMA_BY_KIND: Final = {
    "source_candidate": "alberta.forager_matched_v3.source_observation_candidate.v2",
    "runtime_candidate": "alberta.forager_matched_v3.runtime_observation_candidate.v2",
    "seed_candidate": "alberta.forager_matched_v3.seed_observation_candidate.v2",
    "candidate_candidate": "alberta.forager_matched_v3.candidate_observation_candidate.v2",
    "fresh_replay_candidate": ("alberta.forager_matched_v3.fresh_replay_observation_candidate.v2"),
}

HOST_OPERATIONAL_PHASES: Final = (
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
HOST_LIFECYCLE_PHASES: Final = (
    *HOST_OPERATIONAL_PHASES,
    "cleanup_reconciliation_committed",
    "terminal_metadata_committed",
    "lifecycle_committed",
    "receipt_committed",
    "handoff_committed",
)
HOST_PHASE_STATES: Final = (
    "not_started",
    "failed_before_commit",
    "commit_uncertain",
    "committed",
)
HOST_OPERATIONAL_FAILURE_EFFECT_STATES: Final = (
    "failed_before_commit",
    "commit_uncertain",
)
HOST_RECOVERY_NODE_NAMES: Final = (
    "precleanup_cgroup_sample",
    "cgroup_kill",
    "cgroup_empty",
    "container_absence",
    "post_container_remove_cgroup_sample",
    "cgroup_counter_fds_closed",
    "outer_cgroup_absence",
    "final_cgroup_proof",
)
HOST_RECOVERY_NODE_STATES: Final = (
    "not_applicable",
    "committed",
    "commit_uncertain",
    "failed_before_commit",
)
HOST_RECOVERY_NODE_DEPENDENCIES: Final[Mapping[str, tuple[str, ...]]] = {
    "precleanup_cgroup_sample": (),
    "cgroup_kill": ("precleanup_cgroup_sample",),
    "cgroup_empty": ("precleanup_cgroup_sample",),
    "container_absence": (),
    "post_container_remove_cgroup_sample": ("container_absence",),
    "cgroup_counter_fds_closed": ("post_container_remove_cgroup_sample",),
    "outer_cgroup_absence": ("cgroup_counter_fds_closed",),
    "final_cgroup_proof": (
        "precleanup_cgroup_sample",
        "cgroup_kill",
        "cgroup_empty",
        "container_absence",
        "post_container_remove_cgroup_sample",
        "cgroup_counter_fds_closed",
        "outer_cgroup_absence",
    ),
}
HOST_RECOVERY_NODE_SCHEMAS: Final[Mapping[str, str]] = {
    "precleanup_cgroup_sample": HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
    "cgroup_kill": HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
    "cgroup_empty": HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
    "container_absence": HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
    "post_container_remove_cgroup_sample": HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
    "cgroup_counter_fds_closed": HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
    "outer_cgroup_absence": HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION,
    "final_cgroup_proof": HOST_CGROUP_PROOF_SCHEMA_VERSION,
}
HOST_UNCERTAINTY_DIMENSIONS: Final = (
    "operational_state",
    "publication_state",
    "storage_state",
    "cleanup_state",
    "terminalization_state",
)

PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256: Final = (
    "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
)
PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256: Final = (
    "4d34951ccb4b265caa29794457cdd8a5dd837ecf4b73b7a44e4f849bf8c8106e"
)
INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
)
INCOMPATIBLE_ADAPTER_SOURCE_SHA256S: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
)
INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S: Final = frozenset(
    (*INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S, *INCOMPATIBLE_ADAPTER_SOURCE_SHA256S)
)
NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256: Final = (
    "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
)
NONEXECUTING_HOST_EXECUTOR_SOURCE_SHA256: Final = (
    "d8bbc666a49e252662807f256c7f212c9a7c8c3be279b928a6a93ed77532a2e1"
)
SOURCE_ONLY_QUICKNET_VERIFIER_DESCRIPTOR_SHA256: Final = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)
SOURCE_ONLY_QUICKNET_VERIFIER_SOURCE_SHA256: Final = (
    "3e13009c1843c3341e5a0eb8b2f84ea903b8e5315fbdef347549757710fd3623"
)
SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256: Final = (
    "61345825673afb16bc1942c4b8c84e763fb14530a68225caffa94d98e733a03d"
)
SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256: Final = (
    "1e08a04b8c3120978867999b5316d57ac5361771b018496535a5ab5a77a61023"
)
SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256: Final = (
    "513ed21d7411f65a3d605f38eedc6da5cf6d6764203ebcf38210439a4724cb87"
)
SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256: Final = (
    "e82eab0c471deda4a1abd63fe9634bc27884474ef782b587712c5f0eaadaf065"
)
HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256S: Final = (
    "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733",
)
HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256S: Final = (
    "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8",
)
FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256: Final = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)
FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256: Final = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)
ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S: Final = frozenset(
    (
        FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256,
        FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
        *HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256S,
        *HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256S,
    )
)
FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
)
FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256: Final = (
    "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
)
FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256: Final = (
    "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
)
FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256: Final = (
    "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"
)
HISTORICAL_IMAGE_IDS: Final = (
    "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269",
    "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768",
)

_MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 250_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PORTABLE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\Z")
_CAMEL_BOUNDARY_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_KEY_TOKEN_RE: Final = re.compile(r"[^A-Za-z0-9]+")
_SAFE_TYPE_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")

_FORBIDDEN_KEY_SUBSTRINGS: Final = ("score", "reward")
_FORBIDDEN_KEY_TOKENS: Final = frozenset(
    {
        "acceptance",
        "accepted",
        "content",
        "payload",
        "rank",
        "ranking",
        "raw",
    }
)
_FORBIDDEN_NORMALIZED_KEYS: Final = frozenset(
    {
        "database_bytes",
        "file_bytes",
        "npz_bytes",
        "result_bytes",
        "stderr_bytes",
        "stdout_bytes",
        "trace_bytes",
        "video_bytes",
    }
)


class ForagerMatchedV3QualificationObservationsV2Error(ValueError):
    """A v2 candidate batch or independent replay pin failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationObservationsV2Error(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label)


def _require_image_id(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact sha256 image ID")
    if value == "sha256:" + "0" * 64:
        _fail(f"{label} must not use the zero SHA-256 sentinel")
    if value in HISTORICAL_IMAGE_IDS:
        _fail(f"{label} is a permanently excluded historical image")
    return value


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact boolean")
    return value


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be bounded nonempty printable ASCII")
    return value


def _require_identifier(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _PORTABLE_ID_RE.fullmatch(text) is None:
        _fail(f"{label} must be one portable identifier")
    return text


def _require_exact_literal(value: object, expected: str, label: str) -> str:
    text = _require_text(value, label)
    if not hmac.compare_digest(text, expected):
        _fail(f"{label} differs from its exact literal")
    return text


def _require_one_of(value: object, allowed: tuple[str, ...], label: str) -> str:
    text = _require_text(value, label)
    if text not in allowed:
        _fail(f"{label} differs from its exact vocabulary")
    return text


def _require_exact_string_tuple(
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not str for item in value)
        or value != expected
    ):
        _fail(f"{label} differs from its exact string tuple")
    return cast(tuple[str, ...], value)


def _require_string_tuple_from_vocabulary(
    value: object,
    allowed: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        _fail(f"{label} must be one exact string tuple")
    exact = cast(tuple[str, ...], value)
    if len(set(exact)) != len(exact) or any(item not in allowed for item in exact):
        _fail(f"{label} differs from its exact vocabulary")
    return exact


def _require_relative_path(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _RELATIVE_PATH_RE.fullmatch(text) is None or any(
        component in {".", ".."} for component in text.split("/")
    ):
        _fail(f"{label} must be one safe relative path")
    return text


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"candidate-batch JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"candidate-batch JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("candidate-batch JSON integer exceeds its lexical bound")
    parsed = int(value, 10)
    if not -_MAX_INTEGER <= parsed <= _MAX_INTEGER:
        _fail("candidate-batch JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("candidate-batch JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("candidate-batch JSON structure exceeds its bound")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, "candidate-batch JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("candidate-batch JSON strings must be bounded printable ASCII")
            return
        if type(item) not in {dict, list}:
            _fail("candidate-batch JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("candidate-batch JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("candidate-batch JSON keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _normalized_key(value: str) -> tuple[str, tuple[str, ...]]:
    split = _CAMEL_BOUNDARY_RE.sub("_", value)
    normalized = _NON_KEY_TOKEN_RE.sub("_", split).strip("_").lower()
    return normalized, tuple(token for token in normalized.split("_") if token)


def _reject_forbidden_metadata_keys(value: object) -> None:
    _assert_plain_unaliased_json(value)
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if type(current) is dict:
            for key, child in cast(dict[str, object], current).items():
                normalized, tokens = _normalized_key(key)
                if (
                    any(fragment in normalized for fragment in _FORBIDDEN_KEY_SUBSTRINGS)
                    or any(token in _FORBIDDEN_KEY_TOKENS for token in tokens)
                    or normalized in _FORBIDDEN_NORMALIZED_KEYS
                ):
                    _fail(f"candidate-batch metadata contains forbidden key {key!r}")
                pending.append(child)
        elif type(current) is list:
            pending.extend(cast(list[object], current))


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return frozenset(left_map) == frozenset(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(a, b) for a, b in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationObservationsV2Error(
            "candidate-batch value is not canonical JSON"
        ) from exc
    if newline:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("candidate-batch canonical JSON exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("candidate-batch bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationObservationsV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationObservationsV2Error(
            "candidate-batch bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("candidate-batch JSON root must be one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    if not hmac.compare_digest(_canonical_json(exact), raw):
        _fail("candidate-batch bytes are not canonical")
    return exact


def _body_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value), newline=False)).hexdigest()


def qualification_publication_reload_validation_v1_body_projection(
    *,
    publication_commitment_wrapper_file_sha256: str,
    publication_commitment_wrapper_body_sha256: str,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
    reload_performed: bool,
    reload_read_only: bool,
) -> dict[str, Any]:
    """Return the exact seven-key reload-validation v1 BODY projection."""

    for digest, label in (
        (
            publication_commitment_wrapper_file_sha256,
            "reload-validation wrapper file",
        ),
        (
            publication_commitment_wrapper_body_sha256,
            "reload-validation wrapper body",
        ),
        (expected_reload_observation_sha256, "expected reload observation"),
        (actual_reload_observation_sha256, "actual reload observation"),
    ):
        _require_sha256(digest, label)
    if expected_reload_observation_sha256 != actual_reload_observation_sha256:
        _fail("reload-validation expected and actual observations must match")
    if (
        _require_bool(reload_performed, "reload performed fact") is not True
        or _require_bool(reload_read_only, "reload read-only fact") is not True
    ):
        _fail("reload-validation performed and read-only facts must be exact true")
    return {
        "schema_version": QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "publication_commitment_wrapper_file_sha256": (publication_commitment_wrapper_file_sha256),
        "publication_commitment_wrapper_body_sha256": (publication_commitment_wrapper_body_sha256),
        "expected_reload_observation_sha256": expected_reload_observation_sha256,
        "actual_reload_observation_sha256": actual_reload_observation_sha256,
        "reload_performed": reload_performed,
        "reload_read_only": reload_read_only,
    }


def canonical_qualification_publication_reload_validation_v1_body_bytes(
    *,
    publication_commitment_wrapper_file_sha256: str,
    publication_commitment_wrapper_body_sha256: str,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
    reload_performed: bool,
    reload_read_only: bool,
) -> bytes:
    """Encode the reload-validation v1 BODY as compact sorted ASCII JSON without LF."""

    body = qualification_publication_reload_validation_v1_body_projection(
        publication_commitment_wrapper_file_sha256=(publication_commitment_wrapper_file_sha256),
        publication_commitment_wrapper_body_sha256=(publication_commitment_wrapper_body_sha256),
        expected_reload_observation_sha256=expected_reload_observation_sha256,
        actual_reload_observation_sha256=actual_reload_observation_sha256,
        reload_performed=reload_performed,
        reload_read_only=reload_read_only,
    )
    return _canonical_json(body, newline=False)


def canonical_qualification_publication_reload_validation_v1_file_bytes(
    *,
    publication_commitment_wrapper_file_sha256: str,
    publication_commitment_wrapper_body_sha256: str,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
    reload_performed: bool,
    reload_read_only: bool,
) -> bytes:
    """Encode BODY plus its digest as compact sorted ASCII JSON with exactly one LF."""

    body = qualification_publication_reload_validation_v1_body_projection(
        publication_commitment_wrapper_file_sha256=(publication_commitment_wrapper_file_sha256),
        publication_commitment_wrapper_body_sha256=(publication_commitment_wrapper_body_sha256),
        expected_reload_observation_sha256=expected_reload_observation_sha256,
        actual_reload_observation_sha256=actual_reload_observation_sha256,
        reload_performed=reload_performed,
        reload_read_only=reload_read_only,
    )
    body_sha256 = hashlib.sha256(_canonical_json(body, newline=False)).hexdigest()
    return _canonical_json(
        {
            **body,
            QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_SHA256_FIELD: (body_sha256),
        }
    )


def host_observation_handoff_v2_body_projection(
    *,
    case_spine_sha256: str,
    case_ordinal: int,
    candidate_id: str,
    qualification_case_id: str,
    record_kind: Literal["success", "terminal_failure"],
    terminal_receipt_file_sha256: str,
    terminal_receipt_body_sha256: str,
    terminal_metadata_file_sha256: str,
    terminal_metadata_body_sha256: str,
) -> dict[str, Any]:
    """Return the exact canonical v2 handoff BODY projection."""

    _require_case_projection(
        case_ordinal,
        candidate_id,
        qualification_case_id,
        "handoff",
    )
    _require_sha256(case_spine_sha256, "handoff case spine")
    if type(record_kind) is not str or record_kind not in {"success", "terminal_failure"}:
        _fail("handoff record kind differs")
    for digest, label in (
        (terminal_receipt_file_sha256, "handoff terminal receipt file"),
        (terminal_receipt_body_sha256, "handoff terminal receipt body"),
        (terminal_metadata_file_sha256, "handoff terminal metadata file"),
        (terminal_metadata_body_sha256, "handoff terminal metadata body"),
    ):
        _require_sha256(digest, label)
    return {
        "schema_version": HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
        "case_spine_sha256": case_spine_sha256,
        "case_ordinal": case_ordinal,
        "candidate_id": candidate_id,
        "qualification_case_id": qualification_case_id,
        "record_kind": record_kind,
        "terminal_receipt_file_sha256": terminal_receipt_file_sha256,
        "terminal_receipt_body_sha256": terminal_receipt_body_sha256,
        "terminal_metadata_file_sha256": terminal_metadata_file_sha256,
        "terminal_metadata_body_sha256": terminal_metadata_body_sha256,
    }


def canonical_host_observation_handoff_v2_body_bytes(**facts: Any) -> bytes:
    """Encode one exact v2 handoff BODY without a trailing LF."""

    return _canonical_json(host_observation_handoff_v2_body_projection(**facts), newline=False)


def canonical_host_observation_handoff_v2_file_bytes(**facts: Any) -> bytes:
    """Encode one exact v2 handoff file with its BODY digest and one trailing LF."""

    body = host_observation_handoff_v2_body_projection(**facts)
    return _canonical_json(
        {
            **body,
            HOST_OBSERVATION_HANDOFF_V2_BODY_SHA256_FIELD: _body_sha256(body),
        }
    )


def all_case_sequence_intent_v1_body_projection(
    *,
    candidate_order_sha256: str,
    candidate_order: tuple[str, ...],
    case_count: int,
    claims_completion: bool,
) -> dict[str, Any]:
    """Return the exact canonical pre-execution all-case intent BODY."""

    _require_sha256(candidate_order_sha256, "all-case intent candidate order")
    if (
        type(candidate_order) is not tuple
        or any(type(item) is not str for item in candidate_order)
        or candidate_order != MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS
        or candidate_order_sha256 != _ordered_values_sha256(candidate_order)
    ):
        _fail("all-case intent candidate order differs")
    if _require_int(case_count, "all-case intent count", minimum=28, maximum=28) != 28:
        _fail("all-case intent count differs")
    if _require_bool(claims_completion, "all-case intent completion") is not False:
        _fail("pre-execution all-case intent cannot claim completion")
    return {
        "schema_version": ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION,
        "candidate_order_sha256": candidate_order_sha256,
        "candidate_order": list(candidate_order),
        "case_count": case_count,
        "claims_completion": claims_completion,
    }


def canonical_all_case_sequence_intent_v1_body_bytes(**facts: Any) -> bytes:
    """Encode the exact all-case intent BODY without a trailing LF."""

    return _canonical_json(all_case_sequence_intent_v1_body_projection(**facts), newline=False)


def canonical_all_case_sequence_intent_v1_file_bytes(**facts: Any) -> bytes:
    """Encode the exact all-case intent file with one trailing LF."""

    body = all_case_sequence_intent_v1_body_projection(**facts)
    return _canonical_json(
        {
            **body,
            ALL_CASE_SEQUENCE_INTENT_V1_BODY_SHA256_FIELD: _body_sha256(body),
        }
    )


def all_case_sequence_receipt_v1_body_projection(
    *,
    all_case_sequence_intent_file_sha256: str,
    all_case_sequence_intent_body_sha256: str,
    campaign_spine_body_sha256: str,
    ordered_terminal_handoff_inventory_sha256: str,
    case_count: int,
    terminal_coverage_complete: bool,
) -> dict[str, Any]:
    """Return the exact post-case sequence-receipt BODY projection."""

    for digest, label in (
        (all_case_sequence_intent_file_sha256, "sequence receipt intent file"),
        (all_case_sequence_intent_body_sha256, "sequence receipt intent body"),
        (campaign_spine_body_sha256, "sequence receipt campaign spine"),
        (ordered_terminal_handoff_inventory_sha256, "sequence receipt inventory"),
    ):
        _require_sha256(digest, label)
    if _require_int(case_count, "sequence receipt count", minimum=28, maximum=28) != 28:
        _fail("sequence receipt count differs")
    if _require_bool(terminal_coverage_complete, "sequence receipt coverage") is not True:
        _fail("sequence receipt requires exact terminal coverage")
    return {
        "schema_version": ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION,
        "all_case_sequence_intent_file_sha256": all_case_sequence_intent_file_sha256,
        "all_case_sequence_intent_body_sha256": all_case_sequence_intent_body_sha256,
        "campaign_spine_body_sha256": campaign_spine_body_sha256,
        "ordered_terminal_handoff_inventory_sha256": (ordered_terminal_handoff_inventory_sha256),
        "case_count": case_count,
        "terminal_coverage_complete": terminal_coverage_complete,
    }


def canonical_all_case_sequence_receipt_v1_body_bytes(**facts: Any) -> bytes:
    """Encode the exact all-case receipt BODY without a trailing LF."""

    return _canonical_json(all_case_sequence_receipt_v1_body_projection(**facts), newline=False)


def canonical_all_case_sequence_receipt_v1_file_bytes(**facts: Any) -> bytes:
    """Encode the exact all-case receipt file with one trailing LF."""

    body = all_case_sequence_receipt_v1_body_projection(**facts)
    return _canonical_json(
        {
            **body,
            ALL_CASE_SEQUENCE_RECEIPT_V1_BODY_SHA256_FIELD: _body_sha256(body),
        }
    )


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "batch_issued": False,
        "case_approved": False,
        "decision_evaluated": False,
        "evidence_created": False,
        "execution_authorized": False,
        "performance_claim_allowed": False,
        "production_plan_issued": False,
        "promotion_allowed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _readiness() -> dict[str, bool]:
    return {
        "batch_ready": False,
        "decision_ready": False,
        "execution_ready": False,
        "issuer_available": False,
        "production_ready": False,
        "qualification_ready": False,
    }


def _capabilities() -> dict[str, bool]:
    return {
        "clock": False,
        "default_inputs": False,
        "evaluator": False,
        "executor": False,
        "filesystem": False,
        "issuer": False,
        "network": False,
        "per_case_artifact": False,
    }


def _limitations() -> list[str]:
    return [
        "Only canonical structural metadata candidates are validated.",
        "Referenced artifacts are not read, authenticated, issued, or evaluated here.",
        "Every ordinal is present as success or terminal failure; neither tag is a decision.",
        "Publication file digests and sizes remain admitted metadata side channels.",
        "A future atomic all-case issuer and a separate evaluator remain mandatory.",
        "No candidate batch grants execution, qualification, evidence, or promotion.",
    ]


def _family_for_candidate(candidate_id: str) -> Literal["local", "external", "adapter"]:
    exact_candidate_id = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if exact_candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if exact_candidate_id in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("candidate ID is outside the frozen v3 universe")


def _require_case_projection(
    case_ordinal: object,
    candidate_id: object,
    qualification_case_id: object,
    label: str,
) -> None:
    ordinal = _require_int(
        case_ordinal,
        f"{label} ordinal",
        maximum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS) - 1,
    )
    exact_candidate_id = _require_identifier(candidate_id, f"{label} candidate ID")
    exact_case_id = _require_identifier(qualification_case_id, f"{label} case ID")
    if exact_candidate_id != MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS[ordinal]:
        _fail(f"{label} candidate projection differs")
    if exact_case_id != f"qualification_{ordinal:02d}_{exact_candidate_id}":
        _fail(f"{label} case-ID projection differs")


def _publication_profile(
    family: str,
) -> tuple[tuple[tuple[str, str], ...], str, str]:
    if family == "local":
        return (
            LOCAL_PUBLICATION_ROLE_PATHS,
            LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        )
    if family == "external":
        return (
            EXTERNAL_PUBLICATION_ROLE_PATHS,
            EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION,
            EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        )
    if family == "adapter":
        return (
            ADAPTER_PUBLICATION_ROLE_PATHS,
            STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION,
            STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        )
    _fail("publication family differs")


def _algorithmic_resource_receipt_schema(family: str) -> str:
    if family == "local":
        return LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "adapter":
        return ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    _fail("algorithmic resource-receipt family differs")


def _runner_execution_receipt_schema(candidate_id: str) -> str:
    family = _family_for_candidate(candidate_id)
    if family == "local":
        return LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION
    if candidate_id == "adapted_full_rainbow":
        return FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
    if candidate_id == "adapted_ppo_gru":
        return PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    _fail("adapter candidate has no runner-execution receipt schema")


def _runner_execution_receipt_role(family: str) -> str:
    if family == "local":
        return "local_runner_receipt"
    if family == "external":
        return "execution_receipt"
    if family == "adapter":
        return "runner_result_receipt"
    _fail("runner-execution receipt family differs")


def _ordered_values_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json(list(values), newline=False)).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV2:
    """One canonical JSON artifact identity; no referenced bytes are retained."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact body")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProducerIdentityV2:
    """Detached descriptor/source identity without a source reader."""

    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.descriptor_schema_version, "producer descriptor schema")
        _require_sha256(self.descriptor_sha256, "producer descriptor")
        _require_sha256(self.source_sha256, "producer source")
        if self.descriptor_sha256 == self.source_sha256:
            _fail("producer descriptor and source identities must remain distinct")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


def _require_production_host_executor(value: object, label: str) -> ProducerIdentityV2:
    if type(value) is not ProducerIdentityV2:
        _fail(f"{label} producer identity type differs")
    producer = value
    if producer.descriptor_schema_version != PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION:
        _fail(f"{label} producer descriptor schema differs")
    if any(
        digest
        in {
            NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256,
            NONEXECUTING_HOST_EXECUTOR_SOURCE_SHA256,
        }
        for digest in (producer.descriptor_sha256, producer.source_sha256)
    ):
        _fail(f"nonexecuting host contract cannot fill {label}")
    return producer


def _require_artifact_schema(
    value: ArtifactIdentityV2,
    expected: str,
    label: str,
) -> ArtifactIdentityV2:
    if type(value) is not ArtifactIdentityV2 or value.schema_version != expected:
        _fail(f"{label} artifact schema differs")
    return value


def _require_canonical_artifact_identity(
    value: ArtifactIdentityV2,
    *,
    schema_version: str,
    body_bytes: bytes,
    file_bytes: bytes,
    label: str,
) -> ArtifactIdentityV2:
    _require_artifact_schema(value, schema_version, label)
    if not hmac.compare_digest(
        hashlib.sha256(body_bytes).hexdigest(), value.body_sha256
    ) or not hmac.compare_digest(hashlib.sha256(file_bytes).hexdigest(), value.file_sha256):
        _fail(f"{label} canonical FILE/BODY identity differs")
    return value


def _require_host_observation_handoff_v2_identity(
    value: ArtifactIdentityV2,
    **facts: Any,
) -> ArtifactIdentityV2:
    return _require_canonical_artifact_identity(
        value,
        schema_version=HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
        body_bytes=canonical_host_observation_handoff_v2_body_bytes(**facts),
        file_bytes=canonical_host_observation_handoff_v2_file_bytes(**facts),
        label="host observation handoff",
    )


def _require_all_case_sequence_intent_v1_identity(
    value: ArtifactIdentityV2,
    **facts: Any,
) -> ArtifactIdentityV2:
    return _require_canonical_artifact_identity(
        value,
        schema_version=ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION,
        body_bytes=canonical_all_case_sequence_intent_v1_body_bytes(**facts),
        file_bytes=canonical_all_case_sequence_intent_v1_file_bytes(**facts),
        label="all-case sequence intent",
    )


def _require_all_case_sequence_receipt_v1_identity(
    value: ArtifactIdentityV2,
    **facts: Any,
) -> ArtifactIdentityV2:
    return _require_canonical_artifact_identity(
        value,
        schema_version=ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION,
        body_bytes=canonical_all_case_sequence_receipt_v1_body_bytes(**facts),
        file_bytes=canonical_all_case_sequence_receipt_v1_file_bytes(**facts),
        label="all-case sequence receipt",
    )


def _require_publication_reload_validation_v1_identity(
    value: ArtifactIdentityV2,
    *,
    publication_commitment_wrapper_file_sha256: str,
    publication_commitment_wrapper_body_sha256: str,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
    reload_performed: bool,
    reload_read_only: bool,
    label: str,
) -> ArtifactIdentityV2:
    _require_artifact_schema(
        value,
        QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        label,
    )
    body_bytes = canonical_qualification_publication_reload_validation_v1_body_bytes(
        publication_commitment_wrapper_file_sha256=(publication_commitment_wrapper_file_sha256),
        publication_commitment_wrapper_body_sha256=(publication_commitment_wrapper_body_sha256),
        expected_reload_observation_sha256=expected_reload_observation_sha256,
        actual_reload_observation_sha256=actual_reload_observation_sha256,
        reload_performed=reload_performed,
        reload_read_only=reload_read_only,
    )
    file_bytes = canonical_qualification_publication_reload_validation_v1_file_bytes(
        publication_commitment_wrapper_file_sha256=(publication_commitment_wrapper_file_sha256),
        publication_commitment_wrapper_body_sha256=(publication_commitment_wrapper_body_sha256),
        expected_reload_observation_sha256=expected_reload_observation_sha256,
        actual_reload_observation_sha256=actual_reload_observation_sha256,
        reload_performed=reload_performed,
        reload_read_only=reload_read_only,
    )
    if not hmac.compare_digest(
        hashlib.sha256(body_bytes).hexdigest(), value.body_sha256
    ) or not hmac.compare_digest(hashlib.sha256(file_bytes).hexdigest(), value.file_sha256):
        _fail(f"{label} canonical FILE/BODY identity differs")
    return value


def _with_body_sha256(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: _body_sha256(body)}


@dataclass(frozen=True, slots=True)
class CampaignSpineV2:
    """Caller-carried campaign identities shared by every case candidate."""

    qualification_plan_schema_version: str
    qualification_plan_file_sha256: str
    qualification_plan_body_sha256: str
    observation_registry_schema_version: str
    observation_registry_descriptor_sha256: str
    observation_registry_source_sha256: str
    plan_issuance_receipt: ArtifactIdentityV2
    case_ticket_registry: ArtifactIdentityV2
    publisher_registry: ArtifactIdentityV2
    seed_registry: ArtifactIdentityV2
    seed_pulse_record: ArtifactIdentityV2
    seed_trust_root_receipt: ArtifactIdentityV2
    quicknet_verifier: ProducerIdentityV2
    quicknet_verifier_binary_sha256: str
    quicknet_verifier_receipt: ArtifactIdentityV2
    seed_chronology_receipt: ArtifactIdentityV2
    local_source_candidate: ArtifactIdentityV2
    external_source_candidate: ArtifactIdentityV2
    adapter_source_candidate: ArtifactIdentityV2
    joint_source_closure_candidate: ArtifactIdentityV2
    joint_source_closure_local_file_sha256: str
    joint_source_closure_local_body_sha256: str
    joint_source_closure_external_file_sha256: str
    joint_source_closure_external_body_sha256: str
    joint_source_closure_adapter_file_sha256: str
    joint_source_closure_adapter_body_sha256: str
    sealed_staging_candidate: ArtifactIdentityV2
    sealed_staging_joint_source_closure_file_sha256: str
    sealed_staging_joint_source_closure_body_sha256: str
    fresh_build_candidate: ArtifactIdentityV2
    fresh_build_sealed_staging_file_sha256: str
    fresh_build_sealed_staging_body_sha256: str
    fresh_build_image_id: str
    image_id: str
    runtime_candidate: ArtifactIdentityV2
    runtime_qualification_receipt: ArtifactIdentityV2
    host_provisioning_receipt: ArtifactIdentityV2
    host_executor: ProducerIdentityV2
    full_resource_merger: ProducerIdentityV2
    algorithmic_resource_contract: ProducerIdentityV2
    storage_boundary_contract: ProducerIdentityV2
    all_case_sequence_intent: ArtifactIdentityV2
    all_case_sequence_intent_candidate_order_sha256: str
    all_case_sequence_intent_case_count: int
    all_case_sequence_intent_claims_completion: bool
    candidate_order_sha256: str
    resource_field_order_sha256: str

    def __post_init__(self) -> None:
        _require_exact_literal(
            self.qualification_plan_schema_version,
            QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
            "campaign qualification-plan schema",
        )
        _require_sha256(self.qualification_plan_file_sha256, "qualification plan file")
        _require_sha256(self.qualification_plan_body_sha256, "qualification plan body")
        _require_exact_literal(
            self.observation_registry_schema_version,
            QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION,
            "campaign observation-registry schema",
        )
        _require_sha256(
            self.observation_registry_descriptor_sha256,
            "observation registry descriptor",
        )
        _require_sha256(
            self.observation_registry_source_sha256,
            "independently pinned observation registry source",
        )
        if (
            self.observation_registry_descriptor_sha256
            != QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256
        ):
            _fail("campaign spine observation registry descriptor differs")
        _require_artifact_schema(
            self.plan_issuance_receipt,
            PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION,
            "plan issuance receipt",
        )
        _require_artifact_schema(
            self.case_ticket_registry,
            CASE_TICKET_REGISTRY_SCHEMA_VERSION,
            "case ticket registry",
        )
        _require_artifact_schema(
            self.publisher_registry,
            PUBLISHER_REGISTRY_SCHEMA_VERSION,
            "publisher registry",
        )
        _require_artifact_schema(
            self.seed_registry,
            QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
            "seed registry",
        )
        _require_artifact_schema(
            self.seed_pulse_record,
            QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
            "seed pulse record",
        )
        _require_artifact_schema(
            self.seed_trust_root_receipt,
            QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION,
            "seed trust-root receipt",
        )
        if type(self.quicknet_verifier) is not ProducerIdentityV2:
            _fail("Quicknet verifier producer identity type differs")
        if (
            self.quicknet_verifier.descriptor_schema_version
            != QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
        ):
            _fail("Quicknet verifier must use the future runtime-verifier schema")
        if any(
            digest
            in {
                SOURCE_ONLY_QUICKNET_VERIFIER_DESCRIPTOR_SHA256,
                SOURCE_ONLY_QUICKNET_VERIFIER_SOURCE_SHA256,
                SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256,
                SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256,
                SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256,
                SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256,
            }
            for digest in (
                self.quicknet_verifier.descriptor_sha256,
                self.quicknet_verifier.source_sha256,
            )
        ):
            _fail(
                "source-only or materialization-only Quicknet content cannot fill "
                "the runtime-verifier slot"
            )
        _require_sha256(self.quicknet_verifier_binary_sha256, "Quicknet verifier binary")
        _require_artifact_schema(
            self.quicknet_verifier_receipt,
            QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION,
            "Quicknet verifier receipt",
        )
        _require_artifact_schema(
            self.seed_chronology_receipt,
            SEED_CHRONOLOGY_RECEIPT_SCHEMA_VERSION,
            "seed chronology receipt",
        )
        _require_artifact_schema(
            self.local_source_candidate,
            LOCAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "local source candidate",
        )
        _require_artifact_schema(
            self.external_source_candidate,
            EXTERNAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "external source candidate",
        )
        _require_artifact_schema(
            self.adapter_source_candidate,
            ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "adapter source candidate",
        )
        _require_artifact_schema(
            self.joint_source_closure_candidate,
            JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION,
            "joint source-closure candidate",
        )
        for value, label in (
            (self.joint_source_closure_local_file_sha256, "joint closure local file"),
            (self.joint_source_closure_local_body_sha256, "joint closure local BODY"),
            (self.joint_source_closure_external_file_sha256, "joint closure external file"),
            (self.joint_source_closure_external_body_sha256, "joint closure external BODY"),
            (self.joint_source_closure_adapter_file_sha256, "joint closure adapter file"),
            (self.joint_source_closure_adapter_body_sha256, "joint closure adapter BODY"),
        ):
            _require_sha256(value, label)
        if (
            self.joint_source_closure_local_file_sha256 != self.local_source_candidate.file_sha256
            or self.joint_source_closure_local_body_sha256
            != self.local_source_candidate.body_sha256
            or self.joint_source_closure_external_file_sha256
            != self.external_source_candidate.file_sha256
            or self.joint_source_closure_external_body_sha256
            != self.external_source_candidate.body_sha256
            or self.joint_source_closure_adapter_file_sha256
            != self.adapter_source_candidate.file_sha256
            or self.joint_source_closure_adapter_body_sha256
            != self.adapter_source_candidate.body_sha256
        ):
            _fail("joint source closure is cross-wired from its three source candidates")
        _require_artifact_schema(
            self.sealed_staging_candidate,
            SEALED_STAGING_CANDIDATE_SCHEMA_VERSION,
            "sealed staging candidate",
        )
        _require_sha256(
            self.sealed_staging_joint_source_closure_file_sha256,
            "sealed staging joint-closure file",
        )
        _require_sha256(
            self.sealed_staging_joint_source_closure_body_sha256,
            "sealed staging joint-closure BODY",
        )
        if (
            self.sealed_staging_joint_source_closure_file_sha256
            != self.joint_source_closure_candidate.file_sha256
            or self.sealed_staging_joint_source_closure_body_sha256
            != self.joint_source_closure_candidate.body_sha256
        ):
            _fail("sealed staging candidate is cross-wired from joint source closure")
        _require_artifact_schema(
            self.fresh_build_candidate,
            FRESH_BUILD_CANDIDATE_SCHEMA_VERSION,
            "fresh build candidate",
        )
        _require_sha256(
            self.fresh_build_sealed_staging_file_sha256,
            "fresh build sealed-staging file",
        )
        _require_sha256(
            self.fresh_build_sealed_staging_body_sha256,
            "fresh build sealed-staging BODY",
        )
        if (
            self.fresh_build_sealed_staging_file_sha256 != self.sealed_staging_candidate.file_sha256
            or self.fresh_build_sealed_staging_body_sha256
            != self.sealed_staging_candidate.body_sha256
        ):
            _fail("fresh build candidate is cross-wired from sealed staging")
        _require_image_id(self.fresh_build_image_id, "fresh-build CPU OCI image")
        _require_image_id(self.image_id, "campaign CPU OCI image")
        if self.fresh_build_image_id != self.image_id:
            _fail("fresh build image projection differs from campaign image")
        _require_artifact_schema(
            self.runtime_candidate,
            RUNTIME_CANDIDATE_SCHEMA_VERSION,
            "runtime candidate",
        )
        _require_artifact_schema(
            self.runtime_qualification_receipt,
            RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime qualification receipt",
        )
        _require_artifact_schema(
            self.host_provisioning_receipt,
            HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host provisioning receipt",
        )
        if type(self.host_executor) is not ProducerIdentityV2:
            _fail("host executor producer identity type differs")
        if (
            self.host_executor.descriptor_schema_version
            != PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION
        ):
            _fail("host executor must use the production v2 descriptor schema")
        if any(
            digest
            in {
                NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256,
                NONEXECUTING_HOST_EXECUTOR_SOURCE_SHA256,
            }
            for digest in (
                self.host_executor.descriptor_sha256,
                self.host_executor.source_sha256,
            )
        ):
            _fail("nonexecuting host contract cannot fill the production executor slot")
        if type(self.full_resource_merger) is not ProducerIdentityV2:
            _fail("full resource merger producer identity type differs")
        if (
            self.full_resource_merger.descriptor_schema_version
            != FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION
        ):
            _fail("full resource merger producer descriptor schema differs")
        if self.full_resource_merger.descriptor_sha256 in {
            PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256,
            PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
        } or self.full_resource_merger.source_sha256 in {
            PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256,
            PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
        }:
            _fail("endpoint observer cannot substitute for the full resource merger")
        if (
            self.full_resource_merger.descriptor_sha256
            in ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S
            or self.full_resource_merger.source_sha256
            in ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S
        ):
            _fail("algorithmic validator cannot fill the full-merger slot")
        if self.full_resource_merger.descriptor_sha256 in {
            FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
            FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
        } or self.full_resource_merger.source_sha256 in {
            FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
            FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
        }:
            _fail("storage validator cannot fill the full-merger slot")
        for producer, schema, descriptor_sha256, source_sha256, label in (
            (
                self.algorithmic_resource_contract,
                ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
                "algorithmic resource contract",
            ),
            (
                self.storage_boundary_contract,
                QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
                "storage boundary contract",
            ),
        ):
            if (
                type(producer) is not ProducerIdentityV2
                or producer.descriptor_schema_version != schema
                or producer.descriptor_sha256 != descriptor_sha256
                or producer.source_sha256 != source_sha256
            ):
                _fail(f"{label} finalized implementation identity differs")
        _require_artifact_schema(
            self.all_case_sequence_intent,
            ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION,
            "pre-execution all-case sequence intent",
        )
        _require_sha256(
            self.all_case_sequence_intent_candidate_order_sha256,
            "all-case sequence-intent candidate order",
        )
        _require_sha256(self.candidate_order_sha256, "campaign candidate order")
        _require_sha256(self.resource_field_order_sha256, "campaign resource-field order")
        if (
            self.all_case_sequence_intent_candidate_order_sha256 != self.candidate_order_sha256
            or _require_int(
                self.all_case_sequence_intent_case_count,
                "all-case sequence intent case count",
                minimum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
                maximum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            )
            != len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS)
            or _require_bool(
                self.all_case_sequence_intent_claims_completion,
                "all-case sequence intent completion claim",
            )
            is not False
        ):
            _fail("pre-execution all-case sequence intent projections differ")
        if self.candidate_order_sha256 != _ordered_values_sha256(
            MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS
        ):
            _fail("campaign candidate-order digest differs")
        if self.resource_field_order_sha256 != _ordered_values_sha256(RESOURCE_CEILING_FIELDS):
            _fail("campaign resource-field-order digest differs")
        _require_all_case_sequence_intent_v1_identity(
            self.all_case_sequence_intent,
            candidate_order_sha256=self.all_case_sequence_intent_candidate_order_sha256,
            candidate_order=MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS,
            case_count=self.all_case_sequence_intent_case_count,
            claims_completion=self.all_case_sequence_intent_claims_completion,
        )

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "qualification_plan_schema_version": self.qualification_plan_schema_version,
            "qualification_plan_file_sha256": self.qualification_plan_file_sha256,
            "qualification_plan_body_sha256": self.qualification_plan_body_sha256,
            "observation_registry_schema_version": self.observation_registry_schema_version,
            "observation_registry_descriptor_sha256": (self.observation_registry_descriptor_sha256),
            "observation_registry_source_sha256": self.observation_registry_source_sha256,
            "plan_issuance_receipt": self.plan_issuance_receipt.to_dict(),
            "case_ticket_registry": self.case_ticket_registry.to_dict(),
            "publisher_registry": self.publisher_registry.to_dict(),
            "seed_registry": self.seed_registry.to_dict(),
            "seed_pulse_record": self.seed_pulse_record.to_dict(),
            "seed_trust_root_receipt": self.seed_trust_root_receipt.to_dict(),
            "quicknet_verifier": self.quicknet_verifier.to_dict(),
            "quicknet_verifier_binary_sha256": self.quicknet_verifier_binary_sha256,
            "quicknet_verifier_receipt": self.quicknet_verifier_receipt.to_dict(),
            "seed_chronology_receipt": self.seed_chronology_receipt.to_dict(),
            "local_source_candidate": self.local_source_candidate.to_dict(),
            "external_source_candidate": self.external_source_candidate.to_dict(),
            "adapter_source_candidate": self.adapter_source_candidate.to_dict(),
            "joint_source_closure_candidate": (self.joint_source_closure_candidate.to_dict()),
            "joint_source_closure_local_file_sha256": (self.joint_source_closure_local_file_sha256),
            "joint_source_closure_local_body_sha256": (self.joint_source_closure_local_body_sha256),
            "joint_source_closure_external_file_sha256": (
                self.joint_source_closure_external_file_sha256
            ),
            "joint_source_closure_external_body_sha256": (
                self.joint_source_closure_external_body_sha256
            ),
            "joint_source_closure_adapter_file_sha256": (
                self.joint_source_closure_adapter_file_sha256
            ),
            "joint_source_closure_adapter_body_sha256": (
                self.joint_source_closure_adapter_body_sha256
            ),
            "sealed_staging_candidate": self.sealed_staging_candidate.to_dict(),
            "sealed_staging_joint_source_closure_file_sha256": (
                self.sealed_staging_joint_source_closure_file_sha256
            ),
            "sealed_staging_joint_source_closure_body_sha256": (
                self.sealed_staging_joint_source_closure_body_sha256
            ),
            "fresh_build_candidate": self.fresh_build_candidate.to_dict(),
            "fresh_build_sealed_staging_file_sha256": (self.fresh_build_sealed_staging_file_sha256),
            "fresh_build_sealed_staging_body_sha256": (self.fresh_build_sealed_staging_body_sha256),
            "fresh_build_image_id": self.fresh_build_image_id,
            "image_id": self.image_id,
            "runtime_candidate": self.runtime_candidate.to_dict(),
            "runtime_qualification_receipt": (self.runtime_qualification_receipt.to_dict()),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "host_executor": self.host_executor.to_dict(),
            "full_resource_merger": self.full_resource_merger.to_dict(),
            "algorithmic_resource_contract": self.algorithmic_resource_contract.to_dict(),
            "storage_boundary_contract": self.storage_boundary_contract.to_dict(),
            "all_case_sequence_intent": self.all_case_sequence_intent.to_dict(),
            "all_case_sequence_intent_candidate_order_sha256": (
                self.all_case_sequence_intent_candidate_order_sha256
            ),
            "all_case_sequence_intent_case_count": (self.all_case_sequence_intent_case_count),
            "all_case_sequence_intent_claims_completion": (
                self.all_case_sequence_intent_claims_completion
            ),
            "candidate_order_sha256": self.candidate_order_sha256,
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "candidate_order": list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            "resource_fields": list(RESOURCE_CEILING_FIELDS),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "campaign_spine_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


@dataclass(frozen=True, slots=True)
class CaseSpineV2:
    """Exact single-use case identity shared by one terminal record."""

    campaign_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    qualification_case_manifest: ArtifactIdentityV2
    case_execution_ticket: ArtifactIdentityV2
    plan_issuance_receipt_file_sha256: str
    plan_issuance_receipt_body_sha256: str
    publisher_registry_entry: ArtifactIdentityV2
    resource_requirement_body_sha256: str
    seed_case_record_sha256: str
    seed_derivation_record_sha256: str
    environment_derivation_sha256: str
    agent_derivation_sha256: str
    environment_seed_commitment_sha256: str
    agent_seed_commitment_sha256: str
    attempt_ordinal: int
    ticket_single_use: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.campaign_spine_sha256, "case campaign spine")
        ordinal = _require_int(
            self.case_ordinal,
            "case ordinal",
            maximum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS) - 1,
        )
        _require_case_projection(
            ordinal,
            self.candidate_id,
            self.qualification_case_id,
            "case spine",
        )
        if self.candidate_id != MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS[ordinal]:
            _fail("case ordinal and candidate ID differ from the exact global order")
        expected_family = _family_for_candidate(self.candidate_id)
        _require_exact_literal(
            self.candidate_family,
            expected_family,
            "case candidate family",
        )
        expected_case_id = f"qualification_{ordinal:02d}_{self.candidate_id}"
        if self.qualification_case_id != expected_case_id:
            _fail("qualification case ID differs")
        _require_artifact_schema(
            self.qualification_case_manifest,
            QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION,
            "qualification case manifest",
        )
        _require_artifact_schema(
            self.case_execution_ticket,
            CASE_EXECUTION_TICKET_SCHEMA_VERSION,
            "case execution ticket",
        )
        _require_sha256(
            self.plan_issuance_receipt_file_sha256,
            "case plan issuance receipt file",
        )
        _require_sha256(
            self.plan_issuance_receipt_body_sha256,
            "case plan issuance receipt body",
        )
        _require_artifact_schema(
            self.publisher_registry_entry,
            PUBLISHER_REGISTRY_ENTRY_SCHEMA_VERSION,
            "publisher registry entry",
        )
        for value, label in (
            (self.resource_requirement_body_sha256, "resource requirement body"),
            (self.seed_case_record_sha256, "seed case record"),
            (self.seed_derivation_record_sha256, "seed derivation record"),
            (self.environment_derivation_sha256, "environment derivation"),
            (self.agent_derivation_sha256, "agent derivation"),
            (self.environment_seed_commitment_sha256, "environment seed commitment"),
            (self.agent_seed_commitment_sha256, "agent seed commitment"),
        ):
            _require_sha256(value, label)
        if _require_int(self.attempt_ordinal, "case attempt ordinal", maximum=0) != 0:
            _fail("case attempt ordinal must be exact zero")
        if _require_bool(self.ticket_single_use, "case ticket single-use") is not True:
            _fail("case execution ticket must be single-use")
        if _require_bool(self.same_case_retry_permitted, "same-case retry") is not False:
            _fail("same-case retry can never be permitted")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "campaign_spine_sha256": self.campaign_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "qualification_case_manifest": self.qualification_case_manifest.to_dict(),
            "case_execution_ticket": self.case_execution_ticket.to_dict(),
            "plan_issuance_receipt_file_sha256": (self.plan_issuance_receipt_file_sha256),
            "plan_issuance_receipt_body_sha256": (self.plan_issuance_receipt_body_sha256),
            "publisher_registry_entry": self.publisher_registry_entry.to_dict(),
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "seed_case_record_sha256": self.seed_case_record_sha256,
            "seed_derivation_record_sha256": self.seed_derivation_record_sha256,
            "environment_derivation_sha256": self.environment_derivation_sha256,
            "agent_derivation_sha256": self.agent_derivation_sha256,
            "environment_seed_commitment_sha256": (self.environment_seed_commitment_sha256),
            "agent_seed_commitment_sha256": self.agent_seed_commitment_sha256,
            "attempt_ordinal": self.attempt_ordinal,
            "ticket_single_use": self.ticket_single_use,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "case_spine_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


@dataclass(frozen=True, slots=True)
class ProbeCandidateV2:
    """One digest-only leaf candidate linked to a case and host receipt."""

    probe_kind: str
    schema_version: str
    file_sha256: str
    body_sha256: str
    case_spine_sha256: str
    host_execution_receipt_file_sha256: str
    host_execution_receipt_body_sha256: str

    def __post_init__(self) -> None:
        probe_kind = _require_one_of(self.probe_kind, PROBE_KINDS, "probe candidate kind")
        _require_exact_literal(
            self.schema_version,
            PROBE_SCHEMA_BY_KIND[probe_kind],
            "probe candidate schema",
        )
        for value, label in (
            (self.file_sha256, "probe candidate file"),
            (self.body_sha256, "probe candidate body"),
            (self.case_spine_sha256, "probe case spine"),
            (self.host_execution_receipt_file_sha256, "probe host receipt file"),
            (self.host_execution_receipt_body_sha256, "probe host receipt body"),
        ):
            _require_sha256(value, label)

    def to_dict(self) -> dict[str, str]:
        return {
            "probe_kind": self.probe_kind,
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "host_execution_receipt_file_sha256": (self.host_execution_receipt_file_sha256),
            "host_execution_receipt_body_sha256": (self.host_execution_receipt_body_sha256),
        }


@dataclass(frozen=True, slots=True)
class PublicationFileCandidateV2:
    """One bounded role/name/size/digest record with no file bytes."""

    role: str
    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, "publication file role")
        _require_relative_path(self.name, "publication file name")
        _require_int(
            self.size_bytes,
            "publication file size",
            maximum=MAX_PUBLICATION_FILE_BYTES,
        )
        _require_sha256(self.sha256, "publication file")
        if (self.size_bytes == 0) is not (self.sha256 == EMPTY_FILE_SHA256):
            _fail("publication file zero length and empty digest must agree exactly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _publication_inventory_sha256(
    files: tuple[PublicationFileCandidateV2, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {"files": [item.to_dict() for item in files]},
            newline=False,
        )
    ).hexdigest()


def _normalized_wrapper_capabilities() -> dict[str, bool]:
    return {
        "acceptance_evaluation": False,
        "case_issuance": False,
        "content_value_decoding": False,
        "execution": False,
        "file_publication": False,
        "host_provisioning": False,
        "payload_byte_transport": False,
        "publication_reload": False,
        "reload_digest_equality_validation": False,
    }


def _normalized_wrapper_readiness() -> dict[str, bool]:
    return {
        "host_execution_ready": False,
        "observation_ready": False,
        "publication_ready": False,
        "qualification_ready": False,
        "reload_observed": False,
    }


def _normalized_wrapper_authority() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "observation_issuance_authorized": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "scientific_evidence_created": False,
    }


def _normalized_wrapper_claims() -> dict[str, bool]:
    return {
        "build_qualified": False,
        "performance_claim_allowed": False,
        "publisher_qualified": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "source_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _normalized_wrapper_limitations() -> list[str]:
    return [
        "The wrapper authenticates metadata commitments and never reads publication files.",
        "The reload digest is expected content; this wrapper performs no reload.",
        "A later phase must observe reload output and validate exact digest equality.",
        "Native receipts remain separate artifacts and are not replaced by this wrapper.",
        (
            "Canonical metadata grants no execution, observation, qualification, "
            "or evidence authority."
        ),
    ]


def normalized_publication_commitment_wrapper_v1_body_projection(
    *,
    case_spine_sha256: str,
    case_ordinal: int,
    candidate_id: str,
    candidate_family: Literal["local", "external", "adapter"],
    qualification_case_id: str,
    publisher: ProducerIdentityV2,
    publisher_metadata: ArtifactIdentityV2,
    native_atomic_producer: ProducerIdentityV2,
    native_publication_receipt: ArtifactIdentityV2 | None,
    publication_address_sha256: str,
    publication_manifest_file_sha256: str,
    publication_manifest_body_sha256: str,
    file_inventory_sha256: str,
    published_bundle_sha256: str,
    expected_reload_observation_sha256: str,
    file_count: int,
    total_size_bytes: int,
    maximum_total_size_bytes: int,
    video_slot_mode: str,
    files: tuple[PublicationFileCandidateV2, ...],
) -> dict[str, Any]:
    """Return the exact source-only normalized wrapper v1 BODY projection."""

    _require_case_projection(
        case_ordinal,
        candidate_id,
        qualification_case_id,
        "normalized wrapper",
    )
    if _require_identifier(candidate_family, "normalized wrapper family") != _family_for_candidate(
        candidate_id
    ):
        _fail("normalized wrapper candidate family differs")
    _require_sha256(case_spine_sha256, "normalized wrapper case spine")
    for value, label in (
        (publication_address_sha256, "normalized wrapper publication address"),
        (publication_manifest_file_sha256, "normalized wrapper manifest file"),
        (publication_manifest_body_sha256, "normalized wrapper manifest body"),
        (file_inventory_sha256, "normalized wrapper file inventory"),
        (published_bundle_sha256, "normalized wrapper published bundle"),
        (expected_reload_observation_sha256, "normalized wrapper expected reload"),
    ):
        _require_sha256(value, label)
    if (
        type(publisher) is not ProducerIdentityV2
        or type(native_atomic_producer) is not ProducerIdentityV2
    ):
        _fail("normalized wrapper producer identities differ")
    if type(publisher_metadata) is not ArtifactIdentityV2 or (
        native_publication_receipt is not None
        and type(native_publication_receipt) is not ArtifactIdentityV2
    ):
        _fail("normalized wrapper artifact identities differ")
    if type(files) is not tuple or any(
        type(item) is not PublicationFileCandidateV2 for item in files
    ):
        _fail("normalized wrapper files must be one exact tuple")
    _require_int(file_count, "normalized wrapper file count")
    _require_int(total_size_bytes, "normalized wrapper aggregate size")
    _require_int(maximum_total_size_bytes, "normalized wrapper aggregate ceiling")
    exact_video_mode = _require_identifier(video_slot_mode, "normalized wrapper video mode")
    return {
        "schema_version": NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "status": NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_STATUS,
        "classification": NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_CLASSIFICATION,
        "case_spine_sha256": case_spine_sha256,
        "case_ordinal": case_ordinal,
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "qualification_case_id": qualification_case_id,
        "publisher": publisher.to_dict(),
        "publisher_metadata": publisher_metadata.to_dict(),
        "native_atomic_producer": native_atomic_producer.to_dict(),
        "native_publication_receipt": (
            None if native_publication_receipt is None else native_publication_receipt.to_dict()
        ),
        "publication_address_sha256": publication_address_sha256,
        "publication_manifest_file_sha256": publication_manifest_file_sha256,
        "publication_manifest_body_sha256": publication_manifest_body_sha256,
        "file_inventory_sha256": file_inventory_sha256,
        "published_bundle_sha256": published_bundle_sha256,
        "expected_reload_observation_sha256": expected_reload_observation_sha256,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "maximum_total_size_bytes": maximum_total_size_bytes,
        "video_slot_mode": exact_video_mode,
        "files": [item.to_dict() for item in files],
        "reload_performed_by_wrapper": False,
        "reload_digest_equality_validated_by_wrapper": False,
        "content_values_read_by_wrapper": False,
        "payload_bytes_transported_by_wrapper": False,
        "capabilities": _normalized_wrapper_capabilities(),
        "readiness": _normalized_wrapper_readiness(),
        "authority": _normalized_wrapper_authority(),
        "claims": _normalized_wrapper_claims(),
        "limitations": _normalized_wrapper_limitations(),
    }


def canonical_normalized_publication_commitment_wrapper_v1_body_bytes(**facts: Any) -> bytes:
    """Encode the normalized wrapper BODY without a trailing LF."""

    return _canonical_json(
        normalized_publication_commitment_wrapper_v1_body_projection(**facts),
        newline=False,
    )


def canonical_normalized_publication_commitment_wrapper_v1_file_bytes(**facts: Any) -> bytes:
    """Encode the normalized wrapper full file with exactly one trailing LF."""

    body = normalized_publication_commitment_wrapper_v1_body_projection(**facts)
    return _canonical_json(
        {
            **body,
            NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_SHA256_FIELD: _body_sha256(body),
        }
    )


@dataclass(frozen=True, slots=True)
class PublicationCandidateV2:
    """Exact family publication metadata linked to one host success receipt."""

    schema_version: str
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    case_ordinal: int
    qualification_case_id: str
    case_spine_sha256: str
    host_execution_receipt_file_sha256: str
    host_execution_receipt_body_sha256: str
    host_terminal_metadata_file_sha256: str
    host_terminal_metadata_body_sha256: str
    host_observation_handoff: ArtifactIdentityV2
    publisher_registry_entry_file_sha256: str
    publisher_registry_entry_body_sha256: str
    publisher_metadata: ArtifactIdentityV2
    runner_execution_receipt: ArtifactIdentityV2
    publisher: ProducerIdentityV2
    native_atomic_producer: ProducerIdentityV2
    native_publication_receipt: ArtifactIdentityV2 | None
    publication_commitment_contract_descriptor_sha256: str
    publication_commitment_contract_source_sha256: str
    publication_commitment_wrapper: ArtifactIdentityV2
    wrapper_case_spine_sha256: str
    wrapper_candidate_id: str
    wrapper_candidate_family: Literal["local", "external", "adapter"]
    wrapper_publisher_descriptor_sha256: str
    wrapper_publisher_source_sha256: str
    wrapper_publisher_metadata_file_sha256: str
    wrapper_publisher_metadata_body_sha256: str
    wrapper_native_atomic_producer_descriptor_sha256: str
    wrapper_native_atomic_producer_source_sha256: str
    wrapper_native_publication_receipt_file_sha256: str | None
    wrapper_native_publication_receipt_body_sha256: str | None
    wrapper_publication_address_sha256: str
    wrapper_file_inventory_sha256: str
    wrapper_published_bundle_sha256: str
    wrapper_expected_reload_observation_sha256: str
    wrapper_video_slot_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ]
    publication_address_sha256: str
    publication_manifest_file_sha256: str
    publication_manifest_body_sha256: str
    published_bundle_sha256: str
    reload_observation_sha256: str
    publication_reload_validation: ArtifactIdentityV2
    reload_validation_wrapper_file_sha256: str
    reload_validation_wrapper_body_sha256: str
    reload_validation_expected_reload_observation_sha256: str
    reload_validation_actual_reload_observation_sha256: str
    reload_validation_reload_performed: bool
    reload_validation_read_only: bool
    file_inventory_sha256: str
    file_count: int
    total_size_bytes: int
    maximum_total_size_bytes: int
    video_slot_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ]
    files: tuple[PublicationFileCandidateV2, ...]
    publication_committed: bool
    value_decoding_performed: bool
    byte_transport_performed: bool
    retry_count: int

    def __post_init__(self) -> None:
        _require_exact_literal(
            self.schema_version,
            QUALIFICATION_PUBLICATION_CANDIDATE_V2_SCHEMA_VERSION,
            "publication candidate schema",
        )
        _require_case_projection(
            self.case_ordinal,
            self.candidate_id,
            self.qualification_case_id,
            "publication candidate",
        )
        expected_family = _family_for_candidate(self.candidate_id)
        _require_exact_literal(
            self.candidate_family,
            expected_family,
            "publication candidate family",
        )
        (
            expected_paths,
            expected_metadata_schema,
            expected_descriptor_schema,
        ) = _publication_profile(expected_family)
        for value, label in (
            (self.case_spine_sha256, "publication case spine"),
            (self.host_execution_receipt_file_sha256, "publication host receipt file"),
            (self.host_execution_receipt_body_sha256, "publication host receipt body"),
            (self.host_terminal_metadata_file_sha256, "publication terminal file"),
            (self.host_terminal_metadata_body_sha256, "publication terminal body"),
            (self.publisher_registry_entry_file_sha256, "publication registry entry file"),
            (self.publisher_registry_entry_body_sha256, "publication registry entry body"),
            (self.publication_address_sha256, "publication address"),
            (self.publication_manifest_file_sha256, "publication manifest file"),
            (self.publication_manifest_body_sha256, "publication manifest body"),
            (self.published_bundle_sha256, "published bundle"),
            (self.reload_observation_sha256, "publication reload observation"),
            (
                self.reload_validation_wrapper_file_sha256,
                "reload validation wrapper file",
            ),
            (
                self.reload_validation_wrapper_body_sha256,
                "reload validation wrapper body",
            ),
            (
                self.reload_validation_expected_reload_observation_sha256,
                "reload validation expected observation",
            ),
            (
                self.reload_validation_actual_reload_observation_sha256,
                "reload validation actual observation",
            ),
            (self.file_inventory_sha256, "publication file inventory"),
            (self.wrapper_case_spine_sha256, "wrapper case spine"),
            (
                self.publication_commitment_contract_descriptor_sha256,
                "normalized wrapper contract descriptor",
            ),
            (
                self.publication_commitment_contract_source_sha256,
                "normalized wrapper contract source",
            ),
            (
                self.wrapper_publisher_descriptor_sha256,
                "wrapper publisher descriptor",
            ),
            (self.wrapper_publisher_source_sha256, "wrapper publisher source"),
            (
                self.wrapper_publisher_metadata_file_sha256,
                "wrapper publisher metadata file",
            ),
            (
                self.wrapper_publisher_metadata_body_sha256,
                "wrapper publisher metadata body",
            ),
            (
                self.wrapper_native_atomic_producer_descriptor_sha256,
                "wrapper native producer descriptor",
            ),
            (
                self.wrapper_native_atomic_producer_source_sha256,
                "wrapper native producer source",
            ),
            (self.wrapper_publication_address_sha256, "wrapper publication address"),
            (self.wrapper_file_inventory_sha256, "wrapper file inventory"),
            (self.wrapper_published_bundle_sha256, "wrapper published bundle"),
            (
                self.wrapper_expected_reload_observation_sha256,
                "wrapper expected reload observation",
            ),
        ):
            _require_sha256(value, label)
        if (
            self.publication_commitment_contract_descriptor_sha256
            != FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256
            or self.publication_commitment_contract_source_sha256
            != FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256
        ):
            _fail("normalized publication contract implementation identity differs")
        wrapper_native_receipt_values = (
            self.wrapper_native_publication_receipt_file_sha256,
            self.wrapper_native_publication_receipt_body_sha256,
        )
        for index, native_receipt_value in enumerate(wrapper_native_receipt_values):
            _require_optional_sha256(
                native_receipt_value,
                f"wrapper native receipt identity {index}",
            )
        if (wrapper_native_receipt_values[0] is None) != (wrapper_native_receipt_values[1] is None):
            _fail("wrapper native receipt projections have partial presence")
        _require_artifact_schema(
            self.publisher_metadata,
            expected_metadata_schema,
            "publisher metadata",
        )
        _require_artifact_schema(
            self.host_observation_handoff,
            HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
            "publication host observation handoff",
        )
        _require_artifact_schema(
            self.runner_execution_receipt,
            _runner_execution_receipt_schema(self.candidate_id),
            "publication runner-execution receipt",
        )
        if type(self.publisher) is not ProducerIdentityV2:
            _fail("publication producer identity type differs")
        if self.publisher.descriptor_schema_version != expected_descriptor_schema:
            _fail("publication producer descriptor schema differs from its family")
        if expected_family == "adapter" and (
            self.publisher.descriptor_sha256 in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
            or self.publisher.source_sha256 in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
        ):
            _fail("unqualified adapter implementation cannot fill a strict publisher slot")
        if type(self.native_atomic_producer) is not ProducerIdentityV2:
            _fail("native atomic-publication producer identity type differs")
        expected_atomic_schema = (
            STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            if expected_family == "adapter"
            else ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        )
        if self.native_atomic_producer.descriptor_schema_version != expected_atomic_schema:
            _fail("native atomic-publication producer schema differs from its family")
        if expected_family == "adapter" and (
            self.native_atomic_producer.descriptor_sha256
            in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
            or self.native_atomic_producer.source_sha256
            in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
        ):
            _fail("unqualified adapter implementation cannot fill a strict atomic-publication slot")
        if expected_family == "local":
            if self.native_publication_receipt is not None:
                _fail("local publication exposes atomic producer pins, not a native receipt")
        else:
            expected_native_receipt_schema = (
                EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
                if expected_family == "external"
                else STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
            )
            if self.native_publication_receipt is None:
                _fail("nonlocal publication requires one exact native receipt identity")
            _require_artifact_schema(
                self.native_publication_receipt,
                expected_native_receipt_schema,
                "native publication receipt",
            )
        _require_artifact_schema(
            self.publication_commitment_wrapper,
            NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            "future normalized publication commitment wrapper",
        )
        _require_artifact_schema(
            self.publication_reload_validation,
            QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        )
        _require_exact_literal(
            self.wrapper_candidate_id,
            self.candidate_id,
            "publication wrapper candidate ID",
        )
        _require_exact_literal(
            self.wrapper_candidate_family,
            self.candidate_family,
            "publication wrapper candidate family",
        )
        video_modes = (
            "not_applicable",
            "absent_for_continuing_zero_length_slot",
            "opaque_ppo_video",
        )
        _require_one_of(self.video_slot_mode, video_modes, "publication video-slot mode")
        _require_exact_literal(
            self.wrapper_video_slot_mode,
            self.video_slot_mode,
            "publication wrapper video-slot mode",
        )
        native_receipt_file = (
            None
            if self.native_publication_receipt is None
            else self.native_publication_receipt.file_sha256
        )
        native_receipt_body = (
            None
            if self.native_publication_receipt is None
            else self.native_publication_receipt.body_sha256
        )
        if (
            self.wrapper_case_spine_sha256 != self.case_spine_sha256
            or self.wrapper_candidate_id != self.candidate_id
            or self.wrapper_candidate_family != self.candidate_family
            or self.wrapper_publisher_descriptor_sha256 != self.publisher.descriptor_sha256
            or self.wrapper_publisher_source_sha256 != self.publisher.source_sha256
            or self.wrapper_publisher_metadata_file_sha256 != self.publisher_metadata.file_sha256
            or self.wrapper_publisher_metadata_body_sha256 != self.publisher_metadata.body_sha256
            or self.wrapper_native_atomic_producer_descriptor_sha256
            != self.native_atomic_producer.descriptor_sha256
            or self.wrapper_native_atomic_producer_source_sha256
            != self.native_atomic_producer.source_sha256
            or self.wrapper_native_publication_receipt_file_sha256 != native_receipt_file
            or self.wrapper_native_publication_receipt_body_sha256 != native_receipt_body
            or self.wrapper_publication_address_sha256 != self.publication_address_sha256
            or self.wrapper_file_inventory_sha256 != self.file_inventory_sha256
            or self.wrapper_published_bundle_sha256 != self.published_bundle_sha256
            or self.wrapper_expected_reload_observation_sha256 != self.reload_observation_sha256
            or self.wrapper_video_slot_mode != self.video_slot_mode
        ):
            _fail("publication commitment wrapper projections are cross-wired")
        if (
            self.reload_validation_wrapper_file_sha256
            != self.publication_commitment_wrapper.file_sha256
            or self.reload_validation_wrapper_body_sha256
            != self.publication_commitment_wrapper.body_sha256
            or self.reload_validation_expected_reload_observation_sha256
            != self.wrapper_expected_reload_observation_sha256
            or self.reload_validation_actual_reload_observation_sha256
            != self.reload_observation_sha256
            or _require_bool(
                self.reload_validation_reload_performed,
                "publication reload performed fact",
            )
            is not True
            or _require_bool(
                self.reload_validation_read_only,
                "publication reload read-only fact",
            )
            is not True
        ):
            _fail("publication reload validation projections are cross-wired")
        if type(self.files) is not tuple or any(
            type(item) is not PublicationFileCandidateV2 for item in self.files
        ):
            _fail("publication files must be one exact immutable tuple")
        if tuple((item.role, item.name) for item in self.files) != expected_paths:
            _fail("publication family role/name inventory differs")
        runner_role = _runner_execution_receipt_role(expected_family)
        runner_file = next(item for item in self.files if item.role == runner_role)
        if runner_file.sha256 != self.runner_execution_receipt.file_sha256:
            _fail("publication runner receipt differs from its inventory role")
        if len({item.role for item in self.files}) != len(self.files) or len(
            {item.name for item in self.files}
        ) != len(self.files):
            _fail("publication role or filename is duplicated")
        empty_permitted_roles = {"stdout", "stderr", "upstream_video_slot"}
        if any(
            item.size_bytes == 0 and item.role not in empty_permitted_roles for item in self.files
        ):
            _fail("publication canonical manifest, receipt, or data artifact is empty")
        if self.publication_address_sha256 != self.files[0].sha256:
            _fail("publication address must equal the publication manifest file digest")
        if self.publication_manifest_file_sha256 != self.files[0].sha256:
            _fail("publication manifest identity differs from the exact inventory")
        if self.file_count != len(expected_paths):
            _fail("publication file count differs")
        total = sum(item.size_bytes for item in self.files)
        if self.total_size_bytes != total:
            _fail("publication aggregate size differs from the exact inventory")
        if self.maximum_total_size_bytes != MAX_PUBLICATION_TOTAL_BYTES:
            _fail("publication aggregate ceiling must remain exact 1 GiB")
        if total > self.maximum_total_size_bytes:
            _fail("publication aggregate exceeds its frozen ceiling")
        if self.file_inventory_sha256 != _publication_inventory_sha256(self.files):
            _fail("publication file inventory digest does not replay")
        if expected_family == "external":
            video = self.files[
                EXTERNAL_PUBLICATION_ROLE_PATHS.index(
                    ("upstream_video_slot", "upstream-video-slot.bin")
                )
            ]
            if self.candidate_id in MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS:
                if self.video_slot_mode != "opaque_ppo_video" or video.size_bytes < 1:
                    _fail("PPO publication requires one nonempty opaque video slot")
            elif (
                self.video_slot_mode != "absent_for_continuing_zero_length_slot"
                or video.size_bytes != 0
                or video.sha256 != EMPTY_FILE_SHA256
            ):
                _fail("continuing publication requires the exact empty video sentinel")
        elif self.video_slot_mode != "not_applicable":
            _fail("non-external publication cannot carry an external video-slot mode")
        if _require_bool(self.publication_committed, "publication committed") is not True:
            _fail("success publication must be committed")
        if (
            _require_bool(self.value_decoding_performed, "value decoding") is not False
            or _require_bool(self.byte_transport_performed, "byte transport") is not False
        ):
            _fail("publication candidate cannot decode values or transport file bytes")
        if _require_int(self.retry_count, "publication retry count", maximum=0) != 0:
            _fail("publication candidate cannot record a retry")
        wrapper_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "publisher": self.publisher,
            "publisher_metadata": self.publisher_metadata,
            "native_atomic_producer": self.native_atomic_producer,
            "native_publication_receipt": self.native_publication_receipt,
            "publication_address_sha256": self.publication_address_sha256,
            "publication_manifest_file_sha256": self.publication_manifest_file_sha256,
            "publication_manifest_body_sha256": self.publication_manifest_body_sha256,
            "file_inventory_sha256": self.file_inventory_sha256,
            "published_bundle_sha256": self.published_bundle_sha256,
            "expected_reload_observation_sha256": (self.wrapper_expected_reload_observation_sha256),
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "maximum_total_size_bytes": self.maximum_total_size_bytes,
            "video_slot_mode": self.video_slot_mode,
            "files": self.files,
        }
        _require_canonical_artifact_identity(
            self.publication_commitment_wrapper,
            schema_version=NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            body_bytes=canonical_normalized_publication_commitment_wrapper_v1_body_bytes(
                **wrapper_facts
            ),
            file_bytes=canonical_normalized_publication_commitment_wrapper_v1_file_bytes(
                **wrapper_facts
            ),
            label="normalized publication commitment wrapper",
        )
        _require_publication_reload_validation_v1_identity(
            self.publication_reload_validation,
            publication_commitment_wrapper_file_sha256=(self.reload_validation_wrapper_file_sha256),
            publication_commitment_wrapper_body_sha256=(self.reload_validation_wrapper_body_sha256),
            expected_reload_observation_sha256=(
                self.reload_validation_expected_reload_observation_sha256
            ),
            actual_reload_observation_sha256=(
                self.reload_validation_actual_reload_observation_sha256
            ),
            reload_performed=self.reload_validation_reload_performed,
            reload_read_only=self.reload_validation_read_only,
            label="publication reload validation",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "case_ordinal": self.case_ordinal,
            "qualification_case_id": self.qualification_case_id,
            "case_spine_sha256": self.case_spine_sha256,
            "host_execution_receipt_file_sha256": (self.host_execution_receipt_file_sha256),
            "host_execution_receipt_body_sha256": (self.host_execution_receipt_body_sha256),
            "host_terminal_metadata_file_sha256": (self.host_terminal_metadata_file_sha256),
            "host_terminal_metadata_body_sha256": (self.host_terminal_metadata_body_sha256),
            "host_observation_handoff": self.host_observation_handoff.to_dict(),
            "publisher_registry_entry_file_sha256": (self.publisher_registry_entry_file_sha256),
            "publisher_registry_entry_body_sha256": (self.publisher_registry_entry_body_sha256),
            "publisher_metadata": self.publisher_metadata.to_dict(),
            "runner_execution_receipt": self.runner_execution_receipt.to_dict(),
            "publisher": self.publisher.to_dict(),
            "native_atomic_producer": self.native_atomic_producer.to_dict(),
            "native_publication_receipt": (
                None
                if self.native_publication_receipt is None
                else self.native_publication_receipt.to_dict()
            ),
            "publication_commitment_contract_descriptor_sha256": (
                self.publication_commitment_contract_descriptor_sha256
            ),
            "publication_commitment_contract_source_sha256": (
                self.publication_commitment_contract_source_sha256
            ),
            "publication_commitment_wrapper": (self.publication_commitment_wrapper.to_dict()),
            "wrapper_case_spine_sha256": self.wrapper_case_spine_sha256,
            "wrapper_candidate_id": self.wrapper_candidate_id,
            "wrapper_candidate_family": self.wrapper_candidate_family,
            "wrapper_publisher_descriptor_sha256": (self.wrapper_publisher_descriptor_sha256),
            "wrapper_publisher_source_sha256": self.wrapper_publisher_source_sha256,
            "wrapper_publisher_metadata_file_sha256": (self.wrapper_publisher_metadata_file_sha256),
            "wrapper_publisher_metadata_body_sha256": (self.wrapper_publisher_metadata_body_sha256),
            "wrapper_native_atomic_producer_descriptor_sha256": (
                self.wrapper_native_atomic_producer_descriptor_sha256
            ),
            "wrapper_native_atomic_producer_source_sha256": (
                self.wrapper_native_atomic_producer_source_sha256
            ),
            "wrapper_native_publication_receipt_file_sha256": (
                self.wrapper_native_publication_receipt_file_sha256
            ),
            "wrapper_native_publication_receipt_body_sha256": (
                self.wrapper_native_publication_receipt_body_sha256
            ),
            "wrapper_publication_address_sha256": (self.wrapper_publication_address_sha256),
            "wrapper_file_inventory_sha256": self.wrapper_file_inventory_sha256,
            "wrapper_published_bundle_sha256": self.wrapper_published_bundle_sha256,
            "wrapper_expected_reload_observation_sha256": (
                self.wrapper_expected_reload_observation_sha256
            ),
            "wrapper_video_slot_mode": self.wrapper_video_slot_mode,
            "publication_address_sha256": self.publication_address_sha256,
            "publication_manifest_file_sha256": self.publication_manifest_file_sha256,
            "publication_manifest_body_sha256": self.publication_manifest_body_sha256,
            "published_bundle_sha256": self.published_bundle_sha256,
            "reload_observation_sha256": self.reload_observation_sha256,
            "publication_reload_validation": (self.publication_reload_validation.to_dict()),
            "reload_validation_wrapper_file_sha256": (self.reload_validation_wrapper_file_sha256),
            "reload_validation_wrapper_body_sha256": (self.reload_validation_wrapper_body_sha256),
            "reload_validation_expected_reload_observation_sha256": (
                self.reload_validation_expected_reload_observation_sha256
            ),
            "reload_validation_actual_reload_observation_sha256": (
                self.reload_validation_actual_reload_observation_sha256
            ),
            "reload_validation_reload_performed": (self.reload_validation_reload_performed),
            "reload_validation_read_only": self.reload_validation_read_only,
            "file_inventory_sha256": self.file_inventory_sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "maximum_total_size_bytes": self.maximum_total_size_bytes,
            "video_slot_mode": self.video_slot_mode,
            "files": [item.to_dict() for item in self.files],
            "publication_committed": self.publication_committed,
            "value_decoding_performed": self.value_decoding_performed,
            "byte_transport_performed": self.byte_transport_performed,
            "retry_count": self.retry_count,
        }


@dataclass(frozen=True, slots=True)
class FailurePublicationProjectionV2:
    """Canonical committed-wrapper facts retained by a later host failure."""

    schema_version: str
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    publisher_registry_entry_file_sha256: str
    publisher_registry_entry_body_sha256: str
    algorithmic_resource_receipt: ArtifactIdentityV2
    runner_execution_receipt: ArtifactIdentityV2
    algorithmic_receipt_runner_file_sha256: str
    algorithmic_receipt_runner_body_sha256: str
    publisher: ProducerIdentityV2
    publisher_metadata: ArtifactIdentityV2
    native_atomic_producer: ProducerIdentityV2
    native_publication_receipt: ArtifactIdentityV2 | None
    publication_reconciliation_key_sha256: str
    publication_reconciliation_reference: ArtifactIdentityV2
    publication_commitment_contract_descriptor_sha256: str
    publication_commitment_contract_source_sha256: str
    publication_commitment_wrapper: ArtifactIdentityV2
    publication_address_sha256: str
    publication_manifest_file_sha256: str
    publication_manifest_body_sha256: str
    file_inventory_sha256: str
    published_bundle_sha256: str
    expected_reload_observation_sha256: str
    file_count: int
    total_size_bytes: int
    maximum_total_size_bytes: int
    video_slot_mode: Literal[
        "not_applicable",
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    ]
    files: tuple[PublicationFileCandidateV2, ...]

    def __post_init__(self) -> None:
        _require_exact_literal(
            self.schema_version,
            QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION,
            "failure publication projection schema",
        )
        _require_case_projection(
            self.case_ordinal,
            self.candidate_id,
            self.qualification_case_id,
            "failure publication projection",
        )
        family = _family_for_candidate(self.candidate_id)
        _require_exact_literal(
            self.candidate_family,
            family,
            "failure publication projection family",
        )
        _require_one_of(
            self.video_slot_mode,
            (
                "not_applicable",
                "absent_for_continuing_zero_length_slot",
                "opaque_ppo_video",
            ),
            "failure publication projection video-slot mode",
        )
        expected_paths, metadata_schema, publisher_schema = _publication_profile(family)
        for digest, label in (
            (self.case_spine_sha256, "failure publication case spine"),
            (self.publisher_registry_entry_file_sha256, "failure publisher entry file"),
            (self.publisher_registry_entry_body_sha256, "failure publisher entry body"),
            (self.algorithmic_receipt_runner_file_sha256, "failure algorithmic runner file"),
            (self.algorithmic_receipt_runner_body_sha256, "failure algorithmic runner body"),
            (self.publication_reconciliation_key_sha256, "failure reconciliation key"),
            (
                self.publication_commitment_contract_descriptor_sha256,
                "failure wrapper contract descriptor",
            ),
            (
                self.publication_commitment_contract_source_sha256,
                "failure wrapper contract source",
            ),
            (self.publication_address_sha256, "failure publication address"),
            (self.publication_manifest_file_sha256, "failure manifest file"),
            (self.publication_manifest_body_sha256, "failure manifest body"),
            (self.file_inventory_sha256, "failure file inventory"),
            (self.published_bundle_sha256, "failure published bundle"),
            (self.expected_reload_observation_sha256, "failure expected reload"),
        ):
            _require_sha256(digest, label)
        if (
            self.publication_commitment_contract_descriptor_sha256
            != FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256
            or self.publication_commitment_contract_source_sha256
            != FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256
        ):
            _fail("failure normalized publication implementation identity differs")
        _require_artifact_schema(
            self.algorithmic_resource_receipt,
            _algorithmic_resource_receipt_schema(family),
            "failure projection algorithmic receipt",
        )
        _require_artifact_schema(
            self.runner_execution_receipt,
            _runner_execution_receipt_schema(self.candidate_id),
            "failure projection runner receipt",
        )
        if (
            self.algorithmic_receipt_runner_file_sha256 != self.runner_execution_receipt.file_sha256
            or self.algorithmic_receipt_runner_body_sha256
            != self.runner_execution_receipt.body_sha256
        ):
            _fail("failure algorithmic receipt runner projection is cross-wired")
        _require_artifact_schema(
            self.publisher_metadata,
            metadata_schema,
            "failure projection publisher metadata",
        )
        _require_artifact_schema(
            self.publication_reconciliation_reference,
            QUALIFICATION_PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION,
            "failure publication reconciliation reference",
        )
        _require_artifact_schema(
            self.publication_commitment_wrapper,
            NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            "failure normalized publication wrapper",
        )
        if (
            type(self.publisher) is not ProducerIdentityV2
            or self.publisher.descriptor_schema_version != publisher_schema
            or type(self.native_atomic_producer) is not ProducerIdentityV2
        ):
            _fail("failure publication producer identities differ")
        expected_atomic_schema = (
            STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            if family == "adapter"
            else ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        )
        if self.native_atomic_producer.descriptor_schema_version != expected_atomic_schema:
            _fail("failure native atomic producer schema differs")
        if family == "adapter" and any(
            digest in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
            for digest in (
                self.publisher.descriptor_sha256,
                self.publisher.source_sha256,
                self.native_atomic_producer.descriptor_sha256,
                self.native_atomic_producer.source_sha256,
            )
        ):
            _fail("unqualified adapter implementation cannot fill failure publication slots")
        if family == "local":
            if self.native_publication_receipt is not None:
                _fail("local failure publication cannot carry a native receipt")
        else:
            native_schema = (
                EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
                if family == "external"
                else STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
            )
            if self.native_publication_receipt is None:
                _fail("nonlocal failure publication lacks its native receipt")
            _require_artifact_schema(
                self.native_publication_receipt,
                native_schema,
                "failure native publication receipt",
            )
        if type(self.files) is not tuple or any(
            type(item) is not PublicationFileCandidateV2 for item in self.files
        ):
            _fail("failure publication files must be one exact tuple")
        if tuple((item.role, item.name) for item in self.files) != expected_paths:
            _fail("failure publication role/name inventory differs")
        if len({item.role for item in self.files}) != len(self.files) or len(
            {item.name for item in self.files}
        ) != len(self.files):
            _fail("failure publication role or filename is duplicated")
        if any(
            item.size_bytes == 0 and item.role not in {"stdout", "stderr", "upstream_video_slot"}
            for item in self.files
        ):
            _fail("failure publication required manifest, receipt, or data file is empty")
        runner_file = next(
            item for item in self.files if item.role == _runner_execution_receipt_role(family)
        )
        if runner_file.sha256 != self.runner_execution_receipt.file_sha256:
            _fail("failure publication runner inventory link differs")
        if self.publication_address_sha256 != self.files[0].sha256:
            _fail("failure publication address differs from its manifest file")
        if self.publication_manifest_file_sha256 != self.files[0].sha256:
            _fail("failure publication manifest identity differs from its inventory")
        if _require_int(self.file_count, "failure publication file count", minimum=1) != len(
            expected_paths
        ) or self.file_count != len(self.files):
            _fail("failure publication file count differs")
        total = sum(item.size_bytes for item in self.files)
        if (
            _require_int(
                self.total_size_bytes,
                "failure publication aggregate size",
                maximum=MAX_PUBLICATION_TOTAL_BYTES,
            )
            != total
            or _require_int(
                self.maximum_total_size_bytes,
                "failure publication aggregate ceiling",
                minimum=MAX_PUBLICATION_TOTAL_BYTES,
                maximum=MAX_PUBLICATION_TOTAL_BYTES,
            )
            != MAX_PUBLICATION_TOTAL_BYTES
            or total > self.maximum_total_size_bytes
        ):
            _fail("failure publication aggregate size differs")
        if self.file_inventory_sha256 != _publication_inventory_sha256(self.files):
            _fail("failure publication inventory digest differs")
        if family == "external":
            video = self.files[
                EXTERNAL_PUBLICATION_ROLE_PATHS.index(
                    ("upstream_video_slot", "upstream-video-slot.bin")
                )
            ]
            if self.candidate_id in MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS:
                if self.video_slot_mode != "opaque_ppo_video" or video.size_bytes < 1:
                    _fail("failure PPO publication lacks its opaque video slot")
            elif (
                self.video_slot_mode != "absent_for_continuing_zero_length_slot"
                or video.size_bytes != 0
                or video.sha256 != EMPTY_FILE_SHA256
            ):
                _fail("failure continuing publication lacks its empty video sentinel")
        elif self.video_slot_mode != "not_applicable":
            _fail("failure non-external publication carries an external video mode")
        wrapper_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "publisher": self.publisher,
            "publisher_metadata": self.publisher_metadata,
            "native_atomic_producer": self.native_atomic_producer,
            "native_publication_receipt": self.native_publication_receipt,
            "publication_address_sha256": self.publication_address_sha256,
            "publication_manifest_file_sha256": self.publication_manifest_file_sha256,
            "publication_manifest_body_sha256": self.publication_manifest_body_sha256,
            "file_inventory_sha256": self.file_inventory_sha256,
            "published_bundle_sha256": self.published_bundle_sha256,
            "expected_reload_observation_sha256": self.expected_reload_observation_sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "maximum_total_size_bytes": self.maximum_total_size_bytes,
            "video_slot_mode": self.video_slot_mode,
            "files": self.files,
        }
        _require_canonical_artifact_identity(
            self.publication_commitment_wrapper,
            schema_version=NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            body_bytes=canonical_normalized_publication_commitment_wrapper_v1_body_bytes(
                **wrapper_facts
            ),
            file_bytes=canonical_normalized_publication_commitment_wrapper_v1_file_bytes(
                **wrapper_facts
            ),
            label="failure normalized publication wrapper",
        )

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "publisher_registry_entry_file_sha256": (self.publisher_registry_entry_file_sha256),
            "publisher_registry_entry_body_sha256": (self.publisher_registry_entry_body_sha256),
            "algorithmic_resource_receipt": self.algorithmic_resource_receipt.to_dict(),
            "runner_execution_receipt": self.runner_execution_receipt.to_dict(),
            "algorithmic_receipt_runner_file_sha256": (self.algorithmic_receipt_runner_file_sha256),
            "algorithmic_receipt_runner_body_sha256": (self.algorithmic_receipt_runner_body_sha256),
            "publisher": self.publisher.to_dict(),
            "publisher_metadata": self.publisher_metadata.to_dict(),
            "native_atomic_producer": self.native_atomic_producer.to_dict(),
            "native_publication_receipt": (
                None
                if self.native_publication_receipt is None
                else self.native_publication_receipt.to_dict()
            ),
            "publication_reconciliation_key_sha256": (self.publication_reconciliation_key_sha256),
            "publication_reconciliation_reference": (
                self.publication_reconciliation_reference.to_dict()
            ),
            "publication_commitment_contract_descriptor_sha256": (
                self.publication_commitment_contract_descriptor_sha256
            ),
            "publication_commitment_contract_source_sha256": (
                self.publication_commitment_contract_source_sha256
            ),
            "publication_commitment_wrapper": self.publication_commitment_wrapper.to_dict(),
            "publication_address_sha256": self.publication_address_sha256,
            "publication_manifest_file_sha256": self.publication_manifest_file_sha256,
            "publication_manifest_body_sha256": self.publication_manifest_body_sha256,
            "file_inventory_sha256": self.file_inventory_sha256,
            "published_bundle_sha256": self.published_bundle_sha256,
            "expected_reload_observation_sha256": self.expected_reload_observation_sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "maximum_total_size_bytes": self.maximum_total_size_bytes,
            "video_slot_mode": self.video_slot_mode,
            "files": [item.to_dict() for item in self.files],
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(
            self.to_body_dict(),
            QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD,
        )


@dataclass(frozen=True, slots=True)
class ResourceFieldCandidateV2:
    """One ordered ceiling/observation/provenance record."""

    field_name: str
    declared_ceiling: int
    observed_value: int
    value_semantics: Literal[
        "exact_observation",
        "conservative_observed_upper_bound",
        "conservative_enforced_upper_bound",
    ]
    provenance_kind: str
    provenance_receipt: ArtifactIdentityV2

    def __post_init__(self) -> None:
        field_name = _require_one_of(
            self.field_name,
            RESOURCE_CEILING_FIELDS,
            "resource field name",
        )
        _require_int(self.declared_ceiling, f"resource ceiling {self.field_name}")
        _require_int(self.observed_value, f"resource observation {self.field_name}")
        allowed_semantics: tuple[str, ...]
        if field_name == "max_peak_rss_bytes":
            allowed_semantics = ("conservative_observed_upper_bound",)
        elif field_name in {"max_temporary_peak_bytes", "max_disk_peak_bytes"}:
            allowed_semantics = (
                "exact_observation",
                "conservative_enforced_upper_bound",
            )
        elif field_name == "max_thread_count":
            allowed_semantics = ("conservative_observed_upper_bound",)
        else:
            allowed_semantics = ("exact_observation",)
        _require_one_of(
            self.value_semantics,
            allowed_semantics,
            f"resource value semantics for {field_name}",
        )
        _require_identifier(self.provenance_kind, "resource provenance kind")
        if type(self.provenance_receipt) is not ArtifactIdentityV2:
            _fail("resource provenance receipt type differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "declared_ceiling": self.declared_ceiling,
            "observed_value": self.observed_value,
            "value_semantics": self.value_semantics,
            "provenance_kind": self.provenance_kind,
            "provenance_receipt": self.provenance_receipt.to_dict(),
        }


def _resource_field_inventory_sha256(
    fields: tuple[ResourceFieldCandidateV2, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {"fields": [item.to_dict() for item in fields]},
            newline=False,
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ResourceMergerCandidateV2:
    """Complete ordered merger output candidate; no ceiling decision is made."""

    schema_version: str
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    case_spine_sha256: str
    host_execution_receipt_file_sha256: str
    host_execution_receipt_body_sha256: str
    host_provisioning_receipt: ArtifactIdentityV2
    host_cgroup_proof: ArtifactIdentityV2
    host_terminal_metadata: ArtifactIdentityV2
    host_observation_handoff: ArtifactIdentityV2
    endpoint_corroboration_mode: Literal[
        "absent_unavailable_nonblocking",
        "present_redundant_non_authoritative",
    ]
    endpoint_observer_request: ArtifactIdentityV2 | None
    endpoint_observer_receipt: ArtifactIdentityV2 | None
    algorithmic_resource_contract: ProducerIdentityV2
    algorithmic_measurement_intent: ArtifactIdentityV2
    algorithmic_resource_receipt: ArtifactIdentityV2
    runner_execution_receipt: ArtifactIdentityV2
    storage_boundary_contract: ProducerIdentityV2
    storage_boundary_intent: ArtifactIdentityV2
    storage_write_seal: ArtifactIdentityV2
    storage_boundary_receipt: ArtifactIdentityV2
    merger_receipt: ArtifactIdentityV2
    merger: ProducerIdentityV2
    resource_requirement_body_sha256: str
    field_inventory_sha256: str
    fields: tuple[ResourceFieldCandidateV2, ...]

    def __post_init__(self) -> None:
        _require_exact_literal(
            self.schema_version,
            QUALIFICATION_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION,
            "resource-merger candidate schema",
        )
        _require_exact_literal(
            self.candidate_family,
            _family_for_candidate(self.candidate_id),
            "resource-merger candidate family",
        )
        for value, label in (
            (self.case_spine_sha256, "resource case spine"),
            (self.host_execution_receipt_file_sha256, "resource host receipt file"),
            (self.host_execution_receipt_body_sha256, "resource host receipt body"),
            (self.resource_requirement_body_sha256, "resource requirement body"),
            (self.field_inventory_sha256, "resource field inventory"),
        ):
            _require_sha256(value, label)
        for producer, schema, descriptor_sha256, source_sha256, label in (
            (
                self.algorithmic_resource_contract,
                ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
                "resource-merger algorithmic contract",
            ),
            (
                self.storage_boundary_contract,
                QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
                "resource-merger storage contract",
            ),
        ):
            if (
                type(producer) is not ProducerIdentityV2
                or producer.descriptor_schema_version != schema
                or producer.descriptor_sha256 != descriptor_sha256
                or producer.source_sha256 != source_sha256
            ):
                _fail(f"{label} finalized implementation identity differs")
        _require_artifact_schema(
            self.host_provisioning_receipt,
            HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host provisioning receipt",
        )
        _require_artifact_schema(
            self.host_cgroup_proof,
            HOST_CGROUP_PROOF_SCHEMA_VERSION,
            "host cgroup proof",
        )
        _require_artifact_schema(
            self.host_terminal_metadata,
            HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            "host terminal metadata",
        )
        _require_artifact_schema(
            self.host_observation_handoff,
            HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
            "resource merger host observation handoff",
        )
        endpoint_present = (
            self.endpoint_observer_request is not None
            and self.endpoint_observer_receipt is not None
        )
        if (self.endpoint_observer_request is None) != (self.endpoint_observer_receipt is None):
            _fail("endpoint corroboration request and receipt presence differ")
        expected_endpoint_mode = (
            "present_redundant_non_authoritative"
            if endpoint_present
            else "absent_unavailable_nonblocking"
        )
        _require_exact_literal(
            self.endpoint_corroboration_mode,
            expected_endpoint_mode,
            "endpoint corroboration mode",
        )
        if self.endpoint_observer_request is not None:
            _require_artifact_schema(
                self.endpoint_observer_request,
                ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION,
                "endpoint observer request",
            )
        if self.endpoint_observer_receipt is not None:
            _require_artifact_schema(
                self.endpoint_observer_receipt,
                ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION,
                "endpoint observer receipt",
            )
        _require_artifact_schema(
            self.algorithmic_measurement_intent,
            ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "pre-GO algorithmic resource measurement intent",
        )
        _require_artifact_schema(
            self.algorithmic_resource_receipt,
            _algorithmic_resource_receipt_schema(self.candidate_family),
            "family algorithmic resource receipt",
        )
        _require_artifact_schema(
            self.runner_execution_receipt,
            _runner_execution_receipt_schema(self.candidate_id),
            "merger runner-execution receipt",
        )
        _require_artifact_schema(
            self.storage_boundary_intent,
            QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "case-bound storage boundary intent",
        )
        _require_artifact_schema(
            self.storage_write_seal,
            QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
            "case-bound storage write seal",
        )
        _require_artifact_schema(
            self.storage_boundary_receipt,
            QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
            "case-bound storage boundary receipt",
        )
        _require_artifact_schema(
            self.merger_receipt,
            FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION,
            "full resource merger receipt",
        )
        if type(self.merger) is not ProducerIdentityV2:
            _fail("full resource merger producer identity type differs")
        if self.merger.descriptor_schema_version != FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION:
            _fail("full resource merger producer descriptor schema differs")
        if self.merger.descriptor_sha256 in {
            PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256,
            PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
        } or self.merger.source_sha256 in {
            PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256,
            PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
        }:
            _fail("endpoint observer cannot substitute for the full resource merger")
        if (
            self.merger.descriptor_sha256 in ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S
            or self.merger.source_sha256 in ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S
        ):
            _fail("algorithmic validator cannot fill the full-merger slot")
        if self.merger.descriptor_sha256 in {
            FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
            FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
        } or self.merger.source_sha256 in {
            FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
            FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
        }:
            _fail("storage validator cannot fill the full-merger slot")
        if (
            type(self.fields) is not tuple
            or any(type(item) is not ResourceFieldCandidateV2 for item in self.fields)
            or tuple(item.field_name for item in self.fields) != RESOURCE_CEILING_FIELDS
        ):
            _fail("resource records must use the exact 28-field order")
        host_execution_receipt = ArtifactIdentityV2(
            schema_version=HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
            file_sha256=self.host_execution_receipt_file_sha256,
            body_sha256=self.host_execution_receipt_body_sha256,
        )
        for item, expected_kind in zip(
            self.fields,
            RESOURCE_PROVENANCE_KINDS,
            strict=True,
        ):
            if item.provenance_kind != expected_kind:
                _fail(f"resource provenance kind differs for {item.field_name}")
            expected_receipt = (
                self.storage_boundary_receipt
                if expected_kind == "host_storage_boundary_receipt"
                else self.algorithmic_resource_receipt
                if expected_kind == "algorithmic_resource_receipt"
                else host_execution_receipt
                if expected_kind == "host_execution_lifecycle"
                else self.host_cgroup_proof
            )
            if item.provenance_receipt != expected_receipt:
                _fail(f"resource provenance receipt differs for {item.field_name}")
        by_name = {item.field_name: item for item in self.fields}
        horizon = by_name["max_environment_interactions"]
        attempts = by_name["max_attempt_count"]
        failures = by_name["max_failure_count"]
        if (
            horizon.declared_ceiling != MATCHED_V3_HORIZON
            or horizon.observed_value != MATCHED_V3_HORIZON
        ):
            _fail("resource horizon ceiling and observation must both be exact 499712")
        if attempts.declared_ceiling != 1 or attempts.observed_value != 1:
            _fail("resource attempt ceiling and observation must both be one")
        if failures.declared_ceiling != 0 or failures.observed_value != 0:
            _fail("resource failure ceiling and observation must both be zero")
        if self.field_inventory_sha256 != _resource_field_inventory_sha256(self.fields):
            _fail("resource field inventory digest does not replay")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "case_spine_sha256": self.case_spine_sha256,
            "host_execution_receipt_file_sha256": (self.host_execution_receipt_file_sha256),
            "host_execution_receipt_body_sha256": (self.host_execution_receipt_body_sha256),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "host_cgroup_proof": self.host_cgroup_proof.to_dict(),
            "host_terminal_metadata": self.host_terminal_metadata.to_dict(),
            "host_observation_handoff": self.host_observation_handoff.to_dict(),
            "endpoint_corroboration_mode": self.endpoint_corroboration_mode,
            "endpoint_observer_request": (
                None
                if self.endpoint_observer_request is None
                else self.endpoint_observer_request.to_dict()
            ),
            "endpoint_observer_receipt": (
                None
                if self.endpoint_observer_receipt is None
                else self.endpoint_observer_receipt.to_dict()
            ),
            "algorithmic_resource_contract": self.algorithmic_resource_contract.to_dict(),
            "algorithmic_measurement_intent": (self.algorithmic_measurement_intent.to_dict()),
            "algorithmic_resource_receipt": (self.algorithmic_resource_receipt.to_dict()),
            "runner_execution_receipt": self.runner_execution_receipt.to_dict(),
            "storage_boundary_contract": self.storage_boundary_contract.to_dict(),
            "storage_boundary_intent": self.storage_boundary_intent.to_dict(),
            "storage_write_seal": self.storage_write_seal.to_dict(),
            "storage_boundary_receipt": self.storage_boundary_receipt.to_dict(),
            "merger_receipt": self.merger_receipt.to_dict(),
            "merger": self.merger.to_dict(),
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "field_inventory_sha256": self.field_inventory_sha256,
            "fields": [item.to_dict() for item in self.fields],
        }


def _host_contract_claims() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "execution_performed": False,
        "qualification_granted": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
    }


def _host_contract_authority() -> dict[str, bool]:
    return {
        "issuer_available": False,
        "evaluator_available": False,
        "merger_available": False,
        "production_backend_available": False,
    }


def _derived_operational_boundary_state(
    completed_phases: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
) -> str:
    if phase in completed_phases:
        return "committed"
    if failure_phase == phase and failure_effect_state == "commit_uncertain":
        return "commit_uncertain"
    return "not_started"


def _derived_operational_boundary_count(
    completed_phases: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
) -> tuple[str, int | None]:
    state = _derived_operational_boundary_state(
        completed_phases,
        failure_phase,
        failure_effect_state,
        phase,
    )
    if state == "committed":
        return "exact", 1
    if state == "commit_uncertain":
        return "uncertain", None
    return "exact", 0


def host_operational_frontier_v2_body_projection(
    *,
    case_spine_sha256: str,
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
    case_consumed: bool,
    same_case_retry_permitted: bool,
) -> dict[str, Any]:
    _require_sha256(case_spine_sha256, "operational frontier case spine")
    if type(completed_phases) is not tuple or any(
        type(item) is not str for item in completed_phases
    ):
        _fail("operational frontier phases must be one exact string tuple")
    if len(completed_phases) > len(HOST_OPERATIONAL_PHASES):
        _fail("operational frontier phase prefix is too long")
    _require_exact_string_tuple(
        completed_phases,
        HOST_OPERATIONAL_PHASES[: len(completed_phases)],
        "operational frontier phases",
    )
    if failure_phase is None:
        if failure_effect_state is not None or completed_phases != HOST_OPERATIONAL_PHASES:
            _fail("nonfailure operational frontier must be complete")
    else:
        if len(completed_phases) >= len(HOST_OPERATIONAL_PHASES):
            _fail("operational failure cannot follow a complete frontier")
        _require_exact_literal(
            failure_phase,
            HOST_OPERATIONAL_PHASES[len(completed_phases)],
            "operational failure phase",
        )
        _require_one_of(
            failure_effect_state,
            HOST_OPERATIONAL_FAILURE_EFFECT_STATES,
            "operational failure effect",
        )
    expected_states = {
        "container_create_state": _derived_operational_boundary_state(
            completed_phases, failure_phase, failure_effect_state, "container_created"
        ),
        "container_start_state": _derived_operational_boundary_state(
            completed_phases, failure_phase, failure_effect_state, "container_started"
        ),
        "workload_start_state": _derived_operational_boundary_state(
            completed_phases, failure_phase, failure_effect_state, "workload_started"
        ),
        "workload_exit_state": _derived_operational_boundary_state(
            completed_phases, failure_phase, failure_effect_state, "workload_exited"
        ),
    }
    actual_states = {
        "container_create_state": container_create_state,
        "container_start_state": container_start_state,
        "workload_start_state": workload_start_state,
        "workload_exit_state": workload_exit_state,
    }
    for field_name, expected in expected_states.items():
        _require_exact_literal(actual_states[field_name], expected, f"operational {field_name}")
    expected_counts = {
        "container_create": _derived_operational_boundary_count(
            completed_phases, failure_phase, failure_effect_state, "container_created"
        ),
        "container_start": _derived_operational_boundary_count(
            completed_phases, failure_phase, failure_effect_state, "container_started"
        ),
        "workload_start": _derived_operational_boundary_count(
            completed_phases, failure_phase, failure_effect_state, "workload_started"
        ),
        "workload_exit": _derived_operational_boundary_count(
            completed_phases, failure_phase, failure_effect_state, "workload_exited"
        ),
    }
    actual_counts = {
        "container_create": (container_create_count_state, container_create_count),
        "container_start": (container_start_count_state, container_start_count),
        "workload_start": (workload_start_count_state, workload_start_count),
        "workload_exit": (workload_exit_count_state, workload_exit_count),
    }
    for field_name, (expected_state, expected_count) in expected_counts.items():
        actual_state, actual_count = actual_counts[field_name]
        _require_exact_literal(
            actual_state,
            expected_state,
            f"operational {field_name} count state",
        )
        if actual_count is not None:
            _require_int(actual_count, f"operational {field_name} count", maximum=1)
        if actual_count != expected_count:
            _fail(f"operational {field_name} count differs")
    _require_exact_literal(
        attempt_count_state,
        workload_start_count_state,
        "operational attempt count state",
    )
    if attempt_count is not None:
        _require_int(attempt_count, "operational attempt count", maximum=1)
    if attempt_count != workload_start_count:
        _fail("operational attempt count must mirror workload start")
    expected_failure_count = 0 if failure_phase is None else 1
    if _require_int(failure_count, "operational failure count", maximum=1) != (
        expected_failure_count
    ):
        _fail("operational failure count differs")
    if _require_bool(case_consumed, "operational case consumed") is not True:
        _fail("operational frontier must consume the single-use case")
    if _require_bool(same_case_retry_permitted, "operational retry") is not False:
        _fail("operational frontier cannot permit same-case retry")
    return {
        "schema_version": HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
        "status": "operational_frontier_recorded_nonretryable_non_authorizing",
        "case_spine_sha256": case_spine_sha256,
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
        "case_consumed": case_consumed,
        "same_case_retry_permitted": same_case_retry_permitted,
        "claims": _host_contract_claims(),
    }


HOST_OPERATIONAL_FRONTIER_BODY_SHA256_FIELD: Final = "operational_frontier_body_sha256"


def canonical_host_operational_frontier_v2_body_bytes(**facts: Any) -> bytes:
    return _canonical_json(host_operational_frontier_v2_body_projection(**facts), newline=False)


def canonical_host_operational_frontier_v2_file_bytes(**facts: Any) -> bytes:
    body = host_operational_frontier_v2_body_projection(**facts)
    return _canonical_json(
        {**body, HOST_OPERATIONAL_FRONTIER_BODY_SHA256_FIELD: _body_sha256(body)}
    )


@dataclass(frozen=True, slots=True)
class RecoveryNodeCandidateV2:
    """One ordered conditional cleanup-DAG outcome."""

    node_name: str
    state: Literal[
        "not_applicable",
        "committed",
        "commit_uncertain",
        "failed_before_commit",
    ]
    artifact: ArtifactIdentityV2 | None
    dependencies: tuple[str, ...]
    uncertainty_detail_sha256: str | None

    def __post_init__(self) -> None:
        name = _require_one_of(
            self.node_name,
            HOST_RECOVERY_NODE_NAMES,
            "recovery node name",
        )
        state = _require_one_of(
            self.state,
            HOST_RECOVERY_NODE_STATES,
            "recovery node state",
        )
        _require_exact_string_tuple(
            self.dependencies,
            HOST_RECOVERY_NODE_DEPENDENCIES[name],
            f"recovery node {name} dependencies",
        )
        if state == "committed":
            if self.artifact is None:
                _fail("committed recovery node lacks its artifact")
            _require_artifact_schema(
                self.artifact,
                HOST_RECOVERY_NODE_SCHEMAS[name],
                f"recovery node {name}",
            )
            if self.uncertainty_detail_sha256 is not None:
                _fail("committed recovery node cannot carry uncertainty detail")
        else:
            if self.artifact is not None:
                _fail("uncommitted recovery node cannot carry a committed artifact")
            if state in {"commit_uncertain", "failed_before_commit"}:
                _require_sha256(
                    self.uncertainty_detail_sha256,
                    f"recovery node {name} uncertainty detail",
                )
            elif self.uncertainty_detail_sha256 is not None:
                _fail("inapplicable recovery node cannot carry uncertainty detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "state": self.state,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "dependencies": list(self.dependencies),
            "uncertainty_detail_sha256": self.uncertainty_detail_sha256,
        }


HOST_CLEANUP_RECONCILIATION_BODY_SHA256_FIELD: Final = "cleanup_reconciliation_body_sha256"


def host_cleanup_reconciliation_v2_body_projection(
    *,
    case_spine_sha256: str,
    operational_frontier: ArtifactIdentityV2,
    cgroup_may_exist: bool,
    recovery_nodes: tuple[RecoveryNodeCandidateV2, ...],
    cleanup_proven: bool,
    unresolved_recovery_nodes: tuple[str, ...],
    recovery_complete: bool,
    terminalization_permitted: bool,
    workload_resume_permitted: bool,
    same_case_retry_permitted: bool,
) -> dict[str, Any]:
    _require_sha256(case_spine_sha256, "cleanup reconciliation case spine")
    _require_artifact_schema(
        operational_frontier,
        HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
        "cleanup reconciliation operational frontier",
    )
    may_exist = _require_bool(cgroup_may_exist, "cleanup cgroup may exist")
    if (
        type(recovery_nodes) is not tuple
        or any(type(item) is not RecoveryNodeCandidateV2 for item in recovery_nodes)
        or tuple(item.node_name for item in recovery_nodes) != HOST_RECOVERY_NODE_NAMES
    ):
        _fail("cleanup recovery-node inventory or order differs")
    states = {item.node_name: item.state for item in recovery_nodes}
    if may_exist:
        if any(state == "not_applicable" for state in states.values()):
            _fail("possibly-created cgroup requires every cleanup node outcome")
    elif any(state != "not_applicable" for state in states.values()):
        _fail("cleanup nodes must be inapplicable when the cgroup cannot exist")
    for node in recovery_nodes:
        if node.state == "committed" and any(
            states[dependency] != "committed" for dependency in node.dependencies
        ):
            _fail("committed cleanup node lacks a committed dependency")
    expected_cleanup_proven = not may_exist or all(
        state == "committed" for state in states.values()
    )
    if _require_bool(cleanup_proven, "cleanup proven") is not expected_cleanup_proven:
        _fail("cleanup-proven projection differs")
    expected_unresolved = tuple(
        name
        for name in HOST_RECOVERY_NODE_NAMES
        if states[name] in {"commit_uncertain", "failed_before_commit"}
    )
    _require_exact_string_tuple(
        unresolved_recovery_nodes,
        expected_unresolved,
        "cleanup unresolved nodes",
    )
    if _require_bool(recovery_complete, "cleanup recovery complete") is not True:
        _fail("cleanup reconciliation must terminally classify every node")
    if _require_bool(terminalization_permitted, "cleanup terminalization") is not True:
        _fail("cleanup uncertainty cannot suppress terminalization")
    if _require_bool(workload_resume_permitted, "cleanup workload resume") is not False:
        _fail("cleanup reconciliation cannot resume a workload")
    if _require_bool(same_case_retry_permitted, "cleanup same-case retry") is not False:
        _fail("cleanup reconciliation cannot permit same-case retry")
    return {
        "schema_version": HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        "status": "conditional_recovery_dag_reconciled_non_authorizing",
        "case_spine_sha256": case_spine_sha256,
        "operational_frontier": operational_frontier.to_dict(),
        "cgroup_may_exist": may_exist,
        "recovery_nodes": [item.to_dict() for item in recovery_nodes],
        "cleanup_proven": cleanup_proven,
        "unresolved_recovery_nodes": list(unresolved_recovery_nodes),
        "recovery_complete": recovery_complete,
        "terminalization_permitted": terminalization_permitted,
        "workload_resume_permitted": workload_resume_permitted,
        "same_case_retry_permitted": same_case_retry_permitted,
        "claims": _host_contract_claims(),
    }


def canonical_host_cleanup_reconciliation_v2_body_bytes(**facts: Any) -> bytes:
    return _canonical_json(host_cleanup_reconciliation_v2_body_projection(**facts), newline=False)


def canonical_host_cleanup_reconciliation_v2_file_bytes(**facts: Any) -> bytes:
    body = host_cleanup_reconciliation_v2_body_projection(**facts)
    return _canonical_json(
        {**body, HOST_CLEANUP_RECONCILIATION_BODY_SHA256_FIELD: _body_sha256(body)}
    )


HOST_TERMINAL_METADATA_V2_BODY_KEYS: Final = (
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
HOST_TERMINAL_METADATA_V2_BODY_SHA256_FIELD: Final = "terminal_metadata_body_sha256"


def _optional_artifact_projection(
    value: ArtifactIdentityV2 | None,
    schema_version: str,
    label: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    _require_artifact_schema(value, schema_version, label)
    return value.to_dict()


def host_terminal_metadata_v2_body_projection(
    *,
    case_spine_sha256: str,
    case_ordinal: int,
    candidate_id: str,
    candidate_family: str,
    qualification_case_id: str,
    record_kind: str,
    operational_frontier: ArtifactIdentityV2,
    cleanup_reconciliation: ArtifactIdentityV2,
    driver_terminal: ArtifactIdentityV2 | None,
    algorithmic_resource_receipt: ArtifactIdentityV2 | None,
    publication_commitment_wrapper: ArtifactIdentityV2 | None,
    publication_reload_validation: ArtifactIdentityV2 | None,
    storage_write_seal: ArtifactIdentityV2 | None,
    storage_boundary_receipt: ArtifactIdentityV2 | None,
    returncode: int | None,
    timed_out: bool,
    error_message_sha256: str | None,
    cleanup_proven: bool,
    case_consumed: bool,
    same_case_retry_permitted: bool,
) -> dict[str, Any]:
    _require_sha256(case_spine_sha256, "terminal metadata case spine")
    _require_case_projection(
        case_ordinal,
        candidate_id,
        qualification_case_id,
        "terminal metadata",
    )
    _require_exact_literal(
        candidate_family,
        _family_for_candidate(candidate_id),
        "terminal metadata candidate family",
    )
    kind = _require_one_of(
        record_kind,
        ("success", "terminal_failure"),
        "terminal metadata record kind",
    )
    _require_artifact_schema(
        operational_frontier,
        HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
        "terminal operational frontier",
    )
    _require_artifact_schema(
        cleanup_reconciliation,
        HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        "terminal cleanup reconciliation",
    )
    driver_dict = _optional_artifact_projection(
        driver_terminal,
        IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION,
        "terminal driver record",
    )
    algorithmic_dict = _optional_artifact_projection(
        algorithmic_resource_receipt,
        _algorithmic_resource_receipt_schema(_family_for_candidate(candidate_id)),
        "terminal algorithmic receipt",
    )
    wrapper_dict = _optional_artifact_projection(
        publication_commitment_wrapper,
        NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
        "terminal publication wrapper",
    )
    reload_dict = _optional_artifact_projection(
        publication_reload_validation,
        QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "terminal reload validation",
    )
    seal_dict = _optional_artifact_projection(
        storage_write_seal,
        QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
        "terminal storage write seal",
    )
    storage_dict = _optional_artifact_projection(
        storage_boundary_receipt,
        QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
        "terminal storage receipt",
    )
    if returncode is not None:
        _require_int(returncode, "terminal return code", minimum=-_MAX_INTEGER)
    _require_bool(timed_out, "terminal timed-out fact")
    _require_optional_sha256(error_message_sha256, "terminal error message")
    proven = _require_bool(cleanup_proven, "terminal cleanup proven")
    if _require_bool(case_consumed, "terminal case consumed") is not True:
        _fail("terminal metadata must consume the case")
    if _require_bool(same_case_retry_permitted, "terminal retry") is not False:
        _fail("terminal metadata cannot permit same-case retry")
    if kind == "success" and (
        any(
            value is None
            for value in (
                driver_terminal,
                algorithmic_resource_receipt,
                publication_commitment_wrapper,
                publication_reload_validation,
                storage_write_seal,
                storage_boundary_receipt,
            )
        )
        or returncode != 0
        or timed_out is not False
        or error_message_sha256 is not None
        or proven is not True
    ):
        _fail("success terminal metadata lacks its exact committed facts")
    if kind == "terminal_failure" and error_message_sha256 is None:
        _fail("failure terminal metadata requires an exact error-message identity")
    return {
        "schema_version": HOST_TERMINAL_METADATA_SCHEMA_VERSION,
        "case_spine_sha256": case_spine_sha256,
        "case_ordinal": case_ordinal,
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "qualification_case_id": qualification_case_id,
        "record_kind": kind,
        "operational_frontier": operational_frontier.to_dict(),
        "cleanup_reconciliation": cleanup_reconciliation.to_dict(),
        "driver_terminal": driver_dict,
        "algorithmic_resource_receipt": algorithmic_dict,
        "publication_commitment_wrapper": wrapper_dict,
        "publication_reload_validation": reload_dict,
        "storage_write_seal": seal_dict,
        "storage_boundary_receipt": storage_dict,
        "returncode": returncode,
        "timed_out": timed_out,
        "error_message_sha256": error_message_sha256,
        "cleanup_proven": proven,
        "case_consumed": case_consumed,
        "same_case_retry_permitted": same_case_retry_permitted,
        "authority": _host_contract_authority(),
        "claims": _host_contract_claims(),
    }


def canonical_host_terminal_metadata_v2_body_bytes(**facts: Any) -> bytes:
    return _canonical_json(host_terminal_metadata_v2_body_projection(**facts), newline=False)


def canonical_host_terminal_metadata_v2_file_bytes(**facts: Any) -> bytes:
    body = host_terminal_metadata_v2_body_projection(**facts)
    return _canonical_json(
        {**body, HOST_TERMINAL_METADATA_V2_BODY_SHA256_FIELD: _body_sha256(body)}
    )


@dataclass(frozen=True, slots=True)
class HostSuccessCandidateV2:
    """Exact raw host success-chain identities, still structural inputs only."""

    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_plan_file_sha256: str
    qualification_plan_body_sha256: str
    case_execution_ticket_file_sha256: str
    case_execution_ticket_body_sha256: str
    qualification_case_manifest_file_sha256: str
    qualification_case_manifest_body_sha256: str
    publisher_registry_entry_file_sha256: str
    publisher_registry_entry_body_sha256: str
    resource_requirement_body_sha256: str
    image_id: str
    host_executor: ProducerIdentityV2
    host_provisioning_receipt: ArtifactIdentityV2
    request: ArtifactIdentityV2
    authorization_request_file_sha256: str
    authorization_request_body_sha256: str
    intent: ArtifactIdentityV2
    initial_cgroup_sample: ArtifactIdentityV2
    initial_sample_intent_file_sha256: str
    initial_sample_intent_body_sha256: str
    initial_sample_retained_fd_set_sha256: str
    initial_sample_cgroup_identity_sha256: str
    ready: ArtifactIdentityV2
    ready_initial_cgroup_sample_file_sha256: str
    ready_initial_cgroup_sample_body_sha256: str
    ready_retained_fd_set_sha256: str
    ready_cgroup_identity_sha256: str
    observer_anchor: ArtifactIdentityV2
    observer_initial_cgroup_sample_file_sha256: str
    observer_initial_cgroup_sample_body_sha256: str
    observer_retained_fd_set_sha256: str
    observer_cgroup_identity_sha256: str
    go_commitment: ArtifactIdentityV2
    go_retained_fd_set_sha256: str
    go_cgroup_identity_sha256: str
    operational_frontier: ArtifactIdentityV2
    driver_terminal: ArtifactIdentityV2
    recovery_nodes: tuple[RecoveryNodeCandidateV2, ...]
    cleanup_cgroup_may_exist: bool
    cleanup_proven: bool
    cleanup_unresolved_recovery_nodes: tuple[str, ...]
    cleanup_recovery_complete: bool
    cleanup_terminalization_permitted: bool
    cleanup_workload_resume_permitted: bool
    cleanup_same_case_retry_permitted: bool
    cleanup_reconciliation: ArtifactIdentityV2
    lifecycle: ArtifactIdentityV2
    lifecycle_operational_frontier_file_sha256: str
    lifecycle_operational_frontier_body_sha256: str
    lifecycle_cleanup_reconciliation_file_sha256: str
    lifecycle_cleanup_reconciliation_body_sha256: str
    lifecycle_terminal_metadata_file_sha256: str
    lifecycle_terminal_metadata_body_sha256: str
    completed_phases: tuple[str, ...]
    cgroup_proof: ArtifactIdentityV2
    terminal_metadata: ArtifactIdentityV2
    execution_receipt: ArtifactIdentityV2
    receipt_lifecycle_file_sha256: str
    receipt_lifecycle_body_sha256: str
    receipt_terminal_metadata_file_sha256: str
    receipt_terminal_metadata_body_sha256: str
    observation_handoff: ArtifactIdentityV2
    handoff_execution_receipt_file_sha256: str
    handoff_execution_receipt_body_sha256: str
    handoff_terminal_metadata_file_sha256: str
    handoff_terminal_metadata_body_sha256: str
    endpoint_observer_request_file_sha256: str | None
    endpoint_observer_request_body_sha256: str | None
    endpoint_observer_receipt_file_sha256: str | None
    endpoint_observer_receipt_body_sha256: str | None
    go_ready_file_sha256: str
    go_ready_body_sha256: str
    go_observer_anchor_file_sha256: str
    go_observer_anchor_body_sha256: str
    request_algorithmic_measurement_intent_file_sha256: str
    request_algorithmic_measurement_intent_body_sha256: str
    ready_algorithmic_measurement_intent_file_sha256: str
    ready_algorithmic_measurement_intent_body_sha256: str
    terminal_algorithmic_resource_receipt_file_sha256: str
    terminal_algorithmic_resource_receipt_body_sha256: str
    request_storage_boundary_intent_file_sha256: str
    request_storage_boundary_intent_body_sha256: str
    ready_storage_boundary_intent_file_sha256: str
    ready_storage_boundary_intent_body_sha256: str
    terminal_storage_write_seal: ArtifactIdentityV2
    terminal_storage_boundary_receipt_file_sha256: str
    terminal_storage_boundary_receipt_body_sha256: str
    publication_address_sha256: str
    publication_commitment_wrapper_file_sha256: str
    publication_commitment_wrapper_body_sha256: str
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    terminal_publication_manifest_file_sha256: str
    terminal_publication_manifest_body_sha256: str
    terminal_published_bundle_sha256: str
    terminal_reload_observation_sha256: str
    terminal_publication_reload_validation: ArtifactIdentityV2
    storage_write_seal_reload_validation_file_sha256: str
    storage_write_seal_reload_validation_body_sha256: str
    terminal_file_inventory_sha256: str
    terminal_file_count: int
    terminal_total_size_bytes: int
    terminal_family_metadata: ArtifactIdentityV2
    container_create_count: int
    container_start_count: int
    go_commit_count: int
    workload_start_count: int
    workload_exit_count: int
    attempt_count: int
    failure_count: int
    returncode: int
    timed_out: bool
    execution_state: str
    publication_state: str
    cleanup_state: str
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_case_projection(
            self.case_ordinal,
            self.candidate_id,
            self.qualification_case_id,
            "host-success case",
        )
        for value, label in (
            (self.case_spine_sha256, "host-success case spine"),
            (self.qualification_plan_file_sha256, "host-success plan file"),
            (self.qualification_plan_body_sha256, "host-success plan body"),
            (self.case_execution_ticket_file_sha256, "host-success ticket file"),
            (self.case_execution_ticket_body_sha256, "host-success ticket body"),
            (
                self.qualification_case_manifest_file_sha256,
                "host-success case manifest file",
            ),
            (
                self.qualification_case_manifest_body_sha256,
                "host-success case manifest body",
            ),
            (
                self.publisher_registry_entry_file_sha256,
                "host-success publisher entry file",
            ),
            (
                self.publisher_registry_entry_body_sha256,
                "host-success publisher entry body",
            ),
            (self.resource_requirement_body_sha256, "host-success resource requirement"),
            (self.go_ready_file_sha256, "host-success GO READY file projection"),
            (self.go_ready_body_sha256, "host-success GO READY body projection"),
            (
                self.go_observer_anchor_file_sha256,
                "host-success GO observer-anchor file projection",
            ),
            (
                self.go_observer_anchor_body_sha256,
                "host-success GO observer-anchor body projection",
            ),
            (
                self.handoff_execution_receipt_file_sha256,
                "host-success handoff execution-receipt file projection",
            ),
            (
                self.handoff_execution_receipt_body_sha256,
                "host-success handoff execution-receipt body projection",
            ),
            (
                self.handoff_terminal_metadata_file_sha256,
                "host-success handoff terminal-metadata file projection",
            ),
            (
                self.handoff_terminal_metadata_body_sha256,
                "host-success handoff terminal-metadata body projection",
            ),
            (
                self.request_algorithmic_measurement_intent_file_sha256,
                "host-success request algorithmic intent file",
            ),
            (
                self.request_algorithmic_measurement_intent_body_sha256,
                "host-success request algorithmic intent body",
            ),
            (
                self.ready_algorithmic_measurement_intent_file_sha256,
                "host-success READY algorithmic intent file",
            ),
            (
                self.ready_algorithmic_measurement_intent_body_sha256,
                "host-success READY algorithmic intent body",
            ),
            (
                self.terminal_algorithmic_resource_receipt_file_sha256,
                "host-success terminal algorithmic receipt file",
            ),
            (
                self.terminal_algorithmic_resource_receipt_body_sha256,
                "host-success terminal algorithmic receipt body",
            ),
            (
                self.request_storage_boundary_intent_file_sha256,
                "host-success request storage intent file",
            ),
            (
                self.request_storage_boundary_intent_body_sha256,
                "host-success request storage intent body",
            ),
            (
                self.ready_storage_boundary_intent_file_sha256,
                "host-success READY storage intent file",
            ),
            (
                self.ready_storage_boundary_intent_body_sha256,
                "host-success READY storage intent body",
            ),
            (
                self.terminal_storage_boundary_receipt_file_sha256,
                "host-success terminal storage receipt file",
            ),
            (
                self.terminal_storage_boundary_receipt_body_sha256,
                "host-success terminal storage receipt body",
            ),
            (self.publication_address_sha256, "host-success publication address"),
            (
                self.publication_commitment_wrapper_file_sha256,
                "host-success publication wrapper file",
            ),
            (
                self.publication_commitment_wrapper_body_sha256,
                "host-success publication wrapper body",
            ),
            (self.publisher_descriptor_sha256, "host-success publisher descriptor"),
            (self.publisher_source_sha256, "host-success publisher source"),
            (
                self.terminal_publication_manifest_file_sha256,
                "host-success terminal publication manifest file",
            ),
            (
                self.terminal_publication_manifest_body_sha256,
                "host-success terminal publication manifest body",
            ),
            (self.terminal_published_bundle_sha256, "host-success terminal bundle"),
            (
                self.terminal_reload_observation_sha256,
                "host-success terminal reload observation",
            ),
            (
                self.storage_write_seal_reload_validation_file_sha256,
                "host-success seal reload-validation file projection",
            ),
            (
                self.storage_write_seal_reload_validation_body_sha256,
                "host-success seal reload-validation body projection",
            ),
            (
                self.terminal_file_inventory_sha256,
                "host-success terminal file inventory",
            ),
        ):
            _require_sha256(value, label)
        for value, label in (
            (self.authorization_request_file_sha256, "authorization request file"),
            (self.authorization_request_body_sha256, "authorization request BODY"),
            (self.initial_sample_intent_file_sha256, "initial-sample intent file"),
            (self.initial_sample_intent_body_sha256, "initial-sample intent BODY"),
            (self.initial_sample_retained_fd_set_sha256, "initial retained-FD set"),
            (self.initial_sample_cgroup_identity_sha256, "initial cgroup identity"),
            (
                self.ready_initial_cgroup_sample_file_sha256,
                "READY initial-sample file",
            ),
            (
                self.ready_initial_cgroup_sample_body_sha256,
                "READY initial-sample BODY",
            ),
            (self.ready_retained_fd_set_sha256, "READY retained-FD set"),
            (self.ready_cgroup_identity_sha256, "READY cgroup identity"),
            (
                self.observer_initial_cgroup_sample_file_sha256,
                "observer initial-sample file",
            ),
            (
                self.observer_initial_cgroup_sample_body_sha256,
                "observer initial-sample BODY",
            ),
            (self.observer_retained_fd_set_sha256, "observer retained-FD set"),
            (self.observer_cgroup_identity_sha256, "observer cgroup identity"),
            (self.go_retained_fd_set_sha256, "GO retained-FD set"),
            (self.go_cgroup_identity_sha256, "GO cgroup identity"),
            (
                self.lifecycle_operational_frontier_file_sha256,
                "lifecycle operational-frontier file",
            ),
            (
                self.lifecycle_operational_frontier_body_sha256,
                "lifecycle operational-frontier BODY",
            ),
            (
                self.lifecycle_cleanup_reconciliation_file_sha256,
                "lifecycle cleanup-reconciliation file",
            ),
            (
                self.lifecycle_cleanup_reconciliation_body_sha256,
                "lifecycle cleanup-reconciliation BODY",
            ),
            (
                self.lifecycle_terminal_metadata_file_sha256,
                "lifecycle terminal-metadata file",
            ),
            (
                self.lifecycle_terminal_metadata_body_sha256,
                "lifecycle terminal-metadata BODY",
            ),
            (self.receipt_lifecycle_file_sha256, "success receipt lifecycle file"),
            (self.receipt_lifecycle_body_sha256, "success receipt lifecycle BODY"),
            (
                self.receipt_terminal_metadata_file_sha256,
                "success receipt terminal-metadata file",
            ),
            (
                self.receipt_terminal_metadata_body_sha256,
                "success receipt terminal-metadata BODY",
            ),
        ):
            _require_sha256(value, f"host-success {label}")
        endpoint_values = (
            self.endpoint_observer_request_file_sha256,
            self.endpoint_observer_request_body_sha256,
            self.endpoint_observer_receipt_file_sha256,
            self.endpoint_observer_receipt_body_sha256,
        )
        for index, endpoint_value in enumerate(endpoint_values):
            _require_optional_sha256(
                endpoint_value,
                f"host-success optional endpoint identity {index}",
            )
        if any(value is None for value in endpoint_values) and any(
            value is not None for value in endpoint_values
        ):
            _fail("host-success endpoint corroboration identities have partial presence")
        _require_int(
            self.terminal_file_count,
            "host-success terminal file count",
            minimum=1,
            maximum=len(EXTERNAL_PUBLICATION_ROLE_PATHS),
        )
        _require_int(
            self.terminal_total_size_bytes,
            "host-success terminal aggregate size",
            maximum=MAX_PUBLICATION_TOTAL_BYTES,
        )
        _require_image_id(self.image_id, "host-success image")
        _require_production_host_executor(self.host_executor, "host-success executor")
        for artifact, schema, label in (
            (
                self.host_provisioning_receipt,
                HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
                "host-success provisioning receipt",
            ),
            (self.request, HOST_CASE_REQUEST_SCHEMA_VERSION, "host request"),
            (self.intent, HOST_CASE_INTENT_SCHEMA_VERSION, "host intent"),
            (
                self.initial_cgroup_sample,
                HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
                "host initial cgroup sample",
            ),
            (self.ready, HOST_READY_SCHEMA_VERSION, "host READY metadata"),
            (
                self.observer_anchor,
                HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
                "host observer anchor",
            ),
            (self.go_commitment, HOST_GO_SCHEMA_VERSION, "host GO commitment"),
            (
                self.operational_frontier,
                HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
                "host operational frontier",
            ),
            (
                self.driver_terminal,
                IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION,
                "host driver terminal",
            ),
            (
                self.cleanup_reconciliation,
                HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
                "host cleanup reconciliation",
            ),
            (self.lifecycle, HOST_LIFECYCLE_SCHEMA_VERSION, "host lifecycle"),
            (
                self.terminal_storage_write_seal,
                QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
                "host terminal storage write seal",
            ),
            (
                self.terminal_publication_reload_validation,
                QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "host terminal publication reload validation",
            ),
            (self.cgroup_proof, HOST_CGROUP_PROOF_SCHEMA_VERSION, "host cgroup proof"),
            (
                self.terminal_metadata,
                HOST_TERMINAL_METADATA_SCHEMA_VERSION,
                "host terminal metadata",
            ),
            (
                self.terminal_family_metadata,
                _publication_profile(_family_for_candidate(self.candidate_id))[1],
                "host terminal family metadata",
            ),
            (
                self.execution_receipt,
                HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
                "host success receipt",
            ),
            (
                self.observation_handoff,
                HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
                "host observation handoff",
            ),
        ):
            _require_artifact_schema(artifact, schema, label)
        _require_exact_string_tuple(
            self.completed_phases,
            HOST_OPERATIONAL_PHASES,
            "host-success operational phases",
        )
        frontier_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "completed_phases": self.completed_phases,
            "failure_phase": None,
            "failure_effect_state": None,
            "container_create_state": "committed",
            "container_start_state": "committed",
            "workload_start_state": "committed",
            "workload_exit_state": "committed",
            "container_create_count_state": "exact",
            "container_create_count": self.container_create_count,
            "container_start_count_state": "exact",
            "container_start_count": self.container_start_count,
            "workload_start_count_state": "exact",
            "workload_start_count": self.workload_start_count,
            "workload_exit_count_state": "exact",
            "workload_exit_count": self.workload_exit_count,
            "attempt_count_state": "exact",
            "attempt_count": self.attempt_count,
            "failure_count": self.failure_count,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }
        _require_canonical_artifact_identity(
            self.operational_frontier,
            schema_version=HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            body_bytes=canonical_host_operational_frontier_v2_body_bytes(**frontier_facts),
            file_bytes=canonical_host_operational_frontier_v2_file_bytes(**frontier_facts),
            label="host-success operational frontier",
        )
        if (
            self.authorization_request_file_sha256 != self.request.file_sha256
            or self.authorization_request_body_sha256 != self.request.body_sha256
        ):
            _fail("host-success authorization is cross-wired from its request")
        if (
            self.initial_sample_intent_file_sha256 != self.intent.file_sha256
            or self.initial_sample_intent_body_sha256 != self.intent.body_sha256
            or self.ready_initial_cgroup_sample_file_sha256
            != self.initial_cgroup_sample.file_sha256
            or self.ready_initial_cgroup_sample_body_sha256
            != self.initial_cgroup_sample.body_sha256
            or self.observer_initial_cgroup_sample_file_sha256
            != self.initial_cgroup_sample.file_sha256
            or self.observer_initial_cgroup_sample_body_sha256
            != self.initial_cgroup_sample.body_sha256
            or len(
                {
                    self.initial_sample_retained_fd_set_sha256,
                    self.ready_retained_fd_set_sha256,
                    self.observer_retained_fd_set_sha256,
                    self.go_retained_fd_set_sha256,
                }
            )
            != 1
            or len(
                {
                    self.initial_sample_cgroup_identity_sha256,
                    self.ready_cgroup_identity_sha256,
                    self.observer_cgroup_identity_sha256,
                    self.go_cgroup_identity_sha256,
                }
            )
            != 1
        ):
            _fail("host-success initial-sample or retained-FD chain is cross-wired")
        if (
            type(self.recovery_nodes) is not tuple
            or any(type(item) is not RecoveryNodeCandidateV2 for item in self.recovery_nodes)
            or any(item.state != "committed" for item in self.recovery_nodes)
        ):
            _fail("host-success cleanup DAG must contain exact committed nodes")
        cleanup_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "operational_frontier": self.operational_frontier,
            "cgroup_may_exist": self.cleanup_cgroup_may_exist,
            "recovery_nodes": self.recovery_nodes,
            "cleanup_proven": self.cleanup_proven,
            "unresolved_recovery_nodes": self.cleanup_unresolved_recovery_nodes,
            "recovery_complete": self.cleanup_recovery_complete,
            "terminalization_permitted": self.cleanup_terminalization_permitted,
            "workload_resume_permitted": self.cleanup_workload_resume_permitted,
            "same_case_retry_permitted": self.cleanup_same_case_retry_permitted,
        }
        _require_canonical_artifact_identity(
            self.cleanup_reconciliation,
            schema_version=HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            body_bytes=canonical_host_cleanup_reconciliation_v2_body_bytes(**cleanup_facts),
            file_bytes=canonical_host_cleanup_reconciliation_v2_file_bytes(**cleanup_facts),
            label="host-success cleanup reconciliation",
        )
        final_cleanup_node = self.recovery_nodes[-1]
        if (
            self.cleanup_cgroup_may_exist is not True
            or self.cleanup_proven is not True
            or self.cleanup_unresolved_recovery_nodes != ()
            or final_cleanup_node.artifact != self.cgroup_proof
        ):
            _fail("host-success cleanup proof projection differs")
        algorithmic_receipt = ArtifactIdentityV2(
            schema_version=_algorithmic_resource_receipt_schema(
                _family_for_candidate(self.candidate_id)
            ),
            file_sha256=self.terminal_algorithmic_resource_receipt_file_sha256,
            body_sha256=self.terminal_algorithmic_resource_receipt_body_sha256,
        )
        publication_wrapper = ArtifactIdentityV2(
            schema_version=NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            file_sha256=self.publication_commitment_wrapper_file_sha256,
            body_sha256=self.publication_commitment_wrapper_body_sha256,
        )
        storage_receipt = ArtifactIdentityV2(
            schema_version=QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
            file_sha256=self.terminal_storage_boundary_receipt_file_sha256,
            body_sha256=self.terminal_storage_boundary_receipt_body_sha256,
        )
        terminal_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": _family_for_candidate(self.candidate_id),
            "qualification_case_id": self.qualification_case_id,
            "record_kind": "success",
            "operational_frontier": self.operational_frontier,
            "cleanup_reconciliation": self.cleanup_reconciliation,
            "driver_terminal": self.driver_terminal,
            "algorithmic_resource_receipt": algorithmic_receipt,
            "publication_commitment_wrapper": publication_wrapper,
            "publication_reload_validation": self.terminal_publication_reload_validation,
            "storage_write_seal": self.terminal_storage_write_seal,
            "storage_boundary_receipt": storage_receipt,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "error_message_sha256": None,
            "cleanup_proven": self.cleanup_proven,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }
        _require_canonical_artifact_identity(
            self.terminal_metadata,
            schema_version=HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            body_bytes=canonical_host_terminal_metadata_v2_body_bytes(**terminal_facts),
            file_bytes=canonical_host_terminal_metadata_v2_file_bytes(**terminal_facts),
            label="host-success terminal metadata",
        )
        if (
            self.lifecycle_operational_frontier_file_sha256 != self.operational_frontier.file_sha256
            or self.lifecycle_operational_frontier_body_sha256
            != self.operational_frontier.body_sha256
            or self.lifecycle_cleanup_reconciliation_file_sha256
            != self.cleanup_reconciliation.file_sha256
            or self.lifecycle_cleanup_reconciliation_body_sha256
            != self.cleanup_reconciliation.body_sha256
            or self.lifecycle_terminal_metadata_file_sha256 != self.terminal_metadata.file_sha256
            or self.lifecycle_terminal_metadata_body_sha256 != self.terminal_metadata.body_sha256
            or self.receipt_lifecycle_file_sha256 != self.lifecycle.file_sha256
            or self.receipt_lifecycle_body_sha256 != self.lifecycle.body_sha256
            or self.receipt_terminal_metadata_file_sha256 != self.terminal_metadata.file_sha256
            or self.receipt_terminal_metadata_body_sha256 != self.terminal_metadata.body_sha256
        ):
            _fail("host-success acyclic terminalization chain is cross-wired")
        if (
            self.ready.file_sha256 == self.observer_anchor.file_sha256
            or self.ready.body_sha256 == self.observer_anchor.body_sha256
        ):
            _fail("host READY and observer-anchor identities must remain distinct")
        if (
            self.go_ready_file_sha256 != self.ready.file_sha256
            or self.go_ready_body_sha256 != self.ready.body_sha256
            or self.go_observer_anchor_file_sha256 != self.observer_anchor.file_sha256
            or self.go_observer_anchor_body_sha256 != self.observer_anchor.body_sha256
        ):
            _fail("host GO projections must bind READY and the observer anchor")
        if (
            self.handoff_execution_receipt_file_sha256 != self.execution_receipt.file_sha256
            or self.handoff_execution_receipt_body_sha256 != self.execution_receipt.body_sha256
            or self.handoff_terminal_metadata_file_sha256 != self.terminal_metadata.file_sha256
            or self.handoff_terminal_metadata_body_sha256 != self.terminal_metadata.body_sha256
        ):
            _fail("host observation handoff projections are cross-wired")
        _require_host_observation_handoff_v2_identity(
            self.observation_handoff,
            case_spine_sha256=self.case_spine_sha256,
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            qualification_case_id=self.qualification_case_id,
            record_kind="success",
            terminal_receipt_file_sha256=self.execution_receipt.file_sha256,
            terminal_receipt_body_sha256=self.execution_receipt.body_sha256,
            terminal_metadata_file_sha256=self.terminal_metadata.file_sha256,
            terminal_metadata_body_sha256=self.terminal_metadata.body_sha256,
        )
        if (
            self.storage_write_seal_reload_validation_file_sha256
            != self.terminal_publication_reload_validation.file_sha256
            or self.storage_write_seal_reload_validation_body_sha256
            != self.terminal_publication_reload_validation.body_sha256
        ):
            _fail("host storage write seal is cross-wired from reload validation")
        _require_publication_reload_validation_v1_identity(
            self.terminal_publication_reload_validation,
            publication_commitment_wrapper_file_sha256=(
                self.publication_commitment_wrapper_file_sha256
            ),
            publication_commitment_wrapper_body_sha256=(
                self.publication_commitment_wrapper_body_sha256
            ),
            expected_reload_observation_sha256=(self.terminal_reload_observation_sha256),
            actual_reload_observation_sha256=self.terminal_reload_observation_sha256,
            reload_performed=True,
            reload_read_only=True,
            label="host terminal publication reload validation",
        )
        for count_value, expected, label in (
            (self.container_create_count, 1, "container-create count"),
            (self.container_start_count, 1, "container-start count"),
            (self.go_commit_count, 1, "GO-commit count"),
            (self.workload_start_count, 1, "workload-start count"),
            (self.workload_exit_count, 1, "workload-exit count"),
            (self.attempt_count, 1, "attempt count"),
            (self.failure_count, 0, "failure count"),
        ):
            if _require_int(count_value, f"host-success {label}", maximum=1) != expected:
                _fail(f"host-success {label} differs")
        if _require_int(self.returncode, "host-success return code") != 0:
            _fail("host-success return code must be zero")
        if _require_bool(self.timed_out, "host-success timeout") is not False:
            _fail("host-success candidate cannot be timed out")
        _require_exact_literal(
            self.execution_state,
            "metadata_complete_non_authorizing",
            "host-success execution state",
        )
        _require_exact_literal(
            self.publication_state,
            "committed",
            "host-success publication state",
        )
        _require_exact_literal(
            self.cleanup_state,
            "proven_empty",
            "host-success cleanup state",
        )
        if _require_bool(self.case_consumed, "host-success case consumed") is not True:
            _fail("host-success case must remain consumed")
        if _require_bool(self.same_case_retry_permitted, "host-success retry") is not False:
            _fail("host-success candidate can never permit retry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "qualification_case_id": self.qualification_case_id,
            "qualification_plan_file_sha256": self.qualification_plan_file_sha256,
            "qualification_plan_body_sha256": self.qualification_plan_body_sha256,
            "case_execution_ticket_file_sha256": (self.case_execution_ticket_file_sha256),
            "case_execution_ticket_body_sha256": (self.case_execution_ticket_body_sha256),
            "qualification_case_manifest_file_sha256": (
                self.qualification_case_manifest_file_sha256
            ),
            "qualification_case_manifest_body_sha256": (
                self.qualification_case_manifest_body_sha256
            ),
            "publisher_registry_entry_file_sha256": (self.publisher_registry_entry_file_sha256),
            "publisher_registry_entry_body_sha256": (self.publisher_registry_entry_body_sha256),
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "image_id": self.image_id,
            "host_executor": self.host_executor.to_dict(),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "request": self.request.to_dict(),
            "authorization_request_file_sha256": self.authorization_request_file_sha256,
            "authorization_request_body_sha256": self.authorization_request_body_sha256,
            "intent": self.intent.to_dict(),
            "initial_cgroup_sample": self.initial_cgroup_sample.to_dict(),
            "initial_sample_intent_file_sha256": self.initial_sample_intent_file_sha256,
            "initial_sample_intent_body_sha256": self.initial_sample_intent_body_sha256,
            "initial_sample_retained_fd_set_sha256": (self.initial_sample_retained_fd_set_sha256),
            "initial_sample_cgroup_identity_sha256": self.initial_sample_cgroup_identity_sha256,
            "ready": self.ready.to_dict(),
            "ready_initial_cgroup_sample_file_sha256": (
                self.ready_initial_cgroup_sample_file_sha256
            ),
            "ready_initial_cgroup_sample_body_sha256": (
                self.ready_initial_cgroup_sample_body_sha256
            ),
            "ready_retained_fd_set_sha256": self.ready_retained_fd_set_sha256,
            "ready_cgroup_identity_sha256": self.ready_cgroup_identity_sha256,
            "observer_anchor": self.observer_anchor.to_dict(),
            "observer_initial_cgroup_sample_file_sha256": (
                self.observer_initial_cgroup_sample_file_sha256
            ),
            "observer_initial_cgroup_sample_body_sha256": (
                self.observer_initial_cgroup_sample_body_sha256
            ),
            "observer_retained_fd_set_sha256": self.observer_retained_fd_set_sha256,
            "observer_cgroup_identity_sha256": self.observer_cgroup_identity_sha256,
            "go_commitment": self.go_commitment.to_dict(),
            "go_retained_fd_set_sha256": self.go_retained_fd_set_sha256,
            "go_cgroup_identity_sha256": self.go_cgroup_identity_sha256,
            "operational_frontier": self.operational_frontier.to_dict(),
            "driver_terminal": self.driver_terminal.to_dict(),
            "recovery_nodes": [item.to_dict() for item in self.recovery_nodes],
            "cleanup_cgroup_may_exist": self.cleanup_cgroup_may_exist,
            "cleanup_proven": self.cleanup_proven,
            "cleanup_unresolved_recovery_nodes": list(self.cleanup_unresolved_recovery_nodes),
            "cleanup_recovery_complete": self.cleanup_recovery_complete,
            "cleanup_terminalization_permitted": self.cleanup_terminalization_permitted,
            "cleanup_workload_resume_permitted": self.cleanup_workload_resume_permitted,
            "cleanup_same_case_retry_permitted": self.cleanup_same_case_retry_permitted,
            "cleanup_reconciliation": self.cleanup_reconciliation.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "lifecycle_operational_frontier_file_sha256": (
                self.lifecycle_operational_frontier_file_sha256
            ),
            "lifecycle_operational_frontier_body_sha256": (
                self.lifecycle_operational_frontier_body_sha256
            ),
            "lifecycle_cleanup_reconciliation_file_sha256": (
                self.lifecycle_cleanup_reconciliation_file_sha256
            ),
            "lifecycle_cleanup_reconciliation_body_sha256": (
                self.lifecycle_cleanup_reconciliation_body_sha256
            ),
            "lifecycle_terminal_metadata_file_sha256": (
                self.lifecycle_terminal_metadata_file_sha256
            ),
            "lifecycle_terminal_metadata_body_sha256": (
                self.lifecycle_terminal_metadata_body_sha256
            ),
            "completed_phases": list(self.completed_phases),
            "cgroup_proof": self.cgroup_proof.to_dict(),
            "terminal_metadata": self.terminal_metadata.to_dict(),
            "execution_receipt": self.execution_receipt.to_dict(),
            "receipt_lifecycle_file_sha256": self.receipt_lifecycle_file_sha256,
            "receipt_lifecycle_body_sha256": self.receipt_lifecycle_body_sha256,
            "receipt_terminal_metadata_file_sha256": (self.receipt_terminal_metadata_file_sha256),
            "receipt_terminal_metadata_body_sha256": (self.receipt_terminal_metadata_body_sha256),
            "observation_handoff": self.observation_handoff.to_dict(),
            "handoff_execution_receipt_file_sha256": (self.handoff_execution_receipt_file_sha256),
            "handoff_execution_receipt_body_sha256": (self.handoff_execution_receipt_body_sha256),
            "handoff_terminal_metadata_file_sha256": (self.handoff_terminal_metadata_file_sha256),
            "handoff_terminal_metadata_body_sha256": (self.handoff_terminal_metadata_body_sha256),
            "endpoint_observer_request_file_sha256": (self.endpoint_observer_request_file_sha256),
            "endpoint_observer_request_body_sha256": (self.endpoint_observer_request_body_sha256),
            "endpoint_observer_receipt_file_sha256": (self.endpoint_observer_receipt_file_sha256),
            "endpoint_observer_receipt_body_sha256": (self.endpoint_observer_receipt_body_sha256),
            "go_ready_file_sha256": self.go_ready_file_sha256,
            "go_ready_body_sha256": self.go_ready_body_sha256,
            "go_observer_anchor_file_sha256": (self.go_observer_anchor_file_sha256),
            "go_observer_anchor_body_sha256": (self.go_observer_anchor_body_sha256),
            "request_algorithmic_measurement_intent_file_sha256": (
                self.request_algorithmic_measurement_intent_file_sha256
            ),
            "request_algorithmic_measurement_intent_body_sha256": (
                self.request_algorithmic_measurement_intent_body_sha256
            ),
            "ready_algorithmic_measurement_intent_file_sha256": (
                self.ready_algorithmic_measurement_intent_file_sha256
            ),
            "ready_algorithmic_measurement_intent_body_sha256": (
                self.ready_algorithmic_measurement_intent_body_sha256
            ),
            "terminal_algorithmic_resource_receipt_file_sha256": (
                self.terminal_algorithmic_resource_receipt_file_sha256
            ),
            "terminal_algorithmic_resource_receipt_body_sha256": (
                self.terminal_algorithmic_resource_receipt_body_sha256
            ),
            "request_storage_boundary_intent_file_sha256": (
                self.request_storage_boundary_intent_file_sha256
            ),
            "request_storage_boundary_intent_body_sha256": (
                self.request_storage_boundary_intent_body_sha256
            ),
            "ready_storage_boundary_intent_file_sha256": (
                self.ready_storage_boundary_intent_file_sha256
            ),
            "ready_storage_boundary_intent_body_sha256": (
                self.ready_storage_boundary_intent_body_sha256
            ),
            "terminal_storage_write_seal": self.terminal_storage_write_seal.to_dict(),
            "terminal_storage_boundary_receipt_file_sha256": (
                self.terminal_storage_boundary_receipt_file_sha256
            ),
            "terminal_storage_boundary_receipt_body_sha256": (
                self.terminal_storage_boundary_receipt_body_sha256
            ),
            "publication_address_sha256": self.publication_address_sha256,
            "publication_commitment_wrapper_file_sha256": (
                self.publication_commitment_wrapper_file_sha256
            ),
            "publication_commitment_wrapper_body_sha256": (
                self.publication_commitment_wrapper_body_sha256
            ),
            "publisher_descriptor_sha256": self.publisher_descriptor_sha256,
            "publisher_source_sha256": self.publisher_source_sha256,
            "terminal_publication_manifest_file_sha256": (
                self.terminal_publication_manifest_file_sha256
            ),
            "terminal_publication_manifest_body_sha256": (
                self.terminal_publication_manifest_body_sha256
            ),
            "terminal_published_bundle_sha256": self.terminal_published_bundle_sha256,
            "terminal_reload_observation_sha256": (self.terminal_reload_observation_sha256),
            "terminal_publication_reload_validation": (
                self.terminal_publication_reload_validation.to_dict()
            ),
            "storage_write_seal_reload_validation_file_sha256": (
                self.storage_write_seal_reload_validation_file_sha256
            ),
            "storage_write_seal_reload_validation_body_sha256": (
                self.storage_write_seal_reload_validation_body_sha256
            ),
            "terminal_file_inventory_sha256": self.terminal_file_inventory_sha256,
            "terminal_file_count": self.terminal_file_count,
            "terminal_total_size_bytes": self.terminal_total_size_bytes,
            "terminal_family_metadata": self.terminal_family_metadata.to_dict(),
            "container_create_count": self.container_create_count,
            "container_start_count": self.container_start_count,
            "go_commit_count": self.go_commit_count,
            "workload_start_count": self.workload_start_count,
            "workload_exit_count": self.workload_exit_count,
            "attempt_count": self.attempt_count,
            "failure_count": self.failure_count,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "execution_state": self.execution_state,
            "publication_state": self.publication_state,
            "cleanup_state": self.cleanup_state,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }


type HostOperationalArtifactStateV2 = Literal[
    "not_started",
    "failed_before_commit",
    "commit_uncertain",
    "committed",
]


def _expected_operational_artifact_state(
    completed_phases: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
) -> HostOperationalArtifactStateV2:
    if phase in completed_phases:
        return "committed"
    if failure_phase == phase:
        if failure_effect_state == "failed_before_commit":
            return "failed_before_commit"
        if failure_effect_state == "commit_uncertain":
            return "commit_uncertain"
    return "not_started"


def _require_operational_phase_artifact(
    *,
    completed_phases: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
    state: str,
    artifact: ArtifactIdentityV2 | None,
    schema_version: str,
    label: str,
) -> HostOperationalArtifactStateV2:
    expected = _expected_operational_artifact_state(
        completed_phases,
        failure_phase,
        failure_effect_state,
        phase,
    )
    _require_exact_literal(state, expected, f"{label} state")
    if expected == "committed":
        if artifact is None:
            _fail(f"{label} committed without its artifact")
        _require_artifact_schema(artifact, schema_version, label)
    elif artifact is not None:
        _fail(f"{label} cannot carry an artifact before a known commit")
    return expected


def _require_optional_phase_projection(
    value: str | None,
    *,
    required: bool,
    label: str,
) -> str | None:
    if required:
        return _require_sha256(value, label)
    if value is not None:
        _fail(f"{label} cannot be present before its phase commits")
    return None


@dataclass(frozen=True, slots=True)
class HostTerminalFailureCandidateV2:
    """Consumed nonretryable failure with independent operational and cleanup frontiers."""

    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_plan_file_sha256: str
    qualification_plan_body_sha256: str
    case_execution_ticket_file_sha256: str
    case_execution_ticket_body_sha256: str
    qualification_case_manifest_file_sha256: str
    qualification_case_manifest_body_sha256: str
    publisher_registry_entry_file_sha256: str
    publisher_registry_entry_body_sha256: str
    resource_requirement_body_sha256: str
    image_id: str
    host_executor: ProducerIdentityV2
    host_provisioning_receipt: ArtifactIdentityV2
    algorithmic_resource_contract: ProducerIdentityV2
    algorithmic_measurement_intent: ArtifactIdentityV2
    algorithmic_measurement_intent_case_spine_sha256: str
    storage_boundary_contract: ProducerIdentityV2
    storage_boundary_intent: ArtifactIdentityV2
    storage_boundary_intent_case_spine_sha256: str
    request: ArtifactIdentityV2
    authorization_request_file_sha256: str
    authorization_request_body_sha256: str
    intent_state: HostOperationalArtifactStateV2
    intent: ArtifactIdentityV2 | None
    initial_cgroup_sample_state: HostOperationalArtifactStateV2
    initial_cgroup_sample: ArtifactIdentityV2 | None
    initial_sample_intent_file_sha256: str | None
    initial_sample_intent_body_sha256: str | None
    initial_sample_retained_fd_set_sha256: str | None
    initial_sample_cgroup_identity_sha256: str | None
    ready: ArtifactIdentityV2 | None
    ready_initial_cgroup_sample_file_sha256: str | None
    ready_initial_cgroup_sample_body_sha256: str | None
    ready_retained_fd_set_sha256: str | None
    ready_cgroup_identity_sha256: str | None
    observer_anchor: ArtifactIdentityV2 | None
    observer_initial_cgroup_sample_file_sha256: str | None
    observer_initial_cgroup_sample_body_sha256: str | None
    observer_retained_fd_set_sha256: str | None
    observer_cgroup_identity_sha256: str | None
    go_commitment: ArtifactIdentityV2 | None
    go_ready_file_sha256: str | None
    go_ready_body_sha256: str | None
    go_observer_anchor_file_sha256: str | None
    go_observer_anchor_body_sha256: str | None
    go_retained_fd_set_sha256: str | None
    go_cgroup_identity_sha256: str | None
    operational_frontier: ArtifactIdentityV2
    completed_phases: tuple[str, ...]
    failure_phase: str | None
    failure_effect_state: Literal["failed_before_commit", "commit_uncertain"] | None
    container_create_state: str
    container_start_state: str
    workload_start_state: str
    workload_exit_state: str
    container_create_count_state: str
    container_create_count: int | None
    container_start_count_state: str
    container_start_count: int | None
    workload_start_count_state: str
    workload_start_count: int | None
    workload_exit_count_state: str
    workload_exit_count: int | None
    attempt_count_state: str
    attempt_count: int | None
    failure_count_state: Literal["exact"]
    failure_count: int
    case_consumed: bool
    same_case_retry_permitted: bool
    driver_terminal: ArtifactIdentityV2 | None
    algorithmic_resource_receipt_state: HostOperationalArtifactStateV2
    algorithmic_resource_receipt: ArtifactIdentityV2 | None
    algorithmic_resource_receipt_case_spine_sha256: str | None
    native_publication_state: HostOperationalArtifactStateV2
    native_atomic_producer: ProducerIdentityV2 | None
    native_publication_receipt: ArtifactIdentityV2 | None
    expected_publication_address_sha256: str | None
    publication_reconciliation_key_sha256: str | None
    publication_reconciliation_reference: ArtifactIdentityV2 | None
    failure_publication_projection: FailurePublicationProjectionV2 | None
    publication_commitment_wrapper_state: HostOperationalArtifactStateV2
    publication_commitment_wrapper: ArtifactIdentityV2 | None
    publication_reload_state: HostOperationalArtifactStateV2
    publication_reload_validation: ArtifactIdentityV2 | None
    reload_observation_sha256: str | None
    storage_write_seal_state: HostOperationalArtifactStateV2
    storage_write_seal: ArtifactIdentityV2 | None
    storage_write_seal_case_spine_sha256: str | None
    storage_write_seal_reload_validation_file_sha256: str | None
    storage_write_seal_reload_validation_body_sha256: str | None
    storage_boundary_receipt_state: HostOperationalArtifactStateV2
    storage_boundary_receipt: ArtifactIdentityV2 | None
    storage_boundary_receipt_case_spine_sha256: str | None
    storage_boundary_receipt_write_seal_file_sha256: str | None
    storage_boundary_receipt_write_seal_body_sha256: str | None
    recovery_nodes: tuple[RecoveryNodeCandidateV2, ...]
    cleanup_cgroup_may_exist: bool
    cleanup_proven: bool
    cleanup_unresolved_recovery_nodes: tuple[str, ...]
    cleanup_recovery_complete: bool
    cleanup_terminalization_permitted: bool
    cleanup_workload_resume_permitted: bool
    cleanup_same_case_retry_permitted: bool
    cleanup_reconciliation: ArtifactIdentityV2
    precleanup_cgroup_sample: ArtifactIdentityV2 | None
    cgroup_kill_receipt: ArtifactIdentityV2 | None
    cgroup_empty_observation: ArtifactIdentityV2 | None
    container_absence_observation: ArtifactIdentityV2 | None
    post_container_remove_cgroup_sample: ArtifactIdentityV2 | None
    cgroup_counter_fds_closed_receipt: ArtifactIdentityV2 | None
    outer_cgroup_absence_observation: ArtifactIdentityV2 | None
    cgroup_proof: ArtifactIdentityV2 | None
    post_container_remove_retained_fd_set_sha256: str | None
    post_container_remove_cgroup_identity_sha256: str | None
    post_container_remove_container_identity_sha256: str | None
    cgroup_counter_fds_closed_post_sample_file_sha256: str | None
    cgroup_counter_fds_closed_post_sample_body_sha256: str | None
    cgroup_counter_fds_closed_retained_fd_set_sha256: str | None
    cgroup_counter_fds_closed_cgroup_identity_sha256: str | None
    cgroup_counter_fds_closed_container_identity_sha256: str | None
    outer_cgroup_absence_fd_close_file_sha256: str | None
    outer_cgroup_absence_fd_close_body_sha256: str | None
    outer_cgroup_absence_cgroup_identity_sha256: str | None
    recovery_failure_count_state: Literal["exact"]
    recovery_failure_count: int
    recovery_uncertainty_count_state: Literal["exact"]
    recovery_uncertainty_count: int
    terminal_metadata_state: Literal["committed"]
    terminal_metadata: ArtifactIdentityV2
    returncode: int | None
    timed_out: bool
    exception_type: str
    error_message_sha256: str
    uncertainty_dimensions: tuple[str, ...]
    lifecycle: ArtifactIdentityV2
    lifecycle_operational_frontier_file_sha256: str
    lifecycle_operational_frontier_body_sha256: str
    lifecycle_cleanup_reconciliation_file_sha256: str
    lifecycle_cleanup_reconciliation_body_sha256: str
    lifecycle_terminal_metadata_file_sha256: str
    lifecycle_terminal_metadata_body_sha256: str
    prior_host_execution_receipt_state: Literal["absent", "committed"]
    prior_host_execution_receipt: ArtifactIdentityV2 | None
    failure_receipt_state: Literal["committed"]
    failure_receipt: ArtifactIdentityV2
    receipt_lifecycle_file_sha256: str
    receipt_lifecycle_body_sha256: str
    receipt_terminal_metadata_file_sha256: str
    receipt_terminal_metadata_body_sha256: str
    receipt_operational_failure_count: int
    receipt_recovery_failure_count: int
    handoff_state: Literal["committed"]
    observation_handoff: ArtifactIdentityV2
    classification: str
    ticket_quarantined: bool
    reconciliation_only: bool
    clean_rejection_recorded: bool

    def __post_init__(self) -> None:
        _require_case_projection(
            self.case_ordinal,
            self.candidate_id,
            self.qualification_case_id,
            "host-failure case",
        )
        for value, label in (
            (self.case_spine_sha256, "host-failure case spine"),
            (self.qualification_plan_file_sha256, "host-failure plan file"),
            (self.qualification_plan_body_sha256, "host-failure plan BODY"),
            (self.case_execution_ticket_file_sha256, "host-failure ticket file"),
            (self.case_execution_ticket_body_sha256, "host-failure ticket BODY"),
            (
                self.qualification_case_manifest_file_sha256,
                "host-failure case-manifest file",
            ),
            (
                self.qualification_case_manifest_body_sha256,
                "host-failure case-manifest BODY",
            ),
            (
                self.publisher_registry_entry_file_sha256,
                "host-failure publisher-entry file",
            ),
            (
                self.publisher_registry_entry_body_sha256,
                "host-failure publisher-entry BODY",
            ),
            (self.resource_requirement_body_sha256, "host-failure resource requirement"),
            (
                self.algorithmic_measurement_intent_case_spine_sha256,
                "host-failure algorithmic-intent case spine",
            ),
            (
                self.storage_boundary_intent_case_spine_sha256,
                "host-failure storage-intent case spine",
            ),
            (self.authorization_request_file_sha256, "host-failure authorization request file"),
            (self.authorization_request_body_sha256, "host-failure authorization request BODY"),
            (
                self.lifecycle_operational_frontier_file_sha256,
                "host-failure lifecycle frontier file",
            ),
            (
                self.lifecycle_operational_frontier_body_sha256,
                "host-failure lifecycle frontier BODY",
            ),
            (
                self.lifecycle_cleanup_reconciliation_file_sha256,
                "host-failure lifecycle cleanup file",
            ),
            (
                self.lifecycle_cleanup_reconciliation_body_sha256,
                "host-failure lifecycle cleanup BODY",
            ),
            (
                self.lifecycle_terminal_metadata_file_sha256,
                "host-failure lifecycle terminal file",
            ),
            (
                self.lifecycle_terminal_metadata_body_sha256,
                "host-failure lifecycle terminal BODY",
            ),
            (self.receipt_lifecycle_file_sha256, "host-failure receipt lifecycle file"),
            (self.receipt_lifecycle_body_sha256, "host-failure receipt lifecycle BODY"),
            (
                self.receipt_terminal_metadata_file_sha256,
                "host-failure receipt terminal file",
            ),
            (
                self.receipt_terminal_metadata_body_sha256,
                "host-failure receipt terminal BODY",
            ),
            (self.error_message_sha256, "host-failure error message"),
        ):
            _require_sha256(value, label)
        _require_image_id(self.image_id, "host-failure image")
        _require_production_host_executor(self.host_executor, "host-failure executor")
        _require_artifact_schema(
            self.host_provisioning_receipt,
            HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host-failure provisioning receipt",
        )
        for producer, schema, descriptor_sha256, source_sha256, label in (
            (
                self.algorithmic_resource_contract,
                ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
                "host-failure algorithmic contract",
            ),
            (
                self.storage_boundary_contract,
                QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256,
                FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256,
                "host-failure storage contract",
            ),
        ):
            if (
                type(producer) is not ProducerIdentityV2
                or producer.descriptor_schema_version != schema
                or producer.descriptor_sha256 != descriptor_sha256
                or producer.source_sha256 != source_sha256
            ):
                _fail(f"{label} final identity differs")
        _require_artifact_schema(
            self.algorithmic_measurement_intent,
            ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "host-failure algorithmic intent",
        )
        _require_artifact_schema(
            self.storage_boundary_intent,
            QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "host-failure storage intent",
        )
        _require_artifact_schema(
            self.request, HOST_CASE_REQUEST_SCHEMA_VERSION, "host-failure request"
        )
        if (
            self.algorithmic_measurement_intent_case_spine_sha256 != self.case_spine_sha256
            or self.storage_boundary_intent_case_spine_sha256 != self.case_spine_sha256
            or self.authorization_request_file_sha256 != self.request.file_sha256
            or self.authorization_request_body_sha256 != self.request.body_sha256
        ):
            _fail("host-failure request or intent projections are cross-wired")

        frontier_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "completed_phases": self.completed_phases,
            "failure_phase": self.failure_phase,
            "failure_effect_state": self.failure_effect_state,
            "container_create_state": self.container_create_state,
            "container_start_state": self.container_start_state,
            "workload_start_state": self.workload_start_state,
            "workload_exit_state": self.workload_exit_state,
            "container_create_count_state": self.container_create_count_state,
            "container_create_count": self.container_create_count,
            "container_start_count_state": self.container_start_count_state,
            "container_start_count": self.container_start_count,
            "workload_start_count_state": self.workload_start_count_state,
            "workload_start_count": self.workload_start_count,
            "workload_exit_count_state": self.workload_exit_count_state,
            "workload_exit_count": self.workload_exit_count,
            "attempt_count_state": self.attempt_count_state,
            "attempt_count": self.attempt_count,
            "failure_count": self.failure_count,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }
        _require_exact_literal(
            self.failure_count_state,
            "exact",
            "host-failure operational failure-count state",
        )
        _require_canonical_artifact_identity(
            self.operational_frontier,
            schema_version=HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            body_bytes=canonical_host_operational_frontier_v2_body_bytes(**frontier_facts),
            file_bytes=canonical_host_operational_frontier_v2_file_bytes(**frontier_facts),
            label="host-failure operational frontier",
        )

        _require_operational_phase_artifact(
            completed_phases=self.completed_phases,
            failure_phase=self.failure_phase,
            failure_effect_state=self.failure_effect_state,
            phase="intent_committed",
            state=self.intent_state,
            artifact=self.intent,
            schema_version=HOST_CASE_INTENT_SCHEMA_VERSION,
            label="host-failure intent",
        )
        _require_operational_phase_artifact(
            completed_phases=self.completed_phases,
            failure_phase=self.failure_phase,
            failure_effect_state=self.failure_effect_state,
            phase="initial_cgroup_sample_committed",
            state=self.initial_cgroup_sample_state,
            artifact=self.initial_cgroup_sample,
            schema_version=HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            label="host-failure initial sample",
        )
        phase_artifacts = (
            ("driver_ready", self.ready, HOST_READY_SCHEMA_VERSION, "READY"),
            (
                "observer_anchored",
                self.observer_anchor,
                HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
                "observer anchor",
            ),
            ("go_committed", self.go_commitment, HOST_GO_SCHEMA_VERSION, "GO"),
        )
        for phase, artifact, schema, label in phase_artifacts:
            expected = _expected_operational_artifact_state(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                phase,
            )
            if expected == "committed":
                if artifact is None:
                    _fail(f"host-failure {label} committed without its artifact")
                _require_artifact_schema(artifact, schema, f"host-failure {label}")
            elif artifact is not None:
                _fail(f"host-failure {label} cannot be present before commit")
        driver_required = "workload_exited" in self.completed_phases
        if driver_required:
            if self.driver_terminal is None:
                _fail("host-failure committed workload exit lacks driver terminal")
            _require_artifact_schema(
                self.driver_terminal,
                IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION,
                "host-failure driver terminal",
            )
        elif self.driver_terminal is not None:
            _fail("host-failure driver terminal precedes a committed workload exit")

        initial_committed = self.initial_cgroup_sample_state == "committed"
        for optional_projection, projection_label in (
            (self.initial_sample_intent_file_sha256, "initial-sample intent file"),
            (self.initial_sample_intent_body_sha256, "initial-sample intent BODY"),
            (self.initial_sample_retained_fd_set_sha256, "initial retained-FD set"),
            (self.initial_sample_cgroup_identity_sha256, "initial cgroup identity"),
        ):
            _require_optional_phase_projection(
                optional_projection,
                required=initial_committed,
                label=f"host-failure {projection_label}",
            )
        if initial_committed:
            assert self.intent is not None
            assert self.initial_cgroup_sample is not None
            if (
                self.initial_sample_intent_file_sha256 != self.intent.file_sha256
                or self.initial_sample_intent_body_sha256 != self.intent.body_sha256
            ):
                _fail("host-failure initial sample is cross-wired from intent")

        ready_committed = self.ready is not None
        observer_committed = self.observer_anchor is not None
        go_committed = self.go_commitment is not None
        projection_groups = (
            (
                ready_committed,
                self.ready_initial_cgroup_sample_file_sha256,
                self.ready_initial_cgroup_sample_body_sha256,
                self.ready_retained_fd_set_sha256,
                self.ready_cgroup_identity_sha256,
                "READY",
            ),
            (
                observer_committed,
                self.observer_initial_cgroup_sample_file_sha256,
                self.observer_initial_cgroup_sample_body_sha256,
                self.observer_retained_fd_set_sha256,
                self.observer_cgroup_identity_sha256,
                "observer",
            ),
        )
        for committed, sample_file, sample_body, retained, cgroup, label in projection_groups:
            for optional_projection, projection_suffix in (
                (sample_file, "initial-sample file"),
                (sample_body, "initial-sample BODY"),
                (retained, "retained-FD set"),
                (cgroup, "cgroup identity"),
            ):
                _require_optional_phase_projection(
                    optional_projection,
                    required=committed,
                    label=f"host-failure {label} {projection_suffix}",
                )
            if committed:
                if self.initial_cgroup_sample is None:
                    _fail(f"host-failure {label} lacks its committed initial sample")
                if (
                    sample_file != self.initial_cgroup_sample.file_sha256
                    or sample_body != self.initial_cgroup_sample.body_sha256
                    or retained != self.initial_sample_retained_fd_set_sha256
                    or cgroup != self.initial_sample_cgroup_identity_sha256
                ):
                    _fail(f"host-failure {label} retained-FD chain is cross-wired")
        if ready_committed and observer_committed:
            assert self.ready is not None and self.observer_anchor is not None
            if (
                self.ready.file_sha256 == self.observer_anchor.file_sha256
                or self.ready.body_sha256 == self.observer_anchor.body_sha256
            ):
                _fail("host-failure READY and observer identities must remain distinct")
        for optional_projection, projection_label in (
            (self.go_ready_file_sha256, "GO READY file"),
            (self.go_ready_body_sha256, "GO READY BODY"),
            (self.go_observer_anchor_file_sha256, "GO observer file"),
            (self.go_observer_anchor_body_sha256, "GO observer BODY"),
            (self.go_retained_fd_set_sha256, "GO retained-FD set"),
            (self.go_cgroup_identity_sha256, "GO cgroup identity"),
        ):
            _require_optional_phase_projection(
                optional_projection,
                required=go_committed,
                label=f"host-failure {projection_label}",
            )
        if go_committed:
            if self.ready is None or self.observer_anchor is None:
                _fail("host-failure GO lacks READY or observer anchor")
            if (
                self.go_ready_file_sha256 != self.ready.file_sha256
                or self.go_ready_body_sha256 != self.ready.body_sha256
                or self.go_observer_anchor_file_sha256 != self.observer_anchor.file_sha256
                or self.go_observer_anchor_body_sha256 != self.observer_anchor.body_sha256
                or self.go_retained_fd_set_sha256 != self.initial_sample_retained_fd_set_sha256
                or self.go_cgroup_identity_sha256 != self.initial_sample_cgroup_identity_sha256
            ):
                _fail("host-failure GO chain is cross-wired")

        artifact_phases = (
            (
                "algorithmic_resource_receipt_committed",
                self.algorithmic_resource_receipt_state,
                self.algorithmic_resource_receipt,
                _algorithmic_resource_receipt_schema(_family_for_candidate(self.candidate_id)),
                "algorithmic receipt",
            ),
            (
                "publication_commitment_wrapper_committed",
                self.publication_commitment_wrapper_state,
                self.publication_commitment_wrapper,
                NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
                "publication wrapper",
            ),
            (
                "publication_reload_validated",
                self.publication_reload_state,
                self.publication_reload_validation,
                QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "reload validation",
            ),
            (
                "storage_write_seal_committed",
                self.storage_write_seal_state,
                self.storage_write_seal,
                QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
                "storage write seal",
            ),
            (
                "storage_boundary_receipt_committed",
                self.storage_boundary_receipt_state,
                self.storage_boundary_receipt,
                QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
                "storage receipt",
            ),
        )
        for phase, state, artifact, schema, label in artifact_phases:
            _require_operational_phase_artifact(
                completed_phases=self.completed_phases,
                failure_phase=self.failure_phase,
                failure_effect_state=self.failure_effect_state,
                phase=phase,
                state=state,
                artifact=artifact,
                schema_version=schema,
                label=f"host-failure {label}",
            )
        native_expected = _expected_operational_artifact_state(
            self.completed_phases,
            self.failure_phase,
            self.failure_effect_state,
            "native_publication_committed",
        )
        _require_exact_literal(
            self.native_publication_state,
            native_expected,
            "host-failure native-publication state",
        )
        if native_expected == "committed":
            if (
                self.native_atomic_producer is None
                or self.publication_reconciliation_reference is None
                or self.expected_publication_address_sha256 is None
                or self.publication_reconciliation_key_sha256 is None
            ):
                _fail("committed native publication lacks reconciliation identities")
            _require_sha256(
                self.expected_publication_address_sha256,
                "host-failure expected publication address",
            )
            _require_sha256(
                self.publication_reconciliation_key_sha256,
                "host-failure publication reconciliation key",
            )
            _require_artifact_schema(
                self.publication_reconciliation_reference,
                QUALIFICATION_PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION,
                "host-failure publication reconciliation reference",
            )
            family = _family_for_candidate(self.candidate_id)
            expected_native_producer_schema = (
                STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
                if family == "adapter"
                else ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            )
            if (
                type(self.native_atomic_producer) is not ProducerIdentityV2
                or self.native_atomic_producer.descriptor_schema_version
                != expected_native_producer_schema
            ):
                _fail("host-failure native producer identity differs")
            if family == "adapter" and any(
                digest in INCOMPATIBLE_ADAPTER_IMPLEMENTATION_SHA256S
                for digest in (
                    self.native_atomic_producer.descriptor_sha256,
                    self.native_atomic_producer.source_sha256,
                )
            ):
                _fail("unqualified adapter cannot fill host-failure native producer")
            if family == "local":
                if self.native_publication_receipt is not None:
                    _fail("local host-failure native publication cannot carry a receipt")
            else:
                expected_native_receipt_schema = (
                    EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
                    if family == "external"
                    else STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
                )
                if self.native_publication_receipt is None:
                    _fail("nonlocal host-failure native publication lacks its receipt")
                _require_artifact_schema(
                    self.native_publication_receipt,
                    expected_native_receipt_schema,
                    "host-failure native publication receipt",
                )
        elif any(
            value is not None
            for value in (
                self.native_atomic_producer,
                self.native_publication_receipt,
                self.expected_publication_address_sha256,
                self.publication_reconciliation_key_sha256,
                self.publication_reconciliation_reference,
            )
        ):
            _fail("uncommitted native publication cannot carry committed identities")

        wrapper_committed = self.publication_commitment_wrapper_state == "committed"
        if wrapper_committed:
            if self.failure_publication_projection is None:
                _fail("committed failure wrapper lacks its canonical publication projection")
            projection = self.failure_publication_projection
            assert self.publication_commitment_wrapper is not None
            if (
                projection.case_spine_sha256 != self.case_spine_sha256
                or projection.case_ordinal != self.case_ordinal
                or projection.candidate_id != self.candidate_id
                or projection.qualification_case_id != self.qualification_case_id
                or projection.publisher_registry_entry_file_sha256
                != self.publisher_registry_entry_file_sha256
                or projection.publisher_registry_entry_body_sha256
                != self.publisher_registry_entry_body_sha256
                or projection.publication_commitment_wrapper != self.publication_commitment_wrapper
                or projection.algorithmic_resource_receipt != self.algorithmic_resource_receipt
                or projection.native_atomic_producer != self.native_atomic_producer
                or projection.native_publication_receipt != self.native_publication_receipt
                or projection.publication_reconciliation_key_sha256
                != self.publication_reconciliation_key_sha256
                or projection.publication_reconciliation_reference
                != self.publication_reconciliation_reference
                or projection.publication_address_sha256 != self.expected_publication_address_sha256
            ):
                _fail("failure publication projection is cross-wired")
        elif self.failure_publication_projection is not None:
            _fail("failure publication projection requires a committed wrapper")

        if self.publication_reload_state == "committed":
            if (
                self.failure_publication_projection is None
                or self.reload_observation_sha256 is None
            ):
                _fail("committed failure reload lacks its publication projection")
            projection_expected = (
                self.failure_publication_projection.expected_reload_observation_sha256
            )
            _require_sha256(self.reload_observation_sha256, "host-failure reload observation")
            if self.reload_observation_sha256 != projection_expected:
                _fail("failure reload differs from the projected expected observation")
            assert self.publication_reload_validation is not None
            assert self.publication_commitment_wrapper is not None
            _require_publication_reload_validation_v1_identity(
                self.publication_reload_validation,
                publication_commitment_wrapper_file_sha256=(
                    self.publication_commitment_wrapper.file_sha256
                ),
                publication_commitment_wrapper_body_sha256=(
                    self.publication_commitment_wrapper.body_sha256
                ),
                expected_reload_observation_sha256=projection_expected,
                actual_reload_observation_sha256=projection_expected,
                reload_performed=True,
                reload_read_only=True,
                label="host-failure publication reload validation",
            )
        elif self.reload_observation_sha256 is not None:
            _fail("failure reload observation precedes committed reload validation")

        if self.algorithmic_resource_receipt_state == "committed":
            if self.algorithmic_resource_receipt_case_spine_sha256 != self.case_spine_sha256:
                _fail("failure algorithmic receipt case spine differs")
        elif self.algorithmic_resource_receipt_case_spine_sha256 is not None:
            _fail("uncommitted algorithmic receipt cannot carry a case-spine projection")
        if self.storage_write_seal_state == "committed":
            assert self.storage_write_seal is not None
            if (
                self.storage_write_seal_case_spine_sha256 != self.case_spine_sha256
                or self.publication_reload_validation is None
                or self.storage_write_seal_reload_validation_file_sha256
                != self.publication_reload_validation.file_sha256
                or self.storage_write_seal_reload_validation_body_sha256
                != self.publication_reload_validation.body_sha256
            ):
                _fail("failure storage write seal is cross-wired")
        elif any(
            value is not None
            for value in (
                self.storage_write_seal_case_spine_sha256,
                self.storage_write_seal_reload_validation_file_sha256,
                self.storage_write_seal_reload_validation_body_sha256,
            )
        ):
            _fail("uncommitted storage write seal cannot carry projections")
        if self.storage_boundary_receipt_state == "committed":
            assert self.storage_boundary_receipt is not None
            if (
                self.storage_boundary_receipt_case_spine_sha256 != self.case_spine_sha256
                or self.storage_write_seal is None
                or self.storage_boundary_receipt_write_seal_file_sha256
                != self.storage_write_seal.file_sha256
                or self.storage_boundary_receipt_write_seal_body_sha256
                != self.storage_write_seal.body_sha256
            ):
                _fail("failure storage receipt is cross-wired from its write seal")
        elif any(
            value is not None
            for value in (
                self.storage_boundary_receipt_case_spine_sha256,
                self.storage_boundary_receipt_write_seal_file_sha256,
                self.storage_boundary_receipt_write_seal_body_sha256,
            )
        ):
            _fail("uncommitted storage receipt cannot carry projections")

        if type(self.recovery_nodes) is not tuple or any(
            type(item) is not RecoveryNodeCandidateV2 for item in self.recovery_nodes
        ):
            _fail("host-failure cleanup nodes require exact candidate types")
        expected_cgroup_may_exist = "fresh_cgroup_created" in self.completed_phases or (
            self.failure_phase == "fresh_cgroup_created"
            and self.failure_effect_state == "commit_uncertain"
        )
        if (
            _require_bool(
                self.cleanup_cgroup_may_exist,
                "host-failure cleanup cgroup may exist",
            )
            is not expected_cgroup_may_exist
        ):
            _fail("host-failure cleanup cgroup-existence projection differs")
        cleanup_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "operational_frontier": self.operational_frontier,
            "cgroup_may_exist": self.cleanup_cgroup_may_exist,
            "recovery_nodes": self.recovery_nodes,
            "cleanup_proven": self.cleanup_proven,
            "unresolved_recovery_nodes": self.cleanup_unresolved_recovery_nodes,
            "recovery_complete": self.cleanup_recovery_complete,
            "terminalization_permitted": self.cleanup_terminalization_permitted,
            "workload_resume_permitted": self.cleanup_workload_resume_permitted,
            "same_case_retry_permitted": self.cleanup_same_case_retry_permitted,
        }
        _require_canonical_artifact_identity(
            self.cleanup_reconciliation,
            schema_version=HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            body_bytes=canonical_host_cleanup_reconciliation_v2_body_bytes(**cleanup_facts),
            file_bytes=canonical_host_cleanup_reconciliation_v2_file_bytes(**cleanup_facts),
            label="host-failure cleanup reconciliation",
        )
        nodes = {item.node_name: item for item in self.recovery_nodes}
        recovery_artifacts = {
            "precleanup_cgroup_sample": self.precleanup_cgroup_sample,
            "cgroup_kill": self.cgroup_kill_receipt,
            "cgroup_empty": self.cgroup_empty_observation,
            "container_absence": self.container_absence_observation,
            "post_container_remove_cgroup_sample": (self.post_container_remove_cgroup_sample),
            "cgroup_counter_fds_closed": self.cgroup_counter_fds_closed_receipt,
            "outer_cgroup_absence": self.outer_cgroup_absence_observation,
            "final_cgroup_proof": self.cgroup_proof,
        }
        for name in HOST_RECOVERY_NODE_NAMES:
            node = nodes[name]
            expected_artifact = node.artifact if node.state == "committed" else None
            if recovery_artifacts[name] != expected_artifact:
                _fail(f"host-failure cleanup artifact differs for {name}")
        recovery_failures = sum(
            item.state == "failed_before_commit" for item in self.recovery_nodes
        )
        recovery_uncertainties = sum(
            item.state == "commit_uncertain" for item in self.recovery_nodes
        )
        _require_exact_literal(
            self.recovery_failure_count_state,
            "exact",
            "host-failure recovery failure-count state",
        )
        _require_exact_literal(
            self.recovery_uncertainty_count_state,
            "exact",
            "host-failure recovery uncertainty-count state",
        )
        if (
            _require_int(
                self.recovery_failure_count,
                "host-failure recovery failure count",
                maximum=len(HOST_RECOVERY_NODE_NAMES),
            )
            != recovery_failures
            or _require_int(
                self.recovery_uncertainty_count,
                "host-failure recovery uncertainty count",
                maximum=len(HOST_RECOVERY_NODE_NAMES),
            )
            != recovery_uncertainties
        ):
            _fail("host-failure recovery counts differ")
        if self.failure_phase is None and recovery_failures + recovery_uncertainties == 0:
            _fail("terminal failure cannot follow clean operational and recovery frontiers")

        post_node = nodes["post_container_remove_cgroup_sample"]
        close_node = nodes["cgroup_counter_fds_closed"]
        outer_node = nodes["outer_cgroup_absence"]
        post_committed = post_node.state == "committed"
        close_committed = close_node.state == "committed"
        outer_committed = outer_node.state == "committed"
        for optional_projection, projection_required, projection_label in (
            (
                self.post_container_remove_retained_fd_set_sha256,
                post_committed,
                "post-container retained-FD set",
            ),
            (
                self.post_container_remove_cgroup_identity_sha256,
                post_committed,
                "post-container cgroup identity",
            ),
            (
                self.post_container_remove_container_identity_sha256,
                post_committed,
                "post-container container identity",
            ),
            (
                self.cgroup_counter_fds_closed_post_sample_file_sha256,
                close_committed,
                "FD-close post-sample file",
            ),
            (
                self.cgroup_counter_fds_closed_post_sample_body_sha256,
                close_committed,
                "FD-close post-sample BODY",
            ),
            (
                self.cgroup_counter_fds_closed_retained_fd_set_sha256,
                close_committed,
                "FD-close retained-FD set",
            ),
            (
                self.cgroup_counter_fds_closed_cgroup_identity_sha256,
                close_committed,
                "FD-close cgroup identity",
            ),
            (
                self.cgroup_counter_fds_closed_container_identity_sha256,
                close_committed,
                "FD-close container identity",
            ),
            (
                self.outer_cgroup_absence_fd_close_file_sha256,
                outer_committed,
                "outer-absence FD-close file",
            ),
            (
                self.outer_cgroup_absence_fd_close_body_sha256,
                outer_committed,
                "outer-absence FD-close BODY",
            ),
            (
                self.outer_cgroup_absence_cgroup_identity_sha256,
                outer_committed,
                "outer-absence cgroup identity",
            ),
        ):
            _require_optional_phase_projection(
                optional_projection,
                required=projection_required,
                label=f"host-failure {projection_label}",
            )
        if post_committed:
            if self.initial_cgroup_sample is not None and (
                self.post_container_remove_retained_fd_set_sha256
                != self.initial_sample_retained_fd_set_sha256
                or self.post_container_remove_cgroup_identity_sha256
                != self.initial_sample_cgroup_identity_sha256
            ):
                _fail("post-container retained-FD sample drifts from the initial sample")
        if close_committed:
            assert self.post_container_remove_cgroup_sample is not None
            if (
                self.cgroup_counter_fds_closed_post_sample_file_sha256
                != self.post_container_remove_cgroup_sample.file_sha256
                or self.cgroup_counter_fds_closed_post_sample_body_sha256
                != self.post_container_remove_cgroup_sample.body_sha256
                or self.cgroup_counter_fds_closed_retained_fd_set_sha256
                != self.post_container_remove_retained_fd_set_sha256
                or self.cgroup_counter_fds_closed_cgroup_identity_sha256
                != self.post_container_remove_cgroup_identity_sha256
                or self.cgroup_counter_fds_closed_container_identity_sha256
                != self.post_container_remove_container_identity_sha256
            ):
                _fail("retained-FD close chain is cross-wired")
        if outer_committed:
            assert self.cgroup_counter_fds_closed_receipt is not None
            if (
                self.outer_cgroup_absence_fd_close_file_sha256
                != self.cgroup_counter_fds_closed_receipt.file_sha256
                or self.outer_cgroup_absence_fd_close_body_sha256
                != self.cgroup_counter_fds_closed_receipt.body_sha256
                or self.outer_cgroup_absence_cgroup_identity_sha256
                != self.cgroup_counter_fds_closed_cgroup_identity_sha256
            ):
                _fail("outer-cgroup absence chain is cross-wired")

        _require_exact_literal(
            self.terminal_metadata_state,
            "committed",
            "host-failure terminal-metadata state",
        )
        _require_artifact_schema(
            self.terminal_metadata,
            HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            "host-failure terminal metadata",
        )
        if self.returncode is not None:
            _require_int(
                self.returncode,
                "host-failure return code",
                minimum=-_MAX_INTEGER,
            )
        _require_bool(self.timed_out, "host-failure timed-out fact")
        _require_text(self.exception_type, "host-failure exception type")
        expected_dimensions = tuple(
            dimension
            for dimension, present in (
                (
                    "operational_state",
                    self.failure_effect_state == "commit_uncertain",
                ),
                (
                    "publication_state",
                    any(
                        state == "commit_uncertain"
                        for state in (
                            self.native_publication_state,
                            self.publication_commitment_wrapper_state,
                            self.publication_reload_state,
                        )
                    ),
                ),
                (
                    "storage_state",
                    any(
                        state == "commit_uncertain"
                        for state in (
                            self.storage_write_seal_state,
                            self.storage_boundary_receipt_state,
                        )
                    ),
                ),
                ("cleanup_state", recovery_uncertainties > 0),
                ("terminalization_state", False),
            )
            if present
        )
        _require_exact_string_tuple(
            self.uncertainty_dimensions,
            expected_dimensions,
            "host-failure uncertainty dimensions",
        )
        terminal_facts = {
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": _family_for_candidate(self.candidate_id),
            "qualification_case_id": self.qualification_case_id,
            "record_kind": "terminal_failure",
            "operational_frontier": self.operational_frontier,
            "cleanup_reconciliation": self.cleanup_reconciliation,
            "driver_terminal": self.driver_terminal,
            "algorithmic_resource_receipt": self.algorithmic_resource_receipt,
            "publication_commitment_wrapper": self.publication_commitment_wrapper,
            "publication_reload_validation": self.publication_reload_validation,
            "storage_write_seal": self.storage_write_seal,
            "storage_boundary_receipt": self.storage_boundary_receipt,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "error_message_sha256": self.error_message_sha256,
            "cleanup_proven": self.cleanup_proven,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
        }
        _require_canonical_artifact_identity(
            self.terminal_metadata,
            schema_version=HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            body_bytes=canonical_host_terminal_metadata_v2_body_bytes(**terminal_facts),
            file_bytes=canonical_host_terminal_metadata_v2_file_bytes(**terminal_facts),
            label="host-failure terminal metadata",
        )
        _require_artifact_schema(
            self.lifecycle, HOST_LIFECYCLE_SCHEMA_VERSION, "host-failure lifecycle"
        )
        if (
            self.lifecycle_operational_frontier_file_sha256 != self.operational_frontier.file_sha256
            or self.lifecycle_operational_frontier_body_sha256
            != self.operational_frontier.body_sha256
            or self.lifecycle_cleanup_reconciliation_file_sha256
            != self.cleanup_reconciliation.file_sha256
            or self.lifecycle_cleanup_reconciliation_body_sha256
            != self.cleanup_reconciliation.body_sha256
            or self.lifecycle_terminal_metadata_file_sha256 != self.terminal_metadata.file_sha256
            or self.lifecycle_terminal_metadata_body_sha256 != self.terminal_metadata.body_sha256
        ):
            _fail("host-failure lifecycle is cross-wired from terminal metadata")

        _require_exact_literal(
            self.prior_host_execution_receipt_state,
            "absent" if self.prior_host_execution_receipt is None else "committed",
            "host-failure prior-receipt state",
        )
        if self.prior_host_execution_receipt is not None:
            _require_artifact_schema(
                self.prior_host_execution_receipt,
                HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
                "host-failure prior success receipt",
            )
        _require_exact_literal(
            self.failure_receipt_state,
            "committed",
            "host-failure receipt state",
        )
        _require_artifact_schema(
            self.failure_receipt,
            HOST_FAILURE_RECEIPT_SCHEMA_VERSION,
            "host-failure receipt",
        )
        if (
            self.receipt_lifecycle_file_sha256 != self.lifecycle.file_sha256
            or self.receipt_lifecycle_body_sha256 != self.lifecycle.body_sha256
            or self.receipt_terminal_metadata_file_sha256 != self.terminal_metadata.file_sha256
            or self.receipt_terminal_metadata_body_sha256 != self.terminal_metadata.body_sha256
            or self.receipt_operational_failure_count != self.failure_count
            or self.receipt_recovery_failure_count != self.recovery_failure_count
        ):
            _fail("host-failure receipt is cross-wired from lifecycle or counts")
        _require_int(
            self.receipt_operational_failure_count,
            "receipt operational failure count",
            maximum=1,
        )
        _require_int(
            self.receipt_recovery_failure_count,
            "receipt recovery failure count",
            maximum=len(HOST_RECOVERY_NODE_NAMES),
        )
        _require_exact_literal(self.handoff_state, "committed", "host-failure handoff state")
        _require_artifact_schema(
            self.observation_handoff,
            HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION,
            "host-failure observation handoff",
        )
        _require_host_observation_handoff_v2_identity(
            self.observation_handoff,
            case_spine_sha256=self.case_spine_sha256,
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            qualification_case_id=self.qualification_case_id,
            record_kind="terminal_failure",
            terminal_receipt_file_sha256=self.failure_receipt.file_sha256,
            terminal_receipt_body_sha256=self.failure_receipt.body_sha256,
            terminal_metadata_file_sha256=self.terminal_metadata.file_sha256,
            terminal_metadata_body_sha256=self.terminal_metadata.body_sha256,
        )

        expected_classification = (
            "recovery_failure_after_complete_operational_frontier_nonretryable"
            if self.failure_phase is None
            else "operational_failed_before_commit_ticket_quarantined_nonretryable"
            if self.failure_effect_state == "failed_before_commit"
            else "operational_commit_uncertain_ticket_quarantined_nonretryable"
        )
        _require_exact_literal(
            self.classification,
            expected_classification,
            "host-failure classification",
        )
        for boolean_value, expected_boolean, boolean_label in (
            (self.ticket_quarantined, True, "ticket quarantined"),
            (self.reconciliation_only, True, "reconciliation only"),
            (self.case_consumed, True, "case consumed"),
            (self.same_case_retry_permitted, False, "same-case retry"),
            (self.clean_rejection_recorded, False, "clean rejection"),
        ):
            if (
                _require_bool(boolean_value, f"host-failure {boolean_label}")
                is not expected_boolean
            ):
                _fail(f"host-failure {boolean_label} differs")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if type(value) is ArtifactIdentityV2:
                result[field_name] = value.to_dict()
            elif type(value) is ProducerIdentityV2:
                result[field_name] = value.to_dict()
            elif type(value) is FailurePublicationProjectionV2:
                result[field_name] = value.to_dict()
            elif field_name == "recovery_nodes":
                result[field_name] = [item.to_dict() for item in self.recovery_nodes]
            elif type(value) is tuple:
                result[field_name] = list(value)
            else:
                result[field_name] = value
        return result


def _validate_case_spine_host_links(
    spine: CaseSpineV2,
    host: HostSuccessCandidateV2 | HostTerminalFailureCandidateV2,
) -> None:
    expected = {
        "case_spine_sha256": spine.body_sha256,
        "case_ordinal": spine.case_ordinal,
        "candidate_id": spine.candidate_id,
        "qualification_case_id": spine.qualification_case_id,
        "case_execution_ticket_file_sha256": spine.case_execution_ticket.file_sha256,
        "case_execution_ticket_body_sha256": spine.case_execution_ticket.body_sha256,
        "qualification_case_manifest_file_sha256": (spine.qualification_case_manifest.file_sha256),
        "qualification_case_manifest_body_sha256": (spine.qualification_case_manifest.body_sha256),
        "publisher_registry_entry_file_sha256": (spine.publisher_registry_entry.file_sha256),
        "publisher_registry_entry_body_sha256": (spine.publisher_registry_entry.body_sha256),
        "resource_requirement_body_sha256": spine.resource_requirement_body_sha256,
    }
    for field, value in expected.items():
        if getattr(host, field) != value:
            _fail(f"host candidate is cross-wired at {field}")


@dataclass(frozen=True, slots=True)
class QualificationCaseSuccessCandidateV2:
    """One complete host-success metadata candidate; not an approved case."""

    case_spine: CaseSpineV2
    host_success: HostSuccessCandidateV2
    probes: tuple[ProbeCandidateV2, ...]
    resource_merger: ResourceMergerCandidateV2
    publication: PublicationCandidateV2

    def __post_init__(self) -> None:
        if type(self.case_spine) is not CaseSpineV2:
            _fail("success case spine type differs")
        if type(self.host_success) is not HostSuccessCandidateV2:
            _fail("success host candidate type differs")
        _validate_case_spine_host_links(self.case_spine, self.host_success)
        spine_sha256 = self.case_spine.body_sha256
        host_receipt = self.host_success.execution_receipt
        if (
            type(self.probes) is not tuple
            or any(type(item) is not ProbeCandidateV2 for item in self.probes)
            or tuple(item.probe_kind for item in self.probes) != PROBE_KINDS
        ):
            _fail("success probes must use the exact five-kind order")
        for probe in self.probes:
            if (
                probe.case_spine_sha256 != spine_sha256
                or probe.host_execution_receipt_file_sha256 != host_receipt.file_sha256
                or probe.host_execution_receipt_body_sha256 != host_receipt.body_sha256
            ):
                _fail("success probe is cross-wired from its case or host receipt")
        if type(self.resource_merger) is not ResourceMergerCandidateV2:
            _fail("success resource-merger candidate type differs")
        if type(self.publication) is not PublicationCandidateV2:
            _fail("success publication candidate type differs")
        endpoint_request_file = (
            None
            if self.resource_merger.endpoint_observer_request is None
            else self.resource_merger.endpoint_observer_request.file_sha256
        )
        endpoint_request_body = (
            None
            if self.resource_merger.endpoint_observer_request is None
            else self.resource_merger.endpoint_observer_request.body_sha256
        )
        endpoint_receipt_file = (
            None
            if self.resource_merger.endpoint_observer_receipt is None
            else self.resource_merger.endpoint_observer_receipt.file_sha256
        )
        endpoint_receipt_body = (
            None
            if self.resource_merger.endpoint_observer_receipt is None
            else self.resource_merger.endpoint_observer_receipt.body_sha256
        )
        resource_fields = {item.field_name: item for item in self.resource_merger.fields}
        if (
            self.resource_merger.candidate_id != self.case_spine.candidate_id
            or self.resource_merger.candidate_family != self.case_spine.candidate_family
            or self.resource_merger.case_spine_sha256 != spine_sha256
            or self.resource_merger.host_execution_receipt_file_sha256 != host_receipt.file_sha256
            or self.resource_merger.host_execution_receipt_body_sha256 != host_receipt.body_sha256
            or self.resource_merger.resource_requirement_body_sha256
            != self.case_spine.resource_requirement_body_sha256
            or self.resource_merger.host_provisioning_receipt
            != self.host_success.host_provisioning_receipt
            or self.resource_merger.host_cgroup_proof != self.host_success.cgroup_proof
            or self.resource_merger.host_terminal_metadata != self.host_success.terminal_metadata
            or self.resource_merger.host_observation_handoff
            != self.host_success.observation_handoff
            or self.resource_merger.storage_write_seal
            != self.host_success.terminal_storage_write_seal
            or endpoint_request_file != self.host_success.endpoint_observer_request_file_sha256
            or endpoint_request_body != self.host_success.endpoint_observer_request_body_sha256
            or endpoint_receipt_file != self.host_success.endpoint_observer_receipt_file_sha256
            or endpoint_receipt_body != self.host_success.endpoint_observer_receipt_body_sha256
            or self.resource_merger.algorithmic_measurement_intent.file_sha256
            != self.host_success.request_algorithmic_measurement_intent_file_sha256
            or self.resource_merger.algorithmic_measurement_intent.body_sha256
            != self.host_success.request_algorithmic_measurement_intent_body_sha256
            or self.resource_merger.algorithmic_measurement_intent.file_sha256
            != self.host_success.ready_algorithmic_measurement_intent_file_sha256
            or self.resource_merger.algorithmic_measurement_intent.body_sha256
            != self.host_success.ready_algorithmic_measurement_intent_body_sha256
            or self.resource_merger.algorithmic_resource_receipt.file_sha256
            != self.host_success.terminal_algorithmic_resource_receipt_file_sha256
            or self.resource_merger.algorithmic_resource_receipt.body_sha256
            != self.host_success.terminal_algorithmic_resource_receipt_body_sha256
            or self.resource_merger.runner_execution_receipt
            != self.publication.runner_execution_receipt
            or self.resource_merger.storage_boundary_intent.file_sha256
            != self.host_success.request_storage_boundary_intent_file_sha256
            or self.resource_merger.storage_boundary_intent.body_sha256
            != self.host_success.request_storage_boundary_intent_body_sha256
            or self.resource_merger.storage_boundary_intent.file_sha256
            != self.host_success.ready_storage_boundary_intent_file_sha256
            or self.resource_merger.storage_boundary_intent.body_sha256
            != self.host_success.ready_storage_boundary_intent_body_sha256
            or self.resource_merger.storage_boundary_receipt.file_sha256
            != self.host_success.terminal_storage_boundary_receipt_file_sha256
            or self.resource_merger.storage_boundary_receipt.body_sha256
            != self.host_success.terminal_storage_boundary_receipt_body_sha256
            or resource_fields["max_attempt_count"].observed_value
            != self.host_success.attempt_count
            or resource_fields["max_failure_count"].observed_value
            != self.host_success.failure_count
        ):
            _fail("success resource merger is cross-wired")
        if (
            self.publication.candidate_id != self.case_spine.candidate_id
            or self.publication.candidate_family != self.case_spine.candidate_family
            or self.publication.case_ordinal != self.case_spine.case_ordinal
            or self.publication.qualification_case_id != self.case_spine.qualification_case_id
            or self.publication.case_spine_sha256 != spine_sha256
            or self.publication.host_execution_receipt_file_sha256 != host_receipt.file_sha256
            or self.publication.host_execution_receipt_body_sha256 != host_receipt.body_sha256
            or self.publication.host_terminal_metadata_file_sha256
            != self.host_success.terminal_metadata.file_sha256
            or self.publication.host_terminal_metadata_body_sha256
            != self.host_success.terminal_metadata.body_sha256
            or self.publication.host_observation_handoff != self.host_success.observation_handoff
            or self.publication.publisher_registry_entry_file_sha256
            != self.case_spine.publisher_registry_entry.file_sha256
            or self.publication.publisher_registry_entry_body_sha256
            != self.case_spine.publisher_registry_entry.body_sha256
            or self.publication.publication_address_sha256
            != self.host_success.publication_address_sha256
            or self.publication.publication_commitment_wrapper.file_sha256
            != self.host_success.publication_commitment_wrapper_file_sha256
            or self.publication.publication_commitment_wrapper.body_sha256
            != self.host_success.publication_commitment_wrapper_body_sha256
            or self.publication.publisher.descriptor_sha256
            != self.host_success.publisher_descriptor_sha256
            or self.publication.publisher.source_sha256 != self.host_success.publisher_source_sha256
            or self.publication.publication_manifest_file_sha256
            != self.host_success.terminal_publication_manifest_file_sha256
            or self.publication.publication_manifest_body_sha256
            != self.host_success.terminal_publication_manifest_body_sha256
            or self.publication.published_bundle_sha256
            != self.host_success.terminal_published_bundle_sha256
            or self.publication.reload_observation_sha256
            != self.host_success.terminal_reload_observation_sha256
            or self.publication.publication_reload_validation
            != self.host_success.terminal_publication_reload_validation
            or self.publication.file_inventory_sha256
            != self.host_success.terminal_file_inventory_sha256
            or self.publication.file_count != self.host_success.terminal_file_count
            or self.publication.total_size_bytes != self.host_success.terminal_total_size_bytes
            or self.publication.publisher_metadata != self.host_success.terminal_family_metadata
        ):
            _fail("success publication is cross-wired")

    @property
    def record_kind(self) -> Literal["success"]:
        return "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFICATION_CASE_SUCCESS_CANDIDATE_V2_SCHEMA_VERSION,
            "record_kind": self.record_kind,
            "case_spine": self.case_spine.to_dict(),
            "success": {
                "host_success": self.host_success.to_dict(),
                "probes": [item.to_dict() for item in self.probes],
                "resource_merger": self.resource_merger.to_dict(),
                "publication": self.publication.to_dict(),
            },
            "claims": _claims(),
            "readiness": _readiness(),
        }


@dataclass(frozen=True, slots=True)
class QualificationCaseTerminalFailureCandidateV2:
    """One consumed terminal failure slot; never a clean rejection."""

    case_spine: CaseSpineV2
    terminal_failure: HostTerminalFailureCandidateV2

    def __post_init__(self) -> None:
        if type(self.case_spine) is not CaseSpineV2:
            _fail("terminal-failure case spine type differs")
        if type(self.terminal_failure) is not HostTerminalFailureCandidateV2:
            _fail("terminal-failure host candidate type differs")
        _validate_case_spine_host_links(self.case_spine, self.terminal_failure)

    @property
    def record_kind(self) -> Literal["terminal_failure"]:
        return "terminal_failure"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (QUALIFICATION_CASE_TERMINAL_FAILURE_CANDIDATE_V2_SCHEMA_VERSION),
            "record_kind": self.record_kind,
            "case_spine": self.case_spine.to_dict(),
            "terminal_failure": self.terminal_failure.to_dict(),
            "claims": _claims(),
            "readiness": _readiness(),
        }


type QualificationCaseCandidateV2 = (
    QualificationCaseSuccessCandidateV2 | QualificationCaseTerminalFailureCandidateV2
)


@dataclass(frozen=True, slots=True)
class QualificationObservationCandidateReplayPinsV2:
    """Independent caller pins required to replay one full batch."""

    batch_file_sha256: str
    qualification_plan_file_sha256: str
    qualification_plan_body_sha256: str
    observation_registry_source_sha256: str
    plan_issuance_receipt_file_sha256: str
    plan_issuance_receipt_body_sha256: str
    case_ticket_registry_file_sha256: str
    case_ticket_registry_body_sha256: str
    publisher_registry_file_sha256: str
    publisher_registry_body_sha256: str
    seed_registry_file_sha256: str
    seed_registry_body_sha256: str
    seed_pulse_record_file_sha256: str
    seed_pulse_record_body_sha256: str
    seed_trust_root_receipt_file_sha256: str
    seed_trust_root_receipt_body_sha256: str
    quicknet_verifier_descriptor_sha256: str
    quicknet_verifier_source_sha256: str
    quicknet_verifier_binary_sha256: str
    quicknet_verifier_receipt_file_sha256: str
    quicknet_verifier_receipt_body_sha256: str
    seed_chronology_receipt_file_sha256: str
    seed_chronology_receipt_body_sha256: str
    local_source_candidate_file_sha256: str
    local_source_candidate_body_sha256: str
    external_source_candidate_file_sha256: str
    external_source_candidate_body_sha256: str
    adapter_source_candidate_file_sha256: str
    adapter_source_candidate_body_sha256: str
    joint_source_closure_candidate_file_sha256: str
    joint_source_closure_candidate_body_sha256: str
    sealed_staging_candidate_file_sha256: str
    sealed_staging_candidate_body_sha256: str
    fresh_build_candidate_file_sha256: str
    fresh_build_candidate_body_sha256: str
    runtime_candidate_file_sha256: str
    runtime_candidate_body_sha256: str
    runtime_qualification_receipt_file_sha256: str
    runtime_qualification_receipt_body_sha256: str
    host_provisioning_receipt_file_sha256: str
    host_provisioning_receipt_body_sha256: str
    host_executor_descriptor_sha256: str
    host_executor_source_sha256: str
    full_resource_merger_descriptor_sha256: str
    full_resource_merger_source_sha256: str
    algorithmic_resource_contract_descriptor_sha256: str
    algorithmic_resource_contract_source_sha256: str
    storage_boundary_contract_descriptor_sha256: str
    storage_boundary_contract_source_sha256: str
    normalized_publication_contract_descriptor_sha256: str
    normalized_publication_contract_source_sha256: str
    all_case_sequence_intent_file_sha256: str
    all_case_sequence_intent_body_sha256: str
    all_case_sequence_receipt_file_sha256: str
    all_case_sequence_receipt_body_sha256: str
    all_case_sequence_cases_inventory_sha256: str
    candidate_order_sha256: str
    resource_field_order_sha256: str
    image_id: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "image_id":
                _require_image_id(getattr(self, field_name), "replay pin image ID")
            else:
                _require_sha256(getattr(self, field_name), f"replay pin {field_name}")


def _case_sequence_inventory_sha256(
    cases: tuple[QualificationCaseCandidateV2, ...],
) -> str:
    records: list[dict[str, Any]] = []
    for item in cases:
        host: HostSuccessCandidateV2 | HostTerminalFailureCandidateV2
        terminal_receipt: ArtifactIdentityV2
        if type(item) is QualificationCaseSuccessCandidateV2:
            success_item = item
            host = success_item.host_success
            terminal_receipt = host.execution_receipt
        else:
            failure_item = cast(QualificationCaseTerminalFailureCandidateV2, item)
            host = failure_item.terminal_failure
            if (
                host.failure_receipt_state != "committed"
                or host.terminal_metadata_state != "committed"
                or host.handoff_state != "committed"
            ):
                _fail("batch failure case lacks committed terminal recovery coverage")
            terminal_receipt = host.failure_receipt
        terminal_metadata = host.terminal_metadata
        handoff = host.observation_handoff
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
    return hashlib.sha256(_canonical_json({"cases": records}, newline=False)).hexdigest()


@dataclass(frozen=True, slots=True)
class MatchedV3QualificationObservationCandidateBatchV2:
    """The sole canonical candidate artifact: all 28 terminal case records."""

    campaign_spine: CampaignSpineV2
    cases: tuple[QualificationCaseCandidateV2, ...]
    all_case_sequence_receipt: ArtifactIdentityV2
    all_case_sequence_receipt_intent_file_sha256: str
    all_case_sequence_receipt_intent_body_sha256: str
    all_case_sequence_receipt_campaign_spine_sha256: str
    all_case_sequence_receipt_cases_inventory_sha256: str
    all_case_sequence_receipt_case_count: int
    all_case_sequence_receipt_terminal_coverage_complete: bool

    def __post_init__(self) -> None:
        if type(self.campaign_spine) is not CampaignSpineV2:
            _fail("candidate batch campaign spine type differs")
        if (
            type(self.cases) is not tuple
            or len(self.cases) != len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS)
            or any(
                type(item)
                not in {
                    QualificationCaseSuccessCandidateV2,
                    QualificationCaseTerminalFailureCandidateV2,
                }
                for item in self.cases
            )
        ):
            _fail("candidate batch must contain 28 exact success-or-failure records")
        _require_artifact_schema(
            self.all_case_sequence_receipt,
            ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION,
            "post-case all-case sequence receipt",
        )
        if (
            self.all_case_sequence_receipt_intent_file_sha256
            != self.campaign_spine.all_case_sequence_intent.file_sha256
            or self.all_case_sequence_receipt_intent_body_sha256
            != self.campaign_spine.all_case_sequence_intent.body_sha256
            or self.all_case_sequence_receipt_campaign_spine_sha256
            != self.campaign_spine.body_sha256
            or self.all_case_sequence_receipt_cases_inventory_sha256
            != _case_sequence_inventory_sha256(self.cases)
            or _require_int(
                self.all_case_sequence_receipt_case_count,
                "all-case sequence receipt case count",
                minimum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
                maximum=len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            )
            != len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS)
            or _require_bool(
                self.all_case_sequence_receipt_terminal_coverage_complete,
                "all-case sequence receipt terminal coverage",
            )
            is not True
        ):
            _fail("post-case all-case sequence receipt projections differ")
        _require_all_case_sequence_receipt_v1_identity(
            self.all_case_sequence_receipt,
            all_case_sequence_intent_file_sha256=(
                self.all_case_sequence_receipt_intent_file_sha256
            ),
            all_case_sequence_intent_body_sha256=(
                self.all_case_sequence_receipt_intent_body_sha256
            ),
            campaign_spine_body_sha256=(self.all_case_sequence_receipt_campaign_spine_sha256),
            ordered_terminal_handoff_inventory_sha256=(
                self.all_case_sequence_receipt_cases_inventory_sha256
            ),
            case_count=self.all_case_sequence_receipt_case_count,
            terminal_coverage_complete=(self.all_case_sequence_receipt_terminal_coverage_complete),
        )
        campaign_sha256 = self.campaign_spine.body_sha256
        ticket_files: list[str] = []
        ticket_bodies: list[str] = []
        case_ids: list[str] = []
        manifest_files: list[str] = []
        manifest_bodies: list[str] = []
        publisher_entry_files: list[str] = []
        publisher_entry_bodies: list[str] = []
        seed_case_records: list[str] = []
        seed_derivation_records: list[str] = []
        environment_derivations: list[str] = []
        agent_derivations: list[str] = []
        host_artifact_files: dict[str, list[str]] = {
            role: []
            for role in (
                "request",
                "intent",
                "initial_cgroup_sample",
                "operational_frontier",
                "driver_terminal",
                "lifecycle",
                "terminal_receipt",
                "ready",
                "observer_anchor",
                "go_commitment",
                "cgroup_proof",
                "terminal_metadata",
                "prior_host_execution_receipt",
                "observation_handoff",
                "cleanup_reconciliation",
                "publication_reconciliation_reference",
                "storage_boundary_intent",
                "storage_write_seal",
                "storage_boundary_receipt",
                "endpoint_observer_request",
                "endpoint_observer_receipt",
                "algorithmic_measurement_intent",
                "algorithmic_resource_receipt",
                "runner_execution_receipt",
                "full_merger_receipt",
                "publisher_metadata",
                "publication_commitment_wrapper",
                "publication_reload_validation",
                "native_publication_receipt",
                "precleanup_cgroup_sample",
                "cgroup_kill_receipt",
                "cgroup_empty_observation",
                "container_absence_observation",
                "post_container_remove_cgroup_sample",
                "cgroup_counter_fds_closed_receipt",
                "outer_cgroup_absence_observation",
                *(f"probe_{kind}" for kind in PROBE_KINDS),
            )
        }
        host_artifact_bodies: dict[str, list[str]] = {role: [] for role in host_artifact_files}
        publication_manifest_files: list[str] = []
        publication_manifest_bodies: list[str] = []
        for ordinal, item in enumerate(self.cases):
            spine = item.case_spine
            if (
                spine.case_ordinal != ordinal
                or spine.candidate_id != MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS[ordinal]
                or spine.campaign_spine_sha256 != campaign_sha256
                or spine.plan_issuance_receipt_file_sha256
                != self.campaign_spine.plan_issuance_receipt.file_sha256
                or spine.plan_issuance_receipt_body_sha256
                != self.campaign_spine.plan_issuance_receipt.body_sha256
            ):
                _fail("candidate batch case is reordered or cross-wired from its campaign")
            success_item: QualificationCaseSuccessCandidateV2 | None
            failure_item: QualificationCaseTerminalFailureCandidateV2 | None
            host: HostSuccessCandidateV2 | HostTerminalFailureCandidateV2
            if type(item) is QualificationCaseSuccessCandidateV2:
                success_item = item
                failure_item = None
                host = success_item.host_success
            else:
                success_item = None
                failure_item = cast(QualificationCaseTerminalFailureCandidateV2, item)
                host = failure_item.terminal_failure
            if (
                host.qualification_plan_file_sha256
                != self.campaign_spine.qualification_plan_file_sha256
                or host.qualification_plan_body_sha256
                != self.campaign_spine.qualification_plan_body_sha256
                or host.image_id != self.campaign_spine.image_id
                or host.host_executor != self.campaign_spine.host_executor
            ):
                _fail("candidate batch host record is cross-wired from its campaign")
            if success_item is not None and (
                success_item.resource_merger.merger != self.campaign_spine.full_resource_merger
                or success_item.resource_merger.algorithmic_resource_contract
                != self.campaign_spine.algorithmic_resource_contract
                or success_item.resource_merger.storage_boundary_contract
                != self.campaign_spine.storage_boundary_contract
                or success_item.resource_merger.host_provisioning_receipt
                != self.campaign_spine.host_provisioning_receipt
                or success_item.host_success.host_provisioning_receipt
                != self.campaign_spine.host_provisioning_receipt
            ):
                _fail("success resource merger differs from the campaign resource pins")
            if failure_item is not None and (
                failure_item.terminal_failure.host_provisioning_receipt
                != self.campaign_spine.host_provisioning_receipt
                or failure_item.terminal_failure.algorithmic_resource_contract
                != self.campaign_spine.algorithmic_resource_contract
                or failure_item.terminal_failure.storage_boundary_contract
                != self.campaign_spine.storage_boundary_contract
            ):
                _fail("terminal failure differs from the campaign provisioning pin")

            storage_write_seal: ArtifactIdentityV2 | None
            storage_boundary_receipt: ArtifactIdentityV2 | None
            algorithmic_resource_receipt: ArtifactIdentityV2 | None
            runner_execution_receipt: ArtifactIdentityV2 | None
            publisher_metadata: ArtifactIdentityV2 | None
            publication_commitment_wrapper: ArtifactIdentityV2 | None
            publication_reload_validation: ArtifactIdentityV2 | None
            if success_item is not None:
                success_host = success_item.host_success
                terminal_receipt = success_host.execution_receipt
                prior_host_execution_receipt = None
                cleanup_reconciliation = success_host.cleanup_reconciliation
                publication_reconciliation_reference = None
                storage_boundary_intent = success_item.resource_merger.storage_boundary_intent
                storage_write_seal = success_item.resource_merger.storage_write_seal
                storage_boundary_receipt = success_item.resource_merger.storage_boundary_receipt
                endpoint_observer_request = success_item.resource_merger.endpoint_observer_request
                endpoint_observer_receipt = success_item.resource_merger.endpoint_observer_receipt
                algorithmic_resource_receipt = (
                    success_item.resource_merger.algorithmic_resource_receipt
                )
                algorithmic_measurement_intent = (
                    success_item.resource_merger.algorithmic_measurement_intent
                )
                runner_execution_receipt = success_item.resource_merger.runner_execution_receipt
                full_merger_receipt = success_item.resource_merger.merger_receipt
                publisher_metadata = success_item.publication.publisher_metadata
                publication_commitment_wrapper = (
                    success_item.publication.publication_commitment_wrapper
                )
                publication_reload_validation = (
                    success_item.publication.publication_reload_validation
                )
                native_publication_receipt = success_item.publication.native_publication_receipt
                success_recovery_nodes = {
                    node.node_name: node for node in success_host.recovery_nodes
                }
                precleanup_cgroup_sample = success_recovery_nodes[
                    "precleanup_cgroup_sample"
                ].artifact
                cgroup_kill_receipt = success_recovery_nodes["cgroup_kill"].artifact
                cgroup_empty_observation = success_recovery_nodes["cgroup_empty"].artifact
                container_absence_observation = success_recovery_nodes["container_absence"].artifact
                post_container_remove_cgroup_sample = success_recovery_nodes[
                    "post_container_remove_cgroup_sample"
                ].artifact
                cgroup_counter_fds_closed_receipt = success_recovery_nodes[
                    "cgroup_counter_fds_closed"
                ].artifact
                outer_cgroup_absence_observation = success_recovery_nodes[
                    "outer_cgroup_absence"
                ].artifact
            else:
                assert failure_item is not None
                failure_host = failure_item.terminal_failure
                if (
                    failure_host.failure_receipt_state != "committed"
                    or failure_host.terminal_metadata_state != "committed"
                    or failure_host.handoff_state != "committed"
                ):
                    _fail("terminal failure lacks committed receipt, terminal, or handoff")
                terminal_receipt = failure_host.failure_receipt
                prior_host_execution_receipt = failure_host.prior_host_execution_receipt
                cleanup_reconciliation = failure_host.cleanup_reconciliation
                publication_reconciliation_reference = (
                    failure_host.publication_reconciliation_reference
                )
                storage_boundary_intent = failure_host.storage_boundary_intent
                storage_write_seal = failure_host.storage_write_seal
                storage_boundary_receipt = failure_host.storage_boundary_receipt
                endpoint_observer_request = None
                endpoint_observer_receipt = None
                algorithmic_resource_receipt = failure_host.algorithmic_resource_receipt
                algorithmic_measurement_intent = failure_host.algorithmic_measurement_intent
                failure_publication_projection = failure_host.failure_publication_projection
                runner_execution_receipt = (
                    None
                    if failure_publication_projection is None
                    else failure_publication_projection.runner_execution_receipt
                )
                full_merger_receipt = None
                publisher_metadata = (
                    None
                    if failure_publication_projection is None
                    else failure_publication_projection.publisher_metadata
                )
                publication_commitment_wrapper = failure_host.publication_commitment_wrapper
                publication_reload_validation = failure_host.publication_reload_validation
                native_publication_receipt = failure_host.native_publication_receipt
                precleanup_cgroup_sample = failure_host.precleanup_cgroup_sample
                cgroup_kill_receipt = failure_host.cgroup_kill_receipt
                cgroup_empty_observation = failure_host.cgroup_empty_observation
                container_absence_observation = failure_host.container_absence_observation
                post_container_remove_cgroup_sample = (
                    failure_host.post_container_remove_cgroup_sample
                )
                cgroup_counter_fds_closed_receipt = failure_host.cgroup_counter_fds_closed_receipt
                outer_cgroup_absence_observation = failure_host.outer_cgroup_absence_observation
            role_artifacts: dict[str, ArtifactIdentityV2 | None] = {
                "request": host.request,
                "intent": host.intent,
                "initial_cgroup_sample": host.initial_cgroup_sample,
                "operational_frontier": host.operational_frontier,
                "driver_terminal": host.driver_terminal,
                "lifecycle": host.lifecycle,
                "terminal_receipt": terminal_receipt,
                "ready": host.ready,
                "observer_anchor": host.observer_anchor,
                "go_commitment": host.go_commitment,
                "cgroup_proof": host.cgroup_proof,
                "terminal_metadata": host.terminal_metadata,
                "prior_host_execution_receipt": prior_host_execution_receipt,
                "observation_handoff": host.observation_handoff,
                "cleanup_reconciliation": cleanup_reconciliation,
                "publication_reconciliation_reference": publication_reconciliation_reference,
                "storage_boundary_intent": storage_boundary_intent,
                "storage_write_seal": storage_write_seal,
                "storage_boundary_receipt": storage_boundary_receipt,
                "endpoint_observer_request": endpoint_observer_request,
                "endpoint_observer_receipt": endpoint_observer_receipt,
                "algorithmic_resource_receipt": algorithmic_resource_receipt,
                "algorithmic_measurement_intent": algorithmic_measurement_intent,
                "runner_execution_receipt": runner_execution_receipt,
                "full_merger_receipt": full_merger_receipt,
                "publisher_metadata": publisher_metadata,
                "publication_commitment_wrapper": publication_commitment_wrapper,
                "publication_reload_validation": publication_reload_validation,
                "native_publication_receipt": native_publication_receipt,
                "precleanup_cgroup_sample": precleanup_cgroup_sample,
                "cgroup_kill_receipt": cgroup_kill_receipt,
                "cgroup_empty_observation": cgroup_empty_observation,
                "container_absence_observation": container_absence_observation,
                "post_container_remove_cgroup_sample": post_container_remove_cgroup_sample,
                "cgroup_counter_fds_closed_receipt": cgroup_counter_fds_closed_receipt,
                "outer_cgroup_absence_observation": outer_cgroup_absence_observation,
            }
            if success_item is not None:
                for probe in success_item.probes:
                    role_artifacts[f"probe_{probe.probe_kind}"] = ArtifactIdentityV2(
                        schema_version=probe.schema_version,
                        file_sha256=probe.file_sha256,
                        body_sha256=probe.body_sha256,
                    )
                publication_manifest_files.append(
                    success_item.publication.publication_manifest_file_sha256
                )
                publication_manifest_bodies.append(
                    success_item.publication.publication_manifest_body_sha256
                )
            elif failure_host.failure_publication_projection is not None:
                publication_manifest_files.append(
                    failure_host.failure_publication_projection.publication_manifest_file_sha256
                )
                publication_manifest_bodies.append(
                    failure_host.failure_publication_projection.publication_manifest_body_sha256
                )
            for role, artifact in role_artifacts.items():
                if artifact is not None:
                    host_artifact_files[role].append(artifact.file_sha256)
                    host_artifact_bodies[role].append(artifact.body_sha256)
            ticket_files.append(spine.case_execution_ticket.file_sha256)
            ticket_bodies.append(spine.case_execution_ticket.body_sha256)
            case_ids.append(spine.qualification_case_id)
            manifest_files.append(spine.qualification_case_manifest.file_sha256)
            manifest_bodies.append(spine.qualification_case_manifest.body_sha256)
            publisher_entry_files.append(spine.publisher_registry_entry.file_sha256)
            publisher_entry_bodies.append(spine.publisher_registry_entry.body_sha256)
            seed_case_records.append(spine.seed_case_record_sha256)
            seed_derivation_records.append(spine.seed_derivation_record_sha256)
            environment_derivations.append(spine.environment_derivation_sha256)
            agent_derivations.append(spine.agent_derivation_sha256)
        if (
            len(set(ticket_files)) != len(ticket_files)
            or len(set(ticket_bodies)) != len(ticket_bodies)
            or len(set(case_ids)) != len(case_ids)
            or len(set(manifest_files)) != len(manifest_files)
            or len(set(manifest_bodies)) != len(manifest_bodies)
            or len(set(publisher_entry_files)) != len(publisher_entry_files)
            or len(set(publisher_entry_bodies)) != len(publisher_entry_bodies)
            or len(set(seed_case_records)) != len(seed_case_records)
            or len(set(seed_derivation_records)) != len(seed_derivation_records)
            or len(set(environment_derivations)) != len(environment_derivations)
            or len(set(agent_derivations)) != len(agent_derivations)
            or len(set(publication_manifest_files)) != len(publication_manifest_files)
            or len(set(publication_manifest_bodies)) != len(publication_manifest_bodies)
            or any(
                len(set(values)) != len(values)
                for values in (
                    *host_artifact_files.values(),
                    *host_artifact_bodies.values(),
                )
            )
        ):
            _fail("candidate batch reuses one per-case registry identity")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (QUALIFICATION_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION),
            "status": QUALIFICATION_OBSERVATION_REGISTRY_V2_STATUS,
            "classification": QUALIFICATION_OBSERVATION_REGISTRY_V2_CLASSIFICATION,
            "registry_descriptor_sha256": (QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256),
            "campaign_spine": self.campaign_spine.to_dict(),
            "cases": [item.to_dict() for item in self.cases],
            "all_case_sequence_receipt": self.all_case_sequence_receipt.to_dict(),
            "all_case_sequence_receipt_intent_file_sha256": (
                self.all_case_sequence_receipt_intent_file_sha256
            ),
            "all_case_sequence_receipt_intent_body_sha256": (
                self.all_case_sequence_receipt_intent_body_sha256
            ),
            "all_case_sequence_receipt_campaign_spine_sha256": (
                self.all_case_sequence_receipt_campaign_spine_sha256
            ),
            "all_case_sequence_receipt_cases_inventory_sha256": (
                self.all_case_sequence_receipt_cases_inventory_sha256
            ),
            "all_case_sequence_receipt_case_count": (self.all_case_sequence_receipt_case_count),
            "all_case_sequence_receipt_terminal_coverage_complete": (
                self.all_case_sequence_receipt_terminal_coverage_complete
            ),
            "terminal_coverage": {
                "required_case_count": len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
                "same_case_retry_permitted": False,
                "selective_omission_permitted": False,
                "metadata_based_ordering_permitted": False,
                "terminal_failure_is_clean_rejection": False,
            },
            "claims": _claims(),
            "readiness": _readiness(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "batch_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


def _registry_descriptor() -> dict[str, Any]:
    publication_contracts = []
    for family in ("local", "external", "adapter"):
        paths, metadata_schema, descriptor_schema = _publication_profile(family)
        native_receipt_schema = (
            None
            if family == "local"
            else EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
            if family == "external"
            else STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
        )
        native_atomic_descriptor_schema = (
            STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            if family == "adapter"
            else ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        )
        publication_contracts.append(
            {
                "family": family,
                "publisher_descriptor_schema_version": descriptor_schema,
                "publisher_metadata_schema_version": metadata_schema,
                "native_atomic_producer_descriptor_schema_version": (
                    native_atomic_descriptor_schema
                ),
                "native_publication_receipt_schema_version": native_receipt_schema,
                "role_paths": [{"role": role, "path": path} for role, path in paths],
                "file_count": len(paths),
            }
        )
    return {
        "schema_version": QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION,
        "status": QUALIFICATION_OBSERVATION_REGISTRY_V2_STATUS,
        "classification": QUALIFICATION_OBSERVATION_REGISTRY_V2_CLASSIFICATION,
        "batch_contract": {
            "schema_version": (QUALIFICATION_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION),
            "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
            "full_file_caller_pin_required": True,
            "batch_only_public_serialization": True,
            "per_case_public_artifacts": False,
            "required_case_count": len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            "one_terminal_tag_per_ordinal": True,
            "success_tag_is_decision": False,
            "terminal_failure_tag_is_clean_rejection": False,
            "same_case_retry_permitted": False,
            "selective_omission_permitted": False,
            "pre_execution_sequence_intent_schema_version": (
                ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION
            ),
            "pre_execution_sequence_intent_claims_completion": False,
            "post_case_sequence_receipt_schema_version": (ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION),
            "post_case_sequence_receipt_binds_campaign_spine": True,
            "post_case_sequence_receipt_binds_ordered_terminal_inventory": True,
            "post_case_sequence_receipt_binds_batch_file_or_body": False,
            "sequence_intent_to_terminal_cases_to_sequence_receipt": True,
        },
        "plan_v3_literal_binding": {
            "schema_version": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
            "candidate_order": list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            "candidate_order_sha256": _ordered_values_sha256(
                MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS
            ),
            "families": {
                "local": list(MATCHED_V3_LOCAL_CANDIDATE_IDS),
                "external": list(MATCHED_V3_EXTERNAL_CANDIDATE_IDS),
                "adapter": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
            },
            "resource_fields": list(RESOURCE_CEILING_FIELDS),
            "resource_field_order_sha256": _ordered_values_sha256(RESOURCE_CEILING_FIELDS),
            "plan_descriptor_pin_embedded": False,
            "plan_source_pin_embedded": False,
            "plan_artifact_caller_pins_required": True,
        },
        "seed_identity_contract": {
            "pulse_record_schema_version": (QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION),
            "trust_root_receipt_schema_version": (
                QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION
            ),
            "seed_registry_schema_version": QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
            "runtime_verifier_descriptor_schema_version": (
                QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "runtime_verifier_receipt_schema_version": (QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION),
            "verifier_binary_pin_required": True,
            "raw_numeric_seeds_retained": False,
            "numeric_seed_inequality_required": False,
            "commitment_inequality_required": False,
            "case_derivation_identities_unique_per_role": True,
            "cross_role_derivation_digest_inequality_required": False,
        },
        "source_build_supply_chain_contract": {
            "local_source_schema_version": LOCAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "external_source_schema_version": EXTERNAL_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "adapter_source_schema_version": ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION,
            "joint_source_closure_schema_version": (JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION),
            "sealed_staging_schema_version": SEALED_STAGING_CANDIDATE_SCHEMA_VERSION,
            "fresh_build_schema_version": FRESH_BUILD_CANDIDATE_SCHEMA_VERSION,
            "exact_dag": (
                "local_external_adapter_sources_to_joint_closure_to_sealed_"
                "staging_to_fresh_build_to_image"
            ),
            "joint_closure_binds_all_three_source_file_and_body_identities": True,
            "sealed_staging_binds_joint_closure_file_and_body": True,
            "fresh_build_binds_sealed_staging_file_and_body": True,
            "fresh_build_binds_exact_image_id": True,
            "historical_build_or_image_can_substitute": False,
            "production_source_closure_available": False,
            "production_sealed_staging_available": False,
            "production_fresh_build_available": False,
        },
        "host_contract": {
            "production_executor_descriptor_schema_version": (
                PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION
            ),
            "success_receipt_schema_version": HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
            "failure_receipt_schema_version": HOST_FAILURE_RECEIPT_SCHEMA_VERSION,
            "request_schema_version": HOST_CASE_REQUEST_SCHEMA_VERSION,
            "intent_schema_version": HOST_CASE_INTENT_SCHEMA_VERSION,
            "ready_schema_version": HOST_READY_SCHEMA_VERSION,
            "observer_anchor_schema_version": HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            "go_schema_version": HOST_GO_SCHEMA_VERSION,
            "operational_frontier_schema_version": HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "initial_cgroup_sample_schema_version": HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            "driver_terminal_schema_version": IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION,
            "lifecycle_schema_version": HOST_LIFECYCLE_SCHEMA_VERSION,
            "terminal_schema_version": HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            "observation_handoff_schema_version": (HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION),
            "provisioning_schema_version": HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "cgroup_proof_schema_version": HOST_CGROUP_PROOF_SCHEMA_VERSION,
            "incompatible_v1_schema_versions": list(INCOMPATIBLE_HOST_V1_SCHEMA_VERSIONS),
            "legacy_handoff_schema_version": (INCOMPATIBLE_HOST_HANDOFF_V1_SCHEMA_VERSION),
            "legacy_handoff_compatible": False,
            "success_handoff_binds_execution_receipt_file_and_body": True,
            "success_handoff_binds_terminal_metadata_file_and_body": True,
            "failure_handoff_binds_failure_receipt_file_and_body": True,
            "failure_handoff_binds_terminal_metadata_file_and_body": True,
            "terminal_metadata_envelope_location": "outer_host",
            "terminal_metadata_envelope_common_to_success_and_failure": True,
            "terminal_metadata_envelope_canonical": True,
            "terminal_metadata_is_in_container_record": False,
            "failure_receipt_is_distinct_from_terminal_metadata": True,
            "failure_receipt_can_alias_terminal_metadata": False,
            "batch_failure_requires_committed_receipt_terminal_and_handoff": True,
            "failure_states_derived_from_operational_frontier_and_cleanup_dag": True,
            "failure_operational_phases": list(HOST_OPERATIONAL_PHASES),
            "failure_recovery_nodes": list(HOST_RECOVERY_NODE_NAMES),
            "failure_recovery_node_states": list(HOST_RECOVERY_NODE_STATES),
            "failure_recovery_node_dependencies": {
                name: list(HOST_RECOVERY_NODE_DEPENDENCIES[name])
                for name in HOST_RECOVERY_NODE_NAMES
            },
            "cleanup_nodes_not_applicable_only_when_fresh_cgroup_cannot_exist": True,
            "lifecycle_phases": list(HOST_LIFECYCLE_PHASES),
            "phase_failure_side_effect_states": list(HOST_OPERATIONAL_FAILURE_EFFECT_STATES),
            "failed_before_commit_means_no_committed_side_effect": True,
            "failure_phase_excluded_from_completed_prefix": True,
            "authorization_ack_carried_by_request": True,
            "separate_authorization_artifact": False,
            "intent_states": list(HOST_PHASE_STATES),
            "intent_artifact_present_only_when_committed": True,
            "initial_cgroup_sample_carries_retained_fd_inventory": True,
            "separate_retained_fd_inventory_artifact": False,
            "container_create_start_and_workload_boundaries_live_in_operational_frontier": True,
            "intent_commit_uncertainty_consumes_and_quarantines_ticket": True,
            "intent_commit_uncertainty_reconciliation_only": True,
            "uncertainty_dimensions": list(HOST_UNCERTAINTY_DIMENSIONS),
            "simultaneous_uncertainty_dimensions_permitted": True,
            "ready_contains_host_sample": False,
            "observer_anchor_required_before_go": True,
            "ready_and_observer_anchor_identities_distinct": True,
            "go_binds_ready_file_and_body": True,
            "go_binds_observer_anchor_file_and_body": True,
            "success_exact_counts": {
                "container_create_count": 1,
                "container_start_count": 1,
                "go_commit_count": 1,
                "workload_start_count": 1,
                "workload_exit_count": 1,
                "attempt_count": 1,
                "failure_count": 0,
            },
            "operational_failure_count_independent_of_recovery_failures": True,
            "recovery_only_failure_operational_failure_count": 0,
            "recovery_failure_and_uncertainty_counts_separate": True,
            "failure_counts_are_typed_exact_or_uncertain": True,
            "failure_committed_phase_requires_artifact": True,
            "failure_uncertain_phase_forbids_commit_claim": True,
            "failure_algorithmic_storage_terminal_case_projections_required": True,
            "terminal_metadata_body_keys": list(HOST_TERMINAL_METADATA_V2_BODY_KEYS),
            "terminal_timed_out_is_exact_boolean_for_both_record_kinds": True,
            "terminal_failure_error_message_sha256_required": True,
            "terminal_metadata_does_not_reverse_pin_lifecycle": True,
            "acyclic_terminalization_order": (
                "operational_frontier_to_cleanup_reconciliation_to_terminal_metadata_to_"
                "lifecycle_to_receipt_to_handoff"
            ),
            "failure_storage_receipt_binds_write_seal_file_and_body": True,
            "publication_attempt_requires_precommitted_expected_address": True,
            "publication_attempt_requires_reconciliation_key_and_reference": True,
            "reload_validation_schema_version": (
                QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION
            ),
            "cleanup_reconciliation_schema_version": (HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION),
            "cleanup_node_artifact_schema_versions": {
                "precleanup_cgroup_sample": (HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION),
                "cgroup_kill": HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
                "cgroup_empty": (HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION),
                "container_absence": HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
                "post_container_remove_cgroup_sample": (
                    HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION
                ),
                "cgroup_counter_fds_closed": (
                    HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION
                ),
                "outer_cgroup_absence": (HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION),
                "final_cgroup_proof": HOST_CGROUP_PROOF_SCHEMA_VERSION,
            },
            "cleanup_reconciliation_canonical_body_and_file": True,
            "cleanup_reconciliation_authenticates_ordered_nodes_states_artifacts_deps": True,
            "cleanup_container_branch_has_no_hidden_initial_sample_dependency": True,
            "cleanup_residual_uncertainty_does_not_block_terminalization": True,
            "full_cgroup_proof_only_after_all_cleanup_dependencies": True,
            "production_containment_qualification_alternatives": [
                "event_complete_cgroup_membership_and_migration_evidence",
                ("exclusive_host_provisioning_denies_cgroup_delegation_and_privileged_movers"),
            ],
            "current_v1_containment_qualified": False,
            "same_case_retry_permitted": False,
            "terminal_failure_is_clean_rejection": False,
            "success_terminal_projection_exact": True,
        },
        "resource_merger_contract": {
            "producer_descriptor_schema_version": (FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION),
            "candidate_schema_version": (QUALIFICATION_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION),
            "receipt_schema_version": FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION,
            "resource_fields": list(RESOURCE_CEILING_FIELDS),
            "provenance_kinds": list(RESOURCE_PROVENANCE_KINDS),
            "endpoint_observer_can_substitute": False,
            "endpoint_request_schema_version": ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION,
            "endpoint_receipt_schema_version": ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION,
            "plan_v3_endpoint_producer_available": False,
            "endpoint_corroboration_required": False,
            "endpoint_corroboration_authoritative": False,
            "endpoint_absence_blocks_merger": False,
            "host_terminal_pin_required": True,
            "host_observation_handoff_pin_required": True,
            "merger_is_external_and_opaque_to_this_registry": True,
            "merger_internal_authentication_performed_by_this_registry": False,
            "merger_receipt_and_producer_identity_consumed": True,
            "algorithmic_family_receipt_required": True,
            "algorithmic_contract_descriptor_schema_version": (
                ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
            ),
            "algorithmic_contract_descriptor_sha256": (
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256
            ),
            "algorithmic_contract_source_sha256": (
                FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256
            ),
            "algorithmic_measurement_intent_schema_version": (
                ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION
            ),
            "algorithmic_intent_bound_in_request_and_ready": True,
            "algorithmic_intent_precedes_go": True,
            "algorithmic_intent_contains_observed_values": False,
            "algorithmic_receipt_bound_at_terminal": True,
            "runner_execution_receipt_schema_by_candidate_class": {
                "local": LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION,
                "external": EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
                "adapted_full_rainbow": FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION,
                "adapted_ppo_gru": PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
            },
            "runner_execution_receipt_extracted_from_algorithmic_receipt": True,
            "runner_execution_receipt_shared_with_publisher_metadata": True,
            "runner_execution_receipt_crosslinked_to_publication": True,
            "storage_boundary_receipt_schema_version": (
                QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION
            ),
            "storage_boundary_contract_descriptor_schema_version": (
                QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
            ),
            "storage_boundary_contract_descriptor_sha256": (
                FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256
            ),
            "storage_boundary_contract_source_sha256": (
                FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256
            ),
            "storage_boundary_intent_schema_version": (
                QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION
            ),
            "storage_write_seal_schema_version": (QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION),
            "storage_intent_bound_at_ready": True,
            "storage_write_seal_precedes_storage_receipt": True,
            "storage_write_seal_binds_reload_validation_file_and_body": True,
            "storage_receipt_bound_at_terminal": True,
            "storage_receipt_precedes_terminal_metadata_emission": True,
            "terminal_metadata_is_sole_post_receipt_delivery_emission_evidence": True,
            "receipt_reverse_pins_to_terminal_or_merger": False,
            "storage_boundary_receipt_producer_available": False,
            "storage_exact_mode_requires_event_complete_high_water": True,
            "storage_bound_mode_requires_non_bypass_quota_proof": True,
            "runner_algorithmic_receipt_can_assert_storage_peaks": False,
            "all_28_records_mirrored_in_candidate": True,
            "missing_values_synthesized": False,
            "unobserved_values_default_to_zero": False,
            "default_value_semantics": "exact_observation",
            "max_peak_rss_value_semantics": "conservative_observed_upper_bound",
            "max_peak_rss_observed_quantity": "fresh_cgroup_memory_charge_high_water",
            "max_thread_count_value_semantics": "conservative_observed_upper_bound",
            "max_thread_count_observed_quantity": "fresh_cgroup_linux_task_high_water",
            "storage_value_semantics_allowed": [
                "exact_observation",
                "conservative_enforced_upper_bound",
            ],
            "cpu_delta_envelope": "fresh_cgroup_initial_empty_to_post_container_remove",
            "wall_delta_envelope": "fresh_cgroup_initial_empty_to_post_container_remove",
            "periodic_poll_peak_sufficient": False,
            "within_ceiling_decision_made": False,
            "production_receipt_available": False,
        },
        "publication_contracts": publication_contracts,
        "publication_normalization": {
            "wrapper_schema_version": (NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION),
            "wrapper_canonical_builder_implemented": True,
            "wrapper_production_available": False,
            "wrapper_status": NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_STATUS,
            "wrapper_classification": NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_CLASSIFICATION,
            "wrapper_contract_descriptor_sha256": (FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256),
            "wrapper_contract_source_sha256": FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256,
            "wrapper_replaces_native_family_receipt": False,
            "wrapper_binds_case_spine": True,
            "publication_consumes_exact_host_observation_handoff": True,
            "wrapper_binds_candidate_id_and_family": True,
            "wrapper_binds_outer_publisher_descriptor_and_source": True,
            "wrapper_binds_publisher_metadata_file_and_body": True,
            "wrapper_binds_native_producer_descriptor_and_source": True,
            "wrapper_binds_native_receipt_file_and_body_when_present": True,
            "wrapper_binds_publication_address": True,
            "wrapper_binds_file_inventory": True,
            "wrapper_binds_published_bundle": True,
            "wrapper_precommits_expected_reload_observation_digest": True,
            "wrapper_does_not_claim_reload_performed": True,
            "reload_validation_binds_wrapper_file_and_body": True,
            "reload_validation_binds_expected_and_actual_reload_digests": True,
            "reload_validation_requires_expected_equals_actual": True,
            "reload_validation_requires_performed_and_read_only_facts": True,
            "reload_validation_canonical_encoding": {
                "schema_version": (QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION),
                "body_projection_keys": list(
                    QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_KEYS
                ),
                "body_json_encoding": "compact_sorted_ascii_without_lf",
                "body_sha256_field": (
                    QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_SHA256_FIELD
                ),
                "file_projection_is_body_plus_body_sha256_field": True,
                "file_json_encoding": "compact_sorted_ascii_with_exactly_one_trailing_lf",
                "file_and_body_sha256_independently_derived": True,
            },
            "reload_validation_precedes_storage_write_seal": True,
            "wrapper_binds_video_slot_mode": True,
            "wrapper_canonical_encoding": {
                "body_projection_keys": list(NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_KEYS),
                "body_json_encoding": "compact_sorted_ascii_without_lf",
                "body_sha256_field": (NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_SHA256_FIELD),
                "file_projection_is_body_plus_body_sha256_field": True,
                "file_json_encoding": "compact_sorted_ascii_with_exactly_one_trailing_lf",
                "file_and_body_sha256_independently_derived": True,
            },
            "failure_projection_schema_version": (
                QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION
            ),
            "failure_projection_body_sha256_field": (
                QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD
            ),
            "failure_projection_required_for_committed_wrapper": True,
            "failure_projection_binds_runner_and_algorithmic_receipts": True,
            "failure_projection_binds_reconciliation_reference": True,
            "failure_projection_replays_canonical_wrapper_identity": True,
            "local_native_form": "atomic_producer_descriptor_and_source",
            "external_native_form": "external_atomic_publication_receipt",
            "adapter_native_form": "future_strict_adapter_atomic_receipt",
            "maximum_file_bytes": MAX_PUBLICATION_FILE_BYTES,
            "maximum_total_bytes": MAX_PUBLICATION_TOTAL_BYTES,
        },
        "strict_json": {
            "duplicate_keys_allowed": False,
            "floats_allowed": False,
            "nonfinite_numbers_allowed": False,
            "unknown_keys_allowed": False,
            "noncanonical_bytes_allowed": False,
            "container_aliases_or_cycles_allowed": False,
            "recursive_normalized_forbidden_key_policy": True,
            "maximum_depth": _MAX_JSON_DEPTH,
            "maximum_nodes": _MAX_JSON_NODES,
            "maximum_text_length": _MAX_TEXT_LENGTH,
            "maximum_artifact_bytes": _MAX_ARTIFACT_BYTES,
        },
        "pin_graph": {
            "registry_descriptor_literal_pin_required": True,
            "registry_source_self_pin_embedded": False,
            "registry_source_independent_caller_pin_required": True,
            "plan_descriptor_or_source_pin_embedded": False,
            "campaign_spine_caller_pins_replayed": True,
            "one_way_no_source_hash_cycle": True,
            "artifact_direction": (
                "intents_to_receipts_publication_and_seal_to_terminal_to_"
                "post_container_remove_cgroup_to_host_receipt_to_merger"
            ),
            "receipt_reverse_pins_forbidden": True,
        },
        "explicit_incompatibilities": {
            "nonexecuting_host_descriptor_sha256": (NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256),
            "nonexecuting_host_source_sha256": (NONEXECUTING_HOST_EXECUTOR_SOURCE_SHA256),
            "source_only_quicknet_descriptor_sha256": (
                SOURCE_ONLY_QUICKNET_VERIFIER_DESCRIPTOR_SHA256
            ),
            "source_only_quicknet_source_sha256": (SOURCE_ONLY_QUICKNET_VERIFIER_SOURCE_SHA256),
            "source_materialization_quicknet_descriptor_sha256": (
                SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256
            ),
            "source_materialization_quicknet_source_sha256": (
                SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256
            ),
            "source_only_quicknet_build_descriptor_sha256": (
                SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256
            ),
            "source_only_quicknet_build_source_sha256": (SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256),
            "historical_algorithmic_validator_descriptor_sha256s": list(
                HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256S
            ),
            "historical_algorithmic_validator_source_sha256s": list(
                HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256S
            ),
            "algorithmic_validator_cross_kind_merger_substitution_rejected": True,
            "adapter_descriptor_sha256s": list(INCOMPATIBLE_ADAPTER_DESCRIPTOR_SHA256S),
            "adapter_source_sha256s": list(INCOMPATIBLE_ADAPTER_SOURCE_SHA256S),
            "adapter_cross_kind_identity_substitution_rejected": True,
            "historical_image_ids": list(HISTORICAL_IMAGE_IDS),
        },
        "capabilities": _capabilities(),
        "claims": _claims(),
        "readiness": _readiness(),
        "limitations": _limitations(),
    }


_REGISTRY_DESCRIPTOR_BYTES: Final = _canonical_json(_registry_descriptor())

# Root replaces this literal only after the implementation and tests are statically frozen.
QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256: Final = "0" * 64


def _require_registry_descriptor_pin() -> str:
    pinned = QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256
    if pinned == "0" * 64:
        _fail("qualification observation registry v2 descriptor pin is not finalized")
    _require_sha256(pinned, "qualification observation registry v2 descriptor")
    observed = hashlib.sha256(_REGISTRY_DESCRIPTOR_BYTES).hexdigest()
    if not hmac.compare_digest(observed, pinned):
        _fail("qualification observation registry v2 descriptor identity drifted")
    return pinned


def _require_body_digest(
    value: dict[str, Any],
    field_name: str,
    label: str,
) -> dict[str, Any]:
    body = dict(value)
    supplied = _require_sha256(body.pop(field_name, None), f"{label} body")
    expected = _body_sha256(body)
    if not hmac.compare_digest(supplied, expected):
        _fail(f"{label} body digest differs")
    return body


def _require_false_mapping(value: object, expected: dict[str, bool], label: str) -> None:
    exact = _require_exact_keys(value, frozenset(expected), label)
    if not _exact_json_equal(exact, expected) or any(item is not False for item in exact.values()):
        _fail(f"{label} became true or differs")


def _artifact_identity_from_dict(value: object, label: str) -> ArtifactIdentityV2:
    item = _require_exact_keys(
        value,
        frozenset(ArtifactIdentityV2.__dataclass_fields__),
        label,
    )
    return ArtifactIdentityV2(**item)


def _optional_artifact_identity_from_dict(
    value: object,
    label: str,
) -> ArtifactIdentityV2 | None:
    if value is None:
        return None
    return _artifact_identity_from_dict(value, label)


def _producer_identity_from_dict(value: object, label: str) -> ProducerIdentityV2:
    item = _require_exact_keys(
        value,
        frozenset(ProducerIdentityV2.__dataclass_fields__),
        label,
    )
    return ProducerIdentityV2(**item)


def _campaign_spine_from_dict(value: object) -> CampaignSpineV2:
    keys = frozenset(
        {
            *CampaignSpineV2.__dataclass_fields__,
            "candidate_order",
            "resource_fields",
            "campaign_spine_body_sha256",
        }
    )
    exact = _require_exact_keys(value, keys, "campaign spine")
    item = _require_body_digest(exact, "campaign_spine_body_sha256", "campaign spine")
    if not _exact_json_equal(
        item.pop("candidate_order"),
        list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
    ):
        _fail("campaign spine candidate order differs")
    if not _exact_json_equal(item.pop("resource_fields"), list(RESOURCE_CEILING_FIELDS)):
        _fail("campaign spine resource-field order differs")
    for field_name in (
        "plan_issuance_receipt",
        "case_ticket_registry",
        "publisher_registry",
        "seed_registry",
        "seed_pulse_record",
        "seed_trust_root_receipt",
        "quicknet_verifier_receipt",
        "seed_chronology_receipt",
        "local_source_candidate",
        "external_source_candidate",
        "adapter_source_candidate",
        "joint_source_closure_candidate",
        "sealed_staging_candidate",
        "fresh_build_candidate",
        "runtime_candidate",
        "runtime_qualification_receipt",
        "host_provisioning_receipt",
        "all_case_sequence_intent",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"campaign spine {field_name}",
        )
    for field_name in (
        "quicknet_verifier",
        "host_executor",
        "full_resource_merger",
        "algorithmic_resource_contract",
        "storage_boundary_contract",
    ):
        item[field_name] = _producer_identity_from_dict(
            item[field_name],
            f"campaign spine {field_name}",
        )
    result = CampaignSpineV2(**item)
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("campaign spine canonical reconstruction differs")
    return result


def _case_spine_from_dict(value: object) -> CaseSpineV2:
    keys = frozenset({*CaseSpineV2.__dataclass_fields__, "case_spine_body_sha256"})
    exact = _require_exact_keys(value, keys, "case spine")
    item = _require_body_digest(exact, "case_spine_body_sha256", "case spine")
    for field_name in (
        "qualification_case_manifest",
        "case_execution_ticket",
        "publisher_registry_entry",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"case spine {field_name}",
        )
    result = CaseSpineV2(**item)
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("case spine canonical reconstruction differs")
    return result


def _probe_candidate_from_dict(value: object) -> ProbeCandidateV2:
    item = _require_exact_keys(
        value,
        frozenset(ProbeCandidateV2.__dataclass_fields__),
        "probe candidate",
    )
    return ProbeCandidateV2(**item)


def _publication_file_from_dict(value: object) -> PublicationFileCandidateV2:
    item = _require_exact_keys(
        value,
        frozenset(PublicationFileCandidateV2.__dataclass_fields__),
        "publication file candidate",
    )
    return PublicationFileCandidateV2(**item)


def _publication_candidate_from_dict(value: object) -> PublicationCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(PublicationCandidateV2.__dataclass_fields__),
        "publication candidate",
    )
    item = dict(exact)
    item["publisher_metadata"] = _artifact_identity_from_dict(
        item["publisher_metadata"],
        "publication publisher metadata",
    )
    item["host_observation_handoff"] = _artifact_identity_from_dict(
        item["host_observation_handoff"],
        "publication host observation handoff",
    )
    item["runner_execution_receipt"] = _artifact_identity_from_dict(
        item["runner_execution_receipt"],
        "publication runner-execution receipt",
    )
    item["publisher"] = _producer_identity_from_dict(
        item["publisher"],
        "publication producer",
    )
    item["native_atomic_producer"] = _producer_identity_from_dict(
        item["native_atomic_producer"],
        "publication native atomic producer",
    )
    item["native_publication_receipt"] = _optional_artifact_identity_from_dict(
        item["native_publication_receipt"],
        "publication native receipt",
    )
    item["publication_commitment_wrapper"] = _artifact_identity_from_dict(
        item["publication_commitment_wrapper"],
        "publication commitment wrapper",
    )
    item["publication_reload_validation"] = _artifact_identity_from_dict(
        item["publication_reload_validation"],
        "publication reload validation",
    )
    raw_files = item.pop("files")
    if type(raw_files) is not list:
        _fail("publication files must be one exact list")
    result = PublicationCandidateV2(
        **item,
        files=tuple(_publication_file_from_dict(child) for child in raw_files),
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("publication candidate canonical reconstruction differs")
    return result


def _resource_field_from_dict(value: object) -> ResourceFieldCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(ResourceFieldCandidateV2.__dataclass_fields__),
        "resource field candidate",
    )
    item = dict(exact)
    item["provenance_receipt"] = _artifact_identity_from_dict(
        item["provenance_receipt"],
        "resource field provenance receipt",
    )
    return ResourceFieldCandidateV2(**item)


def _resource_merger_from_dict(value: object) -> ResourceMergerCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(ResourceMergerCandidateV2.__dataclass_fields__),
        "resource merger candidate",
    )
    item = dict(exact)
    for field_name in (
        "host_provisioning_receipt",
        "host_cgroup_proof",
        "host_terminal_metadata",
        "host_observation_handoff",
        "algorithmic_measurement_intent",
        "algorithmic_resource_receipt",
        "runner_execution_receipt",
        "storage_boundary_intent",
        "storage_write_seal",
        "storage_boundary_receipt",
        "merger_receipt",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"resource merger {field_name}",
        )
    for field_name in ("endpoint_observer_request", "endpoint_observer_receipt"):
        item[field_name] = _optional_artifact_identity_from_dict(
            item[field_name],
            f"resource merger {field_name}",
        )
    item["merger"] = _producer_identity_from_dict(
        item["merger"],
        "resource merger producer",
    )
    for field_name in ("algorithmic_resource_contract", "storage_boundary_contract"):
        item[field_name] = _producer_identity_from_dict(
            item[field_name],
            f"resource merger {field_name}",
        )
    raw_fields = item.pop("fields")
    if type(raw_fields) is not list:
        _fail("resource merger fields must be one exact list")
    result = ResourceMergerCandidateV2(
        **item,
        fields=tuple(_resource_field_from_dict(child) for child in raw_fields),
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("resource merger canonical reconstruction differs")
    return result


def _host_success_from_dict(value: object) -> HostSuccessCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(HostSuccessCandidateV2.__dataclass_fields__),
        "host success candidate",
    )
    item = dict(exact)
    item["host_executor"] = _producer_identity_from_dict(
        item["host_executor"],
        "host success executor",
    )
    for field_name in (
        "host_provisioning_receipt",
        "request",
        "intent",
        "initial_cgroup_sample",
        "ready",
        "observer_anchor",
        "go_commitment",
        "operational_frontier",
        "driver_terminal",
        "cleanup_reconciliation",
        "lifecycle",
        "cgroup_proof",
        "terminal_metadata",
        "terminal_storage_write_seal",
        "terminal_publication_reload_validation",
        "terminal_family_metadata",
        "execution_receipt",
        "observation_handoff",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"host success {field_name}",
        )
    raw_phases = item.pop("completed_phases")
    raw_recovery_nodes = item.pop("recovery_nodes")
    raw_unresolved = item.pop("cleanup_unresolved_recovery_nodes")
    if type(raw_phases) is not list or any(type(phase) is not str for phase in raw_phases):
        _fail("host success completed phases must be one exact text list")
    if type(raw_recovery_nodes) is not list or type(raw_unresolved) is not list:
        _fail("host success cleanup inventories must be exact lists")
    result = HostSuccessCandidateV2(
        **item,
        completed_phases=tuple(raw_phases),
        recovery_nodes=tuple(
            _recovery_node_candidate_from_dict(child) for child in raw_recovery_nodes
        ),
        cleanup_unresolved_recovery_nodes=tuple(raw_unresolved),
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("host success canonical reconstruction differs")
    return result


def _failure_publication_projection_from_dict(
    value: object,
) -> FailurePublicationProjectionV2:
    exact = _require_exact_keys(
        value,
        frozenset(
            {
                *FailurePublicationProjectionV2.__dataclass_fields__,
                QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD,
            }
        ),
        "failure publication projection",
    )
    item = _require_body_digest(
        dict(exact),
        QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD,
        "failure publication projection",
    )
    for field_name in (
        "algorithmic_resource_receipt",
        "runner_execution_receipt",
        "publisher_metadata",
        "publication_reconciliation_reference",
        "publication_commitment_wrapper",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"failure publication projection {field_name}",
        )
    item["native_publication_receipt"] = _optional_artifact_identity_from_dict(
        item["native_publication_receipt"],
        "failure publication projection native receipt",
    )
    for field_name in ("publisher", "native_atomic_producer"):
        item[field_name] = _producer_identity_from_dict(
            item[field_name],
            f"failure publication projection {field_name}",
        )
    raw_files = item.pop("files")
    if type(raw_files) is not list:
        _fail("failure publication projection files must be one exact list")
    result = FailurePublicationProjectionV2(
        **item,
        files=tuple(_publication_file_from_dict(child) for child in raw_files),
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("failure publication projection canonical reconstruction differs")
    return result


def _recovery_node_candidate_from_dict(value: object) -> RecoveryNodeCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(RecoveryNodeCandidateV2.__dataclass_fields__),
        "recovery node candidate",
    )
    item = dict(exact)
    item["artifact"] = _optional_artifact_identity_from_dict(
        item["artifact"],
        "recovery node artifact",
    )
    raw_dependencies = item.pop("dependencies")
    if type(raw_dependencies) is not list or any(
        type(dependency) is not str for dependency in raw_dependencies
    ):
        _fail("recovery node dependencies must be one exact text list")
    result = RecoveryNodeCandidateV2(**item, dependencies=tuple(raw_dependencies))
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("recovery node canonical reconstruction differs")
    return result


def _host_terminal_failure_from_dict(value: object) -> HostTerminalFailureCandidateV2:
    exact = _require_exact_keys(
        value,
        frozenset(HostTerminalFailureCandidateV2.__dataclass_fields__),
        "host terminal-failure candidate",
    )
    item = dict(exact)
    item["host_executor"] = _producer_identity_from_dict(
        item["host_executor"],
        "host failure executor",
    )
    for field_name in ("algorithmic_resource_contract", "storage_boundary_contract"):
        item[field_name] = _producer_identity_from_dict(
            item[field_name],
            f"host failure {field_name}",
        )
    for field_name in (
        "host_provisioning_receipt",
        "algorithmic_measurement_intent",
        "storage_boundary_intent",
        "request",
        "operational_frontier",
        "cleanup_reconciliation",
        "terminal_metadata",
        "lifecycle",
        "failure_receipt",
        "observation_handoff",
    ):
        item[field_name] = _artifact_identity_from_dict(
            item[field_name],
            f"host failure {field_name}",
        )
    item["intent"] = _optional_artifact_identity_from_dict(
        item["intent"],
        "host failure intent",
    )
    for field_name in (
        "ready",
        "observer_anchor",
        "go_commitment",
        "initial_cgroup_sample",
        "driver_terminal",
        "algorithmic_resource_receipt",
        "native_publication_receipt",
        "publication_reconciliation_reference",
        "publication_commitment_wrapper",
        "publication_reload_validation",
        "storage_write_seal",
        "storage_boundary_receipt",
        "precleanup_cgroup_sample",
        "cgroup_kill_receipt",
        "cgroup_empty_observation",
        "container_absence_observation",
        "post_container_remove_cgroup_sample",
        "cgroup_counter_fds_closed_receipt",
        "outer_cgroup_absence_observation",
        "cgroup_proof",
        "prior_host_execution_receipt",
    ):
        item[field_name] = _optional_artifact_identity_from_dict(
            item[field_name],
            f"host failure {field_name}",
        )
    raw_native_producer = item["native_atomic_producer"]
    item["native_atomic_producer"] = (
        None
        if raw_native_producer is None
        else _producer_identity_from_dict(
            raw_native_producer,
            "host failure native atomic producer",
        )
    )
    raw_failure_publication_projection = item["failure_publication_projection"]
    item["failure_publication_projection"] = (
        None
        if raw_failure_publication_projection is None
        else _failure_publication_projection_from_dict(raw_failure_publication_projection)
    )
    raw_phases = item.pop("completed_phases")
    raw_recovery_nodes = item.pop("recovery_nodes")
    raw_unresolved = item.pop("cleanup_unresolved_recovery_nodes")
    if type(raw_phases) is not list or any(type(phase) is not str for phase in raw_phases):
        _fail("host failure completed phases must be one exact text list")
    raw_dimensions = item.pop("uncertainty_dimensions")
    if type(raw_dimensions) is not list or any(
        type(dimension) is not str for dimension in raw_dimensions
    ):
        _fail("host failure uncertainty dimensions must be one exact text list")
    if type(raw_recovery_nodes) is not list or type(raw_unresolved) is not list:
        _fail("host failure cleanup inventories must be exact lists")
    result = HostTerminalFailureCandidateV2(
        **item,
        completed_phases=tuple(raw_phases),
        recovery_nodes=tuple(
            _recovery_node_candidate_from_dict(child) for child in raw_recovery_nodes
        ),
        cleanup_unresolved_recovery_nodes=tuple(raw_unresolved),
        uncertainty_dimensions=tuple(raw_dimensions),
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("host terminal-failure canonical reconstruction differs")
    return result


def _case_candidate_from_dict(value: object) -> QualificationCaseCandidateV2:
    if type(value) is not dict:
        _fail("case candidate must be one exact object")
    raw = cast(dict[str, Any], value)
    record_kind = raw.get("record_kind")
    if record_kind == "success":
        exact = _require_exact_keys(
            raw,
            frozenset(
                {
                    "schema_version",
                    "record_kind",
                    "case_spine",
                    "success",
                    "claims",
                    "readiness",
                }
            ),
            "success case candidate",
        )
        if exact["schema_version"] != QUALIFICATION_CASE_SUCCESS_CANDIDATE_V2_SCHEMA_VERSION:
            _fail("success case candidate schema differs")
        _require_false_mapping(exact["claims"], _claims(), "success case claims")
        _require_false_mapping(exact["readiness"], _readiness(), "success case readiness")
        success = _require_exact_keys(
            exact["success"],
            frozenset({"host_success", "probes", "resource_merger", "publication"}),
            "success case body",
        )
        raw_probes = success["probes"]
        if type(raw_probes) is not list:
            _fail("success case probes must be one exact list")
        result: QualificationCaseCandidateV2 = QualificationCaseSuccessCandidateV2(
            case_spine=_case_spine_from_dict(exact["case_spine"]),
            host_success=_host_success_from_dict(success["host_success"]),
            probes=tuple(_probe_candidate_from_dict(child) for child in raw_probes),
            resource_merger=_resource_merger_from_dict(success["resource_merger"]),
            publication=_publication_candidate_from_dict(success["publication"]),
        )
    elif record_kind == "terminal_failure":
        exact = _require_exact_keys(
            raw,
            frozenset(
                {
                    "schema_version",
                    "record_kind",
                    "case_spine",
                    "terminal_failure",
                    "claims",
                    "readiness",
                }
            ),
            "terminal-failure case candidate",
        )
        if (
            exact["schema_version"]
            != QUALIFICATION_CASE_TERMINAL_FAILURE_CANDIDATE_V2_SCHEMA_VERSION
        ):
            _fail("terminal-failure case candidate schema differs")
        _require_false_mapping(
            exact["claims"],
            _claims(),
            "terminal-failure case claims",
        )
        _require_false_mapping(
            exact["readiness"],
            _readiness(),
            "terminal-failure case readiness",
        )
        result = QualificationCaseTerminalFailureCandidateV2(
            case_spine=_case_spine_from_dict(exact["case_spine"]),
            terminal_failure=_host_terminal_failure_from_dict(exact["terminal_failure"]),
        )
    else:
        _fail("case candidate must carry one exact terminal tag")
    if not _exact_json_equal(result.to_dict(), raw):
        _fail("case candidate canonical reconstruction differs")
    return result


def _candidate_batch_from_dict(
    value: object,
) -> MatchedV3QualificationObservationCandidateBatchV2:
    _require_registry_descriptor_pin()
    exact = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "registry_descriptor_sha256",
                "campaign_spine",
                "cases",
                "all_case_sequence_receipt",
                "all_case_sequence_receipt_intent_file_sha256",
                "all_case_sequence_receipt_intent_body_sha256",
                "all_case_sequence_receipt_campaign_spine_sha256",
                "all_case_sequence_receipt_cases_inventory_sha256",
                "all_case_sequence_receipt_case_count",
                "all_case_sequence_receipt_terminal_coverage_complete",
                "terminal_coverage",
                "claims",
                "readiness",
                "limitations",
                "batch_body_sha256",
            }
        ),
        "qualification observation candidate batch",
    )
    if (
        exact["schema_version"] != QUALIFICATION_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION
        or exact["status"] != QUALIFICATION_OBSERVATION_REGISTRY_V2_STATUS
        or exact["classification"] != QUALIFICATION_OBSERVATION_REGISTRY_V2_CLASSIFICATION
        or exact["registry_descriptor_sha256"]
        != QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256
    ):
        _fail("qualification observation candidate batch identity differs")
    terminal_coverage = _require_exact_keys(
        exact["terminal_coverage"],
        frozenset(
            {
                "required_case_count",
                "same_case_retry_permitted",
                "selective_omission_permitted",
                "metadata_based_ordering_permitted",
                "terminal_failure_is_clean_rejection",
            }
        ),
        "candidate batch terminal coverage",
    )
    expected_terminal_coverage = {
        "required_case_count": len(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
        "same_case_retry_permitted": False,
        "selective_omission_permitted": False,
        "metadata_based_ordering_permitted": False,
        "terminal_failure_is_clean_rejection": False,
    }
    if not _exact_json_equal(terminal_coverage, expected_terminal_coverage):
        _fail("candidate batch terminal coverage differs")
    _require_false_mapping(exact["claims"], _claims(), "candidate batch claims")
    _require_false_mapping(exact["readiness"], _readiness(), "candidate batch readiness")
    if not _exact_json_equal(exact["limitations"], _limitations()):
        _fail("candidate batch limitations differ")
    _require_body_digest(exact, "batch_body_sha256", "candidate batch")
    raw_cases = exact["cases"]
    if type(raw_cases) is not list:
        _fail("candidate batch cases must be one exact list")
    result = MatchedV3QualificationObservationCandidateBatchV2(
        campaign_spine=_campaign_spine_from_dict(exact["campaign_spine"]),
        cases=tuple(_case_candidate_from_dict(child) for child in raw_cases),
        all_case_sequence_receipt=_artifact_identity_from_dict(
            exact["all_case_sequence_receipt"],
            "post-case all-case sequence receipt",
        ),
        all_case_sequence_receipt_intent_file_sha256=exact[
            "all_case_sequence_receipt_intent_file_sha256"
        ],
        all_case_sequence_receipt_intent_body_sha256=exact[
            "all_case_sequence_receipt_intent_body_sha256"
        ],
        all_case_sequence_receipt_campaign_spine_sha256=exact[
            "all_case_sequence_receipt_campaign_spine_sha256"
        ],
        all_case_sequence_receipt_cases_inventory_sha256=exact[
            "all_case_sequence_receipt_cases_inventory_sha256"
        ],
        all_case_sequence_receipt_case_count=exact["all_case_sequence_receipt_case_count"],
        all_case_sequence_receipt_terminal_coverage_complete=exact[
            "all_case_sequence_receipt_terminal_coverage_complete"
        ],
    )
    if not _exact_json_equal(result.to_dict(), exact):
        _fail("candidate batch canonical reconstruction differs")
    return result


def _validate_replay_pins(
    batch: MatchedV3QualificationObservationCandidateBatchV2,
    pins: QualificationObservationCandidateReplayPinsV2,
) -> None:
    if type(pins) is not QualificationObservationCandidateReplayPinsV2:
        _fail("candidate batch replay pins must use the exact immutable type")
    spine = batch.campaign_spine
    expected = {
        "qualification_plan_file_sha256": spine.qualification_plan_file_sha256,
        "qualification_plan_body_sha256": spine.qualification_plan_body_sha256,
        "observation_registry_source_sha256": (spine.observation_registry_source_sha256),
        "plan_issuance_receipt_file_sha256": (spine.plan_issuance_receipt.file_sha256),
        "plan_issuance_receipt_body_sha256": (spine.plan_issuance_receipt.body_sha256),
        "case_ticket_registry_file_sha256": spine.case_ticket_registry.file_sha256,
        "case_ticket_registry_body_sha256": spine.case_ticket_registry.body_sha256,
        "publisher_registry_file_sha256": spine.publisher_registry.file_sha256,
        "publisher_registry_body_sha256": spine.publisher_registry.body_sha256,
        "seed_registry_file_sha256": spine.seed_registry.file_sha256,
        "seed_registry_body_sha256": spine.seed_registry.body_sha256,
        "seed_pulse_record_file_sha256": spine.seed_pulse_record.file_sha256,
        "seed_pulse_record_body_sha256": spine.seed_pulse_record.body_sha256,
        "seed_trust_root_receipt_file_sha256": (spine.seed_trust_root_receipt.file_sha256),
        "seed_trust_root_receipt_body_sha256": (spine.seed_trust_root_receipt.body_sha256),
        "quicknet_verifier_descriptor_sha256": (spine.quicknet_verifier.descriptor_sha256),
        "quicknet_verifier_source_sha256": spine.quicknet_verifier.source_sha256,
        "quicknet_verifier_binary_sha256": spine.quicknet_verifier_binary_sha256,
        "quicknet_verifier_receipt_file_sha256": (spine.quicknet_verifier_receipt.file_sha256),
        "quicknet_verifier_receipt_body_sha256": (spine.quicknet_verifier_receipt.body_sha256),
        "seed_chronology_receipt_file_sha256": (spine.seed_chronology_receipt.file_sha256),
        "seed_chronology_receipt_body_sha256": (spine.seed_chronology_receipt.body_sha256),
        "local_source_candidate_file_sha256": (spine.local_source_candidate.file_sha256),
        "local_source_candidate_body_sha256": (spine.local_source_candidate.body_sha256),
        "external_source_candidate_file_sha256": (spine.external_source_candidate.file_sha256),
        "external_source_candidate_body_sha256": (spine.external_source_candidate.body_sha256),
        "adapter_source_candidate_file_sha256": (spine.adapter_source_candidate.file_sha256),
        "adapter_source_candidate_body_sha256": (spine.adapter_source_candidate.body_sha256),
        "joint_source_closure_candidate_file_sha256": (
            spine.joint_source_closure_candidate.file_sha256
        ),
        "joint_source_closure_candidate_body_sha256": (
            spine.joint_source_closure_candidate.body_sha256
        ),
        "sealed_staging_candidate_file_sha256": (spine.sealed_staging_candidate.file_sha256),
        "sealed_staging_candidate_body_sha256": (spine.sealed_staging_candidate.body_sha256),
        "fresh_build_candidate_file_sha256": spine.fresh_build_candidate.file_sha256,
        "fresh_build_candidate_body_sha256": spine.fresh_build_candidate.body_sha256,
        "runtime_candidate_file_sha256": spine.runtime_candidate.file_sha256,
        "runtime_candidate_body_sha256": spine.runtime_candidate.body_sha256,
        "runtime_qualification_receipt_file_sha256": (
            spine.runtime_qualification_receipt.file_sha256
        ),
        "runtime_qualification_receipt_body_sha256": (
            spine.runtime_qualification_receipt.body_sha256
        ),
        "host_provisioning_receipt_file_sha256": (spine.host_provisioning_receipt.file_sha256),
        "host_provisioning_receipt_body_sha256": (spine.host_provisioning_receipt.body_sha256),
        "host_executor_descriptor_sha256": spine.host_executor.descriptor_sha256,
        "host_executor_source_sha256": spine.host_executor.source_sha256,
        "full_resource_merger_descriptor_sha256": (spine.full_resource_merger.descriptor_sha256),
        "full_resource_merger_source_sha256": (spine.full_resource_merger.source_sha256),
        "algorithmic_resource_contract_descriptor_sha256": (
            spine.algorithmic_resource_contract.descriptor_sha256
        ),
        "algorithmic_resource_contract_source_sha256": (
            spine.algorithmic_resource_contract.source_sha256
        ),
        "storage_boundary_contract_descriptor_sha256": (
            spine.storage_boundary_contract.descriptor_sha256
        ),
        "storage_boundary_contract_source_sha256": spine.storage_boundary_contract.source_sha256,
        "normalized_publication_contract_descriptor_sha256": (
            FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256
        ),
        "normalized_publication_contract_source_sha256": (
            FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256
        ),
        "all_case_sequence_intent_file_sha256": (spine.all_case_sequence_intent.file_sha256),
        "all_case_sequence_intent_body_sha256": (spine.all_case_sequence_intent.body_sha256),
        "all_case_sequence_receipt_file_sha256": (batch.all_case_sequence_receipt.file_sha256),
        "all_case_sequence_receipt_body_sha256": (batch.all_case_sequence_receipt.body_sha256),
        "all_case_sequence_cases_inventory_sha256": (
            batch.all_case_sequence_receipt_cases_inventory_sha256
        ),
        "candidate_order_sha256": spine.candidate_order_sha256,
        "resource_field_order_sha256": spine.resource_field_order_sha256,
        "image_id": spine.image_id,
    }
    for field_name, expected_value in expected.items():
        supplied = getattr(pins, field_name)
        if not hmac.compare_digest(supplied, expected_value):
            _fail(f"candidate batch independent replay pin {field_name} differs")


def canonical_matched_v3_qualification_observation_candidate_batch_v2_bytes(
    batch: MatchedV3QualificationObservationCandidateBatchV2,
) -> bytes:
    """Validate and encode the only public artifact: one complete 28-case batch."""

    if type(batch) is not MatchedV3QualificationObservationCandidateBatchV2:
        _fail("candidate batch serialization requires the exact batch type")
    _require_registry_descriptor_pin()
    value = batch.to_dict()
    _reject_forbidden_metadata_keys(value)
    _candidate_batch_from_dict(value)
    return _canonical_json(value)


def parse_matched_v3_qualification_observation_candidate_batch_v2(
    raw: bytes,
    *,
    pins: QualificationObservationCandidateReplayPinsV2,
) -> MatchedV3QualificationObservationCandidateBatchV2:
    """Replay one canonical full batch under all independent campaign pins."""

    if type(pins) is not QualificationObservationCandidateReplayPinsV2:
        _fail("candidate batch replay pins must use the exact immutable type")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        pins.batch_file_sha256,
    ):
        _fail("qualification observation candidate batch full-file digest differs")
    value = _strict_json_load(raw)
    _reject_forbidden_metadata_keys(value)
    batch = _candidate_batch_from_dict(value)
    _validate_replay_pins(batch, pins)
    return batch


def replay_matched_v3_qualification_observation_candidate_batch_v2(
    raw: bytes,
    *,
    pins: QualificationObservationCandidateReplayPinsV2,
) -> MatchedV3QualificationObservationCandidateBatchV2:
    """Replay alias retaining the exact batch-only independent-pin contract."""

    return parse_matched_v3_qualification_observation_candidate_batch_v2(
        raw,
        pins=pins,
    )


def matched_v3_qualification_observation_registry_v2_descriptor() -> dict[str, Any]:
    """Return detached registry content without issuing or evaluating a batch."""

    return _strict_json_load(_canonical_json(_registry_descriptor()))


def canonical_matched_v3_qualification_observation_registry_v2_descriptor_bytes() -> bytes:
    """Return canonical descriptor bytes; the literal identity may still be zero-pinned."""

    return bytes(_REGISTRY_DESCRIPTOR_BYTES)


def matched_v3_qualification_observation_registry_v2_descriptor_sha256() -> str:
    """Return the audited literal descriptor identity, failing while it is zero-pinned."""

    return _require_registry_descriptor_pin()


def parse_matched_v3_qualification_observation_registry_v2_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Parse only the exact finalized registry-v2 descriptor bytes."""

    pinned = _require_registry_descriptor_pin()
    value = _strict_json_load(raw)
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), pinned) or not _exact_json_equal(
        value, _registry_descriptor()
    ):
        _fail("qualification observation registry v2 descriptor differs")
    return value


__all__ = [
    "ADAPTER_SOURCE_CANDIDATE_SCHEMA_VERSION",
    "ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "ADAPTER_PUBLICATION_ROLE_PATHS",
    "ALGORITHMIC_RESOURCE_VALIDATOR_IMPLEMENTATION_SHA256S",
    "ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION",
    "ALL_CASE_SEQUENCE_INTENT_V1_BODY_KEYS",
    "ALL_CASE_SEQUENCE_INTENT_V1_BODY_SHA256_FIELD",
    "ALL_CASE_SEQUENCE_INTENT_SCHEMA_VERSION",
    "ALL_CASE_SEQUENCE_RECEIPT_V1_BODY_KEYS",
    "ALL_CASE_SEQUENCE_RECEIPT_V1_BODY_SHA256_FIELD",
    "ALL_CASE_SEQUENCE_RECEIPT_SCHEMA_VERSION",
    "ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "ArtifactIdentityV2",
    "CASE_EXECUTION_TICKET_SCHEMA_VERSION",
    "CASE_TICKET_REGISTRY_SCHEMA_VERSION",
    "CampaignSpineV2",
    "CaseSpineV2",
    "EMPTY_FILE_SHA256",
    "ENDPOINT_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "ENDPOINT_RESOURCE_REQUEST_SCHEMA_VERSION",
    "EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_METADATA_SCHEMA_VERSION",
    "EXTERNAL_PUBLICATION_ROLE_PATHS",
    "EXTERNAL_SOURCE_CANDIDATE_SCHEMA_VERSION",
    "FRESH_BUILD_CANDIDATE_SCHEMA_VERSION",
    "FailurePublicationProjectionV2",
    "ForagerMatchedV3QualificationObservationsV2Error",
    "FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION",
    "FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION",
    "FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION",
    "FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256",
    "FINAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256",
    "FINAL_NORMALIZED_PUBLICATION_DESCRIPTOR_SHA256",
    "FINAL_NORMALIZED_PUBLICATION_SOURCE_SHA256",
    "FINAL_STORAGE_BOUNDARY_VALIDATOR_DESCRIPTOR_SHA256",
    "FINAL_STORAGE_BOUNDARY_VALIDATOR_SOURCE_SHA256",
    "HOST_CASE_INTENT_SCHEMA_VERSION",
    "HOST_CASE_REQUEST_SCHEMA_VERSION",
    "HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION",
    "HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION",
    "HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION",
    "HOST_CGROUP_PROOF_SCHEMA_VERSION",
    "HOST_CLEANUP_RECONCILIATION_BODY_SHA256_FIELD",
    "HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION",
    "HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION",
    "HOST_FAILURE_RECEIPT_SCHEMA_VERSION",
    "HOST_GO_SCHEMA_VERSION",
    "HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256S",
    "HISTORICAL_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256S",
    "HOST_LIFECYCLE_PHASES",
    "HOST_LIFECYCLE_SCHEMA_VERSION",
    "HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION",
    "HOST_OPERATIONAL_FAILURE_EFFECT_STATES",
    "HOST_OPERATIONAL_FRONTIER_BODY_SHA256_FIELD",
    "HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION",
    "HOST_OPERATIONAL_PHASES",
    "HOST_OBSERVER_ANCHOR_SCHEMA_VERSION",
    "HOST_OBSERVATION_HANDOFF_V2_SCHEMA_VERSION",
    "HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION",
    "HOST_PHASE_STATES",
    "HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION",
    "HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION",
    "HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION",
    "HOST_READY_SCHEMA_VERSION",
    "HOST_RECOVERY_NODE_DEPENDENCIES",
    "HOST_RECOVERY_NODE_NAMES",
    "HOST_RECOVERY_NODE_SCHEMAS",
    "HOST_RECOVERY_NODE_STATES",
    "HOST_SUCCESS_RECEIPT_SCHEMA_VERSION",
    "HOST_TERMINAL_METADATA_SCHEMA_VERSION",
    "HOST_TERMINAL_METADATA_V2_BODY_KEYS",
    "HOST_TERMINAL_METADATA_V2_BODY_SHA256_FIELD",
    "HOST_UNCERTAINTY_DIMENSIONS",
    "HostSuccessCandidateV2",
    "HostTerminalFailureCandidateV2",
    "IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION",
    "JOINT_SOURCE_CLOSURE_CANDIDATE_SCHEMA_VERSION",
    "LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION",
    "LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_PUBLICATION_METADATA_SCHEMA_VERSION",
    "LOCAL_PUBLICATION_ROLE_PATHS",
    "LOCAL_SOURCE_CANDIDATE_SCHEMA_VERSION",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "MATCHED_V3_PPO_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS",
    "MAX_PUBLICATION_FILE_BYTES",
    "MAX_PUBLICATION_TOTAL_BYTES",
    "MatchedV3QualificationObservationCandidateBatchV2",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_STATUS",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_CLASSIFICATION",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_KEYS",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_BODY_SHA256_FIELD",
    "NONEXECUTING_HOST_EXECUTOR_DESCRIPTOR_SHA256",
    "NONEXECUTING_HOST_EXECUTOR_SOURCE_SHA256",
    "PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256",
    "PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256",
    "PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION",
    "PROBE_KINDS",
    "PROBE_SCHEMA_BY_KIND",
    "PRODUCTION_HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION",
    "PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION",
    "ProducerIdentityV2",
    "ProbeCandidateV2",
    "PublicationCandidateV2",
    "PublicationFileCandidateV2",
    "PUBLISHER_REGISTRY_ENTRY_SCHEMA_VERSION",
    "PUBLISHER_REGISTRY_SCHEMA_VERSION",
    "QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION",
    "QualificationCaseSuccessCandidateV2",
    "QualificationCaseTerminalFailureCandidateV2",
    "QualificationObservationCandidateReplayPinsV2",
    "QUALIFICATION_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION",
    "QUALIFICATION_OBSERVATION_REGISTRY_V2_CLASSIFICATION",
    "QUALIFICATION_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SHA256",
    "QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION",
    "QUALIFICATION_OBSERVATION_REGISTRY_V2_STATUS",
    "QUALIFICATION_PLAN_V3_SCHEMA_VERSION",
    "QUALIFICATION_PUBLICATION_CANDIDATE_V2_SCHEMA_VERSION",
    "QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION",
    "QUALIFICATION_FAILURE_PUBLICATION_PROJECTION_BODY_SHA256_FIELD",
    "QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION",
    "QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_KEYS",
    "QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_V1_BODY_SHA256_FIELD",
    "QUALIFICATION_PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION",
    "QUALIFICATION_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION",
    "QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION",
    "QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION",
    "QUALIFICATION_SEED_TRUST_ROOT_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION",
    "QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION",
    "QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION",
    "RESOURCE_CEILING_FIELDS",
    "RESOURCE_PROVENANCE_KINDS",
    "RUNTIME_CANDIDATE_SCHEMA_VERSION",
    "RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "ResourceFieldCandidateV2",
    "ResourceMergerCandidateV2",
    "RecoveryNodeCandidateV2",
    "SEED_CHRONOLOGY_RECEIPT_SCHEMA_VERSION",
    "SEALED_STAGING_CANDIDATE_SCHEMA_VERSION",
    "SOURCE_MATERIALIZATION_QUICKNET_DESCRIPTOR_SHA256",
    "SOURCE_MATERIALIZATION_QUICKNET_SOURCE_SHA256",
    "SOURCE_ONLY_QUICKNET_BUILD_DESCRIPTOR_SHA256",
    "SOURCE_ONLY_QUICKNET_BUILD_SOURCE_SHA256",
    "SOURCE_ONLY_QUICKNET_VERIFIER_DESCRIPTOR_SHA256",
    "SOURCE_ONLY_QUICKNET_VERIFIER_SOURCE_SHA256",
    "STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "STRICT_ADAPTER_PUBLICATION_METADATA_SCHEMA_VERSION",
    "all_case_sequence_intent_v1_body_projection",
    "all_case_sequence_receipt_v1_body_projection",
    "canonical_all_case_sequence_intent_v1_body_bytes",
    "canonical_all_case_sequence_intent_v1_file_bytes",
    "canonical_all_case_sequence_receipt_v1_body_bytes",
    "canonical_all_case_sequence_receipt_v1_file_bytes",
    "canonical_host_cleanup_reconciliation_v2_body_bytes",
    "canonical_host_cleanup_reconciliation_v2_file_bytes",
    "canonical_host_observation_handoff_v2_body_bytes",
    "canonical_host_observation_handoff_v2_file_bytes",
    "canonical_host_operational_frontier_v2_body_bytes",
    "canonical_host_operational_frontier_v2_file_bytes",
    "canonical_host_terminal_metadata_v2_body_bytes",
    "canonical_host_terminal_metadata_v2_file_bytes",
    "canonical_matched_v3_qualification_observation_candidate_batch_v2_bytes",
    "canonical_matched_v3_qualification_observation_registry_v2_descriptor_bytes",
    "canonical_normalized_publication_commitment_wrapper_v1_body_bytes",
    "canonical_normalized_publication_commitment_wrapper_v1_file_bytes",
    "canonical_qualification_publication_reload_validation_v1_body_bytes",
    "canonical_qualification_publication_reload_validation_v1_file_bytes",
    "host_cleanup_reconciliation_v2_body_projection",
    "host_observation_handoff_v2_body_projection",
    "host_operational_frontier_v2_body_projection",
    "host_terminal_metadata_v2_body_projection",
    "matched_v3_qualification_observation_registry_v2_descriptor",
    "matched_v3_qualification_observation_registry_v2_descriptor_sha256",
    "parse_matched_v3_qualification_observation_candidate_batch_v2",
    "parse_matched_v3_qualification_observation_registry_v2_descriptor",
    "normalized_publication_commitment_wrapper_v1_body_projection",
    "qualification_publication_reload_validation_v1_body_projection",
    "replay_matched_v3_qualification_observation_candidate_batch_v2",
]
