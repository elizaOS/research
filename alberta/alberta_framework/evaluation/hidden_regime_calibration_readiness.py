"""Fail-closed readiness receipts for hidden-regime factorial calibration.

This module cannot run a calibration.  It prepares a source/runtime-bound
draft, derives certification records by running an exact list of tests only
after explicit authorization, and publishes a finalized receipt only after a
second explicit authorization.  A future calibration runner may be bound, but
only when its source module already exists.

The receipt and its deterministic source ZIP are content addressed.  The ZIP
is executable source, not merely a hash list: a worker launched through
``execute_bound_calibration_worker`` starts in an empty directory with the ZIP
as the first and sole project source path, and rejects any project module whose
loader or ``__file__`` does not originate inside that ZIP.

Nothing here constructs a hidden-regime world, advances a calibration seed,
derives a protected seed namespace, observes a learner outcome, freezes a
threshold, or promotes a scientific claim.
"""

from __future__ import annotations

import ast
import csv
import ctypes
import dataclasses
import errno
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from alberta_framework.evaluation.hidden_regime_checkpoint import (
    HIDDEN_REGIME_CHECKPOINT_SCHEMA,
    HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
    MANAGED_EXECUTION_BOUNDARY_SCOPE,
    READINESS_EXECUTION_GOVERNANCE_FIELD,
    build_calibration_execution_genesis,
    calibration_execution_genesis_receipt_binding,
    require_valid_calibration_execution_genesis,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
    BOUND_PRIMITIVE_TRACE_SCHEMA,
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_READINESS_RECEIPT_SCHEMA,
    CONSUMED_CALIBRATION_NAMESPACE,
    DESIGN_ENVELOPE_SCHEMA,
    DESIGN_SCHEMA,
    N_MATCHED_CASES,
    PROTOCOL_STATUS,
    SEED_SNAPSHOT_SHA256,
    calibration_design_payload,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    canonical_sha256 as _protocol_canonical_sha256,
)
from alberta_framework.evaluation.hidden_regime_lineage_oracle import (
    HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_summary_oracle import (
    HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
    HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_world_oracle import (
    HIDDEN_REGIME_WORLD_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    SLOT_ROLE_TRANSITION_ORACLE_SCHEMA,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED,
    PROTECTED_CANDIDATE_PARTITION,
)

READINESS_SOURCE_SCHEMA = "alberta.hidden-regime-calibration.source-closure.v1"
READINESS_ARCHIVE_SCHEMA = "alberta.hidden-regime-calibration.source-archive.v1"
READINESS_RUNTIME_SCHEMA = "alberta.hidden-regime-calibration.runtime-identity.v5"
READINESS_CERTIFICATION_SCHEMA = "alberta.hidden-regime-calibration.certification.v6"
READINESS_CERTIFICATION_NODE_MANIFEST_SCHEMA = (
    "alberta.hidden-regime-calibration.certification-node-manifest.v2"
)
READINESS_CERTIFICATION_RUNTIME_CONTRACT_SCHEMA = (
    "alberta.hidden-regime-calibration.certification-runtime-contract.v1"
)
READINESS_CERTIFICATION_EXECUTION_MANIFEST_SCHEMA = (
    "alberta.hidden-regime-calibration.certification-execution-manifest.v1"
)
READINESS_RUNTIME_RECONSTRUCTION_SCHEMA = (
    "alberta.hidden-regime-calibration.runtime-reconstruction.v1"
)
READINESS_RUNTIME_EXECUTION_SCHEMA = (
    "alberta.hidden-regime-calibration.runtime-execution-identity.v1"
)
READINESS_ENVELOPE_SCHEMA = "alberta.hidden-regime-calibration.readiness-envelope.v1"
READINESS_STATUS = "calibration_ready_outcomes_unexecuted"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_FILES = (Path("pyproject.toml"), Path("uv.lock"))
_MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = stat.S_IFREG | 0o444
_PROCESS_SEAL_KEY = secrets.token_bytes(32)
_PROCESS_START_NONCE = secrets.token_hex(32)
_ACTIVE_RUNTIME_BATCH_GUARDS: dict[str, object] = {}


def _reset_process_local_readiness_state_after_fork() -> None:
    global _PROCESS_SEAL_KEY, _PROCESS_START_NONCE
    _PROCESS_SEAL_KEY = secrets.token_bytes(32)
    _PROCESS_START_NONCE = secrets.token_hex(32)
    _ACTIVE_RUNTIME_BATCH_GUARDS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_local_readiness_state_after_fork)
_CALIBRATION_RUNNER_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_calibration"
_THRESHOLD_FREEZE_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_thresholds"
_PROTECTED_PLAN_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_protected_plan"
_EXECUTION_GOVERNANCE_MODULE = "alberta_framework.evaluation.hidden_regime_execution_governance"
_SUMMARY_ORACLE_MODULE = "alberta_framework.evaluation.hidden_regime_summary_oracle"
_ALLOWED_WORKER_ENTRYPOINT_MODES = (
    "--worker-case-v1",
    "--worker-preflight-v1",
    "--worker-aggregate-v1",
    "--worker-threshold-freeze-v1",
    "--worker-protected-plan-v1",
)

_BASE_SOURCE_ROOT_MODULES = (
    "alberta_framework.evaluation.hidden_regime_calibration_readiness",
    "alberta_framework.evaluation.hidden_regime_factorial_protocol",
    "alberta_framework.evaluation.hidden_regime_signaling_development",
    "alberta_framework.evaluation.hidden_regime_trace_audit",
    "alberta_framework.evaluation.hidden_regime_checkpoint",
    "alberta_framework.evaluation.slot_signaling_lifecycle_oracle",
    "alberta_framework.evaluation.hidden_regime_world_oracle",
    "alberta_framework.evaluation.hidden_regime_lineage_oracle",
    _CALIBRATION_RUNNER_MODULE,
    _THRESHOLD_FREEZE_MODULE,
    _PROTECTED_PLAN_MODULE,
    _EXECUTION_GOVERNANCE_MODULE,
    _SUMMARY_ORACLE_MODULE,
)

_ENVIRONMENT_PREFIXES = (
    "JAX_",
    "XLA_",
    "CUDA_",
    "NVIDIA_",
    "OMP_",
    "MKL_",
    "TF_",
    "TPU_",
)
_ENVIRONMENT_NAMES = (
    "LD_LIBRARY_PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)
_FORBIDDEN_CHILD_ENVIRONMENT_NAMES = frozenset(
    {
        "LD_AUDIT",
        "LD_PRELOAD",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
    }
)
_BOUND_LIBRARY_ENVIRONMENT_SIDE_EFFECTS = {
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TPU_SKIP_MDS_QUERY": "1",
}
_KEY_DISTRIBUTIONS = (
    "alberta-framework",
    "chex",
    "jax",
    "jaxlib",
    "numpy",
    "pytest",
    "scipy",
)
_INSTALLED_DISTRIBUTION_FILE_SCOPE = (
    "all_RECORD_listed_files_for_every_distribution_discovered_only_on_the_exact_bound_"
    "purelib_and_platlib_paths_including_python_native_metadata_and_scripts_are_stable_"
    "open_hashed_as_semantic_distribution_provenance"
)
_DEPENDENCY_IMPORT_TREE_FILE_SCOPE = (
    "every_root_and_descendant_directory_regular_file_and_every_symlink_plus_its_regular_"
    "target_in_the_exact_unique_"
    "purelib_platlib_import_trees_including_RECORD_unowned_python_bytecode_native_data_"
    "and_cache_files_are_stable_open_hashed_and_end_scan_identity_checked"
)
_STDLIB_FILE_SCOPE = (
    "the_root_and_all_descendant_directories_regular_files_and_symlink_targets_in_the_"
    "exact_bound_standard_library_and_"
    "lib-dynload_tree_plus_the_present_or_absent_state_of_other_exact_no-site_search_"
    "paths_are_stable_open_hashed_including___pycache___pyc_pyo_and_native_files;_"
    "any_nested_site-packages_or_dist-packages_are_included_even_when_redundantly_bound_"
    "by_the_dependency_import_tree_inventory"
)
_BYTECODE_CACHE_POLICY = (
    "isolated_workers_and_certifications_use_command_line_-B_and_a_fresh_empty_separate_"
    "-X_pycache_prefix_so_no_new_bytecode_is_written_and_normal_adjacent_caches_are_not_"
    "read;_legacy_sourceless_and_other_importable_bytecode_may_be_read_only_when_its_"
    "exact_bytes_and_path_are_bound_by_the_complete_import_tree_inventories"
)
_RUNTIME_PATH_POLICY = (
    "safe-path_no-site_startup_path_must_exactly_match_the_receipt_then_sys_prefix_and_"
    "sys_exec_prefix_are_restored_and_sys_path_is_replaced_by_source_first_bound_stdlib_"
    "paths_then_unique_bound_purelib_platlib_without_site_or_pth_processing;_-P_replaces_"
    "-I_because_-I_ignores_the_required_deterministic_PYTHONHASHSEED"
)
_RUNTIME_MUTATION_POLICY = (
    "the_complete_runtime_identity_is_rehashed_before_and_after_the_certification_batch_"
    "and_before_and_after_each_explicitly_authorized_bounded_calibration_batch_or_each_"
    "unbatched_worker;_every_case_child_rechecks_the_projected_python_platform_version_"
    "JAX_device_config_and_exact_environment_identity_before_and_after_learner_execution;_"
    "each_case_child_also_rehashes_the_complete_runtime_immediately_before_managed_"
    "finalization_so_a_failed_batch_endpoint_scan_cannot_leave_a_trusted_final_record;_"
    "this_is_cooperative_drift_detection_not_a_same-uid_hostile_kernel_or_"
    "swap-and-restore_sandbox_and_does_not_content-address_transitive_OS_DSOs_CPU_"
    "microcode_affinity_or_driver_state"
)
_CERTIFICATION_ENVIRONMENT_POLICY = (
    "startup_environment_is_reconstructed_only_from_the_receipt_bound_allowlist_then_"
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1_is_added;_no_other_inherited_variable_is_passed;_"
    "PYTHONHASHSEED=0_is_applied_before_interpreter_start;_after_imports_only_the_exact_"
    "bound_JAX_library_side_effects_may_be_added"
)
_CHILD_ENVIRONMENT_POLICY = (
    "startup_environment_is_exactly_LC_ALL=C_and_PYTHONHASHSEED=0;_PATH_HOME_and_all_"
    "other_PYTHON_JAX_XLA_CUDA_NVIDIA_OMP_MKL_TF_TPU_variables_LD_PRELOAD_LD_AUDIT_"
    "DYLD_loader_variables_and_every_other_parent_variable_are_absent;_after_imports_"
    "only_TF_CPP_MIN_LOG_LEVEL=1_and_TPU_SKIP_MDS_QUERY=1_may_be_added_by_the_bound_"
    "JAX_runtime"
)
_LOADED_MODULE_ORIGIN_POLICY = (
    "at_pre_and_post_execution_checkpoints_every_registered_non_builtin_non_frozen_"
    "module_has_a_file_origin_or_namespace_location_within_bound_source_stdlib_or_"
    "dependency_roots;_originless_extension-created_modules_are_object-identity_"
    "snapshotted_immediately_after_the_trusted_content-bound_runner_import_and_no_new_"
    "originless_module_may_remain_registered_after_execution;_this_is_a_cooperative_"
    "sys.modules_checkpoint_not_containment_of_transient_exec_file_reads_or_ctypes"
)


class ReadinessError(RuntimeError):
    """A readiness contract, integrity, or publication check failed."""


@dataclass(frozen=True, slots=True)
class CertificationSpec:
    """One exact, outcome-free readiness certification test group."""

    certification_id: str
    node_ids: tuple[str, ...]
    exact_file_test_inventory: bool = False
    checkpoint_cut_runtime_node_id: str | None = None
    checkpoint_cut_fixture_name: str | None = None
    checkpoint_cut_trace_fixture_name: str | None = None
    checkpoint_cut_ids: tuple[str, ...] = ()
    checkpoint_cut_semantics: tuple[tuple[str, str, str, int, int], ...] = ()

    @property
    def runtime_contract(self) -> dict[str, object]:
        """Return the exact runtime-only semantic contract for this group."""

        checkpoint_cut_contract: dict[str, object] | None = None
        if self.checkpoint_cut_runtime_node_id is not None:
            checkpoint_cut_contract = {
                "node_id": self.checkpoint_cut_runtime_node_id,
                "fixture_name": self.checkpoint_cut_fixture_name,
                "fixture_tuple_index": 1,
                "trace_fixture_name": self.checkpoint_cut_trace_fixture_name,
                "expected_cut_ids": list(self.checkpoint_cut_ids),
                "cut_semantics": [
                    {
                        "cut_id": cut_id,
                        "trace_field": trace_field,
                        "predicate": predicate,
                        "occurrence_index": occurrence_index,
                        "index_offset": index_offset,
                    }
                    for (
                        cut_id,
                        trace_field,
                        predicate,
                        occurrence_index,
                        index_offset,
                    ) in self.checkpoint_cut_semantics
                ],
                "cut_value_contract": (
                    "exact_independent_reconstruction_from_bound_trace_events"
                ),
            }
        return {
            "schema": READINESS_CERTIFICATION_RUNTIME_CONTRACT_SCHEMA,
            "certification_id": self.certification_id,
            "checkpoint_cut_contract": checkpoint_cut_contract,
        }

    @property
    def runtime_contract_sha256(self) -> str:
        """Return the canonical digest of the runtime-only semantic contract."""

        return canonical_sha256(self.runtime_contract)

    @property
    def node_manifest(self) -> dict[str, object]:
        """Return the fail-closed semantic test-node and checkpoint-cut inventory."""

        return {
            "schema": READINESS_CERTIFICATION_NODE_MANIFEST_SCHEMA,
            "certification_id": self.certification_id,
            "node_ids": list(self.node_ids),
            "exact_file_test_inventory": self.exact_file_test_inventory,
            "runtime_contract": self.runtime_contract,
            "runtime_contract_sha256": self.runtime_contract_sha256,
        }

    @property
    def node_manifest_sha256(self) -> str:
        """Return the canonical digest of the semantic node manifest."""

        return canonical_sha256(self.node_manifest)

    @property
    def semantic_command(self) -> tuple[str, ...]:
        """Return the portable command bound into a receipt."""

        return (
            "{runtime_python}",
            "-S",
            "-B",
            "-P",
            "-X",
            "pycache_prefix={fresh_empty_separate_bytecode_cache_root}",
            "-c",
            "{readiness_certification_harness_v1}",
            "{fresh_empty_separate_bytecode_cache_root}",
            "{verified_extracted_source_root}",
            "{bound_runtime_prefix}",
            "{bound_runtime_exec_prefix}",
            "{bound_runtime_purelib}",
            "{bound_runtime_platlib}",
            "{bound_runtime_stdlib}",
            "{bound_no_site_stdlib_search_paths_json}",
            "{receipt_derived_certification_environment_json}",
            "{fresh_certification_execution_manifest_path}",
            "{certification_runtime_contract_json}",
            *self.node_ids,
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
        )


_CHECKPOINT_EQUIVALENCE_CUT_IDS = (
    "inside_lease",
    "lease_boundary",
    "regime_boundary",
    "scratch_retest",
    "commit",
    "replacement",
)
_CHECKPOINT_EQUIVALENCE_CUT_SEMANTICS = (
    (
        "inside_lease",
        "helper_lease_offset_post",
        "strictly_between_zero_and_three",
        0,
        1,
    ),
    ("lease_boundary", "helper_lease_boundary", "truthy", 0, 1),
    ("regime_boundary", "segment_index", "adjacent_change_new_index", 0, 0),
    ("scratch_retest", "helper_scratch_retest_started", "truthy", 0, 1),
    ("commit", "helper_committed_slot", "nonnegative", 1, 1),
    ("replacement", "helper_retired_slot", "nonnegative", 0, 1),
)
_CHECKPOINT_EQUIVALENCE_CUT_RUNTIME_NODE_ID = (
    "tests/test_hidden_regime_checkpoint.py::"
    "test_json_roundtripped_lifecycle_chunks_equal_one_shot_bit_for_bit"
)
_CHECKPOINT_EQUIVALENCE_NODE_IDS = (
    "tests/test_hidden_regime_checkpoint.py::"
    "test_checkpoint_payload_is_complete_exact_and_has_no_runtime_oracle_state",
    _CHECKPOINT_EQUIVALENCE_CUT_RUNTIME_NODE_ID,
    "tests/test_hidden_regime_checkpoint.py::"
    "test_resume_api_completes_from_a_json_roundtripped_midlife_checkpoint",
    "tests/test_hidden_regime_checkpoint.py::"
    "test_shuffled_channel_checkpoint_chunks_are_bit_exact",
    "tests/test_hidden_regime_checkpoint.py::"
    "test_factorial_condition_binding_survives_roundtrip_and_execution",
    "tests/test_hidden_regime_checkpoint.py::test_checkpoint_tampering_fails_closed",
    "tests/test_hidden_regime_checkpoint.py::"
    "test_chunk_schema_digest_order_gap_overlap_and_endpoint_tampering_fail_closed",
    "tests/test_hidden_regime_checkpoint.py::"
    "test_chunk_bounds_initial_terminal_and_protected_worlds_fail_closed",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_matches_atomic_replacement_for_all_factorial_axes",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_separates_durable_write_policy_from_replacement_policy",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_reconstructs_exhaustive_durable_search_and_scratch_reset",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_reconstructs_failed_scratch_residency_and_retest",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_reconstructs_consecutive_candidate_confirmation_gate",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_reconstructs_bias_corrected_relevance_and_mass_saturation",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_oracle_reconstructs_vacancy_fill_and_generation_exhaustion",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_strict_json_round_trip_and_named_role_streams",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_every_persistent_state_family_is_fail_closed_under_single_field_tampering",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_decision_key_consumption_and_diagnostics_reject_single_field_tampering",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_external_and_lifecycle_permits_are_independently_reconstructed",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_serialized_old_state_tampering_cannot_be_hidden_by_rehash_or_replay",
    "tests/test_slot_signaling_lifecycle_oracle.py::"
    "test_state_snapshots_validate_directly_and_adjacent_records_require_full_continuity",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_pre_delivered_contract_preserves_explicit_symbol_but_direct_is_strict",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_named_channel_contracts_match_runtime_and_advance_both_keys",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_channel_key_advances_identically_even_when_channel_does_not_draw",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_oracle_reconstructs_hold_and_repeat_cursors",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_oracle_reconstructs_segment_boundary_and_saturated_global_counter",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_strict_json_round_trip_and_schedule_permutation_binding",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_every_world_state_field_rejects_single_field_tampering",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_every_diagnostic_rejects_single_field_tampering",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_action_and_direct_message_tampering_changes_reconstructed_transition",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_state_validation_and_bit_exact_continuity_are_standalone",
    "tests/test_hidden_regime_world_oracle.py::"
    "test_oracle_matches_jit_produced_runtime_transition",
    "tests/test_slot_signaling_agent.py::test_config_is_strict_and_explicitly_development_only",
    "tests/test_slot_signaling_agent.py::"
    "test_explicit_factorial_policies_are_independent_and_legacy_resolves_exactly",
    "tests/test_slot_signaling_agent.py::"
    "test_compatibility_and_explicit_policies_are_transition_bit_exact",
    "tests/test_slot_signaling_agent.py::"
    "test_zero_state_has_three_vacancies_one_scratch_and_exact_resources",
    "tests/test_slot_signaling_agent.py::"
    "test_policy_api_has_no_regime_target_schedule_or_other_role_input",
    "tests/test_slot_signaling_agent.py::"
    "test_success_stays_then_failure_exhausts_stored_slots_before_scratch",
    "tests/test_slot_signaling_agent.py::"
    "test_default_one_scratch_lease_reproduces_immediate_retest",
    "tests/test_slot_signaling_agent.py::"
    "test_scratch_training_residency_is_write_mask_and_ablation_neutral",
    "tests/test_slot_signaling_agent.py::"
    "test_scratch_failure_counter_resets_on_exhaustion_success_and_commit",
    "tests/test_slot_signaling_agent.py::"
    "test_scratch_failure_counter_saturates_safely_and_routes_under_jit",
    "tests/test_slot_signaling_agent.py::"
    "test_search_skips_vacancies_and_never_retests_before_scratch",
    "tests/test_slot_signaling_agent.py::"
    "test_candidate_is_always_formed_while_external_mask_preserves_values",
    "tests/test_slot_signaling_agent.py::"
    "test_scratch_learns_and_sustained_reward_commits_then_resets_it",
    "tests/test_slot_signaling_agent.py::"
    "test_commit_activates_the_actual_nonfirst_vacancy",
    "tests/test_slot_signaling_agent.py::"
    "test_selective_durable_values_close_but_relevance_remains_separate",
    "tests/test_slot_signaling_agent.py::"
    "test_relevance_history_materially_decides_stay_or_search",
    "tests/test_slot_signaling_agent.py::"
    "test_hysteresis_relocks_moderate_durable_but_only_high_scratch_confirms",
    "tests/test_slot_signaling_agent.py::"
    "test_one_frozen_role_blocks_vacancy_commit_and_atomic_replacement",
    "tests/test_slot_signaling_agent.py::"
    "test_selective_durable_table_is_bitwise_immutable_while_being_tested",
    "tests/test_slot_signaling_agent.py::"
    "test_durable_mismatch_records_failure_without_deletion_or_value_change",
    "tests/test_slot_signaling_agent.py::"
    "test_confirmed_candidate_atomically_replaces_stale_generation",
    "tests/test_slot_signaling_agent.py::"
    "test_default_short_transient_cannot_replace_full_durable_bank",
    "tests/test_slot_signaling_agent.py::"
    "test_ab_recurrence_relocks_stored_module_without_scratch_replacement",
    "tests/test_slot_signaling_agent.py::"
    "test_writable_lru_ablation_replaces_oldest_slot_with_identical_resources",
    "tests/test_slot_signaling_agent.py::"
    "test_factorial_axes_independently_control_durable_writes_and_replacement_target",
    "tests/test_slot_signaling_agent.py::"
    "test_generation_exhaustion_blocks_commit_without_reusing_identity",
    "tests/test_slot_signaling_agent.py::"
    "test_public_role_transition_exactly_reproduces_joint_role_updates",
    "tests/test_slot_signaling_agent.py::"
    "test_separate_role_instances_reproduce_a_full_joint_lifecycle",
    "tests/test_slot_signaling_agent.py::"
    "test_selective_full_bank_waits_for_candidate_confirmation",
    "tests/test_slot_signaling_agent.py::"
    "test_named_keys_randomize_zero_ties_and_greedy_probe_is_read_only",
    "tests/test_slot_signaling_agent.py::"
    "test_joint_agent_is_jittable_scannable_finite_and_resource_constant",
)


CERTIFICATION_SPECS = (
    CertificationSpec(
        "complete_factorial_protocol_manifest_recurrence_gate_and_digest_contract",
        ("tests/test_hidden_regime_factorial_protocol.py",),
    ),
    CertificationSpec(
        "complete_development_producer_lineage_serialization_and_actual_transition_contract",
        ("tests/test_hidden_regime_signaling_development.py",),
    ),
    CertificationSpec(
        "complete_independent_generation_lineage_oracle",
        ("tests/test_hidden_regime_lineage_oracle.py",),
    ),
    CertificationSpec(
        "complete_role_world_summary_and_lineage_trace_audit",
        ("tests/test_hidden_regime_trace_audit.py",),
    ),
    CertificationSpec(
        "complete_independent_summary_and_resource_oracle",
        ("tests/test_hidden_regime_summary_oracle.py",),
    ),
    CertificationSpec(
        "managed_execution_authorization_consumption_and_protected_boundary",
        ("tests/test_hidden_regime_execution_governance.py",),
    ),
    CertificationSpec(
        "factorial_runner_worker_shard_ledger_coordinator_and_publication",
        ("tests/test_hidden_regime_factorial_calibration.py",),
    ),
    CertificationSpec(
        "threshold_freeze_engine_and_exact_worker_main_dispatch",
        (
            "tests/test_hidden_regime_factorial_thresholds.py",
            "tests/test_hidden_regime_factorial_calibration.py::"
            "test_threshold_worker_main_dispatches_exact_content_addressed_inputs",
        ),
    ),
    CertificationSpec(
        "protected_plan_derivation_and_exact_worker_main_dispatch",
        (
            "tests/test_hidden_regime_factorial_protected_plan.py",
            "tests/test_hidden_regime_factorial_calibration.py::"
            "test_protected_plan_worker_main_dispatches_exact_nonauthorizing_inputs",
        ),
    ),
    CertificationSpec(
        "checkpoint_resume_and_decentralized_role_bit_exact_equivalence",
        _CHECKPOINT_EQUIVALENCE_NODE_IDS,
        exact_file_test_inventory=True,
        checkpoint_cut_runtime_node_id=_CHECKPOINT_EQUIVALENCE_CUT_RUNTIME_NODE_ID,
        checkpoint_cut_fixture_name="direct_lifecycle_chunks",
        checkpoint_cut_trace_fixture_name="direct_one_shot",
        checkpoint_cut_ids=_CHECKPOINT_EQUIVALENCE_CUT_IDS,
        checkpoint_cut_semantics=_CHECKPOINT_EQUIVALENCE_CUT_SEMANTICS,
    ),
)


@dataclass(frozen=True, slots=True)
class ReadinessDraft:
    """Prepared source/protocol/runtime context, not an authorization receipt."""

    base_body: dict[str, object]
    source_archive: bytes
    repository_root: Path
    seal: str


@dataclass(frozen=True, slots=True)
class VerifiedCertificationBundle:
    """Process-sealed records emitted only by the certification verifier."""

    records: tuple[dict[str, object], ...]
    runtime_reconstruction_record: dict[str, object]
    source_manifest_sha256: str
    runtime_identity_sha256: str
    protocol_payload_sha256: str
    seal: str


@dataclass(frozen=True, slots=True)
class PreparedReadinessReceipt:
    """Final canonical receipt bytes and the exact bound source archive."""

    payload: dict[str, object]
    source_archive: bytes
    repository_root: Path
    seal: str


@dataclass(frozen=True, slots=True)
class ReadinessValidation:
    """Fail-closed validation result."""

    valid: bool
    ready_for_calibration: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedReadinessReceipt:
    """Paths of one immutable, content-addressed publication."""

    directory: Path
    receipt_path: Path
    source_archive_path: Path
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedReadinessBundle:
    """Strictly validated identities consumed by a bound calibration runner."""

    payload: dict[str, object]
    receipt_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    runtime_identity_sha256: str
    calibration_runner_module: str
    execution_genesis_sha256: str


@dataclass(frozen=True, slots=True)
class BoundCalibrationRuntimeBatch:
    """Non-transferable process-local authority for one full-scan-bracketed batch."""

    directory: Path
    receipt_sha256: str
    source_archive_sha256: str
    runtime_identity_sha256: str
    pid: int
    process_start_nonce: str
    nonce: str
    seal: str


@dataclass(frozen=True, slots=True)
class _DistributionRecordFileInventory:
    """Canonical content identity of exact files declared by distribution RECORDs."""

    versions: tuple[tuple[str, str], ...]
    file_count: int
    total_bytes: int
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class _StdlibFileInventory:
    """Canonical content identity of the bound Python standard-library tree."""

    file_count: int
    directory_count: int
    total_bytes: int
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class _ImportTreeFileInventory:
    """Canonical actual-byte identity of complete dependency import trees."""

    root_count: int
    file_count: int
    directory_count: int
    total_bytes: int
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class _RuntimePathBinding:
    """Exact interpreter and import-search paths used by isolated workers."""

    prefix: Path
    exec_prefix: Path
    purelib: Path
    platlib: Path
    stdlib: Path
    no_site_stdlib_search_paths: tuple[Path, ...]


def _fail(message: str) -> NoReturn:
    raise ReadinessError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_value(value: object, *, location: str = "$") -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{location} contains a non-string key")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise TypeError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return the receipt's ASCII, integer-only canonical JSON encoding."""

    _validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    """Hash canonical receipt JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_loads(raw: bytes) -> dict[str, object]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReadinessError("receipt JSON is not ASCII") from exc

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ReadinessError(f"non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReadinessError("receipt JSON is invalid") from exc
    _require(type(parsed) is dict, "receipt JSON must contain one plain object")
    result = cast(dict[str, object], parsed)
    _require(raw == canonical_json_bytes(result), "receipt JSON bytes are not canonical")
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_dict(value: object, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a plain object")
    return cast(dict[str, object], value)


def _expect_list(value: object, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a plain array")
    return cast(list[object], value)


def _expect_exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys differ from the exact schema")


def _seal(kind: str, payload: object, repository_root: Path) -> str:
    preimage = b"|".join(
        (
            kind.encode("ascii"),
            canonical_json_bytes(payload),
            os.fsencode(repository_root.absolute()),
            str(os.getpid()).encode("ascii"),
            _PROCESS_START_NONCE.encode("ascii"),
        )
    )
    return hmac.new(_PROCESS_SEAL_KEY, preimage, hashlib.sha256).hexdigest()


def _read_source_file(path: Path, repository_root: Path) -> bytes:
    """Read one stable regular non-symlink source member under the repository root."""

    root = repository_root.resolve(strict=True)
    try:
        relative = path.relative_to(repository_root)
    except ValueError as exc:
        raise ReadinessError("source member is outside the repository root") from exc
    _require(".." not in relative.parts, "source member traverses a parent")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ReadinessError(f"source member is missing: {relative.as_posix()}") from exc
    _require(stat.S_ISREG(before.st_mode), f"source member is not regular: {relative.as_posix()}")
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(root), f"source member escapes root: {relative.as_posix()}")
    _require(before.st_size <= _MAX_SOURCE_FILE_BYTES, f"source member is too large: {relative}")
    raw = path.read_bytes()
    after = path.lstat()
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"source member changed while read: {relative.as_posix()}",
    )
    return raw


def _module_candidates(repository_root: Path, module: str) -> tuple[Path, Path]:
    relative = Path(*module.split("."))
    return (
        repository_root / relative.with_suffix(".py"),
        repository_root / relative / "__init__.py",
    )


def _module_path(repository_root: Path, module: str) -> Path | None:
    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    for candidate in _module_candidates(repository_root, module):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            return candidate
        _fail(f"local module is not a regular file: {candidate.relative_to(repository_root)}")
    return None


def _module_name(repository_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(repository_root)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _parent_packages(module: str) -> set[str]:
    parts = module.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts))}


def _resolve_local_imports(repository_root: Path, path: Path, raw: bytes) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=path.as_posix())
    except (SyntaxError, UnicodeError) as exc:
        raise ReadinessError(f"cannot parse source member {path}: {exc}") from exc
    module, is_package = _module_name(repository_root, path)
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("alberta_framework"):
                    _require(
                        _module_path(repository_root, alias.name) is not None,
                        f"local source import is missing: {alias.name}",
                    )
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                if base.startswith("alberta_framework"):
                    _require(
                        _module_path(repository_root, base) is not None,
                        f"local source import is missing: {base}",
                    )
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for candidate in candidates:
            parts = candidate.split(".")
            while parts:
                possible = ".".join(parts)
                if _module_path(repository_root, possible) is not None:
                    found.add(possible)
                    found.update(_parent_packages(possible))
                    break
                parts.pop()
    return found


def _certification_source_paths() -> tuple[Path, ...]:
    paths = {Path("tests/conftest.py")}
    for spec in CERTIFICATION_SPECS:
        paths.update(Path(node_id.split("::", 1)[0]) for node_id in spec.node_ids)
    return tuple(sorted(paths, key=Path.as_posix))


def _validate_certification_node_sources(repository_root: Path) -> None:
    functions_by_path: dict[
        Path,
        dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    ] = {}
    certification_ids: set[str] = set()
    for spec in CERTIFICATION_SPECS:
        _require(spec.certification_id not in certification_ids, "duplicate certification ID")
        certification_ids.add(spec.certification_id)
        _require(bool(spec.node_ids), "certification node inventory is empty")
        _require(
            len(spec.node_ids) == len(set(spec.node_ids)),
            f"certification node inventory contains duplicates: {spec.certification_id}",
        )
        expected_tests_by_path: dict[Path, set[str]] = {}
        for node_id in spec.node_ids:
            locator_text, separator, function_name = node_id.partition("::")
            _require(
                not separator
                or (
                    separator == "::"
                    and bool(function_name)
                    and "::" not in function_name
                ),
                f"invalid node ID: {node_id}",
            )
            locator = Path(locator_text)
            if locator not in functions_by_path:
                raw = _read_source_file(repository_root / locator, repository_root)
                try:
                    tree = ast.parse(raw.decode("utf-8"), filename=locator.as_posix())
                except (SyntaxError, UnicodeError) as exc:
                    raise ReadinessError(f"cannot parse certification source {locator}") from exc
                function_nodes = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                seen_test_names: set[str] = set()
                duplicate_test_names: set[str] = set()
                for function_node in function_nodes:
                    if not function_node.name.startswith("test_"):
                        continue
                    if function_node.name in seen_test_names:
                        duplicate_test_names.add(function_node.name)
                    seen_test_names.add(function_node.name)
                _require(
                    not duplicate_test_names,
                    "certification source contains duplicate top-level test definitions: "
                    f"{locator.as_posix()}: {sorted(duplicate_test_names)!r}",
                )
                functions_by_path[locator] = {node.name: node for node in function_nodes}
            if separator:
                _require(
                    function_name in functions_by_path[locator],
                    f"certification node is absent from its source: {node_id}",
                )
                expected_tests_by_path.setdefault(locator, set()).add(function_name)
            else:
                _require(
                    any(name.startswith("test_") for name in functions_by_path[locator]),
                    f"full-file certification source contains no tests: {node_id}",
                )
                _require(
                    not spec.exact_file_test_inventory,
                    f"exact certification inventory cannot use a file selector: {node_id}",
                )

        if spec.exact_file_test_inventory:
            for locator, expected_tests in expected_tests_by_path.items():
                actual_tests = {
                    name for name in functions_by_path[locator] if name.startswith("test_")
                }
                _require(
                    actual_tests == expected_tests,
                    "exact certification test inventory differs for "
                    f"{locator.as_posix()}: expected={sorted(expected_tests)!r}; "
                    f"actual={sorted(actual_tests)!r}",
                )

        cut_node_id = spec.checkpoint_cut_runtime_node_id
        cut_fixture_name = spec.checkpoint_cut_fixture_name
        cut_trace_fixture_name = spec.checkpoint_cut_trace_fixture_name
        cut_ids = spec.checkpoint_cut_ids
        cut_semantics = spec.checkpoint_cut_semantics
        _require(
            len(
                {
                    cut_node_id is None,
                    cut_fixture_name is None,
                    cut_trace_fixture_name is None,
                    not cut_ids,
                    not cut_semantics,
                }
            )
            == 1,
            f"checkpoint-cut manifest is incomplete: {spec.certification_id}",
        )
        _require(
            len(cut_ids) == len(set(cut_ids)),
            f"checkpoint-cut manifest contains duplicates: {spec.certification_id}",
        )
        if cut_node_id is not None:
            _require(
                cut_node_id in spec.node_ids,
                f"checkpoint-cut runtime node is not certified: {cut_node_id}",
            )
            _require(
                type(cut_fixture_name) is str and bool(cut_fixture_name),
                f"checkpoint-cut fixture name is invalid: {spec.certification_id}",
            )
            _require(
                type(cut_trace_fixture_name) is str and bool(cut_trace_fixture_name),
                f"checkpoint-cut trace fixture name is invalid: {spec.certification_id}",
            )
            _require(
                tuple(semantic[0] for semantic in cut_semantics) == cut_ids,
                f"checkpoint-cut semantic order differs: {spec.certification_id}",
            )
            for semantic in cut_semantics:
                _require(
                    len(semantic) == 5
                    and all(type(field) is str and bool(field) for field in semantic[:3])
                    and semantic[2]
                    in {
                        "strictly_between_zero_and_three",
                        "truthy",
                        "adjacent_change_new_index",
                        "nonnegative",
                    }
                    and _is_strict_int(semantic[3])
                    and semantic[3] >= 0
                    and _is_strict_int(semantic[4])
                    and semantic[4] >= 0,
                    f"checkpoint-cut semantic contract is invalid: {spec.certification_id}",
                )


def _build_source_bundle(
    repository_root: Path,
) -> tuple[dict[str, object], bytes]:
    repository_root = repository_root.absolute()
    _require(repository_root.is_dir(), "repository root must be a directory")
    _validate_certification_node_sources(repository_root)
    roots = _BASE_SOURCE_ROOT_MODULES
    _require(
        _module_path(repository_root, _CALIBRATION_RUNNER_MODULE) is not None,
        "mandatory calibration runner module does not exist",
    )
    _require(
        _module_path(repository_root, _EXECUTION_GOVERNANCE_MODULE) is not None,
        "mandatory execution governance module does not exist",
    )
    _require(
        _module_path(repository_root, _SUMMARY_ORACLE_MODULE) is not None,
        "mandatory independent summary oracle module does not exist",
    )

    pending = set(roots)
    pending.update(parent for module in roots for parent in _parent_packages(module))
    visited: set[str] = set()
    bytes_by_module: dict[str, bytes] = {}
    path_by_module: dict[str, Path] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        path = _module_path(repository_root, module)
        _require(path is not None, f"source closure module is missing: {module}")
        assert path is not None
        raw = _read_source_file(path, repository_root)
        visited.add(module)
        path_by_module[module] = path
        bytes_by_module[module] = raw
        pending.update(_resolve_local_imports(repository_root, path, raw) - visited)

    source_entries: list[dict[str, object]] = []
    archive_members: dict[str, bytes] = {}
    for module in sorted(visited):
        path = path_by_module[module]
        locator = path.relative_to(repository_root).as_posix()
        raw = bytes_by_module[module]
        source_entries.append(
            {
                "module": module,
                "locator": locator,
                "byte_size": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
        archive_members[locator] = raw

    support_entries: list[dict[str, object]] = []
    support_paths = (
        *((path, "dependency_lock") for path in _LOCK_FILES),
        *((path, "certification_source") for path in _certification_source_paths()),
    )
    for relative, role in support_paths:
        locator = relative.as_posix()
        _require(locator not in archive_members, f"duplicate archive member: {locator}")
        raw = _read_source_file(repository_root / relative, repository_root)
        support_entries.append(
            {
                "locator": locator,
                "role": role,
                "byte_size": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
        archive_members[locator] = raw

    manifest: dict[str, object] = {
        "schema": READINESS_SOURCE_SCHEMA,
        "closure_kind": "static_transitive_local_python_imports",
        "repository_subtree": "research/alberta",
        "root_modules": list(roots),
        "calibration_runner_module": _CALIBRATION_RUNNER_MODULE,
        "files": source_entries,
        "support_files": support_entries,
    }
    archive = _deterministic_source_zip(archive_members)
    return manifest, archive


def _deterministic_source_zip(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as zf:
        zf.comment = b""
        for locator in sorted(members):
            pure = PurePosixPath(locator)
            _require(
                not pure.is_absolute()
                and ".." not in pure.parts
                and "\\" not in locator
                and pure.as_posix() == locator,
                f"unsafe ZIP member locator: {locator}",
            )
            info = zipfile.ZipInfo(locator, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _ZIP_FILE_MODE << 16
            info.extra = b""
            info.comment = b""
            buffer_data = members[locator]
            zf.writestr(info, buffer_data)
    raw = buffer.getvalue()
    _require(len(raw) <= _MAX_ARCHIVE_BYTES, "source archive exceeds the size limit")
    return raw


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return mutation- and substitution-sensitive regular-file stat fields."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_hash_record_file(
    path: Path,
    *,
    capture_limit: int | None = None,
) -> tuple[int, str, bytes | None, tuple[int, int, int, int, int, int, int]]:
    """Hash one RECORD target through a stable non-symlink descriptor.

    The pathname is checked before opening, the final component is opened with
    ``O_NOFOLLOW``, and the descriptor and pathname identities must still agree
    after the complete read.  Parent components are opened one at a time with
    ``O_NOFOLLOW`` by ``_open_directory_without_symlinks``.
    """

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before_path = absolute.lstat()
    except FileNotFoundError as exc:
        raise ReadinessError(f"RECORD-listed file is missing: {absolute}") from exc
    _require(
        stat.S_ISREG(before_path.st_mode),
        f"RECORD-listed file is not a regular non-symlink file: {absolute}",
    )
    if capture_limit is not None:
        _require(
            before_path.st_size <= capture_limit,
            f"RECORD-listed metadata file exceeds its size limit: {absolute}",
        )

    parent_fd, absolute_parent = _open_directory_without_symlinks(absolute.parent)
    normalized = absolute_parent / absolute.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ReadinessError(
                    f"RECORD-listed file is symlinked or substituted: {normalized}"
                ) from exc
            if exc.errno == errno.ENOENT:
                raise ReadinessError(f"RECORD-listed file is missing: {normalized}") from exc
            raise
        try:
            before_descriptor = os.fstat(descriptor)
            before_locator = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            initial_identity = _stat_identity(before_path)
            _require(
                stat.S_ISREG(before_descriptor.st_mode)
                and initial_identity
                == _stat_identity(before_descriptor)
                == _stat_identity(before_locator),
                f"RECORD-listed path was substituted before hashing: {normalized}",
            )
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if capture_limit is not None else None
            total = 0
            while True:
                remaining_with_mutation_probe = before_descriptor.st_size - total + 1
                chunk = os.read(descriptor, min(1024 * 1024, max(1, remaining_with_mutation_probe)))
                if not chunk:
                    break
                total += len(chunk)
                _require(
                    total <= before_descriptor.st_size,
                    f"RECORD-listed file changed size while hashed: {normalized}",
                )
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after_descriptor = os.fstat(descriptor)
            after_locator = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            try:
                after_path = absolute.lstat()
            except FileNotFoundError as exc:
                raise ReadinessError(
                    f"RECORD-listed file was removed while hashed: {normalized}"
                ) from exc
            _require(
                total == before_descriptor.st_size
                and initial_identity
                == _stat_identity(after_descriptor)
                == _stat_identity(after_locator)
                == _stat_identity(after_path),
                f"RECORD-listed file changed or was replaced while hashed: {normalized}",
            )
            raw = b"".join(chunks) if chunks is not None else None
            return (
                total,
                digest.hexdigest(),
                raw,
                initial_identity,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _normalize_distribution_name(name: str) -> str:
    _require(bool(name), "distribution metadata name is empty")
    normalized = re.sub(r"[-_.]+", "-", name).casefold()
    _require(bool(normalized), "distribution metadata name normalizes to empty")
    return normalized


def _path_identity_without_symlinks(
    path: Path,
) -> tuple[int, int, int, int, int, int, int]:
    parent_fd, absolute_parent = _open_directory_without_symlinks(path.parent)
    try:
        metadata = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ReadinessError(f"inventory path is missing during final recheck: {path}") from exc
    finally:
        os.close(parent_fd)
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"inventory path is not a regular non-symlink file: {absolute_parent / path.name}",
    )
    return _stat_identity(metadata)


def _directory_identity_without_symlinks(
    path: Path,
) -> tuple[int, int, int, int, int, int, int]:
    try:
        descriptor, absolute = _open_directory_without_symlinks(path)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ReadinessError(
            f"inventory directory is missing or symlinked during final recheck: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISDIR(metadata.st_mode), f"inventory path is not a directory: {absolute}")
        return _stat_identity(metadata)
    finally:
        os.close(descriptor)


def _validated_record_locator(raw: object) -> tuple[str, PurePosixPath]:
    if type(raw) is str:
        locator = raw
    elif isinstance(raw, os.PathLike):
        located_raw = os.fspath(raw)
        _require(type(located_raw) is str, "RECORD path locator is not text")
        locator = cast(str, located_raw)
    else:
        _fail("RECORD path locator is not path-like")
    _require(type(locator) is str and bool(locator), "RECORD path locator is empty or non-text")
    _require("\x00" not in locator and "\\" not in locator, "RECORD path locator is unsafe")
    _require(not re.match(r"^[A-Za-z]:", locator), "RECORD path locator is absolute")
    pure = PurePosixPath(locator)
    _require(not pure.is_absolute(), "RECORD path locator is absolute")
    _require(
        pure.as_posix() == locator and pure.name not in {"", ".", ".."},
        f"RECORD path locator is not canonical: {locator}",
    )
    return locator, pure


def _located_record_path(
    distribution: importlib.metadata.Distribution,
    locator: str,
    *,
    runtime_prefix: Path,
) -> Path:
    try:
        located_raw = distribution.locate_file(importlib.metadata.PackagePath(locator))
    except (OSError, TypeError, ValueError) as exc:
        raise ReadinessError(f"cannot locate RECORD path: {locator}") from exc
    located = Path(os.path.abspath(str(located_raw)))
    _require(
        located.is_relative_to(runtime_prefix),
        f"RECORD path locator escapes the runtime prefix: {locator}",
    )
    return located


def _parse_record_locators(raw: bytes) -> list[tuple[str, PurePosixPath]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReadinessError("distribution RECORD is not strict UTF-8 CSV") from exc
    _require(bool(rows), "distribution RECORD is empty")
    parsed: list[tuple[str, PurePosixPath]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        _require(len(row) == 3, f"distribution RECORD row {row_number} is not three fields")
        locator, pure = _validated_record_locator(row[0])
        _require(locator not in seen, f"duplicate RECORD path locator: {locator}")
        seen.add(locator)
        parsed.append((locator, pure))
    return parsed


def _distribution_record_file_inventory(
    distributions: Sequence[importlib.metadata.Distribution],
    *,
    runtime_prefix: Path,
) -> _DistributionRecordFileInventory:
    """Build the canonical actual-byte inventory for explicit distributions."""

    prefix = Path(os.path.abspath(os.fspath(runtime_prefix)))
    _require(prefix.is_dir(), "runtime prefix is not a directory")
    inventory_rows: list[dict[str, object]] = []
    versions: list[tuple[str, str]] = []
    normalized_names: set[str] = set()
    resolved_targets: set[Path] = set()
    resolved_file_identities: set[tuple[int, int]] = set()
    final_path_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    total_bytes = 0

    for distribution in distributions:
        declared_files = distribution.files
        _require(declared_files is not None, "installed distribution has no RECORD file inventory")
        assert declared_files is not None
        record_candidates: list[tuple[str, PurePosixPath]] = []
        for declared in declared_files:
            locator, pure = _validated_record_locator(declared)
            if pure.name == "RECORD" and pure.parent.name.endswith(".dist-info"):
                record_candidates.append((locator, pure))
        _require(
            len(record_candidates) == 1,
            "installed distribution must expose exactly one .dist-info/RECORD locator",
        )
        record_locator, record_pure = record_candidates[0]
        record_path = _located_record_path(
            distribution,
            record_locator,
            runtime_prefix=prefix,
        )
        _record_size, record_sha256, record_raw, _record_identity = _stable_hash_record_file(
            record_path,
            capture_limit=_MAX_SOURCE_FILE_BYTES,
        )
        assert record_raw is not None
        record_rows = _parse_record_locators(record_raw)
        record_rows_by_locator = {locator: pure for locator, pure in record_rows}
        _require(
            record_rows_by_locator.get(record_locator) == record_pure,
            "distribution RECORD does not list its exact RECORD locator",
        )
        nested_records = [
            locator
            for locator, pure in record_rows
            if pure.name == "RECORD" and pure.parent.name.endswith(".dist-info")
        ]
        _require(
            nested_records == [record_locator],
            "distribution RECORD declares an ambiguous RECORD locator",
        )
        metadata_locator = (record_pure.parent / "METADATA").as_posix()
        _require(
            metadata_locator in record_rows_by_locator,
            "distribution RECORD does not list its exact METADATA file",
        )

        distribution_rows: list[tuple[str, int, str]] = []
        metadata_raw: bytes | None = None
        for locator, _pure in record_rows:
            located = _located_record_path(distribution, locator, runtime_prefix=prefix)
            _require(
                located not in resolved_targets,
                f"duplicate or colliding resolved RECORD path: {locator}",
            )
            resolved_targets.add(located)
            capture_limit = _MAX_SOURCE_FILE_BYTES if locator == metadata_locator else None
            byte_size, sha256, captured, file_identity = _stable_hash_record_file(
                located,
                capture_limit=capture_limit,
            )
            resolved_file_identity = (file_identity[0], file_identity[1])
            _require(
                resolved_file_identity not in resolved_file_identities,
                f"duplicate or hard-linked RECORD target: {locator}",
            )
            resolved_file_identities.add(resolved_file_identity)
            final_path_identities[located] = file_identity
            if locator == metadata_locator:
                assert captured is not None
                metadata_raw = captured
            distribution_rows.append((locator, byte_size, sha256))

        final_size, final_sha256, final_record_raw, final_record_identity = (
            _stable_hash_record_file(
                record_path,
                capture_limit=_MAX_SOURCE_FILE_BYTES,
            )
        )
        _require(
            final_size == _record_size
            and final_sha256 == record_sha256
            and final_record_raw == record_raw
            and final_record_identity == _record_identity,
            "distribution RECORD changed while its file inventory was hashed",
        )
        assert metadata_raw is not None
        message = BytesParser().parsebytes(metadata_raw, headersonly=True)
        name_headers = message.get_all("Name", [])
        version_headers = message.get_all("Version", [])
        _require(len(name_headers) == 1, "distribution METADATA must contain one Name")
        _require(len(version_headers) == 1, "distribution METADATA must contain one Version")
        name = str(name_headers[0])
        version = str(version_headers[0])
        normalized_name = _normalize_distribution_name(name)
        _require(
            normalized_name not in normalized_names,
            f"duplicate normalized installed distribution name: {normalized_name}",
        )
        normalized_names.add(normalized_name)
        api_name = distribution.metadata.get("Name")
        api_version = distribution.version
        _require(
            type(api_name) is str and _normalize_distribution_name(api_name) == normalized_name,
            f"distribution API name differs from stable METADATA: {normalized_name}",
        )
        _require(
            type(api_version) is str and api_version == version,
            f"distribution API version differs from stable METADATA: {normalized_name}",
        )
        versions.append((normalized_name, version))
        for locator, byte_size, sha256 in distribution_rows:
            inventory_rows.append(
                {
                    "normalized_distribution_name": normalized_name,
                    "distribution_version": version,
                    "record_path_locator": locator,
                    "byte_size": byte_size,
                    "sha256": sha256,
                }
            )
            total_bytes += byte_size

    inventory_rows.sort(
        key=lambda item: (
            cast(str, item["normalized_distribution_name"]),
            cast(str, item["distribution_version"]),
            cast(str, item["record_path_locator"]),
        )
    )
    for path, expected_identity in sorted(
        final_path_identities.items(),
        key=lambda item: item[0].as_posix(),
    ):
        _require(
            _path_identity_without_symlinks(path) == expected_identity,
            f"installed distribution file drifted after hashing: {path}",
        )
    return _DistributionRecordFileInventory(
        versions=tuple(sorted(versions)),
        file_count=len(inventory_rows),
        total_bytes=total_bytes,
        inventory_sha256=canonical_sha256(inventory_rows),
    )


def _absolute_runtime_directory(raw: str, label: str) -> Path:
    path = Path(os.path.abspath(raw))
    try:
        descriptor, absolute = _open_directory_without_symlinks(path)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ReadinessError(f"runtime {label} is not an exact non-symlink directory") from exc
    try:
        _require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"runtime {label} is not a directory")
    finally:
        os.close(descriptor)
    return absolute


def _runtime_path_binding() -> _RuntimePathBinding:
    prefix = _absolute_runtime_directory(sys.prefix, "prefix")
    exec_prefix = _absolute_runtime_directory(sys.exec_prefix, "exec prefix")
    purelib = _absolute_runtime_directory(sysconfig.get_path("purelib"), "purelib")
    platlib = _absolute_runtime_directory(sysconfig.get_path("platlib"), "platlib")
    stdlib = _absolute_runtime_directory(sysconfig.get_path("stdlib"), "stdlib")
    _require(
        purelib.is_relative_to(prefix),
        "runtime purelib is outside the exact interpreter prefix",
    )
    _require(
        platlib.is_relative_to(exec_prefix),
        "runtime platlib is outside the exact interpreter exec prefix",
    )
    python_zip = stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    lib_dynload = stdlib / "lib-dynload"
    _require(lib_dynload.is_dir(), "runtime standard-library dynamic-loader path is absent")
    no_site_paths = (python_zip, stdlib, lib_dynload)
    _require(
        len(no_site_paths) == len(set(no_site_paths)),
        "runtime no-site standard-library paths are not unique",
    )
    return _RuntimePathBinding(
        prefix=prefix,
        exec_prefix=exec_prefix,
        purelib=purelib,
        platlib=platlib,
        stdlib=stdlib,
        no_site_stdlib_search_paths=no_site_paths,
    )


def _installed_distribution_record_file_inventory(
    paths: _RuntimePathBinding,
) -> _DistributionRecordFileInventory:
    search_paths = tuple(dict.fromkeys((paths.purelib.as_posix(), paths.platlib.as_posix())))
    distributions = tuple(importlib.metadata.distributions(path=list(search_paths)))
    _require(bool(distributions), "bound runtime paths contain no installed distributions")
    inventory = _distribution_record_file_inventory(
        distributions,
        runtime_prefix=paths.prefix,
    )
    available_names = {name for name, _version in inventory.versions}
    _require(
        set(_KEY_DISTRIBUTIONS).issubset(available_names),
        "installed distribution names omit a required key runtime distribution",
    )
    return inventory


def _import_tree_entry_paths(
    root: Path,
    *,
    label: str,
) -> tuple[Path, ...]:
    """Return the root and every directory, regular file, and symlink in an import tree."""

    paths: list[Path] = [root]
    for directory, raw_directories, raw_files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(raw_directories):
            path = parent / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                paths.append(path)
            else:
                _require(
                    stat.S_ISDIR(metadata.st_mode),
                    f"{label} tree entry is not a directory: {path}",
                )
                paths.append(path)
                kept_directories.append(name)
        raw_directories[:] = kept_directories
        for name in sorted(raw_files):
            path = parent / name
            metadata = path.lstat()
            _require(
                stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode),
                f"{label} tree entry is neither regular nor symlink: {path}",
            )
            paths.append(path)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _stdlib_entry_paths(root: Path) -> tuple[Path, ...]:
    return _import_tree_entry_paths(root, label="standard-library")


def _stable_hash_import_tree_symlink(
    path: Path,
    *,
    label: str,
) -> tuple[
    int,
    str,
    str,
    str,
    tuple[int, int, int, int, int, int, int],
    tuple[int, int, int, int, int, int, int],
]:
    before = path.lstat()
    _require(stat.S_ISLNK(before.st_mode), f"{label} link is not a symlink: {path}")
    link_target = os.readlink(path)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ReadinessError(f"{label} symlink target is invalid: {path}") from exc
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{label} symlink does not resolve to a regular file: {path}",
    )
    byte_size, sha256, _raw, target_identity = _stable_hash_record_file(resolved)
    after = path.lstat()
    _require(
        _stat_identity(before) == _stat_identity(after) and os.readlink(path) == link_target,
        f"{label} symlink changed while hashed: {path}",
    )
    return (
        byte_size,
        sha256,
        link_target,
        resolved.as_posix(),
        _stat_identity(before),
        target_identity,
    )


def _require_import_tree_symlink_identity(
    path: Path,
    *,
    label: str,
    expected_link_target: str,
    expected_resolved_target: str,
    expected_link_identity: tuple[int, int, int, int, int, int, int],
    expected_target_identity: tuple[int, int, int, int, int, int, int],
) -> None:
    metadata = path.lstat()
    _require(
        stat.S_ISLNK(metadata.st_mode)
        and _stat_identity(metadata) == expected_link_identity
        and os.readlink(path) == expected_link_target,
        f"{label} symlink drifted after hashing: {path}",
    )
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ReadinessError(f"{label} symlink target drifted after hashing: {path}") from exc
    _require(
        resolved.as_posix() == expected_resolved_target
        and _path_identity_without_symlinks(resolved) == expected_target_identity,
        f"{label} symlink target drifted after hashing: {path}",
    )


def _dependency_import_tree_file_inventory(
    paths: _RuntimePathBinding,
) -> _ImportTreeFileInventory:
    """Bind every importable-tree file, including files unowned by a distribution."""

    roots = tuple(dict.fromkeys((paths.purelib, paths.platlib)))
    _require(bool(roots), "dependency import tree has no roots")
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            _require(
                not first.is_relative_to(second) and not second.is_relative_to(first),
                "dependency import roots overlap",
            )
    initial_paths_by_root = {
        root: _import_tree_entry_paths(root, label="dependency import") for root in roots
    }
    rows: list[dict[str, object]] = []
    directory_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    regular_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    symlink_identities: dict[
        Path,
        tuple[
            str,
            str,
            tuple[int, int, int, int, int, int, int],
            tuple[int, int, int, int, int, int, int],
        ],
    ] = {}
    total_bytes = 0
    for root_index, root in enumerate(roots):
        for path in initial_paths_by_root[root]:
            locator = path.relative_to(root).as_posix()
            metadata = path.lstat()
            row: dict[str, object]
            if stat.S_ISDIR(metadata.st_mode):
                directory_identities[path] = _directory_identity_without_symlinks(path)
                row = {
                    "root_index": root_index,
                    "locator": locator,
                    "entry_kind": "directory",
                }
                byte_size = 0
            elif stat.S_ISLNK(metadata.st_mode):
                (
                    byte_size,
                    sha256,
                    link_target,
                    resolved_target,
                    link_identity,
                    target_identity,
                ) = _stable_hash_import_tree_symlink(path, label="dependency import")
                symlink_identities[path] = (
                    link_target,
                    resolved_target,
                    link_identity,
                    target_identity,
                )
                row = {
                    "root_index": root_index,
                    "locator": locator,
                    "entry_kind": "symlink_to_regular_file",
                    "link_target": link_target,
                    "resolved_target": resolved_target,
                    "target_byte_size": byte_size,
                    "target_sha256": sha256,
                }
            else:
                byte_size, sha256, _raw, identity = _stable_hash_record_file(path)
                regular_identities[path] = identity
                row = {
                    "root_index": root_index,
                    "locator": locator,
                    "entry_kind": "regular_file",
                    "byte_size": byte_size,
                    "sha256": sha256,
                }
            total_bytes += byte_size
            rows.append(row)
    for root in roots:
        _require(
            _import_tree_entry_paths(root, label="dependency import")
            == initial_paths_by_root[root],
            f"dependency import tree changed while hashed: {root}",
        )
    for path, expected_identity in regular_identities.items():
        _require(
            _path_identity_without_symlinks(path) == expected_identity,
            f"dependency import file drifted after hashing: {path}",
        )
    for path, expected_identity in directory_identities.items():
        _require(
            _directory_identity_without_symlinks(path) == expected_identity,
            f"dependency import directory drifted after hashing: {path}",
        )
    for path, (
        expected_link_target,
        expected_resolved_target,
        expected_link_identity,
        expected_target_identity,
    ) in symlink_identities.items():
        _require_import_tree_symlink_identity(
            path,
            label="dependency import",
            expected_link_target=expected_link_target,
            expected_resolved_target=expected_resolved_target,
            expected_link_identity=expected_link_identity,
            expected_target_identity=expected_target_identity,
        )
    return _ImportTreeFileInventory(
        root_count=len(roots),
        file_count=len(regular_identities) + len(symlink_identities),
        directory_count=len(directory_identities),
        total_bytes=total_bytes,
        inventory_sha256=canonical_sha256(rows),
    )


def _stdlib_file_inventory(paths: _RuntimePathBinding) -> _StdlibFileInventory:
    initial_paths = _stdlib_entry_paths(paths.stdlib)
    rows: list[dict[str, object]] = []
    directory_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    regular_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    symlink_identities: dict[
        Path,
        tuple[
            str,
            str,
            tuple[int, int, int, int, int, int, int],
            tuple[int, int, int, int, int, int, int],
        ],
    ] = {}
    absent_paths: set[Path] = set()
    total_bytes = 0
    for path in initial_paths:
        locator = path.relative_to(paths.stdlib).as_posix()
        metadata = path.lstat()
        row: dict[str, object]
        if stat.S_ISDIR(metadata.st_mode):
            directory_identities[path] = _directory_identity_without_symlinks(path)
            row = {
                "locator": locator,
                "entry_kind": "directory",
            }
            byte_size = 0
        elif stat.S_ISLNK(metadata.st_mode):
            (
                byte_size,
                sha256,
                link_target,
                resolved_target,
                link_identity,
                target_identity,
            ) = _stable_hash_import_tree_symlink(path, label="standard-library")
            symlink_identities[path] = (
                link_target,
                resolved_target,
                link_identity,
                target_identity,
            )
            row = {
                "locator": locator,
                "entry_kind": "symlink_to_regular_file",
                "link_target": link_target,
                "resolved_target": resolved_target,
                "target_byte_size": byte_size,
                "target_sha256": sha256,
            }
        else:
            byte_size, sha256, _raw, identity = _stable_hash_record_file(path)
            regular_identities[path] = identity
            row = {
                "locator": locator,
                "entry_kind": "regular_file",
                "byte_size": byte_size,
                "sha256": sha256,
            }
        total_bytes += byte_size
        rows.append(row)
    for path in paths.no_site_stdlib_search_paths:
        if path == paths.stdlib or path.is_relative_to(paths.stdlib):
            continue
        locator = f"@no-site-search-path:{path.as_posix()}"
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            absent_paths.add(path)
            rows.append({"locator": locator, "entry_kind": "absent_path"})
            continue
        if stat.S_ISLNK(metadata.st_mode):
            (
                byte_size,
                sha256,
                link_target,
                resolved_target,
                link_identity,
                target_identity,
            ) = _stable_hash_import_tree_symlink(path, label="standard-library")
            symlink_identities[path] = (
                link_target,
                resolved_target,
                link_identity,
                target_identity,
            )
            rows.append(
                {
                    "locator": locator,
                    "entry_kind": "symlink_to_regular_file",
                    "link_target": link_target,
                    "resolved_target": resolved_target,
                    "target_byte_size": byte_size,
                    "target_sha256": sha256,
                }
            )
        else:
            _require(
                stat.S_ISREG(metadata.st_mode),
                f"no-site standard-library search path is not regular: {path}",
            )
            byte_size, sha256, _raw, identity = _stable_hash_record_file(path)
            regular_identities[path] = identity
            rows.append(
                {
                    "locator": locator,
                    "entry_kind": "regular_file",
                    "byte_size": byte_size,
                    "sha256": sha256,
                }
            )
        total_bytes += byte_size
    _require(
        _stdlib_entry_paths(paths.stdlib) == initial_paths,
        "standard-library tree changed while its inventory was hashed",
    )
    for path, expected_identity in regular_identities.items():
        _require(
            _path_identity_without_symlinks(path) == expected_identity,
            f"standard-library file drifted after hashing: {path}",
        )
    for path, expected_identity in directory_identities.items():
        _require(
            _directory_identity_without_symlinks(path) == expected_identity,
            f"standard-library directory drifted after hashing: {path}",
        )
    for path, (
        expected_link_target,
        expected_resolved_target,
        expected_link_identity,
        expected_target_identity,
    ) in symlink_identities.items():
        _require_import_tree_symlink_identity(
            path,
            label="standard-library",
            expected_link_target=expected_link_target,
            expected_resolved_target=expected_resolved_target,
            expected_link_identity=expected_link_identity,
            expected_target_identity=expected_target_identity,
        )
    for path in absent_paths:
        _require(
            not path.exists() and not path.is_symlink(),
            f"absent runtime path appeared: {path}",
        )
    return _StdlibFileInventory(
        file_count=len(regular_identities) + len(symlink_identities),
        directory_count=len(directory_identities),
        total_bytes=total_bytes,
        inventory_sha256=canonical_sha256(rows),
    )


def _runtime_value(value: object) -> str | int | bool | None:
    if value is None or type(value) in (str, int, bool):
        return cast(str | int | bool | None, value)
    if type(value) is float:
        _require(math.isfinite(value), "runtime configuration is non-finite")
        return repr(value)
    return repr(value)


def _build_child_environment() -> dict[str, str]:
    """Construct the complete child environment without copying the parent."""

    return {"LC_ALL": "C", "PYTHONHASHSEED": "0"}


def _validated_child_environment(value: object) -> dict[str, str]:
    child_raw = _expect_dict(value, "child environment")
    child: dict[str, str] = {}
    for name, raw_value in child_raw.items():
        _require(
            bool(name)
            and "=" not in name
            and "\x00" not in name
            and name not in _FORBIDDEN_CHILD_ENVIRONMENT_NAMES,
            f"child environment name is forbidden: {name!r}",
        )
        _require(type(raw_value) is str and "\x00" not in raw_value, "child environment value")
        _require(
            name in {"LC_ALL", "PYTHONHASHSEED"},
            f"child environment name is outside the allowlist: {name}",
        )
        child[name] = cast(str, raw_value)
    _require(
        child == {"LC_ALL": "C", "PYTHONHASHSEED": "0"},
        "child environment must be exactly LC_ALL=C and PYTHONHASHSEED=0",
    )
    _require(
        list(child) == sorted(child),
        "child environment fields are not in canonical sorted order",
    )
    return child


def _build_runtime_identity() -> dict[str, object]:
    import jax

    runtime_paths = _runtime_path_binding()
    distribution_inventory = _installed_distribution_record_file_inventory(runtime_paths)
    dependency_tree_inventory = _dependency_import_tree_file_inventory(runtime_paths)
    stdlib_inventory = _stdlib_file_inventory(runtime_paths)
    versions = dict(distribution_inventory.versions)
    key_versions = {name: versions[name] for name in _KEY_DISTRIBUTIONS}
    library_environment_side_effects = _validated_library_environment_side_effects()
    environment: list[dict[str, object]] = []
    executable = Path(sys.executable).resolve(strict=True)
    _executable_size, executable_sha256, _executable_raw, _executable_identity = (
        _stable_hash_record_file(executable)
    )
    config_values = {
        name: _runtime_value(value) for name, value in sorted(jax.config.values.items())
    }
    devices = _jax_device_identity(jax)
    return {
        "schema": READINESS_RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "hexversion": sys.hexversion,
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
            "executable_sha256": executable_sha256,
            "prefix": runtime_paths.prefix.as_posix(),
            "exec_prefix": runtime_paths.exec_prefix.as_posix(),
            "purelib": runtime_paths.purelib.as_posix(),
            "platlib": runtime_paths.platlib.as_posix(),
            "stdlib": runtime_paths.stdlib.as_posix(),
            "no_site_stdlib_search_paths": [
                path.as_posix() for path in runtime_paths.no_site_stdlib_search_paths
            ],
            "stdlib_file_scope": _STDLIB_FILE_SCOPE,
            "stdlib_file_count": stdlib_inventory.file_count,
            "stdlib_directory_count": stdlib_inventory.directory_count,
            "stdlib_file_total_bytes": stdlib_inventory.total_bytes,
            "stdlib_file_inventory_sha256": stdlib_inventory.inventory_sha256,
        },
        "platform": _platform_runtime_identity(),
        "dependencies": {
            "key_versions": key_versions,
            "installed_distribution_count": len(distribution_inventory.versions),
            "installed_distribution_file_scope": _INSTALLED_DISTRIBUTION_FILE_SCOPE,
            "installed_distribution_file_count": distribution_inventory.file_count,
            "installed_distribution_file_total_bytes": distribution_inventory.total_bytes,
            "installed_distribution_file_inventory_sha256": (
                distribution_inventory.inventory_sha256
            ),
            "dependency_import_tree_file_scope": _DEPENDENCY_IMPORT_TREE_FILE_SCOPE,
            "dependency_import_tree_root_count": dependency_tree_inventory.root_count,
            "dependency_import_tree_file_count": dependency_tree_inventory.file_count,
            "dependency_import_tree_directory_count": (
                dependency_tree_inventory.directory_count
            ),
            "dependency_import_tree_file_total_bytes": dependency_tree_inventory.total_bytes,
            "dependency_import_tree_file_inventory_sha256": (
                dependency_tree_inventory.inventory_sha256
            ),
        },
        "jax": {
            "default_backend": str(jax.default_backend()),
            "enable_x64": bool(jax.config.jax_enable_x64),
            "config_sha256": canonical_sha256(config_values),
            "devices": devices,
        },
        "environment": environment,
        "child_environment": _build_child_environment(),
        "library_environment_side_effects": library_environment_side_effects,
    }


def _validated_library_environment_side_effects() -> dict[str, str]:
    environment_names = sorted(
        name
        for name in os.environ
        if name in _ENVIRONMENT_NAMES or name.startswith(_ENVIRONMENT_PREFIXES)
    )
    parent_hash_seed = os.environ.get("PYTHONHASHSEED")
    _require(
        parent_hash_seed in {None, "0"},
        "readiness runtime PYTHONHASHSEED must be absent or exactly zero",
    )
    library_environment_side_effects = {
        name: os.environ[name]
        for name in environment_names
        if name in _BOUND_LIBRARY_ENVIRONMENT_SIDE_EFFECTS
    }
    _require(
        library_environment_side_effects == _BOUND_LIBRARY_ENVIRONMENT_SIDE_EFFECTS
        and set(environment_names)
        == set(_BOUND_LIBRARY_ENVIRONMENT_SIDE_EFFECTS)
        | ({"PYTHONHASHSEED"} if parent_hash_seed is not None else set()),
        "readiness runtime environment differs from clean-start bound library side effects: "
        + ",".join(environment_names),
    )
    return library_environment_side_effects


def _platform_runtime_identity() -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version_sha256": _sha256_bytes(platform.version().encode("utf-8")),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "cpu_count": os.cpu_count(),
    }


def _jax_device_identity(jax_module: object) -> list[dict[str, object]]:
    devices: list[dict[str, object]] = []
    for device in cast(Any, jax_module).devices():
        devices.append(
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "local_hardware_id": int(getattr(device, "local_hardware_id", device.id)),
            }
        )
    return devices


def build_runtime_execution_identity() -> dict[str, object]:
    """Build the cheap per-process identity; full file bytes are batch-bracketed."""

    import jax

    runtime_paths = _runtime_path_binding()
    executable = Path(sys.executable).resolve(strict=True)
    _size, executable_sha256, _raw, _identity = _stable_hash_record_file(executable)
    config_values = {
        name: _runtime_value(value) for name, value in sorted(jax.config.values.items())
    }
    return {
        "schema": READINESS_RUNTIME_EXECUTION_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "hexversion": sys.hexversion,
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
            "executable_sha256": executable_sha256,
            "prefix": runtime_paths.prefix.as_posix(),
            "exec_prefix": runtime_paths.exec_prefix.as_posix(),
            "purelib": runtime_paths.purelib.as_posix(),
            "platlib": runtime_paths.platlib.as_posix(),
            "stdlib": runtime_paths.stdlib.as_posix(),
            "no_site_stdlib_search_paths": [
                path.as_posix() for path in runtime_paths.no_site_stdlib_search_paths
            ],
        },
        "platform": _platform_runtime_identity(),
        "dependencies": {
            "key_versions": {
                name: importlib.metadata.version(name) for name in _KEY_DISTRIBUTIONS
            },
        },
        "jax": {
            "default_backend": str(jax.default_backend()),
            "enable_x64": bool(jax.config.jax_enable_x64),
            "config_sha256": canonical_sha256(config_values),
            "devices": _jax_device_identity(jax),
        },
        "child_environment": _build_child_environment(),
        "library_environment_side_effects": _validated_library_environment_side_effects(),
    }


def _protocol_binding() -> dict[str, object]:
    payload = calibration_design_payload()
    _require(
        _protocol_canonical_sha256(payload) == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "factorial protocol payload differs from its frozen digest",
    )
    manifest_bindings = cast(list[object], payload["manifest_bindings"])
    recurrence_bindings = cast(list[object], payload["recurrence_eligibility_bindings"])
    gate_matrix_sha256 = payload["gate_matrix_sha256"]
    _require(_is_sha256(gate_matrix_sha256), "protocol gate matrix digest is invalid")
    return {
        "receipt_schema": CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "design_schema": DESIGN_SCHEMA,
        "design_envelope_schema": DESIGN_ENVELOPE_SCHEMA,
        "protocol_status": PROTOCOL_STATUS,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "manifest_bindings": manifest_bindings,
        "manifest_bindings_sha256": _protocol_canonical_sha256(manifest_bindings),
        "recurrence_eligibility_sha256": _protocol_canonical_sha256(recurrence_bindings),
        "gate_matrix_sha256": gate_matrix_sha256,
        "development_summary_schema": BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace_schema": BOUND_PRIMITIVE_TRACE_SCHEMA,
        "consumed_calibration_namespace_sha256": _sha256_bytes(
            CONSUMED_CALIBRATION_NAMESPACE.encode("ascii")
        ),
        "matched_case_count": N_MATCHED_CASES,
    }


def _component_schema_binding() -> dict[str, object]:
    return {
        "development_summary": BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace": BOUND_PRIMITIVE_TRACE_SCHEMA,
        "trace_audit_input": HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
        "trace_audit_report": HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
        "checkpoint": HIDDEN_REGIME_CHECKPOINT_SCHEMA,
        "trace_chunk": HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
        "role_lifecycle_oracle": SLOT_ROLE_TRANSITION_ORACLE_SCHEMA,
        "world_oracle": HIDDEN_REGIME_WORLD_ORACLE_SCHEMA,
        "lineage_oracle": HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
        "summary_oracle": HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
        "runtime_execution_identity": READINESS_RUNTIME_EXECUTION_SCHEMA,
        "runtime_reconstruction": READINESS_RUNTIME_RECONSTRUCTION_SCHEMA,
        "execution_genesis_binding": CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
    }


def _protected_guard() -> dict[str, object]:
    entries = tuple(
        entry
        for entry in HIDDEN_REGIME_MANIFEST_USE_LEDGER.values()
        if entry.use_partition == PROTECTED_CANDIDATE_PARTITION
    )
    _require(bool(entries), "protected-candidate ledger is empty")
    _require(
        PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED is False,
        "protected-candidate outcome constant is no longer false",
    )
    _require(
        all(entry.learner_outcomes_executed is False for entry in entries),
        "protected-candidate outcome ledger is no longer uniformly false",
    )
    return {
        "scope": "source_literals_only_not_managed_or_external_execution_history",
        "learner_outcome_constant": False,
        "ledger_all_false": True,
        "ledger_entry_count": len(entries),
        "execution_absence_attested": False,
    }


def _base_body(
    source_manifest: dict[str, object],
    source_archive: bytes,
    runtime_identity: dict[str, object],
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    source_manifest_sha256 = canonical_sha256(source_manifest)
    archive_binding = {
        "schema": READINESS_ARCHIVE_SCHEMA,
        "format": "zip-stored-deterministic-v1",
        "file_name": "source.zip",
        "byte_size": len(source_archive),
        "sha256": _sha256_bytes(source_archive),
        "member_count": len(cast(list[object], source_manifest["files"]))
        + len(cast(list[object], source_manifest["support_files"])),
        "member_timestamp": list(_ZIP_TIMESTAMP),
        "member_mode_octal": "100444",
    }
    genesis = build_calibration_execution_genesis(
        source_archive_sha256=cast(str, archive_binding["sha256"]),
        source_manifest_sha256=source_manifest_sha256,
        runtime_identity_sha256=canonical_sha256(runtime_identity),
    )
    validated_genesis = require_valid_calibration_execution_genesis(genesis)
    governance_binding = calibration_execution_genesis_receipt_binding(validated_genesis)
    _require(
        governance_binding["protocol_payload_sha256"]
        == protocol_binding["protocol_payload_sha256"],
        "execution governance protocol binding differs",
    )
    _require(
        governance_binding["seed_snapshot_sha256"] == protocol_binding["seed_snapshot_sha256"],
        "execution governance seed binding differs",
    )
    runner = source_manifest["calibration_runner_module"]
    return {
        "receipt_schema": CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "envelope_schema": READINESS_ENVELOPE_SCHEMA,
        "status": READINESS_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "protocol_binding": protocol_binding,
        "component_schema_binding": _component_schema_binding(),
        "source_snapshot": {
            "manifest": source_manifest,
            "manifest_sha256": source_manifest_sha256,
            "archive": archive_binding,
        },
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        READINESS_EXECUTION_GOVERNANCE_FIELD: governance_binding,
        "worker_execution": {
            "calibration_runner_module": runner,
            "entrypoint": "main" if runner is not None else None,
            "allowed_entrypoint_modes": list(_ALLOWED_WORKER_ENTRYPOINT_MODES),
            "isolated_flag": None,
            "no_site_flag": "-S",
            "dont_write_bytecode_flag": "-B",
            "safe_path_flag": "-P",
            "pycache_prefix_option": (
                "-X pycache_prefix={fresh_empty_separate_bytecode_cache_root}"
            ),
            "bytecode_cache_policy": _BYTECODE_CACHE_POLICY,
            "runtime_path_policy": _RUNTIME_PATH_POLICY,
            "runtime_mutation_policy": _RUNTIME_MUTATION_POLICY,
            "child_environment_policy": _CHILD_ENVIRONMENT_POLICY,
            "loaded_module_origin_policy": _LOADED_MODULE_ORIGIN_POLICY,
            "working_directory": "fresh_empty_temporary_directory",
            "project_source_path": "content_addressed_source_zip_first_and_sole",
            "project_module_provenance_required": "zipimport_loader_and_file_inside_source_zip",
            "explicit_execution_authorization_required": True,
        },
        "source_literal_outcome_guard": _protected_guard(),
        "claim_scope": (
            "nonpromoting readiness for a finite consumed hidden-regime calibration factorial; "
            "not a calibration outcome, threshold freeze, protected evaluation, general "
            "continual-learning result, or Alberta Plan completion; a managed local execution "
            "ledger cannot prove that equivalent source or seeds were never executed in an "
            "external clone"
        ),
    }


def build_readiness_draft(
    *,
    repository_root: Path = _REPO_ROOT,
) -> ReadinessDraft:
    """Build a non-authorizing draft without running any certification or calibration."""

    root = repository_root.absolute()
    source_manifest, archive = _build_source_bundle(root)
    protocol_binding = _protocol_binding()
    runtime_identity = _build_runtime_identity()
    base_body = _base_body(source_manifest, archive, runtime_identity, protocol_binding)
    seal = _seal("readiness-draft-v1", base_body, root)
    return ReadinessDraft(base_body, archive, root, seal)


def _validate_draft_seal(draft: ReadinessDraft) -> None:
    expected = _seal("readiness-draft-v1", draft.base_body, draft.repository_root)
    _require(hmac.compare_digest(draft.seal, expected), "readiness draft seal is invalid")


_CERTIFICATION_BOOTSTRAP = r"""
import hashlib
import json
import os
import sys
import types

(
    bytecode_cache_root,
    source_root,
    runtime_prefix,
    runtime_exec_prefix,
    purelib,
    platlib,
    stdlib,
    no_site_paths_json,
    expected_environment_json,
    execution_manifest_path,
    runtime_contract_json,
    *pytest_argv,
) = sys.argv[1:]
bytecode_cache_root = os.path.abspath(bytecode_cache_root)
source_root = os.path.abspath(source_root)
runtime_prefix = os.path.abspath(runtime_prefix)
runtime_exec_prefix = os.path.abspath(runtime_exec_prefix)
purelib = os.path.abspath(purelib)
platlib = os.path.abspath(platlib)
stdlib = os.path.abspath(stdlib)
no_site_paths = json.loads(no_site_paths_json)
expected_environment = json.loads(expected_environment_json)
runtime_contract = json.loads(runtime_contract_json)
execution_manifest_path = os.path.abspath(execution_manifest_path)
if sys.flags.no_site != 1:
    raise SystemExit("certification interpreter did not start with -S")
if "_virtualenv" in sys.modules:
    raise SystemExit("certification executed a pre-bootstrap virtualenv pth hook")
if expected_environment != {
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}:
    raise SystemExit("certification expected environment binding differs")
if dict(os.environ) != expected_environment:
    raise SystemExit("certification process environment differs from its exact binding")
if sys.flags.isolated != 0 or not sys.flags.safe_path or sys.flags.hash_randomization != 0:
    raise SystemExit("certification interpreter isolation or hash-seed flags differ")
bound_library_environment_side_effects = {
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TPU_SKIP_MDS_QUERY": "1",
}

def require_bound_post_import_environment(initial):
    actual = dict(os.environ)
    if any(actual.get(name) != value for name, value in initial.items()):
        raise SystemExit("certification startup environment changed after imports")
    additions = {name: value for name, value in actual.items() if name not in initial}
    if any(
        name not in bound_library_environment_side_effects
        or bound_library_environment_side_effects[name] != value
        for name, value in additions.items()
    ):
        raise SystemExit("certification environment gained an unbound value after imports")
    return actual

if not isinstance(no_site_paths, list) or not all(isinstance(path, str) for path in no_site_paths):
    raise SystemExit("certification no-site path binding is invalid")
no_site_paths = [os.path.abspath(path) for path in no_site_paths]
if sys.path != no_site_paths:
    raise SystemExit("certification startup path differs from bound no-site paths")
if len(no_site_paths) != 3 or no_site_paths[1:] != [stdlib, os.path.join(stdlib, "lib-dynload")]:
    raise SystemExit("certification standard-library path binding differs")
runtime_directories = (runtime_prefix, runtime_exec_prefix, purelib, platlib, stdlib)
if not all(os.path.isdir(path) for path in runtime_directories):
    raise SystemExit("certification bound runtime directory is absent")
if not sys.dont_write_bytecode:
    raise SystemExit("certification bytecode writes are not disabled")
if not isinstance(sys.pycache_prefix, str):
    raise SystemExit("certification has no isolated bytecode-cache prefix")
if os.path.abspath(sys.pycache_prefix) != bytecode_cache_root:
    raise SystemExit("certification bytecode-cache prefix differs")
if bytecode_cache_root == source_root:
    raise SystemExit("certification bytecode cache is not separate from source")
if os.path.exists(bytecode_cache_root) and os.listdir(bytecode_cache_root):
    raise SystemExit("certification bytecode-cache prefix is not fresh and empty")
if os.path.abspath(os.getcwd()) != source_root:
    raise SystemExit("certification cwd is not the extracted source root")
if os.path.exists(execution_manifest_path):
    raise SystemExit("certification execution manifest path is not fresh")

site_paths = []
for path in (purelib, platlib):
    if path not in site_paths:
        site_paths.append(path)
sys.prefix = runtime_prefix
sys.exec_prefix = runtime_exec_prefix
sys.path[:] = [source_root, *no_site_paths, *site_paths]
expected_sys_path = list(sys.path)

def path_is_within(path, root):
    if not isinstance(path, str) or not path or path.startswith("<"):
        return False
    absolute = os.path.abspath(path)
    root = os.path.abspath(root)
    try:
        return os.path.commonpath((absolute, root)) == root
    except ValueError:
        return False

if path_is_within(execution_manifest_path, source_root):
    raise SystemExit("certification execution manifest path overlaps source")
if path_is_within(execution_manifest_path, bytecode_cache_root):
    raise SystemExit("certification execution manifest path overlaps bytecode cache")

allowed_roots = [source_root, *site_paths]
python_zip = no_site_paths[0]

def stdlib_origin_is_bound(path):
    return path_is_within(path, stdlib)

def origin_is_bound(path):
    return (
        any(path_is_within(path, root) for root in allowed_roots)
        or stdlib_origin_is_bound(path)
        or (
        isinstance(path, str)
        and os.path.abspath(path).startswith(os.path.abspath(python_zip) + os.sep)
        )
    )

def audit_loaded_module_origins(*, allow_originless):
    for name, loaded in tuple(sys.modules.items()):
        if loaded is None:
            continue
        if not isinstance(loaded, types.ModuleType):
            continue
        spec = getattr(loaded, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        file_origin = getattr(loaded, "__file__", None)
        if spec_origin in ("built-in", "frozen"):
            continue
        if name == "__main__" and spec is None and file_origin is None:
            continue
        origins = []
        for origin in (file_origin, spec_origin):
            if isinstance(origin, str) and origin not in origins:
                origins.append(origin)
        if origins:
            for origin in origins:
                if not origin_is_bound(origin):
                    raise SystemExit("loaded module origin is outside bound roots: " + name)
            continue
        locations = getattr(spec, "submodule_search_locations", None)
        if locations is None:
            if allow_originless:
                continue
            raise SystemExit("loaded module has no auditable origin: " + name)
        location_list = list(locations)
        if not location_list or not all(origin_is_bound(path) for path in location_list):
            raise SystemExit("namespace module location is outside bound roots: " + name)

import pytest
import numpy as np

def canonical_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

class CertificationExecutionPlugin:
    def __init__(self, contract):
        self.contract = contract
        self.collected = []
        self.deselected = []
        self.reports = {}
        self.cut_observation = None
        self.cut_observation_error = None
        self.harness_exception = False

    def pytest_collection_finish(self, session):
        self.collected.extend(item.nodeid for item in session.items)

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report):
        phase_reports = self.reports.setdefault(report.nodeid, {})
        phase_reports.setdefault(report.when, []).append(
            {
                "outcome": report.outcome,
                "was_xfail": getattr(report, "wasxfail", None) is not None,
            }
        )

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_call(self, item):
        outcome = yield
        cut_contract = self.contract.get("checkpoint_cut_contract")
        if not isinstance(cut_contract, dict) or item.nodeid != cut_contract.get("node_id"):
            return
        try:
            fixture_name = cut_contract["fixture_name"]
            tuple_index = cut_contract["fixture_tuple_index"]
            trace_fixture_name = cut_contract["trace_fixture_name"]
            fixture_value = item.funcargs[fixture_name]
            trace_result = item.funcargs[trace_fixture_name]
            if type(tuple_index) is not int or type(fixture_value) is not tuple:
                raise TypeError
            cuts = fixture_value[tuple_index]
            if type(cuts) is not dict:
                raise TypeError
            cut_ids = list(cuts)
            cut_values = [cuts[cut_id] for cut_id in cut_ids]
            if not all(type(cut_id) is str for cut_id in cut_ids):
                raise TypeError
            if not all(type(value) is int for value in cut_values):
                raise TypeError
            trace = trace_result.trace
            semantic_observations = []
            for semantic in cut_contract["cut_semantics"]:
                cut_id = semantic["cut_id"]
                trace_field = semantic["trace_field"]
                predicate = semantic["predicate"]
                occurrence_index = semantic["occurrence_index"]
                index_offset = semantic["index_offset"]
                values = np.asarray(getattr(trace, trace_field))
                if values.ndim != 1 or type(occurrence_index) is not int:
                    raise TypeError
                if type(index_offset) is not int:
                    raise TypeError
                if predicate == "strictly_between_zero_and_three":
                    event_indices = np.flatnonzero(
                        np.logical_and(values > 0, values < 3)
                    ).tolist()
                elif predicate == "truthy":
                    event_indices = np.flatnonzero(values).tolist()
                elif predicate == "adjacent_change_new_index":
                    event_indices = (np.flatnonzero(values[1:] != values[:-1]) + 1).tolist()
                elif predicate == "nonnegative":
                    event_indices = np.flatnonzero(values >= 0).tolist()
                else:
                    raise ValueError
                selected_event_index = int(event_indices[occurrence_index])
                expected_cut_value = selected_event_index + index_offset
                semantic_observations.append(
                    {
                        "cut_id": cut_id,
                        "trace_field": trace_field,
                        "predicate": predicate,
                        "occurrence_index": occurrence_index,
                        "index_offset": index_offset,
                        "matching_event_count": len(event_indices),
                        "selected_event_index": selected_event_index,
                        "expected_cut_value": expected_cut_value,
                        "observed_cut_value": cuts[cut_id],
                    }
                )
            self.cut_observation = {
                "node_id": item.nodeid,
                "fixture_name": fixture_name,
                "trace_fixture_name": trace_fixture_name,
                "cut_ids": cut_ids,
                "cut_values": cut_values,
                "semantic_observations": semantic_observations,
            }
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            self.cut_observation_error = "checkpoint_cut_fixture_shape_invalid"
        if outcome.excinfo is not None and self.cut_observation_error is None:
            self.cut_observation_error = "checkpoint_cut_test_call_failed"

    def build_manifest(self, pytest_exit_code):
        violations = []
        collected = sorted(self.collected)
        deselected = sorted(self.deselected)
        if not collected:
            violations.append("zero_collection")
        if len(collected) != len(set(collected)):
            violations.append("duplicate_collected_node_id")
        if deselected:
            violations.append("deselection_observed")
        if self.harness_exception:
            violations.append("pytest_harness_exception")
        if pytest_exit_code != 0:
            violations.append("pytest_exit_code_nonzero")

        per_node = []
        collected_set = set(collected)
        if set(self.reports) != collected_set:
            violations.append("report_node_inventory_differs")
        expected_phases = ("setup", "call", "teardown")
        for node_id in sorted(collected_set):
            phase_reports = self.reports.get(node_id, {})
            if set(phase_reports) != set(expected_phases):
                violations.append("phase_inventory_differs:" + node_id)
            phase_payload = {"node_id": node_id}
            for phase in expected_phases:
                entries = phase_reports.get(phase, [])
                entry = entries[0] if len(entries) == 1 else None
                phase_payload[phase] = entry
                if len(entries) != 1:
                    violations.append("phase_cardinality_differs:" + node_id + ":" + phase)
                else:
                    if entry["outcome"] != "passed":
                        violations.append("phase_not_passed:" + node_id + ":" + phase)
                    if entry["was_xfail"]:
                        violations.append("xfail_or_xpass_observed:" + node_id + ":" + phase)
            per_node.append(phase_payload)

        cut_contract = self.contract.get("checkpoint_cut_contract")
        if cut_contract is None:
            if self.cut_observation is not None or self.cut_observation_error is not None:
                violations.append("unexpected_checkpoint_cut_observation")
        elif isinstance(cut_contract, dict):
            if self.cut_observation_error is not None:
                violations.append(self.cut_observation_error)
            observation = self.cut_observation
            if observation is None:
                violations.append("checkpoint_cut_observation_missing")
            else:
                if observation["node_id"] != cut_contract.get("node_id"):
                    violations.append("checkpoint_cut_node_differs")
                if observation["fixture_name"] != cut_contract.get("fixture_name"):
                    violations.append("checkpoint_cut_fixture_differs")
                if observation["trace_fixture_name"] != cut_contract.get(
                    "trace_fixture_name"
                ):
                    violations.append("checkpoint_cut_trace_fixture_differs")
                expected_cut_ids = cut_contract.get("expected_cut_ids")
                if observation["cut_ids"] != expected_cut_ids:
                    violations.append("checkpoint_cut_ids_differ")
                cut_values = observation["cut_values"]
                if (
                    not cut_values
                    or not all(type(value) is int and value > 0 for value in cut_values)
                    or len(cut_values) != len(set(cut_values))
                ):
                    violations.append("checkpoint_cut_values_invalid")
                expected_semantics = cut_contract.get("cut_semantics")
                semantic_observations = observation["semantic_observations"]
                if (
                    not isinstance(expected_semantics, list)
                    or len(semantic_observations) != len(expected_semantics)
                ):
                    violations.append("checkpoint_cut_semantic_count_differs")
                else:
                    for semantic, observed, cut_value in zip(
                        expected_semantics,
                        semantic_observations,
                        cut_values,
                    ):
                        contract_fields = (
                            "cut_id",
                            "trace_field",
                            "predicate",
                            "occurrence_index",
                            "index_offset",
                        )
                        if any(observed[field] != semantic[field] for field in contract_fields):
                            violations.append("checkpoint_cut_semantic_contract_differs")
                            continue
                        if observed["matching_event_count"] <= semantic["occurrence_index"]:
                            violations.append("checkpoint_cut_event_occurrence_missing")
                        if observed["selected_event_index"] < 0:
                            violations.append("checkpoint_cut_event_index_invalid")
                        if observed["expected_cut_value"] != (
                            observed["selected_event_index"] + semantic["index_offset"]
                        ):
                            violations.append("checkpoint_cut_offset_relation_differs")
                        if (
                            observed["observed_cut_value"]
                            != observed["expected_cut_value"]
                            or cut_value != observed["expected_cut_value"]
                        ):
                            violations.append("checkpoint_cut_trace_semantics_differ")
        else:
            violations.append("checkpoint_cut_contract_invalid")

        violations = sorted(set(violations))
        return {
            "schema": "alberta.hidden-regime-calibration.certification-execution-manifest.v1",
            "certification_id": self.contract.get("certification_id"),
            "runtime_contract_sha256": hashlib.sha256(
                canonical_bytes(self.contract)
            ).hexdigest(),
            "status": "passed" if not violations else "rejected",
            "pytest_exit_code": pytest_exit_code,
            "collected_node_count": len(collected),
            "collected_node_ids": collected,
            "deselected_node_ids": deselected,
            "per_node": per_node,
            "checkpoint_cut_observation": self.cut_observation,
            "violations": violations,
        }

plugin = CertificationExecutionPlugin(runtime_contract)
try:
    exit_code = int(pytest.main(pytest_argv, plugins=[plugin]))
except BaseException:
    plugin.harness_exception = True
    exit_code = 3
execution_manifest = plugin.build_manifest(exit_code)
if execution_manifest["status"] != "passed":
    exit_code = 1
execution_manifest_raw = canonical_bytes(execution_manifest)
manifest_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    manifest_flags |= os.O_NOFOLLOW
manifest_fd = os.open(execution_manifest_path, manifest_flags, 0o600)
try:
    offset = 0
    while offset < len(execution_manifest_raw):
        offset += os.write(manifest_fd, execution_manifest_raw[offset:])
    os.fsync(manifest_fd)
finally:
    os.close(manifest_fd)
prefix = source_root + os.sep
for name, loaded in tuple(sys.modules.items()):
    if name != "alberta_framework" and not name.startswith("alberta_framework."):
        continue
    origin = getattr(loaded, "__file__", None)
    spec_origin = getattr(getattr(loaded, "__spec__", None), "origin", None)
    if not isinstance(origin, str) or not os.path.abspath(origin).startswith(prefix):
        raise SystemExit("certified project module origin is outside snapshot: " + name)
    if not isinstance(spec_origin, str) or not os.path.abspath(spec_origin).startswith(prefix):
        raise SystemExit("certified project module spec is outside snapshot: " + name)
audit_loaded_module_origins(allow_originless=True)
if sys.flags.no_site != 1 or "_virtualenv" in sys.modules:
    raise SystemExit("certification no-site boundary changed")
if sys.prefix != runtime_prefix or sys.exec_prefix != runtime_exec_prefix:
    raise SystemExit("certification runtime prefix changed")
if sys.path != expected_sys_path:
    raise SystemExit("certification exact import path changed")
require_bound_post_import_environment(expected_environment)
if os.path.exists(bytecode_cache_root) and os.listdir(bytecode_cache_root):
    raise SystemExit("certification wrote into its bytecode-cache prefix")
raise SystemExit(exit_code)
"""
_CERTIFICATION_BOOTSTRAP_SHA256 = _sha256_bytes(_CERTIFICATION_BOOTSTRAP.encode("utf-8"))

_RUNTIME_RECONSTRUCTION_BOOTSTRAP = r"""
import importlib
import json
import os
import sys

(
    bytecode_cache_root,
    source_root,
    runtime_prefix,
    runtime_exec_prefix,
    purelib,
    platlib,
    stdlib,
    no_site_paths_json,
    expected_environment_json,
) = sys.argv[1:]
bytecode_cache_root = os.path.abspath(bytecode_cache_root)
source_root = os.path.abspath(source_root)
runtime_prefix = os.path.abspath(runtime_prefix)
runtime_exec_prefix = os.path.abspath(runtime_exec_prefix)
purelib = os.path.abspath(purelib)
platlib = os.path.abspath(platlib)
stdlib = os.path.abspath(stdlib)
no_site_paths = json.loads(no_site_paths_json)
expected_environment = json.loads(expected_environment_json)
if expected_environment != {"LC_ALL": "C", "PYTHONHASHSEED": "0"}:
    raise SystemExit("runtime reconstruction expected environment differs")
if dict(os.environ) != expected_environment:
    raise SystemExit("runtime reconstruction process environment differs")
if (
    sys.flags.no_site != 1
    or sys.flags.isolated != 0
    or not sys.flags.safe_path
    or sys.flags.hash_randomization != 0
):
    raise SystemExit("runtime reconstruction interpreter flags differ")
if not sys.dont_write_bytecode or not isinstance(sys.pycache_prefix, str):
    raise SystemExit("runtime reconstruction bytecode policy differs")
if os.path.abspath(sys.pycache_prefix) != bytecode_cache_root:
    raise SystemExit("runtime reconstruction bytecode-cache prefix differs")
if os.path.exists(bytecode_cache_root) and os.listdir(bytecode_cache_root):
    raise SystemExit("runtime reconstruction bytecode-cache prefix is not empty")
if os.path.abspath(os.getcwd()) != source_root:
    raise SystemExit("runtime reconstruction cwd differs")
if not isinstance(no_site_paths, list) or not all(isinstance(path, str) for path in no_site_paths):
    raise SystemExit("runtime reconstruction no-site paths differ")
no_site_paths = [os.path.abspath(path) for path in no_site_paths]
if sys.path != no_site_paths:
    raise SystemExit("runtime reconstruction startup path differs")
if len(no_site_paths) != 3 or no_site_paths[1:] != [stdlib, os.path.join(stdlib, "lib-dynload")]:
    raise SystemExit("runtime reconstruction standard-library path differs")
site_paths = []
for path in (purelib, platlib):
    if path not in site_paths:
        site_paths.append(path)
if not all(
    os.path.isdir(path)
    for path in (runtime_prefix, runtime_exec_prefix, *site_paths, stdlib)
):
    raise SystemExit("runtime reconstruction bound directory is absent")
sys.prefix = runtime_prefix
sys.exec_prefix = runtime_exec_prefix
sys.path[:] = [source_root, *no_site_paths, *site_paths]
expected_sys_path = list(sys.path)
module = importlib.import_module(
    "alberta_framework.evaluation.hidden_regime_calibration_readiness"
)
origin = getattr(module, "__file__", None)
if not isinstance(origin, str) or not os.path.abspath(origin).startswith(source_root + os.sep):
    raise SystemExit("runtime reconstruction readiness module is outside snapshot")
runtime = module._build_runtime_identity()
if dict(os.environ) != {
    **expected_environment,
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TPU_SKIP_MDS_QUERY": "1",
}:
    raise SystemExit("runtime reconstruction post-import environment differs")
if (
    sys.path != expected_sys_path
    or sys.prefix != runtime_prefix
    or sys.exec_prefix != runtime_exec_prefix
):
    raise SystemExit("runtime reconstruction path binding changed")
if os.path.exists(bytecode_cache_root) and os.listdir(bytecode_cache_root):
    raise SystemExit("runtime reconstruction wrote into its bytecode-cache prefix")
sys.stdout.buffer.write(module.canonical_json_bytes(runtime))
sys.stdout.buffer.flush()
"""
_RUNTIME_RECONSTRUCTION_BOOTSTRAP_SHA256 = _sha256_bytes(
    _RUNTIME_RECONSTRUCTION_BOOTSTRAP.encode("utf-8")
)
_RUNTIME_RECONSTRUCTION_SEMANTIC_COMMAND = (
    "{runtime_python}",
    "-S",
    "-B",
    "-P",
    "-X",
    "pycache_prefix={fresh_empty_separate_bytecode_cache_root}",
    "-c",
    "{runtime_reconstruction_harness_v1}",
    "{fresh_empty_separate_bytecode_cache_root}",
    "{verified_extracted_source_root}",
    "{bound_runtime_prefix}",
    "{bound_runtime_exec_prefix}",
    "{bound_runtime_purelib}",
    "{bound_runtime_platlib}",
    "{bound_runtime_stdlib}",
    "{bound_no_site_stdlib_search_paths_json}",
    "{receipt_derived_child_environment_json}",
)


def _spec_payload() -> list[dict[str, object]]:
    return [
        {
            "certification_id": spec.certification_id,
            "node_ids": list(spec.node_ids),
            "node_manifest": spec.node_manifest,
            "node_manifest_sha256": spec.node_manifest_sha256,
            "command": list(spec.semantic_command),
            "harness_sha256": _CERTIFICATION_BOOTSTRAP_SHA256,
            "environment_policy": _CERTIFICATION_ENVIRONMENT_POLICY,
        }
        for spec in CERTIFICATION_SPECS
    ]


def _certification_selector_matches_node(selector: str, node_id: str) -> bool:
    if "::" not in selector:
        return node_id.startswith(selector + "::")
    return node_id == selector or node_id.startswith(selector + "[")


def _validate_certification_execution_manifest(
    manifest_raw: object,
    spec: CertificationSpec,
) -> dict[str, object]:
    manifest = _expect_dict(manifest_raw, "certification execution manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "certification_id",
            "runtime_contract_sha256",
            "status",
            "pytest_exit_code",
            "collected_node_count",
            "collected_node_ids",
            "deselected_node_ids",
            "per_node",
            "checkpoint_cut_observation",
            "violations",
        },
        "certification execution manifest",
    )
    _require(
        manifest["schema"] == READINESS_CERTIFICATION_EXECUTION_MANIFEST_SCHEMA,
        "certification execution manifest schema differs",
    )
    _require(
        manifest["certification_id"] == spec.certification_id,
        "certification execution manifest identifier differs",
    )
    _require(
        manifest["runtime_contract_sha256"] == spec.runtime_contract_sha256,
        "certification execution runtime contract digest differs",
    )
    _require(
        manifest["status"] == "passed",
        "certification execution manifest is not passed",
    )
    _require(
        _is_strict_int(manifest["pytest_exit_code"])
        and manifest["pytest_exit_code"] == 0,
        "certification pytest exit code is not strict integer zero",
    )
    collected = _expect_list(manifest["collected_node_ids"], "collected certification nodes")
    _require(bool(collected), "certification collected zero test nodes")
    _require(
        all(type(node_id) is str and bool(node_id) for node_id in collected),
        "collected certification node ID is invalid",
    )
    _require(
        collected == sorted(cast(list[str], collected))
        and len(collected) == len(set(cast(list[str], collected))),
        "collected certification node inventory is not sorted and unique",
    )
    _require(
        _is_strict_int(manifest["collected_node_count"])
        and manifest["collected_node_count"] == len(collected),
        "collected certification node count differs",
    )
    _require(
        all(
            any(
                _certification_selector_matches_node(selector, cast(str, node_id))
                for selector in spec.node_ids
            )
            for node_id in collected
        ),
        "collected certification node is outside the frozen selectors",
    )
    _require(
        all(
            any(
                _certification_selector_matches_node(selector, cast(str, node_id))
                for node_id in collected
            )
            for selector in spec.node_ids
        ),
        "frozen certification selector collected no test node",
    )
    _require(
        _expect_list(manifest["deselected_node_ids"], "deselected certification nodes") == [],
        "certification deselected test nodes",
    )
    _require(
        _expect_list(manifest["violations"], "certification execution violations") == [],
        "certification execution manifest contains violations",
    )

    per_node = _expect_list(manifest["per_node"], "per-node certification outcomes")
    _require(len(per_node) == len(collected), "per-node certification outcome count differs")
    for index, raw in enumerate(per_node):
        item = _expect_dict(raw, f"per-node certification outcome {index}")
        _expect_exact_keys(item, {"node_id", "setup", "call", "teardown"}, "per-node outcome")
        _require(item["node_id"] == collected[index], "per-node certification order differs")
        for phase in ("setup", "call", "teardown"):
            phase_outcome = _expect_dict(item[phase], f"certification {phase} outcome")
            _expect_exact_keys(
                phase_outcome,
                {"outcome", "was_xfail"},
                f"certification {phase} outcome",
            )
            _require(
                phase_outcome["outcome"] == "passed",
                f"certification {phase} phase is not passed",
            )
            _require(
                phase_outcome["was_xfail"] is False,
                f"certification {phase} phase observed xfail or xpass",
            )

    cut_contract = spec.runtime_contract["checkpoint_cut_contract"]
    cut_observation_raw = manifest["checkpoint_cut_observation"]
    if cut_contract is None:
        _require(
            cut_observation_raw is None,
            "unexpected checkpoint-cut runtime observation",
        )
    else:
        contract = _expect_dict(cut_contract, "checkpoint-cut runtime contract")
        observation = _expect_dict(
            cut_observation_raw,
            "checkpoint-cut runtime observation",
        )
        _expect_exact_keys(
            observation,
            {
                "node_id",
                "fixture_name",
                "trace_fixture_name",
                "cut_ids",
                "cut_values",
                "semantic_observations",
            },
            "checkpoint-cut runtime observation",
        )
        _require(
            observation["node_id"] == contract["node_id"]
            and observation["node_id"] in collected,
            "checkpoint-cut runtime node differs",
        )
        _require(
            observation["fixture_name"] == contract["fixture_name"],
            "checkpoint-cut runtime fixture differs",
        )
        _require(
            observation["trace_fixture_name"] == contract["trace_fixture_name"],
            "checkpoint-cut trace fixture differs",
        )
        cut_ids = _expect_list(observation["cut_ids"], "observed checkpoint-cut IDs")
        _require(
            cut_ids == contract["expected_cut_ids"],
            "observed checkpoint-cut IDs differ from the exact runtime contract",
        )
        cut_values = _expect_list(observation["cut_values"], "observed checkpoint-cut values")
        _require(
            len(cut_values) == len(cut_ids)
            and all(_is_strict_int(value) and cast(int, value) > 0 for value in cut_values)
            and len(cut_values) == len(set(cast(list[int], cut_values))),
            "observed checkpoint-cut values violate the runtime contract",
        )
        expected_semantics = _expect_list(
            contract["cut_semantics"],
            "checkpoint-cut semantic contracts",
        )
        semantic_observations = _expect_list(
            observation["semantic_observations"],
            "checkpoint-cut semantic observations",
        )
        _require(
            len(semantic_observations) == len(expected_semantics) == len(cut_values),
            "checkpoint-cut semantic observation count differs",
        )
        contract_fields = {
            "cut_id",
            "trace_field",
            "predicate",
            "occurrence_index",
            "index_offset",
        }
        for index, (semantic_raw, observed_raw) in enumerate(
            zip(expected_semantics, semantic_observations, strict=True)
        ):
            semantic = _expect_dict(semantic_raw, f"checkpoint-cut semantic {index}")
            _expect_exact_keys(semantic, contract_fields, "checkpoint-cut semantic")
            observed = _expect_dict(
                observed_raw,
                f"checkpoint-cut semantic observation {index}",
            )
            _expect_exact_keys(
                observed,
                {
                    *contract_fields,
                    "matching_event_count",
                    "selected_event_index",
                    "expected_cut_value",
                    "observed_cut_value",
                },
                "checkpoint-cut semantic observation",
            )
            _require(
                all(observed[field] == semantic[field] for field in contract_fields),
                "checkpoint-cut semantic observation contract differs",
            )
            occurrence_index = semantic["occurrence_index"]
            index_offset = semantic["index_offset"]
            matching_count = observed["matching_event_count"]
            selected_index = observed["selected_event_index"]
            expected_value = observed["expected_cut_value"]
            observed_value = observed["observed_cut_value"]
            _require(
                _is_strict_int(occurrence_index)
                and cast(int, occurrence_index) >= 0
                and _is_strict_int(index_offset)
                and cast(int, index_offset) >= 0
                and _is_strict_int(matching_count)
                and cast(int, matching_count) > cast(int, occurrence_index)
                and _is_strict_int(selected_index)
                and cast(int, selected_index) >= 0,
                "checkpoint-cut event selection is invalid",
            )
            _require(
                _is_strict_int(expected_value)
                and expected_value == cast(int, selected_index) + cast(int, index_offset)
                and _is_strict_int(observed_value)
                and observed_value == expected_value == cut_values[index],
                "checkpoint-cut value is not the independently reconstructed trace event",
            )
    return manifest


def _load_certification_execution_manifest(
    path: Path,
    spec: CertificationSpec,
) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReadinessError("certification execution manifest is absent") from exc
    _require(stat.S_ISREG(metadata.st_mode), "certification execution manifest is not regular")
    _require(
        metadata.st_size <= _MAX_RECEIPT_BYTES,
        "certification execution manifest is too large",
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReadinessError("cannot read certification execution manifest") from exc
    _require(len(raw) == metadata.st_size, "certification execution manifest size changed")
    return _validate_certification_execution_manifest(_strict_json_loads(raw), spec)


def _extract_verified_source_archive(draft: ReadinessDraft, destination: Path) -> None:
    source = _expect_dict(draft.base_body["source_snapshot"], "source snapshot")
    manifest = _validate_source_manifest_shape(source["manifest"])
    _validate_archive(draft.source_archive, manifest, source["archive"])
    _require(not destination.exists(), "source extraction destination already exists")
    destination.mkdir(mode=0o700)
    directories = {destination}
    with zipfile.ZipFile(io.BytesIO(draft.source_archive), "r") as zf:
        for info in zf.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            directories.update((target.parent, *target.parents[:-1]))
            with target.open("xb") as handle:
                handle.write(zf.read(info))
            target.chmod(0o444)
    for directory in sorted(
        (item for item in directories if item.is_relative_to(destination)),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    _verify_extracted_source_tree(destination, manifest)


def _verify_extracted_source_tree(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    entries = [
        *_expect_list(manifest["files"], "source files"),
        *_expect_list(manifest["support_files"], "support files"),
    ]
    expected = {
        cast(str, _expect_dict(raw, "source entry")["locator"]): _expect_dict(raw, "source entry")
        for raw in entries
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(actual_files == set(expected), "extracted certification source members differ")
    for locator, entry in expected.items():
        path = root / locator
        metadata = path.lstat()
        _require(stat.S_ISREG(metadata.st_mode), f"extracted source is not regular: {locator}")
        _require(
            stat.S_IMODE(metadata.st_mode) == 0o444, f"extracted source mode differs: {locator}"
        )
        raw = path.read_bytes()
        _require(len(raw) == entry["byte_size"], f"extracted source size differs: {locator}")
        _require(
            _sha256_bytes(raw) == entry["sha256"], f"extracted source digest differs: {locator}"
        )


def _make_extracted_tree_removable(root: Path) -> None:
    if not root.exists():
        return
    for directory, subdirectories, _files in os.walk(root, topdown=False):
        for subdirectory in subdirectories:
            Path(directory, subdirectory).chmod(0o700)
        Path(directory).chmod(0o700)


def _run_clean_runtime_reconstruction(
    *,
    extracted_root: Path,
    bytecode_cache_root: Path,
    runtime: Mapping[str, object],
    runtime_prefix: str,
    runtime_exec_prefix: str,
    runtime_purelib: str,
    runtime_platlib: str,
    runtime_stdlib: str,
    no_site_paths_json: str,
    child_environment: Mapping[str, str],
    timeout_seconds: int,
    source_manifest_sha256: str,
    protocol_payload_sha256: str,
) -> dict[str, object]:
    """Reconstruct the exact runtime once in a clean snapshot-importing child."""

    _require(
        not bytecode_cache_root.exists(),
        "runtime reconstruction bytecode-cache prefix is not fresh",
    )
    environment = dict(child_environment)
    environment_json = canonical_json_bytes(environment).decode("ascii")
    command = (
        sys.executable,
        "-S",
        "-B",
        "-P",
        "-X",
        f"pycache_prefix={bytecode_cache_root.as_posix()}",
        "-c",
        _RUNTIME_RECONSTRUCTION_BOOTSTRAP,
        bytecode_cache_root.as_posix(),
        extracted_root.as_posix(),
        runtime_prefix,
        runtime_exec_prefix,
        runtime_purelib,
        runtime_platlib,
        runtime_stdlib,
        no_site_paths_json,
        environment_json,
    )
    completed = subprocess.run(
        command,
        cwd=extracted_root,
        env=environment,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    _require(
        not bytecode_cache_root.exists() or not any(bytecode_cache_root.iterdir()),
        "runtime reconstruction populated its isolated bytecode-cache prefix",
    )
    stdout = bytes(completed.stdout)
    stderr = bytes(completed.stderr)
    record: dict[str, object] = {
        "schema": READINESS_RUNTIME_RECONSTRUCTION_SCHEMA,
        "command": list(_RUNTIME_RECONSTRUCTION_SEMANTIC_COMMAND),
        "harness_sha256": _RUNTIME_RECONSTRUCTION_BOOTSTRAP_SHA256,
        "environment_policy": _CHILD_ENVIRONMENT_POLICY,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": int(completed.returncode),
        "stdout": {"byte_size": len(stdout), "sha256": _sha256_bytes(stdout)},
        "stderr": {"byte_size": len(stderr), "sha256": _sha256_bytes(stderr)},
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "protocol_payload_sha256": protocol_payload_sha256,
    }
    if completed.returncode != 0:
        _fail(
            "clean-child runtime reconstruction failed; "
            f"stdout={record['stdout']!r}; stderr={record['stderr']!r}"
        )
    reconstructed = _validate_runtime_shape(_strict_json_loads(stdout))
    _require(
        reconstructed == runtime,
        "clean-child runtime identity differs from readiness runtime identity",
    )
    _require(
        stdout == canonical_json_bytes(reconstructed),
        "clean-child runtime identity output is not canonical",
    )
    return record


def run_readiness_certifications(
    draft: ReadinessDraft,
    *,
    authorize_certification_execution: bool,
    timeout_seconds_per_group: int = 1800,
) -> VerifiedCertificationBundle:
    """Run only the frozen certification node IDs and derive sealed records.

    This does not run a calibration runner. The exact source, runtime, protocol,
    and uniformly-false protected ledger are checked around the complete batch;
    the immutable extracted source is additionally checked after every group.
    """

    _require(
        authorize_certification_execution is True,
        "certification execution requires explicit authorization",
    )
    _require(
        _is_strict_int(timeout_seconds_per_group) and timeout_seconds_per_group > 0,
        "certification timeout must be a positive strict integer",
    )
    _validate_draft_seal(draft)
    source_snapshot = _expect_dict(draft.base_body["source_snapshot"], "source_snapshot")
    source_digest = cast(str, source_snapshot["manifest_sha256"])
    runtime_digest = cast(str, draft.base_body["runtime_identity_sha256"])
    runtime = _expect_dict(draft.base_body["runtime_identity"], "runtime identity")
    runtime_python = _expect_dict(runtime["python"], "runtime Python")
    runtime_prefix = cast(str, runtime_python["prefix"])
    runtime_exec_prefix = cast(str, runtime_python["exec_prefix"])
    runtime_purelib = cast(str, runtime_python["purelib"])
    runtime_platlib = cast(str, runtime_python["platlib"])
    runtime_stdlib = cast(str, runtime_python["stdlib"])
    child_environment = _validated_child_environment(runtime["child_environment"])
    certification_environment = {
        **child_environment,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    no_site_paths_json = canonical_json_bytes(
        runtime_python["no_site_stdlib_search_paths"]
    ).decode("ascii")
    certification_environment_json = canonical_json_bytes(
        certification_environment
    ).decode("ascii")
    protocol = _expect_dict(draft.base_body["protocol_binding"], "protocol_binding")
    protocol_digest = cast(str, protocol["protocol_payload_sha256"])
    current_manifest, current_archive = _build_source_bundle(draft.repository_root)
    _require(
        canonical_sha256(current_manifest) == source_digest,
        "source drift before certification",
    )
    _require(current_archive == draft.source_archive, "source archive drift before certification")
    _require(
        canonical_sha256(_build_runtime_identity()) == runtime_digest,
        "runtime drift before certification",
    )
    _require(_protocol_binding() == protocol, "protocol drift before certification")
    guard_before = _protected_guard()

    records: list[dict[str, object]] = []
    runtime_reconstruction_record: dict[str, object]
    with tempfile.TemporaryDirectory(prefix="alberta-readiness-certification-") as temporary:
        extracted_root = Path(temporary) / "source"
        _extract_verified_source_archive(draft, extracted_root)
        try:
            runtime_reconstruction_record = _run_clean_runtime_reconstruction(
                extracted_root=extracted_root,
                bytecode_cache_root=Path(temporary) / "runtime-bytecode-cache",
                runtime=runtime,
                runtime_prefix=runtime_prefix,
                runtime_exec_prefix=runtime_exec_prefix,
                runtime_purelib=runtime_purelib,
                runtime_platlib=runtime_platlib,
                runtime_stdlib=runtime_stdlib,
                no_site_paths_json=no_site_paths_json,
                child_environment=child_environment,
                timeout_seconds=timeout_seconds_per_group,
                source_manifest_sha256=source_digest,
                protocol_payload_sha256=protocol_digest,
            )
            _verify_extracted_source_tree(extracted_root, current_manifest)
            for certification_index, spec in enumerate(CERTIFICATION_SPECS):
                bytecode_cache_root = Path(temporary) / "bytecode-cache"
                execution_manifest_path = (
                    Path(temporary)
                    / f"certification-execution-manifest-{certification_index}.json"
                )
                _require(
                    not bytecode_cache_root.exists(),
                    "certification bytecode-cache prefix is not fresh",
                )
                _require(
                    not execution_manifest_path.exists(),
                    "certification execution manifest path is not fresh",
                )
                actual_command = (
                    sys.executable,
                    "-S",
                    "-B",
                    "-P",
                    "-X",
                    f"pycache_prefix={bytecode_cache_root.as_posix()}",
                    "-c",
                    _CERTIFICATION_BOOTSTRAP,
                    bytecode_cache_root.as_posix(),
                    extracted_root.as_posix(),
                    runtime_prefix,
                    runtime_exec_prefix,
                    runtime_purelib,
                    runtime_platlib,
                    runtime_stdlib,
                    no_site_paths_json,
                    certification_environment_json,
                    execution_manifest_path.as_posix(),
                    canonical_json_bytes(spec.runtime_contract).decode("ascii"),
                    *spec.node_ids,
                    "-q",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                )
                completed = subprocess.run(
                    actual_command,
                    cwd=extracted_root,
                    env=certification_environment,
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds_per_group,
                )
                _require(
                    not bytecode_cache_root.exists()
                    or not any(bytecode_cache_root.iterdir()),
                    "certification populated its isolated bytecode-cache prefix",
                )
                _verify_extracted_source_tree(extracted_root, current_manifest)
                execution_manifest = _load_certification_execution_manifest(
                    execution_manifest_path,
                    spec,
                )
                stdout = bytes(completed.stdout)
                stderr = bytes(completed.stderr)
                status = "passed" if completed.returncode == 0 else "failed"
                record: dict[str, object] = {
                    "schema": READINESS_CERTIFICATION_SCHEMA,
                    "certification_id": spec.certification_id,
                    "node_ids": list(spec.node_ids),
                    "node_manifest": spec.node_manifest,
                    "node_manifest_sha256": spec.node_manifest_sha256,
                    "execution_manifest": execution_manifest,
                    "execution_manifest_sha256": canonical_sha256(execution_manifest),
                    "command": list(spec.semantic_command),
                    "harness_sha256": _CERTIFICATION_BOOTSTRAP_SHA256,
                    "environment_policy": _CERTIFICATION_ENVIRONMENT_POLICY,
                    "status": status,
                    "exit_code": int(completed.returncode),
                    "stdout": {"byte_size": len(stdout), "sha256": _sha256_bytes(stdout)},
                    "stderr": {"byte_size": len(stderr), "sha256": _sha256_bytes(stderr)},
                    "source_manifest_sha256": source_digest,
                    "runtime_identity_sha256": runtime_digest,
                    "protocol_payload_sha256": protocol_digest,
                }
                if completed.returncode != 0:
                    _fail(
                        f"readiness certification failed: {spec.certification_id}; "
                        f"stdout={record['stdout']!r}; stderr={record['stderr']!r}"
                    )
                records.append(record)
        finally:
            _make_extracted_tree_removable(extracted_root)

    current_manifest, current_archive = _build_source_bundle(draft.repository_root)
    _require(
        canonical_sha256(current_manifest) == source_digest,
        "source drift after certification",
    )
    _require(current_archive == draft.source_archive, "source archive drift after certification")
    _require(
        canonical_sha256(_build_runtime_identity()) == runtime_digest,
        "runtime drift after certification",
    )
    _require(_protocol_binding() == protocol, "protocol drift after certification")
    _require(_protected_guard() == guard_before, "protected ledger changed during certification")

    record_tuple = tuple(records)
    seal_payload = {
        "records": list(record_tuple),
        "runtime_reconstruction_record": runtime_reconstruction_record,
        "source_manifest_sha256": source_digest,
        "runtime_identity_sha256": runtime_digest,
        "protocol_payload_sha256": protocol_digest,
    }
    seal = _seal("readiness-certifications-v1", seal_payload, draft.repository_root)
    return VerifiedCertificationBundle(
        record_tuple,
        runtime_reconstruction_record,
        source_digest,
        runtime_digest,
        protocol_digest,
        seal,
    )


def _validate_certification_bundle(
    draft: ReadinessDraft,
    bundle: VerifiedCertificationBundle,
) -> None:
    payload = {
        "records": list(bundle.records),
        "runtime_reconstruction_record": bundle.runtime_reconstruction_record,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "runtime_identity_sha256": bundle.runtime_identity_sha256,
        "protocol_payload_sha256": bundle.protocol_payload_sha256,
    }
    expected_seal = _seal("readiness-certifications-v1", payload, draft.repository_root)
    _require(
        hmac.compare_digest(bundle.seal, expected_seal),
        "certification bundle seal is invalid",
    )
    source = _expect_dict(draft.base_body["source_snapshot"], "source_snapshot")
    protocol = _expect_dict(draft.base_body["protocol_binding"], "protocol_binding")
    _require(
        bundle.source_manifest_sha256 == source["manifest_sha256"],
        "certification source drift",
    )
    _require(
        bundle.runtime_identity_sha256 == draft.base_body["runtime_identity_sha256"],
        "certification runtime drift",
    )
    _require(
        bundle.protocol_payload_sha256 == protocol["protocol_payload_sha256"],
        "certification protocol drift",
    )
    _validate_runtime_reconstruction_record(
        bundle.runtime_reconstruction_record,
        draft.base_body,
    )
    _validate_certification_records(list(bundle.records), draft.base_body)


def _validate_runtime_reconstruction_record(
    record_raw: object,
    base_body: Mapping[str, object],
) -> dict[str, object]:
    record = _expect_dict(record_raw, "runtime reconstruction record")
    _expect_exact_keys(
        record,
        {
            "schema",
            "command",
            "harness_sha256",
            "environment_policy",
            "status",
            "exit_code",
            "stdout",
            "stderr",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "protocol_payload_sha256",
        },
        "runtime reconstruction record",
    )
    _require(
        record["schema"] == READINESS_RUNTIME_RECONSTRUCTION_SCHEMA,
        "runtime reconstruction schema differs",
    )
    _require(
        record["command"] == list(_RUNTIME_RECONSTRUCTION_SEMANTIC_COMMAND),
        "runtime reconstruction command differs",
    )
    _require(
        record["harness_sha256"] == _RUNTIME_RECONSTRUCTION_BOOTSTRAP_SHA256,
        "runtime reconstruction harness digest differs",
    )
    _require(
        record["environment_policy"] == _CHILD_ENVIRONMENT_POLICY,
        "runtime reconstruction environment policy differs",
    )
    _require(record["status"] == "passed", "runtime reconstruction did not pass")
    _require(
        _is_strict_int(record["exit_code"]) and record["exit_code"] == 0,
        "runtime reconstruction exit code is not strict integer zero",
    )
    for stream_name in ("stdout", "stderr"):
        stream = _expect_dict(record[stream_name], f"runtime reconstruction {stream_name}")
        _expect_exact_keys(stream, {"byte_size", "sha256"}, stream_name)
        _require(
            _is_strict_int(stream["byte_size"]) and cast(int, stream["byte_size"]) >= 0,
            f"runtime reconstruction {stream_name} byte size is invalid",
        )
        _require(_is_sha256(stream["sha256"]), f"runtime reconstruction {stream_name} digest")
    expected_stdout = canonical_json_bytes(base_body["runtime_identity"])
    stdout = _expect_dict(record["stdout"], "runtime reconstruction stdout")
    _require(
        stdout
        == {
            "byte_size": len(expected_stdout),
            "sha256": _sha256_bytes(expected_stdout),
        },
        "runtime reconstruction stdout does not bind the exact runtime identity",
    )
    source = _expect_dict(base_body["source_snapshot"], "source snapshot")
    protocol = _expect_dict(base_body["protocol_binding"], "protocol binding")
    _require(
        record["source_manifest_sha256"] == source["manifest_sha256"],
        "runtime reconstruction source binding differs",
    )
    _require(
        record["runtime_identity_sha256"] == base_body["runtime_identity_sha256"],
        "runtime reconstruction identity binding differs",
    )
    _require(
        record["protocol_payload_sha256"] == protocol["protocol_payload_sha256"],
        "runtime reconstruction protocol binding differs",
    )
    return record


def _validate_certification_records(
    records_raw: list[object],
    base_body: Mapping[str, object],
) -> None:
    _require(len(records_raw) == len(CERTIFICATION_SPECS), "certification count differs")
    source = _expect_dict(base_body["source_snapshot"], "source_snapshot")
    protocol = _expect_dict(base_body["protocol_binding"], "protocol_binding")
    for index, (raw, spec) in enumerate(zip(records_raw, CERTIFICATION_SPECS, strict=True)):
        record = _expect_dict(raw, f"certifications[{index}]")
        _expect_exact_keys(
            record,
            {
                "schema",
                "certification_id",
                "node_ids",
                "node_manifest",
                "node_manifest_sha256",
                "execution_manifest",
                "execution_manifest_sha256",
                "command",
                "harness_sha256",
                "environment_policy",
                "status",
                "exit_code",
                "stdout",
                "stderr",
                "source_manifest_sha256",
                "runtime_identity_sha256",
                "protocol_payload_sha256",
            },
            f"certifications[{index}]",
        )
        _require(record["schema"] == READINESS_CERTIFICATION_SCHEMA, "certification schema differs")
        _require(record["certification_id"] == spec.certification_id, "certification order differs")
        _require(record["node_ids"] == list(spec.node_ids), "certification node IDs differ")
        _require(
            record["node_manifest"] == spec.node_manifest,
            "certification node manifest differs",
        )
        _require(
            record["node_manifest_sha256"] == spec.node_manifest_sha256,
            "certification node manifest digest differs",
        )
        execution_manifest = _validate_certification_execution_manifest(
            record["execution_manifest"],
            spec,
        )
        _require(
            record["execution_manifest_sha256"] == canonical_sha256(execution_manifest),
            "certification execution manifest digest differs",
        )
        _require(record["command"] == list(spec.semantic_command), "certification command differs")
        _require(
            record["harness_sha256"] == _CERTIFICATION_BOOTSTRAP_SHA256,
            "certification harness digest differs",
        )
        _require(
            record["environment_policy"] == _CERTIFICATION_ENVIRONMENT_POLICY,
            "certification environment policy differs",
        )
        _require(record["status"] == "passed", "certification status is not passed")
        _require(
            _is_strict_int(record["exit_code"]) and record["exit_code"] == 0,
            "certification exit code is not strict integer zero",
        )
        for stream_name in ("stdout", "stderr"):
            stream = _expect_dict(record[stream_name], f"certification {stream_name}")
            _expect_exact_keys(stream, {"byte_size", "sha256"}, stream_name)
            _require(
                _is_strict_int(stream["byte_size"]) and cast(int, stream["byte_size"]) >= 0,
                f"certification {stream_name} byte size is invalid",
            )
            _require(_is_sha256(stream["sha256"]), f"certification {stream_name} digest invalid")
        _require(
            record["source_manifest_sha256"] == source["manifest_sha256"],
            "certification source binding differs",
        )
        _require(
            record["runtime_identity_sha256"] == base_body["runtime_identity_sha256"],
            "certification runtime binding differs",
        )
        _require(
            record["protocol_payload_sha256"] == protocol["protocol_payload_sha256"],
            "certification protocol binding differs",
        )


def finalize_readiness_receipt(
    draft: ReadinessDraft,
    certifications: VerifiedCertificationBundle,
) -> PreparedReadinessReceipt:
    """Finalize an in-memory receipt; this function performs no publication."""

    _validate_draft_seal(draft)
    _validate_certification_bundle(draft, certifications)
    _require(
        _protected_guard() == draft.base_body["source_literal_outcome_guard"],
        "source literal guard drift before receipt finalization",
    )
    body = dict(draft.base_body)
    body["certification_contract"] = {
        "specifications": _spec_payload(),
        "specifications_sha256": canonical_sha256(_spec_payload()),
        "runtime_reconstruction_record": certifications.runtime_reconstruction_record,
        "runtime_reconstruction_record_sha256": canonical_sha256(
            certifications.runtime_reconstruction_record
        ),
        "records": list(certifications.records),
        "records_sha256": canonical_sha256(list(certifications.records)),
        "all_required_certifications_passed": True,
    }
    body["authorization"] = {
        "ready_for_calibration": True,
        "calibration_execution_requires_separate_explicit_authorization": True,
        "calibration_outcomes_observed": False,
        "thresholds_frozen": False,
        "protected_candidate_execution_permitted": False,
        "scientific_promotion_permitted": False,
    }
    digest = canonical_sha256(body)
    payload: dict[str, object] = {
        "body": body,
        "receipt_sha256": digest,
    }
    seal = _seal("prepared-readiness-receipt-v1", payload, draft.repository_root)
    prepared = PreparedReadinessReceipt(payload, draft.source_archive, draft.repository_root, seal)
    validation = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=draft.repository_root,
        recheck_current=True,
        recheck_runtime=True,
    )
    _require(validation.valid, "; ".join(validation.errors))
    return prepared


def _validate_source_manifest_shape(manifest_raw: object) -> dict[str, object]:
    manifest = _expect_dict(manifest_raw, "source manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "closure_kind",
            "repository_subtree",
            "root_modules",
            "calibration_runner_module",
            "files",
            "support_files",
        },
        "source manifest",
    )
    _require(manifest["schema"] == READINESS_SOURCE_SCHEMA, "source manifest schema differs")
    _require(
        manifest["closure_kind"] == "static_transitive_local_python_imports",
        "source closure kind differs",
    )
    _require(manifest["repository_subtree"] == "research/alberta", "source subtree differs")
    runner = manifest["calibration_runner_module"]
    _require(runner == _CALIBRATION_RUNNER_MODULE, "calibration runner binding is invalid")
    _require(
        manifest["root_modules"] == list(_BASE_SOURCE_ROOT_MODULES),
        "source root modules differ",
    )
    seen_locators: set[str] = set()
    file_items = _expect_list(manifest["files"], "source files")
    _require(bool(file_items), "source closure is empty")
    modules: list[str] = []
    for index, raw in enumerate(file_items):
        item = _expect_dict(raw, f"source files[{index}]")
        _expect_exact_keys(item, {"module", "locator", "byte_size", "sha256"}, "source file")
        module = item["module"]
        _require(type(module) is str and bool(module), "source module is invalid")
        module_text = cast(str, module)
        locator = cast(str, item["locator"])
        module_path = PurePosixPath(*module_text.split("."))
        _require(
            locator
            in {
                module_path.with_suffix(".py").as_posix(),
                (module_path / "__init__.py").as_posix(),
            },
            "source module and locator disagree",
        )
        modules.append(module_text)
        _validate_source_entry(item, seen_locators)
    _require(modules == sorted(modules), "source modules are not sorted")
    _require(len(modules) == len(set(modules)), "source modules are duplicated")
    support_items = _expect_list(manifest["support_files"], "support files")
    roles: list[tuple[str, str]] = []
    for index, raw in enumerate(support_items):
        item = _expect_dict(raw, f"support files[{index}]")
        _expect_exact_keys(item, {"locator", "role", "byte_size", "sha256"}, "support file")
        role = item["role"]
        _require(role in {"dependency_lock", "certification_source"}, "support role is invalid")
        _validate_source_entry(item, seen_locators)
        roles.append((cast(str, item["locator"]), cast(str, role)))
    expected_support = [
        *((path.as_posix(), "dependency_lock") for path in _LOCK_FILES),
        *((path.as_posix(), "certification_source") for path in _certification_source_paths()),
    ]
    _require(roles == expected_support, "support file set differs")
    return manifest


def _validate_source_entry(item: Mapping[str, object], seen: set[str]) -> None:
    locator = item["locator"]
    _require(type(locator) is str and bool(locator), "source locator is invalid")
    locator_text = cast(str, locator)
    pure = PurePosixPath(locator_text)
    _require(
        not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in locator_text
        and pure.as_posix() == locator_text,
        "source locator is unsafe",
    )
    _require(locator_text not in seen, "source locator is duplicated")
    seen.add(locator_text)
    _require(
        _is_strict_int(item["byte_size"]) and cast(int, item["byte_size"]) >= 0,
        "source byte size is invalid",
    )
    _require(_is_sha256(item["sha256"]), "source digest is invalid")


def _archived_local_imports(
    module: str,
    locator: str,
    raw: bytes,
    available_modules: set[str],
) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=locator)
    except (SyntaxError, UnicodeError) as exc:
        raise ReadinessError(f"cannot parse archived source member {locator}") from exc
    is_package = locator.endswith("/__init__.py")
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()

    def include_exact(candidate: str, *, required: bool) -> None:
        if not candidate.startswith("alberta_framework"):
            return
        if candidate not in available_modules:
            _require(not required, f"archived local import is missing: {candidate}")
            return
        found.add(candidate)
        found.update(_parent_packages(candidate))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                include_exact(alias.name, required=alias.name.startswith("alberta_framework"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                _require(keep >= 0, f"invalid archived relative import in {locator}")
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if not base:
                continue
            include_exact(base, required=base.startswith("alberta_framework"))
            for alias in node.names:
                include_exact(f"{base}.{alias.name}", required=False)
    return found


def _validate_archived_source_closure(
    manifest: Mapping[str, object],
    members: Mapping[str, bytes],
) -> None:
    files = [
        _expect_dict(raw, "source file") for raw in _expect_list(manifest["files"], "source files")
    ]
    locator_by_module = {cast(str, item["module"]): cast(str, item["locator"]) for item in files}
    available = set(locator_by_module)
    pending = set(cast(list[str], manifest["root_modules"]))
    pending.update(parent for module in tuple(pending) for parent in _parent_packages(module))
    visited: set[str] = set()
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        _require(module in available, f"archived source root/import is missing: {module}")
        visited.add(module)
        locator = locator_by_module[module]
        pending.update(
            _archived_local_imports(
                module,
                locator,
                members[locator],
                available,
            )
            - visited
        )
    _require(visited == available, "archived source closure contains unreachable local modules")


def _validate_archive(
    archive: bytes,
    manifest: Mapping[str, object],
    archive_binding_raw: object,
) -> dict[str, bytes]:
    _require(len(archive) <= _MAX_ARCHIVE_BYTES, "source archive exceeds size limit")
    binding = _expect_dict(archive_binding_raw, "source archive binding")
    _expect_exact_keys(
        binding,
        {
            "schema",
            "format",
            "file_name",
            "byte_size",
            "sha256",
            "member_count",
            "member_timestamp",
            "member_mode_octal",
        },
        "source archive binding",
    )
    _require(binding["schema"] == READINESS_ARCHIVE_SCHEMA, "archive schema differs")
    _require(binding["format"] == "zip-stored-deterministic-v1", "archive format differs")
    _require(binding["file_name"] == "source.zip", "archive file name differs")
    _require(binding["byte_size"] == len(archive), "archive byte size differs")
    _require(binding["sha256"] == _sha256_bytes(archive), "archive digest differs")
    _require(binding["member_timestamp"] == list(_ZIP_TIMESTAMP), "archive timestamp differs")
    _require(binding["member_mode_octal"] == "100444", "archive member mode differs")

    entries = [
        *_expect_list(manifest["files"], "source files"),
        *_expect_list(manifest["support_files"], "support files"),
    ]
    expected = {
        cast(str, _expect_dict(raw, "source entry")["locator"]): _expect_dict(raw, "source entry")
        for raw in entries
    }
    _require(binding["member_count"] == len(expected), "archive member count differs")
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ReadinessError("source archive is not a valid ZIP") from exc
    with zf:
        _require(zf.comment == b"", "source ZIP comment is not empty")
        infos = zf.infolist()
        names = [info.filename for info in infos]
        _require(names == sorted(expected), "source ZIP member order or set differs")
        _require(len(names) == len(set(names)), "source ZIP contains duplicate names")
        members: dict[str, bytes] = {}
        for info in infos:
            _require(info.date_time == _ZIP_TIMESTAMP, f"ZIP timestamp differs: {info.filename}")
            _require(info.compress_type == zipfile.ZIP_STORED, "ZIP compression is not stored")
            _require(info.create_system == 3, "ZIP creator system differs")
            _require(info.extra == b"" and info.comment == b"", "ZIP member metadata differs")
            _require((info.external_attr >> 16) == _ZIP_FILE_MODE, "ZIP member mode differs")
            entry = expected[info.filename]
            raw = zf.read(info)
            _require(info.file_size == entry["byte_size"] == len(raw), "ZIP member size differs")
            _require(_sha256_bytes(raw) == entry["sha256"], "ZIP member digest differs")
            members[info.filename] = raw
    _validate_archived_source_closure(manifest, members)
    return members


def _validate_runtime_shape(runtime_raw: object) -> dict[str, object]:
    runtime = _expect_dict(runtime_raw, "runtime identity")
    _expect_exact_keys(
        runtime,
        {
            "schema",
            "python",
            "platform",
            "dependencies",
            "jax",
            "environment",
            "child_environment",
            "library_environment_side_effects",
        },
        "runtime identity",
    )
    _require(runtime["schema"] == READINESS_RUNTIME_SCHEMA, "runtime schema differs")
    python = _expect_dict(runtime["python"], "runtime python")
    _expect_exact_keys(
        python,
        {
            "implementation",
            "version",
            "hexversion",
            "cache_tag",
            "byteorder",
            "executable_sha256",
            "prefix",
            "exec_prefix",
            "purelib",
            "platlib",
            "stdlib",
            "no_site_stdlib_search_paths",
            "stdlib_file_scope",
            "stdlib_file_count",
            "stdlib_directory_count",
            "stdlib_file_total_bytes",
            "stdlib_file_inventory_sha256",
        },
        "runtime python",
    )
    _require(_is_sha256(python["executable_sha256"]), "runtime executable digest invalid")
    for field in ("prefix", "exec_prefix", "purelib", "platlib", "stdlib"):
        value = python[field]
        _require(
            type(value) is str
            and bool(value)
            and Path(value).is_absolute()
            and os.path.abspath(value) == value,
            f"runtime Python path is not exact absolute text: {field}",
        )
    prefix = Path(cast(str, python["prefix"]))
    exec_prefix = Path(cast(str, python["exec_prefix"]))
    purelib = Path(cast(str, python["purelib"]))
    platlib = Path(cast(str, python["platlib"]))
    stdlib = Path(cast(str, python["stdlib"]))
    _require(purelib.is_relative_to(prefix), "runtime purelib is outside prefix")
    _require(platlib.is_relative_to(exec_prefix), "runtime platlib is outside exec prefix")
    no_site_paths = _expect_list(
        python["no_site_stdlib_search_paths"],
        "runtime no-site standard-library search paths",
    )
    expected_no_site_paths = [
        (stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip").as_posix(),
        stdlib.as_posix(),
        (stdlib / "lib-dynload").as_posix(),
    ]
    _require(
        no_site_paths == expected_no_site_paths,
        "runtime no-site standard-library search paths differ from the exact schema",
    )
    _require(python["stdlib_file_scope"] == _STDLIB_FILE_SCOPE, "stdlib file scope differs")
    _require(
        _is_strict_int(python["stdlib_file_count"])
        and cast(int, python["stdlib_file_count"]) > 0,
        "stdlib file count is invalid",
    )
    _require(
        _is_strict_int(python["stdlib_directory_count"])
        and cast(int, python["stdlib_directory_count"]) > 0,
        "stdlib directory count is invalid",
    )
    _require(
        _is_strict_int(python["stdlib_file_total_bytes"])
        and cast(int, python["stdlib_file_total_bytes"]) > 0,
        "stdlib file total bytes is invalid",
    )
    _require(
        _is_sha256(python["stdlib_file_inventory_sha256"]),
        "stdlib file inventory digest is invalid",
    )
    system = _expect_dict(runtime["platform"], "runtime platform")
    _expect_exact_keys(
        system,
        {"system", "release", "version_sha256", "machine", "libc", "cpu_count"},
        "runtime platform",
    )
    _require(_is_sha256(system["version_sha256"]), "platform version digest invalid")
    dependencies = _expect_dict(runtime["dependencies"], "runtime dependencies")
    _expect_exact_keys(
        dependencies,
        {
            "key_versions",
            "installed_distribution_count",
            "installed_distribution_file_scope",
            "installed_distribution_file_count",
            "installed_distribution_file_total_bytes",
            "installed_distribution_file_inventory_sha256",
            "dependency_import_tree_file_scope",
            "dependency_import_tree_root_count",
            "dependency_import_tree_file_count",
            "dependency_import_tree_directory_count",
            "dependency_import_tree_file_total_bytes",
            "dependency_import_tree_file_inventory_sha256",
        },
        "runtime dependencies",
    )
    key_versions = _expect_dict(dependencies["key_versions"], "key dependency versions")
    _expect_exact_keys(key_versions, set(_KEY_DISTRIBUTIONS), "key dependency versions")
    _require(
        all(type(version) is str and bool(version) for version in key_versions.values()),
        "key dependency version is invalid",
    )
    _require(
        _is_strict_int(dependencies["installed_distribution_count"])
        and cast(int, dependencies["installed_distribution_count"]) > 0,
        "installed distribution count is invalid",
    )
    _require(
        dependencies["installed_distribution_file_scope"]
        == _INSTALLED_DISTRIBUTION_FILE_SCOPE,
        "installed distribution file scope differs",
    )
    _require(
        _is_strict_int(dependencies["installed_distribution_file_count"])
        and cast(int, dependencies["installed_distribution_file_count"]) > 0,
        "installed distribution file count is invalid",
    )
    _require(
        _is_strict_int(dependencies["installed_distribution_file_total_bytes"])
        and cast(int, dependencies["installed_distribution_file_total_bytes"]) > 0,
        "installed distribution file total bytes is invalid",
    )
    _require(
        _is_sha256(dependencies["installed_distribution_file_inventory_sha256"]),
        "installed distribution file inventory digest invalid",
    )
    _require(
        dependencies["dependency_import_tree_file_scope"]
        == _DEPENDENCY_IMPORT_TREE_FILE_SCOPE,
        "dependency import tree file scope differs",
    )
    expected_root_count = len({cast(str, python["purelib"]), cast(str, python["platlib"])})
    _require(
        dependencies["dependency_import_tree_root_count"] == expected_root_count,
        "dependency import tree root count differs",
    )
    _require(
        _is_strict_int(dependencies["dependency_import_tree_file_count"])
        and cast(int, dependencies["dependency_import_tree_file_count"]) > 0,
        "dependency import tree file count is invalid",
    )
    _require(
        _is_strict_int(dependencies["dependency_import_tree_directory_count"])
        and cast(int, dependencies["dependency_import_tree_directory_count"]) > 0,
        "dependency import tree directory count is invalid",
    )
    _require(
        _is_strict_int(dependencies["dependency_import_tree_file_total_bytes"])
        and cast(int, dependencies["dependency_import_tree_file_total_bytes"]) > 0,
        "dependency import tree file total bytes is invalid",
    )
    _require(
        _is_sha256(dependencies["dependency_import_tree_file_inventory_sha256"]),
        "dependency import tree file inventory digest invalid",
    )
    jax = _expect_dict(runtime["jax"], "runtime jax")
    _expect_exact_keys(jax, {"default_backend", "enable_x64", "config_sha256", "devices"}, "jax")
    _require(_is_sha256(jax["config_sha256"]), "JAX config digest invalid")
    _require(bool(_expect_list(jax["devices"], "JAX devices")), "JAX device list is empty")
    environment = _expect_list(runtime["environment"], "runtime environment")
    names: list[str] = []
    for raw in environment:
        item = _expect_dict(raw, "runtime environment item")
        _expect_exact_keys(item, {"name", "present", "value_sha256", "value_length"}, "environment")
        _require(type(item["name"]) is str, "environment name is invalid")
        _require(item["present"] is True, "environment presence marker differs")
        _require(_is_sha256(item["value_sha256"]), "environment value digest invalid")
        _require(_is_strict_int(item["value_length"]), "environment value length invalid")
        names.append(cast(str, item["name"]))
    _require(names == sorted(names) and len(names) == len(set(names)), "environment names differ")
    _require(environment == [], "runtime environment must not serialize parent values")
    _validated_child_environment(runtime["child_environment"])
    _require(
        runtime["library_environment_side_effects"]
        == _BOUND_LIBRARY_ENVIRONMENT_SIDE_EFFECTS,
        "runtime library environment side-effect binding differs",
    )
    return runtime


def runtime_execution_identity_from_receipt(runtime_raw: object) -> dict[str, object]:
    """Project the full attested runtime onto the cheap per-process identity."""

    runtime = _validate_runtime_shape(runtime_raw)
    python = _expect_dict(runtime["python"], "runtime Python")
    dependencies = _expect_dict(runtime["dependencies"], "runtime dependencies")
    return {
        "schema": READINESS_RUNTIME_EXECUTION_SCHEMA,
        "python": {
            key: python[key]
            for key in (
                "implementation",
                "version",
                "hexversion",
                "cache_tag",
                "byteorder",
                "executable_sha256",
                "prefix",
                "exec_prefix",
                "purelib",
                "platlib",
                "stdlib",
                "no_site_stdlib_search_paths",
            )
        },
        "platform": runtime["platform"],
        "dependencies": {"key_versions": dependencies["key_versions"]},
        "jax": runtime["jax"],
        "child_environment": runtime["child_environment"],
        "library_environment_side_effects": runtime[
            "library_environment_side_effects"
        ],
    }


def require_current_full_runtime_identity(runtime_raw: object) -> None:
    """Require a complete current byte inventory immediately before finalization."""

    expected = _validate_runtime_shape(runtime_raw)
    _require(
        _build_runtime_identity() == expected,
        "complete runtime identity differs immediately before case finalization",
    )


def _validate_protocol_shape(protocol_raw: object) -> dict[str, object]:
    protocol = _expect_dict(protocol_raw, "protocol binding")
    _expect_exact_keys(
        protocol,
        {
            "receipt_schema",
            "design_schema",
            "design_envelope_schema",
            "protocol_status",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "manifest_bindings",
            "manifest_bindings_sha256",
            "recurrence_eligibility_sha256",
            "gate_matrix_sha256",
            "development_summary_schema",
            "primitive_trace_schema",
            "consumed_calibration_namespace_sha256",
            "matched_case_count",
        },
        "protocol binding",
    )
    for key in (
        "protocol_payload_sha256",
        "seed_snapshot_sha256",
        "manifest_bindings_sha256",
        "recurrence_eligibility_sha256",
        "gate_matrix_sha256",
        "consumed_calibration_namespace_sha256",
    ):
        _require(_is_sha256(protocol[key]), f"protocol digest is invalid: {key}")
    manifests = _expect_list(protocol["manifest_bindings"], "manifest bindings")
    _require(
        protocol["manifest_bindings_sha256"] == _protocol_canonical_sha256(manifests),
        "manifest binding digest differs",
    )
    _require(protocol["matched_case_count"] == N_MATCHED_CASES, "matched case count differs")
    return protocol


def _expected_execution_governance_binding(
    body: Mapping[str, object],
) -> dict[str, object]:
    source = _expect_dict(body["source_snapshot"], "source snapshot")
    archive = _expect_dict(source["archive"], "source archive")
    genesis = build_calibration_execution_genesis(
        source_archive_sha256=cast(str, archive["sha256"]),
        source_manifest_sha256=cast(str, source["manifest_sha256"]),
        runtime_identity_sha256=cast(str, body["runtime_identity_sha256"]),
    )
    validated = require_valid_calibration_execution_genesis(genesis)
    return calibration_execution_genesis_receipt_binding(validated)


def validate_readiness_receipt(
    payload: Mapping[str, object],
    source_archive: bytes,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = True,
    recheck_runtime: bool = True,
) -> ReadinessValidation:
    """Validate an in-memory receipt and source archive without running outcomes."""

    errors: list[str] = []
    try:
        _require(type(payload) is dict, "readiness payload must be a plain object")
        _expect_exact_keys(payload, {"body", "receipt_sha256"}, "readiness envelope")
        body = _expect_dict(payload["body"], "readiness body")
        _expect_exact_keys(
            body,
            {
                "receipt_schema",
                "envelope_schema",
                "status",
                "development_only",
                "scientific_promotion_allowed",
                "protocol_binding",
                "component_schema_binding",
                "source_snapshot",
                "runtime_identity",
                "runtime_identity_sha256",
                READINESS_EXECUTION_GOVERNANCE_FIELD,
                "worker_execution",
                "source_literal_outcome_guard",
                "claim_scope",
                "certification_contract",
                "authorization",
            },
            "readiness body",
        )
        _require(
            body["receipt_schema"] == CALIBRATION_READINESS_RECEIPT_SCHEMA,
            "receipt schema differs",
        )
        _require(body["envelope_schema"] == READINESS_ENVELOPE_SCHEMA, "envelope schema differs")
        _require(body["status"] == READINESS_STATUS, "readiness status differs")
        _require(body["development_only"] is True, "receipt must remain development-only")
        _require(body["scientific_promotion_allowed"] is False, "receipt cannot promote")
        _require(_is_sha256(payload["receipt_sha256"]), "receipt digest is invalid")
        _require(payload["receipt_sha256"] == canonical_sha256(body), "receipt body digest differs")

        protocol = _validate_protocol_shape(body["protocol_binding"])
        _require(
            body["component_schema_binding"] == _component_schema_binding(),
            "component schema binding differs",
        )
        source = _expect_dict(body["source_snapshot"], "source snapshot")
        _expect_exact_keys(source, {"manifest", "manifest_sha256", "archive"}, "source snapshot")
        manifest = _validate_source_manifest_shape(source["manifest"])
        _require(
            source["manifest_sha256"] == canonical_sha256(manifest),
            "source manifest digest differs",
        )
        _validate_archive(source_archive, manifest, source["archive"])
        runtime = _validate_runtime_shape(body["runtime_identity"])
        _require(
            body["runtime_identity_sha256"] == canonical_sha256(runtime),
            "runtime identity digest differs",
        )
        governance = _expect_dict(
            body[READINESS_EXECUTION_GOVERNANCE_FIELD],
            READINESS_EXECUTION_GOVERNANCE_FIELD,
        )
        _require(
            governance.get("schema") == CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
            "execution governance binding schema differs",
        )
        _require(
            governance == _expected_execution_governance_binding(body),
            "execution governance binding differs from pristine deterministic genesis",
        )
        _require(
            governance.get("managed_boundary_scope") == MANAGED_EXECUTION_BOUNDARY_SCOPE,
            "execution governance boundary scope differs",
        )
        worker = _expect_dict(body["worker_execution"], "worker execution")
        _expect_exact_keys(
            worker,
            {
                "calibration_runner_module",
                "entrypoint",
                "allowed_entrypoint_modes",
                "isolated_flag",
                "no_site_flag",
                "dont_write_bytecode_flag",
                "safe_path_flag",
                "pycache_prefix_option",
                "bytecode_cache_policy",
                "runtime_path_policy",
                "runtime_mutation_policy",
                "child_environment_policy",
                "loaded_module_origin_policy",
                "working_directory",
                "project_source_path",
                "project_module_provenance_required",
                "explicit_execution_authorization_required",
            },
            "worker execution",
        )
        runner = manifest["calibration_runner_module"]
        _require(worker["calibration_runner_module"] == runner, "worker runner binding differs")
        _require(
            worker["entrypoint"] == ("main" if runner is not None else None),
            "entrypoint differs",
        )
        _require(
            worker["allowed_entrypoint_modes"] == list(_ALLOWED_WORKER_ENTRYPOINT_MODES),
            "allowed worker entrypoint modes differ",
        )
        _require(worker["isolated_flag"] is None, "worker isolation flag must be absent")
        _require(worker["no_site_flag"] == "-S", "worker no-site flag differs")
        _require(
            worker["dont_write_bytecode_flag"] == "-B",
            "worker no-bytecode-write flag differs",
        )
        _require(worker["safe_path_flag"] == "-P", "worker safe-path flag differs")
        _require(
            worker["pycache_prefix_option"]
            == "-X pycache_prefix={fresh_empty_separate_bytecode_cache_root}",
            "worker bytecode-cache option differs",
        )
        _require(
            worker["bytecode_cache_policy"] == _BYTECODE_CACHE_POLICY,
            "worker bytecode-cache policy differs",
        )
        _require(
            worker["runtime_path_policy"] == _RUNTIME_PATH_POLICY,
            "worker runtime-path policy differs",
        )
        _require(
            worker["runtime_mutation_policy"] == _RUNTIME_MUTATION_POLICY,
            "worker runtime-mutation policy differs",
        )
        _require(
            worker["child_environment_policy"] == _CHILD_ENVIRONMENT_POLICY,
            "worker child-environment policy differs",
        )
        _require(
            worker["loaded_module_origin_policy"] == _LOADED_MODULE_ORIGIN_POLICY,
            "worker loaded-module origin policy differs",
        )
        _require(
            worker["working_directory"] == "fresh_empty_temporary_directory",
            "worker working directory contract differs",
        )
        _require(
            worker["project_source_path"] == "content_addressed_source_zip_first_and_sole",
            "worker project source path contract differs",
        )
        _require(
            worker["project_module_provenance_required"]
            == "zipimport_loader_and_file_inside_source_zip",
            "worker provenance contract differs",
        )
        _require(worker["explicit_execution_authorization_required"] is True, "worker auth differs")

        guard = _expect_dict(body["source_literal_outcome_guard"], "source literal guard")
        _expect_exact_keys(
            guard,
            {
                "scope",
                "learner_outcome_constant",
                "ledger_all_false",
                "ledger_entry_count",
                "execution_absence_attested",
            },
            "source literal guard",
        )
        _require(
            guard["scope"] == "source_literals_only_not_managed_or_external_execution_history"
            and guard["learner_outcome_constant"] is False
            and guard["ledger_all_false"] is True
            and guard["execution_absence_attested"] is False,
            "source literal guard overclaims execution history",
        )
        certification = _expect_dict(body["certification_contract"], "certification contract")
        _expect_exact_keys(
            certification,
            {
                "specifications",
                "specifications_sha256",
                "runtime_reconstruction_record",
                "runtime_reconstruction_record_sha256",
                "records",
                "records_sha256",
                "all_required_certifications_passed",
            },
            "certification contract",
        )
        _require(certification["specifications"] == _spec_payload(), "certification specs differ")
        _require(
            certification["specifications_sha256"] == canonical_sha256(_spec_payload()),
            "certification specification digest differs",
        )
        runtime_reconstruction = _validate_runtime_reconstruction_record(
            certification["runtime_reconstruction_record"],
            body,
        )
        _require(
            certification["runtime_reconstruction_record_sha256"]
            == canonical_sha256(runtime_reconstruction),
            "runtime reconstruction record digest differs",
        )
        records = _expect_list(certification["records"], "certification records")
        _validate_certification_records(records, body)
        _require(
            certification["records_sha256"] == canonical_sha256(records),
            "certification records digest differs",
        )
        _require(
            certification["all_required_certifications_passed"] is True,
            "certifications incomplete",
        )
        authorization = _expect_dict(body["authorization"], "authorization")
        _expect_exact_keys(
            authorization,
            {
                "ready_for_calibration",
                "calibration_execution_requires_separate_explicit_authorization",
                "calibration_outcomes_observed",
                "thresholds_frozen",
                "protected_candidate_execution_permitted",
                "scientific_promotion_permitted",
            },
            "authorization",
        )
        _require(
            authorization
            == {
                "ready_for_calibration": True,
                "calibration_execution_requires_separate_explicit_authorization": True,
                "calibration_outcomes_observed": False,
                "thresholds_frozen": False,
                "protected_candidate_execution_permitted": False,
                "scientific_promotion_permitted": False,
            },
            "authorization policy differs",
        )
        if recheck_current:
            root = repository_root.absolute()
            current_manifest, current_archive = _build_source_bundle(root)
            _require(
                current_manifest == manifest,
                "source closure no longer matches current source",
            )
            _require(
                current_archive == source_archive,
                "source archive no longer matches current source",
            )
            _require(
                _protocol_binding() == protocol,
                "protocol binding no longer matches current source",
            )
            _require(_protected_guard() == guard, "protected ledger no longer matches receipt")
        if recheck_runtime:
            _require(
                _build_runtime_identity() == runtime,
                "runtime/JAX/device/dependency/environment identity drift",
            )
    except (ReadinessError, KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(str(exc))
    return ReadinessValidation(not errors, not errors, tuple(errors))


def require_validated_readiness_receipt(
    payload: Mapping[str, object],
    source_archive: bytes,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = False,
    recheck_runtime: bool = True,
) -> ValidatedReadinessBundle:
    """Return runner-consumable identities or raise on any receipt/ZIP defect.

    ``recheck_current=False`` is intentional for an isolated worker: the exact
    executable checkout is the already-validated ZIP, not whatever mutable
    checkout happens to exist on the host.  Callers issuing or publishing a
    receipt use ``recheck_current=True`` instead.
    """

    validation = validate_readiness_receipt(
        payload,
        source_archive,
        repository_root=repository_root,
        recheck_current=recheck_current,
        recheck_runtime=recheck_runtime,
    )
    _require(validation.valid, "; ".join(validation.errors))
    normalized = dict(payload)
    body = _expect_dict(normalized["body"], "readiness body")
    source = _expect_dict(body["source_snapshot"], "source snapshot")
    archive = _expect_dict(source["archive"], "source archive")
    worker = _expect_dict(body["worker_execution"], "worker execution")
    governance = _expect_dict(
        body[READINESS_EXECUTION_GOVERNANCE_FIELD],
        READINESS_EXECUTION_GOVERNANCE_FIELD,
    )
    return ValidatedReadinessBundle(
        payload=normalized,
        receipt_sha256=cast(str, normalized["receipt_sha256"]),
        source_archive_sha256=cast(str, archive["sha256"]),
        source_manifest_sha256=cast(str, source["manifest_sha256"]),
        runtime_identity_sha256=cast(str, body["runtime_identity_sha256"]),
        calibration_runner_module=cast(str, worker["calibration_runner_module"]),
        execution_genesis_sha256=cast(str, governance["genesis_sha256"]),
    )


def _open_directory_without_symlinks(path: Path) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ReadinessError(
                        f"symlinked readiness directory is forbidden: {absolute}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, absolute


def _open_immutable_regular(path: Path, *, max_bytes: int) -> bytes:
    parent_fd, absolute_parent = _open_directory_without_symlinks(path.parent)
    parent_status = os.fstat(parent_fd)
    absolute = absolute_parent / path.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ReadinessError(f"symlinked readiness path is forbidden: {absolute}") from exc
        raise
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"readiness path is not regular: {path}")
        _require(stat.S_IMODE(before.st_mode) == 0o444, f"readiness file mode is not 0444: {path}")
        _require(before.st_nlink == 1, f"readiness file has multiple hard links: {path}")
        _require(before.st_size <= max_bytes, f"readiness file exceeds size limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= max_bytes, f"readiness file exceeds size limit: {path}")
        after = os.fstat(descriptor)
        locator = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            _stat_identity(before) == _stat_identity(after) == _stat_identity(locator),
            f"readiness file changed or was replaced during read: {path}",
        )
        reopened_parent_fd, reopened_parent = _open_directory_without_symlinks(
            absolute_parent
        )
        try:
            reopened_status = os.fstat(reopened_parent_fd)
            _require(
                reopened_parent == absolute_parent
                and (reopened_status.st_dev, reopened_status.st_ino)
                == (parent_status.st_dev, parent_status.st_ino),
                f"readiness file parent changed during read: {path}",
            )
        finally:
            os.close(reopened_parent_fd)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _write_new_immutable(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing readiness file")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_directory_new_only(
    root_fd: int,
    staging_name: str,
    final_name: str,
) -> None:
    """Atomically expose a complete directory without replacing any destination."""

    renameat2 = cast(Any, getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None))
    _require(renameat2 is not None, "atomic new-only directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        root_fd,
        os.fsencode(staging_name),
        root_fd,
        os.fsencode(final_name),
        1,  # Linux RENAME_NOREPLACE.
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, "refusing to overwrite readiness receipt", final_name)
    raise OSError(error_number, os.strerror(error_number), final_name)


def _discard_readiness_staging_directory(root_fd: int, staging_name: str) -> None:
    """Best-effort cleanup of the exact private staging directory from this call."""

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(staging_name, flags, dir_fd=root_fd)
    except FileNotFoundError:
        return
    try:
        os.fchmod(directory_fd, 0o700)
        for member in ("readiness.json", "source.zip"):
            try:
                os.unlink(member, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(staging_name, dir_fd=root_fd)
    except FileNotFoundError:
        pass


def publish_readiness_receipt(
    prepared: PreparedReadinessReceipt,
    publication_root: Path,
    *,
    authorize_publication: bool,
) -> PublishedReadinessReceipt:
    """Publish ``<root>/<digest>/{readiness.json,source.zip}`` new-only and 0444."""

    _require(authorize_publication is True, "readiness publication requires explicit authorization")
    expected_seal = _seal(
        "prepared-readiness-receipt-v1",
        prepared.payload,
        prepared.repository_root,
    )
    _require(hmac.compare_digest(prepared.seal, expected_seal), "prepared receipt seal is invalid")
    validation = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=prepared.repository_root,
        recheck_current=True,
        recheck_runtime=True,
    )
    _require(validation.valid, "; ".join(validation.errors))
    digest = cast(str, prepared.payload["receipt_sha256"])
    root_fd, root = _open_directory_without_symlinks(publication_root)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    staging_name = f".staging-readiness-{digest}-{os.getpid()}-{secrets.token_hex(16)}"
    directory_fd: int | None = None
    staging_exists = False
    try:
        try:
            os.mkdir(staging_name, 0o700, dir_fd=root_fd)
            staging_exists = True
        except FileExistsError as exc:
            raise ReadinessError("private readiness staging-name collision") from exc
        directory_fd = os.open(staging_name, flags, dir_fd=root_fd)
        _write_new_immutable(directory_fd, "readiness.json", canonical_json_bytes(prepared.payload))
        _write_new_immutable(directory_fd, "source.zip", prepared.source_archive)
        os.fsync(directory_fd)
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
        os.fsync(root_fd)
        _install_directory_new_only(root_fd, staging_name, digest)
        staging_exists = False
        os.fsync(root_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite readiness receipt: {root / digest}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if staging_exists:
            _discard_readiness_staging_directory(root_fd, staging_name)
            os.fsync(root_fd)
        os.close(root_fd)
    directory = root / digest
    return PublishedReadinessReceipt(
        directory,
        directory / "readiness.json",
        directory / "source.zip",
        digest,
    )


def validate_published_readiness_receipt(
    directory: Path,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = True,
    recheck_runtime: bool = True,
) -> ReadinessValidation:
    """Validate immutable modes, content address, canonical JSON, ZIP, and drift."""

    errors: list[str] = []
    try:
        _read_validated_published_readiness_bytes(
            directory,
            repository_root=repository_root,
            recheck_current=recheck_current,
            recheck_runtime=recheck_runtime,
        )
    except (ReadinessError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        errors.append(str(exc))
    return ReadinessValidation(not errors, not errors, tuple(errors))


def _read_validated_published_readiness_bytes(
    directory: Path,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool,
    recheck_runtime: bool,
) -> tuple[dict[str, object], bytes, bytes, ValidatedReadinessBundle]:
    """Read one exact publication snapshot and validate those same bytes."""

    absolute = directory.absolute()
    mode = absolute.lstat().st_mode
    _require(stat.S_ISDIR(mode), "readiness publication path is not a directory")
    _require(stat.S_IMODE(mode) == 0o555, "readiness publication directory mode is not 0555")
    expected_members = ["readiness.json", "source.zip"]
    _require(
        sorted(path.name for path in absolute.iterdir()) == expected_members,
        "readiness publication members differ",
    )
    receipt_raw = _open_immutable_regular(
        absolute / "readiness.json",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    archive_raw = _open_immutable_regular(
        absolute / "source.zip",
        max_bytes=_MAX_ARCHIVE_BYTES,
    )
    payload = _strict_json_loads(receipt_raw)
    digest = payload.get("receipt_sha256")
    _require(
        type(digest) is str and absolute.name == digest,
        "content-addressed directory differs",
    )
    validated = require_validated_readiness_receipt(
        payload,
        archive_raw,
        repository_root=repository_root,
        recheck_current=recheck_current,
        recheck_runtime=recheck_runtime,
    )
    _require(
        receipt_raw == canonical_json_bytes(validated.payload),
        "validated readiness receipt bytes changed during normalization",
    )
    _require(
        sorted(path.name for path in absolute.iterdir()) == expected_members
        and stat.S_IMODE(absolute.lstat().st_mode) == 0o555,
        "readiness publication changed while its exact bytes were validated",
    )
    return payload, receipt_raw, archive_raw, validated


def load_validated_published_readiness_bundle(
    directory: Path,
    *,
    recheck_current: bool = True,
    recheck_runtime: bool = True,
) -> ValidatedReadinessBundle:
    """Read one coherent publication byte pair once and return its strict binding."""

    _payload, _receipt_raw, _archive_raw, validated = (
        _read_validated_published_readiness_bytes(
            directory,
            repository_root=_REPO_ROOT,
            recheck_current=recheck_current,
            recheck_runtime=recheck_runtime,
        )
    )
    return validated


def _runtime_batch_guard_payload(
    guard: BoundCalibrationRuntimeBatch,
) -> dict[str, object]:
    return {
        "directory": guard.directory.as_posix(),
        "receipt_sha256": guard.receipt_sha256,
        "source_archive_sha256": guard.source_archive_sha256,
        "runtime_identity_sha256": guard.runtime_identity_sha256,
        "pid": guard.pid,
        "process_start_nonce": guard.process_start_nonce,
        "nonce": guard.nonce,
    }


def _require_active_runtime_batch_guard(
    guard: BoundCalibrationRuntimeBatch,
    *,
    directory: Path,
    validated: ValidatedReadinessBundle,
) -> None:
    _require(
        type(guard) is BoundCalibrationRuntimeBatch,
        "runtime batch guard has an invalid type",
    )
    _require(
        guard.pid == os.getpid()
        and guard.process_start_nonce == _PROCESS_START_NONCE,
        "runtime batch guard belongs to another process",
    )
    _require(
        _ACTIVE_RUNTIME_BATCH_GUARDS.get(guard.nonce) is guard,
        "runtime batch guard is inactive",
    )
    _require(
        hmac.compare_digest(
            guard.seal,
            _seal("bound-runtime-batch-v1", _runtime_batch_guard_payload(guard), _REPO_ROOT),
        ),
        "runtime batch guard seal is invalid",
    )
    _require(
        guard.directory == directory.absolute(),
        "runtime batch guard publication directory differs",
    )
    _require(
        (
            guard.receipt_sha256,
            guard.source_archive_sha256,
            guard.runtime_identity_sha256,
        )
        == (
            validated.receipt_sha256,
            validated.source_archive_sha256,
            validated.runtime_identity_sha256,
        ),
        "runtime batch guard readiness binding differs",
    )


@contextmanager
def bound_calibration_runtime_batch(
    directory: Path,
    *,
    authorize_batch_execution: bool,
) -> Iterator[BoundCalibrationRuntimeBatch]:
    """Bracket a bounded worker batch with two complete runtime byte inventories."""

    _require(
        authorize_batch_execution is True,
        "runtime batch guard requires explicit authorization",
    )
    absolute = directory.absolute()
    before = load_validated_published_readiness_bundle(
        absolute,
        recheck_current=False,
        recheck_runtime=True,
    )
    nonce = secrets.token_hex(32)
    provisional = BoundCalibrationRuntimeBatch(
        directory=absolute,
        receipt_sha256=before.receipt_sha256,
        source_archive_sha256=before.source_archive_sha256,
        runtime_identity_sha256=before.runtime_identity_sha256,
        pid=os.getpid(),
        process_start_nonce=_PROCESS_START_NONCE,
        nonce=nonce,
        seal="",
    )
    guard = dataclasses.replace(
        provisional,
        seal=_seal(
            "bound-runtime-batch-v1",
            _runtime_batch_guard_payload(provisional),
            _REPO_ROOT,
        ),
    )
    _require(nonce not in _ACTIVE_RUNTIME_BATCH_GUARDS, "runtime batch nonce collision")
    _ACTIVE_RUNTIME_BATCH_GUARDS[nonce] = guard
    try:
        yield guard
    finally:
        _ACTIVE_RUNTIME_BATCH_GUARDS.pop(nonce, None)
        after = load_validated_published_readiness_bundle(
            absolute,
            recheck_current=False,
            recheck_runtime=True,
        )
        _require(after == before, "readiness/runtime identity drifted during worker batch")


_BOUND_WORKER_BOOTSTRAP = r"""
import hashlib
import importlib
import json
import os
import sys
import types
import zipimport

(
    expected_bytecode_cache_root,
    archive,
    receipt_path,
    expected_receipt,
    expected_archive,
    expected_environment_json,
    module_name,
    entrypoint,
    *argv,
) = sys.argv[1:]
expected_bytecode_cache_root = os.path.abspath(expected_bytecode_cache_root)
archive = os.path.abspath(archive)
receipt_path = os.path.abspath(receipt_path)
expected_environment = json.loads(expected_environment_json)
if sys.flags.no_site != 1:
    raise SystemExit("bound worker interpreter did not start with -S")
if "_virtualenv" in sys.modules:
    raise SystemExit("bound worker executed a pre-bootstrap virtualenv pth hook")
if expected_environment != {"LC_ALL": "C", "PYTHONHASHSEED": "0"}:
    raise SystemExit("bound worker expected environment binding differs")
if dict(os.environ) != expected_environment:
    raise SystemExit("bound worker process environment differs from its exact binding")
if sys.flags.isolated != 0 or not sys.flags.safe_path or sys.flags.hash_randomization != 0:
    raise SystemExit("bound worker interpreter isolation or hash-seed flags differ")
bound_library_environment_side_effects = {
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TPU_SKIP_MDS_QUERY": "1",
}

def require_bound_post_import_environment(initial):
    actual = dict(os.environ)
    if any(actual.get(name) != value for name, value in initial.items()):
        raise SystemExit("bound worker startup environment changed after imports")
    additions = {name: value for name, value in actual.items() if name not in initial}
    if any(
        name not in bound_library_environment_side_effects
        or bound_library_environment_side_effects[name] != value
        for name, value in additions.items()
    ):
        raise SystemExit("bound worker environment gained an unbound value after imports")
    return actual

if not sys.dont_write_bytecode:
    raise SystemExit("bound worker bytecode writes are not disabled")
if not isinstance(sys.pycache_prefix, str):
    raise SystemExit("bound worker has no isolated bytecode-cache prefix")
if os.path.abspath(sys.pycache_prefix) != expected_bytecode_cache_root:
    raise SystemExit("bound worker bytecode-cache prefix differs")
if expected_bytecode_cache_root == os.path.abspath(os.getcwd()):
    raise SystemExit("bound worker bytecode cache is not separate from its cwd")
if os.path.exists(expected_bytecode_cache_root) and os.listdir(expected_bytecode_cache_root):
    raise SystemExit("bound worker bytecode-cache prefix is not fresh and empty")
with open(archive, "rb") as archive_handle:
    archive_raw = archive_handle.read()
if hashlib.sha256(archive_raw).hexdigest() != expected_archive:
    raise SystemExit("bound source archive digest mismatch")
with open(receipt_path, "rb") as receipt_handle:
    receipt_raw = receipt_handle.read()

def validate_json_value(value):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is list:
        for item in value:
            validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise SystemExit("bound receipt contains a non-string JSON key")
            validate_json_value(item)
        return
    raise SystemExit("bound receipt contains a non-canonical JSON value")

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit("bound receipt contains a duplicate JSON key")
        result[key] = value
    return result

def reject_constant(value):
    raise SystemExit("bound receipt contains a non-finite JSON constant: " + value)

try:
    payload = json.loads(
        receipt_raw.decode("ascii"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )
except (UnicodeDecodeError, json.JSONDecodeError):
    raise SystemExit("bound readiness receipt JSON is invalid") from None
validate_json_value(payload)
canonical_receipt = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
if receipt_raw != canonical_receipt:
    raise SystemExit("bound readiness receipt JSON is not canonical")
if type(payload) is not dict or set(payload) != {"body", "receipt_sha256"}:
    raise SystemExit("bound readiness receipt envelope differs")
body = payload.get("body")
if type(body) is not dict:
    raise SystemExit("bound readiness receipt body differs")
canonical_body = json.dumps(
    body,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
body_digest = hashlib.sha256(canonical_body).hexdigest()
if payload.get("receipt_sha256") != body_digest or body_digest != expected_receipt:
    raise SystemExit("bound readiness receipt digest mismatch")
try:
    runtime_identity = body["runtime_identity"]
    runtime_python = runtime_identity["python"]
    runtime_prefix = runtime_python["prefix"]
    runtime_exec_prefix = runtime_python["exec_prefix"]
    purelib = runtime_python["purelib"]
    platlib = runtime_python["platlib"]
    stdlib = runtime_python["stdlib"]
    no_site_paths = runtime_python["no_site_stdlib_search_paths"]
    receipt_environment = runtime_identity["child_environment"]
    receipt_library_environment_side_effects = runtime_identity[
        "library_environment_side_effects"
    ]
    archive_binding = body["source_snapshot"]["archive"]
    worker_binding = body["worker_execution"]
except (KeyError, TypeError):
    raise SystemExit("bound worker runtime path binding is absent") from None
if receipt_environment != expected_environment:
    raise SystemExit("bound worker receipt environment differs")
if receipt_library_environment_side_effects != bound_library_environment_side_effects:
    raise SystemExit("bound worker receipt library environment binding differs")
if type(archive_binding) is not dict or archive_binding.get("sha256") != expected_archive:
    raise SystemExit("bound worker receipt archive binding differs")
if type(worker_binding) is not dict or (
    worker_binding.get("calibration_runner_module") != module_name
    or worker_binding.get("entrypoint") != entrypoint
):
    raise SystemExit("bound worker receipt entrypoint binding differs")
bound_paths = [runtime_prefix, runtime_exec_prefix, purelib, platlib, stdlib]
if not all(
    isinstance(path, str) and os.path.isabs(path) and os.path.abspath(path) == path
    for path in bound_paths
):
    raise SystemExit("bound worker runtime path is not exact absolute text")
if not isinstance(no_site_paths, list) or not all(isinstance(path, str) for path in no_site_paths):
    raise SystemExit("bound worker no-site path binding is invalid")
no_site_paths = [os.path.abspath(path) for path in no_site_paths]
if sys.path != no_site_paths:
    raise SystemExit("bound worker startup path differs from bound no-site paths")
if len(no_site_paths) != 3 or no_site_paths[1:] != [stdlib, os.path.join(stdlib, "lib-dynload")]:
    raise SystemExit("bound worker standard-library path binding differs")
if not all(os.path.isdir(path) for path in bound_paths):
    raise SystemExit("bound worker runtime directory is absent")
site_paths = []
for path in (purelib, platlib):
    if path not in site_paths:
        site_paths.append(path)
sys.prefix = runtime_prefix
sys.exec_prefix = runtime_exec_prefix
sys.path[:] = [archive, *no_site_paths, *site_paths]
expected_sys_path = list(sys.path)

def path_is_within(path, root):
    if not isinstance(path, str) or not path or path.startswith("<"):
        return False
    absolute = os.path.abspath(path)
    root = os.path.abspath(root)
    try:
        return os.path.commonpath((absolute, root)) == root
    except ValueError:
        return False

python_zip = no_site_paths[0]
allowed_roots = list(site_paths)
archive_prefix = archive + os.sep

def stdlib_origin_is_bound(path):
    return path_is_within(path, stdlib)

def origin_is_bound(path):
    return (
        isinstance(path, str)
        and os.path.abspath(path).startswith(archive_prefix)
    ) or any(path_is_within(path, root) for root in allowed_roots) or (
        stdlib_origin_is_bound(path)
    ) or (
        isinstance(path, str)
        and os.path.abspath(path).startswith(os.path.abspath(python_zip) + os.sep)
    )

def audit_loaded_module_origins(*, captured_originless=None, capture_originless=False):
    originless = {}
    for name, loaded in tuple(sys.modules.items()):
        if loaded is None:
            continue
        if not isinstance(loaded, types.ModuleType):
            continue
        spec = getattr(loaded, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        file_origin = getattr(loaded, "__file__", None)
        if spec_origin in ("built-in", "frozen"):
            continue
        if name == "__main__" and spec is None and file_origin is None:
            continue
        origins = []
        for origin in (file_origin, spec_origin):
            if isinstance(origin, str) and origin not in origins:
                origins.append(origin)
        if origins:
            for origin in origins:
                if not origin_is_bound(origin):
                    raise SystemExit("loaded module origin is outside bound roots: " + name)
            continue
        locations = getattr(spec, "submodule_search_locations", None)
        if locations is None:
            if capture_originless:
                originless[name] = loaded
                continue
            if captured_originless is not None and captured_originless.get(name) is loaded:
                continue
            raise SystemExit("loaded module has no auditable origin: " + name)
        location_list = list(locations)
        if not location_list or not all(origin_is_bound(path) for path in location_list):
            raise SystemExit("namespace module location is outside bound roots: " + name)
    if captured_originless is not None and any(
        sys.modules.get(name) is not loaded for name, loaded in captured_originless.items()
    ):
        raise SystemExit("captured originless extension module changed after execution")
    return originless

audit_loaded_module_origins()
module = importlib.import_module(module_name)
post_import_environment = require_bound_post_import_environment(expected_environment)
prefix = archive_prefix
def audit_project_module_origins():
    for name, loaded in tuple(sys.modules.items()):
        if name != "alberta_framework" and not name.startswith("alberta_framework."):
            continue
        origin = getattr(loaded, "__file__", None)
        loader = getattr(loaded, "__loader__", None)
        if not isinstance(loader, zipimport.zipimporter):
            raise SystemExit("project module loader is not zipimport: " + name)
        if not isinstance(origin, str) or not os.path.abspath(origin).startswith(prefix):
            raise SystemExit("project module origin is outside bound ZIP: " + name)

audit_project_module_origins()
captured_originless = audit_loaded_module_origins(capture_originless=True)
if sys.path != expected_sys_path:
    raise SystemExit("bound worker exact import path changed")
target = getattr(module, entrypoint, None)
if not callable(target):
    raise SystemExit("bound calibration entrypoint is not callable")
result = target(argv)
audit_project_module_origins()
audit_loaded_module_origins(captured_originless=captured_originless)
if sys.flags.no_site != 1 or "_virtualenv" in sys.modules:
    raise SystemExit("bound worker no-site boundary changed")
if sys.prefix != runtime_prefix or sys.exec_prefix != runtime_exec_prefix:
    raise SystemExit("bound worker runtime prefix changed")
if sys.path != expected_sys_path:
    raise SystemExit("bound worker exact import path changed after execution")
if dict(os.environ) != post_import_environment:
    raise SystemExit("bound worker process environment changed after execution")
with open(archive, "rb") as archive_handle:
    if hashlib.sha256(archive_handle.read()).hexdigest() != expected_archive:
        raise SystemExit("bound source archive changed during execution")
with open(receipt_path, "rb") as receipt_handle:
    if receipt_handle.read() != receipt_raw:
        raise SystemExit("bound readiness receipt changed during execution")
if os.path.exists(expected_bytecode_cache_root) and os.listdir(expected_bytecode_cache_root):
    raise SystemExit("bound worker wrote into its bytecode-cache prefix")
raise SystemExit(0 if result is None else int(result))
"""


def execute_bound_calibration_worker(
    directory: Path,
    arguments: Sequence[str] = (),
    *,
    authorize_calibration_execution: bool,
    timeout_seconds: int | None = None,
    runtime_batch_guard: BoundCalibrationRuntimeBatch | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Explicitly execute a bound future runner from the ZIP, never the checkout."""

    _require(
        authorize_calibration_execution is True,
        "calibration worker execution requires explicit authorization",
    )
    _require(
        timeout_seconds is None or (_is_strict_int(timeout_seconds) and timeout_seconds > 0),
        "worker timeout must be null or a positive strict integer",
    )
    _require(
        all(type(argument) is str and "\x00" not in argument for argument in arguments),
        "bound calibration worker arguments must be plain NUL-free strings",
    )
    payload, receipt_raw, archive_raw, validated = _read_validated_published_readiness_bytes(
        directory,
        repository_root=_REPO_ROOT,
        recheck_current=False,
        recheck_runtime=False,
    )
    body = _expect_dict(payload["body"], "readiness body")
    worker = _expect_dict(body["worker_execution"], "worker execution")
    module = validated.calibration_runner_module
    entrypoint = worker["entrypoint"]
    _require(type(module) is str and type(entrypoint) is str, "receipt binds no calibration runner")
    allowed_modes = _expect_list(worker["allowed_entrypoint_modes"], "allowed worker modes")
    _require(bool(arguments), "bound calibration worker requires an entrypoint mode")
    _require(
        arguments[0] in allowed_modes,
        "bound calibration worker entrypoint mode is not allowed by the receipt",
    )
    _require(len(arguments) > 1, "bound worker has no readiness publication argument")
    normalized_directory = os.path.normpath(os.path.abspath(os.fspath(directory)))
    normalized_argument_directory = os.path.normpath(os.path.abspath(arguments[1]))
    _require(
        normalized_argument_directory == normalized_directory,
        "bound worker readiness argument differs from the executed publication",
    )
    runtime = _expect_dict(body["runtime_identity"], "runtime identity")
    if runtime_batch_guard is None:
        _require(
            _build_runtime_identity() == runtime,
            "runtime identity drift immediately before bound worker launch",
        )
    else:
        _require_active_runtime_batch_guard(
            runtime_batch_guard,
            directory=directory,
            validated=validated,
        )
    environment = _validated_child_environment(runtime["child_environment"])
    environment_json = canonical_json_bytes(environment).decode("ascii")
    with (
        tempfile.TemporaryDirectory(prefix="alberta-hidden-regime-worker-") as temporary,
        tempfile.TemporaryDirectory(
            prefix="alberta-hidden-regime-bytecode-cache-"
        ) as bytecode_cache,
        tempfile.TemporaryDirectory(
            prefix="alberta-hidden-regime-staged-readiness-"
        ) as staging_root,
    ):
        _require(not any(Path(temporary).iterdir()), "worker temporary directory is not empty")
        _require(
            not any(Path(bytecode_cache).iterdir()),
            "worker bytecode-cache directory is not empty",
        )
        _require(not any(Path(staging_root).iterdir()), "readiness staging root is not empty")
        staged_directory = Path(staging_root) / validated.receipt_sha256
        staged_directory.mkdir(mode=0o700)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        staged_fd = os.open(staged_directory, directory_flags)
        try:
            _write_new_immutable(staged_fd, "readiness.json", receipt_raw)
            _write_new_immutable(staged_fd, "source.zip", archive_raw)
            os.fsync(staged_fd)
        finally:
            os.close(staged_fd)
        staged_directory.chmod(0o555)
        receipt_path = staged_directory / "readiness.json"
        archive_path = staged_directory / "source.zip"
        worker_arguments = list(arguments)
        worker_arguments[1] = staged_directory.as_posix()
        command = (
            sys.executable,
            "-S",
            "-B",
            "-P",
            "-X",
            f"pycache_prefix={bytecode_cache}",
            "-c",
            _BOUND_WORKER_BOOTSTRAP,
            bytecode_cache,
            archive_path.as_posix(),
            receipt_path.as_posix(),
            validated.receipt_sha256,
            validated.source_archive_sha256,
            environment_json,
            module,
            cast(str, entrypoint),
            *tuple(worker_arguments),
        )
        try:
            completed = subprocess.run(
                command,
                cwd=temporary,
                env=environment,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        finally:
            if runtime_batch_guard is None:
                _require(
                    _build_runtime_identity() == runtime,
                    "runtime identity drift during bound worker execution",
                )
            else:
                _require_active_runtime_batch_guard(
                    runtime_batch_guard,
                    directory=directory,
                    validated=validated,
                )
        return completed


__all__ = [
    "BoundCalibrationRuntimeBatch",
    "CALIBRATION_READINESS_RECEIPT_SCHEMA",
    "CERTIFICATION_SPECS",
    "CertificationSpec",
    "PreparedReadinessReceipt",
    "PublishedReadinessReceipt",
    "READINESS_ARCHIVE_SCHEMA",
    "READINESS_CERTIFICATION_EXECUTION_MANIFEST_SCHEMA",
    "READINESS_CERTIFICATION_NODE_MANIFEST_SCHEMA",
    "READINESS_CERTIFICATION_RUNTIME_CONTRACT_SCHEMA",
    "READINESS_CERTIFICATION_SCHEMA",
    "READINESS_ENVELOPE_SCHEMA",
    "READINESS_RUNTIME_SCHEMA",
    "READINESS_RUNTIME_EXECUTION_SCHEMA",
    "READINESS_RUNTIME_RECONSTRUCTION_SCHEMA",
    "READINESS_SOURCE_SCHEMA",
    "ReadinessDraft",
    "ReadinessError",
    "ReadinessValidation",
    "ValidatedReadinessBundle",
    "VerifiedCertificationBundle",
    "bound_calibration_runtime_batch",
    "build_runtime_execution_identity",
    "build_readiness_draft",
    "canonical_json_bytes",
    "canonical_sha256",
    "execute_bound_calibration_worker",
    "finalize_readiness_receipt",
    "load_validated_published_readiness_bundle",
    "publish_readiness_receipt",
    "require_validated_readiness_receipt",
    "require_current_full_runtime_identity",
    "run_readiness_certifications",
    "runtime_execution_identity_from_receipt",
    "validate_published_readiness_receipt",
    "validate_readiness_receipt",
]
