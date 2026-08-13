"""Source-only contract tests for the matched-v3 host executor v2 metadata.

The fixtures model canonical bytes and validation edges only.  They do not
invoke a workload, process API, container runtime, cgroup API, or benchmark.
"""

from __future__ import annotations

import ast
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_host_qualification_executor_v2 as host,
)

DESCRIPTOR_SCHEMA = "alberta.forager_matched_v3.host_qualification_executor_descriptor.v2"
REQUEST_SCHEMA = "alberta.forager_matched_v3.host_qualification_case_request.v2"
INTENT_SCHEMA = "alberta.forager_matched_v3.host_qualification_case_intent.v2"
READY_SCHEMA = "alberta.forager_matched_v3.in_container_qualification_driver_ready.v2"
ANCHOR_SCHEMA = "alberta.forager_matched_v3.host_observer_anchor.v2"
GO_SCHEMA = "alberta.forager_matched_v3.host_qualification_go_commitment.v2"
FRONTIER_SCHEMA = "alberta.forager_matched_v3.host_qualification_operational_frontier.v2"
INITIAL_SCHEMA = "alberta.forager_matched_v3.host_initial_cgroup_sample.v2"
PRECLEANUP_SCHEMA = "alberta.forager_matched_v3.host_precleanup_cgroup_sample.v2"
KILL_SCHEMA = "alberta.forager_matched_v3.host_cgroup_kill_receipt.v2"
EMPTY_SCHEMA = "alberta.forager_matched_v3.host_cgroup_empty_observation.v2"
ABSENCE_SCHEMA = "alberta.forager_matched_v3.host_container_absence_observation.v2"
POST_SCHEMA = "alberta.forager_matched_v3.host_post_container_remove_cgroup_sample.v2"
CLOSE_SCHEMA = "alberta.forager_matched_v3.host_cgroup_counter_fds_closed_receipt.v2"
OUTER_SCHEMA = "alberta.forager_matched_v3.host_outer_cgroup_absence_observation.v2"
EVENT_LOG_SCHEMA = "alberta.forager_matched_v3.host_cgroup_membership_event_log.v2"
PROOF_SCHEMA = "alberta.forager_matched_v3.host_cgroup_v2_boundary_proof.v1"
CLEANUP_SCHEMA = "alberta.forager_matched_v3.host_cleanup_reconciliation.v2"
TERMINAL_SCHEMA = "alberta.forager_matched_v3.host_terminal_metadata.v2"
LIFECYCLE_SCHEMA = "alberta.forager_matched_v3.host_qualification_lifecycle_record.v2"
SUCCESS_SCHEMA = "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v2"
FAILURE_SCHEMA = "alberta.forager_matched_v3.host_qualification_failure_receipt.v2"
HANDOFF_SCHEMA = "alberta.forager_matched_v3.host_qualification_observation_handoff.v2"

EXPECTED_DESCRIPTOR_BODY_SHA256 = (
    "e3aa649722e306a4b869db18854a9bc79508cf76f4693468e4abbd3347e5007f"
)
EXPECTED_DESCRIPTOR_FILE_SHA256 = (
    "24c205cc4e3d189b4580512c281a84764d81dce51f43e5d959b8650a516343a4"
)
ZERO_SHA256 = "0" * 64
ACKNOWLEDGEMENT = "AUTHORIZE ONE MATCHED-V3 HOST OCI QUALIFICATION CASE EXECUTION"

PHASES = (
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
ENDPOINTS = (
    "cpu.stat",
    "memory.current",
    "memory.peak",
    "memory.events",
    "pids.current",
    "pids.peak",
    "pids.events",
    "cgroup.events",
    "cgroup.stat",
    "cgroup.kill",
)
RECOVERY_NAMES = (
    "precleanup_cgroup_sample",
    "cgroup_kill",
    "cgroup_empty",
    "container_absence",
    "post_container_remove_cgroup_sample",
    "cgroup_counter_fds_closed",
    "outer_cgroup_absence",
    "final_cgroup_proof",
)
RECOVERY_DEPENDENCIES = {
    "precleanup_cgroup_sample": (),
    "cgroup_kill": ("precleanup_cgroup_sample",),
    "cgroup_empty": ("precleanup_cgroup_sample",),
    "container_absence": (),
    "post_container_remove_cgroup_sample": ("container_absence",),
    "cgroup_counter_fds_closed": ("post_container_remove_cgroup_sample",),
    "outer_cgroup_absence": ("cgroup_counter_fds_closed",),
    "final_cgroup_proof": RECOVERY_NAMES[:-1],
}
RECOVERY_SCHEMAS = dict(
    zip(
        RECOVERY_NAMES,
        (
            PRECLEANUP_SCHEMA,
            KILL_SCHEMA,
            EMPTY_SCHEMA,
            ABSENCE_SCHEMA,
            POST_SCHEMA,
            CLOSE_SCHEMA,
            OUTER_SCHEMA,
            PROOF_SCHEMA,
        ),
        strict=True,
    )
)
RESOURCE_FIELDS = (
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

FINAL_ALGORITHM_DESCRIPTOR = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)
FINAL_ALGORITHM_SOURCE = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)
FINAL_PUBLICATION_DESCRIPTOR = (
    "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
)
FINAL_PUBLICATION_SOURCE = (
    "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
)
FINAL_STORAGE_DESCRIPTOR = (
    "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
)
FINAL_STORAGE_SOURCE = (
    "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(schema: str, label: str) -> host.ArtifactIdentityV2:
    return host.ArtifactIdentityV2(schema, _hash(label + ":file"), _hash(label + ":body"))


def _producer(
    schema: str,
    label: str,
    *,
    descriptor: str | None = None,
    source: str | None = None,
) -> host.ProducerIdentityV2:
    return host.ProducerIdentityV2(
        schema,
        _hash(label + ":descriptor") if descriptor is None else descriptor,
        _hash(label + ":source") if source is None else source,
    )


def _identity(
    value: Any,
    schema: str,
    body_builder: Any,
    file_builder: Any,
) -> host.ArtifactIdentityV2:
    body = body_builder(value)
    file = file_builder(value)
    return host.ArtifactIdentityV2(
        schema,
        hashlib.sha256(file).hexdigest(),
        hashlib.sha256(body).hexdigest(),
    )


def _descriptor_hashes() -> tuple[str, str]:
    descriptor = host.HostExecutorDescriptorV2()
    body = host.canonical_host_executor_descriptor_v2_body_bytes(descriptor)
    file = host.canonical_host_executor_descriptor_v2_file_bytes(descriptor)
    return hashlib.sha256(body).hexdigest(), hashlib.sha256(file).hexdigest()


@contextmanager
def _observed_descriptor_pin() -> Any:
    _, observed_file = _descriptor_hashes()
    previous_pin = host.PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256
    setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", observed_file)
    try:
        yield observed_file
    finally:
        setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", previous_pin)


def _ceilings() -> tuple[tuple[str, int], ...]:
    values = {field: 1 for field in RESOURCE_FIELDS}
    values["max_environment_interactions"] = 499_712
    values["max_failure_count"] = 0
    return tuple((field, values[field]) for field in RESOURCE_FIELDS)


def _case(ordinal: int) -> tuple[str, str, str]:
    cases = {
        0: ("causal_e025_q050", "local"),
        1: ("causal_e025_q075", "local"),
        14: ("external_dqn_plain", "external"),
        23: ("adapted_full_rainbow", "adapter"),
        24: ("adapted_ppo_gru", "adapter"),
    }
    candidate, family = cases[ordinal]
    return candidate, family, f"qualification_{ordinal:02d}_{candidate}"


def _producer_schemas(ordinal: int) -> tuple[str, str, str]:
    _, family, _ = _case(ordinal)
    publisher = {
        "local": "alberta.forager_matched_v3.local_reward_publication_descriptor.v1",
        "external": "alberta.forager_matched_v3.external_reward_publication_descriptor.v1",
        "adapter": "alberta.forager_matched_v3.adapter_qualification_publication_descriptor.v1",
    }[family]
    atomic = (
        "alberta.forager_matched_v3.adapter_qualification_atomic_publication_descriptor.v1"
        if family == "adapter"
        else "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
    )
    driver = {
        0: "alberta.forager_matched_v3.local_runner_descriptor.v1",
        1: "alberta.forager_matched_v3.local_runner_descriptor.v1",
        14: "alberta.forager_matched_v3.external_execution_runner_descriptor.v1",
        23: "alberta.forager_matched_v3.full_rainbow_runner.v1",
        24: "alberta.forager_matched_v3.ppo_gru_runner.v1",
    }[ordinal]
    return publisher, atomic, driver


def _request(ordinal: int = 0) -> host.HostQualificationCaseRequestV2:
    candidate, family, case_id = _case(ordinal)
    publisher_schema, atomic_schema, driver_schema = _producer_schemas(ordinal)
    spine = _hash(f"case-spine:{ordinal}")
    _, descriptor_file = _descriptor_hashes()
    previous_pin = host.PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256
    setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", descriptor_file)
    try:
        return host.HostQualificationCaseRequestV2(
            case_spine_sha256=spine,
            case_ordinal=ordinal,
            candidate_id=candidate,
            candidate_family=family,
            qualification_case_id=case_id,
            container_name="alberta-mv3-" + spine,
            qualification_plan=_artifact(
                "alberta.forager_matched_v3.qualification_plan.v3", "plan"
            ),
            plan_issuance_receipt=_artifact(
                "alberta.forager_matched_v3.qualification_plan_issuance_receipt.v1",
                "issuance",
            ),
            case_execution_ticket=_artifact(
                "alberta.forager_matched_v3.qualification_case_execution_ticket.v1",
                "ticket",
            ),
            qualification_case_manifest=_artifact(
                "alberta.forager_matched_v3.qualification_case_manifest.v2", "manifest"
            ),
            observation_registry=_producer(
                "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v2",
                "observer",
            ),
            joint_source_closure=_artifact(
                "alberta.forager_matched_v3.qualification_joint_source_closure_candidate.v1",
                "closure",
            ),
            sealed_staging=_artifact(
                "alberta.forager_matched_v3.qualification_sealed_staging_candidate.v1",
                "staging",
            ),
            fresh_build=_artifact(
                "alberta.forager_matched_v3.fresh_cpu_oci_build_candidate.v2",
                "fresh-build",
            ),
            build_context_receipt=_artifact(
                "alberta.forager_matched_v3.cpu_oci_build_context_receipt.v1",
                "build-context",
            ),
            build_execution_receipt=_artifact(
                "alberta.forager_matched_v3.cpu_oci_build_execution_receipt.v1",
                "build-execution",
            ),
            build_publication_receipt=_artifact(
                "alberta.forager_matched_v3.cpu_oci_build_publication.v1",
                "build-publication",
            ),
            image_id="sha256:" + _hash("fresh-image"),
            runtime_qualification_receipt=_artifact(
                "alberta.forager_matched_v3.runtime_qualification_receipt.v1",
                "runtime",
            ),
            host_provisioning_receipt=_artifact(
                "alberta.forager_matched_v3.host_provisioning_receipt.v2",
                "provisioning",
            ),
            algorithmic_contract=_producer(
                "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1",
                "algorithm-contract",
                descriptor=FINAL_ALGORITHM_DESCRIPTOR,
                source=FINAL_ALGORITHM_SOURCE,
            ),
            algorithmic_measurement_intent=_artifact(
                "alberta.forager_matched_v3.algorithmic_resource_measurement_intent.v1",
                "algorithm-intent",
            ),
            publication_contract=_producer(
                "alberta.forager_matched_v3.qualification_publication_commitment_contract_descriptor.v1",
                "publication-contract",
                descriptor=FINAL_PUBLICATION_DESCRIPTOR,
                source=FINAL_PUBLICATION_SOURCE,
            ),
            storage_contract=_producer(
                "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1",
                "storage-contract",
                descriptor=FINAL_STORAGE_DESCRIPTOR,
                source=FINAL_STORAGE_SOURCE,
            ),
            storage_boundary_intent=_artifact(
                "alberta.forager_matched_v3.qualification_storage_boundary_intent.v1",
                "storage-intent",
            ),
            host_executor=_producer(
                DESCRIPTOR_SCHEMA,
                "host-v2",
                descriptor=descriptor_file,
                source=_hash("host-v2-source"),
            ),
            full_resource_merger=_producer(
                "alberta.forager_matched_v3.full_resource_merger_descriptor.v1",
                "merger",
            ),
            publisher=_producer(publisher_schema, "publisher"),
            native_atomic_producer=_producer(atomic_schema, "native"),
            in_container_driver=_producer(driver_schema, "driver"),
            resource_requirement_body_sha256=_hash("resource-requirement"),
            declared_ceilings=_ceilings(),
            horizon=499_712,
            attempt_ordinal=0,
            exact_acknowledgement=ACKNOWLEDGEMENT,
        )
    finally:
        setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", previous_pin)


def _intent(request: host.HostQualificationCaseRequestV2) -> host.HostQualificationCaseIntentV2:
    return host.HostQualificationCaseIntentV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        request=_identity(
            request,
            REQUEST_SCHEMA,
            host.canonical_host_case_request_v2_body_bytes,
            host.canonical_host_case_request_v2_file_bytes,
        ),
        case_execution_ticket=request.case_execution_ticket,
        image_id=request.image_id,
        algorithmic_measurement_intent=request.algorithmic_measurement_intent,
        storage_boundary_intent=request.storage_boundary_intent,
        container_name=request.container_name,
        retained_fd_policy_body_sha256=_hash("retained-policy"),
        cleanup_policy_body_sha256=_hash("cleanup-policy"),
        exact_acknowledgement_sha256=hashlib.sha256(ACKNOWLEDGEMENT.encode()).hexdigest(),
        intent_committed=True,
        same_case_retry_permitted=False,
    )


def _case_identity(request: host.HostQualificationCaseRequestV2) -> host.HostCgroupCaseIdentityV2:
    spine = request.case_spine_sha256
    return host.HostCgroupCaseIdentityV2(
        case_spine_sha256=spine,
        route_kind="cgroupfs_qualified_host",
        delegate_root_path="/sys/fs/cgroup/alberta-qualified-host",
        delegate_cgroup_device=101,
        delegate_cgroup_inode=1_001,
        case_cgroup_path=f"/sys/fs/cgroup/alberta-qualified-host/case-{spine}",
        case_cgroup_device=101,
        case_cgroup_inode=1_002,
        docker_cgroup_parent=f"/alberta-qualified-host/case-{spine}",
        enabled_controllers=("cpu", "memory", "pids"),
        subtree_control=("cpu", "memory", "pids"),
        max_depth=1,
        max_descendants=1,
    )


def _fds() -> tuple[host.RetainedCgroupCounterFdV2, ...]:
    return tuple(
        host.RetainedCgroupCounterFdV2(
            endpoint_name=name,
            endpoint_device=101,
            endpoint_inode=2_000 + index,
            open_monotonic_ns=10 + index,
            open_flags=(
                "O_CLOEXEC",
                "O_NOFOLLOW",
                "O_WRONLY" if name == "cgroup.kill" else "O_RDONLY",
            ),
            reset_performed=False,
            reopened=False,
            retained_through_post_container_remove_sample=True,
        )
        for index, name in enumerate(ENDPOINTS)
    )


def _facts(
    *,
    monotonic_ns: int,
    cgroup_identity_sha256: str,
    retained_fd_set_sha256: str,
    cpu_usage_usec: int,
    memory_current_bytes: int,
    memory_peak_bytes: int,
    pids_current: int,
    pids_peak: int,
    populated: bool,
    nr_descendants: int,
) -> host.CgroupSampleFactsV2:
    return host.CgroupSampleFactsV2(
        monotonic_ns=monotonic_ns,
        cgroup_identity_sha256=cgroup_identity_sha256,
        retained_fd_set_sha256=retained_fd_set_sha256,
        cpu_usage_usec=cpu_usage_usec,
        memory_current_bytes=memory_current_bytes,
        memory_peak_bytes=memory_peak_bytes,
        memory_oom_kill_count=0,
        pids_current=pids_current,
        pids_peak=pids_peak,
        pids_max_event_count=0,
        populated=populated,
        nr_descendants=nr_descendants,
        nr_dying_descendants=0,
    )


def _operational_prefix(
    request: host.HostQualificationCaseRequestV2,
) -> tuple[
    host.HostQualificationCaseIntentV2,
    host.HostInitialCgroupSampleV2,
    host.HostReadyV2,
    host.HostObserverAnchorV2,
    host.HostGoCommitmentV2,
]:
    intent = _intent(request)
    intent_id = _identity(
        intent,
        INTENT_SCHEMA,
        host.canonical_host_case_intent_v2_body_bytes,
        host.canonical_host_case_intent_v2_file_bytes,
    )
    case_identity = _case_identity(request)
    fds = _fds()
    fd_set = host.retained_cgroup_fd_inventory_sha256_v2(fds)
    cgroup_identity = host.cgroup_case_identity_sha256_v2(case_identity)
    initial = host.HostInitialCgroupSampleV2(
        case_spine_sha256=request.case_spine_sha256,
        intent=intent_id,
        cgroup_case_identity=case_identity,
        counter_fds=fds,
        facts=_facts(
            monotonic_ns=100,
            cgroup_identity_sha256=cgroup_identity,
            retained_fd_set_sha256=fd_set,
            cpu_usage_usec=0,
            memory_current_bytes=0,
            memory_peak_bytes=0,
            pids_current=0,
            pids_peak=0,
            populated=False,
            nr_descendants=0,
        ),
    )
    initial_id = _identity(
        initial,
        INITIAL_SCHEMA,
        host.canonical_host_initial_cgroup_sample_v2_body_bytes,
        host.canonical_host_initial_cgroup_sample_v2_file_bytes,
    )
    container_id = _hash("actual-container-id")
    ready = host.HostReadyV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        intent=intent_id,
        algorithmic_measurement_intent=request.algorithmic_measurement_intent,
        storage_boundary_intent=request.storage_boundary_intent,
        initial_cgroup_sample=initial_id,
        retained_fd_set_sha256=fd_set,
        cgroup_identity_sha256=cgroup_identity,
        container_identity_sha256=host.container_runtime_identity_sha256_v2(
            request.case_spine_sha256,
            request.container_name,
            container_id,
        ),
        container_id=container_id,
        container_name=request.container_name,
        container_cgroup_path=f"{case_identity.case_cgroup_path}/{container_id}",
        container_cgroup_device=101,
        container_cgroup_inode=3_000,
        host_pid=1_234,
        host_process_start_time_ticks=4_567,
        inner_pid=1,
        ready_monotonic_ns=200,
        candidate_code_loaded=False,
        go_committed=False,
    )
    ready_id = _identity(
        ready,
        READY_SCHEMA,
        host.canonical_host_ready_v2_body_bytes,
        host.canonical_host_ready_v2_file_bytes,
    )
    anchor = host.HostObserverAnchorV2(
        case_spine_sha256=request.case_spine_sha256,
        ready=ready_id,
        initial_cgroup_sample=initial_id,
        retained_fd_set_sha256=fd_set,
        cgroup_identity_sha256=cgroup_identity,
        observer_descriptor_sha256=request.observation_registry.descriptor_sha256,
        observer_source_sha256=request.observation_registry.source_sha256,
        observer_started_monotonic_ns=300,
        observation_loss_detected=False,
    )
    anchor_id = _identity(
        anchor,
        ANCHOR_SCHEMA,
        host.canonical_host_observer_anchor_v2_body_bytes,
        host.canonical_host_observer_anchor_v2_file_bytes,
    )
    go = host.HostGoCommitmentV2(
        case_spine_sha256=request.case_spine_sha256,
        ready=ready_id,
        observer_anchor=anchor_id,
        retained_fd_set_sha256=fd_set,
        cgroup_identity_sha256=cgroup_identity,
        go_payload_sha256=_hash("go-payload"),
        go_committed_monotonic_ns=400,
        go_commit_count=1,
        one_way=True,
        same_case_retry_permitted=False,
    )
    return intent, initial, ready, anchor, go


def _frontier(
    spine: str,
    *,
    failure_index: int | None = None,
    effect: str | None = None,
) -> host.HostOperationalFrontierV2:
    completed = PHASES if failure_index is None else PHASES[:failure_index]
    failure_phase = None if failure_index is None else PHASES[failure_index]

    def state(phase: str) -> str:
        if phase in completed:
            return "committed"
        if phase == failure_phase and effect == "commit_uncertain":
            return "commit_uncertain"
        return "not_started"

    def count(phase: str) -> tuple[str, int | None]:
        value = state(phase)
        if value == "committed":
            return "exact", 1
        if value == "commit_uncertain":
            return "uncertain", None
        return "exact", 0

    create_count_state, create_count = count("container_created")
    start_count_state, start_count = count("container_started")
    workload_count_state, workload_count = count("workload_started")
    exit_count_state, exit_count = count("workload_exited")
    return host.HostOperationalFrontierV2(
        case_spine_sha256=spine,
        completed_phases=completed,
        failure_phase=failure_phase,
        failure_effect_state=effect,
        container_create_state=state("container_created"),
        container_start_state=state("container_started"),
        workload_start_state=state("workload_started"),
        workload_exit_state=state("workload_exited"),
        container_create_count_state=create_count_state,
        container_create_count=create_count,
        container_start_count_state=start_count_state,
        container_start_count=start_count,
        workload_start_count_state=workload_count_state,
        workload_start_count=workload_count,
        workload_exit_count_state=exit_count_state,
        workload_exit_count=exit_count,
        attempt_count_state=workload_count_state,
        attempt_count=workload_count,
        failure_count=0 if failure_index is None else 1,
        case_consumed=True,
        same_case_retry_permitted=False,
    )


def _body_hash(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _recovery_artifacts(
    request: host.HostQualificationCaseRequestV2,
    frontier: host.HostOperationalFrontierV2,
    *,
    initial: host.HostInitialCgroupSampleV2 | None,
    ready: host.HostReadyV2 | None,
    kill_committed: bool = True,
    pre_fact_overrides: dict[str, Any] | None = None,
    post_fact_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontier_id = _identity(
        frontier,
        FRONTIER_SCHEMA,
        host.canonical_host_operational_frontier_v2_body_bytes,
        host.canonical_host_operational_frontier_v2_file_bytes,
    )
    case_identity = _case_identity(request)
    cgroup_identity = host.cgroup_case_identity_sha256_v2(case_identity)
    fd_set = (
        initial.facts.retained_fd_set_sha256
        if initial is not None
        else host.retained_cgroup_fd_inventory_sha256_v2(_fds())
    )
    cgroup_may_exist = (
        "fresh_cgroup_created" in frontier.completed_phases
        or (
            frontier.failure_phase == "fresh_cgroup_created"
            and frontier.failure_effect_state == "commit_uncertain"
        )
    )
    empty: host.HostCgroupEmptyObservationV2 | None = None
    pre: host.HostPrecleanupCgroupSampleV2 | None = None
    kill: host.HostCgroupKillReceiptV2 | None = None
    post: host.HostPostContainerRemoveCgroupSampleV2 | None = None
    close: host.HostCgroupCounterFdsClosedReceiptV2 | None = None
    outer: host.HostOuterCgroupAbsenceObservationV2 | None = None
    if cgroup_may_exist:
        pre_facts = _facts(
            monotonic_ns=500,
            cgroup_identity_sha256=cgroup_identity,
            retained_fd_set_sha256=fd_set,
            cpu_usage_usec=5,
            memory_current_bytes=4,
            memory_peak_bytes=12,
            pids_current=1,
            pids_peak=3,
            populated=True,
            nr_descendants=1,
        )
        if pre_fact_overrides is not None:
            pre_facts = replace(pre_facts, **pre_fact_overrides)
        pre = host.HostPrecleanupCgroupSampleV2(
            case_spine_sha256=request.case_spine_sha256,
            operational_frontier=frontier_id,
            facts=pre_facts,
        )
        pre_id = _identity(
            pre,
            PRECLEANUP_SCHEMA,
            host.canonical_host_precleanup_cgroup_sample_v2_body_bytes,
            host.canonical_host_precleanup_cgroup_sample_v2_file_bytes,
        )
        if kill_committed:
            kill = host.HostCgroupKillReceiptV2(
                case_spine_sha256=request.case_spine_sha256,
                precleanup_sample=pre_id,
                retained_fd_set_sha256=fd_set,
                cgroup_identity_sha256=cgroup_identity,
                kill_monotonic_ns=600,
                cgroup_kill_value=1,
                entire_subtree_targeted=True,
            )
        empty = host.HostCgroupEmptyObservationV2(
            case_spine_sha256=request.case_spine_sha256,
            precleanup_sample=pre_id,
            cgroup_kill_receipt=(
                None
                if kill is None
                else _identity(
                    kill,
                    KILL_SCHEMA,
                    host.canonical_host_cgroup_kill_receipt_v2_body_bytes,
                    host.canonical_host_cgroup_kill_receipt_v2_file_bytes,
                )
            ),
            retained_fd_set_sha256=fd_set,
            cgroup_identity_sha256=cgroup_identity,
            observed_monotonic_ns=700,
            populated=False,
            pids_current=0,
            recursive_process_count=0,
        )
    if frontier.container_create_state == "committed":
        actual_id = _hash("actual-container-id")
        actual_identity = host.container_runtime_identity_sha256_v2(
            request.case_spine_sha256,
            request.container_name,
            actual_id,
        )
        resolution = "created_removed"
        remove_count = 1
    elif frontier.container_create_state == "commit_uncertain":
        actual_id = None
        actual_identity = None
        resolution = "create_uncertain_resolved_absent"
        remove_count = 0
    else:
        actual_id = None
        actual_identity = None
        resolution = "never_created"
        remove_count = 0
    absence = host.HostContainerAbsenceObservationV2(
        case_spine_sha256=request.case_spine_sha256,
        operational_frontier=frontier_id,
        cgroup_empty_observation=(
            None
            if empty is None
            else _identity(
                empty,
                EMPTY_SCHEMA,
                host.canonical_host_cgroup_empty_observation_v2_body_bytes,
                host.canonical_host_cgroup_empty_observation_v2_file_bytes,
            )
        ),
        container_name=request.container_name,
        container_lookup_identity_sha256=host.container_lookup_identity_sha256_v2(
            request.case_spine_sha256,
            request.container_name,
        ),
        resolution_state=resolution,
        actual_runtime_container_identity_sha256=actual_identity,
        actual_container_id=actual_id,
        removal_monotonic_ns=800,
        container_remove_count=remove_count,
        container_absent=True,
    )
    if cgroup_may_exist:
        absence_id = _identity(
            absence,
            ABSENCE_SCHEMA,
            host.canonical_host_container_absence_observation_v2_body_bytes,
            host.canonical_host_container_absence_observation_v2_file_bytes,
        )
        post_facts = _facts(
            monotonic_ns=900,
            cgroup_identity_sha256=cgroup_identity,
            retained_fd_set_sha256=fd_set,
            cpu_usage_usec=7,
            memory_current_bytes=0,
            memory_peak_bytes=12,
            pids_current=0,
            pids_peak=3,
            populated=False,
            nr_descendants=0,
        )
        if post_fact_overrides is not None:
            post_facts = replace(post_facts, **post_fact_overrides)
        post = host.HostPostContainerRemoveCgroupSampleV2(
            case_spine_sha256=request.case_spine_sha256,
            container_absence_observation=absence_id,
            container_name=request.container_name,
            container_lookup_identity_sha256=absence.container_lookup_identity_sha256,
            actual_runtime_container_identity_sha256=actual_identity,
            actual_container_id=actual_id,
            facts=post_facts,
            retained_fds_still_open=True,
        )
        post_id = _identity(
            post,
            POST_SCHEMA,
            host.canonical_host_post_container_remove_cgroup_sample_v2_body_bytes,
            host.canonical_host_post_container_remove_cgroup_sample_v2_file_bytes,
        )
        close = host.HostCgroupCounterFdsClosedReceiptV2(
            case_spine_sha256=request.case_spine_sha256,
            post_container_remove_sample=post_id,
            retained_fd_set_sha256=post.facts.retained_fd_set_sha256,
            cgroup_identity_sha256=post.facts.cgroup_identity_sha256,
            container_name=request.container_name,
            container_lookup_identity_sha256=absence.container_lookup_identity_sha256,
            actual_runtime_container_identity_sha256=actual_identity,
            actual_container_id=actual_id,
            closed_endpoint_names=ENDPOINTS,
            close_monotonic_ns=1_000,
            all_fds_closed=True,
            reopen_permitted=False,
        )
        close_id = _identity(
            close,
            CLOSE_SCHEMA,
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes,
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes,
        )
        outer = host.HostOuterCgroupAbsenceObservationV2(
            case_spine_sha256=request.case_spine_sha256,
            cgroup_counter_fds_closed_receipt=close_id,
            cgroup_case_identity=case_identity,
            cgroup_identity_sha256=cgroup_identity,
            case_cgroup_path=case_identity.case_cgroup_path,
            removal_monotonic_ns=1_100,
            outer_cgroup_remove_count=1,
            outer_cgroup_absent=True,
        )
    if ready is not None and actual_identity != ready.container_identity_sha256:
        raise AssertionError("fixture actual runtime identity drifted from READY")
    return {
        "cgroup_may_exist": cgroup_may_exist,
        "pre": pre,
        "kill": kill,
        "empty": empty,
        "absence": absence,
        "post": post,
        "close": close,
        "outer": outer,
    }


def _proof_and_event_log(
    request: host.HostQualificationCaseRequestV2,
    initial: host.HostInitialCgroupSampleV2,
    ready: host.HostReadyV2,
    anchor: host.HostObserverAnchorV2,
    frontier: host.HostOperationalFrontierV2,
    recovery: dict[str, Any],
) -> tuple[host.HostCgroupMembershipEventLogV2, host.HostCgroupBoundaryProofV2]:
    pre = recovery["pre"]
    kill = recovery["kill"]
    empty = recovery["empty"]
    absence = recovery["absence"]
    post = recovery["post"]
    close = recovery["close"]
    outer = recovery["outer"]
    if any(value is None for value in (pre, kill, empty, post, close, outer)):
        raise AssertionError("proof fixture requires every recovery artifact")
    initial_id = _identity(
        initial,
        INITIAL_SCHEMA,
        host.canonical_host_initial_cgroup_sample_v2_body_bytes,
        host.canonical_host_initial_cgroup_sample_v2_file_bytes,
    )
    anchor_id = _identity(
        anchor,
        ANCHOR_SCHEMA,
        host.canonical_host_observer_anchor_v2_body_bytes,
        host.canonical_host_observer_anchor_v2_file_bytes,
    )
    post_id = _identity(
        post,
        POST_SCHEMA,
        host.canonical_host_post_container_remove_cgroup_sample_v2_body_bytes,
        host.canonical_host_post_container_remove_cgroup_sample_v2_file_bytes,
    )
    events = (
        host.HostCgroupMembershipEventV2(
            0, "initial_empty_boundary", 100, 0, 0, 0
        ),
        host.HostCgroupMembershipEventV2(
            1, "anchored_container_membership", 300, 1, 0, 0
        ),
        host.HostCgroupMembershipEventV2(
            2, "post_remove_empty_boundary", 900, 0, 0, 0
        ),
    )
    inventory = _body_hash({"events": [event.to_dict() for event in events]})
    event_log = host.HostCgroupMembershipEventLogV2(
        case_spine_sha256=request.case_spine_sha256,
        cgroup_case_identity_sha256=initial.facts.cgroup_identity_sha256,
        container_name=request.container_name,
        container_lookup_identity_sha256=absence.container_lookup_identity_sha256,
        actual_runtime_container_identity_sha256=ready.container_identity_sha256,
        actual_container_id=ready.container_id,
        container_cgroup_path=ready.container_cgroup_path,
        container_cgroup_device=ready.container_cgroup_device,
        container_cgroup_inode=ready.container_cgroup_inode,
        host_pid=ready.host_pid,
        host_process_start_time_ticks=ready.host_process_start_time_ticks,
        observer_descriptor_sha256=anchor.observer_descriptor_sha256,
        observer_source_sha256=anchor.observer_source_sha256,
        host_provisioning_receipt=request.host_provisioning_receipt,
        observer_anchor=anchor_id,
        initial_sample=initial_id,
        post_container_remove_sample=post_id,
        events=events,
        event_inventory_sha256=inventory,
        continuous_all_descendant_membership_proven=False,
        provisioning_receipt_semantics_validated=False,
        provisioning_receipt_producer_authenticated=False,
        production_containment_eligible=False,
    )
    event_log_id = _identity(
        event_log,
        EVENT_LOG_SCHEMA,
        host.canonical_host_cgroup_membership_event_log_v2_body_bytes,
        host.canonical_host_cgroup_membership_event_log_v2_file_bytes,
    )
    evidence = host.HostObserverTerminalEvidenceV2(
        case_spine_sha256=request.case_spine_sha256,
        observer_anchor=anchor_id,
        host_provisioning_receipt=request.host_provisioning_receipt,
        initial_sample=initial_id,
        post_container_remove_sample=post_id,
        membership_event_log=event_log_id,
        membership_event_inventory_sha256=inventory,
        continuous_all_descendant_membership_proven=False,
        provisioning_validated=False,
        provisioning_producer_authenticated=False,
        production_containment_eligible=False,
    )
    proof = host.HostCgroupBoundaryProofV2(
        case_spine_sha256=request.case_spine_sha256,
        operational_frontier=_identity(
            frontier,
            FRONTIER_SCHEMA,
            host.canonical_host_operational_frontier_v2_body_bytes,
            host.canonical_host_operational_frontier_v2_file_bytes,
        ),
        initial_sample=initial_id,
        precleanup_sample=_identity(
            pre,
            PRECLEANUP_SCHEMA,
            host.canonical_host_precleanup_cgroup_sample_v2_body_bytes,
            host.canonical_host_precleanup_cgroup_sample_v2_file_bytes,
        ),
        cgroup_kill_receipt=_identity(
            kill,
            KILL_SCHEMA,
            host.canonical_host_cgroup_kill_receipt_v2_body_bytes,
            host.canonical_host_cgroup_kill_receipt_v2_file_bytes,
        ),
        cgroup_empty_observation=_identity(
            empty,
            EMPTY_SCHEMA,
            host.canonical_host_cgroup_empty_observation_v2_body_bytes,
            host.canonical_host_cgroup_empty_observation_v2_file_bytes,
        ),
        container_absence_observation=_identity(
            absence,
            ABSENCE_SCHEMA,
            host.canonical_host_container_absence_observation_v2_body_bytes,
            host.canonical_host_container_absence_observation_v2_file_bytes,
        ),
        post_container_remove_sample=post_id,
        cgroup_counter_fds_closed_receipt=_identity(
            close,
            CLOSE_SCHEMA,
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes,
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes,
        ),
        outer_cgroup_absence_observation=_identity(
            outer,
            OUTER_SCHEMA,
            host.canonical_host_outer_cgroup_absence_observation_v2_body_bytes,
            host.canonical_host_outer_cgroup_absence_observation_v2_file_bytes,
        ),
        cgroup_case_identity=initial.cgroup_case_identity,
        retained_fd_set_sha256=initial.facts.retained_fd_set_sha256,
        cgroup_identity_sha256=initial.facts.cgroup_identity_sha256,
        container_name=request.container_name,
        container_lookup_identity_sha256=absence.container_lookup_identity_sha256,
        actual_runtime_container_identity_sha256=ready.container_identity_sha256,
        actual_container_id=ready.container_id,
        observer_terminal_evidence=evidence,
        observer_terminal_evidence_sha256=host.observer_terminal_evidence_sha256_v2(
            evidence
        ),
        resources=host.HostRawResourceMeasurementsV2(
            memory_peak_bytes=post.facts.memory_peak_bytes,
            memory_peak_semantics="conservative_observed_upper_bound",
            memory_oom_kill_count=post.facts.memory_oom_kill_count,
            memory_oom_kill_count_semantics="exact_observation",
            initial_cpu_usage_usec=initial.facts.cpu_usage_usec,
            post_remove_cpu_usage_usec=post.facts.cpu_usage_usec,
            cpu_delta_usec=(
                post.facts.cpu_usage_usec - initial.facts.cpu_usage_usec
            ),
            cpu_time_ns=(
                post.facts.cpu_usage_usec - initial.facts.cpu_usage_usec
            )
            * 1_000,
            cpu_time_semantics="exact_observation",
            initial_monotonic_ns=initial.facts.monotonic_ns,
            post_remove_monotonic_ns=post.facts.monotonic_ns,
            wall_time_ns=(post.facts.monotonic_ns - initial.facts.monotonic_ns),
            wall_time_semantics="exact_observation",
            pids_peak=post.facts.pids_peak,
            pids_peak_semantics="conservative_observed_upper_bound",
            pids_max_event_count=post.facts.pids_max_event_count,
            pids_max_event_count_semantics="exact_observation",
            attempt_count=1,
            attempt_count_semantics="exact_observation",
            failure_count=frontier.failure_count,
            failure_count_semantics="exact_observation",
            structural_measurements_only=True,
            production_qualified=False,
        ),
        continuous_all_descendant_membership_proven=False,
        provisioning_validated=False,
        provisioning_producer_authenticated=False,
        production_containment_eligible=False,
    )
    return event_log, proof


def _artifact_identity_for_recovery(name: str, value: Any) -> host.ArtifactIdentityV2:
    builders = {
        "precleanup_cgroup_sample": (
            host.canonical_host_precleanup_cgroup_sample_v2_body_bytes,
            host.canonical_host_precleanup_cgroup_sample_v2_file_bytes,
        ),
        "cgroup_kill": (
            host.canonical_host_cgroup_kill_receipt_v2_body_bytes,
            host.canonical_host_cgroup_kill_receipt_v2_file_bytes,
        ),
        "cgroup_empty": (
            host.canonical_host_cgroup_empty_observation_v2_body_bytes,
            host.canonical_host_cgroup_empty_observation_v2_file_bytes,
        ),
        "container_absence": (
            host.canonical_host_container_absence_observation_v2_body_bytes,
            host.canonical_host_container_absence_observation_v2_file_bytes,
        ),
        "post_container_remove_cgroup_sample": (
            host.canonical_host_post_container_remove_cgroup_sample_v2_body_bytes,
            host.canonical_host_post_container_remove_cgroup_sample_v2_file_bytes,
        ),
        "cgroup_counter_fds_closed": (
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes,
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes,
        ),
        "outer_cgroup_absence": (
            host.canonical_host_outer_cgroup_absence_observation_v2_body_bytes,
            host.canonical_host_outer_cgroup_absence_observation_v2_file_bytes,
        ),
        "final_cgroup_proof": (
            host.canonical_host_cgroup_boundary_proof_v2_body_bytes,
            host.canonical_host_cgroup_boundary_proof_v2_file_bytes,
        ),
    }
    body_builder, file_builder = builders[name]
    return _identity(value, RECOVERY_SCHEMAS[name], body_builder, file_builder)


def _cleanup(
    frontier: host.HostOperationalFrontierV2,
    recovery: dict[str, Any],
    *,
    proof: host.HostCgroupBoundaryProofV2 | None = None,
    kill_uncertain: bool = False,
) -> host.HostCleanupReconciliationV2:
    values = {
        "precleanup_cgroup_sample": recovery["pre"],
        "cgroup_kill": recovery["kill"],
        "cgroup_empty": recovery["empty"],
        "container_absence": recovery["absence"],
        "post_container_remove_cgroup_sample": recovery["post"],
        "cgroup_counter_fds_closed": recovery["close"],
        "outer_cgroup_absence": recovery["outer"],
        "final_cgroup_proof": proof,
    }
    nodes: list[host.RecoveryNodeV2] = []
    for name in RECOVERY_NAMES:
        value = values[name]
        if not recovery["cgroup_may_exist"] and name != "container_absence":
            state = "not_applicable"
        elif name == "cgroup_kill" and kill_uncertain:
            state = "commit_uncertain"
        elif name == "final_cgroup_proof" and proof is None:
            state = "failed_before_commit"
        elif value is not None:
            state = "committed"
        else:
            raise AssertionError(f"fixture lacks recovery artifact {name}")
        nodes.append(
            host.RecoveryNodeV2(
                node_name=name,
                state=state,
                artifact=(
                    _artifact_identity_for_recovery(name, value)
                    if state == "committed"
                    else None
                ),
                dependencies=RECOVERY_DEPENDENCIES[name],
                uncertainty_detail_sha256=(
                    _hash("recovery-detail:" + name)
                    if state in {"commit_uncertain", "failed_before_commit"}
                    else None
                ),
            )
        )
    unresolved = tuple(
        node.node_name
        for node in nodes
        if node.state in {"commit_uncertain", "failed_before_commit"}
    )
    cleanup_proven = all(
        node.state == "committed"
        for node in nodes
        if recovery["cgroup_may_exist"] or node.node_name == "container_absence"
    )
    return host.HostCleanupReconciliationV2(
        case_spine_sha256=frontier.case_spine_sha256,
        operational_frontier=_identity(
            frontier,
            FRONTIER_SCHEMA,
            host.canonical_host_operational_frontier_v2_body_bytes,
            host.canonical_host_operational_frontier_v2_file_bytes,
        ),
        cgroup_may_exist=recovery["cgroup_may_exist"],
        recovery_nodes=tuple(nodes),
        cleanup_proven=cleanup_proven,
        unresolved_recovery_nodes=unresolved,
        recovery_complete=True,
        terminalization_permitted=True,
        workload_resume_permitted=False,
        same_case_retry_permitted=False,
    )


def _native_publication(
    request: host.HostQualificationCaseRequestV2,
    frontier: host.HostOperationalFrontierV2,
    *,
    terminal_failure: bool,
) -> host.HostNativePublicationProjectionV2:
    if "native_publication_committed" in frontier.completed_phases:
        state = "committed"
        count_state = "exact"
        count = 1
    elif (
        frontier.failure_phase == "native_publication_committed"
        and frontier.failure_effect_state == "commit_uncertain"
    ):
        state = "commit_uncertain"
        count_state = "uncertain"
        count = None
    else:
        state = "not_started"
        count_state = "exact"
        count = 0
    if state == "not_started":
        address = None
        key = None
        reference = None
    else:
        address = _hash("publication-address")
        key = host.publication_reconciliation_key_sha256_v2(
            request.case_spine_sha256,
            address,
            request.native_atomic_producer,
        )
        reference = _artifact(
            "alberta.forager_matched_v3.qualification_publication_reconciliation_reference.v1",
            "publication-reconciliation",
        )
    native_receipt = None
    if state == "committed" and request.candidate_family != "local":
        schema = {
            "external": "alberta.forager_matched_v3.external_atomic_publication_receipt.v1",
            "adapter": "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1",
        }[request.candidate_family]
        native_receipt = _artifact(schema, "native-publication")
    wrapper_committed = "publication_commitment_wrapper_committed" in frontier.completed_phases
    failure_projection = (
        _artifact(
            "alberta.forager_matched_v3.qualification_failure_publication_projection.v2",
            "failure-publication-projection",
        )
        if terminal_failure and wrapper_committed
        else None
    )
    return host.HostNativePublicationProjectionV2(
        case_spine_sha256=request.case_spine_sha256,
        native_publication_state=state,
        native_atomic_producer=request.native_atomic_producer,
        native_publication_receipt=native_receipt,
        expected_publication_address_sha256=address,
        publication_reconciliation_key_sha256=key,
        publication_reconciliation_reference=reference,
        failure_publication_projection=failure_projection,
        native_publication_commit_count_state=count_state,
        native_publication_commit_count=count,
    )


def _terminal_outputs(
    request: host.HostQualificationCaseRequestV2,
    frontier: host.HostOperationalFrontierV2,
) -> dict[str, host.ArtifactIdentityV2 | None]:
    completed = set(frontier.completed_phases)
    algorithm_schema = {
        "local": "alberta.forager_matched_v3.local_algorithmic_resource_receipt.v1",
        "external": "alberta.forager_matched_v3.external_algorithmic_resource_receipt.v1",
        "adapter": "alberta.forager_matched_v3.adapter_algorithmic_resource_receipt.v1",
    }[request.candidate_family]
    return {
        "driver_terminal": (
            _artifact(
                "alberta.forager_matched_v3.in_container_qualification_driver_terminal.v2",
                "driver-terminal",
            )
            if "workload_exited" in completed
            else None
        ),
        "algorithmic_resource_receipt": (
            _artifact(algorithm_schema, "algorithmic-receipt")
            if "algorithmic_resource_receipt_committed" in completed
            else None
        ),
        "publication_commitment_wrapper": (
            _artifact(
                "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1",
                "publication-wrapper",
            )
            if "publication_commitment_wrapper_committed" in completed
            else None
        ),
        "publication_reload_validation": (
            _artifact(
                "alberta.forager_matched_v3.qualification_publication_reload_validation.v1",
                "publication-reload",
            )
            if "publication_reload_validated" in completed
            else None
        ),
        "storage_write_seal": (
            _artifact(
                "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1",
                "storage-seal",
            )
            if "storage_write_seal_committed" in completed
            else None
        ),
        "storage_boundary_receipt": (
            _artifact(
                "alberta.forager_matched_v3.qualification_storage_boundary_receipt.v1",
                "storage-receipt",
            )
            if "storage_boundary_receipt_committed" in completed
            else None
        ),
    }


def _terminal(
    request: host.HostQualificationCaseRequestV2,
    frontier: host.HostOperationalFrontierV2,
    cleanup: host.HostCleanupReconciliationV2,
) -> host.HostTerminalMetadataV2:
    structural_success = frontier.failure_phase is None and cleanup.cleanup_proven
    workload_exited = "workload_exited" in frontier.completed_phases
    return host.HostTerminalMetadataV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        record_kind="success" if structural_success else "terminal_failure",
        operational_frontier=_identity(
            frontier,
            FRONTIER_SCHEMA,
            host.canonical_host_operational_frontier_v2_body_bytes,
            host.canonical_host_operational_frontier_v2_file_bytes,
        ),
        cleanup_reconciliation=_identity(
            cleanup,
            CLEANUP_SCHEMA,
            host.canonical_host_cleanup_reconciliation_v2_body_bytes,
            host.canonical_host_cleanup_reconciliation_v2_file_bytes,
        ),
        **_terminal_outputs(request, frontier),
        returncode=0 if workload_exited else 1,
        timed_out=False,
        error_message_sha256=None if structural_success else _hash("terminal-error"),
        cleanup_proven=cleanup.cleanup_proven,
        case_consumed=True,
        same_case_retry_permitted=False,
    )


def _lifecycle(
    request: host.HostQualificationCaseRequestV2,
    frontier: host.HostOperationalFrontierV2,
    cleanup: host.HostCleanupReconciliationV2,
    terminal: host.HostTerminalMetadataV2,
) -> host.HostLifecycleRollupV2:
    return host.HostLifecycleRollupV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        record_kind=terminal.record_kind,
        operational_frontier=terminal.operational_frontier,
        cleanup_reconciliation=terminal.cleanup_reconciliation,
        terminal_metadata=_identity(
            terminal,
            TERMINAL_SCHEMA,
            host.canonical_host_terminal_metadata_v2_body_bytes,
            host.canonical_host_terminal_metadata_v2_file_bytes,
        ),
        native_publication=_native_publication(
            request,
            frontier,
            terminal_failure=terminal.record_kind == "terminal_failure",
        ),
        operational_success=frontier.failure_phase is None,
        cleanup_proven=cleanup.cleanup_proven,
        terminal_metadata_validated=True,
        container_create_state=frontier.container_create_state,
        container_start_state=frontier.container_start_state,
        workload_start_state=frontier.workload_start_state,
        workload_exit_state=frontier.workload_exit_state,
        attempt_count_state=frontier.attempt_count_state,
        attempt_count=frontier.attempt_count,
        operational_failure_count=frontier.failure_count,
        recovery_failure_count=sum(
            node.state == "failed_before_commit" for node in cleanup.recovery_nodes
        ),
        recovery_uncertainty_count=sum(
            node.state == "commit_uncertain" for node in cleanup.recovery_nodes
        ),
        terminal_failure_count=0 if terminal.record_kind == "success" else 1,
        structural_success_shape_only=terminal.record_kind == "success",
        production_execution_success=False,
        production_acceptance_eligible=False,
        evidence_eligible=False,
        case_consumed=True,
        same_case_retry_permitted=False,
    )


def _handoff(
    receipt: host.HostSuccessReceiptV2 | host.HostFailureReceiptV2,
    terminal: host.HostTerminalMetadataV2,
) -> host.HostObservationHandoffV2:
    if type(receipt) is host.HostSuccessReceiptV2:
        record_kind = "success"
        receipt_body = host.canonical_host_success_receipt_v2_body_bytes(receipt)
        receipt_file = host.canonical_host_success_receipt_v2_file_bytes(receipt)
    else:
        record_kind = "terminal_failure"
        receipt_body = host.canonical_host_failure_receipt_v2_body_bytes(receipt)
        receipt_file = host.canonical_host_failure_receipt_v2_file_bytes(receipt)
    terminal_body = host.canonical_host_terminal_metadata_v2_body_bytes(terminal)
    terminal_file = host.canonical_host_terminal_metadata_v2_file_bytes(terminal)
    return host.HostObservationHandoffV2(
        case_spine_sha256=receipt.case_spine_sha256,
        case_ordinal=receipt.case_ordinal,
        candidate_id=receipt.candidate_id,
        qualification_case_id=receipt.qualification_case_id,
        record_kind=record_kind,
        terminal_receipt_file_sha256=hashlib.sha256(receipt_file).hexdigest(),
        terminal_receipt_body_sha256=hashlib.sha256(receipt_body).hexdigest(),
        terminal_metadata_file_sha256=hashlib.sha256(terminal_file).hexdigest(),
        terminal_metadata_body_sha256=hashlib.sha256(terminal_body).hexdigest(),
        structural_success_shape_only=record_kind == "success",
        production_execution_success=False,
        production_acceptance_eligible=False,
        evidence_eligible=False,
    )


def _success_bundle() -> dict[str, Any]:
    request = _request()
    intent, initial, ready, anchor, go = _operational_prefix(request)
    frontier = _frontier(request.case_spine_sha256)
    recovery = _recovery_artifacts(
        request,
        frontier,
        initial=initial,
        ready=ready,
    )
    event_log, proof = _proof_and_event_log(
        request,
        initial,
        ready,
        anchor,
        frontier,
        recovery,
    )
    cleanup = _cleanup(frontier, recovery, proof=proof)
    terminal = _terminal(request, frontier, cleanup)
    lifecycle = _lifecycle(request, frontier, cleanup, terminal)
    receipt = host.HostSuccessReceiptV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        request=_identity(
            request,
            REQUEST_SCHEMA,
            host.canonical_host_case_request_v2_body_bytes,
            host.canonical_host_case_request_v2_file_bytes,
        ),
        intent=_identity(
            intent,
            INTENT_SCHEMA,
            host.canonical_host_case_intent_v2_body_bytes,
            host.canonical_host_case_intent_v2_file_bytes,
        ),
        ready=_identity(
            ready,
            READY_SCHEMA,
            host.canonical_host_ready_v2_body_bytes,
            host.canonical_host_ready_v2_file_bytes,
        ),
        observer_anchor=_identity(
            anchor,
            ANCHOR_SCHEMA,
            host.canonical_host_observer_anchor_v2_body_bytes,
            host.canonical_host_observer_anchor_v2_file_bytes,
        ),
        go_commitment=_identity(
            go,
            GO_SCHEMA,
            host.canonical_host_go_v2_body_bytes,
            host.canonical_host_go_v2_file_bytes,
        ),
        operational_frontier=terminal.operational_frontier,
        cleanup_reconciliation=terminal.cleanup_reconciliation,
        cgroup_proof=_artifact_identity_for_recovery("final_cgroup_proof", proof),
        terminal_metadata=lifecycle.terminal_metadata,
        lifecycle=_identity(
            lifecycle,
            LIFECYCLE_SCHEMA,
            host.canonical_host_lifecycle_v2_body_bytes,
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        container_create_count=1,
        container_start_count=1,
        go_commit_count=1,
        workload_start_count=1,
        workload_exit_count=1,
        attempt_count=1,
        failure_count=0,
        returncode=0,
        timed_out=False,
        cleanup_proven=True,
        structural_success_shape_only=True,
        production_execution_success=False,
        production_acceptance_eligible=False,
        evidence_eligible=False,
        case_consumed=True,
        same_case_retry_permitted=False,
    )
    handoff = _handoff(receipt, terminal)
    return {
        "request": request,
        "intent": intent,
        "initial": initial,
        "ready": ready,
        "anchor": anchor,
        "go": go,
        "frontier": frontier,
        "recovery": recovery,
        "event_log": event_log,
        "proof": proof,
        "cleanup": cleanup,
        "terminal": terminal,
        "lifecycle": lifecycle,
        "receipt": receipt,
        "handoff": handoff,
    }


def _failure_bundle(failure_index: int, effect: str) -> dict[str, Any]:
    request = _request()
    full_intent, full_initial, full_ready, full_anchor, full_go = _operational_prefix(request)
    frontier = _frontier(
        request.case_spine_sha256,
        failure_index=failure_index,
        effect=effect,
    )
    intent = full_intent if "intent_committed" in frontier.completed_phases else None
    initial = (
        full_initial
        if "initial_cgroup_sample_committed" in frontier.completed_phases
        else None
    )
    ready = full_ready if "driver_ready" in frontier.completed_phases else None
    anchor = full_anchor if "observer_anchored" in frontier.completed_phases else None
    go = full_go if "go_committed" in frontier.completed_phases else None
    recovery = _recovery_artifacts(
        request,
        frontier,
        initial=initial,
        ready=ready,
    )
    cleanup = _cleanup(frontier, recovery)
    terminal = _terminal(request, frontier, cleanup)
    lifecycle = _lifecycle(request, frontier, cleanup, terminal)
    classification = (
        "operational_commit_uncertain_ticket_quarantined_nonretryable"
        if effect == "commit_uncertain"
        else "operational_failed_before_commit_ticket_quarantined_nonretryable"
    )
    receipt = host.HostFailureReceiptV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        request=_identity(
            request,
            REQUEST_SCHEMA,
            host.canonical_host_case_request_v2_body_bytes,
            host.canonical_host_case_request_v2_file_bytes,
        ),
        intent=(
            None
            if intent is None
            else _identity(
                intent,
                INTENT_SCHEMA,
                host.canonical_host_case_intent_v2_body_bytes,
                host.canonical_host_case_intent_v2_file_bytes,
            )
        ),
        initial_sample=(
            None
            if initial is None
            else _identity(
                initial,
                INITIAL_SCHEMA,
                host.canonical_host_initial_cgroup_sample_v2_body_bytes,
                host.canonical_host_initial_cgroup_sample_v2_file_bytes,
            )
        ),
        ready=(
            None
            if ready is None
            else _identity(
                ready,
                READY_SCHEMA,
                host.canonical_host_ready_v2_body_bytes,
                host.canonical_host_ready_v2_file_bytes,
            )
        ),
        observer_anchor=(
            None
            if anchor is None
            else _identity(
                anchor,
                ANCHOR_SCHEMA,
                host.canonical_host_observer_anchor_v2_body_bytes,
                host.canonical_host_observer_anchor_v2_file_bytes,
            )
        ),
        go_commitment=(
            None
            if go is None
            else _identity(
                go,
                GO_SCHEMA,
                host.canonical_host_go_v2_body_bytes,
                host.canonical_host_go_v2_file_bytes,
            )
        ),
        operational_frontier=terminal.operational_frontier,
        cleanup_reconciliation=terminal.cleanup_reconciliation,
        terminal_metadata=lifecycle.terminal_metadata,
        lifecycle=_identity(
            lifecycle,
            LIFECYCLE_SCHEMA,
            host.canonical_host_lifecycle_v2_body_bytes,
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        native_publication=lifecycle.native_publication,
        failure_receipt_state="committed",
        classification=classification,
        exception_type="HostQualificationFailure",
        error_message_sha256=_hash("terminal-error"),
        operational_failure_phase=frontier.failure_phase,
        operational_failure_effect_state=frontier.failure_effect_state,
        unresolved_recovery_nodes=cleanup.unresolved_recovery_nodes,
        container_create_count_state=frontier.container_create_count_state,
        container_create_count=frontier.container_create_count,
        container_start_count_state=frontier.container_start_count_state,
        container_start_count=frontier.container_start_count,
        workload_start_count_state=frontier.workload_start_count_state,
        workload_start_count=frontier.workload_start_count,
        workload_exit_count_state=frontier.workload_exit_count_state,
        workload_exit_count=frontier.workload_exit_count,
        attempt_count_state=frontier.attempt_count_state,
        attempt_count=frontier.attempt_count,
        operational_failure_count=1,
        recovery_failure_count=sum(
            node.state == "failed_before_commit" for node in cleanup.recovery_nodes
        ),
        recovery_uncertainty_count=sum(
            node.state == "commit_uncertain" for node in cleanup.recovery_nodes
        ),
        terminal_failure_count=1,
        cleanup_proven=cleanup.cleanup_proven,
        ticket_quarantined=True,
        reconciliation_only=True,
        case_consumed=True,
        same_case_retry_permitted=False,
        clean_rejection_recorded=False,
    )
    return {
        "request": request,
        "intent": intent,
        "initial": initial,
        "ready": ready,
        "anchor": anchor,
        "go": go,
        "frontier": frontier,
        "recovery": recovery,
        "event_log": None,
        "proof": None,
        "cleanup": cleanup,
        "terminal": terminal,
        "lifecycle": lifecycle,
        "receipt": receipt,
        "handoff": _handoff(receipt, terminal),
    }


def _recovery_only_failure_bundle() -> dict[str, Any]:
    request = _request()
    intent, initial, ready, anchor, go = _operational_prefix(request)
    frontier = _frontier(request.case_spine_sha256)
    recovery = _recovery_artifacts(
        request,
        frontier,
        initial=initial,
        ready=ready,
        kill_committed=False,
    )
    cleanup = _cleanup(frontier, recovery, kill_uncertain=True)
    terminal = _terminal(request, frontier, cleanup)
    lifecycle = _lifecycle(request, frontier, cleanup, terminal)
    receipt = host.HostFailureReceiptV2(
        case_spine_sha256=request.case_spine_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        request=_identity(
            request,
            REQUEST_SCHEMA,
            host.canonical_host_case_request_v2_body_bytes,
            host.canonical_host_case_request_v2_file_bytes,
        ),
        intent=_identity(
            intent,
            INTENT_SCHEMA,
            host.canonical_host_case_intent_v2_body_bytes,
            host.canonical_host_case_intent_v2_file_bytes,
        ),
        initial_sample=_identity(
            initial,
            INITIAL_SCHEMA,
            host.canonical_host_initial_cgroup_sample_v2_body_bytes,
            host.canonical_host_initial_cgroup_sample_v2_file_bytes,
        ),
        ready=_identity(
            ready,
            READY_SCHEMA,
            host.canonical_host_ready_v2_body_bytes,
            host.canonical_host_ready_v2_file_bytes,
        ),
        observer_anchor=_identity(
            anchor,
            ANCHOR_SCHEMA,
            host.canonical_host_observer_anchor_v2_body_bytes,
            host.canonical_host_observer_anchor_v2_file_bytes,
        ),
        go_commitment=_identity(
            go,
            GO_SCHEMA,
            host.canonical_host_go_v2_body_bytes,
            host.canonical_host_go_v2_file_bytes,
        ),
        operational_frontier=terminal.operational_frontier,
        cleanup_reconciliation=terminal.cleanup_reconciliation,
        terminal_metadata=lifecycle.terminal_metadata,
        lifecycle=_identity(
            lifecycle,
            LIFECYCLE_SCHEMA,
            host.canonical_host_lifecycle_v2_body_bytes,
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        native_publication=lifecycle.native_publication,
        failure_receipt_state="committed",
        classification="recovery_failure_after_complete_operational_frontier_nonretryable",
        exception_type="HostRecoveryFailure",
        error_message_sha256=_hash("terminal-error"),
        operational_failure_phase=None,
        operational_failure_effect_state=None,
        unresolved_recovery_nodes=cleanup.unresolved_recovery_nodes,
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
        operational_failure_count=0,
        recovery_failure_count=1,
        recovery_uncertainty_count=1,
        terminal_failure_count=1,
        cleanup_proven=False,
        ticket_quarantined=True,
        reconciliation_only=True,
        case_consumed=True,
        same_case_retry_permitted=False,
        clean_rejection_recorded=False,
    )
    return {
        "request": request,
        "intent": intent,
        "initial": initial,
        "ready": ready,
        "anchor": anchor,
        "go": go,
        "frontier": frontier,
        "recovery": recovery,
        "event_log": None,
        "proof": None,
        "cleanup": cleanup,
        "terminal": terminal,
        "lifecycle": lifecycle,
        "receipt": receipt,
        "handoff": _handoff(receipt, terminal),
    }


def test_descriptor_has_one_independently_finalized_file_pin() -> None:
    descriptor = host.HostExecutorDescriptorV2()
    body = host.canonical_host_executor_descriptor_v2_body_bytes(descriptor)
    file = host.canonical_host_executor_descriptor_v2_file_bytes(descriptor)
    decoded = json.loads(file)
    assert EXPECTED_DESCRIPTOR_BODY_SHA256 != ZERO_SHA256
    assert EXPECTED_DESCRIPTOR_FILE_SHA256 != ZERO_SHA256
    assert hashlib.sha256(body).hexdigest() == EXPECTED_DESCRIPTOR_BODY_SHA256
    assert hashlib.sha256(file).hexdigest() == EXPECTED_DESCRIPTOR_FILE_SHA256
    assert decoded["descriptor_body_sha256"] == EXPECTED_DESCRIPTOR_BODY_SHA256
    assert host.PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256 == (
        EXPECTED_DESCRIPTOR_FILE_SHA256
    )
    assert host.host_executor_v2_descriptor_sha256() == EXPECTED_DESCRIPTOR_FILE_SHA256
    assert host.parse_host_executor_descriptor_v2(
        file,
        expected_file_sha256=EXPECTED_DESCRIPTOR_FILE_SHA256,
    ) == descriptor
    assert not any("test" in name.lower() and "pin" in name.lower() for name in vars(host))


def test_descriptor_guard_fails_zero_mismatch_and_wrong_caller_pin() -> None:
    descriptor = host.HostExecutorDescriptorV2()
    file = host.canonical_host_executor_descriptor_v2_file_bytes(descriptor)
    previous_pin = host.PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256
    try:
        setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", ZERO_SHA256)
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            host.host_executor_v2_descriptor_sha256()
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            host.parse_host_executor_descriptor_v2(
                file,
                expected_file_sha256=_hash("caller-pin"),
            )
        setattr(
            host,
            "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256",
            _hash("mismatched-repository-pin"),
        )
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            host.host_executor_v2_descriptor_sha256()
    finally:
        setattr(host, "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256", previous_pin)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_executor_descriptor_v2(
            file,
            expected_file_sha256=_hash("wrong-caller-pin"),
        )


def test_descriptor_public_inventory_and_nonproduction_deny_semantics() -> None:
    body = host.HostExecutorDescriptorV2().to_body_dict()
    assert body["schema_version"] == DESCRIPTOR_SCHEMA
    assert body["operational_phases"] == list(PHASES)
    assert len(body["public_api"]["strict_parsers"]) == 23
    assert "parse_host_cgroup_membership_event_log_v2" in body["public_api"][
        "strict_parsers"
    ]
    assert len(body["public_api"]["canonical_builders"]) == 46
    assert body["public_api"]["operational_apis"] == []
    containment = body["containment_contract"]
    assert containment["docker_cgroup_parent"] == "/alberta-qualified-host"
    assert containment["production_containment_claim_available"] is False
    assert containment["provisioning_receipt_semantics_validated"] is False
    assert containment["provisioning_receipt_producer_authenticated"] is False
    assert containment["future_production_backend_requires_additive_new_schema"] is True
    assert all(value is False for value in body["capabilities"].values())
    assert body["acknowledgement_is_execution_authority"] is False


@pytest.mark.parametrize("ordinal", [0, 14, 23, 24])
def test_request_uses_exact_build_and_candidate_role_schemas(ordinal: int) -> None:
    request = _request(ordinal)
    publisher, atomic, driver = _producer_schemas(ordinal)
    assert request.build_context_receipt.schema_version == (
        "alberta.forager_matched_v3.cpu_oci_build_context_receipt.v1"
    )
    assert request.build_execution_receipt.schema_version == (
        "alberta.forager_matched_v3.cpu_oci_build_execution_receipt.v1"
    )
    assert request.build_publication_receipt.schema_version == (
        "alberta.forager_matched_v3.cpu_oci_build_publication.v1"
    )
    assert request.full_resource_merger.descriptor_schema_version == (
        "alberta.forager_matched_v3.full_resource_merger_descriptor.v1"
    )
    assert request.publisher.descriptor_schema_version == publisher
    assert request.native_atomic_producer.descriptor_schema_version == atomic
    assert request.in_container_driver.descriptor_schema_version == driver
    assert request.container_name == "alberta-mv3-" + request.case_spine_sha256
    assert ":" not in request.container_name
    assert len(request.container_name) <= 128
    assert request.host_executor.descriptor_sha256 == _descriptor_hashes()[1]
    assert request.to_body_dict()["case_execution_ticket_is_execution_authority"] is False


def test_request_rejects_zero_image_wrong_roles_and_v1_two_hash_union() -> None:
    request = _request()
    with _observed_descriptor_pin():
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(request, image_id="sha256:" + ZERO_SHA256)
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                request,
                publisher=_producer(
                    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1",
                    "wrong-family-publisher",
                ),
            )
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                request,
                host_executor=_producer(
                    DESCRIPTOR_SCHEMA,
                    "v1-in-source-position",
                    descriptor=_descriptor_hashes()[1],
                    source=(
                        "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
                    ),
                ),
            )
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                request,
                host_executor=_producer(
                    DESCRIPTOR_SCHEMA,
                    "v1-in-descriptor-position",
                    descriptor=(
                        "d8bbc666a49e252662807f256c7f212c9a7c8c3be279b928a6a93ed77532a2e1"
                    ),
                    source=_hash("different-source"),
                ),
            )


def test_container_name_and_absolute_cgroup_parent_are_exact() -> None:
    request = _request()
    identity = _case_identity(request)
    assert identity.docker_cgroup_parent == (
        "/alberta-qualified-host/case-" + request.case_spine_sha256
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(
            identity,
            docker_cgroup_parent=(
                "alberta-qualified-host/case-" + request.case_spine_sha256
            ),
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(request, container_name="relative_or_partial_name")
    intent, _, ready, _, _ = _operational_prefix(request)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(intent, container_name="alberta-mv3-short")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(ready, container_id=ZERO_SHA256)


def test_recovery_schema_and_dependency_mappings_are_immutable() -> None:
    assert dict(host.RECOVERY_NODE_DEPENDENCIES) == RECOVERY_DEPENDENCIES
    with pytest.raises(TypeError):
        host.RECOVERY_NODE_DEPENDENCIES["cgroup_kill"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        host.RECOVERY_NODE_SCHEMAS["cgroup_kill"] = "mutated"  # type: ignore[index]


def _validate_success_bundle(bundle: dict[str, Any]) -> None:
    recovery = bundle["recovery"]
    host.validate_host_request_intent_v2_chain(bundle["request"], bundle["intent"])
    host.validate_host_committed_prefix_v2_chain(
        bundle["request"],
        bundle["intent"],
        bundle["initial"],
        bundle["ready"],
        bundle["anchor"],
        bundle["go"],
        bundle["frontier"],
    )
    host.validate_host_ready_anchor_go_v2_chain(
        bundle["intent"],
        bundle["initial"],
        bundle["ready"],
        bundle["anchor"],
        bundle["go"],
    )
    host.validate_retained_cgroup_fd_inventory_v2(bundle["initial"].counter_fds)
    host.validate_host_cgroup_boundary_proof_v2_chain(
        bundle["frontier"],
        bundle["initial"],
        recovery["pre"],
        recovery["kill"],
        recovery["empty"],
        recovery["absence"],
        recovery["post"],
        recovery["close"],
        recovery["outer"],
        bundle["proof"],
        request=bundle["request"],
        intent=bundle["intent"],
        ready=bundle["ready"],
        observer_anchor=bundle["anchor"],
        go=bundle["go"],
        membership_event_log=bundle["event_log"],
    )
    host.validate_host_cleanup_reconciliation_v2_chain(
        bundle["frontier"],
        bundle["cleanup"],
        request=bundle["request"],
        intent=bundle["intent"],
        ready=bundle["ready"],
        observer_anchor=bundle["anchor"],
        go=bundle["go"],
        membership_event_log=bundle["event_log"],
        initial_sample=bundle["initial"],
        precleanup_sample=recovery["pre"],
        cgroup_kill_receipt=recovery["kill"],
        cgroup_empty_observation=recovery["empty"],
        container_absence_observation=recovery["absence"],
        post_container_remove_sample=recovery["post"],
        cgroup_counter_fds_closed_receipt=recovery["close"],
        outer_cgroup_absence_observation=recovery["outer"],
        cgroup_proof=bundle["proof"],
    )
    host.validate_host_terminal_metadata_v2_chain(
        bundle["frontier"],
        bundle["cleanup"],
        bundle["terminal"],
        request=bundle["request"],
    )
    host.validate_host_lifecycle_v2_chain(
        bundle["frontier"],
        bundle["cleanup"],
        bundle["terminal"],
        bundle["lifecycle"],
        request=bundle["request"],
    )
    host.validate_host_success_receipt_v2_chain(
        bundle["request"],
        bundle["intent"],
        bundle["ready"],
        bundle["anchor"],
        bundle["go"],
        bundle["frontier"],
        bundle["cleanup"],
        bundle["proof"],
        bundle["terminal"],
        bundle["lifecycle"],
        bundle["receipt"],
        initial_sample=bundle["initial"],
        precleanup_sample=recovery["pre"],
        cgroup_kill_receipt=recovery["kill"],
        cgroup_empty_observation=recovery["empty"],
        container_absence_observation=recovery["absence"],
        post_container_remove_sample=recovery["post"],
        cgroup_counter_fds_closed_receipt=recovery["close"],
        outer_cgroup_absence_observation=recovery["outer"],
        membership_event_log=bundle["event_log"],
    )
    host.validate_host_observation_handoff_v2_chain(
        bundle["receipt"], bundle["terminal"], bundle["handoff"]
    )


def _validate_failure_bundle(bundle: dict[str, Any]) -> None:
    recovery = bundle["recovery"]
    host.validate_host_failure_receipt_v2_chain(
        bundle["request"],
        bundle["intent"],
        bundle["frontier"],
        bundle["cleanup"],
        bundle["terminal"],
        bundle["lifecycle"],
        bundle["receipt"],
        initial_sample=bundle["initial"],
        ready=bundle["ready"],
        anchor=bundle["anchor"],
        go=bundle["go"],
        precleanup_sample=recovery["pre"],
        cgroup_kill_receipt=recovery["kill"],
        cgroup_empty_observation=recovery["empty"],
        container_absence_observation=recovery["absence"],
        post_container_remove_sample=recovery["post"],
        cgroup_counter_fds_closed_receipt=recovery["close"],
        outer_cgroup_absence_observation=recovery["outer"],
        membership_event_log=bundle["event_log"],
        cgroup_proof=bundle["proof"],
    )
    host.validate_host_observation_handoff_v2_chain(
        bundle["receipt"], bundle["terminal"], bundle["handoff"]
    )


def test_fully_linked_retained_fd_structural_success_chain() -> None:
    bundle = _success_bundle()
    _validate_success_bundle(bundle)
    proof = bundle["proof"]
    receipt = bundle["receipt"]
    handoff = bundle["handoff"]
    assert proof.continuous_all_descendant_membership_proven is False
    assert proof.provisioning_validated is False
    assert proof.provisioning_producer_authenticated is False
    assert proof.production_containment_eligible is False
    assert proof.resources.structural_measurements_only is True
    assert proof.resources.production_qualified is False
    assert receipt.structural_success_shape_only is True
    assert receipt.production_execution_success is False
    assert receipt.production_acceptance_eligible is False
    assert receipt.evidence_eligible is False
    assert handoff.structural_success_shape_only is True
    assert handoff.production_execution_success is False
    assert handoff.evidence_eligible is False


def test_membership_log_and_provisioning_crosslinks_are_load_bearing() -> None:
    bundle = _success_bundle()
    event_log = bundle["event_log"]
    bad_log = replace(event_log, host_pid=event_log.host_pid + 1)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cgroup_boundary_proof_v2_chain(
            bundle["frontier"],
            bundle["initial"],
            bundle["recovery"]["pre"],
            bundle["recovery"]["kill"],
            bundle["recovery"]["empty"],
            bundle["recovery"]["absence"],
            bundle["recovery"]["post"],
            bundle["recovery"]["close"],
            bundle["recovery"]["outer"],
            bundle["proof"],
            request=bundle["request"],
            intent=bundle["intent"],
            ready=bundle["ready"],
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=bad_log,
        )
    evidence = bundle["proof"].observer_terminal_evidence
    bad_evidence = replace(
        evidence,
        host_provisioning_receipt=_artifact(
            "alberta.forager_matched_v3.host_provisioning_receipt.v2",
            "crosswired-provisioning",
        ),
    )
    bad_proof = replace(
        bundle["proof"],
        observer_terminal_evidence=bad_evidence,
        observer_terminal_evidence_sha256=host.observer_terminal_evidence_sha256_v2(
            bad_evidence
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cgroup_boundary_proof_v2_chain(
            bundle["frontier"],
            bundle["initial"],
            bundle["recovery"]["pre"],
            bundle["recovery"]["kill"],
            bundle["recovery"]["empty"],
            bundle["recovery"]["absence"],
            bundle["recovery"]["post"],
            bundle["recovery"]["close"],
            bundle["recovery"]["outer"],
            bad_proof,
            request=bundle["request"],
            intent=bundle["intent"],
            ready=bundle["ready"],
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=bundle["event_log"],
        )


@pytest.mark.parametrize(
    ("dependency", "dependent"),
    [
        (dependency, name)
        for name in RECOVERY_NAMES
        for dependency in RECOVERY_DEPENDENCIES[name]
    ],
)
def test_every_recovery_dag_edge_requires_committed_dependency(
    dependency: str,
    dependent: str,
) -> None:
    cleanup = _success_bundle()["cleanup"]
    nodes = list(cleanup.recovery_nodes)
    index = RECOVERY_NAMES.index(dependency)
    nodes[index] = replace(
        nodes[index],
        state="failed_before_commit",
        artifact=None,
        uncertainty_detail_sha256=_hash("dependency-blocked:" + dependency),
    )
    assert nodes[RECOVERY_NAMES.index(dependent)].state == "committed"
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(
            cleanup,
            recovery_nodes=tuple(nodes),
            cleanup_proven=False,
            unresolved_recovery_nodes=(dependency,),
        )


def test_kill_uncertainty_does_not_block_empty_or_container_branches() -> None:
    bundle = _recovery_only_failure_bundle()
    states = {node.node_name: node.state for node in bundle["cleanup"].recovery_nodes}
    assert states["cgroup_kill"] == "commit_uncertain"
    assert states["cgroup_empty"] == "committed"
    assert states["container_absence"] == "committed"
    assert states["post_container_remove_cgroup_sample"] == "committed"
    assert states["final_cgroup_proof"] == "failed_before_commit"
    _validate_failure_bundle(bundle)
    assert bundle["receipt"].classification == (
        "recovery_failure_after_complete_operational_frontier_nonretryable"
    )


@pytest.mark.parametrize("failure_index", range(len(PHASES)))
@pytest.mark.parametrize("effect", ["failed_before_commit", "commit_uncertain"])
def test_all_nineteen_operational_phases_terminalize_for_both_effects(
    failure_index: int,
    effect: str,
) -> None:
    bundle = _failure_bundle(failure_index, effect)
    _validate_failure_bundle(bundle)
    frontier = bundle["frontier"]
    assert frontier.completed_phases == PHASES[:failure_index]
    assert frontier.failure_phase == PHASES[failure_index]
    assert frontier.failure_effect_state == effect
    expected = (
        "operational_commit_uncertain_ticket_quarantined_nonretryable"
        if effect == "commit_uncertain"
        else "operational_failed_before_commit_ticket_quarantined_nonretryable"
    )
    assert bundle["receipt"].classification == expected
    assert bundle["receipt"].ticket_quarantined is True
    assert bundle["receipt"].same_case_retry_permitted is False


@pytest.mark.parametrize(
    ("failure_index", "effect", "resolution", "remove_count", "actual_present"),
    [
        (3, "failed_before_commit", "never_created", 0, False),
        (3, "commit_uncertain", "never_created", 0, False),
        (6, "failed_before_commit", "never_created", 0, False),
        (6, "commit_uncertain", "create_uncertain_resolved_absent", 0, False),
        (7, "failed_before_commit", "created_removed", 1, True),
        (7, "commit_uncertain", "created_removed", 1, True),
    ],
)
def test_early_container_absence_is_derived_from_create_frontier(
    failure_index: int,
    effect: str,
    resolution: str,
    remove_count: int,
    actual_present: bool,
) -> None:
    bundle = _failure_bundle(failure_index, effect)
    absence = bundle["recovery"]["absence"]
    assert absence.container_name == bundle["request"].container_name
    assert absence.resolution_state == resolution
    assert absence.container_remove_count == remove_count
    assert (absence.actual_container_id is not None) is actual_present
    assert (
        absence.actual_runtime_container_identity_sha256 is not None
    ) is actual_present
    if not bundle["recovery"]["cgroup_may_exist"]:
        states = {
            node.node_name: node.state for node in bundle["cleanup"].recovery_nodes
        }
        assert states["container_absence"] == "committed"
        assert all(
            state == "not_applicable"
            for name, state in states.items()
            if name != "container_absence"
        )


def test_create_uncertain_found_and_removed_branch_is_correlated() -> None:
    bundle = _failure_bundle(6, "commit_uncertain")
    absence = bundle["recovery"]["absence"]
    actual_id = _hash("resolved-uncertain-container")
    found = replace(
        absence,
        resolution_state="create_uncertain_found_removed",
        actual_container_id=actual_id,
        actual_runtime_container_identity_sha256=host.container_runtime_identity_sha256_v2(
            bundle["request"].case_spine_sha256,
            bundle["request"].container_name,
            actual_id,
        ),
        container_remove_count=1,
    )
    assert found.container_remove_count == 1
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(found, container_remove_count=0)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(found, actual_container_id=ZERO_SHA256)


@pytest.mark.parametrize(
    ("failure_index", "effect", "state", "count_state", "count"),
    [
        (14, "failed_before_commit", "not_started", "exact", 0),
        (14, "commit_uncertain", "commit_uncertain", "uncertain", None),
        (15, "failed_before_commit", "committed", "exact", 1),
        (15, "commit_uncertain", "committed", "exact", 1),
    ],
)
def test_late_native_publication_state_survives_terminal_failure(
    failure_index: int,
    effect: str,
    state: str,
    count_state: str,
    count: int | None,
) -> None:
    bundle = _failure_bundle(failure_index, effect)
    projection = bundle["lifecycle"].native_publication
    assert projection.native_publication_state == state
    assert projection.native_publication_commit_count_state == count_state
    assert projection.native_publication_commit_count == count
    assert projection == bundle["receipt"].native_publication
    assert projection.failure_publication_projection is None
    if state == "commit_uncertain":
        assert projection.publication_reconciliation_reference is not None
        assert projection.native_publication_receipt is None


def test_committed_wrapper_failure_has_exact_observations_v2_failure_projection_schema() -> None:
    bundle = _failure_bundle(16, "failed_before_commit")
    projection = bundle["receipt"].native_publication
    assert projection.native_publication_state == "committed"
    assert projection.failure_publication_projection is not None
    assert projection.failure_publication_projection.schema_version == (
        "alberta.forager_matched_v3.qualification_failure_publication_projection.v2"
    )


@pytest.mark.parametrize(
    ("ordinal", "receipt_schema"),
    [
        (0, None),
        (14, "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"),
        (23, "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1"),
    ],
)
def test_native_publication_receipt_role_is_exact_by_family(
    ordinal: int,
    receipt_schema: str | None,
) -> None:
    request = _request(ordinal)
    projection = _native_publication(
        request,
        _frontier(request.case_spine_sha256),
        terminal_failure=False,
    )
    assert (
        None
        if projection.native_publication_receipt is None
        else projection.native_publication_receipt.schema_version
    ) == receipt_schema


def _parser_cases() -> tuple[tuple[str, Any, Any, Any], ...]:
    success = _success_bundle()
    failure = _failure_bundle(0, "failed_before_commit")
    recovery = success["recovery"]
    return (
        (
            "parse_host_executor_descriptor_v2",
            host.parse_host_executor_descriptor_v2,
            host.HostExecutorDescriptorV2(),
            host.canonical_host_executor_descriptor_v2_file_bytes,
        ),
        (
            "parse_host_case_request_v2",
            host.parse_host_case_request_v2,
            success["request"],
            host.canonical_host_case_request_v2_file_bytes,
        ),
        (
            "parse_host_case_intent_v2",
            host.parse_host_case_intent_v2,
            success["intent"],
            host.canonical_host_case_intent_v2_file_bytes,
        ),
        (
            "parse_host_ready_v2",
            host.parse_host_ready_v2,
            success["ready"],
            host.canonical_host_ready_v2_file_bytes,
        ),
        (
            "parse_host_observer_anchor_v2",
            host.parse_host_observer_anchor_v2,
            success["anchor"],
            host.canonical_host_observer_anchor_v2_file_bytes,
        ),
        (
            "parse_host_go_v2",
            host.parse_host_go_v2,
            success["go"],
            host.canonical_host_go_v2_file_bytes,
        ),
        (
            "parse_host_operational_frontier_v2",
            host.parse_host_operational_frontier_v2,
            success["frontier"],
            host.canonical_host_operational_frontier_v2_file_bytes,
        ),
        (
            "parse_host_cgroup_membership_event_log_v2",
            host.parse_host_cgroup_membership_event_log_v2,
            success["event_log"],
            host.canonical_host_cgroup_membership_event_log_v2_file_bytes,
        ),
        (
            "parse_host_initial_cgroup_sample_v2",
            host.parse_host_initial_cgroup_sample_v2,
            success["initial"],
            host.canonical_host_initial_cgroup_sample_v2_file_bytes,
        ),
        (
            "parse_host_precleanup_cgroup_sample_v2",
            host.parse_host_precleanup_cgroup_sample_v2,
            recovery["pre"],
            host.canonical_host_precleanup_cgroup_sample_v2_file_bytes,
        ),
        (
            "parse_host_cgroup_kill_receipt_v2",
            host.parse_host_cgroup_kill_receipt_v2,
            recovery["kill"],
            host.canonical_host_cgroup_kill_receipt_v2_file_bytes,
        ),
        (
            "parse_host_cgroup_empty_observation_v2",
            host.parse_host_cgroup_empty_observation_v2,
            recovery["empty"],
            host.canonical_host_cgroup_empty_observation_v2_file_bytes,
        ),
        (
            "parse_host_container_absence_observation_v2",
            host.parse_host_container_absence_observation_v2,
            recovery["absence"],
            host.canonical_host_container_absence_observation_v2_file_bytes,
        ),
        (
            "parse_host_post_container_remove_cgroup_sample_v2",
            host.parse_host_post_container_remove_cgroup_sample_v2,
            recovery["post"],
            host.canonical_host_post_container_remove_cgroup_sample_v2_file_bytes,
        ),
        (
            "parse_host_cgroup_counter_fds_closed_receipt_v2",
            host.parse_host_cgroup_counter_fds_closed_receipt_v2,
            recovery["close"],
            host.canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes,
        ),
        (
            "parse_host_outer_cgroup_absence_observation_v2",
            host.parse_host_outer_cgroup_absence_observation_v2,
            recovery["outer"],
            host.canonical_host_outer_cgroup_absence_observation_v2_file_bytes,
        ),
        (
            "parse_host_cgroup_boundary_proof_v2",
            host.parse_host_cgroup_boundary_proof_v2,
            success["proof"],
            host.canonical_host_cgroup_boundary_proof_v2_file_bytes,
        ),
        (
            "parse_host_cleanup_reconciliation_v2",
            host.parse_host_cleanup_reconciliation_v2,
            success["cleanup"],
            host.canonical_host_cleanup_reconciliation_v2_file_bytes,
        ),
        (
            "parse_host_terminal_metadata_v2",
            host.parse_host_terminal_metadata_v2,
            success["terminal"],
            host.canonical_host_terminal_metadata_v2_file_bytes,
        ),
        (
            "parse_host_lifecycle_v2",
            host.parse_host_lifecycle_v2,
            success["lifecycle"],
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        (
            "parse_host_success_receipt_v2",
            host.parse_host_success_receipt_v2,
            success["receipt"],
            host.canonical_host_success_receipt_v2_file_bytes,
        ),
        (
            "parse_host_failure_receipt_v2",
            host.parse_host_failure_receipt_v2,
            failure["receipt"],
            host.canonical_host_failure_receipt_v2_file_bytes,
        ),
        (
            "parse_host_observation_handoff_v2",
            host.parse_host_observation_handoff_v2,
            success["handoff"],
            host.canonical_host_observation_handoff_v2_file_bytes,
        ),
    )


def test_every_public_parser_roundtrips_and_rejects_wrong_caller_file_pin() -> None:
    cases = _parser_cases()
    expected_names = (
        "parse_host_executor_descriptor_v2",
        "parse_host_case_request_v2",
        "parse_host_case_intent_v2",
        "parse_host_ready_v2",
        "parse_host_observer_anchor_v2",
        "parse_host_go_v2",
        "parse_host_operational_frontier_v2",
        "parse_host_cgroup_membership_event_log_v2",
        "parse_host_initial_cgroup_sample_v2",
        "parse_host_precleanup_cgroup_sample_v2",
        "parse_host_cgroup_kill_receipt_v2",
        "parse_host_cgroup_empty_observation_v2",
        "parse_host_container_absence_observation_v2",
        "parse_host_post_container_remove_cgroup_sample_v2",
        "parse_host_cgroup_counter_fds_closed_receipt_v2",
        "parse_host_outer_cgroup_absence_observation_v2",
        "parse_host_cgroup_boundary_proof_v2",
        "parse_host_cleanup_reconciliation_v2",
        "parse_host_terminal_metadata_v2",
        "parse_host_lifecycle_v2",
        "parse_host_success_receipt_v2",
        "parse_host_failure_receipt_v2",
        "parse_host_observation_handoff_v2",
    )
    assert tuple(name for name, _, _, _ in cases) == expected_names
    assert host.HOST_EXECUTOR_PUBLIC_PARSERS == expected_names
    for _, parser, value, file_builder in cases:
        raw = file_builder(value)
        file_pin = hashlib.sha256(raw).hexdigest()
        assert parser(raw, expected_file_sha256=file_pin) == value
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            parser(raw, expected_file_sha256=_hash("wrong-caller-file-pin"))


def _rebuild_file(item: dict[str, Any], body_field: str) -> bytes:
    body = {key: value for key, value in item.items() if key != body_field}
    rebuilt = {**body, body_field: _body_hash(body)}
    return (
        json.dumps(
            rebuilt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def test_parser_rejects_embedded_body_envelope_content_and_exact_type_errors() -> None:
    request = _request()
    raw = host.canonical_host_case_request_v2_file_bytes(request)
    decoded = json.loads(raw)
    decoded["request_body_sha256"] = ZERO_SHA256
    bad_body = (
        json.dumps(decoded, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_case_request_v2(
            bad_body,
            expected_file_sha256=hashlib.sha256(bad_body).hexdigest(),
        )
    decoded = json.loads(raw)
    decoded["status"] = "wrong_envelope"
    bad_envelope = _rebuild_file(decoded, "request_body_sha256")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_case_request_v2(
            bad_envelope,
            expected_file_sha256=hashlib.sha256(bad_envelope).hexdigest(),
        )
    decoded = json.loads(raw)
    decoded["horizon"] = 1
    bad_content = _rebuild_file(decoded, "request_body_sha256")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_case_request_v2(
            bad_content,
            expected_file_sha256=hashlib.sha256(bad_content).hexdigest(),
        )
    decoded = json.loads(raw)
    decoded["extra"] = "forbidden"
    bad_keys = _rebuild_file(decoded, "request_body_sha256")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_case_request_v2(
            bad_keys,
            expected_file_sha256=hashlib.sha256(bad_keys).hexdigest(),
        )
    with pytest.raises(TypeError):
        host.canonical_host_case_request_v2_file_bytes(object())  # type: ignore[arg-type]


def test_membership_event_log_parser_rejects_inventory_migration_and_claim_upgrade() -> None:
    event_log = _success_bundle()["event_log"]
    raw = host.canonical_host_cgroup_membership_event_log_v2_file_bytes(event_log)
    for mutate in ("missing_event", "migration", "production_claim"):
        decoded = json.loads(raw)
        if mutate == "missing_event":
            decoded["events"].pop()
            decoded["event_inventory_sha256"] = _body_hash(
                {"events": decoded["events"]}
            )
        elif mutate == "migration":
            decoded["events"][1]["migration_event_count"] = 1
            decoded["event_inventory_sha256"] = _body_hash(
                {"events": decoded["events"]}
            )
        else:
            decoded["production_containment_eligible"] = True
        changed = _rebuild_file(decoded, "membership_event_log_body_sha256")
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            host.parse_host_cgroup_membership_event_log_v2(
                changed,
                expected_file_sha256=hashlib.sha256(changed).hexdigest(),
            )


def test_common_terminal_body_keys_and_no_status_remain_exact() -> None:
    body = _success_bundle()["terminal"].to_body_dict()
    assert set(body) == {
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
    }
    assert "status" not in body
    assert all(value is False for value in body["authority"].values())
    assert all(value is False for value in body["claims"].values())


def test_public_validator_inventory_is_exercised_by_linked_bundles() -> None:
    assert host.HOST_EXECUTOR_PUBLIC_VALIDATORS == (
        "validate_host_request_intent_v2_chain",
        "validate_host_committed_prefix_v2_chain",
        "validate_host_ready_anchor_go_v2_chain",
        "validate_retained_cgroup_fd_inventory_v2",
        "validate_host_cgroup_boundary_proof_v2_chain",
        "validate_host_cleanup_reconciliation_v2_chain",
        "validate_host_terminal_metadata_v2_chain",
        "validate_host_lifecycle_v2_chain",
        "validate_host_success_receipt_v2_chain",
        "validate_host_failure_receipt_v2_chain",
        "validate_host_observation_handoff_v2_chain",
    )
    _validate_success_bundle(_success_bundle())
    _validate_failure_bundle(_failure_bundle(0, "failed_before_commit"))


def test_aggregate_rejects_committed_prefix_and_recovery_crosswires() -> None:
    bundle = _failure_bundle(10, "failed_before_commit")
    bad_anchor = replace(
        bundle["anchor"],
        ready=_artifact(READY_SCHEMA, "crosswired-ready"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_committed_prefix_v2_chain(
            bundle["request"],
            bundle["intent"],
            bundle["initial"],
            bundle["ready"],
            bad_anchor,
            bundle["go"],
            bundle["frontier"],
        )
    success = _success_bundle()
    recovery = success["recovery"]
    bad_post = replace(
        recovery["post"],
        container_absence_observation=_artifact(
            ABSENCE_SCHEMA,
            "crosswired-container-absence",
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cleanup_reconciliation_v2_chain(
            success["frontier"],
            success["cleanup"],
            request=success["request"],
            intent=success["intent"],
            ready=success["ready"],
            observer_anchor=success["anchor"],
            go=success["go"],
            membership_event_log=success["event_log"],
            initial_sample=success["initial"],
            precleanup_sample=recovery["pre"],
            cgroup_kill_receipt=recovery["kill"],
            cgroup_empty_observation=recovery["empty"],
            container_absence_observation=recovery["absence"],
            post_container_remove_sample=bad_post,
            cgroup_counter_fds_closed_receipt=recovery["close"],
            outer_cgroup_absence_observation=recovery["outer"],
            cgroup_proof=success["proof"],
        )


def test_structural_artifacts_reject_production_claim_upgrades() -> None:
    bundle = _success_bundle()
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(bundle["event_log"], production_containment_eligible=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(
            bundle["proof"].observer_terminal_evidence,
            continuous_all_descendant_membership_proven=True,
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(bundle["proof"], provisioning_validated=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(bundle["proof"].resources, production_qualified=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(bundle["receipt"], production_execution_success=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(bundle["handoff"], evidence_eligible=True)


@pytest.mark.parametrize(
    ("pre_fact_overrides", "post_fact_overrides"),
    (
        ({}, {"cpu_usage_usec": 4}),
        ({}, {"memory_peak_bytes": 11}),
        ({}, {"pids_peak": 2}),
        (
            {"memory_oom_kill_count": 1},
            {"memory_oom_kill_count": 0},
        ),
        (
            {"pids_max_event_count": 1},
            {"pids_max_event_count": 0},
        ),
        (
            {},
            {"retained_fd_set_sha256": _hash("crosswired-retained-fd-set")},
        ),
    ),
    ids=(
        "cpu-rollback",
        "memory-peak-rollback",
        "pids-peak-rollback",
        "oom-kill-rollback",
        "pids-max-event-rollback",
        "retained-fd-crosswire",
    ),
)
def test_early_cleanup_rejects_fully_rehashed_pre_to_post_rollback_or_crosswire(
    pre_fact_overrides: dict[str, Any],
    post_fact_overrides: dict[str, Any],
) -> None:
    bundle = _failure_bundle(5, "failed_before_commit")
    recovery = _recovery_artifacts(
        bundle["request"],
        bundle["frontier"],
        initial=None,
        ready=None,
        pre_fact_overrides=pre_fact_overrides,
        post_fact_overrides=post_fact_overrides,
    )
    cleanup = _cleanup(bundle["frontier"], recovery)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cleanup_reconciliation_v2_chain(
            bundle["frontier"],
            cleanup,
            request=bundle["request"],
            precleanup_sample=recovery["pre"],
            cgroup_kill_receipt=recovery["kill"],
            cgroup_empty_observation=recovery["empty"],
            container_absence_observation=recovery["absence"],
            post_container_remove_sample=recovery["post"],
            cgroup_counter_fds_closed_receipt=recovery["close"],
            outer_cgroup_absence_observation=recovery["outer"],
        )


@pytest.mark.parametrize(
    "event_field",
    ("memory_oom_kill_count", "pids_max_event_count"),
)
def test_fully_rehashed_structural_success_rejects_kernel_quota_breach(
    event_field: str,
) -> None:
    bundle = _success_bundle()
    recovery = _recovery_artifacts(
        bundle["request"],
        bundle["frontier"],
        initial=bundle["initial"],
        ready=bundle["ready"],
        pre_fact_overrides={event_field: 1},
        post_fact_overrides={event_field: 1},
    )
    event_log, proof = _proof_and_event_log(
        bundle["request"],
        bundle["initial"],
        bundle["ready"],
        bundle["anchor"],
        bundle["frontier"],
        recovery,
    )
    cleanup = _cleanup(bundle["frontier"], recovery, proof=proof)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cleanup_reconciliation_v2_chain(
            bundle["frontier"],
            cleanup,
            request=bundle["request"],
            intent=bundle["intent"],
            ready=bundle["ready"],
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=event_log,
            initial_sample=bundle["initial"],
            precleanup_sample=recovery["pre"],
            cgroup_kill_receipt=recovery["kill"],
            cgroup_empty_observation=recovery["empty"],
            container_absence_observation=recovery["absence"],
            post_container_remove_sample=recovery["post"],
            cgroup_counter_fds_closed_receipt=recovery["close"],
            outer_cgroup_absence_observation=recovery["outer"],
            cgroup_proof=proof,
        )


def test_raw_resources_project_kernel_quota_breach_counters() -> None:
    fields = host.HostRawResourceMeasurementsV2.__dataclass_fields__
    assert "memory_oom_kill_count" in fields
    assert "pids_max_event_count" in fields


def test_failure_terminal_and_lifecycle_cannot_crosswire_candidate_on_same_spine() -> None:
    bundle = _failure_bundle(0, "failed_before_commit")
    alternate = _request(1)
    with _observed_descriptor_pin():
        alternate = replace(
            alternate,
            case_spine_sha256=bundle["request"].case_spine_sha256,
            container_name=bundle["request"].container_name,
        )
    terminal = replace(
        bundle["terminal"],
        case_ordinal=alternate.case_ordinal,
        candidate_id=alternate.candidate_id,
        candidate_family=alternate.candidate_family,
        qualification_case_id=alternate.qualification_case_id,
    )
    lifecycle = _lifecycle(
        alternate,
        bundle["frontier"],
        bundle["cleanup"],
        terminal,
    )
    receipt = replace(
        bundle["receipt"],
        terminal_metadata=_identity(
            terminal,
            TERMINAL_SCHEMA,
            host.canonical_host_terminal_metadata_v2_body_bytes,
            host.canonical_host_terminal_metadata_v2_file_bytes,
        ),
        lifecycle=_identity(
            lifecycle,
            LIFECYCLE_SCHEMA,
            host.canonical_host_lifecycle_v2_body_bytes,
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        native_publication=lifecycle.native_publication,
    )
    recovery = bundle["recovery"]
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_failure_receipt_v2_chain(
            bundle["request"],
            bundle["intent"],
            bundle["frontier"],
            bundle["cleanup"],
            terminal,
            lifecycle,
            receipt,
            initial_sample=bundle["initial"],
            ready=bundle["ready"],
            anchor=bundle["anchor"],
            go=bundle["go"],
            precleanup_sample=recovery["pre"],
            cgroup_kill_receipt=recovery["kill"],
            cgroup_empty_observation=recovery["empty"],
            container_absence_observation=recovery["absence"],
            post_container_remove_sample=recovery["post"],
            cgroup_counter_fds_closed_receipt=recovery["close"],
            outer_cgroup_absence_observation=recovery["outer"],
            membership_event_log=bundle["event_log"],
            cgroup_proof=bundle["proof"],
        )
    handoff = _handoff(receipt, terminal)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_observation_handoff_v2_chain(receipt, terminal, handoff)


@pytest.mark.parametrize(
    ("failure_index", "bool_count"),
    ((None, True), (0, False)),
    ids=("committed-true", "not-started-false"),
)
def test_native_publication_count_rejects_bool(
    failure_index: int | None,
    bool_count: bool,
) -> None:
    request = _request()
    frontier = _frontier(
        request.case_spine_sha256,
        failure_index=failure_index,
        effect="failed_before_commit" if failure_index is not None else None,
    )
    projection = _native_publication(
        request,
        frontier,
        terminal_failure=failure_index is not None,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(projection, native_publication_commit_count=bool_count)


def test_cleanup_and_lifecycle_reject_duck_typed_auxiliary_inputs() -> None:
    class Duck:
        def __init__(self, value: object) -> None:
            self.value = value

        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)

    bundle = _failure_bundle(0, "failed_before_commit")
    recovery = bundle["recovery"]
    with pytest.raises(TypeError):
        host.validate_host_cleanup_reconciliation_v2_chain(
            bundle["frontier"],
            bundle["cleanup"],
            request=Duck(bundle["request"]),  # type: ignore[arg-type]
            container_absence_observation=recovery["absence"],
        )
    with pytest.raises(TypeError):
        host.validate_host_lifecycle_v2_chain(
            bundle["frontier"],
            bundle["cleanup"],
            bundle["terminal"],
            bundle["lifecycle"],
            request=Duck(bundle["request"]),  # type: ignore[arg-type]
        )


def test_descriptor_discloses_raw_container_id_path_is_nonproduction() -> None:
    body = host.HostExecutorDescriptorV2().to_body_dict()
    containment = body["containment_contract"]
    assert containment["container_cgroup_child_path_is_derived"] is True
    assert containment["actual_container_cgroup_path_observed"] is False
    assert containment["actual_container_cgroup_identity_authenticated"] is False
    assert containment["production_v3_requires_observed_container_cgroup_path"] is True
    assert "raw_container_id_child_path_is_nonportable_source_only" in body[
        "limitations"
    ]


def test_public_exports_are_unique_and_include_retained_fd_digest_helper() -> None:
    assert len(host.__all__) == len(set(host.__all__))
    assert "retained_cgroup_fd_inventory_sha256_v2" in host.__all__


def test_every_structural_nonproduction_prohibition_is_constructor_enforced() -> None:
    bundle = _success_bundle()
    mutations = (
        *(
            (bundle["event_log"], field, True)
            for field in (
                "continuous_all_descendant_membership_proven",
                "provisioning_receipt_semantics_validated",
                "provisioning_receipt_producer_authenticated",
                "production_containment_eligible",
            )
        ),
        *(
            (bundle["proof"].observer_terminal_evidence, field, True)
            for field in (
                "continuous_all_descendant_membership_proven",
                "provisioning_validated",
                "provisioning_producer_authenticated",
                "production_containment_eligible",
            )
        ),
        *(
            (bundle["proof"], field, True)
            for field in (
                "continuous_all_descendant_membership_proven",
                "provisioning_validated",
                "provisioning_producer_authenticated",
                "production_containment_eligible",
            )
        ),
        (bundle["proof"].resources, "structural_measurements_only", False),
        (bundle["proof"].resources, "production_qualified", True),
        *(
            (bundle["lifecycle"], field, True)
            for field in (
                "production_execution_success",
                "production_acceptance_eligible",
                "evidence_eligible",
            )
        ),
        (bundle["lifecycle"], "structural_success_shape_only", False),
        *(
            (bundle["receipt"], field, True)
            for field in (
                "production_execution_success",
                "production_acceptance_eligible",
                "evidence_eligible",
            )
        ),
        (bundle["receipt"], "structural_success_shape_only", False),
        *(
            (bundle["handoff"], field, True)
            for field in (
                "production_execution_success",
                "production_acceptance_eligible",
                "evidence_eligible",
            )
        ),
        (bundle["handoff"], "structural_success_shape_only", False),
    )
    for value, field, replacement in mutations:
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(value, **{field: replacement})


def test_retained_fd_policy_and_inventory_fail_closed() -> None:
    first = _fds()[0]
    for changes in (
        {"endpoint_device": 0},
        {"open_flags": ("O_RDONLY",)},
        {"reset_performed": True},
        {"reopened": True},
        {"retained_through_post_container_remove_sample": False},
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(first, **changes)
    fds = _fds()
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_retained_cgroup_fd_inventory_v2(tuple(reversed(fds)))
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_retained_cgroup_fd_inventory_v2(
            (fds[0], replace(fds[1], endpoint_inode=fds[0].endpoint_inode), *fds[2:])
        )


def test_recovery_leaf_facts_and_order_fail_closed() -> None:
    recovery = _success_bundle()["recovery"]
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(recovery["kill"], cgroup_kill_value=0)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(recovery["kill"], entire_subtree_targeted=False)
    for changes in (
        {"populated": True},
        {"pids_current": 1},
        {"recursive_process_count": 1},
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(recovery["empty"], **changes)
    for changes in (
        {"populated": True},
        {"memory_current_bytes": 1, "memory_peak_bytes": 1},
        {"pids_current": 1, "pids_peak": 1},
        {"nr_descendants": 1},
        {"nr_dying_descendants": 1},
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                recovery["post"],
                facts=replace(recovery["post"].facts, **changes),
            )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(recovery["close"], all_fds_closed=False)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(recovery["close"], reopen_permitted=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(recovery["outer"], outer_cgroup_absent=False)


def test_operational_frontier_rejects_malformed_prefix_and_skipped_failure_phase() -> None:
    request = _request()
    frontier = _frontier(request.case_spine_sha256, failure_index=3, effect="failed_before_commit")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(frontier, completed_phases=(PHASES[1],))
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        replace(frontier, failure_phase=PHASES[4])


def test_request_permanent_image_build_and_adapter_exclusions_are_load_bearing() -> None:
    with _observed_descriptor_pin():
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                _request(),
                image_id=(
                    "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269"
                ),
            )
        request = _request()
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                request,
                build_context_receipt=replace(
                    request.build_context_receipt,
                    file_sha256=(
                        "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330"
                    ),
                ),
            )
        adapter = _request(23)
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
            replace(
                adapter,
                publisher=replace(
                    adapter.publisher,
                    descriptor_sha256=(
                        "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
                    ),
                ),
            )


def test_operational_failure_can_record_fully_proven_cleanup() -> None:
    bundle = _failure_bundle(13, "failed_before_commit")
    recovery = bundle["recovery"]
    event_log, proof = _proof_and_event_log(
        bundle["request"],
        bundle["initial"],
        bundle["ready"],
        bundle["anchor"],
        bundle["frontier"],
        recovery,
    )
    cleanup = _cleanup(bundle["frontier"], recovery, proof=proof)
    terminal = _terminal(bundle["request"], bundle["frontier"], cleanup)
    lifecycle = _lifecycle(
        bundle["request"],
        bundle["frontier"],
        cleanup,
        terminal,
    )
    receipt = replace(
        bundle["receipt"],
        cleanup_reconciliation=terminal.cleanup_reconciliation,
        terminal_metadata=lifecycle.terminal_metadata,
        lifecycle=_identity(
            lifecycle,
            LIFECYCLE_SCHEMA,
            host.canonical_host_lifecycle_v2_body_bytes,
            host.canonical_host_lifecycle_v2_file_bytes,
        ),
        native_publication=lifecycle.native_publication,
        unresolved_recovery_nodes=(),
        recovery_failure_count=0,
        recovery_uncertainty_count=0,
        cleanup_proven=True,
    )
    proven = {
        **bundle,
        "event_log": event_log,
        "proof": proof,
        "cleanup": cleanup,
        "terminal": terminal,
        "lifecycle": lifecycle,
        "receipt": receipt,
        "handoff": _handoff(receipt, terminal),
    }
    _validate_failure_bundle(proven)


def _rewire_success_after_container_absence(
    bundle: dict[str, Any],
    absence: host.HostContainerAbsenceObservationV2,
) -> tuple[dict[str, Any], host.HostCgroupMembershipEventLogV2, host.HostCgroupBoundaryProofV2]:
    recovery = dict(bundle["recovery"])
    post = replace(
        recovery["post"],
        container_absence_observation=_artifact_identity_for_recovery(
            "container_absence",
            absence,
        ),
        actual_runtime_container_identity_sha256=(
            absence.actual_runtime_container_identity_sha256
        ),
        actual_container_id=absence.actual_container_id,
    )
    close = replace(
        recovery["close"],
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
        actual_runtime_container_identity_sha256=(
            absence.actual_runtime_container_identity_sha256
        ),
        actual_container_id=absence.actual_container_id,
    )
    outer = replace(
        recovery["outer"],
        cgroup_counter_fds_closed_receipt=_artifact_identity_for_recovery(
            "cgroup_counter_fds_closed",
            close,
        ),
    )
    event_log = replace(
        bundle["event_log"],
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
    )
    evidence = replace(
        bundle["proof"].observer_terminal_evidence,
        post_container_remove_sample=event_log.post_container_remove_sample,
        membership_event_log=_identity(
            event_log,
            EVENT_LOG_SCHEMA,
            host.canonical_host_cgroup_membership_event_log_v2_body_bytes,
            host.canonical_host_cgroup_membership_event_log_v2_file_bytes,
        ),
    )
    proof = replace(
        bundle["proof"],
        container_absence_observation=_artifact_identity_for_recovery(
            "container_absence",
            absence,
        ),
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
        cgroup_counter_fds_closed_receipt=_artifact_identity_for_recovery(
            "cgroup_counter_fds_closed",
            close,
        ),
        outer_cgroup_absence_observation=_artifact_identity_for_recovery(
            "outer_cgroup_absence",
            outer,
        ),
        container_lookup_identity_sha256=absence.container_lookup_identity_sha256,
        actual_runtime_container_identity_sha256=(
            absence.actual_runtime_container_identity_sha256
        ),
        actual_container_id=absence.actual_container_id,
        observer_terminal_evidence=evidence,
        observer_terminal_evidence_sha256=host.observer_terminal_evidence_sha256_v2(
            evidence
        ),
    )
    recovery.update(
        {
            "absence": absence,
            "post": post,
            "close": close,
            "outer": outer,
        }
    )
    return recovery, event_log, proof


def test_direct_proof_rejects_fully_rehashed_empty_to_removal_chronology() -> None:
    bundle = _success_bundle()
    absence = replace(
        bundle["recovery"]["absence"],
        removal_monotonic_ns=(
            bundle["recovery"]["empty"].observed_monotonic_ns - 1
        ),
    )
    recovery, event_log, proof = _rewire_success_after_container_absence(
        bundle,
        absence,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cgroup_boundary_proof_v2_chain(
            bundle["frontier"],
            bundle["initial"],
            recovery["pre"],
            recovery["kill"],
            recovery["empty"],
            recovery["absence"],
            recovery["post"],
            recovery["close"],
            recovery["outer"],
            proof,
            request=bundle["request"],
            intent=bundle["intent"],
            ready=bundle["ready"],
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=event_log,
        )


def test_direct_proof_rejects_fully_rehashed_ready_container_crosswire() -> None:
    bundle = _success_bundle()
    recovery = dict(bundle["recovery"])
    different_id = _hash("different-actual-container")
    different_identity = host.container_runtime_identity_sha256_v2(
        bundle["request"].case_spine_sha256,
        bundle["request"].container_name,
        different_id,
    )
    absence = replace(
        recovery["absence"],
        actual_runtime_container_identity_sha256=different_identity,
        actual_container_id=different_id,
    )
    post = replace(
        recovery["post"],
        container_absence_observation=_artifact_identity_for_recovery(
            "container_absence",
            absence,
        ),
        actual_runtime_container_identity_sha256=different_identity,
        actual_container_id=different_id,
    )
    close = replace(
        recovery["close"],
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
        actual_runtime_container_identity_sha256=different_identity,
        actual_container_id=different_id,
    )
    outer = replace(
        recovery["outer"],
        cgroup_counter_fds_closed_receipt=_artifact_identity_for_recovery(
            "cgroup_counter_fds_closed",
            close,
        ),
    )
    event_log = replace(
        bundle["event_log"],
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
    )
    evidence = replace(
        bundle["proof"].observer_terminal_evidence,
        post_container_remove_sample=event_log.post_container_remove_sample,
        membership_event_log=_identity(
            event_log,
            EVENT_LOG_SCHEMA,
            host.canonical_host_cgroup_membership_event_log_v2_body_bytes,
            host.canonical_host_cgroup_membership_event_log_v2_file_bytes,
        ),
    )
    proof = replace(
        bundle["proof"],
        container_absence_observation=_artifact_identity_for_recovery(
            "container_absence",
            absence,
        ),
        post_container_remove_sample=_artifact_identity_for_recovery(
            "post_container_remove_cgroup_sample",
            post,
        ),
        cgroup_counter_fds_closed_receipt=_artifact_identity_for_recovery(
            "cgroup_counter_fds_closed",
            close,
        ),
        outer_cgroup_absence_observation=_artifact_identity_for_recovery(
            "outer_cgroup_absence",
            outer,
        ),
        actual_runtime_container_identity_sha256=different_identity,
        actual_container_id=different_id,
        observer_terminal_evidence=evidence,
        observer_terminal_evidence_sha256=host.observer_terminal_evidence_sha256_v2(
            evidence
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cgroup_boundary_proof_v2_chain(
            bundle["frontier"],
            bundle["initial"],
            recovery["pre"],
            recovery["kill"],
            recovery["empty"],
            absence,
            post,
            close,
            outer,
            proof,
            request=bundle["request"],
            intent=bundle["intent"],
            ready=bundle["ready"],
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=event_log,
        )


def test_request_rejects_build_receipts_with_distinct_schemas_but_aliased_bytes() -> None:
    request = _request()
    aliased_execution = replace(
        request.build_execution_receipt,
        file_sha256=request.build_context_receipt.file_sha256,
        body_sha256=request.build_context_receipt.body_sha256,
    )
    with _observed_descriptor_pin():
        with pytest.raises(
            host.ForagerMatchedV3HostQualificationExecutorV2Error,
            match="cannot alias",
        ):
            replace(request, build_execution_receipt=aliased_execution)


def test_success_rejects_go_committed_after_post_remove_measurement() -> None:
    bundle = _success_bundle()
    late_go = replace(bundle["go"], go_committed_monotonic_ns=950)
    late_receipt = replace(
        bundle["receipt"],
        go_commitment=_identity(
            late_go,
            GO_SCHEMA,
            host.canonical_host_go_v2_body_bytes,
            host.canonical_host_go_v2_file_bytes,
        ),
    )
    late = {
        **bundle,
        "go": late_go,
        "receipt": late_receipt,
        "handoff": _handoff(late_receipt, bundle["terminal"]),
    }
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        _validate_success_bundle(late)


def test_direct_proof_rejects_same_spine_alternate_candidate_ready() -> None:
    bundle = _success_bundle()
    candidate, family, case_id = _case(1)
    alternate_ready = replace(
        bundle["ready"],
        case_ordinal=1,
        candidate_id=candidate,
        candidate_family=family,
        qualification_case_id=case_id,
    )
    recovery = bundle["recovery"]
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_cgroup_boundary_proof_v2_chain(
            bundle["frontier"],
            bundle["initial"],
            recovery["pre"],
            recovery["kill"],
            recovery["empty"],
            recovery["absence"],
            recovery["post"],
            recovery["close"],
            recovery["outer"],
            bundle["proof"],
            request=bundle["request"],
            intent=bundle["intent"],
            ready=alternate_ready,
            observer_anchor=bundle["anchor"],
            go=bundle["go"],
            membership_event_log=bundle["event_log"],
        )


def test_failure_receipt_rejects_coherent_alternate_committed_handshake_prefix() -> None:
    bundle = _failure_bundle(13, "failed_before_commit")
    alternate_ready = replace(bundle["ready"], ready_monotonic_ns=201)
    alternate_ready_id = _identity(
        alternate_ready,
        READY_SCHEMA,
        host.canonical_host_ready_v2_body_bytes,
        host.canonical_host_ready_v2_file_bytes,
    )
    alternate_anchor = replace(
        bundle["anchor"],
        ready=alternate_ready_id,
        observer_started_monotonic_ns=301,
    )
    alternate_anchor_id = _identity(
        alternate_anchor,
        ANCHOR_SCHEMA,
        host.canonical_host_observer_anchor_v2_body_bytes,
        host.canonical_host_observer_anchor_v2_file_bytes,
    )
    alternate_go = replace(
        bundle["go"],
        ready=alternate_ready_id,
        observer_anchor=alternate_anchor_id,
        go_committed_monotonic_ns=401,
    )
    recovery = bundle["recovery"]
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.validate_host_failure_receipt_v2_chain(
            bundle["request"],
            bundle["intent"],
            bundle["frontier"],
            bundle["cleanup"],
            bundle["terminal"],
            bundle["lifecycle"],
            bundle["receipt"],
            initial_sample=bundle["initial"],
            ready=alternate_ready,
            anchor=alternate_anchor,
            go=alternate_go,
            precleanup_sample=recovery["pre"],
            cgroup_kill_receipt=recovery["kill"],
            cgroup_empty_observation=recovery["empty"],
            container_absence_observation=recovery["absence"],
            post_container_remove_sample=recovery["post"],
            cgroup_counter_fds_closed_receipt=recovery["close"],
            outer_cgroup_absence_observation=recovery["outer"],
            membership_event_log=None,
            cgroup_proof=None,
        )


@pytest.mark.parametrize(
    "ready_changes",
    (
        {"ready_monotonic_ns": 50},
        {"container_cgroup_inode": 1_002},
    ),
)
def test_failure_chain_rejects_impossible_partial_ready(
    ready_changes: dict[str, Any],
) -> None:
    bundle = _failure_bundle(9, "failed_before_commit")
    impossible_ready = replace(bundle["ready"], **ready_changes)
    receipt = replace(
        bundle["receipt"],
        ready=_identity(
            impossible_ready,
            READY_SCHEMA,
            host.canonical_host_ready_v2_body_bytes,
            host.canonical_host_ready_v2_file_bytes,
        ),
    )
    impossible = {
        **bundle,
        "ready": impossible_ready,
        "receipt": receipt,
        "handoff": _handoff(receipt, bundle["terminal"]),
    }
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        _validate_failure_bundle(impossible)


def test_failure_chain_rejects_go_after_cleanup_measurement() -> None:
    bundle = _failure_bundle(13, "failed_before_commit")
    late_go = replace(bundle["go"], go_committed_monotonic_ns=950)
    receipt = replace(
        bundle["receipt"],
        go_commitment=_identity(
            late_go,
            GO_SCHEMA,
            host.canonical_host_go_v2_body_bytes,
            host.canonical_host_go_v2_file_bytes,
        ),
    )
    late = {
        **bundle,
        "go": late_go,
        "receipt": receipt,
        "handoff": _handoff(receipt, bundle["terminal"]),
    }
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        _validate_failure_bundle(late)


def test_strict_json_rejects_duplicate_float_depth_alias_and_noncanonical_bytes() -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host._strict_json(b'{"a":1,"a":2}\n')
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host._strict_json(b'{"a":1.5}\n')
    nested: dict[str, Any] = {"leaf": "x"}
    for _ in range(65):
        nested = {"next": nested}
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host._canonical_json(nested, newline=True)
    child: dict[str, Any] = {"value": "x"}
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host._canonical_json({"left": child, "right": child}, newline=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host._canonical_json({"control\x00key": "x"}, newline=True)
    request_raw = host.canonical_host_case_request_v2_file_bytes(_request())
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.parse_host_case_request_v2(
            request_raw + b"\n",
            expected_file_sha256=hashlib.sha256(request_raw + b"\n").hexdigest(),
        )


def test_exact_builtin_types_and_nonzero_identities_fail_closed() -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.ArtifactIdentityV2(REQUEST_SCHEMA, ZERO_SHA256, _hash("body"))
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorV2Error):
        host.CgroupSampleFactsV2(
            monotonic_ns=True,
            cgroup_identity_sha256=_hash("cgroup"),
            retained_fd_set_sha256=_hash("fds"),
            cpu_usage_usec=0,
            memory_current_bytes=0,
            memory_peak_bytes=0,
            memory_oom_kill_count=0,
            pids_current=0,
            pids_peak=0,
            pids_max_event_count=0,
            populated=False,
            nr_descendants=0,
            nr_dying_descendants=0,
        )


def test_source_ast_has_no_operational_backend_or_process_container_calls() -> None:
    source_path = Path(host.__file__)
    tree = ast.parse(source_path.read_text())
    forbidden_import_roots = {
        "asyncio",
        "docker",
        "multiprocessing",
        "os",
        "podman",
        "resource",
        "shutil",
        "signal",
        "socket",
        "subprocess",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in forbidden_import_roots
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_import_roots
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {
                    "exec",
                    "eval",
                    "open",
                    "print",
                    "system",
                    "fork",
                }
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "Popen",
                    "run",
                    "system",
                    "fork",
                    "execv",
                    "execve",
                    "create_container",
                    "start_container",
                    "kill",
                }
