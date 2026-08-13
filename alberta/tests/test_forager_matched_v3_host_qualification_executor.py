"""Tests for the nonexecuting matched-v3 host qualification contract."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_host_qualification_executor as host,
)


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


def _rebody(value: dict[str, Any], field: str) -> bytes:
    body = copy.deepcopy(value)
    body.pop(field)
    value[field] = hashlib.sha256(_canonical(body, newline=False)).hexdigest()
    return _canonical(value)


def _ceilings() -> tuple[tuple[str, int], ...]:
    values = {field: 10_000_000_000 for field in host.RESOURCE_CEILING_FIELDS}
    values["max_environment_interactions"] = host.MATCHED_V3_HORIZON
    values["max_thread_count"] = 256
    values["max_attempt_count"] = 1
    values["max_failure_count"] = 0
    return tuple((field, values[field]) for field in host.RESOURCE_CEILING_FIELDS)


def _request(
    *,
    case_ordinal: int = 0,
    image_id: str | None = None,
    qualification_plan_schema_version: str = host.QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
    observation_registry_schema_version: str = (
        host.QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION
    ),
    publisher_registry_complete: bool = False,
    plan_issuance_receipt_sha256: str | None = None,
    case_execution_ticket_sha256: str | None = None,
) -> host.MatchedV3HostQualificationCaseRequest:
    candidate_id = host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[case_ordinal]
    return host.MatchedV3HostQualificationCaseRequest(
        qualification_plan_schema_version=qualification_plan_schema_version,
        qualification_plan_sha256=_sha("plan"),
        qualification_plan_body_sha256=_sha("plan-body"),
        plan_issuance_receipt_sha256=plan_issuance_receipt_sha256,
        observation_registry_schema_version=observation_registry_schema_version,
        observation_registry_descriptor_sha256=_sha("observation-registry"),
        case_execution_ticket_sha256=case_execution_ticket_sha256,
        case_ordinal=case_ordinal,
        candidate_id=candidate_id,
        qualification_case_id=f"matched-v3-q-{case_ordinal:02d}-{candidate_id}",
        qualification_case_manifest_sha256=_sha(f"case-{case_ordinal}"),
        resource_requirement_body_sha256=_sha(f"resource-{case_ordinal}"),
        declared_ceilings=_ceilings(),
        horizon=host.MATCHED_V3_HORIZON,
        attempt_ordinal=0,
        external_source_tree_sha256=_sha("external-tree"),
        local_source_tree_sha256=_sha("local-tree"),
        build_context_receipt_sha256=_sha("build-context"),
        build_execution_receipt_sha256=_sha("build-execution"),
        build_publication_receipt_sha256=_sha("build-publication"),
        image_id=image_id or f"sha256:{_sha('fresh-image')}",
        runtime_identity_sha256=_sha("runtime-identity"),
        runtime_profile_sha256=_sha("runtime-profile"),
        runtime_qualification_receipt_sha256=None,
        resource_observer_descriptor_sha256=(
            host.PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256
        ),
        resource_observer_source_sha256=host.PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
        full_resource_merger_descriptor_sha256=None,
        publisher_registry_sha256=_sha("publisher-registry"),
        publisher_registry_complete=publisher_registry_complete,
        publisher_descriptor_sha256=_sha("publisher-descriptor"),
        publisher_source_sha256=_sha("publisher-source"),
        in_container_driver_descriptor_sha256=_sha("driver-descriptor"),
        in_container_driver_source_sha256=_sha("driver-source"),
        host_provisioning_receipt_sha256=None,
        exact_acknowledgement=host.HOST_QUALIFICATION_EXECUTION_ACKNOWLEDGEMENT,
    )


def _counter_fds() -> tuple[host.HostCgroupV2CounterFdIdentity, ...]:
    result: list[host.HostCgroupV2CounterFdIdentity] = []
    for index, endpoint in enumerate(host.CGROUP_COUNTER_ENDPOINTS, start=1):
        writable = endpoint == "cgroup.kill"
        result.append(
            host.HostCgroupV2CounterFdIdentity(
                endpoint_name=endpoint,
                endpoint_device=31,
                endpoint_inode=1000 + index,
                open_monotonic_ns=100 + index,
                open_flags=(
                    "O_CLOEXEC",
                    "O_NOFOLLOW",
                    "O_WRONLY" if writable else "O_RDONLY",
                ),
                counter_semantics=host.CGROUP_COUNTER_SEMANTICS[endpoint],
                reset_performed=False,
                retained_through_final_sample=True,
                reopened=False,
            )
        )
    return tuple(result)


def _descendant() -> host.HostCgroupV2DescendantIdentity:
    return host.HostCgroupV2DescendantIdentity(
        relative_path="docker-" + "1" * 64 + ".scope",
        device=31,
        inode=303,
    )


def _process() -> host.HostCgroupV2ProcessIdentity:
    return host.HostCgroupV2ProcessIdentity(
        pid=4321,
        start_time_ticks=987_654,
        cgroup_device=31,
        cgroup_inode=303,
    )


def _sample(
    phase: str,
    *,
    timestamp: int,
    cpu: int,
    memory_current: int,
    memory_peak: int,
    pids_current: int,
    pids_peak: int,
    populated: bool,
    descendants: tuple[host.HostCgroupV2DescendantIdentity, ...],
    processes: tuple[host.HostCgroupV2ProcessIdentity, ...],
    threads: tuple[int, ...],
) -> host.HostCgroupV2Sample:
    return host.HostCgroupV2Sample(
        phase=phase,
        monotonic_ns=timestamp,
        cgroup_device=31,
        cgroup_inode=202,
        cpu_usage_usec=cpu,
        memory_current_bytes=memory_current,
        memory_peak_bytes=memory_peak,
        memory_oom_kill_count=0,
        pids_current=pids_current,
        pids_peak=pids_peak,
        pids_max_event_count=0,
        populated=populated,
        frozen=False,
        nr_descendants=len(descendants),
        nr_dying_descendants=0,
        descendant_cgroups=descendants,
        recursive_processes=processes,
        recursive_thread_ids=threads,
    )


def _proof() -> host.MatchedV3HostCgroupV2BoundaryProof:
    descendant = _descendant()
    process = _process()
    samples = (
        _sample(
            "initial_empty",
            timestamp=1000,
            cpu=0,
            memory_current=0,
            memory_peak=0,
            pids_current=0,
            pids_peak=0,
            populated=False,
            descendants=(),
            processes=(),
            threads=(),
        ),
        _sample(
            "driver_ready",
            timestamp=2000,
            cpu=10,
            memory_current=100,
            memory_peak=100,
            pids_current=1,
            pids_peak=1,
            populated=True,
            descendants=(descendant,),
            processes=(process,),
            threads=(4321,),
        ),
        _sample(
            "pre_cleanup",
            timestamp=3000,
            cpu=500,
            memory_current=250,
            memory_peak=500,
            pids_current=1,
            pids_peak=4,
            populated=True,
            descendants=(descendant,),
            processes=(process,),
            threads=(4321,),
        ),
        _sample(
            "post_kill_empty",
            timestamp=4000,
            cpu=550,
            memory_current=10,
            memory_peak=500,
            pids_current=0,
            pids_peak=4,
            populated=False,
            descendants=(descendant,),
            processes=(),
            threads=(),
        ),
        _sample(
            "post_container_remove",
            timestamp=5000,
            cpu=550,
            memory_current=10,
            memory_peak=500,
            pids_current=0,
            pids_peak=4,
            populated=False,
            descendants=(),
            processes=(),
            threads=(),
        ),
    )
    return host.MatchedV3HostCgroupV2BoundaryProof(
        qualification_plan_sha256=_sha("plan"),
        intent_sha256=_sha("intent"),
        case_ordinal=0,
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        delegate_root_path="/sys/fs/cgroup/alberta-matched-v3",
        delegate_root_device=31,
        delegate_root_inode=101,
        case_cgroup_path="/sys/fs/cgroup/alberta-matched-v3/case-00",
        case_cgroup_device=31,
        case_cgroup_inode=202,
        docker_cgroup_parent_argument="alberta-matched-v3/case-00",
        container_id="1" * 64,
        container_name="alberta-matched-v3-q-" + "2" * 32,
        container_cgroup_relative_path=descendant.relative_path,
        container_cgroup_device=descendant.device,
        container_cgroup_inode=descendant.inode,
        target_pid=process.pid,
        target_process_start_time_ticks=process.start_time_ticks,
        pidfd_opened=True,
        proc_cgroup_path="/alberta-matched-v3/case-00/" + descendant.relative_path,
        controllers=("cpu", "memory", "pids"),
        subtree_control=("cpu", "memory", "pids"),
        cgroup_max_depth=1,
        cgroup_max_descendants=1,
        counter_fds=_counter_fds(),
        samples=samples,
        cgroup_namespace_private=True,
        pid_namespace_private=True,
        writable_cgroup_mount_observed=False,
        setsid_changes_cgroup=False,
        fork_clone_inherit_cgroup=True,
        cgroup_kill_supported=True,
        cgroup_kill_written=True,
        direct_cli_child_waited=True,
        direct_cli_child_reaped=True,
        daemon_owned_container_init_directly_reaped=False,
        container_absent_after_cleanup=True,
        case_cgroup_path_absent_after_cleanup=True,
        continuous_membership_proven=False,
        external_privileged_migration_excluded=False,
    )


def _file_records() -> tuple[host.MatchedV3HostPublishedFileMetadata, ...]:
    return (
        host.MatchedV3HostPublishedFileMetadata(
            role="manifest",
            name="publication.json",
            size_bytes=101,
            sha256=_sha("publication-json"),
        ),
        host.MatchedV3HostPublishedFileMetadata(
            role="receipt",
            name="publication-receipt.json",
            size_bytes=202,
            sha256=_sha("publication-receipt-file"),
        ),
    )


def _terminal() -> host.MatchedV3HostContainerTerminalMetadata:
    files = _file_records()
    return host.MatchedV3HostContainerTerminalMetadata(
        operation="publish_and_strict_reload_metadata_only",
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        case_ordinal=0,
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        qualification_plan_sha256=_sha("plan"),
        interaction_horizon=host.MATCHED_V3_HORIZON,
        image_id=f"sha256:{_sha('fresh-image')}",
        driver_descriptor_sha256=_sha("driver-descriptor"),
        driver_source_sha256=_sha("driver-source"),
        execution_receipt_sha256=_sha("candidate-execution"),
        resource_metadata_sha256=_sha("resource-metadata"),
        publisher_descriptor_sha256=_sha("publisher-descriptor"),
        publisher_source_sha256=_sha("publisher-source"),
        publication_address_sha256=_sha("publication-address"),
        publication_manifest_sha256=_sha("publication-manifest"),
        publication_receipt_sha256=_sha("publication-receipt"),
        published_bundle_sha256=_sha("published-bundle"),
        reload_observation_sha256=_sha("reload-observation"),
        file_inventory_sha256=host.matched_v3_host_published_file_inventory_sha256(files),
        file_count=len(files),
        total_size_bytes=sum(item.size_bytes for item in files),
        files=files,
        family_metadata_sha256=_sha("family-metadata"),
        publication_committed=True,
        raw_content_transported=False,
        score_or_reward_decoded=False,
    )


def _ready() -> host.MatchedV3HostContainerReadyMetadata:
    return host.MatchedV3HostContainerReadyMetadata(
        qualification_plan_sha256=_sha("plan"),
        intent_sha256=_sha("intent"),
        case_ordinal=0,
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        image_id=f"sha256:{_sha('fresh-image')}",
        container_id="1" * 64,
        container_name="alberta-matched-v3-q-" + "2" * 32,
        host_pid=4321,
        host_process_start_time_ticks=987_654,
        inner_pid=1,
        proc_cgroup_path=(
            "/alberta-matched-v3/case-00/docker-" + "1" * 64 + ".scope"
        ),
        container_cgroup_device=31,
        container_cgroup_inode=303,
        driver_descriptor_sha256=_sha("driver-descriptor"),
        driver_source_sha256=_sha("driver-source"),
        runtime_identity_sha256=_sha("runtime-identity"),
        runtime_profile_sha256=_sha("runtime-profile"),
        sandbox_observation_sha256=_sha("sandbox-observation"),
        ready_cgroup_sample_sha256=_sha("ready-cgroup-sample"),
        expected_go_payload_sha256=_sha("expected-go"),
        ready_monotonic_ns=2_000,
        stdout_frame_ordinal=0,
        candidate_code_loaded=False,
        outcome_capability_issued=False,
        go_committed=False,
    )


def _handoff() -> host.MatchedV3HostQualificationObservationHandoff:
    return host.MatchedV3HostQualificationObservationHandoff(
        qualification_plan_schema_version=host.QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
        qualification_plan_sha256=_sha("plan"),
        observation_registry_schema_version=(
            host.QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION
        ),
        observation_registry_descriptor_sha256=_sha("observation-registry"),
        case_ordinal=0,
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        intent_sha256=_sha("intent"),
        lifecycle_record_sha256=_sha("lifecycle"),
        host_execution_receipt_sha256=_sha("host-execution"),
        cgroup_boundary_proof_sha256=_sha("cgroup-proof"),
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
        terminal_metadata_sha256=_sha("terminal"),
        runtime_observation_candidate_sha256=_sha("runtime-observation-candidate"),
        candidate_observation_candidate_sha256=_sha("candidate-observation-candidate"),
        fresh_replay_observation_candidate_sha256=_sha("replay-observation-candidate"),
        publication_address_sha256=_sha("publication-address"),
        publication_manifest_sha256=_sha("publication-manifest"),
        publication_receipt_sha256=_sha("publication-receipt"),
        published_bundle_sha256=_sha("published-bundle"),
        reload_observation_sha256=_sha("reload-observation"),
    )


def _linked_chain() -> dict[str, Any]:
    request = _request()
    intent = host.build_matched_v3_host_qualification_case_intent(request)
    intent_sha256 = hashlib.sha256(
        host.canonical_matched_v3_host_qualification_case_intent_bytes(intent)
    ).hexdigest()
    proof = dataclasses.replace(
        _proof(),
        qualification_plan_sha256=request.qualification_plan_sha256,
        intent_sha256=intent_sha256,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
    )
    ready = dataclasses.replace(
        _ready(),
        qualification_plan_sha256=request.qualification_plan_sha256,
        intent_sha256=intent_sha256,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        image_id=request.image_id,
        driver_descriptor_sha256=request.in_container_driver_descriptor_sha256,
        driver_source_sha256=request.in_container_driver_source_sha256,
        runtime_profile_sha256=request.runtime_profile_sha256,
        ready_cgroup_sample_sha256=host.matched_v3_host_cgroup_sample_sha256(
            proof.samples[1]
        ),
    )
    commitment = host.build_matched_v3_host_qualification_go_commitment(
        ready,
        go_commitment_monotonic_ns=2_100,
    )
    terminal = dataclasses.replace(
        _terminal(),
        qualification_plan_sha256=request.qualification_plan_sha256,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        image_id=request.image_id,
        driver_descriptor_sha256=request.in_container_driver_descriptor_sha256,
        driver_source_sha256=request.in_container_driver_source_sha256,
        publisher_descriptor_sha256=request.publisher_descriptor_sha256,
        publisher_source_sha256=request.publisher_source_sha256,
    )
    pre_receipt = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_PRE_RECEIPT_PHASES,
        failure_phase=None,
        uncertainty_kind=None,
    )
    execution_receipt = host.build_matched_v3_host_qualification_case_execution_receipt(
        request=request,
        intent=intent,
        ready=ready,
        commitment=commitment,
        cgroup_proof=proof,
        pre_receipt_lifecycle=pre_receipt,
        terminal=terminal,
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
    )
    pre_handoff = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_PRE_HANDOFF_PHASES,
        failure_phase=None,
        uncertainty_kind=None,
    )
    handoff = host.build_matched_v3_host_qualification_observation_handoff(
        request=request,
        intent=intent,
        ready=ready,
        commitment=commitment,
        cgroup_proof=proof,
        pre_receipt_lifecycle=pre_receipt,
        execution_receipt=execution_receipt,
        pre_handoff_lifecycle=pre_handoff,
        terminal=terminal,
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
        runtime_observation_candidate_sha256=_sha("runtime-observation-candidate"),
        candidate_observation_candidate_sha256=_sha("candidate-observation-candidate"),
        fresh_replay_observation_candidate_sha256=_sha("replay-observation-candidate"),
    )
    return {
        "request": request,
        "intent": intent,
        "ready": ready,
        "commitment": commitment,
        "proof": proof,
        "pre_receipt": pre_receipt,
        "terminal": terminal,
        "execution_receipt": execution_receipt,
        "pre_handoff": pre_handoff,
        "handoff": handoff,
    }


def test_descriptor_is_pinned_nonexecuting_and_incompatible() -> None:
    descriptor = host.matched_v3_host_qualification_executor_descriptor()
    raw = host.canonical_matched_v3_host_qualification_executor_descriptor_bytes()

    assert hashlib.sha256(raw).hexdigest() == (
        host.PINNED_HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256
    )
    assert host.parse_matched_v3_host_qualification_executor_descriptor(raw) == descriptor
    assert descriptor["candidate_order"] == list(
        host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS
    )
    assert descriptor["horizon"] == 499_712
    assert descriptor["readiness"]["execution_ready"] is False
    assert descriptor["runtime_surface"] == {
        "cgroup_writes_implemented": False,
        "docker_or_oci_invocation_implemented": False,
        "production_process_runner_implemented": False,
        "production_provisioner_implemented": False,
        "subprocess_imported": False,
    }
    assert descriptor["compatibility"][host.QUALIFICATION_PLAN_V2_SCHEMA_VERSION] is False
    assert (
        descriptor["compatibility"][
            host.QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION
        ]
        is False
    )
    observer = descriptor["dependency_pins"]["endpoint_resource_observer"]
    assert observer["descriptor_sha256"] == (
        host.PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256
    )
    assert observer["source_sha256"] == host.PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256
    assert observer["complete_28_field_observation"] is False
    assert "resource_observer_v1_cannot_complete_28_field_observation" in (
        descriptor["readiness"]["blockers"]
    )
    assert "parse_matched_v3_host_qualification_case_execution_receipt" in (
        descriptor["public_api"]["structural_parsers"]
    )
    assert "build_matched_v3_host_qualification_case_execution_receipt" in (
        descriptor["public_api"]["cross_linked_content_builders"]
    )
    assert "validate_matched_v3_host_qualification_observation_handoff_chain" in (
        descriptor["public_api"]["cross_link_validators"]
    )
    assert not any(descriptor["authority"].values())
    assert not any(descriptor["claims"].values())


def test_descriptor_parser_rejects_noncanonical_or_drifted_content() -> None:
    raw = host.canonical_matched_v3_host_qualification_executor_descriptor_bytes()
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.parse_matched_v3_host_qualification_executor_descriptor(raw.rstrip(b"\n"))

    value = json.loads(raw)
    value["readiness"]["execution_ready"] = True
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.parse_matched_v3_host_qualification_executor_descriptor(_canonical(value))


def test_request_is_exact_roundtrippable_and_never_ready() -> None:
    request = _request()
    raw = host.canonical_matched_v3_host_qualification_case_request_bytes(request)
    replayed = host.parse_matched_v3_host_qualification_case_request(raw)
    assert replayed == request
    assert host.replay_matched_v3_host_qualification_case_request(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    ) == request

    readiness = host.matched_v3_host_qualification_readiness(request)
    assert readiness["execution_ready"] is False
    assert readiness["production_mutation_permitted"] is False
    assert "qualification_plan_v2_is_content_only_and_incomplete" in readiness["blockers"]
    assert "qualification_observation_registry_v1_has_no_issuer" in readiness["blockers"]
    assert "no_plan_issuance_receipt" in readiness["blockers"]
    assert "no_case_execution_ticket" in readiness["blockers"]
    assert "publisher_registry_28_of_28_not_bound" in readiness["blockers"]


@pytest.mark.parametrize(
    "ordinal",
    (0, 13, 14, 22, 23, 24, 25, 26, 27),
)
def test_request_binds_literal_global_candidate_order(ordinal: int) -> None:
    request = _request(case_ordinal=ordinal)
    assert request.candidate_id == host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[ordinal]

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            request,
            candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[
                (ordinal + 1) % len(host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS)
            ],
        )


def test_request_rejects_wrong_horizon_ceiling_order_ack_and_attempt() -> None:
    request = _request()
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, horizon=499_711)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, attempt_ordinal=1)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, exact_acknowledgement="yes")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, declared_ceilings=tuple(reversed(request.declared_ceilings)))

    larger_horizon = dict(request.declared_ceilings)
    larger_horizon["max_environment_interactions"] = host.MATCHED_V3_HORIZON + 1
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            request,
            declared_ceilings=tuple(
                (field, larger_horizon[field]) for field in host.RESOURCE_CEILING_FIELDS
            ),
        )

    retriable = dict(request.declared_ceilings)
    retriable["max_attempt_count"] = 2
    retriable["max_failure_count"] = 1
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            request,
            declared_ceilings=tuple(
                (field, retriable[field]) for field in host.RESOURCE_CEILING_FIELDS
            ),
        )


def test_request_pins_existing_endpoint_observer_but_keeps_full_merger_blocker() -> None:
    request = _request()
    assert request.resource_observer_descriptor_sha256 == (
        "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
    )
    assert request.resource_observer_source_sha256 == (
        "4d34951ccb4b265caa29794457cdd8a5dd837ecf4b73b7a44e4f849bf8c8106e"
    )
    assert request.full_resource_merger_descriptor_sha256 is None
    assert "resource_observer_v1_cannot_complete_28_field_observation" in (
        host.matched_v3_host_qualification_readiness(request)["blockers"]
    )

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, resource_observer_descriptor_sha256=_sha("wrong"))
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(request, resource_observer_source_sha256=_sha("wrong"))


@pytest.mark.parametrize("image_id", host.STALE_IMAGE_IDS)
def test_request_rejects_every_historical_image(image_id: str) -> None:
    with pytest.raises(
        host.ForagerMatchedV3HostQualificationExecutorError,
        match="historical|stale",
    ):
        _request(image_id=image_id)


def test_request_rejects_each_component_of_stale_source_closure_lineage() -> None:
    request = _request()
    stale = host.STALE_BUILD_LINEAGES[0]
    replacements: dict[str, Any] = {
        "build_context_receipt_sha256": stale["context_receipt_sha256"],
        "build_execution_receipt_sha256": stale["execution_receipt_sha256"],
        "build_publication_receipt_sha256": stale["publication_receipt_sha256"],
    }
    for field, value in replacements.items():
        with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
            dataclasses.replace(request, **{field: value})


def test_intent_is_deterministic_nonmutating_and_nonretrying() -> None:
    request = _request()
    intent = host.build_matched_v3_host_qualification_case_intent(request)
    raw = host.canonical_matched_v3_host_qualification_case_intent_bytes(intent)
    assert host.parse_matched_v3_host_qualification_case_intent(raw) == intent
    assert host.build_matched_v3_host_qualification_case_intent(request) == intent
    assert intent.runtime_identity_sha256 == request.runtime_identity_sha256
    value = json.loads(raw)
    assert value["authority"]["execution_authorized"] is False
    assert value["policy"]["same_case_retry_permitted"] is False
    assert value["policy"]["production_mutation_permitted"] is False


def test_ready_record_precedes_one_way_go_and_binds_pid_start_cgroup() -> None:
    ready = _ready()
    raw = host.canonical_matched_v3_host_container_ready_metadata_bytes(ready)
    assert host.parse_matched_v3_host_container_ready_metadata(raw) == ready
    assert ready.host_pid == 4321
    assert ready.host_process_start_time_ticks == 987_654
    assert ready.candidate_code_loaded is False
    assert ready.outcome_capability_issued is False
    assert ready.go_committed is False

    go = host.build_matched_v3_host_qualification_go_commitment(
        ready,
        go_commitment_monotonic_ns=2_100,
    )
    go_raw = host.canonical_matched_v3_host_qualification_go_commitment_bytes(go)
    assert host.parse_matched_v3_host_qualification_go_commitment(go_raw) == go
    go_value = json.loads(go_raw)
    assert go.ready_metadata_sha256 == hashlib.sha256(raw).hexdigest()
    assert go.go_payload_sha256 == ready.expected_go_payload_sha256
    assert go_value["policy"]["one_way_commitment"] is True
    assert go_value["policy"]["same_case_retry_permitted"] is False
    assert not any(go_value["authority"].values())

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(ready, candidate_code_loaded=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_go_commitment(
            ready,
            go_commitment_monotonic_ns=1_999,
        )


@pytest.mark.parametrize(
    "proc_path",
    (
        "//alberta-matched-v3/case-00/docker-" + "1" * 64 + ".scope",
        "/alberta-matched-v3/case-00/./docker-" + "1" * 64 + ".scope",
        "/alberta-matched-v3//case-00/docker-" + "1" * 64 + ".scope",
    ),
)
def test_ready_rejects_noncanonical_proc_cgroup_spelling(proc_path: str) -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(_ready(), proc_cgroup_path=proc_path)


def test_cgroup_boundary_proof_roundtrip_and_fresh_counter_semantics() -> None:
    proof = _proof()
    raw = host.canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes(proof)
    assert host.parse_matched_v3_host_cgroup_v2_boundary_proof(raw) == proof
    assert proof.samples[0].pids_peak == 0
    assert proof.samples[0].memory_peak_bytes == 0
    pids = next(item for item in proof.counter_fds if item.endpoint_name == "pids.peak")
    memory = next(item for item in proof.counter_fds if item.endpoint_name == "memory.peak")
    assert pids.counter_semantics == "fresh_cgroup_required_read_only_never_resettable"
    assert memory.counter_semantics == "fresh_cgroup_since_creation_retained_fd_no_reopen"
    assert pids.retained_through_final_sample and not pids.reopened


@pytest.mark.parametrize("path", (".", "..", "a/..", "a/./b", "a//b"))
def test_relative_metadata_paths_reject_dot_and_traversal_spellings(path: str) -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.HostCgroupV2DescendantIdentity(relative_path=path, device=31, inode=303)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.MatchedV3HostPublishedFileMetadata(
            role="artifact",
            name=path,
            size_bytes=0,
            sha256=_sha("artifact"),
        )


def test_cgroup_boundary_rejects_nonempty_initial_nested_drift_and_reopen() -> None:
    proof = _proof()
    initial = dataclasses.replace(proof.samples[0], pids_peak=1)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, samples=(initial, *proof.samples[1:]))

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof.samples[1], nr_descendants=2)

    drifted = dataclasses.replace(proof.samples[2], cgroup_inode=999)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(proof.samples[0], proof.samples[1], drifted, *proof.samples[3:]),
        )

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof.counter_fds[0], reopened=True)


@pytest.mark.parametrize(
    "proc_path",
    (
        "//alberta-matched-v3/case-00/docker-" + "1" * 64 + ".scope",
        "/wrong-prefix/case-00/docker-" + "1" * 64 + ".scope",
        "/alberta-matched-v3/case-00/./docker-" + "1" * 64 + ".scope",
        "/alberta-matched-v3//case-00/docker-" + "1" * 64 + ".scope",
    ),
)
def test_cgroup_boundary_rejects_noncanonical_or_wrong_prefix_proc_path(
    proc_path: str,
) -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(_proof(), proc_cgroup_path=proc_path)


def test_cgroup_boundary_rejects_pid_reuse_escape_and_false_portable_claims() -> None:
    proof = _proof()
    reused = dataclasses.replace(_process(), start_time_ticks=987_655)
    ready = dataclasses.replace(proof.samples[1], recursive_processes=(reused,))
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, samples=(proof.samples[0], ready, *proof.samples[2:]))

    escaped = dataclasses.replace(_process(), cgroup_inode=404)
    escaped_descendant = dataclasses.replace(_descendant(), inode=404)
    ready = dataclasses.replace(
        proof.samples[1],
        descendant_cgroups=(escaped_descendant,),
        recursive_processes=(escaped,),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, samples=(proof.samples[0], ready, *proof.samples[2:]))

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, continuous_membership_proven=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, external_privileged_migration_excluded=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, daemon_owned_container_init_directly_reaped=True)


def test_cgroup_intermediate_samples_retain_one_exact_ready_descendant() -> None:
    proof = _proof()
    ready_descendant = proof.samples[1].descendant_cgroups[0]
    extra = host.HostCgroupV2DescendantIdentity(
        relative_path="other.scope",
        device=31,
        inode=304,
    )
    pre_cleanup_extra = dataclasses.replace(
        proof.samples[2],
        nr_descendants=2,
        descendant_cgroups=(ready_descendant, extra),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(*proof.samples[:2], pre_cleanup_extra, *proof.samples[3:]),
        )

    nested = dataclasses.replace(ready_descendant, relative_path="nested/container.scope")
    pre_cleanup_nested = dataclasses.replace(
        proof.samples[2],
        descendant_cgroups=(nested,),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(*proof.samples[:2], pre_cleanup_nested, *proof.samples[3:]),
        )

    swapped = dataclasses.replace(ready_descendant, inode=304)
    swapped_process = dataclasses.replace(_process(), cgroup_inode=304)
    pre_cleanup_swapped = dataclasses.replace(
        proof.samples[2],
        descendant_cgroups=(swapped,),
        recursive_processes=(swapped_process,),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(*proof.samples[:2], pre_cleanup_swapped, *proof.samples[3:]),
        )

    post_kill_swapped = dataclasses.replace(
        proof.samples[3],
        descendant_cgroups=(swapped,),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(*proof.samples[:3], post_kill_swapped, proof.samples[4]),
        )


def test_cgroup_boundary_rejects_freezer_device_and_inode_alias_claims() -> None:
    proof = _proof()
    frozen = dataclasses.replace(proof.samples[2], frozen=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            proof,
            samples=(*proof.samples[:2], frozen, *proof.samples[3:]),
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, delegate_root_device=32)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, case_cgroup_inode=proof.delegate_root_inode)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, container_cgroup_inode=proof.case_cgroup_inode)
    aliased_fd = dataclasses.replace(
        proof.counter_fds[0],
        endpoint_inode=proof.case_cgroup_inode,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(proof, counter_fds=(aliased_fd, *proof.counter_fds[1:]))


def test_terminal_metadata_is_exact_horizon_metadata_only_and_roundtrippable() -> None:
    terminal = _terminal()
    raw = host.canonical_matched_v3_host_container_terminal_metadata_bytes(terminal)
    assert host.parse_matched_v3_host_container_terminal_metadata(raw) == terminal
    assert terminal.interaction_horizon == 499_712
    assert terminal.raw_content_transported is False
    assert terminal.score_or_reward_decoded is False

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(terminal, interaction_horizon=499_711)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(terminal, publication_committed=False)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(terminal, total_size_bytes=1)


def test_terminal_metadata_roundtrips_empty_upstream_video_slot() -> None:
    base = _terminal()
    empty_video = host.MatchedV3HostPublishedFileMetadata(
        role="upstream_video_slot",
        name="upstream-video-slot.bin",
        size_bytes=0,
        sha256=_sha("empty-video-slot"),
    )
    files = (*base.files, empty_video)
    terminal = dataclasses.replace(
        base,
        files=files,
        file_count=len(files),
        total_size_bytes=sum(item.size_bytes for item in files),
        file_inventory_sha256=host.matched_v3_host_published_file_inventory_sha256(files),
    )
    raw = host.canonical_matched_v3_host_container_terminal_metadata_bytes(terminal)
    assert host.parse_matched_v3_host_container_terminal_metadata(raw) == terminal


def test_terminal_metadata_rejects_over_count_and_over_aggregate_inventory() -> None:
    too_many = tuple(
        host.MatchedV3HostPublishedFileMetadata(
            role=f"role_{index}",
            name=f"files/{index}.bin",
            size_bytes=1,
            sha256=_sha(f"file-{index}"),
        )
        for index in range(129)
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            _terminal(),
            files=too_many,
            file_count=len(too_many),
            total_size_bytes=len(too_many),
        )

    oversized = (
        host.MatchedV3HostPublishedFileMetadata(
            role="oversized",
            name="oversized.bin",
            size_bytes=1024 * 1024 * 1024 + 1,
            sha256=_sha("oversized"),
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(
            _terminal(),
            files=oversized,
            file_count=1,
            total_size_bytes=oversized[0].size_bytes,
            file_inventory_sha256=(
                host.matched_v3_host_published_file_inventory_sha256(oversized)
            ),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"score": 1},
        {"nested": {"cumulative_reward": 2}},
        {"items": [{"raw_result": "bytes"}]},
        {"items": [{"deeper": [{"ranking": [1, 2]}]}]},
        {"items": [[[{"performance_score": 3}]]]},
    ),
)
def test_metadata_forbidden_keys_are_rejected_recursively(payload: dict[str, Any]) -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_metadata_only_mapping(payload)


def test_terminal_parser_rejects_forbidden_nested_key_before_unknown_field() -> None:
    terminal = _terminal()
    raw = host.canonical_matched_v3_host_container_terminal_metadata_bytes(terminal)
    value = json.loads(raw)
    value["extra"] = {"nested": [{"score": 100}]}
    with pytest.raises(
        host.ForagerMatchedV3HostQualificationExecutorError,
        match="forbidden",
    ):
        host.parse_matched_v3_host_container_terminal_metadata(
            _rebody(value, "terminal_metadata_body_sha256")
        )


def test_handoff_is_structural_only_and_current_registry_incompatible() -> None:
    handoff = _handoff()
    raw = host.canonical_matched_v3_host_qualification_observation_handoff_bytes(handoff)
    assert host.parse_matched_v3_host_qualification_observation_handoff(raw) == handoff
    value = json.loads(raw)
    assert value["compatibility"] == {
        "observation_registry_v1_compatible": False,
        "qualification_plan_v2_compatible": False,
    }
    assert value["metadata_policy"]["raw_content_transported"] is False
    assert value["metadata_policy"]["score_or_reward_decoded"] is False
    assert not any(value["authority"].values())
    assert not any(value["claims"].values())


def test_handoff_builder_binds_receipt_then_exact_pre_handoff_checkpoint() -> None:
    chain = _linked_chain()
    handoff = chain["handoff"]
    host.validate_matched_v3_host_qualification_observation_handoff_chain(
        handoff,
        request=chain["request"],
        intent=chain["intent"],
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        pre_receipt_lifecycle=chain["pre_receipt"],
        execution_receipt=chain["execution_receipt"],
        pre_handoff_lifecycle=chain["pre_handoff"],
        terminal=chain["terminal"],
        expected_resource_observation_request_sha256=_sha("resource-request"),
        expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
        expected_runtime_observation_candidate_sha256=_sha(
            "runtime-observation-candidate"
        ),
        expected_candidate_observation_candidate_sha256=_sha(
            "candidate-observation-candidate"
        ),
        expected_fresh_replay_observation_candidate_sha256=_sha(
            "replay-observation-candidate"
        ),
    )
    assert chain["pre_handoff"].completed_phases == (
        host.HOST_QUALIFICATION_PRE_HANDOFF_PHASES
    )
    assert chain["pre_handoff"].terminal_state == (
        "receipt_committed_ready_for_handoff_non_authorizing"
    )

    wrong_handoff = dataclasses.replace(
        handoff,
        publication_manifest_sha256=_sha("unrelated-publication-manifest"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_observation_handoff_chain(
            wrong_handoff,
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            execution_receipt=chain["execution_receipt"],
            pre_handoff_lifecycle=chain["pre_handoff"],
            terminal=chain["terminal"],
            expected_resource_observation_request_sha256=_sha("resource-request"),
            expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
            expected_runtime_observation_candidate_sha256=_sha(
                "runtime-observation-candidate"
            ),
            expected_candidate_observation_candidate_sha256=_sha(
                "candidate-observation-candidate"
            ),
            expected_fresh_replay_observation_candidate_sha256=_sha(
                "replay-observation-candidate"
            ),
        )

    wrong_candidate = dataclasses.replace(
        handoff,
        runtime_observation_candidate_sha256=_sha("unrelated-runtime-candidate"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_observation_handoff_chain(
            wrong_candidate,
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            execution_receipt=chain["execution_receipt"],
            pre_handoff_lifecycle=chain["pre_handoff"],
            terminal=chain["terminal"],
            expected_resource_observation_request_sha256=_sha("resource-request"),
            expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
            expected_runtime_observation_candidate_sha256=_sha(
                "runtime-observation-candidate"
            ),
            expected_candidate_observation_candidate_sha256=_sha(
                "candidate-observation-candidate"
            ),
            expected_fresh_replay_observation_candidate_sha256=_sha(
                "replay-observation-candidate"
            ),
        )

    final_lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES,
        failure_phase=None,
        uncertainty_kind=None,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_observation_handoff(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            execution_receipt=chain["execution_receipt"],
            pre_handoff_lifecycle=final_lifecycle,
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
            runtime_observation_candidate_sha256=_sha("runtime-observation-candidate"),
            candidate_observation_candidate_sha256=_sha(
                "candidate-observation-candidate"
            ),
            fresh_replay_observation_candidate_sha256=_sha(
                "replay-observation-candidate"
            ),
        )


def test_execution_receipt_binds_lifecycle_terminal_cgroup_and_stays_non_authorizing() -> None:
    receipt = host.MatchedV3HostQualificationCaseExecutionReceipt(
        qualification_plan_sha256=_sha("plan"),
        request_sha256=_sha("request"),
        intent_sha256=_sha("intent"),
        ready_metadata_sha256=_sha("ready"),
        go_commitment_sha256=_sha("go"),
        lifecycle_record_sha256=_sha("lifecycle"),
        cgroup_boundary_proof_sha256=_sha("cgroup-proof"),
        terminal_metadata_sha256=_sha("terminal"),
        case_ordinal=0,
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        image_id=f"sha256:{_sha('fresh-image')}",
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
        publication_address_sha256=_sha("publication-address"),
        publication_receipt_sha256=_sha("publication-receipt"),
        returncode=0,
        timed_out=False,
        execution_state="metadata_complete_non_authorizing",
        publication_state="committed",
        cleanup_state="proven_empty",
        case_consumed=True,
        same_case_retry_permitted=False,
    )
    raw = host.canonical_matched_v3_host_qualification_case_execution_receipt_bytes(receipt)
    assert host.parse_matched_v3_host_qualification_case_execution_receipt(raw) == receipt
    value = json.loads(raw)
    assert value["readiness"]["qualification_evaluated"] is False
    assert value["readiness"]["execution_ready"] is False
    assert not any(value["authority"].values())
    assert not any(value["claims"].values())

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(receipt, same_case_retry_permitted=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(receipt, case_consumed=False)


def test_execution_receipt_builder_crosslinks_exact_pre_receipt_chain() -> None:
    chain = _linked_chain()
    receipt = chain["execution_receipt"]
    host.validate_matched_v3_host_qualification_case_execution_receipt_chain(
        receipt,
        request=chain["request"],
        intent=chain["intent"],
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        pre_receipt_lifecycle=chain["pre_receipt"],
        terminal=chain["terminal"],
        expected_resource_observation_request_sha256=_sha("resource-request"),
        expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
    )
    assert chain["pre_receipt"].completed_phases == (
        host.HOST_QUALIFICATION_PRE_RECEIPT_PHASES
    )
    assert chain["pre_receipt"].terminal_state == (
        "postflight_complete_ready_for_receipt_non_authorizing"
    )

    wrong_ready = dataclasses.replace(
        chain["ready"],
        qualification_plan_sha256=_sha("wrong-plan"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=wrong_ready,
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    wrong_terminal = dataclasses.replace(
        chain["terminal"],
        publisher_descriptor_sha256=_sha("wrong-publisher"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=wrong_terminal,
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_case_execution_receipt_chain(
            dataclasses.replace(receipt, request_sha256=_sha("unrelated-request")),
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            expected_resource_observation_request_sha256=_sha("resource-request"),
            expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_case_execution_receipt_chain(
            dataclasses.replace(
                receipt,
                resource_observation_receipt_sha256=_sha("unrelated-resource-receipt"),
            ),
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            expected_resource_observation_request_sha256=_sha("resource-request"),
            expected_resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    final_lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES,
        failure_phase=None,
        uncertainty_kind=None,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=final_lifecycle,
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )


def test_execution_builder_rejects_one_field_go_proof_and_cgroup_crosswires() -> None:
    chain = _linked_chain()
    wrong_go = dataclasses.replace(chain["commitment"], host_pid=9999)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=wrong_go,
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    late_go = host.build_matched_v3_host_qualification_go_commitment(
        chain["ready"],
        go_commitment_monotonic_ns=3_500,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=late_go,
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    wrong_runtime_ready = dataclasses.replace(
        chain["ready"],
        runtime_identity_sha256=_sha("unrelated-runtime-identity"),
    )
    wrong_runtime_go = host.build_matched_v3_host_qualification_go_commitment(
        wrong_runtime_ready,
        go_commitment_monotonic_ns=2_100,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=wrong_runtime_ready,
            commitment=wrong_runtime_go,
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    wrong_proof = dataclasses.replace(
        chain["proof"],
        qualification_plan_sha256=_sha("unrelated-plan"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=wrong_proof,
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )

    wrong_ready_cgroup = dataclasses.replace(chain["ready"], container_cgroup_device=32)
    wrong_ready_go = host.build_matched_v3_host_qualification_go_commitment(
        wrong_ready_cgroup,
        go_commitment_monotonic_ns=2_100,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_execution_receipt(
            request=chain["request"],
            intent=chain["intent"],
            ready=wrong_ready_cgroup,
            commitment=wrong_ready_go,
            cgroup_proof=chain["proof"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            terminal=chain["terminal"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )


def test_failure_receipt_preserves_consumed_uncertain_nonretryable_state() -> None:
    failure = host.MatchedV3HostQualificationCaseFailureReceipt(
        qualification_plan_sha256=_sha("plan"),
        request_sha256=_sha("request"),
        intent_sha256=_sha("intent"),
        lifecycle_record_sha256=_sha("lifecycle"),
        ready_metadata_sha256=_sha("ready"),
        go_commitment_sha256=_sha("go"),
        host_execution_receipt_sha256=None,
        host_execution_receipt_state="not_committed",
        case_ordinal=0,
        candidate_id=host.MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[0],
        qualification_case_id="matched-v3-q-00-causal-e025-q050",
        qualification_case_manifest_sha256=_sha("case-0"),
        image_id=f"sha256:{_sha('fresh-image')}",
        failure_phase="terminal_metadata_validated",
        exception_type="TerminalMetadataError",
        error_message_sha256=_sha("safe-error-message"),
        case_start_state="started",
        container_state="uncertain",
        publication_state="uncertain",
        cgroup_cleanup_state="uncertain",
        cgroup_boundary_proof_sha256=None,
        terminal_metadata_sha256=None,
        case_consumed=True,
        same_case_retry_permitted=False,
    )
    raw = host.canonical_matched_v3_host_qualification_case_failure_receipt_bytes(failure)
    assert host.parse_matched_v3_host_qualification_case_failure_receipt(raw) == failure
    value = json.loads(raw)
    assert value["classification"] == "case_state_uncertain_non_retriable"
    assert value["policy"]["same_case_retry_permitted"] is False
    assert value["policy"]["clean_rejection_recorded"] is False
    assert not any(value["authority"].values())

    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(failure, same_case_retry_permitted=True)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        dataclasses.replace(failure, case_consumed=False)


def test_failure_builder_derives_phase_states_and_rejects_unrelated_links() -> None:
    chain = _linked_chain()
    terminal_index = host.HOST_QUALIFICATION_LIFECYCLE_PHASES.index(
        "terminal_metadata_validated"
    )
    lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES[:terminal_index],
        failure_phase="terminal_metadata_validated",
        uncertainty_kind="publication_state",
    )
    failure = host.build_matched_v3_host_qualification_case_failure_receipt(
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        exception_type="TerminalMetadataError",
        error_message_sha256=_sha("safe-error-message"),
        ready=chain["ready"],
        commitment=chain["commitment"],
    )
    assert failure.case_start_state == "started"
    assert failure.publication_state == "committed"
    assert failure.cgroup_cleanup_state == "uncertain"
    host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
        failure,
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        expected_exception_type="TerminalMetadataError",
        expected_error_message_sha256=_sha("safe-error-message"),
        ready=chain["ready"],
        commitment=chain["commitment"],
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
            dataclasses.replace(failure, request_sha256=_sha("unrelated-request")),
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=lifecycle,
            expected_exception_type="TerminalMetadataError",
            expected_error_message_sha256=_sha("safe-error-message"),
            ready=chain["ready"],
            commitment=chain["commitment"],
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
            dataclasses.replace(failure, exception_type="UnrelatedFailure"),
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=lifecycle,
            expected_exception_type="TerminalMetadataError",
            expected_error_message_sha256=_sha("safe-error-message"),
            ready=chain["ready"],
            commitment=chain["commitment"],
        )

    early_lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=("request_validated", "intent_committed"),
        failure_phase="fresh_cgroup_created",
        uncertainty_kind="cleanup_state",
    )
    early_failure = host.build_matched_v3_host_qualification_case_failure_receipt(
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=early_lifecycle,
        exception_type="FreshCgroupError",
        error_message_sha256=_sha("fresh-cgroup-error"),
    )
    assert early_failure.case_start_state == "not_started"
    assert early_failure.container_state == "known_absent"
    assert early_failure.publication_state == "not_started"
    assert early_failure.cgroup_cleanup_state == "uncertain"

    intent_boundary = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=("request_validated",),
        failure_phase="intent_committed",
        uncertainty_kind="receipt_state",
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_failure_receipt(
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=intent_boundary,
            exception_type="IntentCommitError",
            error_message_sha256=_sha("intent-error"),
        )


def test_handoff_phase_failure_requires_exact_committed_host_receipt() -> None:
    chain = _linked_chain()
    lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_PRE_HANDOFF_PHASES,
        failure_phase="handoff_committed",
        uncertainty_kind="receipt_state",
    )
    failure = host.build_matched_v3_host_qualification_case_failure_receipt(
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        exception_type="HandoffCommitError",
        error_message_sha256=_sha("handoff-commit-error"),
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        terminal=chain["terminal"],
        execution_receipt=chain["execution_receipt"],
        pre_receipt_lifecycle=chain["pre_receipt"],
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
    )
    expected_receipt_sha256 = hashlib.sha256(
        host.canonical_matched_v3_host_qualification_case_execution_receipt_bytes(
            chain["execution_receipt"]
        )
    ).hexdigest()
    assert failure.host_execution_receipt_sha256 == expected_receipt_sha256
    assert failure.host_execution_receipt_state == "committed"
    host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
        failure,
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        expected_exception_type="HandoffCommitError",
        expected_error_message_sha256=_sha("handoff-commit-error"),
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        terminal=chain["terminal"],
        execution_receipt=chain["execution_receipt"],
        pre_receipt_lifecycle=chain["pre_receipt"],
        resource_observation_request_sha256=_sha("resource-request"),
        resource_observation_receipt_sha256=_sha("resource-receipt"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
            dataclasses.replace(
                failure,
                host_execution_receipt_sha256=_sha("unrelated-prior-receipt"),
            ),
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=lifecycle,
            expected_exception_type="HandoffCommitError",
            expected_error_message_sha256=_sha("handoff-commit-error"),
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            terminal=chain["terminal"],
            execution_receipt=chain["execution_receipt"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_failure_receipt(
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=lifecycle,
            exception_type="HandoffCommitError",
            error_message_sha256=_sha("handoff-commit-error"),
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            terminal=chain["terminal"],
        )


def test_receipt_commit_boundary_is_uncertain_without_self_claimed_receipt() -> None:
    chain = _linked_chain()
    lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_PRE_RECEIPT_PHASES,
        failure_phase="receipt_committed",
        uncertainty_kind="receipt_state",
    )
    failure = host.build_matched_v3_host_qualification_case_failure_receipt(
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        exception_type="ReceiptCommitError",
        error_message_sha256=_sha("receipt-commit-error"),
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        terminal=chain["terminal"],
    )
    assert failure.host_execution_receipt_state == "commit_uncertain"
    assert failure.host_execution_receipt_sha256 is None
    host.validate_matched_v3_host_qualification_case_failure_receipt_chain(
        failure,
        request=chain["request"],
        intent=chain["intent"],
        lifecycle=lifecycle,
        expected_exception_type="ReceiptCommitError",
        expected_error_message_sha256=_sha("receipt-commit-error"),
        ready=chain["ready"],
        commitment=chain["commitment"],
        cgroup_proof=chain["proof"],
        terminal=chain["terminal"],
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.build_matched_v3_host_qualification_case_failure_receipt(
            request=chain["request"],
            intent=chain["intent"],
            lifecycle=lifecycle,
            exception_type="ReceiptCommitError",
            error_message_sha256=_sha("receipt-commit-error"),
            ready=chain["ready"],
            commitment=chain["commitment"],
            cgroup_proof=chain["proof"],
            terminal=chain["terminal"],
            execution_receipt=chain["execution_receipt"],
            pre_receipt_lifecycle=chain["pre_receipt"],
            resource_observation_request_sha256=_sha("resource-request"),
            resource_observation_receipt_sha256=_sha("resource-receipt"),
        )


def test_lifecycle_success_is_still_non_authorizing() -> None:
    lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES,
        failure_phase=None,
        uncertainty_kind=None,
    )
    raw = host.canonical_matched_v3_host_qualification_lifecycle_record_bytes(lifecycle)
    assert host.parse_matched_v3_host_qualification_lifecycle_record(raw) == lifecycle
    assert lifecycle.terminal_state == "metadata_handoff_recorded_non_authorizing"
    assert lifecycle.handoff_committed is True
    assert lifecycle.same_case_retry_permitted is False
    assert lifecycle.qualification_evaluated is False


@pytest.mark.parametrize(
    ("completed_count", "failure_phase", "uncertainty_kind"),
    (
        (1, "intent_committed", "receipt_state"),
        (8, "workload_exited", "observation_state"),
        (9, "publication_committed", "publication_state"),
        (10, "terminal_metadata_validated", "publication_state"),
        (14, "receipt_committed", "receipt_state"),
        (15, "handoff_committed", "receipt_state"),
    ),
)
def test_post_intent_and_post_execution_failures_are_nonretryable_uncertain(
    completed_count: int,
    failure_phase: str,
    uncertainty_kind: str,
) -> None:
    lifecycle = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES[:completed_count],
        failure_phase=failure_phase,
        uncertainty_kind=uncertainty_kind,
    )
    assert lifecycle.same_case_retry_permitted is False
    assert lifecycle.qualification_evaluated is False
    if "case_started" in lifecycle.completed_phases:
        assert lifecycle.case_may_have_started is True
        assert lifecycle.terminal_state == "case_state_uncertain_non_retriable"


def test_lifecycle_boundary_uncertainty_is_ordinal_and_conservative() -> None:
    intent_boundary = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=("request_validated",),
        failure_phase="intent_committed",
        uncertainty_kind="receipt_state",
    )
    assert intent_boundary.intent_committed is False
    assert intent_boundary.intent_commit_may_have_occurred is True
    assert intent_boundary.case_may_have_started is False
    assert intent_boundary.publication_may_be_visible is False
    assert intent_boundary.terminal_state == "intent_commit_state_uncertain_non_retriable"

    early_uncertainty = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=("request_validated", "intent_committed"),
        failure_phase="fresh_cgroup_created",
        uncertainty_kind="container_state",
    )
    assert early_uncertainty.case_may_have_started is False
    assert early_uncertainty.publication_may_be_visible is False
    assert early_uncertainty.terminal_state == (
        "intent_consumed_no_workload_started_non_retriable"
    )

    case_index = host.HOST_QUALIFICATION_LIFECYCLE_PHASES.index("case_started")
    case_boundary = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES[:case_index],
        failure_phase="case_started",
        uncertainty_kind="observation_state",
    )
    assert case_boundary.case_may_have_started is True
    assert case_boundary.publication_may_be_visible is True

    publication_index = host.HOST_QUALIFICATION_LIFECYCLE_PHASES.index(
        "publication_committed"
    )
    publication_boundary = host.classify_matched_v3_host_qualification_lifecycle(
        completed_phases=host.HOST_QUALIFICATION_LIFECYCLE_PHASES[:publication_index],
        failure_phase="publication_committed",
        uncertainty_kind="publication_state",
    )
    assert publication_boundary.case_may_have_started is True
    assert publication_boundary.publication_may_be_visible is True


def test_lifecycle_rejects_skipped_or_reordered_phases() -> None:
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.classify_matched_v3_host_qualification_lifecycle(
            completed_phases=("request_validated", "fresh_cgroup_created"),
            failure_phase=None,
            uncertainty_kind=None,
        )


def test_module_exports_no_execution_or_provisioning_api() -> None:
    forbidden = {
        "authorize_matched_v3_host_qualification_case",
        "execute_matched_v3_host_qualification_case",
        "execute_and_publish_matched_v3_host_qualification_case",
        "provision_matched_v3_host_cgroup",
        "retry_matched_v3_host_qualification_case",
    }
    assert forbidden.isdisjoint(vars(host))
    descriptor = host.matched_v3_host_qualification_executor_descriptor()
    assert descriptor["public_api"]["mutation_apis"] == []
    assert descriptor["public_api"]["execution_capability_issued"] is False


def test_strict_parsers_reject_duplicate_keys_floats_and_wrong_body_digest() -> None:
    request = _request()
    raw = host.canonical_matched_v3_host_qualification_case_request_bytes(request)
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.parse_matched_v3_host_qualification_case_request(
            raw.replace(b'"attempt_ordinal":0', b'"attempt_ordinal":0,"attempt_ordinal":0')
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.parse_matched_v3_host_qualification_case_request(
            raw.replace(b'"attempt_ordinal":0', b'"attempt_ordinal":0.0')
        )

    value = json.loads(raw)
    value["request_body_sha256"] = _sha("wrong")
    with pytest.raises(host.ForagerMatchedV3HostQualificationExecutorError):
        host.parse_matched_v3_host_qualification_case_request(_canonical(value))
