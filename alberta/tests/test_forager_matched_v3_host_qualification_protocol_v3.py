"""Pure metadata tests for the additive matched-v3 host protocol.

The fixtures construct and corrupt canonical in-memory records only.  They do
not inspect a host, verify a signature, access storage, launch a process or
container, issue a case, or execute a benchmark.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_host_provisioning_v3 as provisioning_v3,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_host_qualification_executor_v2 as executor_v2,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_host_qualification_protocol_v3 as host,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_plan_v3 as plan_v3,
)
from alberta_framework.benchmarks._forager_matched_v3_canonical_evidence import (
    PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE,
    PRODUCER_ROLES,
    ArtifactRefV1,
    CaseSubjectV1,
    ProducerRefV1,
    canonical_file_bytes,
    canonical_json_bytes,
)


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _StringEqualitySpoof(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _IntegerEqualitySpoof(int):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = int.__hash__


def _replace_unchecked(value: Any, /, **changes: Any) -> Any:
    """Exercise runtime values that intentionally violate static field types."""

    return replace(value, **changes)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _ref(schema: str, label: str) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema,
        _hash(label + ":file"),
        _hash(label + ":body"),
    )


def _component(label: str) -> host.PinnedHostComponentRefV1:
    return host.PinnedHostComponentRefV1(
        component_id=label,
        descriptor_schema_version=f"alberta.forager_matched_v3.{label}_descriptor.v1",
        descriptor_file_sha256=_hash(label + ":descriptor-file"),
        descriptor_body_sha256=_hash(label + ":descriptor-body"),
        source_sha256=_hash(label + ":source"),
        runtime_artifact_sha256=_hash(label + ":runtime"),
    )


def _producers() -> tuple[ProducerRefV1, ...]:
    return tuple(
        ProducerRefV1(
            role=role,
            descriptor_schema_version=PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE[role],
            descriptor_file_sha256=_hash(role + ":descriptor-file"),
            descriptor_body_sha256=_hash(role + ":descriptor-body"),
            source_sha256=_hash(role + ":source"),
        )
        for role in PRODUCER_ROLES
    )


def _producer_facts() -> tuple[host.HostStorageProducerRuntimeFactV1, ...]:
    return tuple(
        host.HostStorageProducerRuntimeFactV1(
            producer=producer,
            component_id=f"component-{producer.role}",
            runtime_artifact_sha256=_hash(producer.role + ":runtime"),
        )
        for producer in _producers()
    )


def _producer_chain() -> host.HostStorageProducerPreGoBundleV1:
    facts = _producer_facts()
    policy = host.HostStorageProducerTrustPolicyV1(
        policy_id="six-role-policy-01",
        policy_nonce_sha256=_hash("six-role-policy-nonce"),
        qualification_plan=_ref(host.QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "plan"),
        storage_backend_policy=_ref(host.STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION, "storage"),
        host_trust_policy=_ref(
            provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
            "host-policy",
        ),
        issued_at_unix_ns=100,
        valid_from_unix_ns=100,
        valid_until_unix_ns=10_000,
        signer_key_id="six-role-signer-01",
        signer_public_key_sha256=_hash("six-role-public-key"),
        independent_verifier=_component("six-role-verifier"),
        live_validator=_component("six-role-validator"),
        expected_producer_facts=facts,
    )
    statement = host.HostStorageProducerInventoryStatementV1(
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        qualification_plan=policy.qualification_plan,
        storage_backend_policy=policy.storage_backend_policy,
        observed_at_unix_ns=1_000,
        observed_at_monotonic_ns=1_000,
        signer_key_id=policy.signer_key_id,
        signer_public_key_sha256=policy.signer_public_key_sha256,
        producer_facts=facts,
        signature_hex="ab" * 64,
    )
    verification = host.HostStorageProducerInventorySignatureVerificationReceiptV1(
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        verifier=policy.independent_verifier,
        verification_run_id_sha256=_hash("six-role-verification-run"),
        verification_started_at_unix_ns=1_100,
        verification_completed_at_unix_ns=1_200,
        verification_started_at_monotonic_ns=1_100,
        verification_completed_at_monotonic_ns=1_200,
        signer_key_id=statement.signer_key_id,
        signer_public_key_sha256=statement.signer_public_key_sha256,
        signed_payload_sha256=statement.signed_payload_sha256,
        signature_sha256=hashlib.sha256(bytes.fromhex(statement.signature_hex)).hexdigest(),
    )
    verification_ref = (
        host.host_storage_producer_inventory_signature_verification_receipt_v1_identity(
            verification
        )
    )
    pre_capability = host.HostStorageProducerLiveValidationReceiptV1(
        checkpoint="pre_capability",
        checkpoint_ordinal=0,
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=None,
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("producer-live-pre-capability"),
        validated_at_unix_ns=1_300,
        validated_at_monotonic_ns=1_300,
        observed_producer_facts=facts,
    )
    pre_go = host.HostStorageProducerLiveValidationReceiptV1(
        checkpoint="pre_go",
        checkpoint_ordinal=1,
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=(
            host.host_storage_producer_live_validation_receipt_v1_identity(pre_capability)
        ),
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("producer-live-pre-go"),
        validated_at_unix_ns=3_000,
        validated_at_monotonic_ns=3_000,
        observed_producer_facts=facts,
    )
    return host.HostStorageProducerPreGoBundleV1(
        policy=policy,
        statement=statement,
        signature_verification_receipt=verification,
        pre_capability_live_validation=pre_capability,
        pre_go_live_validation=pre_go,
    )


def _legacy_artifact(
    schema: str,
    label: str,
) -> provisioning_v3.ArtifactIdentityV1:
    return provisioning_v3.ArtifactIdentityV1(
        schema_version=schema,
        file_sha256=_hash(label + ":file"),
        body_sha256=_hash(label + ":body"),
    )


def _legacy_component(
    label: str,
    *,
    descriptor_schema: str | None = None,
) -> provisioning_v3.PinnedComponentIdentityV1:
    return provisioning_v3.PinnedComponentIdentityV1(
        component_id=label,
        descriptor_schema_version=(
            f"alberta.forager_matched_v3.{label}_descriptor.v1"
            if descriptor_schema is None
            else descriptor_schema
        ),
        descriptor_file_sha256=_hash(label + ":descriptor-file"),
        source_sha256=_hash(label + ":source"),
        runtime_artifact_sha256=_hash(label + ":runtime"),
    )


def _host_component_from_legacy(
    component: provisioning_v3.PinnedComponentIdentityV1,
) -> host.PinnedHostComponentRefV1:
    return host.PinnedHostComponentRefV1(
        component_id=component.component_id,
        descriptor_schema_version=component.descriptor_schema_version,
        descriptor_file_sha256=component.descriptor_file_sha256,
        descriptor_body_sha256=_hash(component.component_id + ":descriptor-body"),
        source_sha256=component.source_sha256,
        runtime_artifact_sha256=component.runtime_artifact_sha256,
    )


def _host_facts() -> provisioning_v3.HostFactsInventoryV1:
    return provisioning_v3.HostFactsInventoryV1(
        kernel=provisioning_v3.HostKernelIdentityV1(
            host_identity_sha256=_hash("host"),
            machine_id_sha256=_hash("machine"),
            boot_id="01234567-89ab-4def-8123-456789abcdef",
            architecture="x86_64",
            kernel_release="6.8.0-qualified",
            kernel_build_sha256=_hash("kernel-build"),
            kernel_command_line_sha256=_hash("kernel-command-line"),
        ),
        cgroup=provisioning_v3.CgroupV2IdentityV1(
            mount_path="/sys/fs/cgroup",
            mount_device_major=0,
            mount_device_minor=29,
            mount_inode=101,
            filesystem_magic="0x63677270",
            unified_hierarchy=True,
            delegate_path="/sys/fs/cgroup/alberta-qualified-host",
            delegate_device_major=0,
            delegate_device_minor=29,
            delegate_inode=202,
            delegate_uid=0,
            delegate_gid=0,
            delegate_mode=0o750,
            delegated_controllers=("cpu", "memory", "pids"),
            subtree_control=("cpu", "memory", "pids"),
        ),
        docker=provisioning_v3.DockerDaemonIdentityV1(
            socket_path="/run/docker.sock",
            socket_device_major=0,
            socket_device_minor=8,
            socket_inode=303,
            socket_uid=0,
            socket_gid=0,
            socket_mode=0o660,
            daemon_id="qualified-daemon-01",
            daemon_pid=404,
            daemon_start_ticks=505,
            rootful=True,
            cgroup_driver="cgroupfs",
            version="29.0.1",
            api_version="1.52",
            config_sha256=_hash("docker-config"),
            root_dir_path="/var/lib/docker-qualified",
            root_dir_device_major=8,
            root_dir_device_minor=1,
            root_dir_inode=606,
        ),
        components=provisioning_v3.HostComponentInventoryV1(
            oci_runtime=_legacy_component("oci-runtime"),
            membership_observer=_legacy_component("membership-observer"),
            storage_measurement_producer=_legacy_component(
                "storage-measurement",
                descriptor_schema=PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE["measurement_producer"],
            ),
            storage_terminal_relay=_legacy_component(
                "storage-terminal-relay",
                descriptor_schema=PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE["terminal_relay"],
            ),
            security_profile=_legacy_component("security-profile"),
        ),
    )


def _host_provisioning_bundle() -> host.HostProvisioningPreGoBundleV3:
    facts = _host_facts()
    policy = provisioning_v3.HostTrustPolicyV1(
        policy_id="matched-v3-qualified-host-policy-01",
        policy_nonce_sha256=_hash("host-policy-nonce"),
        qualification_plan=_legacy_artifact(host.QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "plan"),
        issued_at_unix_ns=100,
        valid_from_unix_ns=100,
        valid_until_unix_ns=10_000,
        signer_key_id="host-provisioner-key-01",
        signer_public_key_sha256=_hash("host-public-key"),
        independent_verifier=_legacy_component("six-role-verifier"),
        live_validator=_legacy_component("six-role-validator"),
        supported_host_tuple=provisioning_v3.SupportedHostTupleV1.from_facts(
            "qualified-linux-docker-cgroupfs-tuple-01",
            facts,
        ),
        expected_facts=facts,
    )
    statement = provisioning_v3.HostProvisioningStatementV1(
        policy=provisioning_v3.host_trust_policy_identity_v1(policy),
        observed_at_unix_ns=1_000,
        observed_at_monotonic_ns=1_000,
        signer_key_id=policy.signer_key_id,
        signer_public_key_sha256=policy.signer_public_key_sha256,
        facts=facts,
        signature_hex="ab" * 64,
    )
    verification = provisioning_v3.HostSignatureVerificationReceiptV1(
        policy=provisioning_v3.host_trust_policy_identity_v1(policy),
        statement=provisioning_v3.host_provisioning_statement_identity_v1(statement),
        verifier=policy.independent_verifier,
        verification_run_id_sha256=_hash("host-verification-run"),
        verification_started_at_unix_ns=1_100,
        verification_completed_at_unix_ns=1_200,
        verification_started_at_monotonic_ns=1_100,
        verification_completed_at_monotonic_ns=1_200,
        signer_key_id=statement.signer_key_id,
        signer_public_key_sha256=statement.signer_public_key_sha256,
        signed_payload_sha256=statement.signed_payload_sha256,
        signature_sha256=hashlib.sha256(bytes.fromhex(statement.signature_hex)).hexdigest(),
    )
    verification_ref = provisioning_v3.host_signature_verification_receipt_identity_v1(verification)
    pre_capability = provisioning_v3.HostLiveValidationReceiptV1(
        checkpoint="pre_capability",
        checkpoint_ordinal=0,
        policy=provisioning_v3.host_trust_policy_identity_v1(policy),
        statement=provisioning_v3.host_provisioning_statement_identity_v1(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=None,
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("host-live-pre-capability"),
        validated_at_unix_ns=1_300,
        validated_at_monotonic_ns=1_300,
        facts=facts,
    )
    pre_go = provisioning_v3.HostLiveValidationReceiptV1(
        checkpoint="pre_go",
        checkpoint_ordinal=1,
        policy=provisioning_v3.host_trust_policy_identity_v1(policy),
        statement=provisioning_v3.host_provisioning_statement_identity_v1(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=(
            provisioning_v3.host_live_validation_receipt_identity_v1(pre_capability)
        ),
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("host-live-pre-go"),
        validated_at_unix_ns=3_000,
        validated_at_monotonic_ns=3_000,
        facts=facts,
    )
    return host.HostProvisioningPreGoBundleV3(
        policy=policy,
        statement=statement,
        signature_verification_receipt=verification,
        pre_capability_live_validation=pre_capability,
        pre_go_live_validation=pre_go,
    )


def _qualified_producer_bundle(
    host_bundle: host.HostProvisioningPreGoBundleV3,
) -> host.HostStorageProducerPreGoBundleV1:
    legacy_components = (
        host_bundle.policy.expected_facts.components.storage_measurement_producer,
        host_bundle.policy.expected_facts.components.storage_terminal_relay,
    )
    facts = tuple(
        host.HostStorageProducerRuntimeFactV1(
            producer=ProducerRefV1(
                role=role,
                descriptor_schema_version=PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE[role],
                descriptor_file_sha256=(
                    legacy_components[index].descriptor_file_sha256
                    if index < len(legacy_components)
                    else _hash(role + ":descriptor-file")
                ),
                descriptor_body_sha256=_hash(role + ":descriptor-body"),
                source_sha256=(
                    legacy_components[index].source_sha256
                    if index < len(legacy_components)
                    else _hash(role + ":source")
                ),
            ),
            component_id=(
                legacy_components[index].component_id
                if index < len(legacy_components)
                else f"component-{role}"
            ),
            runtime_artifact_sha256=(
                legacy_components[index].runtime_artifact_sha256
                if index < len(legacy_components)
                else _hash(role + ":runtime")
            ),
        )
        for index, role in enumerate(PRODUCER_ROLES)
    )
    policy = host.HostStorageProducerTrustPolicyV1(
        policy_id="six-role-policy-01",
        policy_nonce_sha256=_hash("six-role-policy-nonce"),
        qualification_plan=_ref(host.QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "plan"),
        storage_backend_policy=_ref(host.STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION, "storage"),
        host_trust_policy=host_bundle.policy_ref,
        issued_at_unix_ns=100,
        valid_from_unix_ns=100,
        valid_until_unix_ns=10_000,
        signer_key_id="six-role-signer-01",
        signer_public_key_sha256=_hash("six-role-public-key"),
        independent_verifier=_host_component_from_legacy(host_bundle.policy.independent_verifier),
        live_validator=_host_component_from_legacy(host_bundle.policy.live_validator),
        expected_producer_facts=facts,
    )
    statement = host.HostStorageProducerInventoryStatementV1(
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        qualification_plan=policy.qualification_plan,
        storage_backend_policy=policy.storage_backend_policy,
        observed_at_unix_ns=1_000,
        observed_at_monotonic_ns=1_000,
        signer_key_id=policy.signer_key_id,
        signer_public_key_sha256=policy.signer_public_key_sha256,
        producer_facts=facts,
        signature_hex="cd" * 64,
    )
    verification = host.HostStorageProducerInventorySignatureVerificationReceiptV1(
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        verifier=policy.independent_verifier,
        verification_run_id_sha256=_hash("qualified-six-role-verification-run"),
        verification_started_at_unix_ns=1_100,
        verification_completed_at_unix_ns=1_200,
        verification_started_at_monotonic_ns=1_100,
        verification_completed_at_monotonic_ns=1_200,
        signer_key_id=statement.signer_key_id,
        signer_public_key_sha256=statement.signer_public_key_sha256,
        signed_payload_sha256=statement.signed_payload_sha256,
        signature_sha256=hashlib.sha256(bytes.fromhex(statement.signature_hex)).hexdigest(),
    )
    verification_ref = (
        host.host_storage_producer_inventory_signature_verification_receipt_v1_identity(
            verification
        )
    )
    pre_capability = host.HostStorageProducerLiveValidationReceiptV1(
        checkpoint="pre_capability",
        checkpoint_ordinal=0,
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=None,
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("qualified-producer-live-pre-capability"),
        validated_at_unix_ns=1_300,
        validated_at_monotonic_ns=1_300,
        observed_producer_facts=facts,
    )
    pre_go = host.HostStorageProducerLiveValidationReceiptV1(
        checkpoint="pre_go",
        checkpoint_ordinal=1,
        policy=host.host_storage_producer_trust_policy_v1_identity(policy),
        statement=host.host_storage_producer_inventory_statement_v1_identity(statement),
        signature_verification_receipt=verification_ref,
        previous_live_validation_receipt=(
            host.host_storage_producer_live_validation_receipt_v1_identity(pre_capability)
        ),
        validator=policy.live_validator,
        validation_run_id_sha256=_hash("qualified-producer-live-pre-go"),
        validated_at_unix_ns=3_000,
        validated_at_monotonic_ns=3_000,
        observed_producer_facts=facts,
    )
    return host.HostStorageProducerPreGoBundleV1(
        policy=policy,
        statement=statement,
        signature_verification_receipt=verification,
        pre_capability_live_validation=pre_capability,
        pre_go_live_validation=pre_go,
    )


def _ceilings(max_temporary_peak_bytes: int) -> tuple[tuple[str, int], ...]:
    values = {name: 1 for name in host.RESOURCE_FIELDS}
    values["max_environment_interactions"] = host.MATCHED_V3_HORIZON
    values["max_temporary_peak_bytes"] = max_temporary_peak_bytes
    values["max_failure_count"] = 0
    return tuple((name, values[name]) for name in host.RESOURCE_FIELDS)


def _executor_artifact(schema: str, label: str) -> executor_v2.ArtifactIdentityV2:
    return executor_v2.ArtifactIdentityV2(
        schema_version=schema,
        file_sha256=_hash(label + ":file"),
        body_sha256=_hash(label + ":body"),
    )


def _executor_producer(
    schema: str,
    label: str,
    *,
    descriptor_sha256: str | None = None,
    source_sha256: str | None = None,
) -> executor_v2.ProducerIdentityV2:
    return executor_v2.ProducerIdentityV2(
        descriptor_schema_version=schema,
        descriptor_sha256=(
            _hash(label + ":descriptor") if descriptor_sha256 is None else descriptor_sha256
        ),
        source_sha256=_hash(label + ":source") if source_sha256 is None else source_sha256,
    )


def _executor_identity(
    value: Any,
    schema: str,
    body_builder: Callable[[Any], bytes],
    file_builder: Callable[[Any], bytes],
) -> executor_v2.ArtifactIdentityV2:
    return executor_v2.ArtifactIdentityV2(
        schema_version=schema,
        file_sha256=hashlib.sha256(file_builder(value)).hexdigest(),
        body_sha256=hashlib.sha256(body_builder(value)).hexdigest(),
    )


def _host_ref_from_executor(identity: executor_v2.ArtifactIdentityV2) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version=identity.schema_version,
        file_sha256=identity.file_sha256,
        body_sha256=identity.body_sha256,
    )


@dataclass(frozen=True, slots=True)
class _BaseV2Chain:
    request: executor_v2.HostQualificationCaseRequestV2
    intent: executor_v2.HostQualificationCaseIntentV2
    initial_sample: executor_v2.HostInitialCgroupSampleV2
    ready: executor_v2.HostReadyV2
    anchor: executor_v2.HostObserverAnchorV2
    go: executor_v2.HostGoCommitmentV2


def _base_v2_chain() -> _BaseV2Chain:
    spine = _hash("case-spine")
    container_name = executor_v2.expected_container_name_v2(spine)
    request = executor_v2.HostQualificationCaseRequestV2(
        case_spine_sha256=spine,
        case_ordinal=0,
        candidate_id="causal_e025_q050",
        candidate_family="local",
        qualification_case_id="qualification_00_causal_e025_q050",
        container_name=container_name,
        qualification_plan=_executor_artifact(host.QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "plan"),
        plan_issuance_receipt=_executor_artifact(
            executor_v2.PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION,
            "plan-issuance",
        ),
        case_execution_ticket=_executor_artifact(
            host.CASE_EXECUTION_TICKET_SCHEMA_VERSION,
            "ticket",
        ),
        qualification_case_manifest=_executor_artifact(
            executor_v2.QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION,
            "case-manifest",
        ),
        observation_registry=_executor_producer(
            executor_v2.QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION,
            "observation-registry",
        ),
        joint_source_closure=_executor_artifact(
            executor_v2.JOINT_SOURCE_CLOSURE_SCHEMA_VERSION,
            "joint-source-closure",
        ),
        sealed_staging=_executor_artifact(
            executor_v2.SEALED_STAGING_SCHEMA_VERSION,
            "sealed-staging",
        ),
        fresh_build=_executor_artifact(
            executor_v2.FRESH_BUILD_SCHEMA_VERSION,
            "fresh-build",
        ),
        build_context_receipt=_executor_artifact(
            executor_v2.CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
            "build-context",
        ),
        build_execution_receipt=_executor_artifact(
            executor_v2.CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "build-execution",
        ),
        build_publication_receipt=_executor_artifact(
            executor_v2.CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "build-publication",
        ),
        image_id="sha256:" + _hash("image"),
        runtime_qualification_receipt=_executor_artifact(
            host.RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime-receipt",
        ),
        host_provisioning_receipt=_executor_artifact(
            executor_v2.HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host-provisioning-receipt",
        ),
        algorithmic_contract=_executor_producer(
            executor_v2.ALGORITHMIC_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
            "algorithmic-contract",
            descriptor_sha256=executor_v2.FINAL_ALGORITHMIC_CONTRACT_DESCRIPTOR_SHA256,
            source_sha256=executor_v2.FINAL_ALGORITHMIC_CONTRACT_SOURCE_SHA256,
        ),
        algorithmic_measurement_intent=_executor_artifact(
            executor_v2.ALGORITHMIC_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "algorithmic-intent",
        ),
        publication_contract=_executor_producer(
            executor_v2.PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
            "publication-contract",
            descriptor_sha256=executor_v2.FINAL_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256,
            source_sha256=executor_v2.FINAL_PUBLICATION_CONTRACT_SOURCE_SHA256,
        ),
        storage_contract=_executor_producer(
            executor_v2.STORAGE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
            "storage-contract",
            descriptor_sha256=executor_v2.FINAL_STORAGE_CONTRACT_DESCRIPTOR_SHA256,
            source_sha256=executor_v2.FINAL_STORAGE_CONTRACT_SOURCE_SHA256,
        ),
        storage_boundary_intent=_executor_artifact(
            executor_v2.STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "storage-boundary-intent",
        ),
        host_executor=_executor_producer(
            executor_v2.HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
            "host-executor-v2",
            descriptor_sha256=host.AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256,
            source_sha256=host.AUDITED_HOST_EXECUTOR_V2_SOURCE_SHA256,
        ),
        full_resource_merger=_executor_producer(
            executor_v2.FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION,
            "full-resource-merger",
        ),
        publisher=_executor_producer(
            "alberta.forager_matched_v3.local_reward_publication_descriptor.v1",
            "publisher",
        ),
        native_atomic_producer=_executor_producer(
            "alberta.forager_matched_v3.atomic_publication_descriptor.v1",
            "native-atomic",
        ),
        in_container_driver=_executor_producer(
            "alberta.forager_matched_v3.local_runner_descriptor.v1",
            "in-container-driver",
        ),
        resource_requirement_body_sha256=_hash("resource-requirement-body"),
        declared_ceilings=_ceilings(4096),
        horizon=host.MATCHED_V3_HORIZON,
        attempt_ordinal=0,
        exact_acknowledgement=executor_v2.HOST_EXECUTION_ACKNOWLEDGEMENT,
    )
    request_ref = _executor_identity(
        request,
        host.HOST_CASE_REQUEST_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_case_request_v2_body_bytes,
        executor_v2.canonical_host_case_request_v2_file_bytes,
    )
    intent = executor_v2.HostQualificationCaseIntentV2(
        case_spine_sha256=spine,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        request=request_ref,
        case_execution_ticket=request.case_execution_ticket,
        image_id=request.image_id,
        algorithmic_measurement_intent=request.algorithmic_measurement_intent,
        storage_boundary_intent=request.storage_boundary_intent,
        container_name=request.container_name,
        retained_fd_policy_body_sha256=_hash("retained-fd-policy"),
        cleanup_policy_body_sha256=_hash("cleanup-policy"),
        exact_acknowledgement_sha256=hashlib.sha256(
            executor_v2.HOST_EXECUTION_ACKNOWLEDGEMENT.encode("ascii")
        ).hexdigest(),
        intent_committed=True,
        same_case_retry_permitted=False,
    )
    intent_ref = _executor_identity(
        intent,
        host.HOST_CASE_INTENT_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_case_intent_v2_body_bytes,
        executor_v2.canonical_host_case_intent_v2_file_bytes,
    )
    cgroup_identity = executor_v2.HostCgroupCaseIdentityV2(
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
    counter_fds = tuple(
        executor_v2.RetainedCgroupCounterFdV2(
            endpoint_name=name,
            endpoint_device=101,
            endpoint_inode=2_000 + index,
            open_monotonic_ns=100 + index,
            open_flags=(
                "O_CLOEXEC",
                "O_NOFOLLOW",
                "O_WRONLY" if name == "cgroup.kill" else "O_RDONLY",
            ),
            reset_performed=False,
            reopened=False,
            retained_through_post_container_remove_sample=True,
        )
        for index, name in enumerate(executor_v2.CGROUP_COUNTER_ENDPOINTS)
    )
    retained_fd_set = executor_v2.retained_cgroup_fd_inventory_sha256_v2(counter_fds)
    cgroup_identity_sha256 = executor_v2.cgroup_case_identity_sha256_v2(cgroup_identity)
    initial_sample = executor_v2.HostInitialCgroupSampleV2(
        case_spine_sha256=spine,
        intent=intent_ref,
        cgroup_case_identity=cgroup_identity,
        counter_fds=counter_fds,
        facts=executor_v2.CgroupSampleFactsV2(
            monotonic_ns=2_500,
            cgroup_identity_sha256=cgroup_identity_sha256,
            retained_fd_set_sha256=retained_fd_set,
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
        ),
    )
    initial_ref = _executor_identity(
        initial_sample,
        host.HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_initial_cgroup_sample_v2_body_bytes,
        executor_v2.canonical_host_initial_cgroup_sample_v2_file_bytes,
    )
    container_id = _hash("container-id")
    ready = executor_v2.HostReadyV2(
        case_spine_sha256=spine,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        candidate_family=request.candidate_family,
        qualification_case_id=request.qualification_case_id,
        intent=intent_ref,
        algorithmic_measurement_intent=request.algorithmic_measurement_intent,
        storage_boundary_intent=request.storage_boundary_intent,
        initial_cgroup_sample=initial_ref,
        retained_fd_set_sha256=retained_fd_set,
        cgroup_identity_sha256=cgroup_identity_sha256,
        container_identity_sha256=executor_v2.container_runtime_identity_sha256_v2(
            spine,
            container_name,
            container_id,
        ),
        container_id=container_id,
        container_name=container_name,
        container_cgroup_path=f"{cgroup_identity.case_cgroup_path}/{container_id}",
        container_cgroup_device=cgroup_identity.case_cgroup_device,
        container_cgroup_inode=3_000,
        host_pid=1_234,
        host_process_start_time_ticks=4_567,
        inner_pid=1,
        ready_monotonic_ns=2_800,
        candidate_code_loaded=False,
        go_committed=False,
    )
    ready_ref = _executor_identity(
        ready,
        host.HOST_READY_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_ready_v2_body_bytes,
        executor_v2.canonical_host_ready_v2_file_bytes,
    )
    anchor = executor_v2.HostObserverAnchorV2(
        case_spine_sha256=spine,
        ready=ready_ref,
        initial_cgroup_sample=initial_ref,
        retained_fd_set_sha256=retained_fd_set,
        cgroup_identity_sha256=cgroup_identity_sha256,
        observer_descriptor_sha256=request.observation_registry.descriptor_sha256,
        observer_source_sha256=request.observation_registry.source_sha256,
        observer_started_monotonic_ns=2_900,
        observation_loss_detected=False,
    )
    anchor_ref = _executor_identity(
        anchor,
        host.HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_observer_anchor_v2_body_bytes,
        executor_v2.canonical_host_observer_anchor_v2_file_bytes,
    )
    go = executor_v2.HostGoCommitmentV2(
        case_spine_sha256=spine,
        ready=ready_ref,
        observer_anchor=anchor_ref,
        retained_fd_set_sha256=retained_fd_set,
        cgroup_identity_sha256=cgroup_identity_sha256,
        go_payload_sha256=_hash("base-go-payload"),
        go_committed_monotonic_ns=3_300,
        go_commit_count=1,
        one_way=True,
        same_case_retry_permitted=False,
    )
    return _BaseV2Chain(request, intent, initial_sample, ready, anchor, go)


def _handshake() -> tuple[
    host.HostQualificationCaseRequestV3,
    host.HostQualificationCaseIntentV3,
    host.HostReadyV3,
    host.HostObserverAnchorV3,
    host.HostProvisioningValidatedPreGoPrefixV3,
    host.HostGoCommitmentV3,
]:
    producer_chain = _producer_chain()
    plan = producer_chain.policy.qualification_plan
    storage_policy = producer_chain.policy.storage_backend_policy
    producers = producer_chain.policy.storage_producers
    subject = CaseSubjectV1.for_ordinal(0)
    spine = _hash("case-spine")
    image = "sha256:" + _hash("image")
    container_name = executor_v2.expected_container_name_v2(spine)
    maximum = 4096
    host_statement = _ref(
        provisioning_v3.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
        "host-statement",
    )
    host_verification = _ref(
        provisioning_v3.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        "host-verification",
    )
    host_pre_capability = _ref(
        provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "host-pre-capability",
    )
    host_pre_go = _ref(
        provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "host-pre-go",
    )
    request = host.HostQualificationCaseRequestV3(
        campaign_id="campaign-01",
        case_spine_sha256=spine,
        subject=subject,
        base_request_v2=_ref(host.HOST_CASE_REQUEST_V2_SCHEMA_VERSION, "base-request"),
        qualification_plan=plan,
        qualification_plan_descriptor=_ref(
            host.QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
            "plan-descriptor",
        ),
        case_execution_ticket=_ref(host.CASE_EXECUTION_TICKET_SCHEMA_VERSION, "ticket"),
        runtime_qualification_receipt=_ref(
            host.RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime-receipt",
        ),
        storage_backend_descriptor=_ref(
            host.STORAGE_BACKEND_CONTRACT_DESCRIPTOR_V2_SCHEMA_VERSION,
            "storage-descriptor",
        ),
        storage_backend_source_sha256=_hash("storage-source"),
        storage_backend_policy=storage_policy,
        host_provisioning_descriptor=ArtifactRefV1(
            provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
            host.AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256,
            host.AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256,
        ),
        host_provisioning_source_sha256=host.AUDITED_HOST_PROVISIONING_V3_SOURCE_SHA256,
        host_executor_v2_descriptor=ArtifactRefV1(
            executor_v2.HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
            host.AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256,
            host.AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_BODY_SHA256,
        ),
        host_executor_v2_source_sha256=host.AUDITED_HOST_EXECUTOR_V2_SOURCE_SHA256,
        host_protocol_descriptor=host.host_qualification_protocol_descriptor_v3_identity(),
        host_protocol_source_sha256=_hash("host-protocol-source"),
        host_trust_policy=producer_chain.policy.host_trust_policy,
        host_provisioning_statement=host_statement,
        host_signature_verification_receipt=host_verification,
        host_pre_capability_live_validation=host_pre_capability,
        storage_producer_trust_policy=producer_chain.policy_ref,
        storage_producer_inventory_statement=producer_chain.statement_ref,
        storage_producer_signature_verification_receipt=producer_chain.verification_ref,
        storage_producer_pre_capability_live_validation=producer_chain.pre_capability_ref,
        storage_producers=producers,
        resource_field_order_sha256=host.RESOURCE_FIELD_ORDER_SHA256,
        candidate_order_sha256=host.CANDIDATE_ORDER_SHA256,
        resource_requirement_body_sha256=_hash("resource-requirement-body"),
        declared_ceilings=_ceilings(maximum),
        image_id=image,
        container_name=container_name,
        max_temporary_peak_bytes=maximum,
        request_validated_monotonic_ns=2_000,
    )
    intent = host.HostQualificationCaseIntentV3(
        campaign_id=request.campaign_id,
        case_spine_sha256=spine,
        subject=subject,
        request=host.host_case_request_v3_identity(request),
        base_intent_v2=_ref(host.HOST_CASE_INTENT_V2_SCHEMA_VERSION, "base-intent"),
        qualification_plan=plan,
        case_execution_ticket=request.case_execution_ticket,
        storage_backend_policy=storage_policy,
        storage_producer_trust_policy=producer_chain.policy_ref,
        storage_producer_inventory_statement=producer_chain.statement_ref,
        storage_producer_signature_verification_receipt=producer_chain.verification_ref,
        storage_producer_pre_capability_live_validation=producer_chain.pre_capability_ref,
        storage_producers=producers,
        image_id=image,
        container_name=container_name,
        max_temporary_peak_bytes=maximum,
        authorization_validated_monotonic_ns=2_100,
        intent_committed_monotonic_ns=2_200,
        intent_committed=True,
        same_case_retry_permitted=False,
    )
    container_id = _hash("container-id")
    container_commitment = executor_v2.container_runtime_identity_sha256_v2(
        spine,
        container_name,
        container_id,
    )
    ready = host.HostReadyV3(
        campaign_id=request.campaign_id,
        case_spine_sha256=spine,
        subject=subject,
        intent=host.host_case_intent_v3_identity(intent),
        base_initial_cgroup_sample_v2=_ref(
            host.HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
            "base-initial-sample",
        ),
        base_ready_v2=_ref(host.HOST_READY_V2_SCHEMA_VERSION, "base-ready"),
        qualification_plan=plan,
        storage_backend_policy=storage_policy,
        storage_producers=producers,
        image_id=image,
        container_id=container_id,
        container_name=container_name,
        container_id_commitment_sha256=container_commitment,
        outer_cgroup_identity_sha256=_hash("outer-cgroup"),
        max_temporary_peak_bytes=maximum,
        aggregate_root_case_exclusive=True,
        fresh_cgroup_created_monotonic_ns=2_300,
        retained_counter_fds_opened_monotonic_ns=2_400,
        initial_cgroup_sample_committed_monotonic_ns=2_500,
        container_created_monotonic_ns=2_600,
        container_started_monotonic_ns=2_700,
        driver_ready_monotonic_ns=2_800,
        candidate_code_loaded=False,
        go_committed=False,
    )
    anchor = host.HostObserverAnchorV3(
        campaign_id=request.campaign_id,
        case_spine_sha256=spine,
        subject=subject,
        ready=host.host_ready_v3_identity(ready),
        base_observer_anchor_v2=_ref(
            host.HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
            "base-anchor",
        ),
        qualification_plan=plan,
        storage_backend_policy=storage_policy,
        storage_producers=producers,
        image_id=image,
        container_name=container_name,
        container_id_commitment_sha256=container_commitment,
        outer_cgroup_identity_sha256=ready.outer_cgroup_identity_sha256,
        max_temporary_peak_bytes=maximum,
        aggregate_root_case_exclusive=True,
        membership_observer=_component("membership-observer"),
        observer_anchored_monotonic_ns=2_900,
        observation_loss_detected=False,
    )
    prefix = host.HostProvisioningValidatedPreGoPrefixV3(
        campaign_id=request.campaign_id,
        case_spine_sha256=spine,
        subject=subject,
        qualification_plan=plan,
        storage_backend_policy=storage_policy,
        request=host.host_case_request_v3_identity(request),
        intent=host.host_case_intent_v3_identity(intent),
        ready=host.host_ready_v3_identity(ready),
        observer_anchor=host.host_observer_anchor_v3_identity(anchor),
        host_provisioning_descriptor=request.host_provisioning_descriptor,
        host_provisioning_source_sha256=request.host_provisioning_source_sha256,
        host_trust_policy=request.host_trust_policy,
        host_provisioning_statement=host_statement,
        host_signature_verification_receipt=host_verification,
        host_pre_capability_live_validation=host_pre_capability,
        host_pre_go_live_validation=host_pre_go,
        storage_producer_trust_policy=producer_chain.policy_ref,
        storage_producer_inventory_statement=producer_chain.statement_ref,
        storage_producer_signature_verification_receipt=producer_chain.verification_ref,
        storage_producer_pre_capability_live_validation=producer_chain.pre_capability_ref,
        storage_producer_pre_go_live_validation=producer_chain.pre_go_ref,
        storage_producers=producers,
        image_id=image,
        container_name=container_name,
        container_id_commitment_sha256=container_commitment,
        outer_cgroup_identity_sha256=ready.outer_cgroup_identity_sha256,
        max_temporary_peak_bytes=maximum,
        aggregate_root_case_exclusive=True,
        prefix_committed_monotonic_ns=3_100,
    )
    base_go = _ref(host.HOST_GO_V2_SCHEMA_VERSION, "base-go")
    storage_runtime_intent = _ref(
        host.STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION,
        "storage-runtime-intent",
    )
    go_arguments: dict[str, Any] = {
        "campaign_id": request.campaign_id,
        "case_spine_sha256": spine,
        "subject": subject,
        "base_go_v2": base_go,
        "qualification_plan": plan,
        "storage_backend_policy": storage_policy,
        "ready": host.host_ready_v3_identity(ready),
        "observer_anchor": host.host_observer_anchor_v3_identity(anchor),
        "validated_pre_go_prefix": (
            host.host_provisioning_validated_pre_go_prefix_v3_identity(prefix)
        ),
        "storage_runtime_intent": storage_runtime_intent,
        "storage_producers": producers,
        "image_id": image,
        "container_name": container_name,
        "container_id_commitment_sha256": container_commitment,
        "outer_cgroup_identity_sha256": ready.outer_cgroup_identity_sha256,
        "max_temporary_peak_bytes": maximum,
        "aggregate_root_case_exclusive": True,
        "resource_field_order_sha256": host.RESOURCE_FIELD_ORDER_SHA256,
        "candidate_order_sha256": host.CANDIDATE_ORDER_SHA256,
        "storage_runtime_intent_committed_monotonic_ns": 3_200,
        "go_committed_monotonic_ns": 3_300,
        "go_commit_count": 1,
        "one_way": True,
        "same_case_retry_permitted": False,
        "storage_runtime_intent_committed_before_go": True,
        "exact_six_producers_committed": True,
    }
    go_payload = host.host_go_payload_sha256_v3(
        **{
            key: value
            for key, value in go_arguments.items()
            if key
            not in {
                "aggregate_root_case_exclusive",
                "exact_six_producers_committed",
                "go_commit_count",
                "go_committed_monotonic_ns",
                "one_way",
                "same_case_retry_permitted",
                "storage_runtime_intent_committed_before_go",
                "storage_runtime_intent_committed_monotonic_ns",
            }
        }
    )
    go = host.HostGoCommitmentV3(**go_arguments, go_payload_sha256=go_payload)
    return request, intent, ready, anchor, prefix, go


def _rebuild_go(
    go: host.HostGoCommitmentV3,
    **changes: object,
) -> host.HostGoCommitmentV3:
    arguments = {field.name: getattr(go, field.name) for field in fields(go)}
    arguments.update(changes)
    payload_arguments = {
        key: value
        for key, value in arguments.items()
        if key
        not in {
            "aggregate_root_case_exclusive",
            "committed_host_phase_prefix",
            "exact_six_producers_committed",
            "go_commit_count",
            "go_committed_monotonic_ns",
            "go_payload_sha256",
            "one_way",
            "same_case_retry_permitted",
            "storage_runtime_intent_committed_before_go",
            "storage_runtime_intent_committed_monotonic_ns",
        }
    }
    arguments["go_payload_sha256"] = host.host_go_payload_sha256_v3(**payload_arguments)
    return host.HostGoCommitmentV3(**arguments)


def _replace_prefix_artifact(
    prefix: host.HostProvisioningValidatedPreGoPrefixV3,
    field_name: str,
    artifact: ArtifactRefV1,
) -> host.HostProvisioningValidatedPreGoPrefixV3:
    arguments: dict[str, Any] = {
        field.name: getattr(prefix, field.name) for field in fields(prefix)
    }
    arguments[field_name] = artifact
    return host.HostProvisioningValidatedPreGoPrefixV3(**arguments)


@dataclass(frozen=True, slots=True)
class _FullChain:
    base: _BaseV2Chain
    host_bundle: host.HostProvisioningPreGoBundleV3
    producer_bundle: host.HostStorageProducerPreGoBundleV1
    request: host.HostQualificationCaseRequestV3
    intent: host.HostQualificationCaseIntentV3
    ready: host.HostReadyV3
    anchor: host.HostObserverAnchorV3
    pre_go: host.HostQualificationPreGoBundleV3
    go: host.HostGoCommitmentV3


def _full_chain() -> _FullChain:
    base = _base_v2_chain()
    host_bundle = _host_provisioning_bundle()
    producer_bundle = _qualified_producer_bundle(host_bundle)
    request, intent, ready, anchor, prefix, go = _handshake()
    base_request_ref = _host_ref_from_executor(
        _executor_identity(
            base.request,
            host.HOST_CASE_REQUEST_V2_SCHEMA_VERSION,
            executor_v2.canonical_host_case_request_v2_body_bytes,
            executor_v2.canonical_host_case_request_v2_file_bytes,
        )
    )
    request = replace(
        request,
        base_request_v2=base_request_ref,
        qualification_plan=_host_ref_from_executor(base.request.qualification_plan),
        case_execution_ticket=_host_ref_from_executor(base.request.case_execution_ticket),
        runtime_qualification_receipt=_host_ref_from_executor(
            base.request.runtime_qualification_receipt
        ),
        storage_backend_policy=producer_bundle.policy.storage_backend_policy,
        host_trust_policy=host_bundle.policy_ref,
        host_provisioning_statement=host_bundle.statement_ref,
        host_signature_verification_receipt=host_bundle.verification_ref,
        host_pre_capability_live_validation=host_bundle.pre_capability_ref,
        storage_producer_trust_policy=producer_bundle.policy_ref,
        storage_producer_inventory_statement=producer_bundle.statement_ref,
        storage_producer_signature_verification_receipt=producer_bundle.verification_ref,
        storage_producer_pre_capability_live_validation=producer_bundle.pre_capability_ref,
        storage_producers=producer_bundle.policy.storage_producers,
        resource_requirement_body_sha256=base.request.resource_requirement_body_sha256,
        declared_ceilings=base.request.declared_ceilings,
        image_id=base.request.image_id,
        container_name=base.request.container_name,
        max_temporary_peak_bytes=base.request.declared_ceilings[23][1],
    )
    intent = replace(
        intent,
        request=host.host_case_request_v3_identity(request),
        base_intent_v2=_host_ref_from_executor(
            _executor_identity(
                base.intent,
                host.HOST_CASE_INTENT_V2_SCHEMA_VERSION,
                executor_v2.canonical_host_case_intent_v2_body_bytes,
                executor_v2.canonical_host_case_intent_v2_file_bytes,
            )
        ),
        qualification_plan=request.qualification_plan,
        case_execution_ticket=request.case_execution_ticket,
        storage_backend_policy=request.storage_backend_policy,
        storage_producer_trust_policy=request.storage_producer_trust_policy,
        storage_producer_inventory_statement=request.storage_producer_inventory_statement,
        storage_producer_signature_verification_receipt=(
            request.storage_producer_signature_verification_receipt
        ),
        storage_producer_pre_capability_live_validation=(
            request.storage_producer_pre_capability_live_validation
        ),
        storage_producers=request.storage_producers,
        image_id=request.image_id,
        container_name=request.container_name,
        max_temporary_peak_bytes=request.max_temporary_peak_bytes,
    )
    ready = replace(
        ready,
        intent=host.host_case_intent_v3_identity(intent),
        base_initial_cgroup_sample_v2=_host_ref_from_executor(
            _executor_identity(
                base.initial_sample,
                host.HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
                executor_v2.canonical_host_initial_cgroup_sample_v2_body_bytes,
                executor_v2.canonical_host_initial_cgroup_sample_v2_file_bytes,
            )
        ),
        base_ready_v2=_host_ref_from_executor(
            _executor_identity(
                base.ready,
                host.HOST_READY_V2_SCHEMA_VERSION,
                executor_v2.canonical_host_ready_v2_body_bytes,
                executor_v2.canonical_host_ready_v2_file_bytes,
            )
        ),
        qualification_plan=request.qualification_plan,
        storage_backend_policy=request.storage_backend_policy,
        storage_producers=request.storage_producers,
        image_id=request.image_id,
        container_id=base.ready.container_id,
        container_name=request.container_name,
        container_id_commitment_sha256=base.ready.container_identity_sha256,
        outer_cgroup_identity_sha256=base.ready.cgroup_identity_sha256,
        max_temporary_peak_bytes=request.max_temporary_peak_bytes,
        initial_cgroup_sample_committed_monotonic_ns=base.initial_sample.facts.monotonic_ns,
        driver_ready_monotonic_ns=base.ready.ready_monotonic_ns,
    )
    anchor = replace(
        anchor,
        ready=host.host_ready_v3_identity(ready),
        base_observer_anchor_v2=_host_ref_from_executor(
            _executor_identity(
                base.anchor,
                host.HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
                executor_v2.canonical_host_observer_anchor_v2_body_bytes,
                executor_v2.canonical_host_observer_anchor_v2_file_bytes,
            )
        ),
        qualification_plan=request.qualification_plan,
        storage_backend_policy=request.storage_backend_policy,
        storage_producers=request.storage_producers,
        image_id=request.image_id,
        container_name=request.container_name,
        container_id_commitment_sha256=ready.container_id_commitment_sha256,
        outer_cgroup_identity_sha256=ready.outer_cgroup_identity_sha256,
        max_temporary_peak_bytes=request.max_temporary_peak_bytes,
        membership_observer=_host_component_from_legacy(
            host_bundle.policy.expected_facts.components.membership_observer
        ),
        observer_anchored_monotonic_ns=base.anchor.observer_started_monotonic_ns,
    )
    prefix = replace(
        prefix,
        qualification_plan=request.qualification_plan,
        storage_backend_policy=request.storage_backend_policy,
        request=host.host_case_request_v3_identity(request),
        intent=host.host_case_intent_v3_identity(intent),
        ready=host.host_ready_v3_identity(ready),
        observer_anchor=host.host_observer_anchor_v3_identity(anchor),
        host_trust_policy=host_bundle.policy_ref,
        host_provisioning_statement=host_bundle.statement_ref,
        host_signature_verification_receipt=host_bundle.verification_ref,
        host_pre_capability_live_validation=host_bundle.pre_capability_ref,
        host_pre_go_live_validation=host_bundle.pre_go_ref,
        storage_producer_trust_policy=producer_bundle.policy_ref,
        storage_producer_inventory_statement=producer_bundle.statement_ref,
        storage_producer_signature_verification_receipt=producer_bundle.verification_ref,
        storage_producer_pre_capability_live_validation=producer_bundle.pre_capability_ref,
        storage_producer_pre_go_live_validation=producer_bundle.pre_go_ref,
        storage_producers=request.storage_producers,
        image_id=request.image_id,
        container_name=request.container_name,
        container_id_commitment_sha256=ready.container_id_commitment_sha256,
        outer_cgroup_identity_sha256=ready.outer_cgroup_identity_sha256,
        max_temporary_peak_bytes=request.max_temporary_peak_bytes,
    )
    pre_go = host.HostQualificationPreGoBundleV3(
        host_provisioning=host_bundle,
        storage_producers=producer_bundle,
        validated_prefix=prefix,
    )
    go = _rebuild_go(
        go,
        base_go_v2=_host_ref_from_executor(
            _executor_identity(
                base.go,
                host.HOST_GO_V2_SCHEMA_VERSION,
                executor_v2.canonical_host_go_v2_body_bytes,
                executor_v2.canonical_host_go_v2_file_bytes,
            )
        ),
        qualification_plan=request.qualification_plan,
        storage_backend_policy=request.storage_backend_policy,
        ready=prefix.ready,
        observer_anchor=prefix.observer_anchor,
        validated_pre_go_prefix=(
            host.host_provisioning_validated_pre_go_prefix_v3_identity(prefix)
        ),
        storage_producers=request.storage_producers,
        image_id=request.image_id,
        container_name=request.container_name,
        container_id_commitment_sha256=ready.container_id_commitment_sha256,
        outer_cgroup_identity_sha256=ready.outer_cgroup_identity_sha256,
        max_temporary_peak_bytes=request.max_temporary_peak_bytes,
        go_committed_monotonic_ns=base.go.go_committed_monotonic_ns,
    )
    return _FullChain(
        base=base,
        host_bundle=host_bundle,
        producer_bundle=producer_bundle,
        request=request,
        intent=intent,
        ready=ready,
        anchor=anchor,
        pre_go=pre_go,
        go=go,
    )


def _validate_full_chain(chain: _FullChain) -> None:
    host.validate_host_qualification_v3_chain(
        base_request=chain.base.request,
        base_intent=chain.base.intent,
        base_initial_sample=chain.base.initial_sample,
        base_ready=chain.base.ready,
        base_anchor=chain.base.anchor,
        base_go=chain.base.go,
        request=chain.request,
        intent=chain.intent,
        ready=chain.ready,
        anchor=chain.anchor,
        pre_go=chain.pre_go,
        go=chain.go,
    )


def _round_trip(
    value: Any,
    body_builder: Callable[[Any], bytes],
    file_builder: Callable[[Any], bytes],
    parser: Callable[..., Any],
) -> None:
    body = body_builder(value)
    raw = file_builder(value)
    parsed = parser(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_body_sha256=hashlib.sha256(body).hexdigest(),
    )
    assert parsed == value
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        parser(
            raw,
            expected_file_sha256=_hash("wrong-file"),
            expected_body_sha256=hashlib.sha256(body).hexdigest(),
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        parser(
            raw,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_body_sha256=_hash("wrong-body"),
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        parser(
            raw,
            expected_file_sha256=hashlib.sha256(body).hexdigest(),
            expected_body_sha256=hashlib.sha256(raw).hexdigest(),
        )


def _schema_cases() -> tuple[
    tuple[
        Any,
        Callable[[Any], bytes],
        Callable[[Any], bytes],
        Callable[..., Any],
    ],
    ...,
]:
    descriptor = host.HostQualificationProtocolDescriptorV3()
    producer = _producer_chain()
    request, intent, ready, anchor, prefix, go = _handshake()
    return (
        (
            descriptor,
            host.canonical_host_qualification_protocol_descriptor_v3_body_bytes,
            host.canonical_host_qualification_protocol_descriptor_v3_file_bytes,
            host.parse_host_qualification_protocol_descriptor_v3,
        ),
        (
            producer.policy,
            host.canonical_host_storage_producer_trust_policy_v1_body_bytes,
            host.canonical_host_storage_producer_trust_policy_v1_file_bytes,
            host.parse_host_storage_producer_trust_policy_v1,
        ),
        (
            producer.statement,
            host.canonical_host_storage_producer_inventory_statement_v1_body_bytes,
            host.canonical_host_storage_producer_inventory_statement_v1_file_bytes,
            host.parse_host_storage_producer_inventory_statement_v1,
        ),
        (
            producer.signature_verification_receipt,
            host.canonical_host_storage_producer_inventory_signature_verification_receipt_v1_body_bytes,
            host.canonical_host_storage_producer_inventory_signature_verification_receipt_v1_file_bytes,
            host.parse_host_storage_producer_inventory_signature_verification_receipt_v1,
        ),
        (
            producer.pre_capability_live_validation,
            host.canonical_host_storage_producer_live_validation_receipt_v1_body_bytes,
            host.canonical_host_storage_producer_live_validation_receipt_v1_file_bytes,
            host.parse_host_storage_producer_live_validation_receipt_v1,
        ),
        (
            prefix,
            host.canonical_host_provisioning_validated_pre_go_prefix_v3_body_bytes,
            host.canonical_host_provisioning_validated_pre_go_prefix_v3_file_bytes,
            host.parse_host_provisioning_validated_pre_go_prefix_v3,
        ),
        (
            request,
            host.canonical_host_case_request_v3_body_bytes,
            host.canonical_host_case_request_v3_file_bytes,
            host.parse_host_case_request_v3,
        ),
        (
            intent,
            host.canonical_host_case_intent_v3_body_bytes,
            host.canonical_host_case_intent_v3_file_bytes,
            host.parse_host_case_intent_v3,
        ),
        (
            ready,
            host.canonical_host_ready_v3_body_bytes,
            host.canonical_host_ready_v3_file_bytes,
            host.parse_host_ready_v3,
        ),
        (
            anchor,
            host.canonical_host_observer_anchor_v3_body_bytes,
            host.canonical_host_observer_anchor_v3_file_bytes,
            host.parse_host_observer_anchor_v3,
        ),
        (
            go,
            host.canonical_host_go_v3_body_bytes,
            host.canonical_host_go_v3_file_bytes,
            host.parse_host_go_v3,
        ),
    )


def test_all_eleven_public_schemas_round_trip_with_independent_identities() -> None:
    cases = _schema_cases()
    assert len(cases) == 11
    for value, body_builder, file_builder, parser in cases:
        _round_trip(value, body_builder, file_builder, parser)


def _rewrite(
    raw: bytes,
    body_field: str,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[bytes, str, str]:
    body = json.loads(raw)
    body.pop(body_field)
    mutate(body)
    rewritten = canonical_file_bytes(body, body_digest_field=body_field)
    body_sha256 = hashlib.sha256(canonical_json_bytes(body, final_lf=False)).hexdigest()
    return rewritten, hashlib.sha256(rewritten).hexdigest(), body_sha256


def _alias_nested_artifact_body(body: dict[str, Any]) -> None:
    body["qualification_plan"]["body_sha256"] = body["qualification_plan"]["file_sha256"]


def _alias_nested_producer_body(body: dict[str, Any]) -> None:
    producer = body["storage_producers"][0]
    producer["descriptor_body_sha256"] = producer["descriptor_file_sha256"]


def _corrupt_nested_subject(body: dict[str, Any]) -> None:
    body["subject"]["case_ordinal"] = True


def _alias_nested_component_body(body: dict[str, Any]) -> None:
    component = body["membership_observer"]
    component["descriptor_body_sha256"] = component["descriptor_file_sha256"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool_as_int"])
def test_every_public_parser_rejects_exact_key_and_bool_integer_mutations(
    mutation: str,
) -> None:
    for value, body_builder, file_builder, parser in _schema_cases():
        raw = file_builder(value)
        document = json.loads(raw)
        body_keys = set(json.loads(body_builder(value)))
        digest_fields = set(document) - body_keys
        assert len(digest_fields) == 1
        body_field = digest_fields.pop()

        def mutate(body: dict[str, Any]) -> None:
            if mutation == "missing":
                body.pop("status")
            elif mutation == "extra":
                body["unexpected_field"] = "forbidden"
            else:
                authority = body["authority"]
                assert type(authority) is dict
                first = next(iter(authority))
                authority[first] = int(authority[first])

        rewritten, file_sha256, body_sha256 = _rewrite(raw, body_field, mutate)
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            parser(
                rewritten,
                expected_file_sha256=file_sha256,
                expected_body_sha256=body_sha256,
            )


@pytest.mark.parametrize("nested_kind", ["artifact", "producer", "subject", "component"])
def test_public_parsers_reject_rehashed_invalid_nested_records(nested_kind: str) -> None:
    chain = _full_chain()
    parser: Callable[..., object]
    mutate: Callable[[dict[str, Any]], None]
    if nested_kind == "artifact":
        raw = host.canonical_host_case_request_v3_file_bytes(chain.request)
        body_field = "host_qualification_case_request_v3_body_sha256"
        parser = host.parse_host_case_request_v3
        mutate = _alias_nested_artifact_body
    elif nested_kind == "producer":
        raw = host.canonical_host_go_v3_file_bytes(chain.go)
        body_field = "host_go_v3_body_sha256"
        parser = host.parse_host_go_v3
        mutate = _alias_nested_producer_body
    elif nested_kind == "subject":
        raw = host.canonical_host_go_v3_file_bytes(chain.go)
        body_field = "host_go_v3_body_sha256"
        parser = host.parse_host_go_v3
        mutate = _corrupt_nested_subject
    else:
        raw = host.canonical_host_observer_anchor_v3_file_bytes(chain.anchor)
        body_field = "host_observer_anchor_v3_body_sha256"
        parser = host.parse_host_observer_anchor_v3
        mutate = _alias_nested_component_body

    rewritten, file_sha256, body_sha256 = _rewrite(raw, body_field, mutate)
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        parser(
            rewritten,
            expected_file_sha256=file_sha256,
            expected_body_sha256=body_sha256,
        )


def test_go_is_phase_ten_direct_binding_and_rejects_future_phase_or_seal() -> None:
    *_, go = _handshake()
    body = go.to_body_dict()
    assert body["committed_host_phase_prefix"] == list(host.HOST_OPERATIONAL_PHASES_V3[:11])
    assert body["storage_runtime_intent"] == go.storage_runtime_intent.to_dict()
    assert len(body["storage_producers"]) == 6
    assert "write_seal" not in body
    assert "host_go_storage_intent_binding" not in body

    raw = host.canonical_host_go_v3_file_bytes(go)
    rewritten, file_sha256, body_sha256 = _rewrite(
        raw,
        "host_go_v3_body_sha256",
        lambda value: value.__setitem__(
            "committed_host_phase_prefix",
            list(host.HOST_OPERATIONAL_PHASES_V3),
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.parse_host_go_v3(
            rewritten,
            expected_file_sha256=file_sha256,
            expected_body_sha256=body_sha256,
        )

    rewritten, file_sha256, body_sha256 = _rewrite(
        raw,
        "host_go_v3_body_sha256",
        lambda value: value.__setitem__(
            "write_seal",
            _ref(host.IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION, "future-seal").to_dict(),
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.parse_host_go_v3(
            rewritten,
            expected_file_sha256=file_sha256,
            expected_body_sha256=body_sha256,
        )


def test_six_role_policy_has_separate_domain_and_chain_fails_closed() -> None:
    bundle = _producer_chain()
    host.validate_host_storage_producer_pre_go_bundle_v1(bundle)
    assert host.HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL != (  # type: ignore[comparison-overlap]
        provisioning_v3.ED25519_SIGNATURE_DOMAIN_LABEL
    )
    assert bundle.policy.storage_producers == _producers()
    assert (
        bundle.statement.signed_payload_sha256
        == hashlib.sha256(
            host.canonical_host_storage_producer_inventory_statement_v1_signed_payload_bytes(
                bundle.statement
            )
        ).hexdigest()
    )

    crosswired = replace(
        bundle.signature_verification_receipt,
        signed_payload_sha256=_hash("crosswired-signed-payload"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.HostStorageProducerPreGoBundleV1(
            policy=bundle.policy,
            statement=bundle.statement,
            signature_verification_receipt=crosswired,
            pre_capability_live_validation=bundle.pre_capability_live_validation,
            pre_go_live_validation=bundle.pre_go_live_validation,
        )


def test_both_live_receipt_ordinals_round_trip_and_bind_the_exact_predecessor() -> None:
    bundle = _producer_chain()
    for receipt in (
        bundle.pre_capability_live_validation,
        bundle.pre_go_live_validation,
    ):
        _round_trip(
            receipt,
            host.canonical_host_storage_producer_live_validation_receipt_v1_body_bytes,
            host.canonical_host_storage_producer_live_validation_receipt_v1_file_bytes,
            host.parse_host_storage_producer_live_validation_receipt_v1,
        )
    assert bundle.pre_capability_live_validation.checkpoint_ordinal == 0
    assert bundle.pre_capability_live_validation.previous_live_validation_receipt is None
    assert bundle.pre_go_live_validation.checkpoint_ordinal == 1
    assert bundle.pre_go_live_validation.previous_live_validation_receipt == (
        bundle.pre_capability_ref
    )


def test_typed_host_and_six_role_bundles_close_a_full_base_v2_host_v3_chain() -> None:
    chain = _full_chain()
    host.validate_host_storage_producer_pre_go_bundle_v1(chain.producer_bundle)
    host.validate_host_pre_go_prefix_v3(
        chain.pre_go.validated_prefix,
        chain.host_bundle,
        chain.producer_bundle,
    )
    _validate_full_chain(chain)
    assert chain.request.base_request_v2 == _host_ref_from_executor(
        _executor_identity(
            chain.base.request,
            host.HOST_CASE_REQUEST_V2_SCHEMA_VERSION,
            executor_v2.canonical_host_case_request_v2_body_bytes,
            executor_v2.canonical_host_case_request_v2_file_bytes,
        )
    )
    assert chain.ready.container_id_commitment_sha256 == (
        chain.base.ready.container_identity_sha256
    )
    assert chain.anchor.membership_observer.matches_legacy(
        chain.host_bundle.policy.expected_facts.components.membership_observer
    )
    assert chain.go.go_payload_sha256 == host.host_go_payload_sha256_v3(
        **{
            field.name: getattr(chain.go, field.name)
            for field in fields(chain.go)
            if field.name
            not in {
                "aggregate_root_case_exclusive",
                "committed_host_phase_prefix",
                "exact_six_producers_committed",
                "go_commit_count",
                "go_committed_monotonic_ns",
                "go_payload_sha256",
                "one_way",
                "same_case_retry_permitted",
                "storage_runtime_intent_committed_before_go",
                "storage_runtime_intent_committed_monotonic_ns",
            }
        }
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "request",
        "host_trust_policy",
        "host_provisioning_statement",
        "host_signature_verification_receipt",
        "host_pre_capability_live_validation",
        "host_pre_go_live_validation",
        "storage_producer_trust_policy",
        "storage_producer_inventory_statement",
        "storage_producer_signature_verification_receipt",
        "storage_producer_pre_capability_live_validation",
        "storage_producer_pre_go_live_validation",
    ],
)
def test_every_pre_go_predecessor_crosswire_fails_the_public_full_chain(
    field_name: str,
) -> None:
    chain = _full_chain()
    current = getattr(chain.pre_go.validated_prefix, field_name)
    assert type(current) is ArtifactRefV1
    crosswired_prefix = _replace_prefix_artifact(
        chain.pre_go.validated_prefix,
        field_name,
        _ref(current.schema_version, "crosswire:" + field_name),
    )
    crosswired_go = _rebuild_go(
        chain.go,
        validated_pre_go_prefix=(
            host.host_provisioning_validated_pre_go_prefix_v3_identity(crosswired_prefix)
        ),
    )
    object.__setattr__(chain.pre_go, "validated_prefix", crosswired_prefix)
    crosswired_chain = replace(chain, pre_go=chain.pre_go, go=crosswired_go)
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        _validate_full_chain(crosswired_chain)

    shared_with_intent = {
        "request",
        "storage_producer_trust_policy",
        "storage_producer_inventory_statement",
        "storage_producer_signature_verification_receipt",
        "storage_producer_pre_capability_live_validation",
    }
    if field_name in shared_with_intent:
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            host.validate_host_ready_anchor_go_v3_chain(
                chain.intent,
                chain.ready,
                chain.anchor,
                crosswired_prefix,
                crosswired_go,
            )


def test_pre_go_prefix_rejects_exact_duplicate_and_file_or_body_aliases() -> None:
    *_, prefix, _ = _handshake()
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        replace(
            prefix,
            host_pre_go_live_validation=prefix.host_pre_capability_live_validation,
        )
    file_alias = ArtifactRefV1(
        schema_version=prefix.host_pre_go_live_validation.schema_version,
        file_sha256=prefix.host_pre_capability_live_validation.file_sha256,
        body_sha256=_hash("host-pre-go-file-alias-body"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        replace(prefix, host_pre_go_live_validation=file_alias)
    body_alias = ArtifactRefV1(
        schema_version=prefix.storage_producer_pre_go_live_validation.schema_version,
        file_sha256=_hash("producer-pre-go-body-alias-file"),
        body_sha256=prefix.storage_producer_pre_capability_live_validation.body_sha256,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        replace(prefix, storage_producer_pre_go_live_validation=body_alias)


@pytest.mark.parametrize(
    "spoof",
    [_AlwaysEqual(), _StringEqualitySpoof(_hash("wrong exact string"))],
    ids=["generic", "str_subclass"],
)
def test_every_exact_string_semantic_rejects_equality_spoofs(spoof: object) -> None:
    request, intent, ready, anchor, prefix, go = _handshake()
    for value in (request, intent, ready, anchor, prefix, go):
        phase_prefix = value.committed_host_phase_prefix
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            _replace_unchecked(
                value,
                committed_host_phase_prefix=(spoof, *phase_prefix[1:]),
            )

    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        _replace_unchecked(prefix, host_provisioning_source_sha256=spoof)

    for field_name in (
        "host_provisioning_source_sha256",
        "host_executor_v2_source_sha256",
        "resource_field_order_sha256",
        "candidate_order_sha256",
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            _replace_unchecked(request, **{field_name: spoof})

    first_name, first_ceiling = request.declared_ceilings[0]
    assert first_name == host.RESOURCE_FIELDS[0]
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        _replace_unchecked(
            request,
            declared_ceilings=((spoof, first_ceiling), *request.declared_ceilings[1:]),
        )

    for field_name in ("resource_field_order_sha256", "candidate_order_sha256"):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            _replace_unchecked(go, **{field_name: spoof})

        payload_arguments = {
            field.name: getattr(go, field.name)
            for field in fields(go)
            if field.name
            not in {
                "aggregate_root_case_exclusive",
                "committed_host_phase_prefix",
                "exact_six_producers_committed",
                "go_commit_count",
                "go_committed_monotonic_ns",
                "go_payload_sha256",
                "one_way",
                "same_case_retry_permitted",
                "storage_runtime_intent_committed_before_go",
                "storage_runtime_intent_committed_monotonic_ns",
            }
        }
        payload_arguments[field_name] = spoof
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            host.host_go_payload_sha256_v3(**payload_arguments)


def test_endpoint_ordinals_reject_generic_and_integer_subclass_equality_spoofs() -> None:
    *_, go = _handshake()
    projection = host.HostGoWriteSealEndpointProjectionV1(
        host_go=host.host_go_v3_identity(go),
        write_seal=_ref(host.IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION, "write-seal-spoof"),
    )
    for spoof in (_AlwaysEqual(), _IntegerEqualitySpoof(999)):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            _replace_unchecked(projection, host_go_phase_ordinal=spoof)
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            _replace_unchecked(projection, write_seal_phase_ordinal=spoof)


def test_component_and_producer_properties_revalidate_forced_current_state() -> None:
    legacy = _legacy_component("property-membership-observer")
    component = _host_component_from_legacy(legacy)
    object.__setattr__(component, "component_id", _AlwaysEqual())
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        component.matches_legacy(legacy)
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        component.to_dict()

    exact_component = _host_component_from_legacy(legacy)
    wrong_legacy: Any = _AlwaysEqual()
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        exact_component.matches_legacy(wrong_legacy)
    object.__setattr__(legacy, "source_sha256", "0" * 64)
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        exact_component.matches_legacy(legacy)

    bundle = _producer_chain()
    producer = bundle.policy.expected_producer_facts[0].producer
    object.__setattr__(
        producer,
        "descriptor_body_sha256",
        producer.descriptor_file_sha256,
    )
    property_reads: tuple[Callable[[], object], ...] = (
        lambda: bundle.policy.expected_producer_inventory_sha256,
        lambda: bundle.policy.storage_producers,
        lambda: bundle.statement.producer_inventory_sha256,
        lambda: bundle.statement.storage_producers,
        lambda: bundle.statement.signed_payload_sha256,
        lambda: bundle.pre_capability_live_validation.observed_producer_inventory_sha256,
    )
    method_calls: tuple[Callable[[], object], ...] = (
        bundle.policy.to_body_dict,
        bundle.statement.to_unsigned_dict,
        bundle.statement.to_body_dict,
        bundle.pre_capability_live_validation.to_body_dict,
        bundle.policy.expected_producer_facts[0].to_dict,
    )
    for consumer in (*property_reads, *method_calls):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            consumer()

    verification_bundle = _producer_chain()
    verifier = verification_bundle.signature_verification_receipt.verifier
    object.__setattr__(
        verifier,
        "descriptor_body_sha256",
        verifier.descriptor_file_sha256,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        verification_bundle.signature_verification_receipt.to_body_dict()

    chain = _full_chain()
    shared_producer = chain.go.storage_producers[0]
    object.__setattr__(
        shared_producer,
        "descriptor_body_sha256",
        shared_producer.descriptor_file_sha256,
    )
    for artifact in (
        chain.request,
        chain.intent,
        chain.ready,
        chain.anchor,
        chain.pre_go.validated_prefix,
        chain.go,
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            artifact.to_body_dict()


def test_every_storage_bundle_identity_property_revalidates_complete_bundle() -> None:
    bundle = _producer_chain()
    crosswired_statement = replace(bundle.statement, signature_hex="01" * 64)
    object.__setattr__(bundle, "statement", crosswired_statement)
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.validate_host_storage_producer_pre_go_bundle_v1(bundle)
    for property_name in (
        "policy_ref",
        "statement_ref",
        "verification_ref",
        "pre_capability_ref",
        "pre_go_ref",
    ):
        with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
            getattr(bundle, property_name)


def test_six_producer_order_and_duplicates_fail_closed() -> None:
    *_, ready, _, _, _ = _handshake()
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        replace(ready, storage_producers=tuple(reversed(ready.storage_producers)))
    duplicated = (
        ready.storage_producers[0],
        ready.storage_producers[0],
        *ready.storage_producers[2:],
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        replace(ready, storage_producers=duplicated)


def test_bundle_and_full_chain_chronology_fail_closed_at_boundaries() -> None:
    producer_bundle = _producer_chain()
    too_early = replace(
        producer_bundle.pre_go_live_validation,
        validated_at_unix_ns=(producer_bundle.pre_capability_live_validation.validated_at_unix_ns),
        validated_at_monotonic_ns=(
            producer_bundle.pre_capability_live_validation.validated_at_monotonic_ns
        ),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.HostStorageProducerPreGoBundleV1(
            policy=producer_bundle.policy,
            statement=producer_bundle.statement,
            signature_verification_receipt=(producer_bundle.signature_verification_receipt),
            pre_capability_live_validation=(producer_bundle.pre_capability_live_validation),
            pre_go_live_validation=too_early,
        )

    chain = _full_chain()
    object.__setattr__(
        chain.go,
        "go_committed_monotonic_ns",
        chain.go.storage_runtime_intent_committed_monotonic_ns,
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        _validate_full_chain(chain)


@pytest.mark.parametrize(
    "tamper_kind",
    ["artifact", "producer", "subject", "component", "bundle"],
)
def test_public_identity_and_chain_validation_reject_nested_frozen_record_tampering(
    tamper_kind: str,
) -> None:
    chain = _full_chain()
    if tamper_kind == "artifact":
        object.__setattr__(chain.request.qualification_plan, "file_sha256", "0" * 64)
    elif tamper_kind == "producer":
        producer = chain.go.storage_producers[0]
        object.__setattr__(
            producer,
            "descriptor_body_sha256",
            producer.descriptor_file_sha256,
        )
    elif tamper_kind == "subject":
        object.__setattr__(chain.go.subject, "candidate_family", "external")
    elif tamper_kind == "component":
        component = chain.anchor.membership_observer
        object.__setattr__(
            component,
            "descriptor_body_sha256",
            component.descriptor_file_sha256,
        )
    else:
        object.__setattr__(
            chain.host_bundle.pre_go_live_validation,
            "checkpoint_ordinal",
            0,
        )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        if tamper_kind == "artifact":
            host.host_case_request_v3_identity(chain.request)
        elif tamper_kind in {"producer", "subject"}:
            host.host_go_v3_identity(chain.go)
        elif tamper_kind == "component":
            host.host_observer_anchor_v3_identity(chain.anchor)
        else:
            _ = chain.host_bundle.pre_go_ref
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        _validate_full_chain(chain)


def test_v3_handshake_chain_closes_crosslinks_and_first_eleven_phase_order() -> None:
    request, intent, ready, anchor, prefix, go = _handshake()
    host.validate_host_request_intent_v3_chain(request, intent)
    host.validate_host_ready_anchor_go_v3_chain(intent, ready, anchor, prefix, go)
    times = (
        request.request_validated_monotonic_ns,
        intent.authorization_validated_monotonic_ns,
        intent.intent_committed_monotonic_ns,
        ready.fresh_cgroup_created_monotonic_ns,
        ready.retained_counter_fds_opened_monotonic_ns,
        ready.initial_cgroup_sample_committed_monotonic_ns,
        ready.container_created_monotonic_ns,
        ready.container_started_monotonic_ns,
        ready.driver_ready_monotonic_ns,
        anchor.observer_anchored_monotonic_ns,
        go.go_committed_monotonic_ns,
    )
    assert all(later > earlier for earlier, later in zip(times, times[1:], strict=False))

    crosswired_anchor = replace(
        anchor,
        ready=_ref(host.HOST_READY_V3_SCHEMA_VERSION, "crosswired-ready"),
    )
    with pytest.raises(host.ForagerMatchedV3HostQualificationProtocolV3Error):
        host.validate_host_ready_anchor_go_v3_chain(
            intent,
            ready,
            crosswired_anchor,
            prefix,
            go,
        )


def test_descriptor_delegates_repository_closure_and_is_permanently_nonready() -> None:
    descriptor = host.HostQualificationProtocolDescriptorV3().to_body_dict()
    assert len(descriptor["public_schemas"]) == 11
    assert len({item["body_sha256_field"] for item in descriptor["public_schemas"]}) == 11
    assert descriptor["go_serializes_host_storage_binding_artifact"] is False
    assert descriptor["go_names_future_write_seal"] is False
    assert descriptor["operational_apis"] == []
    assert all(value is False for value in descriptor["safety_posture"]["capabilities"].values())
    assert descriptor["repository_closure"] == {
        "delegated_to": "later_typed_provider_bundle",
        "full_descriptor_source_dependency_closure_delegated": True,
        "host_protocol_repository_pins_serialized": False,
        "storage_repository_pins_serialized": False,
    }
    assert "audited_dependencies" not in descriptor
    assert "protocol_repository_pins" not in descriptor
    assert "storage_repository_pins" not in descriptor
    assert host.repository_protocol_ready_v3() is False
    with pytest.raises(
        host.ForagerMatchedV3HostQualificationProtocolV3Error,
        match="permanently delegated to the later typed provider bundle",
    ):
        host.validate_repository_protocol_pins_v3()
    assert plan_v3.QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256 == "0" * 64
    assert not any(name.startswith("PINNED_HOST_QUALIFICATION_PROTOCOL") for name in vars(host))
    assert not any(name.startswith("PINNED_STORAGE_BACKEND") for name in vars(host))


def test_finalized_dependencies_and_order_digests_are_exact() -> None:
    assert host.AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256 == (
        "1ff3b76662504333749529926120c0f9a49dfd7aa010f5fc5951282feed4cf56"
    )
    assert host.AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256 == (
        "0e75dc103dc9b5b4f6d50b35e0832a11396a5f18b839deb05604548b1aacc54a"
    )
    assert host.AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256 == (
        "24c205cc4e3d189b4580512c281a84764d81dce51f43e5d959b8650a516343a4"
    )
    assert host.AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_BODY_SHA256 == (
        "e3aa649722e306a4b869db18854a9bc79508cf76f4693468e4abbd3347e5007f"
    )
    assert host.RESOURCE_FIELD_ORDER_SHA256 == (
        "8048ec1a1402b45d8bb4c67684ee7216b242bfb6d3ed9e196c0cfb262c3b93cc"
    )
    assert host.CANDIDATE_ORDER_SHA256 == (
        "d93aaf66053aaf9a7b1c6d268a47740078dd2c1007f7287bd80908707e40b858"
    )


def test_endpoint_projection_is_frozen_nonserialized_and_phase_ordered() -> None:
    *_, go = _handshake()
    projection = host.HostGoWriteSealEndpointProjectionV1(
        host_go=host.host_go_v3_identity(go),
        write_seal=_ref(host.IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION, "write-seal"),
    )
    assert projection.host_go_phase_ordinal == 10
    assert projection.write_seal_phase_ordinal == 17
    assert not hasattr(projection, "to_body_dict")
    with pytest.raises(FrozenInstanceError):
        projection.host_go_phase_ordinal = 11  # type: ignore[misc]


def test_public_frozen_records_reject_runtime_subclasses() -> None:
    with pytest.raises(TypeError):

        class BadGo(host.HostGoCommitmentV3):  # type: ignore[misc]
            pass

    with pytest.raises(TypeError):

        class BadPolicy(host.HostStorageProducerTrustPolicyV1):  # type: ignore[misc]
            pass


def test_source_dependency_is_acyclic_and_has_no_storage_v2_import() -> None:
    source_path = Path(host.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert all("qualification_storage_backend_v2" not in name for name in imports)
    assert "HostGoStorageIntentBindingV1" not in source_path.read_text(encoding="utf-8")
