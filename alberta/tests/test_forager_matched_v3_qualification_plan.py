"""Tests for the unexecuted, score-blind matched-v3 qualification contract."""

from __future__ import annotations

import builtins
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_qualification_plan as plan

_DESCRIPTOR_SHA256 = "258b9e376b82127f912bf2828a6d4e5c7a257ed2a990cd15bf4c9cbd81c17788"
_CONFIGURATION_PLAN_SHA256 = "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
_UNIVERSE_SHA256 = "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
_METRIC_SHA256 = "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
_TRIAL_PLAN_SHA256 = "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
_MATERIALIZER_SHA256 = "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
_PUBLICATION_SHA256 = "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
_PUBLICATION_SOURCE_SHA256 = "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
_FORAGAX_TREE_SHA256 = "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
_QUICKNET_PUBLIC_KEY_HEX = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183"
    "c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4"
    "bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
_QUICKNET_PUBLIC_KEY_RAW_SHA256 = hashlib.sha256(
    bytes.fromhex(_QUICKNET_PUBLIC_KEY_HEX)
).hexdigest()
_QUICKNET_GENESIS_TIME_UNIX = 1_692_803_367
_QUICKNET_PERIOD_SECONDS = 3
_QUICKNET_ROUND = 12_345_678
_QUICKNET_ROUND_TIME_UNIX = (
    _QUICKNET_GENESIS_TIME_UNIX + (_QUICKNET_ROUND - 1) * _QUICKNET_PERIOD_SECONDS
)
_QUICKNET_ROUND_RANDOMNESS = "47175ae9652cb6704c2b509e5d2aa03b609e67039251ff1e162282136c515bf0"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_TRUST_ROOT_RECEIPT_FILE_SHA256 = _sha("qualification-seed-trust-root-receipt-file")


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _canonical_allow_nan(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=True,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _expected_case_derivation(
    receipt: plan.QualificationSeedTrustRootReceiptBinding,
    candidate_id: str,
    ordinal: int,
) -> tuple[str, int, int, str, str]:
    payload = {
        "schema_version": plan.QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        "domain": plan.QUALIFICATION_SEED_DERIVATION_DOMAIN,
        "algorithm": "sha256_canonical_json_high31_v1",
        "provider_chain_hash": receipt.provider_chain_hash,
        "beacon_round": receipt.beacon_round,
        "beacon_randomness_hex": receipt.beacon_randomness_hex,
        "candidate_id": candidate_id,
        "registry_case_ordinal": ordinal,
    }
    payload_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    digests = [
        hashlib.sha256(_canonical({**payload, "lane": lane})).digest()
        for lane in ("environment", "agent")
    ]
    return (
        payload_sha256,
        int.from_bytes(digests[0][:4], "big") & (2**31 - 1),
        int.from_bytes(digests[1][:4], "big") & (2**31 - 1),
        digests[0].hex(),
        digests[1].hex(),
    )


def _expected_registry_digests(
    receipt: plan.QualificationSeedTrustRootReceiptBinding,
) -> tuple[str, str]:
    cases = []
    for ordinal, candidate_id in enumerate(plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS):
        derived = _expected_case_derivation(receipt, candidate_id, ordinal)
        cases.append(
            {
                "case_id": f"qualification_{ordinal:02d}_{candidate_id}",
                "candidate_id": candidate_id,
                "material_class": "public_nonbenchmark_permanently_consumed",
                "registry_case_ordinal": ordinal,
                "derivation_payload_sha256": derived[0],
                "environment_seed": derived[1],
                "agent_seed": derived[2],
                "environment_seed_derivation_sha256": derived[3],
                "agent_seed_derivation_sha256": derived[4],
            }
        )
    body = {
        "schema_version": plan.QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        "material_class": "public_nonbenchmark_permanently_consumed",
        "provider": {
            "provider_id": receipt.provider_id,
            "provider_chain_hash": receipt.provider_chain_hash,
            "signature_scheme": receipt.signature_scheme,
            "provider_public_key_sha256": receipt.provider_public_key_sha256,
            "beacon_round": receipt.beacon_round,
            "beacon_time_unix": receipt.beacon_time_unix,
            "observation_cutoff_unix": receipt.observation_cutoff_unix,
            "beacon_signature_sha256": receipt.beacon_signature_sha256,
            "beacon_randomness_hex": receipt.beacon_randomness_hex,
            "pulse_record_schema_version": receipt.pulse_record_schema_version,
            "pulse_record_file_sha256": receipt.pulse_record_file_sha256,
            "pulse_record_body_sha256": receipt.pulse_record_body_sha256,
        },
        "derivation": {
            "schema_version": plan.QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
            "domain": plan.QUALIFICATION_SEED_DERIVATION_DOMAIN,
            "algorithm": "sha256_canonical_json_high31_v1",
        },
        "candidate_order": list(plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS),
        "cases": cases,
    }
    body_sha256 = hashlib.sha256(_canonical(body)).hexdigest()
    file_sha256 = hashlib.sha256(
        _canonical({**body, "registry_body_sha256": body_sha256})
    ).hexdigest()
    return file_sha256, body_sha256


class SyntheticInputs(NamedTuple):
    sources: tuple[plan.SourceRequirement, ...]
    runtime: plan.RuntimeRequirement
    trust_root_receipt: plan.QualificationSeedTrustRootReceiptBinding
    expected_trust_root_receipt_file_sha256: str
    expected_trust_root_receipt_binding_sha256: str
    seed_registry: plan.QualificationSeedRegistryBinding
    cases: tuple[plan.QualificationCase, ...]
    resources: tuple[plan.CandidateResourceRequirement, ...]
    publications: tuple[plan.CandidatePublicationBinding, ...]


def _resource(
    candidate_id: str, overrides: dict[str, object] | None = None
) -> plan.CandidateResourceRequirement:
    values: dict[str, object] = {
        "max_environment_interactions": 499_712,
        "max_optimizer_updates": 1,
        "max_gradient_updates": 1,
        "max_sample_updates": 1,
        "max_trainable_parameters": 1,
        "max_frozen_parameters": 1,
        "max_optimizer_state_elements": 1,
        "max_optimizer_state_bytes": 1,
        "max_target_copy_elements": 1,
        "max_target_copy_bytes": 1,
        "max_replay_capacity_transitions": 1,
        "max_replay_peak_bytes": 1,
        "max_rollout_storage_elements": 1,
        "max_rollout_peak_bytes": 1,
        "max_recurrent_carry_elements": 1,
        "max_recurrent_carry_bytes": 1,
        "max_rtrl_sensitivity_elements": 1,
        "max_rtrl_sensitivity_bytes": 1,
        "max_eligibility_elements": 1,
        "max_eligibility_bytes": 1,
        "max_peak_rss_bytes": 1,
        "max_cpu_time_ns": 1,
        "max_wall_time_ns": 1,
        "max_temporary_peak_bytes": 1,
        "max_disk_peak_bytes": 1,
        "max_thread_count": 1,
        "max_attempt_count": 1,
        "max_failure_count": 0,
    }
    if overrides is not None:
        values.update(overrides)
    constructor = cast(Any, plan.CandidateResourceRequirement)
    return cast(plan.CandidateResourceRequirement, constructor(candidate_id=candidate_id, **values))


def _publication(
    candidate_id: str,
    source_tree_sha256: str,
) -> plan.CandidatePublicationBinding:
    if candidate_id in {"adapted_full_rainbow", "adapted_ppo_gru"}:
        return plan.CandidatePublicationBinding(
            candidate_id=candidate_id,
            publisher_kind="adapter_reward_publication_v1",
            descriptor_schema_version=(
                "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            ),
            descriptor_sha256=_PUBLICATION_SHA256,
            publication_schema_version=("alberta.forager_matched_v3.adapter_reward_publication.v1"),
            source_id="local_alberta",
            source_tree_sha256=source_tree_sha256,
            implementation_path=(
                "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
            ),
            implementation_source_sha256=_PUBLICATION_SOURCE_SHA256,
            reload_validator_schema_version=(
                "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            ),
            reload_validator_descriptor_sha256=_PUBLICATION_SHA256,
            reload_validator_implementation_path=(
                "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
            ),
            reload_validator_source_sha256=_PUBLICATION_SOURCE_SHA256,
        )
    return plan.CandidatePublicationBinding(
        candidate_id=candidate_id,
        publisher_kind="synthetic_content_publisher_v1",
        descriptor_schema_version="alberta.synthetic.descriptor.v1",
        descriptor_sha256=_sha(f"publication-descriptor:{candidate_id}"),
        publication_schema_version="alberta.synthetic.publication.v1",
        source_id="local_alberta",
        source_tree_sha256=source_tree_sha256,
        implementation_path=f"publishers/{candidate_id}.py",
        implementation_source_sha256=_sha(f"publication-source:{candidate_id}"),
        reload_validator_schema_version="alberta.synthetic.reload_validator.v1",
        reload_validator_descriptor_sha256=_sha(f"reload-validator-descriptor:{candidate_id}"),
        reload_validator_implementation_path=f"validators/{candidate_id}.py",
        reload_validator_source_sha256=_sha(f"reload-validator-source:{candidate_id}"),
    )


def _synthetic_inputs() -> SyntheticInputs:
    sources = (
        plan.SourceRequirement(
            source_id="external_foragax_agents",
            closure_kind="derived_checkout_manifest_tree",
            manifest_schema_version=("alberta.forager_matched_v3_external_materialization.v1"),
            manifest_file_sha256=_sha("external-manifest-file"),
            manifest_body_sha256=_sha("external-manifest-body"),
            source_tree_sha256=_sha("external-source-tree"),
            inventory_sha256=_sha("external-inventory"),
            file_count=10,
            directory_count=2,
            total_bytes=1_000,
        ),
        plan.SourceRequirement(
            source_id="local_alberta",
            closure_kind="normalized_local_source_snapshot",
            manifest_schema_version="alberta.forager_matched_v3.local_source_snapshot.v1",
            manifest_file_sha256=_sha("local-manifest-file"),
            manifest_body_sha256=_sha("local-manifest-body"),
            source_tree_sha256=_sha("local-source-tree"),
            inventory_sha256=_sha("local-inventory"),
            file_count=20,
            directory_count=3,
            total_bytes=2_000,
        ),
    )
    runtime = plan.RuntimeRequirement(
        executor_kind="networkless_oci_cpu",
        runtime_executable_sha256=_sha("runtime-executable"),
        runtime_version_output_sha256=_sha("runtime-version-output"),
        image_digest=f"sha256:{_sha('image')}",
        image_config_sha256=_sha("image-config"),
        runtime_profile_sha256=_sha("runtime-profile"),
        python_implementation="CPython",
        python_version="3.12.11",
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        foragax_version="0.55.0",
        foragax_install_tree_sha256=_FORAGAX_TREE_SHA256,
        platform="linux/amd64",
        default_prng_impl="threefry2x32",
        jax_enable_x64=False,
        threefry_partitionable=True,
        sandbox_descriptor_sha256=_sha("sandbox"),
        helper_bindings=(
            plan.RuntimeHelperBinding(
                helper_id="drand_verify",
                executable_sha256=_sha("drand-verify-executable"),
                version_output_sha256=_sha("drand-verify-version"),
            ),
            plan.RuntimeHelperBinding(
                helper_id="oci_runtime",
                executable_sha256=_sha("oci-runtime-executable"),
                version_output_sha256=_sha("oci-runtime-version"),
            ),
            plan.RuntimeHelperBinding(
                helper_id="resource_observer",
                executable_sha256=_sha("resource-observer-executable"),
                version_output_sha256=_sha("resource-observer-version"),
            ),
        ),
    )
    registry_file_sha256 = _sha("qualification-seed-registry-file")
    registry_body_sha256 = _sha("qualification-seed-registry-body")
    trust_root_receipt = plan.QualificationSeedTrustRootReceiptBinding(
        receipt_schema_version=plan.QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION,
        receipt_file_sha256=_sha("qualification-seed-trust-root-receipt-file"),
        receipt_body_sha256=_sha("qualification-seed-trust-root-receipt-body"),
        provider_id="drand_quicknet",
        provider_chain_hash=("52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"),
        signature_scheme="bls-unchained-g1-rfc9380",
        provider_public_key_sha256=_QUICKNET_PUBLIC_KEY_RAW_SHA256,
        beacon_round=_QUICKNET_ROUND,
        beacon_time_unix=_QUICKNET_ROUND_TIME_UNIX,
        observation_cutoff_unix=_QUICKNET_ROUND_TIME_UNIX + 3_600,
        beacon_signature_sha256=_QUICKNET_ROUND_RANDOMNESS,
        beacon_randomness_hex=_QUICKNET_ROUND_RANDOMNESS,
        pulse_record_schema_version=plan.QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
        pulse_record_file_sha256=_sha("qualification-seed-pulse-record-file"),
        pulse_record_body_sha256=_sha("qualification-seed-pulse-record-body"),
        seed_registry_schema_version=plan.QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        seed_registry_file_sha256=registry_file_sha256,
        seed_registry_body_sha256=registry_body_sha256,
        seed_derivation_algorithm="sha256_canonical_json_high31_v1",
        timestamp_source="drand_quicknet_round_time",
        offline_verifier_schema_version=(plan.QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION),
        offline_verifier_descriptor_sha256=_sha("quicknet-offline-verifier-descriptor"),
        offline_verifier_source_id="local_alberta",
        offline_verifier_source_tree_sha256=sources[1].source_tree_sha256,
        offline_verifier_implementation_path="verifiers/drand_quicknet_offline.py",
        offline_verifier_source_sha256=_sha("quicknet-offline-verifier-source"),
        offline_verifier_runtime_helper_id="drand_verify",
        offline_verifier_executable_sha256=runtime.helper_bindings[0].executable_sha256,
        offline_verifier_version_output_sha256=(runtime.helper_bindings[0].version_output_sha256),
        offline_verifier_implementation_status=(
            "required_external_preaccepted_not_implemented_here"
        ),
        offline_signature_verification_required=True,
        external_preacceptance_required=True,
    )
    registry_file_sha256, registry_body_sha256 = _expected_registry_digests(trust_root_receipt)
    trust_root_receipt = dataclasses.replace(
        trust_root_receipt,
        seed_registry_file_sha256=registry_file_sha256,
        seed_registry_body_sha256=registry_body_sha256,
    )
    seed_registry = plan.QualificationSeedRegistryBinding(
        registry_schema_version=plan.QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        registry_file_sha256=registry_file_sha256,
        registry_body_sha256=registry_body_sha256,
        derivation_schema_version=plan.QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        derivation_domain=plan.QUALIFICATION_SEED_DERIVATION_DOMAIN,
        provider_id=trust_root_receipt.provider_id,
        provider_identity_sha256=trust_root_receipt.provider_chain_hash,
        provider_receipt_schema_version=(plan.QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION),
        provider_receipt_file_sha256=trust_root_receipt.receipt_file_sha256,
        provider_receipt_body_sha256=trust_root_receipt.receipt_body_sha256,
        trust_root_receipt_binding_sha256=hashlib.sha256(
            _canonical(trust_root_receipt.to_dict())
        ).hexdigest(),
        external_authentication_required=True,
        issued_before_observation_required=True,
    )
    cases_list: list[plan.QualificationCase] = []
    for index, candidate_id in enumerate(plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS):
        (
            derivation_payload_sha256,
            environment_seed,
            agent_seed,
            environment_seed_derivation_sha256,
            agent_seed_derivation_sha256,
        ) = _expected_case_derivation(trust_root_receipt, candidate_id, index)
        cases_list.append(
            plan.QualificationCase(
                case_id=f"qualification_{index:02d}_{candidate_id}",
                candidate_id=candidate_id,
                material_class="public_nonbenchmark_permanently_consumed",
                registry_case_ordinal=index,
                seed_registry_binding_sha256=hashlib.sha256(
                    _canonical(seed_registry.to_dict())
                ).hexdigest(),
                seed_registry_file_sha256=seed_registry.registry_file_sha256,
                seed_registry_body_sha256=seed_registry.registry_body_sha256,
                provider_identity_sha256=seed_registry.provider_identity_sha256,
                provider_receipt_file_sha256=(seed_registry.provider_receipt_file_sha256),
                provider_receipt_body_sha256=(seed_registry.provider_receipt_body_sha256),
                derivation_schema_version=seed_registry.derivation_schema_version,
                derivation_domain=seed_registry.derivation_domain,
                derivation_payload_sha256=derivation_payload_sha256,
                environment_seed=environment_seed,
                agent_seed=agent_seed,
                environment_seed_derivation_sha256=(environment_seed_derivation_sha256),
                agent_seed_derivation_sha256=agent_seed_derivation_sha256,
            )
        )
    cases = tuple(cases_list)
    resources = tuple(
        _resource(candidate_id) for candidate_id in plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    )
    local_source_tree_sha256 = sources[1].source_tree_sha256
    publications = tuple(
        _publication(
            candidate_id,
            local_source_tree_sha256,
        )
        for candidate_id in plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    )
    return SyntheticInputs(
        sources,
        runtime,
        trust_root_receipt,
        trust_root_receipt.receipt_file_sha256,
        hashlib.sha256(_canonical(trust_root_receipt.to_dict())).hexdigest(),
        seed_registry,
        cases,
        resources,
        publications,
    )


def _trusted_receipt_binding_sha256() -> str:
    return hashlib.sha256(_canonical(_synthetic_inputs().trust_root_receipt.to_dict())).hexdigest()


def _build(inputs: SyntheticInputs | None = None) -> dict[str, Any]:
    selected = _synthetic_inputs() if inputs is None else inputs
    return plan.build_matched_v3_qualification_plan(
        source_requirements=selected.sources,
        runtime_requirement=selected.runtime,
        qualification_seed_registry=selected.seed_registry,
        qualification_seed_trust_root_receipt=selected.trust_root_receipt,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            selected.expected_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            selected.expected_trust_root_receipt_binding_sha256
        ),
        qualification_cases=selected.cases,
        resource_requirements=selected.resources,
        result_publication_bindings=selected.publications,
    )


def _rehash_body(value: dict[str, Any]) -> bytes:
    body = copy.deepcopy(value)
    body.pop("plan_body_sha256", None)
    value["plan_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(value)


def _parse_with_actual_digest(raw: bytes) -> dict[str, Any]:
    return plan.parse_matched_v3_qualification_plan_artifact(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )


def test_descriptor_has_exact_literal_identity_and_canonical_newline() -> None:
    raw = plan.canonical_matched_v3_qualification_plan_descriptor_bytes()
    assert len(raw) == 9_608
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert plan.QUALIFICATION_PLAN_DESCRIPTOR_SHA256 == _DESCRIPTOR_SHA256
    assert plan.matched_v3_qualification_plan_descriptor_sha256() == _DESCRIPTOR_SHA256
    assert plan.parse_matched_v3_qualification_plan_descriptor(raw) == (
        plan.matched_v3_qualification_plan_descriptor()
    )


def test_descriptor_snapshot_is_detached() -> None:
    first = plan.matched_v3_qualification_plan_descriptor()
    first["claims"]["production_plan_issued"] = True
    second = plan.matched_v3_qualification_plan_descriptor()
    assert second["claims"]["production_plan_issued"] is False


def test_descriptor_root_dependencies_and_candidate_order_are_exact() -> None:
    descriptor = plan.matched_v3_qualification_plan_descriptor()
    assert set(descriptor) == {
        "schema_version",
        "status",
        "classification",
        "dependencies",
        "required_source_ids",
        "candidate_order",
        "receipt_schemas",
        "probe_profile_ids",
        "runtime_contract",
        "qualification_seed_registry_contract",
        "publication_roundtrip_contract",
        "canonicalization",
        "authentication_policy",
        "claims",
        "limitations",
    }
    assert descriptor["status"] == "contract_implemented_no_production_plan"
    assert descriptor["classification"] == "content_only_unexecuted_non_authorizing"
    assert descriptor["required_source_ids"] == ["external_foragax_agents", "local_alberta"]
    assert descriptor["candidate_order"] == list(plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS)
    assert len(descriptor["candidate_order"]) == 28
    dependencies = descriptor["dependencies"]
    assert dependencies["configuration_plan"]["sha256"] == _CONFIGURATION_PLAN_SHA256
    assert dependencies["candidate_universe"]["sha256"] == _UNIVERSE_SHA256
    assert dependencies["cumulative_reward_metric"]["sha256"] == _METRIC_SHA256
    assert dependencies["trial_block_generator_plan"]["sha256"] == _TRIAL_PLAN_SHA256
    assert dependencies["external_materializer"]["sha256"] == _MATERIALIZER_SHA256
    assert dependencies["external_materializer"]["source_sha256"] == (
        "5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"
    )
    assert dependencies["adapter_reward_publication"]["descriptor_sha256"] == _PUBLICATION_SHA256
    assert dependencies["adapter_reward_publication"]["source_sha256"] == (
        _PUBLICATION_SOURCE_SHA256
    )
    assert dependencies["adapter_reward_bundle"] == {
        "schema_version": "alberta.forager_matched_v3.adapter_reward_bundle_descriptor.v1",
        "descriptor_sha256": ("1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5"),
        "source_sha256": ("22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469"),
    }
    assert dependencies["reward_scorer"]["source_sha256"] == (
        "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert dependencies["foragax_bridge"]["descriptor_sha256"] == (
        "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
    )
    assert dependencies["foragax_bridge"]["source_sha256"] == (
        "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
    )
    assert dependencies["full_rainbow_runner"]["descriptor_sha256"] == (
        "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
    )
    assert dependencies["full_rainbow_runner"]["source_sha256"] == (
        "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
    )
    assert dependencies["ppo_gru_runner"]["descriptor_sha256"] == (
        "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
    )
    assert dependencies["ppo_gru_runner"]["source_sha256"] == (
        "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
    )


def test_descriptor_lower_dependency_digest_mutation_fails() -> None:
    descriptor = plan.matched_v3_qualification_plan_descriptor()
    descriptor["dependencies"]["configuration_plan"]["sha256"] = "0" * 64
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        plan.parse_matched_v3_qualification_plan_descriptor(_canonical(descriptor))


def test_descriptor_receipt_and_probe_schemas_are_frozen() -> None:
    descriptor = plan.matched_v3_qualification_plan_descriptor()
    assert descriptor["probe_profile_ids"] == [
        "qualification_seed_provenance_v1",
        "content_import_v1",
        "environment_rng_replay_v1",
        "candidate_seed_transport_v1",
        "full_horizon_resource_v1",
        "result_publication_roundtrip_v1",
    ]
    assert descriptor["receipt_schemas"] == {
        "source_observation": "alberta.forager_matched_v3.source_observation.v1",
        "runtime_observation": "alberta.forager_matched_v3.runtime_observation.v1",
        "qualification_seed_observation": (
            "alberta.forager_matched_v3.qualification_seed_observation.v1"
        ),
        "candidate_observation": "alberta.forager_matched_v3.candidate_observation.v1",
        "resource_observation": "alberta.forager_matched_v3.resource_observation.v1",
        "publication_observation": ("alberta.forager_matched_v3.result_publication_observation.v1"),
        "fresh_replay_observation": ("alberta.forager_matched_v3.fresh_replay_observation.v1"),
        "qualification_bundle": "alberta.forager_matched_v3.qualification_bundle.v1",
    }


def test_runtime_seed_registry_and_publication_contracts_are_exact() -> None:
    descriptor = plan.matched_v3_qualification_plan_descriptor()
    assert descriptor["runtime_contract"] == {
        "executor_kind": "networkless_oci_cpu",
        "python_implementation": "CPython",
        "python_version_series": "3.12.x",
        "jax_version": "0.11.0",
        "jaxlib_version": "0.11.0",
        "foragax_version": "0.55.0",
        "required_helper_ids": ["drand_verify", "oci_runtime", "resource_observer"],
        "networkless_oci_workflow": [
            "verify_image_runtime_and_helper_content_identities",
            "disable_network_before_candidate_import",
            "record_exact_version_outputs_and_runtime_profile",
            "run_only_score_blind_qualification_probes",
        ],
    }
    assert descriptor["qualification_seed_registry_contract"] == {
        "registry_schema_version": plan.QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        "derivation_schema_version": plan.QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        "derivation_domain": plan.QUALIFICATION_SEED_DERIVATION_DOMAIN,
        "provider_receipt_schema_version": (
            plan.QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION
        ),
        "pulse_record_schema_version": plan.QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
        "offline_verifier_schema_version": (
            plan.QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION
        ),
        "provider_id": "drand_quicknet",
        "provider_chain_hash": ("52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"),
        "signature_scheme": "bls-unchained-g1-rfc9380",
        "provider_public_key_hex": _QUICKNET_PUBLIC_KEY_HEX,
        "provider_public_key_raw_sha256": _QUICKNET_PUBLIC_KEY_RAW_SHA256,
        "provider_public_key_hash_input": "hex_decoded_96_bytes",
        "genesis_time_unix": _QUICKNET_GENESIS_TIME_UNIX,
        "period_seconds": _QUICKNET_PERIOD_SECONDS,
        "bls_message_scope": "unchained_round_only",
        "randomness_derivation": "sha256_raw_signature_bytes",
        "timestamp_source": "drand_quicknet_round_time",
        "seed_pair_derivation_algorithm": "sha256_canonical_json_high31_v1",
        "offline_verifier_runtime_helper_id": "drand_verify",
        "offline_verifier_runtime_helper_content_bound": True,
        "offline_verifier_implementation_status": (
            "required_external_preaccepted_not_implemented_here"
        ),
        "full_file_and_body_digests_required": True,
        "every_case_binds_registry_receipt_and_derivation_payload": True,
        "receipt_binds_pulse_record_key_round_time_signature_randomness": True,
        "pulse_record_requires_raw_public_key_and_signature": True,
        "quicknet_signature_authenticates_seed_registry": False,
        "quicknet_signature_authenticates_trust_root_receipt": False,
        "registry_seed_pairs_deterministically_derived_from_pulse": True,
        "deterministic_seed_derivation_implemented_here": True,
        "canonical_registry_file_and_body_derivation_implemented_here": True,
        "independent_trust_root_receipt_file_pin_required": True,
        "independent_trust_root_receipt_binding_pin_required": True,
        "offline_signature_verification_required": True,
        "offline_signature_verification_implemented_here": False,
        "pulse_time_alone_proves_preobservation": False,
        "external_preacceptance_chronology_required": True,
        "external_preacceptance_chronology_implemented_here": False,
        "external_authentication_required": True,
        "issued_before_observation_required": True,
        "issuer_api_exposed": False,
    }
    assert descriptor["publication_roundtrip_contract"] == {
        "all_candidates_require_source_closure_membership": True,
        "reload_validator_binding_required": True,
        "atomic_publication_required": True,
        "strict_reload_required": True,
        "full_digest_equivalence_required": True,
        "score_or_reward_magnitude_observed": False,
    }


def test_build_has_exact_root_body_and_full_file_digest() -> None:
    built = _build()
    assert set(built) == {
        "schema_version",
        "status",
        "classification",
        "bindings",
        "source_requirements",
        "runtime_requirement",
        "qualification_seed_trust_root_receipt",
        "qualification_seed_registry",
        "probe_profiles",
        "candidate_requirements",
        "resource_contract",
        "seed_boundary",
        "failure_policy",
        "authentication_policy",
        "claims",
        "limitations",
        "plan_body_sha256",
    }
    body = copy.deepcopy(built)
    supplied = body.pop("plan_body_sha256")
    assert supplied == hashlib.sha256(_canonical(body)).hexdigest()
    raw = plan.canonical_matched_v3_qualification_plan_bytes(
        built,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )
    file_sha256 = plan.matched_v3_qualification_plan_sha256(
        built,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )
    assert file_sha256 == hashlib.sha256(raw).hexdigest()
    assert (
        plan.parse_matched_v3_qualification_plan_artifact(
            raw,
            expected_file_sha256=file_sha256,
            expected_qualification_seed_trust_root_receipt_file_sha256=(
                _TRUST_ROOT_RECEIPT_FILE_SHA256
            ),
            expected_qualification_seed_trust_root_receipt_binding_sha256=(
                _trusted_receipt_binding_sha256()
            ),
        )
        == built
    )


def test_build_output_and_inputs_are_detached() -> None:
    inputs = _synthetic_inputs()
    first = _build(inputs)
    first["candidate_requirements"][0]["probe_profile_ids"].append("mutation")
    second = _build(inputs)
    assert second["candidate_requirements"][0]["probe_profile_ids"] == [
        "qualification_seed_provenance_v1",
        "content_import_v1",
        "environment_rng_replay_v1",
        "candidate_seed_transport_v1",
        "full_horizon_resource_v1",
        "result_publication_roundtrip_v1",
    ]


def test_candidate_configuration_order_source_split_and_coverage_are_exact() -> None:
    candidates = _build()["candidate_requirements"]
    assert [item["candidate_id"] for item in candidates] == list(
        plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    )
    external = {
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
    }
    assert len({item["configuration_record_sha256"] for item in candidates}) == 28
    for item in candidates:
        assert item["source_id"] == (
            "external_foragax_agents" if item["candidate_id"] in external else "local_alberta"
        )
        assert len(item["configuration_record_sha256"]) == 64


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered"])
def test_candidate_case_coverage_fails_closed(kind: str) -> None:
    inputs = _synthetic_inputs()
    cases = list(inputs.cases)
    if kind == "missing":
        cases.pop()
    elif kind == "duplicate":
        cases[-1] = cases[0]
    else:
        cases[0], cases[1] = cases[1], cases[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(cases=tuple(cases)))


def test_unknown_candidate_fails_at_typed_boundary() -> None:
    case = _synthetic_inputs().cases[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(case, case_id="qual_unknown", candidate_id="unknown")


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered"])
def test_source_coverage_fails_closed(kind: str) -> None:
    inputs = _synthetic_inputs()
    sources = list(inputs.sources)
    if kind == "missing":
        sources.pop()
    elif kind == "duplicate":
        sources[1] = sources[0]
    else:
        sources.reverse()
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(sources=tuple(sources)))


def test_external_and_local_source_trees_remain_distinct_under_coherent_rebinding() -> None:
    inputs = _synthetic_inputs()
    shared_tree_sha256 = inputs.sources[0].source_tree_sha256
    sources = (
        inputs.sources[0],
        dataclasses.replace(inputs.sources[1], source_tree_sha256=shared_tree_sha256),
    )
    receipt = dataclasses.replace(
        inputs.trust_root_receipt,
        receipt_file_sha256=_sha("equal-tree-trust-root-receipt-file"),
        receipt_body_sha256=_sha("equal-tree-trust-root-receipt-body"),
        offline_verifier_source_tree_sha256=shared_tree_sha256,
    )
    receipt_binding_sha256 = hashlib.sha256(_canonical(receipt.to_dict())).hexdigest()
    registry = dataclasses.replace(
        inputs.seed_registry,
        provider_receipt_file_sha256=receipt.receipt_file_sha256,
        provider_receipt_body_sha256=receipt.receipt_body_sha256,
        trust_root_receipt_binding_sha256=receipt_binding_sha256,
    )
    registry_binding_sha256 = hashlib.sha256(_canonical(registry.to_dict())).hexdigest()
    cases = tuple(
        dataclasses.replace(
            case,
            seed_registry_binding_sha256=registry_binding_sha256,
            provider_receipt_file_sha256=receipt.receipt_file_sha256,
            provider_receipt_body_sha256=receipt.receipt_body_sha256,
        )
        for case in inputs.cases
    )
    publications = tuple(
        dataclasses.replace(item, source_tree_sha256=shared_tree_sha256)
        for item in inputs.publications
    )
    coherent = inputs._replace(
        sources=sources,
        trust_root_receipt=receipt,
        expected_trust_root_receipt_file_sha256=receipt.receipt_file_sha256,
        expected_trust_root_receipt_binding_sha256=receipt_binding_sha256,
        seed_registry=registry,
        cases=cases,
        publications=publications,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="must be distinct"):
        _build(coherent)


def test_unknown_source_id_fails_at_typed_boundary() -> None:
    constructor = cast(Any, plan.SourceRequirement)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        constructor(
            source_id="unknown_source",
            closure_kind="derived_checkout_manifest_tree",
            manifest_schema_version="alberta.synthetic.source.v1",
            manifest_file_sha256=_sha("a"),
            manifest_body_sha256=_sha("b"),
            source_tree_sha256=_sha("c"),
            inventory_sha256=_sha("d"),
            file_count=1,
            directory_count=0,
            total_bytes=1,
        )


@pytest.mark.parametrize("field", ["cases", "resources", "publications"])
def test_every_candidate_scoped_input_is_required(field: str) -> None:
    inputs = _synthetic_inputs()
    values = getattr(inputs, field)[:-1]
    updated = inputs._replace(**{field: values})
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(updated)


def test_runtime_fields_have_no_defaults_and_helper_binding_is_complete() -> None:
    signature = inspect.signature(plan.RuntimeRequirement)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError):
        cast(Any, plan.RuntimeRequirement)(executor_kind="networkless_oci_cpu")
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(_synthetic_inputs().runtime, helper_bindings=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_implementation", "PyPy"),
        ("python_implementation", "cpython"),
        ("python_implementation", True),
        ("python_version", True),
        ("python_version", "3.12"),
        ("python_version", "3.12.01"),
        ("python_version", "3.12.-1"),
        ("python_version", "3.12.1rc1"),
        ("python_version", "3.11.9"),
        ("python_version", "3.13.0"),
        ("python_version", "3.12.1000"),
    ],
)
def test_runtime_requires_exact_cpython_312_patch_release(field: str, value: object) -> None:
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(_synthetic_inputs().runtime, **{field: value})


@pytest.mark.parametrize("kind", ["missing", "reordered", "unknown", "duplicate", "list"])
def test_runtime_helper_identity_set_is_exact_nonempty_and_sorted(kind: str) -> None:
    runtime = _synthetic_inputs().runtime
    helpers: object
    if kind == "missing":
        helpers = runtime.helper_bindings[:1]
    elif kind == "reordered":
        helpers = tuple(reversed(runtime.helper_bindings))
    elif kind == "unknown":
        helpers = (
            runtime.helper_bindings[0],
            dataclasses.replace(runtime.helper_bindings[1], helper_id="shell"),
        )
    elif kind == "duplicate":
        helpers = (runtime.helper_bindings[0], runtime.helper_bindings[0])
    else:
        helpers = list(runtime.helper_bindings)
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(runtime, helper_bindings=helpers)


@pytest.mark.parametrize("field", ["executable_sha256", "version_output_sha256"])
def test_runtime_helper_executable_and_version_hashes_are_mandatory(field: str) -> None:
    helper = _synthetic_inputs().runtime.helper_bindings[0]
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(helper, **{field: "0" * 64})


@pytest.mark.parametrize("field", ["executable_sha256", "version_output_sha256"])
def test_drand_verifier_runtime_helper_is_bound_by_independent_receipt(field: str) -> None:
    inputs = _synthetic_inputs()
    helpers = list(inputs.runtime.helper_bindings)
    helpers[0] = dataclasses.replace(
        helpers[0],
        **{field: _sha(f"substituted-drand-verifier-{field}")},
    )
    runtime = dataclasses.replace(inputs.runtime, helper_bindings=tuple(helpers))
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="runtime helper"):
        _build(inputs._replace(runtime=runtime))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jax_enable_x64", 0),
        ("threefry_partitionable", 1),
        ("jax_version", "0.11.1"),
        ("foragax_install_tree_sha256", "0" * 64),
        ("platform", "linux/arm64"),
    ],
)
def test_runtime_identity_drift_fails_closed(field: str, value: object) -> None:
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(_synthetic_inputs().runtime, **{field: value})


def test_all_zero_digest_sentinel_fails_across_typed_content_boundaries() -> None:
    inputs = _synthetic_inputs()
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.sources[0], inventory_sha256="0" * 64)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.runtime, runtime_profile_sha256="0" * 64)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.runtime, image_digest=f"sha256:{'0' * 64}")
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.trust_root_receipt, beacon_signature_sha256="0" * 64)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.seed_registry, provider_identity_sha256="0" * 64)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(inputs.publications[0], descriptor_sha256="0" * 64)


def test_serialized_runtime_omission_fails_even_with_rehashed_body_and_file() -> None:
    built = _build()
    del built["runtime_requirement"]["sandbox_descriptor_sha256"]
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


def test_serialized_helper_reordering_fails_with_coherent_rehash() -> None:
    built = _build()
    built["runtime_requirement"]["helper_bindings"].reverse()
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


@pytest.mark.parametrize("bad", [True, 1.0, -1, 2**63])
def test_resource_integer_bounds_reject_bool_float_negative_and_overflow(bad: object) -> None:
    candidate_id = plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _resource(candidate_id, {"max_optimizer_updates": bad})


def test_resource_horizon_and_attempt_failure_bounds_fail_closed() -> None:
    candidate_id = plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _resource(candidate_id, {"max_environment_interactions": 499_711})
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _resource(candidate_id, {"max_attempt_count": 1, "max_failure_count": 1})


@pytest.mark.parametrize(
    "case_id",
    ["future_case", "held_out_case", "trial_block_case", "protected_case", "confirmatory_case"],
)
def test_qualification_case_forbids_nonpublic_or_scientific_material(case_id: str) -> None:
    case = _synthetic_inputs().cases[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(case, case_id=case_id)


def test_qualification_derivation_identities_are_distinct_and_globally_unique() -> None:
    inputs = _synthetic_inputs()
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(
            inputs.cases[0],
            agent_seed_derivation_sha256=inputs.cases[0].environment_seed_derivation_sha256,
        )
    cases = list(inputs.cases)
    cases[1] = dataclasses.replace(
        cases[1],
        environment_seed_derivation_sha256=cases[0].environment_seed_derivation_sha256,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(cases=tuple(cases)))

    cases = list(inputs.cases)
    cases[1] = dataclasses.replace(
        cases[1],
        derivation_payload_sha256=cases[0].derivation_payload_sha256,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(cases=tuple(cases)))


def test_qualification_seed_pairs_are_exact_deterministic_pulse_derivations() -> None:
    inputs = _synthetic_inputs()
    cases = list(inputs.cases)
    cases[0] = dataclasses.replace(
        cases[0],
        environment_seed=(cases[0].environment_seed + 1) & (2**31 - 1),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="pulse derivation"):
        _build(inputs._replace(cases=tuple(cases)))


def test_seed_registry_and_every_case_are_bound_to_exact_authenticated_provenance() -> None:
    inputs = _synthetic_inputs()
    built = _build(inputs)
    receipt = built["qualification_seed_trust_root_receipt"]
    registry = built["qualification_seed_registry"]
    assert receipt == inputs.trust_root_receipt.to_dict()
    assert registry == inputs.seed_registry.to_dict()
    receipt_binding_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    binding_sha256 = hashlib.sha256(_canonical(registry)).hexdigest()
    assert built["seed_boundary"]["registry_binding_sha256"] == binding_sha256
    assert built["seed_boundary"]["trust_root_receipt_binding_sha256"] == (receipt_binding_sha256)
    assert built["seed_boundary"]["trust_root_receipt_external_pin_required"] is True
    assert built["seed_boundary"]["offline_signature_verification_required"] is True
    assert built["seed_boundary"]["offline_signature_verification_implemented_here"] is False
    assert built["seed_boundary"]["preacceptance_chronology_implemented_here"] is False
    assert receipt["provider_id"] == "drand_quicknet"
    assert receipt["provider_chain_hash"] == (
        "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
    )
    assert receipt["signature_scheme"] == "bls-unchained-g1-rfc9380"
    assert receipt["beacon_time_unix"] < receipt["observation_cutoff_unix"]
    assert receipt["seed_registry_file_sha256"] == registry["registry_file_sha256"]
    assert receipt["seed_registry_body_sha256"] == registry["registry_body_sha256"]
    assert (
        registry["registry_file_sha256"],
        registry["registry_body_sha256"],
    ) == _expected_registry_digests(inputs.trust_root_receipt)
    assert registry["trust_root_receipt_binding_sha256"] == receipt_binding_sha256
    assert built["seed_boundary"]["external_authentication_required"] is True
    assert built["seed_boundary"]["issued_before_observation_required"] is True
    cases = built["seed_boundary"]["cases"]
    assert [item["registry_case_ordinal"] for item in cases] == list(range(28))
    for ordinal, item in enumerate(cases):
        assert item["seed_registry_binding_sha256"] == binding_sha256
        assert item["seed_registry_file_sha256"] == registry["registry_file_sha256"]
        assert item["seed_registry_body_sha256"] == registry["registry_body_sha256"]
        assert item["provider_identity_sha256"] == registry["provider_identity_sha256"]
        assert item["provider_receipt_file_sha256"] == (registry["provider_receipt_file_sha256"])
        assert item["provider_receipt_body_sha256"] == (registry["provider_receipt_body_sha256"])
        assert item["derivation_schema_version"] == registry["derivation_schema_version"]
        assert item["derivation_domain"] == plan.QUALIFICATION_SEED_DERIVATION_DOMAIN
        expected = _expected_case_derivation(
            inputs.trust_root_receipt,
            item["candidate_id"],
            ordinal,
        )
        assert (
            item["derivation_payload_sha256"],
            item["environment_seed"],
            item["agent_seed"],
            item["environment_seed_derivation_sha256"],
            item["agent_seed_derivation_sha256"],
        ) == expected
    provenance_profile = next(
        item
        for item in built["probe_profiles"]
        if item["profile_id"] == "qualification_seed_provenance_v1"
    )
    assert provenance_profile == {
        "profile_id": "qualification_seed_provenance_v1",
        "required_observation_schema": (
            "alberta.forager_matched_v3.qualification_seed_observation.v1"
        ),
        "acceptance_fields": [
            "registry_full_file_and_body_digests_exact",
            "independent_trust_root_receipt_file_pin_exact",
            "independent_trust_root_receipt_binding_pin_exact",
            "provider_chain_public_key_and_signature_scheme_exact",
            "pulse_record_exact",
            "beacon_round_time_signature_and_randomness_exact",
            "offline_verifier_source_closure_membership_exact",
            "offline_signature_verification_exact",
            "deterministic_28_case_seed_pair_derivation_exact",
            "deterministic_registry_file_and_body_digests_exact",
            "derivation_schema_and_domain_exact",
            "case_derivation_payload_membership_exact",
            "beacon_time_precedes_observation_cutoff_exact",
            "external_receipt_preacceptance_chronology_exact",
        ],
        "reward_magnitude_is_acceptance_input": False,
        "score_is_acceptance_input": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registry_schema_version", "alberta.synthetic.registry.v1"),
        ("derivation_schema_version", "alberta.synthetic.derivation.v1"),
        ("derivation_domain", "alberta.forager.matched_v3.held_out.seed.v1"),
        ("provider_id", "self_asserted_provider"),
        ("provider_identity_sha256", _sha("self-asserted-provider")),
        ("provider_receipt_schema_version", "alberta.synthetic.receipt.v1"),
        ("external_authentication_required", False),
        ("external_authentication_required", 1),
        ("issued_before_observation_required", False),
        ("issued_before_observation_required", 1),
    ],
)
def test_seed_registry_schema_domain_and_authentication_are_exact(
    field: str, value: object
) -> None:
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(_synthetic_inputs().seed_registry, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt_schema_version", "alberta.synthetic.receipt.v1"),
        ("provider_id", "self_asserted_provider"),
        ("provider_chain_hash", _sha("self-asserted-chain")),
        ("signature_scheme", "self-asserted-signature"),
        ("provider_public_key_sha256", _sha("self-asserted-public-key")),
        ("pulse_record_schema_version", "alberta.synthetic.pulse_record.v1"),
        ("seed_derivation_algorithm", "self_asserted_derivation"),
        ("timestamp_source", "self_asserted_time"),
        ("offline_verifier_schema_version", "alberta.synthetic.verifier.v1"),
        ("offline_verifier_source_id", "external_foragax_agents"),
        ("offline_verifier_runtime_helper_id", "oci_runtime"),
        ("offline_verifier_implementation_status", "implemented"),
        ("offline_signature_verification_required", False),
        ("external_preacceptance_required", False),
    ],
)
def test_seed_trust_root_receipt_contract_is_exact(field: str, value: object) -> None:
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(_synthetic_inputs().trust_root_receipt, **{field: value})


def test_seed_trust_root_receipt_time_must_precede_observation_cutoff() -> None:
    receipt = _synthetic_inputs().trust_root_receipt
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(receipt, beacon_time_unix=receipt.observation_cutoff_unix)


def test_seed_trust_root_receipt_round_time_is_derived_from_quicknet_chain_info() -> None:
    receipt = _synthetic_inputs().trust_root_receipt
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="round time"):
        dataclasses.replace(receipt, beacon_time_unix=receipt.beacon_time_unix + 1)


def test_seed_trust_root_receipt_randomness_is_raw_signature_sha256() -> None:
    receipt = _synthetic_inputs().trust_root_receipt
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="raw signature"):
        dataclasses.replace(receipt, beacon_randomness_hex=_sha("unbound-randomness"))


@pytest.mark.parametrize(
    "field",
    [
        "seed_registry_binding_sha256",
        "seed_registry_file_sha256",
        "seed_registry_body_sha256",
        "provider_identity_sha256",
        "provider_receipt_file_sha256",
        "provider_receipt_body_sha256",
    ],
)
def test_case_registry_and_provider_cross_wiring_fails_closed(field: str) -> None:
    inputs = _synthetic_inputs()
    cases = list(inputs.cases)
    replace = cast(Any, dataclasses.replace)
    cases[0] = replace(cases[0], **{field: _sha(f"cross-wire:{field}")})
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(cases=tuple(cases)))


def test_changed_registry_cannot_be_paired_with_old_cases() -> None:
    inputs = _synthetic_inputs()
    changed_registry = dataclasses.replace(
        inputs.seed_registry,
        registry_file_sha256=_sha("different-authenticated-registry-file"),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(seed_registry=changed_registry))


def test_coherent_post_observation_seed_relabel_fails_without_independent_receipt_pin() -> None:
    inputs = _synthetic_inputs()
    registry_file_sha256 = _sha("post-observation-registry-file")
    registry_body_sha256 = _sha("post-observation-registry-body")
    changed_receipt = dataclasses.replace(
        inputs.trust_root_receipt,
        receipt_file_sha256=_sha("post-observation-trust-root-receipt-file"),
        receipt_body_sha256=_sha("post-observation-trust-root-receipt-body"),
        beacon_round=inputs.trust_root_receipt.beacon_round + 1,
        beacon_time_unix=(inputs.trust_root_receipt.beacon_time_unix + _QUICKNET_PERIOD_SECONDS),
        beacon_signature_sha256=_sha("post-observation-beacon-signature"),
        beacon_randomness_hex=_sha("post-observation-beacon-signature"),
        pulse_record_file_sha256=_sha("post-observation-pulse-record-file"),
        pulse_record_body_sha256=_sha("post-observation-pulse-record-body"),
        seed_registry_file_sha256=registry_file_sha256,
        seed_registry_body_sha256=registry_body_sha256,
    )
    changed_registry = dataclasses.replace(
        inputs.seed_registry,
        registry_file_sha256=registry_file_sha256,
        registry_body_sha256=registry_body_sha256,
        provider_receipt_file_sha256=changed_receipt.receipt_file_sha256,
        provider_receipt_body_sha256=changed_receipt.receipt_body_sha256,
        trust_root_receipt_binding_sha256=hashlib.sha256(
            _canonical(changed_receipt.to_dict())
        ).hexdigest(),
    )
    changed_registry_binding = hashlib.sha256(_canonical(changed_registry.to_dict())).hexdigest()
    changed_cases = tuple(
        dataclasses.replace(
            case,
            seed_registry_binding_sha256=changed_registry_binding,
            seed_registry_file_sha256=changed_registry.registry_file_sha256,
            seed_registry_body_sha256=changed_registry.registry_body_sha256,
            provider_receipt_file_sha256=changed_registry.provider_receipt_file_sha256,
            provider_receipt_body_sha256=changed_registry.provider_receipt_body_sha256,
        )
        for case in inputs.cases
    )
    changed_inputs = inputs._replace(
        trust_root_receipt=changed_receipt,
        seed_registry=changed_registry,
        cases=changed_cases,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="independent pin"):
        _build(changed_inputs)


def test_new_caller_pins_do_not_claim_preacceptance_chronology() -> None:
    inputs = _synthetic_inputs()
    changed_randomness = _sha("different-internally-consistent-pulse-signature")
    provisional_receipt = dataclasses.replace(
        inputs.trust_root_receipt,
        receipt_file_sha256=_sha("different-trust-root-receipt-file"),
        receipt_body_sha256=_sha("different-trust-root-receipt-body"),
        beacon_round=inputs.trust_root_receipt.beacon_round + 1,
        beacon_time_unix=(inputs.trust_root_receipt.beacon_time_unix + _QUICKNET_PERIOD_SECONDS),
        beacon_signature_sha256=changed_randomness,
        beacon_randomness_hex=changed_randomness,
        pulse_record_file_sha256=_sha("different-pulse-record-file"),
        pulse_record_body_sha256=_sha("different-pulse-record-body"),
    )
    registry_file_sha256, registry_body_sha256 = _expected_registry_digests(provisional_receipt)
    receipt = dataclasses.replace(
        provisional_receipt,
        seed_registry_file_sha256=registry_file_sha256,
        seed_registry_body_sha256=registry_body_sha256,
    )
    receipt_binding_sha256 = hashlib.sha256(_canonical(receipt.to_dict())).hexdigest()
    registry = dataclasses.replace(
        inputs.seed_registry,
        registry_file_sha256=registry_file_sha256,
        registry_body_sha256=registry_body_sha256,
        provider_receipt_file_sha256=receipt.receipt_file_sha256,
        provider_receipt_body_sha256=receipt.receipt_body_sha256,
        trust_root_receipt_binding_sha256=receipt_binding_sha256,
    )
    registry_binding_sha256 = hashlib.sha256(_canonical(registry.to_dict())).hexdigest()
    cases: list[plan.QualificationCase] = []
    for ordinal, old_case in enumerate(inputs.cases):
        derived = _expected_case_derivation(receipt, old_case.candidate_id, ordinal)
        cases.append(
            dataclasses.replace(
                old_case,
                seed_registry_binding_sha256=registry_binding_sha256,
                seed_registry_file_sha256=registry.registry_file_sha256,
                seed_registry_body_sha256=registry.registry_body_sha256,
                provider_receipt_file_sha256=registry.provider_receipt_file_sha256,
                provider_receipt_body_sha256=registry.provider_receipt_body_sha256,
                derivation_payload_sha256=derived[0],
                environment_seed=derived[1],
                agent_seed=derived[2],
                environment_seed_derivation_sha256=derived[3],
                agent_seed_derivation_sha256=derived[4],
            )
        )
    changed = inputs._replace(
        trust_root_receipt=receipt,
        expected_trust_root_receipt_file_sha256=receipt.receipt_file_sha256,
        expected_trust_root_receipt_binding_sha256=receipt_binding_sha256,
        seed_registry=registry,
        cases=tuple(cases),
    )

    built = _build(changed)

    assert built["claims"]["qualification_executed"] is False
    assert built["claims"]["execution_authorized"] is False
    assert (
        built["authentication_policy"]["qualification_seed_preacceptance_chronology_verified_here"]
        is False
    )
    assert built["seed_boundary"]["preacceptance_chronology_implemented_here"] is False


def test_receipt_metadata_relabel_with_same_file_claim_fails_binding_pin() -> None:
    inputs = _synthetic_inputs()
    changed_receipt = dataclasses.replace(
        inputs.trust_root_receipt,
        beacon_round=inputs.trust_root_receipt.beacon_round + 1,
        beacon_time_unix=(inputs.trust_root_receipt.beacon_time_unix + _QUICKNET_PERIOD_SECONDS),
        beacon_randomness_hex=_sha("same-file-claim-different-signature"),
        beacon_signature_sha256=_sha("same-file-claim-different-signature"),
    )
    changed_registry = dataclasses.replace(
        inputs.seed_registry,
        trust_root_receipt_binding_sha256=hashlib.sha256(
            _canonical(changed_receipt.to_dict())
        ).hexdigest(),
    )
    changed_registry_binding = hashlib.sha256(_canonical(changed_registry.to_dict())).hexdigest()
    changed_cases: list[plan.QualificationCase] = []
    for ordinal, case in enumerate(inputs.cases):
        derived = _expected_case_derivation(changed_receipt, case.candidate_id, ordinal)
        changed_cases.append(
            dataclasses.replace(
                case,
                seed_registry_binding_sha256=changed_registry_binding,
                derivation_payload_sha256=derived[0],
                environment_seed=derived[1],
                agent_seed=derived[2],
                environment_seed_derivation_sha256=derived[3],
                agent_seed_derivation_sha256=derived[4],
            )
        )
    changed_inputs = inputs._replace(
        trust_root_receipt=changed_receipt,
        seed_registry=changed_registry,
        cases=tuple(changed_cases),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="binding differs"):
        _build(changed_inputs)


def test_preaccepted_receipt_cannot_bind_a_nonderived_seed_registry() -> None:
    inputs = _synthetic_inputs()
    wrong_file_sha256 = _sha("nonderived-seed-registry-file")
    wrong_body_sha256 = _sha("nonderived-seed-registry-body")
    receipt = dataclasses.replace(
        inputs.trust_root_receipt,
        seed_registry_file_sha256=wrong_file_sha256,
        seed_registry_body_sha256=wrong_body_sha256,
    )
    receipt_binding_sha256 = hashlib.sha256(_canonical(receipt.to_dict())).hexdigest()
    registry = dataclasses.replace(
        inputs.seed_registry,
        registry_file_sha256=wrong_file_sha256,
        registry_body_sha256=wrong_body_sha256,
        trust_root_receipt_binding_sha256=receipt_binding_sha256,
    )
    registry_binding_sha256 = hashlib.sha256(_canonical(registry.to_dict())).hexdigest()
    cases = tuple(
        dataclasses.replace(
            case,
            seed_registry_binding_sha256=registry_binding_sha256,
            seed_registry_file_sha256=wrong_file_sha256,
            seed_registry_body_sha256=wrong_body_sha256,
        )
        for case in inputs.cases
    )
    changed = inputs._replace(
        trust_root_receipt=receipt,
        expected_trust_root_receipt_binding_sha256=receipt_binding_sha256,
        seed_registry=registry,
        cases=cases,
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="pulse-derived"):
        _build(changed)


def test_serialized_case_registry_cross_wiring_fails_with_coherent_rehash() -> None:
    built = _build()
    built["seed_boundary"]["cases"][0]["provider_receipt_body_sha256"] = _sha(
        "cross-wired-provider-receipt"
    )
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


def test_module_exposes_no_seed_registry_issuer_or_qualification_executor_api() -> None:
    assert (
        plan.matched_v3_qualification_plan_descriptor()["qualification_seed_registry_contract"][
            "issuer_api_exposed"
        ]
        is False
    )
    for name in (
        "issue_qualification_seed_registry",
        "build_qualification_seed_registry",
        "qualify_candidate",
        "execute_qualification_plan",
    ):
        assert not hasattr(plan, name)


def test_all_28_publication_bindings_are_embedded_and_adapters_are_exact() -> None:
    built = _build()
    candidates = built["candidate_requirements"]
    bindings = [item["result_publication_binding"] for item in candidates]
    assert [item["candidate_id"] for item in bindings] == list(
        plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    )
    adapters = [item for item in bindings if item["candidate_id"].startswith("adapted_")]
    assert len(adapters) == 2
    assert {item["descriptor_sha256"] for item in adapters} == {_PUBLICATION_SHA256}
    assert {item["implementation_source_sha256"] for item in adapters} == {
        _PUBLICATION_SOURCE_SHA256
    }
    local_tree_sha256 = next(
        item["source_tree_sha256"]
        for item in built["source_requirements"]
        if item["source_id"] == "local_alberta"
    )
    expected_keys = {
        "candidate_id",
        "publisher_kind",
        "descriptor_schema_version",
        "descriptor_sha256",
        "publication_schema_version",
        "source_id",
        "source_tree_sha256",
        "implementation_path",
        "implementation_source_sha256",
        "reload_validator_schema_version",
        "reload_validator_descriptor_sha256",
        "reload_validator_implementation_path",
        "reload_validator_source_sha256",
    }
    for item in bindings:
        assert set(item) == expected_keys
        assert item["source_id"] == "local_alberta"
        assert item["source_tree_sha256"] == local_tree_sha256
        assert item["implementation_path"]
        assert len(item["implementation_source_sha256"]) == 64
        assert item["reload_validator_implementation_path"]
        assert len(item["reload_validator_descriptor_sha256"]) == 64
        assert len(item["reload_validator_source_sha256"]) == 64
    adapter_path = "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
    for item in adapters:
        assert item["implementation_path"] == adapter_path
        assert item["reload_validator_schema_version"] == (
            "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
        )
        assert item["reload_validator_descriptor_sha256"] == _PUBLICATION_SHA256
        assert item["reload_validator_implementation_path"] == adapter_path
        assert item["reload_validator_source_sha256"] == _PUBLICATION_SOURCE_SHA256


def test_candidate_execution_source_and_local_publisher_source_cannot_be_conflated() -> None:
    inputs = _synthetic_inputs()
    built = _build(inputs)
    external = next(
        item
        for item in built["candidate_requirements"]
        if item["candidate_id"] == "external_dqn_plain"
    )
    assert external["source_id"] == "external_foragax_agents"
    assert external["result_publication_binding"]["source_id"] == "local_alberta"
    publication_index = plan.MATCHED_V3_QUALIFICATION_CANDIDATE_IDS.index("external_dqn_plain")
    publications = list(inputs.publications)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(
            publications[publication_index],
            source_id="external_foragax_agents",
            source_tree_sha256=inputs.sources[0].source_tree_sha256,
        )


@pytest.mark.parametrize(
    "bad_path",
    ["", "/absolute.py", "../escape.py", "publishers/../escape.py", "./publisher.py", "a//b.py"],
)
def test_publication_and_reload_paths_must_be_normalized_relative_paths(
    bad_path: str,
) -> None:
    publication = _synthetic_inputs().publications[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(publication, implementation_path=bad_path)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(publication, reload_validator_implementation_path=bad_path)


def test_publication_source_tree_mismatch_fails_at_build_boundary() -> None:
    inputs = _synthetic_inputs()
    publications = list(inputs.publications)
    publications[0] = dataclasses.replace(
        publications[0], source_tree_sha256=_sha("unbound-local-source-tree")
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(publications=tuple(publications)))


def test_serialized_publication_source_cross_wiring_fails_with_coherent_rehash() -> None:
    built = _build()
    local = built["candidate_requirements"][0]["result_publication_binding"]
    external_tree = next(
        item["source_tree_sha256"]
        for item in built["source_requirements"]
        if item["source_id"] == "external_foragax_agents"
    )
    local["source_tree_sha256"] = external_tree
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


def test_adapter_publication_substitution_fails_at_typed_boundary() -> None:
    adapter = next(
        item
        for item in _synthetic_inputs().publications
        if item.candidate_id == "adapted_full_rainbow"
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        dataclasses.replace(
            adapter,
            publisher_kind="synthetic_content_publisher_v1",
            descriptor_schema_version="alberta.synthetic.descriptor.v1",
            descriptor_sha256=_sha("wrong-adapter-descriptor"),
            publication_schema_version="alberta.synthetic.publication.v1",
            implementation_source_sha256=_sha("wrong-adapter-source"),
        )


@pytest.mark.parametrize("kind", ["duplicate", "reordered"])
def test_publication_candidate_coverage_fails_closed(kind: str) -> None:
    inputs = _synthetic_inputs()
    publications = list(inputs.publications)
    if kind == "duplicate":
        publications[1] = publications[0]
    else:
        publications[0], publications[1] = publications[1], publications[0]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _build(inputs._replace(publications=tuple(publications)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reload_validator_schema_version", "alberta.synthetic.validator.v1"),
        ("reload_validator_descriptor_sha256", _sha("wrong-reload-descriptor")),
        ("reload_validator_implementation_path", "validators/wrong.py"),
        ("reload_validator_source_sha256", _sha("wrong-reload-source")),
    ],
)
def test_adapter_reload_validator_substitution_fails_closed(field: str, value: object) -> None:
    adapter = next(
        item for item in _synthetic_inputs().publications if item.candidate_id == "adapted_ppo_gru"
    )
    replace = cast(Any, dataclasses.replace)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        replace(adapter, **{field: value})


def test_publication_roundtrip_probe_is_required_score_blind_and_fail_closed() -> None:
    built = _build()
    profile = next(
        item
        for item in built["probe_profiles"]
        if item["profile_id"] == "result_publication_roundtrip_v1"
    )
    assert profile == {
        "profile_id": "result_publication_roundtrip_v1",
        "required_observation_schema": (
            "alberta.forager_matched_v3.result_publication_observation.v1"
        ),
        "acceptance_fields": [
            "publisher_descriptor_membership_exact",
            "publisher_source_closure_membership_exact",
            "reload_validator_membership_exact",
            "atomic_publication_exact",
            "strict_reload_exact",
            "full_file_digest_equivalence_exact",
            "score_and_reward_magnitude_not_decoded",
        ],
        "reward_magnitude_is_acceptance_input": False,
        "score_is_acceptance_input": False,
    }
    for candidate in built["candidate_requirements"]:
        assert "result_publication_roundtrip_v1" in candidate["probe_profile_ids"]
        acceptance = candidate["acceptance"]
        assert acceptance["result_publication_roundtrip_exact"] is True
        assert acceptance["publication_full_file_digest_equivalence_exact"] is True
        assert acceptance["reward_magnitude_is_acceptance_input"] is False
        assert acceptance["score_is_acceptance_input"] is False


def test_synthetic_publishers_only_build_an_unexecuted_plan_pending_roundtrip_probe() -> None:
    built = _build()
    synthetic = [
        item["result_publication_binding"]
        for item in built["candidate_requirements"]
        if not item["candidate_id"].startswith("adapted_")
    ]
    assert synthetic
    assert {item["publisher_kind"] for item in synthetic} == {"synthetic_content_publisher_v1"}
    assert all(value is False for value in built["claims"].values())
    assert built["claims"]["candidate_qualified"] is False
    assert built["claims"]["qualification_executed"] is False
    assert not hasattr(plan, "qualify_matched_v3_candidate")


def test_authority_claims_are_recursively_false_and_authentication_is_separate() -> None:
    built = _build()
    descriptor = plan.matched_v3_qualification_plan_descriptor()
    for artifact in (built, descriptor):
        assert artifact["claims"]
        assert all(value is False for value in artifact["claims"].values())
        policy = artifact["authentication_policy"]
        assert policy["development_external_authentication_required"] is False
        assert policy["qualification_case_external_authentication_required"] is True
        assert policy["qualification_seed_registry_issued_before_observation_required"] is True
        assert policy["qualification_seed_trust_root_receipt_external_pin_required"] is True
        assert policy["qualification_seed_offline_signature_verification_required"] is True
        assert policy["qualification_seed_offline_signature_verification_implemented_here"] is False
        assert policy["qualification_seed_preacceptance_chronology_verified_here"] is False
        assert policy["confirmatory_external_authentication_required"] is True
        assert policy["execution_authority_separate_from_content_validation"] is True
        assert policy["serialized_artifact_grants_execution_capability"] is False
        assert any(
            "Quicknet pulse does not prove receipt preacceptance timing" in item
            for item in artifact["limitations"]
        )
        assert any(
            "Quicknet signs only its unchained round message" in item
            for item in artifact["limitations"]
        )
    assert built["resource_contract"]["compute_efficiency_claimed"] is False
    assert built["resource_contract"]["resource_matched_claimed"] is False


def test_acceptance_and_failure_policy_are_preobservation_and_score_blind() -> None:
    built = _build()
    assert built["failure_policy"]["fixed_before_observation_required"] is True
    assert built["failure_policy"]["fixed_before_observation_verified_here"] is False
    assert built["failure_policy"]["fail_closed"] is True

    def visit(value: Any) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if "reward_magnitude_is_" in key or "score_is_" in key or "ranking_is_" in key:
                    assert child is False
                visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)

    visit(built)


def test_full_file_digest_is_mandatory_and_cannot_be_replaced_by_body_digest() -> None:
    built = _build()
    raw = plan.canonical_matched_v3_qualification_plan_bytes(
        built,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        plan.parse_matched_v3_qualification_plan_artifact(
            raw,
            expected_file_sha256=built["plan_body_sha256"],
            expected_qualification_seed_trust_root_receipt_file_sha256=(
                _TRUST_ROOT_RECEIPT_FILE_SHA256
            ),
            expected_qualification_seed_trust_root_receipt_binding_sha256=(
                _trusted_receipt_binding_sha256()
            ),
        )
    with pytest.raises(TypeError):
        cast(Any, plan.parse_matched_v3_qualification_plan_artifact)(raw)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        cast(Any, plan.parse_matched_v3_qualification_plan_artifact)(
            bytearray(raw),
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_qualification_seed_trust_root_receipt_file_sha256=(
                _TRUST_ROOT_RECEIPT_FILE_SHA256
            ),
            expected_qualification_seed_trust_root_receipt_binding_sha256=(
                _trusted_receipt_binding_sha256()
            ),
        )


def test_semantic_mutation_fails_with_coherent_body_and_external_file_digest() -> None:
    built = _build()
    built["bindings"]["dependencies"]["reward_scorer"]["source_sha256"] = "0" * 64
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


def test_parser_rejects_duplicate_keys() -> None:
    raw = plan.canonical_matched_v3_qualification_plan_bytes(
        _build(),
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )
    duplicate = b'{"schema_version":"duplicate",' + raw[1:]
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="duplicate"):
        _parse_with_actual_digest(duplicate)


def test_parser_rejects_noncanonical_and_nonascii_bytes() -> None:
    raw = plan.canonical_matched_v3_qualification_plan_bytes(
        _build(),
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            _TRUST_ROOT_RECEIPT_FILE_SHA256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            _trusted_receipt_binding_sha256()
        ),
    )
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(b" " + raw)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw[:-2] + b"\xff\n")


def test_parser_rejects_nonfinite_and_exact_bool_int_substitution() -> None:
    built = _build()
    built["runtime_requirement"]["jax_enable_x64"] = 0
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)
    built = _build()
    built["resource_contract"]["requirements"][0]["max_optimizer_updates"] = float("nan")
    body = copy.deepcopy(built)
    body.pop("plan_body_sha256")
    built["plan_body_sha256"] = "0" * 64
    raw = _canonical_allow_nan(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="non-finite"):
        _parse_with_actual_digest(raw)


@pytest.mark.parametrize("location", ["authentication", "failure", "probe", "acceptance"])
def test_parser_rejects_bool_int_aliases_in_fixed_policy_templates(location: str) -> None:
    built = _build()
    if location == "authentication":
        built["authentication_policy"]["qualification_case_external_authentication_required"] = 1
    elif location == "failure":
        built["failure_policy"]["fixed_before_observation_required"] = 1
    elif location == "probe":
        built["probe_profiles"][0]["score_is_acceptance_input"] = 0
    else:
        built["candidate_requirements"][0]["acceptance"]["runtime_membership_exact"] = 1
    raw = _rehash_body(built)
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError):
        _parse_with_actual_digest(raw)


def test_parser_rejects_depth_node_and_byte_limit_attacks() -> None:
    nested: Any = 0
    for _ in range(70):
        nested = [nested]
    depth_raw = _canonical({"x": nested})
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="depth"):
        _parse_with_actual_digest(depth_raw)
    node_raw = _canonical({"x": [0] * 100_001})
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="node"):
        _parse_with_actual_digest(node_raw)
    byte_raw = b'{"x":"' + b"a" * (2 * 1024 * 1024) + b'"}\n'
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="byte"):
        _parse_with_actual_digest(byte_raw)


def test_canonicalizer_rejects_container_aliases() -> None:
    shared: list[object] = []
    aliased = {"first": shared, "second": shared}
    with pytest.raises(plan.ForagerMatchedV3QualificationPlanError, match="alias"):
        cast(Any, plan._canonical_json)(aliased)


def test_no_default_or_production_plan_exists() -> None:
    signature = inspect.signature(plan.build_matched_v3_qualification_plan)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )
    assert not hasattr(plan, "DEFAULT_QUALIFICATION_PLAN")
    assert not hasattr(plan, "PRODUCTION_QUALIFICATION_PLAN")


def test_construction_invokes_no_runtime_filesystem_probe_or_publication_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs()

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("content construction invoked a forbidden side-effect API")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    built = _build(inputs)
    assert built["claims"]["qualification_executed"] is False
    assert built["claims"]["benchmark_executed"] is False
    assert built["claims"]["result_observed"] is False
