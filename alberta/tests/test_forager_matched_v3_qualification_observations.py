"""Tests for score-blind matched-v3 qualification observation structures."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Iterable
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_observations as observations,
)
from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v2 as plan_v2

_PLAN_SHA256 = hashlib.sha256(b"caller-supplied-qualification-plan").hexdigest()


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


def _rebody(value: dict[str, Any]) -> bytes:
    body = copy.deepcopy(value)
    body.pop("envelope_body_sha256")
    value["envelope_body_sha256"] = hashlib.sha256(
        _canonical(body, newline=False)
    ).hexdigest()
    return _canonical(value)


def _helpers() -> tuple[observations.RuntimeHelperIdentity, ...]:
    return tuple(
        observations.RuntimeHelperIdentity(
            helper_id=helper_id,
            descriptor_schema_version=f"alberta.test.{helper_id}.descriptor.v1",
            descriptor_sha256=_sha(f"{helper_id}-descriptor"),
            implementation_path=f"helpers/{helper_id}.py",
            implementation_source_sha256=_sha(f"{helper_id}-source"),
            entrypoint=f"helpers.{helper_id}:main",
            entrypoint_sha256=_sha(f"{helper_id}-entrypoint"),
            executable_sha256=_sha(f"{helper_id}-executable"),
            version_output_sha256=_sha(f"{helper_id}-version-output"),
        )
        for helper_id in ("drand_verify", "oci_runtime", "resource_observer")
    )


def _resource_pairs(offset: int) -> tuple[tuple[str, int], ...]:
    return tuple(
        (field, index + offset)
        for index, field in enumerate(plan_v2.RESOURCE_CEILING_FIELDS, start=1)
    )


def _payloads() -> tuple[observations.QualificationObservationPayload, ...]:
    return (
        observations.ExternalSourceObservationPayload(
            source_id="external_foragax_agents",
            producer_kind="durable_external_source_publication_v1_materialization_v2",
            publication_receipt_schema_version=(
                plan_v2.EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
            ),
            publication_receipt_file_sha256=_sha("external-receipt-file"),
            publication_receipt_body_sha256=_sha("external-receipt-body"),
            publication_contract_descriptor_sha256=_sha("external-contract"),
            materialization_manifest_schema_version=(
                plan_v2.EXTERNAL_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
            ),
            materialization_manifest_file_sha256=_sha("external-manifest-file"),
            materialization_manifest_body_sha256=_sha("external-manifest-body"),
            materialization_payload_sha256=_sha("external-payload"),
            source_tree_sha256=_sha("external-tree"),
            source_inventory_sha256=_sha("external-inventory"),
            staging_manifest_schema_version=plan_v2.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
            staging_manifest_file_sha256=_sha("staging-file"),
            staging_manifest_body_sha256=_sha("staging-body"),
            archive_file_sha256=_sha("external-archive"),
            archive_inventory_sha256=_sha("external-archive-inventory"),
            tracked_entry_count=11,
            materialized_file_count=10,
            excluded_gitlink_count=1,
            archive_member_count=12,
            materialized_total_size_bytes=1000,
            archive_size_bytes=20_480,
            producer_receipt_replay_exact=True,
            manifest_file_body_binding_exact=False,
            source_tree_inventory_exact=True,
            archive_inventory_exact=False,
            counts_exact=True,
        ),
        observations.LocalSourceObservationPayload(
            source_id="local_alberta",
            producer_kind="local_source_snapshot_and_retained_bundle_v1",
            snapshot_descriptor_sha256=_sha("snapshot-descriptor"),
            snapshot_manifest_schema_version=(
                plan_v2.LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
            ),
            snapshot_manifest_file_sha256=_sha("snapshot-file"),
            snapshot_manifest_body_sha256=_sha("snapshot-body"),
            snapshot_tree_schema_version=plan_v2.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
            snapshot_tree_sha256=_sha("snapshot-tree"),
            bundle_descriptor_sha256=_sha("bundle-descriptor"),
            bundle_receipt_schema_version=plan_v2.LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION,
            bundle_receipt_file_sha256=_sha("bundle-receipt-file"),
            bundle_receipt_body_sha256=_sha("bundle-receipt-body"),
            archive_file_sha256=_sha("local-archive"),
            member_inventory_sha256=_sha("local-inventory"),
            directory_count=3,
            file_count=5,
            total_size_bytes=500,
            archive_member_count=5,
            archive_size_bytes=10_240,
            producer_receipt_replay_exact=False,
            manifest_file_body_binding_exact=True,
            source_tree_inventory_exact=False,
            archive_inventory_exact=True,
            counts_exact=False,
        ),
        observations.RuntimeObservationPayload(
            executor_kind="networkless_oci_cpu",
            executor_descriptor_sha256=_sha("executor-descriptor"),
            executor_source_sha256=_sha("executor-source"),
            executor_receipt_schema_version="alberta.test.executor_receipt.v1",
            executor_receipt_file_sha256=_sha("executor-receipt"),
            executor_receipt_body_sha256=_sha("executor-receipt-body"),
            runtime_executable_sha256=_sha("runtime-executable"),
            runtime_version_output_sha256=_sha("runtime-version-output"),
            image_id=f"sha256:{_sha('caller-image')}",
            image_config_sha256=_sha("caller-image"),
            runtime_profile_sha256=_sha("runtime-profile"),
            runtime_identity_sha256=_sha("runtime-identity"),
            runtime_inventory_sha256=_sha("runtime-inventory"),
            source_import_inventory_sha256=_sha("source-import-inventory"),
            python_implementation="CPython",
            platform="linux/amd64",
            python_version="3.12.3",
            jax_version="0.11.0",
            jaxlib_version="0.11.0",
            foragax_version="0.55.0",
            foragax_install_tree_sha256=(
                "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
            ),
            jax_backend="cpu",
            default_prng_impl="threefry2x32",
            jax_enable_x64=False,
            threefry_partitionable=True,
            sandbox_policy_sha256=_sha("sandbox-policy"),
            sandbox_observation_sha256=_sha("sandbox-observation"),
            helpers=_helpers(),
            executor_identity_exact=True,
            image_identity_exact=False,
            runtime_inventory_exact=True,
            sandbox_policy_exact=False,
            helper_order_and_identity_exact=True,
            fresh_process_observation=False,
            network_disabled_observed=True,
            cpu_only_observed=False,
            unprivileged_user_observed=True,
            read_only_source_observed=False,
            bytecode_cache_disabled_observed=True,
        ),
        observations.QualificationSeedObservationPayload(
            qualification_case_id="public-case-0001",
            qualification_case_manifest_sha256=_sha("case-manifest"),
            trust_root_receipt_sha256=_sha("trust-root"),
            signature_bundle_sha256=_sha("signature-bundle"),
            derivation_descriptor_sha256=_sha("derivation-descriptor"),
            seed_commitment_sha256=_sha("seed-commitment"),
            draw_index=1,
            registry_full_file_and_body_digests_exact=True,
            independent_trust_root_receipt_file_pin_exact=False,
            independent_trust_root_receipt_binding_pin_exact=True,
            provider_chain_public_key_and_signature_scheme_exact=False,
            pulse_record_exact=True,
            beacon_round_time_signature_and_randomness_exact=False,
            offline_verifier_source_closure_membership_exact=True,
            offline_signature_verification_exact=False,
            deterministic_28_case_seed_pair_derivation_exact=True,
            deterministic_registry_file_and_body_digests_exact=False,
            derivation_schema_and_domain_exact=True,
            case_derivation_payload_membership_exact=False,
            beacon_time_precedes_observation_cutoff_exact=True,
            external_receipt_preacceptance_chronology_exact=False,
        ),
        observations.CandidateObservationPayload(
            candidate_id=plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[0],
            qualification_case_manifest_sha256=_sha("candidate-case"),
            source_tree_sha256=_sha("candidate-tree"),
            configuration_record_sha256=_sha("candidate-configuration"),
            entrypoint_source_sha256=_sha("candidate-entrypoint"),
            agent_seed_commitment_sha256=_sha("candidate-agent-seed"),
            environment_seed_commitment_sha256=_sha("candidate-environment-seed"),
            candidate_rng_trace_sha256=_sha("candidate-rng-trace"),
            source_membership_exact=True,
            configuration_membership_exact=False,
            entrypoint_import_exact=True,
            agent_seed_transport_exact=False,
            environment_agent_derivations_distinct=True,
            candidate_rng_membership_exact=False,
        ),
        observations.ResourceObservationPayload(
            candidate_id=plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[1],
            qualification_case_manifest_sha256=_sha("resource-case"),
            resource_requirement_body_sha256=_sha("resource-requirement"),
            resource_observation_sha256=_sha("resource-observation"),
            declared_ceilings=_resource_pairs(100),
            observed_values=_resource_pairs(0),
            horizon_accounting_exact=True,
            reward_membership_structural_only=False,
            all_resource_observations_within_predeclared_integer_ceilings=True,
        ),
        observations.ResultPublicationObservationPayload(
            candidate_id=plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[2],
            qualification_case_manifest_sha256=_sha("publication-case"),
            publisher_descriptor_sha256=_sha("publisher-descriptor"),
            publisher_source_sha256=_sha("publisher-source"),
            publisher_source_tree_sha256=_sha("publisher-tree"),
            publication_manifest_sha256=_sha("publication-manifest"),
            publication_receipt_sha256=_sha("publication-receipt"),
            published_bundle_sha256=_sha("publication-bundle"),
            reload_observation_sha256=_sha("reload-observation"),
            publication_file_count=5,
            publication_total_size_bytes=100,
            publisher_descriptor_membership_exact=True,
            publisher_source_closure_membership_exact=False,
            reload_validator_membership_exact=True,
            atomic_publication_exact=True,
            strict_reload_exact=False,
            full_file_digest_equivalence_exact=True,
            score_and_reward_magnitude_not_decoded=False,
        ),
        observations.FreshReplayObservationPayload(
            candidate_id=plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS[3],
            qualification_case_manifest_sha256=_sha("replay-case"),
            runtime_identity_sha256=_sha("replay-runtime"),
            environment_seed_commitment_sha256=_sha("replay-environment-seed"),
            key_schedule_descriptor_sha256=_sha("replay-key-schedule"),
            first_structure_trace_sha256=_sha("first-structure"),
            replay_structure_trace_sha256=_sha("replay-structure"),
            interaction_count=499_712,
            environment_seed_transport_exact=True,
            reset_step_key_schedule_exact=False,
            structural_replay_exact=True,
        ),
    )


def _observation(
    payload: observations.QualificationObservationPayload,
) -> observations.MatchedV3QualificationObservation:
    return observations.MatchedV3QualificationObservation(
        qualification_plan_sha256=_PLAN_SHA256,
        payload=payload,
    )


def _encoded(
    payload: observations.QualificationObservationPayload,
) -> tuple[dict[str, Any], bytes, str]:
    observation = _observation(payload)
    raw = observations.canonical_matched_v3_qualification_observation_bytes(observation)
    return observation.to_dict(), raw, hashlib.sha256(raw).hexdigest()


def _parse(raw: bytes, digest: str) -> observations.MatchedV3QualificationObservation:
    return observations.parse_matched_v3_qualification_observation(
        raw,
        expected_file_sha256=digest,
        expected_qualification_plan_sha256=_PLAN_SHA256,
    )


def test_registry_descriptor_is_frozen_complete_and_nonauthorizing() -> None:
    raw = observations.canonical_matched_v3_qualification_observation_registry_descriptor_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = observations.parse_matched_v3_qualification_observation_registry_descriptor(raw)

    assert digest == observations.QUALIFICATION_OBSERVATION_REGISTRY_DESCRIPTOR_SHA256
    assert digest == observations.matched_v3_qualification_observation_registry_descriptor_sha256()
    assert value == observations.matched_v3_qualification_observation_registry_descriptor()
    assert value["status"] == "implemented_structural_validators_no_observation_issuer"
    assert [item["kind"] for item in value["payload_contracts"]] == list(
        observations.QUALIFICATION_OBSERVATION_KINDS
    )
    assert value["qualification_plan_binding"]["candidate_order"] == list(
        plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    assert value["qualification_plan_binding"]["resource_fields"] == list(
        plan_v2.RESOURCE_CEILING_FIELDS
    )
    assert all(claim is False for claim in value["claims"].values())
    assert all(capability is False for capability in value["capabilities"].values())
    assert "sha256:a1f491fc" not in raw.decode("ascii")


@pytest.mark.parametrize("payload", _payloads())
def test_every_payload_roundtrips_immutably_without_evaluating_booleans(
    payload: observations.QualificationObservationPayload,
) -> None:
    observation = _observation(payload)
    raw = observations.canonical_matched_v3_qualification_observation_bytes(observation)
    digest = hashlib.sha256(raw).hexdigest()

    parsed = _parse(raw, digest)
    replayed = observations.replay_matched_v3_qualification_observation(
        raw,
        expected_file_sha256=digest,
        expected_qualification_plan_sha256=_PLAN_SHA256,
    )

    assert parsed == observation
    assert replayed == observation
    assert all(claim is False for claim in parsed.to_dict()["claims"].values())
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.qualification_plan_sha256 = _sha("mutation")  # type: ignore[misc]


def test_seven_kinds_use_the_exact_v1_schema_bindings() -> None:
    observed = {
        _observation(payload).observation_kind: _observation(payload).observation_schema_version
        for payload in _payloads()
    }
    assert tuple(observed) == observations.QUALIFICATION_OBSERVATION_KINDS
    assert tuple(observed.values()) == observations.QUALIFICATION_OBSERVATION_SCHEMA_VERSIONS


def test_runtime_helpers_are_exactly_ordered_and_have_no_defaults() -> None:
    helpers = _helpers()
    runtime = _payloads()[2]
    assert isinstance(runtime, observations.RuntimeObservationPayload)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(runtime, helpers=tuple(reversed(helpers)))
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(runtime, helpers=helpers[:2])


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("executor_kind", "host_process"),
        ("image_config_sha256", _sha("wrong-image-config")),
        ("python_implementation", "PyPy"),
        ("python_version", "3.12.4"),
        ("jax_version", "0.10.0"),
        ("jaxlib_version", "0.10.0"),
        ("foragax_version", "0.54.0"),
        ("foragax_install_tree_sha256", _sha("wrong-foragax-tree")),
        ("platform", "linux/arm64"),
        ("jax_backend", "gpu"),
        ("default_prng_impl", "rbg"),
        ("jax_enable_x64", True),
        ("threefry_partitionable", False),
    ),
)
def test_runtime_observation_rejects_exact_production_identity_drift(
    field_name: str,
    value: object,
) -> None:
    runtime = _payloads()[2]
    assert isinstance(runtime, observations.RuntimeObservationPayload)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(runtime, **cast(Any, {field_name: value}))


def test_registry_exposes_complete_exact_acceptance_field_contracts() -> None:
    descriptor = observations.matched_v3_qualification_observation_registry_descriptor()
    contracts = {item["kind"]: item for item in descriptor["payload_contracts"]}

    assert contracts[observations.QUALIFICATION_SEED_OBSERVATION_KIND][
        "acceptance_fields"
    ] == [
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
    ]
    assert contracts[observations.RESOURCE_OBSERVATION_KIND]["acceptance_fields"] == [
        "horizon_accounting_exact",
        "reward_membership_structural_only",
        "all_resource_observations_within_predeclared_integer_ceilings",
    ]
    assert contracts[observations.RESULT_PUBLICATION_OBSERVATION_KIND][
        "acceptance_fields"
    ] == [
        "publisher_descriptor_membership_exact",
        "publisher_source_closure_membership_exact",
        "reload_validator_membership_exact",
        "atomic_publication_exact",
        "strict_reload_exact",
        "full_file_digest_equivalence_exact",
        "score_and_reward_magnitude_not_decoded",
    ]

    seed = _payloads()[3].to_dict()
    resource = _payloads()[5].to_dict()
    publication = _payloads()[6].to_dict()
    assert seed["independent_trust_root_receipt_pins_exact"] is False
    assert seed["preobservation_chronology_exact"] is False
    assert seed["deterministic_case_seed_derivation_exact"] is False
    assert resource["resource_observations_within_predeclared_integer_ceilings"] is True
    assert publication["reload_validator_membership_exact"] is True


def test_every_qualification_plan_v2_profile_field_is_serialized() -> None:
    payload_by_schema = {
        observations.QUALIFICATION_SEED_OBSERVATION_SCHEMA_VERSION: _payloads()[3].to_dict(),
        observations.CANDIDATE_OBSERVATION_SCHEMA_VERSION: _payloads()[4].to_dict(),
        observations.RESOURCE_OBSERVATION_SCHEMA_VERSION: _payloads()[5].to_dict(),
        observations.RESULT_PUBLICATION_OBSERVATION_SCHEMA_VERSION: _payloads()[6].to_dict(),
        observations.FRESH_REPLAY_OBSERVATION_SCHEMA_VERSION: _payloads()[7].to_dict(),
    }
    for profile in plan_v2._probe_profiles():
        payload = payload_by_schema[profile["required_observation_schema"]]
        assert set(profile["acceptance_fields"]) <= set(payload)

    compatibility = (
        observations.matched_v3_qualification_observation_registry_descriptor()[
            "v2_compatibility"
        ]
    )
    assert compatibility["summary_fields_are_caller_controlled"] is False
    assert compatibility["acceptance_evaluation_performed_here"] is False
    assert compatibility["authority_granted"] is False


@pytest.mark.parametrize(
    ("summary_field", "input_fields"),
    (
        (
            "independent_trust_root_receipt_pins_exact",
            (
                "independent_trust_root_receipt_file_pin_exact",
                "independent_trust_root_receipt_binding_pin_exact",
            ),
        ),
        (
            "preobservation_chronology_exact",
            (
                "beacon_time_precedes_observation_cutoff_exact",
                "external_receipt_preacceptance_chronology_exact",
            ),
        ),
        (
            "deterministic_case_seed_derivation_exact",
            (
                "deterministic_28_case_seed_pair_derivation_exact",
                "deterministic_registry_file_and_body_digests_exact",
                "derivation_schema_and_domain_exact",
                "case_derivation_payload_membership_exact",
            ),
        ),
    ),
)
def test_v2_seed_summaries_are_exact_conjunctions_with_complete_false_cases(
    summary_field: str,
    input_fields: tuple[str, ...],
) -> None:
    seed = _payloads()[3]
    assert isinstance(seed, observations.QualificationSeedObservationPayload)
    all_true = dataclasses.replace(
        seed,
        **cast(Any, {field_name: True for field_name in input_fields}),
    )
    assert getattr(all_true, summary_field) is True
    for field_name in input_fields:
        one_false = dataclasses.replace(
            all_true,
            **cast(Any, {field_name: False}),
        )
        assert getattr(one_false, summary_field) is False


@pytest.mark.parametrize(
    ("payload_index", "field_name"),
    (
        (3, "independent_trust_root_receipt_pins_exact"),
        (3, "preobservation_chronology_exact"),
        (3, "deterministic_case_seed_derivation_exact"),
        (5, "resource_observations_within_predeclared_integer_ceilings"),
    ),
)
@pytest.mark.parametrize("bad_value", (True, False, 1))
def test_v2_compatibility_summaries_reject_mismatch_and_integer_alias(
    payload_index: int,
    field_name: str,
    bad_value: object,
) -> None:
    value, _, _ = _encoded(_payloads()[payload_index])
    actual = value["payload"][field_name]
    if type(bad_value) is bool and bad_value is actual:
        bad_value = not bad_value
    value["payload"][field_name] = bad_value
    raw = _rebody(value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize(
    "field_name",
    (
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
    ),
)
def test_every_seed_audit_flag_rejects_integer_boolean_alias(field_name: str) -> None:
    seed = _payloads()[3]
    assert isinstance(seed, observations.QualificationSeedObservationPayload)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(seed, **cast(Any, {field_name: 1}))


def test_candidate_and_resource_bindings_fail_closed_against_plan_v2() -> None:
    candidate = _payloads()[4]
    resource = _payloads()[5]
    assert isinstance(candidate, observations.CandidateObservationPayload)
    assert isinstance(resource, observations.ResourceObservationPayload)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(candidate, candidate_id="unknown-candidate")
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        dataclasses.replace(resource, declared_ceilings=tuple(reversed(resource.declared_ceilings)))


def test_independent_full_file_and_plan_pins_are_required() -> None:
    _, raw, digest = _encoded(_payloads()[4])
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        observations.parse_matched_v3_qualification_observation(
            raw,
            expected_file_sha256=_sha("wrong-file"),
            expected_qualification_plan_sha256=_PLAN_SHA256,
        )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        observations.parse_matched_v3_qualification_observation(
            raw,
            expected_file_sha256=digest,
            expected_qualification_plan_sha256=_sha("wrong-plan"),
        )


def _mutations() -> Iterable[dict[str, Any]]:
    value, _, _ = _encoded(_payloads()[4])

    unknown = copy.deepcopy(value)
    unknown["payload"]["unknown"] = _sha("unknown")
    yield unknown

    true_claim = copy.deepcopy(value)
    true_claim["claims"]["qualification_authority_granted"] = True
    yield true_claim

    wrong_schema = copy.deepcopy(value)
    wrong_schema["observation_schema_version"] = observations.RUNTIME_OBSERVATION_SCHEMA_VERSION
    yield wrong_schema

    integer_boolean = copy.deepcopy(value)
    integer_boolean["payload"]["source_membership_exact"] = 1
    yield integer_boolean


@pytest.mark.parametrize("value", tuple(_mutations()))
def test_unknown_keys_true_authority_schema_drift_and_bool_int_confusion_fail(
    value: dict[str, Any],
) -> None:
    raw = _rebody(value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize(
    "forbidden",
    ("score", "ranking", "reward_magnitude", "cumulative_reward", "total_reward"),
)
def test_score_ranking_and_reward_magnitude_fields_are_forbidden(forbidden: str) -> None:
    value, _, _ = _encoded(_payloads()[6])
    value["payload"][forbidden] = 0
    raw = _rebody(value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())


def test_resource_integer_fields_reject_booleans() -> None:
    value, _, _ = _encoded(_payloads()[5])
    field = plan_v2.RESOURCE_CEILING_FIELDS[0]
    value["payload"]["observed_values"][field] = True
    raw = _rebody(value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())


def test_duplicate_float_and_noncanonical_json_are_rejected() -> None:
    value, raw, _ = _encoded(_payloads()[6])

    duplicate = raw.replace(
        b'"status":',
        b'"status":"duplicate","status":',
        1,
    )
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(duplicate, hashlib.sha256(duplicate).hexdigest())

    value["payload"]["publication_file_count"] = 1.0
    floating = _canonical(value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(floating, hashlib.sha256(floating).hexdigest())

    pretty = json.dumps(json.loads(raw), indent=2, sort_keys=True).encode("ascii") + b"\n"
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(pretty, hashlib.sha256(pretty).hexdigest())


def test_depth_node_text_alias_and_cycle_bounds_fail_closed() -> None:
    value, _, _ = _encoded(_payloads()[4])

    deep: object = None
    for _ in range(70):
        deep = [deep]
    deep_value = copy.deepcopy(value)
    deep_value["payload"]["unknown"] = deep
    deep_raw = _canonical(deep_value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(deep_raw, hashlib.sha256(deep_raw).hexdigest())

    nodes_value = copy.deepcopy(value)
    nodes_value["payload"]["unknown"] = [None] * 100_001
    nodes_raw = _canonical(nodes_value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(nodes_raw, hashlib.sha256(nodes_raw).hexdigest())

    text_value = copy.deepcopy(value)
    text_value["payload"]["unknown"] = "x" * 16_385
    text_raw = _canonical(text_value)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(text_raw, hashlib.sha256(text_raw).hexdigest())

    shared: list[object] = []
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        observations._canonical_json({"left": shared, "right": shared})
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        observations._canonical_json({"cycle": cycle})


def test_source_variants_require_producer_specific_exact_keys() -> None:
    external, _, _ = _encoded(_payloads()[0])
    external["payload"].pop("staging_manifest_body_sha256")
    raw = _rebody(external)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())

    local, _, _ = _encoded(_payloads()[1])
    local["payload"]["producer_kind"] = (
        "durable_external_source_publication_v1_materialization_v2"
    )
    raw = _rebody(local)
    with pytest.raises(observations.ForagerMatchedV3QualificationObservationError):
        _parse(raw, hashlib.sha256(raw).hexdigest())
