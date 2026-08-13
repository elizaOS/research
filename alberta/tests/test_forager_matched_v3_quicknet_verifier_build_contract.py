"""Tests for the detached matched-v3 Quicknet build and wire contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_quicknet_verifier_build_contract as contract,
)

pytestmark = pytest.mark.unit

_EXPECTED_DESCRIPTOR_SHA256 = "513ed21d7411f65a3d605f38eedc6da5cf6d6764203ebcf38210439a4724cb87"
_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
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


def _descriptor() -> dict[str, Any]:
    return contract.matched_v3_quicknet_verifier_build_contract_descriptor()


def _request(
    *,
    round_number: int = 1_000,
    public_key: bytes = bytes(range(96)),
    signature: bytes = bytes(range(48)),
) -> bytes:
    return contract.canonical_matched_v3_quicknet_verifier_request_bytes(
        round_number=round_number,
        public_key=public_key,
        signature=signature,
    )


def test_descriptor_is_canonical_detached_and_zero_pinned_for_root_audit() -> None:
    raw = contract.canonical_matched_v3_quicknet_verifier_build_contract_descriptor_bytes()
    assert raw == _canonical(json.loads(raw))
    assert contract.MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        contract.parse_matched_v3_quicknet_verifier_build_contract_descriptor(raw) == _descriptor()
    )

    first = _descriptor()
    first["state"]["build_ready"] = True
    first["authority"]["execution_authority_granted"] = True
    first["wire_contract"]["request"]["total_size_bytes"] = 1
    second = _descriptor()
    assert second["state"]["build_ready"] is False
    assert second["authority"]["execution_authority_granted"] is False
    assert second["wire_contract"]["request"]["total_size_bytes"] == 160


def test_descriptor_binds_finalized_prerequisite_and_source_registry_identities() -> None:
    prerequisites = _descriptor()["source_prerequisites"]
    assert prerequisites["verifier_source_registry"] == {
        "descriptor_schema_version": (
            "alberta.forager_matched_v3.quicknet_verifier_source_descriptor.v2"
        ),
        "descriptor_sha256": ("4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"),
        "source_sha256": ("3e13009c1843c3341e5a0eb8b2f84ea903b8e5315fbdef347549757710fd3623"),
        "imported_or_executed_here": False,
    }
    materialization = prerequisites["source_materialization_contract"]
    assert materialization["descriptor_sha256"] == (
        "61345825673afb16bc1942c4b8c84e763fb14530a68225caffa94d98e733a03d"
    )
    assert materialization["plan_sha256"] == (
        "5ccbed13f70ed15355c6849d732801db6e372864ac570fc62f4acf4a78cde0e7"
    )
    assert materialization["source_sha256"] == (
        "1e08a04b8c3120978867999b5316d57ac5361771b018496535a5ab5a77a61023"
    )
    assert materialization["test_sha256"] == (
        "6aa06c38345d833cbd6e716469ea1871fa249ea4da321648ffc283c66c423f69"
    )


def test_bound_prerequisite_source_and_test_files_still_match() -> None:
    verifier_source_path = (
        _ROOT / "alberta_framework/benchmarks/forager_matched_v3_quicknet_verifier_source.py"
    )
    source_path = (
        _ROOT / "alberta_framework/benchmarks/forager_matched_v3_quicknet_source_materialization.py"
    )
    test_path = _ROOT / "tests/test_forager_matched_v3_quicknet_source_materialization.py"
    assert hashlib.sha256(verifier_source_path.read_bytes()).hexdigest() == (
        contract.QUICKNET_SOURCE_REGISTRY_SOURCE_SHA256
    )
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        contract.QUICKNET_SOURCE_MATERIALIZATION_SOURCE_SHA256
    )
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == (
        contract.QUICKNET_SOURCE_MATERIALIZATION_TEST_SHA256
    )


def test_prerequisite_inventory_is_not_relabelled_as_materialized_vendor_or_build_input() -> None:
    prerequisite = _descriptor()["source_prerequisites"]["source_materialization_contract"]
    assert prerequisite["manifest_or_receipt_instance_bound_here"] is False
    assert prerequisite["caller_supplied_archives_inventoried_in_memory_only"] is True
    assert prerequisite["filesystem_tree_materialized"] is False
    assert prerequisite["archive_member_extracted_to_path"] is False
    assert prerequisite["dependency_vendor_closure_provided"] is False
    assert prerequisite["build_input_selected"] is False
    assert prerequisite["build_or_runtime_authority_granted"] is False
    assert prerequisite["trust_or_chronology_authority_granted"] is False


def test_future_artifact_schema_names_are_exact_and_all_are_absent() -> None:
    assert _descriptor()["artifact_schemas"] == {
        "adapter_descriptor": (
            "alberta.forager_matched_v3.quicknet_verifier_adapter_descriptor.v1"
        ),
        "build_plan": "alberta.forager_matched_v3.quicknet_verifier_build_plan.v1",
        "build_receipt": "alberta.forager_matched_v3.quicknet_verifier_build_receipt.v1",
        "cargo_vendor_closure": ("alberta.forager_matched_v3.quicknet_cargo_vendor_closure.v1"),
        "final_image_identity": (
            "alberta.forager_matched_v3.quicknet_verifier_final_image_identity.v1"
        ),
        "final_image_vector_run_receipt": (
            "alberta.forager_matched_v3.quicknet_verifier_final_image_vector_run_receipt.v1"
        ),
        "oci_build_context_receipt_v2": (
            "alberta.forager_matched_v3.quicknet_verifier_oci_build_context_receipt.v2"
        ),
        "oci_build_execution_receipt_v2": (
            "alberta.forager_matched_v3.quicknet_verifier_oci_build_execution_receipt.v2"
        ),
        "oci_build_plan_v2": ("alberta.forager_matched_v3.quicknet_verifier_oci_build_plan.v2"),
        "oci_build_publication_v2": (
            "alberta.forager_matched_v3.quicknet_verifier_oci_build_publication.v2"
        ),
        "reproducibility_receipt": (
            "alberta.forager_matched_v3.quicknet_verifier_reproducibility_receipt.v1"
        ),
        "request": "alberta.forager_matched_v3.quicknet_verifier_request.v1",
        "runtime_verifier_descriptor": (
            "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
        ),
        "rust_toolchain_manifest": (
            "alberta.forager_matched_v3.quicknet_rust_toolchain_manifest.v1"
        ),
        "rustsec_audit_receipt": ("alberta.forager_matched_v3.quicknet_rustsec_audit_receipt.v1"),
        "source_tree_materialization_receipt": (
            "alberta.forager_matched_v3.quicknet_source_tree_materialization_receipt.v1"
        ),
        "verifier_outcome": "alberta.forager_matched_v3.quicknet_verifier_outcome.v1",
        "verifier_receipt": "alberta.forager_matched_v3.quicknet_verifier_receipt.v1",
        "vector_run_receipt": (
            "alberta.forager_matched_v3.quicknet_verifier_vector_run_receipt.v1"
        ),
        "vector_suite": "alberta.forager_matched_v3.quicknet_verifier_vector_suite.v1",
    }
    build = _descriptor()["future_build_contract"]
    assert build["all_artifacts_absent_here"] is True
    for name, stage in build.items():
        if name != "all_artifacts_absent_here":
            assert stage["satisfied_here"] is False


def test_request_wire_layout_is_exact_160_bytes_and_big_endian() -> None:
    public_key = bytes(range(96))
    signature = bytes(range(48))
    raw = _request(round_number=0x0102030405060708, public_key=public_key, signature=signature)
    assert len(raw) == 160
    assert raw[:8] == b"ALBQNV1\0"
    assert raw[8:16] == bytes.fromhex("0102030405060708")
    assert raw[16:112] == public_key
    assert raw[112:160] == signature
    assert contract.parse_matched_v3_quicknet_verifier_request(raw) == (
        contract.MatchedV3QuicknetVerifierRequest(
            round_number=0x0102030405060708,
            public_key=public_key,
            signature=signature,
        )
    )


@pytest.mark.parametrize("round_number", [0, 1, (1 << 64) - 1])
def test_request_round_accepts_the_full_exact_u64_domain(round_number: int) -> None:
    raw = _request(round_number=round_number)
    assert int.from_bytes(raw[8:16], byteorder="big", signed=False) == round_number
    assert contract.parse_matched_v3_quicknet_verifier_request(raw).round_number == round_number


@pytest.mark.parametrize("round_number", [-1, 1 << 64, True])
def test_request_encoder_rejects_out_of_range_and_boolean_rounds(round_number: Any) -> None:
    with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
        _request(round_number=round_number)


@pytest.mark.parametrize(
    ("public_key", "signature"),
    [
        (b"k" * 95, b"s" * 48),
        (b"k" * 97, b"s" * 48),
        (b"k" * 96, b"s" * 47),
        (b"k" * 96, b"s" * 49),
        (bytearray(96), b"s" * 48),
        (b"k" * 96, bytearray(48)),
    ],
)
def test_request_encoder_rejects_width_and_container_aliases(
    public_key: Any,
    signature: Any,
) -> None:
    with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
        contract.canonical_matched_v3_quicknet_verifier_request_bytes(
            round_number=1,
            public_key=public_key,
            signature=signature,
        )


def test_request_structural_parser_deliberately_does_not_validate_curve_points() -> None:
    raw = _request(public_key=b"\0" * 96, signature=b"\0" * 48)
    parsed = contract.parse_matched_v3_quicknet_verifier_request(raw)
    assert parsed.public_key == b"\0" * 96
    assert parsed.signature == b"\0" * 48
    assert (
        _descriptor()["wire_contract"]["request"]["structural_parser_validates_curve_points"]
        is False
    )


def test_request_parser_rejects_wrong_type_width_magic_and_trailing_bytes() -> None:
    raw = _request()
    candidates: list[Any] = [
        bytearray(raw),
        raw[:-1],
        raw + b"\0",
        b"BADMAGIC" + raw[8:],
    ]
    for candidate in candidates:
        with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
            contract.parse_matched_v3_quicknet_verifier_request(candidate)


def test_verified_outcome_is_exactly_41_bytes_and_binds_signature_randomness() -> None:
    request = _request(signature=b"signature".ljust(48, b"!"))
    signature = request[112:160]
    randomness = hashlib.sha256(signature).digest()
    raw = b"ALBQNO1\0" + b"\x00" + randomness
    assert len(raw) == 41
    parsed = contract.parse_matched_v3_quicknet_verifier_outcome(
        raw,
        request_bytes=request,
    )
    assert parsed == contract.MatchedV3QuicknetVerifierOutcome(
        outcome="verified",
        outcome_tag=0,
        randomness=randomness,
    )


def test_verified_outcome_rejects_wrong_randomness_short_long_and_other_request() -> None:
    request = _request(signature=b"a" * 48)
    good = b"ALBQNO1\0" + b"\x00" + hashlib.sha256(b"a" * 48).digest()
    candidates = [
        good[:9] + b"\0" * 32,
        good[:-1],
        good + b"\0",
    ]
    for raw in candidates:
        with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
            contract.parse_matched_v3_quicknet_verifier_outcome(
                raw,
                request_bytes=request,
            )
    with pytest.raises(
        contract.ForagerMatchedV3QuicknetVerifierBuildContractError,
        match="randomness relation",
    ):
        contract.parse_matched_v3_quicknet_verifier_outcome(
            good,
            request_bytes=_request(signature=b"b" * 48),
        )


@pytest.mark.parametrize(
    ("tag", "name"),
    [
        (1, "ok_false"),
        (2, "verification_error"),
        (3, "invalid_public_key"),
    ],
)
def test_negative_outcomes_are_exactly_nine_bytes_with_no_payload(
    tag: int,
    name: str,
) -> None:
    raw = b"ALBQNO1\0" + bytes([tag])
    assert len(raw) == 9
    parsed = contract.parse_matched_v3_quicknet_verifier_outcome(
        raw,
        request_bytes=_request(),
    )
    assert parsed == contract.MatchedV3QuicknetVerifierOutcome(
        outcome=name,
        outcome_tag=tag,
        randomness=None,
    )
    for leakage in (b"x", b"\0" * 32, b"free form error"):
        with pytest.raises(
            contract.ForagerMatchedV3QuicknetVerifierBuildContractError,
            match="exactly 9 bytes with no payload",
        ):
            contract.parse_matched_v3_quicknet_verifier_outcome(
                raw + leakage,
                request_bytes=_request(),
            )


@pytest.mark.parametrize("tag", [1, 2, 3])
def test_negative_outcomes_do_not_derive_or_expose_randomness(
    tag: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_sha256(_: bytes) -> Any:
        raise AssertionError("negative outcome attempted randomness derivation")

    monkeypatch.setattr(hashlib, "sha256", forbidden_sha256)
    parsed = contract.parse_matched_v3_quicknet_verifier_outcome(
        b"ALBQNO1\0" + bytes([tag]),
        request_bytes=_request(),
    )
    assert parsed.randomness is None


def test_outcome_parser_rejects_type_short_magic_unknown_tag_and_trailing_data() -> None:
    request = _request()
    candidates: list[Any] = [
        bytearray(b"ALBQNO1\0\x01"),
        b"ALBQNO1\0",
        b"BADMAGIC\x01",
        b"ALBQNO1\0\xff",
        b"ALBQNO1\0\x01x",
    ]
    for candidate in candidates:
        with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
            contract.parse_matched_v3_quicknet_verifier_outcome(
                candidate,
                request_bytes=request,
            )


def test_wire_descriptor_matches_codecs_and_future_process_protocol() -> None:
    wire = _descriptor()["wire_contract"]
    request = wire["request"]
    assert request["total_size_bytes"] == 160
    request_fields = [
        (field["name"], field["offset_bytes"], field["size_bytes"]) for field in request["fields"]
    ]
    assert request_fields == [
        ("magic", 0, 8),
        ("round", 8, 8),
        ("public_key", 16, 96),
        ("signature", 112, 48),
    ]
    outcome = wire["outcome"]
    outcome_tags = [
        (entry["tag"], entry["name"], entry["total_size_bytes"]) for entry in outcome["tags"]
    ]
    assert outcome_tags == [
        (0, "verified", 41),
        (1, "ok_false", 9),
        (2, "verification_error", 9),
        (3, "invalid_public_key", 9),
    ]
    assert outcome["randomness_derived_only_after_upstream_ok_true"] is True
    assert outcome["randomness_on_ok_false_allowed"] is False
    assert outcome["randomness_on_error_allowed"] is False
    assert outcome["free_form_error_payload_allowed"] is False
    assert outcome["structural_parser_authenticates_verifier_truth"] is False

    process = wire["future_process_protocol"]
    assert process["argv_argument_count"] == 0
    assert process["stdin_request_count"] == 1
    assert process["stdin_eof_required_after_request"] is True
    assert process["stdout_outcome_count"] == 1
    assert process["stderr_must_be_empty_for_typed_outcomes"] is True
    assert process["typed_outcome_exit_code"] == 0
    assert process["framing_or_internal_error_exit_code_nonzero"] is True
    assert process["accepted_outcome_on_nonzero_exit_allowed"] is False
    for denial in (
        "file_input_allowed",
        "network_input_allowed",
        "clock_input_allowed",
        "environment_input_allowed",
        "default_pulse_input_allowed",
        "child_process_allowed",
    ):
        assert process[denial] is False


def test_build_contract_requires_real_vendor_toolchain_rustsec_and_two_clean_builds() -> None:
    build = _descriptor()["future_build_contract"]
    source_tree = build["source_tree_materialization"]
    assert source_tree["must_replay_exact_caller_archive_bytes"] is True
    assert source_tree["must_materialize_to_separate_immutable_tree"] is True
    assert source_tree["satisfied_here"] is False

    adapter = build["adapter_source"]
    assert adapter["separate_source_bytes_and_sha256_required"] is True
    assert adapter["producer_identity_requires_descriptor_and_source_sha256"] is True
    assert adapter["wire_contract_sha256_required"] is True
    assert adapter["source_implemented_here"] is False

    vendor = build["cargo_vendor_closure"]
    assert vendor["cargo_lock_sha256"] == (
        "6dd200178128e6e02788b194c856ff3668abf2916b321431790760c950739767"
    )
    assert vendor["locked_dependency_count"] == 36
    assert vendor["all_locked_crate_archives_and_full_inventories_required"] is True
    assert vendor["every_cargo_checksum_json_required"] is True
    assert vendor["dependency_and_target_feature_graph_required"] is True
    assert vendor["exact_default_feature_policy_required"] is True
    assert vendor["path_or_git_dependencies_allowed"] is False
    assert vendor["filesystem_tree_or_vendor_closure_available_here"] is False

    toolchain = build["toolchain"]
    assert toolchain["rustc_cargo_target_linker_and_runtime_bytes_required"] is True
    assert toolchain["exact_versions_sizes_sha256_and_argv_required"] is True
    assert toolchain["toolchain_available_here"] is False

    audit = build["fresh_rustsec"]
    assert audit["fresh_scan_required"] is True
    assert audit["historical_source_registry_scan_may_substitute"] is False
    assert audit["network_during_scan_allowed"] is False
    assert audit["receipt_available_here"] is False

    offline = build["offline_build"]
    assert offline["required_cargo_flags"] == ["--frozen", "--offline"]
    assert offline["clean_builder_count"] == 2
    assert offline["distinct_builder_identity_required"] is True
    assert offline["shared_target_directory_allowed"] is False
    assert offline["writable_source_or_vendor_allowed"] is False
    assert offline["network_during_build_allowed"] is False
    assert offline["build_executed_here"] is False


def test_reproducibility_contract_rejects_binary_or_toolchain_substitution_by_design() -> None:
    reproducibility = _descriptor()["future_build_contract"]["reproducibility"]
    assert reproducibility["two_distinct_build_receipts_required"] is True
    assert reproducibility["same_plan_toolchain_source_vendor_and_adapter_required"] is True
    assert reproducibility["byte_identical_binary_required"] is True
    assert reproducibility["binary_sha256_size_and_elf_identity_required"] is True
    assert reproducibility["binary_may_embed_its_own_sha256"] is False
    assert reproducibility["reproduced_binary_available_here"] is False
    assert reproducibility["satisfied_here"] is False


def test_vector_contract_requires_positive_adversarial_and_leakage_cases() -> None:
    vectors = _descriptor()["future_build_contract"]["standalone_vectors"]
    assert vectors["official_round_1000_positive_required"] is True
    assert vectors["negative_classes_required"] == [
        "wrong_round",
        "wrong_signature",
        "malformed_compression",
        "noncanonical_point",
        "invalid_subgroup",
        "ok_false_randomness_leakage",
        "error_randomness_leakage",
    ]
    assert vectors["network_during_vector_run_allowed"] is False
    assert vectors["suite_or_receipt_available_here"] is False


def test_fresh_oci_v2_must_copy_and_recheck_exact_reproduced_binary() -> None:
    oci = _descriptor()["future_oci_v2_contract"]
    assert oci["fresh_lineage_required"] is True
    assert oci["historical_cpu_oci_v1_plan_accepted"] is False
    assert oci["historical_twelve_member_context_extended_or_reinterpreted"] is False
    binary = oci["binary_copy"]
    assert binary == {
        "source": "exact_reproducibility_receipt_binary",
        "context_path": "inputs/alberta-quicknet-verify",
        "image_path": "/opt/elizaos/bin/alberta-quicknet-verify",
        "mode": "0555",
        "uid": 0,
        "gid": 0,
        "reproducibility_binary_sha256_and_size_must_match_context_member": True,
        "context_member_sha256_and_size_must_match_final_image_file": True,
        "final_image_file_sha256_must_be_rechecked_before_runtime_use": True,
        "elf_identity_must_match_at_every_boundary": True,
        "post_hash_binary_substitution_allowed": False,
    }
    assert oci["rust_build_inside_final_image_allowed"] is False
    assert oci["rustc_or_cargo_in_final_image_allowed"] is False
    assert oci["source_or_vendor_tree_in_final_image_allowed"] is False
    assert oci["dockerfile_rust_build_step_allowed"] is False
    assert oci["network_during_oci_build_allowed"] is False
    assert oci["pull_during_oci_build_allowed"] is False
    assert oci["exact_base_image_digest_required"] is True
    assert oci["exact_runtime_loader_and_library_identity_required"] is True
    assert oci["final_image_vector_replay_required"] is True
    assert oci["final_image_vector_replay_must_use_copied_binary"] is True
    assert oci["final_image_vector_replay_network_allowed"] is False
    assert oci["final_image_or_receipt_available_here"] is False
    assert oci["satisfied_here"] is False


def test_future_runtime_verifier_descriptor_is_exact_distinct_and_unsatisfied() -> None:
    runtime = _descriptor()["future_runtime_verifier_descriptor"]
    assert runtime["schema_version"] == (
        "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
    )
    assert runtime["role"] == "production_quicknet_runtime_verifier_producer_identity"
    assert runtime["adapter_binding"] == {
        "schema_version": ("alberta.forager_matched_v3.quicknet_verifier_adapter_descriptor.v1"),
        "descriptor_full_sha256_required": True,
        "adapter_source_sha256_required": True,
    }
    assert runtime["reproduced_binary_binding"]["exact_binary_sha256_required"] is True
    assert runtime["reproduced_binary_binding"]["exact_binary_size_bytes_required"] is True
    assert runtime["reproduced_binary_binding"]["exact_elf_identity_required"] is True
    assert (
        runtime["final_image_binding"][
            "fresh_oci_v2_plan_context_execution_publication_chain_required"
        ]
        is True
    )
    assert runtime["final_image_binding"]["exact_image_digest_required"] is True
    assert (
        runtime["final_image_binding"]["in_image_binary_sha256_size_and_elf_match_required"] is True
    )
    vectors = runtime["vector_binding"]
    assert vectors["standalone_schema_version"] == (
        "alberta.forager_matched_v3.quicknet_verifier_vector_run_receipt.v1"
    )
    assert vectors["final_image_schema_version"] == (
        "alberta.forager_matched_v3.quicknet_verifier_final_image_vector_run_receipt.v1"
    )
    assert vectors["both_receipt_full_and_body_sha256_required"] is True
    assert vectors["both_receipts_must_bind_same_vector_suite"] is True
    assert vectors["both_receipts_must_bind_same_reproduced_binary"] is True
    wire = runtime["wire_and_process_binding"]
    assert wire["request_schema_version"] == (
        "alberta.forager_matched_v3.quicknet_verifier_request.v1"
    )
    assert wire["outcome_schema_version"] == (
        "alberta.forager_matched_v3.quicknet_verifier_outcome.v1"
    )
    assert (
        wire["wire_contract_canonical_sha256"]
        == hashlib.sha256(_canonical(_descriptor()["wire_contract"])).hexdigest()
    )
    assert wire["exact_wire_contract_projection_required"] is True
    assert wire["exact_process_protocol_projection_required"] is True
    assert wire["build_contract_descriptor_full_sha256_required"] is True
    assert runtime["producer_source_sha256_required"] is True
    assert runtime["descriptor_issued_here"] is False
    assert runtime["producer_available_here"] is False
    assert runtime["satisfied_here"] is False
    assert set(runtime["authority"].values()) == {False}


def test_future_verifier_receipt_requires_runtime_producer_request_outcome_and_trust() -> None:
    receipt = _descriptor()["future_verifier_receipt"]
    assert receipt == {
        "schema_version": "alberta.forager_matched_v3.quicknet_verifier_receipt.v1",
        "must_bind_runtime_verifier_descriptor_full_and_body_sha256": True,
        "must_bind_runtime_verifier_producer_source_sha256": True,
        "must_bind_exact_request_and_outcome_bytes": True,
        "must_bind_separate_trust_policy_and_pulse": True,
        "does_not_authenticate_seed_registry_or_chronology": True,
        "issued_here": False,
    }


def test_future_pin_graph_is_forward_only_and_every_stage_is_unsatisfied() -> None:
    graph = _descriptor()["future_pin_graph"]
    assert [stage["stage"] for stage in graph] == [
        "source_tree_materialization",
        "cargo_vendor_closure",
        "adapter_source",
        "offline_build_plan",
        "independent_builds",
        "binary_reproducibility",
        "standalone_vector_run",
        "fresh_oci_v2_lineage",
        "final_image_vector_run",
        "runtime_verifier_descriptor",
        "future_verifier_receipt",
    ]
    assert all(stage["satisfied_here"] is False for stage in graph)
    assert graph[0]["output"] == (
        "alberta.forager_matched_v3.quicknet_source_tree_materialization_receipt.v1"
    )
    assert graph[5]["output"] == (
        "alberta.forager_matched_v3.quicknet_verifier_reproducibility_receipt.v1"
    )
    assert graph[-2]["output"] == (
        "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
    )
    assert graph[-1]["output"] == ("alberta.forager_matched_v3.quicknet_verifier_receipt.v1")
    assert graph[-1]["inputs"][0] == (
        "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
    )
    for stage in graph:
        assert stage["output"] not in stage["inputs"]


def test_every_authority_and_every_readiness_state_remain_false() -> None:
    descriptor = _descriptor()
    assert descriptor["authority"]
    assert set(descriptor["authority"].values()) == {False}
    readiness = {key: value for key, value in descriptor["state"].items() if key.endswith("_ready")}
    assert readiness == {
        "build_ready": False,
        "runtime_ready": False,
        "qualification_ready": False,
    }
    for key in (
        "filesystem_source_tree_materialized",
        "dependency_vendor_closure_available",
        "adapter_source_available",
        "toolchain_available",
        "fresh_rustsec_receipt_available",
        "build_plan_issued",
        "rust_build_executed",
        "reproduced_binary_available",
        "standalone_vectors_executed",
        "oci_v2_plan_issued",
        "final_image_available",
        "final_image_vectors_executed",
        "runtime_verifier_descriptor_issued",
        "runtime_verifier_producer_available",
        "verifier_invoked",
        "verifier_receipt_issued",
    ):
        assert descriptor["state"][key] is False


def test_capabilities_expose_only_pure_wire_operations() -> None:
    capabilities = _descriptor()["capabilities"]
    assert capabilities["request_encoder_api_exposed"] is True
    assert capabilities["request_parser_api_exposed"] is True
    assert capabilities["outcome_parser_api_exposed"] is True
    for key, value in capabilities.items():
        if key not in {
            "request_encoder_api_exposed",
            "request_parser_api_exposed",
            "outcome_parser_api_exposed",
        }:
            assert value is False
    assert not hasattr(contract, "canonical_matched_v3_quicknet_verifier_outcome_bytes")


def test_request_and_outcome_records_are_frozen() -> None:
    request = contract.parse_matched_v3_quicknet_verifier_request(_request())
    with pytest.raises((FrozenInstanceError, AttributeError)):
        request.round_number = 2  # type: ignore[misc]
    outcome = contract.parse_matched_v3_quicknet_verifier_outcome(
        b"ALBQNO1\0\x01",
        request_bytes=_request(),
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        outcome.outcome = "verified"  # type: ignore[misc]


def test_descriptor_parser_accepts_only_exact_canonical_frozen_identity() -> None:
    raw = contract.canonical_matched_v3_quicknet_verifier_build_contract_descriptor_bytes()
    for noncanonical in (b" " + raw, raw[:-1], raw + b"\n", raw.replace(b":", b": ", 1)):
        with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
            contract.parse_matched_v3_quicknet_verifier_build_contract_descriptor(noncanonical)
    mutated = _descriptor()
    mutated["status"] = "different_but_canonical"
    with pytest.raises(
        contract.ForagerMatchedV3QuicknetVerifierBuildContractError,
        match="identity",
    ):
        contract.parse_matched_v3_quicknet_verifier_build_contract_descriptor(_canonical(mutated))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":1.0}\n',
        b'{"x":-1}\n',
        b'{"x":999999999999999999999}\n',
        b'{"x":"\xff"}\n',
        b"[]\n",
    ],
)
def test_descriptor_parser_rejects_duplicate_nonfinite_float_bounds_and_nonobject(
    raw: bytes,
) -> None:
    with pytest.raises(contract.ForagerMatchedV3QuicknetVerifierBuildContractError):
        contract.parse_matched_v3_quicknet_verifier_build_contract_descriptor(raw)


def test_source_module_is_stdlib_only_and_has_no_operational_imports() -> None:
    module_path = Path(contract.__file__)
    raw_source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(raw_source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "typing",
    }
    assert imported_roots.isdisjoint(
        {
            "ctypes",
            "fcntl",
            "httpx",
            "importlib",
            "mmap",
            "os",
            "pathlib",
            "requests",
            "secrets",
            "socket",
            "subprocess",
            "tempfile",
            "time",
            "urllib",
        }
    )
    assert "forager_matched_v3_quicknet_source_materialization" not in raw_source
    assert "forager_matched_v3_quicknet_verifier_source" not in raw_source


def test_limitations_explicitly_deny_materialization_vendor_execution_and_authority() -> None:
    limitations = " ".join(_descriptor()["limitations"])
    for phrase in (
        "inventories caller bytes in memory and materializes no filesystem tree",
        "No concrete prerequisite-1 manifest or receipt instance is accepted",
        "not a Cargo dependency vendor closure or build input",
        "not BLS truth",
        "artifacts are absent",
        "must copy a separately reproduced binary and must not build Rust",
        "runtime-verifier descriptor is a future artifact distinct",
        "cannot be extended or reinterpreted",
        "grants readiness, authority, evidence, or qualification",
    ):
        assert phrase in limitations


def test_frozen_descriptor_copy_does_not_share_nested_mutable_state() -> None:
    first = copy.deepcopy(_descriptor())
    first["future_oci_v2_contract"]["binary_copy"]["mode"] = "0777"
    assert _descriptor()["future_oci_v2_contract"]["binary_copy"]["mode"] == "0555"
