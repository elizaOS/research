"""Tests for the detached, nonauthorizing matched-v3 Quicknet source registry."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_seed_registry as seed_registry,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_quicknet_verifier_source as source,
)

pytestmark = pytest.mark.unit

_EXPECTED_DESCRIPTOR_SHA256 = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)

_EXPECTED_FILE_PINS = {
    ".cargo_vcs_info.json": (
        "f304ef56e003d4cf1c29f052279ffafed2fd83d7a49ffce07377fc66687060fa",
        "crates_io_package",
    ),
    "CHANGELOG.md": (
        "f0393d5e35a54a5ad9181307fb4b242912925542bf15636cb817da12dafeab94",
        "release_source",
    ),
    "Cargo.lock": (
        "6dd200178128e6e02788b194c856ff3668abf2916b321431790760c950739767",
        "release_source",
    ),
    "Cargo.toml": (
        "499d25339dc90d107633cab975404ea1fdac8e03b4f784fa64e5f06dccf04cc1",
        "release_source",
    ),
    "LICENSE": (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "release_source",
    ),
    "MAINTENANCE.md": (
        "f6b31cf4f31d718253bfa3a1ddbcc32cf83ff71a745006f7fc6b9cd89c0a5272",
        "release_source",
    ),
    "NOTICE": (
        "3322338486638129acbae928b2025eac84395088ff92cdc2f07528e00312ac8e",
        "release_source",
    ),
    "README.md": (
        "b45615e1648c152d60bc784b28e9ba0d0ec2618ecc84acef7eb09d7e9618b3a5",
        "release_source",
    ),
    "src/lib.rs": (
        "5aab4357c622f089cb0a25825356f6cece18259c42af8aca1069c8708a15b97c",
        "release_source",
    ),
    "src/points.rs": (
        "7fa15e818aead5758a44306b304d35c8ff3ab3d9491e8370fa4698546f545eee",
        "release_source",
    ),
    "src/randomness.rs": (
        "d66781a3e78b61fe64b3e734b8f159e73926f13dc4badd546498446c1418575b",
        "release_source",
    ),
    "src/verify.rs": (
        "47c7a755b4bc226371df83ec8dc430c7a37f0f0ef2bcf55f898cad52e100f9a6",
        "release_source",
    ),
}


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
    return source.matched_v3_quicknet_verifier_source_descriptor()


def test_descriptor_is_canonical_detached_and_sha_frozen() -> None:
    raw = source.canonical_matched_v3_quicknet_verifier_source_descriptor_bytes()
    assert raw == _canonical(json.loads(raw))
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256
    assert source.MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert _descriptor()["schema_version"].endswith(".v2")
    assert source.parse_matched_v3_quicknet_verifier_source_descriptor(raw) == _descriptor()

    first = _descriptor()
    first["status"] = "mutated"
    first["authority"]["execution_authority_granted"] = True
    second = _descriptor()
    assert second["status"] == source.MATCHED_V3_QUICKNET_VERIFIER_SOURCE_STATUS
    assert second["authority"]["execution_authority_granted"] is False


def test_parser_accepts_only_exact_bytes_and_exact_frozen_identity() -> None:
    raw = source.canonical_matched_v3_quicknet_verifier_source_descriptor_bytes()
    for noncanonical in (b" " + raw, raw[:-1], raw + b"\n", raw.replace(b":", b": ", 1)):
        with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError):
            source.parse_matched_v3_quicknet_verifier_source_descriptor(noncanonical)

    mutated = _descriptor()
    mutated["status"] = "different_but_canonical"
    with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError, match="identity"):
        source.parse_matched_v3_quicknet_verifier_source_descriptor(_canonical(mutated))

    with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError, match="exact bytes"):
        source.parse_matched_v3_quicknet_verifier_source_descriptor(
            cast(Any, bytearray(raw))
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":1.0}\n',
        b'{"x":999999999999999999999}\n',
        b'{"x":"\xff"}\n',
        b"[]\n",
    ],
)
def test_parser_rejects_duplicates_nonfinite_floats_bounds_nonascii_and_nonobject(
    raw: bytes,
) -> None:
    with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError):
        source.parse_matched_v3_quicknet_verifier_source_descriptor(raw)


def test_parser_rejects_oversize_and_excessive_depth() -> None:
    with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError, match="byte bound"):
        source.parse_matched_v3_quicknet_verifier_source_descriptor(b"{" + b" " * 300_000)
    deep = b'{"x":' + b"[" * 60 + b"0" + b"]" * 60 + b"}\n"
    with pytest.raises(source.ForagerMatchedV3QuicknetVerifierSourceError):
        source.parse_matched_v3_quicknet_verifier_source_descriptor(deep)


def test_release_commit_tree_archives_and_canonical_redirect_are_exact() -> None:
    identity = _descriptor()["source_identity"]
    assert identity["canonical_repository_url"] == "https://github.com/CosmWasm/drand-verify"
    assert identity["historical_cargo_repository_redirect"] == {
        "historical_url": "https://github.com/noislabs/drand-verify",
        "redirect_target": "https://github.com/CosmWasm/drand-verify",
        "redirect_is_an_authentication_authority": False,
    }
    assert identity["release_tag"] == "v0.6.2"
    assert identity["commit_git_sha1"] == "1db2248afac44fc2e5c9c78f896b4412d8679914"
    assert identity["tree_git_sha1"] == "35c957bc3466992194df43fc597014791dd7abe4"
    assert identity["commit_archive"] == {
        "sha256": "633408b2d2adca4d9986e765ee2ece148b26de50f7440db5c5f3f7054edfe760",
        "size_bytes": 18_727,
        "source_bytes_fetched_here": False,
    }
    assert identity["license_spdx"] == "Apache-2.0"


def test_crates_io_package_and_cargo_vcs_declaration_are_exact_but_nonauthenticating() -> None:
    package = _descriptor()["source_identity"]["crates_io_package"]
    assert package == {
        "crate_name": "drand-verify",
        "version": "0.6.2",
        "sha256": "4c1d531704590bbfce3433cd735378d135cabc9e318d8aa52c5dccf7b80178ee",
        "size_bytes": 18_961,
        "cargo_vcs_info_logical_name": ".cargo_vcs_info",
        "cargo_vcs_info_archive_path": ".cargo_vcs_info.json",
        "cargo_vcs_info_sha256": (
            "f304ef56e003d4cf1c29f052279ffafed2fd83d7a49ffce07377fc66687060fa"
        ),
        "cargo_vcs_info_declared_commit_git_sha1": (
            "1db2248afac44fc2e5c9c78f896b4412d8679914"
        ),
        "cargo_vcs_info_is_upstream_package_declaration": True,
        "cargo_vcs_info_authenticates_commit": False,
        "cargo_vcs_info_authenticates_crate_bytes": False,
        "package_downloaded_here": False,
    }


def test_all_relevant_file_pins_are_exact_and_ordered() -> None:
    records = _descriptor()["source_identity"]["relevant_file_sha256"]
    observed = {
        record["path"]: (record["sha256"], record["artifact_scope"]) for record in records
    }
    assert observed == _EXPECTED_FILE_PINS
    assert list(observed) == sorted(observed, key=lambda path: path.encode("ascii"))
    assert len(records) == 12
    assert tuple(
        (record["path"], record["sha256"], record["artifact_scope"])
        for record in records
    ) == source.PINNED_RELEVANT_FILE_SHA256


def test_audited_api_semantics_are_narrow_offline_and_source_only() -> None:
    semantics = _descriptor()["audited_api_semantics"]
    assert semantics == {
        "public_key_type": "G2PubkeyRfc",
        "verification_call": "Pubkey::verify(round,b'',signature)",
        "unchained_previous_signature_argument": "empty_byte_string",
        "message_construction": {
            "hash": "sha256",
            "formula": "SHA256(empty_previous_signature||round_u64_big_endian)",
            "preimage_order": ["empty_previous_signature", "round_u64_big_endian"],
            "previous_signature_width_bytes": 0,
            "round_integer_type": "u64",
            "round_encoding": "big_endian",
            "round_width_bytes": 8,
            "message_digest_width_bytes": 32,
            "chain_hash_included": False,
            "public_key_included": False,
            "signature_included": False,
        },
        "public_key_parsing": {
            "constructor": "G2PubkeyRfc::from_fixed",
            "group": "G2",
            "compressed_width_bytes": 96,
            "compressed_parser": "G2Affine::from_compressed",
            "checked": True,
            "unchecked_constructor_allowed": False,
            "result": "Result<G2PubkeyRfc,InvalidPoint>",
        },
        "signature_parsing": {
            "parser": "g1_from_variable_then_G1Affine::from_compressed",
            "group": "G1",
            "compressed_width_bytes": 48,
            "checked": True,
            "unchecked_parser_allowed": False,
        },
        "rfc9380_g1_hash_domain": (
            "BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"
        ),
        "verification_result": "Result<bool,VerificationError>",
        "randomness_function": "derive_randomness",
        "randomness_derivation": "sha256_raw_signature_bytes",
        "required_adapter_outcome_policy": {
            "ok_true": "derive_randomness_then_emit_verified_outcome",
            "ok_false": "reject_without_randomness",
            "err": "reject_without_randomness",
            "err_type": "VerificationError",
            "derive_randomness_only_after_ok_true": True,
            "randomness_before_verification_allowed": False,
            "randomness_on_ok_false_allowed": False,
            "randomness_on_error_allowed": False,
        },
        "upstream_network_api_present": False,
        "upstream_json_api_present": False,
        "caller_must_supply_round_public_key_and_signature": True,
        "semantics_observed_from_pinned_source_only": True,
    }


def test_round_message_and_verification_outcome_order_are_exact_and_fail_closed() -> None:
    semantics = _descriptor()["audited_api_semantics"]
    message = semantics["message_construction"]
    round_1000_preimage = b"" + (1_000).to_bytes(8, byteorder="big", signed=False)
    assert len(round_1000_preimage) == 8
    assert hashlib.sha256(round_1000_preimage).hexdigest() == (
        _descriptor()["official_round_1000_vector"]["message_sha256"]
    )
    assert message["chain_hash_included"] is False

    policy = semantics["required_adapter_outcome_policy"]
    assert policy["ok_true"] == "derive_randomness_then_emit_verified_outcome"
    assert policy["ok_false"] == "reject_without_randomness"
    assert policy["err"] == "reject_without_randomness"
    assert policy["err_type"] == "VerificationError"
    assert policy["derive_randomness_only_after_ok_true"] is True
    assert policy["randomness_before_verification_allowed"] is False
    assert policy["randomness_on_ok_false_allowed"] is False
    assert policy["randomness_on_error_allowed"] is False


def test_quicknet_constants_are_copied_exactly_without_source_import_dependency() -> None:
    quicknet = _descriptor()["quicknet_binding"]
    assert source.QUICKNET_CHAIN_HASH == seed_registry.QUICKNET_CHAIN_HASH
    assert source.QUICKNET_PUBLIC_KEY_HEX == seed_registry.QUICKNET_PUBLIC_KEY_HEX
    assert source.QUICKNET_SIGNATURE_SCHEME == seed_registry.QUICKNET_SIGNATURE_SCHEME
    assert quicknet["chain_hash"] == seed_registry.QUICKNET_CHAIN_HASH
    assert quicknet["public_key_hex"] == seed_registry.QUICKNET_PUBLIC_KEY_HEX
    assert quicknet["signature_scheme"] == seed_registry.QUICKNET_SIGNATURE_SCHEME
    assert quicknet["constants_copied_not_imported_from_seed_registry"] is True
    assert quicknet["chain_hash_in_verification_message"] is False
    assert quicknet["chain_info_independently_authenticated_here"] is False
    assert quicknet["public_key_rotation_authenticated_here"] is False
    assert quicknet["signature_verified_here"] is False


def test_official_round_1000_is_exact_unverified_data_only() -> None:
    vector = _descriptor()["official_round_1000_vector"]
    assert vector["round"] == 1_000
    assert vector["round_time_unix"] == 1_692_806_364
    assert vector["signature_hex"] == source.QUICKNET_ROUND_1000_SIGNATURE_HEX
    assert vector["randomness_hex"] == source.QUICKNET_ROUND_1000_RANDOMNESS_HEX
    assert vector["message_sha256"] == (
        "f652498d092acd949bad74e40683bf3824fb817980504a0c7e6722cfc5a9c0a3"
    )
    assert vector["positive_vector_required_in_future_adapter_suite"] is True
    assert vector["cryptographically_verified_here"] is False
    assert vector["verifier_receipt_emitted"] is False


def test_rustsec_record_is_exactly_point_in_time_and_has_no_future_security_claim() -> None:
    audit = _descriptor()["rustsec_point_in_time_audit"]
    assert audit["tool"] == "cargo-audit"
    assert audit["tool_version"] == "0.22.1"
    assert audit["advisory_database_commit_git_sha1"] == (
        "d91a8fc9492378f23cba86b81770c6d16de6ebba"
    )
    assert audit["advisory_database_tree_git_sha1"] == (
        "35c42e42572140462a3711931db61c4a84cbd350"
    )
    assert audit["scan_date"] == "2026-08-03"
    assert audit["cargo_lock_locked_dependency_count"] == 36
    assert audit["advisories_reported"] == 0
    assert audit["result_wording"] == "no_advisories_reported_at_the_pinned_scan"
    assert audit["point_in_time_only"] is True
    assert audit["future_security_claim_granted"] is False
    assert audit["dependency_source_bytes_vendored_here"] is False
    assert audit["audit_executed_by_this_module"] is False
    assert audit["fresh_pinned_scan_required_before_any_invocation"] is True
    assert audit["stored_scan_may_substitute_for_fresh_scan"] is False
    assert audit["receipt_gaps"] == {
        "versioned_receipt_emitted": False,
        "scanner_binary_digest_bound": False,
        "command_argv_bound": False,
        "cargo_lock_bytes_bound_in_receipt": False,
        "advisory_database_materialization_bound": False,
        "stdout_stderr_bound": False,
        "host_runtime_identity_bound": False,
    }


def test_every_future_integration_requirement_is_explicit_and_unsatisfied() -> None:
    requirements = _descriptor()["future_integration_requirements"]
    assert [record["requirement_id"] for record in requirements] == [
        "primary_crate_materialization_and_exact_inventory",
        "separately_pinned_adapter_source",
        "vendored_locked_dependency_closure",
        "cargo_feature_and_offline_build_policy",
        "compiler_target_runtime_identity",
        "reproducible_build_receipt_and_binary_digest",
        "official_and_negative_quicknet_vectors",
        "authenticated_quicknet_chain_info_and_key_rotation",
        "bounded_canonical_adapter_io_and_outcome",
        "offline_invocation_sandbox",
        "fresh_pinned_rustsec_scan_and_receipt",
        "chronology_and_receipt_integration",
    ]
    assert all(record["required_before_any_invocation"] is True for record in requirements)
    assert all(record["satisfied_by_this_descriptor"] is False for record in requirements)
    vectors = next(
        record
        for record in requirements
        if record["requirement_id"] == "official_and_negative_quicknet_vectors"
    )
    assert vectors["official_round_1000_required"] is True
    assert vectors["negative_vector_bytes_pinned_here"] is False
    assert vectors["negative_classes_required"] == [
        "wrong_round",
        "wrong_signature",
        "malformed_compression",
        "noncanonical_point",
        "invalid_subgroup",
    ]

    by_id = {record["requirement_id"]: record for record in requirements}
    materialization = by_id["primary_crate_materialization_and_exact_inventory"]
    assert materialization["primary_artifact"] == "crates_io_package_drand_verify_0_6_2"
    assert materialization["safe_materialization_receipt_required"] is True
    assert materialization["full_inventory_required"] is True
    assert materialization["relevant_file_subset_is_full_inventory"] is False

    cargo_policy = by_id["cargo_feature_and_offline_build_policy"]
    assert cargo_policy["exact_feature_set_pinned_here"] is False
    assert cargo_policy["default_features_policy_pinned_here"] is False
    assert cargo_policy["required_cargo_flags"] == ["--frozen", "--offline"]
    assert cargo_policy["network_access_allowed"] is False

    chain_info = by_id["authenticated_quicknet_chain_info_and_key_rotation"]
    assert chain_info["copied_constants_are_independent_authentication"] is False
    assert chain_info["key_rotation_policy_pinned_here"] is False
    assert chain_info["chain_info_receipt_available_here"] is False

    adapter_io = by_id["bounded_canonical_adapter_io_and_outcome"]
    assert adapter_io["input_maximum_bytes_pinned_here"] is False
    assert adapter_io["output_maximum_bytes_pinned_here"] is False
    assert adapter_io["canonical_encoding_pinned_here"] is False
    assert adapter_io["outcome_enumeration_pinned_here"] is False
    assert adapter_io["randomness_only_in_verified_outcome_required"] is True


def test_source_is_unbuilt_uninvoked_unqualified_and_all_authority_is_false() -> None:
    descriptor = _descriptor()
    state = descriptor["state"]
    assert state["source_only"] is True
    assert state["primary_crate_materialized"] is False
    assert state["primary_crate_full_inventory_verified"] is False
    assert state["adapter_io_outcome_contract_pinned"] is False
    assert state["cargo_feature_policy_pinned"] is False
    assert state["rust_built"] is False
    assert state["binary_digest_available"] is False
    assert state["verifier_invoked"] is False
    assert state["quicknet_signature_verified"] is False
    assert state["quicknet_chain_info_independently_authenticated"] is False
    assert state["quicknet_key_rotation_policy_pinned"] is False
    assert state["fresh_rustsec_scan_receipt_available"] is False
    assert state["qualification_ready"] is False
    assert descriptor["authority"]
    assert set(descriptor["authority"].values()) == {False}
    assert descriptor["authority"]["chronology_authority_granted"] is False
    assert descriptor["authority"]["seed_issuer_authority_granted"] is False
    assert descriptor["authority"]["trust_root_receipt_issued"] is False
    assert descriptor["authority"]["verifier_receipt_issued"] is False
    assert descriptor["capabilities"]
    assert set(descriptor["capabilities"].values()) == {False}


def test_import_claim_is_narrowly_scoped_to_leaf_body_after_parent_initialization() -> None:
    boundary = _descriptor()["import_boundary"]
    assert boundary["claim_scope"] == (
        "leaf_module_body_after_parent_package_initialization"
    )
    assert boundary["leaf_module_body"] == {
        "constructs_deterministic_in_memory_descriptor_only": True,
        "explicit_filesystem_read": False,
        "explicit_network_read": False,
        "explicit_process_execution": False,
        "explicit_clock_read": False,
        "explicit_environment_read": False,
        "explicit_randomness_read": False,
        "explicit_default_pulse_read": False,
    }
    assert boundary["dotted_import"] == {
        "hermetic": False,
        "parent_packages_initialized_first": True,
        "transitively_executes_parent_package_initializers": True,
        "parent_initializer_behavior_audited_here": False,
        "leaf_no_read_claim_applies_to_parent_initializers": False,
        "dependency_initializer_behavior_audited_here": False,
    }


@pytest.mark.integration
def test_fresh_process_records_dotted_import_and_leaf_body_scope_distinction() -> None:
    module_name = (
        "alberta_framework.benchmarks.forager_matched_v3_quicknet_verifier_source"
    )
    script = "\n".join(
        [
            "import importlib",
            "import json",
            "import sys",
            f"module_name = {module_name!r}",
            "parent_absent_before = 'alberta_framework' not in sys.modules",
            "leaf_absent_before = module_name not in sys.modules",
            "module = importlib.import_module(module_name)",
            "boundary = module.matched_v3_quicknet_verifier_source_descriptor()['import_boundary']",
            "record = {",
            "    'parent_absent_before': parent_absent_before,",
            "    'leaf_absent_before': leaf_absent_before,",
            "    'parent_present_after': 'alberta_framework' in sys.modules,",
            "    'leaf_present_after': module_name in sys.modules,",
            "    'claim_scope': boundary['claim_scope'],",
            "    'dotted_import_hermetic': boundary['dotted_import']['hermetic'],",
            (
                "    'parent_initializers_execute': "
                "boundary['dotted_import']['transitively_executes_parent_package_initializers'],"
            ),
            "}",
            "print(json.dumps(record, sort_keys=True))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    # Parent initializers may emit any number of diagnostics; that is deliberately not asserted.
    record = json.loads(completed.stdout.strip().splitlines()[-1])
    assert record == {
        "parent_absent_before": True,
        "leaf_absent_before": True,
        "parent_present_after": True,
        "leaf_present_after": True,
        "claim_scope": "leaf_module_body_after_parent_package_initialization",
        "dotted_import_hermetic": False,
        "parent_initializers_execute": True,
    }


def test_source_module_has_no_operational_or_seed_registry_imports() -> None:
    module_path = Path(source.__file__)
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
        "hashlib",
        "hmac",
        "json",
        "re",
        "typing",
    }
    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "httpx",
            "os",
            "pathlib",
            "random",
            "requests",
            "secrets",
            "socket",
            "subprocess",
            "time",
            "urllib",
        }
    )
    assert "forager_matched_v3_qualification_seed_registry" not in raw_source


def test_limitations_deny_source_authentication_security_chronology_and_qualification() -> None:
    limitations = " ".join(_descriptor()["limitations"])
    for required_phrase in (
        "leaf-body no-read statement begins after parent package initialization",
        "normal dotted import is non-hermetic and executes package initializers",
        "do not fetch, authenticate, unpack, or verify upstream bytes",
        ".cargo_vcs_info is an upstream declaration and does not authenticate any bytes",
        "not a vendored or independently hashed dependency closure",
        "has no complete receipt",
        "makes no claim about later advisories",
        "fresh pinned scan is required",
        "not independent authentication",
        "have not been cryptographically verified here",
        "not observation or receipt chronology",
        "not a seed issuer, trust receipt, verifier receipt, or authority",
        "Nothing here qualifies",
    ):
        assert required_phrase in limitations
