"""Frozen, nonauthorizing source registry for a future Quicknet verifier.

The descriptor pins one audited release of the Rust ``drand-verify`` crate and
records the narrow API semantics relevant to drand Quicknet.  It is source-only:
this module does not fetch source, build or invoke Rust, verify a signature,
issue a seed, establish chronology, or emit any trust or verifier receipt.

After Python has initialized this module's parent packages, this leaf module
body constructs deterministic in-memory JSON only.  The leaf body has no
explicit filesystem, process, network, clock, environment, randomness, or
default-pulse read.  That narrow statement is not a hermetic dotted-import
claim: importing this dotted name transitively executes parent package
initializers before this leaf body, and their behavior is outside this
descriptor's scope.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from typing import Any, Final, NoReturn, cast

MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_source_descriptor.v2"
)
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_STATUS: Final = (
    "source_only_pinned_unbuilt_uninvoked_nonauthorizing"
)
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_CLASSIFICATION: Final = (
    "detached_source_registry_nonproduction_nonqualifying"
)

UPSTREAM_CANONICAL_REPOSITORY_URL: Final = "https://github.com/CosmWasm/drand-verify"
UPSTREAM_HISTORICAL_REPOSITORY_URL: Final = "https://github.com/noislabs/drand-verify"
UPSTREAM_RELEASE_TAG: Final = "v0.6.2"
UPSTREAM_CRATE_NAME: Final = "drand-verify"
UPSTREAM_CRATE_VERSION: Final = "0.6.2"
UPSTREAM_COMMIT_GIT_SHA1: Final = "1db2248afac44fc2e5c9c78f896b4412d8679914"
UPSTREAM_TREE_GIT_SHA1: Final = "35c957bc3466992194df43fc597014791dd7abe4"
UPSTREAM_COMMIT_ARCHIVE_SHA256: Final = (
    "633408b2d2adca4d9986e765ee2ece148b26de50f7440db5c5f3f7054edfe760"
)
UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES: Final = 18_727
UPSTREAM_CRATE_ARCHIVE_SHA256: Final = (
    "4c1d531704590bbfce3433cd735378d135cabc9e318d8aa52c5dccf7b80178ee"
)
UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES: Final = 18_961

QUICKNET_CHAIN_HASH: Final = (
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
QUICKNET_PUBLIC_KEY_HEX: Final = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183"
    "c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4"
    "bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
QUICKNET_SIGNATURE_SCHEME: Final = "bls-unchained-g1-rfc9380"
QUICKNET_ROUND_1000_SIGNATURE_HEX: Final = (
    "b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125"
    "e342b73a8dd2bacbe47e4b6b63ed5e39"
)
QUICKNET_ROUND_1000_RANDOMNESS_HEX: Final = (
    "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd"
)

RUSTSEC_CARGO_AUDIT_VERSION: Final = "0.22.1"
RUSTSEC_ADVISORY_DB_COMMIT_GIT_SHA1: Final = (
    "d91a8fc9492378f23cba86b81770c6d16de6ebba"
)
RUSTSEC_ADVISORY_DB_TREE_GIT_SHA1: Final = "35c42e42572140462a3711931db61c4a84cbd350"
RUSTSEC_SCAN_DATE: Final = "2026-08-03"
RUSTSEC_LOCKED_DEPENDENCY_COUNT: Final = 36

# The final element names the artifact surface on which the pin was observed.
# ``.cargo_vcs_info.json`` is Cargo's package-generated spelling; the task-level
# shorthand ``.cargo_vcs_info`` is retained as its logical label in the descriptor.
PINNED_RELEVANT_FILE_SHA256: Final = (
    (
        ".cargo_vcs_info.json",
        "f304ef56e003d4cf1c29f052279ffafed2fd83d7a49ffce07377fc66687060fa",
        "crates_io_package",
    ),
    (
        "CHANGELOG.md",
        "f0393d5e35a54a5ad9181307fb4b242912925542bf15636cb817da12dafeab94",
        "release_source",
    ),
    (
        "Cargo.lock",
        "6dd200178128e6e02788b194c856ff3668abf2916b321431790760c950739767",
        "release_source",
    ),
    (
        "Cargo.toml",
        "499d25339dc90d107633cab975404ea1fdac8e03b4f784fa64e5f06dccf04cc1",
        "release_source",
    ),
    (
        "LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "release_source",
    ),
    (
        "MAINTENANCE.md",
        "f6b31cf4f31d718253bfa3a1ddbcc32cf83ff71a745006f7fc6b9cd89c0a5272",
        "release_source",
    ),
    (
        "NOTICE",
        "3322338486638129acbae928b2025eac84395088ff92cdc2f07528e00312ac8e",
        "release_source",
    ),
    (
        "README.md",
        "b45615e1648c152d60bc784b28e9ba0d0ec2618ecc84acef7eb09d7e9618b3a5",
        "release_source",
    ),
    (
        "src/lib.rs",
        "5aab4357c622f089cb0a25825356f6cece18259c42af8aca1069c8708a15b97c",
        "release_source",
    ),
    (
        "src/points.rs",
        "7fa15e818aead5758a44306b304d35c8ff3ab3d9491e8370fa4698546f545eee",
        "release_source",
    ),
    (
        "src/randomness.rs",
        "d66781a3e78b61fe64b3e734b8f159e73926f13dc4badd546498446c1418575b",
        "release_source",
    ),
    (
        "src/verify.rs",
        "47c7a755b4bc226371df83ec8dc430c7a37f0f0ef2bcf55f898cad52e100f9a6",
        "release_source",
    ),
)

_MAX_DESCRIPTOR_BYTES: Final = 256 * 1024
_MAX_JSON_DEPTH: Final = 48
_MAX_JSON_NODES: Final = 20_000
_MAX_JSON_TEXT_LENGTH: Final = 8_192
_MAX_JSON_INTEGER: Final = (1 << 63) - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")


class ForagerMatchedV3QuicknetVerifierSourceError(ValueError):
    """The detached verifier-source descriptor failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QuicknetVerifierSourceError(message)


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"Quicknet verifier-source JSON contains non-finite constant {value!r}")


def _reject_json_float(value: str) -> NoReturn:
    _fail(f"Quicknet verifier-source JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("Quicknet verifier-source JSON integer exceeds its lexical bound")
    parsed = int(value)
    if not -_MAX_JSON_INTEGER <= parsed <= _MAX_JSON_INTEGER:
        _fail("Quicknet verifier-source JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Quicknet verifier-source JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("Quicknet verifier-source JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("Quicknet verifier-source JSON exceeds its depth bound")
        if type(item) is str:
            text = item
            if len(text) > _MAX_JSON_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in text
            ):
                _fail("Quicknet verifier-source JSON strings must be bounded printable ASCII")
            return
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            integer = item
            if not -_MAX_JSON_INTEGER <= integer <= _MAX_JSON_INTEGER:
                _fail("Quicknet verifier-source JSON integer exceeds its value bound")
            return
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("Quicknet verifier-source JSON object keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        _fail("Quicknet verifier-source JSON contains a non-JSON value")

    visit(value, 0)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        _fail("Quicknet verifier-source JSON root must be one plain object")
    _validate_json_tree(value)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QuicknetVerifierSourceError(
            "Quicknet verifier-source descriptor is not canonical ASCII JSON"
        ) from exc
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        _fail("Quicknet verifier-source descriptor exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail("Quicknet verifier-source descriptor must be exact bytes")
    if not raw or len(raw) > _MAX_DESCRIPTOR_BYTES:
        _fail("Quicknet verifier-source descriptor violates its byte bound")
    try:
        decoded = raw.decode("ascii", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMatchedV3QuicknetVerifierSourceError:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QuicknetVerifierSourceError(
            "Quicknet verifier-source descriptor is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("Quicknet verifier-source descriptor root must be one plain object")
    _validate_json_tree(value)
    result = cast(dict[str, Any], value)
    if not hmac.compare_digest(_canonical_json_bytes(result), raw):
        _fail("Quicknet verifier-source descriptor is not canonical")
    return result


def _file_pin_records() -> list[dict[str, Any]]:
    return [
        {
            "artifact_scope": artifact_scope,
            "path": path,
            "sha256": sha256,
        }
        for path, sha256, artifact_scope in PINNED_RELEVANT_FILE_SHA256
    ]


def _future_requirement(
    requirement_id: str,
    detail: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "detail": detail,
        "requirement_id": requirement_id,
        "required_before_any_invocation": True,
        "satisfied_by_this_descriptor": False,
    }
    if extra is not None:
        record.update(extra)
    return record


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "status": MATCHED_V3_QUICKNET_VERIFIER_SOURCE_STATUS,
        "classification": MATCHED_V3_QUICKNET_VERIFIER_SOURCE_CLASSIFICATION,
        "import_boundary": {
            "claim_scope": "leaf_module_body_after_parent_package_initialization",
            "leaf_module_body": {
                "constructs_deterministic_in_memory_descriptor_only": True,
                "explicit_filesystem_read": False,
                "explicit_network_read": False,
                "explicit_process_execution": False,
                "explicit_clock_read": False,
                "explicit_environment_read": False,
                "explicit_randomness_read": False,
                "explicit_default_pulse_read": False,
            },
            "dotted_import": {
                "hermetic": False,
                "parent_packages_initialized_first": True,
                "transitively_executes_parent_package_initializers": True,
                "parent_initializer_behavior_audited_here": False,
                "leaf_no_read_claim_applies_to_parent_initializers": False,
                "dependency_initializer_behavior_audited_here": False,
            },
        },
        "source_identity": {
            "canonical_repository_url": UPSTREAM_CANONICAL_REPOSITORY_URL,
            "historical_cargo_repository_redirect": {
                "historical_url": UPSTREAM_HISTORICAL_REPOSITORY_URL,
                "redirect_target": UPSTREAM_CANONICAL_REPOSITORY_URL,
                "redirect_is_an_authentication_authority": False,
            },
            "release_tag": UPSTREAM_RELEASE_TAG,
            "commit_git_sha1": UPSTREAM_COMMIT_GIT_SHA1,
            "tree_git_sha1": UPSTREAM_TREE_GIT_SHA1,
            "license_spdx": "Apache-2.0",
            "commit_archive": {
                "sha256": UPSTREAM_COMMIT_ARCHIVE_SHA256,
                "size_bytes": UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES,
                "source_bytes_fetched_here": False,
            },
            "crates_io_package": {
                "crate_name": UPSTREAM_CRATE_NAME,
                "version": UPSTREAM_CRATE_VERSION,
                "sha256": UPSTREAM_CRATE_ARCHIVE_SHA256,
                "size_bytes": UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES,
                "cargo_vcs_info_logical_name": ".cargo_vcs_info",
                "cargo_vcs_info_archive_path": ".cargo_vcs_info.json",
                "cargo_vcs_info_sha256": (
                    "f304ef56e003d4cf1c29f052279ffafed2fd83d7a49ffce07377fc66687060fa"
                ),
                "cargo_vcs_info_declared_commit_git_sha1": UPSTREAM_COMMIT_GIT_SHA1,
                "cargo_vcs_info_is_upstream_package_declaration": True,
                "cargo_vcs_info_authenticates_commit": False,
                "cargo_vcs_info_authenticates_crate_bytes": False,
                "package_downloaded_here": False,
            },
            "relevant_file_sha256": _file_pin_records(),
            "relevant_file_count": len(PINNED_RELEVANT_FILE_SHA256),
        },
        "audited_api_semantics": {
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
        },
        "quicknet_binding": {
            "chain_hash": QUICKNET_CHAIN_HASH,
            "public_key_hex": QUICKNET_PUBLIC_KEY_HEX,
            "signature_scheme": QUICKNET_SIGNATURE_SCHEME,
            "message_scope": "unchained_round_only",
            "public_key_group": "G2",
            "signature_group": "G1",
            "randomness_derivation": "sha256_raw_signature_bytes",
            "constants_copied_not_imported_from_seed_registry": True,
            "chain_hash_in_verification_message": False,
            "chain_info_independently_authenticated_here": False,
            "public_key_rotation_authenticated_here": False,
            "signature_verified_here": False,
        },
        "official_round_1000_vector": {
            "round": 1_000,
            "round_time_unix": 1_692_806_364,
            "signature_hex": QUICKNET_ROUND_1000_SIGNATURE_HEX,
            "randomness_hex": QUICKNET_ROUND_1000_RANDOMNESS_HEX,
            "message_sha256": (
                "f652498d092acd949bad74e40683bf3824fb817980504a0c7e6722cfc5a9c0a3"
            ),
            "randomness_derivation": "sha256_raw_signature_bytes",
            "positive_vector_required_in_future_adapter_suite": True,
            "cryptographically_verified_here": False,
            "verifier_receipt_emitted": False,
        },
        "rustsec_point_in_time_audit": {
            "tool": "cargo-audit",
            "tool_version": RUSTSEC_CARGO_AUDIT_VERSION,
            "advisory_database_commit_git_sha1": RUSTSEC_ADVISORY_DB_COMMIT_GIT_SHA1,
            "advisory_database_tree_git_sha1": RUSTSEC_ADVISORY_DB_TREE_GIT_SHA1,
            "scan_date": RUSTSEC_SCAN_DATE,
            "cargo_lock_locked_dependency_count": RUSTSEC_LOCKED_DEPENDENCY_COUNT,
            "advisories_reported": 0,
            "result_wording": "no_advisories_reported_at_the_pinned_scan",
            "point_in_time_only": True,
            "future_security_claim_granted": False,
            "dependency_source_bytes_vendored_here": False,
            "audit_executed_by_this_module": False,
            "fresh_pinned_scan_required_before_any_invocation": True,
            "stored_scan_may_substitute_for_fresh_scan": False,
            "receipt_gaps": {
                "versioned_receipt_emitted": False,
                "scanner_binary_digest_bound": False,
                "command_argv_bound": False,
                "cargo_lock_bytes_bound_in_receipt": False,
                "advisory_database_materialization_bound": False,
                "stdout_stderr_bound": False,
                "host_runtime_identity_bound": False,
            },
        },
        "future_integration_requirements": [
            _future_requirement(
                "primary_crate_materialization_and_exact_inventory",
                (
                    "Materialize the pinned crates.io package safely and bind every extracted "
                    "path, file type, size, mode policy, and digest in an exact inventory."
                ),
                extra={
                    "primary_artifact": "crates_io_package_drand_verify_0_6_2",
                    "safe_materialization_receipt_required": True,
                    "full_inventory_required": True,
                    "relevant_file_subset_is_full_inventory": False,
                },
            ),
            _future_requirement(
                "separately_pinned_adapter_source",
                "Freeze a reviewed adapter that exposes only the required offline operation.",
            ),
            _future_requirement(
                "vendored_locked_dependency_closure",
                (
                    "Vendor and digest every Cargo.lock dependency source byte; "
                    "Cargo.lock alone is insufficient."
                ),
            ),
            _future_requirement(
                "cargo_feature_and_offline_build_policy",
                (
                    "Pin the exact Cargo feature set and default-feature policy, then require "
                    "both --frozen and --offline for every build."
                ),
                extra={
                    "exact_feature_set_pinned_here": False,
                    "default_features_policy_pinned_here": False,
                    "required_cargo_flags": ["--frozen", "--offline"],
                    "network_access_allowed": False,
                },
            ),
            _future_requirement(
                "compiler_target_runtime_identity",
                (
                    "Pin Rust compiler, target, linker, flags, host runtime, "
                    "and CPU execution identity."
                ),
            ),
            _future_requirement(
                "reproducible_build_receipt_and_binary_digest",
                "Require independent reproducibility, a build receipt, and an exact binary digest.",
            ),
            _future_requirement(
                "official_and_negative_quicknet_vectors",
                "Freeze official positive and adversarial negative Quicknet vectors.",
                extra={
                    "official_round_1000_required": True,
                    "negative_classes_required": [
                        "wrong_round",
                        "wrong_signature",
                        "malformed_compression",
                        "noncanonical_point",
                        "invalid_subgroup",
                    ],
                    "negative_vector_bytes_pinned_here": False,
                },
            ),
            _future_requirement(
                "authenticated_quicknet_chain_info_and_key_rotation",
                (
                    "Independently authenticate Quicknet chain-info, public-key identity, "
                    "scheme, and every key-rotation cutover before accepting a pulse."
                ),
                extra={
                    "copied_constants_are_independent_authentication": False,
                    "key_rotation_policy_pinned_here": False,
                    "chain_info_receipt_available_here": False,
                },
            ),
            _future_requirement(
                "bounded_canonical_adapter_io_and_outcome",
                (
                    "Freeze bounded canonical request/outcome schemas and expose randomness "
                    "only in the verified Ok(true) outcome."
                ),
                extra={
                    "input_maximum_bytes_pinned_here": False,
                    "output_maximum_bytes_pinned_here": False,
                    "canonical_encoding_pinned_here": False,
                    "outcome_enumeration_pinned_here": False,
                    "randomness_only_in_verified_outcome_required": True,
                },
            ),
            _future_requirement(
                "offline_invocation_sandbox",
                "Invoke a pinned binary with no network and a bounded filesystem/process surface.",
            ),
            _future_requirement(
                "fresh_pinned_rustsec_scan_and_receipt",
                (
                    "Run a fresh pinned advisory scan over the materialized locked closure "
                    "and bind scanner, database, argv, inputs, outputs, and runtime in a receipt."
                ),
            ),
            _future_requirement(
                "chronology_and_receipt_integration",
                (
                    "Bind verification to independent preacceptance chronology "
                    "and versioned trust/verifier receipts."
                ),
            ),
        ],
        "state": {
            "source_only": True,
            "source_fetched_by_module": False,
            "primary_crate_materialized": False,
            "primary_crate_full_inventory_verified": False,
            "dependency_closure_vendored": False,
            "adapter_implemented": False,
            "adapter_io_outcome_contract_pinned": False,
            "cargo_feature_policy_pinned": False,
            "rust_built": False,
            "binary_digest_available": False,
            "verifier_invoked": False,
            "quicknet_signature_verified": False,
            "quicknet_chain_info_independently_authenticated": False,
            "quicknet_key_rotation_policy_pinned": False,
            "fresh_rustsec_scan_receipt_available": False,
            "offline_sandbox_available": False,
            "chronology_integration_available": False,
            "qualification_ready": False,
        },
        "authority": {
            "acceptance_authority_granted": False,
            "build_authority_granted": False,
            "chronology_authority_granted": False,
            "execution_authority_granted": False,
            "network_authority_granted": False,
            "publication_authority_granted": False,
            "qualification_authority_granted": False,
            "scientific_evidence_authority_granted": False,
            "seed_issuer_authority_granted": False,
            "trust_root_authority_granted": False,
            "trust_root_receipt_issued": False,
            "verifier_receipt_issued": False,
            "promotion_authority_granted": False,
            "performance_or_sota_claim_authority_granted": False,
        },
        "capabilities": {
            "filesystem_api_exposed": False,
            "network_api_exposed": False,
            "process_api_exposed": False,
            "clock_api_exposed": False,
            "randomness_api_exposed": False,
            "default_pulse_exposed": False,
            "source_inventory_verifier_exposed": False,
            "bls_signature_verifier_exposed": False,
            "seed_issuer_api_exposed": False,
            "chronology_acceptor_exposed": False,
            "receipt_issuer_exposed": False,
        },
        "limitations": [
            (
                "The leaf-body no-read statement begins after parent package initialization; "
                "a normal dotted import is non-hermetic and executes package initializers."
            ),
            "Static hashes do not fetch, authenticate, unpack, or verify upstream bytes.",
            ".cargo_vcs_info is an upstream declaration and does not authenticate any bytes.",
            "A Cargo.lock pin is not a vendored or independently hashed dependency closure.",
            (
                "The RustSec result has no complete receipt, is historical, and makes no "
                "claim about later advisories; a fresh pinned scan is required."
            ),
            "Copied Quicknet chain-info and key bytes are not independent authentication.",
            "No compiler, target, linker, runtime, reproducible build, or binary is identified.",
            "No bounded canonical adapter request or outcome contract is identified.",
            "The recorded API semantics have not been executed by this module.",
            "The round-1000 bytes are data only and have not been cryptographically verified here.",
            "Quicknet authenticates an unchained round, not observation or receipt chronology.",
            "This descriptor is not a seed issuer, trust receipt, verifier receipt, or authority.",
            (
                "Nothing here qualifies a source, binary, seed, run, result, "
                "evidence artifact, or claim."
            ),
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json_bytes(_descriptor())
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256: Final = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 Quicknet verifier-source descriptor drifted")


def matched_v3_quicknet_verifier_source_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen, authority-denying descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_quicknet_verifier_source_descriptor_bytes() -> bytes:
    """Return the exact canonical source-registry descriptor bytes."""

    return _DESCRIPTOR_BYTES


def parse_matched_v3_quicknet_verifier_source_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen descriptor without granting authority."""

    value = _strict_json_load(raw)
    if not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        _fail("Quicknet verifier-source descriptor identity drifted")
    return value


__all__ = [
    "ForagerMatchedV3QuicknetVerifierSourceError",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_CLASSIFICATION",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_STATUS",
    "PINNED_RELEVANT_FILE_SHA256",
    "QUICKNET_CHAIN_HASH",
    "QUICKNET_PUBLIC_KEY_HEX",
    "QUICKNET_ROUND_1000_RANDOMNESS_HEX",
    "QUICKNET_ROUND_1000_SIGNATURE_HEX",
    "QUICKNET_SIGNATURE_SCHEME",
    "RUSTSEC_ADVISORY_DB_COMMIT_GIT_SHA1",
    "RUSTSEC_ADVISORY_DB_TREE_GIT_SHA1",
    "RUSTSEC_CARGO_AUDIT_VERSION",
    "RUSTSEC_LOCKED_DEPENDENCY_COUNT",
    "RUSTSEC_SCAN_DATE",
    "UPSTREAM_CANONICAL_REPOSITORY_URL",
    "UPSTREAM_COMMIT_ARCHIVE_SHA256",
    "UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES",
    "UPSTREAM_COMMIT_GIT_SHA1",
    "UPSTREAM_CRATE_ARCHIVE_SHA256",
    "UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES",
    "UPSTREAM_CRATE_NAME",
    "UPSTREAM_CRATE_VERSION",
    "UPSTREAM_HISTORICAL_REPOSITORY_URL",
    "UPSTREAM_RELEASE_TAG",
    "UPSTREAM_TREE_GIT_SHA1",
    "canonical_matched_v3_quicknet_verifier_source_descriptor_bytes",
    "matched_v3_quicknet_verifier_source_descriptor",
    "parse_matched_v3_quicknet_verifier_source_descriptor",
]
