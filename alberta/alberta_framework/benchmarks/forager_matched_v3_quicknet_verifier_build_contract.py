"""Detached build and wire contract for a future Quicknet verifier.

This module freezes prerequisite identities, future artifact schemas, a
reproducible-build/OCI-v2 pin graph, and the exact binary request/outcome wire
formats for a future Rust adapter.  It does not materialize a filesystem tree,
vendor dependencies, scan advisories, build or invoke Rust, inspect a binary,
construct an image, verify a signature, issue a receipt, or grant authority.

The only operational surface is pure in-memory framing and parsing of
caller-supplied bytes.  After parent package initialization, this leaf module
has no explicit filesystem, process, network, clock, environment, randomness,
dynamic-library, or Rust API.  Normal dotted import still executes parent
package initializers first; their behavior is outside this leaf-body claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_build_contract_descriptor.v1"
)
MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_STATUS: Final = (
    "source_only_nonexecuting_build_and_wire_contract_nonauthorizing"
)

QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_source_descriptor.v2"
)
QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SHA256: Final = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)
QUICKNET_SOURCE_REGISTRY_SOURCE_SHA256: Final = (
    "3e13009c1843c3341e5a0eb8b2f84ea903b8e5315fbdef347549757710fd3623"
)

QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_descriptor.v1"
)
QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256: Final = (
    "61345825673afb16bc1942c4b8c84e763fb14530a68225caffa94d98e733a03d"
)
QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_plan.v1"
)
QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256: Final = (
    "5ccbed13f70ed15355c6849d732801db6e372864ac570fc62f4acf4a78cde0e7"
)
QUICKNET_SOURCE_MATERIALIZATION_SOURCE_SHA256: Final = (
    "1e08a04b8c3120978867999b5316d57ac5361771b018496535a5ab5a77a61023"
)
QUICKNET_SOURCE_MATERIALIZATION_TEST_SHA256: Final = (
    "6aa06c38345d833cbd6e716469ea1871fa249ea4da321648ffc283c66c423f69"
)
QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_manifest.v1"
)
QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_receipt.v1"
)

QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_tree_materialization_receipt.v1"
)
QUICKNET_CARGO_VENDOR_CLOSURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_cargo_vendor_closure.v1"
)
QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_adapter_descriptor.v1"
)
QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_runtime_verifier_descriptor.v1"
)
QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_request.v1"
)
QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_outcome.v1"
)
QUICKNET_RUST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_rust_toolchain_manifest.v1"
)
QUICKNET_RUSTSEC_AUDIT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_rustsec_audit_receipt.v1"
)
QUICKNET_VERIFIER_BUILD_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_build_plan.v1"
)
QUICKNET_VERIFIER_BUILD_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_build_receipt.v1"
)
QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_reproducibility_receipt.v1"
)
QUICKNET_VERIFIER_VECTOR_SUITE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_vector_suite.v1"
)
QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_vector_run_receipt.v1"
)
QUICKNET_VERIFIER_OCI_BUILD_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_oci_build_plan.v2"
)
QUICKNET_VERIFIER_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_oci_build_context_receipt.v2"
)
QUICKNET_VERIFIER_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_oci_build_execution_receipt.v2"
)
QUICKNET_VERIFIER_OCI_BUILD_PUBLICATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_oci_build_publication.v2"
)
QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_final_image_identity.v1"
)
QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_final_image_vector_run_receipt.v1"
)
QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_receipt.v1"
)

QUICKNET_VERIFIER_REQUEST_MAGIC: Final = b"ALBQNV1\0"
QUICKNET_VERIFIER_OUTCOME_MAGIC: Final = b"ALBQNO1\0"
QUICKNET_VERIFIER_REQUEST_SIZE_BYTES: Final = 160
QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES: Final = 9
QUICKNET_VERIFIER_OUTCOME_VERIFIED_SIZE_BYTES: Final = 41
QUICKNET_VERIFIER_PUBLIC_KEY_SIZE_BYTES: Final = 96
QUICKNET_VERIFIER_SIGNATURE_SIZE_BYTES: Final = 48
QUICKNET_VERIFIER_RANDOMNESS_SIZE_BYTES: Final = 32
QUICKNET_VERIFIER_ROUND_MAXIMUM: Final = (1 << 64) - 1

QUICKNET_VERIFIER_OUTCOME_TAG_VERIFIED: Final = 0x00
QUICKNET_VERIFIER_OUTCOME_TAG_OK_FALSE: Final = 0x01
QUICKNET_VERIFIER_OUTCOME_TAG_VERIFICATION_ERROR: Final = 0x02
QUICKNET_VERIFIER_OUTCOME_TAG_INVALID_PUBLIC_KEY: Final = 0x03

_OUTCOME_NAMES: Final = {
    QUICKNET_VERIFIER_OUTCOME_TAG_VERIFIED: "verified",
    QUICKNET_VERIFIER_OUTCOME_TAG_OK_FALSE: "ok_false",
    QUICKNET_VERIFIER_OUTCOME_TAG_VERIFICATION_ERROR: "verification_error",
    QUICKNET_VERIFIER_OUTCOME_TAG_INVALID_PUBLIC_KEY: "invalid_public_key",
}
_CARGO_LOCK_SHA256: Final = "6dd200178128e6e02788b194c856ff3668abf2916b321431790760c950739767"
_LOCKED_DEPENDENCY_COUNT: Final = 36
_FINAL_BINARY_CONTEXT_PATH: Final = "inputs/alberta-quicknet-verify"
_FINAL_BINARY_IMAGE_PATH: Final = "/opt/elizaos/bin/alberta-quicknet-verify"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MAX_DESCRIPTOR_BYTES: Final = 256 * 1024
_MAX_JSON_DEPTH: Final = 48
_MAX_JSON_NODES: Final = 40_000
_MAX_JSON_TEXT_LENGTH: Final = 16_384
_MAX_JSON_INTEGER: Final = (1 << 64) - 1


class ForagerMatchedV3QuicknetVerifierBuildContractError(ValueError):
    """The detached Quicknet build or wire contract failed closed."""


@dataclass(frozen=True)
class MatchedV3QuicknetVerifierRequest:
    """One structurally valid request; no cryptographic validity is implied."""

    round_number: int
    public_key: bytes
    signature: bytes


@dataclass(frozen=True)
class MatchedV3QuicknetVerifierOutcome:
    """One structurally valid adapter claim; no verifier truth is authenticated."""

    outcome: str
    outcome_tag: int
    randomness: bytes | None


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QuicknetVerifierBuildContractError(message)


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"Quicknet build-contract JSON contains non-finite constant {value!r}")


def _reject_json_float(value: str) -> NoReturn:
    _fail(f"Quicknet build-contract JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        _fail("Quicknet build-contract JSON integer exceeds its lexical bound")
    parsed = int(value)
    if not 0 <= parsed <= _MAX_JSON_INTEGER:
        _fail("Quicknet build-contract JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Quicknet build-contract JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("Quicknet build-contract JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("Quicknet build-contract JSON exceeds its depth bound")
        if type(item) is str:
            if len(item) > _MAX_JSON_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("Quicknet build-contract JSON strings must be bounded printable ASCII")
            return
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not 0 <= item <= _MAX_JSON_INTEGER:
                _fail("Quicknet build-contract JSON integer exceeds its value bound")
            return
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("Quicknet build-contract JSON object keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        _fail("Quicknet build-contract JSON contains a non-JSON value")

    visit(value, 0)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        _fail("Quicknet build-contract JSON root must be one plain object")
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
        raise ForagerMatchedV3QuicknetVerifierBuildContractError(
            "Quicknet build-contract descriptor is not canonical ASCII JSON"
        ) from exc
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        _fail("Quicknet build-contract descriptor exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail("Quicknet build-contract descriptor must be exact bytes")
    if not raw or len(raw) > _MAX_DESCRIPTOR_BYTES:
        _fail("Quicknet build-contract descriptor violates its byte bound")
    try:
        decoded = raw.decode("ascii", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMatchedV3QuicknetVerifierBuildContractError:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QuicknetVerifierBuildContractError(
            "Quicknet build-contract descriptor is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("Quicknet build-contract descriptor root must be one plain object")
    result = cast(dict[str, Any], value)
    _validate_json_tree(result)
    if not hmac.compare_digest(_canonical_json_bytes(result), raw):
        _fail("Quicknet build-contract descriptor is not canonical")
    return result


def _denied_authority() -> dict[str, bool]:
    return {
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
        "promotion_authority_granted": False,
        "performance_or_sota_claim_authority_granted": False,
    }


def _artifact_schemas() -> dict[str, str]:
    return {
        "adapter_descriptor": QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
        "build_plan": QUICKNET_VERIFIER_BUILD_PLAN_SCHEMA_VERSION,
        "build_receipt": QUICKNET_VERIFIER_BUILD_RECEIPT_SCHEMA_VERSION,
        "cargo_vendor_closure": QUICKNET_CARGO_VENDOR_CLOSURE_SCHEMA_VERSION,
        "final_image_identity": QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
        "final_image_vector_run_receipt": (
            QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION
        ),
        "oci_build_context_receipt_v2": (
            QUICKNET_VERIFIER_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION
        ),
        "oci_build_execution_receipt_v2": (
            QUICKNET_VERIFIER_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION
        ),
        "oci_build_plan_v2": QUICKNET_VERIFIER_OCI_BUILD_PLAN_SCHEMA_VERSION,
        "oci_build_publication_v2": QUICKNET_VERIFIER_OCI_BUILD_PUBLICATION_SCHEMA_VERSION,
        "reproducibility_receipt": (QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION),
        "request": QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
        "runtime_verifier_descriptor": QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION,
        "rust_toolchain_manifest": QUICKNET_RUST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
        "rustsec_audit_receipt": QUICKNET_RUSTSEC_AUDIT_RECEIPT_SCHEMA_VERSION,
        "source_tree_materialization_receipt": (
            QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION
        ),
        "verifier_outcome": QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
        "verifier_receipt": QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION,
        "vector_run_receipt": QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
        "vector_suite": QUICKNET_VERIFIER_VECTOR_SUITE_SCHEMA_VERSION,
    }


def _source_prerequisites() -> dict[str, Any]:
    return {
        "verifier_source_registry": {
            "descriptor_schema_version": QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SHA256,
            "source_sha256": QUICKNET_SOURCE_REGISTRY_SOURCE_SHA256,
            "imported_or_executed_here": False,
        },
        "source_materialization_contract": {
            "descriptor_schema_version": (
                QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256,
            "plan_schema_version": QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
            "plan_sha256": QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256,
            "source_sha256": QUICKNET_SOURCE_MATERIALIZATION_SOURCE_SHA256,
            "test_sha256": QUICKNET_SOURCE_MATERIALIZATION_TEST_SHA256,
            "manifest_schema_version": (QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION),
            "receipt_schema_version": QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            "contract_imported_or_executed_here": False,
            "manifest_or_receipt_instance_bound_here": False,
            "caller_supplied_archives_inventoried_in_memory_only": True,
            "filesystem_tree_materialized": False,
            "archive_member_extracted_to_path": False,
            "dependency_vendor_closure_provided": False,
            "build_input_selected": False,
            "build_or_runtime_authority_granted": False,
            "trust_or_chronology_authority_granted": False,
        },
    }


def _wire_contract() -> dict[str, Any]:
    return {
        "request": {
            "schema_version": QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
            "encoding": "fixed_width_binary",
            "total_size_bytes": QUICKNET_VERIFIER_REQUEST_SIZE_BYTES,
            "magic_ascii_with_terminal_nul": "ALBQNV1\\0",
            "fields": [
                {
                    "name": "magic",
                    "offset_bytes": 0,
                    "size_bytes": 8,
                    "encoding": "exact_bytes",
                },
                {
                    "name": "round",
                    "offset_bytes": 8,
                    "size_bytes": 8,
                    "encoding": "u64_big_endian",
                },
                {
                    "name": "public_key",
                    "offset_bytes": 16,
                    "size_bytes": 96,
                    "encoding": "compressed_bls12_381_g2_candidate_bytes",
                },
                {
                    "name": "signature",
                    "offset_bytes": 112,
                    "size_bytes": 48,
                    "encoding": "compressed_bls12_381_g1_candidate_bytes",
                },
            ],
            "round_minimum": 0,
            "round_maximum": QUICKNET_VERIFIER_ROUND_MAXIMUM,
            "structural_parser_validates_curve_points": False,
            "trailing_bytes_allowed": False,
        },
        "outcome": {
            "schema_version": QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
            "encoding": "fixed_magic_tag_and_tag_dependent_payload",
            "magic_ascii_with_terminal_nul": "ALBQNO1\\0",
            "magic_size_bytes": 8,
            "tag_offset_bytes": 8,
            "tags": [
                {
                    "name": "verified",
                    "tag": QUICKNET_VERIFIER_OUTCOME_TAG_VERIFIED,
                    "total_size_bytes": QUICKNET_VERIFIER_OUTCOME_VERIFIED_SIZE_BYTES,
                    "payload": "sha256_raw_request_signature_bytes",
                    "payload_size_bytes": QUICKNET_VERIFIER_RANDOMNESS_SIZE_BYTES,
                },
                {
                    "name": "ok_false",
                    "tag": QUICKNET_VERIFIER_OUTCOME_TAG_OK_FALSE,
                    "total_size_bytes": QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES,
                    "payload": "none",
                    "payload_size_bytes": 0,
                },
                {
                    "name": "verification_error",
                    "tag": QUICKNET_VERIFIER_OUTCOME_TAG_VERIFICATION_ERROR,
                    "total_size_bytes": QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES,
                    "payload": "none",
                    "payload_size_bytes": 0,
                },
                {
                    "name": "invalid_public_key",
                    "tag": QUICKNET_VERIFIER_OUTCOME_TAG_INVALID_PUBLIC_KEY,
                    "total_size_bytes": QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES,
                    "payload": "none",
                    "payload_size_bytes": 0,
                },
            ],
            "randomness_derived_only_after_upstream_ok_true": True,
            "randomness_on_ok_false_allowed": False,
            "randomness_on_error_allowed": False,
            "free_form_error_payload_allowed": False,
            "trailing_bytes_allowed": False,
            "structural_parser_authenticates_verifier_truth": False,
        },
        "future_process_protocol": {
            "argv_argument_count": 0,
            "stdin_request_count": 1,
            "stdin_eof_required_after_request": True,
            "stdout_outcome_count": 1,
            "stderr_must_be_empty_for_typed_outcomes": True,
            "typed_outcome_exit_code": 0,
            "framing_or_internal_error_exit_code_nonzero": True,
            "accepted_outcome_on_nonzero_exit_allowed": False,
            "file_input_allowed": False,
            "network_input_allowed": False,
            "clock_input_allowed": False,
            "environment_input_allowed": False,
            "default_pulse_input_allowed": False,
            "child_process_allowed": False,
        },
    }


def _future_build_contract() -> dict[str, Any]:
    return {
        "all_artifacts_absent_here": True,
        "source_tree_materialization": {
            "schema_version": QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            "required_inputs": [
                QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
                QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            ],
            "must_replay_exact_caller_archive_bytes": True,
            "must_materialize_to_separate_immutable_tree": True,
            "satisfied_here": False,
        },
        "cargo_vendor_closure": {
            "schema_version": QUICKNET_CARGO_VENDOR_CLOSURE_SCHEMA_VERSION,
            "cargo_lock_sha256": _CARGO_LOCK_SHA256,
            "locked_dependency_count": _LOCKED_DEPENDENCY_COUNT,
            "all_locked_crate_archives_and_full_inventories_required": True,
            "every_cargo_checksum_json_required": True,
            "dependency_and_target_feature_graph_required": True,
            "exact_default_feature_policy_required": True,
            "offline_cargo_configuration_required": True,
            "path_or_git_dependencies_allowed": False,
            "filesystem_tree_or_vendor_closure_available_here": False,
            "satisfied_here": False,
        },
        "adapter_source": {
            "schema_version": QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "separate_source_bytes_and_sha256_required": True,
            "producer_identity_requires_descriptor_and_source_sha256": True,
            "wire_contract_sha256_required": True,
            "source_implemented_here": False,
            "satisfied_here": False,
        },
        "toolchain": {
            "schema_version": QUICKNET_RUST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
            "rustc_cargo_target_linker_and_runtime_bytes_required": True,
            "exact_versions_sizes_sha256_and_argv_required": True,
            "target": "x86_64_unknown_linux_gnu_elf64",
            "toolchain_available_here": False,
            "satisfied_here": False,
        },
        "fresh_rustsec": {
            "schema_version": QUICKNET_RUSTSEC_AUDIT_RECEIPT_SCHEMA_VERSION,
            "fresh_scan_required": True,
            "historical_source_registry_scan_may_substitute": False,
            "scanner_binary_database_archive_inventory_commit_argv_bound": True,
            "cargo_lock_vendor_stdout_stderr_runtime_bound": True,
            "network_during_scan_allowed": False,
            "receipt_available_here": False,
            "satisfied_here": False,
        },
        "offline_build": {
            "plan_schema_version": QUICKNET_VERIFIER_BUILD_PLAN_SCHEMA_VERSION,
            "receipt_schema_version": QUICKNET_VERIFIER_BUILD_RECEIPT_SCHEMA_VERSION,
            "required_cargo_flags": ["--frozen", "--offline"],
            "clean_builder_count": 2,
            "distinct_builder_identity_required": True,
            "shared_target_directory_allowed": False,
            "writable_source_or_vendor_allowed": False,
            "network_during_build_allowed": False,
            "build_executed_here": False,
            "satisfied_here": False,
        },
        "reproducibility": {
            "schema_version": QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
            "two_distinct_build_receipts_required": True,
            "same_plan_toolchain_source_vendor_and_adapter_required": True,
            "byte_identical_binary_required": True,
            "binary_sha256_size_and_elf_identity_required": True,
            "binary_may_embed_its_own_sha256": False,
            "reproduced_binary_available_here": False,
            "satisfied_here": False,
        },
        "standalone_vectors": {
            "suite_schema_version": QUICKNET_VERIFIER_VECTOR_SUITE_SCHEMA_VERSION,
            "run_receipt_schema_version": QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
            "official_round_1000_positive_required": True,
            "negative_classes_required": [
                "wrong_round",
                "wrong_signature",
                "malformed_compression",
                "noncanonical_point",
                "invalid_subgroup",
                "ok_false_randomness_leakage",
                "error_randomness_leakage",
            ],
            "network_during_vector_run_allowed": False,
            "suite_or_receipt_available_here": False,
            "satisfied_here": False,
        },
    }


def _future_oci_v2_contract() -> dict[str, Any]:
    return {
        "fresh_lineage_required": True,
        "schemas": {
            "plan": QUICKNET_VERIFIER_OCI_BUILD_PLAN_SCHEMA_VERSION,
            "context_receipt": QUICKNET_VERIFIER_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
            "execution_receipt": QUICKNET_VERIFIER_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "publication": QUICKNET_VERIFIER_OCI_BUILD_PUBLICATION_SCHEMA_VERSION,
            "final_image_identity": QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
            "final_image_vector_run_receipt": (
                QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION
            ),
        },
        "historical_cpu_oci_v1_plan_accepted": False,
        "historical_twelve_member_context_extended_or_reinterpreted": False,
        "binary_copy": {
            "source": "exact_reproducibility_receipt_binary",
            "context_path": _FINAL_BINARY_CONTEXT_PATH,
            "image_path": _FINAL_BINARY_IMAGE_PATH,
            "mode": "0555",
            "uid": 0,
            "gid": 0,
            "reproducibility_binary_sha256_and_size_must_match_context_member": True,
            "context_member_sha256_and_size_must_match_final_image_file": True,
            "final_image_file_sha256_must_be_rechecked_before_runtime_use": True,
            "elf_identity_must_match_at_every_boundary": True,
            "post_hash_binary_substitution_allowed": False,
        },
        "rust_build_inside_final_image_allowed": False,
        "rustc_or_cargo_in_final_image_allowed": False,
        "source_or_vendor_tree_in_final_image_allowed": False,
        "dockerfile_rust_build_step_allowed": False,
        "network_during_oci_build_allowed": False,
        "pull_during_oci_build_allowed": False,
        "exact_base_image_digest_required": True,
        "exact_runtime_loader_and_library_identity_required": True,
        "final_image_vector_replay_required": True,
        "final_image_vector_replay_must_use_copied_binary": True,
        "final_image_vector_replay_network_allowed": False,
        "final_image_or_receipt_available_here": False,
        "satisfied_here": False,
    }


def _future_runtime_verifier_descriptor_contract() -> dict[str, Any]:
    wire_contract = _wire_contract()
    return {
        "schema_version": QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION,
        "role": "production_quicknet_runtime_verifier_producer_identity",
        "adapter_binding": {
            "schema_version": QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_full_sha256_required": True,
            "adapter_source_sha256_required": True,
        },
        "reproduced_binary_binding": {
            "schema_version": QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
            "receipt_full_and_body_sha256_required": True,
            "exact_binary_sha256_required": True,
            "exact_binary_size_bytes_required": True,
            "exact_elf_identity_required": True,
        },
        "final_image_binding": {
            "schema_version": QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
            "identity_full_and_body_sha256_required": True,
            "fresh_oci_v2_plan_context_execution_publication_chain_required": True,
            "exact_image_digest_required": True,
            "in_image_binary_sha256_size_and_elf_match_required": True,
        },
        "vector_binding": {
            "standalone_schema_version": QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
            "final_image_schema_version": (
                QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION
            ),
            "both_receipt_full_and_body_sha256_required": True,
            "both_receipts_must_bind_same_vector_suite": True,
            "both_receipts_must_bind_same_reproduced_binary": True,
        },
        "wire_and_process_binding": {
            "request_schema_version": QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
            "outcome_schema_version": QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
            "wire_contract_canonical_sha256": hashlib.sha256(
                _canonical_json_bytes(wire_contract)
            ).hexdigest(),
            "exact_wire_contract_projection_required": True,
            "exact_process_protocol_projection_required": True,
            "build_contract_descriptor_full_sha256_required": True,
        },
        "producer_source_sha256_required": True,
        "descriptor_issued_here": False,
        "producer_available_here": False,
        "satisfied_here": False,
        "authority": _denied_authority(),
    }


def _future_pin_graph() -> list[dict[str, Any]]:
    return [
        {
            "stage": "source_tree_materialization",
            "inputs": [
                QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SCHEMA_VERSION,
                QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
                QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            ],
            "output": QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "cargo_vendor_closure",
            "inputs": [
                QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
                "exact_cargo_lock_and_all_36_locked_crate_archives",
            ],
            "output": QUICKNET_CARGO_VENDOR_CLOSURE_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "adapter_source",
            "inputs": [
                QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SCHEMA_VERSION,
                QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
                QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
            ],
            "output": QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "offline_build_plan",
            "inputs": [
                QUICKNET_SOURCE_TREE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
                QUICKNET_CARGO_VENDOR_CLOSURE_SCHEMA_VERSION,
                QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
                QUICKNET_RUST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
                QUICKNET_RUSTSEC_AUDIT_RECEIPT_SCHEMA_VERSION,
            ],
            "output": QUICKNET_VERIFIER_BUILD_PLAN_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "independent_builds",
            "inputs": [QUICKNET_VERIFIER_BUILD_PLAN_SCHEMA_VERSION],
            "output": "two_distinct_quicknet_verifier_build_receipt_v1_artifacts",
            "satisfied_here": False,
        },
        {
            "stage": "binary_reproducibility",
            "inputs": ["two_distinct_quicknet_verifier_build_receipt_v1_artifacts"],
            "output": QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "standalone_vector_run",
            "inputs": [
                QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_VECTOR_SUITE_SCHEMA_VERSION,
            ],
            "output": QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "fresh_oci_v2_lineage",
            "inputs": [
                QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
                "fresh_cpu_runtime_and_source_inputs",
            ],
            "output": QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "final_image_vector_run",
            "inputs": [
                QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
                QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_VECTOR_SUITE_SCHEMA_VERSION,
            ],
            "output": QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "runtime_verifier_descriptor",
            "inputs": [
                QUICKNET_VERIFIER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
                QUICKNET_VERIFIER_REPRODUCIBILITY_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_FINAL_IMAGE_IDENTITY_SCHEMA_VERSION,
                QUICKNET_VERIFIER_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_FINAL_IMAGE_VECTOR_RUN_RECEIPT_SCHEMA_VERSION,
                QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
                QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
            ],
            "output": QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION,
            "satisfied_here": False,
        },
        {
            "stage": "future_verifier_receipt",
            "inputs": [
                QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION,
                "separately_authenticated_trust_policy_and_pulse",
                QUICKNET_VERIFIER_REQUEST_SCHEMA_VERSION,
                QUICKNET_VERIFIER_OUTCOME_SCHEMA_VERSION,
            ],
            "output": QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION,
            "satisfied_here": False,
        },
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": (MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SCHEMA_VERSION),
        "status": MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_STATUS,
        "classification": "detached_source_only_future_build_contract_nonqualifying",
        "import_boundary": {
            "claim_scope": "leaf_module_body_after_parent_package_initialization",
            "leaf_module_body_has_explicit_filesystem_api": False,
            "leaf_module_body_has_explicit_process_api": False,
            "leaf_module_body_has_explicit_network_api": False,
            "leaf_module_body_has_explicit_clock_api": False,
            "leaf_module_body_has_explicit_environment_api": False,
            "leaf_module_body_has_explicit_randomness_api": False,
            "leaf_module_body_has_explicit_dynamic_library_api": False,
            "leaf_module_body_has_explicit_rust_api": False,
            "dotted_import_is_hermetic": False,
            "parent_packages_initialize_first": True,
            "parent_initializer_behavior_audited_here": False,
        },
        "source_prerequisites": _source_prerequisites(),
        "artifact_schemas": _artifact_schemas(),
        "wire_contract": _wire_contract(),
        "future_build_contract": _future_build_contract(),
        "future_oci_v2_contract": _future_oci_v2_contract(),
        "future_runtime_verifier_descriptor": (_future_runtime_verifier_descriptor_contract()),
        "future_pin_graph": _future_pin_graph(),
        "future_verifier_receipt": {
            "schema_version": QUICKNET_VERIFIER_RECEIPT_SCHEMA_VERSION,
            "must_bind_runtime_verifier_descriptor_full_and_body_sha256": True,
            "must_bind_runtime_verifier_producer_source_sha256": True,
            "must_bind_exact_request_and_outcome_bytes": True,
            "must_bind_separate_trust_policy_and_pulse": True,
            "does_not_authenticate_seed_registry_or_chronology": True,
            "issued_here": False,
        },
        "state": {
            "source_registry_identity_bound": True,
            "source_materialization_contract_identity_bound": True,
            "wire_format_frozen": True,
            "filesystem_source_tree_materialized": False,
            "dependency_vendor_closure_available": False,
            "adapter_source_available": False,
            "toolchain_available": False,
            "fresh_rustsec_receipt_available": False,
            "build_plan_issued": False,
            "rust_build_executed": False,
            "reproduced_binary_available": False,
            "standalone_vectors_executed": False,
            "oci_v2_plan_issued": False,
            "final_image_available": False,
            "final_image_vectors_executed": False,
            "runtime_verifier_descriptor_issued": False,
            "runtime_verifier_producer_available": False,
            "verifier_invoked": False,
            "verifier_receipt_issued": False,
            "build_ready": False,
            "runtime_ready": False,
            "qualification_ready": False,
        },
        "authority": _denied_authority(),
        "capabilities": {
            "request_encoder_api_exposed": True,
            "request_parser_api_exposed": True,
            "outcome_parser_api_exposed": True,
            "outcome_issuer_api_exposed": False,
            "filesystem_api_exposed": False,
            "process_api_exposed": False,
            "network_api_exposed": False,
            "rust_api_exposed": False,
            "dynamic_library_api_exposed": False,
            "build_executor_api_exposed": False,
            "oci_executor_api_exposed": False,
            "verifier_api_exposed": False,
            "receipt_issuer_api_exposed": False,
            "trust_root_api_exposed": False,
            "chronology_api_exposed": False,
            "seed_issuer_api_exposed": False,
        },
        "limitations": [
            (
                "Prerequisite 1 inventories caller bytes in memory and materializes "
                "no filesystem tree."
            ),
            "No concrete prerequisite-1 manifest or receipt instance is accepted by this module.",
            "The source inventory is not a Cargo dependency vendor closure or build input.",
            "Wire parsing validates framing and a claimed randomness relation, not BLS truth.",
            (
                "All future source, vendor, toolchain, audit, build, vector, "
                "and OCI artifacts are absent."
            ),
            "The final image must copy a separately reproduced binary and must not build Rust.",
            (
                "The production runtime-verifier descriptor is a future artifact distinct "
                "from the earlier adapter-source descriptor."
            ),
            "Historical CPU OCI-v1 plans and contexts cannot be extended or reinterpreted.",
            (
                "No structure in this descriptor grants readiness, authority, "
                "evidence, or qualification."
            ),
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json_bytes(_descriptor())
MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "513ed21d7411f65a3d605f38eedc6da5cf6d6764203ebcf38210439a4724cb87"
)


def matched_v3_quicknet_verifier_build_contract_descriptor() -> dict[str, Any]:
    """Return a detached copy of the nonexecuting build and wire contract."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_quicknet_verifier_build_contract_descriptor_bytes() -> bytes:
    """Return the exact canonical build-contract descriptor bytes."""

    return _DESCRIPTOR_BYTES


def parse_matched_v3_quicknet_verifier_build_contract_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Accept only the exact frozen build-contract descriptor."""

    value = _strict_json_load(raw)
    if not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        _fail("Quicknet build-contract descriptor identity differs")
    return value


def canonical_matched_v3_quicknet_verifier_request_bytes(
    *,
    round_number: int,
    public_key: bytes,
    signature: bytes,
) -> bytes:
    """Encode one fixed-width request without checking curve or signature validity."""

    if type(round_number) is not int or not 0 <= round_number <= QUICKNET_VERIFIER_ROUND_MAXIMUM:
        _fail("Quicknet verifier request round must be one exact u64")
    if type(public_key) is not bytes or len(public_key) != QUICKNET_VERIFIER_PUBLIC_KEY_SIZE_BYTES:
        _fail("Quicknet verifier request public key must be exactly 96 bytes")
    if type(signature) is not bytes or len(signature) != QUICKNET_VERIFIER_SIGNATURE_SIZE_BYTES:
        _fail("Quicknet verifier request signature must be exactly 48 bytes")
    raw = (
        QUICKNET_VERIFIER_REQUEST_MAGIC
        + round_number.to_bytes(8, byteorder="big", signed=False)
        + public_key
        + signature
    )
    if len(raw) != QUICKNET_VERIFIER_REQUEST_SIZE_BYTES:
        raise AssertionError("Quicknet verifier request encoder width drifted")
    return raw


def parse_matched_v3_quicknet_verifier_request(
    raw: bytes,
) -> MatchedV3QuicknetVerifierRequest:
    """Parse exactly one request frame; cryptographic candidates remain untrusted."""

    if type(raw) is not bytes:
        _fail("Quicknet verifier request must be exact bytes")
    if len(raw) != QUICKNET_VERIFIER_REQUEST_SIZE_BYTES:
        _fail("Quicknet verifier request must be exactly 160 bytes")
    if not hmac.compare_digest(raw[:8], QUICKNET_VERIFIER_REQUEST_MAGIC):
        _fail("Quicknet verifier request magic differs")
    round_number = int.from_bytes(raw[8:16], byteorder="big", signed=False)
    public_key = raw[16:112]
    signature = raw[112:160]
    replay = canonical_matched_v3_quicknet_verifier_request_bytes(
        round_number=round_number,
        public_key=public_key,
        signature=signature,
    )
    if not hmac.compare_digest(replay, raw):
        _fail("Quicknet verifier request is not canonical")
    return MatchedV3QuicknetVerifierRequest(
        round_number=round_number,
        public_key=public_key,
        signature=signature,
    )


def parse_matched_v3_quicknet_verifier_outcome(
    raw: bytes,
    *,
    request_bytes: bytes,
) -> MatchedV3QuicknetVerifierOutcome:
    """Parse an untrusted typed outcome and enforce its payload-leakage contract.

    The verified tag's randomness is checked against SHA-256 of the request's
    raw signature bytes.  That relation is not BLS verification and does not
    authenticate that an adapter executed or returned a truthful tag.
    """

    request = parse_matched_v3_quicknet_verifier_request(request_bytes)
    if type(raw) is not bytes:
        _fail("Quicknet verifier outcome must be exact bytes")
    if len(raw) < QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES:
        _fail("Quicknet verifier outcome is shorter than its minimum frame")
    if not hmac.compare_digest(raw[:8], QUICKNET_VERIFIER_OUTCOME_MAGIC):
        _fail("Quicknet verifier outcome magic differs")
    tag = raw[8]
    name = _OUTCOME_NAMES.get(tag)
    if name is None:
        _fail("Quicknet verifier outcome tag is unknown")
    if tag == QUICKNET_VERIFIER_OUTCOME_TAG_VERIFIED:
        if len(raw) != QUICKNET_VERIFIER_OUTCOME_VERIFIED_SIZE_BYTES:
            _fail("verified Quicknet verifier outcome must be exactly 41 bytes")
        randomness = raw[9:41]
        expected_randomness = hashlib.sha256(request.signature).digest()
        if not hmac.compare_digest(randomness, expected_randomness):
            _fail("verified Quicknet verifier outcome randomness relation differs")
        return MatchedV3QuicknetVerifierOutcome(
            outcome=name,
            outcome_tag=tag,
            randomness=randomness,
        )
    if len(raw) != QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES:
        _fail("rejected Quicknet verifier outcome must be exactly 9 bytes with no payload")
    return MatchedV3QuicknetVerifierOutcome(
        outcome=name,
        outcome_tag=tag,
        randomness=None,
    )


if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 Quicknet verifier build-contract descriptor drifted")


__all__ = [
    "ForagerMatchedV3QuicknetVerifierBuildContractError",
    "MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_DESCRIPTOR_SHA256",
    "MATCHED_V3_QUICKNET_VERIFIER_BUILD_CONTRACT_STATUS",
    "MatchedV3QuicknetVerifierOutcome",
    "MatchedV3QuicknetVerifierRequest",
    "QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256",
    "QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256",
    "QUICKNET_SOURCE_MATERIALIZATION_SOURCE_SHA256",
    "QUICKNET_SOURCE_MATERIALIZATION_TEST_SHA256",
    "QUICKNET_SOURCE_REGISTRY_DESCRIPTOR_SHA256",
    "QUICKNET_SOURCE_REGISTRY_SOURCE_SHA256",
    "QUICKNET_RUNTIME_VERIFIER_DESCRIPTOR_SCHEMA_VERSION",
    "QUICKNET_VERIFIER_OUTCOME_MAGIC",
    "QUICKNET_VERIFIER_OUTCOME_REJECTED_SIZE_BYTES",
    "QUICKNET_VERIFIER_OUTCOME_TAG_INVALID_PUBLIC_KEY",
    "QUICKNET_VERIFIER_OUTCOME_TAG_OK_FALSE",
    "QUICKNET_VERIFIER_OUTCOME_TAG_VERIFICATION_ERROR",
    "QUICKNET_VERIFIER_OUTCOME_TAG_VERIFIED",
    "QUICKNET_VERIFIER_OUTCOME_VERIFIED_SIZE_BYTES",
    "QUICKNET_VERIFIER_PUBLIC_KEY_SIZE_BYTES",
    "QUICKNET_VERIFIER_RANDOMNESS_SIZE_BYTES",
    "QUICKNET_VERIFIER_REQUEST_MAGIC",
    "QUICKNET_VERIFIER_REQUEST_SIZE_BYTES",
    "QUICKNET_VERIFIER_ROUND_MAXIMUM",
    "QUICKNET_VERIFIER_SIGNATURE_SIZE_BYTES",
    "canonical_matched_v3_quicknet_verifier_build_contract_descriptor_bytes",
    "canonical_matched_v3_quicknet_verifier_request_bytes",
    "matched_v3_quicknet_verifier_build_contract_descriptor",
    "parse_matched_v3_quicknet_verifier_build_contract_descriptor",
    "parse_matched_v3_quicknet_verifier_outcome",
    "parse_matched_v3_quicknet_verifier_request",
]
