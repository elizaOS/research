"""Capability-gated, score-bearing local result inspection for matched-v3.

This module is descriptor-only under an ordinary package import.  Bundle
issuance requires its exact isolated direct-byte load beside the audited local
result handoff.  Issuance consumes one authentic, unconsumed handoff
capability, converts the retained raw trace through the frozen scorer's byte
API, and returns a fresh PID-bound single-use capability.  A separate explicit
inspection opt-in consumes that capability and only then constructs an
immutable nine-file in-memory bundle.  Until one path wins, the registry
retains only a private sealed payload.

The public bundle is score-bearing: callers can parse its uncompressed
``reward-trace.npz`` and ``score-receipt.json``.  This module itself does not
decode either artifact, but public inspection is permanently nonqualifying.
The exact score-payload-opaque atomic publisher consumes the still-live capability
through the private sink captured when this module was loaded; it never accepts
a :class:`MatchedV3LocalRewardBundle` object or its serialized bytes.  Choosing
public inspection consumes the same single-use capability and therefore closes
the direct publisher path, and publication closes inspection in the same way.

This module performs no workload execution or filesystem publication and
grants no qualification, evidence, promotion, or publication authority.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import re
import stat
import struct
import sys
import threading
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_bundle_descriptor.v1"
)
LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_bundle_manifest.v1"
)
LOCAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_payload.v1"
)
LOCAL_REWARD_BUNDLE_STATUS: Final = "implemented_unexecuted_non_authorizing"
LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_reward_bundle_isolated_v1"
)

PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_result_handoff_isolated_v1"
)
PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_result_handoff_descriptor.v1"
)
PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256: Final = (
    "dc488f74d50ef224309e89968559df4671f4a3f954144530a9e4424e3cabba03"
)
PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256: Final = (
    "a5275d77d9b0870214b19c31acad73841f12c217f6eb411a6f8c56e317cc0819"
)
PINNED_LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_result_handoff_record.v1"
)

PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_reward_publication_isolated_v1"
)
PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f"
)
PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256: Final = (
    "48640a7e352383eac58fed24c8c36c77fcf3bbed8baf78ce663394d1f7e90200"
)
PINNED_LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_metadata.v1"
)

PINNED_SCORER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
)
PINNED_SCORER_SOURCE_SHA256: Final = (
    "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
)
PINNED_SCORER_PROTOCOL_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_protocol.py"
)
PINNED_SCORER_PROTOCOL_SOURCE_SHA256: Final = (
    "dd5db9a657ad167abf192942489642130b08bd065f724f7ad1b80743b1103720"
)
PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_cumulative_reward_metric.v1"
)
PINNED_SCORER_METRIC_DESCRIPTOR_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
PINNED_SCORE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_score_receipt.v1"
)
PINNED_REWARD_NPZ_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_reward_npz.v1"
)
PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_raw_reward_trace.int8.v1"
)
PINNED_RAW_TRACE_DIGEST_DOMAIN: Final = (
    b"alberta.forager.matched_v3.raw_reward_trace.int8.v1"
)
PINNED_CANONICAL_NPZ_SIZE_BYTES: Final = 499_980

PUBLICATION_MANIFEST_FILENAME: Final = "publication.json"
LOCAL_BUNDLE_MANIFEST_FILENAME: Final = "local-bundle-manifest.json"
BOOTSTRAP_RECEIPT_FILENAME: Final = "bootstrap-receipt.json"
BOOTSTRAP_CHILD_RECORD_FILENAME: Final = "bootstrap-child-record.json"
LOCAL_RUNNER_RECEIPT_FILENAME: Final = "local-runner-receipt.json"
REWARD_TRACE_FILENAME: Final = "reward-trace.npz"
SCORE_RECEIPT_FILENAME: Final = "score-receipt.json"
STDOUT_FILENAME: Final = "stdout.bin"
STDERR_FILENAME: Final = "stderr.bin"

_ROLE_PATHS: Final[tuple[tuple[str, str], ...]] = (
    ("publication_manifest", PUBLICATION_MANIFEST_FILENAME),
    ("local_bundle_manifest", LOCAL_BUNDLE_MANIFEST_FILENAME),
    ("bootstrap_receipt", BOOTSTRAP_RECEIPT_FILENAME),
    ("bootstrap_child_record", BOOTSTRAP_CHILD_RECORD_FILENAME),
    ("local_runner_receipt", LOCAL_RUNNER_RECEIPT_FILENAME),
    ("reward_trace", REWARD_TRACE_FILENAME),
    ("score_receipt", SCORE_RECEIPT_FILENAME),
    ("stdout", STDOUT_FILENAME),
    ("stderr", STDERR_FILENAME),
)
_PAYLOAD_ROLE_PATHS: Final[tuple[tuple[str, str], ...]] = _ROLE_PATHS[2:]
_PUBLICATION_BOUND_ROLE_PATHS: Final[tuple[tuple[str, str], ...]] = _ROLE_PATHS[1:]
_EXACT_FILENAMES: Final = tuple(path for _role, path in _ROLE_PATHS)

_BUNDLE_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_REWARD_BUNDLE_SOURCE_SHA256"
)
_HANDOFF_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_RESULT_HANDOFF_SOURCE_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")

_FORBIDDEN_PREFIXES: Final = (
    "alberta_framework",
    "chex",
    "foragax",
    "jax",
    "jaxlib",
    "ml_dtypes",
    "numpy",
    "scipy",
)
_MODULE_KEYS_AT_LOAD: Final = tuple(sys.modules)
_NONEXACT_MODULE_KEYS_AT_LOAD: Final = tuple(
    type(name).__name__ for name in _MODULE_KEYS_AT_LOAD if type(name) is not str
)
_PRELOADED_FORBIDDEN_AT_LOAD: Final = tuple(
    sorted(
        name
        for name in _MODULE_KEYS_AT_LOAD
        if type(name) is str
        and any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES)
    )
)
_SELF_MODULE_AT_LOAD: Final = (
    sys.modules.get(_MODULE_NAME_INPUT) if type(_MODULE_NAME_INPUT) is str else None
)
_ISOLATED_BUNDLE_BOUNDARY: Final = (
    type(_MODULE_NAME_INPUT) is str
    and _MODULE_NAME_INPUT == LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME
    and (
        _MODULE_PACKAGE_INPUT is None
        or (type(_MODULE_PACKAGE_INPUT) is str and _MODULE_PACKAGE_INPUT == "")
    )
    and type(_SELF_MODULE_AT_LOAD) is types.ModuleType
    and _SELF_MODULE_AT_LOAD.__dict__ is globals()
    and not _NONEXACT_MODULE_KEYS_AT_LOAD
    and not _PRELOADED_FORBIDDEN_AT_LOAD
)

_HANDOFF_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
)
_HANDOFF_CONSUMER_AT_LOAD: Final = getattr(
    _HANDOFF_MODULE_AT_LOAD,
    "consume_matched_v3_local_result_handoff",
    None,
)
_HANDOFF_PARSER_AT_LOAD: Final = getattr(
    _HANDOFF_MODULE_AT_LOAD,
    "parse_matched_v3_local_result_handoff_record",
    None,
)
_HANDOFF_CONTENT_TYPE_AT_LOAD: Final = getattr(
    _HANDOFF_MODULE_AT_LOAD,
    "MatchedV3LocalResultHandoffContent",
    None,
)
_PUBLISHER_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
)
_PUBLISHER_SINK_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD,
    "_publish_consumed_local_reward_payload",
    None,
)
_PUBLISHER_METADATA_TYPE_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD,
    "MatchedV3LocalRewardPublicationMetadata",
    None,
)
_PUBLISHER_REQUIRE_BOUNDARY_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD,
    "_require_publication_boundary",
    None,
)
_PUBLISHER_PARENT_PREFLIGHT_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD,
    "_preflight_publication_parent",
    None,
)
_HANDOFF_FUNCTION_IDENTITIES_AT_LOAD: Final = (
    tuple(
        sorted(
            (
                (name, value, value.__code__)
                for name, value in vars(_HANDOFF_MODULE_AT_LOAD).items()
                if type(_HANDOFF_MODULE_AT_LOAD) is types.ModuleType
                and type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )
    if type(_HANDOFF_MODULE_AT_LOAD) is types.ModuleType
    else ()
)

_PINNED_HANDOFF_CONSUMER_CODE_SHA256: Final = (
    "0d3c508e4261b1d5f1f7438e011d4f9c714cd553bdeb56db433ef7b35d431140"
)
_PINNED_HANDOFF_PARSER_CODE_SHA256: Final = (
    "5dd6de033b8046bec37b01b38f3eaa4cb9f7249528b2673de9681ab496ef067a"
)
_PINNED_SCORER_ENCODER_CODE_SHA256: Final = (
    "a87f85bdc96dcae4da51bd24381d6e47c31a2bbad94bc4765de3ae4d0ecda83d"
)
_PINNED_SCORER_INGEST_CODE_SHA256: Final = (
    "0341f7c0bf6c9abd4f60c9ef3399ee54a4bc8fdc9efd38512934fdabd0ccad6d"
)
_PINNED_SCORER_PARSE_CODE_SHA256: Final = (
    "d7a273e731f905fe9e99132d345c9294635f5a2755dcf126ceb32aacb46cc67b"
)
_PINNED_SCORER_RECEIPT_JSON_CODE_SHA256: Final = (
    "f050000a66c93df1ae29eac5489acfcf3293708f4caaa18d8606509edc36d488"
)
PINNED_SCORER_SEMANTIC_SURFACE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.scorer_semantic_surface.v1"
)
PINNED_SCORER_SEMANTIC_SURFACE_SHA256: Final = (
    "008d2650a369794dc7fa3c87a2ddb2e8d7ca61f41635c36a3ad57acfb18defa6"
)

_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 1024 * 1024
_MAX_BOOTSTRAP_RECEIPT_BYTES: Final = 32 * 1024 * 1024
_MAX_CHILD_RECORD_BYTES: Final = 32 * 1024 * 1024
_MAX_LOCAL_RECEIPT_BYTES: Final = 32 * 1024 * 1024
_MAX_RAW_TRACE_BYTES: Final = 32 * 1024 * 1024
_MAX_STDIO_BYTES: Final = 32 * 1024 * 1024
_MAX_SCORE_RECEIPT_BYTES: Final = 64 * 1024
_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_STRING_BYTES: Final = 24 * 1024 * 1024
_UINT31_MAX: Final = 2**31 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PORTABLE_FILENAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ForagerMatchedV3LocalRewardBundleError(RuntimeError):
    """A live capability, byte binding, scorer replay, or manifest failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalRewardBundleError(
        f"local reward bundle JSON contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalRewardBundleError(
        f"local reward bundle JSON contains forbidden float {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalRewardBundleError(
                f"local reward bundle JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle JSON exceeds its node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle JSON exceeds its depth bound"
            )
        if type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3LocalRewardBundleError(
                    "local reward bundle JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES or any(
                byte < 0x20 or byte > 0x7E for byte in encoded
            ):
                raise ForagerMatchedV3LocalRewardBundleError(
                    "local reward bundle JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle JSON contains a non-plain value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalRewardBundleError(
                        "local reward bundle JSON keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: dict[str, Any], *, maximum_bytes: int) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle canonical JSON root must be a plain object"
        )
    _assert_plain_unaliased_json(value)
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
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle value is not canonical finite ASCII JSON"
        ) from exc
    if not 0 < len(raw) <= maximum_bytes:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle canonical artifact exceeds its byte ceiling"
        )
    return raw


def _strict_json_object(raw: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must be bounded nonempty exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must have exactly one trailing newline"
        )
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalRewardBundleError:
        raise
    except (RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} root must be a plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(
        _canonical_json(result, maximum_bytes=maximum_bytes),
        raw,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} is not exactly canonical")
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        right_list = cast(list[Any], right)
        return len(left) == len(right_list) and all(
            _exact_json_equal(item, right_item)
            for item, right_item in zip(left, right_list, strict=True)
        )
    if type(left) is dict:
        left_dict = cast(dict[str, Any], left)
        right_dict = cast(dict[str, Any], right)
        return set(left_dict) == set(right_dict) and all(
            _exact_json_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    return False


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_uint31(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must be an exact unsigned 31-bit integer"
        )
    return value


def _require_candidate_id(value: Any) -> str:
    if type(value) is not str or _CANDIDATE_RE.fullmatch(value) is None:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle candidate ID is invalid"
        )
    return value


def _require_portable_filename(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or _PORTABLE_FILENAME_RE.fullmatch(value) is None
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must be one flat portable filename"
        )
    return value


def _require_exact_bytes(
    value: Any,
    *,
    label: str,
    maximum_bytes: int,
    require_nonempty: bool,
) -> bytes:
    if type(value) is not bytes or len(value) > maximum_bytes:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} must be bounded exact bytes"
        )
    if require_nonempty and not value:
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} must be nonempty")
    return value


def _claims() -> dict[str, bool]:
    return {
        "authorization_issuer": False,
        "execution_authority_granted": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "public_bundle_qualification_allowed": False,
        "qualification_publisher_may_accept_public_bundle_object": False,
    }


def _limitations() -> list[str]:
    return [
        "This bundle retains one unqualified local result and is not scientific evidence.",
        "Serialized bytes cannot reconstruct either process-local capability.",
        "Public inspection exposes a parseable score receipt and uncompressed reward NPZ.",
        "This module does not decode those artifacts; that does not make them score-blind.",
        "Every public bundle object is permanently nonqualifying inspection content.",
        (
            "A qualification publisher that returns no score fields or payload bytes must "
            "consume the live capability directly, must never accept a public bundle object, "
            "and still requires an externally enforced no-branch policy for digest metadata."
        ),
        "No file has been published, fsynced, or made durable by this layer.",
        (
            "The pre-claim parent preflight does not eliminate filesystem TOCTOU; "
            "the atomic commit independently reopens and revalidates the parent."
        ),
        "Runtime, source, hardware, qualification, publication, and promotion remain external.",
        (
            "Same-process Python state remains mutable; pinned semantic and function surfaces "
            "detect enumerated drift but cannot make a hostile interpreter trustworthy."
        ),
    ]


def _inventory_descriptor() -> dict[str, Any]:
    return {
        "file_count": len(_ROLE_PATHS),
        "exact_filenames": list(_EXACT_FILENAMES),
        "flat_directory_only": True,
        "path_traversal_allowed": False,
        "symlinks_allowed": False,
        "stdout_and_stderr_may_be_zero_length": True,
    }


def _inspection_contract() -> dict[str, bool]:
    return {
        "canonical_scorer_byte_api_only": True,
        "canonical_npz_reingested": True,
        "score_receipt_replayed": True,
        "module_decodes_score_fields": False,
        "module_decodes_reward_npz": False,
        "public_bundle_is_score_bearing": True,
        "public_bundle_is_permanently_nonqualifying": True,
        "score_receipt_is_parseable_by_caller": True,
        "reward_npz_is_uncompressed_and_parseable_by_caller": True,
    }


def _qualification_publisher_contract() -> dict[str, bool]:
    return {
        "must_consume_live_capability_directly": True,
        "may_accept_public_bundle_object": False,
        "may_accept_serialized_public_bundle_bytes": False,
        "public_inspection_consumes_and_closes_direct_path": True,
        "publication_consumes_and_closes_inspection_path": True,
        "exact_publisher_sink_captured_at_bundle_load": True,
        "exact_parent_preflight_captured_at_bundle_load": True,
        "safe_parent_preflight_precedes_capability_claim": True,
        "atomic_commit_reopens_and_reverifies_parent": True,
        "parent_preflight_eliminates_toctou": False,
        "publisher_implemented_in_captured_external_module": True,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_REWARD_BUNDLE_STATUS,
        "classification": (
            "score_bearing_permanently_nonqualifying_inspection_plumbing_non_authorizing"
        ),
        "handoff": {
            "descriptor_schema_version": (
                PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256,
            "isolated_module_name": PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME,
            "authentic_unconsumed_capability_required": True,
            "consumed_internally": True,
        },
        "capability": {
            "issuance_explicit_opt_in": True,
            "content_access_separate_explicit_opt_in": True,
            "opaque": True,
            "pid_bound": True,
            "single_use": True,
            "weak_registry": True,
            "serializable": False,
            "copyable": False,
            "single_use_path_choice": (
                "public_score_bearing_inspection_or_exact_direct_publisher"
            ),
            "registry_retains_private_sealed_payload_not_public_bundle": True,
        },
        "scorer": {
            "source_path": PINNED_SCORER_SOURCE_PATH,
            "source_sha256": PINNED_SCORER_SOURCE_SHA256,
            "protocol_source_path": PINNED_SCORER_PROTOCOL_SOURCE_PATH,
            "protocol_source_sha256": PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
            "metric_descriptor_schema_version": (
                PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION
            ),
            "metric_descriptor_sha256": PINNED_SCORER_METRIC_DESCRIPTOR_SHA256,
            "score_receipt_schema_version": PINNED_SCORE_RECEIPT_SCHEMA_VERSION,
            "reward_npz_schema_version": PINNED_REWARD_NPZ_SCHEMA_VERSION,
            "canonical_npz_size_bytes": PINNED_CANONICAL_NPZ_SIZE_BYTES,
            "semantic_surface_schema_version": (
                PINNED_SCORER_SEMANTIC_SURFACE_SCHEMA_VERSION
            ),
            "semantic_surface_sha256": PINNED_SCORER_SEMANTIC_SURFACE_SHA256,
        },
        "inspection": _inspection_contract(),
        "qualification_publisher": _qualification_publisher_contract(),
        "publisher_binding": {
            "descriptor_schema_version": (
                PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256,
            "metadata_schema_version": (
                PINNED_LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION
            ),
            "isolated_module_name": (
                PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
            ),
            "one_way_bundle_to_publisher_source_trust": True,
            "live_capability_direct_sink_only": True,
        },
        "filesystem": {
            "writes": False,
            "publication_performed": False,
            "captured_atomic_publisher_performs_writes_only_after_path_choice": True,
        },
        "inventory": _inventory_descriptor(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(),
    maximum_bytes=_MAX_DESCRIPTOR_BYTES,
)
LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "f1fb7d28f0508c38b0d53173707ea5cb006b669793d3401091a942874ee3b878"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
):
    raise AssertionError(
        "matched-v3 local reward bundle descriptor identity drifted: "
        f"{hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest()}"
    )


def _live_forbidden_modules() -> tuple[str, ...]:
    try:
        names = tuple(sys.modules)
    except RuntimeError as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "runtime module registry changed during bundle boundary observation"
        ) from exc
    if any(type(name) is not str for name in names):
        raise ForagerMatchedV3LocalRewardBundleError(
            "runtime module registry contains a non-exact-string key"
        )
    return tuple(
        sorted(
            name
            for name in names
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_PREFIXES
            )
        )
    )


_SELF_FUNCTION_SURFACE_AT_READY: tuple[
    tuple[str, types.FunctionType, types.CodeType], ...
] | None = None


def _current_self_function_surface() -> tuple[
    tuple[str, types.FunctionType, types.CodeType], ...
]:
    module_name = _MODULE_NAME_INPUT
    if type(module_name) is not str:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle module name is not exact"
        )
    return tuple(
        sorted(
            (
                (name, value, value.__code__)
                for name, value in globals().items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == module_name
            ),
            key=lambda item: item[0],
        )
    )


def _require_self_function_surface() -> None:
    expected = _SELF_FUNCTION_SURFACE_AT_READY
    if expected is None:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle own-function surface is not ready"
        )
    current = _current_self_function_surface()
    if len(current) != len(expected) or any(
        current_name != expected_name
        or current_function is not expected_function
        or current_code is not expected_code
        for (current_name, current_function, current_code), (
            expected_name,
            expected_function,
            expected_code,
        ) in zip(current, expected, strict=True)
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle own-function surface drifted in memory"
        )


def _read_exact_source_sha256(
    module: types.ModuleType,
    *,
    label: str,
    expected_suffix: str | None = None,
) -> str:
    raw_path = getattr(module, "__file__", None)
    if type(raw_path) is not str:
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} has no exact source path")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.anchor != os.sep
        or path == Path(path.anchor)
        or os.path.abspath(raw_path) != raw_path
        or (expected_suffix is not None and not raw_path.endswith(expected_suffix))
    ):
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} source path is not exact")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAX_SOURCE_BYTES
            or before_identity != opened_identity
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} is not one stable bounded single-link regular source file"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            except InterruptedError:
                continue
            if not chunk:
                raise ForagerMatchedV3LocalRewardBundleError(
                    f"{label} source ended while being hashed"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} source grew while being hashed"
            )
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if before_identity != after_identity or before_identity != current_identity:
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} source changed while being hashed"
            )
        return digest.hexdigest()
    except OSError as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} source could not be read exactly"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_exact_source_bytes(
    path: Path,
    *,
    label: str,
    expected_suffix: str,
    expected_sha256: str,
) -> bytes:
    raw_path = str(path)
    if (
        not path.is_absolute()
        or path.anchor != os.sep
        or path == Path(path.anchor)
        or os.path.abspath(raw_path) != raw_path
        or not raw_path.endswith(expected_suffix)
    ):
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} source path is not exact")
    expected = _require_sha256(expected_sha256, f"{label} expected source")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAX_SOURCE_BYTES
            or before_identity != opened_identity
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} is not one stable bounded single-link regular source file"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            except InterruptedError:
                continue
            if not chunk:
                raise ForagerMatchedV3LocalRewardBundleError(
                    f"{label} source ended while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} source grew while being read"
            )
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if (
            before_identity != after_identity
            or before_identity != current_identity
            or len(raw) != before.st_size
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected)
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} source identity changed or disagrees with its pin"
            )
        return raw
    except OSError as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} source could not be read exactly"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stable_code_constant(value: Any) -> tuple[Any, ...]:
    if type(value) is types.CodeType:
        return ("code", _code_shape(value))
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value.hex())
    if type(value) is tuple:
        return ("tuple", tuple(_stable_code_constant(item) for item in value))
    if type(value) is frozenset:
        items = (_stable_code_constant(item) for item in value)
        return ("frozenset", tuple(sorted(items, key=repr)))
    return ("other", type(value).__module__, type(value).__qualname__, repr(value))


def _code_shape(code: types.CodeType) -> tuple[Any, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code.hex(),
        tuple(_stable_code_constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )


def _function_code_sha256(function: Any) -> str:
    if type(function) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardBundleError(
            "dependency boundary callable is not one exact Python function"
        )
    return hashlib.sha256(repr(_code_shape(function.__code__)).encode("ascii")).hexdigest()


def _validated_function(
    function: Any,
    *,
    module: types.ModuleType,
    module_name: str,
    name: str,
    code_sha256: str,
    captured: Any | None = None,
) -> types.FunctionType:
    if type(function) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"dependency function identity drifted: {name}"
        )
    exact = function
    if (
        (captured is not None and exact is not captured)
        or exact.__name__ != name
        or exact.__qualname__ != name
        or exact.__module__ != module_name
        or exact.__globals__ is not module.__dict__
        or not hmac.compare_digest(_function_code_sha256(exact), code_sha256)
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            f"dependency function identity drifted: {name}"
        )
    return exact


def _require_bundle_boundary(*, reject_runtime_modules: bool) -> str:
    _require_self_function_surface()
    if not _ISOLATED_BUNDLE_BOUNDARY:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle requires its exact isolated direct-byte module boundary"
        )
    expected = _require_sha256(
        _BUNDLE_SOURCE_SHA256_INPUT,
        "local reward bundle direct-byte source",
    )
    handoff_source = _require_sha256(
        _HANDOFF_SOURCE_SHA256_INPUT,
        "local result handoff source injection",
    )
    if not hmac.compare_digest(
        handoff_source,
        PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local handoff source injection differs from the pinned dependency"
        )
    if reject_runtime_modules:
        forbidden = _live_forbidden_modules()
        if forbidden:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle rejects preloaded runtime dependencies: "
                f"{', '.join(forbidden[:8])}"
            )
    current = sys.modules.get(LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME)
    if type(current) is not types.ModuleType or current is not _SELF_MODULE_AT_LOAD:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle isolated module identity is stale"
        )
    observed = _read_exact_source_sha256(current, label="local reward bundle")
    if not hmac.compare_digest(observed, expected):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle direct-byte source identity is stale or forged"
        )
    return expected


def _current_handoff_function_surface(module: types.ModuleType) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (
                (name, value)
                for name, value in module.__dict__.items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
            ),
            key=lambda item: item[0],
        )
    )


def _require_exact_handoff_module() -> tuple[
    types.ModuleType,
    types.FunctionType,
    types.FunctionType,
    type[Any],
]:
    _require_self_function_surface()
    current = sys.modules.get(PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME)
    if (
        type(_HANDOFF_MODULE_AT_LOAD) is not types.ModuleType
        or current is not _HANDOFF_MODULE_AT_LOAD
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff module is absent, replaced, or was not preloaded"
        )
    module = current
    if (
        module.__name__ != PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
        or module.__package__ not in {None, ""}
        or getattr(module, "LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256", None)
        != PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256
        or getattr(module, "LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION", None)
        != PINNED_LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION
        or getattr(module, "LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME", None)
        != PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
        or getattr(module, "_HANDOFF_SOURCE_SHA256_INPUT", None)
        != PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff module identity drifted"
        )
    observed = _read_exact_source_sha256(module, label="local result handoff")
    if not hmac.compare_digest(observed, PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff source bytes drifted"
        )
    current_functions = _current_handoff_function_surface(module)
    if (
        len(current_functions) != len(_HANDOFF_FUNCTION_IDENTITIES_AT_LOAD)
        or any(
            current_name != expected_name
            or current_function is not expected_function
            or current_function.__code__ is not expected_code
            for (current_name, current_function), (
                expected_name,
                expected_function,
                expected_code,
            ) in zip(
                current_functions,
                _HANDOFF_FUNCTION_IDENTITIES_AT_LOAD,
                strict=True,
            )
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff in-memory function surface drifted"
        )
    consumer = _validated_function(
        getattr(module, "consume_matched_v3_local_result_handoff", None),
        module=module,
        module_name=PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME,
        name="consume_matched_v3_local_result_handoff",
        code_sha256=_PINNED_HANDOFF_CONSUMER_CODE_SHA256,
        captured=_HANDOFF_CONSUMER_AT_LOAD,
    )
    parser = _validated_function(
        getattr(module, "parse_matched_v3_local_result_handoff_record", None),
        module=module,
        module_name=PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME,
        name="parse_matched_v3_local_result_handoff_record",
        code_sha256=_PINNED_HANDOFF_PARSER_CODE_SHA256,
        captured=_HANDOFF_PARSER_AT_LOAD,
    )
    content_type = getattr(module, "MatchedV3LocalResultHandoffContent", None)
    if (
        type(content_type) is not type
        or content_type is not _HANDOFF_CONTENT_TYPE_AT_LOAD
        or content_type.__module__ != PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff content type drifted"
        )
    descriptor_raw = getattr(
        module,
        "canonical_matched_v3_local_result_handoff_descriptor_bytes",
    )()
    if (
        type(descriptor_raw) is not bytes
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_raw).hexdigest(),
            PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff descriptor bytes drifted"
        )
    try:
        parsed_descriptor = getattr(
            module,
            "parse_matched_v3_local_result_handoff_descriptor",
        )(descriptor_raw)
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff descriptor replay failed"
        ) from exc
    if type(parsed_descriptor) is not dict:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff descriptor parser returned a non-plain object"
        )
    return module, consumer, parser, cast(type[Any], content_type)


def _require_exact_publisher_module() -> tuple[
    types.ModuleType,
    types.FunctionType,
    type[Any],
]:
    """Replay the publisher captured before this bundle was direct-loaded."""

    _require_self_function_surface()
    current = sys.modules.get(PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME)
    module = _PUBLISHER_MODULE_AT_LOAD
    if type(module) is not types.ModuleType or current is not module:
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local reward publisher is absent or replaced"
        )
    sink = getattr(module, "_publish_consumed_local_reward_payload", None)
    metadata_type = getattr(module, "MatchedV3LocalRewardPublicationMetadata", None)
    require_boundary = getattr(module, "_require_publication_boundary", None)
    parent_preflight = getattr(module, "_preflight_publication_parent", None)
    if (
        module.__name__ != PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
        or module.__package__ not in {None, ""}
        or getattr(module, "LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256", None)
        != PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or getattr(module, "LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION", None)
        != PINNED_LOCAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION
        or getattr(module, "LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME", None)
        != PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
        or getattr(module, "_PUBLICATION_SOURCE_SHA256_INPUT", None)
        != PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256
        or sink is not _PUBLISHER_SINK_AT_LOAD
        or metadata_type is not _PUBLISHER_METADATA_TYPE_AT_LOAD
        or require_boundary is not _PUBLISHER_REQUIRE_BOUNDARY_AT_LOAD
        or parent_preflight is not _PUBLISHER_PARENT_PREFLIGHT_AT_LOAD
        or type(sink) is not types.FunctionType
        or type(require_boundary) is not types.FunctionType
        or type(parent_preflight) is not types.FunctionType
        or type(metadata_type) is not type
        or sink.__module__ != PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
        or sink.__globals__ is not module.__dict__
        or parent_preflight.__module__
        != PINNED_LOCAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
        or parent_preflight.__globals__ is not module.__dict__
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local reward publisher identity drifted"
        )
    observed = _read_exact_source_sha256(
        module,
        label="local reward publisher",
        expected_suffix=(
            "/alberta_framework/benchmarks/forager_matched_v3_local_reward_publication.py"
        ),
    )
    if not hmac.compare_digest(observed, PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local reward publisher source bytes drifted"
        )
    try:
        guarded_source = require_boundary(reject_runtime_modules=False)
        descriptor_raw = getattr(
            module,
            "canonical_matched_v3_local_reward_publication_descriptor_bytes",
        )()
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local reward publisher guard replay failed"
        ) from exc
    if (
        guarded_source != PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256
        or type(descriptor_raw) is not bytes
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_raw).hexdigest(),
            PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local reward publisher source or descriptor drifted"
        )
    return module, sink, cast(type[Any], metadata_type)


@dataclass(frozen=True, slots=True)
class _ScorerAPI:
    module: types.ModuleType
    protocol_module: types.ModuleType
    encode: types.FunctionType
    ingest: types.FunctionType
    parse: types.FunctionType
    receipt_type: type[Any]
    receipt_canonical_json: types.FunctionType
    semantic_surface_sha256: str
    scorer_function_surface: tuple[tuple[str, types.FunctionType, types.CodeType], ...]
    protocol_function_surface: tuple[tuple[str, types.FunctionType, types.CodeType], ...]
    receipt_class_surface: tuple[tuple[str, object], ...]


_SCORER_LOAD_LOCK: Final = threading.Lock()
_SCORER_API_CACHE: _ScorerAPI | None = None


def _execute_exact_source_module(
    *,
    raw: bytes,
    path: Path,
    module_name: str,
    package_name: str,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = package_name
    sys.modules[module_name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def _direct_load_stdlib_scorer() -> types.ModuleType:
    _require_self_function_surface()
    if threading.active_count() != 1:
        raise ForagerMatchedV3LocalRewardBundleError(
            "isolated scorer direct-byte loading requires a single-threaded process"
        )
    self_path = getattr(_SELF_MODULE_AT_LOAD, "__file__", None)
    if type(self_path) is not str or not os.path.isabs(self_path):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle cannot resolve its scorer source siblings"
        )
    source_directory = Path(self_path).parent
    protocol_path = source_directory / "forager_matched_v3_protocol.py"
    scorer_path = source_directory / "_forager_matched_v3_scorer.py"
    protocol_raw = _read_exact_source_bytes(
        protocol_path,
        label="matched-v3 scorer protocol",
        expected_suffix=PINNED_SCORER_PROTOCOL_SOURCE_PATH,
        expected_sha256=PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
    )
    scorer_raw = _read_exact_source_bytes(
        scorer_path,
        label="canonical matched-v3 scorer",
        expected_suffix=PINNED_SCORER_SOURCE_PATH,
        expected_sha256=PINNED_SCORER_SOURCE_SHA256,
    )
    root_name = "alberta_framework"
    package_name = "alberta_framework.benchmarks"
    protocol_name = f"{package_name}.forager_matched_v3_protocol"
    scorer_name = f"{package_name}._forager_matched_v3_scorer"
    occupied = tuple(
        name
        for name in (root_name, package_name, protocol_name, scorer_name)
        if name in sys.modules
    )
    if occupied:
        raise ForagerMatchedV3LocalRewardBundleError(
            "isolated scorer direct-byte module names are already occupied"
        )
    root = types.ModuleType(root_name)
    root.__package__ = root_name
    setattr(root, "__path__", [])
    package = types.ModuleType(package_name)
    package.__package__ = package_name
    setattr(package, "__path__", [])
    protocol: types.ModuleType | None = None
    scorer: types.ModuleType | None = None
    sys.modules[root_name] = root
    sys.modules[package_name] = package
    setattr(root, "benchmarks", package)
    try:
        protocol = _execute_exact_source_module(
            raw=protocol_raw,
            path=protocol_path,
            module_name=protocol_name,
            package_name=package_name,
        )
        setattr(package, "forager_matched_v3_protocol", protocol)
        scorer = _execute_exact_source_module(
            raw=scorer_raw,
            path=scorer_path,
            module_name=scorer_name,
            package_name=package_name,
        )
    except BaseException as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer exact-source loading failed"
        ) from exc
    finally:
        for name, expected in (
            (scorer_name, scorer),
            (protocol_name, protocol),
            (package_name, package),
            (root_name, root),
        ):
            if expected is not None and sys.modules.get(name) is expected:
                sys.modules.pop(name, None)
    if type(scorer) is not types.ModuleType:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer exact-source load returned no module"
        )
    return scorer


def _load_scorer_module_once() -> types.ModuleType:
    existing = sys.modules.get(
        "alberta_framework.benchmarks._forager_matched_v3_scorer"
    )
    if existing is None:
        return _direct_load_stdlib_scorer()
    try:
        imported = importlib.import_module(
            "alberta_framework.benchmarks._forager_matched_v3_scorer"
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "the canonical matched-v3 scorer could not be loaded"
        ) from exc
    if type(imported) is not types.ModuleType or imported is not existing:
        raise ForagerMatchedV3LocalRewardBundleError(
            "the canonical matched-v3 scorer module is not exact"
        )
    return imported


def _module_function_surface(
    module: types.ModuleType,
) -> tuple[tuple[str, types.FunctionType, types.CodeType], ...]:
    return tuple(
        sorted(
            (
                (name, value, value.__code__)
                for name, value in module.__dict__.items()
                if type(name) is str
                and type(value) is types.FunctionType
                and value.__module__ == module.__name__
            ),
            key=lambda item: item[0],
        )
    )


def _semantic_sort_key(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "scorer semantic surface contains a nonsortable value"
        ) from exc


def _semantic_value(value: Any, *, active: set[int]) -> Any:
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, re.Pattern):
        return {
            "kind": "regex",
            "pattern": _semantic_value(value.pattern, active=active),
            "flags": str(value.flags),
        }
    if isinstance(value, struct.Struct):
        return {
            "kind": "struct",
            "format": value.format,
            "size": str(value.size),
        }
    if type(value) in {tuple, list, frozenset, set, dict}:
        identity = id(value)
        if identity in active:
            raise ForagerMatchedV3LocalRewardBundleError(
                "scorer semantic surface contains a container cycle"
            )
        active.add(identity)
        try:
            if type(value) in {tuple, list}:
                return {
                    "kind": "tuple" if type(value) is tuple else "list",
                    "items": [
                        _semantic_value(item, active=active)
                        for item in cast(tuple[Any, ...] | list[Any], value)
                    ],
                }
            if type(value) in {frozenset, set}:
                items = [
                    _semantic_value(item, active=active)
                    for item in cast(frozenset[Any] | set[Any], value)
                ]
                return {
                    "kind": "frozenset" if type(value) is frozenset else "set",
                    "items": sorted(items, key=_semantic_sort_key),
                }
            entries = [
                [
                    _semantic_value(key, active=active),
                    _semantic_value(item, active=active),
                ]
                for key, item in cast(dict[Any, Any], value).items()
            ]
            return {
                "kind": "dict",
                "items": sorted(entries, key=lambda entry: _semantic_sort_key(entry[0])),
            }
        finally:
            active.remove(identity)
    raise ForagerMatchedV3LocalRewardBundleError(
        "scorer semantic surface contains unsupported behavior global type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _semantic_behavior_globals(module: types.ModuleType) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in sorted(module.__dict__.items()):
        if name.startswith("__") or name == "annotations":
            continue
        if (
            type(value) is types.ModuleType
            or type(value) is types.FunctionType
            or type(value) is type
            or callable(value)
        ):
            continue
        result[name] = _semantic_value(value, active=set())
    return result


def _semantic_function_codes(module: types.ModuleType) -> dict[str, str]:
    return {
        name: _function_code_sha256(function)
        for name, function, _code in _module_function_surface(module)
    }


def _receipt_accessor_codes(receipt_type: type[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    attribute_types: dict[str, str] = {}
    for name, value in sorted(vars(receipt_type).items()):
        attribute_types[name] = f"{type(value).__module__}.{type(value).__qualname__}"
        if type(value) is types.FunctionType:
            result[name] = {
                "kind": "method",
                "code_sha256": _function_code_sha256(value),
            }
        elif type(value) is property:
            accessors: dict[str, str | None] = {}
            for accessor_name, accessor in (
                ("get", value.fget),
                ("set", value.fset),
                ("delete", value.fdel),
            ):
                accessors[accessor_name] = (
                    None if accessor is None else _function_code_sha256(accessor)
                )
            result[name] = {"kind": "property", "accessors": accessors}
        elif type(value) in {staticmethod, classmethod}:
            function = value.__func__
            if type(function) is not types.FunctionType:
                raise ForagerMatchedV3LocalRewardBundleError(
                    "score receipt static/class method is not exact"
                )
            result[name] = {
                "kind": "staticmethod" if type(value) is staticmethod else "classmethod",
                "code_sha256": _function_code_sha256(function),
            }
    return {"attribute_types": attribute_types, "callable_accessors": result}


def _scorer_semantic_surface_sha256(
    *,
    scorer: types.ModuleType,
    protocol: types.ModuleType,
    receipt_type: type[Any],
) -> str:
    surface = {
        "schema_version": PINNED_SCORER_SEMANTIC_SURFACE_SCHEMA_VERSION,
        "source_binding": {
            "scorer_sha256": PINNED_SCORER_SOURCE_SHA256,
            "protocol_sha256": PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
        },
        "scorer_functions": _semantic_function_codes(scorer),
        "protocol_functions": _semantic_function_codes(protocol),
        "score_receipt": _receipt_accessor_codes(receipt_type),
        "scorer_behavior_globals": _semantic_behavior_globals(scorer),
        "protocol_behavior_globals": _semantic_behavior_globals(protocol),
    }
    return hashlib.sha256(
        _canonical_json(surface, maximum_bytes=_MAX_DESCRIPTOR_BYTES)
    ).hexdigest()


def _validate_scorer_module(module: types.ModuleType) -> _ScorerAPI:
    _require_self_function_surface()
    observed = _read_exact_source_sha256(
        module,
        label="canonical matched-v3 scorer",
        expected_suffix=PINNED_SCORER_SOURCE_PATH,
    )
    if not hmac.compare_digest(observed, PINNED_SCORER_SOURCE_SHA256):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer source identity drifted"
        )
    module_name = "alberta_framework.benchmarks._forager_matched_v3_scorer"
    encode = _validated_function(
        getattr(module, "canonical_reward_npz_bytes", None),
        module=module,
        module_name=module_name,
        name="canonical_reward_npz_bytes",
        code_sha256=_PINNED_SCORER_ENCODER_CODE_SHA256,
    )
    ingest = _validated_function(
        getattr(module, "ingest_reward_npz_bytes", None),
        module=module,
        module_name=module_name,
        name="ingest_reward_npz_bytes",
        code_sha256=_PINNED_SCORER_INGEST_CODE_SHA256,
    )
    parse = _validated_function(
        getattr(module, "parse_score_receipt", None),
        module=module,
        module_name=module_name,
        name="parse_score_receipt",
        code_sha256=_PINNED_SCORER_PARSE_CODE_SHA256,
    )
    receipt_type = getattr(module, "MatchedV3ScoreReceipt", None)
    if (
        type(receipt_type) is not type
        or receipt_type.__module__ != module_name
        or getattr(module, "SCORE_RECEIPT_SCHEMA_VERSION", None)
        != PINNED_SCORE_RECEIPT_SCHEMA_VERSION
        or getattr(module, "NPZ_CONTAINER_SCHEMA_VERSION", None)
        != PINNED_REWARD_NPZ_SCHEMA_VERSION
        or getattr(module, "RAW_TRACE_ENCODING_SCHEMA_VERSION", None)
        != PINNED_RAW_TRACE_ENCODING_SCHEMA_VERSION
        or getattr(module, "CANONICAL_NPZ_SIZE_BYTES", None)
        != PINNED_CANONICAL_NPZ_SIZE_BYTES
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer type or schema constants drifted"
        )
    canonical_json_method = getattr(receipt_type, "canonical_json", None)
    if type(canonical_json_method) is not types.FunctionType or not hmac.compare_digest(
        _function_code_sha256(canonical_json_method),
        _PINNED_SCORER_RECEIPT_JSON_CODE_SHA256,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 score receipt byte API drifted"
        )
    protocol = getattr(module, "protocol", None)
    if type(protocol) is not types.ModuleType:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer protocol module is not exact"
        )
    observed_protocol = _read_exact_source_sha256(
        protocol,
        label="matched-v3 scorer protocol",
        expected_suffix=PINNED_SCORER_PROTOCOL_SOURCE_PATH,
    )
    if not hmac.compare_digest(
        observed_protocol,
        PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer protocol source identity drifted"
        )
    semantic_surface_sha256 = _scorer_semantic_surface_sha256(
        scorer=module,
        protocol=protocol,
        receipt_type=cast(type[Any], receipt_type),
    )
    if (
        getattr(module, "RAW_TRACE_DIGEST_DOMAIN", None)
        != PINNED_RAW_TRACE_DIGEST_DOMAIN
        or not hmac.compare_digest(
            semantic_surface_sha256,
            PINNED_SCORER_SEMANTIC_SURFACE_SHA256,
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer pinned semantic surface drifted"
        )
    descriptor_function = getattr(protocol, "cumulative_reward_metric_descriptor", None)
    descriptor_bytes_function = getattr(
        protocol,
        "canonical_cumulative_reward_metric_bytes",
        None,
    )
    if not callable(descriptor_function) or not callable(descriptor_bytes_function):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer metric descriptor API is absent"
        )
    try:
        descriptor_raw = descriptor_bytes_function()
        descriptor_value = descriptor_function()
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer metric descriptor replay failed"
        ) from exc
    if (
        type(descriptor_raw) is not bytes
        or type(descriptor_value) is not dict
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_raw).hexdigest(),
            PINNED_SCORER_METRIC_DESCRIPTOR_SHA256,
        )
        or getattr(protocol, "CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION", None)
        != PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION
        or getattr(protocol, "CUMULATIVE_REWARD_METRIC_SHA256", None)
        != PINNED_SCORER_METRIC_DESCRIPTOR_SHA256
        or _canonical_json(
            cast(dict[str, Any], descriptor_value),
            maximum_bytes=_MAX_DESCRIPTOR_BYTES,
        )
        != descriptor_raw
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical matched-v3 scorer metric descriptor identity drifted"
        )
    return _ScorerAPI(
        module=module,
        protocol_module=protocol,
        encode=encode,
        ingest=ingest,
        parse=parse,
        receipt_type=cast(type[Any], receipt_type),
        receipt_canonical_json=canonical_json_method,
        semantic_surface_sha256=semantic_surface_sha256,
        scorer_function_surface=_module_function_surface(module),
        protocol_function_surface=_module_function_surface(protocol),
        receipt_class_surface=tuple(sorted(vars(receipt_type).items())),
    )


def _require_scorer_api() -> _ScorerAPI:
    global _SCORER_API_CACHE

    _require_self_function_surface()
    with _SCORER_LOAD_LOCK:
        cached = _SCORER_API_CACHE
        if cached is None:
            cached = _validate_scorer_module(_load_scorer_module_once())
            _SCORER_API_CACHE = cached
        validated = _validate_scorer_module(cached.module)
        if not _same_scorer_api(cached, validated):
            raise ForagerMatchedV3LocalRewardBundleError(
                "canonical matched-v3 scorer cached API identity drifted"
            )
        return cached


@dataclass(frozen=True, slots=True)
class _HandoffFacts:
    candidate_id: str
    environment_seed: int
    agent_seed: int
    creation_pid: int
    handoff_source_sha256: str
    handoff_record_sha256: str
    handoff_record_body_sha256: str
    bootstrap_source_sha256: str
    bootstrap_descriptor_schema_version: str
    bootstrap_descriptor_sha256: str
    bootstrap_receipt_sha256: str
    bootstrap_receipt_body_sha256: str
    bootstrap_child_record_sha256: str
    bootstrap_child_record_body_sha256: str
    local_runner_receipt_sha256: str
    local_runner_receipt_body_sha256: str
    local_source_descriptor_sha256: str
    local_source_snapshot_source_sha256: str
    local_source_full_sha256: str
    local_source_tree_sha256: str
    local_runner_descriptor_sha256: str
    local_runner_source_sha256: str
    raw_trace_content_sha256: str


def _facts_from_handoff_content(
    *,
    content: Any,
    parser: types.FunctionType,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_full_sha256: str,
    expected_local_source_tree_sha256: str,
) -> tuple[_HandoffFacts, dict[str, bytes]]:
    byte_fields = {
        "bootstrap_receipt": _require_exact_bytes(
            getattr(content, "canonical_bootstrap_receipt_bytes", None),
            label="handoff bootstrap receipt",
            maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
            require_nonempty=True,
        ),
        "bootstrap_child_record": _require_exact_bytes(
            getattr(content, "canonical_bootstrap_child_record_bytes", None),
            label="handoff bootstrap child record",
            maximum_bytes=_MAX_CHILD_RECORD_BYTES,
            require_nonempty=True,
        ),
        "local_runner_receipt": _require_exact_bytes(
            getattr(content, "canonical_local_runner_receipt_bytes", None),
            label="handoff local runner receipt",
            maximum_bytes=_MAX_LOCAL_RECEIPT_BYTES,
            require_nonempty=True,
        ),
        "raw_trace": _require_exact_bytes(
            getattr(content, "raw_reward_trace_bytes", None),
            label="handoff raw reward trace",
            maximum_bytes=_MAX_RAW_TRACE_BYTES,
            require_nonempty=False,
        ),
        "stdout": _require_exact_bytes(
            getattr(content, "stdout_bytes", None),
            label="handoff stdout",
            maximum_bytes=_MAX_STDIO_BYTES,
            require_nonempty=False,
        ),
        "stderr": _require_exact_bytes(
            getattr(content, "stderr_bytes", None),
            label="handoff stderr",
            maximum_bytes=_MAX_STDIO_BYTES,
            require_nonempty=False,
        ),
    }
    record_raw = _require_exact_bytes(
        getattr(content, "canonical_handoff_record_bytes", None),
        label="canonical handoff record",
        maximum_bytes=_MAX_MANIFEST_BYTES,
        require_nonempty=True,
    )
    record_sha256 = _require_sha256(
        getattr(content, "handoff_record_sha256", None),
        "handoff record",
    )
    try:
        record_value = parser(
            record_raw,
            expected_record_sha256=record_sha256,
            bootstrap_receipt_bytes=byte_fields["bootstrap_receipt"],
            bootstrap_child_record_bytes=byte_fields["bootstrap_child_record"],
            local_runner_receipt_bytes=byte_fields["local_runner_receipt"],
            raw_reward_trace_bytes=byte_fields["raw_trace"],
            stdout_bytes=byte_fields["stdout"],
            stderr_bytes=byte_fields["stderr"],
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "authentic handoff record strict replay failed"
        ) from exc
    if type(record_value) is not dict:
        raise ForagerMatchedV3LocalRewardBundleError(
            "authentic handoff parser returned a non-plain record"
        )
    record = cast(dict[str, Any], record_value)
    cell = _require_exact_keys(
        record.get("cell"),
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "handoff cell",
    )
    candidate_id = _require_candidate_id(cell["candidate_id"])
    environment_seed = _require_uint31(cell["environment_seed"], "handoff environment seed")
    agent_seed = _require_uint31(cell["agent_seed"], "handoff agent seed")
    if (
        candidate_id != expected_candidate_id
        or environment_seed != expected_environment_seed
        or agent_seed != expected_agent_seed
        or getattr(content, "candidate_id", None) != candidate_id
        or getattr(content, "environment_seed", None) != environment_seed
        or getattr(content, "agent_seed", None) != agent_seed
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "handoff capability is cross-paired with a different candidate or seed cell"
        )
    provenance = _require_exact_keys(
        record.get("provenance"),
        frozenset(
            {
                "creation_pid",
                "authentic_bootstrap_outcome_consumed",
                "bootstrap_completion_returned_to_creation_caller",
                "handoff_pid_bound",
                "handoff_single_use",
                "content_access_requires_second_explicit_opt_in",
            }
        ),
        "handoff provenance",
    )
    creation_pid = provenance["creation_pid"]
    if (
        type(creation_pid) is not int
        or creation_pid != os.getpid()
        or getattr(content, "creation_pid", None) != creation_pid
        or provenance["authentic_bootstrap_outcome_consumed"] is not True
        or provenance["handoff_pid_bound"] is not True
        or provenance["handoff_single_use"] is not True
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "handoff capability provenance or PID binding drifted"
        )
    source_binding = _require_exact_keys(
        record.get("source_binding"),
        frozenset({"handoff_source_sha256", "bootstrap"}),
        "handoff source binding",
    )
    handoff_source = _require_sha256(
        source_binding["handoff_source_sha256"],
        "handoff source",
    )
    if not hmac.compare_digest(handoff_source, PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256):
        raise ForagerMatchedV3LocalRewardBundleError("handoff record source binding drifted")
    bootstrap_binding = _require_exact_keys(
        source_binding["bootstrap"],
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "source_sha256",
                "isolated_module_name",
            }
        ),
        "handoff bootstrap binding",
    )
    bootstrap_source = _require_sha256(
        bootstrap_binding["source_sha256"],
        "bootstrap source",
    )
    bootstrap_receipt = _strict_json_object(
        byte_fields["bootstrap_receipt"],
        label="bootstrap receipt",
        maximum_bytes=_MAX_BOOTSTRAP_RECEIPT_BYTES,
    )
    bootstrap_cell = _require_exact_keys(
        bootstrap_receipt.get("cell"),
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "bootstrap receipt cell",
    )
    if not _exact_json_equal(bootstrap_cell, cell):
        raise ForagerMatchedV3LocalRewardBundleError(
            "bootstrap receipt cell is cross-paired with the handoff"
        )
    snapshot = _require_exact_keys(
        bootstrap_receipt.get("source_snapshot"),
        frozenset(
            {
                "descriptor_sha256",
                "source_sha256",
                "expected_full_sha256",
                "expected_tree_sha256",
                "pre_full_sha256",
                "pre_tree_sha256",
                "post_full_sha256",
                "post_tree_sha256",
                "continuous_immutability_attested",
            }
        ),
        "bootstrap source snapshot",
    )
    snapshot_digests = {
        key: _require_sha256(snapshot[key], f"bootstrap source snapshot {key}")
        for key in (
            "descriptor_sha256",
            "source_sha256",
            "expected_full_sha256",
            "expected_tree_sha256",
            "pre_full_sha256",
            "pre_tree_sha256",
            "post_full_sha256",
            "post_tree_sha256",
        )
    }
    if (
        snapshot_digests["expected_full_sha256"] != expected_local_source_full_sha256
        or snapshot_digests["expected_tree_sha256"] != expected_local_source_tree_sha256
        or snapshot_digests["expected_full_sha256"] != snapshot_digests["pre_full_sha256"]
        or snapshot_digests["expected_full_sha256"] != snapshot_digests["post_full_sha256"]
        or snapshot_digests["expected_tree_sha256"] != snapshot_digests["pre_tree_sha256"]
        or snapshot_digests["expected_tree_sha256"] != snapshot_digests["post_tree_sha256"]
        or snapshot["continuous_immutability_attested"] is not False
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "handoff capability is cross-paired with a different local source tree"
        )
    runner = _require_exact_keys(
        bootstrap_receipt.get("runner"),
        frozenset({"descriptor_sha256", "source_sha256"}),
        "bootstrap local runner binding",
    )
    runner_descriptor = _require_sha256(
        runner["descriptor_sha256"],
        "local runner descriptor",
    )
    runner_source = _require_sha256(runner["source_sha256"], "local runner source")
    record_body_sha256 = _require_sha256(
        record.get("handoff_record_body_sha256"),
        "handoff record body",
    )
    bootstrap_body_sha256 = _require_sha256(
        bootstrap_receipt.get("receipt_body_sha256"),
        "bootstrap receipt body",
    )
    facts = _HandoffFacts(
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        creation_pid=creation_pid,
        handoff_source_sha256=handoff_source,
        handoff_record_sha256=record_sha256,
        handoff_record_body_sha256=record_body_sha256,
        bootstrap_source_sha256=bootstrap_source,
        bootstrap_descriptor_schema_version=cast(
            str,
            bootstrap_binding["descriptor_schema_version"],
        ),
        bootstrap_descriptor_sha256=_require_sha256(
            bootstrap_binding["descriptor_sha256"],
            "bootstrap descriptor",
        ),
        bootstrap_receipt_sha256=hashlib.sha256(byte_fields["bootstrap_receipt"]).hexdigest(),
        bootstrap_receipt_body_sha256=bootstrap_body_sha256,
        bootstrap_child_record_sha256=hashlib.sha256(
            byte_fields["bootstrap_child_record"]
        ).hexdigest(),
        bootstrap_child_record_body_sha256=_require_sha256(
            getattr(content, "bootstrap_child_record_body_sha256", None),
            "bootstrap child record body",
        ),
        local_runner_receipt_sha256=hashlib.sha256(
            byte_fields["local_runner_receipt"]
        ).hexdigest(),
        local_runner_receipt_body_sha256=_require_sha256(
            getattr(content, "local_runner_receipt_body_sha256", None),
            "local runner receipt body",
        ),
        local_source_descriptor_sha256=snapshot_digests["descriptor_sha256"],
        local_source_snapshot_source_sha256=snapshot_digests["source_sha256"],
        local_source_full_sha256=snapshot_digests["expected_full_sha256"],
        local_source_tree_sha256=snapshot_digests["expected_tree_sha256"],
        local_runner_descriptor_sha256=runner_descriptor,
        local_runner_source_sha256=runner_source,
        raw_trace_content_sha256=hashlib.sha256(byte_fields["raw_trace"]).hexdigest(),
    )
    return facts, byte_fields


def _file_size_bounds(role: str) -> tuple[int, int]:
    bounds = {
        "publication_manifest": (1, _MAX_MANIFEST_BYTES),
        "local_bundle_manifest": (1, _MAX_MANIFEST_BYTES),
        "bootstrap_receipt": (1, _MAX_BOOTSTRAP_RECEIPT_BYTES),
        "bootstrap_child_record": (1, _MAX_CHILD_RECORD_BYTES),
        "local_runner_receipt": (1, _MAX_LOCAL_RECEIPT_BYTES),
        "reward_trace": (
            PINNED_CANONICAL_NPZ_SIZE_BYTES,
            PINNED_CANONICAL_NPZ_SIZE_BYTES,
        ),
        "score_receipt": (1, _MAX_SCORE_RECEIPT_BYTES),
        "stdout": (0, _MAX_STDIO_BYTES),
        "stderr": (0, _MAX_STDIO_BYTES),
    }
    try:
        return bounds[role]
    except KeyError as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"unknown local reward bundle file role {role!r}"
        ) from exc


def _file_record(role: str, path: str, raw: bytes) -> dict[str, Any]:
    expected_path = dict(_ROLE_PATHS).get(role)
    if expected_path != path:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"local reward bundle role/path pairing drifted: {role}"
        )
    _require_portable_filename(path, f"local reward bundle {role} path")
    minimum, maximum = _file_size_bounds(role)
    if type(raw) is not bytes or not minimum <= len(raw) <= maximum:
        raise ForagerMatchedV3LocalRewardBundleError(
            f"local reward bundle {role} byte size is invalid"
        )
    return {
        "path": path,
        "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _payload_file_records(payloads: dict[str, bytes]) -> dict[str, Any]:
    if type(payloads) is not dict or set(payloads) != {
        role for role, _path in _PAYLOAD_ROLE_PATHS
    }:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle payload byte roles are not exact"
        )
    return {
        role: _file_record(role, path, payloads[role])
        for role, path in _PAYLOAD_ROLE_PATHS
    }


def _manifest_source_binding(
    *,
    facts: _HandoffFacts,
    bundle_source_sha256: str,
) -> dict[str, Any]:
    return {
        "bundle_source_sha256": bundle_source_sha256,
        "handoff": {
            "descriptor_schema_version": (
                PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256,
            "source_sha256": facts.handoff_source_sha256,
            "record_schema_version": PINNED_LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION,
            "record_full_file_sha256": facts.handoff_record_sha256,
            "record_body_sha256": facts.handoff_record_body_sha256,
        },
        "bootstrap": {
            "descriptor_schema_version": facts.bootstrap_descriptor_schema_version,
            "descriptor_sha256": facts.bootstrap_descriptor_sha256,
            "source_sha256": facts.bootstrap_source_sha256,
            "receipt_full_file_sha256": facts.bootstrap_receipt_sha256,
            "receipt_body_sha256": facts.bootstrap_receipt_body_sha256,
            "child_record_full_file_sha256": facts.bootstrap_child_record_sha256,
            "child_record_body_sha256": facts.bootstrap_child_record_body_sha256,
        },
        "local_source_tree": {
            "descriptor_sha256": facts.local_source_descriptor_sha256,
            "snapshot_source_sha256": facts.local_source_snapshot_source_sha256,
            "full_sha256": facts.local_source_full_sha256,
            "tree_sha256": facts.local_source_tree_sha256,
            "pre_post_equal": True,
            "continuous_immutability_attested": False,
        },
        "local_runner": {
            "descriptor_sha256": facts.local_runner_descriptor_sha256,
            "source_sha256": facts.local_runner_source_sha256,
            "receipt_full_file_sha256": facts.local_runner_receipt_sha256,
            "receipt_body_sha256": facts.local_runner_receipt_body_sha256,
        },
        "scorer": {
            "source_path": PINNED_SCORER_SOURCE_PATH,
            "source_sha256": PINNED_SCORER_SOURCE_SHA256,
            "protocol_source_path": PINNED_SCORER_PROTOCOL_SOURCE_PATH,
            "protocol_source_sha256": PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
            "metric_descriptor_schema_version": (
                PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION
            ),
            "metric_descriptor_sha256": PINNED_SCORER_METRIC_DESCRIPTOR_SHA256,
            "score_receipt_schema_version": PINNED_SCORE_RECEIPT_SCHEMA_VERSION,
            "reward_npz_schema_version": PINNED_REWARD_NPZ_SCHEMA_VERSION,
            "semantic_surface_schema_version": (
                PINNED_SCORER_SEMANTIC_SURFACE_SCHEMA_VERSION
            ),
            "semantic_surface_sha256": PINNED_SCORER_SEMANTIC_SURFACE_SHA256,
        },
    }


def _local_manifest_body(
    *,
    facts: _HandoffFacts,
    bundle_source_sha256: str,
    payloads: dict[str, bytes],
) -> dict[str, Any]:
    files = _payload_file_records(payloads)
    reward_record = cast(dict[str, Any], files["reward_trace"])
    score_record = cast(dict[str, Any], files["score_receipt"])
    return {
        "schema_version": LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "classification": (
            "score_bearing_permanently_nonqualifying_inspection_content_non_authorizing"
        ),
        "descriptor_binding": {
            "schema_version": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
        },
        "cell": {
            "candidate_id": facts.candidate_id,
            "environment_seed": facts.environment_seed,
            "agent_seed": facts.agent_seed,
        },
        "source_binding": _manifest_source_binding(
            facts=facts,
            bundle_source_sha256=bundle_source_sha256,
        ),
        "provenance": {
            "creation_pid": facts.creation_pid,
            "authentic_handoff_capability_consumed": True,
            "handoff_content_access_opt_in_applied_internally": True,
            "bundle_issuance_explicit_opt_in": True,
            "bundle_capability_pid_bound": True,
            "bundle_capability_single_use": True,
            "bundle_content_access_requires_separate_explicit_opt_in": True,
        },
        "files": files,
        "inventory": _inventory_descriptor(),
        "inspection": _inspection_contract(),
        "qualification_publisher": _qualification_publisher_contract(),
        "scorer_output": {
            "reward_artifact_sha256": reward_record["sha256"],
            "reward_artifact_size_bytes": reward_record["size_bytes"],
            "score_receipt_full_file_sha256": score_record["sha256"],
            "score_receipt_size_bytes": score_record["size_bytes"],
            "raw_trace_content_sha256_before_scorer_byte_conversion": (
                facts.raw_trace_content_sha256
            ),
            "module_decodes_score_fields": False,
            "module_decodes_reward_npz": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _with_body_digest(
    body: dict[str, Any],
    *,
    digest_key: str,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    body_raw = _canonical_json(body, maximum_bytes=maximum_bytes)
    body_sha256 = hashlib.sha256(body_raw).hexdigest()
    payload = dict(body)
    payload[digest_key] = body_sha256
    return _canonical_json(payload, maximum_bytes=maximum_bytes), body_sha256


def _publication_body(
    *,
    facts: _HandoffFacts,
    bundle_source_sha256: str,
    local_manifest: bytes,
    local_manifest_body_sha256: str,
    payloads: dict[str, bytes],
) -> dict[str, Any]:
    bound_payloads = {"local_bundle_manifest": local_manifest, **payloads}
    files = {
        role: _file_record(role, path, bound_payloads[role])
        for role, path in _PUBLICATION_BOUND_ROLE_PATHS
    }
    return {
        "schema_version": LOCAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "classification": (
            "score_bearing_prepublication_payload_plan_permanently_nonqualifying"
        ),
        "cell": {
            "candidate_id": facts.candidate_id,
            "environment_seed": facts.environment_seed,
            "agent_seed": facts.agent_seed,
        },
        "bundle_binding": {
            "descriptor_schema_version": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
            "implementation_source_sha256": bundle_source_sha256,
            "manifest_schema_version": LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "manifest_body_sha256": local_manifest_body_sha256,
            "manifest_full_file_sha256": hashlib.sha256(local_manifest).hexdigest(),
            "public_bundle_object_permanently_nonqualifying": True,
            "qualification_publisher_may_accept_public_bundle_object": False,
            "qualification_publisher_must_consume_live_capability_directly": True,
        },
        "handoff_record_sha256": facts.handoff_record_sha256,
        "local_source_tree_sha256": facts.local_source_tree_sha256,
        "files": files,
        "inventory": _inventory_descriptor(),
        "inspection": _inspection_contract(),
        "qualification_publisher": _qualification_publisher_contract(),
        "writer_contract": {
            "filesystem_writes_performed": False,
            "durability_claimed": False,
            "atomic_no_replace_publisher_required": True,
            "flat_owned_directory_required": True,
            "symlinks_allowed": False,
            "publication_manifest_self_digest_in_file": True,
            "publication_manifest_omitted_from_its_files_map_to_avoid_self_reference": True,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _validate_claims_and_limitations(value: dict[str, Any], *, label: str) -> None:
    claims = _require_exact_keys(value.get("claims"), frozenset(_claims()), f"{label} claims")
    if any(item is not False for item in claims.values()):
        raise ForagerMatchedV3LocalRewardBundleError(
            f"{label} claims must remain exact false booleans"
        )
    if not _exact_json_equal(value.get("limitations"), _limitations()):
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} limitations drifted")


def _validate_inventory(value: Any, *, label: str) -> None:
    exact = _require_exact_keys(
        value,
        frozenset(
            {
                "file_count",
                "exact_filenames",
                "flat_directory_only",
                "path_traversal_allowed",
                "symlinks_allowed",
                "stdout_and_stderr_may_be_zero_length",
            }
        ),
        label,
    )
    if not _exact_json_equal(exact, _inventory_descriptor()):
        raise ForagerMatchedV3LocalRewardBundleError(f"{label} drifted")
    for filename in _EXACT_FILENAMES:
        _require_portable_filename(filename, f"{label} filename")


def _validate_file_records(
    value: Any,
    *,
    role_paths: tuple[tuple[str, str], ...],
    label: str,
) -> dict[str, Any]:
    expected_roles = frozenset(role for role, _path in role_paths)
    files = _require_exact_keys(value, expected_roles, label)
    for role, path in role_paths:
        record = _require_exact_keys(
            files[role],
            frozenset({"path", "role", "sha256", "size_bytes"}),
            f"{label} {role}",
        )
        if record["role"] != role or record["path"] != path:
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} {role} path or role drifted"
            )
        _require_portable_filename(record["path"], f"{label} {role} path")
        _require_sha256(record["sha256"], f"{label} {role} digest")
        minimum, maximum = _file_size_bounds(role)
        size = record["size_bytes"]
        if type(size) is not int or not minimum <= size <= maximum:
            raise ForagerMatchedV3LocalRewardBundleError(
                f"{label} {role} size is invalid"
            )
    return files


def _validate_cell(value: Any, *, label: str) -> dict[str, Any]:
    cell = _require_exact_keys(
        value,
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        label,
    )
    _require_candidate_id(cell["candidate_id"])
    _require_uint31(cell["environment_seed"], f"{label} environment seed")
    _require_uint31(cell["agent_seed"], f"{label} agent seed")
    return cell


def _validate_source_binding(value: Any) -> dict[str, Any]:
    source = _require_exact_keys(
        value,
        frozenset(
            {
                "bundle_source_sha256",
                "handoff",
                "bootstrap",
                "local_source_tree",
                "local_runner",
                "scorer",
            }
        ),
        "local bundle source binding",
    )
    _require_sha256(source["bundle_source_sha256"], "local bundle source")
    handoff = _require_exact_keys(
        source["handoff"],
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "source_sha256",
                "record_schema_version",
                "record_full_file_sha256",
                "record_body_sha256",
            }
        ),
        "local bundle handoff binding",
    )
    if (
        handoff["descriptor_schema_version"]
        != PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION
        or handoff["descriptor_sha256"]
        != PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256
        or handoff["source_sha256"] != PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256
        or handoff["record_schema_version"]
        != PINNED_LOCAL_RESULT_HANDOFF_RECORD_SCHEMA_VERSION
    ):
        raise ForagerMatchedV3LocalRewardBundleError("local bundle handoff binding drifted")
    _require_sha256(handoff["record_full_file_sha256"], "handoff record file")
    _require_sha256(handoff["record_body_sha256"], "handoff record body")
    bootstrap = _require_exact_keys(
        source["bootstrap"],
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "source_sha256",
                "receipt_full_file_sha256",
                "receipt_body_sha256",
                "child_record_full_file_sha256",
                "child_record_body_sha256",
            }
        ),
        "local bundle bootstrap binding",
    )
    if type(bootstrap["descriptor_schema_version"]) is not str:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle bootstrap descriptor schema is not exact"
        )
    for key in (
        "descriptor_sha256",
        "source_sha256",
        "receipt_full_file_sha256",
        "receipt_body_sha256",
        "child_record_full_file_sha256",
        "child_record_body_sha256",
    ):
        _require_sha256(bootstrap[key], f"local bundle bootstrap {key}")
    local_source = _require_exact_keys(
        source["local_source_tree"],
        frozenset(
            {
                "descriptor_sha256",
                "snapshot_source_sha256",
                "full_sha256",
                "tree_sha256",
                "pre_post_equal",
                "continuous_immutability_attested",
            }
        ),
        "local bundle source tree binding",
    )
    for key in ("descriptor_sha256", "snapshot_source_sha256", "full_sha256", "tree_sha256"):
        _require_sha256(local_source[key], f"local bundle source tree {key}")
    if (
        local_source["pre_post_equal"] is not True
        or local_source["continuous_immutability_attested"] is not False
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle source tree claims drifted"
        )
    runner = _require_exact_keys(
        source["local_runner"],
        frozenset(
            {
                "descriptor_sha256",
                "source_sha256",
                "receipt_full_file_sha256",
                "receipt_body_sha256",
            }
        ),
        "local bundle runner binding",
    )
    for key in runner:
        _require_sha256(runner[key], f"local bundle runner {key}")
    scorer = _require_exact_keys(
        source["scorer"],
        frozenset(
            {
                "source_path",
                "source_sha256",
                "protocol_source_path",
                "protocol_source_sha256",
                "metric_descriptor_schema_version",
                "metric_descriptor_sha256",
                "score_receipt_schema_version",
                "reward_npz_schema_version",
                "semantic_surface_schema_version",
                "semantic_surface_sha256",
            }
        ),
        "local bundle scorer binding",
    )
    if not _exact_json_equal(
        scorer,
        {
            "source_path": PINNED_SCORER_SOURCE_PATH,
            "source_sha256": PINNED_SCORER_SOURCE_SHA256,
            "protocol_source_path": PINNED_SCORER_PROTOCOL_SOURCE_PATH,
            "protocol_source_sha256": PINNED_SCORER_PROTOCOL_SOURCE_SHA256,
            "metric_descriptor_schema_version": (
                PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION
            ),
            "metric_descriptor_sha256": PINNED_SCORER_METRIC_DESCRIPTOR_SHA256,
            "score_receipt_schema_version": PINNED_SCORE_RECEIPT_SCHEMA_VERSION,
            "reward_npz_schema_version": PINNED_REWARD_NPZ_SCHEMA_VERSION,
            "semantic_surface_schema_version": (
                PINNED_SCORER_SEMANTIC_SURFACE_SCHEMA_VERSION
            ),
            "semantic_surface_sha256": PINNED_SCORER_SEMANTIC_SURFACE_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalRewardBundleError("local bundle scorer binding drifted")
    return source


def parse_matched_v3_local_reward_bundle_manifest(
    raw: bytes,
    *,
    expected_full_file_sha256: str,
) -> dict[str, Any]:
    """Strictly replay one detached local bundle manifest without authority."""

    _require_self_function_surface()
    expected = _require_sha256(expected_full_file_sha256, "expected local manifest file")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        expected,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle manifest disagrees with its caller-carried file digest"
        )
    value = _strict_json_object(
        raw,
        label="local reward bundle manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "classification",
                "descriptor_binding",
                "cell",
                "source_binding",
                "provenance",
                "files",
                "inventory",
                "inspection",
                "qualification_publisher",
                "scorer_output",
                "claims",
                "limitations",
                "manifest_body_sha256",
            }
        ),
        "local reward bundle manifest",
    )
    if (
        manifest["schema_version"] != LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        or manifest["classification"]
        != (
            "score_bearing_permanently_nonqualifying_inspection_content_non_authorizing"
        )
        or not _exact_json_equal(
            manifest["descriptor_binding"],
            {
                "schema_version": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
            },
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle manifest fixed identity drifted"
        )
    _validate_cell(manifest["cell"], label="local bundle cell")
    source = _validate_source_binding(manifest["source_binding"])
    provenance = _require_exact_keys(
        manifest["provenance"],
        frozenset(
            {
                "creation_pid",
                "authentic_handoff_capability_consumed",
                "handoff_content_access_opt_in_applied_internally",
                "bundle_issuance_explicit_opt_in",
                "bundle_capability_pid_bound",
                "bundle_capability_single_use",
                "bundle_content_access_requires_separate_explicit_opt_in",
            }
        ),
        "local bundle provenance",
    )
    if (
        type(provenance["creation_pid"]) is not int
        or provenance["creation_pid"] <= 0
        or any(
            provenance[key] is not True
            for key in provenance
            if key != "creation_pid"
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError("local bundle provenance drifted")
    files = _validate_file_records(
        manifest["files"],
        role_paths=_PAYLOAD_ROLE_PATHS,
        label="local bundle files",
    )
    _validate_inventory(manifest["inventory"], label="local bundle inventory")
    if not _exact_json_equal(manifest["inspection"], _inspection_contract()):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle inspection contract drifted"
        )
    if not _exact_json_equal(
        manifest["qualification_publisher"],
        _qualification_publisher_contract(),
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle qualification publisher contract drifted"
        )
    scorer_output = _require_exact_keys(
        manifest["scorer_output"],
        frozenset(
            {
                "reward_artifact_sha256",
                "reward_artifact_size_bytes",
                "score_receipt_full_file_sha256",
                "score_receipt_size_bytes",
                "raw_trace_content_sha256_before_scorer_byte_conversion",
                "module_decodes_score_fields",
                "module_decodes_reward_npz",
            }
        ),
        "local bundle scorer output",
    )
    reward_record = cast(dict[str, Any], files["reward_trace"])
    score_record = cast(dict[str, Any], files["score_receipt"])
    if (
        scorer_output["reward_artifact_sha256"] != reward_record["sha256"]
        or scorer_output["reward_artifact_size_bytes"] != reward_record["size_bytes"]
        or scorer_output["score_receipt_full_file_sha256"] != score_record["sha256"]
        or scorer_output["score_receipt_size_bytes"] != score_record["size_bytes"]
        or scorer_output["module_decodes_score_fields"] is not False
        or scorer_output["module_decodes_reward_npz"] is not False
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle scorer output binding drifted"
        )
    _require_sha256(
        scorer_output["raw_trace_content_sha256_before_scorer_byte_conversion"],
        "local bundle pre-conversion raw trace content",
    )
    runner = cast(dict[str, Any], source["local_runner"])
    bootstrap = cast(dict[str, Any], source["bootstrap"])
    if (
        cast(dict[str, Any], files["local_runner_receipt"])["sha256"]
        != runner["receipt_full_file_sha256"]
        or cast(dict[str, Any], files["bootstrap_receipt"])["sha256"]
        != bootstrap["receipt_full_file_sha256"]
        or cast(dict[str, Any], files["bootstrap_child_record"])["sha256"]
        != bootstrap["child_record_full_file_sha256"]
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle receipt file/source bindings disagree"
        )
    _validate_claims_and_limitations(manifest, label="local bundle manifest")
    supplied_body = _require_sha256(
        manifest["manifest_body_sha256"],
        "local bundle manifest body",
    )
    body = dict(manifest)
    del body["manifest_body_sha256"]
    if not hmac.compare_digest(
        hashlib.sha256(
            _canonical_json(body, maximum_bytes=_MAX_MANIFEST_BYTES)
        ).hexdigest(),
        supplied_body,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local bundle manifest body digest does not replay"
        )
    return manifest


def parse_matched_v3_local_reward_publication_manifest(
    raw: bytes,
    *,
    expected_full_file_sha256: str,
) -> dict[str, Any]:
    """Replay the not-yet-published outer payload manifest without authority."""

    _require_self_function_surface()
    expected = _require_sha256(expected_full_file_sha256, "expected publication file")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        expected,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication payload manifest disagrees with its caller-carried digest"
        )
    value = _strict_json_object(
        raw,
        label="local reward publication payload manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    publication = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "classification",
                "cell",
                "bundle_binding",
                "handoff_record_sha256",
                "local_source_tree_sha256",
                "files",
                "inventory",
                "inspection",
                "qualification_publisher",
                "writer_contract",
                "claims",
                "limitations",
                "publication_body_sha256",
            }
        ),
        "local reward publication payload manifest",
    )
    if (
        publication["schema_version"]
        != LOCAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION
        or publication["classification"]
        != (
            "score_bearing_prepublication_payload_plan_permanently_nonqualifying"
        )
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward publication payload identity drifted"
        )
    _validate_cell(publication["cell"], label="publication payload cell")
    binding = _require_exact_keys(
        publication["bundle_binding"],
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "implementation_source_sha256",
                "manifest_schema_version",
                "manifest_body_sha256",
                "manifest_full_file_sha256",
                "public_bundle_object_permanently_nonqualifying",
                "qualification_publisher_may_accept_public_bundle_object",
                "qualification_publisher_must_consume_live_capability_directly",
            }
        ),
        "publication bundle binding",
    )
    if (
        binding["descriptor_schema_version"]
        != LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        or binding["descriptor_sha256"] != LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256
        or binding["manifest_schema_version"]
        != LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        or binding["public_bundle_object_permanently_nonqualifying"] is not True
        or binding["qualification_publisher_may_accept_public_bundle_object"] is not False
        or binding["qualification_publisher_must_consume_live_capability_directly"]
        is not True
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication bundle fixed binding drifted"
        )
    for key in (
        "implementation_source_sha256",
        "manifest_body_sha256",
        "manifest_full_file_sha256",
    ):
        _require_sha256(binding[key], f"publication bundle {key}")
    _require_sha256(publication["handoff_record_sha256"], "publication handoff record")
    _require_sha256(
        publication["local_source_tree_sha256"],
        "publication local source tree",
    )
    files = _validate_file_records(
        publication["files"],
        role_paths=_PUBLICATION_BOUND_ROLE_PATHS,
        label="publication payload files",
    )
    manifest_record = cast(dict[str, Any], files["local_bundle_manifest"])
    if binding["manifest_full_file_sha256"] != manifest_record["sha256"]:
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication bundle manifest digest bindings disagree"
        )
    _validate_inventory(publication["inventory"], label="publication inventory")
    if not _exact_json_equal(publication["inspection"], _inspection_contract()):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication inspection contract drifted"
        )
    if not _exact_json_equal(
        publication["qualification_publisher"],
        _qualification_publisher_contract(),
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication qualification publisher contract drifted"
        )
    writer = _require_exact_keys(
        publication["writer_contract"],
        frozenset(
            {
                "filesystem_writes_performed",
                "durability_claimed",
                "atomic_no_replace_publisher_required",
                "flat_owned_directory_required",
                "symlinks_allowed",
                "publication_manifest_self_digest_in_file",
                "publication_manifest_omitted_from_its_files_map_to_avoid_self_reference",
            }
        ),
        "publication writer contract",
    )
    expected_writer = {
        "filesystem_writes_performed": False,
        "durability_claimed": False,
        "atomic_no_replace_publisher_required": True,
        "flat_owned_directory_required": True,
        "symlinks_allowed": False,
        "publication_manifest_self_digest_in_file": True,
        "publication_manifest_omitted_from_its_files_map_to_avoid_self_reference": True,
    }
    if not _exact_json_equal(writer, expected_writer):
        raise ForagerMatchedV3LocalRewardBundleError("publication writer contract drifted")
    _validate_claims_and_limitations(publication, label="publication payload manifest")
    supplied_body = _require_sha256(
        publication["publication_body_sha256"],
        "publication payload body",
    )
    body = dict(publication)
    del body["publication_body_sha256"]
    if not hmac.compare_digest(
        hashlib.sha256(
            _canonical_json(body, maximum_bytes=_MAX_MANIFEST_BYTES)
        ).hexdigest(),
        supplied_body,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication payload body digest does not replay"
        )
    return publication


@dataclass(frozen=True, slots=True)
class MatchedV3LocalRewardBundleFile:
    """One immutable file identity in the exact local bundle inventory."""

    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or dict(_ROLE_PATHS).get(self.role) != self.path:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle inventory role/path pairing is invalid"
            )
        _require_portable_filename(self.path, f"local bundle inventory {self.role}")
        minimum, maximum = _file_size_bounds(self.role)
        if type(self.size_bytes) is not int or not minimum <= self.size_bytes <= maximum:
            raise ForagerMatchedV3LocalRewardBundleError(
                f"local reward bundle inventory {self.role} size is invalid"
            )
        _require_sha256(self.sha256, f"local reward bundle inventory {self.role}")


@dataclass(frozen=True, slots=True)
class MatchedV3LocalRewardBundle:
    """Score-bearing, permanently nonqualifying public inspection bytes.

    The score receipt and uncompressed reward NPZ are parseable by callers even
    though this module does not decode them.  A qualification publisher must
    never accept this object or its serialized bytes; it must consume the live
    capability directly before public inspection consumes that single-use path.
    """

    candidate_id: str
    environment_seed: int
    agent_seed: int
    creation_pid: int
    bundle_source_sha256: str
    handoff_record_sha256: str
    local_source_full_sha256: str
    local_source_tree_sha256: str
    publication_manifest_bytes: bytes
    local_bundle_manifest_bytes: bytes
    bootstrap_receipt_bytes: bytes
    bootstrap_child_record_bytes: bytes
    local_runner_receipt_bytes: bytes
    reward_artifact_bytes: bytes
    score_receipt_bytes: bytes
    stdout_bytes: bytes
    stderr_bytes: bytes
    inventory: tuple[MatchedV3LocalRewardBundleFile, ...]

    def __post_init__(self) -> None:
        _validate_bundle_structure(self, scorer_api=None)

    def file_bytes(self, path: str) -> bytes:
        """Return score-bearing inspection bytes; this grants no authority."""

        _require_self_function_surface()
        _require_portable_filename(path, "local reward bundle file lookup")
        by_path = {
            PUBLICATION_MANIFEST_FILENAME: self.publication_manifest_bytes,
            LOCAL_BUNDLE_MANIFEST_FILENAME: self.local_bundle_manifest_bytes,
            BOOTSTRAP_RECEIPT_FILENAME: self.bootstrap_receipt_bytes,
            BOOTSTRAP_CHILD_RECORD_FILENAME: self.bootstrap_child_record_bytes,
            LOCAL_RUNNER_RECEIPT_FILENAME: self.local_runner_receipt_bytes,
            REWARD_TRACE_FILENAME: self.reward_artifact_bytes,
            SCORE_RECEIPT_FILENAME: self.score_receipt_bytes,
            STDOUT_FILENAME: self.stdout_bytes,
            STDERR_FILENAME: self.stderr_bytes,
        }
        try:
            return by_path[path]
        except KeyError as exc:
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle file lookup is outside the exact inventory"
            ) from exc


def _bundle_role_bytes(bundle: MatchedV3LocalRewardBundle) -> dict[str, bytes]:
    return {
        "publication_manifest": bundle.publication_manifest_bytes,
        "local_bundle_manifest": bundle.local_bundle_manifest_bytes,
        "bootstrap_receipt": bundle.bootstrap_receipt_bytes,
        "bootstrap_child_record": bundle.bootstrap_child_record_bytes,
        "local_runner_receipt": bundle.local_runner_receipt_bytes,
        "reward_trace": bundle.reward_artifact_bytes,
        "score_receipt": bundle.score_receipt_bytes,
        "stdout": bundle.stdout_bytes,
        "stderr": bundle.stderr_bytes,
    }


def _inventory_from_role_bytes(
    role_bytes: dict[str, bytes],
) -> tuple[MatchedV3LocalRewardBundleFile, ...]:
    if type(role_bytes) is not dict or set(role_bytes) != {role for role, _path in _ROLE_PATHS}:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle role byte inventory is not exact"
        )
    return tuple(
        MatchedV3LocalRewardBundleFile(
            role=role,
            path=path,
            size_bytes=len(role_bytes[role]),
            sha256=hashlib.sha256(role_bytes[role]).hexdigest(),
        )
        for role, path in _ROLE_PATHS
    )


def _same_scorer_api(first: _ScorerAPI, second: _ScorerAPI) -> bool:
    scorer_surface_equal = (
        len(first.scorer_function_surface) == len(second.scorer_function_surface)
        and all(
            first_name == second_name
            and first_function is second_function
            and first_code is second_code
            for (first_name, first_function, first_code), (
                second_name,
                second_function,
                second_code,
            ) in zip(
                first.scorer_function_surface,
                second.scorer_function_surface,
                strict=True,
            )
        )
    )
    protocol_surface_equal = (
        len(first.protocol_function_surface) == len(second.protocol_function_surface)
        and all(
            first_name == second_name
            and first_function is second_function
            and first_code is second_code
            for (first_name, first_function, first_code), (
                second_name,
                second_function,
                second_code,
            ) in zip(
                first.protocol_function_surface,
                second.protocol_function_surface,
                strict=True,
            )
        )
    )
    receipt_surface_equal = (
        len(first.receipt_class_surface) == len(second.receipt_class_surface)
        and all(
            first_name == second_name and first_value is second_value
            for (first_name, first_value), (second_name, second_value) in zip(
                first.receipt_class_surface,
                second.receipt_class_surface,
                strict=True,
            )
        )
    )
    return bool(
        first.module is second.module
        and first.protocol_module is second.protocol_module
        and first.encode is second.encode
        and first.ingest is second.ingest
        and first.parse is second.parse
        and first.receipt_type is second.receipt_type
        and first.receipt_canonical_json is second.receipt_canonical_json
        and hmac.compare_digest(
            first.semantic_surface_sha256,
            second.semantic_surface_sha256,
        )
        and scorer_surface_equal
        and protocol_surface_equal
        and receipt_surface_equal
    )


def _scorer_byte_replay_without_module_decode(
    *,
    scorer_api: _ScorerAPI,
    reward_artifact: bytes,
    score_receipt: bytes,
) -> None:
    if (
        type(reward_artifact) is not bytes
        or len(reward_artifact) != PINNED_CANONICAL_NPZ_SIZE_BYTES
        or type(score_receipt) is not bytes
        or not 1 <= len(score_receipt) <= _MAX_SCORE_RECEIPT_BYTES
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "scorer byte-replay artifact shapes are invalid"
        )
    try:
        parsed = scorer_api.parse(score_receipt)
        replayed = scorer_api.ingest(reward_artifact)
        if (
            type(parsed) is not scorer_api.receipt_type
            or type(replayed) is not scorer_api.receipt_type
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                "scorer byte replay returned a non-exact receipt type"
            )
        parsed_raw = scorer_api.receipt_canonical_json(parsed)
        replayed_raw = scorer_api.receipt_canonical_json(replayed)
    except ForagerMatchedV3LocalRewardBundleError:
        raise
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer byte replay rejected the bundle"
        ) from exc
    if (
        type(parsed_raw) is not bytes
        or type(replayed_raw) is not bytes
        or not hmac.compare_digest(parsed_raw, score_receipt)
        or not hmac.compare_digest(replayed_raw, score_receipt)
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer byte receipt replay disagrees"
        )


def _validate_bundle_structure(
    bundle: object,
    *,
    scorer_api: _ScorerAPI | None,
) -> MatchedV3LocalRewardBundle:
    _require_self_function_surface()
    if type(bundle) is not MatchedV3LocalRewardBundle:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle must be the exact immutable bundle type"
        )
    exact = bundle
    candidate_id = _require_candidate_id(exact.candidate_id)
    environment_seed = _require_uint31(exact.environment_seed, "bundle environment seed")
    agent_seed = _require_uint31(exact.agent_seed, "bundle agent seed")
    if type(exact.creation_pid) is not int or exact.creation_pid <= 0:
        raise ForagerMatchedV3LocalRewardBundleError("bundle creation PID is invalid")
    bundle_source = _require_sha256(exact.bundle_source_sha256, "bundle source")
    handoff_record = _require_sha256(exact.handoff_record_sha256, "bundle handoff record")
    local_full = _require_sha256(exact.local_source_full_sha256, "bundle source full")
    local_tree = _require_sha256(exact.local_source_tree_sha256, "bundle source tree")
    role_bytes = _bundle_role_bytes(exact)
    expected_inventory = _inventory_from_role_bytes(role_bytes)
    if (
        type(exact.inventory) is not tuple
        or len(exact.inventory) != len(expected_inventory)
        or any(type(item) is not MatchedV3LocalRewardBundleFile for item in exact.inventory)
        or exact.inventory != expected_inventory
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle immutable inventory does not replay from its bytes"
        )
    local_manifest = parse_matched_v3_local_reward_bundle_manifest(
        exact.local_bundle_manifest_bytes,
        expected_full_file_sha256=hashlib.sha256(
            exact.local_bundle_manifest_bytes
        ).hexdigest(),
    )
    publication = parse_matched_v3_local_reward_publication_manifest(
        exact.publication_manifest_bytes,
        expected_full_file_sha256=hashlib.sha256(
            exact.publication_manifest_bytes
        ).hexdigest(),
    )
    expected_cell = {
        "candidate_id": candidate_id,
        "environment_seed": environment_seed,
        "agent_seed": agent_seed,
    }
    source = cast(dict[str, Any], local_manifest["source_binding"])
    source_tree = cast(dict[str, Any], source["local_source_tree"])
    handoff = cast(dict[str, Any], source["handoff"])
    provenance = cast(dict[str, Any], local_manifest["provenance"])
    publication_binding = cast(dict[str, Any], publication["bundle_binding"])
    if not hmac.compare_digest(
        publication_binding["manifest_body_sha256"],
        local_manifest["manifest_body_sha256"],
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "publication/local manifest body binding drifted"
        )
    if (
        not _exact_json_equal(local_manifest["cell"], expected_cell)
        or not _exact_json_equal(publication["cell"], expected_cell)
        or source["bundle_source_sha256"] != bundle_source
        or source_tree["full_sha256"] != local_full
        or source_tree["tree_sha256"] != local_tree
        or handoff["record_full_file_sha256"] != handoff_record
        or publication["handoff_record_sha256"] != handoff_record
        or publication["local_source_tree_sha256"] != local_tree
        or provenance["creation_pid"] != exact.creation_pid
        or publication_binding["implementation_source_sha256"] != bundle_source
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle cell, source, PID, or handoff binding drifted"
        )
    local_files = cast(dict[str, Any], local_manifest["files"])
    for role, _path in _PAYLOAD_ROLE_PATHS:
        if not _exact_json_equal(
            local_files[role],
            _file_record(role, dict(_ROLE_PATHS)[role], role_bytes[role]),
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"local reward bundle manifest file binding drifted: {role}"
            )
    publication_files = cast(dict[str, Any], publication["files"])
    for role, _path in _PUBLICATION_BOUND_ROLE_PATHS:
        if not _exact_json_equal(
            publication_files[role],
            _file_record(role, dict(_ROLE_PATHS)[role], role_bytes[role]),
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"local reward publication file binding drifted: {role}"
            )
    if scorer_api is not None:
        _scorer_byte_replay_without_module_decode(
            scorer_api=scorer_api,
            reward_artifact=exact.reward_artifact_bytes,
            score_receipt=exact.score_receipt_bytes,
        )
    return exact


class _LocalRewardBundleCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local reward bundle capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("local reward bundle capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("local reward bundle capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("local reward bundle capabilities cannot be serialized")


@dataclass(frozen=True, slots=True)
class _SealedLocalRewardPayload:
    """Private score-bearing payload retained until exactly one path wins."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    creation_pid: int
    bundle_source_sha256: str
    handoff_record_sha256: str
    local_source_full_sha256: str
    local_source_tree_sha256: str
    role_payloads: tuple[tuple[str, str, bytes], ...]


@dataclass(slots=True)
class _BundleState:
    pid: int
    status: Literal["live", "consumed"]
    handoff_capability: object
    handoff_capability_identity: int
    handoff_content: object
    handoff_content_identity: int
    bundle_source_sha256: str
    scorer_api: _ScorerAPI
    sealed_payload: _SealedLocalRewardPayload
    sealed_payload_identity: int
    file_sha256: tuple[str, ...]


_CAPABILITY_LOCK: Final = threading.Lock()
_BUNDLE_CAPABILITIES: Final[
    weakref.WeakKeyDictionary[_LocalRewardBundleCapability, _BundleState]
] = weakref.WeakKeyDictionary()


def _sealed_role_bytes(payload: _SealedLocalRewardPayload) -> dict[str, bytes]:
    if type(payload.role_payloads) is not tuple or len(payload.role_payloads) != len(
        _ROLE_PATHS
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "sealed local reward payload inventory is not exact"
        )
    result: dict[str, bytes] = {}
    for item, (expected_role, expected_path) in zip(
        payload.role_payloads,
        _ROLE_PATHS,
        strict=True,
    ):
        if (
            type(item) is not tuple
            or len(item) != 3
            or item[0] != expected_role
            or item[1] != expected_path
            or type(item[2]) is not bytes
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                "sealed local reward payload order, path, or byte type drifted"
            )
        result[expected_role] = item[2]
    return result


def _validate_sealed_payload(
    payload: object,
    *,
    scorer_api: _ScorerAPI,
) -> _SealedLocalRewardPayload:
    """Replay a private payload without constructing the public bundle type."""

    _require_self_function_surface()
    if type(payload) is not _SealedLocalRewardPayload:
        raise ForagerMatchedV3LocalRewardBundleError(
            "sealed local reward payload type is not exact"
        )
    exact = payload
    candidate_id = _require_candidate_id(exact.candidate_id)
    environment_seed = _require_uint31(exact.environment_seed, "sealed environment seed")
    agent_seed = _require_uint31(exact.agent_seed, "sealed agent seed")
    if type(exact.creation_pid) is not int or exact.creation_pid <= 0:
        raise ForagerMatchedV3LocalRewardBundleError("sealed creation PID is invalid")
    bundle_source = _require_sha256(exact.bundle_source_sha256, "sealed bundle source")
    handoff_record = _require_sha256(exact.handoff_record_sha256, "sealed handoff")
    local_full = _require_sha256(exact.local_source_full_sha256, "sealed source full")
    local_tree = _require_sha256(exact.local_source_tree_sha256, "sealed source tree")
    role_bytes = _sealed_role_bytes(exact)
    for role, path in _ROLE_PATHS:
        _file_record(role, path, role_bytes[role])
    local_raw = role_bytes["local_bundle_manifest"]
    publication_raw = role_bytes["publication_manifest"]
    local_manifest = parse_matched_v3_local_reward_bundle_manifest(
        local_raw,
        expected_full_file_sha256=hashlib.sha256(local_raw).hexdigest(),
    )
    publication = parse_matched_v3_local_reward_publication_manifest(
        publication_raw,
        expected_full_file_sha256=hashlib.sha256(publication_raw).hexdigest(),
    )
    expected_cell = {
        "candidate_id": candidate_id,
        "environment_seed": environment_seed,
        "agent_seed": agent_seed,
    }
    source = cast(dict[str, Any], local_manifest["source_binding"])
    source_tree = cast(dict[str, Any], source["local_source_tree"])
    handoff = cast(dict[str, Any], source["handoff"])
    provenance = cast(dict[str, Any], local_manifest["provenance"])
    publication_binding = cast(dict[str, Any], publication["bundle_binding"])
    if (
        publication_binding["manifest_body_sha256"]
        != local_manifest["manifest_body_sha256"]
        or not _exact_json_equal(local_manifest["cell"], expected_cell)
        or not _exact_json_equal(publication["cell"], expected_cell)
        or source["bundle_source_sha256"] != bundle_source
        or source_tree["full_sha256"] != local_full
        or source_tree["tree_sha256"] != local_tree
        or handoff["record_full_file_sha256"] != handoff_record
        or publication["handoff_record_sha256"] != handoff_record
        or publication["local_source_tree_sha256"] != local_tree
        or provenance["creation_pid"] != exact.creation_pid
        or publication_binding["implementation_source_sha256"] != bundle_source
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "sealed local reward cell, source, PID, or handoff binding drifted"
        )
    local_files = cast(dict[str, Any], local_manifest["files"])
    for role, path in _PAYLOAD_ROLE_PATHS:
        if not _exact_json_equal(
            local_files[role],
            _file_record(role, path, role_bytes[role]),
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"sealed local manifest binding drifted: {role}"
            )
    publication_files = cast(dict[str, Any], publication["files"])
    for role, path in _PUBLICATION_BOUND_ROLE_PATHS:
        if not _exact_json_equal(
            publication_files[role],
            _file_record(role, path, role_bytes[role]),
        ):
            raise ForagerMatchedV3LocalRewardBundleError(
                f"sealed publication manifest binding drifted: {role}"
            )
    _scorer_byte_replay_without_module_decode(
        scorer_api=scorer_api,
        reward_artifact=role_bytes["reward_trace"],
        score_receipt=role_bytes["score_receipt"],
    )
    return exact


def _public_bundle_from_sealed(
    payload: _SealedLocalRewardPayload,
    *,
    scorer_api: _ScorerAPI,
) -> MatchedV3LocalRewardBundle:
    """Construct public score-bearing inspection content only after path selection."""

    exact = _validate_sealed_payload(payload, scorer_api=scorer_api)
    role_bytes = _sealed_role_bytes(exact)
    bundle = MatchedV3LocalRewardBundle(
        candidate_id=exact.candidate_id,
        environment_seed=exact.environment_seed,
        agent_seed=exact.agent_seed,
        creation_pid=exact.creation_pid,
        bundle_source_sha256=exact.bundle_source_sha256,
        handoff_record_sha256=exact.handoff_record_sha256,
        local_source_full_sha256=exact.local_source_full_sha256,
        local_source_tree_sha256=exact.local_source_tree_sha256,
        publication_manifest_bytes=role_bytes["publication_manifest"],
        local_bundle_manifest_bytes=role_bytes["local_bundle_manifest"],
        bootstrap_receipt_bytes=role_bytes["bootstrap_receipt"],
        bootstrap_child_record_bytes=role_bytes["bootstrap_child_record"],
        local_runner_receipt_bytes=role_bytes["local_runner_receipt"],
        reward_artifact_bytes=role_bytes["reward_trace"],
        score_receipt_bytes=role_bytes["score_receipt"],
        stdout_bytes=role_bytes["stdout"],
        stderr_bytes=role_bytes["stderr"],
        inventory=_inventory_from_role_bytes(role_bytes),
    )
    return _validate_bundle_structure(bundle, scorer_api=scorer_api)


def _build_bundle_bytes(
    *,
    facts: _HandoffFacts,
    bundle_source_sha256: str,
    handoff_bytes: dict[str, bytes],
    scorer_api: _ScorerAPI,
) -> _SealedLocalRewardPayload:
    try:
        reward_artifact = scorer_api.encode(handoff_bytes["raw_trace"])
        if type(reward_artifact) is not bytes:
            raise ForagerMatchedV3LocalRewardBundleError(
                "canonical scorer returned a non-byte reward artifact"
            )
        score_receipt_object = scorer_api.ingest(reward_artifact)
        if type(score_receipt_object) is not scorer_api.receipt_type:
            raise ForagerMatchedV3LocalRewardBundleError(
                "canonical scorer returned a non-exact score receipt"
            )
        score_receipt = scorer_api.receipt_canonical_json(score_receipt_object)
    except ForagerMatchedV3LocalRewardBundleError:
        raise
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer rejected the handoff raw trace"
        ) from exc
    if type(score_receipt) is not bytes:
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer returned non-byte score receipt content"
        )
    _scorer_byte_replay_without_module_decode(
        scorer_api=scorer_api,
        reward_artifact=reward_artifact,
        score_receipt=score_receipt,
    )
    scorer_after = _require_scorer_api()
    if not _same_scorer_api(scorer_api, scorer_after):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer API identity changed during byte conversion"
        )
    payloads = {
        "bootstrap_receipt": handoff_bytes["bootstrap_receipt"],
        "bootstrap_child_record": handoff_bytes["bootstrap_child_record"],
        "local_runner_receipt": handoff_bytes["local_runner_receipt"],
        "reward_trace": reward_artifact,
        "score_receipt": score_receipt,
        "stdout": handoff_bytes["stdout"],
        "stderr": handoff_bytes["stderr"],
    }
    local_body = _local_manifest_body(
        facts=facts,
        bundle_source_sha256=bundle_source_sha256,
        payloads=payloads,
    )
    local_manifest, local_body_sha256 = _with_body_digest(
        local_body,
        digest_key="manifest_body_sha256",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    publication_body = _publication_body(
        facts=facts,
        bundle_source_sha256=bundle_source_sha256,
        local_manifest=local_manifest,
        local_manifest_body_sha256=local_body_sha256,
        payloads=payloads,
    )
    publication, _publication_body_sha256 = _with_body_digest(
        publication_body,
        digest_key="publication_body_sha256",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    role_bytes = {
        "publication_manifest": publication,
        "local_bundle_manifest": local_manifest,
        **payloads,
    }
    sealed = _SealedLocalRewardPayload(
        candidate_id=facts.candidate_id,
        environment_seed=facts.environment_seed,
        agent_seed=facts.agent_seed,
        creation_pid=facts.creation_pid,
        bundle_source_sha256=bundle_source_sha256,
        handoff_record_sha256=facts.handoff_record_sha256,
        local_source_full_sha256=facts.local_source_full_sha256,
        local_source_tree_sha256=facts.local_source_tree_sha256,
        role_payloads=tuple(
            (role, path, role_bytes[role]) for role, path in _ROLE_PATHS
        ),
    )
    return _validate_sealed_payload(sealed, scorer_api=scorer_api)


def issue_matched_v3_local_reward_bundle(
    *,
    handoff_capability: object,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_full_sha256: str,
    expected_local_source_tree_sha256: str,
    explicit_bundle_opt_in: bool,
) -> object:
    """Consume one handoff and issue a capability with two exclusive future paths.

    Public inspection consumes the capability here.  The captured score-payload-opaque
    publisher instead consumes this live capability directly.
    """

    _require_self_function_surface()
    if type(explicit_bundle_opt_in) is not bool or explicit_bundle_opt_in is not True:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle issuance requires exact explicit opt-in"
        )
    candidate_id = _require_candidate_id(expected_candidate_id)
    environment_seed = _require_uint31(
        expected_environment_seed,
        "expected environment seed",
    )
    agent_seed = _require_uint31(expected_agent_seed, "expected agent seed")
    source_full = _require_sha256(
        expected_local_source_full_sha256,
        "expected local source full",
    )
    source_tree = _require_sha256(
        expected_local_source_tree_sha256,
        "expected local source tree",
    )
    bundle_source = _require_bundle_boundary(reject_runtime_modules=True)
    _require_exact_publisher_module()
    _module, consumer, parser, content_type = _require_exact_handoff_module()
    try:
        content = consumer(
            handoff_capability=handoff_capability,
            explicit_content_access_opt_in=True,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalRewardBundleError(
            "authentic unconsumed local result handoff consumption failed"
        ) from exc
    if type(content) is not content_type:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local result handoff returned non-authentic content"
        )
    facts, handoff_bytes = _facts_from_handoff_content(
        content=content,
        parser=parser,
        expected_candidate_id=candidate_id,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
        expected_local_source_full_sha256=source_full,
        expected_local_source_tree_sha256=source_tree,
    )
    _require_exact_handoff_module()
    if not hmac.compare_digest(
        _require_bundle_boundary(reject_runtime_modules=True),
        bundle_source,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle source changed after handoff consumption"
        )
    scorer_api = _require_scorer_api()
    sealed_payload = _build_bundle_bytes(
        facts=facts,
        bundle_source_sha256=bundle_source,
        handoff_bytes=handoff_bytes,
        scorer_api=scorer_api,
    )
    if not hmac.compare_digest(
        _require_bundle_boundary(reject_runtime_modules=False),
        bundle_source,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle source changed before capability issuance"
        )
    _require_exact_handoff_module()
    _require_exact_publisher_module()
    current_scorer = _require_scorer_api()
    if not _same_scorer_api(scorer_api, current_scorer):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer API changed before capability issuance"
        )
    capability = _LocalRewardBundleCapability()
    role_bytes = _sealed_role_bytes(sealed_payload)
    with _CAPABILITY_LOCK:
        _BUNDLE_CAPABILITIES[capability] = _BundleState(
            pid=os.getpid(),
            status="live",
            handoff_capability=handoff_capability,
            handoff_capability_identity=id(handoff_capability),
            handoff_content=content,
            handoff_content_identity=id(content),
            bundle_source_sha256=bundle_source,
            scorer_api=scorer_api,
            sealed_payload=sealed_payload,
            sealed_payload_identity=id(sealed_payload),
            file_sha256=tuple(
                hashlib.sha256(role_bytes[role]).hexdigest()
                for role, _path in _ROLE_PATHS
            ),
        )
    return capability


def _claim_and_validate_live_bundle_state(
    bundle_capability: object,
) -> tuple[_BundleState, _ScorerAPI, dict[str, bytes]]:
    """Irreversibly select one path, then replay every retained live binding."""

    _require_self_function_surface()
    if type(bundle_capability) is not _LocalRewardBundleCapability:
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward path selection requires an authentic opaque capability"
        )
    exact_capability = bundle_capability
    with _CAPABILITY_LOCK:
        state = _BUNDLE_CAPABILITIES.get(exact_capability)
        if state is None or state.status != "live":
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle capability is unknown, stale, or already consumed"
            )
        if state.pid != os.getpid():
            state.status = "consumed"
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle capability cannot cross a PID boundary"
            )
        if (
            id(state.handoff_capability) != state.handoff_capability_identity
            or id(state.handoff_content) != state.handoff_content_identity
            or id(state.sealed_payload) != state.sealed_payload_identity
        ):
            state.status = "consumed"
            raise ForagerMatchedV3LocalRewardBundleError(
                "local reward bundle live provenance identity is stale"
            )
        state.status = "consumed"
    if not hmac.compare_digest(
        _require_bundle_boundary(reject_runtime_modules=False),
        state.bundle_source_sha256,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle source is stale"
        )
    _require_exact_handoff_module()
    _require_exact_publisher_module()
    current_scorer = _require_scorer_api()
    if not _same_scorer_api(state.scorer_api, current_scorer):
        raise ForagerMatchedV3LocalRewardBundleError(
            "canonical scorer API is stale"
        )
    sealed = _validate_sealed_payload(state.sealed_payload, scorer_api=current_scorer)
    role_bytes = _sealed_role_bytes(sealed)
    observed = tuple(
        hashlib.sha256(role_bytes[role]).hexdigest() for role, _path in _ROLE_PATHS
    )
    if len(observed) != len(state.file_sha256) or any(
        not hmac.compare_digest(actual, expected)
        for actual, expected in zip(observed, state.file_sha256, strict=True)
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle bytes are stale"
        )
    return state, current_scorer, role_bytes


def consume_matched_v3_local_reward_bundle(
    *,
    bundle_capability: object,
    explicit_content_access_opt_in: bool,
) -> MatchedV3LocalRewardBundle:
    """Choose public score-bearing inspection and close the publisher path.

    The public object is constructed only after inspection irreversibly wins
    the live capability.  It is permanently nonqualifying.
    """

    _require_self_function_surface()
    if (
        type(explicit_content_access_opt_in) is not bool
        or explicit_content_access_opt_in is not True
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle content access requires exact explicit opt-in"
        )
    state, scorer_api, _role_bytes = _claim_and_validate_live_bundle_state(
        bundle_capability
    )
    return _public_bundle_from_sealed(state.sealed_payload, scorer_api=scorer_api)


def _consume_matched_v3_local_reward_capability_to_captured_sink(
    *,
    bundle_capability: object,
    publication_parent: Path,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_local_source_tree_sha256: str,
    explicit_publication_opt_in: bool,
) -> object:
    """Choose the exact captured publisher path without exposing payload bytes."""

    _require_self_function_surface()
    if (
        type(explicit_publication_opt_in) is not bool
        or explicit_publication_opt_in is not True
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publication requires exact explicit opt-in"
        )
    candidate_id = _require_candidate_id(expected_candidate_id)
    environment_seed = _require_uint31(expected_environment_seed, "environment seed")
    agent_seed = _require_uint31(expected_agent_seed, "agent seed")
    source_tree = _require_sha256(expected_local_source_tree_sha256, "local source tree")
    if type(publication_parent) is not type(Path()):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publication parent must be an exact pathlib path"
        )
    _publisher, _preclaim_sink, _preclaim_metadata_type = _require_exact_publisher_module()
    parent_preflight = _PUBLISHER_PARENT_PREFLIGHT_AT_LOAD
    if type(parent_preflight) is not types.FunctionType:
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publisher parent preflight is unavailable"
        )
    preflighted_parent = parent_preflight(publication_parent=publication_parent)
    if type(preflighted_parent) is not type(Path()) or preflighted_parent != publication_parent:
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publisher parent preflight result differs"
        )
    state, _scorer_api, role_bytes = _claim_and_validate_live_bundle_state(
        bundle_capability
    )
    sealed = state.sealed_payload
    if (
        sealed.candidate_id != candidate_id
        or sealed.environment_seed != environment_seed
        or sealed.agent_seed != agent_seed
        or sealed.local_source_tree_sha256 != source_tree
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publication expected cell or source tree disagrees"
        )
    _publisher, sink, metadata_type = _require_exact_publisher_module()
    metadata = sink(
        publication_parent=preflighted_parent,
        role_payloads=tuple(
            (path, role_bytes[role]) for role, path in _ROLE_PATHS
        ),
        expected_candidate_id=candidate_id,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
        expected_local_source_tree_sha256=source_tree,
    )
    if type(metadata) is not metadata_type:
        raise ForagerMatchedV3LocalRewardBundleError(
            "captured local publisher returned non-exact metadata"
        )
    return metadata


def matched_v3_local_reward_bundle_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing descriptor content."""

    _require_self_function_surface()
    return _strict_json_object(
        _DESCRIPTOR_BYTES,
        label="local reward bundle descriptor",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )


def canonical_matched_v3_local_reward_bundle_descriptor_bytes() -> bytes:
    """Return the exact canonical local reward bundle descriptor bytes."""

    _require_self_function_surface()
    return _DESCRIPTOR_BYTES


def matched_v3_local_reward_bundle_descriptor_sha256() -> str:
    """Return the frozen local reward bundle descriptor digest."""

    _require_self_function_surface()
    return LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256


def parse_matched_v3_local_reward_bundle_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen, nonauthorizing bundle descriptor."""

    _require_self_function_surface()
    value = _strict_json_object(
        raw,
        label="local reward bundle descriptor",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3LocalRewardBundleError(
            "local reward bundle descriptor differs from its frozen identity"
        )
    return value


__all__ = [
    "BOOTSTRAP_CHILD_RECORD_FILENAME",
    "BOOTSTRAP_RECEIPT_FILENAME",
    "ForagerMatchedV3LocalRewardBundleError",
    "LOCAL_BUNDLE_MANIFEST_FILENAME",
    "LOCAL_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256",
    "LOCAL_REWARD_BUNDLE_ISOLATED_MODULE_NAME",
    "LOCAL_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "LOCAL_REWARD_BUNDLE_STATUS",
    "LOCAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION",
    "LOCAL_RUNNER_RECEIPT_FILENAME",
    "MatchedV3LocalRewardBundle",
    "MatchedV3LocalRewardBundleFile",
    "PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SCHEMA_VERSION",
    "PINNED_LOCAL_RESULT_HANDOFF_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_RESULT_HANDOFF_ISOLATED_MODULE_NAME",
    "PINNED_LOCAL_RESULT_HANDOFF_SOURCE_SHA256",
    "PINNED_SCORER_METRIC_DESCRIPTOR_SCHEMA_VERSION",
    "PINNED_SCORER_METRIC_DESCRIPTOR_SHA256",
    "PINNED_SCORER_PROTOCOL_SOURCE_PATH",
    "PINNED_SCORER_PROTOCOL_SOURCE_SHA256",
    "PINNED_SCORER_SOURCE_PATH",
    "PINNED_SCORER_SOURCE_SHA256",
    "PUBLICATION_MANIFEST_FILENAME",
    "REWARD_TRACE_FILENAME",
    "SCORE_RECEIPT_FILENAME",
    "STDERR_FILENAME",
    "STDOUT_FILENAME",
    "canonical_matched_v3_local_reward_bundle_descriptor_bytes",
    "consume_matched_v3_local_reward_bundle",
    "issue_matched_v3_local_reward_bundle",
    "matched_v3_local_reward_bundle_descriptor",
    "matched_v3_local_reward_bundle_descriptor_sha256",
    "parse_matched_v3_local_reward_bundle_descriptor",
    "parse_matched_v3_local_reward_bundle_manifest",
    "parse_matched_v3_local_reward_publication_manifest",
]


_SELF_FUNCTION_SURFACE_AT_READY = _current_self_function_surface()
