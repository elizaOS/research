"""Strict in-memory reward conversion for a live compiled PPO-GRU outcome.

This additive module does not amend the v1 adapter bundle, its publication surface, or any
qualification plan.  It accepts a compiled-runner outcome only through the runner's public
process-local capability validator, checks the complete signed-int8 reward trace, receipt,
score, accounting, and runtime identity, and converts the trace to the sole canonical NPZ
layout accepted by the matched-v3 scorer.  The canonical artifact is immediately reingested
before an immutable content bundle is returned.

No capability is serialized.  Bundle contents are structural, remain in memory, and grant
no execution, readiness, ingestion, qualification, evidence, publication, promotion, or
performance authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_compiled_runner as compiled_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_bundle_descriptor.v1"
)
COMPILED_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_bundle_manifest.v1"
)
COMPILED_REWARD_BUNDLE_STATUS: Final = "implemented_unqualified"

_COMPILED_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_compiled_runner.py"
)
_COMPILED_RUNNER_SOURCE_SHA256: Final = (
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f"
)
_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_runner_descriptor.v1"
)
_COMPILED_RUNNER_DESCRIPTOR_SHA256: Final = (
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
)
_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_result_receipt.v2"
)
_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_runtime_identity.v1"
)
_SCORER_SOURCE_PATH: Final = "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
_SCORER_SOURCE_SHA256: Final = "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
_SCORE_RECEIPT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_score_receipt.v1"
_NPZ_CONTAINER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_reward_npz.v1"
_RAW_TRACE_ENCODING_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_raw_reward_trace.int8.v1"
_RAW_TRACE_ENCODING: Final = "signed_int8_twos_complement_c_order_one_byte_per_step"
_CANONICAL_NPZ_SIZE_BYTES: Final = 499_980
_METRIC_SCHEMA_VERSION: Final = "alberta.forager_cumulative_reward_metric.v1"
_METRIC_SHA256: Final = "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
_HORIZON: Final = 499_712
_RAW_REWARD_VALUES: Final = (-1, 0, 1, 30)

_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_DESCRIPTOR_BYTES: Final = 64 * 1024
_MAX_RUNTIME_IDENTITY_BYTES: Final = 256 * 1024
_MAX_RUNNER_RECEIPT_BYTES: Final = 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 256 * 1024
_MAX_JSON_DEPTH: Final = 128
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_CHARACTERS: Final = 16 * 1024
_MAX_JSON_INTEGER_DIGITS: Final = 19


class ForagerMatchedV3CompiledRewardBundleError(ValueError):
    """A compiled outcome, conversion, receipt, or bundle failed closed."""


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open exact source bytes for {expected_suffix}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_SOURCE_BYTES
        ):
            raise RuntimeError(
                f"source is not one bounded single-link regular file: {expected_suffix}"
            )
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise RuntimeError(f"source ended while reading {expected_suffix}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"source grew while reading {expected_suffix}")
        after = os.fstat(descriptor)

        def identity(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise RuntimeError(f"source changed while reading {expected_suffix}")
        return digest.hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    finally:
        os.close(descriptor)


def _check_frozen_bindings() -> None:
    for module_file, path, expected in (
        (
            compiled_runner.__file__,
            _COMPILED_RUNNER_SOURCE_PATH,
            _COMPILED_RUNNER_SOURCE_SHA256,
        ),
        (scorer.__file__, _SCORER_SOURCE_PATH, _SCORER_SOURCE_SHA256),
    ):
        if not hmac.compare_digest(_source_sha256(module_file, path), expected):
            raise RuntimeError(f"compiled reward bundle source binding drifted: {path}")
    metric_bytes = protocol.canonical_cumulative_reward_metric_bytes()
    if (
        compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION
        != _COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION
        or compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
        != _COMPILED_RUNNER_DESCRIPTOR_SHA256
        or compiled_runner.PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION
        != _COMPILED_RESULT_RECEIPT_SCHEMA_VERSION
        or compiled_runner.PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION
        != _COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION
        or compiled_runner.MATCHED_V3_HORIZON != _HORIZON
        or scorer.SCORE_RECEIPT_SCHEMA_VERSION != _SCORE_RECEIPT_SCHEMA_VERSION
        or scorer.NPZ_CONTAINER_SCHEMA_VERSION != _NPZ_CONTAINER_SCHEMA_VERSION
        or scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION != _RAW_TRACE_ENCODING_SCHEMA_VERSION
        or scorer.RAW_TRACE_ENCODING != _RAW_TRACE_ENCODING
        or scorer.CANONICAL_NPZ_SIZE_BYTES != _CANONICAL_NPZ_SIZE_BYTES
        or protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION != _METRIC_SCHEMA_VERSION
        or protocol.CUMULATIVE_REWARD_METRIC_SHA256 != _METRIC_SHA256
        or hashlib.sha256(metric_bytes).hexdigest() != _METRIC_SHA256
        or protocol.MATCHED_V3_HORIZON != _HORIZON
        or protocol.MATCHED_V3_RAW_REWARD_VALUES != _RAW_REWARD_VALUES
    ):
        raise RuntimeError("compiled reward bundle descriptor or metric binding drifted")


_check_frozen_bindings()


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledRewardBundleError(
        f"compiled reward bundle JSON contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledRewardBundleError(
        f"compiled reward bundle JSON contains forbidden float {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3CompiledRewardBundleError(
                f"compiled reward bundle JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _validate_json_lexical_bounds(text: str, *, label: str) -> None:
    depth = 0
    nodes = 0
    in_string = False
    escaped = False
    in_primitive = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            in_primitive = False
            nodes += 1
        elif character in "[{":
            depth += 1
            nodes += 1
            in_primitive = False
            if depth > _MAX_JSON_DEPTH:
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} exceeds its JSON nesting-depth bound"
                )
        elif character in "]}":
            depth -= 1
            in_primitive = False
        elif character in ",:":
            in_primitive = False
        elif character in " \t\r\n":
            in_primitive = False
        elif not in_primitive:
            nodes += 1
            in_primitive = True
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CompiledRewardBundleError(f"{label} exceeds its JSON node bound")


def _assert_plain_unaliased_json(value: object, *, label: str) -> None:
    pending = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CompiledRewardBundleError(f"{label} exceeds its JSON node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3CompiledRewardBundleError(
                f"{label} exceeds its JSON nesting-depth bound"
            )
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} contains a non-string object key"
                )
            pending.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        elif type(item) is str:
            if len(item) > _MAX_JSON_STRING_CHARACTERS:
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} contains an oversized string"
                )
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3CompiledRewardBundleError(
                    f"{label} contains a non-finite float"
                )
            raise ForagerMatchedV3CompiledRewardBundleError(f"{label} contains a forbidden float")
        elif item is not None and type(item) not in {bool, int}:
            raise ForagerMatchedV3CompiledRewardBundleError(
                f"{label} contains non-plain type {type(item).__name__}"
            )


def _exact_json_equal(first: object, second: object) -> bool:
    """Compare JSON trees without Python's bool/int equality alias."""

    pending = [(first, second)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            left_mapping = cast(dict[str, object], left)
            right_mapping = cast(dict[str, object], right)
            if set(left_mapping) != set(right_mapping):
                return False
            pending.extend((left_mapping[key], right_mapping[key]) for key in left_mapping)
        elif type(left) is list:
            left_list = cast(list[object], left)
            right_list = cast(list[object], right)
            if len(left_list) != len(right_list):
                return False
            pending.extend(zip(left_list, right_list, strict=True))
        elif left != right:
            return False
    return True


def _canonical_json(value: object, *, label: str, maximum_bytes: int) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} root must be a plain object")
    _assert_plain_unaliased_json(value, label=label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(
            f"{label} is not finite canonical ASCII JSON"
        ) from exc
    if not 0 < len(raw) <= maximum_bytes:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} exceeds its canonical byte bound")
    return raw


def _strict_json_object(raw: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum_bytes:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} must be bounded exact bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} must be ASCII") from exc
    _validate_json_lexical_bounds(text, label=label)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3CompiledRewardBundleError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} is not strict JSON") from exc
    if type(parsed) is not dict:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} root must be a plain object")
    result = cast(dict[str, Any], parsed)
    _assert_plain_unaliased_json(result, label=label)
    if not hmac.compare_digest(
        _canonical_json(result, label=label, maximum_bytes=maximum_bytes), raw
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} is not exactly canonical")
    return result


def _require_exact_keys(value: object, *, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} is below its bound")
    if maximum is not None and value > maximum:
        raise ForagerMatchedV3CompiledRewardBundleError(f"{label} exceeds its bound")
    return value


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "execution_authorized": False,
        "execution_ready": False,
        "ingestion_authorized": False,
        "performance_claim_allowed": False,
        "publication_authorized": False,
        "qualification_claim_allowed": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The live compiled-outcome capability is not serialized into this bundle.",
        "Receipt and runtime-identity bytes do not independently attest execution.",
        "The bundle is in memory and makes no durable or atomic publication claim.",
        "Caller-supplied seed provenance remains unverified.",
        "Conversion grants no ingestion, qualification, evidence, or promotion authority.",
        "This additive bundle does not amend the v1 bundle, publication, or qualification files.",
    ]


def _runner_accounting() -> dict[str, int]:
    return {
        "action_draws": 499_712,
        "automatic_resets": 0,
        "bridge_environment_key_uses": 499_713,
        "bridge_resets": 1,
        "compiled_chunk_count": 976,
        "compiled_chunk_steps": 512,
        "environment_interactions": 499_712,
        "optimizer_updates": 15_616,
        "parameter_initialization_draws": 1,
        "permutation_draws": 3_904,
        "segment_steps": 128,
        "segments_per_rollout": 4,
        "total_agent_draws": 503_617,
        "update_epochs": 4,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
        "status": COMPILED_REWARD_BUNDLE_STATUS,
        "classification": "in_process_compiled_conversion_non_authorizing",
        "candidate_id": "adapted_ppo_gru",
        "compiled_runner": {
            "descriptor_schema_version": _COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_RUNNER_DESCRIPTOR_SHA256,
            "source_path": _COMPILED_RUNNER_SOURCE_PATH,
            "source_sha256": _COMPILED_RUNNER_SOURCE_SHA256,
            "result_receipt_schema_version": _COMPILED_RESULT_RECEIPT_SCHEMA_VERSION,
            "runtime_identity_schema_version": _COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION,
        },
        "scorer": {
            "source_path": _SCORER_SOURCE_PATH,
            "source_sha256": _SCORER_SOURCE_SHA256,
            "score_receipt_schema_version": _SCORE_RECEIPT_SCHEMA_VERSION,
            "npz_container_schema_version": _NPZ_CONTAINER_SCHEMA_VERSION,
            "canonical_npz_size_bytes": _CANONICAL_NPZ_SIZE_BYTES,
            "raw_trace_encoding_schema_version": _RAW_TRACE_ENCODING_SCHEMA_VERSION,
        },
        "metric": {
            "schema_version": _METRIC_SCHEMA_VERSION,
            "sha256": _METRIC_SHA256,
            "horizon": _HORIZON,
            "raw_reward_values": list(_RAW_REWARD_VALUES),
            "accumulation": "ordered_exact_integer_sum",
        },
        "conversion": {
            "public_live_outcome_receipt_capability_required": True,
            "complete_raw_trace_required": True,
            "runner_receipt_and_runtime_identity_cross_checked": True,
            "canonical_npz_reingested_before_return": True,
            "runner_and_scorer_scores_must_match": True,
            "filesystem_writes": False,
            "runtime_opened_or_workload_executed": False,
            "persisted_content_independently_attests_execution": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(),
    label="compiled reward bundle descriptor",
    maximum_bytes=_MAX_DESCRIPTOR_BYTES,
)
COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256,
):
    raise RuntimeError("compiled reward bundle descriptor identity drifted")


def compiled_reward_bundle_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen nonauthorizing descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_compiled_reward_bundle_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_compiled_reward_bundle_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen descriptor identity."""

    parsed = _strict_json_object(
        raw,
        label="compiled reward bundle descriptor",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    if (
        not hmac.compare_digest(raw, _DESCRIPTOR_BYTES)
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256,
        )
        or not _exact_json_equal(parsed.get("claims"), _claims())
        or not _exact_json_equal(parsed.get("limitations"), _limitations())
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle descriptor identity drifted"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class MatchedV3CompiledRewardBundle:
    """Immutable in-memory compiled receipt, runtime identity, and scorer content."""

    candidate_id: str
    runner_receipt_bytes: bytes
    runtime_identity_bytes: bytes
    reward_artifact_bytes: bytes
    score_receipt_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        """Strictly parse and detach this bundle's structural manifest."""

        return parse_compiled_reward_bundle_manifest(
            self.manifest_bytes,
            expected_manifest_sha256=self.manifest_sha256,
        )


def _parse_runner_receipt(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_RUNNER_RECEIPT_BYTES:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled runner receipt must be bounded exact bytes"
        )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        parsed = compiled_runner.parse_ppo_gru_compiled_result_receipt(
            raw,
            expected_receipt_sha256=digest,
        )
    except compiled_runner.ForagerMatchedV3PPOGRUCompiledRunnerError as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled runner receipt failed its frozen structural parser"
        ) from exc
    expected_implementation = {
        "path": _COMPILED_RUNNER_SOURCE_PATH,
        "source_self_hash_bound": False,
        "source_digest_requires_external_binding": True,
    }
    expected_completion = {
        "exact_horizon_complete": True,
        "full_horizon_compiled_execution": True,
        "production_runtime_complete": True,
        "violation_mask": 0,
        "first_invalid_offset": None,
        "content_independently_proves_execution": False,
    }
    if (
        not _exact_json_equal(parsed.get("implementation"), expected_implementation)
        or not _exact_json_equal(parsed.get("accounting"), _runner_accounting())
        or not _exact_json_equal(parsed.get("completion"), expected_completion)
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled runner receipt contains an exact-type contract alias"
        )
    return parsed


def _receipt_runtime_bytes(receipt: dict[str, Any]) -> bytes:
    runtime = _require_exact_keys(
        receipt.get("runtime_identity"),
        expected=frozenset(
            {"schema_version", "classification", "bindings", "runtime", "kernel", "claims"}
        ),
        label="compiled receipt runtime identity",
    )
    observed_runtime = _require_exact_keys(
        runtime["runtime"],
        expected=frozenset(
            {
                "jax_version",
                "jaxlib_version",
                "default_prng_impl",
                "threefry_partitionable",
                "jax_enable_x64",
                "backend",
                "foragax_version",
                "foragax_install_tree_sha256",
                "foragax_package_root",
                "runtime_qualified",
            }
        ),
        label="compiled receipt observed runtime",
    )
    string_fields = (
        "jax_version",
        "jaxlib_version",
        "default_prng_impl",
        "backend",
        "foragax_version",
        "foragax_install_tree_sha256",
        "foragax_package_root",
    )
    expected_kernel = {
        "chunk_steps": 512,
        "constructed": True,
        "full_horizon_executed": False,
        "runtime_qualified": False,
    }
    runner_claims = compiled_runner.matched_v3_ppo_gru_compiled_runner_descriptor()["claims"]
    if (
        runtime["schema_version"] != _COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION
        or any(
            type(observed_runtime[field]) is not str or not observed_runtime[field]
            for field in string_fields
        )
        or observed_runtime["threefry_partitionable"] is not True
        or observed_runtime["jax_enable_x64"] is not False
        or observed_runtime["runtime_qualified"] is not False
        or not _exact_json_equal(runtime["kernel"], expected_kernel)
        or not _exact_json_equal(runtime["claims"], runner_claims)
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled receipt runtime identity contains an exact-type contract alias"
        )
    return _canonical_json(
        runtime,
        label="compiled receipt runtime identity",
        maximum_bytes=_MAX_RUNTIME_IDENTITY_BYTES,
    )


def _validate_outcome_against_receipt(
    outcome: compiled_runner.PPOGRUCompiledOutcome,
    runner_receipt_bytes: bytes,
) -> tuple[dict[str, Any], bytes, int]:
    if type(outcome.raw_reward_trace) is not bytes:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome raw reward trace must be exact bytes"
        )
    if type(outcome.runtime_identity_bytes) is not bytes:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome runtime identity must be exact bytes"
        )
    if type(outcome.receipt_bytes) is not bytes or not hmac.compare_digest(
        outcome.receipt_bytes, runner_receipt_bytes
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome and public receipt bytes disagree"
        )
    receipt = _parse_runner_receipt(runner_receipt_bytes)
    receipt_runtime_bytes = _receipt_runtime_bytes(receipt)
    parsed_runtime = _strict_json_object(
        outcome.runtime_identity_bytes,
        label="compiled outcome runtime identity",
        maximum_bytes=_MAX_RUNTIME_IDENTITY_BYTES,
    )
    if (
        not _exact_json_equal(parsed_runtime, receipt["runtime_identity"])
        or not hmac.compare_digest(receipt_runtime_bytes, outcome.runtime_identity_bytes)
        or not hmac.compare_digest(
            cast(str, receipt["runtime_identity_sha256"]),
            hashlib.sha256(outcome.runtime_identity_bytes).hexdigest(),
        )
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome and receipt runtime identities disagree"
        )
    score = _require_exact_keys(
        receipt.get("score"),
        expected=frozenset(
            {
                "raw_reward_trace_encoding",
                "raw_reward_trace_length",
                "raw_reward_trace_sha256",
                "raw_cumulative_score",
                "reward_scaling_applied",
            }
        ),
        label="compiled receipt score",
    )
    trace = outcome.raw_reward_trace
    outcome_score = _require_exact_int(
        outcome.raw_cumulative_score,
        label="compiled outcome score",
        minimum=-_HORIZON,
        maximum=30 * _HORIZON,
    )
    if (
        len(trace) != _HORIZON
        or score["raw_reward_trace_length"] != _HORIZON
        or score["raw_reward_trace_encoding"] != "signed_int8_twos_complement"
        or score["reward_scaling_applied"] is not False
        or score["raw_cumulative_score"] != outcome_score
        or not hmac.compare_digest(
            cast(str, score["raw_reward_trace_sha256"]),
            hashlib.sha256(trace).hexdigest(),
        )
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome, receipt, reward trace, or score disagrees"
        )
    expected_accounting = _runner_accounting()
    if (
        not _exact_json_equal(receipt.get("accounting"), expected_accounting)
        or _require_exact_int(outcome.interactions, label="compiled interactions")
        != expected_accounting["environment_interactions"]
        or _require_exact_int(outcome.rollout_count, label="compiled rollout count")
        != expected_accounting["compiled_chunk_count"]
        or _require_exact_int(
            outcome.optimizer_update_count,
            label="compiled optimizer update count",
        )
        != expected_accounting["optimizer_updates"]
        or _require_exact_int(
            outcome.total_agent_draw_count,
            label="compiled agent draw count",
        )
        != expected_accounting["total_agent_draws"]
        or _require_exact_int(
            outcome.bridge_environment_key_use_count,
            label="compiled bridge key-use count",
        )
        != expected_accounting["bridge_environment_key_uses"]
        or outcome.production_runtime is not True
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome and receipt accounting disagree"
        )
    trace_chain_sha256 = _require_sha256(
        outcome.trace_chain_sha256,
        label="compiled outcome trace-chain digest",
    )
    if not hmac.compare_digest(trace_chain_sha256, cast(str, receipt["trace_chain_sha256"])):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled outcome and receipt trace-chain digests disagree"
        )
    return receipt, trace, outcome_score


def _manifest_bindings() -> dict[str, str]:
    return {
        "compiled_reward_bundle_descriptor_schema_version": (
            COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "compiled_reward_bundle_descriptor_sha256": (COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256),
        "compiled_runner_descriptor_schema_version": (_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION),
        "compiled_runner_descriptor_sha256": _COMPILED_RUNNER_DESCRIPTOR_SHA256,
        "compiled_runner_source_path": _COMPILED_RUNNER_SOURCE_PATH,
        "compiled_runner_source_sha256": _COMPILED_RUNNER_SOURCE_SHA256,
        "compiled_result_receipt_schema_version": _COMPILED_RESULT_RECEIPT_SCHEMA_VERSION,
        "compiled_runtime_identity_schema_version": (_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION),
        "scorer_source_path": _SCORER_SOURCE_PATH,
        "scorer_source_sha256": _SCORER_SOURCE_SHA256,
        "cumulative_reward_metric_schema_version": _METRIC_SCHEMA_VERSION,
        "cumulative_reward_metric_sha256": _METRIC_SHA256,
    }


def _manifest_body(
    *,
    runner_receipt_bytes: bytes,
    runtime_identity_bytes: bytes,
    reward_artifact_bytes: bytes,
    score_receipt: scorer.MatchedV3ScoreReceipt,
    raw_trace: bytes,
    runner_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": COMPILED_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "classification": "converted_compiled_result_non_authorizing",
        "candidate_id": "adapted_ppo_gru",
        "bindings": _manifest_bindings(),
        "compiled_runner_receipt": {
            "schema_version": _COMPILED_RESULT_RECEIPT_SCHEMA_VERSION,
            "sha256": hashlib.sha256(runner_receipt_bytes).hexdigest(),
            "size_bytes": len(runner_receipt_bytes),
            "trace_chain_sha256": runner_receipt["trace_chain_sha256"],
            "structural_content_independently_attests_execution": False,
        },
        "runtime_identity": {
            "schema_version": _COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION,
            "sha256": hashlib.sha256(runtime_identity_bytes).hexdigest(),
            "size_bytes": len(runtime_identity_bytes),
            "structural_content_independently_attests_execution": False,
        },
        "runner_accounting": _runner_accounting(),
        "raw_reward_trace": {
            "encoding_schema_version": _RAW_TRACE_ENCODING_SCHEMA_VERSION,
            "encoding": _RAW_TRACE_ENCODING,
            "length": len(raw_trace),
            "bytes_sha256": hashlib.sha256(raw_trace).hexdigest(),
            "version_framed_sha256": score_receipt.raw_trace_sha256,
            "raw_cumulative_score": score_receipt.cumulative_score,
        },
        "reward_artifact": {
            "container_schema_version": _NPZ_CONTAINER_SCHEMA_VERSION,
            "sha256": hashlib.sha256(reward_artifact_bytes).hexdigest(),
            "size_bytes": len(reward_artifact_bytes),
        },
        "score_receipt": {
            "schema_version": _SCORE_RECEIPT_SCHEMA_VERSION,
            "sha256": hashlib.sha256(score_receipt.canonical_json()).hexdigest(),
            "receipt_body_sha256": score_receipt.receipt_sha256,
            "size_bytes": len(score_receipt.canonical_json()),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _manifest_bytes(body: dict[str, Any]) -> tuple[bytes, str]:
    body_bytes = _canonical_json(
        body,
        label="compiled reward bundle manifest body",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    digest = hashlib.sha256(body_bytes).hexdigest()
    payload = dict(body)
    payload["manifest_body_sha256"] = digest
    return (
        _canonical_json(
            payload,
            label="compiled reward bundle manifest",
            maximum_bytes=_MAX_MANIFEST_BYTES,
        ),
        digest,
    )


def _validate_manifest_body(body: dict[str, Any]) -> None:
    _require_exact_keys(
        body,
        expected=frozenset(
            {
                "schema_version",
                "classification",
                "candidate_id",
                "bindings",
                "compiled_runner_receipt",
                "runtime_identity",
                "runner_accounting",
                "raw_reward_trace",
                "reward_artifact",
                "score_receipt",
                "claims",
                "limitations",
            }
        ),
        label="compiled reward bundle manifest body",
    )
    if (
        body["schema_version"] != COMPILED_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        or body["classification"] != "converted_compiled_result_non_authorizing"
        or body["candidate_id"] != "adapted_ppo_gru"
        or not _exact_json_equal(body["bindings"], _manifest_bindings())
        or not _exact_json_equal(body["runner_accounting"], _runner_accounting())
        or not _exact_json_equal(body["claims"], _claims())
        or not _exact_json_equal(body["limitations"], _limitations())
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle manifest fixed identity drifted"
        )
    runner_receipt = _require_exact_keys(
        body["compiled_runner_receipt"],
        expected=frozenset(
            {
                "schema_version",
                "sha256",
                "size_bytes",
                "trace_chain_sha256",
                "structural_content_independently_attests_execution",
            }
        ),
        label="manifest compiled runner receipt",
    )
    if (
        runner_receipt["schema_version"] != _COMPILED_RESULT_RECEIPT_SCHEMA_VERSION
        or runner_receipt["structural_content_independently_attests_execution"] is not False
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "manifest compiled runner receipt contract drifted"
        )
    _require_sha256(runner_receipt["sha256"], label="compiled runner receipt digest")
    _require_sha256(
        runner_receipt["trace_chain_sha256"],
        label="compiled runner trace-chain digest",
    )
    _require_exact_int(
        runner_receipt["size_bytes"],
        label="compiled runner receipt size",
        minimum=1,
        maximum=_MAX_RUNNER_RECEIPT_BYTES,
    )
    runtime = _require_exact_keys(
        body["runtime_identity"],
        expected=frozenset(
            {
                "schema_version",
                "sha256",
                "size_bytes",
                "structural_content_independently_attests_execution",
            }
        ),
        label="manifest runtime identity",
    )
    if (
        runtime["schema_version"] != _COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION
        or runtime["structural_content_independently_attests_execution"] is not False
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "manifest runtime identity contract drifted"
        )
    _require_sha256(runtime["sha256"], label="runtime identity digest")
    _require_exact_int(
        runtime["size_bytes"],
        label="runtime identity size",
        minimum=1,
        maximum=_MAX_RUNTIME_IDENTITY_BYTES,
    )
    raw_trace = _require_exact_keys(
        body["raw_reward_trace"],
        expected=frozenset(
            {
                "encoding_schema_version",
                "encoding",
                "length",
                "bytes_sha256",
                "version_framed_sha256",
                "raw_cumulative_score",
            }
        ),
        label="manifest raw reward trace",
    )
    if (
        raw_trace["encoding_schema_version"] != _RAW_TRACE_ENCODING_SCHEMA_VERSION
        or raw_trace["encoding"] != _RAW_TRACE_ENCODING
        or raw_trace["length"] != _HORIZON
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "manifest raw reward trace contract drifted"
        )
    _require_sha256(raw_trace["bytes_sha256"], label="raw reward trace byte digest")
    _require_sha256(
        raw_trace["version_framed_sha256"],
        label="version-framed raw reward trace digest",
    )
    _require_exact_int(
        raw_trace["raw_cumulative_score"],
        label="raw cumulative score",
        minimum=-_HORIZON,
        maximum=30 * _HORIZON,
    )
    artifact = _require_exact_keys(
        body["reward_artifact"],
        expected=frozenset({"container_schema_version", "sha256", "size_bytes"}),
        label="manifest reward artifact",
    )
    if artifact["container_schema_version"] != _NPZ_CONTAINER_SCHEMA_VERSION:
        raise ForagerMatchedV3CompiledRewardBundleError("manifest reward artifact schema drifted")
    _require_sha256(artifact["sha256"], label="reward artifact digest")
    _require_exact_int(
        artifact["size_bytes"],
        label="reward artifact size",
        minimum=_CANONICAL_NPZ_SIZE_BYTES,
        maximum=_CANONICAL_NPZ_SIZE_BYTES,
    )
    score_receipt = _require_exact_keys(
        body["score_receipt"],
        expected=frozenset({"schema_version", "sha256", "receipt_body_sha256", "size_bytes"}),
        label="manifest score receipt",
    )
    if score_receipt["schema_version"] != _SCORE_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedV3CompiledRewardBundleError("manifest score receipt schema drifted")
    _require_sha256(score_receipt["sha256"], label="score receipt digest")
    _require_sha256(
        score_receipt["receipt_body_sha256"],
        label="score receipt body digest",
    )
    _require_exact_int(
        score_receipt["size_bytes"],
        label="score receipt size",
        minimum=1,
        maximum=_MAX_MANIFEST_BYTES,
    )
    claims = _require_exact_keys(
        body["claims"], expected=frozenset(_claims()), label="manifest claims"
    )
    if any(value is not False for value in claims.values()):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "manifest claims must be exact false booleans"
        )
    limitations = body["limitations"]
    if (
        type(limitations) is not list
        or not _exact_json_equal(limitations, _limitations())
        or any(type(item) is not str for item in limitations)
    ):
        raise ForagerMatchedV3CompiledRewardBundleError("manifest limitations drifted")


def parse_compiled_reward_bundle_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly parse a bounded structural manifest without granting authority."""

    expected_digest = _require_sha256(
        expected_manifest_sha256,
        label="expected manifest digest",
    )
    payload = _strict_json_object(
        raw,
        label="compiled reward bundle manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    supplied = _require_sha256(
        payload.get("manifest_body_sha256"),
        label="manifest body digest",
    )
    if not hmac.compare_digest(supplied, expected_digest):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle manifest digest binding differs"
        )
    body = dict(payload)
    del body["manifest_body_sha256"]
    calculated = hashlib.sha256(
        _canonical_json(
            body,
            label="compiled reward bundle manifest body",
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    ).hexdigest()
    if not hmac.compare_digest(calculated, expected_digest):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle manifest body digest drifted"
        )
    _validate_manifest_body(body)
    return payload


def build_ppo_gru_compiled_reward_bundle(
    outcome: compiled_runner.PPOGRUCompiledOutcome,
) -> MatchedV3CompiledRewardBundle:
    """Convert only one exact live capability-backed compiled PPO-GRU outcome."""

    try:
        runner_receipt_bytes = compiled_runner.canonical_ppo_gru_compiled_result_receipt_bytes(
            outcome
        )
    except compiled_runner.ForagerMatchedV3PPOGRUCompiledRunnerError as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled PPO-GRU outcome lacks a live authentic completion capability"
        ) from exc
    if type(runner_receipt_bytes) is not bytes:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled runner public receipt API returned non-bytes"
        )
    receipt, raw_trace, expected_score = _validate_outcome_against_receipt(
        outcome,
        runner_receipt_bytes,
    )
    try:
        artifact = scorer.canonical_reward_npz_bytes(raw_trace)
        score_receipt = scorer.ingest_reward_npz_bytes(artifact)
        replayed_trace = scorer.extract_canonical_reward_trace(artifact)
    except scorer.ForagerMatchedV3ScorerError as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "strict scorer rejected the compiled reward trace"
        ) from exc
    if (
        not hmac.compare_digest(replayed_trace, raw_trace)
        or score_receipt.cumulative_score != expected_score
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled runner and strict scorer replay disagree"
        )
    body = _manifest_body(
        runner_receipt_bytes=runner_receipt_bytes,
        runtime_identity_bytes=outcome.runtime_identity_bytes,
        reward_artifact_bytes=artifact,
        score_receipt=score_receipt,
        raw_trace=raw_trace,
        runner_receipt=receipt,
    )
    manifest_bytes, manifest_sha256 = _manifest_bytes(body)
    bundle = MatchedV3CompiledRewardBundle(
        candidate_id="adapted_ppo_gru",
        runner_receipt_bytes=runner_receipt_bytes,
        runtime_identity_bytes=outcome.runtime_identity_bytes,
        reward_artifact_bytes=artifact,
        score_receipt_bytes=score_receipt.canonical_json(),
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )
    return validate_compiled_reward_bundle(bundle)


def validate_compiled_reward_bundle(bundle: object) -> MatchedV3CompiledRewardBundle:
    """Replay every structural content relationship in one in-memory bundle."""

    if type(bundle) is not MatchedV3CompiledRewardBundle:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "bundle must be an exact MatchedV3CompiledRewardBundle"
        )
    if type(bundle.candidate_id) is not str or bundle.candidate_id != "adapted_ppo_gru":
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle candidate identity drifted"
        )
    for name in (
        "runner_receipt_bytes",
        "runtime_identity_bytes",
        "reward_artifact_bytes",
        "score_receipt_bytes",
        "manifest_bytes",
    ):
        if type(getattr(bundle, name)) is not bytes:
            raise ForagerMatchedV3CompiledRewardBundleError(f"bundle {name} must be exact bytes")
    manifest = parse_compiled_reward_bundle_manifest(
        bundle.manifest_bytes,
        expected_manifest_sha256=bundle.manifest_sha256,
    )
    body = dict(manifest)
    del body["manifest_body_sha256"]
    runner_receipt = _parse_runner_receipt(bundle.runner_receipt_bytes)
    receipt_runtime_bytes = _receipt_runtime_bytes(runner_receipt)
    runtime_identity = _strict_json_object(
        bundle.runtime_identity_bytes,
        label="bundle runtime identity",
        maximum_bytes=_MAX_RUNTIME_IDENTITY_BYTES,
    )
    if (
        not _exact_json_equal(runtime_identity, runner_receipt["runtime_identity"])
        or not hmac.compare_digest(receipt_runtime_bytes, bundle.runtime_identity_bytes)
        or not hmac.compare_digest(
            cast(str, runner_receipt["runtime_identity_sha256"]),
            hashlib.sha256(bundle.runtime_identity_bytes).hexdigest(),
        )
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "bundle runtime identity does not replay from its runner receipt"
        )
    try:
        score_receipt = scorer.parse_score_receipt(bundle.score_receipt_bytes)
        replayed_score = scorer.ingest_reward_npz_bytes(bundle.reward_artifact_bytes)
        raw_trace = scorer.extract_canonical_reward_trace(bundle.reward_artifact_bytes)
    except scorer.ForagerMatchedV3ScorerError as exc:
        raise ForagerMatchedV3CompiledRewardBundleError(
            "bundle reward artifact or score receipt failed strict replay"
        ) from exc
    receipt_score = cast(dict[str, Any], runner_receipt["score"])
    if (
        score_receipt.canonical_json() != replayed_score.canonical_json()
        or len(raw_trace) != _HORIZON
        or receipt_score["raw_reward_trace_length"] != len(raw_trace)
        or receipt_score["raw_cumulative_score"] != replayed_score.cumulative_score
        or not hmac.compare_digest(
            cast(str, receipt_score["raw_reward_trace_sha256"]),
            hashlib.sha256(raw_trace).hexdigest(),
        )
    ):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "bundle runner receipt and strict scorer replay disagree"
        )
    expected_body = _manifest_body(
        runner_receipt_bytes=bundle.runner_receipt_bytes,
        runtime_identity_bytes=bundle.runtime_identity_bytes,
        reward_artifact_bytes=bundle.reward_artifact_bytes,
        score_receipt=score_receipt,
        raw_trace=raw_trace,
        runner_receipt=runner_receipt,
    )
    if not _exact_json_equal(body, expected_body):
        raise ForagerMatchedV3CompiledRewardBundleError(
            "compiled reward bundle manifest does not replay from its exact contents"
        )
    return bundle


__all__ = [
    "COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION",
    "COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256",
    "COMPILED_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "COMPILED_REWARD_BUNDLE_STATUS",
    "ForagerMatchedV3CompiledRewardBundleError",
    "MatchedV3CompiledRewardBundle",
    "build_ppo_gru_compiled_reward_bundle",
    "canonical_compiled_reward_bundle_descriptor_bytes",
    "compiled_reward_bundle_descriptor",
    "parse_compiled_reward_bundle_descriptor",
    "parse_compiled_reward_bundle_manifest",
    "validate_compiled_reward_bundle",
]
