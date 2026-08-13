"""Content-only compiled-PPO qualification addendum for matched Forager v3.

This additive artifact binds one strictly parsed qualification-plan-v1 artifact to
the separately versioned compiled PPO-GRU runner, reward bundle, and six-file
publication chain.  It copies only the ``adapted_ppo_gru`` case, resource ceiling,
local source requirement, shared runtime requirement, and static candidate contract.
The base publisher is retained only as an explicitly excluded content identity.

The addendum does not mutate, amend, or supersede qualification-plan v1.  It has no
executor, seed issuer, workload opener, result-acceptance API, or authority token.
Source membership, runtime behavior, resource observations, publication replay, and
qualification acceptance remain work for a future independently versioned executor
and validator.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as _scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_reward_bundle as _compiled_bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_reward_publication as _compiled_publication,
)
from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as _bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as _ppo_gru
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_compiled_runner as _compiled_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru_runner as _v1_runner
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_plan as _base_qualification,
)

COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_qualification_addendum_descriptor.v1"
)
COMPILED_QUALIFICATION_ADDENDUM_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_qualification_addendum.v1"
)
COMPILED_QUALIFICATION_ADDENDUM_STATUS: Final = "implemented_unexecuted_no_production_addendum"
COMPILED_QUALIFICATION_ADDENDUM_CLASSIFICATION: Final = (
    "single_candidate_content_overlay_non_authorizing"
)

_CANDIDATE_ID: Final = "adapted_ppo_gru"
_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 16 * 1024
_MAX_INTEGER_DIGITS: Final = 19
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

_BASE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_qualification_plan_descriptor.v1"
)
_BASE_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_qualification_plan.v1"
_BASE_DESCRIPTOR_SHA256: Final = "258b9e376b82127f912bf2828a6d4e5c7a257ed2a990cd15bf4c9cbd81c17788"
_BASE_SOURCE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_qualification_plan.py"
_BASE_SOURCE_SHA256: Final = "d84eb2322dc902dc912e79d9b14295f5d580bcdedf3e8870027854ca344e1ebf"

_CONFIGURATION_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_configuration_plan.v1"
_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
_CONFIGURATION_PLAN_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_configuration_plan.py"
)
_CONFIGURATION_PLAN_SOURCE_SHA256: Final = (
    "ad711eaa61511c6b1d43b86b867e09ba70f7124d5d67966b22d1f7ef3a556a84"
)
_CONFIGURATION_RECORD_SHA256: Final = (
    "4f8b429ff968213d0c05de87553456be7f2c1a67a806944357543025d725d7ca"
)

_BRIDGE_DESCRIPTOR_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_foragax_bridge.v2"
_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
_BRIDGE_SOURCE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
_BRIDGE_SOURCE_SHA256: Final = "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"

_CORE_CONFIGURATION_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_ppo_gru_configuration.v1"
_CORE_CONFIGURATION_SHA256: Final = (
    "07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70"
)
_CORE_DESCRIPTOR_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_ppo_gru_source.v1"
_CORE_DESCRIPTOR_SHA256: Final = "64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca"
_CORE_SOURCE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py"
_CORE_SOURCE_SHA256: Final = "58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e"

_V1_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_ppo_gru_runner.v1"
_V1_RUNNER_DESCRIPTOR_SHA256: Final = (
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
)
_V1_RUNNER_SOURCE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
_V1_RUNNER_SOURCE_SHA256: Final = "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"

_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_runner_descriptor.v1"
)
_COMPILED_RUNNER_DESCRIPTOR_SHA256: Final = (
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
)
_COMPILED_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_compiled_runner.py"
)
_COMPILED_RUNNER_SOURCE_SHA256: Final = (
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f"
)

_COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_bundle_descriptor.v1"
)
_COMPILED_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
)
_COMPILED_BUNDLE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_bundle.py"
)
_COMPILED_BUNDLE_SOURCE_SHA256: Final = (
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e"
)

_COMPILED_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_publication_descriptor.v1"
)
_COMPILED_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500"
)
_COMPILED_PUBLICATION_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_publication.py"
)
_COMPILED_PUBLICATION_SOURCE_SHA256: Final = (
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71"
)

_SCORER_SOURCE_PATH: Final = "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
_SCORER_SOURCE_SHA256: Final = "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
_METRIC_SCHEMA_VERSION: Final = "alberta.forager_cumulative_reward_metric.v1"
_METRIC_SHA256: Final = "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"

_BASE_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
_BASE_PUBLICATION_SOURCE_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)

_ACCOUNTING: Final = {
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

_RNG_CONTRACT: Final = {
    "agent_action": "continuation,action_key=split(continuation)",
    "agent_initialization": "continuation,init_key=split(agent_root)",
    "agent_permutation": "one continuation split per epoch",
    "automatic_resets": 0,
    "categorical_mode": "low",
    "environment_reset": "continuation,reset_key=split(root)",
    "environment_root": "jax.random.key(environment_seed)",
    "environment_transition": "continuation,step_key=split(continuation)",
    "implementation": "threefry2x32",
    "ppo_environment_chain_consumed": False,
}

_EXACT_PUBLICATION_FILES: Final = {
    "compiled_bundle_manifest": "compiled-bundle-manifest.json",
    "publication_manifest": "publication.json",
    "reward_trace": "reward-trace.npz",
    "runner_result_receipt": "runner-result-receipt.json",
    "runtime_identity": "runtime-identity.json",
    "score_receipt": "score-receipt.json",
}


class ForagerMatchedV3CompiledQualificationAddendumError(ValueError):
    """A compiled qualification-addendum content boundary failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledQualificationAddendumError(
        f"compiled qualification addendum contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledQualificationAddendumError(
        f"compiled qualification addendum contains forbidden float {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_INTEGER_DIGITS:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                f"compiled qualification addendum contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification addendum exceeds its JSON node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification addendum exceeds its JSON depth bound"
            )
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                raise ForagerMatchedV3CompiledQualificationAddendumError(
                    "compiled qualification addendum strings must be bounded printable ASCII"
                )
            continue
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification addendum contains a non-plain JSON value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification addendum contains an aliased or cyclic container"
            )
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
            continue
        mapping = cast(dict[Any, Any], item)
        if any(type(key) is not str for key in mapping):
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification addendum object keys must be exact strings"
            )
        for key, child in mapping.items():
            pending.append((key, depth + 1))
            pending.append((child, depth + 1))


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, Any], left)
        right_map = cast(dict[str, Any], right)
        return left_map.keys() == right_map.keys() and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_list = left
        right_list = cast(list[Any], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum canonical root must be a plain object"
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
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum exceeds its canonical byte bound"
        )
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum input must be bounded exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum must have one canonical trailing newline"
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
    except ForagerMatchedV3CompiledQualificationAddendumError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum is not bounded strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum is not in exact canonical form"
        )
    return result


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _bounded_source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"dependency source path differs from {expected_suffix}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "dependency source verification requires O_NOFOLLOW"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"dependency source cannot be opened safely: {expected_suffix}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_SOURCE_BYTES
        ):
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                f"dependency source is not one bounded single-link file: {expected_suffix}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3CompiledQualificationAddendumError(
                    f"dependency source was truncated while reading: {expected_suffix}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                f"dependency source grew while reading: {expected_suffix}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_before != stable_after:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"dependency source changed while reading: {expected_suffix}"
        )
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def _dependency_bindings() -> dict[str, Any]:
    return {
        "base_qualification_plan": {
            "descriptor_schema_version": _BASE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _BASE_DESCRIPTOR_SHA256,
            "plan_schema_version": _BASE_PLAN_SCHEMA_VERSION,
            "source_path": _BASE_SOURCE_PATH,
            "source_sha256": _BASE_SOURCE_SHA256,
        },
        "configuration_plan": {
            "schema_version": _CONFIGURATION_PLAN_SCHEMA_VERSION,
            "sha256": _CONFIGURATION_PLAN_SHA256,
            "source_path": _CONFIGURATION_PLAN_SOURCE_PATH,
            "source_sha256": _CONFIGURATION_PLAN_SOURCE_SHA256,
            "candidate_id": _CANDIDATE_ID,
            "candidate_record_sha256": _CONFIGURATION_RECORD_SHA256,
            "candidate_record_canonicalization": "sorted_compact_ascii_json_without_newline",
        },
        "foragax_bridge": {
            "descriptor_schema_version": _BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _BRIDGE_DESCRIPTOR_SHA256,
            "source_path": _BRIDGE_SOURCE_PATH,
            "source_sha256": _BRIDGE_SOURCE_SHA256,
        },
        "ppo_gru_core": {
            "configuration_schema_version": _CORE_CONFIGURATION_SCHEMA_VERSION,
            "configuration_sha256": _CORE_CONFIGURATION_SHA256,
            "descriptor_schema_version": _CORE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _CORE_DESCRIPTOR_SHA256,
            "source_path": _CORE_SOURCE_PATH,
            "source_sha256": _CORE_SOURCE_SHA256,
        },
        "v1_semantic_reference_runner": {
            "descriptor_schema_version": _V1_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _V1_RUNNER_DESCRIPTOR_SHA256,
            "source_path": _V1_RUNNER_SOURCE_PATH,
            "source_sha256": _V1_RUNNER_SOURCE_SHA256,
            "execution_path_selected_by_addendum": False,
        },
        "compiled_runner": {
            "descriptor_schema_version": _COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_RUNNER_DESCRIPTOR_SHA256,
            "source_path": _COMPILED_RUNNER_SOURCE_PATH,
            "source_sha256": _COMPILED_RUNNER_SOURCE_SHA256,
            "result_receipt_schema_version": (
                "alberta.forager_matched_v3.ppo_gru_compiled_result_receipt.v2"
            ),
            "runtime_identity_schema_version": (
                "alberta.forager_matched_v3.ppo_gru_compiled_runtime_identity.v1"
            ),
        },
        "compiled_reward_bundle": {
            "descriptor_schema_version": _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
            "source_path": _COMPILED_BUNDLE_SOURCE_PATH,
            "source_sha256": _COMPILED_BUNDLE_SOURCE_SHA256,
            "manifest_schema_version": (
                "alberta.forager_matched_v3.compiled_reward_bundle_manifest.v1"
            ),
        },
        "compiled_reward_publication": {
            "descriptor_schema_version": _COMPILED_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_PUBLICATION_DESCRIPTOR_SHA256,
            "source_path": _COMPILED_PUBLICATION_SOURCE_PATH,
            "source_sha256": _COMPILED_PUBLICATION_SOURCE_SHA256,
            "publication_schema_version": (
                "alberta.forager_matched_v3.compiled_reward_publication.v1"
            ),
        },
        "strict_scorer": {
            "source_path": _SCORER_SOURCE_PATH,
            "source_sha256": _SCORER_SOURCE_SHA256,
            "metric_schema_version": _METRIC_SCHEMA_VERSION,
            "metric_sha256": _METRIC_SHA256,
            "canonical_npz_size_bytes": 499_980,
        },
    }


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "base_plan_amended": False,
        "base_plan_superseded": False,
        "candidate_qualified": False,
        "execution_authorized": False,
        "execution_ready": False,
        "ingestion_authorized": False,
        "performance_claim_allowed": False,
        "production_addendum_issued": False,
        "qualification_executed": False,
        "result_accepted": False,
        "result_loaded": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "source_closure_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This is an additive content overlay and does not amend or supersede the base plan.",
        "The base v1 result publisher is excluded, not reinterpreted as compiled publication.",
        "No workload, compiled chunk, result, publication, or qualification is executed here.",
        "The copied source, runtime, resource, and seed requirements remain unqualified.",
        "A future executor must preserve the exact base seed case and compiled RNG accounting.",
        "A future validator must verify source membership, runtime, resources, and all six files.",
        (
            "Structural content cannot authorize ingestion, evidence, promotion, or "
            "performance claims."
        ),
    ]


def _future_requirements() -> dict[str, bool]:
    return {
        "future_executor_required": True,
        "future_result_validator_required": True,
        "independent_source_membership_validation_required": True,
        "publication_six_file_replay_required": True,
        "resource_observation_required": True,
        "runtime_qualification_required": True,
        "seed_provenance_revalidation_required": True,
        "source_closure_qualification_required": True,
        "implemented_here": False,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SCHEMA_VERSION,
        "status": COMPILED_QUALIFICATION_ADDENDUM_STATUS,
        "classification": COMPILED_QUALIFICATION_ADDENDUM_CLASSIFICATION,
        "candidate_id": _CANDIDATE_ID,
        "dependencies": _dependency_bindings(),
        "base_input_contract": {
            "strict_base_plan_parser_required": True,
            "caller_supplied_base_plan_full_file_sha256_required": True,
            "independent_seed_trust_receipt_file_pin_required": True,
            "independent_seed_trust_receipt_binding_pin_required": True,
            "base_plan_embedded_or_mutated": False,
        },
        "overlay_scope": {
            "candidate_count": 1,
            "candidate_id": _CANDIDATE_ID,
            "operation": "replace_execution_and_result_publication_path_in_addendum_only",
            "reused_base_requirements": [
                "candidate_static_contract",
                "qualification_seed_case",
                "resource_requirement",
                "local_source_requirement",
                "shared_runtime_requirement",
            ],
            "base_result_publisher_reused": False,
            "base_v1_runner_selected_for_execution": False,
            "base_plan_amended": False,
            "base_plan_superseded": False,
        },
        "compiled_chain": {
            "ordered_stages": [
                "compiled_ppo_gru_runner",
                "compiled_reward_bundle",
                "compiled_six_file_publication",
            ],
            "accounting": dict(_ACCOUNTING),
            "rng": dict(_RNG_CONTRACT),
            "exact_publication_files": dict(_EXACT_PUBLICATION_FILES),
        },
        "future_qualification_requirements": _future_requirements(),
        "canonicalization": {
            "format": "sorted_compact_ascii_json_with_one_newline",
            "allow_nan": False,
            "floats_allowed": False,
            "duplicate_keys_rejected": True,
            "exact_scalar_types_required": True,
            "container_aliases_rejected": True,
            "maximum_bytes": _MAX_ARTIFACT_BYTES,
            "maximum_depth": _MAX_JSON_DEPTH,
            "maximum_nodes": _MAX_JSON_NODES,
        },
        "apis": {
            "executor_exposed": False,
            "seed_issuer_exposed": False,
            "default_production_addendum_exposed": False,
            "result_loader_acceptance_exposed": False,
            "workload_opener_exposed": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256: Final = (
    "b5f7df77cd3f6e35126ed7c9f4b7acacdaa8237e8242241f658a95d21e9e3b06"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256,
):
    raise AssertionError("compiled qualification-addendum descriptor identity drifted")


def _source_bindings() -> tuple[tuple[str, object, str, str], ...]:
    base_file = _base_qualification.__file__
    if type(base_file) is not str:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base qualification source path is unavailable"
        )
    configuration_source = os.path.join(
        os.path.dirname(base_file),
        "forager_matched_v3_configuration_plan.py",
    )
    return (
        (
            "base qualification",
            _base_qualification.__file__,
            _BASE_SOURCE_PATH,
            _BASE_SOURCE_SHA256,
        ),
        (
            "configuration plan",
            configuration_source,
            _CONFIGURATION_PLAN_SOURCE_PATH,
            _CONFIGURATION_PLAN_SOURCE_SHA256,
        ),
        ("Foragax bridge", _bridge.__file__, _BRIDGE_SOURCE_PATH, _BRIDGE_SOURCE_SHA256),
        ("PPO-GRU core", _ppo_gru.__file__, _CORE_SOURCE_PATH, _CORE_SOURCE_SHA256),
        (
            "v1 semantic runner",
            _v1_runner.__file__,
            _V1_RUNNER_SOURCE_PATH,
            _V1_RUNNER_SOURCE_SHA256,
        ),
        (
            "compiled runner",
            _compiled_runner.__file__,
            _COMPILED_RUNNER_SOURCE_PATH,
            _COMPILED_RUNNER_SOURCE_SHA256,
        ),
        (
            "compiled bundle",
            _compiled_bundle.__file__,
            _COMPILED_BUNDLE_SOURCE_PATH,
            _COMPILED_BUNDLE_SOURCE_SHA256,
        ),
        (
            "compiled publication",
            _compiled_publication.__file__,
            _COMPILED_PUBLICATION_SOURCE_PATH,
            _COMPILED_PUBLICATION_SOURCE_SHA256,
        ),
        ("strict scorer", _scorer.__file__, _SCORER_SOURCE_PATH, _SCORER_SOURCE_SHA256),
    )


def _verify_live_dependency_bindings() -> None:
    for label, module_file, source_path, expected_sha256 in _source_bindings():
        actual_sha256 = _bounded_source_sha256(module_file, source_path)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                f"{label} source binding drifted"
            )

    descriptor_checks = (
        (
            _base_qualification.QUALIFICATION_PLAN_DESCRIPTOR_SCHEMA_VERSION,
            _BASE_DESCRIPTOR_SCHEMA_VERSION,
            _base_qualification.QUALIFICATION_PLAN_DESCRIPTOR_SHA256,
            _BASE_DESCRIPTOR_SHA256,
            _base_qualification.canonical_matched_v3_qualification_plan_descriptor_bytes(),
        ),
        (
            _bridge.FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
            _BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
            _bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
            _BRIDGE_DESCRIPTOR_SHA256,
            _bridge.canonical_matched_v3_foragax_bridge_descriptor_bytes(),
        ),
        (
            _ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION,
            _CORE_DESCRIPTOR_SCHEMA_VERSION,
            _ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256,
            _CORE_DESCRIPTOR_SHA256,
            _ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes(),
        ),
        (
            _v1_runner.PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            _V1_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            _v1_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
            _V1_RUNNER_DESCRIPTOR_SHA256,
            _v1_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes(),
        ),
        (
            _compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            _COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            _compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256,
            _COMPILED_RUNNER_DESCRIPTOR_SHA256,
            _compiled_runner.canonical_matched_v3_ppo_gru_compiled_runner_descriptor_bytes(),
        ),
        (
            _compiled_bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            _compiled_bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256,
            _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
            _compiled_bundle.canonical_compiled_reward_bundle_descriptor_bytes(),
        ),
        (
            _compiled_publication.COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            _COMPILED_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            _compiled_publication.COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
            _COMPILED_PUBLICATION_DESCRIPTOR_SHA256,
            _compiled_publication.canonical_compiled_reward_publication_descriptor_bytes(),
        ),
    )
    for live_schema, expected_schema, live_sha256, expected_sha256, raw in descriptor_checks:
        if (
            live_schema != expected_schema
            or live_sha256 != expected_sha256
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256)
        ):
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                "compiled qualification-addendum descriptor dependency drifted"
            )

    if (
        _ppo_gru.PPO_GRU_CONFIGURATION_SHA256 != _CORE_CONFIGURATION_SHA256
        or hashlib.sha256(_ppo_gru.canonical_matched_v3_ppo_gru_configuration_bytes()).hexdigest()
        != _CORE_CONFIGURATION_SHA256
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification-addendum configuration dependency drifted"
        )
    runner_descriptor = _compiled_runner.matched_v3_ppo_gru_compiled_runner_descriptor()
    if not _exact_json_equal(
        runner_descriptor.get("geometry"), _ACCOUNTING
    ) or not _exact_json_equal(runner_descriptor.get("rng"), _RNG_CONTRACT):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled runner accounting or RNG contract drifted"
        )
    publication_descriptor = _compiled_publication.compiled_reward_publication_descriptor()
    if not _exact_json_equal(publication_descriptor.get("exact_files"), _EXACT_PUBLICATION_FILES):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled publication six-file contract drifted"
        )


def _parse_base_plan(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_trust_receipt_file_sha256: str,
    expected_trust_receipt_binding_sha256: str,
) -> dict[str, Any]:
    expected_file_sha256 = _require_sha256(expected_file_sha256, "expected base plan file")
    _require_sha256(
        expected_trust_receipt_file_sha256,
        "expected base seed trust-receipt file",
    )
    _require_sha256(
        expected_trust_receipt_binding_sha256,
        "expected base seed trust-receipt binding",
    )
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_file_sha256
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base qualification-plan full-file digest disagrees"
        )
    _verify_live_dependency_bindings()
    try:
        return _base_qualification.parse_matched_v3_qualification_plan_artifact(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_qualification_seed_trust_root_receipt_file_sha256=(
                expected_trust_receipt_file_sha256
            ),
            expected_qualification_seed_trust_root_receipt_binding_sha256=(
                expected_trust_receipt_binding_sha256
            ),
        )
    except _base_qualification.ForagerMatchedV3QualificationPlanError as exc:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base qualification plan failed its frozen strict parser"
        ) from exc


def _single_by_candidate(items: Any, label: str) -> dict[str, Any]:
    if type(items) is not list:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"base {label} must be an exact list"
        )
    matches = [
        item for item in items if type(item) is dict and item.get("candidate_id") == _CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            f"base {label} does not contain exactly one compiled-PPO candidate"
        )
    return cast(dict[str, Any], matches[0])


def _base_reuse(base_plan: Mapping[str, Any]) -> dict[str, Any]:
    bindings = base_plan.get("bindings")
    if type(bindings) is not dict or type(bindings.get("dependencies")) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base qualification dependency bindings are not exact"
        )
    configuration_binding = bindings["dependencies"].get("configuration_plan")
    if type(configuration_binding) is not dict or not _exact_json_equal(
        configuration_binding,
        {
            "schema_version": _CONFIGURATION_PLAN_SCHEMA_VERSION,
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base configuration-plan full artifact binding drifted"
        )
    candidate = _single_by_candidate(base_plan.get("candidate_requirements"), "candidate contract")
    seed_boundary = base_plan.get("seed_boundary")
    resource_contract = base_plan.get("resource_contract")
    if type(seed_boundary) is not dict or type(resource_contract) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base seed or resource contract is not exact"
        )
    seed_case = _single_by_candidate(seed_boundary.get("cases"), "qualification seed cases")
    resource = _single_by_candidate(resource_contract.get("requirements"), "resource requirements")
    sources = base_plan.get("source_requirements")
    if type(sources) is not list:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base source requirements must be an exact list"
        )
    local_sources = [
        item for item in sources if type(item) is dict and item.get("source_id") == "local_alberta"
    ]
    if len(local_sources) != 1:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base plan lacks one exact local source requirement"
        )
    runtime = base_plan.get("runtime_requirement")
    if type(runtime) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base runtime requirement must be one exact object"
        )
    if (
        candidate.get("configuration_record_sha256") != _CONFIGURATION_RECORD_SHA256
        or candidate.get("source_id") != "local_alberta"
        or seed_case.get("material_class") != "public_nonbenchmark_permanently_consumed"
        or seed_case.get("candidate_id") != _CANDIDATE_ID
        or resource.get("candidate_id") != _CANDIDATE_ID
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base compiled-PPO requirement identity drifted"
        )
    for field, minimum in (
        ("max_environment_interactions", _ACCOUNTING["environment_interactions"]),
        ("max_optimizer_updates", _ACCOUNTING["optimizer_updates"]),
        ("max_gradient_updates", _ACCOUNTING["optimizer_updates"]),
    ):
        value = resource.get(field)
        if type(value) is not int or value < minimum:
            raise ForagerMatchedV3CompiledQualificationAddendumError(
                f"base resource ceiling {field} cannot cover compiled accounting"
            )
    publisher = candidate.get("result_publication_binding")
    if type(publisher) is not dict or (
        publisher.get("descriptor_sha256") != _BASE_PUBLICATION_DESCRIPTOR_SHA256
        or publisher.get("implementation_source_sha256") != _BASE_PUBLICATION_SOURCE_SHA256
        or publisher.get("reload_validator_descriptor_sha256")
        != _BASE_PUBLICATION_DESCRIPTOR_SHA256
        or publisher.get("reload_validator_source_sha256") != _BASE_PUBLICATION_SOURCE_SHA256
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "base v1 publisher identity drifted or was silently treated as compiled"
        )
    candidate_static = {
        key: value for key, value in candidate.items() if key != "result_publication_binding"
    }
    return {
        "candidate_static_contract": candidate_static,
        "qualification_seed_case": seed_case,
        "resource_requirement": resource,
        "local_source_requirement": local_sources[0],
        "shared_runtime_requirement": runtime,
        "excluded_base_result_publication": {
            "binding": publisher,
            "binding_sha256": hashlib.sha256(_canonical_json(publisher)).hexdigest(),
            "disposition": "excluded_not_reused_not_reinterpreted_as_compiled",
        },
    }


def _compiled_overlay() -> dict[str, Any]:
    dependencies = _dependency_bindings()
    return {
        "candidate_id": _CANDIDATE_ID,
        "operation": "replace_execution_and_result_publication_path_in_addendum_only",
        "base_plan_mutated": False,
        "base_plan_amended": False,
        "base_plan_superseded": False,
        "base_result_publication_reused": False,
        "v1_runner_selected_for_execution": False,
        "ordered_chain": [
            dependencies["compiled_runner"],
            dependencies["compiled_reward_bundle"],
            dependencies["compiled_reward_publication"],
        ],
        "bridge": dependencies["foragax_bridge"],
        "ppo_gru_core": dependencies["ppo_gru_core"],
        "configuration": dependencies["configuration_plan"],
        "accounting": dict(_ACCOUNTING),
        "rng": dict(_RNG_CONTRACT),
        "publication": {
            "exact_file_count": 6,
            "exact_files": dict(_EXACT_PUBLICATION_FILES),
            "canonical_reward_npz_size_bytes": 499_980,
            "full_file_digest_required_for_future_load": True,
            "structural_replay_required": True,
            "result_accepted_here": False,
        },
    }


def _body_from_base(
    base_plan: Mapping[str, Any],
    *,
    base_plan_file_sha256: str,
    trust_receipt_file_sha256: str,
    trust_receipt_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": COMPILED_QUALIFICATION_ADDENDUM_SCHEMA_VERSION,
        "status": COMPILED_QUALIFICATION_ADDENDUM_STATUS,
        "classification": COMPILED_QUALIFICATION_ADDENDUM_CLASSIFICATION,
        "candidate_id": _CANDIDATE_ID,
        "descriptor_binding": {
            "schema_version": COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256,
        },
        "base_plan_binding": {
            "schema_version": _BASE_PLAN_SCHEMA_VERSION,
            "full_file_sha256": base_plan_file_sha256,
            "body_sha256": base_plan["plan_body_sha256"],
            "descriptor_sha256": _BASE_DESCRIPTOR_SHA256,
            "source_sha256": _BASE_SOURCE_SHA256,
            "trust_receipt_file_sha256": trust_receipt_file_sha256,
            "trust_receipt_binding_sha256": trust_receipt_binding_sha256,
            "strictly_parsed_with_independent_pins": True,
            "embedded": False,
            "amended": False,
            "superseded": False,
        },
        "reused_base_requirements": _base_reuse(base_plan),
        "compiled_execution_and_publication_overlay": _compiled_overlay(),
        "future_qualification_requirements": _future_requirements(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _validate_addendum(
    value: Mapping[str, Any],
    *,
    base_plan: Mapping[str, Any],
    base_plan_file_sha256: str,
    trust_receipt_file_sha256: str,
    trust_receipt_binding_sha256: str,
) -> None:
    if type(value) is not dict:
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum must be a plain object"
        )
    supplied = dict(value)
    supplied_body_sha256 = supplied.pop("addendum_body_sha256", None)
    _require_sha256(supplied_body_sha256, "compiled qualification addendum body")
    expected = _body_from_base(
        base_plan,
        base_plan_file_sha256=base_plan_file_sha256,
        trust_receipt_file_sha256=trust_receipt_file_sha256,
        trust_receipt_binding_sha256=trust_receipt_binding_sha256,
    )
    if not _exact_json_equal(supplied, expected):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum differs from its exact base-bound overlay"
        )
    calculated_body_sha256 = hashlib.sha256(_canonical_json(expected)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, calculated_body_sha256):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum body digest disagrees"
        )
    if any(value is not False for value in cast(dict[str, Any], supplied["claims"]).values()):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification addendum claim became true"
        )
    _assert_plain_unaliased_json(value)
    _canonical_json(value)


def compiled_qualification_addendum_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the frozen non-authorizing descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_compiled_qualification_addendum_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes, including the newline."""

    return _DESCRIPTOR_BYTES


def compiled_qualification_addendum_descriptor_sha256() -> str:
    """Return the exact frozen descriptor digest."""

    return COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256


def parse_compiled_qualification_addendum_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen addendum descriptor."""

    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification-addendum descriptor identity drifted"
        )
    return value


def build_compiled_ppo_qualification_addendum(
    *,
    base_plan_raw: bytes,
    expected_base_plan_file_sha256: str,
    expected_base_seed_trust_receipt_file_sha256: str,
    expected_base_seed_trust_receipt_binding_sha256: str,
) -> dict[str, Any]:
    """Build one detached content overlay from a strictly pinned base plan."""

    base_plan = _parse_base_plan(
        base_plan_raw,
        expected_file_sha256=expected_base_plan_file_sha256,
        expected_trust_receipt_file_sha256=(expected_base_seed_trust_receipt_file_sha256),
        expected_trust_receipt_binding_sha256=(expected_base_seed_trust_receipt_binding_sha256),
    )
    body = _body_from_base(
        base_plan,
        base_plan_file_sha256=expected_base_plan_file_sha256,
        trust_receipt_file_sha256=expected_base_seed_trust_receipt_file_sha256,
        trust_receipt_binding_sha256=expected_base_seed_trust_receipt_binding_sha256,
    )
    result = {
        **body,
        "addendum_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    _validate_addendum(
        result,
        base_plan=base_plan,
        base_plan_file_sha256=expected_base_plan_file_sha256,
        trust_receipt_file_sha256=expected_base_seed_trust_receipt_file_sha256,
        trust_receipt_binding_sha256=expected_base_seed_trust_receipt_binding_sha256,
    )
    return _strict_json_load(_canonical_json(result))


def canonical_compiled_ppo_qualification_addendum_bytes(
    addendum: Mapping[str, Any],
    *,
    base_plan_raw: bytes,
    expected_base_plan_file_sha256: str,
    expected_base_seed_trust_receipt_file_sha256: str,
    expected_base_seed_trust_receipt_binding_sha256: str,
) -> bytes:
    """Validate and encode one base-bound addendum without executing a workload."""

    base_plan = _parse_base_plan(
        base_plan_raw,
        expected_file_sha256=expected_base_plan_file_sha256,
        expected_trust_receipt_file_sha256=(expected_base_seed_trust_receipt_file_sha256),
        expected_trust_receipt_binding_sha256=(expected_base_seed_trust_receipt_binding_sha256),
    )
    _validate_addendum(
        addendum,
        base_plan=base_plan,
        base_plan_file_sha256=expected_base_plan_file_sha256,
        trust_receipt_file_sha256=expected_base_seed_trust_receipt_file_sha256,
        trust_receipt_binding_sha256=expected_base_seed_trust_receipt_binding_sha256,
    )
    return _canonical_json(addendum)


def compiled_ppo_qualification_addendum_sha256(
    addendum: Mapping[str, Any],
    *,
    base_plan_raw: bytes,
    expected_base_plan_file_sha256: str,
    expected_base_seed_trust_receipt_file_sha256: str,
    expected_base_seed_trust_receipt_binding_sha256: str,
) -> str:
    """Return the full-file digest of one validated addendum."""

    return hashlib.sha256(
        canonical_compiled_ppo_qualification_addendum_bytes(
            addendum,
            base_plan_raw=base_plan_raw,
            expected_base_plan_file_sha256=expected_base_plan_file_sha256,
            expected_base_seed_trust_receipt_file_sha256=(
                expected_base_seed_trust_receipt_file_sha256
            ),
            expected_base_seed_trust_receipt_binding_sha256=(
                expected_base_seed_trust_receipt_binding_sha256
            ),
        )
    ).hexdigest()


def parse_compiled_ppo_qualification_addendum_artifact(
    raw: bytes,
    *,
    expected_file_sha256: str,
    base_plan_raw: bytes,
    expected_base_plan_file_sha256: str,
    expected_base_seed_trust_receipt_file_sha256: str,
    expected_base_seed_trust_receipt_binding_sha256: str,
) -> dict[str, Any]:
    """Parse one addendum only with independent addendum, base, and receipt pins."""

    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "expected compiled qualification-addendum file"
    )
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_file_sha256
    ):
        raise ForagerMatchedV3CompiledQualificationAddendumError(
            "compiled qualification-addendum full-file digest disagrees"
        )
    value = _strict_json_load(raw)
    base_plan = _parse_base_plan(
        base_plan_raw,
        expected_file_sha256=expected_base_plan_file_sha256,
        expected_trust_receipt_file_sha256=(expected_base_seed_trust_receipt_file_sha256),
        expected_trust_receipt_binding_sha256=(expected_base_seed_trust_receipt_binding_sha256),
    )
    _validate_addendum(
        value,
        base_plan=base_plan,
        base_plan_file_sha256=expected_base_plan_file_sha256,
        trust_receipt_file_sha256=expected_base_seed_trust_receipt_file_sha256,
        trust_receipt_binding_sha256=expected_base_seed_trust_receipt_binding_sha256,
    )
    return value


__all__ = [
    "COMPILED_QUALIFICATION_ADDENDUM_CLASSIFICATION",
    "COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SCHEMA_VERSION",
    "COMPILED_QUALIFICATION_ADDENDUM_DESCRIPTOR_SHA256",
    "COMPILED_QUALIFICATION_ADDENDUM_SCHEMA_VERSION",
    "COMPILED_QUALIFICATION_ADDENDUM_STATUS",
    "ForagerMatchedV3CompiledQualificationAddendumError",
    "build_compiled_ppo_qualification_addendum",
    "canonical_compiled_ppo_qualification_addendum_bytes",
    "canonical_compiled_qualification_addendum_descriptor_bytes",
    "compiled_ppo_qualification_addendum_sha256",
    "compiled_qualification_addendum_descriptor",
    "compiled_qualification_addendum_descriptor_sha256",
    "parse_compiled_ppo_qualification_addendum_artifact",
    "parse_compiled_qualification_addendum_descriptor",
]
