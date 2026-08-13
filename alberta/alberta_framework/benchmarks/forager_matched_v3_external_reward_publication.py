"""Payload-value-undecoded atomic publication for matched-v3 external outcomes.

An ordinary package import exposes detached descriptor and metadata surfaces only.
The live interface works after this exact source is direct-loaded under its isolated
name after the exact private atomic helper.  It accepts only a live runner outcome
capability plus caller-carried identities; it never accepts a public completion,
bridge conversion, reward bytes, callback, or sink.

The exact outcome consumer is loaded later and calls the private sink captured from
this module.  This one-way trust avoids a source-hash cycle: the consumer pins this
publisher, while this publisher validates caller-injected consumer source and
descriptor pins at call time.  Publication makes exactly one atomic-helper call.
Collision and uncertain states pass through without retry.  Public publish and reload
return frozen digest metadata only; helper mappings and file bytes are discarded.

Persisted content is score/reward-bearing, and file sizes/digests may be side channels.
This module claims controller path discipline, not same-process Python security,
same-UID confidentiality, qualification, evidence, or promotion authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
EXTERNAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_manifest.v1"
)
EXTERNAL_OUTCOME_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_outcome_manifest.v1"
)
EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_metadata.v1"
)
EXTERNAL_REWARD_PUBLICATION_STATUS: Final = "implemented_unexecuted_non_authorizing"
EXTERNAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_reward_publication_isolated_v1"
)

PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_atomic_publication_isolated_v1"
)
PINNED_ATOMIC_PUBLICATION_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_atomic_publication.py"
)
PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256: Final = (
    "8e7ccf6333c7cd8d932a190bc69aed969be93fdad450df7d5b6f8cbb785fc587"
)
PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "b224fe9fdc438ccab0df5bfd3199e1d264feacbb99147970cc68a9c703b9e98e"
)

PINNED_EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_outcome_consumer_isolated_v1"
)
PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_outcome_consumer_descriptor.v1"
)

EXTERNAL_PUBLICATION_ROLE_PATHS: Final = (
    ("publication_manifest", "publication.json"),
    ("outcome_manifest", "external-outcome-manifest.json"),
    ("execution_receipt", "external-execution-receipt.json"),
    ("conversion_receipt", "external-conversion-receipt.json"),
    ("upstream_reward_npz", "upstream-reward.npz"),
    ("upstream_results_database", "upstream-results.db"),
    ("upstream_video_slot", "upstream-video-slot.bin"),
    ("canonical_reward_npz", "reward-trace.npz"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)
EXTERNAL_PUBLICATION_FILENAMES: Final = tuple(
    path for _role, path in EXTERNAL_PUBLICATION_ROLE_PATHS
)
_PAYLOAD_ROLE_PATHS: Final = EXTERNAL_PUBLICATION_ROLE_PATHS[1:]
_PAYLOAD_FILENAMES: Final = EXTERNAL_PUBLICATION_FILENAMES[1:]
_ROLE_BY_PATH: Final = {path: role for role, path in EXTERNAL_PUBLICATION_ROLE_PATHS}

EXTERNAL_PUBLICATION_CANDIDATE_IDS: Final = (
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
)
_PPO_CANDIDATE_IDS: Final = (
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
)

MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES: Final = 1024 * 1024 * 1024
MATCHED_V3_INTERACTION_HORIZON: Final = 499_712
_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
_MAX_METADATA_BYTES: Final = 4 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_TEXT_BYTES: Final = 4 * 1024 * 1024
_UINT31_MAX: Final = 2**31 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_TYPE: Final = type(Path())

_PUBLICATION_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256"
)
_CONSUMER_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256"
)
_CONSUMER_DESCRIPTOR_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")
_SELF_MODULE_AT_LOAD: Final = (
    sys.modules.get(_MODULE_NAME_INPUT) if type(_MODULE_NAME_INPUT) is str else None
)
_ISOLATED_PUBLICATION_BOUNDARY: Final = (
    _MODULE_NAME_INPUT == EXTERNAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
    and (_MODULE_PACKAGE_INPUT is None or _MODULE_PACKAGE_INPUT == "")
    and type(_SELF_MODULE_AT_LOAD) is types.ModuleType
    and _SELF_MODULE_AT_LOAD.__dict__ is globals()
)

_ATOMIC_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
)
_ATOMIC_PUBLISH_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "publish_exact_flat_publication", None
)
_ATOMIC_LOAD_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "load_exact_flat_publication", None
)
_ATOMIC_RECORD_TYPE_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "ExactFileRecord", None)
_ATOMIC_RESULT_TYPE_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "ContentVerifiedFlatPublication", None
)
_ATOMIC_UNCERTAIN_TYPE_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "ForagerMatchedV3AtomicPublicationUncertainError", None
)
_ATOMIC_OPEN_DIRECTORY_TYPE_AT_LOAD: Final = getattr(
    _ATOMIC_MODULE_AT_LOAD, "_OpenDirectory", None
)
_ATOMIC_OPEN_PARENT_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "_open_parent", None)
_ATOMIC_CLOSE_AT_LOAD: Final = getattr(_ATOMIC_MODULE_AT_LOAD, "_close_no_raise", None)
_ATOMIC_FUNCTION_SURFACE_AT_LOAD: Final = (
    tuple(
        sorted(
            (
                name,
                value,
                value.__code__,
                value.__defaults__,
                value.__kwdefaults__,
            )
            for name, value in vars(_ATOMIC_MODULE_AT_LOAD).items()
            if type(_ATOMIC_MODULE_AT_LOAD) is types.ModuleType
            and type(name) is str
            and type(value) is types.FunctionType
            and value.__module__ == PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
        )
    )
    if type(_ATOMIC_MODULE_AT_LOAD) is types.ModuleType
    else ()
)


class ForagerMatchedV3ExternalRewardPublicationError(RuntimeError):
    """A live binding, exact inventory, manifest, or metadata failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3ExternalRewardPublicationError(message)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"external publication JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"external publication JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("external publication JSON integer exceeds its digit bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("external publication JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _assert_plain_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("external publication JSON exceeds its complexity bound")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_TEXT_BYTES:
                _fail("external publication JSON text exceeds its byte bound")
            return
        if type(item) not in {dict, list}:
            _fail("external publication JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("external publication JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _fail("external publication JSON keys must be exact strings")
            for child in item.values():
                visit(child, depth + 1)
        else:
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: Any, *, maximum: int) -> bytes:
    _assert_plain_json(value)
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
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalRewardPublicationError(
            "external publication value is not canonical JSON"
        ) from exc
    if not 1 <= len(raw) <= maximum:
        _fail("external publication canonical JSON exceeds its byte bound")
    return raw


def _strict_json(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        _fail("external publication JSON bytes exceed their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3ExternalRewardPublicationError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalRewardPublicationError(
            "external publication content is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("external publication JSON root must be an object")
    _assert_plain_json(value)
    if not hmac.compare_digest(_canonical_json(value, maximum=maximum), raw):
        _fail("external publication JSON is not canonical")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_uint31(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        _fail(f"{label} must be one exact uint31")
    return value


def _require_candidate(value: object) -> str:
    if type(value) is not str or value not in EXTERNAL_PUBLICATION_CANDIDATE_IDS:
        _fail("candidate ID is not one exact external candidate")
    return value


def _family(candidate_id: str) -> Literal["continuing", "ppo"]:
    return "ppo" if candidate_id in _PPO_CANDIDATE_IDS else "continuing"


def _video_slot_mode(
    *, candidate_id: str, raw: bytes
) -> Literal["absent_for_continuing_zero_length_slot", "opaque_ppo_video"]:
    """Validate the fixed video slot solely from the frozen candidate family."""

    candidate = _require_candidate(candidate_id)
    if type(raw) is not bytes:
        _fail("video slot must be exact immutable bytes")
    if _family(candidate) == "continuing":
        if raw:
            _fail("continuing candidates require the exact empty video slot")
        return "absent_for_continuing_zero_length_slot"
    if not raw:
        _fail("PPO candidates require one nonempty opaque video slot")
    return "opaque_ppo_video"


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        _fail(f"source path differs from {expected_suffix}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise ForagerMatchedV3ExternalRewardPublicationError(
            f"cannot open exact source {expected_suffix}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAX_SOURCE_BYTES
        ):
            _fail(f"source is not one bounded single-link file: {expected_suffix}")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"source truncated while reading: {expected_suffix}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"source grew while reading: {expected_suffix}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    if identity(before) != identity(after):
        _fail(f"source changed while reading: {expected_suffix}")
    return digest.hexdigest()


def _claims() -> dict[str, bool]:
    return {
        "campaign_ingestion_authorized": False,
        "candidate_qualified": False,
        "evidence_authority": False,
        "execution_authorized": False,
        "performance_claim_allowed": False,
        "qualification_authority": False,
        "result_accepted": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "status": EXTERNAL_REWARD_PUBLICATION_STATUS,
        "classification": (
            "payload_score_fields_not_decoded_atomic_external_publication_"
            "metadata_only_non_authorizing"
        ),
        "candidate_count": len(EXTERNAL_PUBLICATION_CANDIDATE_IDS),
        "candidate_order": list(EXTERNAL_PUBLICATION_CANDIDATE_IDS),
        "load_order": {
            "atomic_before_publisher": True,
            "publisher_before_consumer": True,
            "consumer_before_runner": True,
            "bridge_loaded_only_after_runner_outcome_claim": True,
            "static_mutual_source_hash_cycle": False,
        },
        "public_publish_interface": {
            "accepts_live_outcome_capability": True,
            "accepts_public_completion": False,
            "accepts_bridge_conversion": False,
            "accepts_payload_bytes": False,
            "accepts_callback_or_sink": False,
            "returns_immutable_metadata_only": True,
        },
        "publisher_policy": {
            "score_or_reward_magnitude_decoded": False,
            "score_or_reward_value_branching": False,
            "collision_or_uncertain_retry": False,
            "payload_bytes_returned_to_controller": False,
            "metadata_size_and_digest_side_channels_admitted": True,
            "safe_parent_preflight_before_runner_claim": True,
            "atomic_commit_reopens_and_reverifies_parent": True,
            "parent_preflight_eliminates_toctou": False,
        },
        "publication": {
            "manifest_schema_version": EXTERNAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION,
            "outcome_manifest_schema_version": EXTERNAL_OUTCOME_MANIFEST_SCHEMA_VERSION,
            "metadata_schema_version": EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION,
            "exact_file_count": len(EXTERNAL_PUBLICATION_FILENAMES),
            "exact_filenames": list(EXTERNAL_PUBLICATION_FILENAMES),
            "address": "full_sha256_of_publication_json",
            "maximum_total_bytes": MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES,
            "atomic_helper_call_count": 1,
            "collision_retry": False,
            "uncertain_state_retry": False,
            "raw_helper_result_publicly_exposed": False,
            "terminal_metadata": {
                "interaction_horizon_explicit": True,
                "publication_committed_explicit": True,
                "atomic_helper_intent_sha256": True,
                "atomic_publication_receipt_sha256": True,
                "atomic_publication_receipt_semantics": (
                    "canonical_digest_of_content_verified_helper_result_metadata"
                ),
                "reload_observation_sha256": True,
                "score_or_reward_value_fields": False,
                "raw_score_reward_or_payload_fields_recursively_forbidden": True,
            },
        },
        "bindings": {
            "atomic_descriptor_schema_version": (
                PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "atomic_descriptor_sha256": PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
            "atomic_source_sha256": PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
            "consumer_descriptor_schema_version": (
                PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "consumer_source_and_descriptor_caller_injected": True,
        },
        "reload": {
            "caller_carried_address_required": True,
            "caller_carried_exact_file_records_required": True,
            "runner_consumer_bridge_scorer_or_protocol_loaded": False,
            "fresh_process_absence_checked": True,
            "returns_immutable_metadata_only": True,
        },
        "limitations": [
            "Persisted content is score/reward-bearing even though returned metadata is not.",
            "File sizes and digests are not information-theoretically score opaque.",
            "Same-process Python mutation is outside the security boundary.",
            "Same-UID filesystem confidentiality is not claimed.",
            "The pre-claim parent preflight does not eliminate filesystem TOCTOU.",
            "A fresh isolated worker and separate host cgroup proof remain required.",
        ],
        "claims": _claims(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor(), maximum=_MAX_DESCRIPTOR_BYTES)
EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "59d470d6c31e1d3dce8eded401e6331994ca007b94524d8e00714c1f2c66f30b"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
):
    raise AssertionError("external reward publication descriptor identity drifted")


@dataclass(frozen=True, slots=True)
class MatchedV3ExternalPublicationFile:
    """One immutable semantic role/name/size/digest record without content."""

    role: str
    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        expected = dict(EXTERNAL_PUBLICATION_ROLE_PATHS).get(self.role)
        if type(self.role) is not str or expected is None or self.name != expected:
            _fail("external publication file role/name binding is invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            _fail("external publication file size is invalid")
        _require_sha256(self.sha256, "external publication file digest")


@dataclass(frozen=True, slots=True)
class _ExternalPublicationFacts:
    candidate_id: str
    external_candidate_ordinal: int
    family: Literal["continuing", "ppo"]
    qualification_plan_sha256: str
    qualification_case_manifest_sha256: str
    publisher_source_tree_sha256: str
    workload_source_tree_sha256: str
    staging_manifest_sha256: str
    environment_seed_commitment_sha256: str
    agent_seed_commitment_sha256: str
    runner_descriptor_sha256: str
    runner_source_sha256: str
    consumer_descriptor_sha256: str
    consumer_source_sha256: str
    bridge_descriptor_sha256: str
    bridge_source_sha256: str
    scorer_source_sha256: str
    protocol_source_sha256: str
    metric_descriptor_sha256: str
    execution_contract_descriptor_sha256: str
    staging_descriptor_sha256: str
    seed_transport_descriptor_sha256: str
    execution_receipt_sha256: str
    conversion_receipt_sha256: str
    production_runner_exact: bool
    video_slot_mode: Literal[
        "absent_for_continuing_zero_length_slot", "opaque_ppo_video"
    ]
    maximum_publication_total_bytes: int


@dataclass(frozen=True, slots=True)
class MatchedV3ExternalPublicationMetadata:
    """Immutable digest metadata; no published file bytes are retained."""

    schema_version: str
    operation: Literal["published", "reloaded"]
    publication_root: Path
    address: str
    interaction_horizon: int
    publication_committed: bool
    candidate_id: str
    external_candidate_ordinal: int
    family: Literal["continuing", "ppo"]
    qualification_plan_sha256: str
    qualification_case_manifest_sha256: str
    publisher_source_tree_sha256: str
    workload_source_tree_sha256: str
    staging_manifest_sha256: str
    environment_seed_commitment_sha256: str
    agent_seed_commitment_sha256: str
    publication_manifest_body_sha256: str
    outcome_manifest_sha256: str
    outcome_manifest_body_sha256: str
    execution_receipt_sha256: str
    conversion_receipt_sha256: str
    pipeline_binding_sha256: str
    video_slot_mode: Literal[
        "absent_for_continuing_zero_length_slot", "opaque_ppo_video"
    ]
    production_runner_exact: bool
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    consumer_descriptor_sha256: str
    consumer_source_sha256: str
    runner_descriptor_sha256: str
    runner_source_sha256: str
    bridge_descriptor_sha256: str
    bridge_source_sha256: str
    scorer_source_sha256: str
    protocol_source_sha256: str
    metric_descriptor_sha256: str
    execution_contract_descriptor_sha256: str
    staging_descriptor_sha256: str
    seed_transport_descriptor_sha256: str
    atomic_descriptor_sha256: str
    atomic_source_sha256: str
    atomic_helper_intent_sha256: str
    atomic_publication_receipt_sha256: str
    reload_observation_sha256: str
    file_count: int
    total_size_bytes: int
    maximum_publication_total_bytes: int
    inventory_sha256: str
    files: tuple[MatchedV3ExternalPublicationFile, ...]
    metadata_body_sha256: str

    def __post_init__(self) -> None:
        _validate_metadata(self)


def _validate_facts(value: object) -> _ExternalPublicationFacts:
    if type(value) is not _ExternalPublicationFacts:
        _fail("external publication facts type is not exact")
    facts = value
    candidate = _require_candidate(facts.candidate_id)
    if (
        type(facts.external_candidate_ordinal) is not int
        or facts.external_candidate_ordinal
        != EXTERNAL_PUBLICATION_CANDIDATE_IDS.index(candidate)
        or facts.family != _family(candidate)
    ):
        _fail("external publication candidate order or family differs")
    for label, digest in (
        ("qualification plan", facts.qualification_plan_sha256),
        ("qualification case", facts.qualification_case_manifest_sha256),
        ("publisher source tree", facts.publisher_source_tree_sha256),
        ("workload source tree", facts.workload_source_tree_sha256),
        ("staging manifest", facts.staging_manifest_sha256),
        ("environment seed commitment", facts.environment_seed_commitment_sha256),
        ("agent seed commitment", facts.agent_seed_commitment_sha256),
        ("runner descriptor", facts.runner_descriptor_sha256),
        ("runner source", facts.runner_source_sha256),
        ("consumer descriptor", facts.consumer_descriptor_sha256),
        ("consumer source", facts.consumer_source_sha256),
        ("bridge descriptor", facts.bridge_descriptor_sha256),
        ("bridge source", facts.bridge_source_sha256),
        ("scorer source", facts.scorer_source_sha256),
        ("protocol source", facts.protocol_source_sha256),
        ("metric descriptor", facts.metric_descriptor_sha256),
        ("execution contract", facts.execution_contract_descriptor_sha256),
        ("staging descriptor", facts.staging_descriptor_sha256),
        ("seed transport", facts.seed_transport_descriptor_sha256),
        ("execution receipt", facts.execution_receipt_sha256),
        ("conversion receipt", facts.conversion_receipt_sha256),
    ):
        _require_sha256(digest, label)
    if type(facts.production_runner_exact) is not bool:
        _fail("external publication production-runner flag is not exact")
    if facts.video_slot_mode not in {
        "absent_for_continuing_zero_length_slot",
        "opaque_ppo_video",
    }:
        _fail("external publication video-slot mode differs")
    expected_mode = (
        "opaque_ppo_video"
        if facts.family == "ppo"
        else "absent_for_continuing_zero_length_slot"
    )
    if facts.video_slot_mode != expected_mode:
        _fail("external publication video-slot mode disagrees with family")
    if (
        type(facts.maximum_publication_total_bytes) is not int
        or not 1
        <= facts.maximum_publication_total_bytes
        <= MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES
    ):
        _fail("external publication aggregate ceiling is invalid")
    return facts


def _record_body(record: MatchedV3ExternalPublicationFile) -> dict[str, Any]:
    return {
        "role": record.role,
        "name": record.name,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    }


def _records_body(records: tuple[MatchedV3ExternalPublicationFile, ...]) -> list[dict[str, Any]]:
    return [_record_body(record) for record in records]


def _atomic_helper_intent_sha256(
    root: Path,
    records: tuple[MatchedV3ExternalPublicationFile, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": (
                    "alberta.forager_matched_v3.external_atomic_helper_intent.v1"
                ),
                "publication_parent": str(root.parent),
                "publication_root": str(root),
                "address": records[0].sha256,
                "expected_files": _records_body(records),
                "atomic_descriptor_sha256": (
                    PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
                ),
                "atomic_source_sha256": PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
                "exactly_one_call": True,
                "retry_allowed": False,
            },
            maximum=_MAX_METADATA_BYTES,
        )
    ).hexdigest()


def _atomic_publication_receipt_sha256(
    root: Path,
    records: tuple[MatchedV3ExternalPublicationFile, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": (
                    "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"
                ),
                "atomic_helper_intent_sha256": _atomic_helper_intent_sha256(
                    root, records
                ),
                "publication_root": str(root),
                "address": records[0].sha256,
                "publication_committed": True,
                "file_count": len(records),
                "total_size_bytes": sum(record.size_bytes for record in records),
                "inventory_sha256": hashlib.sha256(
                    _canonical_json(
                        _records_body(records), maximum=_MAX_METADATA_BYTES
                    )
                ).hexdigest(),
            },
            maximum=_MAX_METADATA_BYTES,
        )
    ).hexdigest()


def _reload_observation_sha256(
    root: Path,
    records: tuple[MatchedV3ExternalPublicationFile, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": (
                    "alberta.forager_matched_v3.external_reload_observation.v1"
                ),
                "publication_root": str(root),
                "address": records[0].sha256,
                "content_verified": True,
                "exact_files": _records_body(records),
            },
            maximum=_MAX_METADATA_BYTES,
        )
    ).hexdigest()


def _validate_records(
    value: object,
) -> tuple[MatchedV3ExternalPublicationFile, ...]:
    if type(value) is not tuple:
        _fail("external publication file records must be an exact tuple")
    items = cast(tuple[object, ...], value)
    if (
        len(items) != len(EXTERNAL_PUBLICATION_ROLE_PATHS)
        or any(type(item) is not MatchedV3ExternalPublicationFile for item in items)
    ):
        _fail("external publication file record inventory is not exact")
    records = cast(tuple[MatchedV3ExternalPublicationFile, ...], items)
    if tuple((record.role, record.name) for record in records) != EXTERNAL_PUBLICATION_ROLE_PATHS:
        _fail("external publication file record semantic order differs")
    if sum(record.size_bytes for record in records) > MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES:
        _fail("external publication file records exceed the aggregate ceiling")
    return records


def _pipeline_binding(facts: _ExternalPublicationFacts) -> dict[str, Any]:
    return {
        "publisher": {
            "descriptor_sha256": EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
            "source_sha256": _require_sha256(
                _PUBLICATION_SOURCE_SHA256_INPUT, "publisher source"
            ),
        },
        "consumer": {
            "descriptor_sha256": facts.consumer_descriptor_sha256,
            "source_sha256": facts.consumer_source_sha256,
        },
        "runner": {
            "descriptor_sha256": facts.runner_descriptor_sha256,
            "source_sha256": facts.runner_source_sha256,
        },
        "bridge": {
            "descriptor_sha256": facts.bridge_descriptor_sha256,
            "source_sha256": facts.bridge_source_sha256,
        },
        "scorer_source_sha256": facts.scorer_source_sha256,
        "protocol_source_sha256": facts.protocol_source_sha256,
        "metric_descriptor_sha256": facts.metric_descriptor_sha256,
        "atomic": {
            "descriptor_sha256": PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
        },
        "execution_contract_descriptor_sha256": (
            facts.execution_contract_descriptor_sha256
        ),
        "staging_descriptor_sha256": facts.staging_descriptor_sha256,
        "seed_transport_descriptor_sha256": facts.seed_transport_descriptor_sha256,
    }


def _pipeline_binding_sha256(facts: _ExternalPublicationFacts) -> str:
    return hashlib.sha256(
        _canonical_json(_pipeline_binding(facts), maximum=_MAX_MANIFEST_BYTES)
    ).hexdigest()


def _payload_records(payloads: dict[str, bytes]) -> tuple[MatchedV3ExternalPublicationFile, ...]:
    return tuple(
        MatchedV3ExternalPublicationFile(
            role=role,
            name=path,
            size_bytes=len(payloads[path]),
            sha256=hashlib.sha256(payloads[path]).hexdigest(),
        )
        for role, path in _PAYLOAD_ROLE_PATHS
    )


def _outcome_manifest_body(
    facts: _ExternalPublicationFacts,
    content_payloads: dict[str, bytes],
) -> dict[str, Any]:
    facts = _validate_facts(facts)
    expected = _PAYLOAD_FILENAMES[1:]
    if tuple(content_payloads) != expected:
        _fail("external outcome content payload order differs")
    if not hmac.compare_digest(
        hashlib.sha256(
            content_payloads["external-execution-receipt.json"]
        ).hexdigest(),
        facts.execution_receipt_sha256,
    ) or not hmac.compare_digest(
        hashlib.sha256(
            content_payloads["external-conversion-receipt.json"]
        ).hexdigest(),
        facts.conversion_receipt_sha256,
    ):
        _fail("external outcome receipt facts differ from actual content")
    observed_video_mode = _video_slot_mode(
        candidate_id=facts.candidate_id,
        raw=content_payloads["upstream-video-slot.bin"],
    )
    if observed_video_mode != facts.video_slot_mode:
        _fail("external outcome video-slot facts differ from actual content")
    records = tuple(
        MatchedV3ExternalPublicationFile(
            role=_ROLE_BY_PATH[path],
            name=path,
            size_bytes=len(content_payloads[path]),
            sha256=hashlib.sha256(content_payloads[path]).hexdigest(),
        )
        for path in expected
    )
    return {
        "schema_version": EXTERNAL_OUTCOME_MANIFEST_SCHEMA_VERSION,
        "status": "external_outcome_converted_unqualified_non_authorizing",
        "classification": "score_reward_bearing_bundle_manifest_non_authorizing",
        "interaction_horizon": MATCHED_V3_INTERACTION_HORIZON,
        "candidate": {
            "candidate_id": facts.candidate_id,
            "external_candidate_ordinal": facts.external_candidate_ordinal,
            "family": facts.family,
            "production_runner_exact": facts.production_runner_exact,
            "video_slot_mode": facts.video_slot_mode,
        },
        "qualification": {
            "plan_sha256": facts.qualification_plan_sha256,
            "case_manifest_sha256": facts.qualification_case_manifest_sha256,
            "environment_seed_commitment_sha256": (
                facts.environment_seed_commitment_sha256
            ),
            "agent_seed_commitment_sha256": facts.agent_seed_commitment_sha256,
        },
        "sources": {
            "publisher_source_tree_sha256": facts.publisher_source_tree_sha256,
            "workload_source_tree_sha256": facts.workload_source_tree_sha256,
            "staging_manifest_sha256": facts.staging_manifest_sha256,
        },
        "publication_policy": {
            "maximum_total_bytes": facts.maximum_publication_total_bytes,
        },
        "pipeline_binding": _pipeline_binding(facts),
        "files": _records_body(records),
        "execution_receipt_sha256": facts.execution_receipt_sha256,
        "conversion_receipt_sha256": facts.conversion_receipt_sha256,
        "controller_exposure": {
            "metadata_only": True,
            "persisted_content_score_reward_bearing": True,
            "sizes_or_digests_information_theoretically_opaque": False,
        },
        "claims": _claims(),
    }


def _build_external_outcome_manifest(
    *, facts: _ExternalPublicationFacts, content_payloads: tuple[tuple[str, bytes], ...]
) -> bytes:
    """Private captured helper used only by the exact outcome consumer."""

    if type(content_payloads) is not tuple:
        _fail("external outcome manifest content tuple differs")
    items = cast(tuple[object, ...], content_payloads)
    validated: list[tuple[str, bytes]] = []
    for index, item in enumerate(items):
        if (
            index >= len(_PAYLOAD_FILENAMES[1:])
            or type(item) is not tuple
            or len(item) != 2
            or item[0] != _PAYLOAD_FILENAMES[1:][index]
            or type(item[1]) is not bytes
        ):
            _fail("external outcome manifest content tuple differs")
        validated.append((item[0], item[1]))
    if len(validated) != len(_PAYLOAD_FILENAMES[1:]):
        _fail("external outcome manifest content tuple differs")
    payloads = dict(validated)
    body = _outcome_manifest_body(_validate_facts(facts), payloads)
    body_digest = hashlib.sha256(
        _canonical_json(body, maximum=_MAX_MANIFEST_BYTES)
    ).hexdigest()
    return _canonical_json(
        {**body, "manifest_body_sha256": body_digest},
        maximum=_MAX_MANIFEST_BYTES,
    )


def _parse_outcome_manifest(
    raw: bytes,
    *,
    facts: _ExternalPublicationFacts,
    content_payloads: dict[str, bytes],
) -> tuple[dict[str, Any], str]:
    supplied = _strict_json(raw, maximum=_MAX_MANIFEST_BYTES)
    body = _outcome_manifest_body(facts, content_payloads)
    body_digest = hashlib.sha256(
        _canonical_json(body, maximum=_MAX_MANIFEST_BYTES)
    ).hexdigest()
    expected = {**body, "manifest_body_sha256": body_digest}
    if supplied != expected or not hmac.compare_digest(
        _canonical_json(expected, maximum=_MAX_MANIFEST_BYTES), raw
    ):
        _fail("external outcome manifest semantic replay differs")
    return supplied, body_digest


def _publication_body(
    facts: _ExternalPublicationFacts,
    payload_records: tuple[MatchedV3ExternalPublicationFile, ...],
    *,
    outcome_body_sha256: str,
) -> dict[str, Any]:
    facts = _validate_facts(facts)
    if tuple((record.role, record.name) for record in payload_records) != _PAYLOAD_ROLE_PATHS:
        _fail("external publication payload record order differs")
    return {
        "schema_version": EXTERNAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "status": "external_outcome_published_unqualified_non_authorizing",
        "classification": "score_reward_bearing_content_publication_non_authorizing",
        "interaction_horizon": MATCHED_V3_INTERACTION_HORIZON,
        "candidate": {
            "candidate_id": facts.candidate_id,
            "external_candidate_ordinal": facts.external_candidate_ordinal,
            "family": facts.family,
            "production_runner_exact": facts.production_runner_exact,
            "video_slot_mode": facts.video_slot_mode,
        },
        "qualification": {
            "plan_sha256": facts.qualification_plan_sha256,
            "case_manifest_sha256": facts.qualification_case_manifest_sha256,
        },
        "sources": {
            "publisher_source_tree_sha256": facts.publisher_source_tree_sha256,
            "workload_source_tree_sha256": facts.workload_source_tree_sha256,
            "staging_manifest_sha256": facts.staging_manifest_sha256,
        },
        "pipeline_binding_sha256": _pipeline_binding_sha256(facts),
        "outcome_manifest_body_sha256": outcome_body_sha256,
        "payload_files": _records_body(payload_records),
        "exact_total_file_count": len(EXTERNAL_PUBLICATION_FILENAMES),
        "claims": _claims(),
    }


def _publication_manifest(
    facts: _ExternalPublicationFacts,
    payload_records: tuple[MatchedV3ExternalPublicationFile, ...],
    *,
    outcome_body_sha256: str,
) -> tuple[bytes, str]:
    body = _publication_body(
        facts,
        payload_records,
        outcome_body_sha256=outcome_body_sha256,
    )
    digest = hashlib.sha256(
        _canonical_json(body, maximum=_MAX_MANIFEST_BYTES)
    ).hexdigest()
    raw = _canonical_json(
        {**body, "publication_body_sha256": digest},
        maximum=_MAX_MANIFEST_BYTES,
    )
    return raw, digest


def _parse_publication_manifest(
    raw: bytes,
    *,
    facts: _ExternalPublicationFacts,
    payload_records: tuple[MatchedV3ExternalPublicationFile, ...],
    outcome_body_sha256: str,
) -> str:
    expected, body_digest = _publication_manifest(
        facts,
        payload_records,
        outcome_body_sha256=outcome_body_sha256,
    )
    if not hmac.compare_digest(raw, expected):
        _fail("external publication manifest semantic replay differs")
    _strict_json(raw, maximum=_MAX_MANIFEST_BYTES)
    return body_digest


def _metadata_body(metadata: MatchedV3ExternalPublicationMetadata) -> dict[str, Any]:
    return {
        "schema_version": metadata.schema_version,
        "operation": metadata.operation,
        "publication_root": str(metadata.publication_root),
        "address": metadata.address,
        "interaction_horizon": metadata.interaction_horizon,
        "publication_committed": metadata.publication_committed,
        "candidate_id": metadata.candidate_id,
        "external_candidate_ordinal": metadata.external_candidate_ordinal,
        "family": metadata.family,
        "qualification_plan_sha256": metadata.qualification_plan_sha256,
        "qualification_case_manifest_sha256": (
            metadata.qualification_case_manifest_sha256
        ),
        "publisher_source_tree_sha256": metadata.publisher_source_tree_sha256,
        "workload_source_tree_sha256": metadata.workload_source_tree_sha256,
        "staging_manifest_sha256": metadata.staging_manifest_sha256,
        "environment_seed_commitment_sha256": (
            metadata.environment_seed_commitment_sha256
        ),
        "agent_seed_commitment_sha256": metadata.agent_seed_commitment_sha256,
        "publication_manifest_body_sha256": (
            metadata.publication_manifest_body_sha256
        ),
        "outcome_manifest_sha256": metadata.outcome_manifest_sha256,
        "outcome_manifest_body_sha256": metadata.outcome_manifest_body_sha256,
        "execution_receipt_sha256": metadata.execution_receipt_sha256,
        "conversion_receipt_sha256": metadata.conversion_receipt_sha256,
        "pipeline_binding_sha256": metadata.pipeline_binding_sha256,
        "video_slot_mode": metadata.video_slot_mode,
        "production_runner_exact": metadata.production_runner_exact,
        "publisher_descriptor_sha256": metadata.publisher_descriptor_sha256,
        "publisher_source_sha256": metadata.publisher_source_sha256,
        "consumer_descriptor_sha256": metadata.consumer_descriptor_sha256,
        "consumer_source_sha256": metadata.consumer_source_sha256,
        "runner_descriptor_sha256": metadata.runner_descriptor_sha256,
        "runner_source_sha256": metadata.runner_source_sha256,
        "bridge_descriptor_sha256": metadata.bridge_descriptor_sha256,
        "bridge_source_sha256": metadata.bridge_source_sha256,
        "scorer_source_sha256": metadata.scorer_source_sha256,
        "protocol_source_sha256": metadata.protocol_source_sha256,
        "metric_descriptor_sha256": metadata.metric_descriptor_sha256,
        "execution_contract_descriptor_sha256": (
            metadata.execution_contract_descriptor_sha256
        ),
        "staging_descriptor_sha256": metadata.staging_descriptor_sha256,
        "seed_transport_descriptor_sha256": (
            metadata.seed_transport_descriptor_sha256
        ),
        "atomic_descriptor_sha256": metadata.atomic_descriptor_sha256,
        "atomic_source_sha256": metadata.atomic_source_sha256,
        "atomic_helper_intent_sha256": metadata.atomic_helper_intent_sha256,
        "atomic_publication_receipt_sha256": (
            metadata.atomic_publication_receipt_sha256
        ),
        "reload_observation_sha256": metadata.reload_observation_sha256,
        "file_count": metadata.file_count,
        "total_size_bytes": metadata.total_size_bytes,
        "maximum_publication_total_bytes": metadata.maximum_publication_total_bytes,
        "inventory_sha256": metadata.inventory_sha256,
        "files": _records_body(metadata.files),
    }


def _assert_metadata_has_no_raw_score_reward_or_payload_fields(
    value: dict[str, Any],
) -> None:
    forbidden = {
        "content",
        "conversion",
        "cumulative_score",
        "payload",
        "raw_trace",
        "reward",
        "reward_values",
        "rewards",
        "score",
        "trace",
        "trace_bytes",
    }
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(key in forbidden for key in item):
                _fail(
                    "external publication metadata contains a raw score, reward, "
                    "or payload field"
                )
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)


def _validate_metadata(value: object) -> MatchedV3ExternalPublicationMetadata:
    if type(value) is not MatchedV3ExternalPublicationMetadata:
        _fail("external publication metadata type is not exact")
    metadata = value
    if (
        metadata.schema_version != EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION
        or metadata.operation not in {"published", "reloaded"}
        or type(metadata.publication_root) is not _PATH_TYPE
        or not metadata.publication_root.is_absolute()
        or metadata.publication_root == Path("/")
        or metadata.publication_root.name != metadata.address
        or metadata.interaction_horizon != MATCHED_V3_INTERACTION_HORIZON
        or metadata.publication_committed is not True
    ):
        _fail("external publication metadata identity or root differs")
    candidate = _require_candidate(metadata.candidate_id)
    if (
        metadata.external_candidate_ordinal
        != EXTERNAL_PUBLICATION_CANDIDATE_IDS.index(candidate)
        or metadata.family != _family(candidate)
        or type(metadata.production_runner_exact) is not bool
    ):
        _fail("external publication metadata candidate binding differs")
    records = _validate_records(metadata.files)
    for label, digest in (
        ("address", metadata.address),
        ("qualification plan", metadata.qualification_plan_sha256),
        ("qualification case", metadata.qualification_case_manifest_sha256),
        ("publisher source tree", metadata.publisher_source_tree_sha256),
        ("workload source tree", metadata.workload_source_tree_sha256),
        ("staging manifest", metadata.staging_manifest_sha256),
        ("environment seed commitment", metadata.environment_seed_commitment_sha256),
        ("agent seed commitment", metadata.agent_seed_commitment_sha256),
        ("publication body", metadata.publication_manifest_body_sha256),
        ("outcome manifest", metadata.outcome_manifest_sha256),
        ("outcome body", metadata.outcome_manifest_body_sha256),
        ("execution receipt", metadata.execution_receipt_sha256),
        ("conversion receipt", metadata.conversion_receipt_sha256),
        ("pipeline binding", metadata.pipeline_binding_sha256),
        ("publisher descriptor", metadata.publisher_descriptor_sha256),
        ("publisher source", metadata.publisher_source_sha256),
        ("consumer descriptor", metadata.consumer_descriptor_sha256),
        ("consumer source", metadata.consumer_source_sha256),
        ("runner descriptor", metadata.runner_descriptor_sha256),
        ("runner source", metadata.runner_source_sha256),
        ("bridge descriptor", metadata.bridge_descriptor_sha256),
        ("bridge source", metadata.bridge_source_sha256),
        ("scorer source", metadata.scorer_source_sha256),
        ("protocol source", metadata.protocol_source_sha256),
        ("metric descriptor", metadata.metric_descriptor_sha256),
        ("execution contract", metadata.execution_contract_descriptor_sha256),
        ("staging descriptor", metadata.staging_descriptor_sha256),
        ("seed transport descriptor", metadata.seed_transport_descriptor_sha256),
        ("atomic descriptor", metadata.atomic_descriptor_sha256),
        ("atomic source", metadata.atomic_source_sha256),
        ("atomic helper intent", metadata.atomic_helper_intent_sha256),
        ("atomic publication receipt", metadata.atomic_publication_receipt_sha256),
        ("reload observation", metadata.reload_observation_sha256),
        ("inventory", metadata.inventory_sha256),
        ("metadata body", metadata.metadata_body_sha256),
    ):
        _require_sha256(digest, label)
    expected_mode = (
        "opaque_ppo_video"
        if metadata.family == "ppo"
        else "absent_for_continuing_zero_length_slot"
    )
    if metadata.video_slot_mode != expected_mode:
        _fail("external publication metadata video-slot mode differs")
    if (
        metadata.address != records[0].sha256
        or metadata.publisher_descriptor_sha256
        != EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or metadata.atomic_descriptor_sha256
        != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
        or metadata.atomic_source_sha256 != PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256
        or metadata.file_count != len(records)
        or metadata.total_size_bytes != sum(record.size_bytes for record in records)
        or type(metadata.maximum_publication_total_bytes) is not int
        or not 1
        <= metadata.maximum_publication_total_bytes
        <= MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES
        or metadata.total_size_bytes > metadata.maximum_publication_total_bytes
        or metadata.atomic_helper_intent_sha256
        != _atomic_helper_intent_sha256(metadata.publication_root, records)
        or metadata.atomic_publication_receipt_sha256
        != _atomic_publication_receipt_sha256(metadata.publication_root, records)
        or metadata.reload_observation_sha256
        != _reload_observation_sha256(metadata.publication_root, records)
    ):
        _fail("external publication metadata fixed binding differs")
    inventory = hashlib.sha256(
        _canonical_json(_records_body(records), maximum=_MAX_METADATA_BYTES)
    ).hexdigest()
    if not hmac.compare_digest(inventory, metadata.inventory_sha256):
        _fail("external publication metadata inventory digest differs")
    body = hashlib.sha256(
        _canonical_json(_metadata_body(metadata), maximum=_MAX_METADATA_BYTES)
    ).hexdigest()
    _assert_metadata_has_no_raw_score_reward_or_payload_fields(
        _metadata_body(metadata)
    )
    if not hmac.compare_digest(body, metadata.metadata_body_sha256):
        _fail("external publication metadata body digest differs")
    return metadata


def _make_metadata(
    *,
    operation: Literal["published", "reloaded"],
    root: Path,
    records: tuple[MatchedV3ExternalPublicationFile, ...],
    facts: _ExternalPublicationFacts,
    publication_body_sha256: str,
    outcome_body_sha256: str,
) -> MatchedV3ExternalPublicationMetadata:
    facts = _validate_facts(facts)
    source = _require_sha256(_PUBLICATION_SOURCE_SHA256_INPUT, "publisher source")
    body_values = dict(
        schema_version=EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION,
        operation=operation,
        publication_root=root,
        address=records[0].sha256,
        interaction_horizon=MATCHED_V3_INTERACTION_HORIZON,
        publication_committed=True,
        candidate_id=facts.candidate_id,
        external_candidate_ordinal=facts.external_candidate_ordinal,
        family=facts.family,
        qualification_plan_sha256=facts.qualification_plan_sha256,
        qualification_case_manifest_sha256=facts.qualification_case_manifest_sha256,
        publisher_source_tree_sha256=facts.publisher_source_tree_sha256,
        workload_source_tree_sha256=facts.workload_source_tree_sha256,
        staging_manifest_sha256=facts.staging_manifest_sha256,
        environment_seed_commitment_sha256=facts.environment_seed_commitment_sha256,
        agent_seed_commitment_sha256=facts.agent_seed_commitment_sha256,
        publication_manifest_body_sha256=publication_body_sha256,
        outcome_manifest_sha256=records[1].sha256,
        outcome_manifest_body_sha256=outcome_body_sha256,
        execution_receipt_sha256=facts.execution_receipt_sha256,
        conversion_receipt_sha256=facts.conversion_receipt_sha256,
        pipeline_binding_sha256=_pipeline_binding_sha256(facts),
        video_slot_mode=facts.video_slot_mode,
        production_runner_exact=facts.production_runner_exact,
        publisher_descriptor_sha256=EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        publisher_source_sha256=source,
        consumer_descriptor_sha256=facts.consumer_descriptor_sha256,
        consumer_source_sha256=facts.consumer_source_sha256,
        runner_descriptor_sha256=facts.runner_descriptor_sha256,
        runner_source_sha256=facts.runner_source_sha256,
        bridge_descriptor_sha256=facts.bridge_descriptor_sha256,
        bridge_source_sha256=facts.bridge_source_sha256,
        scorer_source_sha256=facts.scorer_source_sha256,
        protocol_source_sha256=facts.protocol_source_sha256,
        metric_descriptor_sha256=facts.metric_descriptor_sha256,
        execution_contract_descriptor_sha256=(
            facts.execution_contract_descriptor_sha256
        ),
        staging_descriptor_sha256=facts.staging_descriptor_sha256,
        seed_transport_descriptor_sha256=facts.seed_transport_descriptor_sha256,
        atomic_descriptor_sha256=PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        atomic_source_sha256=PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
        atomic_helper_intent_sha256=_atomic_helper_intent_sha256(root, records),
        atomic_publication_receipt_sha256=_atomic_publication_receipt_sha256(
            root, records
        ),
        reload_observation_sha256=_reload_observation_sha256(root, records),
        file_count=len(records),
        total_size_bytes=sum(record.size_bytes for record in records),
        maximum_publication_total_bytes=facts.maximum_publication_total_bytes,
        inventory_sha256=hashlib.sha256(
            _canonical_json(_records_body(records), maximum=_MAX_METADATA_BYTES)
        ).hexdigest(),
        files=records,
    )
    provisional = object.__new__(MatchedV3ExternalPublicationMetadata)
    for name, value in body_values.items():
        object.__setattr__(provisional, name, value)
    metadata_digest = hashlib.sha256(
        _canonical_json(_metadata_body(provisional), maximum=_MAX_METADATA_BYTES)
    ).hexdigest()
    return MatchedV3ExternalPublicationMetadata(
        schema_version=EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION,
        operation=operation,
        publication_root=root,
        address=records[0].sha256,
        interaction_horizon=MATCHED_V3_INTERACTION_HORIZON,
        publication_committed=True,
        candidate_id=facts.candidate_id,
        external_candidate_ordinal=facts.external_candidate_ordinal,
        family=facts.family,
        qualification_plan_sha256=facts.qualification_plan_sha256,
        qualification_case_manifest_sha256=facts.qualification_case_manifest_sha256,
        publisher_source_tree_sha256=facts.publisher_source_tree_sha256,
        workload_source_tree_sha256=facts.workload_source_tree_sha256,
        staging_manifest_sha256=facts.staging_manifest_sha256,
        environment_seed_commitment_sha256=(
            facts.environment_seed_commitment_sha256
        ),
        agent_seed_commitment_sha256=facts.agent_seed_commitment_sha256,
        publication_manifest_body_sha256=publication_body_sha256,
        outcome_manifest_sha256=records[1].sha256,
        outcome_manifest_body_sha256=outcome_body_sha256,
        execution_receipt_sha256=facts.execution_receipt_sha256,
        conversion_receipt_sha256=facts.conversion_receipt_sha256,
        pipeline_binding_sha256=_pipeline_binding_sha256(facts),
        video_slot_mode=facts.video_slot_mode,
        production_runner_exact=facts.production_runner_exact,
        publisher_descriptor_sha256=EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        publisher_source_sha256=source,
        consumer_descriptor_sha256=facts.consumer_descriptor_sha256,
        consumer_source_sha256=facts.consumer_source_sha256,
        runner_descriptor_sha256=facts.runner_descriptor_sha256,
        runner_source_sha256=facts.runner_source_sha256,
        bridge_descriptor_sha256=facts.bridge_descriptor_sha256,
        bridge_source_sha256=facts.bridge_source_sha256,
        scorer_source_sha256=facts.scorer_source_sha256,
        protocol_source_sha256=facts.protocol_source_sha256,
        metric_descriptor_sha256=facts.metric_descriptor_sha256,
        execution_contract_descriptor_sha256=(
            facts.execution_contract_descriptor_sha256
        ),
        staging_descriptor_sha256=facts.staging_descriptor_sha256,
        seed_transport_descriptor_sha256=facts.seed_transport_descriptor_sha256,
        atomic_descriptor_sha256=PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        atomic_source_sha256=PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
        atomic_helper_intent_sha256=_atomic_helper_intent_sha256(root, records),
        atomic_publication_receipt_sha256=_atomic_publication_receipt_sha256(
            root, records
        ),
        reload_observation_sha256=_reload_observation_sha256(root, records),
        file_count=len(records),
        total_size_bytes=sum(record.size_bytes for record in records),
        maximum_publication_total_bytes=facts.maximum_publication_total_bytes,
        inventory_sha256=cast(str, body_values["inventory_sha256"]),
        files=records,
        metadata_body_sha256=metadata_digest,
    )


def _require_parent(value: object) -> Path:
    if (
        type(value) is not _PATH_TYPE
        or not value.is_absolute()
        or value == Path("/")
    ):
        _fail("external publication parent must be one exact absolute non-root Path")
    return value


def _require_atomic_module() -> types.ModuleType:
    module = _ATOMIC_MODULE_AT_LOAD
    if type(module) is not types.ModuleType:
        _fail("exact atomic helper was not loaded before the publisher")
    if sys.modules.get(PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME) is not module:
        _fail("exact atomic helper module identity changed")
    if not hmac.compare_digest(
        _source_sha256(module.__file__, PINNED_ATOMIC_PUBLICATION_SOURCE_PATH),
        PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256,
    ):
        _fail("exact atomic helper source identity changed")
    descriptor = getattr(module, "canonical_atomic_publication_descriptor_bytes", None)
    if type(descriptor) is not types.FunctionType:
        _fail("exact atomic helper descriptor API is unavailable")
    raw = descriptor()
    if (
        type(raw) is not bytes
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        )
        or getattr(module, "ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    ):
        _fail("exact atomic helper descriptor identity changed")
    current_surface = tuple(
        sorted(
            (
                name,
                value,
                value.__code__,
                value.__defaults__,
                value.__kwdefaults__,
            )
            for name, value in vars(module).items()
            if type(name) is str
            and type(value) is types.FunctionType
            and value.__module__ == PINNED_ATOMIC_PUBLICATION_ISOLATED_MODULE_NAME
        )
    )
    if len(current_surface) != len(_ATOMIC_FUNCTION_SURFACE_AT_LOAD) or any(
        current[0] != expected[0]
        or current[1] is not expected[1]
        or current[2] is not expected[2]
        or current[3] is not expected[3]
        or current[4] is not expected[4]
        for current, expected in zip(
            current_surface, _ATOMIC_FUNCTION_SURFACE_AT_LOAD, strict=True
        )
    ):
        _fail("exact atomic helper function surface changed")
    return module


def _require_boundary() -> str:
    if not _ISOLATED_PUBLICATION_BOUNDARY:
        _fail("external publication requires its exact isolated direct-load boundary")
    expected = _require_sha256(_PUBLICATION_SOURCE_SHA256_INPUT, "publisher source")
    current = _source_sha256(globals().get("__file__"), __file__)
    if not hmac.compare_digest(current, expected):
        _fail("external publisher direct-loaded source identity differs")
    if (
        EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        != "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
        or not hmac.compare_digest(
            hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
            EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        )
        or not hmac.compare_digest(
            _canonical_json(_descriptor(), maximum=_MAX_DESCRIPTOR_BYTES),
            _DESCRIPTOR_BYTES,
        )
    ):
        _fail("external publisher canonical descriptor replay differs")
    _require_atomic_module()
    return current


def _preflight_external_publication_parent(*, publication_parent: Path) -> Path:
    """Open and validate the exact parent before an outcome can be claimed.

    The atomic helper independently reopens and revalidates the path during commit;
    this preflight deliberately does not claim to eliminate filesystem TOCTOU.
    """

    _require_boundary()
    parent = _require_parent(publication_parent)
    open_parent = _ATOMIC_OPEN_PARENT_AT_LOAD
    close = _ATOMIC_CLOSE_AT_LOAD
    opened_type = _ATOMIC_OPEN_DIRECTORY_TYPE_AT_LOAD
    if (
        type(open_parent) is not types.FunctionType
        or type(close) is not types.FunctionType
        or type(opened_type) is not type
    ):
        _fail("captured atomic parent preflight surface is unavailable")
    opened = open_parent(parent)
    exact_opened = cast(Any, opened)
    try:
        if (
            type(opened) is not opened_type
            or exact_opened.path != parent
            or type(exact_opened.descriptor) is not int
            or exact_opened.descriptor < 0
        ):
            _fail("atomic parent preflight returned a non-exact result")
    finally:
        descriptor = getattr(opened, "descriptor", None)
        if type(descriptor) is int and descriptor >= 0:
            close(descriptor)
    return parent


def _require_fresh_reload_boundary() -> None:
    forbidden = (
        PINNED_EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME,
        "_alberta_forager_matched_v3_external_execution_runner_isolated_v1",
        "alberta_framework.benchmarks._forager_matched_v3_external_result_bridge",
        "alberta_framework.benchmarks._forager_matched_v3_scorer",
        "alberta_framework.benchmarks.forager_matched_v3_protocol",
    )
    if any(name in sys.modules for name in forbidden):
        _fail("external publication reload requires a fresh score-content-free process")
    package = sys.modules.get("alberta_framework.benchmarks")
    forbidden_attributes = (
        "forager_matched_v3_external_outcome_consumer",
        "forager_matched_v3_external_execution_runner",
        "_forager_matched_v3_external_result_bridge",
        "_forager_matched_v3_scorer",
        "forager_matched_v3_protocol",
    )
    if type(package) is types.ModuleType and any(
        hasattr(package, name) for name in forbidden_attributes
    ):
        _fail("external publication reload found a lingering score-content module")


def _require_consumer_module() -> tuple[types.ModuleType, types.FunctionType]:
    module = sys.modules.get(PINNED_EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME)
    if type(module) is not types.ModuleType:
        _fail("exact external outcome consumer is unavailable")
    source = _require_sha256(_CONSUMER_SOURCE_SHA256_INPUT, "consumer source")
    descriptor_sha = _require_sha256(
        _CONSUMER_DESCRIPTOR_SHA256_INPUT, "consumer descriptor"
    )
    if not hmac.compare_digest(
        _source_sha256(module.__file__, "forager_matched_v3_external_outcome_consumer.py"),
        source,
    ):
        _fail("external outcome consumer source identity differs")
    if (
        getattr(module, "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256", None)
        != descriptor_sha
    ):
        _fail("external outcome consumer descriptor identity differs")
    descriptor = getattr(module, "canonical_external_outcome_consumer_descriptor_bytes", None)
    consume = getattr(
        module,
        "_consume_matched_v3_external_outcome_to_captured_publication",
        None,
    )
    guard = getattr(module, "_CONSUMER_GUARD_AT_LOAD", None)
    replay_guard = getattr(module, "_replay_external_outcome_consumer_guard", None)
    if (
        type(descriptor) is not types.FunctionType
        or type(consume) is not types.FunctionType
        or type(replay_guard) is not types.FunctionType
    ):
        _fail("external outcome consumer exact API is unavailable")
    raw = descriptor()
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), descriptor_sha
    ):
        _fail("external outcome consumer descriptor bytes differ")
    guarded = replay_guard(guard)
    if (
        type(guarded) is not tuple
        or len(guarded) != 2
        or guarded[0] is not consume
        or type(guarded[1]) is not types.FunctionType
    ):
        _fail("external outcome consumer guarded function replay differs")
    return module, consume


def _exact_role_payloads(value: object) -> tuple[tuple[str, bytes], ...]:
    if type(value) is not tuple:
        _fail("external publication private payloads must be an exact tuple")
    items = cast(tuple[object, ...], value)
    if len(items) != len(_PAYLOAD_FILENAMES):
        _fail("external publication private payload count differs")
    result: list[tuple[str, bytes]] = []
    for index, item in enumerate(items):
        if (
            type(item) is not tuple
            or len(item) != 2
            or item[0] != _PAYLOAD_FILENAMES[index]
            or type(item[1]) is not bytes
        ):
            _fail("external publication private payload order or type differs")
        result.append((item[0], item[1]))
    return tuple(result)


def _atomic_records(
    records: tuple[MatchedV3ExternalPublicationFile, ...],
) -> tuple[Any, ...]:
    record_type = _ATOMIC_RECORD_TYPE_AT_LOAD
    if type(record_type) is not type:
        _fail("captured atomic record type is unavailable")
    return tuple(
        record_type(name=item.name, size_bytes=item.size_bytes, sha256=item.sha256)
        for item in records
    )


def _validated_atomic_result(
    *,
    result: object,
    parent: Path,
    address: str,
    atomic_records: tuple[Any, ...],
    records: tuple[MatchedV3ExternalPublicationFile, ...],
) -> dict[str, bytes]:
    if type(result) is not _ATOMIC_RESULT_TYPE_AT_LOAD:
        _fail("atomic helper returned a non-exact result type")
    exact = cast(Any, result)
    expected_atomic = tuple(sorted(atomic_records, key=lambda item: os.fsencode(item.name)))
    if (
        type(exact.root) is not _PATH_TYPE
        or exact.root != parent / address
        or exact.address != address
        or exact.records != expected_atomic
        or type(exact.files) is not types.MappingProxyType
        or tuple(exact.files) != tuple(item.name for item in expected_atomic)
    ):
        _fail("atomic helper result identity or normalized inventory differs")
    loaded: dict[str, bytes] = {}
    for record in records:
        raw = exact.files.get(record.name)
        if (
            type(raw) is not bytes
            or len(raw) != record.size_bytes
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record.sha256)
        ):
            _fail(f"atomic helper bytes differ for {record.name}")
        loaded[record.name] = raw
    return loaded


def _metadata_from_loaded(
    *,
    operation: Literal["published", "reloaded"],
    parent: Path,
    records: tuple[MatchedV3ExternalPublicationFile, ...],
    loaded: dict[str, bytes],
    facts: _ExternalPublicationFacts,
) -> MatchedV3ExternalPublicationMetadata:
    payload_records = records[1:]
    outcome_content = {name: loaded[name] for name in _PAYLOAD_FILENAMES[1:]}
    _outcome, outcome_body = _parse_outcome_manifest(
        loaded["external-outcome-manifest.json"],
        facts=facts,
        content_payloads=outcome_content,
    )
    publication_body = _parse_publication_manifest(
        loaded["publication.json"],
        facts=facts,
        payload_records=payload_records,
        outcome_body_sha256=outcome_body,
    )
    if not hmac.compare_digest(
        hashlib.sha256(loaded["external-execution-receipt.json"]).hexdigest(),
        facts.execution_receipt_sha256,
    ) or not hmac.compare_digest(
        hashlib.sha256(loaded["external-conversion-receipt.json"]).hexdigest(),
        facts.conversion_receipt_sha256,
    ):
        _fail("external publication receipt digest binding differs")
    mode = _video_slot_mode(
        candidate_id=facts.candidate_id,
        raw=loaded["upstream-video-slot.bin"],
    )
    if mode != facts.video_slot_mode:
        _fail("external publication video slot replay differs")
    return _make_metadata(
        operation=operation,
        root=parent / records[0].sha256,
        records=records,
        facts=facts,
        publication_body_sha256=publication_body,
        outcome_body_sha256=outcome_body,
    )


def _publish_consumed_external_outcome_payload(
    *,
    publication_parent: Path,
    role_payloads: tuple[tuple[str, bytes], ...],
    facts: _ExternalPublicationFacts,
) -> MatchedV3ExternalPublicationMetadata:
    """Captured private sink; it is not a public byte-ingestion interface."""

    _require_boundary()
    facts = _validate_facts(facts)
    parent = _require_parent(publication_parent)
    payload_items = _exact_role_payloads(role_payloads)
    payloads = dict(payload_items)
    outcome_content = {name: payloads[name] for name in _PAYLOAD_FILENAMES[1:]}
    _outcome, outcome_body_sha256 = _parse_outcome_manifest(
        payloads["external-outcome-manifest.json"],
        facts=facts,
        content_payloads=outcome_content,
    )
    payload_records = _payload_records(payloads)
    publication_raw, _body_sha = _publication_manifest(
        facts,
        payload_records,
        outcome_body_sha256=outcome_body_sha256,
    )
    all_payloads = {"publication.json": publication_raw, **payloads}
    records = tuple(
        MatchedV3ExternalPublicationFile(
            role=role,
            name=name,
            size_bytes=len(all_payloads[name]),
            sha256=hashlib.sha256(all_payloads[name]).hexdigest(),
        )
        for role, name in EXTERNAL_PUBLICATION_ROLE_PATHS
    )
    total = sum(record.size_bytes for record in records)
    if total > facts.maximum_publication_total_bytes:
        _fail("external publication exceeds its caller-declared aggregate ceiling")
    address = records[0].sha256
    atomic_records = _atomic_records(records)
    publish = _ATOMIC_PUBLISH_AT_LOAD
    if type(publish) is not types.FunctionType:
        _fail("captured atomic publish function is unavailable")
    # Exactly one call. Collision and uncertain errors intentionally pass through.
    raw_result = publish(
        parent,
        address=address,
        expected_files=atomic_records,
        payloads=all_payloads,
    )
    try:
        loaded = _validated_atomic_result(
            result=raw_result,
            parent=parent,
            address=address,
            atomic_records=atomic_records,
            records=records,
        )
        metadata = _metadata_from_loaded(
            operation="published",
            parent=parent,
            records=records,
            loaded=loaded,
            facts=facts,
        )
    except BaseException as exc:
        uncertain_type = _ATOMIC_UNCERTAIN_TYPE_AT_LOAD
        if type(uncertain_type) is type:
            raise uncertain_type(
                parent / address,
                address,
                "publisher validation failed after atomic publication returned",
                committed=True,
            ) from exc
        raise
    del loaded
    del raw_result
    return _validate_metadata(metadata)


def publish_matched_v3_external_outcome_capability(
    *,
    outcome_capability: object,
    publication_parent: Path,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_environment_seed_commitment_sha256: str,
    expected_agent_seed_commitment_sha256: str,
    expected_qualification_plan_sha256: str,
    expected_qualification_case_manifest_sha256: str,
    expected_publisher_source_tree_sha256: str,
    expected_workload_source_tree_sha256: str,
    expected_staging_manifest_sha256: str,
    maximum_publication_total_bytes: int,
    explicit_publication_opt_in: bool,
) -> MatchedV3ExternalPublicationMetadata:
    """Publish only through the exact consumer and a still-live outcome capability."""

    _require_boundary()
    if explicit_publication_opt_in is not True:
        _fail("external outcome publication requires exact explicit opt-in")
    _module, consume = _require_consumer_module()
    metadata = consume(
        outcome_capability=outcome_capability,
        publication_parent=_require_parent(publication_parent),
        expected_candidate_id=_require_candidate(expected_candidate_id),
        expected_environment_seed=_require_uint31(
            expected_environment_seed, "expected environment seed"
        ),
        expected_agent_seed=_require_uint31(expected_agent_seed, "expected agent seed"),
        expected_environment_seed_commitment_sha256=_require_sha256(
            expected_environment_seed_commitment_sha256,
            "expected environment seed commitment",
        ),
        expected_agent_seed_commitment_sha256=_require_sha256(
            expected_agent_seed_commitment_sha256,
            "expected agent seed commitment",
        ),
        expected_qualification_plan_sha256=_require_sha256(
            expected_qualification_plan_sha256, "expected qualification plan"
        ),
        expected_qualification_case_manifest_sha256=_require_sha256(
            expected_qualification_case_manifest_sha256,
            "expected qualification case manifest",
        ),
        expected_publisher_source_tree_sha256=_require_sha256(
            expected_publisher_source_tree_sha256,
            "expected publisher source tree",
        ),
        expected_workload_source_tree_sha256=_require_sha256(
            expected_workload_source_tree_sha256,
            "expected workload source tree",
        ),
        expected_staging_manifest_sha256=_require_sha256(
            expected_staging_manifest_sha256, "expected staging manifest"
        ),
        maximum_publication_total_bytes=maximum_publication_total_bytes,
        explicit_publication_opt_in=True,
    )
    return _validate_metadata(metadata)


def _facts_from_loaded_outcome(
    outcome: dict[str, Any],
    *,
    expected_candidate_id: str,
    expected_qualification_plan_sha256: str,
    expected_qualification_case_manifest_sha256: str,
    expected_publisher_source_tree_sha256: str,
    expected_workload_source_tree_sha256: str,
    expected_staging_manifest_sha256: str,
    expected_environment_seed_commitment_sha256: str,
    expected_agent_seed_commitment_sha256: str,
    expected_consumer_descriptor_sha256: str,
    expected_consumer_source_sha256: str,
    expected_runner_descriptor_sha256: str,
    expected_runner_source_sha256: str,
    expected_bridge_descriptor_sha256: str,
    expected_bridge_source_sha256: str,
    expected_scorer_source_sha256: str,
    expected_protocol_source_sha256: str,
    expected_metric_descriptor_sha256: str,
    expected_execution_contract_descriptor_sha256: str,
    expected_staging_descriptor_sha256: str,
    expected_seed_transport_descriptor_sha256: str,
    expected_execution_receipt_sha256: str,
    expected_conversion_receipt_sha256: str,
    expected_production_runner_exact: bool,
    expected_maximum_publication_total_bytes: int,
) -> _ExternalPublicationFacts:
    if type(expected_production_runner_exact) is not bool:
        _fail("expected production-runner binding must be one exact bool")
    if (
        type(expected_maximum_publication_total_bytes) is not int
        or not 1
        <= expected_maximum_publication_total_bytes
        <= MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES
    ):
        _fail("expected external publication aggregate ceiling is invalid")
    candidate = cast(dict[str, Any], outcome["candidate"])
    qualification = cast(dict[str, Any], outcome["qualification"])
    sources = cast(dict[str, Any], outcome["sources"])
    policy = cast(dict[str, Any], outcome["publication_policy"])
    pipeline = cast(dict[str, Any], outcome["pipeline_binding"])
    if (
        candidate["candidate_id"] != expected_candidate_id
        or qualification["plan_sha256"] != expected_qualification_plan_sha256
        or qualification["case_manifest_sha256"]
        != expected_qualification_case_manifest_sha256
        or sources["publisher_source_tree_sha256"]
        != expected_publisher_source_tree_sha256
        or sources["workload_source_tree_sha256"]
        != expected_workload_source_tree_sha256
        or sources["staging_manifest_sha256"] != expected_staging_manifest_sha256
        or qualification["environment_seed_commitment_sha256"]
        != expected_environment_seed_commitment_sha256
        or qualification["agent_seed_commitment_sha256"]
        != expected_agent_seed_commitment_sha256
        or candidate["production_runner_exact"] is not expected_production_runner_exact
        or policy["maximum_total_bytes"] != expected_maximum_publication_total_bytes
        or outcome["execution_receipt_sha256"] != expected_execution_receipt_sha256
        or outcome["conversion_receipt_sha256"] != expected_conversion_receipt_sha256
    ):
        _fail("external publication reload caller-carried identities differ")
    atomic = cast(dict[str, Any], pipeline["atomic"])
    publisher = cast(dict[str, Any], pipeline["publisher"])
    consumer = cast(dict[str, Any], pipeline["consumer"])
    runner = cast(dict[str, Any], pipeline["runner"])
    bridge = cast(dict[str, Any], pipeline["bridge"])
    if (
        atomic["descriptor_sha256"] != PINNED_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
        or atomic["source_sha256"] != PINNED_ATOMIC_PUBLICATION_SOURCE_SHA256
        or publisher["descriptor_sha256"]
        != EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or publisher["source_sha256"]
        != _require_sha256(_PUBLICATION_SOURCE_SHA256_INPUT, "publisher source")
        or consumer["descriptor_sha256"] != expected_consumer_descriptor_sha256
        or consumer["source_sha256"] != expected_consumer_source_sha256
        or runner["descriptor_sha256"] != expected_runner_descriptor_sha256
        or runner["source_sha256"] != expected_runner_source_sha256
        or bridge["descriptor_sha256"] != expected_bridge_descriptor_sha256
        or bridge["source_sha256"] != expected_bridge_source_sha256
        or pipeline["scorer_source_sha256"] != expected_scorer_source_sha256
        or pipeline["protocol_source_sha256"] != expected_protocol_source_sha256
        or pipeline["metric_descriptor_sha256"] != expected_metric_descriptor_sha256
        or pipeline["execution_contract_descriptor_sha256"]
        != expected_execution_contract_descriptor_sha256
        or pipeline["staging_descriptor_sha256"]
        != expected_staging_descriptor_sha256
        or pipeline["seed_transport_descriptor_sha256"]
        != expected_seed_transport_descriptor_sha256
    ):
        _fail("external publication reload pipeline binding differs")
    return _validate_facts(
        _ExternalPublicationFacts(
            candidate_id=cast(str, candidate["candidate_id"]),
            external_candidate_ordinal=cast(int, candidate["external_candidate_ordinal"]),
            family=cast(Literal["continuing", "ppo"], candidate["family"]),
            qualification_plan_sha256=cast(str, qualification["plan_sha256"]),
            qualification_case_manifest_sha256=cast(
                str, qualification["case_manifest_sha256"]
            ),
            publisher_source_tree_sha256=cast(
                str, sources["publisher_source_tree_sha256"]
            ),
            workload_source_tree_sha256=cast(
                str, sources["workload_source_tree_sha256"]
            ),
            staging_manifest_sha256=cast(str, sources["staging_manifest_sha256"]),
            environment_seed_commitment_sha256=cast(
                str, qualification["environment_seed_commitment_sha256"]
            ),
            agent_seed_commitment_sha256=cast(
                str, qualification["agent_seed_commitment_sha256"]
            ),
            runner_descriptor_sha256=cast(str, runner["descriptor_sha256"]),
            runner_source_sha256=cast(str, runner["source_sha256"]),
            consumer_descriptor_sha256=cast(str, consumer["descriptor_sha256"]),
            consumer_source_sha256=cast(str, consumer["source_sha256"]),
            bridge_descriptor_sha256=cast(str, bridge["descriptor_sha256"]),
            bridge_source_sha256=cast(str, bridge["source_sha256"]),
            scorer_source_sha256=cast(str, pipeline["scorer_source_sha256"]),
            protocol_source_sha256=cast(str, pipeline["protocol_source_sha256"]),
            metric_descriptor_sha256=cast(str, pipeline["metric_descriptor_sha256"]),
            execution_contract_descriptor_sha256=cast(
                str, pipeline["execution_contract_descriptor_sha256"]
            ),
            staging_descriptor_sha256=cast(str, pipeline["staging_descriptor_sha256"]),
            seed_transport_descriptor_sha256=cast(
                str, pipeline["seed_transport_descriptor_sha256"]
            ),
            execution_receipt_sha256=cast(str, outcome["execution_receipt_sha256"]),
            conversion_receipt_sha256=cast(str, outcome["conversion_receipt_sha256"]),
            production_runner_exact=candidate["production_runner_exact"],
            video_slot_mode=cast(Any, candidate["video_slot_mode"]),
            maximum_publication_total_bytes=cast(int, policy["maximum_total_bytes"]),
        )
    )


def load_matched_v3_external_reward_publication(
    *,
    publication_parent: Path,
    expected_address: str,
    expected_file_records: tuple[MatchedV3ExternalPublicationFile, ...],
    expected_candidate_id: str,
    expected_qualification_plan_sha256: str,
    expected_qualification_case_manifest_sha256: str,
    expected_publisher_source_tree_sha256: str,
    expected_workload_source_tree_sha256: str,
    expected_staging_manifest_sha256: str,
    expected_environment_seed_commitment_sha256: str,
    expected_agent_seed_commitment_sha256: str,
    expected_consumer_descriptor_sha256: str,
    expected_consumer_source_sha256: str,
    expected_runner_descriptor_sha256: str,
    expected_runner_source_sha256: str,
    expected_bridge_descriptor_sha256: str,
    expected_bridge_source_sha256: str,
    expected_scorer_source_sha256: str,
    expected_protocol_source_sha256: str,
    expected_metric_descriptor_sha256: str,
    expected_execution_contract_descriptor_sha256: str,
    expected_staging_descriptor_sha256: str,
    expected_seed_transport_descriptor_sha256: str,
    expected_production_runner_exact: bool,
    expected_maximum_publication_total_bytes: int,
) -> MatchedV3ExternalPublicationMetadata:
    """Reload caller-addressed exact content without loading runner/consumer/bridge."""

    _require_boundary()
    _require_fresh_reload_boundary()
    parent = _require_parent(publication_parent)
    address = _require_sha256(expected_address, "expected publication address")
    records = _validate_records(expected_file_records)
    if not hmac.compare_digest(records[0].sha256, address):
        _fail("expected publication address differs from publication.json record")
    atomic_records = _atomic_records(records)
    load = _ATOMIC_LOAD_AT_LOAD
    if type(load) is not types.FunctionType:
        _fail("captured atomic load function is unavailable")
    raw_result = load(parent, address=address, expected_files=atomic_records)
    loaded = _validated_atomic_result(
        result=raw_result,
        parent=parent,
        address=address,
        atomic_records=atomic_records,
        records=records,
    )
    outcome = _strict_json(
        loaded["external-outcome-manifest.json"], maximum=_MAX_MANIFEST_BYTES
    )
    try:
        facts = _facts_from_loaded_outcome(
            outcome,
            expected_candidate_id=_require_candidate(expected_candidate_id),
            expected_qualification_plan_sha256=_require_sha256(
                expected_qualification_plan_sha256, "expected qualification plan"
            ),
            expected_qualification_case_manifest_sha256=_require_sha256(
                expected_qualification_case_manifest_sha256,
                "expected qualification case",
            ),
            expected_publisher_source_tree_sha256=_require_sha256(
                expected_publisher_source_tree_sha256,
                "expected publisher source tree",
            ),
            expected_workload_source_tree_sha256=_require_sha256(
                expected_workload_source_tree_sha256,
                "expected workload source tree",
            ),
            expected_staging_manifest_sha256=_require_sha256(
                expected_staging_manifest_sha256, "expected staging manifest"
            ),
            expected_environment_seed_commitment_sha256=_require_sha256(
                expected_environment_seed_commitment_sha256,
                "expected environment seed commitment",
            ),
            expected_agent_seed_commitment_sha256=_require_sha256(
                expected_agent_seed_commitment_sha256,
                "expected agent seed commitment",
            ),
            expected_consumer_descriptor_sha256=_require_sha256(
                expected_consumer_descriptor_sha256,
                "expected consumer descriptor",
            ),
            expected_consumer_source_sha256=_require_sha256(
                expected_consumer_source_sha256, "expected consumer source"
            ),
            expected_runner_descriptor_sha256=_require_sha256(
                expected_runner_descriptor_sha256, "expected runner descriptor"
            ),
            expected_runner_source_sha256=_require_sha256(
                expected_runner_source_sha256, "expected runner source"
            ),
            expected_bridge_descriptor_sha256=_require_sha256(
                expected_bridge_descriptor_sha256, "expected bridge descriptor"
            ),
            expected_bridge_source_sha256=_require_sha256(
                expected_bridge_source_sha256, "expected bridge source"
            ),
            expected_scorer_source_sha256=_require_sha256(
                expected_scorer_source_sha256, "expected scorer source"
            ),
            expected_protocol_source_sha256=_require_sha256(
                expected_protocol_source_sha256, "expected protocol source"
            ),
            expected_metric_descriptor_sha256=_require_sha256(
                expected_metric_descriptor_sha256, "expected metric descriptor"
            ),
            expected_execution_contract_descriptor_sha256=_require_sha256(
                expected_execution_contract_descriptor_sha256,
                "expected execution contract descriptor",
            ),
            expected_staging_descriptor_sha256=_require_sha256(
                expected_staging_descriptor_sha256,
                "expected staging descriptor",
            ),
            expected_seed_transport_descriptor_sha256=_require_sha256(
                expected_seed_transport_descriptor_sha256,
                "expected seed transport descriptor",
            ),
            expected_execution_receipt_sha256=records[2].sha256,
            expected_conversion_receipt_sha256=records[3].sha256,
            expected_production_runner_exact=expected_production_runner_exact,
            expected_maximum_publication_total_bytes=(
                expected_maximum_publication_total_bytes
            ),
        )
    except ForagerMatchedV3ExternalRewardPublicationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalRewardPublicationError(
            "external publication outcome manifest fields differ"
        ) from exc
    metadata = _metadata_from_loaded(
        operation="reloaded",
        parent=parent,
        records=records,
        loaded=loaded,
        facts=facts,
    )
    del loaded
    del raw_result
    return _validate_metadata(metadata)


def canonical_external_publication_metadata_bytes(
    metadata: MatchedV3ExternalPublicationMetadata,
) -> bytes:
    """Return canonical metadata-only host handoff bytes."""

    exact = _validate_metadata(metadata)
    return _canonical_json(
        {**_metadata_body(exact), "metadata_body_sha256": exact.metadata_body_sha256},
        maximum=_MAX_METADATA_BYTES,
    )


def parse_external_publication_metadata(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> MatchedV3ExternalPublicationMetadata:
    """Parse one metadata handoff under a caller-carried full-file digest."""

    expected = _require_sha256(expected_file_sha256, "expected metadata file")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected
    ):
        _fail("external publication metadata full-file digest differs")
    value = _strict_json(raw, maximum=_MAX_METADATA_BYTES)
    files_value = value.pop("files", None)
    metadata_digest = value.pop("metadata_body_sha256", None)
    publication_root = value.pop("publication_root", None)
    if type(files_value) is not list or type(publication_root) is not str:
        _fail("external publication metadata files are invalid")
    try:
        files = tuple(
            MatchedV3ExternalPublicationFile(
                role=item["role"],
                name=item["name"],
                size_bytes=item["size_bytes"],
                sha256=item["sha256"],
            )
            for item in files_value
            if type(item) is dict
        )
        if len(files) != len(files_value):
            _fail("external publication metadata file record differs")
        metadata = MatchedV3ExternalPublicationMetadata(
            **value,
            publication_root=Path(publication_root),
            files=files,
            metadata_body_sha256=metadata_digest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalRewardPublicationError(
            "external publication metadata fields differ"
        ) from exc
    if not hmac.compare_digest(canonical_external_publication_metadata_bytes(metadata), raw):
        _fail("external publication metadata semantic replay differs")
    return metadata


def external_reward_publication_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing descriptor content."""

    return _strict_json(_DESCRIPTOR_BYTES, maximum=_MAX_DESCRIPTOR_BYTES)


def canonical_external_reward_publication_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def parse_external_reward_publication_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact self-pinned descriptor."""

    value = _strict_json(raw, maximum=_MAX_DESCRIPTOR_BYTES)
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    ):
        _fail("external reward publication descriptor identity differs")
    return value


__all__ = [
    "EXTERNAL_PUBLICATION_CANDIDATE_IDS",
    "EXTERNAL_PUBLICATION_FILENAMES",
    "EXTERNAL_PUBLICATION_ROLE_PATHS",
    "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "EXTERNAL_REWARD_PUBLICATION_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_REWARD_PUBLICATION_METADATA_SCHEMA_VERSION",
    "EXTERNAL_REWARD_PUBLICATION_STATUS",
    "ForagerMatchedV3ExternalRewardPublicationError",
    "MATCHED_V3_INTERACTION_HORIZON",
    "MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES",
    "MatchedV3ExternalPublicationFile",
    "MatchedV3ExternalPublicationMetadata",
    "canonical_external_publication_metadata_bytes",
    "canonical_external_reward_publication_descriptor_bytes",
    "external_reward_publication_descriptor",
    "load_matched_v3_external_reward_publication",
    "parse_external_publication_metadata",
    "parse_external_reward_publication_descriptor",
    "publish_matched_v3_external_outcome_capability",
]
