"""Post-claim matched-v3 external outcome conversion and atomic-publication handoff.

This source has no static project import.  A qualification worker direct-loads the
exact atomic helper, external publisher, this consumer, and the external runner in
that order.  The publisher calls the private orchestrator with a live PID-bound
runner outcome.  The runner irrevocably claims that outcome before invoking the
private sink captured from this module.  Before that claim, the captured publisher
uses the atomic helper's exact safe-parent open as a preflight; the atomic commit
still reopens and revalidates the parent, because the preflight cannot remove TOCTOU.
Only after claim does this module direct-load the exact protocol, scorer, and result
bridge sources, convert the opaque upstream NPZ, and pass one fixed role/payload
inventory to the publisher sink captured at load.

The public surface is descriptor-only.  No public completion, conversion, reward
bytes, callback, or sink is accepted.  The returned object is the publisher's frozen
digest metadata, never bridge conversion content.  This path is permanently
nonqualifying and nonauthorizing.  Same-process Python is not a hostile-code boundary;
fresh-process and host cgroup/container proofs remain external duties.
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
from pathlib import Path
from typing import Any, Final, NoReturn, cast

EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_outcome_consumer_descriptor.v1"
)
EXTERNAL_OUTCOME_CONSUMER_STATUS: Final = (
    "implemented_post_claim_score_bearing_nonqualifying_non_authorizing"
)
EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_outcome_consumer_isolated_v1"
)

PINNED_EXTERNAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_reward_publication_isolated_v1"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_reward_publication.py"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256: Final = (
    "645d232134b220f57b466d3f9c3e140ace8bad3835d9ed290fc066a3c257a80c"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "59d470d6c31e1d3dce8eded401e6331994ca007b94524d8e00714c1f2c66f30b"
)

PINNED_EXTERNAL_EXECUTION_RUNNER_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_execution_runner_isolated_v1"
)
PINNED_EXTERNAL_EXECUTION_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_execution_runner.py"
)
PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_runner_descriptor.v1"
)

PINNED_RESULT_BRIDGE_MODULE_NAME: Final = (
    "alberta_framework.benchmarks._forager_matched_v3_external_result_bridge"
)
PINNED_SCORER_MODULE_NAME: Final = (
    "alberta_framework.benchmarks._forager_matched_v3_scorer"
)
PINNED_PROTOCOL_MODULE_NAME: Final = (
    "alberta_framework.benchmarks.forager_matched_v3_protocol"
)
PINNED_RESULT_BRIDGE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_external_result_bridge.py"
)
PINNED_SCORER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
)
PINNED_PROTOCOL_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_protocol.py"
)
PINNED_RESULT_BRIDGE_SOURCE_SHA256: Final = (
    "c1859f0cfb7862e22c470f89ad9d3298a76b1fb419bf1431069f286f593e22f7"
)
PINNED_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_result_bridge_descriptor.v1"
)
PINNED_RESULT_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
)
PINNED_SCORER_SOURCE_SHA256: Final = (
    "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
)
PINNED_PROTOCOL_SOURCE_SHA256: Final = (
    "dd5db9a657ad167abf192942489642130b08bd065f724f7ad1b80743b1103720"
)
PINNED_METRIC_DESCRIPTOR_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
PINNED_EXECUTION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
)
PINNED_STAGING_DESCRIPTOR_SHA256: Final = (
    "ceea86b38822f3add0465788003d349dd221a49fba5f3fa069bfec985537caea"
)
PINNED_SEED_TRANSPORT_DESCRIPTOR_SHA256: Final = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)

EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS: Final = (
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
_CONTENT_ROLE_PATHS: Final = (
    ("execution_receipt", "external-execution-receipt.json"),
    ("conversion_receipt", "external-conversion-receipt.json"),
    ("upstream_reward_npz", "upstream-reward.npz"),
    ("upstream_results_database", "upstream-results.db"),
    ("upstream_video_slot", "upstream-video-slot.bin"),
    ("canonical_reward_npz", "reward-trace.npz"),
    ("stdout", "stdout.bin"),
    ("stderr", "stderr.bin"),
)

MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES: Final = 1024 * 1024 * 1024
_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_TEXT_BYTES: Final = 4 * 1024 * 1024
_UINT31_MAX: Final = 2**31 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_TYPE: Final = type(Path())

_CONSUMER_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256"
)
_RUNNER_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_EXECUTION_RUNNER_SOURCE_SHA256"
)
_RUNNER_DESCRIPTOR_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")
_SELF_MODULE_AT_LOAD: Final = (
    sys.modules.get(_MODULE_NAME_INPUT) if type(_MODULE_NAME_INPUT) is str else None
)
_ISOLATED_CONSUMER_BOUNDARY: Final = (
    _MODULE_NAME_INPUT == EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME
    and (_MODULE_PACKAGE_INPUT is None or _MODULE_PACKAGE_INPUT == "")
    and type(_SELF_MODULE_AT_LOAD) is types.ModuleType
    and _SELF_MODULE_AT_LOAD.__dict__ is globals()
)

_PUBLISHER_MODULE_AT_LOAD: Final = sys.modules.get(
    PINNED_EXTERNAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME
)
_PUBLISHER_SINK_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD, "_publish_consumed_external_outcome_payload", None
)
_PUBLISHER_MANIFEST_BUILDER_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD, "_build_external_outcome_manifest", None
)
_PUBLISHER_PARENT_PREFLIGHT_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD, "_preflight_external_publication_parent", None
)
_PUBLISHER_FACTS_TYPE_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD, "_ExternalPublicationFacts", None
)
_PUBLISHER_METADATA_TYPE_AT_LOAD: Final = getattr(
    _PUBLISHER_MODULE_AT_LOAD, "MatchedV3ExternalPublicationMetadata", None
)
_BENCHMARKS_PACKAGE_AT_LOAD: Final = sys.modules.get("alberta_framework.benchmarks")


class ForagerMatchedV3ExternalOutcomeConsumerError(RuntimeError):
    """The exact post-claim conversion/publication closure failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3ExternalOutcomeConsumerError(message)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"external outcome consumer JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"external outcome consumer JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("external outcome consumer JSON integer exceeds its digit bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("external outcome consumer JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _assert_plain_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("external outcome consumer JSON exceeds its complexity bound")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_TEXT_BYTES:
                _fail("external outcome consumer JSON text exceeds its byte bound")
            return
        if type(item) not in {dict, list}:
            _fail("external outcome consumer JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("external outcome consumer JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _fail("external outcome consumer JSON keys must be exact strings")
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
        raise ForagerMatchedV3ExternalOutcomeConsumerError(
            "external outcome consumer value is not canonical JSON"
        ) from exc
    if not 1 <= len(raw) <= maximum:
        _fail("external outcome consumer canonical JSON exceeds its byte bound")
    return raw


def _strict_json(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        _fail("external outcome consumer JSON bytes exceed their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3ExternalOutcomeConsumerError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalOutcomeConsumerError(
            "external outcome consumer content is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("external outcome consumer JSON root must be an object")
    _assert_plain_json(value)
    if not hmac.compare_digest(_canonical_json(value, maximum=maximum), raw):
        _fail("external outcome consumer JSON is not canonical")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_uint31(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        _fail(f"{label} must be one exact uint31")
    return value


def _require_candidate(value: object) -> str:
    if type(value) is not str or value not in EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS:
        _fail("candidate ID is not one exact external candidate")
    return value


def _require_parent(value: object) -> Path:
    if (
        type(value) is not _PATH_TYPE
        or not value.is_absolute()
        or value == Path("/")
    ):
        _fail("external outcome publication parent must be an absolute non-root Path")
    return value


def _require_maximum(value: object) -> int:
    if (
        type(value) is not int
        or not 1 <= value <= MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES
    ):
        _fail("external outcome publication aggregate ceiling is invalid")
    return value


def _claims() -> dict[str, bool]:
    return {
        "campaign_ingestion_authorized": False,
        "candidate_qualified": False,
        "evidence_authority": False,
        "execution_authorized": False,
        "performance_claim_allowed": False,
        "publication_authority": False,
        "qualification_authority": False,
        "result_accepted": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION,
        "status": EXTERNAL_OUTCOME_CONSUMER_STATUS,
        "classification": (
            "post_claim_score_reward_bearing_conversion_to_captured_atomic_publisher"
        ),
        "candidate_count": len(EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS),
        "candidate_order": list(EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS),
        "load_order": {
            "atomic_before_publisher": True,
            "publisher_before_consumer": True,
            "consumer_before_runner": True,
            "bridge_scorer_protocol_absent_before_runner_claim": True,
            "protocol_then_scorer_then_bridge_after_runner_claim": True,
            "preexisting_parent_package_required": True,
            "parent_package_initializer_executed_post_claim": False,
        },
        "outcome_path": {
            "live_pid_bound_capability_required": True,
            "public_completion_accepted": False,
            "raw_bytes_accepted_by_orchestrator": False,
            "callback_or_sink_accepted": False,
            "safe_parent_preflight_precedes_runner_claim": True,
            "runner_claim_precedes_conversion": True,
            "claim_failure_retry": False,
            "public_completion_and_publication_paths_mutually_exclusive": True,
        },
        "publication": {
            "returns_metadata_only": True,
            "exact_file_count": 10,
            "exact_content_role_paths": [list(item) for item in _CONTENT_ROLE_PATHS],
            "aggregate_ceiling_bytes": MAX_EXTERNAL_PUBLICATION_TOTAL_BYTES,
        },
        "bindings": {
            "publisher": {
                "descriptor_schema_version": (
                    PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
                ),
                "descriptor_sha256": (
                    PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
                ),
                "source_sha256": PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256,
            },
            "runner": {
                "descriptor_schema_version": (
                    PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION
                ),
                "source_and_descriptor_caller_injected": True,
            },
            "bridge": {
                "descriptor_schema_version": (
                    PINNED_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION
                ),
                "descriptor_sha256": PINNED_RESULT_BRIDGE_DESCRIPTOR_SHA256,
                "source_sha256": PINNED_RESULT_BRIDGE_SOURCE_SHA256,
            },
            "scorer_source_sha256": PINNED_SCORER_SOURCE_SHA256,
            "protocol_source_sha256": PINNED_PROTOCOL_SOURCE_SHA256,
            "metric_descriptor_sha256": PINNED_METRIC_DESCRIPTOR_SHA256,
            "execution_contract_descriptor_sha256": (
                PINNED_EXECUTION_CONTRACT_DESCRIPTOR_SHA256
            ),
            "staging_descriptor_sha256": PINNED_STAGING_DESCRIPTOR_SHA256,
            "seed_transport_descriptor_sha256": (
                PINNED_SEED_TRANSPORT_DESCRIPTOR_SHA256
            ),
        },
        "limitations": [
            "This consumer handles score- and reward-bearing content after claim.",
            "Same-process Python is not a hostile-code isolation boundary.",
            "Same-UID filesystem confidentiality is not claimed.",
            "Digests and sizes may reveal information about persisted content.",
            "The pre-claim parent preflight does not eliminate filesystem TOCTOU.",
            "Atomic publication reopens and revalidates the parent at commit time.",
            "Fresh-process and host cgroup/container proofs remain external duties.",
            "The host must authenticate the parent package loaded before this consumer.",
        ],
        "claims": _claims(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor(), maximum=_MAX_DESCRIPTOR_BYTES)
EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256: Final = (
    "7c7d007f29b55d6e4a72467d72c4b793568847930d7eb0c17cc276b027e74ceb"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256,
):
    raise AssertionError("external outcome consumer descriptor identity drifted")


def _source_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_exact_source(path: str, *, expected_suffix: str, expected_sha256: str) -> bytes:
    if (
        type(path) is not str
        or not os.path.isabs(path)
        or not path.endswith(expected_suffix)
    ):
        _fail(f"exact source path differs from {expected_suffix}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedV3ExternalOutcomeConsumerError(
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
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"source truncated while reading: {expected_suffix}")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"source grew while reading: {expected_suffix}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _source_identity(before) != _source_identity(after):
        _fail(f"source changed while reading: {expected_suffix}")
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
        _fail(f"source digest differs: {expected_suffix}")
    return b"".join(chunks)


def _require_boundary() -> str:
    if not _ISOLATED_CONSUMER_BOUNDARY:
        _fail("external outcome consumer requires its isolated direct-load boundary")
    expected = _require_sha256(_CONSUMER_SOURCE_SHA256_INPUT, "consumer source")
    path = globals().get("__file__")
    if type(path) is not str:
        _fail("external outcome consumer source path is unavailable")
    _read_exact_source(
        path,
        expected_suffix="forager_matched_v3_external_outcome_consumer.py",
        expected_sha256=expected,
    )
    return expected


def _require_publisher_module() -> types.ModuleType:
    module = _PUBLISHER_MODULE_AT_LOAD
    if type(module) is not types.ModuleType:
        _fail("exact external publisher was not loaded before the consumer")
    if sys.modules.get(PINNED_EXTERNAL_REWARD_PUBLICATION_ISOLATED_MODULE_NAME) is not module:
        _fail("exact external publisher module identity changed")
    path = getattr(module, "__file__", None)
    if type(path) is not str:
        _fail("exact external publisher source path is unavailable")
    _read_exact_source(
        path,
        expected_suffix=PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_PATH,
        expected_sha256=PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256,
    )
    descriptor = getattr(
        module, "canonical_external_reward_publication_descriptor_bytes", None
    )
    if (
        type(descriptor) is not types.FunctionType
        or getattr(module, "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256", None)
        != PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    ):
        _fail("exact external publisher descriptor API differs")
    raw = descriptor()
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    ):
        _fail("exact external publisher descriptor bytes differ")
    for current, captured, label in (
        (
            getattr(module, "_publish_consumed_external_outcome_payload", None),
            _PUBLISHER_SINK_AT_LOAD,
            "sink",
        ),
        (
            getattr(module, "_build_external_outcome_manifest", None),
            _PUBLISHER_MANIFEST_BUILDER_AT_LOAD,
            "manifest builder",
        ),
        (
            getattr(module, "_preflight_external_publication_parent", None),
            _PUBLISHER_PARENT_PREFLIGHT_AT_LOAD,
            "parent preflight",
        ),
        (
            getattr(module, "_ExternalPublicationFacts", None),
            _PUBLISHER_FACTS_TYPE_AT_LOAD,
            "facts type",
        ),
        (
            getattr(module, "MatchedV3ExternalPublicationMetadata", None),
            _PUBLISHER_METADATA_TYPE_AT_LOAD,
            "metadata type",
        ),
    ):
        if current is not captured:
            _fail(f"exact external publisher captured {label} changed")
    if (
        type(_PUBLISHER_SINK_AT_LOAD) is not types.FunctionType
        or type(_PUBLISHER_MANIFEST_BUILDER_AT_LOAD) is not types.FunctionType
        or type(_PUBLISHER_PARENT_PREFLIGHT_AT_LOAD) is not types.FunctionType
        or type(_PUBLISHER_FACTS_TYPE_AT_LOAD) is not type
        or type(_PUBLISHER_METADATA_TYPE_AT_LOAD) is not type
    ):
        _fail("exact external publisher private surface is unavailable")
    return module


def _score_modules_absent() -> None:
    for name in (
        PINNED_PROTOCOL_MODULE_NAME,
        PINNED_SCORER_MODULE_NAME,
        PINNED_RESULT_BRIDGE_MODULE_NAME,
    ):
        if name in sys.modules:
            _fail("score-bearing module was loaded before runner outcome claim")
    package = sys.modules.get("alberta_framework.benchmarks")
    if type(package) is types.ModuleType and any(
        hasattr(package, name)
        for name in (
            "forager_matched_v3_protocol",
            "_forager_matched_v3_scorer",
            "_forager_matched_v3_external_result_bridge",
        )
    ):
        _fail("score-bearing package alias remained before runner outcome claim")


def _require_runner_module() -> tuple[types.ModuleType, types.FunctionType]:
    _score_modules_absent()
    module = sys.modules.get(PINNED_EXTERNAL_EXECUTION_RUNNER_ISOLATED_MODULE_NAME)
    if type(module) is not types.ModuleType:
        _fail("exact isolated external runner is unavailable")
    runner_source = _require_sha256(_RUNNER_SOURCE_SHA256_INPUT, "runner source")
    runner_descriptor = _require_sha256(
        _RUNNER_DESCRIPTOR_SHA256_INPUT, "runner descriptor"
    )
    source_path = getattr(module, "__file__", None)
    if type(source_path) is not str:
        _fail("exact external runner source path is unavailable")
    _read_exact_source(
        source_path,
        expected_suffix=PINNED_EXTERNAL_EXECUTION_RUNNER_SOURCE_PATH,
        expected_sha256=runner_source,
    )
    descriptor = getattr(module, "canonical_external_execution_runner_descriptor_bytes", None)
    claim = getattr(module, "_consume_outcome_for_captured_external_consumer", None)
    if (
        type(descriptor) is not types.FunctionType
        or type(claim) is not types.FunctionType
        or getattr(module, "EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256", None)
        != runner_descriptor
    ):
        _fail("exact external runner consumer API differs")
    raw = descriptor()
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), runner_descriptor
    ):
        _fail("exact external runner descriptor bytes differ")
    return module, claim


def _load_source_module(
    *,
    module_name: str,
    source_path: str,
    source_sha256: str,
) -> types.ModuleType:
    if module_name in sys.modules:
        _fail(f"score-bearing module was already loaded: {module_name}")
    consumer_file = globals().get("__file__")
    if type(consumer_file) is not str:
        _fail("consumer source path is unavailable during score-closure load")
    repository_root = Path(consumer_file).resolve().parents[2]
    path = repository_root / source_path
    raw = _read_exact_source(
        str(path),
        expected_suffix=source_path,
        expected_sha256=source_sha256,
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[module_name] = module
    package = sys.modules.get("alberta_framework.benchmarks")
    if (
        type(package) is not types.ModuleType
        or package is not _BENCHMARKS_PACKAGE_AT_LOAD
    ):
        sys.modules.pop(module_name, None)
        _fail("captured benchmark parent package is unavailable or changed")
    attribute = module_name.rpartition(".")[2]
    if hasattr(package, attribute):
        sys.modules.pop(module_name, None)
        _fail(f"score-bearing package attribute was already present: {attribute}")
    setattr(package, attribute, module)
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        if getattr(package, attribute, None) is module:
            delattr(package, attribute)
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def _load_score_closure() -> types.ModuleType:
    _score_modules_absent()
    _load_source_module(
        module_name=PINNED_PROTOCOL_MODULE_NAME,
        source_path=PINNED_PROTOCOL_SOURCE_PATH,
        source_sha256=PINNED_PROTOCOL_SOURCE_SHA256,
    )
    _load_source_module(
        module_name=PINNED_SCORER_MODULE_NAME,
        source_path=PINNED_SCORER_SOURCE_PATH,
        source_sha256=PINNED_SCORER_SOURCE_SHA256,
    )
    bridge = _load_source_module(
        module_name=PINNED_RESULT_BRIDGE_MODULE_NAME,
        source_path=PINNED_RESULT_BRIDGE_SOURCE_PATH,
        source_sha256=PINNED_RESULT_BRIDGE_SOURCE_SHA256,
    )
    descriptor = getattr(bridge, "canonical_external_result_bridge_descriptor_bytes", None)
    if (
        type(descriptor) is not types.FunctionType
        or getattr(bridge, "EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION", None)
        != PINNED_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION
        or getattr(bridge, "EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256", None)
        != PINNED_RESULT_BRIDGE_DESCRIPTOR_SHA256
        or getattr(bridge, "SCORER_SOURCE_SHA256", None)
        != PINNED_SCORER_SOURCE_SHA256
        or getattr(bridge, "SCORER_PROTOCOL_SOURCE_SHA256", None)
        != PINNED_PROTOCOL_SOURCE_SHA256
        or getattr(bridge, "SCORER_METRIC_SHA256", None)
        != PINNED_METRIC_DESCRIPTOR_SHA256
    ):
        _fail("exact score-bearing bridge closure identity differs")
    raw = descriptor()
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), PINNED_RESULT_BRIDGE_DESCRIPTOR_SHA256
    ):
        _fail("exact score-bearing bridge descriptor bytes differ")
    return bridge


def _consume_matched_v3_external_outcome_to_captured_publication(
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
) -> object:
    """Private orchestrator called only by the exact captured publisher."""

    _require_boundary()
    _require_publisher_module()
    if explicit_publication_opt_in is not True:
        _fail("external outcome consumer requires exact publication opt-in")
    parent = _require_parent(publication_parent)
    candidate = _require_candidate(expected_candidate_id)
    environment_seed = _require_uint31(
        expected_environment_seed, "expected environment seed"
    )
    agent_seed = _require_uint31(expected_agent_seed, "expected agent seed")
    environment_commitment = _require_sha256(
        expected_environment_seed_commitment_sha256,
        "expected environment seed commitment",
    )
    agent_commitment = _require_sha256(
        expected_agent_seed_commitment_sha256,
        "expected agent seed commitment",
    )
    plan = _require_sha256(
        expected_qualification_plan_sha256, "expected qualification plan"
    )
    case_manifest = _require_sha256(
        expected_qualification_case_manifest_sha256,
        "expected qualification case manifest",
    )
    publisher_source_tree = _require_sha256(
        expected_publisher_source_tree_sha256,
        "expected publisher source tree",
    )
    workload_source_tree = _require_sha256(
        expected_workload_source_tree_sha256,
        "expected workload source tree",
    )
    staging_manifest = _require_sha256(
        expected_staging_manifest_sha256, "expected staging manifest"
    )
    maximum = _require_maximum(maximum_publication_total_bytes)
    preflight = _PUBLISHER_PARENT_PREFLIGHT_AT_LOAD
    if type(preflight) is not types.FunctionType:
        _fail("captured external publisher parent preflight is unavailable")
    preflighted_parent = preflight(publication_parent=parent)
    if type(preflighted_parent) is not _PATH_TYPE or preflighted_parent != parent:
        _fail("captured external publisher parent preflight result differs")
    _runner, claim = _require_runner_module()
    return claim(
        outcome_capability=outcome_capability,
        publication_parent=parent,
        expected_candidate_id=candidate,
        expected_environment_seed=environment_seed,
        expected_agent_seed=agent_seed,
        expected_environment_seed_commitment_sha256=environment_commitment,
        expected_agent_seed_commitment_sha256=agent_commitment,
        expected_qualification_plan_sha256=plan,
        expected_qualification_case_manifest_sha256=case_manifest,
        expected_publisher_source_tree_sha256=publisher_source_tree,
        expected_workload_source_tree_sha256=workload_source_tree,
        expected_staging_manifest_sha256=staging_manifest,
        maximum_publication_total_bytes=maximum,
        explicit_publication_opt_in=True,
    )


def _consume_claimed_matched_v3_external_execution_payload(
    *,
    sealed_payload: object,
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
) -> object:
    """Captured runner callback entered only after irreversible outcome claim."""

    consumer_source = _require_boundary()
    _require_publisher_module()
    runner, _claim = _require_runner_module()
    if explicit_publication_opt_in is not True:
        _fail("claimed external outcome requires exact publication opt-in")
    payload_type = getattr(runner, "_SealedExternalExecutionPayload", None)
    if type(payload_type) is not type or type(sealed_payload) is not payload_type:
        _fail("claimed external execution payload type differs")
    payload = cast(Any, sealed_payload)
    candidate_id = _require_candidate(expected_candidate_id)
    environment_seed = _require_uint31(
        expected_environment_seed, "expected environment seed"
    )
    agent_seed = _require_uint31(expected_agent_seed, "expected agent seed")
    if (
        payload.candidate_id != candidate_id
        or payload.environment_seed != environment_seed
        or payload.agent_seed != agent_seed
    ):
        _fail("claimed external execution candidate or seed binding differs")
    parent = _require_parent(publication_parent)
    maximum = _require_maximum(maximum_publication_total_bytes)
    execution_receipt = payload.execution_receipt_bytes
    execution_receipt_sha256 = _require_sha256(
        payload.execution_receipt_sha256, "execution receipt"
    )
    for value, label in (
        (execution_receipt, "execution receipt"),
        (payload.upstream_reward_npz, "upstream reward NPZ"),
        (payload.upstream_results_database, "upstream results database"),
        (payload.stdout, "stdout"),
        (payload.stderr, "stderr"),
    ):
        if type(value) is not bytes:
            _fail(f"claimed external execution {label} is not immutable bytes")
    if payload.upstream_video is not None and type(payload.upstream_video) is not bytes:
        _fail("claimed external execution video is not immutable optional bytes")
    parser = getattr(runner, "parse_matched_v3_external_execution_receipt", None)
    if type(parser) is not types.FunctionType:
        _fail("exact external runner receipt parser is unavailable")
    parsed = parser(
        execution_receipt,
        expected_receipt_sha256=execution_receipt_sha256,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        upstream_reward_npz=payload.upstream_reward_npz,
        upstream_results_database=payload.upstream_results_database,
        upstream_video=payload.upstream_video,
        stdout=payload.stdout,
        stderr=payload.stderr,
    )
    runtime = cast(dict[str, Any], parsed["runtime"])
    production_runner_exact = runtime["production_runner_exact"]
    if (
        type(production_runner_exact) is not bool
        or production_runner_exact is not payload.production_runner_exact
    ):
        _fail("claimed external execution production-runner binding differs")
    video_slot = payload.upstream_video or b""
    family = "ppo" if candidate_id in _PPO_CANDIDATE_IDS else "continuing"
    if (family == "ppo" and not video_slot) or (family == "continuing" and video_slot):
        _fail("claimed external execution video-slot family binding differs")
    preconversion_total = sum(
        len(value)
        for value in (
            execution_receipt,
            payload.upstream_reward_npz,
            payload.upstream_results_database,
            video_slot,
            payload.stdout,
            payload.stderr,
        )
    )
    if preconversion_total > maximum:
        _fail("claimed external execution content exceeds the publication ceiling")

    bridge = _load_score_closure()
    convert = getattr(bridge, "convert_external_reward_npz", None)
    receipt_bytes = getattr(
        bridge, "canonical_external_reward_conversion_receipt_bytes", None
    )
    receipt_sha = getattr(bridge, "external_reward_conversion_receipt_sha256", None)
    conversion_type = getattr(bridge, "ExternalRewardConversion", None)
    if (
        type(convert) is not types.FunctionType
        or type(receipt_bytes) is not types.FunctionType
        or type(receipt_sha) is not types.FunctionType
        or type(conversion_type) is not type
    ):
        _fail("exact score-bearing bridge conversion API differs")
    conversion: Any = cast(
        Any,
        convert(
            candidate_id=candidate_id,
            external_npz=payload.upstream_reward_npz,
        ),
    )
    if type(conversion) is not conversion_type:
        _fail("score-bearing bridge returned a non-exact conversion")
    exact_conversion = cast(Any, conversion)
    conversion_receipt = receipt_bytes(conversion)
    conversion_receipt_sha256 = receipt_sha(conversion)
    canonical_reward_npz = exact_conversion.canonical_scorer_npz
    canonical_reward_npz_sha256 = _require_sha256(
        exact_conversion.canonical_scorer_npz_sha256,
        "canonical scorer NPZ",
    )
    if (
        type(conversion_receipt) is not bytes
        or type(canonical_reward_npz) is not bytes
        or not hmac.compare_digest(
            hashlib.sha256(conversion_receipt).hexdigest(),
            _require_sha256(conversion_receipt_sha256, "conversion receipt"),
        )
        or not hmac.compare_digest(
            hashlib.sha256(canonical_reward_npz).hexdigest(),
            canonical_reward_npz_sha256,
        )
    ):
        _fail("score-bearing bridge conversion content binding differs")

    runner_source = _require_sha256(_RUNNER_SOURCE_SHA256_INPUT, "runner source")
    runner_descriptor = _require_sha256(
        _RUNNER_DESCRIPTOR_SHA256_INPUT, "runner descriptor"
    )
    facts_type = _PUBLISHER_FACTS_TYPE_AT_LOAD
    if type(facts_type) is not type:
        _fail("captured publisher facts type is unavailable")
    facts = facts_type(
        candidate_id=candidate_id,
        external_candidate_ordinal=EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS.index(
            candidate_id
        ),
        family=family,
        qualification_plan_sha256=_require_sha256(
            expected_qualification_plan_sha256, "expected qualification plan"
        ),
        qualification_case_manifest_sha256=_require_sha256(
            expected_qualification_case_manifest_sha256,
            "expected qualification case manifest",
        ),
        publisher_source_tree_sha256=_require_sha256(
            expected_publisher_source_tree_sha256,
            "expected publisher source tree",
        ),
        workload_source_tree_sha256=_require_sha256(
            expected_workload_source_tree_sha256,
            "expected workload source tree",
        ),
        staging_manifest_sha256=_require_sha256(
            expected_staging_manifest_sha256, "expected staging manifest"
        ),
        environment_seed_commitment_sha256=_require_sha256(
            expected_environment_seed_commitment_sha256,
            "expected environment seed commitment",
        ),
        agent_seed_commitment_sha256=_require_sha256(
            expected_agent_seed_commitment_sha256,
            "expected agent seed commitment",
        ),
        runner_descriptor_sha256=runner_descriptor,
        runner_source_sha256=runner_source,
        consumer_descriptor_sha256=EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256,
        consumer_source_sha256=consumer_source,
        bridge_descriptor_sha256=PINNED_RESULT_BRIDGE_DESCRIPTOR_SHA256,
        bridge_source_sha256=PINNED_RESULT_BRIDGE_SOURCE_SHA256,
        scorer_source_sha256=PINNED_SCORER_SOURCE_SHA256,
        protocol_source_sha256=PINNED_PROTOCOL_SOURCE_SHA256,
        metric_descriptor_sha256=PINNED_METRIC_DESCRIPTOR_SHA256,
        execution_contract_descriptor_sha256=(
            PINNED_EXECUTION_CONTRACT_DESCRIPTOR_SHA256
        ),
        staging_descriptor_sha256=PINNED_STAGING_DESCRIPTOR_SHA256,
        seed_transport_descriptor_sha256=PINNED_SEED_TRANSPORT_DESCRIPTOR_SHA256,
        execution_receipt_sha256=execution_receipt_sha256,
        conversion_receipt_sha256=conversion_receipt_sha256,
        production_runner_exact=production_runner_exact,
        video_slot_mode=(
            "opaque_ppo_video"
            if family == "ppo"
            else "absent_for_continuing_zero_length_slot"
        ),
        maximum_publication_total_bytes=maximum,
    )
    content_payloads = (
        ("external-execution-receipt.json", execution_receipt),
        ("external-conversion-receipt.json", conversion_receipt),
        ("upstream-reward.npz", payload.upstream_reward_npz),
        ("upstream-results.db", payload.upstream_results_database),
        ("upstream-video-slot.bin", video_slot),
        ("reward-trace.npz", canonical_reward_npz),
        ("stdout.bin", payload.stdout),
        ("stderr.bin", payload.stderr),
    )
    builder = _PUBLISHER_MANIFEST_BUILDER_AT_LOAD
    sink = _PUBLISHER_SINK_AT_LOAD
    if type(builder) is not types.FunctionType or type(sink) is not types.FunctionType:
        _fail("captured publisher manifest or sink is unavailable")
    outcome_manifest = builder(facts=facts, content_payloads=content_payloads)
    if type(outcome_manifest) is not bytes:
        _fail("captured publisher returned a non-byte outcome manifest")
    metadata = sink(
        publication_parent=parent,
        role_payloads=(
            ("external-outcome-manifest.json", outcome_manifest),
            *content_payloads,
        ),
        facts=facts,
    )
    if type(metadata) is not _PUBLISHER_METADATA_TYPE_AT_LOAD:
        _fail("captured publisher returned non-exact metadata")
    return metadata


def external_outcome_consumer_descriptor() -> dict[str, Any]:
    """Return detached permanently nonauthorizing descriptor content."""

    return _strict_json(_DESCRIPTOR_BYTES, maximum=_MAX_DESCRIPTOR_BYTES)


def canonical_external_outcome_consumer_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def parse_external_outcome_consumer_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact self-pinned descriptor."""

    value = _strict_json(raw, maximum=_MAX_DESCRIPTOR_BYTES)
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256
    ):
        _fail("external outcome consumer descriptor identity differs")
    return value


_CONSUMER_GUARD_AT_LOAD: Final = object()
_CONSUMER_GUARDED_FUNCTION_NAMES: Final = (
    "_require_boundary",
    "_require_publisher_module",
    "_score_modules_absent",
    "_require_runner_module",
    "_read_exact_source",
    "_load_source_module",
    "_load_score_closure",
    "_consume_matched_v3_external_outcome_to_captured_publication",
    "_consume_claimed_matched_v3_external_execution_payload",
)


def _consumer_function_surface() -> tuple[tuple[Any, ...], ...]:
    surface: list[tuple[Any, ...]] = []
    for name in _CONSUMER_GUARDED_FUNCTION_NAMES:
        value = globals().get(name)
        if type(value) is not types.FunctionType:
            _fail(f"external outcome consumer guarded function is unavailable: {name}")
        surface.append(
            (
                name,
                value,
                value.__code__,
                value.__defaults__,
                value.__kwdefaults__,
            )
        )
    return tuple(surface)


_CONSUMER_FUNCTION_SURFACE_AT_LOAD: Final = _consumer_function_surface()


def _replay_external_outcome_consumer_guard(
    guard: object,
) -> tuple[types.FunctionType, types.FunctionType]:
    """Replay the captured critical function surface for the exact publisher."""

    _require_boundary()
    if guard is not _CONSUMER_GUARD_AT_LOAD:
        _fail("external outcome consumer guard identity differs")
    current = _consumer_function_surface()
    if len(current) != len(_CONSUMER_FUNCTION_SURFACE_AT_LOAD) or any(
        observed[0] != expected[0]
        or observed[1] is not expected[1]
        or observed[2] is not expected[2]
        or observed[3] is not expected[3]
        or observed[4] is not expected[4]
        for observed, expected in zip(
            current, _CONSUMER_FUNCTION_SURFACE_AT_LOAD, strict=True
        )
    ):
        _fail("external outcome consumer guarded function surface changed")
    orchestrator = _consume_matched_v3_external_outcome_to_captured_publication
    claimed_sink = _consume_claimed_matched_v3_external_execution_payload
    return cast(types.FunctionType, orchestrator), cast(
        types.FunctionType, claimed_sink
    )


__all__ = [
    "EXTERNAL_OUTCOME_CONSUMER_CANDIDATE_IDS",
    "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256",
    "EXTERNAL_OUTCOME_CONSUMER_STATUS",
    "ForagerMatchedV3ExternalOutcomeConsumerError",
    "canonical_external_outcome_consumer_descriptor_bytes",
    "external_outcome_consumer_descriptor",
    "parse_external_outcome_consumer_descriptor",
]
