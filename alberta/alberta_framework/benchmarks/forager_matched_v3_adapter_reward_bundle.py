"""In-process, non-authorizing adapter-result bridge to the strict v3 scorer.

The two local adapter runners retain exact raw reward traces.  This module accepts only a
runner's registered production outcome, serializes its structural runner receipt, converts
the trace into the sole canonical NPZ layout accepted by the v3 scorer, and returns an
immutable in-memory bundle.  It performs no filesystem write and grants no execution,
qualification, ingestion, evidence, or promotion authority.

Persisted receipt bytes remain structural checksums.  Only the runner's process-local
completion capability distinguishes a live outcome while this builder is executing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_full_rainbow_runner as full_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_runner as ppo_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_reward_bundle_descriptor.v1"
)
ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_reward_bundle_manifest.v1"
)
ADAPTER_REWARD_BUNDLE_STATUS: Final = "implemented_unqualified"

_SCORER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
)
_SCORER_SOURCE_SHA256: Final = (
    "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
)
_FULL_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
)
_FULL_RUNNER_SOURCE_SHA256: Final = (
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
)
_PPO_RUNNER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
)
_PPO_RUNNER_SOURCE_SHA256: Final = (
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
)
_MAX_MANIFEST_BYTES: Final = 256 * 1024


class ForagerMatchedV3AdapterRewardBundleError(ValueError):
    """An adapter outcome, scorer conversion, receipt, or bundle failed closed."""


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()


for _module_file, _path, _expected in (
    (scorer.__file__, _SCORER_SOURCE_PATH, _SCORER_SOURCE_SHA256),
    (full_runner.__file__, _FULL_RUNNER_SOURCE_PATH, _FULL_RUNNER_SOURCE_SHA256),
    (ppo_runner.__file__, _PPO_RUNNER_SOURCE_PATH, _PPO_RUNNER_SOURCE_SHA256),
):
    if not hmac.compare_digest(_source_sha256(_module_file, _path), _expected):
        raise RuntimeError(f"adapter reward bundle source binding drifted: {_path}")


def _canonical_json(value: object, *, label: str) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} exceeds its canonical byte limit"
        )
    return raw


def _require_exact_keys(
    value: dict[str, Any], *, expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} fields differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
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
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} must be an exact integer"
        )
    if minimum is not None and value < minimum:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} is below its lower bound"
        )
    if maximum is not None and value > maximum:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} exceeds its upper bound"
        )
    return value


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "execution_authorized": False,
        "execution_ready": False,
        "ingestion_authorized": False,
        "performance_claim_allowed": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The live process capability is not serialized into this bundle.",
        "Persisted runner, score, and manifest hashes do not independently prove execution.",
        "The bundle is in memory and does not claim durable or atomic filesystem publication.",
        "Seed provenance remains whatever unverified status the runner receipt records.",
        "A valid conversion grants no ingestion, qualification, evidence, or promotion authority.",
    ]


def _runner_bindings() -> dict[str, dict[str, str]]:
    return {
        "adapted_full_rainbow": {
            "runner_descriptor_schema_version": (
                full_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "runner_descriptor_sha256": (
                full_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
            ),
            "runner_source_path": _FULL_RUNNER_SOURCE_PATH,
            "runner_source_sha256": _FULL_RUNNER_SOURCE_SHA256,
            "runner_receipt_schema_version": (
                full_runner.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
            ),
        },
        "adapted_ppo_gru": {
            "runner_descriptor_schema_version": (
                ppo_runner.PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "runner_descriptor_sha256": ppo_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
            "runner_source_path": _PPO_RUNNER_SOURCE_PATH,
            "runner_source_sha256": _PPO_RUNNER_SOURCE_SHA256,
            "runner_receipt_schema_version": (
                ppo_runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
            ),
        },
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
        "status": ADAPTER_REWARD_BUNDLE_STATUS,
        "classification": "in_process_conversion_non_authorizing",
        "candidate_bindings": _runner_bindings(),
        "scorer": {
            "source_path": _SCORER_SOURCE_PATH,
            "source_sha256": _SCORER_SOURCE_SHA256,
            "score_receipt_schema_version": scorer.SCORE_RECEIPT_SCHEMA_VERSION,
            "npz_container_schema_version": scorer.NPZ_CONTAINER_SCHEMA_VERSION,
            "canonical_npz_size_bytes": scorer.CANONICAL_NPZ_SIZE_BYTES,
            "raw_trace_encoding_schema_version": (
                scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION
            ),
        },
        "metric": {
            "schema_version": protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION,
            "sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
            "horizon": protocol.MATCHED_V3_HORIZON,
            "accumulation": "ordered_exact_integer_sum",
        },
        "conversion": {
            "runner_production_capability_required_in_process": True,
            "complete_raw_trace_required": True,
            "canonical_npz_reingested_before_return": True,
            "runner_and_scorer_scores_must_match": True,
            "filesystem_writes": False,
            "persisted_content_independently_attests_execution": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(), label="adapter reward bundle descriptor"
)
ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256,
):
    raise RuntimeError("adapter reward bundle descriptor identity drifted")


def adapter_reward_bundle_descriptor() -> dict[str, Any]:
    """Return a detached copy of the exact conversion descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_adapter_reward_bundle_descriptor_bytes() -> bytes:
    """Return the exact descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_adapter_reward_bundle_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical descriptor bytes."""

    if type(raw) is not bytes or raw != _DESCRIPTOR_BYTES:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle descriptor bytes do not match"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle descriptor digest drifted"
        )
    return adapter_reward_bundle_descriptor()


@dataclass(frozen=True, slots=True)
class MatchedV3AdapterRewardBundle:
    """Immutable in-memory canonical reward artifact and detached receipts."""

    candidate_id: str
    runner_receipt_bytes: bytes
    reward_artifact_bytes: bytes
    score_receipt_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        return parse_adapter_reward_bundle_manifest(
            self.manifest_bytes,
            expected_manifest_sha256=self.manifest_sha256,
        )


def _runner_receipt_facts(
    candidate_id: str,
    runner_receipt_bytes: bytes,
) -> tuple[dict[str, Any], int, str, int]:
    try:
        if candidate_id == "adapted_full_rainbow":
            parsed = full_runner.parse_full_rainbow_result_receipt(runner_receipt_bytes)
            score = cast(dict[str, Any], parsed["score"])
            return (
                parsed,
                cast(int, score["raw_cumulative_score"]),
                cast(str, score["raw_reward_trace_sha256"]),
                cast(int, score["raw_reward_trace_length"]),
            )
        if candidate_id == "adapted_ppo_gru":
            parsed = ppo_runner.parse_ppo_gru_result_receipt(runner_receipt_bytes)
            trace = cast(dict[str, Any], parsed["raw_reward_trace"])
            return (
                parsed,
                cast(int, parsed["raw_cumulative_score"]),
                cast(str, trace["sha256"]),
                cast(int, trace["length"]),
            )
    except (
        full_runner.FullRainbowRunnerContractError,
        ppo_runner.ForagerMatchedV3PPOGRURunnerError,
    ) as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter runner receipt failed its frozen structural parser"
        ) from exc
    raise ForagerMatchedV3AdapterRewardBundleError(
        "candidate has no adapter reward conversion binding"
    )


def _manifest_bindings(candidate_id: str) -> dict[str, str]:
    try:
        runner = _runner_bindings()[candidate_id]
    except (KeyError, TypeError) as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "candidate has no adapter reward conversion binding"
        ) from exc
    return {
        "adapter_reward_bundle_descriptor_schema_version": (
            ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "adapter_reward_bundle_descriptor_sha256": (
            ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
        ),
        "runner_descriptor_schema_version": runner[
            "runner_descriptor_schema_version"
        ],
        "runner_descriptor_sha256": runner["runner_descriptor_sha256"],
        "runner_source_path": runner["runner_source_path"],
        "runner_source_sha256": runner["runner_source_sha256"],
        "scorer_source_path": _SCORER_SOURCE_PATH,
        "scorer_source_sha256": _SCORER_SOURCE_SHA256,
        "cumulative_reward_metric_schema_version": (
            protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
        ),
        "cumulative_reward_metric_sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
    }


def _manifest_body(
    *,
    candidate_id: str,
    runner_receipt_bytes: bytes,
    reward_artifact_bytes: bytes,
    score_receipt: scorer.MatchedV3ScoreReceipt,
    raw_trace: bytes,
) -> dict[str, Any]:
    runner = _runner_bindings()[candidate_id]
    return {
        "schema_version": ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "classification": "converted_adapter_result_non_authorizing",
        "candidate_id": candidate_id,
        "bindings": _manifest_bindings(candidate_id),
        "runner_receipt": {
            "schema_version": runner["runner_receipt_schema_version"],
            "sha256": hashlib.sha256(runner_receipt_bytes).hexdigest(),
            "size_bytes": len(runner_receipt_bytes),
            "structural_content_independently_attests_execution": False,
        },
        "raw_reward_trace": {
            "encoding_schema_version": scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION,
            "encoding": scorer.RAW_TRACE_ENCODING,
            "length": len(raw_trace),
            "bytes_sha256": hashlib.sha256(raw_trace).hexdigest(),
            "version_framed_sha256": score_receipt.raw_trace_sha256,
            "raw_cumulative_score": score_receipt.cumulative_score,
        },
        "reward_artifact": {
            "container_schema_version": scorer.NPZ_CONTAINER_SCHEMA_VERSION,
            "sha256": hashlib.sha256(reward_artifact_bytes).hexdigest(),
            "size_bytes": len(reward_artifact_bytes),
        },
        "score_receipt": {
            "schema_version": scorer.SCORE_RECEIPT_SCHEMA_VERSION,
            "sha256": hashlib.sha256(score_receipt.canonical_json()).hexdigest(),
            "receipt_body_sha256": score_receipt.receipt_sha256,
            "size_bytes": len(score_receipt.canonical_json()),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _manifest_bytes(body: dict[str, Any]) -> tuple[bytes, str]:
    body_bytes = _canonical_json(body, label="adapter reward bundle manifest body")
    digest = hashlib.sha256(body_bytes).hexdigest()
    payload = dict(body)
    payload["manifest_body_sha256"] = digest
    return _canonical_json(payload, label="adapter reward bundle manifest"), digest


def _require_object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedV3AdapterRewardBundleError(
            f"{label} must be a plain object"
        )
    return cast(dict[str, Any], value)


def _validate_manifest_body(body: dict[str, Any]) -> None:
    _require_exact_keys(
        body,
        expected=frozenset(
            {
                "schema_version",
                "classification",
                "candidate_id",
                "bindings",
                "runner_receipt",
                "raw_reward_trace",
                "reward_artifact",
                "score_receipt",
                "claims",
                "limitations",
            }
        ),
        label="adapter reward bundle manifest body",
    )
    if body["schema_version"] != ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest schema drifted"
        )
    if body["classification"] != "converted_adapter_result_non_authorizing":
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest classification drifted"
        )
    candidate_id = body["candidate_id"]
    if type(candidate_id) is not str:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle candidate ID must be an exact string"
        )
    runner = _runner_bindings().get(candidate_id)
    if runner is None:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "candidate has no adapter reward conversion binding"
        )
    bindings = _require_object(body["bindings"], label="manifest bindings")
    if bindings != _manifest_bindings(candidate_id):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest fixed bindings drifted"
        )

    runner_receipt = _require_object(
        body["runner_receipt"], label="manifest runner receipt"
    )
    _require_exact_keys(
        runner_receipt,
        expected=frozenset(
            {
                "schema_version",
                "sha256",
                "size_bytes",
                "structural_content_independently_attests_execution",
            }
        ),
        label="manifest runner receipt",
    )
    if (
        runner_receipt["schema_version"] != runner["runner_receipt_schema_version"]
        or runner_receipt["structural_content_independently_attests_execution"]
        is not False
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle runner receipt contract drifted"
        )
    _require_sha256(runner_receipt["sha256"], label="runner receipt digest")
    _require_exact_int(
        runner_receipt["size_bytes"],
        label="runner receipt size",
        minimum=1,
        maximum=4 * 1024 * 1024,
    )

    raw_trace = _require_object(
        body["raw_reward_trace"], label="manifest raw reward trace"
    )
    _require_exact_keys(
        raw_trace,
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
        raw_trace["encoding_schema_version"] != scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION
        or raw_trace["encoding"] != scorer.RAW_TRACE_ENCODING
        or raw_trace["length"] != protocol.MATCHED_V3_HORIZON
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle raw trace contract drifted"
        )
    _require_exact_int(
        raw_trace["length"],
        label="raw reward trace length",
        minimum=protocol.MATCHED_V3_HORIZON,
        maximum=protocol.MATCHED_V3_HORIZON,
    )
    _require_sha256(raw_trace["bytes_sha256"], label="raw reward trace byte digest")
    _require_sha256(
        raw_trace["version_framed_sha256"],
        label="version-framed raw reward trace digest",
    )
    _require_exact_int(
        raw_trace["raw_cumulative_score"],
        label="raw cumulative score",
        minimum=-protocol.MATCHED_V3_HORIZON,
        maximum=30 * protocol.MATCHED_V3_HORIZON,
    )

    reward_artifact = _require_object(
        body["reward_artifact"], label="manifest reward artifact"
    )
    _require_exact_keys(
        reward_artifact,
        expected=frozenset({"container_schema_version", "sha256", "size_bytes"}),
        label="manifest reward artifact",
    )
    if reward_artifact["container_schema_version"] != scorer.NPZ_CONTAINER_SCHEMA_VERSION:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward artifact container schema drifted"
        )
    _require_sha256(reward_artifact["sha256"], label="reward artifact digest")
    _require_exact_int(
        reward_artifact["size_bytes"],
        label="reward artifact size",
        minimum=scorer.CANONICAL_NPZ_SIZE_BYTES,
        maximum=scorer.CANONICAL_NPZ_SIZE_BYTES,
    )

    score_receipt = _require_object(
        body["score_receipt"], label="manifest score receipt"
    )
    _require_exact_keys(
        score_receipt,
        expected=frozenset(
            {"schema_version", "sha256", "receipt_body_sha256", "size_bytes"}
        ),
        label="manifest score receipt",
    )
    if score_receipt["schema_version"] != scorer.SCORE_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward score receipt schema drifted"
        )
    _require_sha256(score_receipt["sha256"], label="score receipt digest")
    _require_sha256(
        score_receipt["receipt_body_sha256"], label="score receipt body digest"
    )
    _require_exact_int(
        score_receipt["size_bytes"],
        label="score receipt size",
        minimum=1,
        maximum=_MAX_MANIFEST_BYTES,
    )
    claims = _require_object(body["claims"], label="manifest claims")
    _require_exact_keys(
        claims,
        expected=frozenset(_claims()),
        label="manifest claims",
    )
    if any(value is not False for value in claims.values()):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest claims must be exact false booleans"
        )
    limitations = body["limitations"]
    if (
        type(limitations) is not list
        or any(type(item) is not str for item in limitations)
        or limitations != _limitations()
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest claims or limitations drifted"
        )


def _build_bundle(
    *,
    candidate_id: str,
    runner_receipt_bytes: bytes,
    raw_trace: bytes,
    expected_score: int,
) -> MatchedV3AdapterRewardBundle:
    if type(runner_receipt_bytes) is not bytes or type(raw_trace) is not bytes:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "runner receipt and raw trace must be exact bytes"
        )
    if type(expected_score) is not int:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "expected score must be an exact integer"
        )
    _, receipt_score, receipt_trace_sha256, receipt_trace_length = (
        _runner_receipt_facts(candidate_id, runner_receipt_bytes)
    )
    if (
        receipt_score != expected_score
        or receipt_trace_length != len(raw_trace)
        or not hmac.compare_digest(
            receipt_trace_sha256, hashlib.sha256(raw_trace).hexdigest()
        )
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "runner receipt disagrees with its raw trace or score"
        )
    try:
        artifact = scorer.canonical_reward_npz_bytes(raw_trace)
        score_receipt = scorer.ingest_reward_npz_bytes(artifact)
    except scorer.ForagerMatchedV3ScorerError as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "strict scorer rejected the adapter reward trace"
        ) from exc
    if score_receipt.cumulative_score != expected_score:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "runner and strict scorer cumulative scores disagree"
        )
    body = _manifest_body(
        candidate_id=candidate_id,
        runner_receipt_bytes=runner_receipt_bytes,
        reward_artifact_bytes=artifact,
        score_receipt=score_receipt,
        raw_trace=raw_trace,
    )
    manifest_bytes, manifest_sha256 = _manifest_bytes(body)
    bundle = MatchedV3AdapterRewardBundle(
        candidate_id=candidate_id,
        runner_receipt_bytes=runner_receipt_bytes,
        reward_artifact_bytes=artifact,
        score_receipt_bytes=score_receipt.canonical_json(),
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )
    return validate_adapter_reward_bundle(bundle)


def build_full_rainbow_reward_bundle(
    result: full_runner.FullRainbowRunnerResult,
) -> MatchedV3AdapterRewardBundle:
    """Convert only an exact registered Full Rainbow production result."""

    try:
        runner_receipt = full_runner.canonical_full_rainbow_result_receipt_bytes(
            result
        )
    except full_runner.FullRainbowRunnerContractError as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "Full Rainbow result lacks a live production completion capability"
        ) from exc
    return _build_bundle(
        candidate_id="adapted_full_rainbow",
        runner_receipt_bytes=runner_receipt,
        raw_trace=result.raw_reward_trace,
        expected_score=result.cumulative_raw_score,
    )


def build_ppo_gru_reward_bundle(
    outcome: ppo_runner.PPOGRURunnerOutcome,
) -> MatchedV3AdapterRewardBundle:
    """Convert only an exact registered PPO-GRU production outcome."""

    try:
        runner_receipt = ppo_runner.canonical_ppo_gru_result_receipt_bytes(outcome)
    except ppo_runner.ForagerMatchedV3PPOGRURunnerError as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "PPO-GRU outcome lacks a live production completion capability"
        ) from exc
    return _build_bundle(
        candidate_id="adapted_ppo_gru",
        runner_receipt_bytes=runner_receipt,
        raw_trace=outcome.raw_reward_trace,
        expected_score=outcome.raw_cumulative_score,
    )


def parse_adapter_reward_bundle_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly parse a structural manifest without granting execution authority."""

    if type(raw) is not bytes or type(expected_manifest_sha256) is not str:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "manifest and expected digest must be exact bytes/string"
        )
    if not 0 < len(raw) <= _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest byte length is invalid"
        )
    _require_sha256(expected_manifest_sha256, label="expected manifest digest")
    try:
        parsed = json.loads(raw.decode("ascii"))
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest is not strict ASCII JSON"
        ) from exc
    if type(parsed) is not dict or _canonical_json(
        parsed, label="adapter reward bundle manifest"
    ) != raw:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest is not canonical"
        )
    payload = cast(dict[str, Any], parsed)
    supplied = payload.get("manifest_body_sha256")
    if type(supplied) is not str or supplied != expected_manifest_sha256:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest digest binding differs"
        )
    body = dict(payload)
    del body["manifest_body_sha256"]
    if not hmac.compare_digest(
        hashlib.sha256(
            _canonical_json(body, label="adapter reward bundle manifest body")
        ).hexdigest(),
        expected_manifest_sha256,
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest body digest drifted"
        )
    if body.get("schema_version") != ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest schema drifted"
        )
    if body.get("claims") != _claims() or body.get("limitations") != _limitations():
        raise ForagerMatchedV3AdapterRewardBundleError(
            "adapter reward bundle manifest claims or limitations drifted"
        )
    _validate_manifest_body(body)
    return payload


def validate_adapter_reward_bundle(
    bundle: object,
) -> MatchedV3AdapterRewardBundle:
    """Replay all content relationships in one in-memory bundle."""

    if type(bundle) is not MatchedV3AdapterRewardBundle:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "bundle must be an exact MatchedV3AdapterRewardBundle"
        )
    for name in (
        "runner_receipt_bytes",
        "reward_artifact_bytes",
        "score_receipt_bytes",
        "manifest_bytes",
    ):
        if type(getattr(bundle, name)) is not bytes:
            raise ForagerMatchedV3AdapterRewardBundleError(
                f"bundle {name} must be exact bytes"
            )
    manifest = parse_adapter_reward_bundle_manifest(
        bundle.manifest_bytes,
        expected_manifest_sha256=bundle.manifest_sha256,
    )
    body = dict(manifest)
    del body["manifest_body_sha256"]
    try:
        score_receipt = scorer.parse_score_receipt(bundle.score_receipt_bytes)
        replayed_score = scorer.ingest_reward_npz_bytes(bundle.reward_artifact_bytes)
        raw_trace = scorer.extract_canonical_reward_trace(bundle.reward_artifact_bytes)
    except scorer.ForagerMatchedV3ScorerError as exc:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "bundle reward artifact or score receipt failed strict replay"
        ) from exc
    if score_receipt != replayed_score:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "bundle score receipt does not replay from its reward artifact"
        )
    _, runner_score, runner_trace_sha256, runner_trace_length = _runner_receipt_facts(
        bundle.candidate_id, bundle.runner_receipt_bytes
    )
    if (
        runner_score != score_receipt.cumulative_score
        or runner_trace_length != len(raw_trace)
        or not hmac.compare_digest(
            runner_trace_sha256, hashlib.sha256(raw_trace).hexdigest()
        )
    ):
        raise ForagerMatchedV3AdapterRewardBundleError(
            "bundle runner receipt does not agree with strict scorer replay"
        )
    expected_body = _manifest_body(
        candidate_id=bundle.candidate_id,
        runner_receipt_bytes=bundle.runner_receipt_bytes,
        reward_artifact_bytes=bundle.reward_artifact_bytes,
        score_receipt=score_receipt,
        raw_trace=raw_trace,
    )
    if body != expected_body:
        raise ForagerMatchedV3AdapterRewardBundleError(
            "bundle manifest does not replay from its exact contents"
        )
    return bundle


__all__ = [
    "ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256",
    "ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "ADAPTER_REWARD_BUNDLE_STATUS",
    "ForagerMatchedV3AdapterRewardBundleError",
    "MatchedV3AdapterRewardBundle",
    "adapter_reward_bundle_descriptor",
    "build_full_rainbow_reward_bundle",
    "build_ppo_gru_reward_bundle",
    "canonical_adapter_reward_bundle_descriptor_bytes",
    "parse_adapter_reward_bundle_descriptor",
    "parse_adapter_reward_bundle_manifest",
    "validate_adapter_reward_bundle",
]
