"""Synthetic contracts for the non-authorizing v3 adapter reward bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_full_rainbow_runner as full_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_runner as ppo_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]


def _zero_trace() -> bytes:
    return bytes(protocol.MATCHED_V3_HORIZON)


def _rehashed_manifest(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = dict(payload)
    del body["manifest_body_sha256"]
    body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(body_bytes).hexdigest()
    rewritten = dict(body)
    rewritten["manifest_body_sha256"] = digest
    return (
        json.dumps(rewritten, sort_keys=True, separators=(",", ":")).encode("ascii"),
        digest,
    )


def test_descriptor_is_literal_source_bound_detached_and_non_authorizing() -> None:
    raw = bundle.canonical_adapter_reward_bundle_descriptor_bytes()
    descriptor = bundle.parse_adapter_reward_bundle_descriptor(raw)

    assert len(raw) == 2_754
    assert hashlib.sha256(raw).hexdigest() == (
        bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
    )
    assert bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256 == (
        "1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5"
    )
    assert descriptor["conversion"] == {
        "canonical_npz_reingested_before_return": True,
        "complete_raw_trace_required": True,
        "filesystem_writes": False,
        "persisted_content_independently_attests_execution": False,
        "runner_and_scorer_scores_must_match": True,
        "runner_production_capability_required_in_process": True,
    }
    assert set(descriptor["claims"].values()) == {False}
    for candidate in descriptor["candidate_bindings"].values():
        source = _ROOT / candidate["runner_source_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == candidate[
            "runner_source_sha256"
        ]
    scorer_source = _ROOT / descriptor["scorer"]["source_path"]
    assert hashlib.sha256(scorer_source.read_bytes()).hexdigest() == descriptor[
        "scorer"
    ]["source_sha256"]

    descriptor["claims"]["authority_granted"] = True
    assert bundle.adapter_reward_bundle_descriptor()["claims"][
        "authority_granted"
    ] is False
    with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
        bundle.parse_adapter_reward_bundle_descriptor(b" " + raw)


def test_private_conversion_round_trips_the_strict_scorer_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _zero_trace()
    receipt_bytes = b"synthetic-structural-runner-receipt"

    def facts(
        candidate_id: str, supplied_receipt: bytes
    ) -> tuple[dict[str, Any], int, str, int]:
        assert candidate_id == "adapted_full_rainbow"
        if supplied_receipt != receipt_bytes:
            raise bundle.ForagerMatchedV3AdapterRewardBundleError(
                "synthetic runner receipt differs"
            )
        return {}, 0, hashlib.sha256(trace).hexdigest(), len(trace)

    monkeypatch.setattr(bundle, "_runner_receipt_facts", facts)
    built = bundle._build_bundle(
        candidate_id="adapted_full_rainbow",
        runner_receipt_bytes=receipt_bytes,
        raw_trace=trace,
        expected_score=0,
    )

    assert bundle.validate_adapter_reward_bundle(built) is built
    assert scorer.extract_canonical_reward_trace(built.reward_artifact_bytes) == trace
    score_receipt = scorer.parse_score_receipt(built.score_receipt_bytes)
    assert score_receipt.cumulative_score == 0
    manifest = built.manifest()
    assert manifest["candidate_id"] == "adapted_full_rainbow"
    assert manifest["raw_reward_trace"] == {
        "bytes_sha256": hashlib.sha256(trace).hexdigest(),
        "encoding": scorer.RAW_TRACE_ENCODING,
        "encoding_schema_version": scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION,
        "length": protocol.MATCHED_V3_HORIZON,
        "raw_cumulative_score": 0,
        "version_framed_sha256": score_receipt.raw_trace_sha256,
    }
    assert manifest["runner_receipt"][
        "structural_content_independently_attests_execution"
    ] is False
    assert set(manifest["claims"].values()) == {False}

    changed_artifact = bytearray(built.reward_artifact_bytes)
    changed_artifact[100] ^= 1
    with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
        bundle.validate_adapter_reward_bundle(
            dataclasses.replace(built, reward_artifact_bytes=bytes(changed_artifact))
        )
    with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
        bundle.validate_adapter_reward_bundle(
            dataclasses.replace(built, runner_receipt_bytes=b"changed")
        )


@pytest.mark.parametrize(
    ("receipt_score", "receipt_trace_sha256", "receipt_trace_length"),
    (
        (1, None, None),
        (0, "0" * 64, None),
        (0, None, protocol.MATCHED_V3_HORIZON - 1),
    ),
)
def test_runner_and_scorer_trace_or_score_disagreement_fails_before_bundle(
    monkeypatch: pytest.MonkeyPatch,
    receipt_score: int,
    receipt_trace_sha256: str | None,
    receipt_trace_length: int | None,
) -> None:
    trace = _zero_trace()
    monkeypatch.setattr(
        bundle,
        "_runner_receipt_facts",
        lambda *args: (
            {},
            receipt_score,
            (
                hashlib.sha256(trace).hexdigest()
                if receipt_trace_sha256 is None
                else receipt_trace_sha256
            ),
            len(trace) if receipt_trace_length is None else receipt_trace_length,
        ),
    )
    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError,
        match="receipt disagrees",
    ):
        bundle._build_bundle(
            candidate_id="adapted_ppo_gru",
            runner_receipt_bytes=b"receipt",
            raw_trace=trace,
            expected_score=0,
        )


def test_public_builders_reject_objects_without_live_production_capabilities() -> None:
    constructed_full = full_runner.FullRainbowRunnerResult(
        raw_reward_trace=b"\x00",
        cumulative_raw_score=0,
        interactions=1,
        receipt_bytes=b"{}",
        production_runtime=False,
    )
    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError,
        match="completion capability",
    ):
        bundle.build_full_rainbow_reward_bundle(constructed_full)

    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError,
        match="completion capability",
    ):
        bundle.build_ppo_gru_reward_bundle(object())  # type: ignore[arg-type]


def test_manifest_parser_rejects_mutation_noncanonical_and_wrong_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _zero_trace()
    receipt_bytes = b"receipt"
    monkeypatch.setattr(
        bundle,
        "_runner_receipt_facts",
        lambda *args: ({}, 0, hashlib.sha256(trace).hexdigest(), len(trace)),
    )
    built = bundle._build_bundle(
        candidate_id="adapted_ppo_gru",
        runner_receipt_bytes=receipt_bytes,
        raw_trace=trace,
        expected_score=0,
    )
    parsed = json.loads(built.manifest_bytes)
    parsed["claims"]["authority_granted"] = True
    changed = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    for raw, digest in (
        (b" " + built.manifest_bytes, built.manifest_sha256),
        (changed, built.manifest_sha256),
        (built.manifest_bytes, "0" * 64),
    ):
        with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
            bundle.parse_adapter_reward_bundle_manifest(
                raw,
                expected_manifest_sha256=digest,
            )

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.update({"classification": "authoritative_evidence"}),
        lambda value: value.update({"execution_authorized": True}),
        lambda value: value["bindings"].update(
            {"runner_descriptor_sha256": "0" * 64}
        ),
        lambda value: value["raw_reward_trace"].update({"length": 1}),
        lambda value: value["runner_receipt"].update({"unexpected": False}),
        lambda value: value.update(
            {"claims": {key: 0 for key in value["claims"]}}
        ),
    )
    for mutation in mutations:
        mutated = json.loads(built.manifest_bytes)
        mutation(mutated)
        rehashed, rehashed_digest = _rehashed_manifest(mutated)
        with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
            bundle.parse_adapter_reward_bundle_manifest(
                rehashed,
                expected_manifest_sha256=rehashed_digest,
            )

    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError, match="byte length"
    ):
        bundle.parse_adapter_reward_bundle_manifest(
            b"{" + b" " * bundle._MAX_MANIFEST_BYTES,
            expected_manifest_sha256="0" * 64,
        )
    huge_integer_manifest = b'{"value":' + b"9" * 5_000 + b"}"
    with pytest.raises(bundle.ForagerMatchedV3AdapterRewardBundleError):
        bundle.parse_adapter_reward_bundle_manifest(
            huge_integer_manifest,
            expected_manifest_sha256=hashlib.sha256(huge_integer_manifest).hexdigest(),
        )


def test_runner_parser_failures_are_normalized_to_the_bundle_error_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_full(raw: bytes) -> dict[str, Any]:
        del raw
        raise full_runner.FullRainbowRunnerContractError("invalid Full Rainbow receipt")

    def reject_ppo(raw: bytes) -> dict[str, Any]:
        del raw
        raise ppo_runner.ForagerMatchedV3PPOGRURunnerError("invalid PPO-GRU receipt")

    monkeypatch.setattr(full_runner, "parse_full_rainbow_result_receipt", reject_full)
    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError,
        match="failed its frozen structural parser",
    ):
        bundle._runner_receipt_facts("adapted_full_rainbow", b"{}")

    monkeypatch.setattr(ppo_runner, "parse_ppo_gru_result_receipt", reject_ppo)
    with pytest.raises(
        bundle.ForagerMatchedV3AdapterRewardBundleError,
        match="failed its frozen structural parser",
    ):
        bundle._runner_receipt_facts("adapted_ppo_gru", b"{}")


def test_source_has_no_filesystem_write_or_authorizing_shortcut() -> None:
    source = (
        _ROOT
        / "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py"
    ).read_text(encoding="utf-8")
    assert "open(" not in source
    assert "write_bytes" not in source
    assert "execution_authorized\": True" not in source
    assert "scientific_promotion_allowed\": True" not in source
    assert "universal_sota_claim_allowed\": True" not in source
    assert "TO_BE_" not in source
