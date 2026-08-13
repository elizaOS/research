"""Synthetic contracts for the additive compiled PPO-GRU reward bundle."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_compiled_reward_bundle as bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_compiled_runner as compiled_runner,
)
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _synthetic_runtime_identity_bytes() -> bytes:
    runner_descriptor = compiled_runner.matched_v3_ppo_gru_compiled_runner_descriptor()
    identity = {
        "schema_version": (compiled_runner.PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION),
        "classification": "observed_compiled_runtime_unqualified_non_authorizing",
        "bindings": {
            "compiled_runner_descriptor_sha256": (
                compiled_runner.PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
            ),
            "bridge_descriptor_sha256": compiled_runner.BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_sha256": (compiled_runner.BOUND_BRIDGE_IMPLEMENTATION_SHA256),
            "core_configuration_sha256": (compiled_runner.BOUND_CORE_CONFIGURATION_SHA256),
            "core_implementation_sha256": (compiled_runner.BOUND_CORE_IMPLEMENTATION_SHA256),
            "foragax_install_tree_sha256": (compiled_runner.BOUND_FORAGAX_INSTALL_TREE_SHA256),
        },
        "runtime": {
            "jax_version": "0.11.0",
            "jaxlib_version": "0.11.0",
            "default_prng_impl": "threefry2x32",
            "threefry_partitionable": True,
            "jax_enable_x64": False,
            "backend": "synthetic-test-only",
            "foragax_version": "0.55.0",
            "foragax_install_tree_sha256": (compiled_runner.BOUND_FORAGAX_INSTALL_TREE_SHA256),
            "foragax_package_root": "/synthetic/not-opened",
            "runtime_qualified": False,
        },
        "kernel": {
            "chunk_steps": compiled_runner.PPO_GRU_COMPILED_CHUNK_STEPS,
            "constructed": True,
            "full_horizon_executed": False,
            "runtime_qualified": False,
        },
        "claims": runner_descriptor["claims"],
    }
    return _canonical(identity)


@dataclass(frozen=True, slots=True)
class _SyntheticOutcome:
    raw_reward_trace: bytes
    raw_cumulative_score: int
    interactions: int
    rollout_count: int
    optimizer_update_count: int
    total_agent_draw_count: int
    bridge_environment_key_use_count: int
    trace_chain_sha256: str
    runtime_identity_bytes: bytes
    receipt_bytes: bytes
    production_runtime: bool


@lru_cache(maxsize=1)
def _synthetic_contents() -> tuple[_SyntheticOutcome, bytes]:
    trace = bytes(protocol.MATCHED_V3_HORIZON)
    runtime_identity_bytes = _synthetic_runtime_identity_bytes()
    receipt = compiled_runner._receipt_bytes_from_fields(
        environment_seed=17,
        agent_seed=29,
        runtime_identity_bytes=runtime_identity_bytes,
        raw_reward_trace=trace,
        raw_cumulative_score=0,
        trace_chain_sha256="1" * 64,
    )
    outcome = _SyntheticOutcome(
        raw_reward_trace=trace,
        raw_cumulative_score=0,
        interactions=compiled_runner.MATCHED_V3_HORIZON,
        rollout_count=compiled_runner.PPO_GRU_COMPILED_CHUNK_COUNT,
        optimizer_update_count=compiled_runner.PPO_GRU_OPTIMIZER_UPDATES,
        total_agent_draw_count=compiled_runner.PPO_GRU_TOTAL_AGENT_DRAWS,
        bridge_environment_key_use_count=compiled_runner.PPO_GRU_BRIDGE_KEY_USES,
        trace_chain_sha256="1" * 64,
        runtime_identity_bytes=runtime_identity_bytes,
        receipt_bytes=receipt,
        production_runtime=True,
    )
    return outcome, receipt


def _patch_public_capability_validator(
    monkeypatch: pytest.MonkeyPatch,
    outcome: _SyntheticOutcome,
    receipt: bytes,
) -> list[object]:
    calls: list[object] = []

    def validate(value: object) -> bytes:
        calls.append(value)
        if value is not outcome:
            raise compiled_runner.ForagerMatchedV3PPOGRUCompiledRunnerError(
                "synthetic validator received another object"
            )
        return receipt

    monkeypatch.setattr(
        compiled_runner,
        "canonical_ppo_gru_compiled_result_receipt_bytes",
        validate,
    )
    return calls


def _build_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bundle.MatchedV3CompiledRewardBundle, _SyntheticOutcome]:
    outcome, receipt = _synthetic_contents()
    _patch_public_capability_validator(monkeypatch, outcome, receipt)
    built = bundle.build_ppo_gru_compiled_reward_bundle(cast(Any, outcome))
    return built, outcome


def _rehashed_manifest(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = copy.deepcopy(payload)
    body.pop("manifest_body_sha256", None)
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    rewritten = dict(body)
    rewritten["manifest_body_sha256"] = digest
    return _canonical(rewritten), digest


def _resigned_runner_receipt(payload: dict[str, Any]) -> bytes:
    body = copy.deepcopy(payload)
    body.pop("receipt_body_sha256", None)
    rewritten = dict(body)
    rewritten["receipt_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(rewritten)


def test_descriptor_is_hardcoded_source_bound_detached_and_nonauthorizing() -> None:
    raw = bundle.canonical_compiled_reward_bundle_descriptor_bytes()
    descriptor = bundle.parse_compiled_reward_bundle_descriptor(raw)

    assert len(raw) == 2_681
    assert bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256 == (
        "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
    )
    assert hashlib.sha256(raw).hexdigest() == (bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256)
    assert descriptor["compiled_runner"] == {
        "descriptor_schema_version": (
            "alberta.forager_matched_v3.ppo_gru_compiled_runner_descriptor.v1"
        ),
        "descriptor_sha256": ("3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"),
        "result_receipt_schema_version": (
            "alberta.forager_matched_v3.ppo_gru_compiled_result_receipt.v2"
        ),
        "runtime_identity_schema_version": (
            "alberta.forager_matched_v3.ppo_gru_compiled_runtime_identity.v1"
        ),
        "source_path": (
            "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_compiled_runner.py"
        ),
        "source_sha256": ("08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f"),
    }
    assert descriptor["scorer"]["source_sha256"] == (
        "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert descriptor["metric"] == {
        "accumulation": "ordered_exact_integer_sum",
        "horizon": 499_712,
        "raw_reward_values": [-1, 0, 1, 30],
        "schema_version": "alberta.forager_cumulative_reward_metric.v1",
        "sha256": "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd",
    }
    assert set(descriptor["claims"].values()) == {False}
    assert descriptor["conversion"]["runtime_opened_or_workload_executed"] is False
    assert descriptor["conversion"]["canonical_npz_reingested_before_return"] is True

    for section in ("compiled_runner", "scorer"):
        source = _ROOT / descriptor[section]["source_path"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest() == descriptor[section]["source_sha256"]
        )
    assert (
        hashlib.sha256(protocol.canonical_cumulative_reward_metric_bytes()).hexdigest()
        == (descriptor["metric"]["sha256"])
    )

    descriptor["claims"]["authority_granted"] = True
    assert bundle.compiled_reward_bundle_descriptor()["claims"]["authority_granted"] is False
    with pytest.raises(bundle.ForagerMatchedV3CompiledRewardBundleError):
        bundle.parse_compiled_reward_bundle_descriptor(b" " + raw)
    duplicate = b'{"candidate_id":"duplicate",' + raw[1:]
    with pytest.raises(
        bundle.ForagerMatchedV3CompiledRewardBundleError,
        match="duplicate",
    ):
        bundle.parse_compiled_reward_bundle_descriptor(duplicate)


def test_public_capability_conversion_reingests_immediately_without_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, receipt = _synthetic_contents()
    calls = _patch_public_capability_validator(monkeypatch, outcome, receipt)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("a runtime or workload must not be opened")

    monkeypatch.setattr(
        compiled_runner,
        "open_matched_v3_ppo_gru_compiled_runtime",
        forbidden,
    )
    monkeypatch.setattr(
        compiled_runner,
        "run_matched_v3_ppo_gru_compiled",
        forbidden,
    )
    events: list[str] = []
    original_encode = scorer.canonical_reward_npz_bytes
    original_ingest = scorer.ingest_reward_npz_bytes
    original_extract = scorer.extract_canonical_reward_trace

    def encode(raw_trace: bytes) -> bytes:
        events.append("encode")
        return original_encode(raw_trace)

    def ingest(artifact: bytes) -> scorer.MatchedV3ScoreReceipt:
        events.append("ingest")
        return original_ingest(artifact)

    def extract(artifact: bytes) -> bytes:
        events.append("extract")
        return original_extract(artifact)

    monkeypatch.setattr(scorer, "canonical_reward_npz_bytes", encode)
    monkeypatch.setattr(scorer, "ingest_reward_npz_bytes", ingest)
    monkeypatch.setattr(scorer, "extract_canonical_reward_trace", extract)

    built = bundle.build_ppo_gru_compiled_reward_bundle(cast(Any, outcome))

    assert calls == [outcome]
    assert events[:3] == ["encode", "ingest", "extract"]
    assert bundle.validate_compiled_reward_bundle(built) is built
    assert scorer.extract_canonical_reward_trace(built.reward_artifact_bytes) == (
        outcome.raw_reward_trace
    )
    assert len(built.reward_artifact_bytes) == scorer.CANONICAL_NPZ_SIZE_BYTES
    score_receipt = scorer.parse_score_receipt(built.score_receipt_bytes)
    assert score_receipt.cumulative_score == 0
    manifest = built.manifest()
    assert manifest["compiled_runner_receipt"]["sha256"] == hashlib.sha256(receipt).hexdigest()
    assert (
        manifest["runtime_identity"]["sha256"]
        == hashlib.sha256(outcome.runtime_identity_bytes).hexdigest()
    )
    assert (
        manifest["raw_reward_trace"]["bytes_sha256"]
        == hashlib.sha256(outcome.raw_reward_trace).hexdigest()
    )
    assert set(manifest["claims"].values()) == {False}
    assert (
        manifest["compiled_runner_receipt"]["structural_content_independently_attests_execution"]
        is False
    )
    assert (
        manifest["runtime_identity"]["structural_content_independently_attests_execution"] is False
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.candidate_id = "changed"  # type: ignore[misc]


def test_fabricated_exact_compiled_outcome_lacks_the_live_capability() -> None:
    synthetic, receipt = _synthetic_contents()
    forged = compiled_runner.PPOGRUCompiledOutcome(
        raw_reward_trace=synthetic.raw_reward_trace,
        raw_cumulative_score=synthetic.raw_cumulative_score,
        interactions=synthetic.interactions,
        rollout_count=synthetic.rollout_count,
        optimizer_update_count=synthetic.optimizer_update_count,
        total_agent_draw_count=synthetic.total_agent_draw_count,
        bridge_environment_key_use_count=synthetic.bridge_environment_key_use_count,
        trace_chain_sha256=synthetic.trace_chain_sha256,
        runtime_identity_bytes=synthetic.runtime_identity_bytes,
        receipt_bytes=receipt,
        production_runtime=True,
        _capability=compiled_runner._OutcomeCapability(),
        _pid=os.getpid(),
    )

    with pytest.raises(
        bundle.ForagerMatchedV3CompiledRewardBundleError,
        match="live authentic completion capability",
    ):
        bundle.build_ppo_gru_compiled_reward_bundle(forged)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("raw_reward_trace", b"\x01" + bytes(protocol.MATCHED_V3_HORIZON - 1), "trace"),
        ("raw_cumulative_score", 1, "score"),
        ("runtime_identity_bytes", b"{}", "runtime"),
        ("receipt_bytes", b"{}", "receipt bytes"),
        ("rollout_count", True, "exact integer"),
        ("production_runtime", False, "accounting"),
    ),
)
def test_even_a_validator_return_cannot_hide_outcome_content_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    authentic_shape, receipt = _synthetic_contents()
    replacement_fields: Any = {field: replacement}
    changed = dataclasses.replace(authentic_shape, **replacement_fields)
    _patch_public_capability_validator(monkeypatch, changed, receipt)

    with pytest.raises(
        bundle.ForagerMatchedV3CompiledRewardBundleError,
        match=message,
    ):
        bundle.build_ppo_gru_compiled_reward_bundle(cast(Any, changed))


def test_bundle_replay_rejects_each_detached_content_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built, _outcome = _build_synthetic(monkeypatch)

    class MutableCandidateAlias:
        def __init__(self) -> None:
            self.value = "adapted_ppo_gru"

        def __eq__(self, other: object) -> bool:
            return other == self.value

    changed_artifact = bytearray(built.reward_artifact_bytes)
    changed_artifact[100] ^= 1
    changed_score = bytearray(built.score_receipt_bytes)
    changed_score[-2] ^= 1
    tampered = (
        dataclasses.replace(built, runner_receipt_bytes=b"{}"),
        dataclasses.replace(built, runtime_identity_bytes=b"{}"),
        dataclasses.replace(built, reward_artifact_bytes=bytes(changed_artifact)),
        dataclasses.replace(built, score_receipt_bytes=bytes(changed_score)),
        dataclasses.replace(built, manifest_bytes=b"{}"),
        dataclasses.replace(built, manifest_sha256="0" * 64),
        dataclasses.replace(built, candidate_id="adapted_full_rainbow"),
        dataclasses.replace(built, candidate_id=cast(Any, MutableCandidateAlias())),
    )
    for changed in tampered:
        with pytest.raises(bundle.ForagerMatchedV3CompiledRewardBundleError):
            bundle.validate_compiled_reward_bundle(changed)


def test_manifest_parser_is_bounded_canonical_and_exact_type_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built, _outcome = _build_synthetic(monkeypatch)
    parsed = cast(dict[str, Any], json.loads(built.manifest_bytes))
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["claims"].update({"authority_granted": True}),
        lambda value: value["claims"].update({"authority_granted": 0}),
        lambda value: value["runner_accounting"].update({"automatic_resets": False}),
        lambda value: value["bindings"].update({"compiled_runner_source_sha256": "0" * 64}),
        lambda value: value["runtime_identity"].update({"size_bytes": 0}),
        lambda value: value["raw_reward_trace"].update({"length": 1}),
        lambda value: value["compiled_runner_receipt"].update({"unexpected": False}),
    )
    for mutation in mutations:
        changed = copy.deepcopy(parsed)
        mutation(changed)
        raw, digest = _rehashed_manifest(changed)
        with pytest.raises(bundle.ForagerMatchedV3CompiledRewardBundleError):
            bundle.parse_compiled_reward_bundle_manifest(
                raw,
                expected_manifest_sha256=digest,
            )

    malformed = (
        b" " + built.manifest_bytes,
        b'{"claims":{},' + built.manifest_bytes[1:],
        b'{"value":NaN}',
        b'{"value":1.5}',
        b'{"value":' + b"9" * 20 + b"}",
        b'{"value":' + b"[" * (bundle._MAX_JSON_DEPTH + 1) + b"0",
        b'{"value":[' + b"0," * bundle._MAX_JSON_NODES + b"0]}",
        b"{" + b" " * bundle._MAX_MANIFEST_BYTES,
    )
    for raw in malformed:
        with pytest.raises(bundle.ForagerMatchedV3CompiledRewardBundleError):
            bundle.parse_compiled_reward_bundle_manifest(
                raw,
                expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_runner_receipt_bool_alias_and_parser_error_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _outcome, receipt = _synthetic_contents()
    parsed = json.loads(receipt)
    parsed["completion"]["exact_horizon_complete"] = 1
    aliased = _resigned_runner_receipt(parsed)
    with pytest.raises(
        bundle.ForagerMatchedV3CompiledRewardBundleError,
        match="exact-type contract alias",
    ):
        bundle._parse_runner_receipt(aliased)

    def reject(raw: bytes, *, expected_receipt_sha256: str) -> dict[str, Any]:
        del raw, expected_receipt_sha256
        raise compiled_runner.ForagerMatchedV3PPOGRUCompiledRunnerError(
            "synthetic structural rejection"
        )

    monkeypatch.setattr(
        compiled_runner,
        "parse_ppo_gru_compiled_result_receipt",
        reject,
    )
    with pytest.raises(
        bundle.ForagerMatchedV3CompiledRewardBundleError,
        match="frozen structural parser",
    ):
        bundle._parse_runner_receipt(receipt)


def test_source_has_no_writer_execution_or_v1_authority_shortcut() -> None:
    path = _ROOT / "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_bundle.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_calls = {
        "open_matched_v3_ppo_gru_compiled_runtime",
        "run_matched_v3_ppo_gru_compiled",
        "write",
        "write_bytes",
        "write_text",
        "mkdir",
        "replace",
        "rename",
        "unlink",
    }
    assert called_attributes.isdisjoint(forbidden_calls)
    assert "O_WRONLY" not in source
    assert "O_RDWR" not in source
    assert "O_CREAT" not in source
    assert "O_TRUNC" not in source
    assert "import jax" not in source
    assert "import numpy" not in source
    assert "forager_matched_v3_adapter_reward_bundle" not in source
    assert "forager_matched_v3_adapter_reward_publication" not in source
    assert "forager_matched_v3_qualification_plan" not in source
    assert '"execution_authorized": True' not in source
    assert '"execution_ready": True' not in source
    assert '"runtime_qualified": True' not in source
    assert '"scientific_promotion_allowed": True' not in source
    assert '"universal_sota_claim_allowed": True' not in source
    assert "TO_BE_" not in source
