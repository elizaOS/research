from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_final_analysis as final_analysis
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from alberta_framework.benchmarks import forager_matched_statistics as statistics
from tests import test_forager_matched_executor as executor_fixtures
from tests import test_forager_matched_open_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _raw_bytes(label: str, candidate_id: str, seed: int) -> bytes:
    return f"opaque:{label}:{candidate_id}:{seed}".encode("ascii")


def _qualification_manifest(
    open_protocol_sha256: str,
    qualification_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "structurally_qualified_external_trust_resolution_required",
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "authority": {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
        },
        "reward_blind_boundary": {
            "qualification_seed": qualification.PUBLIC_QUALIFICATION_SEED,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "tuning_seeds_used": [],
            "evaluation_seeds_used": [],
            "environment_resets": len(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS),
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
        },
        "runtime_qualification": {},
        "qualification_probe": {},
        "resource_accounting_semantics": {},
        "executor_qualification_roots": {},
        "frozen_executor_qualification_artifacts": json.loads(
            executor.canonical_json_bytes(qualification_artifacts)
        ),
        "candidate_order": list(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS),
        "sources": {},
        "candidates": {},
        "open_protocol_sha256": open_protocol_sha256,
    }


def _bindings(
    request: executor.VerificationRequest,
    label: str,
) -> evidence.AuthenticatedEvidenceBindings:
    return evidence.AuthenticatedEvidenceBindings(
        stage=request.stage,
        protocol_sha256=request.protocol_sha256,
        score_evidence_sha256=request.score_evidence_sha256,
        source_manifest_sha256=request.source_manifest_sha256,
        executor_manifest_sha256=request.executor_manifest_sha256,
        execution_closure_sha256=request.execution_closure_sha256,
        trust_anchor_identity=request.trust_anchor_identity,
        verification_subject_sha256=request.verification_subject_sha256,
        verification_receipt_sha256=_sha(f"verification-receipt:{label}"),
    )


def _plan(
    frozen: protocol.ForagerMatchedProtocol,
    candidate_ids: tuple[str, ...],
    base: executor.MatchedExecutionPlan,
) -> executor.MatchedExecutionPlan:
    prepared = tuple(
        replace(
            base.candidates[0],
            candidate=frozen.candidate_index[candidate_id],
            capability_receipt_sha256=(
                frozen.candidate_index[
                    candidate_id
                ].runtime_binding.capability_qualification_receipt_sha256
            ),
        )
        for candidate_id in candidate_ids
    )
    base_source = cast(
        dict[str, Any],
        cast(list[Any], base.source_manifest["candidates"])[0],
    )
    source_manifest = {
        "schema_version": executor.MATCHED_SOURCE_MANIFEST_SCHEMA_VERSION,
        "stage": frozen.stage,
        "protocol_sha256": frozen.protocol_sha256,
        "candidates": [
            {
                **json.loads(executor.canonical_json_bytes(base_source)),
                "candidate_id": candidate_id,
                "capability_descriptor_sha256": (
                    protocol.candidate_capability_descriptor_sha256(
                        frozen.candidate_index[candidate_id]
                    )
                ),
                "capability_qualification_receipt_sha256": (
                    frozen.candidate_index[
                        candidate_id
                    ].runtime_binding.capability_qualification_receipt_sha256
                ),
                "source": frozen.candidate_index[candidate_id].source.to_dict(),
                "configuration": (frozen.candidate_index[candidate_id].configuration.to_dict()),
                "entrypoint_family": (frozen.candidate_index[candidate_id].entrypoint_family),
            }
            for candidate_id in candidate_ids
        ],
    }
    executor_manifest = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(base.executor_manifest)),
    )
    executor_manifest.update(
        {
            "protocol_sha256": frozen.protocol_sha256,
            "runtime": frozen.runtime.to_dict(),
            "sandbox": frozen.runtime.sandbox.to_dict(),
        }
    )
    payload = base.to_dict()
    payload.update(
        {
            "stage": frozen.stage,
            "protocol_sha256": frozen.protocol_sha256,
            "active_seeds": list(frozen.active_seeds),
            "horizon": frozen.horizon,
            "candidate_order": list(candidate_ids),
            "source_manifest": source_manifest,
            "source_manifest_sha256": campaign._canonical_sha256(source_manifest),
            "executor_manifest": executor_manifest,
            "executor_manifest_sha256": campaign._canonical_sha256(executor_manifest),
            "candidate_command_templates": [
                {
                    "candidate_id": candidate_id,
                    "argv": ["synthetic", candidate_id, "<ACTIVE_SEED>"],
                }
                for candidate_id in candidate_ids
            ],
        }
    )
    return executor.MatchedExecutionPlan(
        protocol=frozen,
        candidates=prepared,
        source_manifest=MappingProxyType(source_manifest),
        executor_manifest=MappingProxyType(executor_manifest),
        payload=payload,
        candidate_index=MappingProxyType({item.candidate.candidate_id: item for item in prepared}),
        cpu_qualification_root=base.cpu_qualification_root,
        rng_parity_qualification_root=base.rng_parity_qualification_root,
    )


def _panel(
    tmp_path: Path,
    frozen: protocol.ForagerMatchedProtocol,
    candidate_ids: tuple[str, ...],
    base: executor.MatchedExecutionPlan,
    *,
    label: str,
) -> tuple[
    executor.MatchedExecutionPlan,
    executor.LiveRuntimeIdentity,
    dict[str, tuple[executor.SeedExecutionArtifacts, ...]],
    executor.MatchedExecutionReceiptIndex,
    evidence.MatchedScoreEvidence,
    executor.VerificationRequest,
    evidence.AuthenticatedEvidenceBindings,
]:
    plan = _plan(frozen, candidate_ids, base)
    live = executor_fixtures._runtime(tmp_path / f"runtime-{label}", plan)
    artifacts: dict[str, tuple[executor.SeedExecutionArtifacts, ...]] = {}
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate = plan.candidate_index[candidate_id]
        records: list[executor.SeedExecutionArtifacts] = []
        for seed_index, seed in enumerate(frozen.active_seeds):
            raw = _raw_bytes(label, candidate_id, seed)
            records.append(
                executor._artifact_mappings(
                    plan=plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=hashlib.sha256(raw).hexdigest(),
                    raw_archive_size=len(raw),
                    live_runtime=live,
                    scorer_record={
                        "fov_last_10pct_ema_auc": float(candidate_index + 1) + seed_index / 100.0,
                        "npz_sha256": _sha(f"npz:{label}:{candidate_id}:{seed}"),
                        "npz_size_bytes": 4096 + seed_index,
                        "reward_trace_sha256": _sha(f"trace:{label}:{candidate_id}:{seed}"),
                        "reward_dtype": "<f4",
                        "reward_shape": [frozen.horizon],
                    },
                )
            )
        artifacts[candidate_id] = tuple(records)
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    scores = executor.build_score_evidence(plan, artifacts)
    request = executor.build_verification_request(plan, scores)
    return (
        plan,
        live,
        artifacts,
        receipt_index,
        scores,
        request,
        _bindings(request, label),
    )


@dataclass(frozen=True, slots=True)
class _Fixture:
    qualification_root: Path
    seal_root: Path
    evaluation_root: Path
    open_bindings: evidence.AuthenticatedEvidenceBindings
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings
    completed: campaign.CompletedCampaignBundle
    evaluation_context: campaign._CampaignContext
    sealed_inputs: sealed_campaign._SealedInputs


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    install_completed_loader: bool = True,
) -> _Fixture:
    base_root = tmp_path / "base"
    base_root.mkdir()
    base = executor_fixtures._plan(base_root)
    original_open = protocol_fixtures._build()
    analysis_plan = replace(
        original_open.analysis_plan,
        primary=replace(original_open.analysis_plan.primary, resamples=31),
        secondary=replace(
            original_open.analysis_plan.secondary,
            monte_carlo_resamples=64,
        ),
    )
    open_protocol = replace(original_open, analysis_plan=analysis_plan)
    tuning_ids = tuple(
        candidate_id
        for group in open_protocol.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    (
        open_plan,
        open_live,
        open_artifacts,
        open_receipts,
        open_scores,
        open_request,
        open_bindings,
    ) = _panel(
        tmp_path,
        open_protocol,
        tuning_ids,
        base,
        label="open",
    )
    open_root = tmp_path / "open-campaign"
    open_root.mkdir()
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    open_rebuilt = campaign._RebuiltInputs(
        bundle=cast(Any, SimpleNamespace(manifest={})),
        protocol=open_protocol,
        plan=open_plan,
        candidate_ids=tuning_ids,
        assets={},
        schedule={},
    )
    open_context = campaign._CampaignContext(open_root, open_rebuilt, open_live)
    open_summary = campaign._completion_summary(
        open_context,
        open_receipts,
        open_scores,
        open_request,
    )
    open_final_raw = {
        "execution-receipt-index.json": open_receipts.canonical_bytes,
        "score-evidence.json": open_scores.canonical_bytes,
        "verification-request.json": open_request.canonical_bytes,
        "completion-summary.json": seal.canonical_json_bytes(open_summary),
    }
    open_completed = campaign.CompletedCampaignBundle(
        output_root=open_root,
        protocol=open_protocol,
        plan=open_plan,
        live_runtime=open_live,
        candidate_ids=tuning_ids,
        active_seeds=open_protocol.active_seeds,
        schedule={},
        seed_artifacts=open_artifacts,
        execution_receipt_index=open_receipts,
        score_evidence=open_scores,
        verification_request=open_request,
        completion_summary=open_summary,
        final_file_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in open_final_raw.items()
        },
    )
    monkeypatch.setattr(
        campaign,
        "load_completed_open_tuning_campaign",
        lambda *_args, **_kwargs: open_completed,
    )
    seal_root = tmp_path / "seal"
    seal_content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        open_root,
        seal_root,
        resolver=lambda _request: open_bindings,
        expected_trust_anchor_identity=open_bindings.trust_anchor_identity,
        expected_verification_subject_sha256=(open_bindings.verification_subject_sha256),
    )

    transition = protocol.validate_sealed_protocol_transition(
        seal_content.open_protocol,
        seal_content.sealed_protocol,
        seal_content.selection_result,
        seal_content.selection_result.selection_result_sha256,
    )
    evaluation_ids = transition.evaluation_candidate_ids
    (
        evaluation_plan,
        evaluation_live,
        evaluation_artifacts,
        evaluation_receipts,
        evaluation_scores,
        evaluation_request,
        evaluation_bindings,
    ) = _panel(
        tmp_path,
        seal_content.sealed_protocol,
        evaluation_ids,
        base,
        label="evaluation",
    )
    evaluation_schedule = evaluation.build_sealed_evaluation_schedule(
        seal_content.sealed_protocol,
        transition,
    )
    qualification_bundle = cast(
        Any,
        SimpleNamespace(
            output_root=qualification_root,
            manifest=_qualification_manifest(
                seal_content.open_protocol.protocol_sha256,
                cast(
                    Mapping[str, Any],
                    base.executor_manifest["qualification_artifacts"],
                ),
            ),
            cpu_qualification_root=base.cpu_qualification_root,
            rng_parity_qualification_root=base.rng_parity_qualification_root,
        ),
    )
    evaluation_rebuilt = campaign._RebuiltInputs(
        bundle=qualification_bundle,
        protocol=seal_content.sealed_protocol,
        plan=evaluation_plan,
        candidate_ids=evaluation_ids,
        assets={},
        schedule=evaluation_schedule,
    )
    sealed_inputs = sealed_campaign._SealedInputs(
        rebuilt=evaluation_rebuilt,
        seal_content=seal_content,
        transition=transition,
        seal_manifest_payload_sha256=cast(
            str,
            seal_content.manifest["payload_sha256"],
        ),
        open_verification_subject_sha256=(
            seal_content.open_verification_request.verification_subject_sha256
        ),
    )
    evaluation_root = tmp_path / "evaluation"
    evaluation_context = campaign._CampaignContext(
        evaluation_root,
        evaluation_rebuilt,
        evaluation_live,
    )
    prospective = sealed_campaign._prospective_output(
        sealed_inputs,
        evaluation_root,
    )
    sealed_campaign._publish_initial_root(
        sealed_inputs,
        evaluation_live,
        evaluation_root,
        prospective,
    )
    evaluation_summary = sealed_campaign._completion_summary(
        sealed_inputs,
        evaluation_context,
        evaluation_receipts,
        evaluation_scores,
        evaluation_request,
    )
    final_values = {
        "execution-receipt-index.json": evaluation_receipts.to_dict(),
        "score-evidence.json": evaluation_scores.to_dict(),
        "verification-request.json": evaluation_request.to_dict(),
        "completion-summary.json": evaluation_summary,
    }
    final_hashes = {
        name: campaign._publish_json_pair(evaluation_root / name, value)
        for name, value in final_values.items()
    }
    completed = campaign.CompletedCampaignBundle(
        output_root=evaluation_root,
        protocol=seal_content.sealed_protocol,
        plan=evaluation_plan,
        live_runtime=evaluation_live,
        candidate_ids=evaluation_ids,
        active_seeds=seal_content.sealed_protocol.active_seeds,
        schedule=evaluation_schedule,
        seed_artifacts=evaluation_artifacts,
        execution_receipt_index=evaluation_receipts,
        score_evidence=evaluation_scores,
        verification_request=evaluation_request,
        completion_summary=evaluation_summary,
        final_file_sha256=final_hashes,
    )
    if install_completed_loader:
        monkeypatch.setattr(
            sealed_campaign,
            "load_completed_sealed_evaluation_campaign_content",
            lambda *_args, **_kwargs: completed,
        )
    return _Fixture(
        qualification_root=qualification_root,
        seal_root=seal_root,
        evaluation_root=evaluation_root,
        open_bindings=open_bindings,
        evaluation_bindings=evaluation_bindings,
        completed=completed,
        evaluation_context=evaluation_context,
        sealed_inputs=sealed_inputs,
    )


def _materialize_completed_cells(value: _Fixture) -> None:
    context = value.evaluation_context
    for candidate_id, artifacts in value.completed.seed_artifacts.items():
        for artifact in artifacts:
            run_cell, completion_path = campaign._cell_paths(
                context.root,
                candidate_id,
                artifact.seed,
            )
            attempt = run_cell / "attempt-000001"
            attempt.mkdir(parents=True)
            raw = _raw_bytes("evaluation", candidate_id, artifact.seed)
            (attempt / "raw-output.tar").write_bytes(raw)
            binding = campaign._raw_binding(
                context,
                candidate_id,
                artifact.seed,
                attempt.name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
            )
            binding_sha256 = campaign._publish_json_pair(
                attempt / "raw-binding.json",
                binding,
            )
            bundle_sha256 = campaign._publish_json_pair(
                attempt / "bundle.json",
                artifact.to_dict(),
            )
            completion_path.parent.mkdir(parents=True, exist_ok=True)
            campaign._publish_json_pair(
                completion_path,
                campaign._completion_pointer(
                    context,
                    candidate_id,
                    artifact.seed,
                    attempt,
                    binding_sha256,
                    bundle_sha256,
                ),
            )


def _creation_kwargs(value: _Fixture) -> dict[str, Any]:
    seal_content = seal.load_forager_matched_seal_bundle_content(value.seal_root)
    return {
        "open_resolver": cast(Any, lambda _request: value.open_bindings),
        "evaluation_resolver": cast(
            Any,
            lambda _request: value.evaluation_bindings,
        ),
        "expected_open_trust_anchor_identity": (value.open_bindings.trust_anchor_identity),
        "expected_open_verification_subject_sha256": (
            value.open_bindings.verification_subject_sha256
        ),
        "expected_evaluation_trust_anchor_identity": (
            value.evaluation_bindings.trust_anchor_identity
        ),
        "expected_evaluation_verification_subject_sha256": (
            value.evaluation_bindings.verification_subject_sha256
        ),
        "expected_seal_manifest_payload_sha256": seal_content.manifest["payload_sha256"],
    }


def _expected_pair_inventory(names: tuple[str, ...]) -> set[str]:
    return {*(name for name in names), *(f"{name}.sha256" for name in names)}


def _rewrite_json_pair(path: Path, payload: Mapping[str, Any]) -> None:
    raw = seal.canonical_json_bytes(payload)
    sidecar = path.with_name(f"{path.name}.sha256")
    path.chmod(0o600)
    sidecar.chmod(0o600)
    path.write_bytes(raw)
    sidecar.write_bytes(f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"))


def _rewrite_copied_artifact_and_manifest(
    output: Path,
    subtree: str,
    name: str,
    payload: Mapping[str, Any],
) -> None:
    artifact_path = output / subtree / name
    _rewrite_json_pair(artifact_path, payload)
    raw = artifact_path.read_bytes()
    manifest_path = output / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    artifact_sha256 = cast(dict[str, Any], manifest["artifact_sha256"])
    artifact_sha256[f"{subtree}/{name}"] = hashlib.sha256(raw).hexdigest()
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)


def test_publish_orders_authentication_before_analysis_and_replays_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    events: list[str] = []
    original_analyze = statistics.analyze_matched_scores

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open-resolver")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation-resolver")
        return fixture.evaluation_bindings

    def analyze(
        contract: statistics.MatchedComparisonContract,
    ) -> statistics.MatchedComparisonResult:
        events.append("analysis")
        return original_analyze(contract)

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)
    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = evaluation_resolver
    output = tmp_path / "published" / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **kwargs,
    )

    assert events[:3] == ["open-resolver", "evaluation-resolver", "analysis"]
    assert events.count("open-resolver") == 1
    assert events.count("evaluation-resolver") == 1
    assert len(content.evaluation_score_evidence.candidate_scores) == 6
    assert all(
        len(candidate.records) == 30
        for candidate in content.evaluation_score_evidence.candidate_scores
    )
    expected_descriptive = ("exact_ppo", "search_oracle")
    assert (
        tuple(item.candidate_id for item in content.contract.fixed_descriptive_diagnostics)
        == expected_descriptive
    )
    assert (
        tuple(item.candidate_id for item in content.result.fixed_descriptive_exclusions)
        == expected_descriptive
    )
    inferential = {method.method_id for method in content.contract.methods}
    assert inferential.isdisjoint(expected_descriptive)

    assert set(path.name for path in output.iterdir()) == {
        "seal",
        "evaluation",
        "analysis",
        "manifest.json",
        "manifest.json.sha256",
    }
    assert set(path.name for path in (output / "seal").iterdir()) == (
        _expected_pair_inventory(final_analysis._SEAL_ARTIFACTS)
    )
    assert set(path.name for path in (output / "evaluation").iterdir()) == (
        _expected_pair_inventory(final_analysis._EVALUATION_ARTIFACTS)
    )
    assert set(path.name for path in (output / "analysis").iterdir()) == (
        _expected_pair_inventory(final_analysis._ANALYSIS_ARTIFACTS)
    )

    claim = cast(Mapping[str, Any], content.manifest["claim_boundary"])
    scope = cast(Mapping[str, Any], content.manifest["self_contained_scope"])
    assert claim["scope"] == "selected_six_panel_only"
    assert claim["promotion_authorized"] is False
    assert claim["sota_claim_authorized"] is False
    assert claim["secondary_sign_flip_interpretation"] == ("nonconfirmatory_sensitivity_only")
    assert claim["secondary_sign_exchangeability_assumption"] == ("unstated_in_frozen_protocol")
    assert claim["secondary_inferential_claim_authorized"] is False
    assert claim["secondary_reject_flags_are_claim_gates"] is False
    assert tuple(claim["selected_six_candidate_ids"]) == tuple(
        candidate.candidate_id for candidate in content.evaluation_score_evidence.candidate_scores
    )
    assert len(claim["selection_winner_candidate_ids"]) == 4
    assert tuple(claim["fixed_descriptive_candidate_ids"]) == expected_descriptive
    assert content.manifest["promotion_authorized"] is False
    assert content.manifest["sota_claim_authorized"] is False
    assert scope["scope"] == "scalar_statistics_and_digest_closure"
    assert scope["raw_execution_archives_copied"] is False
    assert scope["reward_trace_payloads_copied"] is False
    assert scope["workload_or_score_recomputation_requires_original_qualified_roots"] is True

    fixture.qualification_root.rename(tmp_path / "qualification-moved")
    fixture.seal_root.rename(tmp_path / "seal-moved")
    fixture.evaluation_root.rename(tmp_path / "evaluation-moved")
    offline = final_analysis.load_final_analysis_content(output)
    assert offline.contract == content.contract
    assert offline.result == content.result


def test_pin_failures_and_resolver_failures_write_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "never-created" / "final-analysis"
    calls: list[str] = []
    kwargs = _creation_kwargs(fixture)

    def should_not_resolve_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def should_not_resolve_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("evaluation")
        return fixture.evaluation_bindings

    kwargs.update(
        {
            "open_resolver": should_not_resolve_open,
            "evaluation_resolver": should_not_resolve_evaluation,
            "expected_open_verification_subject_sha256": _sha("wrong-open-subject"),
        }
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="caller-pinned verification subject",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **kwargs,
        )
    assert calls == []
    assert not output.parent.exists()

    kwargs = _creation_kwargs(fixture)

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def fail_evaluation(_request: executor.VerificationRequest) -> Any:
        calls.append("evaluation")
        raise RuntimeError("authority unavailable")

    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = fail_evaluation
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="evaluation trust resolution failed",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **kwargs,
        )
    assert calls == ["open", "evaluation"]
    assert not output.parent.exists()


def test_bindings_cache_parser_rejects_schema_drift() -> None:
    payload = {
        "schema_version": "alberta.authenticated_evidence_bindings.v999",
        "stage": "sealed_evaluation",
        "protocol_sha256": _sha("protocol"),
        "score_evidence_sha256": _sha("scores"),
        "source_manifest_sha256": _sha("sources"),
        "executor_manifest_sha256": _sha("executor"),
        "execution_closure_sha256": _sha("closure"),
        "trust_anchor_identity": "test-anchor",
        "verification_subject_sha256": _sha("subject"),
        "verification_receipt_sha256": _sha("receipt"),
    }
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="schema/stage drifted",
    ):
        final_analysis._parse_bindings(
            executor.canonical_json_bytes(payload),
            expected_stage="sealed_evaluation",
        )


def test_content_loading_is_resolver_free_and_fresh_auth_checks_both_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("content loading must not invoke a resolver")

    with monkeypatch.context() as guarded:
        guarded.setattr(
            seal,
            "authenticate_forager_matched_seal_bundle",
            unexpected,
        )
        guarded.setattr(executor, "resolve_authenticated_bindings", unexpected)
        replayed = final_analysis.load_final_analysis_content(output)
    assert replayed.result == content.result
    authority = cast(Mapping[str, Any], replayed.manifest["authority_boundary"])
    assert authority["persisted_bindings_are_cache_only"] is True
    assert authority["content_loader_authenticates_nothing"] is True

    calls: list[str] = []

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("evaluation")
        return fixture.evaluation_bindings

    auth_kwargs = {
        **_creation_kwargs(fixture),
        "open_resolver": open_resolver,
        "evaluation_resolver": evaluation_resolver,
        "expected_analysis_manifest_payload_sha256": content.manifest["payload_sha256"],
    }
    authenticated = final_analysis.authenticate_final_analysis_content(
        content,
        **auth_kwargs,
    )
    assert calls == ["open", "evaluation"]
    assert authenticated.open_bindings == fixture.open_bindings
    assert authenticated.evaluation_bindings == fixture.evaluation_bindings

    changed_evaluation = replace(
        fixture.evaluation_bindings,
        verification_receipt_sha256=_sha("different-evaluation-receipt"),
    )
    auth_kwargs["evaluation_resolver"] = lambda _request: changed_evaluation
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="differs from persisted cache",
    ):
        final_analysis.authenticate_final_analysis_content(output, **auth_kwargs)


def test_loader_rejects_false_promotion_and_partial_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )

    false_promotion = tmp_path / "false-promotion"
    shutil.copytree(output, false_promotion)
    manifest_path = false_promotion / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    claim = cast(dict[str, Any], manifest["claim_boundary"])
    claim["promotion_authorized"] = True
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="manifest differs from exact replay",
    ):
        final_analysis.load_final_analysis_content(false_promotion)

    false_secondary_inference = tmp_path / "false-secondary-inference"
    shutil.copytree(output, false_secondary_inference)
    manifest_path = false_secondary_inference / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    claim = cast(dict[str, Any], manifest["claim_boundary"])
    claim["secondary_inferential_claim_authorized"] = True
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="manifest differs from exact replay",
    ):
        final_analysis.load_final_analysis_content(false_secondary_inference)

    partial = tmp_path / "partial"
    shutil.copytree(output, partial)
    (partial / "analysis" / "statistics-result.json.sha256").unlink()
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="inventory differs",
    ):
        final_analysis.load_final_analysis_content(partial)


def test_post_publish_failure_is_uncertain_and_preserves_published_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"

    real_load = final_analysis._load_from_open_root
    replay_calls = 0

    def fail_post_publish(
        opened: seal._OpenDirectory,
    ) -> final_analysis.ContentVerifiedFinalAnalysisBundle:
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 2:
            raise RuntimeError("injected final replay failure")
        return real_load(opened)

    with monkeypatch.context() as injected:
        injected.setattr(
            final_analysis,
            "_load_from_open_root",
            fail_post_publish,
        )
        with pytest.raises(
            final_analysis.PublishedFinalAnalysisUncertainError,
            match="post-publication content or inode replay failed",
        ) as caught:
            final_analysis.create_forager_matched_final_analysis_bundle(
                fixture.qualification_root,
                fixture.seal_root,
                fixture.evaluation_root,
                output,
                **_creation_kwargs(fixture),
            )
    assert caught.value.destination == output
    assert output.is_dir()
    assert final_analysis.load_final_analysis_content(output).output_root == output


def test_concurrent_destination_is_preserved_and_owned_staging_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    real_publish = seal._publish_verified_no_replace

    def concurrent_publish(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        destination.mkdir()
        (destination / "owner").write_text("concurrent", encoding="utf-8")
        real_publish(
            parent,
            staging,
            source_name,
            destination_name,
            destination,
        )

    monkeypatch.setattr(seal, "_publish_verified_no_replace", concurrent_publish)
    with pytest.raises(ValueError, match="created concurrently"):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert (output / "owner").read_text(encoding="utf-8") == "concurrent"
    assert not tuple(output.parent.glob(".seal-partial-*"))


def test_each_resolver_is_followed_by_full_input_replay_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    completion_path = fixture.evaluation_root / "completion-summary.json"
    original = cast(dict[str, Any], json.loads(completion_path.read_bytes()))
    mutated = dict(original)
    mutated["status"] = "mutated-during-resolver"
    events: list[str] = []

    def analyze(_contract: statistics.MatchedComparisonContract) -> Any:
        events.append("analysis")
        raise AssertionError("statistics must not run after input mutation")

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)

    def mutate_during_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open")
        _rewrite_json_pair(completion_path, mutated)
        return fixture.open_bindings

    def record_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation")
        return fixture.evaluation_bindings

    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = mutate_during_open
    kwargs["evaluation_resolver"] = record_evaluation
    first_output = tmp_path / "first-output" / "analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="snapshot differs from replayed completion",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            first_output,
            **kwargs,
        )
    assert events == ["open"]
    assert not first_output.parent.exists()

    _rewrite_json_pair(completion_path, original)
    events.clear()

    def stable_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open")
        return fixture.open_bindings

    def mutate_during_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation")
        _rewrite_json_pair(completion_path, mutated)
        return fixture.evaluation_bindings

    kwargs["open_resolver"] = stable_open
    kwargs["evaluation_resolver"] = mutate_during_evaluation
    second_output = tmp_path / "second-output" / "analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="snapshot differs from replayed completion",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            second_output,
            **kwargs,
        )
    assert events == ["open", "evaluation"]
    assert not second_output.parent.exists()


def test_publication_parent_aba_is_rejected_without_touching_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_path = tmp_path / "publication"
    moved_parent = tmp_path / "publication-moved"
    output = parent_path / "final-analysis"
    real_sync = final_analysis._sync_staged_tree

    def sync_then_substitute_parent(staging: seal._OpenDirectory) -> None:
        real_sync(staging)
        parent_path.rename(moved_parent)
        parent_path.mkdir()
        (parent_path / "replacement-owner").write_text(
            "must-survive",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        final_analysis,
        "_sync_staged_tree",
        sync_then_substitute_parent,
    )
    with pytest.raises(ValueError, match="output parent"):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert not output.exists()
    assert (parent_path / "replacement-owner").read_text(encoding="utf-8") == ("must-survive")
    assert not tuple(moved_parent.glob(".seal-partial-*"))


def test_evaluation_resolver_is_last_external_callback_at_real_loader_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        install_completed_loader=False,
    )
    _materialize_completed_cells(fixture)
    events: list[str] = []
    evaluation_resolved = False

    def caller_runner(_command: Sequence[str]) -> executor.ProcessResult:
        events.append("caller-runner")
        raise AssertionError("the context stub must not execute the runner")

    def context_boundary(
        qualification_root: Path,
        seal_root: Path,
        output_root: Path,
        *,
        runtime: str | Path,
        runner: executor.ProcessRunner | None,
    ) -> sealed_campaign._SealedContext:
        del qualification_root, seal_root, output_root, runtime
        events.append("post-evaluation-replay" if evaluation_resolved else "replay")
        if evaluation_resolved and runner is caller_runner:
            raise AssertionError("caller runner reached after evaluation resolver")
        return sealed_campaign._SealedContext(
            engine=fixture.evaluation_context,
            inputs=fixture.sealed_inputs,
        )

    monkeypatch.setattr(sealed_campaign, "_load_context", context_boundary)

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open-resolver")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal evaluation_resolved
        events.append("evaluation-resolver")
        evaluation_resolved = True
        return fixture.evaluation_bindings

    real_analyze = statistics.analyze_matched_scores
    real_identity = final_analysis._analysis_runtime_source_identity

    def identity() -> dict[str, Any]:
        events.append("analysis-identity")
        return real_identity()

    def analyze(
        contract: statistics.MatchedComparisonContract,
    ) -> statistics.MatchedComparisonResult:
        events.append("analysis")
        return real_analyze(contract)

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)
    monkeypatch.setattr(final_analysis, "_analysis_runtime_source_identity", identity)
    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = evaluation_resolver
    kwargs["runner"] = caller_runner
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        tmp_path / "final-analysis",
        **kwargs,
    )
    assert events[:7] == [
        "replay",
        "open-resolver",
        "replay",
        "evaluation-resolver",
        "post-evaluation-replay",
        "analysis-identity",
        "analysis",
    ]
    assert "caller-runner" not in events


def test_post_publish_byte_identical_destination_aba_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    moved = tmp_path / "published-original-inode"
    real_publish = seal._publish_verified_no_replace

    def publish_then_swap(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        real_publish(
            parent,
            staging,
            source_name,
            destination_name,
            destination,
        )
        destination.rename(moved)
        shutil.copytree(moved, destination)

    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish_then_swap)
    with pytest.raises(final_analysis.PublishedFinalAnalysisUncertainError):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert output.is_dir()
    assert moved.is_dir()
    assert output.stat().st_ino != moved.stat().st_ino


@pytest.mark.parametrize("failure_ordinal", [2, 3])
def test_partial_child_open_failure_closes_every_prior_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_ordinal: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )
    real_open_child = final_analysis._open_child
    opened_descriptors: list[int] = []
    calls = 0

    def fail_partial_open(
        parent: seal._OpenDirectory,
        name: str,
        label: str,
    ) -> seal._OpenDirectory:
        nonlocal calls
        calls += 1
        if calls == failure_ordinal:
            raise final_analysis.ForagerMatchedFinalAnalysisError("injected child open failure")
        opened = real_open_child(parent, name, label)
        opened_descriptors.append(opened.descriptor)
        return opened

    monkeypatch.setattr(final_analysis, "_open_child", fail_partial_open)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="injected child open failure",
    ):
        final_analysis.load_final_analysis_content(output)
    leaked: list[int] = []
    for descriptor in opened_descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        leaked.append(descriptor)
        os.close(descriptor)
    assert leaked == []


def test_analysis_runtime_source_artifact_binds_executing_code_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )
    assert "analysis-runtime-source.json" in final_analysis._ANALYSIS_ARTIFACTS
    artifact = cast(
        dict[str, Any],
        json.loads((output / "analysis" / "analysis-runtime-source.json").read_bytes()),
    )
    sources = cast(dict[str, Any], artifact["sources"])
    finalizer_raw = Path(final_analysis.__file__).resolve().read_bytes()
    statistics_raw = Path(statistics.__file__).resolve().read_bytes()
    assert sources["finalizer"] == {
        "path": "alberta_framework/benchmarks/forager_matched_final_analysis.py",
        "sha256": hashlib.sha256(finalizer_raw).hexdigest(),
        "size_bytes": len(finalizer_raw),
    }
    assert sources["statistics"] == {
        "path": "alberta_framework/benchmarks/forager_matched_statistics.py",
        "sha256": hashlib.sha256(statistics_raw).hexdigest(),
        "size_bytes": len(statistics_raw),
    }
    runtime = cast(dict[str, Any], artifact["runtime"])
    python = cast(dict[str, Any], runtime["python"])
    numpy = cast(dict[str, Any], runtime["numpy"])
    assert python["implementation"] == sys.implementation.name
    assert python["version"] == list(sys.version_info[:3])
    assert python["cache_tag"] == sys.implementation.cache_tag
    assert numpy["version"] == np.__version__
    assert (
        cast(Mapping[str, Any], content.manifest["analysis_execution"])[
            "runtime_source_payload_sha256"
        ]
        == artifact["payload_sha256"]
    )

    baseline = final_analysis._analysis_source_records()

    def drifted_sources() -> dict[str, dict[str, Any]]:
        drifted = cast(
            dict[str, dict[str, Any]],
            json.loads(json.dumps(baseline)),
        )
        drifted["finalizer"]["sha256"] = _sha("offline-finalizer-drift")
        return drifted

    with monkeypatch.context() as drifted:
        drifted.setattr(final_analysis, "_analysis_source_records", drifted_sources)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="analysis source changed",
        ):
            final_analysis.load_final_analysis_content(output)


def test_analysis_source_drift_during_statistics_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    baseline = final_analysis._analysis_source_records()
    captures = 0

    def drifting_sources() -> dict[str, dict[str, Any]]:
        nonlocal captures
        captures += 1
        copied = cast(
            dict[str, dict[str, Any]],
            json.loads(json.dumps(baseline)),
        )
        if captures >= 2:
            copied["statistics"]["sha256"] = _sha("changed-statistics-source")
        return copied

    monkeypatch.setattr(final_analysis, "_analysis_source_records", drifting_sources)
    output = tmp_path / "never-created" / "final-analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="analysis source changed",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert captures == 2
    assert not output.parent.exists()


def test_nested_authority_threat_and_reward_tampering_fails_semantic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    pristine = tmp_path / "pristine"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        pristine,
        **_creation_kwargs(fixture),
    )

    mutations: tuple[tuple[str, Any], ...] = (
        (
            "qualification-manifest.json",
            lambda payload: cast(dict[str, Any], payload["reward_blind_boundary"]).__setitem__(
                "reward_arrays_read", 1
            ),
        ),
        (
            "campaign.json",
            lambda payload: cast(
                dict[str, Any], payload["content_capture_threat_boundary"]
            ).__setitem__("noncooperative_same_uid_writers", "trusted"),
        ),
        (
            "completion-summary.json",
            lambda payload: payload.__setitem__("cached_bindings_accepted_as_authority", True),
        ),
    )
    for name, mutate in mutations:
        tampered = tmp_path / f"tampered-{name.removesuffix('.json')}"
        shutil.copytree(pristine, tampered)
        artifact_path = tampered / "evaluation" / name
        payload = cast(dict[str, Any], json.loads(artifact_path.read_bytes()))
        mutate(payload)
        _rewrite_copied_artifact_and_manifest(
            tampered,
            "evaluation",
            name,
            payload,
        )
        with pytest.raises(final_analysis.ForagerMatchedFinalAnalysisError):
            final_analysis.load_final_analysis_content(tampered)


def test_plan_source_executor_live_and_receipt_shapes_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    final_analysis._validate_evaluation_snapshot(
        artifacts,
        seal_content,
        fixture.evaluation_bindings,
    )
    for name in (
        "execution-plan.json",
        "source-manifest.json",
        "executor-manifest.json",
        "live-runtime.json",
        "execution-receipt-index.json",
    ):
        changed = dict(artifacts)
        payload = cast(dict[str, Any], json.loads(changed[name]))
        payload["unexpected_authority_field"] = False
        if name == "execution-receipt-index.json":
            unsigned = dict(payload)
            unsigned.pop("payload_sha256")
            payload["payload_sha256"] = campaign._canonical_sha256(unsigned)
        changed[name] = seal.canonical_json_bytes(payload)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="fields drifted",
        ):
            final_analysis._validate_evaluation_snapshot(
                changed,
                seal_content,
                fixture.evaluation_bindings,
            )

    changed = dict(artifacts)
    executor_payload = cast(
        dict[str, Any],
        json.loads(changed["executor-manifest.json"]),
    )
    qualification_artifacts = cast(dict[str, Any], executor_payload["qualification_artifacts"])
    cpu_qualification = cast(dict[str, Any], qualification_artifacts["cpu_qualification"])
    authority = cast(dict[str, Any], cpu_qualification["authority_boundary"])
    authority["performance_claim"] = True
    changed["executor-manifest.json"] = seal.canonical_json_bytes(executor_payload)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="authority and qualification closure drifted",
    ):
        final_analysis._validate_evaluation_snapshot(
            changed,
            seal_content,
            fixture.evaluation_bindings,
        )
