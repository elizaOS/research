from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as schedule
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from tests import test_forager_matched_executor as executor_fixtures
from tests import test_forager_matched_open_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sealed_context(tmp_path: Path) -> sealed_campaign._SealedContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    open_value = protocol_fixtures._build()
    result_payload = {
        "schema_version": protocol.FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
        "open_protocol_sha256": open_value.protocol_sha256,
        "selection_plan_sha256": open_value.selection_plan.plan_sha256,
        "tuning_seeds": list(open_value.tuning_seeds),
        "ranked_groups": [
            {
                "selection_group": group.selection_group,
                "ranked_candidate_ids": list(reversed(group.candidate_ids)),
                "ranking_evidence_sha256": _sha(
                    f"sealed-ranking:{group.selection_group}"
                ),
            }
            for group in open_value.selection_plan.groups
        ],
    }
    selection = protocol.parse_forager_matched_selection_result(result_payload)
    sealed_protocol = protocol.seal_forager_matched_protocol(open_value, selection)
    transition = protocol.validate_sealed_protocol_transition(
        open_value,
        sealed_protocol,
        selection,
        selection.selection_result_sha256,
    )
    evaluation_schedule = schedule.build_sealed_evaluation_schedule(
        sealed_protocol,
        transition,
    )

    base_plan_root = tmp_path / "base-plan"
    base_plan_root.mkdir()
    base_plan = executor_fixtures._plan(base_plan_root)
    base_candidate = base_plan.candidates[0]
    prepared = tuple(
        replace(
            base_candidate,
            candidate=sealed_protocol.candidate_index[candidate_id],
            capability_receipt_sha256=(
                sealed_protocol.candidate_index[
                    candidate_id
                ].runtime_binding.capability_qualification_receipt_sha256
            ),
        )
        for candidate_id in transition.evaluation_candidate_ids
    )
    plan_payload = base_plan.to_dict()
    plan_payload.update(
        {
            "stage": "sealed_evaluation",
            "protocol_sha256": sealed_protocol.protocol_sha256,
            "active_seeds": list(sealed_protocol.active_seeds),
            "horizon": sealed_protocol.horizon,
            "candidate_order": list(transition.evaluation_candidate_ids),
        }
    )
    plan = executor.MatchedExecutionPlan(
        protocol=sealed_protocol,
        candidates=prepared,
        source_manifest=base_plan.source_manifest,
        executor_manifest=base_plan.executor_manifest,
        payload=plan_payload,
        candidate_index=MappingProxyType(
            {item.candidate.candidate_id: item for item in prepared}
        ),
        cpu_qualification_root=base_plan.cpu_qualification_root,
        rng_parity_qualification_root=base_plan.rng_parity_qualification_root,
    )
    live = executor_fixtures._runtime(tmp_path / "runtime", plan)

    qualification_root = tmp_path / "qualification"
    seal_root = tmp_path / "seal"
    qualification_root.mkdir()
    seal_root.mkdir()
    bundle = cast(
        Any,
        SimpleNamespace(
            output_root=qualification_root,
            manifest={"schema_version": "test.qualification.v1"},
            cpu_qualification_root=base_plan.cpu_qualification_root,
            rng_parity_qualification_root=base_plan.rng_parity_qualification_root,
        ),
    )
    rebuilt = campaign._RebuiltInputs(
        bundle=bundle,
        protocol=sealed_protocol,
        plan=plan,
        candidate_ids=transition.evaluation_candidate_ids,
        assets={},
        schedule=evaluation_schedule,
    )
    open_request_values = {
        "stage": "open_tuning",
        "protocol_sha256": open_value.protocol_sha256,
        "score_evidence_sha256": _sha("open-score"),
        "source_manifest_sha256": _sha("open-source"),
        "executor_manifest_sha256": _sha("open-executor"),
        "execution_closure_sha256": _sha("open-closure"),
        "trust_anchor_identity": (
            open_value.runtime.qualification_trust_anchor_identity
        ),
    }
    open_subject = campaign._canonical_sha256(
        {
            "schema_version": evidence.MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
            **open_request_values,
        }
    )
    open_request = executor.VerificationRequest(
        **cast(Any, open_request_values),
        verification_subject_sha256=open_subject,
        qualification_authority_boundary={
            "endorsement_created": False,
            "endorsements_at_seal": 0,
            "gpu_qualified": False,
            "performance_claim": False,
            "seed_class": "open_development",
            "trust_profile_created": False,
            "trust_profiles_at_seal": 0,
        },
        rng_parity_qualification_status=(
            "content_complete_external_executor_receipt_unverified"
        ),
    )
    transition_descriptor = schedule.build_sealed_transition_descriptor(
        sealed_protocol,
        transition,
    )
    transition_sha256 = schedule.canonical_sealed_transition_descriptor_sha256(
        sealed_protocol,
        transition,
    )
    seal_content = seal.ContentVerifiedSealBundle(
        output_root=seal_root,
        manifest={"payload_sha256": _sha("seal-manifest")},
        open_protocol=open_value,
        open_score_evidence=cast(Any, SimpleNamespace()),
        open_verification_request=open_request,
        recorded_bindings_cache={},
        selection_result=selection,
        selection_report={},
        sealed_protocol=sealed_protocol,
        sealed_transition=transition_descriptor,
        sealed_transition_sha256=transition_sha256,
    )
    inputs = sealed_campaign._SealedInputs(
        rebuilt=rebuilt,
        seal_content=seal_content,
        transition=transition,
        seal_manifest_payload_sha256=_sha("seal-manifest"),
        open_verification_subject_sha256=open_subject,
    )
    engine = campaign._CampaignContext(
        root=tmp_path / "campaign",
        rebuilt=rebuilt,
        live_runtime=live,
    )
    return sealed_campaign._SealedContext(engine=engine, inputs=inputs)


def _status(context: sealed_campaign._SealedContext) -> campaign.CampaignStatus:
    return campaign.CampaignStatus(
        output_root=context.engine.root,
        state="in_progress",
        completed_cells=0,
        total_cells=180,
        next_candidate_id=context.engine.rebuilt.candidate_ids[0],
        next_seed=context.engine.rebuilt.protocol.active_seeds[0],
        protocol_sha256=context.engine.rebuilt.protocol.protocol_sha256,
        plan_sha256=context.engine.rebuilt.plan.plan_sha256,
        live_runtime_identity_sha256=context.engine.live_runtime.identity_sha256,
        score_evidence_sha256=None,
        verification_subject_sha256=None,
    )


def _publish_context(context: sealed_campaign._SealedContext) -> None:
    prospective = sealed_campaign._prospective_output(
        context.inputs,
        context.engine.root,
    )
    sealed_campaign._publish_initial_root(
        context.inputs,
        context.engine.live_runtime,
        context.engine.root,
        prospective,
    )


def _artifact(
    context: sealed_campaign._SealedContext,
    candidate_id: str,
    seed: int,
    raw_sha256: str,
    raw_size: int,
) -> executor.SeedExecutionArtifacts:
    return executor._artifact_mappings(
        plan=context.engine.rebuilt.plan,
        candidate=context.engine.rebuilt.plan.candidate_index[candidate_id],
        seed=seed,
        raw_archive_sha256=raw_sha256,
        raw_archive_size=raw_size,
        live_runtime=context.engine.live_runtime,
        scorer_record={
            "fov_last_10pct_ema_auc": 0.5,
            "npz_sha256": _sha(f"npz:{candidate_id}:{seed}"),
            "npz_size_bytes": 4096,
            "reward_trace_sha256": _sha(f"trace:{candidate_id}:{seed}"),
            "reward_dtype": "<f4",
            "reward_shape": [context.engine.rebuilt.protocol.horizon],
        },
    )


def _pins(context: sealed_campaign._SealedContext) -> dict[str, Any]:
    return {
        "resolver": cast(Any, lambda _request: None),
        "expected_trust_anchor_identity": (
            context.inputs.seal_content.open_verification_request.trust_anchor_identity
        ),
        "expected_seal_manifest_payload_sha256": (
            context.inputs.seal_manifest_payload_sha256
        ),
        "expected_open_verification_subject_sha256": (
            context.inputs.open_verification_subject_sha256
        ),
    }


def test_exact_six_by_thirty_manifest_and_initial_inventory(tmp_path: Path) -> None:
    context = _sealed_context(tmp_path)
    manifest = sealed_campaign._campaign_manifest(
        context.inputs,
        context.engine.live_runtime,
    )

    assert len(context.engine.rebuilt.candidate_ids) == 6
    assert len(context.engine.rebuilt.protocol.active_seeds) == 30
    assert len(context.engine.rebuilt.schedule["cells"]) == 180
    assert manifest["candidate_order"] == list(context.engine.rebuilt.candidate_ids)
    assert manifest["selected_candidate_ids"] == list(
        context.engine.rebuilt.candidate_ids[:4]
    )
    assert manifest["fixed_descriptive_candidate_ids"] == list(
        context.engine.rebuilt.candidate_ids[4:]
    )
    assert manifest["content_capture_threat_boundary"] == {
        "campaign_tree_writers": "cooperative_lock_participants_trusted",
        "root_guard_scope": "top_level_inode_and_path_swap_detection_only",
        "noncooperative_same_uid_writers": "out_of_scope",
        "claim_authority": "independent_final_subject_authentication_required",
    }
    _publish_context(context)
    expected = {
        "runs",
        "completions",
        *sealed_campaign._IMMUTABLE_ARTIFACTS,
        *(f"{name}.sha256" for name in sealed_campaign._IMMUTABLE_ARTIFACTS),
    }
    assert {path.name for path in context.engine.root.iterdir()} == expected


def test_prepare_auth_failure_creates_no_output_or_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "uncreated" / "campaign"
    monkeypatch.setattr(
        sealed_campaign,
        "_rebuild_sealed_inputs",
        lambda *_args: context.inputs,
    )
    monkeypatch.setattr(
        campaign,
        "_qualify_live",
        lambda *_args: context.engine.live_runtime,
    )

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("authority rejected")

    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", reject)
    with pytest.raises(RuntimeError, match="authority rejected"):
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            output,
            **_pins(context),
        )
    assert not output.parent.exists()


def test_prepare_event_order_authenticates_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "campaign"
    events: list[str] = []

    def rebuild(*_args: Any) -> sealed_campaign._SealedInputs:
        events.append("replay")
        return context.inputs

    def qualify(*_args: Any) -> executor.LiveRuntimeIdentity:
        events.append("qualify")
        return context.engine.live_runtime

    def authenticate(*_args: Any, **kwargs: Any) -> None:
        events.append("authenticate")
        assert kwargs["expected_seal_manifest_payload_sha256"] == _sha(
            "seal-manifest"
        )

    def publish(*_args: Any) -> Path:
        events.append("publish")
        return output

    def verify(*_args: Any, **_kwargs: Any) -> campaign.CampaignStatus:
        events.append("verify")
        return _status(context)

    monkeypatch.setattr(sealed_campaign, "_rebuild_sealed_inputs", rebuild)
    monkeypatch.setattr(campaign, "_qualify_live", qualify)
    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", authenticate)
    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        verify,
    )

    sealed_campaign.prepare_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        output,
        **_pins(context),
    )
    assert events == ["replay", "qualify", "authenticate", "publish", "verify"]


def test_postpublication_verify_failure_is_uncertain_and_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        sealed_campaign,
        "_rebuild_sealed_inputs",
        lambda *_args: context.inputs,
    )
    monkeypatch.setattr(
        campaign,
        "_qualify_live",
        lambda *_args: context.engine.live_runtime,
    )
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: None,
    )

    def publish(*_args: Any) -> Path:
        output.mkdir()
        (output / "preserved").write_bytes(b"published")
        return output

    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("replay failed")),
    )
    with pytest.raises(
        sealed_campaign.PublishedSealedEvaluationCampaignUncertainError
    ) as captured:
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            output,
            **_pins(context),
        )
    assert captured.value.destination == output
    assert (output / "preserved").read_bytes() == b"published"


def test_read_only_verification_never_authenticates_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    monkeypatch.setattr(
        sealed_campaign,
        "_load_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: pytest.fail("read-only API resolved authority"),
    )
    before = {
        path.relative_to(context.engine.root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in context.engine.root.rglob("*")
        if path.is_file()
    }
    status = sealed_campaign.verify_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
    )
    after = {
        path.relative_to(context.engine.root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in context.engine.root.rglob("*")
        if path.is_file()
    }
    assert status.total_cells == 180
    assert status.completed_cells == 0
    assert after == before


def test_run_recovers_bound_raw_then_repairs_missing_completion_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    monkeypatch.setattr(
        sealed_campaign,
        "_load_context",
        lambda *_args, **_kwargs: context,
    )
    auth_calls: list[str] = []
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: auth_calls.append("auth"),
    )
    execute_calls: list[tuple[str, int]] = []
    score_calls: list[tuple[str, int]] = []

    def fail_after_raw(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        raw_path: Path,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        execute_calls.append((candidate_id, seed))
        raw_path.write_bytes(f"opaque:{candidate_id}:{seed}".encode("ascii"))
        raise RuntimeError("injected scorer failure")

    def score_raw(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        _raw_path: Path,
        binding: Any,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        score_calls.append((candidate_id, seed))
        raw = binding["raw_archive"]
        return _artifact(
            context,
            candidate_id,
            seed,
            cast(str, raw["sha256"]),
            cast(int, raw["size_bytes"]),
        )

    monkeypatch.setattr(campaign, "_execute_with_optional_runner", fail_after_raw)
    monkeypatch.setattr(campaign, "_score_with_optional_runner", score_raw)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="matched execution"):
        sealed_campaign.run_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            context.engine.root,
            max_cells=1,
            **_pins(context),
        )
    status = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        max_cells=1,
        **_pins(context),
    )
    assert status.completed_cells == 1
    assert len(execute_calls) == 1
    assert len(score_calls) == 1

    candidate_id, seed = execute_calls[0]
    _run, pointer = campaign._cell_paths(context.engine.root, candidate_id, seed)
    pointer.unlink()
    pointer.with_name(f"{pointer.name}.sha256").unlink()
    repaired = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        max_cells=1,
        **_pins(context),
    )
    assert repaired.completed_cells == 1
    assert pointer.is_file()
    assert len(execute_calls) == 1
    assert len(score_calls) == 1
    assert len(auth_calls) == 3


def test_full_evaluation_finalizes_stage_specific_unresolved_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    events: list[str] = []

    def load(*_args: Any, **_kwargs: Any) -> sealed_campaign._SealedContext:
        events.append("replay")
        return context

    def authenticate(*_args: Any, **_kwargs: Any) -> None:
        events.append("authenticate")

    def execute(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        raw_path: Path,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        if not any(event == "execute" for event in events):
            events.append("execute")
        raw = f"opaque:{candidate_id}:{seed}".encode("ascii")
        raw_path.write_bytes(raw)
        return _artifact(
            context,
            candidate_id,
            seed,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        )

    monkeypatch.setattr(sealed_campaign, "_load_context", load)
    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", authenticate)
    monkeypatch.setattr(campaign, "_execute_with_optional_runner", execute)

    status = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        **_pins(context),
    )
    assert events[:3] == ["replay", "authenticate", "execute"]
    assert status.state == "complete_content_only_external_verification_unresolved"
    assert status.completed_cells == status.total_cells == 180
    bundle = sealed_campaign.load_completed_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
    )
    assert bundle.candidate_ids == context.inputs.transition.evaluation_candidate_ids
    assert bundle.completion_summary["stage"] == "sealed_evaluation"
    assert bundle.completion_summary["selection_inherited_from_seal"] is True
    assert bundle.completion_summary["promotion_authorized"] is False
    assert bundle.completion_summary[
        "evaluation_verification_authentication_state"
    ] == "unresolved_external_verifier_required"
    assert bundle.completion_summary["sealed_transition_sha256"] == (
        context.inputs.seal_content.sealed_transition_sha256
    )


def test_false_authority_completion_summary_is_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    context = _sealed_context(tmp_path)
    receipt = cast(Any, SimpleNamespace(payload_sha256=_sha("receipt")))
    scores = cast(Any, SimpleNamespace(payload_sha256=_sha("scores")))
    request = cast(
        Any,
        SimpleNamespace(verification_subject_sha256=_sha("final-subject")),
    )
    summary = sealed_campaign._completion_summary(
        context.inputs,
        context.engine,
        receipt,
        scores,
        request,
    )
    summary["promotion_authorized"] = True
    summary["protocol_sha256"] = _sha("wrong-protocol")

    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="common closure/authority invariants",
    ):
        sealed_campaign._summary_validator(context.inputs)(
            context.engine,
            receipt,
            scores,
            request,
            summary,
        )
    assert not any(
        (context.engine.root / name).exists()
        for name in campaign._FINAL_ARTIFACTS
    )
