from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from tests import test_forager_matched_evidence as evidence_fixtures
from tests import test_forager_matched_seal as seal_fixtures

pytestmark = pytest.mark.integration


class _AuthenticationPins(TypedDict):
    expected_trust_anchor_identity: str
    expected_seal_manifest_payload_sha256: str
    expected_open_verification_subject_sha256: str


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(campaign.canonical_json_bytes(value)).hexdigest()


@cache
def _six_by_thirty_protocol() -> tuple[
    protocol.ForagerMatchedProtocol,
    protocol.ForagerMatchedProtocol,
    protocol.ForagerMatchedSelectionResult,
    protocol.SealedProtocolValidation,
    dict[str, Any],
]:
    payload, toy_open, _scores = evidence_fixtures._open_fixture()
    payload["evaluation_seeds"] = list(range(2_200_001, 2_200_031))
    payload["selection_plan"]["groups"] = [
        {
            "selection_group": "alberta",
            "candidate_ids": ["alberta_causal", "alberta_route"],
            "advance_count": 2,
        },
        {
            "selection_group": "external",
            "candidate_ids": ["external_dqn", "isolated_rtu"],
            "advance_count": 2,
        },
    ]
    for candidate in payload["candidates"]:
        if candidate["candidate_id"] != "isolated_rtu":
            continue
        candidate["selection_group"] = "external"
        original = toy_open.candidate_index["isolated_rtu"]
        candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            protocol.candidate_capability_descriptor_sha256(
                replace(original, selection_group="external")
            )
        )
    payload["evaluation_panel"]["selection_slots"] = [
        {"selection_group": "alberta", "rank": 1},
        {"selection_group": "alberta", "rank": 2},
        {"selection_group": "external", "rank": 1},
        {"selection_group": "external", "rank": 2},
    ]
    payload["secondary_hypotheses"][0]["intervention_slot"] = {
        "selection_group": "external",
        "rank": 2,
    }
    opened = protocol.parse_forager_matched_protocol(payload)
    selection = protocol.ForagerMatchedSelectionResult(
        schema_version="alberta.forager_matched_selection_result.v1",
        open_protocol_sha256=opened.protocol_sha256,
        selection_plan_sha256=opened.selection_plan.plan_sha256,
        tuning_seeds=opened.tuning_seeds,
        ranked_groups=(
            protocol.RankedSelectionGroup(
                selection_group="alberta",
                ranked_candidate_ids=("alberta_route", "alberta_causal"),
                ranking_evidence_sha256=_sha("alberta-ranking"),
            ),
            protocol.RankedSelectionGroup(
                selection_group="external",
                ranked_candidate_ids=("isolated_rtu", "external_dqn"),
                ranking_evidence_sha256=_sha("external-ranking"),
            ),
        ),
    )
    sealed = protocol.seal_forager_matched_protocol(opened, selection)
    transition = protocol.validate_sealed_protocol_transition(
        opened,
        sealed,
        selection,
        selection.selection_result_sha256,
    )
    schedule = evaluation.build_sealed_evaluation_schedule(sealed, transition)
    return opened, sealed, selection, transition, schedule


def _fake_plan(sealed: protocol.ForagerMatchedProtocol, candidate_ids: tuple[str, ...]) -> Any:
    source_manifest = {
        "schema_version": "test.sealed.source-manifest.v1",
        "candidate_order": list(candidate_ids),
    }
    executor_manifest = {
        "schema_version": "test.sealed.executor-manifest.v1",
        "candidate_order": list(candidate_ids),
    }
    payload = {
        "schema_version": "test.sealed.execution-plan.v1",
        "stage": "sealed_evaluation",
        "protocol_sha256": sealed.protocol_sha256,
        "candidate_order": list(candidate_ids),
        "active_seeds": list(sealed.active_seeds),
        "source_manifest_sha256": _canonical_sha(source_manifest),
        "executor_manifest_sha256": _canonical_sha(executor_manifest),
    }
    return SimpleNamespace(
        protocol=sealed,
        candidates=tuple(
            SimpleNamespace(candidate=SimpleNamespace(candidate_id=candidate_id))
            for candidate_id in candidate_ids
        ),
        source_manifest=source_manifest,
        source_manifest_sha256=_canonical_sha(source_manifest),
        executor_manifest=executor_manifest,
        executor_manifest_sha256=_canonical_sha(executor_manifest),
        plan_sha256=_canonical_sha(payload),
        to_dict=lambda: payload,
    )


def _fake_live() -> Any:
    payload = {
        "schema_version": "test.sealed.live-runtime.v1",
        "runtime": "synthetic-no-process",
    }
    return SimpleNamespace(unsigned_dict=payload, identity_sha256=_canonical_sha(payload))


def _synthetic_inputs(
    tmp_path: Path,
    *,
    seal_content: Any | None = None,
) -> sealed_campaign._SealedInputs:
    opened, sealed, selection, transition, schedule = _six_by_thirty_protocol()
    candidate_ids = transition.evaluation_candidate_ids
    plan = _fake_plan(sealed, candidate_ids)
    bundle = SimpleNamespace(
        output_root=tmp_path / "synthetic-qualification",
        manifest={"schema_version": "test.qualification.v1"},
        cpu_qualification_root=tmp_path / "cpu-qualification",
        rng_parity_qualification_root=tmp_path / "rng-qualification",
    )
    if seal_content is None:
        descriptor = evaluation.build_sealed_transition_descriptor(sealed, transition)
        descriptor_sha = evaluation.canonical_sealed_transition_descriptor_sha256(
            sealed,
            transition,
        )
        seal_content = SimpleNamespace(
            output_root=tmp_path / "synthetic-seal",
            manifest={"payload_sha256": _sha("synthetic-seal-manifest")},
            open_protocol=opened,
            open_verification_request=SimpleNamespace(
                verification_subject_sha256=_sha("synthetic-open-subject")
            ),
            selection_result=selection,
            sealed_protocol=sealed,
            sealed_transition=descriptor,
            sealed_transition_sha256=descriptor_sha,
        )
    rebuilt = campaign._RebuiltInputs(
        bundle=cast(Any, bundle),
        protocol=sealed,
        plan=cast(Any, plan),
        candidate_ids=candidate_ids,
        assets={},
        schedule=schedule,
    )
    return sealed_campaign._SealedInputs(
        rebuilt=rebuilt,
        seal_content=cast(Any, seal_content),
        transition=transition,
        seal_manifest_payload_sha256=cast(str, seal_content.manifest["payload_sha256"]),
        open_verification_subject_sha256=cast(
            str,
            seal_content.open_verification_request.verification_subject_sha256,
        ),
    )


def _context(root: Path, inputs: sealed_campaign._SealedInputs) -> sealed_campaign._SealedContext:
    engine = campaign._CampaignContext(
        root=root,
        rebuilt=inputs.rebuilt,
        live_runtime=_fake_live(),
    )
    return sealed_campaign._SealedContext(engine=engine, inputs=inputs)


def _status(root: Path, inputs: sealed_campaign._SealedInputs) -> campaign.CampaignStatus:
    return campaign.CampaignStatus(
        output_root=root,
        state="in_progress",
        completed_cells=0,
        total_cells=180,
        next_candidate_id=inputs.rebuilt.candidate_ids[0],
        next_seed=inputs.rebuilt.protocol.active_seeds[0],
        protocol_sha256=inputs.rebuilt.protocol.protocol_sha256,
        plan_sha256=inputs.rebuilt.plan.plan_sha256,
        live_runtime_identity_sha256=_fake_live().identity_sha256,
        score_evidence_sha256=None,
        verification_subject_sha256=None,
    )


def _minimal_locked_root(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "completions").mkdir()
    campaign._publish_json_pair(
        root / "campaign.json",
        {"schema_version": "test.sealed.lock.v1"},
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, str], ...]:
    if not root.exists():
        return ()
    result: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result.append((relative, "symlink", os.readlink(path)))
        elif path.is_dir():
            result.append((relative, "directory", ""))
        else:
            result.append((relative, "file", hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


@pytest.fixture(scope="module")
def real_seal_bundle(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("sealed-campaign-real-auth")
    completed, bindings = seal_fixtures._completed_campaign(root)
    qualification_root = root / "qualification"
    campaign_root = root / "open-campaign"
    qualification_root.mkdir()
    campaign_root.mkdir()
    output_root = root / "seal"
    patcher = pytest.MonkeyPatch()
    seal_fixtures._install_completed_loader(patcher, completed)
    try:
        content = seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )
    finally:
        patcher.undo()
    return SimpleNamespace(content=content, bindings=bindings)


def _pins(real_seal_bundle: Any) -> _AuthenticationPins:
    return {
        "expected_trust_anchor_identity": (
            real_seal_bundle.bindings.trust_anchor_identity
        ),
        "expected_seal_manifest_payload_sha256": cast(
            str,
            real_seal_bundle.content.manifest["payload_sha256"],
        ),
        "expected_open_verification_subject_sha256": (
            real_seal_bundle.bindings.verification_subject_sha256
        ),
    }


def test_exact_six_by_thirty_rebuild_and_seed_major_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened, sealed, selection, transition, schedule = _six_by_thirty_protocol()
    qualification_root = tmp_path / "qualification"
    seal_root = tmp_path / "seal"
    qualification_root.mkdir()
    seal_root.mkdir()
    candidate_universe_ids = tuple(candidate.candidate_id for candidate in opened.candidates)
    assets = {
        candidate_id: SimpleNamespace(candidate_id=candidate_id)
        for candidate_id in candidate_universe_ids
    }
    bundle = SimpleNamespace(
        output_root=qualification_root,
        runtime_qualification=SimpleNamespace(),
        candidate_qualifications={},
        candidate_assets=assets,
        cpu_qualification_root=tmp_path / "cpu",
        rng_parity_qualification_root=tmp_path / "rng",
        manifest={"schema_version": "test.qualification.v1"},
    )
    descriptor = evaluation.build_sealed_transition_descriptor(sealed, transition)
    descriptor_sha = evaluation.canonical_sealed_transition_descriptor_sha256(
        sealed,
        transition,
    )
    content = SimpleNamespace(
        output_root=seal_root,
        manifest={"payload_sha256": _sha("seal-manifest")},
        open_protocol=opened,
        open_verification_request=SimpleNamespace(
            verification_subject_sha256=_sha("open-subject")
        ),
        selection_result=selection,
        sealed_protocol=sealed,
        sealed_transition=descriptor,
        sealed_transition_sha256=descriptor_sha,
    )
    plan = _fake_plan(sealed, transition.evaluation_candidate_ids)
    observed: dict[str, Any] = {}

    def build_plan(
        frozen: protocol.ForagerMatchedProtocol,
        selected_assets: dict[str, Any],
        *,
        candidate_ids: tuple[str, ...],
        cpu_qualification_root: Path,
        rng_parity_qualification_root: Path,
    ) -> Any:
        observed.update(
            frozen=frozen,
            assets=selected_assets,
            candidate_ids=candidate_ids,
            cpu=cpu_qualification_root,
            rng=rng_parity_qualification_root,
        )
        return plan

    monkeypatch.setattr(
        qualification,
        "load_matched_current_qualification_bundle",
        lambda _root: bundle,
    )
    monkeypatch.setattr(
        open_protocol,
        "MATCHED_CURRENT_CANDIDATE_IDS",
        candidate_universe_ids,
    )
    monkeypatch.setattr(seal, "load_forager_matched_seal_bundle_content", lambda _root: content)
    monkeypatch.setattr(
        open_protocol,
        "build_forager_matched_open_protocol",
        lambda **_kwargs: opened,
    )
    monkeypatch.setattr(
        universe,
        "verify_matched_current_candidate_universe_sources",
        lambda _root: SimpleNamespace(
            candidate_universe_sha256=opened.selection_plan.candidate_universe_sha256
        ),
    )
    monkeypatch.setattr(executor, "build_execution_plan", build_plan)

    rebuilt = sealed_campaign._rebuild_sealed_inputs(qualification_root, seal_root)

    expected_ids = (
        "alberta_route",
        "alberta_causal",
        "isolated_rtu",
        "external_dqn",
        "exact_ppo",
        "search_oracle",
    )
    assert rebuilt.rebuilt.candidate_ids == expected_ids
    assert tuple(observed["assets"]) == expected_ids
    assert observed["candidate_ids"] == expected_ids
    assert len(rebuilt.rebuilt.protocol.active_seeds) == 30
    assert len(schedule["cells"]) == 180
    assert [cell["candidate_id"] for cell in schedule["cells"][:6]] == list(
        expected_ids
    )
    assert {cell["seed"] for cell in schedule["cells"][:6]} == {2_200_001}
    assert [cell["ordinal"] for cell in schedule["cells"]] == list(range(180))


def test_initial_publication_has_exact_immutable_inventory(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    live = _fake_live()
    output_root = tmp_path / "published" / "evaluation"
    prospective = sealed_campaign._prospective_output(inputs, output_root)

    published = sealed_campaign._publish_initial_root(
        inputs,
        live,
        output_root,
        prospective,
    )

    expected = {
        "runs",
        "completions",
        *sealed_campaign._IMMUTABLE_ARTIFACTS,
        *(f"{name}.sha256" for name in sealed_campaign._IMMUTABLE_ARTIFACTS),
    }
    assert {path.name for path in published.iterdir()} == expected
    assert not any((published / name).exists() for name in sealed_campaign._FINAL_ARTIFACTS)
    assert not any((published / "runs").iterdir())
    assert not any((published / "completions").iterdir())
    for name in sealed_campaign._IMMUTABLE_ARTIFACTS:
        raw = (published / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert (published / f"{name}.sha256").read_bytes() == (
            f"{digest}\n".encode("ascii")
        )
    sealed_campaign._validate_root_shape(published)


def test_mandatory_pins_and_resolver_failure_are_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_seal_bundle: Any,
) -> None:
    for function in (
        sealed_campaign.prepare_sealed_evaluation_campaign,
        sealed_campaign.run_sealed_evaluation_campaign,
    ):
        parameters = inspect.signature(function).parameters
        for name in (
            "resolver",
            "expected_trust_anchor_identity",
            "expected_seal_manifest_payload_sha256",
            "expected_open_verification_subject_sha256",
        ):
            assert parameters[name].default is inspect.Parameter.empty

    inputs = _synthetic_inputs(tmp_path, seal_content=real_seal_bundle.content)
    monkeypatch.setattr(sealed_campaign, "_rebuild_sealed_inputs", lambda *_args: inputs)
    monkeypatch.setattr(campaign, "_qualify_live", lambda *_args, **_kwargs: _fake_live())
    resolver_calls = 0

    def resolver(_request: executor.VerificationRequest) -> Any:
        nonlocal resolver_calls
        resolver_calls += 1
        return real_seal_bundle.bindings

    wrong_output = tmp_path / "wrong-pin-parent" / "evaluation"
    wrong_pins = _pins(real_seal_bundle)
    wrong_pins["expected_seal_manifest_payload_sha256"] = _sha("wrong-manifest")
    with pytest.raises(seal.ForagerMatchedSealError, match="caller-pinned digest"):
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            real_seal_bundle.content.output_root,
            wrong_output,
            resolver=resolver,
            **wrong_pins,
        )
    assert resolver_calls == 0
    assert not wrong_output.parent.exists()

    rejected_output = tmp_path / "rejected-parent" / "evaluation"

    def reject(_request: executor.VerificationRequest) -> Any:
        raise RuntimeError("independent authority rejected subject")

    with pytest.raises(seal.ForagerMatchedSealError, match="trust resolution"):
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            real_seal_bundle.content.output_root,
            rejected_output,
            resolver=reject,
            **_pins(real_seal_bundle),
        )
    assert not rejected_output.parent.exists()


def test_resolver_is_last_event_before_prepare_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_seal_bundle: Any,
) -> None:
    inputs = _synthetic_inputs(tmp_path, seal_content=real_seal_bundle.content)
    output_root = tmp_path / "evaluation"
    events: list[str] = []
    expected_status = _status(output_root, inputs)

    monkeypatch.setattr(sealed_campaign, "_rebuild_sealed_inputs", lambda *_args: inputs)

    def qualify(*_args: Any, **_kwargs: Any) -> Any:
        events.append("qualify")
        return _fake_live()

    def resolver(_request: executor.VerificationRequest) -> Any:
        events.append("resolver")
        return real_seal_bundle.bindings

    def publish(*_args: Any, **_kwargs: Any) -> Path:
        events.append("publish")
        return output_root

    monkeypatch.setattr(campaign, "_qualify_live", qualify)
    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        lambda *_args, **_kwargs: expected_status,
    )

    status = sealed_campaign.prepare_sealed_evaluation_campaign(
        tmp_path / "qualification",
        real_seal_bundle.content.output_root,
        output_root,
        resolver=resolver,
        **_pins(real_seal_bundle),
    )

    assert status is expected_status
    assert events == ["qualify", "resolver", "publish"]


def test_run_auth_failure_does_not_repair_and_resolver_precedes_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_seal_bundle: Any,
) -> None:
    inputs = _synthetic_inputs(tmp_path, seal_content=real_seal_bundle.content)
    root = tmp_path / "evaluation"
    _minimal_locked_root(root)
    orphan = root / "runs" / "unpaired.json"
    orphan.write_bytes(campaign.canonical_json_bytes({"orphan": True}))
    context = _context(root, inputs)
    expected_status = _status(root, inputs)
    events: list[str] = []

    def load(*_args: Any, **_kwargs: Any) -> sealed_campaign._SealedContext:
        events.append("load")
        return context

    monkeypatch.setattr(sealed_campaign, "_load_context", load)
    before = _tree_snapshot(root)

    def reject(_request: executor.VerificationRequest) -> Any:
        events.append("resolver")
        raise RuntimeError("rejected")

    with pytest.raises(seal.ForagerMatchedSealError, match="trust resolution"):
        sealed_campaign.run_sealed_evaluation_campaign(
            tmp_path / "qualification",
            real_seal_bundle.content.output_root,
            root,
            resolver=reject,
            max_cells=1,
            **_pins(real_seal_bundle),
        )
    assert events == ["load", "resolver"]
    assert _tree_snapshot(root) == before
    assert not orphan.with_name(f"{orphan.name}.sha256").exists()

    events.clear()

    def resolver(_request: executor.VerificationRequest) -> Any:
        events.append("resolver")
        return real_seal_bundle.bindings

    def run_engine(
        _context: campaign._CampaignContext,
        **kwargs: Any,
    ) -> campaign.CampaignStatus:
        cast(Any, kwargs["mutation_guard"])()
        events.append("engine")
        return expected_status

    monkeypatch.setattr(campaign, "_run_resumable_context_locked", run_engine)
    status = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        real_seal_bundle.content.output_root,
        root,
        resolver=resolver,
        max_cells=1,
        **_pins(real_seal_bundle),
    )
    assert status is expected_status
    assert events == ["load", "resolver", "engine"]


def test_read_only_entry_points_never_resolve_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path)
    root = tmp_path / "evaluation"
    _minimal_locked_root(root)
    context = _context(root, inputs)
    expected_status = _status(root, inputs)
    completed = object()
    resolver_calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal resolver_calls
        resolver_calls += 1
        raise AssertionError("read-only API attempted external trust resolution")

    monkeypatch.setattr(seal, "authenticate_forager_matched_seal_bundle", forbidden)
    monkeypatch.setattr(executor, "resolve_authenticated_bindings", forbidden)
    monkeypatch.setattr(sealed_campaign, "_load_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        sealed_campaign,
        "_derive_content_status",
        lambda _context: expected_status,
    )

    assert sealed_campaign.verify_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        root,
    ) is expected_status
    assert sealed_campaign.sealed_evaluation_campaign_content_status(
        tmp_path / "qualification",
        tmp_path / "seal",
        root,
    ) is expected_status

    monkeypatch.setattr(
        campaign,
        "_scan_all_cells",
        lambda _context: {("done", 1): SimpleNamespace(artifact=object(), pointer_present=True)},
    )
    monkeypatch.setattr(
        campaign,
        "_build_completed_campaign_bundle",
        lambda *_args, **_kwargs: completed,
    )
    assert sealed_campaign.load_completed_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        root,
    ) is completed
    assert resolver_calls == 0


def test_completion_summary_rejects_false_authority_and_closure(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    context = _context(tmp_path / "evaluation", inputs).engine
    receipt = cast(Any, SimpleNamespace(payload_sha256=_sha("receipt-index")))
    scores = cast(Any, SimpleNamespace(payload_sha256=_sha("scores")))
    request = cast(
        Any,
        SimpleNamespace(verification_subject_sha256=_sha("evaluation-subject")),
    )
    builder = sealed_campaign._summary_builder(inputs)
    validator = sealed_campaign._summary_validator(inputs)
    valid = builder(context, receipt, scores, request)
    validator(context, receipt, scores, request, valid)

    tampered = []
    for key, value in (
        ("promotion_authorized", True),
        ("performance_claim", True),
        ("external_verification_required", False),
        ("cached_bindings_accepted_as_authority", True),
        ("score_evidence_sha256", _sha("other-scores")),
        ("seal_manifest_payload_sha256", _sha("other-seal")),
    ):
        changed = dict(valid)
        changed[key] = value
        tampered.append(changed)
    for changed in tampered:
        with pytest.raises(ValueError, match="closure|exact v1 schema"):
            validator(context, receipt, scores, request, changed)


def test_partial_final_artifacts_fail_closed_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path)
    root = tmp_path / "evaluation"
    _minimal_locked_root(root)
    context = _context(root, inputs)
    partial = root / "score-evidence.json"
    partial.write_bytes(b"partial-final-must-not-be-repaired")
    scans = {
        (cast(str, cell["candidate_id"]), cast(int, cell["seed"])): SimpleNamespace(
            artifact=None,
            pointer_present=False,
            resumable_attempt=None,
        )
        for cell in cast(list[dict[str, Any]], inputs.rebuilt.schedule["cells"])
    }
    monkeypatch.setattr(sealed_campaign, "_load_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(campaign, "_scan_all_cells", lambda _context: scans)

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="final evidence exists"):
        sealed_campaign.verify_sealed_evaluation_campaign_content(
            tmp_path / "qualification",
            tmp_path / "seal",
            root,
        )
    with pytest.raises(
        sealed_campaign.ForagerMatchedSealedEvaluationCampaignError,
        match="final evaluation evidence exists",
    ):
        sealed_campaign.load_completed_sealed_evaluation_campaign_content(
            tmp_path / "qualification",
            tmp_path / "seal",
            root,
        )
    assert partial.read_bytes() == b"partial-final-must-not-be-repaired"
    assert not partial.with_name(f"{partial.name}.sha256").exists()


def test_root_and_dynamic_root_substitution_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root-swap" / "evaluation"
    _minimal_locked_root(root)
    moved = root.with_name("evaluation-moved")
    replacement_marker = root / "replacement-owner"
    with pytest.raises(ValueError, match="campaign root|lock"):
        with sealed_campaign._locked_campaign_root(root, exclusive=True) as recheck:
            root.rename(moved)
            _minimal_locked_root(root)
            replacement_marker.write_text("must-survive", encoding="utf-8")
            recheck()
    assert replacement_marker.read_text(encoding="utf-8") == "must-survive"
    assert (moved / "campaign.json").is_file()

    dynamic_root = tmp_path / "runs-swap" / "evaluation"
    _minimal_locked_root(dynamic_root)
    moved_runs = dynamic_root / "runs-moved"
    replacement_runs_marker = dynamic_root / "runs" / "replacement-owner"
    with pytest.raises(ValueError, match="runs root inode changed"):
        with sealed_campaign._locked_campaign_root(
            dynamic_root,
            exclusive=True,
        ) as recheck:
            (dynamic_root / "runs").rename(moved_runs)
            (dynamic_root / "runs").mkdir()
            replacement_runs_marker.write_text("must-survive", encoding="utf-8")
            recheck()
    assert replacement_runs_marker.read_text(encoding="utf-8") == "must-survive"


def test_postpublication_failure_is_explicitly_uncertain_and_retains_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_seal_bundle: Any,
) -> None:
    inputs = _synthetic_inputs(tmp_path, seal_content=real_seal_bundle.content)
    output_root = tmp_path / "evaluation"
    monkeypatch.setattr(sealed_campaign, "_rebuild_sealed_inputs", lambda *_args: inputs)
    monkeypatch.setattr(campaign, "_qualify_live", lambda *_args, **_kwargs: _fake_live())

    def publish(*_args: Any, **_kwargs: Any) -> Path:
        output_root.mkdir()
        (output_root / "published-owner").write_text("retained", encoding="utf-8")
        return output_root

    def fail_replay(*_args: Any, **_kwargs: Any) -> Any:
        raise sealed_campaign.ForagerMatchedSealedEvaluationCampaignError(
            "injected final replay failure"
        )

    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        fail_replay,
    )

    with pytest.raises(
        sealed_campaign.PublishedSealedEvaluationCampaignUncertainError
    ) as caught:
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            real_seal_bundle.content.output_root,
            output_root,
            resolver=lambda _request: real_seal_bundle.bindings,
            **_pins(real_seal_bundle),
        )
    assert caught.value.destination == output_root
    assert (output_root / "published-owner").read_text(encoding="utf-8") == "retained"
