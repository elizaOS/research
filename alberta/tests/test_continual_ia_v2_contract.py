"""Adversarial tests for the unissued development-only continual-IA v2 contract."""

from __future__ import annotations

import copy
import os
import stat
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.continual_ia_v2 as ia_v2
from alberta_framework.evaluation.continual_ia import (
    ContinualIAConfig,
    IAAcceptanceThresholds,
)
from alberta_framework.evaluation.continual_ia_cli import main as v1_main
from alberta_framework.evaluation.continual_ia_v2 import (
    ARTIFACT_SCHEMA,
    NAMESPACE,
    PLAN_SCHEMA,
    SHARD_SCHEMA,
    TRACE_SCHEMA,
    V2_CONDITIONS,
    V2_EVIDENCE_SEEDS,
    ContinualIAV2Error,
    canonical_json_bytes,
    canonical_json_sha256,
    load_artifact,
    load_plan,
    strict_json_loads,
    validate_artifact_payload,
    validate_plan_payload,
    validate_shard_payload,
    validate_trace,
    write_plan,
)
from alberta_framework.evaluation.continual_ia_v2_cli import main as v2_main

pytestmark = pytest.mark.integration


def _reward_trace(count: int) -> list[float]:
    return [1.0] * count + [0.0] * (1_200 - count)


def _record(
    rewards: list[float],
    *,
    recommendation: bool,
    treatment: bool = False,
    accept_always: bool = False,
) -> dict[str, object]:
    actions = [0] * 1_200
    credits = [0] * 1_200
    if not recommendation:
        return {
            "rewards": rewards,
            "executed_actions": actions,
            "credited_actions": credits,
            "pre_update_recommendations": [-1] * 1_200,
            "pre_update_partner_proposals": [-1] * 1_200,
            "accepted_recommendations": [False] * 1_200,
        }
    recommendations = [0] * 1_200
    proposals = [0] * 1_200
    accepted = [False] * 1_200
    if treatment:
        accepted[1:901] = [True] * 900
        recommendations[1:145] = [1] * 144
    if accept_always:
        accepted[1:] = [True] * 1_199
    actions = [
        recommendation_value if acceptance else proposal
        for recommendation_value, proposal, acceptance in zip(
            recommendations, proposals, accepted, strict=True
        )
    ]
    return {
        "rewards": rewards,
        "executed_actions": actions,
        "credited_actions": list(actions),
        "pre_update_recommendations": recommendations,
        "pre_update_partner_proposals": proposals,
        "accepted_recommendations": accepted,
    }


def _synthetic_trace() -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "semantics": {
            "alignment": (
                "index t records reward_t from executed_action_t; the action, partner proposal, "
                "recommendation, and acceptance were selected before transition t and before any "
                "update using reward_t"
            ),
            "credit": "credited_actions[t] is the primitive action credited for transition t",
            "first_transition": (
                "recommendation/proposal equal the initial executed action and acceptance is false"
            ),
            "update_order": "predict/select before transition; observe reward; then update",
        },
        "conditions": {
            "partner_alone": _record(_reward_trace(600), recommendation=False),
            "observe_only": _record(_reward_trace(600), recommendation=True),
            "recommendation_p075": _record(_reward_trace(900), recommendation=True, treatment=True),
            "accept_always": _record(_reward_trace(480), recommendation=True, accept_always=True),
            "augmented_predictions": _record(_reward_trace(960), recommendation=False),
            "augmented_noise": _record(_reward_trace(720), recommendation=False),
        },
    }


def _rehash_plan(plan: dict[str, Any]) -> None:
    body = plan["plan"]
    plan["plan_sha256"] = canonical_json_sha256(body)


def _rehash_shard(shard: dict[str, Any]) -> None:
    body = shard["shard"]
    shard["shard_sha256"] = canonical_json_sha256(body)


def _rehash_artifact(artifact: dict[str, Any]) -> None:
    body = artifact["artifact"]
    artifact["artifact_sha256"] = canonical_json_sha256(body)


def _publish_synthetic_inputs(root: Path) -> dict[str, Any]:
    plan_path = root / "plan.v2.json"
    shard_directory = root / "shards"
    artifact_path = root / "evidence.v2.json"
    plan = ia_v2._build_plan_payload(
        plan_path,
        shard_directory,
        artifact_path,
        issued_unix=int(time.time()),
    )
    ia_v2._atomic_publish_new_json(plan_path, plan)
    plan_raw = canonical_json_bytes(plan)
    trace = _synthetic_trace()
    shards: list[dict[str, Any]] = []
    reservation_paths: list[Path] = []
    for seed in V2_EVIDENCE_SEEDS:
        shard = ia_v2._build_shard_payload(
            plan,
            seed,
            trace,
            copy.deepcopy(trace),
            recheck_current=False,
        )
        path = shard_directory / f"seed-{seed:03d}.v2.json"
        reservation = ia_v2._build_reservation_payload(
            plan,
            plan_raw,
            seed,
            path,
        )
        reservation_path = ia_v2._reservation_path(path)
        ia_v2._atomic_publish_new_json(reservation_path, reservation)
        reservation_paths.append(reservation_path)
        ia_v2._atomic_publish_new_json(path, shard)
        shards.append(shard)
    return {
        "plan_path": plan_path,
        "shard_directory": shard_directory,
        "artifact_path": artifact_path,
        "plan": plan,
        "trace": trace,
        "shards": shards,
        "reservation_paths": reservation_paths,
    }


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("continual_ia_v2")
    inputs = _publish_synthetic_inputs(root)
    plan_path = cast(Path, inputs["plan_path"])
    artifact_path = cast(Path, inputs["artifact_path"])
    trace = cast(dict[str, object], inputs["trace"])
    patch = pytest.MonkeyPatch()
    patch.setattr(ia_v2, "_run_seed_trace", lambda _seed: copy.deepcopy(trace))
    ia_v2._merge_shards_for_testing(
        plan_path,
        artifact_path,
        replay_runner=lambda _seed: copy.deepcopy(trace),
    )
    _raw, artifact, validation = load_artifact(artifact_path)
    assert validation.valid
    assert not validation.internally_accepted
    yield {
        "root": root,
        "artifact": artifact,
        **inputs,
    }
    patch.undo()


def test_plan_binds_exact_namespaced_protocol_and_development_only_claim_scope(
    bundle: dict[str, Any],
) -> None:
    plan = bundle["plan"]
    validation = validate_plan_payload(plan)
    assert validation.valid
    assert not validation.internally_accepted
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["namespace"] == NAMESPACE
    body = plan["plan"]
    spec = body["run_spec"]
    assert spec["seed_schedule"]["seed_ids"] == list(range(60, 90))
    assert spec["seed_schedule"]["seed_count"] == 30
    assert spec["seed_schedule"]["known_consumed_seed_ids"] == [
        *range(12),
        *range(30, 60),
    ]
    assert spec["planned_shard_count"] == 30
    assert spec["configuration"]["recommendation_acceptance_probability"] == 0.75
    assert spec["thresholds"]["evidence_seed_start"] == 60
    assert [item["id"] for item in spec["conditions"]] == list(V2_CONDITIONS)
    treatment = spec["conditions"][2]
    assert treatment == {
        "id": "recommendation_p075",
        "acceptance_probability": 0.75,
        "text": "IA recommendation treatment with exact p_accept=0.75",
    }
    provenance = body["development_selection_provenance"]
    assert provenance["role"] == "development_selection_only_nonpromoting"
    assert provenance["v1_development_and_calibration_seed_ids"] == list(range(12))
    assert provenance["p075_selection_probe_consumed_seed_ids"] == list(range(30, 60))
    probe = provenance["selected_treatment"]["development_probe"]
    assert probe == {
        "changed_action_intervention_rate": 0.1083056,
        "primary_uplift": 0.27453,
        "paired_95_percent_lower_bound": 0.25386,
    }
    policy = plan["evidence_policy"]
    assert policy["development_only"] is True
    assert policy["internal_l2_candidate_if_all_gates_pass"] is False
    assert policy["external_prerun_chronology_attested"] is False
    assert policy["automatic_registry_promotion_allowed"] is False
    assert policy["independent_replication_present"] is False
    assert body["claim_scope"]["general_step12"] == "not established"
    assert body["claim_scope"]["state_of_the_art"] == "not established"
    disabled = validate_plan_payload(plan, recheck_current=False)
    assert not disabled.valid
    assert "current source/runtime" in disabled.errors[0]
    with pytest.raises(ContinualIAV2Error, match="current source/runtime"):
        load_plan(bundle["plan_path"], recheck_current=False)


def test_plan_binds_commands_runtime_devices_static_import_closure_and_lockfiles(
    bundle: dict[str, Any],
) -> None:
    body = bundle["plan"]["plan"]
    source = body["source_manifest"]
    assert source["closure_kind"] == "static_transitive_local_python_imports"
    assert source["root_modules"] == [
        "alberta_framework.evaluation.continual_ia_v2",
        "alberta_framework.evaluation.continual_ia_v2_cli",
    ]
    assert [item["locator"] for item in source["lockfiles"]] == ["pyproject.toml", "uv.lock"]
    assert all(item["sha256"] for item in source["files"])
    runtime = body["runtime_manifest"]
    assert runtime["dependencies"]["jax"]
    assert runtime["python"]["executable"]["sha256"]
    assert runtime["python"]["flags"]
    assert isinstance(runtime["python"]["sys_path"], list)
    assert runtime["distribution_content"]["jaxlib"]["sha256"]
    assert runtime["module_origins"]["jaxlib"]["sha256"]
    assert runtime["jax"]["devices"]
    assert runtime["jax"]["config"]
    assert set(runtime["jax"]["environment"]) == set(ia_v2._RUNTIME_ENVIRONMENT_NAMES)
    assert {"process_index", "runtime_type"} <= set(runtime["jax"]["devices"][0])
    commands = body["commands"]
    assert commands["plan"][-1] == "--attest-fresh-seeds-60-89"
    assert len(commands["shards"]) == 30
    assert commands["shards"][0][-2:] == [
        "--output",
        str(bundle["shard_directory"] / "seed-060.v2.json"),
    ]
    assert body["issuance"]["prescribed_argv"] == commands["plan"]


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": 9007199254740992}',
        '{"a": 1e999}',
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_huge_numbers(text: str) -> None:
    with pytest.raises(ContinualIAV2Error):
        strict_json_loads(text)


def test_atomic_publication_is_0444_no_overwrite_and_no_symlink_traversal(
    tmp_path: Path,
) -> None:
    target = ia_v2._atomic_publish_new(tmp_path / "sealed.json", b"first\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ia_v2._atomic_publish_new(target, b"second\n")
    assert target.read_bytes() == b"first\n"

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        ia_v2._atomic_publish_new(linked / "forbidden.json", b"data\n")
    assert not (real / "forbidden.json").exists()


def test_plan_publication_requires_explicit_fresh_seed_attestation(tmp_path: Path) -> None:
    plan = tmp_path / "must-not-exist.json"
    with pytest.raises(ContinualIAV2Error, match="explicitly attested"):
        write_plan(
            plan,
            tmp_path / "shards",
            tmp_path / "artifact.json",
            attest_fresh_seeds=False,
        )
    assert not plan.exists()


def test_plan_rejects_future_timestamp_colliding_layout_and_disabled_public_binding(
    bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises(ContinualIAV2Error, match="cannot be in the future"):
        ia_v2._build_plan_payload(
            tmp_path / "future-plan.json",
            tmp_path / "future-shards",
            tmp_path / "future-artifact.json",
            issued_unix=int(time.time()) + 60,
        )
    shard_directory = tmp_path / "colliding-shards"
    with pytest.raises(ContinualIAV2Error, match="locators must be distinct"):
        ia_v2._build_plan_payload(
            tmp_path / "collision-plan.json",
            shard_directory,
            shard_directory / "seed-060.v2.json",
        )
    ancestor_plan = tmp_path / "file-used-as-directory"
    with pytest.raises(ContinualIAV2Error, match="cannot be an ancestor"):
        ia_v2._build_plan_payload(
            ancestor_plan,
            ancestor_plan / "shards",
            tmp_path / "ancestor-artifact.json",
        )
    assert not validate_plan_payload(
        bundle["plan"],
        recheck_current=False,
    ).valid


def test_plan_preflights_bound_outputs_before_publication(tmp_path: Path) -> None:
    plan_path = tmp_path / "preflight" / "plan.json"
    artifact_path = tmp_path / "preflight" / "artifact.json"
    ia_v2._atomic_publish_new(artifact_path, b"occupied\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_plan(
            plan_path,
            tmp_path / "preflight" / "shards",
            artifact_path,
            attest_fresh_seeds=True,
        )
    assert not plan_path.exists()


def test_private_lifecycle_builders_and_publishers_are_not_exported() -> None:
    assert not hasattr(ia_v2, "build_plan_payload")
    assert not hasattr(ia_v2, "build_shard_payload")
    assert not hasattr(ia_v2, "build_artifact_payload")
    assert not hasattr(ia_v2, "atomic_publish_new")
    assert not hasattr(ia_v2, "atomic_publish_new_json")
    assert not hasattr(ia_v2, "run_seed_trace")
    for name in (
        "_build_plan_payload",
        "_build_shard_payload",
        "_build_artifact_payload",
        "_atomic_publish_new",
        "_atomic_publish_new_json",
        "_run_seed_trace",
    ):
        assert name not in ia_v2.__all__


def test_public_payload_validators_wrap_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(ia_v2, "_validate_plan_or_raise", fail)
    assert not validate_plan_payload({}).valid
    monkeypatch.undo()

    monkeypatch.setattr(ia_v2, "_validate_trace_or_raise", fail)
    assert not validate_trace({}).valid
    monkeypatch.undo()

    monkeypatch.setattr(ia_v2, "_validate_shard_or_raise", fail)
    assert not validate_shard_payload({}, {}).valid
    monkeypatch.undo()

    monkeypatch.setattr(ia_v2, "_validate_artifact_or_raise", fail)
    assert not validate_artifact_payload({}).valid


def test_atomic_publication_detects_ancestor_replacement_without_leaking_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    original = ia_v2._assert_parent_locator_stable
    replaced = False

    def replace_parent(absolute: Path, directory_fd: int) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(moved)
            parent.mkdir()
        original(absolute, directory_fd)

    monkeypatch.setattr(ia_v2, "_assert_parent_locator_stable", replace_parent)
    with pytest.raises(ContinualIAV2Error, match="ancestor directory changed"):
        ia_v2._atomic_publish_new(parent / "target.json", b"payload\n")
    assert not (parent / "target.json").exists()
    assert not (moved / "target.json").exists()
    assert not list(moved.glob(".*.tmp"))


def test_atomic_publication_cleans_descriptor_owned_target_on_final_check_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_stat = os.stat
    target = tmp_path / "target.json"
    target_stats = 0

    def bad_final_link_count(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal target_stats
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == target.name and kwargs.get("dir_fd") is not None:
            target_stats += 1
            if target_stats >= 2:
                fields = list(result)
                fields[3] = 2
                return os.stat_result(fields)
        return result

    monkeypatch.setattr(os, "stat", bad_final_link_count)
    with pytest.raises(ContinualIAV2Error, match="link count changed"):
        ia_v2._atomic_publish_new(target, b"payload\n")
    assert not target.exists()


def test_atomic_publication_detects_same_byte_substitution_during_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.json"
    moved = tmp_path / "moved.json"
    original_read = ia_v2._read_regular_bytes
    substituted = False

    def substitute_after_read(
        path: Path,
        *,
        require_immutable: bool,
        max_bytes: int = ia_v2._MAX_JSON_BYTES,
    ) -> bytes:
        nonlocal substituted
        raw = original_read(
            path,
            require_immutable=require_immutable,
            max_bytes=max_bytes,
        )
        if path == target and not substituted:
            substituted = True
            target.rename(moved)
            target.write_bytes(raw)
            target.chmod(0o444)
        return raw

    monkeypatch.setattr(ia_v2, "_read_regular_bytes", substitute_after_read)
    with pytest.raises(ContinualIAV2Error, match="changed during byte verification"):
        ia_v2._atomic_publish_new(target, b"payload\n")
    assert target.read_bytes() == b"payload\n"
    assert moved.read_bytes() == b"payload\n"


def test_atomic_publication_never_unlinks_unknown_post_link_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_link = os.link

    def substitute_source_then_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        os.unlink(source, dir_fd=src_dir_fd)
        replacement_fd = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(replacement_fd, b"unknown bytes\n")
        finally:
            os.close(replacement_fd)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", substitute_source_then_link)
    target = tmp_path / "substituted.json"
    with pytest.raises(ContinualIAV2Error, match="descriptor-anchored"):
        ia_v2._atomic_publish_new(target, b"trusted bytes\n")
    assert target.read_bytes() == b"unknown bytes\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert not list(tmp_path.glob(".*.tmp"))


def test_loaders_reject_symlink_mutable_hardlink_and_fifo(
    bundle: dict[str, Any], tmp_path: Path
) -> None:
    linked = tmp_path / "linked-plan.json"
    linked.symlink_to(bundle["plan_path"])
    with pytest.raises((ContinualIAV2Error, OSError)):
        load_plan(linked)

    mutable = tmp_path / "mutable-plan.json"
    mutable.write_bytes(bundle["plan_path"].read_bytes())
    os.chmod(mutable, 0o644)
    with pytest.raises(ContinualIAV2Error, match="mode 0444"):
        load_plan(mutable)

    hardlink_source = tmp_path / "hardlink-source.json"
    hardlink_source.write_bytes(bundle["plan_path"].read_bytes())
    hardlink_source.chmod(0o444)
    hardlink_alias = tmp_path / "hardlink-alias.json"
    os.link(hardlink_source, hardlink_alias)
    with pytest.raises(ContinualIAV2Error, match="exactly one hard link"):
        load_plan(hardlink_alias)

    fifo = tmp_path / "plan.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ContinualIAV2Error, match="not a regular file"):
        load_plan(fifo)


def test_source_manifest_rejects_lockfile_change_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ia_v2._read_regular_bytes
    target = ia_v2._REPO_ROOT / "uv.lock"
    calls = 0

    def unstable_read(path: Path, **kwargs: object) -> bytes:
        nonlocal calls
        raw = original(path, **kwargs)  # type: ignore[arg-type]
        if path == target:
            calls += 1
            if calls >= 2:
                return raw + b"\n"
        return raw

    monkeypatch.setattr(ia_v2, "_read_regular_bytes", unstable_read)
    with pytest.raises(ContinualIAV2Error, match="lockfile changed"):
        ia_v2._build_source_manifest()


def test_descriptor_read_detects_locator_replacement(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_stat = os.stat
    target_name = bundle["plan_path"].name

    def replaced_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == target_name and kwargs.get("dir_fd") is not None:
            fields = list(result)
            fields[1] += 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(os, "stat", replaced_stat)
    with pytest.raises(ContinualIAV2Error, match="locator was replaced"):
        ia_v2._load_plan(bundle["plan_path"], recheck_current=False)


def test_closed_plan_schema_rejects_seed_config_bool_source_and_runtime_tampering(
    bundle: dict[str, Any],
) -> None:
    mutations: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda plan: plan.update({"unknown": True}),
        lambda plan: plan["plan"]["run_spec"]["seed_schedule"]["seed_ids"].__setitem__(0, 59),
        lambda plan: plan["plan"]["run_spec"]["configuration"].__setitem__(
            "recommendation_acceptance_probability", 0.5
        ),
        lambda plan: plan["plan"]["run_spec"]["seed_schedule"].__setitem__("seed_count", True),
    )
    for mutate in mutations:
        plan = copy.deepcopy(bundle["plan"])
        mutate(plan)
        body = plan.get("plan")
        if isinstance(body, dict):
            if isinstance(body.get("run_spec"), dict):
                body["run_spec_sha256"] = canonical_json_sha256(body["run_spec"])
            if isinstance(body.get("source_manifest"), dict):
                body["source_manifest_sha256"] = canonical_json_sha256(body["source_manifest"])
            if isinstance(body.get("runtime_manifest"), dict):
                body["runtime_manifest_sha256"] = canonical_json_sha256(body["runtime_manifest"])
            _rehash_plan(plan)
        validation = validate_plan_payload(plan)
        assert not validation.valid
        assert not validation.internally_accepted


def test_plan_rechecks_current_source_lockfiles_runtime_and_devices(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = bundle["plan"]
    body = cast(dict[str, Any], plan["plan"])
    source = copy.deepcopy(body["source_manifest"])
    source["files"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(ia_v2, "_build_source_manifest", lambda: source)
    source_validation = validate_plan_payload(plan, recheck_current=True)
    assert not source_validation.valid
    assert "source closure" in source_validation.errors[0]

    monkeypatch.setattr(
        ia_v2,
        "_build_source_manifest",
        lambda: copy.deepcopy(body["source_manifest"]),
    )
    runtime = copy.deepcopy(body["runtime_manifest"])
    runtime["jax"]["backend"] = "fabricated"
    monkeypatch.setattr(ia_v2, "_build_runtime_manifest", lambda: runtime)
    runtime_validation = validate_plan_payload(plan, recheck_current=True)
    assert not runtime_validation.valid
    assert "runtime/devices/dependencies" in runtime_validation.errors[0]

    def runtime_failure() -> dict[str, Any]:
        raise RuntimeError("simulated backend discovery failure")

    monkeypatch.setattr(ia_v2, "_build_runtime_manifest", runtime_failure)
    failure_validation = validate_plan_payload(plan, recheck_current=True)
    assert not failure_validation.valid
    assert "runtime identity discovery failed" in failure_validation.errors[0]


def test_one_seed_shard_contains_recomputable_primitives_and_exact_replay(
    bundle: dict[str, Any],
) -> None:
    shard = bundle["shards"][0]
    validation = validate_shard_payload(shard, bundle["plan"])
    assert validation.valid
    assert shard["schema"] == SHARD_SCHEMA
    body = shard["shard"]
    assert body["seed"] == 60
    assert body["deterministic_replay"]["performed"] is True
    assert body["deterministic_replay"]["exact_match"] is True
    trace = body["primitive_trace"]
    assert validate_trace(trace).valid
    treatment = trace["conditions"]["recommendation_p075"]
    assert len(treatment["rewards"]) == 1_200
    assert len(treatment["executed_actions"]) == 1_200
    assert len(treatment["pre_update_recommendations"]) == 1_200
    assert len(treatment["pre_update_partner_proposals"]) == 1_200
    assert len(treatment["accepted_recommendations"]) == 1_200
    assert len(treatment["credited_actions"]) == 1_200


def test_public_shard_validator_computationally_replays_and_rejects_forgery(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def replay(seed: int) -> dict[str, object]:
        calls.append(seed)
        return copy.deepcopy(bundle["trace"])

    monkeypatch.setattr(ia_v2, "_run_seed_trace", replay)
    assert validate_shard_payload(
        bundle["shards"][0],
        bundle["plan"],
    ).valid
    assert calls == [60]

    forged_trace = copy.deepcopy(bundle["trace"])
    forged_trace["conditions"]["partner_alone"]["rewards"][0] = 0.0
    forged = ia_v2._build_shard_payload(
        bundle["plan"],
        60,
        forged_trace,
        copy.deepcopy(forged_trace),
        recheck_current=False,
    )
    validation = validate_shard_payload(forged, bundle["plan"])
    assert not validation.valid
    assert "exact computational replay" in validation.errors[0]
    disabled = validate_shard_payload(
        bundle["shards"][0],
        bundle["plan"],
        recheck_current=False,
    )
    assert not disabled.valid
    assert "current source/runtime" in disabled.errors[0]


def test_shard_output_and_persistent_reservation_precede_seed_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reserved-shards"
    plan_path = root / "plan.json"
    shard_directory = root / "shards"
    artifact_path = root / "artifact.json"
    plan = ia_v2._build_plan_payload(
        plan_path,
        shard_directory,
        artifact_path,
        issued_unix=int(time.time()),
    )
    ia_v2._atomic_publish_new_json(plan_path, plan)
    trace = _synthetic_trace()
    calls: list[int] = []

    def runner(seed: int) -> dict[str, object]:
        calls.append(seed)
        return copy.deepcopy(trace)

    occupied_output = shard_directory / "seed-060.v2.json"
    ia_v2._atomic_publish_new(occupied_output, b"occupied\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ia_v2._write_shard(plan_path, 60, occupied_output, runner=runner)
    assert calls == []

    reserved_output = shard_directory / "seed-061.v2.json"
    reservation = ia_v2._reservation_path(reserved_output)
    ia_v2._atomic_publish_new(reservation, b"reserved\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ia_v2._write_shard(plan_path, 61, reserved_output, runner=runner)
    assert calls == []

    failed_output = shard_directory / "seed-062.v2.json"

    def fail_after_reservation(seed: int) -> dict[str, object]:
        calls.append(seed)
        raise RuntimeError("synthetic worker failure")

    with pytest.raises(ContinualIAV2Error, match="synthetic worker failure"):
        ia_v2._write_shard(
            plan_path,
            62,
            failed_output,
            runner=fail_after_reservation,
        )
    failed_reservation = ia_v2._reservation_path(failed_output)
    assert failed_reservation.exists()
    assert stat.S_IMODE(failed_reservation.stat().st_mode) == 0o444
    assert not failed_output.exists()
    calls_before_retry = list(calls)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ia_v2._write_shard(plan_path, 62, failed_output, runner=runner)
    assert calls == calls_before_retry

    successful_output = shard_directory / "seed-063.v2.json"
    published = ia_v2._write_shard(plan_path, 63, successful_output, runner=runner)
    assert published == successful_output
    assert calls[-2:] == [63, 63]
    assert ia_v2._reservation_path(successful_output).exists()
    assert stat.S_IMODE(successful_output.stat().st_mode) == 0o444


def test_merge_and_artifact_loading_require_exact_reservation_markers(
    tmp_path: Path,
) -> None:
    inputs = _publish_synthetic_inputs(tmp_path / "reservation-required")
    reservation_path = cast(list[Path], inputs["reservation_paths"])[0]
    reservation_raw = reservation_path.read_bytes()
    reservation_path.unlink()
    calls: list[int] = []

    def replay(_seed: int) -> dict[str, object]:
        calls.append(_seed)
        return copy.deepcopy(inputs["trace"])

    with pytest.raises(FileNotFoundError):
        ia_v2._merge_shards_for_testing(
            inputs["plan_path"],
            inputs["artifact_path"],
            replay_runner=replay,
        )
    assert calls == []

    ia_v2._atomic_publish_new(reservation_path, reservation_raw)
    ia_v2._merge_shards_for_testing(
        inputs["plan_path"],
        inputs["artifact_path"],
        replay_runner=replay,
    )
    assert calls == list(V2_EVIDENCE_SEEDS)
    reservation_path.unlink()
    _raw, _payload, validation = load_artifact(inputs["artifact_path"])
    assert not validation.valid
    assert not validation.internally_accepted


def test_seed_runner_uses_exact_p075_config_and_renames_only_the_v1_adapter_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    condition_names = (
        "partner_alone",
        "observe_only",
        "recommendation_p05",
        "accept_always",
        "augmented_predictions",
        "augmented_noise",
    )

    def fake_benchmark(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        results = []
        for condition in condition_names:
            recommendation = condition in {
                "observe_only",
                "recommendation_p05",
                "accept_always",
            }
            sentinel = 0 if recommendation else -1
            results.append(
                SimpleNamespace(
                    seed=60,
                    condition=condition,
                    rewards=np.zeros((1_200,), dtype=np.float64),
                    executed_actions=np.zeros((1_200,), dtype=np.int64),
                    credited_actions=np.zeros((1_200,), dtype=np.int64),
                    recommendations=np.full((1_200,), sentinel, dtype=np.int64),
                    partner_proposals=np.full((1_200,), sentinel, dtype=np.int64),
                    accepted_recommendations=np.zeros((1_200,), dtype=np.bool_),
                )
            )
        return SimpleNamespace(
            config=kwargs["config"],
            thresholds=kwargs["thresholds"],
            condition_results=tuple(results),
        )

    monkeypatch.setattr(ia_v2, "run_continual_ia_benchmark", fake_benchmark)
    trace = ia_v2._run_seed_trace(60)
    assert observed["seeds"] == (60,)
    config = cast(ContinualIAConfig, observed["config"])
    thresholds = cast(IAAcceptanceThresholds, observed["thresholds"])
    conditions = cast(dict[str, object], trace["conditions"])
    assert config.recommendation_acceptance_probability == 0.75
    assert thresholds.evidence_seed_start == 60
    assert set(conditions) == set(V2_CONDITIONS)
    assert "recommendation_p05" not in conditions

    def wrong_seed_benchmark(**kwargs: object) -> SimpleNamespace:
        report = fake_benchmark(**kwargs)
        report.condition_results[0].seed = 61
        return report

    monkeypatch.setattr(ia_v2, "run_continual_ia_benchmark", wrong_seed_benchmark)
    with pytest.raises(ContinualIAV2Error, match="different seed"):
        ia_v2._run_seed_trace(60)

    def wrong_config_benchmark(**kwargs: object) -> SimpleNamespace:
        report = fake_benchmark(**kwargs)
        report.config = ContinualIAConfig(recommendation_acceptance_probability=0.5)
        return report

    monkeypatch.setattr(ia_v2, "run_continual_ia_benchmark", wrong_config_benchmark)
    with pytest.raises(ContinualIAV2Error, match="different v2 configuration"):
        ia_v2._run_seed_trace(60)
    with pytest.raises(ContinualIAV2Error, match="reserved v2 seed"):
        ia_v2._run_seed_trace(59)


def test_trace_and_shard_fail_closed_on_bool_action_provenance_or_replay_fabrication(
    bundle: dict[str, Any],
) -> None:
    boolean = copy.deepcopy(bundle["trace"])
    boolean["conditions"]["partner_alone"]["executed_actions"][0] = True
    assert not validate_trace(boolean).valid

    nonfinite = copy.deepcopy(bundle["trace"])
    nonfinite["conditions"]["partner_alone"]["rewards"][0] = float("nan")
    assert not validate_trace(nonfinite).valid

    provenance = copy.deepcopy(bundle["trace"])
    treatment = provenance["conditions"]["recommendation_p075"]
    treatment["executed_actions"][1] = 0
    assert not validate_trace(provenance).valid

    shard = copy.deepcopy(bundle["shards"][0])
    shard["shard"]["deterministic_replay"]["exact_match"] = False
    _rehash_shard(shard)
    assert not validate_shard_payload(shard, bundle["plan"]).valid


def test_merge_recomputes_every_gate_budget_credit_identity_and_shard_hash(
    bundle: dict[str, Any],
) -> None:
    artifact = bundle["artifact"]
    validation = validate_artifact_payload(artifact)
    assert validation.valid
    assert not validation.internally_accepted
    assert artifact["schema"] == ARTIFACT_SCHEMA
    body = artifact["artifact"]
    assert [item["seed"] for item in body["shard_manifest"]] == list(range(60, 90))
    assert [item["seed"] for item in body["merge_replay_manifest"]] == list(range(60, 90))
    assert all(item["exact_match"] is True for item in body["merge_replay_manifest"])
    assert all(
        item["byte_size"] > 0 and len(item["sha256"]) == 64 for item in body["shard_manifest"]
    )
    aggregate = body["aggregate"]
    assert aggregate["primary_uplift"]["estimate"] == pytest.approx(0.25)
    assert aggregate["primary_uplift"]["lower"] == pytest.approx(0.25)
    assert aggregate["mean_changed_action_intervention_rate"] == pytest.approx(0.12)
    assert aggregate["observe_only_exact_reward_identity"] is True
    assert aggregate["observe_only_exact_action_identity"] is True
    assert aggregate["executed_action_credit_mismatches"] == 0
    assert aggregate["primary_state_budget_matched"] is True
    assert aggregate["primary_interaction_budget_matched"] is True
    acceptance = body["acceptance"]
    assert acceptance["internally_accepted"] is False
    assert acceptance["scientific_gates_passed"] is True
    assert acceptance["chronology_attestation_present"] is False
    assert acceptance["interpretation"] == {
        "if_accepted": (
            "not available in self-issued v2; an externally anchored future protocol is required"
        ),
        "if_scientific_gates_pass": (
            "reproducible development-only diagnostic, not held-out or preregistered evidence"
        ),
        "independent_replication": False,
        "general_step12": False,
        "state_of_the_art": False,
    }


def test_public_artifact_validator_replays_all_thirty_seeds(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def replay(seed: int) -> dict[str, object]:
        calls.append(seed)
        return copy.deepcopy(bundle["trace"])

    monkeypatch.setattr(ia_v2, "_run_seed_trace", replay)
    validation = validate_artifact_payload(bundle["artifact"])
    assert validation.valid
    assert calls == list(V2_EVIDENCE_SEEDS)
    disabled = validate_artifact_payload(
        bundle["artifact"],
        recheck_current=False,
    )
    assert not disabled.valid
    assert "current source/runtime" in disabled.errors[0]


def test_artifact_validator_rechecks_source_after_long_replay(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = cast(
        dict[str, Any],
        copy.deepcopy(bundle["plan"]["plan"]["source_manifest"]),
    )
    calls = 0

    def source_manifest() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        current = copy.deepcopy(expected)
        if calls >= 2:
            current["files"][0]["sha256"] = "0" * 64
        return current

    monkeypatch.setattr(ia_v2, "_build_source_manifest", source_manifest)
    validation = validate_artifact_payload(bundle["artifact"], recheck_current=True)
    assert not validation.valid
    assert calls >= 2
    assert "source closure" in validation.errors[0]


def test_artifact_validator_rereads_external_shards_after_long_replay(
    bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ia_v2._read_canonical_json
    target = cast(Path, bundle["shard_directory"]) / "seed-060.v2.json"
    reads = 0

    def unstable(path: Path) -> tuple[bytes, dict[str, Any]]:
        nonlocal reads
        raw, payload = original(path)
        if path == target:
            reads += 1
            if reads >= 2:
                return raw + b"drift", payload
        return raw, payload

    monkeypatch.setattr(ia_v2, "_read_canonical_json", unstable)
    validation = validate_artifact_payload(
        bundle["artifact"],
        locator=bundle["artifact_path"],
    )
    assert not validation.valid
    assert reads >= 2
    assert "external lifecycle bytes changed" in validation.errors[0]


def test_artifact_rechecks_current_bindings_after_final_external_reads(
    bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = ia_v2._read_canonical_json
    expected_source = cast(
        dict[str, Any],
        copy.deepcopy(bundle["plan"]["plan"]["source_manifest"]),
    )
    target = cast(Path, bundle["shard_directory"]) / "seed-060.v2.json"
    target_reads = 0
    drifted = False

    def tracked_read(path: Path) -> tuple[bytes, dict[str, Any]]:
        nonlocal target_reads, drifted
        result = original_read(path)
        if path == target:
            target_reads += 1
            if target_reads >= 2:
                drifted = True
        return result

    def current_source() -> dict[str, Any]:
        current = copy.deepcopy(expected_source)
        if drifted:
            current["files"][0]["sha256"] = "0" * 64
        return current

    monkeypatch.setattr(ia_v2, "_read_canonical_json", tracked_read)
    monkeypatch.setattr(ia_v2, "_build_source_manifest", current_source)
    validation = validate_artifact_payload(
        bundle["artifact"],
        locator=bundle["artifact_path"],
    )
    assert not validation.valid
    assert drifted
    assert "source closure" in validation.errors[0]


def test_merge_preflights_existing_output_without_replay(bundle: dict[str, Any]) -> None:
    calls: list[int] = []

    def replay(seed: int) -> dict[str, object]:
        calls.append(seed)
        return copy.deepcopy(bundle["trace"])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ia_v2._merge_shards_for_testing(
            bundle["plan_path"],
            bundle["artifact_path"],
            replay_runner=replay,
        )
    assert calls == []


def test_merge_mismatch_fails_then_cli_replays_each_seed_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _publish_synthetic_inputs(tmp_path / "fresh-merge")
    replay = copy.deepcopy(inputs["trace"])
    replay["conditions"]["partner_alone"]["rewards"][0] = 0.0
    with pytest.raises(ContinualIAV2Error, match="exact computational replay"):
        ia_v2._merge_shards_for_testing(
            inputs["plan_path"],
            inputs["artifact_path"],
            replay_runner=lambda _seed: copy.deepcopy(replay),
        )
    assert not cast(Path, inputs["artifact_path"]).exists()

    calls: list[int] = []

    def exact_replay(seed: int) -> dict[str, object]:
        calls.append(seed)
        return copy.deepcopy(inputs["trace"])

    monkeypatch.setattr(ia_v2, "_run_seed_trace", exact_replay)
    assert (
        v2_main(
            [
                "merge",
                "--plan",
                str(inputs["plan_path"]),
                "--output",
                str(inputs["artifact_path"]),
            ]
        )
        == 0
    )
    assert calls == list(V2_EVIDENCE_SEEDS)


def test_rehashed_aggregate_manifest_duplicate_and_missing_shard_tampering_fail(
    bundle: dict[str, Any],
) -> None:
    mutations = []
    aggregate = copy.deepcopy(bundle["artifact"])
    aggregate["artifact"]["aggregate"]["seed_count"] = 1
    mutations.append(aggregate)
    manifest = copy.deepcopy(bundle["artifact"])
    manifest["artifact"]["shard_manifest"][0]["sha256"] = "0" * 64
    mutations.append(manifest)
    duplicate = copy.deepcopy(bundle["artifact"])
    duplicate["artifact"]["shards"][1] = copy.deepcopy(duplicate["artifact"]["shards"][0])
    mutations.append(duplicate)
    missing = copy.deepcopy(bundle["artifact"])
    missing["artifact"]["shards"].pop()
    mutations.append(missing)
    for artifact in mutations:
        _rehash_artifact(artifact)
        validation = validate_artifact_payload(artifact)
        assert not validation.valid
        assert not validation.internally_accepted


def test_credit_failure_is_a_valid_development_rejection_not_an_invalid_artifact(
    bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = copy.deepcopy(bundle["trace"])
    treatment = trace["conditions"]["recommendation_p075"]
    treatment["credited_actions"][0] = 1 - treatment["executed_actions"][0]
    changed = ia_v2._build_shard_payload(
        bundle["plan"],
        60,
        trace,
        copy.deepcopy(trace),
        recheck_current=False,
    )
    shards = [changed, *bundle["shards"][1:]]
    artifact = ia_v2._build_artifact_payload(
        bundle["plan"],
        shards,
        recheck_current=False,
    )
    monkeypatch.setattr(
        ia_v2,
        "_run_seed_trace",
        lambda seed: copy.deepcopy(trace if seed == 60 else bundle["trace"]),
    )
    validation = validate_artifact_payload(artifact)
    assert validation.valid
    assert not validation.internally_accepted
    body = cast(dict[str, Any], artifact["artifact"])
    acceptance = cast(dict[str, Any], body["acceptance"])
    failures = [check["name"] for check in acceptance["checks"] if not check["passed"]]
    assert failures == ["executed_action_credit_mismatches"]


def test_v1_cli_stays_v1_only_and_v2_requires_explicit_subcommand() -> None:
    with pytest.raises(SystemExit) as v1_exit:
        v1_main(["v2"])
    assert v1_exit.value.code == 2
    with pytest.raises(SystemExit) as v2_exit:
        v2_main([])
    assert v2_exit.value.code == 2


def test_artifact_file_is_canonical_immutable_and_independently_loadable(
    bundle: dict[str, Any],
) -> None:
    path = bundle["artifact_path"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    raw, artifact, validation = load_artifact(path)
    assert raw == canonical_json_bytes(artifact)
    assert validation.valid
    assert not validation.internally_accepted
