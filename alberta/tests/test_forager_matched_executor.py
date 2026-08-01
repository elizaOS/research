from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

from alberta_framework.benchmarks import _forager_matched_container as container_helper
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_rng_parity as parity
from tests import test_forager_matched_open_protocol as open_protocol_fixtures
from tests import test_forager_matched_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(executor.canonical_json_bytes(value)).hexdigest()


def _receipt(
    candidate: dict[str, Any],
    *,
    entrypoint: str,
    invocation_style: str,
    result_root: str,
    patch_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": executor.MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION,
        "status": "qualified",
        "candidate_id": candidate["candidate_id"],
        "capability_descriptor_sha256": protocol_fixtures._capability_descriptor_sha256(
            candidate
        ),
        "qualification_trust_anchor_identity": candidate["runtime_binding"][
            "qualification_trust_anchor_identity"
        ],
        "source": copy.deepcopy(candidate["source"]),
        "configuration_sha256": candidate["configuration"]["derived_sha256"],
        "image_sha256": candidate["runtime_binding"]["image_sha256"],
        "runtime_profile_sha256": candidate["runtime_binding"]["runtime_profile_sha256"],
        "task_identity_sha256": candidate["runtime_binding"]["task_identity_sha256"],
        "environment_rng_schedule_sha256": candidate["environment_rng"]["schedule_sha256"],
        "rng_parity_contract_sha256": executor.RNG_PARITY_CONTRACT_SHA256,
        "entrypoint_family": candidate["entrypoint_family"],
        "entrypoint_path": entrypoint,
        "python_import_root": "src",
        "invocation_style": invocation_style,
        "result_root": result_root,
        "agent_rng_identity": candidate["agent_rng"]["identity"],
        "environment_key_shared": candidate["agent_rng"]["environment_key_shared"],
        "rng_isolation_patch_sha256": patch_sha256,
    }


def _fixture(
    tmp_path: Path,
    *,
    candidate_id: str = "alberta_causal",
) -> tuple[
    dict[str, Any],
    protocol.ForagerMatchedProtocol,
    dict[str, executor.CandidateExecutionAssets],
]:
    payload = protocol_fixtures._payload()
    if candidate_id == "isolated_ppo":
        def replace_isolated_rtu(value: Any) -> Any:
            if type(value) is dict:
                return {key: replace_isolated_rtu(item) for key, item in value.items()}
            if type(value) is list:
                return [replace_isolated_rtu(item) for item in value]
            return "isolated_ppo" if value == "isolated_rtu" else value

        payload = cast(dict[str, Any], replace_isolated_rtu(payload))
        isolated = next(
            item for item in payload["candidates"] if item["candidate_id"] == "isolated_ppo"
        )
        isolated["implementation_kind"] = "upstream_ppo_isolated_rng"
    task = parity.task_descriptor()
    payload["task"].update(
        {
            "preset": task["preset"],
            "environment_id": task["environment_id"],
            "foragax_distribution": task["foragax_distribution"],
            "foragax_version": task["foragax_version"],
            "observation_type": task["observation_type"],
            "aperture_size": task["aperture_size"],
            "task_identity_sha256": task["task_sha256"],
            "environment_rng_schedule_sha256": (
                executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
            ),
        }
    )
    payload["analysis_plan"]["metric_implementation_sha256"] = (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    payload["selection_plan"]["metric_implementation_sha256"] = (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    payload["runtime"].update(
        {
            "image_sha256": executor.QUALIFIED_IMAGE_SHA256,
            "runtime_profile_sha256": executor.QUALIFIED_RUNTIME_PROFILE_SHA256,
            "executor_qualification_receipt_sha256": (
                executor.QUALIFIED_EXECUTOR_RECEIPT_SHA256
            ),
        }
    )
    for candidate in payload["candidates"]:
        candidate["runtime_binding"].update(
            {
                "image_sha256": executor.QUALIFIED_IMAGE_SHA256,
                "runtime_profile_sha256": executor.QUALIFIED_RUNTIME_PROFILE_SHA256,
                "task_identity_sha256": task["task_sha256"],
            }
        )
        if not candidate["agent_rng"]["environment_key_shared"]:
            candidate["environment_rng"]["schedule_sha256"] = (
                executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
            )

    candidate = next(item for item in payload["candidates"] if item["candidate_id"] == candidate_id)
    source_root = tmp_path / f"{candidate_id}-source"
    source_root.mkdir()
    source_entrypoint = (
        "src/worker.py" if candidate_id == "alberta_causal" else "src/rtu_ppo.py"
    )
    entrypoint = source_root / source_entrypoint
    entrypoint.parent.mkdir()
    entrypoint.write_text("raise SystemExit('not executed by tests')\n", encoding="utf-8")
    inventory = executor.source_inventory(source_root)
    source_archive = tmp_path / f"{candidate_id}.tar"
    source_archive.write_bytes(f"archive:{candidate_id}".encode("ascii"))
    configuration = tmp_path / f"{candidate_id}.json"
    configuration.write_bytes(b"{}")
    source_update: dict[str, Any] = {
        "archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "inventory_sha256": executor.source_inventory_sha256(source_root),
    }
    candidate["source"].pop("tree_sha256", None)
    if candidate_id in {"alberta_causal", "isolated_ppo", "isolated_rtu"}:
        source_update.update(
            {
                "provenance_kind": "reviewed_snapshot",
                "tree_git_sha1": None,
                "snapshot_descriptor_sha256": _sha(f"snapshot:{candidate_id}"),
            }
        )
    else:
        source_update["snapshot_descriptor_sha256"] = None
    candidate["source"].update(source_update)
    candidate["configuration"].update(
        {
            "original_sha256": hashlib.sha256(b"{}").hexdigest(),
            "derived_sha256": hashlib.sha256(b"{}").hexdigest(),
            "allowed_transforms": [],
        }
    )
    patch_sha256 = (
        executor.QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256
        if candidate_id in {"isolated_ppo", "isolated_rtu"}
        else None
    )
    receipt = _receipt(
        candidate,
        entrypoint=source_entrypoint,
        invocation_style=(
            "alberta_single_seed_v1"
            if candidate_id == "alberta_causal"
            else "official_foragax_ppo_frozen_updates_v1"
        ),
        result_root=("results/alberta" if candidate_id == "alberta_causal" else "results/rtu"),
        patch_sha256=patch_sha256,
    )
    candidate["runtime_binding"]["capability_qualification_receipt_sha256"] = _canonical_sha(
        receipt
    )
    for item in payload["candidates"]:
        item["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            protocol_fixtures._capability_descriptor_sha256(item)
        )
    frozen = protocol.parse_forager_matched_protocol(payload)
    assets = {
        candidate_id: executor.CandidateExecutionAssets(
            candidate_id=candidate_id,
            source_root=source_root,
            source_archive=source_archive,
            source_inventory=inventory,
            original_configuration=configuration,
            configuration=configuration,
            capability_receipt=receipt,
        )
    }
    return payload, frozen, assets


def _plan(
    tmp_path: Path,
    *,
    candidate_id: str = "alberta_causal",
) -> executor.MatchedExecutionPlan:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id=candidate_id)
    return executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=(candidate_id,),
    )


def _runtime_version_payload() -> dict[str, Any]:
    return {"Client": {"Version": "qualified-test-runtime"}}


def _runtime_inspection_payload() -> dict[str, Any]:
    return {
        "Id": f"sha256:{executor.QUALIFIED_IMAGE_SHA256}",
        "Config": {
            "Labels": {
                "io.elizaos.alberta.foragax.launcher-contract": (
                    "oci-read-only-stdout-tar-v4"
                )
            }
        },
    }


def _runtime_reinspection_result(
    command: Sequence[str],
    live: executor.LiveRuntimeIdentity,
) -> executor.ProcessResult | None:
    if len(command) >= 2 and command[1] == "version":
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(live.version),
            b"",
        )
    if len(command) >= 3 and command[1:3] == ("image", "inspect"):
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(live.image_inspection),
            b"",
        )
    return None


def _runtime(
    tmp_path: Path,
    plan: executor.MatchedExecutionPlan,
) -> executor.LiveRuntimeIdentity:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "docker"
    runtime.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    runtime.chmod(0o755)

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        if "version" in command:
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(_runtime_version_payload()),
                b"",
            )
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(_runtime_inspection_payload()),
            b"",
        )

    return executor.qualify_live_runtime(plan, runtime=runtime, runner=runner)


def _scoring_output(plan: executor.MatchedExecutionPlan, seed: int) -> bytes:
    candidate = plan.candidates[0]
    sample_count = (plan.protocol.horizon + 99) // 100
    tail_start = int(0.9 * sample_count)
    return executor.canonical_json_bytes(
        {
            "schema_version": executor.SCORER_OUTPUT_SCHEMA,
            "horizon": plan.protocol.horizon,
            "seeds": [seed],
            "result_root": candidate.result_root,
            "records": [
                {
                    "archive_path": (
                        f"payload/{candidate.result_root}/data/{seed}.npz"
                    ),
                    "ema_sample_count": sample_count,
                    "ema_tail_sample_count": sample_count - tail_start,
                    "ema_tail_start_index": tail_start,
                    "final_unadjusted_ema": 0.25,
                    "fov_last_10pct_ema_auc": 0.5,
                    "npz_sha256": _sha(f"npz:{seed}"),
                    "npz_size_bytes": 4096,
                    "reward_dtype": "<f4",
                    "reward_shape": [plan.protocol.horizon],
                    "reward_sum_float64": 10.0,
                    "reward_trace_sha256": _sha(f"trace-content:{seed}"),
                    "seed": seed,
                }
            ],
        }
    )


def _execute_fixture(
    tmp_path: Path,
) -> tuple[
    executor.MatchedExecutionPlan,
    executor.SeedExecutionArtifacts,
]:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        calls.append(tuple(command))
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(0, _scoring_output(plan, seed), b"")
        return executor.ProcessResult(0, b"opaque-ustar-never-opened-on-host", b"")

    artifacts = executor.execute_seed(
        plan,
        "alberta_causal",
        seed,
        tmp_path / "raw.tar",
        live,
        runner=runner,
    )
    assert sum("run" in command for command in calls) == 2
    assert sum(len(command) >= 2 and command[1] == "version" for command in calls) == 2
    assert sum(command[1:3] == ("image", "inspect") for command in calls) == 2
    return plan, artifacts


def _complete_execution_artifacts(
    plan: executor.MatchedExecutionPlan,
    live: executor.LiveRuntimeIdentity,
) -> dict[str, list[executor.SeedExecutionArtifacts]]:
    blocks: dict[str, list[executor.SeedExecutionArtifacts]] = {}
    for candidate in plan.candidates:
        candidate_id = candidate.candidate.candidate_id
        records: list[executor.SeedExecutionArtifacts] = []
        for seed in plan.protocol.active_seeds:
            records.append(
                executor._artifact_mappings(
                    plan=plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=_sha(f"archive:{candidate_id}:{seed}"),
                    raw_archive_size=1024,
                    live_runtime=live,
                    scorer_record={
                        "fov_last_10pct_ema_auc": 0.5,
                        "npz_sha256": _sha(f"npz:{candidate_id}:{seed}"),
                        "npz_size_bytes": 4096,
                        "reward_trace_sha256": _sha(
                            f"trace-content:{candidate_id}:{seed}"
                        ),
                        "reward_dtype": "<f4",
                        "reward_shape": [plan.protocol.horizon],
                    },
                )
            )
        blocks[candidate_id] = records
    return blocks


def _two_candidate_plan(tmp_path: Path) -> executor.MatchedExecutionPlan:
    causal_root = tmp_path / "causal"
    rtu_root = tmp_path / "rtu"
    causal_root.mkdir()
    rtu_root.mkdir()
    causal_payload, _causal_protocol, causal_assets = _fixture(
        causal_root,
        candidate_id="alberta_causal",
    )
    rtu_payload, _rtu_protocol, rtu_assets = _fixture(
        rtu_root,
        candidate_id="isolated_rtu",
    )
    rtu_candidate = next(
        item for item in rtu_payload["candidates"] if item["candidate_id"] == "isolated_rtu"
    )
    for index, candidate in enumerate(causal_payload["candidates"]):
        if candidate["candidate_id"] == "isolated_rtu":
            causal_payload["candidates"][index] = copy.deepcopy(rtu_candidate)
            break
    frozen = protocol.parse_forager_matched_protocol(causal_payload)
    assets = {
        "alberta_causal": causal_assets["alberta_causal"],
        "isolated_rtu": rtu_assets["isolated_rtu"],
    }
    return executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=tuple(assets),
    )


def _refresh_receipt_index_digest(payload: dict[str, Any]) -> None:
    unsigned = copy.deepcopy(payload)
    del unsigned["payload_sha256"]
    payload["payload_sha256"] = _canonical_sha(unsigned)


def test_plan_is_content_addressed_nonexecuting_and_replayable(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    plan = executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=("alberta_causal",),
    )

    assert plan.payload["promotion_authorized"] is False
    assert plan.payload["external_verification_required"] is True
    assert plan.source_manifest_sha256 == plan.payload["source_manifest_sha256"]
    assert plan.executor_manifest_sha256 == plan.payload["executor_manifest_sha256"]
    assert plan.executor_manifest["scorer"]["sha256"] == (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    qualification = plan.executor_manifest["qualification_artifacts"]
    assert qualification["cpu_qualification"]["authority_boundary"] == {
        "endorsement_created": False,
        "endorsements_at_seal": 0,
        "gpu_qualified": False,
        "performance_claim": False,
        "seed_class": "open_development",
        "trust_profile_created": False,
        "trust_profiles_at_seal": 0,
    }
    assert qualification["rng_parity_qualification"]["status"] == (
        "content_complete_external_executor_receipt_unverified"
    )
    assert qualification["rng_parity_qualification"]["promotion_authorized"] is False
    assert b"<ACTIVE_SEED>" in plan.canonical_bytes
    assert str(tmp_path).encode() not in plan.canonical_bytes

    replayed = executor.parse_execution_plan(
        plan.canonical_bytes,
        protocol=frozen,
        assets=assets,
        expected_plan_sha256=plan.plan_sha256,
    )
    assert replayed.to_dict() == plan.to_dict()


@pytest.mark.parametrize(
    ("qualification_kind", "filename"),
    (("cpu", "receipt.v1.json"), ("rng", "receipt.json")),
)
def test_qualification_loader_rejects_artifact_drift_despite_matching_constants(
    tmp_path: Path,
    qualification_kind: str,
    filename: str,
) -> None:
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    for name in ("receipt.v1.json", "qualification.json", "environment-profile.json"):
        shutil.copy2(executor.DEFAULT_CPU_QUALIFICATION_ROOT / name, cpu_root / name)
    for name in ("plan.json", "receipt.json"):
        shutil.copy2(executor.DEFAULT_RNG_PARITY_QUALIFICATION_ROOT / name, rng_root / name)
    target = (cpu_root if qualification_kind == "cpu" else rng_root) / filename
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="frozen file digest"):
        executor.load_executor_qualification_artifacts(
            cpu_root=cpu_root,
            rng_parity_root=rng_root,
        )


def test_replay_uses_frozen_open_builder_transforms_including_search_oracle() -> None:
    frozen = open_protocol_fixtures._build()
    for candidate_id, requirement in open_protocol_fixtures._UPSTREAM_CONFIGS.items():
        relative_path, _original_sha256, derived_sha256, _transforms = requirement
        if candidate_id == "search_oracle":
            path = open_protocol_fixtures._SEARCH_ORACLE_FIXTURE
        elif candidate_id == "external_dqn_plain":
            path = Path(__file__).parent / "fixtures/forager_matched/DQN.json"
        else:
            path = open_protocol_fixtures._PINNED_CONFIG_ROOT / relative_path
        original = path.read_bytes()
        if candidate_id == "search_oracle":
            original = original.removesuffix(b"\n")
        replayed = executor._replay_configuration_transforms(
            frozen.candidate_index[candidate_id],
            original,
        )
        assert hashlib.sha256(replayed).hexdigest() == derived_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_sha256", "0" * 64),
        ("runtime_profile_sha256", "0" * 64),
        ("executor_qualification_receipt_sha256", "0" * 64),
    ],
)
def test_plan_rejects_unqualified_runtime_lock(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    payload, _frozen, assets = _fixture(tmp_path)
    payload["runtime"][field] = value
    for candidate in payload["candidates"]:
        if field in candidate["runtime_binding"]:
            candidate["runtime_binding"][field] = value
        candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            protocol_fixtures._capability_descriptor_sha256(candidate)
        )

    with pytest.raises(executor.ForagerMatchedExecutorError, match="qualified matched-current"):
        executor.build_execution_plan(
            protocol.parse_forager_matched_protocol(payload),
            assets,
            candidate_ids=("alberta_causal",),
        )


def test_plan_rejects_horizon_task_scorer_and_source_drift(tmp_path: Path) -> None:
    payload, frozen, assets = _fixture(tmp_path)
    mutators: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.__setitem__("horizon", executor.MATCHED_HORIZON - 1),
        lambda value: value["task"].__setitem__("task_identity_sha256", "0" * 64),
        lambda value: value["analysis_plan"].__setitem__(
            "metric_implementation_sha256", "0" * 64
        ),
    )
    for mutator in mutators:
        changed = copy.deepcopy(payload)
        mutator(changed)
        if changed["horizon"] != executor.MATCHED_HORIZON:
            for candidate in changed["candidates"]:
                candidate["execution_semantics"] = {
                    "rollout_steps": None,
                    "num_rollouts": None,
                    "update_semantics": "environment_step_counted",
                }
        if changed["analysis_plan"]["metric_implementation_sha256"] == "0" * 64:
            changed["selection_plan"]["metric_implementation_sha256"] = "0" * 64
            with pytest.raises(executor.ForagerMatchedExecutorError):
                executor.build_execution_plan(
                    changed,
                    assets,
                candidate_ids=("alberta_causal",),
            )

    assets["alberta_causal"].source_archive.write_bytes(b"drift")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="source archive"):
        executor.build_execution_plan(
            frozen,
            assets,
            candidate_ids=("alberta_causal",),
        )


def test_receipt_is_exact_canonical_and_not_a_self_attested_boolean(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["alberta_causal"].capability_receipt))
    )
    receipt["self_attested_trusted"] = True
    changed_assets = {
        "alberta_causal": executor.CandidateExecutionAssets(
            candidate_id="alberta_causal",
            source_root=assets["alberta_causal"].source_root,
            source_archive=assets["alberta_causal"].source_archive,
            source_inventory=assets["alberta_causal"].source_inventory,
            original_configuration=assets["alberta_causal"].original_configuration,
            configuration=assets["alberta_causal"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.build_execution_plan(
            frozen,
            changed_assets,
            candidate_ids=("alberta_causal",),
        )


def test_rtu_requires_separately_bound_isolated_rng_patch(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="isolated_rtu")
    plan = executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=("isolated_rtu",),
    )
    assert plan.candidates[0].rng_isolation_patch_sha256 == (
        executor.QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256
    )

    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["isolated_rtu"].capability_receipt))
    )
    receipt["rng_isolation_patch_sha256"] = None
    changed = {
        "isolated_rtu": executor.CandidateExecutionAssets(
            candidate_id="isolated_rtu",
            source_root=assets["isolated_rtu"].source_root,
            source_archive=assets["isolated_rtu"].source_archive,
            source_inventory=assets["isolated_rtu"].source_inventory,
            original_configuration=assets["isolated_rtu"].original_configuration,
            configuration=assets["isolated_rtu"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="source-bound RNG patch"):
        executor.build_execution_plan(frozen, changed, candidate_ids=("isolated_rtu",))


def test_isolated_ppo_rejects_any_unreviewed_rng_patch(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="isolated_ppo")
    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["isolated_ppo"].capability_receipt))
    )
    receipt["rng_isolation_patch_sha256"] = _sha("unreviewed-isolated-ppo-patch")
    frozen_payload = frozen.to_dict()
    candidate = next(
        item for item in frozen_payload["candidates"] if item["candidate_id"] == "isolated_ppo"
    )
    candidate["runtime_binding"]["capability_qualification_receipt_sha256"] = _canonical_sha(
        receipt
    )
    candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        protocol_fixtures._capability_descriptor_sha256(candidate)
    )
    changed_assets = {
        "isolated_ppo": executor.CandidateExecutionAssets(
            candidate_id="isolated_ppo",
            source_root=assets["isolated_ppo"].source_root,
            source_archive=assets["isolated_ppo"].source_archive,
            source_inventory=assets["isolated_ppo"].source_inventory,
            original_configuration=assets["isolated_ppo"].original_configuration,
            configuration=assets["isolated_ppo"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="reviewed RNG patch"):
        executor.build_execution_plan(
            frozen_payload,
            changed_assets,
            candidate_ids=("isolated_ppo",),
        )


def test_fixed_descriptive_shared_rng_ppo_remains_labeled_noninferential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="exact_ppo")
    monkeypatch.setattr(
        executor,
        "QUALIFIED_UPSTREAM_SOURCE_ARCHIVE_SHA256",
        frozen.candidate_index["exact_ppo"].source.archive_sha256,
    )
    plan = executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=("exact_ppo",),
    )
    candidate = plan.candidates[0].candidate
    assert candidate.pairing.analysis_role == "descriptive_only"
    assert candidate.pairing.eligible is False
    assert candidate.pairing.exclusion_reasons == ("shared_agent_environment_rng",)


def test_live_runtime_and_commands_pin_exact_cpu_sandbox_seed_and_horizon(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    command = executor.build_candidate_command(plan, "alberta_causal", seed, live)

    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--cpus=4.0" in command
    assert "--memory=16g" in command
    assert "--memory-swap=16g" in command
    assert f"sha256:{executor.QUALIFIED_IMAGE_SHA256}" in command
    assert f"--seed={seed}" in command
    assert f"--horizon={executor.MATCHED_HORIZON}" in command
    assert not any("--device" in item for item in command)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="active protocol seed"):
        executor.build_candidate_command(plan, "alberta_causal", seed + 10_000, live)


@pytest.mark.parametrize("failure_kind", ["timeout", "runner_error"])
def test_default_runner_cidfile_force_removes_interrupted_container(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        if len(materialized) >= 2 and materialized[1] == "run":
            cid_argument = next(item for item in materialized if item.startswith("--cidfile="))
            Path(cid_argument.split("=", 1)[1]).write_text(
                container_id + "\n",
                encoding="ascii",
            )
            if failure_kind == "timeout":
                raise subprocess.TimeoutExpired(materialized, timeout=1)
            raise OSError("synthetic runner failure")
        assert materialized == ("/usr/bin/docker", "rm", "--force", container_id)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cleanup=force_removed",
    ):
        executor._default_runner(("/usr/bin/docker", "run", "qualified-image"))

    assert len(calls) == 2


def test_execute_seed_rebinds_daemon_and_image_before_launch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    container_run_called = False

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        nonlocal container_run_called
        if len(command) >= 2 and command[1] == "version":
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(
                    {"Client": {"Version": "drifted-runtime"}}
                ),
                b"",
            )
        if len(command) >= 3 and command[1:3] == ("image", "inspect"):
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(live.image_inspection),
                b"",
            )
        container_run_called = True
        return executor.ProcessResult(0, b"must-not-run", b"")

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="daemon version or image identity changed",
    ):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            tmp_path / "raw-runtime-drift.tar",
            live,
            runner=runner,
        )

    assert container_run_called is False


def test_execute_seed_keeps_raw_archive_opaque_and_emits_hash_only_artifacts(
    tmp_path: Path,
) -> None:
    plan, artifacts = _execute_fixture(tmp_path)
    raw_bytes = (tmp_path / "raw.tar").read_bytes()

    assert raw_bytes == b"opaque-ustar-never-opened-on-host"
    assert artifacts.score == 0.5
    assert artifacts.raw_artifact["container_export_sha256"] == hashlib.sha256(
        raw_bytes
    ).hexdigest()
    assert artifacts.trace_artifact["host_reward_array_access"] == "forbidden_not_performed"
    serialized = executor.canonical_json_bytes(artifacts.to_dict())
    assert b"reward_sum_float64" not in serialized
    assert b"final_unadjusted_ema" not in serialized
    assert b"rewards" not in serialized

    parsed = executor.parse_seed_artifact_bundle(artifacts.to_dict(), plan=plan)
    assert parsed == artifacts


def test_score_seed_archive_resumes_after_scorer_failure_without_rerunning_candidate(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-resume.tar"
    raw_bytes = b"opaque-candidate-output-preserved-for-resume"

    def failing_runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(3, b"", b"")
        return executor.ProcessResult(0, raw_bytes, b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="scoring failed"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            runner=failing_runner,
        )
    assert raw_path.read_bytes() == raw_bytes

    resumed_commands: list[tuple[str, ...]] = []

    def resumed_runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        resumed_commands.append(tuple(command))
        assert "score" in command
        return executor.ProcessResult(0, _scoring_output(plan, seed), b"")

    artifacts = executor.score_seed_archive(
        plan,
        "alberta_causal",
        seed,
        raw_path,
        live,
        expected_raw_archive_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        expected_raw_archive_size=len(raw_bytes),
        runner=resumed_runner,
    )

    assert len(resumed_commands) == 1
    assert artifacts.raw_artifact["container_export_sha256"] == hashlib.sha256(
        raw_bytes
    ).hexdigest()
    assert raw_path.read_bytes() == raw_bytes


def test_score_seed_archive_rejects_expected_binding_mismatch_before_oci(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-binding-mismatch.tar"
    raw_path.write_bytes(b"opaque")
    runner_called = False

    def runner(_command: Sequence[str]) -> executor.ProcessResult:
        nonlocal runner_called
        runner_called = True
        raise AssertionError("OCI must not run for a mismatched archive binding")

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="expected execution binding",
    ):
        executor.score_seed_archive(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            expected_raw_archive_sha256=_sha("different archive"),
            expected_raw_archive_size=raw_path.stat().st_size,
            runner=runner,
        )

    assert runner_called is False


def test_scorer_output_rejects_extra_reward_values_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    payload = executor.decode_strict_json(_scoring_output(plan, seed))
    payload["records"][0]["reward_values"] = [0.0]

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(0, executor.canonical_json_bytes(payload), b"")
        return executor.ProcessResult(0, b"opaque", b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            tmp_path / "raw-extra.tar",
            live,
            runner=runner,
        )


def test_execute_seed_rejects_raw_archive_mutation_during_scoring(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-mutation.tar"

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            raw_path.unlink()
            raw_path.write_bytes(b"mutated-after-scoring-command-construction")
            return executor.ProcessResult(0, _scoring_output(plan, seed), b"")
        return executor.ProcessResult(0, b"original-opaque-ustar", b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="changed during scoring"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            runner=runner,
        )


def test_score_seed_archive_rejects_rename_swap_even_when_path_is_restored(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-swap.tar"
    replacement_path = tmp_path / "raw-swap-attacker.tar"
    held_path = tmp_path / "raw-swap-original-held.tar"
    original = b"opaque-original-archive"
    replacement = b"opaque-attacker-archive"
    assert len(original) == len(replacement)
    raw_path.write_bytes(original)
    replacement_path.write_bytes(replacement)
    expected_digest = hashlib.sha256(original).hexdigest()
    declared_digests: list[str] = []
    scorer_rejected = False
    swap_performed = False

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        nonlocal scorer_rejected, swap_performed
        if len(command) >= 2 and command[1] == "version" and not swap_performed:
            os.replace(raw_path, held_path)
            os.replace(replacement_path, raw_path)
            swap_performed = True
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        assert "score" in command
        declared_digest = next(
            item.split("=", 1)[1]
            for item in command
            if item.startswith("--raw-archive-sha256=")
        )
        declared_digests.append(declared_digest)
        mounted_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        os.replace(raw_path, replacement_path)
        os.replace(held_path, raw_path)
        if mounted_digest != declared_digest:
            scorer_rejected = True
            return executor.ProcessResult(2, b"", b"raw archive digest mismatch")
        return executor.ProcessResult(0, _scoring_output(plan, seed), b"")

    artifacts: executor.SeedExecutionArtifacts | None = None
    with pytest.raises(executor.ForagerMatchedExecutorError, match="scoring failed"):
        artifacts = executor.score_seed_archive(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            expected_raw_archive_sha256=expected_digest,
            expected_raw_archive_size=len(original),
            runner=runner,
        )

    assert artifacts is None
    assert scorer_rejected is True
    assert declared_digests == [expected_digest]
    assert raw_path.read_bytes() == original


def test_build_score_evidence_is_bridge_compatible_and_requires_external_resolver(
    tmp_path: Path,
) -> None:
    plan, first = _execute_fixture(tmp_path)
    records = []
    for position, seed in enumerate(plan.protocol.active_seeds):
        if position == 0:
            records.append(first)
            continue
        scorer_record = executor.decode_strict_json(_scoring_output(plan, seed))["records"][0]
        records.append(
            executor._artifact_mappings(
                plan=plan,
                candidate=plan.candidates[0],
                seed=seed,
                raw_archive_sha256=_sha(f"archive:{seed}"),
                raw_archive_size=1024,
                live_runtime=_runtime(tmp_path / f"runtime-{seed}", plan),
                scorer_record=scorer_record,
            )
        )
    # Evidence requires one live runtime identity for the complete candidate
    # block; replay the second hash-only artifact under the first identity.
    records[1] = executor.SeedExecutionArtifacts(
        candidate_id=records[1].candidate_id,
        seed=records[1].seed,
        score=records[1].score,
        live_runtime_identity_sha256=first.live_runtime_identity_sha256,
        raw_artifact=records[1].raw_artifact,
        trace_artifact=records[1].trace_artifact,
        scoring_record={
            **dict(records[1].scoring_record),
            "live_runtime_identity_sha256": first.live_runtime_identity_sha256,
        },
    )
    scores = executor.build_score_evidence(plan, {"alberta_causal": records})
    request = executor.build_verification_request(plan, scores)

    assert request.verification_subject_sha256 == evidence.matched_verification_subject_sha256(
        plan.protocol, scores
    )
    assert request.to_dict()["authentication_state"] == (
        "unresolved_external_verifier_required"
    )
    assert request.to_dict()["qualification_authority_boundary"][
        "endorsement_created"
    ] is False
    assert request.to_dict()["qualification_authority_boundary"][
        "trust_profile_created"
    ] is False
    assert request.to_dict()["qualification_authority_boundary"][
        "performance_claim"
    ] is False
    assert request.to_dict()["rng_parity_qualification_status"] == (
        "content_complete_external_executor_receipt_unverified"
    )
    assert request.to_dict()["qualification_promotion_authorized"] is False

    def resolver(
        subject: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        return evidence.AuthenticatedEvidenceBindings(
            stage=subject.stage,
            protocol_sha256=subject.protocol_sha256,
            score_evidence_sha256=subject.score_evidence_sha256,
            source_manifest_sha256=subject.source_manifest_sha256,
            executor_manifest_sha256=subject.executor_manifest_sha256,
            execution_closure_sha256=subject.execution_closure_sha256,
            trust_anchor_identity=subject.trust_anchor_identity,
            verification_subject_sha256=subject.verification_subject_sha256,
            verification_receipt_sha256=_sha("external-verifier-receipt"),
        )

    bindings = executor.resolve_authenticated_bindings(request, resolver)
    _frozen, validated = evidence.validate_score_evidence_against_protocol(
        plan.protocol,
        scores,
        authenticated_bindings=bindings,
        expected_candidate_ids=("alberta_causal",),
    )
    assert validated == scores


def test_verification_request_round_trip_loader_and_digest_domains(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    scores = executor.build_score_evidence(plan, artifacts)
    request = executor.build_verification_request(plan, scores)
    raw = request.canonical_bytes

    assert request.request_sha256 == hashlib.sha256(raw).hexdigest()
    assert request.request_sha256 != request.verification_subject_sha256
    assert executor.canonical_verification_request_bytes(request) == raw
    assert executor.canonical_verification_request_sha256(request) == request.request_sha256
    assert executor.parse_verification_request(raw) == request
    assert executor.parse_verification_request(raw.decode("ascii")) == request
    assert executor.parse_verification_request(request.to_dict()) == request

    path = tmp_path / "verification-request.json"
    path.write_bytes(raw)
    assert executor.load_verification_request(
        path,
        expected_request_sha256=request.request_sha256,
    ) == request

    with pytest.raises(executor.ForagerMatchedExecutorError, match="expected digest"):
        executor.load_verification_request(
            path,
            expected_request_sha256="0" * 64,
        )
    with pytest.raises(executor.ForagerMatchedExecutorError, match="canonical"):
        executor.parse_verification_request(json.dumps(request.to_dict(), indent=2))

    symlink = tmp_path / "verification-request-link.json"
    os.symlink(path, symlink)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="single-link regular"):
        executor.load_verification_request(
            symlink,
            expected_request_sha256=request.request_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("authentication_state", "authenticated", "authentication boundary"),
        ("qualification_promotion_authorized", True, "authentication boundary"),
        ("score_evidence_sha256", "0" * 64, "subject"),
        (
            "rng_parity_qualification_status",
            "verified",
            "RNG parity qualification status",
        ),
    ],
)
def test_verification_request_rejects_boundary_and_subject_drift(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    request = executor.build_verification_request(
        plan,
        executor.build_score_evidence(plan, artifacts),
    )
    payload = request.to_dict()
    payload[field] = value

    with pytest.raises(executor.ForagerMatchedExecutorError, match=error):
        executor.parse_verification_request(payload)


def test_execution_receipt_index_exposes_exact_score_evidence_preimages(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    artifacts = _complete_execution_artifacts(plan, live)

    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    scores = executor.build_score_evidence(plan, artifacts)
    score_receipt_hashes = {
        candidate.candidate_id: candidate.execution_receipt_sha256
        for candidate in scores.candidate_scores
    }

    assert receipt_index.protocol_sha256 == plan.protocol.protocol_sha256
    assert receipt_index.plan_sha256 == plan.plan_sha256
    assert receipt_index.source_manifest_sha256 == plan.source_manifest_sha256
    assert receipt_index.executor_manifest_sha256 == plan.executor_manifest_sha256
    assert receipt_index.live_runtime_identity_sha256 == live.identity_sha256
    assert receipt_index.candidate_order == ("alberta_causal",)
    assert receipt_index.to_dict()["promotion_authorized"] is False
    assert receipt_index.payload_sha256 == _canonical_sha(receipt_index.unsigned_dict())
    for item in receipt_index.execution_receipts:
        assert item.execution_receipt_sha256 == _canonical_sha(
            dict(item.receipt_payload)
        )
        assert item.execution_receipt_sha256 == score_receipt_hashes[item.candidate_id]
        assert tuple(
            seed_artifact["seed"]
            for seed_artifact in item.receipt_payload["seed_artifacts"]
        ) == plan.protocol.active_seeds

    parsed = executor.parse_execution_receipt_index(
        receipt_index.canonical_bytes,
        plan=plan,
        artifacts=artifacts,
        expected_payload_sha256=receipt_index.payload_sha256,
    )
    assert parsed == receipt_index
    index_path = tmp_path / "execution-receipt-index.json"
    index_path.write_bytes(receipt_index.canonical_bytes)
    loaded = executor.load_execution_receipt_index(
        index_path,
        plan=plan,
        artifacts=artifacts,
        expected_payload_sha256=receipt_index.payload_sha256,
    )
    assert loaded == receipt_index


def test_execution_receipt_index_rejects_self_consistent_receipt_tamper(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    payload = receipt_index.to_dict()
    entry = payload["execution_receipts"][0]
    receipt_payload = entry["receipt_payload"]
    receipt_payload["seed_artifacts"][0]["raw_artifact_sha256"] = "0" * 64
    entry["execution_receipt_sha256"] = _canonical_sha(receipt_payload)
    _refresh_receipt_index_digest(payload)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="differs from exact plan/artifacts",
    ):
        executor.parse_execution_receipt_index(
            payload,
            plan=plan,
            artifacts=artifacts,
        )


def test_execution_receipt_index_direct_construction_is_strict(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    original = receipt_index.execution_receipts[0]

    extra_field_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    extra_field_payload["unexpected"] = False
    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.IndexedExecutionReceipt(
            candidate_id=original.candidate_id,
            execution_receipt_sha256=_canonical_sha(extra_field_payload),
            receipt_payload=extra_field_payload,
        )

    reordered_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    reordered_payload["seed_artifacts"].reverse()
    reordered_receipt = executor.IndexedExecutionReceipt(
        candidate_id=original.candidate_id,
        execution_receipt_sha256=_canonical_sha(reordered_payload),
        receipt_payload=reordered_payload,
    )
    unsigned = executor._execution_receipt_index_unsigned_dict(
        schema_version=receipt_index.schema_version,
        stage=receipt_index.stage,
        protocol_sha256=receipt_index.protocol_sha256,
        plan_sha256=receipt_index.plan_sha256,
        source_manifest_sha256=receipt_index.source_manifest_sha256,
        executor_manifest_sha256=receipt_index.executor_manifest_sha256,
        live_runtime_identity_sha256=receipt_index.live_runtime_identity_sha256,
        active_seeds=receipt_index.active_seeds,
        horizon=receipt_index.horizon,
        candidate_order=receipt_index.candidate_order,
        execution_receipts=(reordered_receipt,),
    )
    with pytest.raises(executor.ForagerMatchedExecutorError, match="seed order drift"):
        executor.MatchedExecutionReceiptIndex(
            schema_version=receipt_index.schema_version,
            stage=receipt_index.stage,
            protocol_sha256=receipt_index.protocol_sha256,
            plan_sha256=receipt_index.plan_sha256,
            source_manifest_sha256=receipt_index.source_manifest_sha256,
            executor_manifest_sha256=receipt_index.executor_manifest_sha256,
            live_runtime_identity_sha256=receipt_index.live_runtime_identity_sha256,
            active_seeds=receipt_index.active_seeds,
            horizon=receipt_index.horizon,
            candidate_order=receipt_index.candidate_order,
            execution_receipts=(reordered_receipt,),
            payload_sha256=_canonical_sha(unsigned),
        )


def test_indexed_execution_receipt_freezes_caller_owned_payload(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    original = executor.build_execution_receipt_index(
        plan,
        artifacts,
    ).execution_receipts[0]
    caller_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    receipt = executor.IndexedExecutionReceipt(
        candidate_id=original.candidate_id,
        execution_receipt_sha256=original.execution_receipt_sha256,
        receipt_payload=caller_payload,
    )

    caller_payload["seed_artifacts"][0]["raw_artifact_sha256"] = "0" * 64

    assert receipt.to_dict() == original.to_dict()


def test_execution_receipt_index_binds_candidate_order(tmp_path: Path) -> None:
    plan = _two_candidate_plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    payload = receipt_index.to_dict()
    payload["candidate_order"].reverse()
    payload["execution_receipts"].reverse()
    _refresh_receipt_index_digest(payload)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="differs from exact plan/artifacts",
    ):
        executor.parse_execution_receipt_index(
            payload,
            plan=plan,
            artifacts=artifacts,
        )


def test_execution_receipt_index_rejects_live_runtime_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    records = artifacts["alberta_causal"]
    different_runtime_sha256 = _sha("different-live-runtime")
    second = records[1]
    records[1] = executor.SeedExecutionArtifacts(
        candidate_id=second.candidate_id,
        seed=second.seed,
        score=second.score,
        live_runtime_identity_sha256=different_runtime_sha256,
        raw_artifact=second.raw_artifact,
        trace_artifact=second.trace_artifact,
        scoring_record={
            **dict(second.scoring_record),
            "live_runtime_identity_sha256": different_runtime_sha256,
        },
    )

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="multiple live runtime identities",
    ):
        executor.build_execution_receipt_index(plan, artifacts)


def test_score_evidence_requires_one_live_runtime_identity_across_candidates(
    tmp_path: Path,
) -> None:
    panel_plan = _two_candidate_plan(tmp_path)
    first_runtime = _runtime(tmp_path / "runtime-a", panel_plan)
    second_runtime = replace(first_runtime, executable_sha256=_sha("different-runtime"))
    artifact_blocks: dict[str, list[executor.SeedExecutionArtifacts]] = {}
    for candidate, runtime in zip(
        panel_plan.candidates,
        (first_runtime, second_runtime),
        strict=True,
    ):
        records: list[executor.SeedExecutionArtifacts] = []
        for seed in panel_plan.protocol.active_seeds:
            scorer_record = executor.decode_strict_json(
                _scoring_output(panel_plan, seed)
            )["records"][0]
            records.append(
                executor._artifact_mappings(
                    plan=panel_plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=_sha(f"archive:{candidate.candidate.candidate_id}:{seed}"),
                    raw_archive_size=1024,
                    live_runtime=runtime,
                    scorer_record=scorer_record,
                )
            )
        artifact_blocks[candidate.candidate.candidate_id] = records

    with pytest.raises(executor.ForagerMatchedExecutorError, match="candidate panel used multiple"):
        executor.build_score_evidence(panel_plan, artifact_blocks)


def test_plan_and_artifact_loaders_require_external_digests(tmp_path: Path) -> None:
    payload, frozen, assets = _fixture(tmp_path)
    del payload
    plan = executor.build_execution_plan(
        frozen,
        assets,
        candidate_ids=("alberta_causal",),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(plan.canonical_bytes)
    assert executor.load_execution_plan(
        plan_path,
        protocol=frozen,
        assets=assets,
        expected_plan_sha256=plan.plan_sha256,
    ).plan_sha256 == plan.plan_sha256
    with pytest.raises(executor.ForagerMatchedExecutorError, match="external expected"):
        executor.load_execution_plan(
            plan_path,
            protocol=frozen,
            assets=assets,
            expected_plan_sha256="0" * 64,
        )


def test_source_inventory_rejects_symlink_and_inode_alias(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = root / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    (root / "link.py").symlink_to(target)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="symlink"):
        executor.source_inventory(root)
    (root / "link.py").unlink()
    os.link(target, root / "hard.py")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="hardlink"):
        executor.source_inventory(root)


def test_source_inventory_rejects_excessive_directory_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        (root / f"empty-{index}").mkdir()
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 10)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="directory bound"):
        executor.source_inventory(root)


def test_source_inventory_rejects_excessive_recursion_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    current = root
    for index in range(3):
        current = current / f"level-{index}"
        current.mkdir()
    monkeypatch.setattr(executor, "_MAX_SOURCE_DEPTH", 2)
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 10)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="recursion-depth bound"):
        executor.source_inventory(root)


def test_source_inventory_bounds_single_directory_before_eager_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        (root / f"entry-{index}.py").write_bytes(b"pass\n")
    monkeypatch.setattr(executor, "_MAX_SOURCE_FILES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 2)

    def forbid_listdir(_path: object) -> NoReturn:
        raise AssertionError("source inventory must accumulate through bounded scandir")

    monkeypatch.setattr(os, "listdir", forbid_listdir)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="total-entry bound"):
        executor.source_inventory(root)


def test_stable_file_and_source_inventory_reject_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="regular file"):
        executor._read_stable_file(fifo, "FIFO", maximum=1024)

    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "source.fifo")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="non-regular"):
        executor.source_inventory(source)


def test_container_safe_extract_opens_root_once_and_binds_raw_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "raw.tar"
    payloads = {
        "stdout.log": b"",
        "stderr.log": b"warning archived, not emitted by helper",
        "results/rtu/data/7.npz": b"synthetic-not-a-protected-result",
    }
    with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, BytesIO(raw))
    extraction_root = tmp_path / "extract"
    monkeypatch.setattr(container_helper, "EXTRACT_ROOT", extraction_root)

    container_helper._safe_extract(
        archive_path,
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )

    assert (extraction_root / "results/rtu/data/7.npz").read_bytes() == payloads[
        "results/rtu/data/7.npz"
    ]


def test_container_ustar_caps_include_all_header_payload_and_record_padding() -> None:
    member_sizes = [0, 1, 511, 512, 513]
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for index, size in enumerate(member_sizes):
            info = tarfile.TarInfo(f"member-{index}")
            info.size = size
            archive.addfile(info, BytesIO(b"x" * size))

    assert len(stream.getvalue()) == container_helper._ustar_stream_size(member_sizes)
    assert container_helper._maximum_ustar_stream_size(
        payload_bytes=container_helper.MAX_OUTPUT_PAYLOAD_BYTES,
        member_count=container_helper.MAX_MEMBERS,
    ) <= container_helper.MAX_RAW_ARCHIVE_BYTES
    assert container_helper.MAX_RAW_ARCHIVE_BYTES == executor._MAX_RAW_ARCHIVE_BYTES

    descriptor = os.open("/dev/null", os.O_RDONLY)
    oversized = container_helper._OutputMember(
        name="oversized",
        descriptor=descriptor,
        size=container_helper.MAX_RAW_ARCHIVE_BYTES,
        identity=(),
    )
    with pytest.raises(container_helper.ContainerError, match="USTAR stream exceeds"):
        container_helper._write_ustar([oversized])
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_container_ppo_dispatch_omits_max_steps_and_archives_workload_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    entrypoint = source / "src/rtu_ppo.py"
    entrypoint.parent.mkdir(parents=True)
    relative_input = source / "runtime-input.txt"
    relative_input.write_text("snapshot-value", encoding="utf-8")
    entrypoint.write_text(
        f"""from pathlib import Path
import sys
args = sys.argv[1:]
assert '--max_steps' not in args
seed = args[args.index('-i') + 1]
save = Path(args[args.index('--save_path') + 1])
target = save / 'rtu' / 'data' / f'{{seed}}.npz'
target.parent.mkdir(parents=True)
Path({relative_input.as_posix()!r}).write_text('mutated-host-mount')
relative_value = Path('runtime-input.txt').read_text()
target.write_bytes((' '.join(args) + ' relative=' + relative_value).encode())
print('captured stdout')
sys.stderr.write('captured upstream warning')
""",
        encoding="utf-8",
    )
    configuration = tmp_path / "configuration.json"
    configuration.write_bytes(b"{}")
    output_root = tmp_path / "output"
    monkeypatch.setattr(container_helper, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(container_helper, "EXECUTION_SOURCE_ROOT", tmp_path / "snapshot")
    monkeypatch.setattr(container_helper, "EXECUTION_CONFIG", tmp_path / "snapshot.json")
    captured: dict[str, bytes] = {}

    def capture_members(members: list[container_helper._OutputMember]) -> None:
        try:
            for member in members:
                captured[member.name] = os.pread(member.descriptor, member.size, 0)
        finally:
            for member in members:
                os.close(member.descriptor)

    monkeypatch.setattr(container_helper, "_write_ustar", capture_members)
    container_helper._run(
        Namespace(
            source_root=source.resolve().as_posix(),
            entrypoint=entrypoint.resolve().as_posix(),
            python_import_root=(source / "src").resolve().as_posix(),
            config=configuration.resolve().as_posix(),
            python=Path(sys.executable).resolve().as_posix(),
            source_inventory_sha256=executor.source_inventory_sha256(source),
            configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
            invocation_style="official_foragax_ppo_frozen_updates_v1",
            result_root="results/rtu",
            seed=7,
            horizon=executor.MATCHED_HORIZON,
        )
    )

    assert b"--max_steps" not in captured["results/rtu/data/7.npz"]
    assert b"relative=snapshot-value" in captured["results/rtu/data/7.npz"]
    assert captured["stdout.log"] == b"captured stdout\n"
    assert captured["stderr.log"] == b"captured upstream warning"
