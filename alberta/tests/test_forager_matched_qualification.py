from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as builder
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks.forager_matched_protocol import SourceBinding

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _dummy_source(key: qualification.SourceKey, root: Path, binding: SourceBinding) -> Any:
    inventory_path = root / f"{key}-inventory.json"
    archive = root / f"{key}.tar"
    return qualification._StagedSource(  # noqa: SLF001
        key=key,
        root=root,
        archive=archive,
        inventory_path=inventory_path,
        inventory={"schema_version": "test", "files": []},
        binding=binding,
        descriptor_path=None,
        patch_path=None,
    )


def _source_bindings() -> dict[qualification.SourceKey, SourceBinding]:
    return {
        "alberta": SourceBinding(
            provenance_kind="reviewed_snapshot",
            repository=builder.MATCHED_CURRENT_ALBERTA_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
            tree_git_sha1=None,
            archive_sha256=_sha("alberta-archive"),
            inventory_sha256=_sha("alberta-inventory"),
            snapshot_descriptor_sha256=_sha("alberta-snapshot"),
        ),
        "upstream": SourceBinding(
            provenance_kind="git_tree",
            repository=builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
            tree_git_sha1=builder.MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1,
            archive_sha256=builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256,
            inventory_sha256=builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256,
            snapshot_descriptor_sha256=None,
        ),
        "upstream_rng_isolated": SourceBinding(
            provenance_kind="reviewed_snapshot",
            repository=builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
            base_commit=builder.MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
            tree_git_sha1=None,
            archive_sha256=_sha("isolated-archive"),
            inventory_sha256=_sha("isolated-inventory"),
            snapshot_descriptor_sha256=_sha("isolated-snapshot"),
        ),
    }


def _configuration_sources(tmp_path: Path) -> dict[qualification.SourceKey, Any]:
    bindings = _source_bindings()
    upstream = tmp_path / "upstream-source"
    search = upstream / (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Oracle.json"
    )
    search.parent.mkdir(parents=True)
    fixture = (
        _PROJECT_ROOT / "tests/fixtures/forager_matched/Search-Oracle.json"
    ).read_bytes()
    search.write_bytes(fixture.rstrip(b"\n"))
    dqn = upstream / (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json"
    )
    dqn.parent.mkdir(parents=True, exist_ok=True)
    dqn.write_bytes((_PROJECT_ROOT / "tests/fixtures/forager_matched/DQN.json").read_bytes())
    return {
        key: _dummy_source(key, upstream if key != "alberta" else tmp_path, bindings[key])
        for key in ("alberta", "upstream", "upstream_rng_isolated")
    }


def _artifact_tree_inputs(
    root: Path,
) -> tuple[Any, dict[qualification.SourceKey, Any], dict[str, Any], dict[str, Any]]:
    for name in ("manifest.json", "manifest.json.sha256"):
        (root / name).write_bytes(b"bound")
    bindings = _source_bindings()
    sources: dict[qualification.SourceKey, Any] = {}
    for key in ("alberta", "upstream", "upstream_rng_isolated"):
        source_parent = root / "sources" / key
        source_root = source_parent / "source"
        source_root.mkdir(parents=True)
        archive = source_parent / "source.tar"
        inventory = source_parent / "inventory.json"
        archive.write_bytes(b"archive")
        inventory.write_bytes(b"inventory")
        sources[key] = qualification._StagedSource(  # noqa: SLF001
            key,
            source_root,
            archive,
            inventory,
            {},
            bindings[key],
            None,
            None,
        )
    configuration_root = root / "configurations" / "candidate"
    configuration_root.mkdir(parents=True)
    original = configuration_root / "original.json"
    derived = configuration_root / "derived.json"
    original.write_bytes(b"{}")
    derived.write_bytes(b"{}")
    configurations = {
        "candidate": qualification._MaterializedConfiguration(  # noqa: SLF001
            "candidate",
            original,
            derived,
            None,
        )
    }
    probe = root / "probes/candidate.json"
    receipt = root / "receipts/candidate.json"
    probe.parent.mkdir()
    receipt.parent.mkdir()
    probe.write_bytes(b"{}")
    receipt.write_bytes(b"{}")
    records = {
        "candidate": {
            "probe": {"path": "probes/candidate.json"},
            "capability_receipt": {"path": "receipts/candidate.json"},
        }
    }
    cpu_root = root / "executor-qualification/cpu"
    rng_root = root / "executor-qualification/rng-parity"
    cpu_root.mkdir(parents=True)
    rng_root.mkdir(parents=True)
    executor_qualifications = qualification._StagedExecutorQualifications(  # noqa: SLF001
        cpu_root,
        rng_root,
        {},
        {},
    )
    return executor_qualifications, sources, configurations, records


def _stub_qualification_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification,
        "_stage_executor_qualification_roots",
        lambda _root: object(),
    )
    monkeypatch.setattr(qualification, "_stage_sources", lambda *_args: {})
    monkeypatch.setattr(qualification, "_materialize_configurations", lambda *_args: {})
    monkeypatch.setattr(qualification, "_probe_invocations", lambda *_args: ())

    def assemble(root: Path, *_args: Any) -> None:
        (root / "staged.marker").write_bytes(b"staged")

    monkeypatch.setattr(qualification, "_assemble_and_write", assemble)


def _resources(candidate_id: str) -> dict[str, int]:
    if candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS:
        return {
            "parameter_count": 0,
            "optimizer_update_count": 0,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 100,
        }
    if candidate_id in builder.MATCHED_CURRENT_HORDE_CANDIDATE_IDS:
        return {
            "parameter_count": 1_000,
            "optimizer_update_count": builder.MATCHED_CURRENT_HORIZON,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": (
                64 if candidate_id == "alberta_horde_recurrent64" else 0
            ),
        }
    if candidate_id == "alberta_rtu_h08_taylor":
        return {
            "parameter_count": 1_000,
            "optimizer_update_count": builder.MATCHED_CURRENT_HORIZON,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 32,
        }
    if candidate_id == "search_oracle":
        return {
            "parameter_count": 0,
            "optimizer_update_count": 0,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": 0,
        }
    replay = {
        "external_dqn_ln": 10_000,
        "external_dqn_crelu": 10_000,
        "external_dqn_plain": 10_000,
        "external_dqn_redo": 1_000,
        "external_drqn_paper": 1_000,
        "isolated_ppo": 0,
        "isolated_rtu": 0,
        "exact_ppo": 0,
    }[candidate_id]
    optimizer_updates = {
        "external_dqn_ln": 124_920,
        "external_dqn_crelu": 124_920,
        "external_dqn_plain": 124_920,
        "external_dqn_redo": 124_915,
        "external_drqn_paper": 124_915,
    }.get(candidate_id, 100)
    return {
        "parameter_count": 1_000,
        "optimizer_update_count": optimizer_updates,
        "replay_capacity_transitions": replay,
        "recurrent_state_elements": (
            64 if candidate_id in {"external_drqn_paper", "isolated_rtu"} else 0
        ),
    }


def _resource_supplement(candidate_id: str) -> dict[str, Any]:
    return {
        "fixed_substrate_parameter_count": (
            1_000 if candidate_id == "alberta_horde_recurrent64" else 0
        ),
        "target_snapshot_parameter_count": (
            1_000
            if candidate_id.startswith("external_dqn")
            or candidate_id == "external_drqn_paper"
            else 0
        ),
        "non_gradient_operations": {
            "causal_nonparametric_transition_updates": (
                builder.MATCHED_CURRENT_HORIZON
                if candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
                else 0
            ),
            "target_snapshot_refreshes": 0,
            "redo_recycles": 0,
        },
    }


def _probe_payload(invocation: qualification.ProbeInvocation) -> dict[str, Any]:
    parser_identity = {
        "alberta_causal_map": "MatchedAlbertaWorkerConfiguration:CausalMapForagerConfig",
        "alberta_horde_actor_critic": (
            "MatchedAlbertaWorkerConfiguration:AlbertaForagerConfig"
        ),
        "alberta_rtu_rtrl": "MatchedAlbertaWorkerConfiguration:RTURTRLForagerConfig",
        "upstream_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_rtu_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_ppo": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_dqn_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_crelu": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_plain": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_redo_post_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_drqn": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_search_oracle": "PyExpUtils.ExperimentModel+problem.registry",
    }[invocation.implementation_kind]
    required_literals = {
        "alberta_single_seed_v1": [
            "--configuration",
            "--horizon",
            "--output-root",
            "--seed",
        ],
        "official_foragax_continuing_main_v4": [
            "--exp",
            "--idxs",
            "--max_steps",
            "--save_path",
            "-e",
            "-i",
        ],
        "official_foragax_ppo_frozen_updates_v1": [
            "--exp",
            "--idxs",
            "--save_path",
            "-e",
            "-i",
        ],
    }[invocation.invocation_style]
    if invocation.source_key == "upstream_rng_isolated":
        agent_derivation = "fold_in_isolated_agent_namespace_v1"
        agent_words = [2795197240, 2837457689]
    elif invocation.invocation_style == "official_foragax_ppo_frozen_updates_v1":
        agent_derivation = "shared_root_post_reset_split_v1"
        agent_words = [1797259609, 2579123966]
    else:
        agent_derivation = "effective_seed_constructor_input_v1"
        agent_words = [0, 0]
    return {
        "schema_version": qualification.MATCHED_CURRENT_PROBE_SCHEMA_VERSION,
        "status": "structurally_qualified_content_only",
        "candidate_id": invocation.candidate_id,
        "qualification_seed": 0,
        "source_key": invocation.source_key,
        "qualification_probe": {
            "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
            "sha256": invocation.probe_sha256,
        },
        "configuration": {
            "path": "/run/alberta/configuration.json",
            "sha256": invocation.configuration_sha256,
            "parser_identity": parser_identity,
            "round_trip_accepted": True,
        },
        "entrypoint": {
            "family": invocation.entrypoint_family,
            "path": invocation.entrypoint_path,
            "sha256": invocation.entrypoint_sha256,
            "python_ast_parsed": True,
            "ast_node_count": 1,
            "required_cli_literals": required_literals,
            "required_cli_literals_present": True,
        },
        "implementation_kind": invocation.implementation_kind,
        "resolved_agent": invocation.expected_agent,
        "result_root": invocation.result_root,
        "seed_resolution": {
            "candidate_id": invocation.candidate_id,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "requested_seed": 0,
            "stored_seed": 0,
            "offset": 0,
            "effective_seed": 0,
            "transport": invocation.seed_transport,
            "prng_impl": "threefry2x32",
            "effective_seed_key_words": [0, 0],
            "agent_rng_provenance_derivation": agent_derivation,
            "agent_rng_provenance_key_words": agent_words,
            "environment_transition_count": 0,
            "reward_array_read_count": 0,
        },
        "resources": _resources(invocation.candidate_id),
        "resource_supplement": _resource_supplement(invocation.candidate_id),
        "runtime": {
            "image_sha256": builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256,
            "python": "3.12.3",
            "jax_backend": "cpu",
            "device_platforms": ["cpu"],
        },
        "reward_blind_boundary": {
            "environment_resets": 1,
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
            "benchmark_seeds_used": [],
        },
        "authority": {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
            "promotion_authorized": False,
            "performance_claim": False,
        },
    }


def test_deterministic_archive_and_filtered_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "agent.py").write_text("value = 1\n", encoding="utf-8")
    (source / "nested").mkdir()
    executable = source / "nested/run.py"
    executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    executable.chmod(0o755)
    (source / "__pycache__").mkdir()
    (source / "__pycache__/agent.pyc").write_bytes(b"not-source")
    staged = tmp_path / "staged"
    qualification._copy_tree(source, staged, alberta_filter=True)  # noqa: SLF001
    assert not (staged / "__pycache__").exists()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    qualification._archive_tree(staged, first)  # noqa: SLF001
    qualification._archive_tree(staged, second)  # noqa: SLF001
    assert first.read_bytes() == second.read_bytes()
    assert executable.stat().st_mtime_ns != 0


def test_source_tree_rejects_excessive_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        (source / f"empty-{index}").mkdir()
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 10)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="directory bound",
    ):
        qualification._copy_tree(  # noqa: SLF001
            source,
            tmp_path / "staged",
            alberta_filter=False,
        )


def test_tar_extraction_rejects_excessive_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "directories.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for index in range(3):
            member = tarfile.TarInfo(f"empty-{index}/")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 10)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="directory bound",
    ):
        qualification._extract_git_archive(  # noqa: SLF001
            archive_path,
            tmp_path / "extracted",
        )


def test_tar_extraction_rejects_excessive_total_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "members.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for index in range(3):
            member = tarfile.TarInfo(f"empty-{index}/")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_ENTRIES", 2)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="total-member bound",
    ):
        qualification._extract_git_archive(  # noqa: SLF001
            archive_path,
            tmp_path / "extracted",
        )


def test_materializes_every_frozen_worker_envelope_and_external_transform(
    tmp_path: Path,
) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    assert tuple(configurations) == builder.MATCHED_CURRENT_CANDIDATE_IDS
    assert len(builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS) == 14
    assert len(configurations) == 23
    for candidate_id, expected in (
        builder.matched_current_alberta_configuration_fingerprints().items()
    ):
        assert hashlib.sha256(configurations[candidate_id].derived.read_bytes()).hexdigest() == (
            expected
        )
    assert configurations["isolated_ppo"].derived.read_bytes() == configurations[
        "exact_ppo"
    ].derived.read_bytes()
    assert configurations["external_dqn_crelu"].binding.derived_sha256 == (
        "4a969466bd5e7d35b937cbc7fc10354f87dc1e39b42b5ee770cf67a73bb47450"
    )
    assert configurations["external_dqn_plain"].binding.derived_sha256 == (
        "cbcaa3949b3c4e898bc615dc157272bd63603cdf25b579aff3a0dddc2d61c7bc"
    )


def test_probe_commands_are_seed_zero_networkless_and_exact_image(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"# staged probe\n")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source,
        probe_path=probe,
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        configuration=config,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    command = qualification._probe_command("docker", invocation)  # noqa: SLF001
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--seed=0" in command
    assert f"sha256:{builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}" in command
    assert any(f"source={probe}" in item for item in command)
    assert f"--probe-sha256={invocation.probe_sha256}" in command
    assert not any(str(seed) in command for seed in builder.MATCHED_CURRENT_TUNING_SEEDS)
    assert not any(str(seed) in command for seed in builder.MATCHED_CURRENT_EVALUATION_SEEDS)


def test_default_runner_spools_then_rejects_oversized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_MAX_PROBE_OUTPUT_BYTES", 8)

    def fake_run(command: Any, **kwargs: Any) -> Any:
        kwargs["stdout"].write(b"123456789")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="output exceeds",
    ):
        qualification._default_runner(("fake-runtime", "version"))  # noqa: SLF001


def test_default_runner_force_removes_probe_container_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def fake_run(command: Any, **kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        if len(calls) == 1:
            cid_argument = next(item for item in materialized if item.startswith("--cidfile="))
            Path(cid_argument.split("=", 1)[1]).write_text(container_id, encoding="ascii")
            raise subprocess.TimeoutExpired(
                materialized,
                kwargs["timeout"],
            )
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="cleanup=force_removed",
    ):
        qualification._default_runner(("fake-runtime", "run", "image"))  # noqa: SLF001
    assert calls[1] == ("fake-runtime", "rm", "--force", container_id)


def test_probe_result_accepts_only_reward_blind_unendorsed_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"# staged probe\n")
    invocation = qualification.ProbeInvocation(
        candidate_id="external_dqn_plain",
        source_key="upstream",
        source_root=source,
        probe_path=probe,
        probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
        configuration=config,
        configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        entrypoint_path="src/continuing_main.py",
        entrypoint_sha256=_sha("entrypoint"),
        entrypoint_family="continuing_main",
        implementation_kind="upstream_dqn_plain",
        invocation_style="official_foragax_continuing_main_v4",
        result_root="results/results/run/alberta/DQN",
        seed_transport="top_level_seed",
        expected_agent="DQN",
        horizon=builder.MATCHED_CURRENT_HORIZON,
    )
    payload = _probe_payload(invocation)

    def runner(_command: Any) -> qualification.QualificationProcessResult:
        return qualification.QualificationProcessResult(
            0,
            qualification._canonical_json_bytes(payload),  # noqa: SLF001
            b"",
        )

    parsed, stderr_sha = qualification._run_probe(  # noqa: SLF001
        "docker",
        invocation,
        runner,
    )
    assert parsed == payload
    assert stderr_sha == hashlib.sha256(b"").hexdigest()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="unexpected stderr",
    ):
        qualification._run_probe(  # noqa: SLF001
            "docker",
            invocation,
            lambda _command: qualification.QualificationProcessResult(
                0,
                qualification._canonical_json_bytes(payload),  # noqa: SLF001
                b"warning\n",
            ),
        )
    payload["reward_blind_boundary"]["reward_arrays_read"] = 1
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="reward-blind",
    ):
        qualification._run_probe("docker", invocation, runner)  # noqa: SLF001


@pytest.mark.parametrize(
    ("buffer_min_size", "expected"),
    [(32, 124_920), (50, 124_915)],
)
def test_replay_optimizer_count_excludes_warmup_gates(
    buffer_min_size: int,
    expected: int,
) -> None:
    assert qualification._exact_continuing_main_v4_optimizer_update_count(  # noqa: SLF001
        horizon=builder.MATCHED_CURRENT_HORIZON,
        buffer_min_size=buffer_min_size,
        update_frequency=4,
        initial_agent_step=1,
        freeze_steps=float("inf"),
    ) == expected


def test_candidate_qualification_values_close_under_open_protocol(tmp_path: Path) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    source_roots = {
        key: _dummy_source(key, source.root, _source_bindings()[key])
        for key, source in sources.items()
    }
    probes = {
        candidate_id: {
            "seed_resolution": {
                "candidate_id": candidate_id,
                "requested_seed": 0,
                "effective_seed": 0,
            },
            "resources": _resources(candidate_id),
        }
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    receipt_digests = {
        candidate_id: _sha(f"receipt:{candidate_id}")
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    values = qualification._candidate_qualifications(  # noqa: SLF001
        source_roots,
        configurations,
        probes,
        receipt_digests,
    )
    protocol = builder.build_forager_matched_open_protocol(
        runtime=qualification._runtime_qualification(),  # noqa: SLF001
        candidate_qualifications=values,
    )
    assert tuple(protocol.candidate_index) == builder.MATCHED_CURRENT_CANDIDATE_IDS
    assert len(protocol.candidates) == 23
    assert all(
        candidate.runtime_binding.qualification_trust_anchor_identity
        == qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY
        for candidate in protocol.candidates
    )


def test_capability_receipts_remain_content_only_and_patch_both_isolated_arms(
    tmp_path: Path,
) -> None:
    sources = _configuration_sources(tmp_path)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    configurations = qualification._materialize_configurations(  # noqa: SLF001
        _PROJECT_ROOT,
        sources,
        artifact_root,
    )
    for source in sources.values():
        (source.root / "src").mkdir(exist_ok=True)
        (source.root / "src/continuing_main.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('-e', '--exp')\n"
            "parser.add_argument('-i', '--idxs')\n"
            "parser.add_argument('--max_steps')\n"
            "parser.add_argument('--save_path')\n",
            encoding="utf-8",
        )
        (source.root / "src/rtu_ppo.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('-e', '--exp')\n"
            "parser.add_argument('-i', '--idxs')\n"
            "parser.add_argument('--save_path')\n",
            encoding="utf-8",
        )
    alberta_entrypoint = (
        sources["alberta"].root
        / "alberta_framework/benchmarks/_forager_matched_alberta_worker.py"
    )
    alberta_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    alberta_entrypoint.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--configuration')\n"
        "parser.add_argument('--seed')\n"
        "parser.add_argument('--horizon')\n"
        "parser.add_argument('--output-root')\n",
        encoding="utf-8",
    )
    staged_probe = (
        sources["alberta"].root
        / "alberta_framework/benchmarks/forager_matched_qualification.py"
    )
    staged_probe.write_bytes(
        (_PROJECT_ROOT / "alberta_framework/benchmarks/forager_matched_qualification.py")
        .read_bytes()
    )
    invocations = qualification._probe_invocations(sources, configurations)  # noqa: SLF001
    probes = {
        invocation.candidate_id: _probe_payload(invocation)
        for invocation in invocations
    }
    placeholder = {
        candidate_id: _sha(f"placeholder:{candidate_id}")
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    values = qualification._candidate_qualifications(  # noqa: SLF001
        sources,
        configurations,
        probes,
        placeholder,
    )
    protocol = builder.build_forager_matched_open_protocol(
        runtime=qualification._runtime_qualification(),  # noqa: SLF001
        candidate_qualifications=values,
    )
    invocation_index = {item.candidate_id: item for item in invocations}
    receipts = {
        candidate_id: qualification._capability_receipt(  # noqa: SLF001
            protocol.candidate_index[candidate_id],
            invocation_index[candidate_id],
        )
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    assert all(
        receipt["qualification_trust_anchor_identity"]
        == qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY
        for receipt in receipts.values()
    )
    assert receipts["isolated_ppo"]["rng_isolation_patch_sha256"] == (
        qualification._QUALIFIED_RNG_PATCH_SHA256  # noqa: SLF001
    )
    assert receipts["isolated_rtu"]["rng_isolation_patch_sha256"] == (
        qualification._QUALIFIED_RNG_PATCH_SHA256  # noqa: SLF001
    )
    assert receipts["exact_ppo"]["rng_isolation_patch_sha256"] is None
    assert all("signature" not in receipt for receipt in receipts.values())


def test_public_api_rejects_existing_output_before_staging(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    output = tmp_path / "already-present"
    output.mkdir()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="already exists",
    ):
        qualification.qualify_matched_current_candidates(
            _PROJECT_ROOT,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )


@pytest.mark.parametrize("input_name", ["alberta", "upstream", "cpu", "rng"])
def test_public_api_rejects_output_nested_in_every_staged_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    project = tmp_path / "project"
    alberta_source = project / "alberta_framework"
    alberta_source.mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu-qualification"
    cpu_root.mkdir()
    rng_root = tmp_path / "rng-qualification"
    rng_root.mkdir()
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    inputs = {
        "alberta": alberta_source,
        "upstream": upstream,
        "cpu": cpu_root,
        "rng": rng_root,
    }
    output = inputs[input_name] / "uncreated-parent/qualification"
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="output root overlaps",
    ):
        qualification.qualify_matched_current_candidates(
            project,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )
    assert not output.parent.exists()


def test_public_api_allows_intended_outputs_forager_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    forager_root = project / "outputs/forager"
    cpu_root = forager_root / "cpu-qualification"
    rng_root = forager_root / "rng-qualification"
    cpu_root.mkdir(parents=True)
    rng_root.mkdir()
    output = forager_root / "matched-current-qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)

    class StoppedBeforeStagingError(RuntimeError):
        pass

    def stop_before_staging(*_args: Any, **_kwargs: Any) -> str:
        raise StoppedBeforeStagingError

    monkeypatch.setattr(tempfile, "mkdtemp", stop_before_staging)
    with pytest.raises(StoppedBeforeStagingError):
        qualification.qualify_matched_current_candidates(
            project,
            upstream,
            output,
            runner=lambda _command: qualification.QualificationProcessResult(1, b"", b""),
        )
    assert not output.exists()


def test_publication_is_atomic_and_never_replaces_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_bytes(b"{}")
    destination = tmp_path / "final"
    qualification._publish_directory_no_replace(  # noqa: SLF001
        source,
        destination,
        tmp_path,
    )
    assert not source.exists()
    assert (destination / "manifest.json").read_bytes() == b"{}"

    second = tmp_path / "second-partial"
    second.mkdir()
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="created concurrently",
    ):
        qualification._publish_directory_no_replace(  # noqa: SLF001
            second,
            destination,
            tmp_path,
        )
    assert second.is_dir()
    assert (destination / "manifest.json").read_bytes() == b"{}"


def test_verified_tree_fsyncs_files_and_directories_bottom_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staging"
    nested = root / "a-nested"
    nested.mkdir(parents=True)
    nested_file = nested / "payload.bin"
    nested_file.write_bytes(b"nested")
    root_file = root / "z-root.bin"
    root_file.write_bytes(b"root")
    events: list[Path] = []

    def record_fsync(descriptor: int) -> None:
        events.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))

    monkeypatch.setattr(os, "fsync", record_fsync)
    qualification._durably_sync_verified_tree(root)  # noqa: SLF001

    assert events.count(nested_file) == 1
    assert events.count(root_file) == 1
    assert events.count(nested) == 1
    assert events.count(root) == 1
    assert events.index(nested_file) < events.index(nested) < events.index(root)
    assert events.index(root_file) < events.index(root)


@pytest.mark.parametrize(
    ("case", "limit_name", "message"),
    [
        ("files", "_MAX_QUALIFICATION_FILES", "file bound"),
        ("directories", "_MAX_QUALIFICATION_DIRECTORIES", "directory bound"),
        ("entries", "_MAX_QUALIFICATION_ENTRIES", "entry bound"),
        ("depth", "_MAX_QUALIFICATION_DEPTH", "depth bound"),
        ("bytes", "_MAX_QUALIFICATION_BYTES", "byte bound"),
    ],
)
def test_verified_tree_enforces_global_resource_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    limit_name: str,
    message: str,
) -> None:
    root = tmp_path / case
    root.mkdir()
    if case in {"files", "entries"}:
        (root / "first").write_bytes(b"")
        (root / "second").write_bytes(b"")
    elif case == "directories":
        (root / "first").mkdir()
        (root / "second").mkdir()
    elif case == "depth":
        (root / "first" / "second").mkdir(parents=True)
    else:
        (root / "payload").write_bytes(b"12")
    monkeypatch.setattr(qualification, limit_name, 1)

    with pytest.raises(qualification.ForagerMatchedQualificationError, match=message):
        qualification._durably_sync_verified_tree(root)  # noqa: SLF001


def test_verified_tree_rejects_link_and_detects_replacement_during_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    target = linked_root / "target"
    target.write_bytes(b"target")
    (linked_root / "alias").symlink_to(target.name)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="link or special",
    ):
        qualification._durably_sync_verified_tree(linked_root)  # noqa: SLF001

    race_root = tmp_path / "race"
    race_root.mkdir()
    payload = race_root / "payload"
    payload.write_bytes(b"original")
    real_fsync = os.fsync

    def replace_during_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        if Path(os.readlink(f"/proc/self/fd/{descriptor}")) == payload:
            payload.unlink()
            payload.write_bytes(b"replacement")

    monkeypatch.setattr(os, "fsync", replace_during_fsync)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="changed during traversal",
    ):
        qualification._durably_sync_verified_tree(race_root)  # noqa: SLF001


def test_qualification_validates_then_fsyncs_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    events: list[tuple[str, Path]] = []

    def load(path: Path) -> Path:
        events.append(("validate", path))
        return path

    def sync(path: Path) -> None:
        events.append(("fsync", path))

    def publish(source: Path, destination: Path, _parent: Path) -> None:
        events.append(("publish", destination))
        source.rename(destination)

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    monkeypatch.setattr(qualification, "_durably_sync_verified_tree", sync)
    monkeypatch.setattr(qualification, "_publish_directory_no_replace", publish)

    result: Any = qualification.qualify_matched_current_candidates(project, upstream, output)
    assert result == output
    assert [event for event, _path in events] == ["validate", "fsync", "publish", "validate"]
    assert events[0][1] == events[1][1]
    assert events[2][1] == output
    assert events[3][1] == output


def test_qualification_fsync_failure_cleans_staging_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "alberta_framework").mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    output = tmp_path / "qualification"
    monkeypatch.setattr(executor, "DEFAULT_CPU_QUALIFICATION_ROOT", cpu_root)
    monkeypatch.setattr(executor, "DEFAULT_RNG_PARITY_QUALIFICATION_ROOT", rng_root)
    _stub_qualification_stages(monkeypatch)
    temporary_roots: list[Path] = []

    def load(path: Path) -> Path:
        temporary_roots.append(path)
        return path

    def fail_sync(_path: Path) -> None:
        raise qualification.ForagerMatchedQualificationError("injected fsync failure")

    def publish(*_args: Any) -> None:
        raise AssertionError("publication must not run after an fsync failure")

    monkeypatch.setattr(qualification, "load_matched_current_qualification_bundle", load)
    monkeypatch.setattr(qualification, "_durably_sync_verified_tree", fail_sync)
    monkeypatch.setattr(qualification, "_publish_directory_no_replace", publish)

    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="injected fsync failure",
    ):
        qualification.qualify_matched_current_candidates(project, upstream, output)
    assert len(temporary_roots) == 1
    assert not temporary_roots[0].exists()
    assert not output.exists()


def test_artifact_tree_rejects_every_unreferenced_file(tmp_path: Path) -> None:
    executor_qualifications, sources, configurations, records = _artifact_tree_inputs(tmp_path)
    qualification._verify_qualification_artifact_tree(  # noqa: SLF001
        tmp_path,
        executor_qualifications,
        sources,
        configurations,
        records,
    )
    (tmp_path / "unexpected.npz").write_bytes(b"")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="unreferenced",
    ):
        qualification._verify_qualification_artifact_tree(  # noqa: SLF001
            tmp_path,
            executor_qualifications,
            sources,
            configurations,
            records,
        )


def test_artifact_tree_rejects_before_unbounded_non_source_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_qualifications, sources, configurations, records = _artifact_tree_inputs(tmp_path)
    monkeypatch.setattr(qualification, "_MAX_QUALIFICATION_ARTIFACT_ENTRIES", 2)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="entry bound",
    ):
        qualification._verify_qualification_artifact_tree(  # noqa: SLF001
            tmp_path,
            executor_qualifications,
            sources,
            configurations,
            records,
        )


def test_cli_converts_protocol_validator_failure_to_clean_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_arguments: Any) -> int:
        raise builder.ForagerMatchedOpenProtocolBuildError("frozen protocol drift")

    monkeypatch.setattr(qualification, "_cli", fail)
    assert qualification.main(("verify",)) == 2
    assert "frozen protocol drift" in capsys.readouterr().err


def test_copy_tree_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    os.symlink(target.name, source / "linked.py")
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="link or special",
    ):
        qualification._copy_tree(source, tmp_path / "staged", alberta_filter=True)  # noqa: SLF001
