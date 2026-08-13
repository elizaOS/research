from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pickle
import platform
import selectors
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_execution_runner as runner,
)

EXPECTED_CANDIDATE_IDS = (
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


@dataclass(frozen=True)
class InjectedTestRuntime:
    source: Path
    private: Path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _resign_receipt(receipt: dict[str, Any]) -> tuple[bytes, str]:
    body = dict(receipt)
    body.pop("receipt_body_sha256")
    receipt["receipt_body_sha256"] = _sha256(runner._canonical_json(body))
    raw = runner._canonical_json(receipt)
    return raw, _sha256(raw)


@pytest.fixture
def fake_runtime(
    tmp_path: Path,
) -> InjectedTestRuntime:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    return InjectedTestRuntime(source=source, private=private)


def _fake_source_member_identity(
    root: runner._DirectoryAnchor,
    path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    del root
    return {"path": path, "sha256": expected_sha256, "size_bytes": 123}


def _mock_exact_parent_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        platform,
        "python_version",
        lambda: runner._EXPECTED_PYTHON_VERSION,
    )
    monkeypatch.setattr(os, "getuid", lambda: runner._EXPECTED_UID)
    monkeypatch.setattr(os, "geteuid", lambda: runner._EXPECTED_UID)
    monkeypatch.setattr(os, "getgid", lambda: runner._EXPECTED_GID)
    monkeypatch.setattr(os, "getegid", lambda: runner._EXPECTED_GID)
    monkeypatch.setattr(sys, "executable", str(runner._PYTHON_EXECUTABLE))
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda _value: str(runner._PYTHON_EXECUTABLE),
    )
    for key in tuple(os.environ):
        if key in {
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONINSPECT",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        } or key.startswith("LD_AUDIT"):
            monkeypatch.delenv(key, raising=False)


def _write_private_file(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    root = Path(*path.parts[:5])
    current = path.parent
    while True:
        current.chmod(0o700)
        if current == root:
            break
        current = current.parent


def _fake_process_runner(
    spec: runner._CandidateSpec,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    result: runner.BoundedExternalProcessResult | None = None,
    omit_kind: str | None = None,
    extra_file: bool = False,
    checkpoint_file: bool = False,
    symlink_npz: bool = False,
    hardlink_artifacts: bool = False,
    substitute_save_name: bool = False,
    artifact_payloads: dict[str, bytes] | None = None,
    observations: dict[str, object] | None = None,
) -> runner.ExternalProcessRunner:
    payloads = {
        "upstream_reward_npz": b"opaque-upstream-npz",
        "upstream_results_database": b"opaque-results-database",
        "upstream_video": b"opaque-video",
        **(artifact_payloads or {}),
    }

    def fake(
        argv: tuple[str, ...],
        *,
        environment: object,
        executable_descriptor: int,
        inherited_descriptors: tuple[int, ...],
        working_directory: str,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> runner.BoundedExternalProcessResult:
        if observations is not None:
            observations.update(
                {
                    "argv": argv,
                    "environment": environment,
                    "executable_descriptor": executable_descriptor,
                    "inherited_descriptors": inherited_descriptors,
                    "working_directory": working_directory,
                    "timeout_seconds": timeout_seconds,
                    "stdout_limit_bytes": stdout_limit_bytes,
                    "stderr_limit_bytes": stderr_limit_bytes,
                }
            )
        save = Path(argv[argv.index("--save_path") + 1])
        checkpoint = Path(argv[argv.index("--checkpoint_path") + 1])
        written: dict[str, Path] = {}
        for kind, relative in runner._artifact_paths(spec):
            if kind == omit_kind:
                continue
            path = save / relative
            if kind == "upstream_reward_npz" and symlink_npz:
                target = save / "opaque-target"
                _write_private_file(target, payloads[kind])
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.symlink_to(target)
                continue
            _write_private_file(path, payloads[kind])
            written[kind] = path
        if hardlink_artifacts:
            database = written["upstream_results_database"]
            database.unlink()
            os.link(written["upstream_reward_npz"], database)
        if extra_file:
            _write_private_file(save / "unexpected.bin", b"extra")
        if checkpoint_file:
            _write_private_file(checkpoint / "unexpected.bin", b"checkpoint")
        for path in [save, *[item for item in save.rglob("*") if item.is_dir()]]:
            path.chmod(0o700)
        checkpoint.chmod(0o700)
        if substitute_save_name:
            execution_descriptor = inherited_descriptors[1]
            os.rename(
                "save",
                "save-moved",
                src_dir_fd=execution_descriptor,
                dst_dir_fd=execution_descriptor,
            )
            os.mkdir("save", mode=0o700, dir_fd=execution_descriptor)
        if result is not None:
            return result
        return runner.BoundedExternalProcessResult(0, stdout, stderr)

    return fake


def _issue(candidate_id: str, *, environment_seed: int = 11, agent_seed: int = 29) -> object:
    return runner._issue_matched_v3_external_execution_capability_for_test(
        test_only_marker=runner._INJECTED_TEST_ONLY_MARKER,
        explicit_execution_opt_in=runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
    )


def _execute(
    capability: object,
    runtime: InjectedTestRuntime,
    *,
    process_runner: runner.ExternalProcessRunner | None = None,
    cleanup_execution_root: runner._CleanupExecutionRoot | None = None,
    maximum_stdout_bytes: int = 1024,
    maximum_stderr_bytes: int = 1024,
    maximum_external_npz_bytes: int = 1024,
    maximum_results_database_bytes: int = 1024,
    maximum_ppo_video_bytes: int = 0,
) -> object:
    state = runner._EXECUTION_CAPABILITIES.get(
        cast(runner._ExecutionCapability, capability)
    )
    if state is None:
        raise AssertionError("test execution capability state disappeared")
    spec = runner._candidate(state.candidate_id)
    injected_runner = process_runner or _fake_process_runner(spec)
    return runner._execute_matched_v3_external_candidate_for_test(
        test_only_marker=runner._INJECTED_TEST_ONLY_MARKER,
        execution_capability=capability,
        workload_root=runtime.source,
        private_runtime_parent=runtime.private,
        python_executable=Path("/usr/bin/python3.12"),
        python_argv0="/usr/bin/python3.12",
        process_runner=injected_runner,
        source_member_identity=_fake_source_member_identity,
        cleanup_execution_root=cleanup_execution_root,
        timeout_seconds=30,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
        maximum_external_npz_bytes=maximum_external_npz_bytes,
        maximum_results_database_bytes=maximum_results_database_bytes,
        maximum_ppo_video_bytes=maximum_ppo_video_bytes,
    )


def _consume(outcome: object) -> runner.MatchedV3ExternalExecutionCompletion:
    return runner.consume_matched_v3_external_execution_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )


@pytest.mark.unit
def test_descriptor_is_frozen_nondecoding_and_in_container_only() -> None:
    raw = runner.canonical_external_execution_runner_descriptor_bytes()
    descriptor = runner.parse_external_execution_runner_descriptor(raw)
    assert _sha256(raw) == runner.EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256
    assert runner.external_execution_runner_descriptor_sha256() == _sha256(raw)
    assert descriptor["candidate_order"] == list(EXPECTED_CANDIDATE_IDS)
    assert descriptor["runtime_role"]["role"] == "in_container_worker_only"
    assert descriptor["runtime_role"]["host_oci_executor_required"] is True
    assert descriptor["runtime_role"]["host_oci_executor_implemented_here"] is False
    assert descriptor["artifact_contract"]["score_or_reward_magnitude_decoded"] is False
    assert descriptor["process_contract"]["process_group_termination_and_absence_check"] is True
    assert descriptor["process_contract"][
        "ambient_loader_and_python_injection_variables_rejected"
    ] is True
    assert descriptor["process_contract"]["already_started_parent_loader_closure_proven"] is False
    assert descriptor["process_contract"]["all_descendant_cleanup_proven"] is False
    assert descriptor["process_contract"]["cgroup_or_container_empty_proven"] is False
    assert descriptor["test_seam"]["injected_path_can_claim_production_runner_exact"] is False
    assert descriptor["bindings"]["result_bridge"]["imported"] is False
    assert all(value is False for value in descriptor["claims"].values())


@pytest.mark.unit
def test_candidate_mapping_is_exact_and_id_keyed() -> None:
    assert runner.EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS == EXPECTED_CANDIDATE_IDS
    descriptor = runner.external_execution_runner_descriptor()
    records = {record["candidate_id"]: record for record in descriptor["candidates"]}
    assert tuple(records) == EXPECTED_CANDIDATE_IDS
    assert records["isolated_ppo_generic"]["family"] == "ppo"
    assert records["isolated_ppo_generic"]["max_steps"] == 244
    assert records["random_policy"]["family"] == "continuing"
    assert records["random_policy"]["max_steps"] == 499_712
    assert records["search_oracle"]["configuration_sha256"] == (
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89"
    )


@pytest.mark.unit
def test_all_twelve_id_records_match_execution_contract_directly() -> None:
    contract_path = Path(runner.__file__).with_name(
        "forager_matched_v3_external_execution_contract.py"
    )
    contract_raw = contract_path.read_bytes()
    contract_tree = ast.parse(contract_raw, filename=str(contract_path))
    literal_names = {
        "_CANDIDATE_SPECS",
        "_CONTINUING_ENTRYPOINT",
        "_DERIVED_SOURCE_SHA256_BY_PATH",
        "_HORIZON",
        "_PPO_ENTRYPOINT",
        "_PPO_ROLLOUT_COUNT",
        "_PPO_VIDEO_RELATIVE_PATH",
    }
    literals: dict[str, object] = {}
    for node in contract_tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in literal_names
            and node.value is not None
        ):
            literals[node.target.id] = ast.literal_eval(node.value)
    assert set(literals) == literal_names

    candidate_specs = cast(
        tuple[tuple[str, str, str, str, str, str, str], ...],
        literals["_CANDIDATE_SPECS"],
    )
    continuing_entrypoint = cast(str, literals["_CONTINUING_ENTRYPOINT"])
    entrypoint_sha256 = cast(
        dict[str, str],
        literals["_DERIVED_SOURCE_SHA256_BY_PATH"],
    )
    horizon = cast(int, literals["_HORIZON"])
    ppo_entrypoint = cast(str, literals["_PPO_ENTRYPOINT"])
    ppo_rollout_count = cast(int, literals["_PPO_ROLLOUT_COUNT"])
    ppo_video = cast(str, literals["_PPO_VIDEO_RELATIVE_PATH"])

    worker = runner.external_execution_runner_descriptor()
    assert _sha256(contract_raw) == worker["bindings"]["execution_contract"]["source_sha256"]
    contract_by_id: dict[str, dict[str, object]] = {}
    for (
        candidate_id,
        configuration_path,
        _original_sha256,
        configuration_sha256,
        output_stem,
        family,
        _npy_descr,
    ) in candidate_specs:
        entrypoint = ppo_entrypoint if family == "ppo" else continuing_entrypoint
        max_steps = ppo_rollout_count if family == "ppo" else horizon
        configuration_suffix = configuration_path.removeprefix("experiments/")
        configuration_directory = configuration_suffix.rsplit("/", 1)[0]
        result_directory = f"results/{configuration_directory}/{output_stem}"
        artifact_paths = [
            f"{result_directory}/data/0.npz",
            f"{result_directory}/results.db",
        ]
        if family == "ppo":
            artifact_paths.append(f"{result_directory}/{ppo_video}")
        contract_by_id[candidate_id] = {
            "configuration_path": configuration_path,
            "configuration_sha256": configuration_sha256,
            "family": family,
            "entrypoint_path": entrypoint,
            "entrypoint_sha256": entrypoint_sha256[entrypoint],
            "max_steps": max_steps,
            "artifact_paths": artifact_paths,
        }

    worker_by_id = {record["candidate_id"]: record for record in worker["candidates"]}
    assert tuple(contract_by_id) == tuple(worker_by_id) == EXPECTED_CANDIDATE_IDS
    for candidate_id in EXPECTED_CANDIDATE_IDS:
        contract_record = contract_by_id[candidate_id]
        worker_record = worker_by_id[candidate_id]
        assert worker_record["configuration_path"] == contract_record["configuration_path"]
        assert worker_record["configuration_sha256"] == contract_record["configuration_sha256"]
        assert worker_record["family"] == contract_record["family"]
        assert worker_record["entrypoint_path"] == contract_record["entrypoint_path"]
        assert worker_record["entrypoint_sha256"] == contract_record["entrypoint_sha256"]
        assert worker_record["max_steps"] == contract_record["max_steps"]
        assert [item["path"] for item in worker_record["artifact_paths"]] == contract_record[
            "artifact_paths"
        ]


@pytest.mark.unit
def test_module_imports_no_project_or_result_decoding_module() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert all(not name.startswith("alberta_framework") for name in imports)
    assert "convert_external_reward_npz" not in source
    assert "canonical_reward_npz_bytes" not in source
    assert "cumulative_score" not in source


@pytest.mark.unit
def test_score_decoding_parent_boundary_rejects_preloaded_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "alberta_framework.benchmarks._forager_matched_v3_scorer"
    monkeypatch.setitem(sys.modules, name, object())
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="non-decoding boundary",
    ):
        runner._require_score_decoding_modules_absent()


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "LD_AUDIT",
    ],
)
def test_parent_runtime_rejects_every_ambient_injection_variable(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _mock_exact_parent_runtime(monkeypatch)
    monkeypatch.setenv(name, "")
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="ambient interpreter injection",
    ):
        runner._require_in_container_runtime()


@pytest.mark.unit
@pytest.mark.parametrize("effective", ["uid", "gid"])
def test_parent_runtime_rejects_real_effective_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    effective: str,
) -> None:
    _mock_exact_parent_runtime(monkeypatch)
    if effective == "uid":
        monkeypatch.setattr(os, "geteuid", lambda: runner._EXPECTED_UID - 1)
    else:
        monkeypatch.setattr(os, "getegid", lambda: runner._EXPECTED_GID - 1)
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="runtime profile",
    ):
        runner._require_in_container_runtime()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("acknowledgement", "candidate_id", "environment_seed", "agent_seed", "match"),
    [
        ("wrong", "external_dqn_plain", 1, 2, "explicit opt-in"),
        (True, "external_dqn_plain", 1, 2, "explicit opt-in"),
        (
            runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
            "not-a-candidate",
            1,
            2,
            "frozen external candidate",
        ),
        (
            runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
            "external_dqn_plain",
            True,
            2,
            "uint31",
        ),
        (
            runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
            "external_dqn_plain",
            1,
            2**31,
            "uint31",
        ),
    ],
)
def test_capability_issuance_fails_closed(
    acknowledgement: object,
    candidate_id: str,
    environment_seed: object,
    agent_seed: object,
    match: str,
) -> None:
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match=match):
        runner.issue_matched_v3_external_execution_capability(
            explicit_execution_opt_in=acknowledgement,  # type: ignore[arg-type]
            candidate_id=candidate_id,
            environment_seed=environment_seed,  # type: ignore[arg-type]
            agent_seed=agent_seed,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_execution_capability_is_opaque_noncopyable_and_pid_bound(
    monkeypatch: pytest.MonkeyPatch,
    fake_runtime: InjectedTestRuntime,
) -> None:
    capability = _issue("external_dqn_plain")
    with pytest.raises(TypeError, match="copied"):
        copy.copy(capability)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(capability)
    state = runner._EXECUTION_CAPABILITIES[
        cast(runner._ExecutionCapability, capability)
    ]
    monkeypatch.setattr(state, "pid", os.getpid() + 1)
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="PID"):
        _execute(capability, fake_runtime)
    assert state.status == "consumed"


@pytest.mark.unit
def test_continuing_happy_path_is_exact_and_single_use(
    fake_runtime: InjectedTestRuntime,
) -> None:
    observations: dict[str, object] = {}
    spec = runner._candidate("external_dqn_plain")
    process_runner = _fake_process_runner(
        spec,
        stdout=b"",
        stderr=b"",
        observations=observations,
    )
    capability = _issue(spec.candidate_id)
    outcome = _execute(capability, fake_runtime, process_runner=process_runner)
    assert list(fake_runtime.private.iterdir()) == []
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="consumed"):
        _execute(capability, fake_runtime)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(outcome)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(outcome)
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="opt-in"):
        runner.consume_matched_v3_external_execution_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=False,
        )
    completion = _consume(outcome)
    assert completion.candidate_id == spec.candidate_id
    assert completion.environment_seed == 11
    assert completion.agent_seed == 29
    assert completion.upstream_reward_npz == b"opaque-upstream-npz"
    assert completion.upstream_results_database == b"opaque-results-database"
    assert completion.upstream_video is None
    receipt = runner.parse_matched_v3_external_execution_receipt(
        completion.execution_receipt_bytes,
        expected_receipt_sha256=completion.execution_receipt_sha256,
        candidate_id=completion.candidate_id,
        environment_seed=completion.environment_seed,
        agent_seed=completion.agent_seed,
        upstream_reward_npz=completion.upstream_reward_npz,
        upstream_results_database=completion.upstream_results_database,
        upstream_video=completion.upstream_video,
        stdout=completion.stdout,
        stderr=completion.stderr,
    )
    assert receipt["runner_content_handling"]["reward_magnitudes_decoded"] is False
    assert receipt["inventory"]["checkpoint_root_empty_after_execution"] is True
    assert receipt["runtime"]["production_runner_exact"] is False
    assert receipt["runtime"]["closure_integrity_checked"] is True
    assert receipt["runtime"]["test_only_process_runner_injected"] is True
    assert receipt["runtime"]["process_group_cleanup_completed"] is False
    assert receipt["runtime"]["parent_runtime_profile_checked"] is False
    assert receipt["runtime"]["parent_ambient_injection_variables_absent"] is False
    assert receipt["runtime"]["all_descendant_cleanup_proven"] is False
    assert receipt["runtime"]["cgroup_or_container_empty_proven"] is False
    assert receipt["runtime"]["future_host_cgroup_or_container_empty_proof_required"] is True
    argv = observations["argv"]
    assert isinstance(argv, tuple)
    assert argv[2] == "src/continuing_main.py"
    assert argv[4] == spec.configuration_path
    assert argv[-1] == "--silent"
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="consumed"):
        _consume(outcome)


@pytest.mark.unit
def test_ppo_happy_path_requires_and_retains_exact_video(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("isolated_rtu_paper_scale")
    observations: dict[str, object] = {}
    process_runner = _fake_process_runner(
        spec,
        observations=observations,
    )
    outcome = _execute(
        _issue(spec.candidate_id),
        fake_runtime,
        process_runner=process_runner,
        maximum_ppo_video_bytes=1024,
    )
    completion = _consume(outcome)
    assert completion.upstream_video == b"opaque-video"
    argv = observations["argv"]
    assert isinstance(argv, tuple)
    assert argv[2] == "src/rtu_ppo.py"
    assert argv[12] == "244"
    receipt = json.loads(completion.execution_receipt_bytes)
    assert [record["kind"] for record in receipt["artifacts"]] == [
        "upstream_reward_npz",
        "upstream_results_database",
        "upstream_video",
    ]


@pytest.mark.unit
def test_plain_completion_paths_and_bytes_cannot_forge_outcome() -> None:
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="authentic"):
        _consume(object())
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="authentic"):
        _consume(b"serialized completion")


@pytest.mark.unit
def test_outcome_state_is_privately_sealed_and_paths_are_irrevocably_exclusive(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    outcome = _execute(
        _issue(spec.candidate_id),
        fake_runtime,
        process_runner=_fake_process_runner(spec),
    )
    state = runner._OUTCOME_CAPABILITIES[cast(runner._OutcomeCapability, outcome)]
    assert type(state.sealed_payload) is runner._SealedExternalExecutionPayload
    assert not hasattr(state, "completion")
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="consumer.*loaded before the runner",
    ):
        runner._consume_outcome_for_captured_external_consumer(
            outcome_capability=outcome,
            publication_parent=fake_runtime.private,
            expected_candidate_id=spec.candidate_id,
            expected_environment_seed=11,
            expected_agent_seed=29,
            expected_environment_seed_commitment_sha256="1" * 64,
            expected_agent_seed_commitment_sha256="2" * 64,
            expected_qualification_plan_sha256="3" * 64,
            expected_qualification_case_manifest_sha256="4" * 64,
            expected_publisher_source_tree_sha256="5" * 64,
            expected_workload_source_tree_sha256="6" * 64,
            expected_staging_manifest_sha256="7" * 64,
            maximum_publication_total_bytes=1024 * 1024,
            explicit_publication_opt_in=True,
        )
    assert state.status == "consumed"
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="consumed"):
        _consume(outcome)


@pytest.mark.unit
def test_outcome_is_pid_bound(
    monkeypatch: pytest.MonkeyPatch,
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    outcome = _execute(
        _issue(spec.candidate_id),
        fake_runtime,
        process_runner=_fake_process_runner(spec),
    )
    state = runner._OUTCOME_CAPABILITIES[cast(runner._OutcomeCapability, outcome)]
    monkeypatch.setattr(state, "pid", os.getpid() + 1)
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="PID"):
        _consume(outcome)
    assert state.status == "consumed"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("returncode", "error"),
    [
        (False, TypeError),
        (runner._MAX_PROCESS_RETURNCODE + 1, ValueError),
        (runner._MIN_PROCESS_RETURNCODE - 1, ValueError),
    ],
)
def test_process_result_rejects_noninteger_or_implausible_returncode(
    returncode: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        runner.BoundedExternalProcessResult(returncode, b"", b"")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.usefixtures("fake_runtime")
@pytest.mark.parametrize(
    ("process_result", "match"),
    [
        (runner.BoundedExternalProcessResult(7, b"", b""), "nonzero"),
        (runner.BoundedExternalProcessResult(0, b"", b"", timed_out=True), "timed out"),
        (
            runner.BoundedExternalProcessResult(
                0, b"", b"", output_limit_exceeded=True
            ),
            "output ceiling",
        ),
    ],
)
def test_process_failure_flags_fail_closed_and_cleanup(
    process_result: runner.BoundedExternalProcessResult,
    match: str,
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match=match):
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec, result=process_result),
        )
    assert list(fake_runtime.private.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runner_factory", "match"),
    [
        (lambda spec: _fake_process_runner(spec, omit_kind="upstream_reward_npz"), "inventory"),
        (lambda spec: _fake_process_runner(spec, extra_file=True), "inventory"),
        (lambda spec: _fake_process_runner(spec, checkpoint_file=True), "checkpoint"),
        (
            lambda spec: _fake_process_runner(spec, symlink_npz=True),
            "ownership or mode|link or special",
        ),
        (
            lambda spec: _fake_process_runner(spec, hardlink_artifacts=True),
            "inode alias|link or special",
        ),
        (
            lambda spec: _fake_process_runner(spec, substitute_save_name=True),
            "substituted",
        ),
    ],
)
def test_artifact_inventory_substitution_and_checkpoint_fail_closed(
    runner_factory: Callable[[runner._CandidateSpec], runner.ExternalProcessRunner],
    match: str,
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match=match):
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=runner_factory(spec),
        )
    assert list(fake_runtime.private.iterdir()) == []


@pytest.mark.unit
def test_artifact_and_stream_bounds_fail_closed(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    artifact_runner = _fake_process_runner(
        spec,
        artifact_payloads={"upstream_reward_npz": b"12345"},
    )
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="bounded"):
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=artifact_runner,
            maximum_external_npz_bytes=4,
        )

    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="ceiling"):
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec, stdout=b"x"),
            maximum_stdout_bytes=0,
        )


@pytest.mark.unit
def test_family_specific_video_ceiling_is_exact(
    fake_runtime: InjectedTestRuntime,
) -> None:
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="exactly zero"):
        _execute(
            _issue("external_dqn_plain"),
            fake_runtime,
            maximum_ppo_video_bytes=1,
        )
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="exact integer"):
        _execute(
            _issue("isolated_ppo_generic"),
            fake_runtime,
            maximum_ppo_video_bytes=0,
        )


@pytest.mark.unit
def test_runner_exception_marks_process_state_uncertain(
    fake_runtime: InjectedTestRuntime,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> runner.BoundedExternalProcessResult:
        raise RuntimeError("fake boundary failure")

    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError) as captured:
        _execute(
            _issue("external_dqn_plain"),
            fake_runtime,
            process_runner=fail,
        )
    assert captured.value.process_state_uncertain is True
    assert list(fake_runtime.private.iterdir()) == []


@pytest.mark.unit
def test_default_process_boundary_runs_one_tiny_bounded_child(tmp_path: Path) -> None:
    executable = os.open("/usr/bin/python3.12", runner._file_flags())
    working = os.open(tmp_path, runner._directory_flags())
    try:
        result = runner._default_process_runner(
            (
                "/usr/bin/python3.12",
                "-B",
                "-c",
                "import os; os.write(1, b'bounded-child')",
            ),
            environment={"PATH": "/usr/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            executable_descriptor=executable,
            inherited_descriptors=(working,),
            working_directory=f"/proc/self/fd/{working}",
            timeout_seconds=5,
            stdout_limit_bytes=64,
            stderr_limit_bytes=64,
        )
    finally:
        os.close(working)
        os.close(executable)
    assert result == runner.BoundedExternalProcessResult(
        returncode=0,
        stdout=b"bounded-child",
        stderr=b"",
        timed_out=False,
        output_limit_exceeded=False,
    )


@pytest.mark.unit
def test_default_process_boundary_kills_and_reaps_on_timeout(tmp_path: Path) -> None:
    executable = os.open("/usr/bin/python3.12", runner._file_flags())
    working = os.open(tmp_path, runner._directory_flags())
    try:
        result = runner._default_process_runner(
            (
                "/usr/bin/python3.12",
                "-B",
                "-c",
                "import time; time.sleep(30)",
            ),
            environment={"PATH": "/usr/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            executable_descriptor=executable,
            inherited_descriptors=(working,),
            working_directory=f"/proc/self/fd/{working}",
            timeout_seconds=1,
            stdout_limit_bytes=64,
            stderr_limit_bytes=64,
        )
    finally:
        os.close(working)
        os.close(executable)
    assert result.timed_out is True
    assert result.output_limit_exceeded is False
    assert result.returncode < 0


@pytest.mark.unit
def test_cleanup_uncertainty_blocks_outcome(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    original = runner._PRODUCTION_CLEANUP_EXECUTION_ROOT

    def uncertain(
        parent: runner._DirectoryAnchor,
        name: str,
        root: runner._DirectoryAnchor,
    ) -> None:
        original(parent, name, root)
        raise runner.ForagerMatchedV3ExternalExecutionRunnerError(
            "injected cleanup uncertainty",
            filesystem_state_uncertain=False,
        )

    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError) as captured:
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec),
            cleanup_execution_root=uncertain,
        )
    assert captured.value.filesystem_state_uncertain is True
    assert any("injected cleanup uncertainty" in note for note in captured.value.__notes__)


@pytest.mark.unit
def test_linked_absolute_path_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError):
        runner._open_absolute_directory(linked, "linked root")


@pytest.mark.unit
def test_inventory_and_cleanup_limits_fail_closed_with_filesystem_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    parent = runner._open_absolute_directory(private, "bounded cleanup parent")
    name, root = runner._create_execution_root(parent)
    try:
        artifact = Path(f"/proc/self/fd/{root.descriptor}/artifact.bin")
        artifact.write_bytes(b"opaque")
        artifact.chmod(0o600)
        with monkeypatch.context() as patch:
            patch.setattr(
                runner,
                "_remove_tree_contents",
                lambda _descriptor: (_ for _ in ()).throw(RuntimeError("injected")),
            )
            with pytest.raises(
                runner.ForagerMatchedV3ExternalExecutionRunnerError,
                match="cleanup failed",
            ) as arbitrary_failure:
                runner._cleanup_execution_root(parent, name, root)
            assert arbitrary_failure.value.filesystem_state_uncertain is True
        with monkeypatch.context() as patch:
            patch.setattr(runner, "_MAX_INVENTORY_ENTRIES", 0)
            with pytest.raises(
                runner.ForagerMatchedV3ExternalExecutionRunnerError,
                match="inventory entry count",
            ):
                runner._inventory_tree(root.descriptor)
        with monkeypatch.context() as patch:
            patch.setattr(runner, "_MAX_CLEANUP_ENTRIES", 0)
            with pytest.raises(
                runner.ForagerMatchedV3ExternalExecutionRunnerError,
                match="cleanup entry count",
            ) as captured:
                runner._cleanup_execution_root(parent, name, root)
            assert captured.value.filesystem_state_uncertain is True
        runner._cleanup_execution_root(parent, name, root)
    finally:
        if root.descriptor >= 0:
            root.close()
        parent.close()


@pytest.mark.unit
def test_receipt_rejects_digest_tamper_and_resigned_runtime_tamper(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    completion = _consume(
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec),
        )
    )
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="full-file"):
        runner.parse_matched_v3_external_execution_receipt(
            completion.execution_receipt_bytes + b"x",
            expected_receipt_sha256=completion.execution_receipt_sha256,
            candidate_id=completion.candidate_id,
            environment_seed=completion.environment_seed,
            agent_seed=completion.agent_seed,
            upstream_reward_npz=completion.upstream_reward_npz,
            upstream_results_database=completion.upstream_results_database,
            upstream_video=completion.upstream_video,
            stdout=completion.stdout,
            stderr=completion.stderr,
        )

    changed = json.loads(completion.execution_receipt_bytes)
    changed["runtime"]["role"] = "host_executor"
    body = dict(changed)
    body.pop("receipt_body_sha256")
    changed["receipt_body_sha256"] = _sha256(runner._canonical_json(body))
    changed_raw = runner._canonical_json(changed)
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="runtime"):
        runner.parse_matched_v3_external_execution_receipt(
            changed_raw,
            expected_receipt_sha256=_sha256(changed_raw),
            candidate_id=completion.candidate_id,
            environment_seed=completion.environment_seed,
            agent_seed=completion.agent_seed,
            upstream_reward_npz=completion.upstream_reward_npz,
            upstream_results_database=completion.upstream_results_database,
            upstream_video=completion.upstream_video,
            stdout=completion.stdout,
            stderr=completion.stderr,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "tampered_returncode",
    [False, runner._MAX_PROCESS_RETURNCODE + 1, runner._MIN_PROCESS_RETURNCODE - 1],
)
def test_receipt_requires_exact_integer_zero_and_plausible_returncode(
    fake_runtime: InjectedTestRuntime,
    tampered_returncode: object,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    completion = _consume(
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec),
        )
    )
    changed = json.loads(completion.execution_receipt_bytes)
    changed["process"]["returncode"] = tampered_returncode
    changed_raw, changed_sha256 = _resign_receipt(changed)
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="returncode",
    ):
        runner.parse_matched_v3_external_execution_receipt(
            changed_raw,
            expected_receipt_sha256=changed_sha256,
            candidate_id=completion.candidate_id,
            environment_seed=completion.environment_seed,
            agent_seed=completion.agent_seed,
            upstream_reward_npz=completion.upstream_reward_npz,
            upstream_results_database=completion.upstream_results_database,
            upstream_video=completion.upstream_video,
            stdout=completion.stdout,
            stderr=completion.stderr,
        )


@pytest.mark.unit
def test_detached_receipt_uid_gid_bind_frozen_profile_not_replay_process(
    fake_runtime: InjectedTestRuntime,
) -> None:
    spec = runner._candidate("external_dqn_plain")
    completion = _consume(
        _execute(
            _issue(spec.candidate_id),
            fake_runtime,
            process_runner=_fake_process_runner(spec),
        )
    )
    serialized = json.loads(completion.execution_receipt_bytes)
    assert serialized["runtime"]["uid"] == runner._EXPECTED_UID
    assert serialized["runtime"]["gid"] == runner._EXPECTED_GID
    parsed = runner.parse_matched_v3_external_execution_receipt(
        completion.execution_receipt_bytes,
        expected_receipt_sha256=completion.execution_receipt_sha256,
        candidate_id=completion.candidate_id,
        environment_seed=completion.environment_seed,
        agent_seed=completion.agent_seed,
        upstream_reward_npz=completion.upstream_reward_npz,
        upstream_results_database=completion.upstream_results_database,
        upstream_video=completion.upstream_video,
        stdout=completion.stdout,
        stderr=completion.stderr,
    )
    assert parsed["runtime"]["uid"] == runner._EXPECTED_UID
    assert parsed["runtime"]["gid"] == runner._EXPECTED_GID
    changed = json.loads(completion.execution_receipt_bytes)
    replay_uid = os.getuid()
    if replay_uid == runner._EXPECTED_UID:
        replay_uid -= 1
    changed["runtime"]["uid"] = replay_uid
    changed_raw, changed_sha256 = _resign_receipt(changed)
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="runtime identity",
    ):
        runner.parse_matched_v3_external_execution_receipt(
            changed_raw,
            expected_receipt_sha256=changed_sha256,
            candidate_id=completion.candidate_id,
            environment_seed=completion.environment_seed,
            agent_seed=completion.agent_seed,
            upstream_reward_npz=completion.upstream_reward_npz,
            upstream_results_database=completion.upstream_results_database,
            upstream_video=completion.upstream_video,
            stdout=completion.stdout,
            stderr=completion.stderr,
        )


@pytest.mark.unit
def test_source_staleness_consumes_execution_capability(
    fake_runtime: InjectedTestRuntime,
) -> None:
    capability = _issue("external_dqn_plain")
    state = runner._EXECUTION_CAPABILITIES[
        cast(runner._ExecutionCapability, capability)
    ]
    state.source_sha256 = "0" * 64
    with pytest.raises(runner.ForagerMatchedV3ExternalExecutionRunnerError, match="source"):
        _execute(capability, fake_runtime)
    assert state.status == "consumed"


@pytest.mark.unit
def test_loaded_closure_rejects_helper_code_and_constant_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_child_environment", lambda _path: {})
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="closure integrity",
        ):
            _issue("external_dqn_plain")

    def replacement(_spec: object) -> tuple[()]:
        return ()

    with monkeypatch.context() as patch:
        patch.setattr(runner._artifact_paths, "__code__", replacement.__code__)
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="closure integrity",
        ):
            _issue("external_dqn_plain")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_HORIZON", runner._HORIZON - 1)
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="constant content",
        ):
            _issue("external_dqn_plain")


@pytest.mark.unit
def test_loaded_closure_rejects_module_and_transitive_behavior_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutations: tuple[Callable[[pytest.MonkeyPatch], None], ...] = (
        lambda patch: patch.setattr(runner, "subprocess", object()),
        lambda patch: patch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: None),
        lambda patch: patch.setattr(os, "getuid", lambda: runner._EXPECTED_UID),
        lambda patch: patch.setattr(platform, "system", lambda: "Linux"),
        lambda patch: patch.setattr(time, "monotonic", lambda: 0.0),
        lambda patch: patch.setattr(selectors, "DefaultSelector", object()),
        lambda patch: patch.setattr(runner, "Path", object()),
    )
    for mutate in mutations:
        with monkeypatch.context() as patch:
            mutate(patch)
            with pytest.raises(
                runner.ForagerMatchedV3ExternalExecutionRunnerError,
                match="transitive behavior",
            ):
                _issue("external_dqn_plain")


@pytest.mark.unit
def test_loaded_closure_rejects_function_defaults_and_kwdefaults_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            runner._require_current_module_source,
            "__defaults__",
            ("0" * 64,),
        )
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="callable defaults",
        ):
            _issue("external_dqn_plain")

    kwdefaults = runner._canonical_json.__kwdefaults__
    assert kwdefaults is not None
    with monkeypatch.context() as patch:
        patch.setitem(kwdefaults, "maximum", 1)
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="callable defaults",
        ):
            _issue("external_dqn_plain")


@pytest.mark.unit
def test_loaded_closure_is_checked_at_execute_and_consume(
    fake_runtime: InjectedTestRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _issue("external_dqn_plain")
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_read_relative_regular", lambda *_args, **_kwargs: b"")
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="closure integrity",
        ):
            _execute(capability, fake_runtime)

    spec = runner._candidate("external_dqn_plain")
    outcome = _execute(
        capability,
        fake_runtime,
        process_runner=_fake_process_runner(spec),
    )
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_validate_artifact_inputs", lambda *_args, **_kwargs: None)
        with pytest.raises(
            runner.ForagerMatchedV3ExternalExecutionRunnerError,
            match="closure integrity",
        ):
            _consume(outcome)
    assert _consume(outcome).candidate_id == "external_dqn_plain"


@pytest.mark.unit
def test_injected_marker_and_capability_cannot_enter_production_runner() -> None:
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="exact marker",
    ):
        runner._issue_matched_v3_external_execution_capability_for_test(
            test_only_marker=object(),
            explicit_execution_opt_in=runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
            candidate_id="external_dqn_plain",
            environment_seed=1,
            agent_seed=2,
        )
    capability = _issue("external_dqn_plain")
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="cannot enter the production runner",
    ):
        runner.execute_matched_v3_external_candidate(
            execution_capability=capability,
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
            maximum_external_npz_bytes=1,
            maximum_results_database_bytes=1,
            maximum_ppo_video_bytes=0,
        )


@pytest.mark.unit
def test_internal_core_cannot_label_an_injected_runner_as_production() -> None:
    spec = runner._candidate("external_dqn_plain")
    state = runner._ExecutionState(
        pid=os.getpid(),
        status="consumed",
        candidate_id=spec.candidate_id,
        environment_seed=1,
        agent_seed=2,
        source_sha256=runner._MODULE_SOURCE_SHA256_AT_IMPORT,
        test_only_injected=False,
    )
    forged_context = runner._ExecutionContext(
        workload_root=runner._WORKLOAD_ROOT,
        private_runtime_parent=runner._PRIVATE_RUNTIME_PARENT,
        python_executable=runner._PYTHON_EXECUTABLE,
        python_argv0=runner._PYTHON_ARGV0,
        process_runner=_fake_process_runner(spec),
        source_member_identity=runner._PRODUCTION_SOURCE_MEMBER_IDENTITY,
        cleanup_execution_root=runner._PRODUCTION_CLEANUP_EXECUTION_ROOT,
        production_runner_exact=True,
        test_only_process_runner_injected=False,
        closure_integrity_checked=True,
    )
    with pytest.raises(
        runner.ForagerMatchedV3ExternalExecutionRunnerError,
        match="captured closure binding",
    ):
        runner._execute_matched_v3_external_candidate(
            exact_capability=runner._ExecutionCapability(),
            state=state,
            context=forged_context,
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
            maximum_external_npz_bytes=1,
            maximum_results_database_bytes=1,
            maximum_ppo_video_bytes=0,
        )
