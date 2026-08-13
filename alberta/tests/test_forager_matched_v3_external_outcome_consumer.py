from __future__ import annotations

import ast
import hashlib
import io
import json
import stat
import struct
import subprocess
import sys
import types
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_execution_runner as detached_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_outcome_consumer as consumer,
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

_ROOT = Path(__file__).resolve().parents[1]
_ATOMIC_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "_forager_matched_v3_atomic_publication.py"
)
_PUBLISHER_PATH = (
    _ROOT
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_external_reward_publication.py"
)
_CONSUMER_PATH = Path(consumer.__file__).resolve()
_RUNNER_PATH = Path(detached_runner.__file__).resolve()


def _load_exact_module(
    path: Path,
    name: str,
    injections: dict[str, object],
) -> tuple[types.ModuleType, str]:
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__dict__.update(injections)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def _valid_external_npz() -> bytes:
    data = bytearray(499_712 * 2)
    for index, value in ((0, -1), (1, 0), (2, 1), (3, 30), (499_711, -1)):
        struct.pack_into("<e", data, index * 2, value)
    dictionary = repr(
        {"descr": "<f2", "fortran_order": False, "shape": (499_712,)}
    ).encode("ascii")
    prefix_size = 10
    padding = (-((prefix_size + len(dictionary) + 1) % 64)) % 64
    header = dictionary + b" " * padding + b"\n"
    member = (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + bytes(data)
    )
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
            info = zipfile.ZipInfo("rewards.npy")
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, member)
    return output.getvalue()


def _write_private_file(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    current = path.parent
    while current.name not in {"", "fd"}:
        current.chmod(0o700)
        current = current.parent


def _run_isolated_publication_chain(root: Path) -> dict[str, Any]:
    atomic_name = "_alberta_forager_matched_v3_atomic_publication_isolated_v1"
    publisher_name = (
        "_alberta_forager_matched_v3_external_reward_publication_isolated_v1"
    )
    consumer_name = (
        "_alberta_forager_matched_v3_external_outcome_consumer_isolated_v1"
    )
    runner_name = (
        "_alberta_forager_matched_v3_external_execution_runner_isolated_v1"
    )
    consumer_source = hashlib.sha256(_CONSUMER_PATH.read_bytes()).hexdigest()
    runner_source = hashlib.sha256(_RUNNER_PATH.read_bytes()).hexdigest()
    publisher_source = hashlib.sha256(_PUBLISHER_PATH.read_bytes()).hexdigest()
    atomic, _atomic_source = _load_exact_module(_ATOMIC_PATH, atomic_name, {})
    publisher, observed_publisher = _load_exact_module(
        _PUBLISHER_PATH,
        publisher_name,
        {
            "_MATCHED_V3_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256": publisher_source,
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256": consumer_source,
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256": (
                consumer.EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256
            ),
        },
    )
    isolated_consumer, observed_consumer = _load_exact_module(
        _CONSUMER_PATH,
        consumer_name,
        {
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256": consumer_source,
            "_MATCHED_V3_EXTERNAL_EXECUTION_RUNNER_SOURCE_SHA256": runner_source,
            "_MATCHED_V3_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256": (
                detached_runner.EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256
            ),
        },
    )
    isolated_runner, observed_runner = _load_exact_module(
        _RUNNER_PATH,
        runner_name,
        {
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256": consumer_source,
            "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256": (
                isolated_consumer.EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256
            ),
        },
    )
    assert observed_publisher == publisher_source
    assert observed_consumer == consumer_source
    assert observed_runner == runner_source
    assert publisher._ATOMIC_MODULE_AT_LOAD is atomic
    assert isolated_consumer._PUBLISHER_MODULE_AT_LOAD is publisher
    assert isolated_runner._EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD is isolated_consumer

    source_root = root / "source"
    private_root = root / "private"
    publication_parent = root / "output"
    for path in (source_root, private_root, publication_parent):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    external_npz = _valid_external_npz()

    def source_identity(
        _anchor: object, path: str, expected_sha256: str
    ) -> dict[str, Any]:
        return {"path": path, "sha256": expected_sha256, "size_bytes": 123}

    spec = isolated_runner._candidate("external_dqn_plain")

    def process_runner(
        argv: tuple[str, ...],
        *,
        environment: object,
        executable_descriptor: int,
        inherited_descriptors: tuple[int, ...],
        working_directory: str,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> object:
        del (
            environment,
            executable_descriptor,
            inherited_descriptors,
            working_directory,
            timeout_seconds,
            stdout_limit_bytes,
            stderr_limit_bytes,
        )
        save = Path(argv[argv.index("--save_path") + 1])
        payloads = {
            "upstream_reward_npz": external_npz,
            "upstream_results_database": b"opaque-results-database",
        }
        for kind, relative in isolated_runner._artifact_paths(spec):
            _write_private_file(save / relative, payloads[kind])
        return isolated_runner.BoundedExternalProcessResult(0, b"", b"")

    execution = isolated_runner._issue_matched_v3_external_execution_capability_for_test(
        test_only_marker=isolated_runner._INJECTED_TEST_ONLY_MARKER,
        explicit_execution_opt_in=isolated_runner.EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
        candidate_id=spec.candidate_id,
        environment_seed=11,
        agent_seed=29,
    )
    outcome = isolated_runner._execute_matched_v3_external_candidate_for_test(
        test_only_marker=isolated_runner._INJECTED_TEST_ONLY_MARKER,
        execution_capability=execution,
        workload_root=source_root,
        private_runtime_parent=private_root,
        python_executable=Path("/usr/bin/python3.12"),
        python_argv0="/usr/bin/python3.12",
        process_runner=process_runner,
        source_member_identity=source_identity,
        timeout_seconds=30,
        maximum_stdout_bytes=1024,
        maximum_stderr_bytes=1024,
        maximum_external_npz_bytes=len(external_npz),
        maximum_results_database_bytes=1024,
        maximum_ppo_video_bytes=0,
    )
    publish_kwargs = dict(
        outcome_capability=outcome,
        publication_parent=publication_parent,
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
        maximum_publication_total_bytes=8 * 1024 * 1024,
        explicit_publication_opt_in=True,
    )
    publication_parent.chmod(0o755)
    try:
        publisher.publish_matched_v3_external_outcome_capability(**publish_kwargs)
    except atomic.ForagerMatchedV3AtomicPublicationError:
        pass
    else:
        raise AssertionError("unsafe publication parent passed pre-claim preflight")
    assert isolated_runner._OUTCOME_CAPABILITIES[outcome].status == "live"
    publication_parent.chmod(0o700)
    metadata = publisher.publish_matched_v3_external_outcome_capability(**publish_kwargs)
    assert type(metadata) is publisher.MatchedV3ExternalPublicationMetadata
    assert metadata.operation == "published"
    assert metadata.interaction_horizon == 499_712
    assert metadata.publication_committed is True
    assert metadata.file_count == 10
    assert metadata.publication_root == publication_parent / metadata.address
    assert metadata.publication_root.name == metadata.address
    assert not hasattr(metadata, "cumulative_score")
    assert not hasattr(metadata, "conversion")
    raw = publisher.canonical_external_publication_metadata_bytes(metadata)
    parsed = publisher.parse_external_publication_metadata(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert parsed == metadata
    state = isolated_runner._OUTCOME_CAPABILITIES[outcome]
    assert state.status == "consumed"
    try:
        isolated_runner.consume_matched_v3_external_execution_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )
    except isolated_runner.ForagerMatchedV3ExternalExecutionRunnerError:
        pass
    else:
        raise AssertionError("publication path did not consume the public completion path")
    return {
        "address": metadata.address,
        "file_count": metadata.file_count,
        "total_size_bytes": metadata.total_size_bytes,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "invalid_parent_preserved_live_outcome": True,
    }


def test_descriptor_freezes_one_way_post_claim_conversion_and_publication() -> None:
    raw = consumer.canonical_external_outcome_consumer_descriptor_bytes()
    descriptor = consumer.parse_external_outcome_consumer_descriptor(raw)
    assert hashlib.sha256(raw).hexdigest() == (
        consumer.EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256
    )
    assert descriptor["candidate_order"] == list(EXPECTED_CANDIDATE_IDS)
    assert descriptor["load_order"] == {
        "atomic_before_publisher": True,
        "publisher_before_consumer": True,
        "consumer_before_runner": True,
        "bridge_scorer_protocol_absent_before_runner_claim": True,
        "protocol_then_scorer_then_bridge_after_runner_claim": True,
        "preexisting_parent_package_required": True,
        "parent_package_initializer_executed_post_claim": False,
    }
    assert descriptor["outcome_path"]["live_pid_bound_capability_required"] is True
    assert descriptor["outcome_path"]["public_completion_accepted"] is False
    assert descriptor["outcome_path"]["raw_bytes_accepted_by_orchestrator"] is False
    assert descriptor["outcome_path"]["callback_or_sink_accepted"] is False
    assert (
        descriptor["outcome_path"]["safe_parent_preflight_precedes_runner_claim"]
        is True
    )
    assert descriptor["outcome_path"]["runner_claim_precedes_conversion"] is True
    assert descriptor["outcome_path"]["claim_failure_retry"] is False
    assert descriptor["publication"]["returns_metadata_only"] is True
    assert descriptor["publication"]["exact_file_count"] == 10
    assert descriptor["publication"]["aggregate_ceiling_bytes"] == 1024 * 1024 * 1024
    assert all(value is False for value in descriptor["claims"].values())


def test_consumer_source_has_no_static_project_import() -> None:
    source = Path(consumer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert all(not name.startswith("alberta_framework") for name in imports)


def test_descriptor_parser_rejects_reformatted_or_resigned_content() -> None:
    raw = consumer.canonical_external_outcome_consumer_descriptor_bytes()
    changed = json.dumps(json.loads(raw), indent=2, sort_keys=True).encode("ascii") + b"\n"
    with pytest.raises(consumer.ForagerMatchedV3ExternalOutcomeConsumerError):
        consumer.parse_external_outcome_consumer_descriptor(changed)


def test_public_surface_exposes_no_conversion_or_payload_api() -> None:
    exported = set(consumer.__all__)
    assert "_consume_matched_v3_external_outcome_to_captured_publication" not in exported
    assert "_consume_claimed_matched_v3_external_execution_payload" not in exported
    assert not any("convert" in name or "payload" in name for name in exported)


def test_detached_import_cannot_consume_an_outcome(tmp_path: Path) -> None:
    with pytest.raises(
        consumer.ForagerMatchedV3ExternalOutcomeConsumerError,
        match="isolated direct-load boundary",
    ):
        consumer._consume_matched_v3_external_outcome_to_captured_publication(
            outcome_capability=object(),
            publication_parent=tmp_path.resolve(),
            expected_candidate_id="external_dqn_plain",
            expected_environment_seed=1,
            expected_agent_seed=2,
            expected_environment_seed_commitment_sha256="1" * 64,
            expected_agent_seed_commitment_sha256="2" * 64,
            expected_qualification_plan_sha256="3" * 64,
            expected_qualification_case_manifest_sha256="4" * 64,
            expected_publisher_source_tree_sha256="5" * 64,
            expected_workload_source_tree_sha256="6" * 64,
            expected_staging_manifest_sha256="7" * 64,
            maximum_publication_total_bytes=1024,
            explicit_publication_opt_in=True,
        )


def test_fresh_process_claim_convert_publish_returns_metadata_only(tmp_path: Path) -> None:
    isolated_root = tmp_path / "isolated-chain"
    isolated_root.mkdir(mode=0o700)
    script = r'''\
import importlib.util
import json
import sys
from pathlib import Path

test_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
name = '_matched_v3_external_consumer_integration_helpers'
spec = importlib.util.spec_from_file_location(name, test_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
print(json.dumps(module._run_isolated_publication_chain(root), sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(isolated_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["file_count"] == 10
    assert summary["total_size_bytes"] > 0
    assert len(summary["address"]) == 64
    assert len(summary["metadata_sha256"]) == 64
    assert summary["invalid_parent_preserved_live_outcome"] is True


def test_score_closure_never_imports_a_missing_parent_package() -> None:
    script = r'''\
import sys
from alberta_framework.benchmarks import forager_matched_v3_external_outcome_consumer as c

sys.modules.pop('alberta_framework.benchmarks', None)
try:
    c._load_source_module(
        module_name=c.PINNED_PROTOCOL_MODULE_NAME,
        source_path=c.PINNED_PROTOCOL_SOURCE_PATH,
        source_sha256=c.PINNED_PROTOCOL_SOURCE_SHA256,
    )
except c.ForagerMatchedV3ExternalOutcomeConsumerError as exc:
    assert 'parent package' in str(exc)
else:
    raise AssertionError('missing parent package was implicitly imported')
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
