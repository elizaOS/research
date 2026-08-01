"""Fail-closed contracts for hidden-regime calibration readiness receipts."""

from __future__ import annotations

import copy
import dataclasses
import errno
import importlib.metadata
import io
import json
import os
import pickle
import py_compile
import stat
import subprocess
import sys
import sysconfig
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

import alberta_framework.evaluation.hidden_regime_calibration_readiness as readiness
from alberta_framework.evaluation.hidden_regime_calibration_readiness import (
    CERTIFICATION_SPECS,
    PreparedReadinessReceipt,
    ReadinessDraft,
    ReadinessError,
    VerifiedCertificationBundle,
    bound_calibration_runtime_batch,
    build_readiness_draft,
    canonical_json_bytes,
    canonical_sha256,
    execute_bound_calibration_worker,
    finalize_readiness_receipt,
    publish_readiness_receipt,
    require_validated_readiness_receipt,
    run_readiness_certifications,
    validate_published_readiness_receipt,
    validate_readiness_receipt,
)

pytestmark = pytest.mark.unit

_RUNNER_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_calibration"
_REAL_SUBPROCESS_RUN = subprocess.run
_REAL_PROTOCOL_BINDING = readiness._protocol_binding
_SHA = "1" * 64
_RUNTIME_PREFIX = Path(sys.prefix).absolute()
_RUNTIME_EXEC_PREFIX = Path(sys.exec_prefix).absolute()
_RUNTIME_PURELIB = Path(sysconfig.get_path("purelib")).absolute()
_RUNTIME_PLATLIB = Path(sysconfig.get_path("platlib")).absolute()
_RUNTIME_STDLIB = Path(sysconfig.get_path("stdlib")).absolute()
_NO_SITE_STDLIB_PATHS = [
    (
        _RUNTIME_STDLIB.parent
        / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    ).as_posix(),
    _RUNTIME_STDLIB.as_posix(),
    (_RUNTIME_STDLIB / "lib-dynload").as_posix(),
]


def _fake_worker_arguments(
    published: readiness.PublishedReadinessReceipt,
    marker: str,
) -> tuple[str, str, str]:
    return (
        "--worker-preflight-v1",
        published.directory.absolute().as_posix(),
        marker,
    )


def _module_locator(module: str) -> Path:
    return Path(*module.split(".")).with_suffix(".py")


def _fake_repository(root: Path) -> Path:
    (root / "alberta_framework/evaluation").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "alberta_framework/__init__.py").write_text('"""fake package"""\n')
    (root / "alberta_framework/evaluation/__init__.py").write_text(
        '"""fake evaluation package"""\n'
    )
    for module in readiness._BASE_SOURCE_ROOT_MODULES:
        locator = root / _module_locator(module)
        if module == _RUNNER_MODULE:
            source = """\
\"\"\"Bound fake calibration runner.\"\"\"
import json
import os
import sys

def main(argv):
    print(json.dumps({
        \"argv\": list(argv),
        \"cwd_empty\": os.listdir(os.getcwd()) == [],
        \"environment\": dict(os.environ),
        \"origin\": __file__,
        \"dont_write_bytecode\": sys.dont_write_bytecode,
        \"pycache_prefix_set\": isinstance(sys.pycache_prefix, str),
        \"no_site\": sys.flags.no_site == 1,
        \"virtualenv_hook_absent\": \"_virtualenv\" not in sys.modules,
        \"sys_prefix\": sys.prefix,
        \"sys_exec_prefix\": sys.exec_prefix,
        \"sys_path\": list(sys.path),
    }, sort_keys=True))
    return 0
"""
        else:
            source = f'"""Fake source root for {module}."""\n'
        locator.write_text(source, encoding="utf-8")

    functions_by_path: dict[Path, set[str]] = {}
    checkpoint_contracts_by_node: dict[tuple[Path, str], readiness.CertificationSpec] = {}
    for spec in CERTIFICATION_SPECS:
        for node_id in spec.node_ids:
            locator_text, separator, function_name = node_id.partition("::")
            if not separator:
                function_name = "test_full_file_certification"
            locator = Path(locator_text)
            functions_by_path.setdefault(locator, set()).add(function_name)
            if node_id == spec.checkpoint_cut_runtime_node_id:
                checkpoint_contracts_by_node[(locator, function_name)] = spec
    for relative, functions in functions_by_path.items():
        function_sources: list[str] = []
        fixture_sources: list[str] = []
        for function_name in sorted(functions):
            contract = checkpoint_contracts_by_node.get((relative, function_name))
            arguments = ""
            if contract is not None:
                fixture_name = contract.checkpoint_cut_fixture_name
                trace_fixture_name = contract.checkpoint_cut_trace_fixture_name
                assert fixture_name is not None
                assert trace_fixture_name is not None
                arguments = f"{fixture_name}, {trace_fixture_name}"
                cuts = {
                    cut_id: index
                    for index, cut_id in enumerate(contract.checkpoint_cut_ids, start=1)
                }
                fixture_sources.append(
                    f"@pytest.fixture\ndef {fixture_name}():\n    return ((), {cuts!r})"
                )
                fixture_sources.append(
                    f"""@pytest.fixture
def {trace_fixture_name}():
    trace = SimpleNamespace(
        helper_lease_offset_post=[1, 0, 0, 0, 0, 0],
        helper_lease_boundary=[False, True, False, False, False, False],
        segment_index=[0, 0, 0, 1, 1, 1],
        helper_scratch_retest_started=[False, False, False, True, False, False],
        helper_committed_slot=[1, -1, -1, -1, 2, -1],
        helper_retired_slot=[-1, -1, -1, -1, -1, 1],
    )
    return SimpleNamespace(trace=trace)"""
                )
            lines = [
                f"def {function_name}({arguments}):",
                "    assert callable(bound_runner.main)",
            ]
            function_sources.append("\n".join(lines))
        (root / relative).write_text(
            '"""Fake certification source."""\n\n'
            "from types import SimpleNamespace\n"
            "import pytest\n"
            "import alberta_framework.evaluation.hidden_regime_factorial_calibration "
            "as bound_runner\n\n"
            + "\n\n".join(fixture_sources)
            + ("\n\n" if fixture_sources else "")
            + "\n\n".join(function_sources)
            + "\n",
            encoding="utf-8",
        )
    (root / "tests/conftest.py").write_text('"""Fake pytest configuration."""\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "readiness-fixture"\nversion = "0"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def _fake_runtime_identity() -> dict[str, object]:
    return {
        "schema": readiness.READINESS_RUNTIME_SCHEMA,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "hexversion": 1,
            "cache_tag": "cpython-312",
            "byteorder": "little",
            "executable_sha256": _SHA,
            "prefix": _RUNTIME_PREFIX.as_posix(),
            "exec_prefix": _RUNTIME_EXEC_PREFIX.as_posix(),
            "purelib": _RUNTIME_PURELIB.as_posix(),
            "platlib": _RUNTIME_PLATLIB.as_posix(),
            "stdlib": _RUNTIME_STDLIB.as_posix(),
            "no_site_stdlib_search_paths": _NO_SITE_STDLIB_PATHS,
            "stdlib_file_scope": readiness._STDLIB_FILE_SCOPE,
            "stdlib_file_count": 1,
            "stdlib_directory_count": 1,
            "stdlib_file_total_bytes": 1,
            "stdlib_file_inventory_sha256": _SHA,
        },
        "platform": {
            "system": "Linux",
            "release": "test",
            "version_sha256": _SHA,
            "machine": "x86_64",
            "libc": ["glibc", "test"],
            "cpu_count": 1,
        },
        "dependencies": {
            "key_versions": {
                "alberta-framework": "test",
                "chex": "test",
                "jax": "test",
                "jaxlib": "test",
                "numpy": "test",
                "pytest": "test",
                "scipy": "test",
            },
            "installed_distribution_count": 1,
            "installed_distribution_file_scope": (
                readiness._INSTALLED_DISTRIBUTION_FILE_SCOPE
            ),
            "installed_distribution_file_count": 1,
            "installed_distribution_file_total_bytes": 1,
            "installed_distribution_file_inventory_sha256": _SHA,
            "dependency_import_tree_file_scope": (
                readiness._DEPENDENCY_IMPORT_TREE_FILE_SCOPE
            ),
            "dependency_import_tree_root_count": len(
                {_RUNTIME_PURELIB.as_posix(), _RUNTIME_PLATLIB.as_posix()}
            ),
            "dependency_import_tree_file_count": 1,
            "dependency_import_tree_directory_count": 1,
            "dependency_import_tree_file_total_bytes": 1,
            "dependency_import_tree_file_inventory_sha256": _SHA,
        },
        "jax": {
            "default_backend": "cpu",
            "enable_x64": False,
            "config_sha256": _SHA,
            "devices": [
                {
                    "id": 0,
                    "process_index": 0,
                    "platform": "cpu",
                    "device_kind": "test",
                    "local_hardware_id": 0,
                }
            ],
        },
        "environment": [],
        "child_environment": {"LC_ALL": "C", "PYTHONHASHSEED": "0"},
        "library_environment_side_effects": {
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "TPU_SKIP_MDS_QUERY": "1",
        },
    }


def _fake_runtime_reconstruction_record(**kwargs: object) -> dict[str, object]:
    runtime = kwargs["runtime"]
    stdout = canonical_json_bytes(runtime)
    return {
        "schema": readiness.READINESS_RUNTIME_RECONSTRUCTION_SCHEMA,
        "command": list(readiness._RUNTIME_RECONSTRUCTION_SEMANTIC_COMMAND),
        "harness_sha256": readiness._RUNTIME_RECONSTRUCTION_BOOTSTRAP_SHA256,
        "environment_policy": readiness._CHILD_ENVIRONMENT_POLICY,
        "status": "passed",
        "exit_code": 0,
        "stdout": {"byte_size": len(stdout), "sha256": readiness._sha256_bytes(stdout)},
        "stderr": {"byte_size": 0, "sha256": readiness._sha256_bytes(b"")},
        "source_manifest_sha256": kwargs["source_manifest_sha256"],
        "runtime_identity_sha256": canonical_sha256(runtime),
        "protocol_payload_sha256": kwargs["protocol_payload_sha256"],
    }


def _fake_installed_distribution(
    runtime_prefix: Path,
    *,
    name: str,
    version: str,
    payloads: dict[str, bytes],
) -> tuple[importlib.metadata.PathDistribution, Path, Path]:
    site_packages = runtime_prefix / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    metadata_locator = f"{dist_info.name}/METADATA"
    record_locator = f"{dist_info.name}/RECORD"
    metadata_raw = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n").encode()
    members = {**payloads, metadata_locator: metadata_raw}
    for locator, raw in members.items():
        target = Path(os.path.abspath(os.fspath(site_packages / locator)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    record_raw = "".join(f"{locator},,\n" for locator in (*sorted(members), record_locator)).encode(
        "utf-8"
    )
    record_path = dist_info / "RECORD"
    record_path.write_bytes(record_raw)
    return importlib.metadata.PathDistribution(dist_info), site_packages, record_path


def _fake_protocol_binding() -> dict[str, object]:
    manifests: list[object] = [
        {
            "name": "hidden-regime-calibration-a-v1",
            "use_partition": "calibration-only",
            "manifest_payload_sha256": "2" * 64,
        }
    ]
    return {
        "receipt_schema": readiness.CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "design_schema": readiness.DESIGN_SCHEMA,
        "design_envelope_schema": readiness.DESIGN_ENVELOPE_SCHEMA,
        "protocol_status": readiness.PROTOCOL_STATUS,
        "protocol_payload_sha256": readiness.CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": readiness.SEED_SNAPSHOT_SHA256,
        "manifest_bindings": manifests,
        "manifest_bindings_sha256": readiness._protocol_canonical_sha256(manifests),
        "recurrence_eligibility_sha256": "5" * 64,
        "gate_matrix_sha256": "6" * 64,
        "development_summary_schema": readiness.BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace_schema": readiness.BOUND_PRIMITIVE_TRACE_SCHEMA,
        "consumed_calibration_namespace_sha256": "7" * 64,
        "matched_case_count": 240,
    }


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    return _fake_repository(tmp_path / "repository")


@pytest.fixture(autouse=True)
def stable_external_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness, "_build_runtime_identity", _fake_runtime_identity)
    monkeypatch.setattr(readiness, "_protocol_binding", _fake_protocol_binding)


def _passing_execution_manifest(spec: readiness.CertificationSpec) -> dict[str, object]:
    collected = sorted(
        node_id if "::" in node_id else f"{node_id}::test_full_file_certification"
        for node_id in spec.node_ids
    )
    cut_observation: dict[str, object] | None = None
    if spec.checkpoint_cut_runtime_node_id is not None:
        cut_values = list(range(1, len(spec.checkpoint_cut_ids) + 1))
        cut_contract = spec.runtime_contract["checkpoint_cut_contract"]
        assert isinstance(cut_contract, dict)
        semantics = cut_contract["cut_semantics"]
        assert isinstance(semantics, list)
        cut_observation = {
            "node_id": spec.checkpoint_cut_runtime_node_id,
            "fixture_name": spec.checkpoint_cut_fixture_name,
            "trace_fixture_name": spec.checkpoint_cut_trace_fixture_name,
            "cut_ids": list(spec.checkpoint_cut_ids),
            "cut_values": cut_values,
            "semantic_observations": [
                {
                    **semantic,
                    "matching_event_count": semantic["occurrence_index"] + 1,
                    "selected_event_index": cut_value - semantic["index_offset"],
                    "expected_cut_value": cut_value,
                    "observed_cut_value": cut_value,
                }
                for semantic, cut_value in zip(semantics, cut_values, strict=True)
            ],
        }
    phase = {"outcome": "passed", "was_xfail": False}
    return {
        "schema": readiness.READINESS_CERTIFICATION_EXECUTION_MANIFEST_SCHEMA,
        "certification_id": spec.certification_id,
        "runtime_contract_sha256": spec.runtime_contract_sha256,
        "status": "passed",
        "pytest_exit_code": 0,
        "collected_node_count": len(collected),
        "collected_node_ids": collected,
        "deselected_node_ids": [],
        "per_node": [
            {
                "node_id": node_id,
                "setup": dict(phase),
                "call": dict(phase),
                "teardown": dict(phase),
            }
            for node_id in collected
        ],
        "checkpoint_cut_observation": cut_observation,
        "violations": [],
    }


def _successful_certification_run(
    calls: list[tuple[tuple[str, ...], Path]],
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((tuple(command), Path(cwd)))
        assert command[1:5] == ("-S", "-B", "-P", "-X")
        assert command[5].startswith("pycache_prefix=")
        assert command[6] == "-c"
        bytecode_cache_root = Path(command[5].removeprefix("pycache_prefix="))
        assert not bytecode_cache_root.exists() or not any(bytecode_cache_root.iterdir())
        normalized = list(command)
        normalized[0] = "{runtime_python}"
        normalized[5] = "pycache_prefix={fresh_empty_separate_bytecode_cache_root}"
        normalized[7] = "{readiness_certification_harness_v1}"
        normalized[8] = "{fresh_empty_separate_bytecode_cache_root}"
        normalized[9] = "{verified_extracted_source_root}"
        normalized[10] = "{bound_runtime_prefix}"
        normalized[11] = "{bound_runtime_exec_prefix}"
        normalized[12] = "{bound_runtime_purelib}"
        normalized[13] = "{bound_runtime_platlib}"
        normalized[14] = "{bound_runtime_stdlib}"
        normalized[15] = "{bound_no_site_stdlib_search_paths_json}"
        if command[7] == readiness._RUNTIME_RECONSTRUCTION_BOOTSTRAP:
            normalized[7] = "{runtime_reconstruction_harness_v1}"
            normalized[16] = "{receipt_derived_child_environment_json}"
            assert tuple(normalized) == readiness._RUNTIME_RECONSTRUCTION_SEMANTIC_COMMAND
            assert env == {"LC_ALL": "C", "PYTHONHASHSEED": "0"}
            stdout = canonical_json_bytes(readiness._build_runtime_identity())
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")
        spec = CERTIFICATION_SPECS[len(calls) - 2]
        normalized[7] = "{readiness_certification_harness_v1}"
        normalized[16] = "{receipt_derived_certification_environment_json}"
        normalized[17] = "{fresh_certification_execution_manifest_path}"
        normalized[18] = "{certification_runtime_contract_json}"
        assert tuple(normalized) == spec.semantic_command
        assert json.loads(command[18]) == spec.runtime_contract
        Path(command[17]).write_bytes(canonical_json_bytes(_passing_execution_manifest(spec)))
        assert env == {
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"
        }
        assert Path(cwd).name == "source"
        assert (Path(cwd) / "alberta_framework").is_dir()
        return subprocess.CompletedProcess(command, 0, stdout=b"certified\n", stderr=b"")

    return run


def _prepare(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ReadinessDraft, VerifiedCertificationBundle, PreparedReadinessReceipt, list[object]]:
    draft = build_readiness_draft(repository_root=repository)
    calls: list[object] = []
    monkeypatch.setattr(readiness.subprocess, "run", _successful_certification_run(calls))
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
    )
    prepared = finalize_readiness_receipt(draft, bundle)
    return draft, bundle, prepared, calls


def test_record_inventory_binds_python_native_bytes_and_is_distribution_order_stable(
    tmp_path: Path,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    first, first_site, first_record = _fake_installed_distribution(
        runtime_prefix,
        name="Example_Core",
        version="1.2.3",
        payloads={
            "../../../bin/example-core": b"#!/bin/sh\n",
            "example_core/__init__.py": b"VALUE = 1\n",
            "example_core/_native.so": b"native-one",
        },
    )
    second, _second_site, _second_record = _fake_installed_distribution(
        runtime_prefix,
        name="Example-Extra",
        version="4.5",
        payloads={"example_extra.py": b"EXTRA = True\n"},
    )
    forward = readiness._distribution_record_file_inventory(
        (first, second),
        runtime_prefix=runtime_prefix,
    )
    reverse = readiness._distribution_record_file_inventory(
        (second, first),
        runtime_prefix=runtime_prefix,
    )
    assert forward == reverse
    assert forward.versions == (("example-core", "1.2.3"), ("example-extra", "4.5"))
    assert forward.file_count == 8
    assert forward.total_bytes > len(b"native-one")

    metadata_before = (first_site / "Example_Core-1.2.3.dist-info/METADATA").read_bytes()
    record_before = first_record.read_bytes()
    native = first_site / "example_core/_native.so"
    native.write_bytes(b"native-two")
    changed_native = readiness._distribution_record_file_inventory(
        (second, first),
        runtime_prefix=runtime_prefix,
    )
    assert changed_native.file_count == forward.file_count
    assert changed_native.total_bytes == forward.total_bytes
    assert changed_native.inventory_sha256 != forward.inventory_sha256
    assert (first_site / "Example_Core-1.2.3.dist-info/METADATA").read_bytes() == metadata_before
    assert first_record.read_bytes() == record_before

    module = first_site / "example_core/__init__.py"
    module.write_bytes(b"VALUE = 2\n")
    changed_python = readiness._distribution_record_file_inventory(
        (first, second),
        runtime_prefix=runtime_prefix,
    )
    assert changed_python.inventory_sha256 != changed_native.inventory_sha256
    assert first_record.read_bytes() == record_before


def test_record_inventory_rejects_missing_symlink_escape_duplicate_and_collision(
    tmp_path: Path,
) -> None:
    missing_prefix = tmp_path / "missing-runtime"
    missing, missing_site, _missing_record = _fake_installed_distribution(
        missing_prefix,
        name="Missing",
        version="1",
        payloads={"missing/module.py": b"present\n"},
    )
    (missing_site / "missing/module.py").unlink()
    with pytest.raises(ReadinessError, match="missing"):
        readiness._distribution_record_file_inventory(
            (missing,),
            runtime_prefix=missing_prefix,
        )

    symlink_prefix = tmp_path / "symlink-runtime"
    symlinked, symlink_site, _symlink_record = _fake_installed_distribution(
        symlink_prefix,
        name="Symlinked",
        version="1",
        payloads={"symlinked/module.py": b"original\n"},
    )
    target = symlink_site / "symlinked/module.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ReadinessError, match="regular non-symlink"):
        readiness._distribution_record_file_inventory(
            (symlinked,),
            runtime_prefix=symlink_prefix,
        )

    escape_prefix = tmp_path / "escape-runtime"
    escaping, _escape_site, escape_record = _fake_installed_distribution(
        escape_prefix,
        name="Escaping",
        version="1",
        payloads={"escaping.py": b"safe\n"},
    )
    escape_target = tmp_path / "escape.bin"
    escape_target.write_bytes(b"escape\n")
    escape_record.write_bytes(escape_record.read_bytes() + b"../../../../escape.bin,,\n")
    with pytest.raises(ReadinessError, match="escapes the runtime prefix"):
        readiness._distribution_record_file_inventory(
            (escaping,),
            runtime_prefix=escape_prefix,
        )

    absolute_prefix = tmp_path / "absolute-runtime"
    absolute, _absolute_site, absolute_record = _fake_installed_distribution(
        absolute_prefix,
        name="Absolute",
        version="1",
        payloads={"absolute.py": b"safe\n"},
    )
    absolute_target = tmp_path / "absolute.bin"
    absolute_target.write_bytes(b"absolute\n")
    absolute_record.write_bytes(
        absolute_record.read_bytes() + f"{absolute_target.as_posix()},,\n".encode()
    )
    with pytest.raises(ReadinessError, match="locator is absolute"):
        readiness._distribution_record_file_inventory(
            (absolute,),
            runtime_prefix=absolute_prefix,
        )

    duplicate_prefix = tmp_path / "duplicate-runtime"
    duplicate, _duplicate_site, duplicate_record = _fake_installed_distribution(
        duplicate_prefix,
        name="Duplicate",
        version="1",
        payloads={"duplicate/module.py": b"one\n"},
    )
    duplicate_record.write_bytes(duplicate_record.read_bytes() + b"duplicate/module.py,,\n")
    with pytest.raises(ReadinessError, match="duplicate RECORD path locator"):
        readiness._distribution_record_file_inventory(
            (duplicate,),
            runtime_prefix=duplicate_prefix,
        )

    collision_prefix = tmp_path / "collision-runtime"
    colliding, _collision_site, collision_record = _fake_installed_distribution(
        collision_prefix,
        name="Collision",
        version="1",
        payloads={"collision/module.py": b"one\n"},
    )
    collision_record.write_bytes(
        collision_record.read_bytes() + b"collision/../collision/module.py,,\n"
    )
    with pytest.raises(ReadinessError, match="colliding resolved RECORD path"):
        readiness._distribution_record_file_inventory(
            (colliding,),
            runtime_prefix=collision_prefix,
        )


def test_record_inventory_rejects_mutation_during_stable_open_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    distribution, site_packages, _record = _fake_installed_distribution(
        runtime_prefix,
        name="Mutable",
        version="1",
        payloads={"mutable/module.py": b"x" * 64},
    )
    target = site_packages / "mutable/module.py"
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    real_read = os.read
    mutated = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, count)
        opened = os.fstat(descriptor)
        if chunk and not mutated and (opened.st_dev, opened.st_ino) == target_identity:
            mutated = True
            with target.open("ab") as handle:
                handle.write(b"y")
        return chunk

    monkeypatch.setattr(readiness.os, "read", racing_read)
    with pytest.raises(ReadinessError, match="changed size while hashed"):
        readiness._distribution_record_file_inventory(
            (distribution,),
            runtime_prefix=runtime_prefix,
        )
    assert mutated


def test_record_inventory_rejects_earlier_file_drift_after_its_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    distribution, site_packages, _record = _fake_installed_distribution(
        runtime_prefix,
        name="Later-Drift",
        version="1",
        payloads={
            "later_drift/first.py": b"FIRST = 1\n",
            "later_drift/second.py": b"SECOND = 1\n",
        },
    )
    first = site_packages / "later_drift/first.py"
    second = site_packages / "later_drift/second.py"
    real_hash = readiness._stable_hash_record_file
    first_hashed = False
    mutated = False

    def drift_after_earlier_hash(
        path: Path,
        *,
        capture_limit: int | None = None,
    ) -> tuple[int, str, bytes | None, tuple[int, int, int, int, int, int, int]]:
        nonlocal first_hashed, mutated
        result = real_hash(path, capture_limit=capture_limit)
        if path == first:
            first_hashed = True
        elif path == second and first_hashed and not mutated:
            first.write_bytes(b"FIRST = 2\n")
            mutated = True
        return result

    monkeypatch.setattr(readiness, "_stable_hash_record_file", drift_after_earlier_hash)
    with pytest.raises(ReadinessError, match="drifted after hashing"):
        readiness._distribution_record_file_inventory(
            (distribution,),
            runtime_prefix=runtime_prefix,
        )
    assert mutated


def test_installed_inventory_hashes_nonkey_transitive_distribution_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    key, site_packages, _key_record = _fake_installed_distribution(
        runtime_prefix,
        name="Required-Key",
        version="1",
        payloads={"required_key.py": b"VALUE = 1\n"},
    )
    transitive, _site_packages, _transitive_record = _fake_installed_distribution(
        runtime_prefix,
        name="Transitive-Native",
        version="2",
        payloads={"transitive_native/_extension.so": b"native-one"},
    )
    stdlib = tmp_path / "stdlib"
    (stdlib / "lib-dynload").mkdir(parents=True)
    paths = readiness._RuntimePathBinding(
        prefix=runtime_prefix,
        exec_prefix=runtime_prefix,
        purelib=site_packages,
        platlib=site_packages,
        stdlib=stdlib,
        no_site_stdlib_search_paths=(
            tmp_path / "python312.zip",
            stdlib,
            stdlib / "lib-dynload",
        ),
    )
    monkeypatch.setattr(readiness, "_KEY_DISTRIBUTIONS", ("required-key",))
    monkeypatch.setattr(
        readiness.importlib.metadata,
        "distributions",
        lambda *, path: (key, transitive),
    )
    first = readiness._installed_distribution_record_file_inventory(paths)
    assert first.versions == (("required-key", "1"), ("transitive-native", "2"))

    native = site_packages / "transitive_native/_extension.so"
    native.write_bytes(b"native-two")
    second = readiness._installed_distribution_record_file_inventory(paths)
    assert second.file_count == first.file_count
    assert second.total_bytes == first.total_bytes
    assert second.inventory_sha256 != first.inventory_sha256


def test_stdlib_inventory_binds_native_and_symlink_target_bytes(tmp_path: Path) -> None:
    runtime_prefix = tmp_path / "runtime"
    purelib = runtime_prefix / "lib/python3.12/site-packages"
    purelib.mkdir(parents=True)
    stdlib = tmp_path / "stdlib"
    dynamic = stdlib / "lib-dynload"
    dynamic.mkdir(parents=True)
    (stdlib / "module.py").write_bytes(b"VALUE = 1\n")
    native = dynamic / "_native.so"
    native.write_bytes(b"native-one")
    first_target = tmp_path / "first-target.so"
    second_target = tmp_path / "second-target.so"
    first_target.write_bytes(b"target-one")
    second_target.write_bytes(b"target-two")
    link = stdlib / "linked-native.so"
    link.symlink_to(first_target)
    paths = readiness._RuntimePathBinding(
        prefix=runtime_prefix,
        exec_prefix=runtime_prefix,
        purelib=purelib,
        platlib=purelib,
        stdlib=stdlib,
        no_site_stdlib_search_paths=(
            tmp_path / "python312.zip",
            stdlib,
            dynamic,
        ),
    )

    first = readiness._stdlib_file_inventory(paths)
    native.write_bytes(b"native-two")
    changed_native = readiness._stdlib_file_inventory(paths)
    assert changed_native.file_count == first.file_count
    assert changed_native.total_bytes == first.total_bytes
    assert changed_native.inventory_sha256 != first.inventory_sha256

    native.write_bytes(b"native-one")
    link.unlink()
    link.symlink_to(second_target)
    changed_link = readiness._stdlib_file_inventory(paths)
    assert changed_link.file_count == first.file_count
    assert changed_link.total_bytes == first.total_bytes
    assert changed_link.inventory_sha256 != first.inventory_sha256


def test_stdlib_inventory_includes_nested_site_packages_and_root_directory(
    tmp_path: Path,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    purelib = runtime_prefix / "lib/python3.12/site-packages"
    purelib.mkdir(parents=True)
    stdlib = tmp_path / "stdlib"
    dynamic = stdlib / "lib-dynload"
    dynamic.mkdir(parents=True)
    paths = readiness._RuntimePathBinding(
        prefix=runtime_prefix,
        exec_prefix=runtime_prefix,
        purelib=purelib,
        platlib=purelib,
        stdlib=stdlib,
        no_site_stdlib_search_paths=(
            tmp_path / "python312.zip",
            stdlib,
            dynamic,
        ),
    )

    baseline = readiness._stdlib_file_inventory(paths)
    nested = stdlib / "site-packages"
    nested.mkdir()
    (nested / "unowned_shadow.py").write_text("VALUE = 1\n", encoding="utf-8")
    changed = readiness._stdlib_file_inventory(paths)
    assert baseline.directory_count >= 2  # stdlib root plus lib-dynload
    assert changed.directory_count == baseline.directory_count + 1
    assert changed.file_count == baseline.file_count + 1
    assert changed.inventory_sha256 != baseline.inventory_sha256


def test_complete_import_tree_binds_unowned_source_sourceless_pyc_and_namespace_dirs(
    tmp_path: Path,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    purelib = runtime_prefix / "lib/python3.12/site-packages"
    purelib.mkdir(parents=True)
    (purelib / "baseline.py").write_text("VALUE = 1\n", encoding="utf-8")
    stdlib = tmp_path / "stdlib"
    dynamic = stdlib / "lib-dynload"
    dynamic.mkdir(parents=True)
    (stdlib / "baseline.py").write_text("VALUE = 1\n", encoding="utf-8")
    paths = readiness._RuntimePathBinding(
        prefix=runtime_prefix,
        exec_prefix=runtime_prefix,
        purelib=purelib,
        platlib=purelib,
        stdlib=stdlib,
        no_site_stdlib_search_paths=(
            tmp_path / "python312.zip",
            stdlib,
            dynamic,
        ),
    )

    baseline = readiness._dependency_import_tree_file_inventory(paths)
    unowned_source = purelib / "unowned_shadow.py"
    unowned_source.write_text("VALUE = 'shadow'\n", encoding="utf-8")
    with_source = readiness._dependency_import_tree_file_inventory(paths)
    assert with_source.file_count == baseline.file_count + 1
    assert with_source.inventory_sha256 != baseline.inventory_sha256

    unowned_source.unlink()
    pyc_source = tmp_path / "legacy_source.py"
    pyc_source.write_text("VALUE = 'legacy'\n", encoding="utf-8")
    sourceless_pyc = purelib / "legacy_shadow.pyc"
    py_compile.compile(pyc_source.as_posix(), cfile=sourceless_pyc.as_posix(), doraise=True)
    with_pyc = readiness._dependency_import_tree_file_inventory(paths)
    assert with_pyc.file_count == baseline.file_count + 1
    assert with_pyc.inventory_sha256 not in {
        baseline.inventory_sha256,
        with_source.inventory_sha256,
    }

    sourceless_pyc.unlink()
    namespace = purelib / "unowned_namespace"
    namespace.mkdir()
    with_namespace = readiness._dependency_import_tree_file_inventory(paths)
    assert with_namespace.file_count == baseline.file_count
    assert with_namespace.directory_count == baseline.directory_count + 1
    assert with_namespace.inventory_sha256 != baseline.inventory_sha256

    stdlib_baseline = readiness._stdlib_file_inventory(paths)
    stdlib_pyc = stdlib / "legacy_stdlib_shadow.pyc"
    py_compile.compile(pyc_source.as_posix(), cfile=stdlib_pyc.as_posix(), doraise=True)
    stdlib_changed = readiness._stdlib_file_inventory(paths)
    assert stdlib_changed.file_count == stdlib_baseline.file_count + 1
    assert stdlib_changed.inventory_sha256 != stdlib_baseline.inventory_sha256


def test_draft_is_deterministic_nonexecuting_and_binds_exact_source_support(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("draft construction executed a subprocess")

    monkeypatch.setattr(readiness.subprocess, "run", forbidden)
    first = build_readiness_draft(repository_root=repository)
    second = build_readiness_draft(repository_root=repository)
    assert first.base_body == second.base_body
    assert first.source_archive == second.source_archive
    assert "certification_contract" not in first.base_body
    assert "authorization" not in first.base_body
    assert first.base_body["source_literal_outcome_guard"] == {
        "scope": "source_literals_only_not_managed_or_external_execution_history",
        "learner_outcome_constant": False,
        "ledger_all_false": True,
        "ledger_entry_count": 3,
        "execution_absence_attested": False,
    }
    governance = first.base_body["execution_governance"]
    assert governance["initial_started_record_count"] == 0
    assert governance["initial_completed_record_count"] == 0
    assert governance["initial_protected_record_count"] == 0
    assert governance["protected_execution_permitted"] is False
    assert "cannot prove non-execution" in governance["managed_boundary_scope"]
    assert "external clone" in first.base_body["claim_scope"]
    assert first.base_body["worker_execution"]["allowed_entrypoint_modes"] == [
        "--worker-case-v1",
        "--worker-preflight-v1",
        "--worker-aggregate-v1",
        "--worker-threshold-freeze-v1",
        "--worker-protected-plan-v1",
    ]
    assert first.base_body["worker_execution"]["isolated_flag"] is None
    assert first.base_body["worker_execution"]["no_site_flag"] == "-S"
    assert first.base_body["worker_execution"]["dont_write_bytecode_flag"] == "-B"
    assert first.base_body["worker_execution"]["safe_path_flag"] == "-P"
    assert first.base_body["worker_execution"]["pycache_prefix_option"] == (
        "-X pycache_prefix={fresh_empty_separate_bytecode_cache_root}"
    )
    assert (
        "complete_import_tree_inventories"
        in (first.base_body["worker_execution"]["bytecode_cache_policy"])
    )
    assert (
        "without_site_or_pth_processing"
        in first.base_body["worker_execution"]["runtime_path_policy"]
    )

    source = first.base_body["source_snapshot"]
    assert isinstance(source, dict)
    manifest = source["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["calibration_runner_module"] == _RUNNER_MODULE
    assert _RUNNER_MODULE in manifest["root_modules"]
    assert readiness._THRESHOLD_FREEZE_MODULE in manifest["root_modules"]
    assert readiness._PROTECTED_PLAN_MODULE in manifest["root_modules"]
    support = {item["locator"]: item["role"] for item in manifest["support_files"]}
    assert support["pyproject.toml"] == "dependency_lock"
    assert support["uv.lock"] == "dependency_lock"
    assert support["tests/test_hidden_regime_trace_audit.py"] == "certification_source"
    assert support["tests/test_hidden_regime_checkpoint.py"] == "certification_source"
    assert support["tests/test_hidden_regime_factorial_thresholds.py"] == (
        "certification_source"
    )
    assert support["tests/test_hidden_regime_factorial_protected_plan.py"] == (
        "certification_source"
    )
    with zipfile.ZipFile(io.BytesIO(first.source_archive)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert "alberta_framework/evaluation/hidden_regime_calibration_readiness.py" in (
            archive.namelist()
        )
        assert "alberta_framework/evaluation/hidden_regime_factorial_calibration.py" in (
            archive.namelist()
        )
        assert "alberta_framework/evaluation/hidden_regime_factorial_thresholds.py" in (
            archive.namelist()
        )
        assert "alberta_framework/evaluation/hidden_regime_factorial_protected_plan.py" in (
            archive.namelist()
        )
        assert "tests/test_hidden_regime_factorial_thresholds.py" in archive.namelist()
        assert "tests/test_hidden_regime_factorial_protected_plan.py" in archive.namelist()
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())


def test_live_protocol_binding_matches_final_frozen_digest_and_v5_v3_schemas() -> None:
    binding = _REAL_PROTOCOL_BINDING()
    assert binding["protocol_payload_sha256"] == (
        "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
    )
    assert binding["seed_snapshot_sha256"] == readiness.SEED_SNAPSHOT_SHA256
    assert binding["development_summary_schema"].endswith(".development.v5")
    assert binding["primitive_trace_schema"].endswith(".primitive-trace.v3")


def test_missing_or_symlinked_mandatory_runner_fails_closed(
    repository: Path,
) -> None:
    runner = repository / _module_locator(_RUNNER_MODULE)
    original = runner.read_bytes()
    runner.unlink()
    with pytest.raises(ReadinessError, match="mandatory calibration runner"):
        build_readiness_draft(repository_root=repository)

    outside = repository.parent / "outside.py"
    outside.write_bytes(original)
    runner.symlink_to(outside)
    with pytest.raises(ReadinessError, match="not a regular file"):
        build_readiness_draft(repository_root=repository)


def test_checkpoint_equivalence_certification_binds_exact_nodes_cuts_and_digest() -> None:
    spec = next(
        item
        for item in CERTIFICATION_SPECS
        if item.certification_id
        == "checkpoint_resume_and_decentralized_role_bit_exact_equivalence"
    )
    assert spec.exact_file_test_inventory is True
    assert all("::test_" in node_id for node_id in spec.node_ids)
    assert spec.checkpoint_cut_runtime_node_id == (
        "tests/test_hidden_regime_checkpoint.py::"
        "test_json_roundtripped_lifecycle_chunks_equal_one_shot_bit_for_bit"
    )
    assert spec.checkpoint_cut_fixture_name == "direct_lifecycle_chunks"
    assert spec.checkpoint_cut_ids == (
        "inside_lease",
        "lease_boundary",
        "regime_boundary",
        "scratch_retest",
        "commit",
        "replacement",
    )
    assert spec.node_manifest == {
        "schema": readiness.READINESS_CERTIFICATION_NODE_MANIFEST_SCHEMA,
        "certification_id": spec.certification_id,
        "node_ids": list(spec.node_ids),
        "exact_file_test_inventory": True,
        "runtime_contract": spec.runtime_contract,
        "runtime_contract_sha256": spec.runtime_contract_sha256,
    }
    assert spec.node_manifest_sha256 == canonical_sha256(spec.node_manifest)
    payload = next(
        item
        for item in readiness._spec_payload()
        if item["certification_id"] == spec.certification_id
    )
    assert payload["node_manifest"] == spec.node_manifest
    assert payload["node_manifest_sha256"] == spec.node_manifest_sha256


def test_live_checkpoint_equivalence_source_matches_exact_node_and_cut_inventory() -> None:
    readiness._validate_certification_node_sources(readiness._REPO_ROOT)


def test_checkpoint_equivalence_inventory_rejects_missing_or_unbound_test(
    repository: Path,
) -> None:
    spec = next(item for item in CERTIFICATION_SPECS if item.exact_file_test_inventory)
    removed_node = spec.node_ids[-1]
    locator_text, _, function_name = removed_node.partition("::")
    locator = repository / locator_text
    source = locator.read_text(encoding="utf-8")
    locator.write_text(
        source.replace(f"def {function_name}():", f"def removed_{function_name}():", 1),
        encoding="utf-8",
    )
    with pytest.raises(ReadinessError, match="certification node is absent"):
        build_readiness_draft(repository_root=repository)


def test_checkpoint_equivalence_inventory_rejects_new_unbound_test(
    repository: Path,
) -> None:
    spec = next(item for item in CERTIFICATION_SPECS if item.exact_file_test_inventory)
    locator_text = spec.node_ids[0].partition("::")[0]
    locator = repository / locator_text
    locator.write_text(
        locator.read_text(encoding="utf-8")
        + "\n\ndef test_unbound_checkpoint_equivalence():\n    assert True\n",
        encoding="utf-8",
    )
    with pytest.raises(ReadinessError, match="exact certification test inventory differs"):
        build_readiness_draft(repository_root=repository)


def test_certification_source_rejects_duplicate_top_level_test_before_inventory_collapse(
    repository: Path,
) -> None:
    spec = next(item for item in CERTIFICATION_SPECS if item.exact_file_test_inventory)
    duplicate_node = spec.node_ids[0]
    locator_text, _, function_name = duplicate_node.partition("::")
    locator = repository / locator_text
    locator.write_text(
        locator.read_text(encoding="utf-8")
        + f"\n\ndef {function_name}():\n    assert True\n",
        encoding="utf-8",
    )
    with pytest.raises(ReadinessError, match="duplicate top-level test definitions"):
        build_readiness_draft(repository_root=repository)


def _capture_rejected_real_certification_manifest(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: readiness.CertificationSpec,
) -> dict[str, object]:
    monkeypatch.setattr(readiness, "CERTIFICATION_SPECS", (spec,))
    draft = build_readiness_draft(repository_root=repository)
    monkeypatch.setattr(
        readiness,
        "_run_clean_runtime_reconstruction",
        _fake_runtime_reconstruction_record,
    )
    captured: list[dict[str, object]] = []
    real_validate = readiness._validate_certification_execution_manifest

    def capture_then_validate(
        manifest: object,
        certification_spec: readiness.CertificationSpec,
    ) -> dict[str, object]:
        assert isinstance(manifest, dict)
        captured.append(copy.deepcopy(manifest))
        return real_validate(manifest, certification_spec)

    monkeypatch.setattr(
        readiness,
        "_validate_certification_execution_manifest",
        capture_then_validate,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    with pytest.raises(ReadinessError, match="execution manifest is not passed"):
        run_readiness_certifications(
            draft,
            authorize_certification_execution=True,
            timeout_seconds_per_group=30,
        )
    assert len(captured) == 1
    return captured[0]


def test_certification_harness_rejects_zero_collection_and_deselection(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CERTIFICATION_SPECS[0]
    source_path = repository / spec.node_ids[0]
    source_path.write_text("def test_full_file_certification():\n    assert True\n")
    (repository / "tests/conftest.py").write_text(
        """def pytest_collection_modifyitems(config, items):
    config.hook.pytest_deselected(items=list(items))
    items[:] = []
""",
        encoding="utf-8",
    )
    manifest = _capture_rejected_real_certification_manifest(
        repository,
        monkeypatch,
        spec,
    )
    assert manifest["status"] == "rejected"
    assert manifest["collected_node_count"] == 0
    violations = manifest["violations"]
    assert isinstance(violations, list)
    assert "zero_collection" in violations
    assert "deselection_observed" in violations


def test_certification_execution_manifest_is_deterministic_and_phase_complete(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CERTIFICATION_SPECS[0]
    monkeypatch.setattr(readiness, "CERTIFICATION_SPECS", (spec,))
    draft = build_readiness_draft(repository_root=repository)
    monkeypatch.setattr(
        readiness,
        "_run_clean_runtime_reconstruction",
        _fake_runtime_reconstruction_record,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    first = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
        timeout_seconds_per_group=30,
    )
    second = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
        timeout_seconds_per_group=30,
    )
    first_manifest = first.records[0]["execution_manifest"]
    second_manifest = second.records[0]["execution_manifest"]
    assert first_manifest == second_manifest
    assert first.records[0]["execution_manifest_sha256"] == canonical_sha256(
        first_manifest
    )
    assert first_manifest["status"] == "passed"
    assert first_manifest["collected_node_count"] == 1
    assert first_manifest["deselected_node_ids"] == []
    assert first_manifest["violations"] == []
    outcome = first_manifest["per_node"][0]
    assert outcome["setup"] == {"outcome": "passed", "was_xfail": False}
    assert outcome["call"] == {"outcome": "passed", "was_xfail": False}
    assert outcome["teardown"] == {"outcome": "passed", "was_xfail": False}


def test_certification_harness_rejects_every_nonpass_phase_skip_xfail_and_xpass(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CERTIFICATION_SPECS[0]
    source_path = repository / spec.node_ids[0]
    source_path.write_text(
        """import pytest

@pytest.fixture
def setup_failure():
    raise RuntimeError("setup failure")

@pytest.fixture
def teardown_failure():
    yield None
    raise RuntimeError("teardown failure")

@pytest.mark.skip(reason="skip is forbidden")
def test_skip():
    pass

@pytest.mark.xfail(reason="xfail is forbidden")
def test_xfail():
    assert False

@pytest.mark.xfail(reason="xpass is forbidden")
def test_xpass():
    pass

def test_setup_failure(setup_failure):
    pass

def test_call_failure():
    assert False

def test_teardown_failure(teardown_failure):
    pass
""",
        encoding="utf-8",
    )
    manifest = _capture_rejected_real_certification_manifest(
        repository,
        monkeypatch,
        spec,
    )
    violations = manifest["violations"]
    assert isinstance(violations, list)
    assert any(item.startswith("phase_not_passed:") for item in violations)
    assert any(item.endswith(":setup") for item in violations)
    assert any(item.endswith(":call") for item in violations)
    assert any(item.endswith(":teardown") for item in violations)
    assert any(item.startswith("phase_cardinality_differs:") for item in violations)
    assert any(item.startswith("xfail_or_xpass_observed:") for item in violations)


def test_checkpoint_cut_runtime_contract_rejects_arbitrary_positive_unique_constants(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(item for item in CERTIFICATION_SPECS if item.checkpoint_cut_ids)
    cut_node = spec.checkpoint_cut_runtime_node_id
    assert cut_node is not None
    locator_text, _, function_name = cut_node.partition("::")
    locator = repository / locator_text
    source = locator.read_text(encoding="utf-8")
    expected_cuts = {
        cut_id: index for index, cut_id in enumerate(spec.checkpoint_cut_ids, start=1)
    }
    arbitrary_cuts = {cut_id: value + 10 for cut_id, value in expected_cuts.items()}
    source = source.replace(
        f"return ((), {expected_cuts!r})",
        f"return ((), {arbitrary_cuts!r})",
        1,
    )
    arguments = (
        f"{spec.checkpoint_cut_fixture_name}, {spec.checkpoint_cut_trace_fixture_name}"
    )
    anchor = (
        f"def {function_name}({arguments}):\n"
        "    assert callable(bound_runner.main)"
    )
    source = source.replace(
        anchor,
        anchor + "\n" + f"    assert True or set({spec.checkpoint_cut_ids!r})",
        1,
    )
    assert f"return ((), {arbitrary_cuts!r})" in source
    assert "assert True or set(" in source
    locator.write_text(source, encoding="utf-8")
    manifest = _capture_rejected_real_certification_manifest(
        repository,
        monkeypatch,
        spec,
    )
    violations = manifest["violations"]
    assert isinstance(violations, list)
    assert "checkpoint_cut_trace_semantics_differ" in violations


def test_threshold_engine_and_exact_main_dispatch_have_dedicated_certification() -> None:
    spec = next(
        item
        for item in CERTIFICATION_SPECS
        if item.certification_id == "threshold_freeze_engine_and_exact_worker_main_dispatch"
    )
    assert spec.node_ids == (
        "tests/test_hidden_regime_factorial_thresholds.py",
        "tests/test_hidden_regime_factorial_calibration.py::"
        "test_threshold_worker_main_dispatches_exact_content_addressed_inputs",
    )
    assert spec.runtime_contract["checkpoint_cut_contract"] is None


def test_protected_plan_and_exact_main_dispatch_have_dedicated_certification() -> None:
    spec = next(
        item
        for item in CERTIFICATION_SPECS
        if item.certification_id
        == "protected_plan_derivation_and_exact_worker_main_dispatch"
    )
    assert spec.node_ids == (
        "tests/test_hidden_regime_factorial_protected_plan.py",
        "tests/test_hidden_regime_factorial_calibration.py::"
        "test_protected_plan_worker_main_dispatches_exact_nonauthorizing_inputs",
    )
    assert spec.runtime_contract["checkpoint_cut_contract"] is None


def test_certifications_are_derived_from_verified_extraction_and_sealed(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_readiness_draft(repository_root=repository)
    with pytest.raises(ReadinessError, match="explicit authorization"):
        run_readiness_certifications(
            draft,
            authorize_certification_execution=False,
        )

    calls: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(readiness.subprocess, "run", _successful_certification_run(calls))
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
    )
    assert len(calls) == len(CERTIFICATION_SPECS) + 1
    assert all(repository not in cwd.parents and cwd != repository for _, cwd in calls)
    assert tuple(record["status"] for record in bundle.records) == ("passed",) * len(
        CERTIFICATION_SPECS
    )
    assert all(
        record["harness_sha256"] == readiness._CERTIFICATION_BOOTSTRAP_SHA256
        for record in bundle.records
    )
    prepared = finalize_readiness_receipt(draft, bundle)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    ).valid


def test_certification_scrubs_inherited_pytest_environment_and_binds_policy(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1 --pdb")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    draft = build_readiness_draft(repository_root=repository)
    calls: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(readiness.subprocess, "run", _successful_certification_run(calls))
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
    )
    assert len(calls) == len(CERTIFICATION_SPECS) + 1
    assert all(
        record["environment_policy"] == readiness._CERTIFICATION_ENVIRONMENT_POLICY
        for record in bundle.records
    )
    assert all(
        specification["environment_policy"] == readiness._CERTIFICATION_ENVIRONMENT_POLICY
        for specification in readiness._spec_payload()
    )

    forged_records = [dict(record) for record in bundle.records]
    forged_records[0]["status"] = "failed"
    forged = dataclasses.replace(bundle, records=tuple(forged_records))
    with pytest.raises(ReadinessError, match="seal"):
        finalize_readiness_receipt(draft, forged)


def test_certification_rejects_runtime_not_reconstructible_in_clean_child(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_readiness_draft(repository_root=repository)
    calls: list[tuple[tuple[str, ...], Path]] = []
    normal_run = _successful_certification_run(calls)

    def drifted_reconstruction(
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        if command[7] != readiness._RUNTIME_RECONSTRUCTION_BOOTSTRAP:
            return normal_run(command, cwd=cwd, env=env, **kwargs)
        calls.append((tuple(command), Path(cwd)))
        drifted = copy.deepcopy(_fake_runtime_identity())
        dependencies = drifted["dependencies"]
        assert isinstance(dependencies, dict)
        dependencies["dependency_import_tree_file_inventory_sha256"] = "2" * 64
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=canonical_json_bytes(drifted),
            stderr=b"",
        )

    monkeypatch.setattr(readiness.subprocess, "run", drifted_reconstruction)
    with pytest.raises(ReadinessError, match="clean-child runtime identity differs"):
        run_readiness_certifications(
            draft,
            authorize_certification_execution=True,
        )
    assert len(calls) == 1


def test_certification_harness_really_runs_snapshot_tests_without_checkout(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_readiness_draft(repository_root=repository)
    monkeypatch.setattr(
        readiness,
        "_run_clean_runtime_reconstruction",
        _fake_runtime_reconstruction_record,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
        timeout_seconds_per_group=30,
    )
    assert len(bundle.records) == len(CERTIFICATION_SPECS)
    assert all(record["status"] == "passed" for record in bundle.records)
    assert all(record["stdout"]["byte_size"] > 0 for record in bundle.records)


def test_in_memory_validator_rejects_body_and_archive_tampering(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    ).valid

    changed = copy.deepcopy(prepared.payload)
    changed["body"]["source_literal_outcome_guard"]["ledger_all_false"] = False
    changed["receipt_sha256"] = canonical_sha256(changed["body"])
    validation = validate_readiness_receipt(
        changed,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("source literal guard" in error for error in validation.errors)

    governance_tamper = copy.deepcopy(prepared.payload)
    governance_tamper["body"]["execution_governance"]["initial_started_record_count"] = 1
    governance_tamper["receipt_sha256"] = canonical_sha256(governance_tamper["body"])
    validation = validate_readiness_receipt(
        governance_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("pristine deterministic genesis" in error for error in validation.errors)

    worker_mode_tamper = copy.deepcopy(prepared.payload)
    worker_mode_tamper["body"]["worker_execution"]["allowed_entrypoint_modes"] = [
        "--worker-case-v1"
    ]
    worker_mode_tamper["receipt_sha256"] = canonical_sha256(worker_mode_tamper["body"])
    validation = validate_readiness_receipt(
        worker_mode_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("allowed worker entrypoint modes" in error for error in validation.errors)

    exit_code_type_tamper = copy.deepcopy(prepared.payload)
    exit_code_type_tamper["body"]["certification_contract"]["records"][0]["exit_code"] = False
    records = exit_code_type_tamper["body"]["certification_contract"]["records"]
    exit_code_type_tamper["body"]["certification_contract"]["records_sha256"] = canonical_sha256(
        records
    )
    exit_code_type_tamper["receipt_sha256"] = canonical_sha256(exit_code_type_tamper["body"])
    validation = validate_readiness_receipt(
        exit_code_type_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("strict integer zero" in error for error in validation.errors)

    node_manifest_tamper = copy.deepcopy(prepared.payload)
    node_manifest_tamper["body"]["certification_contract"]["records"][-1][
        "node_manifest_sha256"
    ] = "0" * 64
    records = node_manifest_tamper["body"]["certification_contract"]["records"]
    node_manifest_tamper["body"]["certification_contract"]["records_sha256"] = (
        canonical_sha256(records)
    )
    node_manifest_tamper["receipt_sha256"] = canonical_sha256(node_manifest_tamper["body"])
    validation = validate_readiness_receipt(
        node_manifest_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("node manifest digest" in error for error in validation.errors)

    execution_manifest_tamper = copy.deepcopy(prepared.payload)
    execution_record = execution_manifest_tamper["body"]["certification_contract"][
        "records"
    ][-1]
    execution_manifest = execution_record["execution_manifest"]
    execution_manifest["per_node"][0]["call"]["outcome"] = "skipped"
    execution_record["execution_manifest_sha256"] = canonical_sha256(execution_manifest)
    records = execution_manifest_tamper["body"]["certification_contract"]["records"]
    execution_manifest_tamper["body"]["certification_contract"]["records_sha256"] = (
        canonical_sha256(records)
    )
    execution_manifest_tamper["receipt_sha256"] = canonical_sha256(
        execution_manifest_tamper["body"]
    )
    validation = validate_readiness_receipt(
        execution_manifest_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("call phase is not passed" in error for error in validation.errors)

    cut_semantic_tamper = copy.deepcopy(prepared.payload)
    cut_record = cut_semantic_tamper["body"]["certification_contract"]["records"][-1]
    cut_manifest = cut_record["execution_manifest"]
    cut_observation = cut_manifest["checkpoint_cut_observation"]
    cut_observation["semantic_observations"][0]["observed_cut_value"] += 1
    cut_record["execution_manifest_sha256"] = canonical_sha256(cut_manifest)
    records = cut_semantic_tamper["body"]["certification_contract"]["records"]
    cut_semantic_tamper["body"]["certification_contract"]["records_sha256"] = (
        canonical_sha256(records)
    )
    cut_semantic_tamper["receipt_sha256"] = canonical_sha256(cut_semantic_tamper["body"])
    validation = validate_readiness_receipt(
        cut_semantic_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("independently reconstructed trace event" in error for error in validation.errors)

    reconstruction_stdout_tamper = copy.deepcopy(prepared.payload)
    reconstruction = reconstruction_stdout_tamper["body"]["certification_contract"][
        "runtime_reconstruction_record"
    ]
    reconstruction["stdout"] = {"byte_size": 0, "sha256": "0" * 64}
    certification = reconstruction_stdout_tamper["body"]["certification_contract"]
    certification["runtime_reconstruction_record_sha256"] = canonical_sha256(reconstruction)
    reconstruction_stdout_tamper["receipt_sha256"] = canonical_sha256(
        reconstruction_stdout_tamper["body"]
    )
    validation = validate_readiness_receipt(
        reconstruction_stdout_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("stdout does not bind" in error for error in validation.errors)

    archive_tamper = bytearray(prepared.source_archive)
    archive_tamper[-1] ^= 1
    validation = validate_readiness_receipt(
        prepared.payload,
        bytes(archive_tamper),
        repository_root=repository,
    )
    assert not validation.valid
    assert any("archive digest" in error for error in validation.errors)


def test_current_source_drift_is_rejected_but_bound_archive_remains_self_valid(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(runner.read_text(encoding="utf-8") + "\nDRIFT = True\n", encoding="utf-8")
    current = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=True,
    )
    assert not current.valid
    assert any("source closure" in error for error in current.errors)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=False,
    ).valid


def test_runtime_recheck_refuses_installed_distribution_file_inventory_drift(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    drifted = copy.deepcopy(_fake_runtime_identity())
    dependencies = drifted["dependencies"]
    assert isinstance(dependencies, dict)
    dependencies["installed_distribution_file_inventory_sha256"] = "2" * 64
    monkeypatch.setattr(readiness, "_build_runtime_identity", lambda: drifted)

    runtime_checked = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=False,
        recheck_runtime=True,
    )
    assert not runtime_checked.valid
    assert any(
        "runtime/JAX/device/dependency/environment identity drift" in error
        for error in (runtime_checked.errors)
    )
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=False,
        recheck_runtime=False,
    ).valid


def test_runner_facing_validator_returns_only_bound_identities(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    validated = require_validated_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    )
    assert validated.payload == prepared.payload
    assert validated.receipt_sha256 == prepared.payload["receipt_sha256"]
    assert validated.source_archive_sha256 == readiness._sha256_bytes(prepared.source_archive)
    assert validated.calibration_runner_module == _RUNNER_MODULE
    assert (
        validated.execution_genesis_sha256
        == prepared.payload["body"]["execution_governance"]["genesis_sha256"]
    )

    bad = copy.deepcopy(prepared.payload)
    bad["body"]["authorization"]["protected_candidate_execution_permitted"] = True
    bad["receipt_sha256"] = canonical_sha256(bad["body"])
    with pytest.raises(ReadinessError, match="authorization policy"):
        require_validated_readiness_receipt(
            bad,
            prepared.source_archive,
            repository_root=repository,
        )


def test_publication_is_content_addressed_immutable_new_only_and_no_symlink_root(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "published"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    assert published.directory.name == prepared.payload["receipt_sha256"]
    assert stat.S_IMODE(published.directory.stat().st_mode) == 0o555
    assert stat.S_IMODE(published.receipt_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(published.source_archive_path.stat().st_mode) == 0o444
    assert published.receipt_path.read_bytes() == canonical_json_bytes(prepared.payload)
    assert validate_published_readiness_receipt(
        published.directory,
        repository_root=repository,
    ).valid
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_readiness_receipt(
            prepared,
            publication_root,
            authorize_publication=True,
        )

    actual_root = tmp_path / "actual-root"
    actual_root.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(actual_root, target_is_directory=True)
    with pytest.raises(ReadinessError, match="symlink"):
        publish_readiness_receipt(
            prepared,
            symlink_root,
            authorize_publication=True,
        )


def test_readiness_publication_exposes_only_a_complete_atomic_bundle(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "published"
    publication_root.mkdir()
    digest = prepared.payload["receipt_sha256"]
    assert isinstance(digest, str)
    real_install = readiness._install_directory_new_only
    observations: list[str] = []

    def checked_install(root_fd: int, staging_name: str, final_name: str) -> None:
        assert final_name == digest
        assert not (publication_root / final_name).exists()
        staging = publication_root / staging_name
        assert stat.S_IMODE(staging.stat().st_mode) == 0o555
        assert sorted(path.name for path in staging.iterdir()) == ["readiness.json", "source.zip"]
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in staging.iterdir())
        assert (staging / "readiness.json").read_bytes() == canonical_json_bytes(
            prepared.payload
        )
        assert (staging / "source.zip").read_bytes() == prepared.source_archive
        observations.append(staging_name)
        real_install(root_fd, staging_name, final_name)

    monkeypatch.setattr(readiness, "_install_directory_new_only", checked_install)
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    assert observations and published.directory == publication_root / digest
    assert not tuple(publication_root.glob(".staging-readiness-*"))

    failed_root = tmp_path / "failed-publication"
    failed_root.mkdir()

    def fail_install(root_fd: int, staging_name: str, final_name: str) -> None:
        del root_fd, staging_name, final_name
        raise OSError(errno.EIO, "synthetic atomic installation failure")

    monkeypatch.setattr(readiness, "_install_directory_new_only", fail_install)
    with pytest.raises(OSError, match="synthetic atomic installation failure"):
        publish_readiness_receipt(
            prepared,
            failed_root,
            authorize_publication=True,
        )
    assert not (failed_root / digest).exists()
    assert not tuple(failed_root.glob(".staging-readiness-*"))


def test_published_symlink_and_byte_tamper_are_rejected(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = publish_readiness_receipt(prepared, first_root, authorize_publication=True)
    external = tmp_path / "external.json"
    external.write_bytes(canonical_json_bytes(prepared.payload))
    external.chmod(0o444)
    first.directory.chmod(0o755)
    first.receipt_path.unlink()
    first.receipt_path.symlink_to(external)
    first.directory.chmod(0o555)
    validation = validate_published_readiness_receipt(
        first.directory,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("symlink" in error for error in validation.errors)

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = publish_readiness_receipt(prepared, second_root, authorize_publication=True)
    second.directory.chmod(0o755)
    second.source_archive_path.chmod(0o644)
    tampered = bytearray(second.source_archive_path.read_bytes())
    tampered[-1] ^= 1
    second.source_archive_path.write_bytes(tampered)
    second.source_archive_path.chmod(0o444)
    second.directory.chmod(0o555)
    validation = validate_published_readiness_receipt(
        second.directory,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("archive digest" in error for error in validation.errors)


@pytest.mark.parametrize("mutation", ["writable", "hardlink"])
def test_readiness_immutable_reader_rejects_mode_or_link_mutation_during_read(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "readiness.json"
    path.write_bytes(b"immutable-readiness")
    path.chmod(0o444)
    alias = tmp_path / "readiness-alias.json"
    real_read = os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        raw = real_read(descriptor, size)
        if raw and not mutated:
            mutated = True
            if mutation == "writable":
                path.chmod(0o644)
            else:
                os.link(path, alias)
        return raw

    monkeypatch.setattr(readiness.os, "read", mutate_after_first_read)
    with pytest.raises(ReadinessError, match="changed or was replaced during read"):
        readiness._open_immutable_regular(path, max_bytes=1024)


def test_readiness_immutable_reader_rejects_parent_substitution_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "bound"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    parent.mkdir()
    replacement.mkdir()
    path = parent / "readiness.json"
    path.write_bytes(b"original-readiness")
    path.chmod(0o444)
    replacement_path = replacement / path.name
    replacement_path.write_bytes(b"replacement-readiness")
    replacement_path.chmod(0o444)
    real_read = os.read
    mutated = False

    def substitute_parent_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        raw = real_read(descriptor, size)
        if raw and not mutated:
            mutated = True
            parent.rename(moved)
            replacement.rename(parent)
        return raw

    monkeypatch.setattr(readiness.os, "read", substitute_parent_after_first_read)
    with pytest.raises(ReadinessError, match="parent changed during read"):
        readiness._open_immutable_regular(path, max_bytes=1024)


def test_bound_worker_uses_empty_cwd_and_zipimport_snapshot(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "worker"
    publication_root.mkdir()
    published = publish_readiness_receipt(prepared, publication_root, authorize_publication=True)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    with pytest.raises(ReadinessError, match="explicit authorization"):
        execute_bound_calibration_worker(
            published.directory,
            authorize_calibration_execution=False,
        )
    with pytest.raises(ReadinessError, match="entrypoint mode"):
        execute_bound_calibration_worker(
            published.directory,
            ("--unsupported-worker-mode",),
            authorize_calibration_execution=True,
        )
    monkeypatch.setenv("LD_PRELOAD", "/definitely/not/a/bound/library.so")
    monkeypatch.setenv("LD_AUDIT", "/definitely/not/a/bound/auditor.so")
    monkeypatch.setenv("HOME", "/untrusted-parent-home")
    monkeypatch.setenv("PATH", "/untrusted-parent-path")
    monkeypatch.setenv("ALBERTA_UNTRUSTED_PARENT_MARKER", "must-not-cross")
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "7"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    report = json.loads(completed.stdout)
    assert report["argv"][0] == "--worker-preflight-v1"
    assert report["argv"][2] == "7"
    assert Path(report["argv"][1]).name == prepared.payload["receipt_sha256"]
    assert report["argv"][1] != published.directory.as_posix()
    assert report["cwd_empty"] is True
    assert report["environment"] == {"LC_ALL": "C", "PYTHONHASHSEED": "0"}
    assert "/source.zip/alberta_framework/evaluation/" in report["origin"]
    assert report["dont_write_bytecode"] is True
    assert report["pycache_prefix_set"] is True
    assert report["no_site"] is True
    assert report["virtualenv_hook_absent"] is True
    assert report["sys_prefix"] == _RUNTIME_PREFIX.as_posix()
    assert report["sys_exec_prefix"] == _RUNTIME_EXEC_PREFIX.as_posix()
    expected_site_paths = list(
        dict.fromkeys((_RUNTIME_PURELIB.as_posix(), _RUNTIME_PLATLIB.as_posix()))
    )
    assert report["sys_path"][1:] == [*_NO_SITE_STDLIB_PATHS, *expected_site_paths]

    alias_component = published.directory.parent / "alias-component"
    alias_component.mkdir()
    aliased_readiness_argument = alias_component / ".." / published.directory.name
    completed = execute_bound_calibration_worker(
        aliased_readiness_argument,
        _fake_worker_arguments(published, "normalized-alias"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    aliased_argv = json.loads(completed.stdout)["argv"]
    assert aliased_argv[2] == "normalized-alias"
    assert "alias-component" not in aliased_argv[1]

    alternate_root = tmp_path / "alternate-worker"
    alternate_root.mkdir()
    alternate = publish_readiness_receipt(
        prepared,
        alternate_root,
        authorize_publication=True,
    )
    with pytest.raises(ReadinessError, match="readiness argument differs"):
        execute_bound_calibration_worker(
            published.directory,
            _fake_worker_arguments(alternate, "mixed-readiness"),
            authorize_calibration_execution=True,
            timeout_seconds=30,
        )

    drifted = copy.deepcopy(_fake_runtime_identity())
    dependencies = drifted["dependencies"]
    assert isinstance(dependencies, dict)
    dependencies["installed_distribution_file_inventory_sha256"] = "2" * 64
    monkeypatch.setattr(readiness, "_build_runtime_identity", lambda: drifted)
    with pytest.raises(
        ReadinessError,
        match="runtime identity drift immediately before bound worker launch",
    ):
        execute_bound_calibration_worker(
            published.directory,
            _fake_worker_arguments(published, "7"),
            authorize_calibration_execution=True,
            timeout_seconds=30,
        )


def test_bound_worker_fixes_hash_seed_and_allows_only_bound_jax_environment_effects(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(
        '''\
"""Fake runner exercising the real JAX import boundary."""
import json
import os

import jax
from scipy.stats import t

def main(argv):
    print(json.dumps({
        "backend": jax.default_backend(),
        "environment": dict(os.environ),
        "hash": hash("hidden-regime"),
        "scipy_probe": float(t.cdf(0.0, 1.0)),
    }, sort_keys=True))
    return 0
''',
        encoding="utf-8",
    )
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    reports: list[dict[str, object]] = []
    for _ in range(2):
        completed = execute_bound_calibration_worker(
            published.directory,
            _fake_worker_arguments(published, "hash-seed"),
            authorize_calibration_execution=True,
            timeout_seconds=30,
        )
        assert completed.returncode == 0, completed.stderr.decode()
        reports.append(json.loads(completed.stdout))
    assert reports[0]["hash"] == reports[1]["hash"]
    assert reports[0]["environment"] == reports[1]["environment"] == {
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "TPU_SKIP_MDS_QUERY": "1",
    }


def test_runtime_batch_guard_amortizes_full_identity_to_batch_boundaries(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    full_identity_calls = 0

    def counted_identity() -> dict[str, object]:
        nonlocal full_identity_calls
        full_identity_calls += 1
        return _fake_runtime_identity()

    monkeypatch.setattr(readiness, "_build_runtime_identity", counted_identity)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    with bound_calibration_runtime_batch(
        published.directory,
        authorize_batch_execution=True,
    ) as guard:
        validated = readiness.load_validated_published_readiness_bundle(
            published.directory,
            recheck_current=False,
            recheck_runtime=False,
        )
        for transferred in (copy.copy(guard), pickle.loads(pickle.dumps(guard))):
            assert transferred is not guard
            with pytest.raises(ReadinessError, match="inactive"):
                readiness._require_active_runtime_batch_guard(
                    transferred,
                    directory=published.directory,
                    validated=validated,
                )
        if hasattr(os, "fork"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"os.fork\(\) was called")
                warnings.filterwarnings("ignore", message="This process .* is multi-threaded")
                child = os.fork()
            if child == 0:
                try:
                    readiness._require_active_runtime_batch_guard(
                        guard,
                        directory=published.directory,
                        validated=validated,
                    )
                except ReadinessError:
                    os._exit(0)
                os._exit(1)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
        for argument in ("first", "second"):
            completed = execute_bound_calibration_worker(
                published.directory,
                _fake_worker_arguments(published, argument),
                authorize_calibration_execution=True,
                timeout_seconds=30,
                runtime_batch_guard=guard,
            )
            assert completed.returncode == 0, completed.stderr.decode()
    assert full_identity_calls == 2

    with pytest.raises(ReadinessError, match="inactive"):
        execute_bound_calibration_worker(
            published.directory,
            _fake_worker_arguments(published, "stale-guard"),
            authorize_calibration_execution=True,
            timeout_seconds=30,
            runtime_batch_guard=guard,
        )


def test_bound_worker_ignores_malicious_adjacent_pyc_and_never_executes_pth(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "isolated-runtime"
    purelib = runtime_prefix / "lib/python3.12/site-packages"
    purelib.mkdir(parents=True)
    dependency = purelib / "probe_dependency.py"
    dependency.write_text("VALUE = 'malicious-bytecode'\n", encoding="utf-8")
    py_compile.compile(
        dependency.as_posix(),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    dependency.write_text("VALUE = 'benign-source'\n", encoding="utf-8")
    pth_marker = tmp_path / "pth-hook-executed"
    (purelib / "hostile_startup.pth").write_text(
        f"import pathlib; pathlib.Path({pth_marker.as_posix()!r}).touch()\n",
        encoding="utf-8",
    )
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(
        f'''\
"""Bound fake calibration runner with an external dependency probe."""
import json
import os
import sys
import probe_dependency

def main(argv):
    print(json.dumps({{
        "argv": list(argv),
        "dependency_value": probe_dependency.VALUE,
        "pth_marker_exists": os.path.exists({pth_marker.as_posix()!r}),
        "no_site": sys.flags.no_site == 1,
        "virtualenv_hook_absent": "_virtualenv" not in sys.modules,
    }}, sort_keys=True))
    return 0
''',
        encoding="utf-8",
    )
    runtime = copy.deepcopy(_fake_runtime_identity())
    runtime_python = runtime["python"]
    assert isinstance(runtime_python, dict)
    runtime_python["prefix"] = runtime_prefix.as_posix()
    runtime_python["exec_prefix"] = runtime_prefix.as_posix()
    runtime_python["purelib"] = purelib.as_posix()
    runtime_python["platlib"] = purelib.as_posix()
    monkeypatch.setattr(readiness, "_build_runtime_identity", lambda: runtime)

    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "worker-publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "probe"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    report = json.loads(completed.stdout)
    assert report["dependency_value"] == "benign-source"
    assert report["pth_marker_exists"] is False
    assert report["no_site"] is True
    assert report["virtualenv_hook_absent"] is True
    assert not pth_marker.exists()


def test_bound_worker_stages_the_exact_validated_bytes_before_publication_path_drift(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    real_read = readiness._read_validated_published_readiness_bytes
    read_count = 0

    def read_then_drift(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal read_count
        result = real_read(*args, **kwargs)
        read_count += 1
        published.directory.chmod(0o755)
        published.source_archive_path.chmod(0o644)
        published.source_archive_path.write_bytes(b"mutated after exact validation")
        published.source_archive_path.chmod(0o444)
        published.directory.chmod(0o555)
        return result

    monkeypatch.setattr(readiness, "_read_validated_published_readiness_bytes", read_then_drift)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "staged"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert read_count == 1
    assert completed.returncode == 0, completed.stderr.decode()
    staged_argv = json.loads(completed.stdout)["argv"]
    assert staged_argv[0] == "--worker-preflight-v1"
    assert staged_argv[2] == "staged"
    assert Path(staged_argv[1]).name == prepared.payload["receipt_sha256"]


def test_bound_worker_bootstrap_rejects_duplicate_receipt_keys_after_parent_validation(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    real_read = readiness._read_validated_published_readiness_bytes

    def inject_duplicate(*args: object, **kwargs: object) -> tuple[object, ...]:
        payload, receipt_raw, archive_raw, validated = real_read(*args, **kwargs)
        duplicate = (
            b'{"receipt_sha256":"'
            + validated.receipt_sha256.encode("ascii")
            + b'",'
            + receipt_raw[1:]
        )
        return payload, duplicate, archive_raw, validated

    monkeypatch.setattr(readiness, "_read_validated_published_readiness_bytes", inject_duplicate)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "duplicate"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode != 0
    assert b"duplicate JSON key" in completed.stderr


def test_bound_worker_rejects_post_target_project_module_substitution(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside_project_module.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(
        f'''\
"""Fake runner that attempts a post-target project-module substitution."""
import importlib.util
import sys

def main(argv):
    name = "alberta_framework.injected_outside"
    spec = importlib.util.spec_from_file_location(name, {outside.as_posix()!r})
    if spec is None or spec.loader is None:
        return 2
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return 0
''',
        encoding="utf-8",
    )
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "origin"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode != 0
    assert b"project module loader is not zipimport" in completed.stderr


def test_bound_worker_rejects_post_target_external_module_origin(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside_dependency.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(
        f'''\
"""Fake runner that attempts an explicit external dependency import."""
import importlib.util
import sys

def main(argv):
    name = "injected_external_dependency"
    spec = importlib.util.spec_from_file_location(name, {outside.as_posix()!r})
    if spec is None or spec.loader is None:
        return 2
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return 0
''',
        encoding="utf-8",
    )
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    completed = execute_bound_calibration_worker(
        published.directory,
        _fake_worker_arguments(published, "origin"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode != 0
    assert b"loaded module origin is outside bound roots" in completed.stderr


def test_bound_worker_rechecks_runtime_immediately_before_launch(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    stable = _fake_runtime_identity()
    drifted = copy.deepcopy(stable)
    dependencies = drifted["dependencies"]
    assert isinstance(dependencies, dict)
    dependencies["dependency_import_tree_file_inventory_sha256"] = "2" * 64
    call_count = 0

    def drifted_runtime() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return copy.deepcopy(drifted)

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("worker launched after runtime drift")

    monkeypatch.setattr(readiness, "_build_runtime_identity", drifted_runtime)
    monkeypatch.setattr(readiness.subprocess, "run", forbidden_subprocess)
    with pytest.raises(ReadinessError, match="immediately before bound worker launch"):
        execute_bound_calibration_worker(
            published.directory,
            _fake_worker_arguments(published, "runtime-drift"),
            authorize_calibration_execution=True,
            timeout_seconds=30,
        )
    assert call_count == 1


def test_receipt_json_is_canonical_and_digest_excludes_only_self_digest(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    raw = canonical_json_bytes(prepared.payload)
    parsed = json.loads(raw)
    assert raw == canonical_json_bytes(parsed)
    assert parsed["receipt_sha256"] == canonical_sha256(parsed["body"])
    assert "receipt_sha256" not in parsed["body"]
    with pytest.raises(TypeError, match="unsupported JSON type"):
        canonical_json_bytes({"float": 1.0})
