from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_container as container

pytestmark = pytest.mark.unit


def _output_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir(parents=True)
    (root / "stdout.log").write_bytes(b"")
    (root / "stderr.log").write_bytes(b"")
    monkeypatch.setattr(container, "OUTPUT_ROOT", root)
    return root


def _descriptor_count() -> int:
    descriptor_root = Path("/proc/self/fd")
    if not descriptor_root.is_dir():
        pytest.skip("descriptor-count regression requires procfs")
    return len(os.listdir(descriptor_root))


def test_output_member_walk_closes_root_when_initial_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _output_tree(tmp_path, monkeypatch)
    before = _descriptor_count()

    def fail_root_fstat(_descriptor: int) -> os.stat_result:
        raise OSError("injected root fstat failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fstat", fail_root_fstat)
        with pytest.raises(OSError, match="injected root fstat failure"):
            container._output_members(root / "stdout.log", root / "stderr.log")

    assert _descriptor_count() == before


def test_output_member_walk_closes_member_when_initial_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _output_tree(tmp_path, monkeypatch)
    original_fstat = os.fstat
    calls = 0

    def fail_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected member fstat failure")
        return original_fstat(descriptor)

    before = _descriptor_count()
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fstat", fail_second_fstat)
        with pytest.raises(OSError, match="injected member fstat failure"):
            container._output_members(root / "stdout.log", root / "stderr.log")

    assert calls == 2
    assert _descriptor_count() == before


def test_output_member_walk_enforces_directory_and_depth_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_root = _output_tree(tmp_path / "directories", monkeypatch)
    (directory_root / "first").mkdir()
    (directory_root / "second").mkdir()
    monkeypatch.setattr(container, "MAX_OUTPUT_DIRECTORIES", 1)
    with pytest.raises(container.ContainerError, match="directory-count"):
        container._output_members(
            directory_root / "stdout.log",
            directory_root / "stderr.log",
        )

    depth_root = _output_tree(tmp_path / "depth", monkeypatch)
    (depth_root / "one" / "two").mkdir(parents=True)
    monkeypatch.setattr(container, "MAX_OUTPUT_DIRECTORIES", 20_000)
    monkeypatch.setattr(container, "MAX_OUTPUT_DEPTH", 1)
    with pytest.raises(container.ContainerError, match="directory-depth"):
        container._output_members(
            depth_root / "stdout.log",
            depth_root / "stderr.log",
        )


def test_output_member_walk_bounds_one_bushy_directory_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _output_tree(tmp_path, monkeypatch)
    (root / "extra-a").write_bytes(b"")
    (root / "extra-b").write_bytes(b"")
    monkeypatch.setattr(container, "MAX_MEMBERS", 2)
    monkeypatch.setattr(container, "MAX_OUTPUT_DIRECTORIES", 1)

    with pytest.raises(container.ContainerError, match="entry-count"):
        container._output_members(root / "stdout.log", root / "stderr.log")


def test_workload_command_sets_non_raiseable_live_file_limit_before_runpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    entrypoint = source / "worker.py"
    source.mkdir()
    entrypoint.write_text("pass\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="ascii")
    snapshot = tmp_path / "snapshot"
    copied_config = tmp_path / "copied-config.json"
    output = tmp_path / "output"
    monkeypatch.setattr(container, "EXECUTION_SOURCE_ROOT", snapshot)
    monkeypatch.setattr(container, "EXECUTION_CONFIG", copied_config)
    monkeypatch.setattr(container, "OUTPUT_ROOT", output)

    def fake_snapshot(_root: Path, _digest: str) -> None:
        snapshot.mkdir()
        (snapshot / "worker.py").write_text("pass\n", encoding="utf-8")

    def fake_copy(_source: Path, destination: Path, _digest: str) -> None:
        destination.write_text("{}", encoding="ascii")

    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(container, "_snapshot_source", fake_snapshot)
    monkeypatch.setattr(container, "_copy_stable_file", fake_copy)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(container.ContainerError, match="workload exited"):
        container._run(
            Namespace(
                source_root=source.resolve().as_posix(),
                entrypoint=entrypoint.resolve().as_posix(),
                python_import_root=source.resolve().as_posix(),
                config=config.resolve().as_posix(),
                python=Path("/usr/bin/python3").resolve().as_posix(),
                source_inventory_sha256="0" * 64,
                configuration_sha256="1" * 64,
                invocation_style="alberta_single_seed_v1",
                result_root="results/worker",
                seed=0,
                horizon=1,
            )
        )

    command = observed["command"]
    wrapper = command[command.index("-c") + 1]
    assert "preexec_fn" not in observed
    assert wrapper.index("resource.setrlimit") < wrapper.index("runpy.run_path")
    assert observed["stdout"].name.endswith("stdout.log")
    assert observed["stderr"].name.endswith("stderr.log")


def test_runpy_wrapper_enforces_file_limit_in_real_child_without_fork_hook(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "record_limit.py"
    result = tmp_path / "limit.txt"
    entrypoint.write_text(
        """import resource
import sys
from pathlib import Path

soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
raise_failed = False
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (OSError, ValueError):
    raise_failed = True
Path(sys.argv[1]).write_text(f"{soft},{hard},{raise_failed}", encoding="ascii")
""",
        encoding="utf-8",
    )
    command = container._runpy_command(
        python=Path(sys.executable).resolve(),
        python_import_root=tmp_path,
        entrypoint=entrypoint,
        arguments=[result.as_posix()],
    )

    completed = subprocess.run(command, check=False, capture_output=True)

    assert completed.returncode == 0
    assert completed.stderr == b""
    assert result.read_text(encoding="ascii") == (
        f"{container.MAX_MEMBER_BYTES},{container.MAX_MEMBER_BYTES},True"
    )
