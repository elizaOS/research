"""Process-boundary checks for matched-current qualification replay and I/O caps."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_qualification as qualification

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_bounded_process_enforces_both_stream_limits_without_deadlock() -> None:
    exact = qualification._run_bounded_process(  # noqa: SLF001
        (
            sys.executable,
            "-I",
            "-c",
            "import os; os.write(1, b'o' * 8); os.write(2, b'e' * 8)",
        ),
        timeout=10,
        maximum_stdout_bytes=8,
        maximum_stderr_bytes=8,
    )
    assert exact.returncode == 0
    assert exact.stdout == b"o" * 8
    assert exact.stderr == b"e" * 8

    for descriptor in (1, 2):
        with pytest.raises(
            qualification._BoundedProcessOutputError,  # noqa: SLF001
            match="active byte limit",
        ):
            qualification._run_bounded_process(  # noqa: SLF001
                (
                    sys.executable,
                    "-I",
                    "-c",
                    f"import os; os.write({descriptor}, b'x' * 9)",
                ),
                timeout=10,
                maximum_stdout_bytes=8,
                maximum_stderr_bytes=8,
            )


def test_bounded_process_stream_sink_never_persists_the_overflow_witness() -> None:
    sink = io.BytesIO()
    with pytest.raises(
        qualification._BoundedProcessOutputError,  # noqa: SLF001
        match="active byte limit",
    ):
        qualification._run_bounded_process(  # noqa: SLF001
            (
                sys.executable,
                "-I",
                "-c",
                "import os; os.write(1, b'x' * 9)",
            ),
            timeout=10,
            maximum_stdout_bytes=8,
            maximum_stderr_bytes=8,
            stdout_sink=sink,
        )
    assert sink.getvalue() == b"x" * 8


def test_bounded_process_concurrently_drains_more_than_pipe_capacity() -> None:
    size = 256 * 1024
    script = (
        "import os, threading\n"
        "def emit(descriptor, value):\n"
        f"    remaining = value * {size}\n"
        "    while remaining:\n"
        "        remaining = remaining[os.write(descriptor, remaining):]\n"
        "threads = [\n"
        "    threading.Thread(target=emit, args=(1, b'o')),\n"
        "    threading.Thread(target=emit, args=(2, b'e')),\n"
        "]\n"
        "[thread.start() for thread in threads]\n"
        "[thread.join() for thread in threads]\n"
    )
    completed = qualification._run_bounded_process(  # noqa: SLF001
        (sys.executable, "-I", "-c", script),
        timeout=10,
        maximum_stdout_bytes=size,
        maximum_stderr_bytes=size,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"o" * size
    assert completed.stderr == b"e" * size


def test_bounded_process_timeout_kills_and_reaps_child() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        qualification._run_bounded_process(  # noqa: SLF001
            (
                sys.executable,
                "-I",
                "-c",
                "import time; time.sleep(60)",
            ),
            timeout=0.05,
            maximum_stdout_bytes=8,
            maximum_stderr_bytes=8,
        )


def test_bounded_process_timeout_leaves_no_live_child_pid() -> None:
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        qualification._run_bounded_process(  # noqa: SLF001
            (
                sys.executable,
                "-I",
                "-c",
                "import os, time; os.write(1, f'{os.getpid()}\\n'.encode()); time.sleep(60)",
            ),
            timeout=1,
            maximum_stdout_bytes=128,
            maximum_stderr_bytes=128,
        )
    pid = int(caught.value.output)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_builtin_git_tar_ignores_repository_local_archive_commands(
    tmp_path: Path,
) -> None:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        pytest.skip("system Git is unavailable")
    repository = tmp_path / "repository"
    environment = qualification._git_environment()  # noqa: SLF001
    commands = (
        (git, "init", "--quiet", repository.as_posix()),
        (git, "-C", repository.as_posix(), "add", "payload.txt"),
        (
            git,
            "-C",
            repository.as_posix(),
            "-c",
            "user.name=Qualification Test",
            "-c",
            "user.email=qualification@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ),
        (
            git,
            "-C",
            repository.as_posix(),
            "config",
            "--local",
            "tar.tar.command",
            "false",
        ),
    )
    for index, command in enumerate(commands):
        if index == 1:
            (repository / "payload.txt").write_bytes(b"payload\n")
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            env=environment,
        )
    identity = qualification._bind_git_runtime(Path(git))  # noqa: SLF001
    completed = qualification._run_bounded_process(  # noqa: SLF001
        qualification._git_command(  # noqa: SLF001
            identity,
            repository,
            "archive",
            "--format=tar",
            "HEAD",
        ),
        timeout=10,
        maximum_stdout_bytes=1024 * 1024,
        maximum_stderr_bytes=4096,
        environment=environment,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        assert archive.getnames() == ["payload.txt"]


def test_fresh_replay_script_uses_staged_artifacts_and_transitive_module(
    tmp_path: Path,
) -> None:
    qualification_root = tmp_path / "qualification"
    source_root = qualification_root / "sources" / "alberta" / "source"
    package = source_root / "alberta_framework" / "benchmarks"
    package.mkdir(parents=True)
    (source_root / "alberta_framework" / "__init__.py").write_bytes(b"")
    (package / "__init__.py").write_bytes(b"")
    manifest_sha256 = _sha("staged-manifest")
    protocol_sha256 = _sha("staged-protocol")
    plan_sha256 = _sha("staged-plan")
    fixture = {
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan_sha256,
    }
    (qualification_root / "fixture.json").write_bytes(
        qualification._canonical_json_bytes(fixture)  # noqa: SLF001
    )
    (package / "fresh_replay_support.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def load(root):\n"
        "    return json.loads((Path(root) / 'fixture.json').read_text('utf-8'))\n",
        encoding="utf-8",
    )
    module = package / "forager_matched_qualification.py"
    module.write_text(
        "from .fresh_replay_support import load\n"
        "class Bundle:\n"
        "    def __init__(self, value, root):\n"
        "        self.manifest_sha256 = value['manifest_sha256']\n"
        "        self.root = root\n"
        "class Protocol:\n"
        "    def __init__(self, value): self.protocol_sha256 = value['protocol_sha256']\n"
        "class Plan:\n"
        "    def __init__(self, value):\n"
        "        self.plan_sha256 = value['plan_sha256']\n"
        "        self.qualification_manifest_sha256 = value['manifest_sha256']\n"
        "def load_matched_current_qualification_bundle(root):\n"
        "    return Bundle(load(root), root)\n"
        "def build_open_protocol_and_execution_plan(bundle):\n"
        "    value = load(bundle.root)\n"
        "    return Protocol(value), Plan(value)\n",
        encoding="utf-8",
    )
    qualification._normalize_qualification_tree_permissions(  # noqa: SLF001
        qualification_root
    )
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    runtime = qualification._ProbeRuntimeIdentity(  # noqa: SLF001
        executable=tmp_path / "docker",
        executable_sha256=_sha("runtime"),
        version={},
        image_inspection={},
    )
    observed_commands: list[tuple[str, ...]] = []

    def runner(command: Any) -> qualification.QualificationProcessResult:
        materialized = tuple(command)
        observed_commands.append(materialized)
        python_index = materialized.index(qualification._QUALIFIED_PYTHON)  # noqa: SLF001
        child_arguments = list(materialized[python_index + 1 :])
        replacements = {
            qualification._CONTAINER_REPLAY_SOURCE_ROOT: source_root.as_posix(),  # noqa: SLF001
            qualification._CONTAINER_BUNDLE_ROOT: qualification_root.as_posix(),  # noqa: SLF001
        }
        child_arguments = [replacements.get(item, item) for item in child_arguments]
        return qualification._run_bounded_process(  # noqa: SLF001
            (sys.executable, *child_arguments),
            timeout=30,
            maximum_stdout_bytes=qualification._MAX_PROBE_OUTPUT_BYTES,  # noqa: SLF001
            maximum_stderr_bytes=qualification._MAX_PROBE_OUTPUT_BYTES,  # noqa: SLF001
        )

    qualification._run_fresh_snapshot_replay(  # noqa: SLF001
        qualification_root,
        runtime,
        runner,
        expected_manifest_sha256=manifest_sha256,
        expected_protocol_sha256=protocol_sha256,
        expected_plan_sha256=plan_sha256,
        expected_qualification_module_sha256=module_sha256,
    )

    assert len(observed_commands) == 1
    assert "--network=none" in observed_commands[0]
    assert "--read-only" in observed_commands[0]
    assert f"sha256:{qualification._QUALIFIED_IMAGE_SHA256}" in observed_commands[0]  # noqa: SLF001
