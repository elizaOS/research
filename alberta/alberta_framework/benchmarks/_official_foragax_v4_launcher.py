#!/opt/alberta-runtime/bin/python
"""Immutable container-side launcher for the official Foragax OCI contract.

This file deliberately depends only on the Python standard library.  The OCI
builder copies its exact bytes into the image and records their SHA-256.  On a
successful workload invocation, stdout is *only* an uncompressed USTAR stream;
all workload output is captured as bounded members of that stream.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

CONTRACT = "oci-read-only-stdout-tar-v4"
PATH_MODE = "isolated-runpy-prepend-v1"
SOURCE_ROOT = Path("/opt/continual-foragax-agents")
RUNTIME_ROOT = Path("/opt/alberta-runtime")
OUTPUT_ROOT = Path("/run/alberta/output")
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_MEMBERS = 20_000

_RUNPY_WRAPPER = """\
import runpy
import sys
trusted_path, entrypoint, *arguments = sys.argv[1:]
sys.path.insert(0, trusted_path)
sys.argv = [entrypoint, *arguments]
runpy.run_path(entrypoint, run_name="__main__")
"""


class LauncherError(RuntimeError):
    """Fail-closed launcher validation error."""


def _absolute_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\x00" in value
    ):
        raise LauncherError(f"{label} is not a canonical absolute path")
    return path


def _under(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LauncherError(f"{label} escapes {root}") from exc
    return path


def _regular_immutable(path: Path, *, label: str, executable: bool = False) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != (0o555 if executable else 0o444)
    ):
        raise LauncherError(f"{label} is not a canonical immutable regular file")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-flag", action="append", required=True)
    parser.add_argument("--trusted-python-path", required=True)
    parser.add_argument("--trusted-python-path-mode", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--export-format", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.python_flag != ["-I", "-B"]:
        raise LauncherError("Python flags must be exactly -I then -B")
    if args.trusted_python_path_mode != PATH_MODE:
        raise LauncherError("trusted Python path mode is unsupported")
    if args.export_format != CONTRACT:
        raise LauncherError("export format is unsupported")
    if args.max_steps is not None and args.max_steps < 1:
        raise LauncherError("max steps must be positive")

    python = _absolute_path(args.python, label="Python executable")
    if python != RUNTIME_ROOT / "bin/python":
        raise LauncherError("Python executable is not the sealed virtual environment")
    trusted = _absolute_path(
        args.trusted_python_path,
        label="trusted Python path",
    )
    if trusted != SOURCE_ROOT / "src":
        raise LauncherError("trusted Python path is not the immutable source tree")
    entrypoint = _under(
        _absolute_path(args.entrypoint, label="entrypoint"),
        trusted,
        label="entrypoint",
    )
    if entrypoint not in {
        trusted / "continuing_main.py",
        trusted / "rtu_ppo.py",
    }:
        raise LauncherError("entrypoint is not allowlisted")
    config = _under(
        _absolute_path(args.config, label="configuration"),
        SOURCE_ROOT,
        label="configuration",
    )
    save_path = _absolute_path(args.save_path, label="save path")
    checkpoint_path = _absolute_path(
        args.checkpoint_path,
        label="checkpoint path",
    )
    stdout_log = _absolute_path(args.stdout_log, label="stdout log")
    stderr_log = _absolute_path(args.stderr_log, label="stderr log")
    expected = (
        OUTPUT_ROOT / "official-results",
        OUTPUT_ROOT / "official-checkpoints",
        OUTPUT_ROOT / "stdout.log",
        OUTPUT_ROOT / "stderr.log",
    )
    if (save_path, checkpoint_path, stdout_log, stderr_log) != expected:
        raise LauncherError("output paths differ from the sealed v4 layout")
    single = re.fullmatch(r"(?:0|[1-9][0-9]*)", args.index)
    range_match = re.fullmatch(
        r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)",
        args.index,
    )
    valid_range = (
        range_match is not None
        and 0 <= int(range_match.group(1)) < int(range_match.group(2))
        <= (1 << 32)
    )
    valid_single = single is not None and int(args.index) <= (1 << 32) - 1
    if not valid_single and not valid_range:
        raise LauncherError("index expression is not canonical")

    _regular_immutable(entrypoint, label="entrypoint")
    _regular_immutable(config, label="configuration")
    trusted_metadata = trusted.stat()
    if (
        not trusted.is_dir()
        or stat.S_IMODE(trusted_metadata.st_mode) != 0o555
        or os.access(trusted, os.W_OK)
    ):
        raise LauncherError("trusted source directory is writable or noncanonical")
    return (
        python,
        trusted,
        entrypoint,
        config,
        save_path,
        checkpoint_path,
        stdout_log,
        stderr_log,
    )


def _mkdir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def _workload_command(
    args: argparse.Namespace,
    *,
    python: Path,
    trusted: Path,
    entrypoint: Path,
    config: Path,
    save_path: Path,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    command = [
        str(python),
        "-I",
        "-B",
        "-c",
        _RUNPY_WRAPPER,
        str(trusted),
        str(entrypoint),
        "-e",
        str(config),
        "-i",
        args.index,
        "--save_path",
        str(save_path),
        "--checkpoint_path",
        str(checkpoint_path),
    ]
    if args.gpu:
        command.append("--gpu")
    if args.max_steps is not None:
        command.extend(("--max_steps", str(args.max_steps)))
    return tuple(command)


def _output_members(stdout_log: Path, stderr_log: Path) -> list[tuple[str, Path]]:
    members: list[tuple[str, Path]] = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise LauncherError(
                f"workload output contains a symlink: "
                f"{path.relative_to(OUTPUT_ROOT)}"
            )
        if path in {stdout_log, stderr_log} or stat.S_ISDIR(metadata.st_mode):
            continue
        relative = path.relative_to(OUTPUT_ROOT).as_posix()
        parsed = PurePosixPath(relative)
        if (
            not parsed.parts
            or parsed.is_absolute()
            or ".." in parsed.parts
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_MEMBER_BYTES
        ):
            raise LauncherError(f"unsafe workload output: {relative}")
        members.append((relative, path))
    members.extend((("stdout.log", stdout_log), ("stderr.log", stderr_log)))
    if len(members) > _MAX_MEMBERS:
        raise LauncherError("workload output contains too many members")
    if len({name for name, _path in members}) != len(members):
        raise LauncherError("workload output contains duplicate members")
    total = sum(path.stat().st_size for _name, path in members)
    if total > _MAX_TOTAL_BYTES:
        raise LauncherError("workload output exceeds the launcher size bound")
    return members


def _write_ustar(members: list[tuple[str, Path]]) -> None:
    with tarfile.open(
        fileobj=sys.stdout.buffer,
        mode="w|",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for name, path in members:
            metadata = path.stat()
            info = tarfile.TarInfo(name)
            info.type = tarfile.REGTYPE
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.size = metadata.st_size
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def main() -> int:
    try:
        args = _arguments()
        (
            python,
            trusted,
            entrypoint,
            config,
            save_path,
            checkpoint_path,
            stdout_log,
            stderr_log,
        ) = _validate_arguments(args)
        if OUTPUT_ROOT.exists():
            raise LauncherError("output root unexpectedly exists")
        _mkdir(OUTPUT_ROOT)
        _mkdir(save_path)
        _mkdir(checkpoint_path)
        stdout_log.touch(mode=0o600, exist_ok=False)
        stderr_log.touch(mode=0o600, exist_ok=False)
        command = _workload_command(
            args,
            python=python,
            trusted=trusted,
            entrypoint=entrypoint,
            config=config,
            save_path=save_path,
            checkpoint_path=checkpoint_path,
        )
        with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=SOURCE_ROOT,
                env=dict(os.environ),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            raise LauncherError(
                f"workload exited with status {completed.returncode}"
            )
        _write_ustar(_output_members(stdout_log, stderr_log))
        return 0
    except (LauncherError, OSError, tarfile.TarError) as exc:
        sys.stderr.write(f"official Foragax launcher: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
