"""The packaged ``alberta_framework.benchmarks`` subpackage must always win.

A legacy compatibility shim in ``alberta_framework/__init__.py`` used to alias
a repository-root ``benchmarks`` package to the ``alberta_framework.benchmarks``
name.  Now that the real subpackage ships inside the wheel (forager family,
official Foragax bindings), an unrelated top-level ``benchmarks`` directory on
``sys.path`` — for example the upstream repository's root benchmark tree —
must never shadow it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROBE = """
import pathlib

import alberta_framework
import alberta_framework.benchmarks as bench

path = pathlib.Path(bench.__file__).resolve()
assert not getattr(bench, "IS_DUMMY_ROOT_BENCHMARKS", False), (
    "root-level benchmarks package shadowed the real subpackage: " + str(path)
)
assert path.parent.name == "benchmarks", path
assert path.parent.parent.name == "alberta_framework", path

from alberta_framework.benchmarks import official_foragax  # noqa: F401

print("REAL_SUBPACKAGE_OK")
"""


def _run_probe(extra_sys_path: Path | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if extra_sys_path is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(extra_sys_path) if not existing else f"{extra_sys_path}{os.pathsep}{existing}"
        )
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )


def test_real_benchmarks_subpackage_importable() -> None:
    result = _run_probe(extra_sys_path=None)
    assert result.returncode == 0, result.stderr
    assert "REAL_SUBPACKAGE_OK" in result.stdout


def test_root_benchmarks_directory_does_not_shadow_subpackage(tmp_path: Path) -> None:
    dummy = tmp_path / "benchmarks"
    dummy.mkdir()
    (dummy / "__init__.py").write_text("IS_DUMMY_ROOT_BENCHMARKS = True\n")
    result = _run_probe(extra_sys_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "REAL_SUBPACKAGE_OK" in result.stdout
