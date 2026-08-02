"""Descriptor-leak regressions for the executor and qualification tree walkers.

Same before/after ``/proc/self/fd`` counting pattern as
``test_forager_matched_container_hardening`` (see its module docstring),
applied to :func:`forager_matched_executor.source_inventory` and
``forager_matched_qualification._durably_sync_verified_tree``: an injected
``os.fstat`` failure on the freshly opened root must not leak the root
descriptor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_qualification as qualification

pytestmark = pytest.mark.unit


def _open_fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def test_executor_source_root_fstat_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "source.py").write_text("pass\n", encoding="utf-8")
    original = os.fstat
    failed = False

    def fail_first(descriptor: int) -> os.stat_result:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected root fstat failure")
        return original(descriptor)

    before = _open_fd_count()
    monkeypatch.setattr(os, "fstat", fail_first)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="opened source root"):
        executor.source_inventory(root)
    assert _open_fd_count() == before


def test_qualification_root_fstat_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "qualification"
    root.mkdir()
    (root / "artifact.json").write_text("{}", encoding="ascii")
    original = os.fstat
    failed = False

    def fail_first(descriptor: int) -> os.stat_result:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected root fstat failure")
        return original(descriptor)

    before = _open_fd_count()
    monkeypatch.setattr(os, "fstat", fail_first)
    with pytest.raises(
        qualification.ForagerMatchedQualificationError,
        match="opened root",
    ):
        qualification._durably_sync_verified_tree(root)
    assert _open_fd_count() == before
