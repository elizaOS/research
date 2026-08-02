"""Overwrite-refusal contract of the recurring multi-agent evidence CLI.

The pinned canonical artifact ``outputs/continual_multiagent/evidence.json``
is immutable.  Generation must refuse the pinned path and any
already-existing file BEFORE the benchmark runs.  Happy-path generation and
verification are covered in ``test_continual_multiagent_benchmark.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alberta_framework.evaluation import continual_multiagent_cli
from alberta_framework.evaluation.continual_multiagent import (
    ContinualMultiAgentReport,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _forbidden_run(**_: object) -> ContinualMultiAgentReport:
    raise AssertionError("the benchmark must not run when output is refused")


def test_bare_invocation_requires_explicit_output_before_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPOSITORY_ROOT)
    pinned = continual_multiagent_cli.DEFAULT_OUTPUT.resolve()
    assert pinned.is_file()
    original = pinned.read_bytes()
    monkeypatch.setattr(
        continual_multiagent_cli,
        "run_continual_multiagent_benchmark",
        _forbidden_run,
    )

    status = continual_multiagent_cli.main([])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert emitted["accepted"] is False
    assert "generation requires --output with a new path" in emitted["errors"][0]
    assert "pass --output with a new path" in emitted["errors"][0]
    assert pinned.read_bytes() == original


def test_reserved_canonical_path_is_refused_even_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reserved = tmp_path / "reserved" / "evidence.json"
    assert not reserved.exists()
    monkeypatch.setattr(continual_multiagent_cli, "DEFAULT_OUTPUT", reserved)
    monkeypatch.setattr(
        continual_multiagent_cli,
        "run_continual_multiagent_benchmark",
        _forbidden_run,
    )

    status = continual_multiagent_cli.main(["--output", str(reserved)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "pinned canonical artifact path" in emitted["errors"][0]
    assert not reserved.exists()


def test_existing_output_path_is_refused_before_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "existing.json"
    sentinel = b"existing artifact must survive"
    path.write_bytes(sentinel)
    monkeypatch.setattr(
        continual_multiagent_cli,
        "run_continual_multiagent_benchmark",
        _forbidden_run,
    )

    status = continual_multiagent_cli.main(["--output", str(path)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "existing output path" in emitted["errors"][0]
    assert path.read_bytes() == sentinel


def test_refusal_happens_before_threshold_and_seed_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A refused output wins even when other flags are themselves invalid."""

    path = tmp_path / "existing.json"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        continual_multiagent_cli,
        "run_continual_multiagent_benchmark",
        _forbidden_run,
    )

    status = continual_multiagent_cli.main(
        ["--output", str(path), "--seed-count", "0"]
    )
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert "existing output path" in emitted["errors"][0]
