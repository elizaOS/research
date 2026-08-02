"""Overwrite-refusal contract of the FTL decision-fidelity evidence CLI.

The pinned canonical artifact ``outputs/ftl_decision/evidence.v1.json`` is
immutable.  Generation must refuse the pinned path and any already-existing
file BEFORE the scientific protocol runs, and the artifact write site itself
must refuse existing paths.  Happy-path generation and verification are
covered in ``test_ftl_decision_artifact.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation import ftl_decision_artifact, ftl_decision_cli
from alberta_framework.evaluation.ftl_decision_artifact import (
    write_ftl_decision_artifact,
)
from alberta_framework.evaluation.ftl_decision_fidelity import (
    DecisionFidelityReport,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _forbidden_run() -> DecisionFidelityReport:
    raise AssertionError("the frozen protocol must not run when output is refused")


def test_bare_invocation_requires_explicit_output_before_protocol(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPOSITORY_ROOT)
    pinned = ftl_decision_cli.DEFAULT_OUTPUT.resolve()
    assert pinned.is_file()
    original = pinned.read_bytes()
    monkeypatch.setattr(
        ftl_decision_cli,
        "run_ftl_decision_fidelity_evaluation",
        _forbidden_run,
    )

    status = ftl_decision_cli.main([])
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
    reserved = tmp_path / "reserved" / "evidence.v1.json"
    assert not reserved.exists()
    monkeypatch.setattr(ftl_decision_cli, "DEFAULT_OUTPUT", reserved)
    monkeypatch.setattr(
        ftl_decision_cli,
        "run_ftl_decision_fidelity_evaluation",
        _forbidden_run,
    )

    status = ftl_decision_cli.main(["--output", str(reserved)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "pinned canonical artifact path" in emitted["errors"][0]
    assert not reserved.exists()


def test_existing_output_path_is_refused_before_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "existing.json"
    sentinel = b"existing artifact must survive"
    path.write_bytes(sentinel)
    monkeypatch.setattr(
        ftl_decision_cli,
        "run_ftl_decision_fidelity_evaluation",
        _forbidden_run,
    )

    status = ftl_decision_cli.main(["--output", str(path)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "existing output path" in emitted["errors"][0]
    assert path.read_bytes() == sentinel


def test_write_site_refuses_existing_artifact_path(tmp_path: Path) -> None:
    """The guard fires before the report is even inspected or built."""

    path = tmp_path / "already-there.json"
    sentinel = b"immutable bytes"
    path.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="existing evidence artifact path"):
        write_ftl_decision_artifact(
            path,
            cast(DecisionFidelityReport, object()),
            evaluation_wall_seconds=0.0,
        )
    assert path.read_bytes() == sentinel


def test_write_site_never_overwrites_a_concurrently_created_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "raced.json"
    sentinel = b"concurrent writer owns this path"

    def build(*_args: object, **_kwargs: object) -> dict[str, object]:
        path.write_bytes(sentinel)
        return {}

    monkeypatch.setattr(ftl_decision_artifact, "build_ftl_decision_artifact", build)
    monkeypatch.setattr(
        ftl_decision_artifact,
        "validate_ftl_decision_artifact",
        lambda _artifact: ftl_decision_artifact.ArtifactValidation(
            valid=True,
            accepted=False,
            errors=(),
        ),
    )
    monkeypatch.setattr(ftl_decision_artifact, "artifact_json", lambda _artifact: "new\n")

    with pytest.raises(FileExistsError, match="existing evidence artifact path"):
        write_ftl_decision_artifact(
            path,
            cast(DecisionFidelityReport, object()),
            evaluation_wall_seconds=0.0,
        )
    assert path.read_bytes() == sentinel
