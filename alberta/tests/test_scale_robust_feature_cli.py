"""Actionable refusal messages of the scale-robust pair-feature CLI.

The pinned canonical artifact ``outputs/scale_robust_feature/evidence.v2.json``
is immutable and a bare invocation must be refused, but the refusal has to
tell the caller what to do instead of exiting with an opaque status 2.  The
full generation/verification contract lives in
``test_scale_robust_feature_artifact.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alberta_framework.evaluation import scale_robust_feature_cli
from alberta_framework.evaluation.scale_robust_feature import (
    ScaleRobustFeatureReport,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _forbidden_run() -> ScaleRobustFeatureReport:
    raise AssertionError("the frozen protocol must not run when output is refused")


def test_bare_invocation_requires_explicit_output_before_protocol(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPOSITORY_ROOT)
    pinned = scale_robust_feature_cli.DEFAULT_OUTPUT.resolve()
    assert pinned.is_file()
    original = pinned.read_bytes()
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        _forbidden_run,
    )

    status = scale_robust_feature_cli.main([])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    message = emitted["errors"][0]
    assert "generation requires --output with a new path" in message
    assert "pass --output with a new path" in message
    assert pinned.read_bytes() == original


def test_explicit_pinned_output_is_refused_before_protocol(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPOSITORY_ROOT)
    pinned = scale_robust_feature_cli.DEFAULT_OUTPUT.resolve()
    original = pinned.read_bytes()
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        _forbidden_run,
    )

    status = scale_robust_feature_cli.main(["--output", str(pinned)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert "pinned canonical artifact path" in emitted["errors"][0]
    assert pinned.read_bytes() == original


def test_existing_output_refusal_names_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "existing.json"
    sentinel = b"existing artifact must survive"
    path.write_bytes(sentinel)
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        _forbidden_run,
    )

    status = scale_robust_feature_cli.main(["--output", str(path)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    message = emitted["errors"][0]
    assert "existing output path" in message
    assert "pass --output with a new path" in message
    assert path.read_bytes() == sentinel
