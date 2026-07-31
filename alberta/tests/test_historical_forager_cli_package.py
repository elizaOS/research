"""Public API, read-only CLI, and distribution checks for the historical lane."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import alberta_framework.benchmarks as benchmark_api
import alberta_framework.benchmarks.historical_forager as historical_module
from alberta_framework import forager_cli
from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerRunConfig,
    HistoricalUpdateKernel,
    development_historical_environment_adapter,
    run_historical_forager,
)
from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[1]
provenance_module = importlib.import_module(
    "alberta_framework.benchmarks.historical_forager_provenance"
)
_DISTRIBUTION_FILES = (
    "alberta_framework/benchmarks/__init__.py",
    "alberta_framework/benchmarks/historical_forager.py",
    "alberta_framework/benchmarks/historical_forager_provenance.py",
    "alberta_framework/forager_cli.py",
)


class _TinyHistoricalEnvironment:
    def __init__(self) -> None:
        self._offset = 0

    def start(self) -> int:
        return 0

    def step(self, action: int) -> tuple[float, int, bool, Mapping[str, Any]]:
        reward = (1.0, -0.5, 2.0, 0.25)[self._offset]
        self._offset += 1
        return reward, self._offset, False, {}


def _write_tiny_artifact(output_directory: Path, *, seed: int) -> None:
    def factory(_seed: int, _aperture_size: int) -> _TinyHistoricalEnvironment:
        return _TinyHistoricalEnvironment()

    kernel = HistoricalUpdateKernel[int](
        name="historical_cli_test_kernel",
        start_kernel=lambda _observation: (0, 0),
        update_kernel=lambda state, _reward, _observation: (state + 1, (state + 1) % 4),
        metadata={"purpose": "cli_contract_test"},
    )
    run_historical_forager(
        development_historical_environment_adapter(factory),
        kernel,
        HistoricalForagerRunConfig(
            seed=seed,
            steps=4,
            aperture_size=9,
            output_directory=output_directory,
            allow_unverified_development_adapter=True,
        ),
    )


def _run_historical_cli(
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    original_handlers = list(forager_cli.LOGGER.handlers)
    original_level = forager_cli.LOGGER.level
    original_propagate = forager_cli.LOGGER.propagate
    forager_cli.LOGGER.handlers.clear()
    try:
        try:
            returncode = forager_cli.main(("historical", *arguments))
        except SystemExit as exc:
            returncode = exc.code if isinstance(exc.code, int) else 1
        captured = capsys.readouterr()
    finally:
        forager_cli.LOGGER.handlers.clear()
        forager_cli.LOGGER.handlers.extend(original_handlers)
        forager_cli.LOGGER.setLevel(original_level)
        forager_cli.LOGGER.propagate = original_propagate
    return subprocess.CompletedProcess(arguments, returncode, captured.out, captured.err)


def test_benchmark_package_exports_complete_historical_public_api() -> None:
    expected = set(historical_module.__all__) | set(provenance_module.__all__)

    assert expected <= set(benchmark_api.__all__)
    for name in expected:
        source_module = (
            historical_module if name in historical_module.__all__ else provenance_module
        )
        assert getattr(benchmark_api, name) is getattr(source_module, name)


def test_historical_subcommand_help_is_read_only_and_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_help = forager_cli._parser().format_help()
    historical_parser = forager_cli._historical_parser(prog="alberta-forager-benchmark historical")
    historical_help = historical_parser.format_help()
    command_action = next(
        action
        for action in historical_parser._actions
        if isinstance(action, forager_cli.argparse._SubParsersAction)
    )

    assert "historical --help" in root_help
    assert set(command_action.choices) == {"provenance", "validate", "pair"}
    assert HISTORICAL_FORAGER_FAMILY_ID in historical_help
    assert "explicitly unattested" in historical_help
    assert "never launches a benchmark run" in historical_help

    completed = _run_historical_cli(capsys, "--help")
    assert completed.returncode == 0
    assert "{provenance,validate,pair}" in completed.stdout
    assert completed.stderr == ""


def test_historical_cli_reports_provenance_and_validates_strict_pairing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    wrong_seed = tmp_path / "wrong-seed"
    _write_tiny_artifact(left, seed=17)
    _write_tiny_artifact(right, seed=17)
    _write_tiny_artifact(wrong_seed, seed=18)

    provenance = _run_historical_cli(capsys, "provenance")
    assert provenance.returncode == 0
    provenance_payload = json.loads(provenance.stdout)
    assert provenance_payload["family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert provenance_payload["environment_resolution_attested"] is False
    assert provenance_payload["pairable_with_current_foragax"] is False
    assert provenance_payload["provenance_sha256"] == HISTORICAL_FORAGER_PROVENANCE_SHA256
    assert provenance_payload["provenance"]["environment_resolution_attested"] is False

    validated = _run_historical_cli(capsys, "validate", str(left))
    assert validated.returncode == 0
    validated_payload = json.loads(validated.stdout)
    assert validated_payload["artifact"]["status"] == "complete"
    assert validated_payload["artifact"]["family_id"] == HISTORICAL_FORAGER_FAMILY_ID

    paired = _run_historical_cli(capsys, "pair", str(left), str(right))
    assert paired.returncode == 0
    paired_payload = json.loads(paired.stdout)
    assert paired_payload["pairable"] is True
    assert paired_payload["pairing_identity"]["seed"] == 17
    assert paired_payload["pairing_identity"]["family_id"] == HISTORICAL_FORAGER_FAMILY_ID

    rejected = _run_historical_cli(capsys, "pair", str(left), str(wrong_seed))
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert "identical provenance, seed, aperture" in rejected.stderr

    missing = _run_historical_cli(capsys, "validate", str(tmp_path / "missing"))
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "must be a real directory" in missing.stderr


def test_wheel_and_sdist_ship_historical_surfaces_and_sdist_document(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the distribution integration check")
    output_directory = tmp_path / "dist"
    completed = subprocess.run(
        (
            uv,
            "build",
            "--offline",
            "--no-build-logs",
            "--no-create-gitignore",
            "--no-python-downloads",
            "--out-dir",
            str(output_directory),
            str(_REPO_ROOT),
        ),
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    wheel_path = output_directory / f"alberta_framework-{version}-py3-none-any.whl"
    sdist_path = output_directory / f"alberta_framework-{version}.tar.gz"
    assert wheel_path.is_file()
    assert sdist_path.is_file()

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = set(wheel.namelist())
        for relative_path in _DISTRIBUTION_FILES:
            assert relative_path in wheel_names
            assert wheel.read(relative_path) == (_REPO_ROOT / relative_path).read_bytes()
        assert "HISTORICAL_FORAGER_RECONSTRUCTED.md" not in wheel_names
        entry_points_name = next(
            name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = wheel.read(entry_points_name).decode("utf-8")
        assert (
            "alberta-historical-forager = alberta_framework.forager_cli:historical_main"
            in entry_points
        )
        assert "alberta-foragax-oci" in entry_points

    prefix = f"alberta_framework-{version}"
    with tarfile.open(sdist_path, mode="r:gz") as source_distribution:
        sdist_names = set(source_distribution.getnames())
        for relative_path in _DISTRIBUTION_FILES:
            archived_path = f"{prefix}/{relative_path}"
            assert archived_path in sdist_names
            extracted = source_distribution.extractfile(archived_path)
            assert extracted is not None
            assert extracted.read() == (_REPO_ROOT / relative_path).read_bytes()
        document_path = f"{prefix}/HISTORICAL_FORAGER_RECONSTRUCTED.md"
        assert document_path in sdist_names
        document = source_distribution.extractfile(document_path)
        assert document is not None
        assert document.read() == (_REPO_ROOT / "HISTORICAL_FORAGER_RECONSTRUCTED.md").read_bytes()
