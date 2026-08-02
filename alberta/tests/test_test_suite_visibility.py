"""Contracts for standalone-checkout test visibility."""

from pathlib import Path

import conftest as pytest_config
import pytest


def _configure_optional_root(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> Path:
    optional_root = project_root / "examples"
    monkeypatch.setattr(pytest_config, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pytest_config, "_OPTIONAL_SCRIPT_ROOTS", (optional_root,))
    return optional_root


def test_load_script_reports_absent_optional_root_as_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_root = _configure_optional_root(monkeypatch, tmp_path)

    with pytest.raises(pytest.skip.Exception, match="standalone checkout omits examples"):
        pytest_config.load_script(optional_root / "missing.py", "absent_optional_script")


def test_load_script_fails_for_missing_file_inside_present_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_root = _configure_optional_root(monkeypatch, tmp_path)
    optional_root.mkdir()

    with pytest.raises(FileNotFoundError):
        pytest_config.load_script(optional_root / "missing.py", "missing_optional_script")


def test_load_script_imports_file_inside_present_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_root = _configure_optional_root(monkeypatch, tmp_path)
    optional_root.mkdir()
    script_path = optional_root / "available.py"
    script_path.write_text("VALUE = 42\n", encoding="utf-8")

    module = pytest_config.load_script(script_path, "available_optional_script")

    assert module.VALUE == 42
