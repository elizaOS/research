"""Release metadata must move as one versioned transaction."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import alberta_framework

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


def _citation_version() -> str:
    matches: list[str] = re.findall(
        r"(?m)^version:\s*[\"']?([^\"'#\s]+)[\"']?\s*$",
        (_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise AssertionError("CITATION.cff must contain exactly one scalar version")
    return matches[0]


def test_release_version_carriers_and_lockfile_are_synchronized() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["version"]
    assert isinstance(expected, str)
    assert _SEMVER.fullmatch(expected)

    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_versions = [
        package["version"]
        for package in lock["package"]
        if package.get("name") == "alberta-framework"
    ]

    assert alberta_framework.__version__ == expected
    assert _citation_version() == expected
    assert root_versions == [expected]
    assert f"## [{expected}] - " in (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
