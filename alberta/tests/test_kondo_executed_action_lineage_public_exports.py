"""Public identity contracts for executed-action Kondo lineage."""

from __future__ import annotations

import pytest

import alberta_framework
import alberta_framework.core as core
from alberta_framework.core import kondo_executed_action_lineage_bridge as implementation

pytestmark = pytest.mark.unit


def test_every_declared_lineage_symbol_is_publicly_reexported() -> None:
    assert implementation.__all__
    assert len(implementation.__all__) == len(set(implementation.__all__))
    for name in implementation.__all__:
        expected = getattr(implementation, name)
        assert name in core.__all__
        assert name in alberta_framework.__all__
        assert getattr(core, name) is expected
        assert getattr(alberta_framework, name) is expected


def test_package_export_manifests_remain_duplicate_free() -> None:
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta_framework.__all__) == len(set(alberta_framework.__all__))
