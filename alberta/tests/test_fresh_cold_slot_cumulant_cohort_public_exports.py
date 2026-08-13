"""Public-package identity contracts for the opt-in v2 cold-slot filter."""

import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core import fresh_cold_slot_cumulant_cohort as implementation

pytestmark = pytest.mark.unit


def test_root_and_core_export_exact_declared_cold_slot_filter_surface() -> None:
    assert implementation.__all__
    assert len(implementation.__all__) == len(set(implementation.__all__))
    for name in implementation.__all__:
        expected = getattr(implementation, name)
        assert getattr(core, name) is expected
        assert getattr(alberta, name) is expected
        assert name in core.__all__
        assert name in alberta.__all__


def test_package_export_manifests_remain_duplicate_free() -> None:
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta.__all__) == len(set(alberta.__all__))
