"""Public identity contracts for the continual dyad and actor-owned P route."""

from __future__ import annotations

import pytest

import alberta_framework
import alberta_framework.core as core
from alberta_framework.core import hccl_continual_dyad_transaction as dyad
from alberta_framework.core import hccl_kondo_continual_dyad_route as route
from alberta_framework.core import kondo_protected_td as protected_td

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("implementation", (dyad, route, protected_td))
def test_every_declared_dyad_symbol_is_publicly_reexported(
    implementation: object,
) -> None:
    declared = getattr(implementation, "__all__")
    assert declared
    assert len(declared) == len(set(declared))
    for name in declared:
        expected = getattr(implementation, name)
        assert name in core.__all__
        assert name in alberta_framework.__all__
        assert getattr(core, name) is expected
        assert getattr(alberta_framework, name) is expected


def test_package_export_manifests_remain_duplicate_free() -> None:
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta_framework.__all__) == len(set(alberta_framework.__all__))
