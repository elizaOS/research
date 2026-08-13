"""Public import parity for the L0 embodied safety envelope."""

from __future__ import annotations

import pytest

import alberta_framework
import alberta_framework.core as core
from alberta_framework.core import embodied_safety_envelope

pytestmark = pytest.mark.unit


def test_every_declared_embodied_safety_symbol_is_publicly_reexported() -> None:
    for name in embodied_safety_envelope.__all__:
        expected = getattr(embodied_safety_envelope, name)
        assert name in core.__all__
        assert name in alberta_framework.__all__
        assert getattr(core, name) is expected
        assert getattr(alberta_framework, name) is expected
