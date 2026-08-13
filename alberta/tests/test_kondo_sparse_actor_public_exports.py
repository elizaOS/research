"""Public import parity for the sparse Kondo actor consumer."""

from __future__ import annotations

import pytest

import alberta_framework
import alberta_framework.core as core
from alberta_framework.core import kondo_sparse_actor

pytestmark = pytest.mark.unit


def test_every_declared_sparse_actor_symbol_is_publicly_reexported() -> None:
    for name in kondo_sparse_actor.__all__:
        expected = getattr(kondo_sparse_actor, name)
        assert name in core.__all__
        assert name in alberta_framework.__all__
        assert getattr(core, name) is expected
        assert getattr(alberta_framework, name) is expected

