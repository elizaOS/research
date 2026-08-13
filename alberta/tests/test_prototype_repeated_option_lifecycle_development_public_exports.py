"""Public identities for the nonwriting repeated-option development harness."""

from __future__ import annotations

import pytest

import alberta_framework as alberta
import alberta_framework.evaluation as evaluation
from alberta_framework.evaluation import (
    prototype_repeated_option_lifecycle_development as development,
)

pytestmark = pytest.mark.unit


def test_every_declared_development_symbol_has_one_public_identity() -> None:
    for name in development.__all__:
        direct = getattr(development, name)
        assert getattr(evaluation, name) is direct
        assert getattr(alberta, name) is direct
        assert name in evaluation.__all__
        assert name in alberta.__all__
