"""Public identity contracts for the versioned repeated Prototype bridge."""

from __future__ import annotations

import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core import prototype_option_authority_bridge as v1
from alberta_framework.core import prototype_repeated_option_authority_bridge as v2

pytestmark = pytest.mark.unit


def test_every_declared_v2_symbol_has_one_core_and_top_level_identity() -> None:
    for name in v2.__all__:
        direct = getattr(v2, name)
        assert getattr(core, name) is direct
        assert getattr(alberta, name) is direct
        assert name in core.__all__
        assert name in alberta.__all__


def test_v2_exports_do_not_relabel_or_replace_the_v1_bridge() -> None:
    assert v1.PROTOTYPE_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA.endswith("state.v1")
    assert v2.PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA.endswith("state.v2")
    assert v1.PrototypeOptionAuthorityBridge is core.PrototypeOptionAuthorityBridge
    assert v2.PrototypeRepeatedOptionAuthorityBridge is core.PrototypeRepeatedOptionAuthorityBridge
    legacy: object = v1.PrototypeOptionAuthorityBridge
    versioned: object = v2.PrototypeRepeatedOptionAuthorityBridge
    assert legacy is not versioned
