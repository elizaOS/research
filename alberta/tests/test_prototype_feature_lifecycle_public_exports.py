"""Public identity checks for the bounded Prototype pair-feature lifecycle."""

from __future__ import annotations

import pytest

import alberta_framework as alberta
import alberta_framework.core as core
import alberta_framework.core.prototype_agent as prototype_agent_module
import alberta_framework.core.prototype_feature_lifecycle as lifecycle_module
import alberta_framework.core.prototype_feature_utility as utility_module

pytestmark = pytest.mark.unit


def test_prototype_feature_lifecycle_public_exports_are_identity_preserving() -> None:
    for name in lifecycle_module.__all__:
        implementation = getattr(lifecycle_module, name)
        assert core.__all__.count(name) == 1
        assert alberta.__all__.count(name) == 1
        assert getattr(core, name) is implementation
        assert getattr(alberta, name) is implementation


def test_prototype_feature_utility_public_exports_are_identity_preserving() -> None:
    for name in utility_module.__all__:
        implementation = getattr(utility_module, name)
        assert core.__all__.count(name) == 1
        assert alberta.__all__.count(name) == 1
        assert getattr(core, name) is implementation
        assert getattr(alberta, name) is implementation


@pytest.mark.parametrize(
    "name",
    (
        "PrototypeFeatureOaKHordeState",
        "PrototypeFeatureOaKHordeUtilityState",
        "PrototypeFeatureOaKState",
        "PrototypeFeatureRepresentationState",
        "PrototypeFeatureLifecycleIntegrationDiagnostics",
        "PrototypeFeatureUtilityIntegrationDiagnostics",
        "PrototypeRTUTransitionPreparation",
        "PrototypeRTUFinalizationReceipt",
        "PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA",
    ),
)
def test_prototype_integration_types_are_public_identity_exports(name: str) -> None:
    implementation = getattr(prototype_agent_module, name)
    assert prototype_agent_module.__all__.count(name) == 1
    assert core.__all__.count(name) == 1
    assert alberta.__all__.count(name) == 1
    assert getattr(core, name) is implementation
    assert getattr(alberta, name) is implementation
