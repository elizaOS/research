"""Public package exports for opt-in cumulant option installation."""

import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core import cumulant_option_installation as implementation

pytestmark = pytest.mark.unit


def test_root_and_core_export_the_exact_installation_surface() -> None:
    names = (
        "CUMULANT_OPTION_INSTALLATION_CONTROL_HOST_ONLY",
        "CumulantOptionInstallation",
        "CumulantOptionInstallationConfig",
        "CumulantOptionInstallationResourceBudget",
        "CumulantOptionInstallationResult",
        "CumulantOptionInstallationState",
        "CumulantOptionLiveInputs",
        "CumulantOptionMaterialization",
        "CumulantOptionMaterializationResult",
        "CumulantOptionStartResult",
        "CumulantOptionUpdateResult",
    )
    for name in names:
        expected = getattr(implementation, name)
        assert getattr(core, name) is expected
        assert getattr(alberta, name) is expected
        assert name in core.__all__
        assert name in alberta.__all__
