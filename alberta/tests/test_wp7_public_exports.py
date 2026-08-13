"""Public-package contracts for connected continual-agent mechanism surfaces."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

import alberta_framework as alberta
import alberta_framework.core as core

pytestmark = pytest.mark.unit

MODULE_NAMES = (
    "alberta_framework.core.authorized_option_retirement",
    "alberta_framework.core.balanced_state_objectives",
    "alberta_framework.core.calibrated_extended_search_control",
    "alberta_framework.core.causal_exploration_estimator",
    "alberta_framework.core.comprehensive_state_objectives",
    "alberta_framework.core.consolidated_memory_policy",
    "alberta_framework.core.cumulant_option_scheduler",
    "alberta_framework.core.cumulant_subtask_discovery",
    "alberta_framework.core.prospective_exploration",
    "alberta_framework.core.ensemble_short_rollouts",
    "alberta_framework.core.nonlinear_average_reward_actor_critic",
    "alberta_framework.core.nonlinear_off_policy_actor_critic",
    "alberta_framework.core.one_step_dyna",
    "alberta_framework.core.option_lifecycle_audit",
    "alberta_framework.core.prototype_balanced_state_objectives",
    "alberta_framework.core.prototype_comprehensive_state_objectives",
    "alberta_framework.core.prototype_feature_memory",
    "alberta_framework.core.prototype_routed_linear_world_model",
    "alberta_framework.core.self_normalized_resets",
    "alberta_framework.core.stomp_option_lifecycle",
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_declared_connected_symbol_is_exported_from_both_package_roots(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    assert isinstance(module, ModuleType)
    declared = getattr(module, "__all__")
    assert isinstance(declared, list)
    assert declared
    assert len(declared) == len(set(declared))
    for name in declared:
        value = getattr(module, name)
        assert getattr(core, name) is value
        assert getattr(alberta, name) is value
        assert name in core.__all__
        assert name in alberta.__all__


def test_package_export_manifests_remain_duplicate_free() -> None:
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta.__all__) == len(set(alberta.__all__))
