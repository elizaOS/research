"""Canonical terminology for the non-gradient prospective exploration score."""

from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.unit


def test_prospective_exploration_api_never_names_its_proxy_delight() -> None:
    """Expected-improvement scoring is not evidence of an actor backward."""

    from alberta_framework.core import prospective_exploration as exploration

    result_fields = {
        field.name for field in dataclasses.fields(exploration.ProspectiveExplorationResult)
    }
    scan_fields = {
        field.name
        for field in dataclasses.fields(exploration.ProspectiveExplorationScanResult)
    }

    assert exploration.PROSPECTIVE_EXPLORATION_MODES[0] == (
        "expected_improvement_surprisal"
    )
    assert exploration.ProspectiveExploration.__name__ == "ProspectiveExploration"
    assert exploration.ProspectiveExplorationConfig.__name__ == (
        "ProspectiveExplorationConfig"
    )
    assert "prospective_delight" not in exploration.PROSPECTIVE_EXPLORATION_MODES
    assert "expected_improvement_surprisal_score" in result_fields
    assert "selected_expected_improvement_surprisal_score" in result_fields
    assert "prospective_delight" not in result_fields
    assert "selected_prospective_delight" not in result_fields
    assert "selected_expected_improvement_surprisal_score" in scan_fields


def test_prospective_exploration_contract_disclaims_gradient_delight() -> None:
    from alberta_framework.core import prospective_exploration as exploration

    assert exploration.PROSPECTIVE_EXPLORATION_GRADIENT_DELIGHT_SEMANTICS is False
    assert exploration.PROSPECTIVE_EXPLORATION_EXECUTES_ACTOR_BACKWARD is False
    assert exploration.PROSPECTIVE_EXPLORATION_SCORE_SEMANTICS == (
        "expected-improvement-times-capped-host-relative-surprisal"
    )


def test_prospective_exploration_current_errors_use_canonical_names() -> None:
    from alberta_framework.core import prospective_exploration as exploration

    with pytest.raises(TypeError, match="ProspectiveExplorationConfig") as config_error:
        exploration.ProspectiveExploration(object())  # type: ignore[arg-type]
    assert "DelightfulExploration" not in str(config_error.value)

    with pytest.raises(TypeError, match="ProspectiveExploration") as controller_error:
        exploration.run_prospective_exploration_from_batches(  # type: ignore[arg-type]
            object(),
            object(),
            object(),
        )
    assert "DelightfulExploration" not in str(controller_error.value)
