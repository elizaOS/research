"""Integration checks for WP1 Prototype continual-control execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation.continual_control_evaluator import (
    validate_continual_control_report,
)
from alberta_framework.evaluation.prototype_continual_control_development import (
    DEVELOPMENT_SEEDS,
    build_prototype_continual_control_development_report,
    prototype_continual_control_evaluator_for_seed,
    validate_prototype_continual_control_development_report,
)

pytestmark = pytest.mark.slow


@pytest.mark.integration
def test_end_to_end_raw_runs_validate_and_report_exact_logical_resources() -> None:
    report = build_prototype_continual_control_development_report()
    assert validate_prototype_continual_control_development_report(report).valid
    for run in cast(list[Mapping[str, object]], report["runs"]):
        control = cast(Mapping[str, object], run["control_report"])
        assert validate_continual_control_report(control).valid
        for condition in cast(list[Mapping[str, object]], control["conditions"]):
            resources = cast(Mapping[str, object], condition["resources"])
            assert cast(int, resources["persistent_state_bytes_high_water"]) > 0
            counts = cast(
                Mapping[str, object],
                cast(Mapping[str, object], condition["operations"])["counts"],
            )
            assert counts["processed_transitions"] == 6
            assert counts["dropped_transitions"] == 0
            latency = cast(
                Mapping[str, object],
                cast(Mapping[str, object], condition["operations"])["latency_ms"],
            )
            assert latency["measurement_method"] == (
                "deterministic logical test clock: exactly 1000 ns per measured call; "
                "not wall-clock latency"
            )


@pytest.mark.integration
def test_exact_seed_factory_preserves_core_checkpoint_resume(tmp_path: Path) -> None:
    seed = DEVELOPMENT_SEEDS[0]
    uninterrupted = prototype_continual_control_evaluator_for_seed(seed)
    full_state = uninterrupted.advance(uninterrupted.init(), steps=6)
    full_report = uninterrupted.build_report(full_state)

    first = prototype_continual_control_evaluator_for_seed(seed)
    partial = first.advance(first.init(), steps=3)
    checkpoint = tmp_path / "wp1-control-checkpoint.json"
    first.save_checkpoint(partial, checkpoint)

    resumed = prototype_continual_control_evaluator_for_seed(seed)
    restored = resumed.load_checkpoint(checkpoint)
    resumed_state = resumed.advance(restored, steps=3)
    assert resumed.build_report(resumed_state) == full_report

    foreign = prototype_continual_control_evaluator_for_seed(DEVELOPMENT_SEEDS[1])
    with pytest.raises(ValueError, match="config does not match"):
        foreign.load_checkpoint(checkpoint)
