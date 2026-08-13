"""Contracts for the historical behavioral-KL descriptive weight sweep."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import struct
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from alberta_framework.evaluation import (
    historical_behavioral_kl_retention_sweep_development as sweep_module,
)
from alberta_framework.evaluation.historical_behavioral_kl_retention_development import (
    HistoricalBehavioralKLRetentionConfig,
)
from alberta_framework.evaluation.historical_behavioral_kl_retention_sweep_development import (
    ARTIFACT_AUTHORITY,
    BENCHMARK_EXECUTION_AUTHORITY,
    DEVELOPMENT_ONLY,
    EVIDENCE_CLAIMED,
    EXPECTED_CELL_COUNT,
    HIDDEN_RETRIES_USED,
    MAX_SWEEP_REPORT_BYTES,
    OUTPUT_WRITES_ALLOWED,
    RNG_USED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SWEEP_AXES,
    SWEEP_SCHEMA,
    WEIGHT_GRID,
    BoundedFailureReceipt,
    CompletedCellReceipt,
    CompletedLaneHashes,
    HistoricalBehavioralKLRetentionSweepReport,
    InterventionWeights,
    RawTargetArmCoordinates,
    ResourceSummary,
    SweepCell,
    WorkVector,
    run_historical_behavioral_kl_retention_sweep_development,
    validate_historical_behavioral_kl_retention_sweep_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@dataclasses.dataclass(frozen=True, slots=True)
class _ObservedSweep:
    report: HistoricalBehavioralKLRetentionSweepReport
    configs: tuple[HistoricalBehavioralKLRetentionConfig, ...]
    outcomes: tuple[object, ...]
    runner_calls: int


@pytest.fixture(scope="module")
def observed_sweep() -> _ObservedSweep:
    original = cast(Any, sweep_module).run_historical_behavioral_kl_retention_development
    configs: list[HistoricalBehavioralKLRetentionConfig] = []
    outcomes: list[object] = []

    def counted(config: HistoricalBehavioralKLRetentionConfig) -> Any:
        configs.append(config)
        try:
            outcome = original(config)
        except RuntimeError as error:
            outcomes.append(error)
            raise
        outcomes.append(outcome)
        return outcome

    with mock.patch.object(
        sweep_module,
        "run_historical_behavioral_kl_retention_development",
        side_effect=counted,
    ) as runner:
        report = run_historical_behavioral_kl_retention_sweep_development()
    return _ObservedSweep(
        report=report,
        configs=tuple(configs),
        outcomes=tuple(outcomes),
        runner_calls=runner.call_count,
    )


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _assert_float_exact(left: float, right: float) -> None:
    assert _float_bits(left) == _float_bits(right)


def _all_work_values(work: WorkVector) -> tuple[int, ...]:
    return tuple(getattr(work, field.name) for field in dataclasses.fields(WorkVector))


def test_sweep_imports_only_stdlib_and_three_completed_lane_apis() -> None:
    source_path = Path(inspect.getsourcefile(sweep_module) or "")
    parsed = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    lane_imports: list[ast.ImportFrom] = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
            if node.module.endswith("historical_behavioral_kl_retention_development"):
                lane_imports.append(node)
    assert imported_roots <= {
        "__future__",
        "alberta_framework",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "struct",
        "typing",
    }
    assert len(lane_imports) == 1
    assert {alias.name for alias in lane_imports[0].names} == {
        "HistoricalBehavioralKLRetentionConfig",
        "run_historical_behavioral_kl_retention_development",
        "validate_historical_behavioral_kl_retention_report",
    }


def test_grid_is_frozen_ordered_and_one_at_a_time(
    observed_sweep: _ObservedSweep,
) -> None:
    report = observed_sweep.report
    base = HistoricalBehavioralKLRetentionConfig()
    assert WEIGHT_GRID == (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert report.weight_grid == WEIGHT_GRID
    assert report.axes == SWEEP_AXES
    assert len(report.cells) == EXPECTED_CELL_COUNT == 27
    assert len(observed_sweep.configs) == EXPECTED_CELL_COUNT
    assert len(observed_sweep.outcomes) == EXPECTED_CELL_COUNT
    assert observed_sweep.runner_calls == EXPECTED_CELL_COUNT

    for ordinal, (cell, config) in enumerate(
        zip(report.cells, observed_sweep.configs, strict=True)
    ):
        axis = SWEEP_AXES[ordinal // len(WEIGHT_GRID)]
        grid_index = ordinal % len(WEIGHT_GRID)
        weight = WEIGHT_GRID[grid_index]
        assert cell.ordinal == ordinal
        assert cell.axis == axis
        assert cell.grid_index == grid_index
        _assert_float_exact(cell.varied_weight, weight)
        assert cell.target_arm_name == axis.target_arm_name
        assert cell.complete_report_attempts == 1
        assert cell.retry_attempts == 0
        assert cell.config_sha256 == sweep_module._sha256(config)

        expected_weights = {
            "historical_kl_weight": base.historical_kl_weight,
            "current_kl_weight": base.current_kl_weight,
            "movement_l2_weight": base.movement_l2_weight,
        }
        expected_weights[axis.config_field_name] = weight
        for name, expected in expected_weights.items():
            _assert_float_exact(getattr(config, name), expected)
            _assert_float_exact(getattr(cell.weights, name), expected)
        for field in dataclasses.fields(base):
            if field.name not in expected_weights:
                assert sweep_module._exact(getattr(config, field.name), getattr(base, field.name))


def test_each_completed_cell_retains_only_exact_projection_of_target_arm(
    observed_sweep: _ObservedSweep,
) -> None:
    for cell, outcome in zip(
        observed_sweep.report.cells,
        observed_sweep.outcomes,
        strict=True,
    ):
        if cell.completed is None:
            assert isinstance(outcome, RuntimeError)
            continue
        completed = cell.completed
        lane_report = cast(Any, outcome)
        target_arm = next(
            arm for arm in lane_report.arms if arm.name == cell.target_arm_name
        )
        metrics = target_arm.metrics
        assert completed.hashes.complete_report_sha256 == sweep_module._sha256(lane_report)
        assert completed.hashes.target_arm_sha256 == sweep_module._sha256(target_arm)
        assert completed.hashes.target_arm_metrics_sha256 == sweep_module._sha256(metrics)
        assert completed.hashes.source_sha256 == lane_report.source_sha256
        assert (
            completed.hashes.source_generator_contract_sha256
            == lane_report.source.generator_contract_sha256
        )
        assert completed.hashes.source_input_sha256 == lane_report.source.input_sha256
        assert completed.work == sweep_module._copy_work(lane_report.work)
        assert completed.resource == sweep_module._copy_resource(lane_report.resource)
        for name in (
            "a_return_before_b",
            "a_return_after_b",
            "a_return_delta",
            "a_forgetting",
            "b_return_before_b",
            "b_return_after_b",
            "b_return_delta",
            "b_plasticity_gain",
        ):
            _assert_float_exact(
                getattr(completed.raw_coordinates, name),
                getattr(metrics, name),
            )

    retained_types = {
        SweepCell,
        CompletedCellReceipt,
        CompletedLaneHashes,
        RawTargetArmCoordinates,
        WorkVector,
        ResourceSummary,
        InterventionWeights,
    }
    retained_field_names = {
        field.name
        for retained_type in retained_types
        for field in dataclasses.fields(retained_type)
    }
    assert "trace" not in retained_field_names
    assert "source" not in retained_field_names
    assert "arms" not in retained_field_names
    assert "complete_report" not in retained_field_names


def test_status_counts_and_l2_cap_failures_are_raw_single_attempt_outcomes(
    observed_sweep: _ObservedSweep,
) -> None:
    report = observed_sweep.report
    assert tuple((item.status, item.count) for item in report.status_counts) == (
        ("completed_report", 25),
        ("bounded_failure", 2),
    )
    assert tuple((item.kind, item.count) for item in report.failure_kind_counts) == (
        ("candidate_parameter_cap", 2),
    )
    failed = tuple(cell for cell in report.cells if cell.failure is not None)
    assert tuple((cell.axis.name, cell.varied_weight) for cell in failed) == (
        ("parameter_movement_l2_axis", 8.0),
        ("parameter_movement_l2_axis", 16.0),
    )
    for cell in failed:
        assert cell.completed is None
        failure = cell.failure
        assert type(failure) is BoundedFailureReceipt
        assert failure.kind == "candidate_parameter_cap"
        assert failure.exception_module == "builtins"
        assert failure.exception_type == "RuntimeError"
        assert failure.message == "candidate actor parameters exceed the configured cap"
        assert len(failure.message) <= sweep_module.MAX_FAILURE_MESSAGE_CHARS
        assert failure.report_returned is False
        assert failure.validator_called is False
        assert failure.retry_attempts == 0
        assert _all_work_values(failure.minimum_attempted_work) == (0,) * len(
            dataclasses.fields(WorkVector)
        )
        config = observed_sweep.configs[cell.ordinal]
        assert failure.maximum_attempted_work == sweep_module._planned_work(config)


def test_aggregate_work_includes_completed_exact_work_and_failed_bounds(
    observed_sweep: _ObservedSweep,
) -> None:
    report = observed_sweep.report
    aggregate = report.aggregate_attempted_work
    completed_work = tuple(
        cell.completed.work for cell in report.cells if cell.completed is not None
    )
    failure_minimum = tuple(
        cell.failure.minimum_attempted_work
        for cell in report.cells
        if cell.failure is not None
    )
    failure_maximum = tuple(
        cell.failure.maximum_attempted_work
        for cell in report.cells
        if cell.failure is not None
    )
    assert aggregate.cell_attempts == 27
    assert aggregate.complete_report_attempts == 27
    assert aggregate.complete_reports_returned == 25
    assert aggregate.bounded_failure_receipts == 2
    assert aggregate.retry_attempts == 0
    assert aggregate.completed_work == sweep_module._sum_work(completed_work)
    assert aggregate.failure_minimum_work == sweep_module._sum_work(failure_minimum)
    assert aggregate.failure_maximum_work == sweep_module._sum_work(failure_maximum)
    assert aggregate.total_minimum_work == sweep_module._add_work(
        aggregate.completed_work,
        aggregate.failure_minimum_work,
    )
    assert aggregate.total_maximum_work == sweep_module._add_work(
        aggregate.completed_work,
        aggregate.failure_maximum_work,
    )
    assert aggregate.completed_work.total_candidate_objective_evaluations == 19_200
    assert aggregate.total_minimum_work.total_candidate_objective_evaluations == 19_200
    assert aggregate.total_maximum_work.total_candidate_objective_evaluations == 20_736
    for minimum, maximum in zip(
        _all_work_values(aggregate.total_minimum_work),
        _all_work_values(aggregate.total_maximum_work),
        strict=True,
    ):
        assert minimum <= maximum


def test_report_is_nonpromoting_nonwriting_and_has_no_selection_fields(
    observed_sweep: _ObservedSweep,
) -> None:
    report = observed_sweep.report
    assert report.schema == SWEEP_SCHEMA
    assert DEVELOPMENT_ONLY is True
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert BENCHMARK_EXECUTION_AUTHORITY is False
    assert ARTIFACT_AUTHORITY is False
    assert OUTPUT_WRITES_ALLOWED is False
    assert EVIDENCE_CLAIMED is False
    assert RNG_USED is False
    assert HIDDEN_RETRIES_USED is False
    assert report.development_only is True
    assert report.scientific_promotion_allowed is False
    assert report.benchmark_execution_authority is False
    assert report.artifact_authority is False
    assert report.output_writes_allowed is False
    assert report.evidence_claimed is False
    assert report.rng_used is False
    assert report.hidden_retries_used is False

    forbidden = {
        "ranking",
        "rank",
        "frontier",
        "winner",
        "default",
        "threshold",
        "verdict",
        "accepted",
        "dominated",
    }
    for name in sweep_module.__all__:
        value = getattr(sweep_module, name)
        if isinstance(value, type) and dataclasses.is_dataclass(value):
            assert forbidden.isdisjoint(field.name for field in dataclasses.fields(value))
    assert not any(name.startswith("write_") for name in sweep_module.__all__)


def test_hashes_bytes_and_canonical_type_identity_are_exact(
    observed_sweep: _ObservedSweep,
) -> None:
    report = observed_sweep.report
    assert report.base_config_sha256 == sweep_module._sha256(
        HistoricalBehavioralKLRetentionConfig()
    )
    assert report.cells_sha256 == sweep_module._sha256(report.cells)
    assert report.aggregate_attempted_work_sha256 == sweep_module._sha256(
        report.aggregate_attempted_work
    )
    assert report.implementation_source_sha256 == sweep_module._implementation_source_sha256()
    assert report.report_payload_sha256 == sweep_module._sha256(
        sweep_module._report_payload(dataclasses.replace(report, report_payload_sha256=""))
    )
    assert report.canonical_cells_nbytes == len(sweep_module._canonical_bytes(report.cells))
    assert len(sweep_module._canonical_bytes(report)) <= MAX_SWEEP_REPORT_BYTES
    assert report.report_cap_enforced is True
    assert all(
        len(getattr(report, name)) == 64
        for name in (
            "base_config_sha256",
            "sweep_contract_sha256",
            "cells_sha256",
            "aggregate_attempted_work_sha256",
            "implementation_source_sha256",
            "report_payload_sha256",
        )
    )
    assert sweep_module._sha256(0.0) != sweep_module._sha256(-0.0)
    assert sweep_module._sha256(False) != sweep_module._sha256(0)
    assert sweep_module._exact(0.0, -0.0) is False
    assert sweep_module._exact(False, 0) is False
    with pytest.raises(ValueError, match="finite"):
        sweep_module._canonical_bytes(float("nan"))


def test_validator_reconstructs_every_cell_exactly(
    observed_sweep: _ObservedSweep,
) -> None:
    assert validate_historical_behavioral_kl_retention_sweep_report(
        observed_sweep.report
    ) == ()
    assert validate_historical_behavioral_kl_retention_sweep_report(
        cast(HistoricalBehavioralKLRetentionSweepReport, object())
    ) == ("report type differs",)


def test_validator_rejects_bool_alias_and_digest_tamper(
    observed_sweep: _ObservedSweep,
) -> None:
    tampered = dataclasses.replace(
        observed_sweep.report,
        hidden_retries_used=cast(Any, 0),
        cells_sha256="0" * 64,
    )
    errors = validate_historical_behavioral_kl_retention_sweep_report(tampered)
    assert "report does not reconstruct bit-exactly" in errors
    assert "cell digest differs" in errors


def test_unknown_runner_errors_surface_immediately_without_retry() -> None:
    with mock.patch.object(
        sweep_module,
        "run_historical_behavioral_kl_retention_development",
        side_effect=RuntimeError("unexpected runner fault"),
    ) as runner:
        with pytest.raises(RuntimeError, match="unexpected runner fault"):
            run_historical_behavioral_kl_retention_sweep_development()
    assert runner.call_count == 1


def test_public_runner_and_validator_reject_no_configuration_aliases() -> None:
    assert tuple(
        inspect.signature(run_historical_behavioral_kl_retention_sweep_development).parameters
    ) == ()
    assert tuple(
        inspect.signature(
            validate_historical_behavioral_kl_retention_sweep_report
        ).parameters
    ) == ("report",)
    assert all(math.isfinite(weight) and weight > 0.0 for weight in WEIGHT_GRID)
