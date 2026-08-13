"""Pure-stdlib one-at-a-time sweep for historical behavioral-KL retention.

This development-only companion runs the complete four-arm historical
behavioral-KL retention probe once for every cell in a frozen three-axis
weight grid.  Only one intervention weight changes in a cell; the other two
retain the base configuration's values.  Every cell invokes the complete
probe even though most of that work is deliberately redundant.

Completed cells retain digests, the addressed arm's raw A/B coordinates, and
exact work/resource summaries.  Known bounded runner failures retain a typed
receipt and conservative attempted-work bounds.  A failed cell is never
retried.  In particular, parameter-cap failures are descriptive outcomes;
the sweep does not change the cap in response.

The sweep is an in-memory L0 diagnostic.  It writes no output, creates no
artifact, promotes no evidence, and makes no comparative selection claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Final, cast

from alberta_framework.evaluation.historical_behavioral_kl_retention_development import (
    HistoricalBehavioralKLRetentionConfig,
    run_historical_behavioral_kl_retention_development,
    validate_historical_behavioral_kl_retention_report,
)

SWEEP_SCHEMA: Final = "alberta.historical-behavioral-kl-retention-sweep.development.v1"
CELL_SCHEMA: Final = "alberta.historical-behavioral-kl-retention-sweep.cell.v1"
SUCCESS_SCHEMA: Final = "alberta.historical-behavioral-kl-retention-sweep.success.v1"
FAILURE_SCHEMA: Final = "alberta.historical-behavioral-kl-retention-sweep.failure.v1"

DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
BENCHMARK_EXECUTION_AUTHORITY: Final = False
ARTIFACT_AUTHORITY: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_CLAIMED: Final = False
RNG_USED: Final = False
HIDDEN_RETRIES_USED: Final = False

WEIGHT_GRID: Final = (
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
AXIS_COUNT: Final = 3
EXPECTED_CELL_COUNT: Final = AXIS_COUNT * len(WEIGHT_GRID)
ARM_COUNT: Final = 4
CANDIDATE_COMPONENT_COUNT: Final = 4
PARAMETER_COUNT: Final = 2
MAX_SWEEP_REPORT_BYTES: Final = 1_000_000
HARD_MAX_SWEEP_REPORT_BYTES: Final = 4_000_000
MAX_FAILURE_MESSAGE_CHARS: Final = 256

_KNOWN_RUNTIME_FAILURES: Final = {
    "candidate actor parameters exceed the configured cap": "candidate_parameter_cap",
    "candidate actor parameters are non-finite": "nonfinite_candidate_parameters",
    "canonical in-memory report exceeds max_report_bytes": "configured_report_byte_cap",
    "canonical in-memory report exceeds the hard byte cap": "hard_report_byte_cap",
}

_LIMITATIONS: Final = (
    "the frozen grid is a descriptive one-at-a-time sweep, not a joint interaction study",
    "every cell reruns the complete four-arm probe and intentionally repeats shared work",
    "completed cells retain hashes and summaries rather than the complete underlying report",
    "failure work is conservatively bounded because the public runner exposes no partial trace",
    "a bounded failure is retained once and never retried with a changed cap or configuration",
    "raw target-arm coordinates carry no comparative selection or deployment authority",
    "the underlying analytic binary bandit remains an L0 diagnostic rather than sampled RL",
    "this in-memory sweep writes no output and has no artifact or evidence-promotion role",
)


@dataclasses.dataclass(frozen=True, slots=True)
class SweepAxis:
    """One frozen weight axis and the arm whose raw coordinates are retained."""

    ordinal: int
    name: str
    config_field_name: str
    target_arm_name: str


SWEEP_AXES: Final = (
    SweepAxis(
        ordinal=0,
        name="historical_a_state_kl_axis",
        config_field_name="historical_kl_weight",
        target_arm_name="historical_a_state_kl",
    ),
    SweepAxis(
        ordinal=1,
        name="current_b_state_kl_axis",
        config_field_name="current_kl_weight",
        target_arm_name="current_b_state_kl",
    ),
    SweepAxis(
        ordinal=2,
        name="parameter_movement_l2_axis",
        config_field_name="movement_l2_weight",
        target_arm_name="parameter_movement_l2",
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class InterventionWeights:
    """All three exact intervention weights used by one cell."""

    historical_kl_weight: float
    current_kl_weight: float
    movement_l2_weight: float


@dataclasses.dataclass(frozen=True, slots=True)
class CompletedLaneHashes:
    """Exact source, report, and component digests retained from a completed cell."""

    lane_implementation_source_sha256: str
    config_sha256: str
    source_sha256: str
    source_generator_contract_sha256: str
    source_input_sha256: str
    initial_state_sha256: str
    frozen_a_state_sha256: str
    retained_anchor_sha256: str
    arm_states_sha256: str
    trace_sha256: str
    work_sha256: str
    resource_sha256: str
    scaling_sha256: str
    target_arm_sha256: str
    target_arm_metrics_sha256: str
    complete_report_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class RawTargetArmCoordinates:
    """Unclassified A retention/forgetting and B plasticity values."""

    a_return_before_b: float
    a_return_after_b: float
    a_return_delta: float
    a_forgetting: float
    b_return_before_b: float
    b_return_after_b: float
    b_return_delta: float
    b_plasticity_gain: float


@dataclasses.dataclass(frozen=True, slots=True)
class WorkVector:
    """Exact logical work vector, also used for conservative failure bounds."""

    prefix_task_objective_evaluations: int
    b_task_objective_evaluations: int
    historical_kl_objective_evaluations: int
    current_kl_objective_evaluations: int
    movement_l2_objective_evaluations: int
    total_candidate_objective_evaluations: int
    total_candidate_gradient_float64_scalars: int
    prefix_parameter_updates: int
    routed_parameter_updates: int
    total_parameter_updates: int
    addressed_parameter_float64_scalars: int
    frozen_policy_probability_evaluations: int
    b_pre_post_probe_probability_evaluations: int
    rng_draws: int
    global_shrink_evaluations: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Exact bounded resource summary copied from one completed report."""

    parameter_count: int
    actor_state_logical_nbytes: int
    retained_anchor_float64_scalars: int
    retained_anchor_logical_nbytes: int
    frozen_actor_parameter_float64_scalars: int
    per_arm_actor_state_logical_nbytes: int
    prefix_trace_records: int
    b_trace_records_per_arm: int
    total_trace_records: int
    canonical_source_nbytes: int
    canonical_trace_nbytes: int
    max_report_bytes: int
    hard_max_report_bytes: int
    report_cap_enforced: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CompletedCellReceipt:
    """The permitted retained projection of one validated complete report."""

    schema: str
    hashes: CompletedLaneHashes
    raw_coordinates: RawTargetArmCoordinates
    work: WorkVector
    resource: ResourceSummary


@dataclasses.dataclass(frozen=True, slots=True)
class BoundedFailureReceipt:
    """One known runner failure with no retry and conservative work bounds."""

    schema: str
    kind: str
    exception_module: str
    exception_type: str
    message: str
    report_returned: bool
    validator_called: bool
    retry_attempts: int
    minimum_attempted_work: WorkVector
    maximum_attempted_work: WorkVector


@dataclasses.dataclass(frozen=True, slots=True)
class SweepCell:
    """Exactly one full-report invocation on one frozen grid cell."""

    schema: str
    ordinal: int
    axis: SweepAxis
    grid_index: int
    varied_weight: float
    weights: InterventionWeights
    target_arm_name: str
    config_sha256: str
    complete_report_attempts: int
    retry_attempts: int
    status: str
    completed: CompletedCellReceipt | None
    failure: BoundedFailureReceipt | None


@dataclasses.dataclass(frozen=True, slots=True)
class StatusCount:
    """A raw outcome-label count."""

    status: str
    count: int


@dataclasses.dataclass(frozen=True, slots=True)
class FailureKindCount:
    """A raw bounded-failure-kind count."""

    kind: str
    count: int


@dataclasses.dataclass(frozen=True, slots=True)
class AggregateAttemptedWork:
    """Exact completed work plus conservative failed-attempt work bounds."""

    cell_attempts: int
    complete_report_attempts: int
    complete_reports_returned: int
    bounded_failure_receipts: int
    retry_attempts: int
    completed_work: WorkVector
    failure_minimum_work: WorkVector
    failure_maximum_work: WorkVector
    total_minimum_work: WorkVector
    total_maximum_work: WorkVector


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalBehavioralKLRetentionSweepReport:
    """Strict in-memory descriptive sweep with no evidence authority."""

    schema: str
    status: str
    development_only: bool
    scientific_promotion_allowed: bool
    benchmark_execution_authority: bool
    artifact_authority: bool
    output_writes_allowed: bool
    evidence_claimed: bool
    rng_used: bool
    hidden_retries_used: bool
    weight_grid: tuple[float, ...]
    axes: tuple[SweepAxis, SweepAxis, SweepAxis]
    cells: tuple[SweepCell, ...]
    status_counts: tuple[StatusCount, ...]
    failure_kind_counts: tuple[FailureKindCount, ...]
    aggregate_attempted_work: AggregateAttemptedWork
    base_config_sha256: str
    sweep_contract_sha256: str
    cells_sha256: str
    aggregate_attempted_work_sha256: str
    implementation_source_sha256: str
    report_payload_sha256: str
    canonical_cells_nbytes: int
    max_sweep_report_bytes: int
    hard_max_sweep_report_bytes: int
    report_cap_enforced: bool
    limitations: tuple[str, ...]


def _canonical_value(value: object) -> object:
    """Return type-explicit JSON data; float hex preserves signed zero."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            "fields": [
                [field.name, _canonical_value(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        }
    if type(value) is tuple:
        return {"__tuple__": [_canonical_value(item) for item in cast(tuple[object, ...], value)]}
    if type(value) is str:
        return {"__str__": value}
    if type(value) is bool:
        return {"__bool__": value}
    if type(value) is int:
        return {"__int__": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return {"__float64_hex__": value.hex()}
    if value is None:
        return {"__none__": True}
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _implementation_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return all(
            _exact(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    if type(left) is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return len(left_tuple) == len(right_tuple) and all(
            _exact(a, b) for a, b in zip(left_tuple, right_tuple, strict=True)
        )
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    return bool(left == right)


def _all_finite(value: object) -> bool:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(_all_finite(getattr(value, field.name)) for field in dataclasses.fields(value))
    if type(value) is tuple:
        return all(_all_finite(item) for item in cast(tuple[object, ...], value))
    if type(value) is float:
        return math.isfinite(value)
    return True


def _zero_work() -> WorkVector:
    return WorkVector(*(0 for _ in dataclasses.fields(WorkVector)))


def _planned_work(config: HistoricalBehavioralKLRetentionConfig) -> WorkVector:
    b_arm_steps = ARM_COUNT * config.b_interference_steps
    candidate_evaluations = CANDIDATE_COMPONENT_COUNT * b_arm_steps
    total_updates = config.a_prefix_steps + b_arm_steps
    return WorkVector(
        prefix_task_objective_evaluations=config.a_prefix_steps,
        b_task_objective_evaluations=b_arm_steps,
        historical_kl_objective_evaluations=b_arm_steps,
        current_kl_objective_evaluations=b_arm_steps,
        movement_l2_objective_evaluations=b_arm_steps,
        total_candidate_objective_evaluations=candidate_evaluations,
        total_candidate_gradient_float64_scalars=candidate_evaluations * PARAMETER_COUNT,
        prefix_parameter_updates=config.a_prefix_steps,
        routed_parameter_updates=b_arm_steps,
        total_parameter_updates=total_updates,
        addressed_parameter_float64_scalars=total_updates * PARAMETER_COUNT,
        frozen_policy_probability_evaluations=2 * ARM_COUNT,
        b_pre_post_probe_probability_evaluations=4 * b_arm_steps,
        rng_draws=0,
        global_shrink_evaluations=0,
    )


def _copy_work(lane_work: object) -> WorkVector:
    return WorkVector(
        **{
            field.name: getattr(lane_work, field.name)
            for field in dataclasses.fields(WorkVector)
        }
    )


def _copy_resource(lane_resource: object) -> ResourceSummary:
    return ResourceSummary(
        **{
            field.name: getattr(lane_resource, field.name)
            for field in dataclasses.fields(ResourceSummary)
        }
    )


def _add_work(left: WorkVector, right: WorkVector) -> WorkVector:
    return WorkVector(
        **{
            field.name: getattr(left, field.name) + getattr(right, field.name)
            for field in dataclasses.fields(WorkVector)
        }
    )


def _sum_work(values: tuple[WorkVector, ...]) -> WorkVector:
    total = _zero_work()
    for value in values:
        total = _add_work(total, value)
    return total


def _weights(config: HistoricalBehavioralKLRetentionConfig) -> InterventionWeights:
    return InterventionWeights(
        historical_kl_weight=config.historical_kl_weight,
        current_kl_weight=config.current_kl_weight,
        movement_l2_weight=config.movement_l2_weight,
    )


def _cell_config(
    base: HistoricalBehavioralKLRetentionConfig,
    axis: SweepAxis,
    weight: float,
) -> HistoricalBehavioralKLRetentionConfig:
    if axis.config_field_name == "historical_kl_weight":
        return dataclasses.replace(base, historical_kl_weight=weight)
    if axis.config_field_name == "current_kl_weight":
        return dataclasses.replace(base, current_kl_weight=weight)
    if axis.config_field_name == "movement_l2_weight":
        return dataclasses.replace(base, movement_l2_weight=weight)
    raise RuntimeError("sweep axis has no configuration route")


def _bounded_failure(
    error: RuntimeError,
    planned_work: WorkVector,
) -> BoundedFailureReceipt:
    message = str(error)
    kind = _KNOWN_RUNTIME_FAILURES.get(message)
    if kind is None:
        raise error
    if len(message) > MAX_FAILURE_MESSAGE_CHARS:
        raise RuntimeError("known failure message exceeds its bounded length") from error
    return BoundedFailureReceipt(
        schema=FAILURE_SCHEMA,
        kind=kind,
        exception_module=type(error).__module__,
        exception_type=type(error).__name__,
        message=message,
        report_returned=False,
        validator_called=False,
        retry_attempts=0,
        minimum_attempted_work=_zero_work(),
        maximum_attempted_work=planned_work,
    )


def _attempt_cell(
    *,
    ordinal: int,
    axis: SweepAxis,
    grid_index: int,
    weight: float,
    config: HistoricalBehavioralKLRetentionConfig,
) -> SweepCell:
    config_sha = _sha256(config)
    planned_work = _planned_work(config)
    try:
        lane_report = run_historical_behavioral_kl_retention_development(config)
    except RuntimeError as error:
        failure = _bounded_failure(error, planned_work)
        return SweepCell(
            schema=CELL_SCHEMA,
            ordinal=ordinal,
            axis=axis,
            grid_index=grid_index,
            varied_weight=weight,
            weights=_weights(config),
            target_arm_name=axis.target_arm_name,
            config_sha256=config_sha,
            complete_report_attempts=1,
            retry_attempts=0,
            status="bounded_failure",
            completed=None,
            failure=failure,
        )

    validation_errors = validate_historical_behavioral_kl_retention_report(lane_report)
    if validation_errors:
        raise RuntimeError(
            "completed-lane validator rejected a sweep cell: " + " | ".join(validation_errors)
        )
    if lane_report.config_sha256 != config_sha:
        raise RuntimeError("completed-lane config digest differs from sweep digest")

    matching_arms = tuple(arm for arm in lane_report.arms if arm.name == axis.target_arm_name)
    if len(matching_arms) != 1:
        raise RuntimeError("target arm cardinality differs")
    target_arm = matching_arms[0]
    metrics = target_arm.metrics
    work = _copy_work(lane_report.work)
    if not _exact(work, planned_work):
        raise RuntimeError("completed-lane work differs from frozen sweep accounting")
    completed = CompletedCellReceipt(
        schema=SUCCESS_SCHEMA,
        hashes=CompletedLaneHashes(
            lane_implementation_source_sha256=lane_report.implementation_source_sha256,
            config_sha256=lane_report.config_sha256,
            source_sha256=lane_report.source_sha256,
            source_generator_contract_sha256=lane_report.source.generator_contract_sha256,
            source_input_sha256=lane_report.source.input_sha256,
            initial_state_sha256=lane_report.initial_state_sha256,
            frozen_a_state_sha256=lane_report.frozen_a_state_sha256,
            retained_anchor_sha256=lane_report.retained_anchor_sha256,
            arm_states_sha256=lane_report.arm_states_sha256,
            trace_sha256=lane_report.trace_sha256,
            work_sha256=lane_report.work_sha256,
            resource_sha256=lane_report.resource_sha256,
            scaling_sha256=lane_report.scaling_sha256,
            target_arm_sha256=_sha256(target_arm),
            target_arm_metrics_sha256=_sha256(metrics),
            complete_report_sha256=_sha256(lane_report),
        ),
        raw_coordinates=RawTargetArmCoordinates(
            a_return_before_b=metrics.a_return_before_b,
            a_return_after_b=metrics.a_return_after_b,
            a_return_delta=metrics.a_return_delta,
            a_forgetting=metrics.a_forgetting,
            b_return_before_b=metrics.b_return_before_b,
            b_return_after_b=metrics.b_return_after_b,
            b_return_delta=metrics.b_return_delta,
            b_plasticity_gain=metrics.b_plasticity_gain,
        ),
        work=work,
        resource=_copy_resource(lane_report.resource),
    )
    return SweepCell(
        schema=CELL_SCHEMA,
        ordinal=ordinal,
        axis=axis,
        grid_index=grid_index,
        varied_weight=weight,
        weights=_weights(config),
        target_arm_name=axis.target_arm_name,
        config_sha256=config_sha,
        complete_report_attempts=1,
        retry_attempts=0,
        status="completed_report",
        completed=completed,
        failure=None,
    )


def _status_counts(cells: tuple[SweepCell, ...]) -> tuple[StatusCount, ...]:
    return (
        StatusCount(
            status="completed_report",
            count=sum(cell.status == "completed_report" for cell in cells),
        ),
        StatusCount(
            status="bounded_failure",
            count=sum(cell.status == "bounded_failure" for cell in cells),
        ),
    )


def _failure_kind_counts(cells: tuple[SweepCell, ...]) -> tuple[FailureKindCount, ...]:
    kinds = tuple(sorted({cell.failure.kind for cell in cells if cell.failure is not None}))
    return tuple(
        FailureKindCount(
            kind=kind,
            count=sum(cell.failure is not None and cell.failure.kind == kind for cell in cells),
        )
        for kind in kinds
    )


def _aggregate_work(cells: tuple[SweepCell, ...]) -> AggregateAttemptedWork:
    completed_work = _sum_work(
        tuple(cell.completed.work for cell in cells if cell.completed is not None)
    )
    failure_minimum = _sum_work(
        tuple(
            cell.failure.minimum_attempted_work
            for cell in cells
            if cell.failure is not None
        )
    )
    failure_maximum = _sum_work(
        tuple(
            cell.failure.maximum_attempted_work
            for cell in cells
            if cell.failure is not None
        )
    )
    return AggregateAttemptedWork(
        cell_attempts=len(cells),
        complete_report_attempts=sum(cell.complete_report_attempts for cell in cells),
        complete_reports_returned=sum(cell.completed is not None for cell in cells),
        bounded_failure_receipts=sum(cell.failure is not None for cell in cells),
        retry_attempts=sum(cell.retry_attempts for cell in cells),
        completed_work=completed_work,
        failure_minimum_work=failure_minimum,
        failure_maximum_work=failure_maximum,
        total_minimum_work=_add_work(completed_work, failure_minimum),
        total_maximum_work=_add_work(completed_work, failure_maximum),
    )


def _report_payload(report: HistoricalBehavioralKLRetentionSweepReport) -> tuple[object, ...]:
    return tuple(
        getattr(report, field.name)
        for field in dataclasses.fields(report)
        if field.name != "report_payload_sha256"
    )


def _execute_unchecked() -> HistoricalBehavioralKLRetentionSweepReport:
    base = HistoricalBehavioralKLRetentionConfig()
    cells_list: list[SweepCell] = []
    for axis in SWEEP_AXES:
        for grid_index, weight in enumerate(WEIGHT_GRID):
            ordinal = axis.ordinal * len(WEIGHT_GRID) + grid_index
            config = _cell_config(base, axis, weight)
            cells_list.append(
                _attempt_cell(
                    ordinal=ordinal,
                    axis=axis,
                    grid_index=grid_index,
                    weight=weight,
                    config=config,
                )
            )
    cells = tuple(cells_list)
    if len(cells) != EXPECTED_CELL_COUNT:
        raise RuntimeError("sweep cell cardinality differs")
    aggregate = _aggregate_work(cells)
    contract = (WEIGHT_GRID, SWEEP_AXES, _sha256(base), EXPECTED_CELL_COUNT)
    preliminary = HistoricalBehavioralKLRetentionSweepReport(
        schema=SWEEP_SCHEMA,
        status="development_only_descriptive_sweep",
        development_only=DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        benchmark_execution_authority=BENCHMARK_EXECUTION_AUTHORITY,
        artifact_authority=ARTIFACT_AUTHORITY,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        evidence_claimed=EVIDENCE_CLAIMED,
        rng_used=RNG_USED,
        hidden_retries_used=HIDDEN_RETRIES_USED,
        weight_grid=WEIGHT_GRID,
        axes=SWEEP_AXES,
        cells=cells,
        status_counts=_status_counts(cells),
        failure_kind_counts=_failure_kind_counts(cells),
        aggregate_attempted_work=aggregate,
        base_config_sha256=_sha256(base),
        sweep_contract_sha256=_sha256(contract),
        cells_sha256=_sha256(cells),
        aggregate_attempted_work_sha256=_sha256(aggregate),
        implementation_source_sha256=_implementation_source_sha256(),
        report_payload_sha256="",
        canonical_cells_nbytes=len(_canonical_bytes(cells)),
        max_sweep_report_bytes=MAX_SWEEP_REPORT_BYTES,
        hard_max_sweep_report_bytes=HARD_MAX_SWEEP_REPORT_BYTES,
        report_cap_enforced=True,
        limitations=_LIMITATIONS,
    )
    report = dataclasses.replace(
        preliminary,
        report_payload_sha256=_sha256(_report_payload(preliminary)),
    )
    report_nbytes = len(_canonical_bytes(report))
    if report_nbytes > MAX_SWEEP_REPORT_BYTES:
        raise RuntimeError("canonical sweep report exceeds its configured byte cap")
    if report_nbytes > HARD_MAX_SWEEP_REPORT_BYTES:
        raise RuntimeError("canonical sweep report exceeds its hard byte cap")
    return report


def run_historical_behavioral_kl_retention_sweep_development(
) -> HistoricalBehavioralKLRetentionSweepReport:
    """Run all 27 complete-report attempts exactly once in memory."""

    return _execute_unchecked()


def _hash_is_canonical(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _structure_errors(
    report: HistoricalBehavioralKLRetentionSweepReport,
) -> tuple[str, ...]:
    errors: list[str] = []
    if type(report.weight_grid) is not tuple or any(
        type(weight) is not float for weight in report.weight_grid
    ):
        errors.append("weight grid types differ")
    if type(report.axes) is not tuple or len(report.axes) != AXIS_COUNT or any(
        type(axis) is not SweepAxis for axis in report.axes
    ):
        errors.append("axis types or cardinality differ")
    if type(report.cells) is not tuple or len(report.cells) != EXPECTED_CELL_COUNT or any(
        type(cell) is not SweepCell for cell in report.cells
    ):
        errors.append("cell types or cardinality differ")
        return tuple(errors)
    if type(report.status_counts) is not tuple or any(
        type(item) is not StatusCount for item in report.status_counts
    ):
        errors.append("status count types differ")
    if type(report.failure_kind_counts) is not tuple or any(
        type(item) is not FailureKindCount for item in report.failure_kind_counts
    ):
        errors.append("failure kind count types differ")
    if type(report.aggregate_attempted_work) is not AggregateAttemptedWork:
        errors.append("aggregate attempted work type differs")
    if type(report.limitations) is not tuple or any(
        type(item) is not str for item in report.limitations
    ):
        errors.append("limitation types differ")

    for cell in report.cells:
        if type(cell.axis) is not SweepAxis:
            errors.append(f"cell {cell.ordinal!r} axis type differs")
        if type(cell.weights) is not InterventionWeights:
            errors.append(f"cell {cell.ordinal!r} weight assignment type differs")
        completed_valid = cell.completed is None or type(cell.completed) is CompletedCellReceipt
        failure_valid = cell.failure is None or type(cell.failure) is BoundedFailureReceipt
        if not completed_valid:
            errors.append(f"cell {cell.ordinal!r} completed receipt type differs")
        if not failure_valid:
            errors.append(f"cell {cell.ordinal!r} failure receipt type differs")
        if not completed_valid or not failure_valid:
            continue
        if cell.completed is not None:
            if type(cell.completed.hashes) is not CompletedLaneHashes:
                errors.append(f"cell {cell.ordinal!r} hash bundle type differs")
            if type(cell.completed.raw_coordinates) is not RawTargetArmCoordinates:
                errors.append(f"cell {cell.ordinal!r} coordinate type differs")
            if type(cell.completed.work) is not WorkVector:
                errors.append(f"cell {cell.ordinal!r} work type differs")
            if type(cell.completed.resource) is not ResourceSummary:
                errors.append(f"cell {cell.ordinal!r} resource type differs")
        if cell.failure is not None:
            if type(cell.failure.minimum_attempted_work) is not WorkVector:
                errors.append(f"cell {cell.ordinal!r} minimum work type differs")
            if type(cell.failure.maximum_attempted_work) is not WorkVector:
                errors.append(f"cell {cell.ordinal!r} maximum work type differs")
    return tuple(errors)


def validate_historical_behavioral_kl_retention_sweep_report(
    report: HistoricalBehavioralKLRetentionSweepReport,
) -> tuple[str, ...]:
    """Reconstruct all attempts and verify the strict in-memory report."""

    if type(report) is not HistoricalBehavioralKLRetentionSweepReport:
        return ("report type differs",)
    errors = list(_structure_errors(report))
    if errors:
        return tuple(errors)
    report_finite = _all_finite(report)
    if not report_finite:
        errors.append("report contains non-finite values")

    expected = _execute_unchecked()
    if not _exact(report, expected):
        errors.append("report does not reconstruct bit-exactly")
    if report.implementation_source_sha256 != _implementation_source_sha256():
        errors.append("implementation source digest differs")

    digest_values = (
        report.base_config_sha256,
        report.sweep_contract_sha256,
        report.cells_sha256,
        report.aggregate_attempted_work_sha256,
        report.implementation_source_sha256,
        report.report_payload_sha256,
    )
    if any(not _hash_is_canonical(value) for value in digest_values):
        errors.append("top-level digest format differs")
    for cell in report.cells:
        if not _hash_is_canonical(cell.config_sha256):
            errors.append(f"cell {cell.ordinal!r} config digest format differs")
        if cell.completed is not None and any(
            not _hash_is_canonical(getattr(cell.completed.hashes, field.name))
            for field in dataclasses.fields(CompletedLaneHashes)
        ):
            errors.append(f"cell {cell.ordinal!r} completed digest format differs")

    if report_finite:
        base = HistoricalBehavioralKLRetentionConfig()
        try:
            recomputed = (
                (report.base_config_sha256, _sha256(base), "base config digest differs"),
                (
                    report.sweep_contract_sha256,
                    _sha256((WEIGHT_GRID, SWEEP_AXES, _sha256(base), EXPECTED_CELL_COUNT)),
                    "sweep contract digest differs",
                ),
                (report.cells_sha256, _sha256(report.cells), "cell digest differs"),
                (
                    report.aggregate_attempted_work_sha256,
                    _sha256(report.aggregate_attempted_work),
                    "aggregate attempted work digest differs",
                ),
                (
                    report.report_payload_sha256,
                    _sha256(_report_payload(dataclasses.replace(report, report_payload_sha256=""))),
                    "report payload digest differs",
                ),
            )
        except (TypeError, ValueError):
            errors.append("report digest payload is not canonical")
        else:
            for actual, calculated, message in recomputed:
                if type(actual) is not str or actual != calculated:
                    errors.append(message)
        try:
            cells_nbytes = len(_canonical_bytes(report.cells))
            report_nbytes = len(_canonical_bytes(report))
        except (TypeError, ValueError):
            errors.append("report cannot be canonically serialized")
        else:
            if type(report.canonical_cells_nbytes) is not int:
                errors.append("canonical cell byte count type differs")
            elif report.canonical_cells_nbytes != cells_nbytes:
                errors.append("canonical cell byte count differs")
            if report_nbytes > MAX_SWEEP_REPORT_BYTES:
                errors.append("report exceeds its configured byte cap")
            if report_nbytes > HARD_MAX_SWEEP_REPORT_BYTES:
                errors.append("report exceeds its hard byte cap")
    return tuple(errors)


__all__ = [
    "ARTIFACT_AUTHORITY",
    "AggregateAttemptedWork",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "BoundedFailureReceipt",
    "CELL_SCHEMA",
    "CompletedCellReceipt",
    "CompletedLaneHashes",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_CLAIMED",
    "EXPECTED_CELL_COUNT",
    "FAILURE_SCHEMA",
    "FailureKindCount",
    "HIDDEN_RETRIES_USED",
    "HistoricalBehavioralKLRetentionSweepReport",
    "InterventionWeights",
    "MAX_SWEEP_REPORT_BYTES",
    "OUTPUT_WRITES_ALLOWED",
    "RNG_USED",
    "RawTargetArmCoordinates",
    "ResourceSummary",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SUCCESS_SCHEMA",
    "SWEEP_AXES",
    "SWEEP_SCHEMA",
    "StatusCount",
    "SweepAxis",
    "SweepCell",
    "WEIGHT_GRID",
    "WorkVector",
    "run_historical_behavioral_kl_retention_sweep_development",
    "validate_historical_behavioral_kl_retention_sweep_report",
]
