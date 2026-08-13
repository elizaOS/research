"""Validated exact reward-count projection for the bound v3 development source.

This private module can summarize an already-produced event trace.  It cannot
construct experience, execute an arm or panel, issue a root, write output, or
authorize evidence or scientific promotion.  The public v3 wrapper binds every
source array and task-semantic argument before the mandatory experience
validator runs; only a successful validation can reach count projection.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np

from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as protocol,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_source as source_binding,
)

DEVELOPMENT_ONLY: Final = True
EXECUTION_AUTHORIZED: Final = False
PANEL_EXECUTION_AUTHORIZED: Final = False
ROOT_ISSUANCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
REWARD_COUNT_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-cadence-separated."
    "exact-reward-counts.v1"
)

_RECORD_FIELDS: Final = (
    "steps",
    "executed_reward_sum",
    "greedy_reward_sum",
    "executed_action_one_count",
    "greedy_action_one_count",
    "explored_count",
)
_PHASE_COUNT: Final = 10
_EXACT_WINDOW_STEPS: Final = 64


def _exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact integer")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class _ExperienceSemanticsSnapshot:
    """One immutable host snapshot shared by validation and projection."""

    action: np.ndarray[Any, Any]
    greedy_action: np.ndarray[Any, Any]
    explored: np.ndarray[Any, Any]
    target_value: np.ndarray[Any, Any]
    executed_reward: np.ndarray[Any, Any]
    greedy_reward: np.ndarray[Any, Any]
    executed_regret: np.ndarray[Any, Any]
    greedy_regret: np.ndarray[Any, Any]
    full_q: np.ndarray[Any, Any]
    raw_q: np.ndarray[Any, Any]
    behavior_q: np.ndarray[Any, Any]


def _snapshot_experience_events(events: object) -> _ExperienceSemanticsSnapshot:
    scan = cast(Any, events)

    def snapshot(name: str) -> np.ndarray[Any, Any]:
        value = np.array(getattr(scan, name), copy=True)
        value.setflags(write=False)
        return value

    return _ExperienceSemanticsSnapshot(
        action=snapshot("action"),
        greedy_action=snapshot("greedy_action"),
        explored=snapshot("explored"),
        target_value=snapshot("target_value"),
        executed_reward=snapshot("executed_reward"),
        greedy_reward=snapshot("greedy_reward"),
        executed_regret=snapshot("executed_regret"),
        greedy_regret=snapshot("greedy_regret"),
        full_q=snapshot("full_q"),
        raw_q=snapshot("raw_q"),
        behavior_q=snapshot("behavior_q"),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ExactRewardCountRecord:
    """Six sufficient integers for exact binary rewards and two-action behavior."""

    steps: int
    executed_reward_sum: int
    greedy_reward_sum: int
    executed_action_one_count: int
    greedy_action_one_count: int
    explored_count: int

    def __post_init__(self) -> None:
        for field in _RECORD_FIELDS:
            _exact_int(getattr(self, field), field=field)
        if self.steps < 1:
            raise ValueError("steps must be positive")
        for field in ("executed_reward_sum", "greedy_reward_sum"):
            reward_sum = cast(int, getattr(self, field))
            if not -self.steps <= reward_sum <= self.steps:
                raise ValueError(f"{field} is outside the exact reward range")
            if (self.steps + reward_sum) % 2:
                raise ValueError(f"{field} has invalid binary-reward parity")
        for field in (
            "executed_action_one_count",
            "greedy_action_one_count",
            "explored_count",
        ):
            count = cast(int, getattr(self, field))
            if not 0 <= count <= self.steps:
                raise ValueError(f"{field} is outside its exact count bounds")

    def to_config(self) -> dict[str, object]:
        """Return only the six stored exact integers as a fresh JSON record."""

        return {
            "steps": self.steps,
            "executed_reward_sum": self.executed_reward_sum,
            "greedy_reward_sum": self.greedy_reward_sum,
            "executed_action_one_count": self.executed_action_one_count,
            "greedy_action_one_count": self.greedy_action_one_count,
            "explored_count": self.explored_count,
        }

    @property
    def executed_positive_reward_count(self) -> int:
        return (self.steps + self.executed_reward_sum) // 2

    @property
    def executed_negative_reward_count(self) -> int:
        return (self.steps - self.executed_reward_sum) // 2

    @property
    def greedy_positive_reward_count(self) -> int:
        return (self.steps + self.greedy_reward_sum) // 2

    @property
    def greedy_negative_reward_count(self) -> int:
        return (self.steps - self.greedy_reward_sum) // 2

    @property
    def executed_action_zero_count(self) -> int:
        return self.steps - self.executed_action_one_count

    @property
    def greedy_action_zero_count(self) -> int:
        return self.steps - self.greedy_action_one_count

    @property
    def non_explored_count(self) -> int:
        return self.steps - self.explored_count


@dataclasses.dataclass(frozen=True, slots=True)
class ExactRewardCountProjection:
    """Lifetime, whole-phase, entry, and tail exact-count records."""

    phase_order: tuple[str, ...]
    lifetime: ExactRewardCountRecord
    whole_phases: tuple[ExactRewardCountRecord, ...]
    entry_windows: tuple[ExactRewardCountRecord, ...]
    tail_windows: tuple[ExactRewardCountRecord, ...]
    experience_semantics_validated: bool
    development_only: bool = True
    execution_authorized: bool = False
    output_writes_allowed: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if self.phase_order != protocol.PHASE_ORDER:
            raise ValueError("phase order must be the exact v3 ten-phase order")
        collections = (
            ("whole phases", self.whole_phases),
            ("entry windows", self.entry_windows),
            ("tail windows", self.tail_windows),
        )
        for label, records in collections:
            if type(records) is not tuple or len(records) != _PHASE_COUNT:
                raise ValueError(f"{label} must contain exactly ten records")
            if any(type(record) is not ExactRewardCountRecord for record in records):
                raise TypeError(f"{label} must contain exact reward-count records")
        if type(self.lifetime) is not ExactRewardCountRecord:
            raise TypeError("lifetime must be an exact reward-count record")
        if any(record.steps != _EXACT_WINDOW_STEPS for record in self.entry_windows):
            raise ValueError("every entry-window record must contain exactly 64 steps")
        if any(record.steps != _EXACT_WINDOW_STEPS for record in self.tail_windows):
            raise ValueError("every tail-window record must contain exactly 64 steps")
        for field in _RECORD_FIELDS:
            phase_sum = sum(
                cast(int, getattr(record, field)) for record in self.whole_phases
            )
            if getattr(self.lifetime, field) != phase_sum:
                raise ValueError(
                    f"lifetime {field} must equal the exact sum of whole phases"
                )
        if self.experience_semantics_validated is not True:
            raise ValueError("experience semantics must be validated before projection")
        if self.development_only is not True:
            raise ValueError("reward-count projection must remain development-only")
        if (
            self.execution_authorized is not False
            or self.output_writes_allowed is not False
            or self.evidence_authorized is not False
            or self.scientific_promotion_allowed is not False
        ):
            raise ValueError("reward-count projection cannot acquire execution or claim authority")

    def to_config(self) -> dict[str, object]:
        """Return one fresh strict-JSON projection record under the frozen schema."""

        return {
            "schema": REWARD_COUNT_SCHEMA,
            "phase_order": list(self.phase_order),
            "lifetime": self.lifetime.to_config(),
            "whole_phases": [record.to_config() for record in self.whole_phases],
            "entry_windows": [record.to_config() for record in self.entry_windows],
            "tail_windows": [record.to_config() for record in self.tail_windows],
            "experience_semantics_validated": self.experience_semantics_validated,
            "development_only": self.development_only,
            "execution_authorized": self.execution_authorized,
            "output_writes_allowed": self.output_writes_allowed,
            "evidence_authorized": self.evidence_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


def _exact_event_array(
    events: object,
    *,
    name: str,
    steps: int,
    dtype: np.dtype[Any],
) -> np.ndarray[Any, Any]:
    value = np.asarray(getattr(cast(Any, events), name))
    if value.shape != (steps,) or value.dtype != dtype:
        raise RuntimeError(
            f"reward-count field {name} has shape/dtype {value.shape}/{value.dtype}, "
            f"expected {(steps,)}/{dtype}"
        )
    return value


def _record_from_arrays(
    *,
    executed_reward: np.ndarray[Any, Any],
    greedy_reward: np.ndarray[Any, Any],
    action: np.ndarray[Any, Any],
    greedy_action: np.ndarray[Any, Any],
    explored: np.ndarray[Any, Any],
    start: int,
    stop: int,
) -> ExactRewardCountRecord:
    steps = stop - start
    executed_slice = executed_reward[start:stop]
    greedy_slice = greedy_reward[start:stop]
    return ExactRewardCountRecord(
        steps=steps,
        executed_reward_sum=(
            int(np.count_nonzero(executed_slice == np.float32(1.0)))
            - int(np.count_nonzero(executed_slice == np.float32(-1.0)))
        ),
        greedy_reward_sum=(
            int(np.count_nonzero(greedy_slice == np.float32(1.0)))
            - int(np.count_nonzero(greedy_slice == np.float32(-1.0)))
        ),
        executed_action_one_count=int(
            np.count_nonzero(action[start:stop] == np.int32(1))
        ),
        greedy_action_one_count=int(
            np.count_nonzero(greedy_action[start:stop] == np.int32(1))
        ),
        explored_count=int(np.count_nonzero(explored[start:stop])),
    )


def _project_exact_reward_count_records(
    geometry: engine.FutureUtilityEndpointGeometry,
    events: object,
    *,
    entry_window: int,
    tail_window: int,
) -> ExactRewardCountProjection:
    """Project arrays only after the caller has completed semantic validation."""

    steps = geometry.total_steps
    executed_reward = _exact_event_array(
        events,
        name="executed_reward",
        steps=steps,
        dtype=np.dtype(np.float32),
    )
    greedy_reward = _exact_event_array(
        events,
        name="greedy_reward",
        steps=steps,
        dtype=np.dtype(np.float32),
    )
    action = _exact_event_array(
        events,
        name="action",
        steps=steps,
        dtype=np.dtype(np.int32),
    )
    greedy_action = _exact_event_array(
        events,
        name="greedy_action",
        steps=steps,
        dtype=np.dtype(np.int32),
    )
    explored = _exact_event_array(
        events,
        name="explored",
        steps=steps,
        dtype=np.dtype(np.bool_),
    )
    if (
        not np.all(np.isin(executed_reward, (-1.0, 1.0)))
        or not np.all(np.isin(greedy_reward, (-1.0, 1.0)))
    ):
        raise RuntimeError("reward-count rewards must be exact negative or positive one")
    if not np.all(np.isin(action, (0, 1))) or not np.all(
        np.isin(greedy_action, (0, 1))
    ):
        raise RuntimeError("reward-count actions must be exact zero or one")

    def record(start: int, stop: int) -> ExactRewardCountRecord:
        return _record_from_arrays(
            executed_reward=executed_reward,
            greedy_reward=greedy_reward,
            action=action,
            greedy_action=greedy_action,
            explored=explored,
            start=start,
            stop=stop,
        )

    boundaries = geometry.phase_boundaries
    phase_ranges = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
    lifetime = record(0, steps)
    whole_phases = tuple(record(start, stop) for start, stop in phase_ranges)
    entry_windows = tuple(
        record(start, start + entry_window) for start, _stop in phase_ranges
    )
    tail_windows = tuple(
        record(stop - tail_window, stop) for _start, stop in phase_ranges
    )
    if tuple(item.steps for item in whole_phases) != geometry.phase_lengths:
        raise RuntimeError("whole-phase record lengths do not match endpoint geometry")
    return ExactRewardCountProjection(
        phase_order=geometry.phase_order,
        lifetime=lifetime,
        whole_phases=whole_phases,
        entry_windows=entry_windows,
        tail_windows=tail_windows,
        experience_semantics_validated=True,
    )


def _validate_semantic_receipt(
    receipt: object,
    *,
    geometry: engine.FutureUtilityEndpointGeometry,
    composed_readout_enabled: bool,
) -> None:
    if type(receipt) is not dict:
        raise RuntimeError("experience validator did not return its exact receipt")
    values = cast(dict[str, object], receipt)
    if (
        values.get("all_experience_semantics_match") is not True
        or type(values.get("steps")) is not int
        or values.get("steps") != geometry.total_steps
        or values.get("composed_readout_enabled") is not composed_readout_enabled
    ):
        raise RuntimeError("experience validator receipt is not exact and accepted")
    for field in ("explored_step_count",):
        value = values.get(field)
        if type(value) is not int or not 0 <= value <= geometry.total_steps:
            raise RuntimeError(f"experience validator receipt has invalid {field}")
    for field in ("executed_action_counts", "greedy_action_counts"):
        counts = values.get(field)
        if (
            type(counts) is not list
            or len(counts) != control.ACTION_HEADS
            or any(type(value) is not int or value < 0 for value in counts)
            or sum(cast(list[int], counts)) != geometry.total_steps
        ):
            raise RuntimeError(f"experience validator receipt has invalid {field}")


def _validate_projection_against_receipt(
    projection: ExactRewardCountProjection,
    receipt: dict[str, object],
) -> None:
    executed_counts = cast(list[int], receipt["executed_action_counts"])
    greedy_counts = cast(list[int], receipt["greedy_action_counts"])
    lifetime = projection.lifetime
    if (
        receipt["explored_step_count"] != lifetime.explored_count
        or executed_counts
        != [lifetime.executed_action_zero_count, lifetime.executed_action_one_count]
        or greedy_counts
        != [lifetime.greedy_action_zero_count, lifetime.greedy_action_one_count]
    ):
        raise RuntimeError("reward-count projection does not close against validator receipt")


def _validated_exact_reward_count_projection(
    geometry: engine.FutureUtilityEndpointGeometry,
    source: control.BoundCompositionalControlLifeSource,
    events: object,
    *,
    phase_target_raw_indices: Sequence[Sequence[int]],
    action_reward_multipliers: Sequence[float],
    composed_readout_enabled: bool,
    entry_window: int,
    tail_window: int,
) -> ExactRewardCountProjection:
    """Private short-test seam that still makes semantic validation mandatory."""

    if type(geometry) is not engine.FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    if type(source) is not control.BoundCompositionalControlLifeSource:
        raise TypeError("source must be an exact BoundCompositionalControlLifeSource")
    if geometry.phase_order != protocol.PHASE_ORDER:
        raise ValueError("geometry must use the exact v3 phase order")
    if (
        type(entry_window) is not int
        or type(tail_window) is not int
        or entry_window != _EXACT_WINDOW_STEPS
        or tail_window != _EXACT_WINDOW_STEPS
    ):
        raise ValueError("entry and tail windows must both be exactly 64 steps")
    if any(length < _EXACT_WINDOW_STEPS for length in geometry.phase_lengths):
        raise ValueError("every phase must contain both exact 64-step windows")

    event_snapshot = _snapshot_experience_events(events)
    receipt = engine.validate_future_utility_experience_semantics(
        geometry,
        event_snapshot,
        observations=source.observations,
        phase_indices=source.phase_indices,
        exploration_mask=source.exploration_mask,
        random_actions=source.random_actions,
        phase_target_raw_indices=phase_target_raw_indices,
        action_reward_multipliers=action_reward_multipliers,
        composed_readout_enabled=composed_readout_enabled,
    )
    _validate_semantic_receipt(
        receipt,
        geometry=geometry,
        composed_readout_enabled=composed_readout_enabled,
    )
    projection = _project_exact_reward_count_records(
        geometry,
        event_snapshot,
        entry_window=entry_window,
        tail_window=tail_window,
    )
    _validate_projection_against_receipt(projection, receipt)
    return projection


def project_v3_exact_reward_counts(
    bound_source: source_binding.BoundV3Source,
    events: object,
) -> ExactRewardCountProjection:
    """Validate and project one event trace against the exact bound v3 source."""

    if type(bound_source) is not source_binding.BoundV3Source:
        raise TypeError("bound_source must be an exact BoundV3Source")
    source_binding.validate_protocol_and_source_constants()
    canonical_bound = dataclasses.replace(bound_source)
    if (
        protocol.ENTRY_WINDOW != _EXACT_WINDOW_STEPS
        or protocol.TAIL_WINDOW != _EXACT_WINDOW_STEPS
        or len(protocol.PHASE_ORDER) != _PHASE_COUNT
        or protocol.SOURCE_ARM_CONFIG.get("composed_readout_enabled") is not True
    ):
        raise RuntimeError("v3 reward-count protocol bindings have drifted")
    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=protocol.PHASE_ORDER,
        phase_lengths=protocol.PHASE_LENGTHS,
        target_names=protocol.TARGET_NAMES,
        curation_interval=protocol.CURATION_INTERVAL,
    )
    if (
        geometry.total_steps != protocol.TOTAL_STEPS
        or geometry.phase_boundaries != protocol.PHASE_BOUNDARIES
        or canonical_bound.control_protocol.phase_lengths != geometry.phase_lengths
    ):
        raise RuntimeError("bound v3 source geometry differs from the count contract")
    return _validated_exact_reward_count_projection(
        geometry,
        canonical_bound.source,
        events,
        phase_target_raw_indices=protocol.PHASE_TARGET_RAW_INDICES,
        action_reward_multipliers=(-1.0, 1.0),
        composed_readout_enabled=True,
        entry_window=64,
        tail_window=64,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "ROOT_ISSUANCE_AUTHORIZED",
    "REWARD_COUNT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "ExactRewardCountProjection",
    "ExactRewardCountRecord",
    "project_v3_exact_reward_counts",
]
