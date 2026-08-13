# mypy: disable-error-code="call-arg"
"""Strict development-only matched experiential-memory transfer diagnostics.

The evaluator owns a fixed recurring A/B/A schedule and evaluates every event
causally: query the bounded memory first, score the retrieved outcome against
an evaluator-only target and a stateless matched reference, then write the
current typed entry.  Context, case, and expected-outcome annotations never
cross the memory boundary.  The reference receives the same event, query, and
write *opportunity* budget but keeps no state and always uses its declared base
prediction/action.

The resulting report is descriptive.  It deliberately has no success gate,
scientific claim, promotion path, or SOTA verdict.  In particular, transfer
measurements here are unrelated to the candidate-update safety audit and to
paper-defined actor-sample delight.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
    ExperientialMemoryStepResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIENTIAL_MEMORY_TRANSFER_CONFIG_SCHEMA = (
    "alberta.experiential-memory-transfer.config.v1"
)
EXPERIENTIAL_MEMORY_TRANSFER_PROTOCOL_SCHEMA = (
    "alberta.experiential-memory-transfer.protocol.v1"
)
EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA = (
    "alberta.experiential-memory-transfer.report.v1"
)
EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA = (
    "alberta.experiential-memory-transfer.snapshot.v1"
)
DEVELOPMENT_STATUS = "development-only-not-assessed"

MAX_ABSOLUTE_EVENTS = 256
MAX_ABSOLUTE_PHASES = 16
MAX_ABSOLUTE_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_ABSOLUTE_REPORT_BYTES = 64 * 1024 * 1024

SOURCE_PATHS = (
    Path("alberta_framework/core/checkpoints.py"),
    Path("alberta_framework/core/experiential_memory.py"),
    Path("alberta_framework/evaluation/experiential_memory_transfer.py"),
)

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "the fixed evaluator-owned A/B/A trace does not establish external validity",
    "the matched reference is stateless and receives equal opportunities, not memory storage",
    "signed transfer descriptions and negative-transfer counts are descriptive, not thresholds",
    "one bounded run establishes no efficacy, retention, scientific-promotion, or SOTA claim",
    "the diagnostic is unrelated to the candidate-update audit and paper-defined delight",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a canonical JSON array")
    return cast(list[object], value)


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase canonical identifier")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a canonical positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a canonical non-negative integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    values = _list(value, name=name)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return tuple(
        _finite_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(values)
    )


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a canonical boolean")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiential_memory_transfer_source_snapshot(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Hash the complete local source closure used by this diagnostic."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


def _tree_nbytes(tree: object) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
    )


def frozen_experiential_memory_state_sha256(state: ExperientialMemoryState) -> str:
    """Hash every persistent leaf with explicit order, shape, dtype, and bytes."""
    if not isinstance(state, ExperientialMemoryState):
        raise TypeError("state must be ExperientialMemoryState")
    digest = hashlib.sha256()
    for index, raw_leaf in enumerate(jax.tree.leaves(state)):
        leaf = np.asarray(jax.device_get(raw_leaf))
        descriptor = {
            "index": index,
            "shape": list(leaf.shape),
            "dtype": leaf.dtype.str,
            "nbytes": int(leaf.nbytes),
        }
        digest.update(_canonical_json_bytes(descriptor))
        digest.update(b"\0")
        digest.update(leaf.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryTransferConfig:
    """Memory construction, stateless reference, and hard diagnostic bounds."""

    memory_config: ExperientialMemoryConfig
    reference_prediction: tuple[float, ...]
    reference_action: tuple[float, ...]
    max_events: int
    max_phases: int
    max_initial_snapshot_bytes: int
    max_report_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.memory_config, ExperientialMemoryConfig):
            raise TypeError("memory_config must be ExperientialMemoryConfig")
        # Reconstructing the core object applies the complete core validation.
        ExperientialMemory(self.memory_config)
        if not isinstance(self.reference_prediction, tuple):
            raise TypeError("reference_prediction must be a tuple")
        if not isinstance(self.reference_action, tuple):
            raise TypeError("reference_action must be a tuple")
        if len(self.reference_prediction) != self.memory_config.outcome_dim:
            raise ValueError("reference_prediction must match memory outcome_dim")
        if len(self.reference_action) != self.memory_config.action_dim:
            raise ValueError("reference_action must match memory action_dim")
        for index, value in enumerate(self.reference_prediction):
            _finite_float(value, name=f"reference_prediction[{index}]")
        for index, value in enumerate(self.reference_action):
            _finite_float(value, name=f"reference_action[{index}]")
        _positive_int(self.max_events, name="max_events")
        _positive_int(self.max_phases, name="max_phases")
        _positive_int(
            self.max_initial_snapshot_bytes,
            name="max_initial_snapshot_bytes",
        )
        _positive_int(self.max_report_bytes, name="max_report_bytes")
        if self.max_events > MAX_ABSOLUTE_EVENTS:
            raise ValueError("max_events exceeds the hard evaluator ceiling")
        if self.max_phases > MAX_ABSOLUTE_PHASES:
            raise ValueError("max_phases exceeds the hard evaluator ceiling")
        if self.max_initial_snapshot_bytes > MAX_ABSOLUTE_SNAPSHOT_BYTES:
            raise ValueError("max_initial_snapshot_bytes exceeds the hard evaluator ceiling")
        if self.max_report_bytes > MAX_ABSOLUTE_REPORT_BYTES:
            raise ValueError("max_report_bytes exceeds the hard evaluator ceiling")
        configured_bytes = ExperientialMemory(self.memory_config).persistent_bytes
        if configured_bytes > self.max_initial_snapshot_bytes:
            raise ValueError("configured memory exceeds the initial snapshot byte bound")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "memory_config": self.memory_config.to_config(),
            "reference_prediction": list(self.reference_prediction),
            "reference_action": list(self.reference_action),
            "max_events": self.max_events,
            "max_phases": self.max_phases,
            "max_initial_snapshot_bytes": self.max_initial_snapshot_bytes,
            "max_report_bytes": self.max_report_bytes,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExperientialMemoryTransferConfig:
        expected = {
            "schema",
            "type",
            "development_status",
            "assessment_status",
            "scientific_promotion_allowed",
            "performance_thresholds_applied",
            "memory_config",
            "reference_prediction",
            "reference_action",
            "max_events",
            "max_phases",
            "max_initial_snapshot_bytes",
            "max_report_bytes",
        }
        if set(payload) != expected:
            raise ValueError("experiential-memory transfer config fields do not match v1")
        fixed = {
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_CONFIG_SCHEMA,
            "type": cls.__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
        }
        for name, expected_value in fixed.items():
            if not _strict_json_equal(payload.get(name), expected_value):
                raise ValueError(f"experiential-memory transfer config {name} is invalid")
        result = cls(
            memory_config=ExperientialMemoryConfig.from_config(
                dict(_mapping(payload.get("memory_config"), name="memory_config"))
            ),
            reference_prediction=_float_tuple(
                payload.get("reference_prediction"), name="reference_prediction"
            ),
            reference_action=_float_tuple(
                payload.get("reference_action"), name="reference_action"
            ),
            max_events=_positive_int(payload.get("max_events"), name="max_events"),
            max_phases=_positive_int(payload.get("max_phases"), name="max_phases"),
            max_initial_snapshot_bytes=_positive_int(
                payload.get("max_initial_snapshot_bytes"),
                name="max_initial_snapshot_bytes",
            ),
            max_report_bytes=_positive_int(
                payload.get("max_report_bytes"), name="max_report_bytes"
            ),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("experiential-memory transfer config is noncanonical")
        return result


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryTransferEvent:
    """One query-before-write event with evaluator-only truth annotations."""

    event_id: str
    case_id: str
    query_key: tuple[float, ...]
    representation_version: int
    query_uncertainty: float
    query_uncertainty_available: bool
    entry_observation: tuple[float, ...]
    entry_key: tuple[float, ...]
    entry_action: tuple[float, ...]
    entry_outcome: tuple[float, ...]
    entry_reward: float
    entry_uncertainty: float
    entry_uncertainty_available: bool
    entry_safety_cost: float
    entry_safety_cost_available: bool
    entry_reliability: float
    entry_utility: float
    entry_utility_available: bool
    entry_representation_version: int
    entry_valid: bool
    entry_age: int
    entry_provenance_id: int
    entry_source_id: int
    expected_outcome: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.event_id, name="event_id")
        _identifier(self.case_id, name="case_id")
        for name in (
            "query_key",
            "entry_observation",
            "entry_key",
            "entry_action",
            "entry_outcome",
            "expected_outcome",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise TypeError(f"{name} must be a non-empty tuple")
            for index, item in enumerate(value):
                _finite_float(item, name=f"{name}[{index}]")
        for name in (
            "query_uncertainty",
            "entry_reward",
            "entry_uncertainty",
            "entry_safety_cost",
            "entry_reliability",
            "entry_utility",
        ):
            _finite_float(getattr(self, name), name=name)
        for name in (
            "query_uncertainty_available",
            "entry_uncertainty_available",
            "entry_safety_cost_available",
            "entry_utility_available",
            "entry_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "representation_version",
            "entry_representation_version",
            "entry_age",
            "entry_provenance_id",
            "entry_source_id",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.query_uncertainty < 0.0 or self.entry_uncertainty < 0.0:
            raise ValueError("uncertainty values must be non-negative")
        if self.entry_safety_cost < 0.0:
            raise ValueError("entry_safety_cost must be non-negative")
        if not 0.0 <= self.entry_reliability <= 1.0:
            raise ValueError("entry_reliability must be in [0, 1]")
        if self.entry_utility < 0.0:
            raise ValueError("entry_utility must be non-negative")
        if not self.entry_uncertainty_available and self.entry_uncertainty != 0.0:
            raise ValueError("unavailable entry uncertainty must be explicit zero")
        if not self.entry_safety_cost_available and self.entry_safety_cost != 0.0:
            raise ValueError("unavailable entry safety cost must be explicit zero")
        if not self.entry_utility_available and self.entry_utility != 0.0:
            raise ValueError("unavailable entry utility must be explicit zero")
        if self.entry_outcome != self.expected_outcome:
            raise ValueError("entry outcome must equal evaluator truth written after query")

    def learner_input_config(self) -> dict[str, object]:
        """Return exactly the fields crossing the memory boundary."""
        return {
            "query_key": list(self.query_key),
            "representation_version": self.representation_version,
            "query_uncertainty": self.query_uncertainty,
            "query_uncertainty_available": self.query_uncertainty_available,
            "entry": {
                "observation": list(self.entry_observation),
                "key": list(self.entry_key),
                "action": list(self.entry_action),
                "outcome": list(self.entry_outcome),
                "reward": self.entry_reward,
                "uncertainty": self.entry_uncertainty,
                "uncertainty_available": self.entry_uncertainty_available,
                "safety_cost": self.entry_safety_cost,
                "safety_cost_available": self.entry_safety_cost_available,
                "reliability": self.entry_reliability,
                "utility": self.entry_utility,
                "utility_available": self.entry_utility_available,
                "representation_version": self.entry_representation_version,
                "valid": self.entry_valid,
                "age": self.entry_age,
                "provenance_id": self.entry_provenance_id,
                "source_id": self.entry_source_id,
            },
        }

    def learner_case_config(self) -> dict[str, object]:
        """Return recurrence-comparable learner input, excluding unique provenance."""
        payload = self.learner_input_config()
        entry = dict(cast(dict[str, object], payload["entry"]))
        entry.pop("provenance_id")
        payload["entry"] = entry
        return payload

    def to_config(self) -> dict[str, object]:
        return {
            "learner_input": self.learner_input_config(),
            "evaluator_only": {
                "event_id": self.event_id,
                "case_id": self.case_id,
                "expected_outcome": list(self.expected_outcome),
            },
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExperientialMemoryTransferEvent:
        if set(payload) != {"learner_input", "evaluator_only"}:
            raise ValueError("transfer event fields do not match v1")
        learner = _mapping(payload.get("learner_input"), name="learner_input")
        if set(learner) != {
            "query_key",
            "representation_version",
            "query_uncertainty",
            "query_uncertainty_available",
            "entry",
        }:
            raise ValueError("transfer event learner fields do not match v1")
        entry = _mapping(learner.get("entry"), name="entry")
        expected_entry_fields = {
            "observation",
            "key",
            "action",
            "outcome",
            "reward",
            "uncertainty",
            "uncertainty_available",
            "safety_cost",
            "safety_cost_available",
            "reliability",
            "utility",
            "utility_available",
            "representation_version",
            "valid",
            "age",
            "provenance_id",
            "source_id",
        }
        if set(entry) != expected_entry_fields:
            raise ValueError("transfer event entry fields do not match v1")
        evaluator = _mapping(payload.get("evaluator_only"), name="evaluator_only")
        if set(evaluator) != {"event_id", "case_id", "expected_outcome"}:
            raise ValueError("transfer event evaluator fields do not match v1")
        result = cls(
            event_id=_identifier(evaluator.get("event_id"), name="event_id"),
            case_id=_identifier(evaluator.get("case_id"), name="case_id"),
            query_key=_float_tuple(learner.get("query_key"), name="query_key"),
            representation_version=_nonnegative_int(
                learner.get("representation_version"), name="representation_version"
            ),
            query_uncertainty=_finite_float(
                learner.get("query_uncertainty"), name="query_uncertainty"
            ),
            query_uncertainty_available=_strict_bool(
                learner.get("query_uncertainty_available"),
                name="query_uncertainty_available",
            ),
            entry_observation=_float_tuple(entry.get("observation"), name="observation"),
            entry_key=_float_tuple(entry.get("key"), name="key"),
            entry_action=_float_tuple(entry.get("action"), name="action"),
            entry_outcome=_float_tuple(entry.get("outcome"), name="outcome"),
            entry_reward=_finite_float(entry.get("reward"), name="reward"),
            entry_uncertainty=_finite_float(
                entry.get("uncertainty"), name="uncertainty"
            ),
            entry_uncertainty_available=_strict_bool(
                entry.get("uncertainty_available"), name="uncertainty_available"
            ),
            entry_safety_cost=_finite_float(
                entry.get("safety_cost"), name="safety_cost"
            ),
            entry_safety_cost_available=_strict_bool(
                entry.get("safety_cost_available"), name="safety_cost_available"
            ),
            entry_reliability=_finite_float(
                entry.get("reliability"), name="reliability"
            ),
            entry_utility=_finite_float(entry.get("utility"), name="utility"),
            entry_utility_available=_strict_bool(
                entry.get("utility_available"), name="utility_available"
            ),
            entry_representation_version=_nonnegative_int(
                entry.get("representation_version"), name="entry representation_version"
            ),
            entry_valid=_strict_bool(entry.get("valid"), name="valid"),
            entry_age=_nonnegative_int(entry.get("age"), name="age"),
            entry_provenance_id=_nonnegative_int(
                entry.get("provenance_id"), name="provenance_id"
            ),
            entry_source_id=_nonnegative_int(entry.get("source_id"), name="source_id"),
            expected_outcome=_float_tuple(
                evaluator.get("expected_outcome"), name="expected_outcome"
            ),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("transfer event is noncanonical")
        return result

    def to_entry(self) -> ExperientialMemoryEntry:
        """Build the exact typed entry supplied after this event's query."""
        return ExperientialMemoryEntry(
            observation=jnp.asarray(self.entry_observation, dtype=jnp.float32),
            key=jnp.asarray(self.entry_key, dtype=jnp.float32),
            action=jnp.asarray(self.entry_action, dtype=jnp.float32),
            outcome=jnp.asarray(self.entry_outcome, dtype=jnp.float32),
            reward=jnp.asarray(self.entry_reward, dtype=jnp.float32),
            uncertainty=jnp.asarray(self.entry_uncertainty, dtype=jnp.float32),
            uncertainty_available=jnp.asarray(
                self.entry_uncertainty_available, dtype=jnp.bool_
            ),
            safety_cost=jnp.asarray(self.entry_safety_cost, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(
                self.entry_safety_cost_available, dtype=jnp.bool_
            ),
            reliability=jnp.asarray(self.entry_reliability, dtype=jnp.float32),
            utility=jnp.asarray(self.entry_utility, dtype=jnp.float32),
            utility_available=jnp.asarray(self.entry_utility_available, dtype=jnp.bool_),
            representation_version=jnp.asarray(
                self.entry_representation_version, dtype=jnp.int32
            ),
            valid=jnp.asarray(self.entry_valid, dtype=jnp.bool_),
            age=jnp.asarray(self.entry_age, dtype=jnp.int32),
            provenance_id=jnp.asarray(self.entry_provenance_id, dtype=jnp.int32),
            source_id=jnp.asarray(self.entry_source_id, dtype=jnp.int32),
        )


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryTransferPhase:
    """Evaluator-only contiguous phase annotation."""

    phase_id: str
    evaluator_regime_id: str
    event_count: int

    def __post_init__(self) -> None:
        _identifier(self.phase_id, name="phase_id")
        _identifier(self.evaluator_regime_id, name="evaluator_regime_id")
        _positive_int(self.event_count, name="event_count")

    def to_config(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "evaluator_regime_id": self.evaluator_regime_id,
            "event_count": self.event_count,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExperientialMemoryTransferPhase:
        if set(payload) != {"phase_id", "evaluator_regime_id", "event_count"}:
            raise ValueError("transfer phase fields do not match v1")
        result = cls(
            phase_id=_identifier(payload.get("phase_id"), name="phase_id"),
            evaluator_regime_id=_identifier(
                payload.get("evaluator_regime_id"), name="evaluator_regime_id"
            ),
            event_count=_positive_int(payload.get("event_count"), name="event_count"),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("transfer phase is noncanonical")
        return result


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryTransferProtocol:
    """Fixed evaluator-owned recurring A/B/A query-before-write schedule."""

    protocol_id: str
    events: tuple[ExperientialMemoryTransferEvent, ...]
    phases: tuple[ExperientialMemoryTransferPhase, ...]

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id")
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a non-empty tuple")
        if not all(isinstance(event, ExperientialMemoryTransferEvent) for event in self.events):
            raise TypeError("events must contain ExperientialMemoryTransferEvent")
        if not isinstance(self.phases, tuple) or len(self.phases) != 3:
            raise ValueError("transfer protocol requires exactly three A/B/A phases")
        if not all(isinstance(phase, ExperientialMemoryTransferPhase) for phase in self.phases):
            raise TypeError("phases must contain ExperientialMemoryTransferPhase")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event_id values must be unique")
        if len({event.entry_provenance_id for event in self.events}) != len(self.events):
            raise ValueError("entry provenance identifiers must be unique")
        if len({phase.phase_id for phase in self.phases}) != len(self.phases):
            raise ValueError("phase_id values must be unique")
        if sum(phase.event_count for phase in self.phases) != len(self.events):
            raise ValueError("phase event counts must exactly cover events")
        regimes = [phase.evaluator_regime_id for phase in self.phases]
        if regimes[0] != regimes[2] or regimes[0] == regimes[1]:
            raise ValueError("transfer protocol phases must have exact A/B/A regime recurrence")
        first, _, returned = self.phase_events()
        if len(first) != len(returned) or [event.case_id for event in first] != [
            event.case_id for event in returned
        ]:
            raise ValueError("recurring phases must reuse exact evaluator case ordering")
        if [event.learner_case_config() for event in first] != [
            event.learner_case_config() for event in returned
        ]:
            raise ValueError("recurring phases must reuse exact learner-visible cases")
        if len({event.entry_source_id for event in self.events}) != 1:
            raise ValueError("entry source_id must be constant and cannot encode regime labels")
        if any(not event.entry_valid for event in self.events):
            raise ValueError("fixed transfer protocol requires valid write entries")
        if len({event.expected_outcome for event in self.events}) < 2:
            raise ValueError("transfer protocol requires nonconstant evaluator outcomes")
        if all(event.query_uncertainty_available for event in self.events):
            raise ValueError("transfer protocol must exercise unavailable query uncertainty")
        if len({event.representation_version for event in self.events}) < 2:
            raise ValueError("transfer protocol must exercise representation-version changes")

    def phase_events(
        self,
    ) -> tuple[tuple[ExperientialMemoryTransferEvent, ...], ...]:
        groups: list[tuple[ExperientialMemoryTransferEvent, ...]] = []
        start = 0
        for phase in self.phases:
            end = start + phase.event_count
            groups.append(self.events[start:end])
            start = end
        return tuple(groups)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_PROTOCOL_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "ownership": "evaluator-owned-fixed-recurring-query-before-write",
            "learner_visible_fields": [
                "query_key",
                "representation_version",
                "query_uncertainty",
                "query_uncertainty_available",
                "entry",
            ],
            "evaluator_only_fields": [
                "phase_id",
                "evaluator_regime_id",
                "case_id",
                "expected_outcome",
            ],
            "regime_identifiers_visible_to_memory": False,
            "protocol_id": self.protocol_id,
            "events": [event.to_config() for event in self.events],
            "phases": [phase.to_config() for phase in self.phases],
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExperientialMemoryTransferProtocol:
        expected = {
            "schema",
            "type",
            "development_status",
            "ownership",
            "learner_visible_fields",
            "evaluator_only_fields",
            "regime_identifiers_visible_to_memory",
            "protocol_id",
            "events",
            "phases",
        }
        if set(payload) != expected:
            raise ValueError("experiential-memory transfer protocol fields do not match v1")
        fixed = {
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_PROTOCOL_SCHEMA,
            "type": cls.__name__,
            "development_status": DEVELOPMENT_STATUS,
            "ownership": "evaluator-owned-fixed-recurring-query-before-write",
            "learner_visible_fields": [
                "query_key",
                "representation_version",
                "query_uncertainty",
                "query_uncertainty_available",
                "entry",
            ],
            "evaluator_only_fields": [
                "phase_id",
                "evaluator_regime_id",
                "case_id",
                "expected_outcome",
            ],
            "regime_identifiers_visible_to_memory": False,
        }
        for name, expected_value in fixed.items():
            if not _strict_json_equal(payload.get(name), expected_value):
                raise ValueError(f"experiential-memory transfer protocol {name} is invalid")
        raw_events = _list(payload.get("events"), name="events")
        raw_phases = _list(payload.get("phases"), name="phases")
        if any(not isinstance(value, Mapping) for value in (*raw_events, *raw_phases)):
            raise ValueError("events and phases must contain objects")
        result = cls(
            protocol_id=_identifier(payload.get("protocol_id"), name="protocol_id"),
            events=tuple(
                ExperientialMemoryTransferEvent.from_config(
                    cast(Mapping[str, object], value)
                )
                for value in raw_events
            ),
            phases=tuple(
                ExperientialMemoryTransferPhase.from_config(
                    cast(Mapping[str, object], value)
                )
                for value in raw_phases
            ),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("experiential-memory transfer protocol is noncanonical")
        return result


def default_experiential_memory_transfer_config() -> ExperientialMemoryTransferConfig:
    """Return the bounded v1 development configuration; it is not a threshold set."""
    return ExperientialMemoryTransferConfig(
        memory_config=ExperientialMemoryConfig(
            capacity=4,
            observation_dim=2,
            key_dim=2,
            action_dim=1,
            outcome_dim=1,
            top_k=2,
            min_neighbors=1,
            distance_scale=0.05,
            min_similarity=0.8,
            min_effective_reliability=0.05,
            max_uncertainty=0.5,
            max_safety_cost=0.5,
            max_age=3,
            staleness_scale=100.0,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=1.0,
            recency_scale=10.0,
        ),
        reference_prediction=(0.0,),
        reference_action=(0.5,),
        max_events=16,
        max_phases=3,
        max_initial_snapshot_bytes=1_000_000,
        max_report_bytes=2_000_000,
    )


def _event(
    event_id: str,
    case_id: str,
    *,
    key: tuple[float, float],
    version: int,
    outcome: float,
    provenance_id: int,
    query_uncertainty: float = 0.1,
    query_uncertainty_available: bool = True,
    safety_cost: float = 0.0,
    utility: float = 1.0,
    action: float = 0.25,
) -> ExperientialMemoryTransferEvent:
    return ExperientialMemoryTransferEvent(
        event_id=event_id,
        case_id=case_id,
        query_key=key,
        representation_version=version,
        query_uncertainty=query_uncertainty,
        query_uncertainty_available=query_uncertainty_available,
        entry_observation=key,
        entry_key=key,
        entry_action=(action,),
        entry_outcome=(outcome,),
        entry_reward=outcome,
        entry_uncertainty=0.1,
        entry_uncertainty_available=True,
        entry_safety_cost=safety_cost,
        entry_safety_cost_available=True,
        entry_reliability=1.0,
        entry_utility=utility,
        entry_utility_available=True,
        entry_representation_version=version,
        entry_valid=True,
        entry_age=0,
        entry_provenance_id=provenance_id,
        entry_source_id=0,
        expected_outcome=(outcome,),
    )


def default_experiential_memory_transfer_protocol() -> ExperientialMemoryTransferProtocol:
    """Return the fixed A/B/A trace used for bounded mechanism diagnostics."""
    first_a = (
        _event(
            "a1-stale-anchor",
            "stale-anchor",
            key=(0.0, 1.0),
            version=3,
            outcome=0.5,
            provenance_id=100,
            utility=100.0,
            action=0.1,
        ),
        _event(
            "a1-left-train",
            "left-train",
            key=(1.0, 0.0),
            version=1,
            outcome=1.0,
            provenance_id=101,
            action=0.2,
        ),
        _event(
            "a1-left-repeat",
            "left-repeat",
            key=(1.0, 0.0),
            version=1,
            outcome=1.0,
            provenance_id=102,
            utility=2.0,
            action=0.3,
        ),
        _event(
            "a1-uncertainty-missing",
            "uncertainty-missing",
            key=(1.0, 0.0),
            version=1,
            outcome=1.0,
            provenance_id=103,
            query_uncertainty=0.0,
            query_uncertainty_available=False,
            action=0.4,
        ),
    )
    interference_b = (
        _event(
            "b-negative-recall",
            "negative-recall",
            key=(1.0, 0.0),
            version=1,
            outcome=-1.0,
            provenance_id=200,
            action=0.8,
        ),
        _event(
            "b-version-shift",
            "version-shift",
            key=(1.0, 0.0),
            version=2,
            outcome=-1.0,
            provenance_id=201,
            safety_cost=1.0,
            action=0.9,
        ),
        _event(
            "b-unsafe-neighbor",
            "unsafe-neighbor",
            key=(1.0, 0.0),
            version=2,
            outcome=-1.0,
            provenance_id=202,
            action=1.0,
        ),
    )
    returned_a = tuple(
        dataclasses.replace(
            event,
            event_id=event.event_id.replace("a1-", "a2-"),
            entry_provenance_id=300 + index,
        )
        for index, event in enumerate(first_a)
    )
    return ExperientialMemoryTransferProtocol(
        protocol_id="experiential-memory-aba-v1",
        events=(*first_a, *interference_b, *returned_a),
        phases=(
            ExperientialMemoryTransferPhase("first-a", "context-a", len(first_a)),
            ExperientialMemoryTransferPhase(
                "interference-b", "context-b", len(interference_b)
            ),
            ExperientialMemoryTransferPhase("return-a", "context-a", len(returned_a)),
        ),
    )


def _validate_protocol_dimensions(
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
) -> None:
    memory_config = config.memory_config
    if len(protocol.events) > config.max_events:
        raise ValueError("transfer protocol event count exceeds max_events")
    if len(protocol.phases) > config.max_phases:
        raise ValueError("transfer protocol phase count exceeds max_phases")
    for event in protocol.events:
        dimensions = {
            "query_key": (len(event.query_key), memory_config.key_dim),
            "entry_observation": (
                len(event.entry_observation),
                memory_config.observation_dim,
            ),
            "entry_key": (len(event.entry_key), memory_config.key_dim),
            "entry_action": (len(event.entry_action), memory_config.action_dim),
            "entry_outcome": (len(event.entry_outcome), memory_config.outcome_dim),
            "expected_outcome": (
                len(event.expected_outcome),
                memory_config.outcome_dim,
            ),
        }
        for name, (actual, expected) in dimensions.items():
            if actual != expected:
                raise ValueError(
                    f"event {event.event_id} {name} dimension {actual} != {expected}"
                )


def _state_dynamic_valid(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
) -> bool:
    retrieval = memory.query(
        state,
        jnp.zeros((memory.config.key_dim,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    return bool(jax.device_get(retrieval.state_valid))


def _accounting_config(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
) -> dict[str, object]:
    accounting = memory.accounting(state)
    return {
        "active_entries": int(jax.device_get(accounting.active_entries)),
        "capacity_entries": int(jax.device_get(accounting.capacity_entries)),
        "slot_bytes": int(jax.device_get(accounting.slot_bytes)),
        "persistent_bytes": int(jax.device_get(accounting.persistent_bytes)),
        "queries": int(jax.device_get(accounting.queries)),
        "accepted_queries": int(jax.device_get(accounting.accepted_queries)),
        "writes": int(jax.device_get(accounting.writes)),
        "rejected_writes": int(jax.device_get(accounting.rejected_writes)),
        "evictions": int(jax.device_get(accounting.evictions)),
    }


def _snapshot_descriptor(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
) -> dict[str, object]:
    if not isinstance(memory, ExperientialMemory):
        raise TypeError("memory must be ExperientialMemory")
    if not isinstance(state, ExperientialMemoryState):
        raise TypeError("state must be ExperientialMemoryState")
    state_bytes = _tree_nbytes(state)
    if state_bytes > MAX_ABSOLUTE_SNAPSHOT_BYTES:
        raise ValueError("experiential-memory snapshot exceeds the hard byte ceiling")
    if not _state_dynamic_valid(memory, state):
        raise ValueError("experiential-memory snapshot state is invalid")
    memory_config = memory.to_config()
    accounting = _accounting_config(memory, state)
    return {
        "memory_config": memory_config,
        "memory_config_sha256": _canonical_sha256(memory_config),
        "state_sha256": frozen_experiential_memory_state_sha256(state),
        "state_bytes": state_bytes,
        "accounting": accounting,
        "accounting_sha256": _canonical_sha256(accounting),
        "empty_state": frozen_experiential_memory_state_sha256(state)
        == frozen_experiential_memory_state_sha256(memory.init()),
    }


def _validate_evaluation_inputs(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
) -> dict[str, object]:
    if not isinstance(config, ExperientialMemoryTransferConfig):
        raise TypeError("config must be ExperientialMemoryTransferConfig")
    if not isinstance(protocol, ExperientialMemoryTransferProtocol):
        raise TypeError("protocol must be ExperientialMemoryTransferProtocol")
    if not _strict_json_equal(memory.config.to_config(), config.memory_config.to_config()):
        raise ValueError("memory construction does not match transfer config")
    _validate_protocol_dimensions(config, protocol)
    descriptor = _snapshot_descriptor(memory, state)
    if descriptor["empty_state"] is not True:
        raise ValueError("v1 transfer evaluation requires an exact empty memory snapshot")
    state_bytes = cast(int, descriptor["state_bytes"])
    if state_bytes > config.max_initial_snapshot_bytes:
        raise ValueError("initial memory snapshot exceeds max_initial_snapshot_bytes")
    return descriptor


def _stack_entries(
    events: Sequence[ExperientialMemoryTransferEvent],
) -> ExperientialMemoryEntry:
    entries = [event.to_entry() for event in events]
    return cast(
        ExperientialMemoryEntry,
        jax.tree.map(lambda *leaves: jnp.stack(leaves), *entries),
    )


def _compiled_protocol_scan(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    query_keys: jax.Array,
    versions: jax.Array,
    uncertainties: jax.Array,
    uncertainty_available: jax.Array,
    entries: ExperientialMemoryEntry,
) -> tuple[ExperientialMemoryState, ExperientialMemoryStepResult]:
    def body(
        current: ExperientialMemoryState,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, ExperientialMemoryEntry],
    ) -> tuple[ExperientialMemoryState, ExperientialMemoryStepResult]:
        query_key, version, uncertainty, available, entry = inputs
        result = memory.step(
            current,
            query_key,
            version,
            uncertainty,
            available,
            entry,
        )
        return result.state, result

    return cast(
        tuple[ExperientialMemoryState, ExperientialMemoryStepResult],
        jax.lax.scan(
            body,
            state,
            (query_keys, versions, uncertainties, uncertainty_available, entries),
        ),
    )


def _trees_exactly_equal(left: object, right: object) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(
            np.asarray(jax.device_get(left_leaf)),
            np.asarray(jax.device_get(right_leaf)),
        )
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _execute_protocol(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    protocol: ExperientialMemoryTransferProtocol,
) -> tuple[
    ExperientialMemoryState,
    list[ExperientialMemoryStepResult],
    dict[str, object],
]:
    before = frozen_experiential_memory_state_sha256(state)
    eager_state = state
    eager_results: list[ExperientialMemoryStepResult] = []
    for event in protocol.events:
        result = memory.step(
            eager_state,
            jnp.asarray(event.query_key, dtype=jnp.float32),
            jnp.asarray(event.representation_version, dtype=jnp.int32),
            jnp.asarray(event.query_uncertainty, dtype=jnp.float32),
            jnp.asarray(event.query_uncertainty_available, dtype=jnp.bool_),
            event.to_entry(),
        )
        eager_results.append(result)
        eager_state = result.state

    query_keys = jnp.asarray([event.query_key for event in protocol.events], dtype=jnp.float32)
    versions = jnp.asarray(
        [event.representation_version for event in protocol.events], dtype=jnp.int32
    )
    uncertainties = jnp.asarray(
        [event.query_uncertainty for event in protocol.events], dtype=jnp.float32
    )
    uncertainty_available = jnp.asarray(
        [event.query_uncertainty_available for event in protocol.events],
        dtype=jnp.bool_,
    )
    entries = _stack_entries(protocol.events)
    compiled = jax.jit(
        lambda initial: _compiled_protocol_scan(
            memory,
            initial,
            query_keys,
            versions,
            uncertainties,
            uncertainty_available,
            entries,
        )
    )
    compiled_state, compiled_results = compiled(state)
    parity = _trees_exactly_equal(eager_state, compiled_state)
    for index, eager_result in enumerate(eager_results):
        compiled_result = cast(
            ExperientialMemoryStepResult,
            jax.tree.map(lambda leaf: leaf[index], compiled_results),
        )
        parity = parity and _trees_exactly_equal(eager_result, compiled_result)
    if not parity:
        raise RuntimeError("eager and isolated compiled memory kernels diverged")
    if frozen_experiential_memory_state_sha256(state) != before:
        raise RuntimeError("protocol execution mutated the supplied memory snapshot")
    return (
        eager_state,
        eager_results,
        {
            "compiled_kernel_parity_checked": True,
            "compiled_kernel_parity_exact": True,
            "external_snapshot_mutations": 0,
        },
    )


def _event_annotations(
    protocol: ExperientialMemoryTransferProtocol,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    index = 0
    occurrences: dict[str, int] = {}
    for phase_index, (phase, events) in enumerate(
        zip(protocol.phases, protocol.phase_events(), strict=True)
    ):
        occurrence = occurrences.get(phase.evaluator_regime_id, 0)
        occurrences[phase.evaluator_regime_id] = occurrence + 1
        for phase_event_index, event in enumerate(events):
            result.append(
                {
                    "event_index": index,
                    "event_id": event.event_id,
                    "case_id": event.case_id,
                    "phase_index": phase_index,
                    "phase_event_index": phase_event_index,
                    "phase_id": phase.phase_id,
                    "evaluator_regime_id": phase.evaluator_regime_id,
                    "evaluator_regime_occurrence_index": occurrence,
                    "expected_outcome": list(event.expected_outcome),
                    "evaluator_owned": True,
                    "learner_visible": False,
                }
            )
            index += 1
    return result


def _array_floats(value: object) -> list[float]:
    array = np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1)
    result = [float(item) for item in array]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("non-finite kernel value cannot enter transfer report")
    return result


def _scalar_float(value: object) -> float:
    result = float(np.asarray(jax.device_get(value)).reshape(()))
    if not math.isfinite(result):
        raise ValueError("non-finite kernel scalar cannot enter transfer report")
    return result


def _scalar_int(value: object) -> int:
    return int(np.asarray(jax.device_get(value)).reshape(()))


def _scalar_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)).reshape(()))


def _mean_squared_error(prediction: Sequence[float], target: Sequence[float]) -> float:
    if len(prediction) != len(target) or not prediction:
        raise ValueError("prediction and target dimensions must match")
    return math.fsum((left - right) ** 2 for left, right in zip(prediction, target)) / len(
        target
    )


def _provenance_catalog(
    protocol: ExperientialMemoryTransferProtocol,
    annotations: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    return {
        event.entry_provenance_id: {
            "source_event_id": event.event_id,
            "source_case_id": event.case_id,
            "source_evaluator_regime_id": annotation["evaluator_regime_id"],
            "source_expected_outcome": list(event.expected_outcome),
        }
        for event, annotation in zip(protocol.events, annotations, strict=True)
    }


def _primary_abstention_reason(
    retrieval: Mapping[str, object],
    *,
    pre_active_entries: int,
) -> str | None:
    if retrieval["accepted"] is True:
        return None
    if pre_active_entries == 0:
        return "empty-memory"
    if retrieval["version_compatible"] is False:
        return "representation-version-mismatch"
    if retrieval["freshness_ok"] is False:
        return "stale"
    if (
        retrieval["query_valid"] is False
        or retrieval["uncertainty_available"] is False
        or retrieval["uncertainty_ok"] is False
    ):
        return "uncertain-or-unavailable"
    if retrieval["safety_cost_available"] is False or retrieval["safety_ok"] is False:
        return "unsafe-or-unavailable"
    if retrieval["has_neighbors"] is False:
        return "insufficient-eligible-neighbors"
    if retrieval["state_valid"] is False:
        return "invalid-state"
    return "aggregate-gate"


def _raw_trace(
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
    results: Sequence[ExperientialMemoryStepResult],
) -> list[dict[str, object]]:
    annotations = _event_annotations(protocol)
    catalog = _provenance_catalog(protocol, annotations)
    trace: list[dict[str, object]] = []
    previous_accounting: dict[str, object] = {
        "steps": 0,
        "queries": 0,
        "accepted_queries": 0,
        "writes": 0,
        "rejected_writes": 0,
        "evictions": 0,
        "active_entries": 0,
    }
    for event, annotation, result in zip(
        protocol.events, annotations, results, strict=True
    ):
        retrieval = result.retrieval
        accepted = _scalar_bool(retrieval.accepted)
        neighbor_indices = [int(value) for value in _array_floats(retrieval.neighbor_indices)]
        neighbor_mask = [
            bool(value)
            for value in np.asarray(jax.device_get(retrieval.neighbor_mask))
        ]
        neighbor_weights = _array_floats(retrieval.neighbor_weights)
        neighbor_provenance_ids = [
            int(value) for value in _array_floats(retrieval.neighbor_provenance_ids)
        ]
        retrieval_payload: dict[str, object] = {
            "accepted": accepted,
            "observation": _array_floats(retrieval.observation),
            "action": _array_floats(retrieval.action),
            "outcome": _array_floats(retrieval.outcome),
            "reward": _scalar_float(retrieval.reward),
            "uncertainty": _scalar_float(retrieval.uncertainty),
            "safety_cost": _scalar_float(retrieval.safety_cost),
            "effective_reliability": _scalar_float(retrieval.effective_reliability),
            "neighbor_indices": neighbor_indices,
            "neighbor_mask": neighbor_mask,
            "neighbor_weights": neighbor_weights,
            "neighbor_similarities": _array_floats(retrieval.neighbor_similarities),
            "neighbor_reliabilities": _array_floats(retrieval.neighbor_reliabilities),
            "neighbor_ages": [
                int(value) for value in _array_floats(retrieval.neighbor_ages)
            ],
            "neighbor_provenance_ids": neighbor_provenance_ids,
            "state_valid": _scalar_bool(retrieval.state_valid),
            "query_valid": _scalar_bool(retrieval.query_valid),
            "version_compatible": _scalar_bool(retrieval.version_compatible),
            "freshness_ok": _scalar_bool(retrieval.freshness_ok),
            "uncertainty_available": _scalar_bool(retrieval.uncertainty_available),
            "safety_cost_available": _scalar_bool(retrieval.safety_cost_available),
            "uncertainty_ok": _scalar_bool(retrieval.uncertainty_ok),
            "safety_ok": _scalar_bool(retrieval.safety_ok),
            "has_neighbors": _scalar_bool(retrieval.has_neighbors),
        }
        expected = list(event.expected_outcome)
        reference_prediction = list(config.reference_prediction)
        memory_prediction = (
            cast(list[float], retrieval_payload["outcome"])
            if accepted
            else reference_prediction
        )
        reference_action = list(config.reference_action)
        memory_action = (
            cast(list[float], retrieval_payload["action"])
            if accepted
            else reference_action
        )
        memory_error = _mean_squared_error(memory_prediction, expected)
        reference_error = _mean_squared_error(reference_prediction, expected)
        excess = max(0.0, memory_error - reference_error) if accepted else 0.0

        neighbor_sources: list[dict[str, object] | None] = []
        correct_weight = 0.0
        same_regime_weight = 0.0
        for provenance_id, used, weight in zip(
            neighbor_provenance_ids, neighbor_mask, neighbor_weights, strict=True
        ):
            source = catalog.get(provenance_id) if used else None
            neighbor_sources.append(source)
            if source is None:
                continue
            if source["source_evaluator_regime_id"] == annotation["evaluator_regime_id"]:
                same_regime_weight += weight
                if source["source_expected_outcome"] == expected:
                    correct_weight += weight

        evicted_id = _scalar_int(result.evicted_provenance_id)
        retrieval_error: float | None = None
        if accepted:
            retrieval_error = _mean_squared_error(
                cast(list[float], retrieval_payload["outcome"]), expected
            )
        retrieval_payload["primary_abstention_reason"] = _primary_abstention_reason(
            retrieval_payload,
            pre_active_entries=cast(int, previous_accounting["active_entries"]),
        )
        post_step_accounting: dict[str, object] = {
            "steps": _scalar_int(result.state.step_count),
            "queries": _scalar_int(result.state.query_count),
            "accepted_queries": _scalar_int(result.state.accepted_query_count),
            "writes": _scalar_int(result.state.write_count),
            "rejected_writes": _scalar_int(result.state.rejected_write_count),
            "evictions": _scalar_int(result.state.eviction_count),
            "active_entries": _scalar_int(result.state.active_count),
        }
        record = {
            "event_index": annotation["event_index"],
            "event_id": event.event_id,
            "query_before_write": True,
            "learner_input_sha256": _canonical_sha256(event.learner_input_config()),
            "evaluator_annotation": dict(annotation),
            "pre_step_accounting": dict(previous_accounting),
            "retrieval": retrieval_payload,
            "retrieval_provenance": {
                "neighbor_sources": neighbor_sources,
                "same_evaluator_regime_neighbor_weight": same_regime_weight,
                "correct_neighbor_weight": correct_weight,
                "retrieval_mean_squared_error": retrieval_error,
            },
            "reference_prediction": {
                "outcome": reference_prediction,
                "action": reference_action,
                "mean_squared_error": reference_error,
                "persistent_state_bytes": 0,
            },
            "memory_prediction": {
                "outcome": memory_prediction,
                "action": memory_action,
                "used_retrieval": accepted,
                "mean_squared_error": memory_error,
            },
            "negative_transfer": {
                "occurred": accepted and excess > 0.0,
                "excess_squared_error": excess,
            },
            "write_after_query": {
                "attempted": True,
                "wrote": _scalar_bool(result.wrote),
                "slot": _scalar_int(result.slot),
                "entry_provenance_id": event.entry_provenance_id,
                "evicted": _scalar_bool(result.evicted),
                "evicted_provenance_id": evicted_id,
                "evicted_provenance": catalog.get(evicted_id),
            },
            "post_step_accounting": post_step_accounting,
        }
        trace.append(record)
        previous_accounting = post_step_accounting
    return trace


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def reconstruct_experiential_memory_transfer_summary(
    trace: Sequence[Mapping[str, object]],
    protocol: ExperientialMemoryTransferProtocol,
) -> dict[str, object]:
    """Reconstruct every descriptive metric from raw query-before-write records."""
    if len(trace) != len(protocol.events):
        raise ValueError("raw trace length does not match transfer protocol")
    annotations = _event_annotations(protocol)
    for record, annotation in zip(trace, annotations, strict=True):
        if record.get("event_id") != annotation["event_id"]:
            raise ValueError("raw trace event order does not match transfer protocol")

    phase_descriptions: list[dict[str, object]] = []
    start = 0
    occurrences: dict[str, int] = {}
    for phase in protocol.phases:
        end = start + phase.event_count
        records = list(trace[start:end])
        occurrence = occurrences.get(phase.evaluator_regime_id, 0)
        occurrences[phase.evaluator_regime_id] = occurrence + 1
        memory_errors = [
            cast(float, _mapping(record["memory_prediction"], name="memory_prediction")[
                "mean_squared_error"
            ])
            for record in records
        ]
        reference_errors = [
            cast(float, _mapping(record["reference_prediction"], name="reference_prediction")[
                "mean_squared_error"
            ])
            for record in records
        ]
        phase_descriptions.append(
            {
                "phase_id": phase.phase_id,
                "evaluator_regime_id": phase.evaluator_regime_id,
                "evaluator_regime_occurrence_index": occurrence,
                "event_count": len(records),
                "memory_mean_squared_error": _mean(memory_errors),
                "reference_mean_squared_error": _mean(reference_errors),
                "memory_minus_reference_mean_squared_error": _mean(memory_errors)
                - _mean(reference_errors),
                "entry_memory_squared_error": memory_errors[0],
                "entry_reference_squared_error": reference_errors[0],
                "interpretation": "descriptive-signed-difference-with-no-threshold",
            }
        )
        start = end

    forward_descriptions: list[dict[str, object]] = []
    first_by_regime: dict[str, dict[str, object]] = {}
    return_descriptions: list[dict[str, object]] = []
    for phase_description in phase_descriptions:
        regime = cast(str, phase_description["evaluator_regime_id"])
        if regime not in first_by_regime:
            first_by_regime[regime] = phase_description
            forward_descriptions.append(
                {
                    "evaluator_regime_id": regime,
                    "first_phase_id": phase_description["phase_id"],
                    "memory_minus_reference_mean_squared_error": phase_description[
                        "memory_minus_reference_mean_squared_error"
                    ],
                    "entry_memory_minus_reference_squared_error": cast(
                        float, phase_description["entry_memory_squared_error"]
                    )
                    - cast(float, phase_description["entry_reference_squared_error"]),
                    "assessment": "not-assessed",
                }
            )
        else:
            first = first_by_regime[regime]
            return_descriptions.append(
                {
                    "evaluator_regime_id": regime,
                    "first_phase_id": first["phase_id"],
                    "return_phase_id": phase_description["phase_id"],
                    "memory_mean_squared_error_delta_vs_first_occurrence": cast(
                        float, phase_description["memory_mean_squared_error"]
                    )
                    - cast(float, first["memory_mean_squared_error"]),
                    "reference_mean_squared_error_delta_vs_first_occurrence": cast(
                        float, phase_description["reference_mean_squared_error"]
                    )
                    - cast(float, first["reference_mean_squared_error"]),
                    "entry_memory_squared_error_delta_vs_first_occurrence": cast(
                        float, phase_description["entry_memory_squared_error"]
                    )
                    - cast(float, first["entry_memory_squared_error"]),
                    "assessment": "not-assessed",
                }
            )

    accepted_records = [
        record
        for record in trace
        if _mapping(record["retrieval"], name="retrieval")["accepted"] is True
    ]
    correct_weights = [
        cast(
            float,
            _mapping(record["retrieval_provenance"], name="retrieval_provenance")[
                "correct_neighbor_weight"
            ],
        )
        for record in accepted_records
    ]
    retrieval_errors = [
        cast(
            float,
            _mapping(record["retrieval_provenance"], name="retrieval_provenance")[
                "retrieval_mean_squared_error"
            ],
        )
        for record in accepted_records
    ]
    negative_records = [
        record
        for record in trace
        if _mapping(record["negative_transfer"], name="negative_transfer")[
            "occurred"
        ]
        is True
    ]
    negative_magnitudes = [
        cast(
            float,
            _mapping(record["negative_transfer"], name="negative_transfer")[
                "excess_squared_error"
            ],
        )
        for record in negative_records
    ]
    reasons = [
        _mapping(record["retrieval"], name="retrieval")["primary_abstention_reason"]
        for record in trace
    ]
    memory_outcomes = [
        tuple(
            cast(
                list[float],
                _mapping(record["memory_prediction"], name="memory_prediction")["outcome"],
            )
        )
        for record in trace
    ]
    memory_actions = [
        tuple(
            cast(
                list[float],
                _mapping(record["memory_prediction"], name="memory_prediction")["action"],
            )
        )
        for record in trace
    ]
    expected_outcomes = [event.expected_outcome for event in protocol.events]
    eviction_records = [
        {
            "event_id": record["event_id"],
            "evicted_provenance_id": _mapping(
                record["write_after_query"], name="write_after_query"
            )["evicted_provenance_id"],
            "evicted_provenance": _mapping(
                record["write_after_query"], name="write_after_query"
            )["evicted_provenance"],
        }
        for record in trace
        if _mapping(record["write_after_query"], name="write_after_query")[
            "evicted"
        ]
        is True
    ]
    return {
        "assessment_status": "not-assessed",
        "metric_direction": "lower_squared_error_is_better-description-only",
        "matched_reference": "stateless-fixed-outcome-and-action",
        "phase_descriptions": phase_descriptions,
        "forward_transfer_descriptions": forward_descriptions,
        "return_transfer_descriptions": return_descriptions,
        "retrieval_diagnostics": {
            "query_count": len(trace),
            "accepted_count": len(accepted_records),
            "abstained_count": len(trace) - len(accepted_records),
            "weighted_retrieval_precision_when_accepted": _mean(correct_weights),
            "mean_correct_neighbor_weight_when_accepted": _mean(correct_weights),
            "mean_retrieval_squared_error_when_accepted": _mean(retrieval_errors),
            "retrieval_precision_definition": (
                "neighbor weight from the same evaluator regime with the same expected outcome"
            ),
        },
        "abstention_diagnostics": {
            "empty_memory_count": reasons.count("empty-memory"),
            "representation_version_mismatch_count": reasons.count(
                "representation-version-mismatch"
            ),
            "stale_count": reasons.count("stale"),
            "unsafe_count": reasons.count("unsafe-or-unavailable"),
            "uncertain_or_unavailable_count": reasons.count(
                "uncertain-or-unavailable"
            ),
            "insufficient_eligible_neighbor_count": reasons.count(
                "insufficient-eligible-neighbors"
            ),
            "primary_reason_counts_sum": sum(reason is not None for reason in reasons),
        },
        "negative_transfer": {
            "count": len(negative_records),
            "total_excess_squared_error": math.fsum(negative_magnitudes),
            "mean_excess_squared_error": _mean(negative_magnitudes),
            "maximum_excess_squared_error": max(negative_magnitudes, default=0.0),
            "event_ids": [record["event_id"] for record in negative_records],
        },
        "eviction_provenance": {
            "eviction_count": len(eviction_records),
            "events": eviction_records,
        },
        "loophole_diagnostics": {
            "always_abstained": not accepted_records,
            "always_retrieved": len(accepted_records) == len(trace),
            "memory_prediction_constant": len(set(memory_outcomes)) == 1,
            "memory_action_policy_constant": len(set(memory_actions)) == 1,
            "reference_prediction_constant_by_design": True,
            "reference_action_policy_constant_by_design": True,
            "expected_outcome_constant": len(set(expected_outcomes)) == 1,
            "distinct_memory_predictions": len(set(memory_outcomes)),
            "distinct_memory_actions": len(set(memory_actions)),
            "distinct_expected_outcomes": len(set(expected_outcomes)),
        },
        "claims": {
            "transfer_established": False,
            "retention_established": False,
            "efficacy_established": False,
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "sota_claimed": False,
        },
    }


def _resource_accounting(
    *,
    memory: ExperientialMemory,
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
    initial: Mapping[str, object],
    final: Mapping[str, object],
    sources: Mapping[str, str],
    kernel_audit: Mapping[str, object],
    canonical_report_bytes: int,
) -> dict[str, object]:
    final_accounting = _mapping(final.get("accounting"), name="final accounting")
    event_count = len(protocol.events)
    return {
        "initial_snapshot_state_bytes": cast(int, initial["state_bytes"]),
        "initial_snapshot_state_byte_limit": config.max_initial_snapshot_bytes,
        "bounded_event_capacity": config.max_events,
        "recorded_event_count": event_count,
        "bounded_phase_capacity": config.max_phases,
        "recorded_phase_count": len(protocol.phases),
        "memory_query_opportunities": event_count,
        "memory_write_opportunities": event_count,
        "reference_query_opportunities": event_count,
        "reference_write_opportunities": event_count,
        "matched_event_query_write_opportunity_budgets": True,
        "memory_executed_queries": _nonnegative_int(
            final_accounting.get("queries"), name="final queries"
        ),
        "memory_executed_write_attempts": _nonnegative_int(
            final_accounting.get("writes"), name="final writes"
        )
        + _nonnegative_int(
            final_accounting.get("rejected_writes"), name="final rejected writes"
        ),
        "reference_physical_memory_queries": 0,
        "reference_physical_memory_writes": 0,
        "reference_persistent_state_bytes": 0,
        "memory_active_entries": _nonnegative_int(
            final_accounting.get("active_entries"), name="final active entries"
        ),
        "memory_capacity_entries": memory.config.capacity,
        "memory_slot_bytes": memory.slot_bytes,
        "memory_persistent_state_bytes": memory.persistent_bytes,
        "memory_accepted_queries": _nonnegative_int(
            final_accounting.get("accepted_queries"), name="final accepted queries"
        ),
        "memory_writes": _nonnegative_int(
            final_accounting.get("writes"), name="final writes"
        ),
        "memory_rejected_writes": _nonnegative_int(
            final_accounting.get("rejected_writes"), name="final rejected writes"
        ),
        "memory_evictions": _nonnegative_int(
            final_accounting.get("evictions"), name="final evictions"
        ),
        "source_manifest_entries": len(sources),
        "canonical_report_bytes": canonical_report_bytes,
        "canonical_report_byte_limit": config.max_report_bytes,
        "compiled_kernel_parity_checked": kernel_audit[
            "compiled_kernel_parity_checked"
        ],
        "compiled_kernel_parity_exact": kernel_audit["compiled_kernel_parity_exact"],
        "external_snapshot_mutations": kernel_audit["external_snapshot_mutations"],
        "learner_visible_evaluator_label_reads": 0,
        "regime_identifier_reads_by_memory": 0,
        "performance_threshold_comparisons": 0,
    }


def _assemble_report(
    *,
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
    sources: Mapping[str, str],
    initial: Mapping[str, object],
    final: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    config_payload = config.to_config()
    protocol_payload = protocol.to_config()
    trace_payload = [dict(record) for record in trace]
    hashes = {
        "config_sha256": _canonical_sha256(config_payload),
        "protocol_sha256": _canonical_sha256(protocol_payload),
        "source_manifest_sha256": _canonical_sha256(sources),
        "initial_snapshot_sha256": _canonical_sha256(initial),
        "final_isolated_memory_state_sha256": _canonical_sha256(final),
        "raw_query_before_write_trace_sha256": _canonical_sha256(trace_payload),
        "summary_sha256": _canonical_sha256(summary),
        "resource_accounting_sha256": _canonical_sha256(resources),
    }
    payload: dict[str, object] = {
        "development_only": True,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "transfer_claimed": False,
        "retention_claimed": False,
        "efficacy_claimed": False,
        "sota_claimed": False,
        "evaluation_protocol": "matched-stateless-reference-query-before-write-aba",
        "config": config_payload,
        "source_sha256": dict(sources),
        "protocol": protocol_payload,
        "initial_snapshot": dict(initial),
        "final_isolated_memory_state": dict(final),
        "raw_query_before_write_trace": trace_payload,
        "summary": dict(summary),
        "resource_accounting": dict(resources),
        "hashes": hashes,
        "limitations": list(_LIMITATIONS),
    }
    return {
        "schema": EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def build_experiential_memory_transfer_report(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Run the matched diagnostic on an isolated copy of an empty snapshot."""
    initial = _validate_evaluation_inputs(memory, state, config, protocol)
    before = frozen_experiential_memory_state_sha256(state)
    final_state, results, kernel_audit = _execute_protocol(memory, state, protocol)
    if frozen_experiential_memory_state_sha256(state) != before:
        raise RuntimeError("transfer evaluation mutated the supplied memory snapshot")
    final = _snapshot_descriptor(memory, final_state)
    trace = _raw_trace(config, protocol, results)
    summary = reconstruct_experiential_memory_transfer_summary(trace, protocol)
    sources = experiential_memory_transfer_source_snapshot(root)

    report_size = 0
    report: dict[str, object] | None = None
    for _ in range(16):
        resources = _resource_accounting(
            memory=memory,
            config=config,
            protocol=protocol,
            initial=initial,
            final=final,
            sources=sources,
            kernel_audit=kernel_audit,
            canonical_report_bytes=report_size,
        )
        report = _assemble_report(
            config=config,
            protocol=protocol,
            sources=sources,
            initial=initial,
            final=final,
            trace=trace,
            summary=summary,
            resources=resources,
        )
        measured = len(_canonical_json_bytes(report)) + 1
        if measured == report_size:
            break
        report_size = measured
    else:
        raise RuntimeError("canonical experiential-memory transfer report size did not converge")
    assert report is not None
    if report_size > config.max_report_bytes:
        raise ValueError("canonical report exceeds configured report byte bound")
    validation = validate_experiential_memory_transfer_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "internal experiential-memory transfer report validation failed: "
            + "; ".join(validation.errors)
        )
    return report


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryTransferValidation:
    """Fail-closed artifact validation with no performance verdict."""

    valid: bool
    assessment_status: str
    errors: tuple[str, ...]


def _validate_fixed_payload(payload: Mapping[str, object], errors: list[str]) -> None:
    expected_fields = {
        "development_only",
        "assessment_status",
        "scientific_promotion_allowed",
        "performance_thresholds_applied",
        "transfer_claimed",
        "retention_claimed",
        "efficacy_claimed",
        "sota_claimed",
        "evaluation_protocol",
        "config",
        "source_sha256",
        "protocol",
        "initial_snapshot",
        "final_isolated_memory_state",
        "raw_query_before_write_trace",
        "summary",
        "resource_accounting",
        "hashes",
        "limitations",
    }
    if set(payload) != expected_fields:
        errors.append("experiential-memory transfer payload fields do not match v1")
    fixed = {
        "development_only": True,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "transfer_claimed": False,
        "retention_claimed": False,
        "efficacy_claimed": False,
        "sota_claimed": False,
        "evaluation_protocol": "matched-stateless-reference-query-before-write-aba",
        "limitations": list(_LIMITATIONS),
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(payload.get(name), expected):
            errors.append(f"experiential-memory transfer {name} is not the fixed v1 value")


def _replay_report_components(
    config: ExperientialMemoryTransferConfig,
    protocol: ExperientialMemoryTransferProtocol,
) -> tuple[
    ExperientialMemory,
    ExperientialMemoryState,
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    memory = ExperientialMemory(config.memory_config)
    state = memory.init()
    initial = _validate_evaluation_inputs(memory, state, config, protocol)
    final_state, results, audit = _execute_protocol(memory, state, protocol)
    final = _snapshot_descriptor(memory, final_state)
    trace = _raw_trace(config, protocol, results)
    summary = reconstruct_experiential_memory_transfer_summary(trace, protocol)
    return memory, state, initial, final, trace, summary, audit


def validate_experiential_memory_transfer_report(
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    memory: ExperientialMemory | None = None,
    state: ExperientialMemoryState | None = None,
    protocol: ExperientialMemoryTransferProtocol | None = None,
) -> ExperientialMemoryTransferValidation:
    """Reject schema, source, hash, reconstruction, bounds, or live-replay drift."""
    errors: list[str] = []
    if set(report) != {"schema", "payload", "payload_sha256"}:
        errors.append("experiential-memory transfer report top-level fields do not match v1")
    if report.get("schema") != EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA:
        errors.append("experiential-memory transfer report schema is unsupported")
    payload_value = report.get("payload")
    if not isinstance(payload_value, Mapping):
        errors.append("experiential-memory transfer payload must be an object")
        payload: Mapping[str, object] = {}
    else:
        payload = cast(Mapping[str, object], payload_value)
    _validate_fixed_payload(payload, errors)
    try:
        if report.get("payload_sha256") != _canonical_sha256(payload):
            errors.append("experiential-memory transfer payload digest does not match")
    except (TypeError, ValueError) as error:
        errors.append(f"experiential-memory transfer payload is noncanonical: {error}")

    reconstructed_config: ExperientialMemoryTransferConfig | None = None
    config_value = payload.get("config")
    if not isinstance(config_value, Mapping):
        errors.append("experiential-memory transfer config must be an object")
    else:
        try:
            reconstructed_config = ExperientialMemoryTransferConfig.from_config(config_value)
        except (TypeError, ValueError) as error:
            errors.append(f"experiential-memory transfer config is invalid: {error}")

    reconstructed_protocol: ExperientialMemoryTransferProtocol | None = None
    protocol_value = payload.get("protocol")
    if not isinstance(protocol_value, Mapping):
        errors.append("experiential-memory transfer protocol must be an object")
    else:
        try:
            reconstructed_protocol = ExperientialMemoryTransferProtocol.from_config(
                protocol_value
            )
        except (TypeError, ValueError) as error:
            errors.append(f"experiential-memory transfer protocol is invalid: {error}")

    try:
        current_sources = experiential_memory_transfer_source_snapshot(root)
    except OSError as error:
        errors.append(f"cannot hash experiential-memory transfer sources: {error}")
        current_sources = {}
    source_value = payload.get("source_sha256")
    if not _strict_json_equal(source_value, current_sources):
        errors.append("experiential-memory transfer source hashes do not match current sources")

    replay_memory: ExperientialMemory | None = None
    replay_state: ExperientialMemoryState | None = None
    replay_initial: dict[str, object] | None = None
    replay_final: dict[str, object] | None = None
    replay_trace: list[dict[str, object]] | None = None
    replay_summary: dict[str, object] | None = None
    replay_audit: dict[str, object] | None = None
    if reconstructed_config is not None and reconstructed_protocol is not None:
        try:
            (
                replay_memory,
                replay_state,
                replay_initial,
                replay_final,
                replay_trace,
                replay_summary,
                replay_audit,
            ) = _replay_report_components(reconstructed_config, reconstructed_protocol)
            comparisons = {
                "initial snapshot": (payload.get("initial_snapshot"), replay_initial),
                "final isolated memory state": (
                    payload.get("final_isolated_memory_state"),
                    replay_final,
                ),
                "raw query-before-write trace": (
                    payload.get("raw_query_before_write_trace"),
                    replay_trace,
                ),
                "summary": (payload.get("summary"), replay_summary),
            }
            for name, (actual, expected) in comparisons.items():
                if not _strict_json_equal(actual, expected):
                    raise ValueError(f"{name} does not reconstruct exactly")
        except (TypeError, ValueError, RuntimeError) as error:
            errors.append(f"experiential-memory transfer reconstruction failed: {error}")

    resources_value = payload.get("resource_accounting")
    if (
        reconstructed_config is not None
        and reconstructed_protocol is not None
        and replay_memory is not None
        and replay_initial is not None
        and replay_final is not None
        and replay_audit is not None
    ):
        try:
            expected_resources = _resource_accounting(
                memory=replay_memory,
                config=reconstructed_config,
                protocol=reconstructed_protocol,
                initial=replay_initial,
                final=replay_final,
                sources=current_sources,
                kernel_audit=replay_audit,
                canonical_report_bytes=len(_canonical_json_bytes(report)) + 1,
            )
            if not _strict_json_equal(resources_value, expected_resources):
                raise ValueError("resource accounting does not reconstruct exactly")
            if len(_canonical_json_bytes(report)) + 1 > reconstructed_config.max_report_bytes:
                raise ValueError("canonical report exceeds configured report byte bound")
        except (TypeError, ValueError) as error:
            errors.append(f"experiential-memory transfer resources are invalid: {error}")
    elif not isinstance(resources_value, Mapping):
        errors.append("experiential-memory transfer resources must be an object")

    hashes_value = payload.get("hashes")
    hash_inputs = {
        "config_sha256": config_value,
        "protocol_sha256": protocol_value,
        "source_manifest_sha256": source_value,
        "initial_snapshot_sha256": payload.get("initial_snapshot"),
        "final_isolated_memory_state_sha256": payload.get(
            "final_isolated_memory_state"
        ),
        "raw_query_before_write_trace_sha256": payload.get(
            "raw_query_before_write_trace"
        ),
        "summary_sha256": payload.get("summary"),
        "resource_accounting_sha256": resources_value,
    }
    if not isinstance(hashes_value, Mapping) or set(hashes_value) != set(hash_inputs):
        errors.append("experiential-memory transfer component hash fields do not match v1")
    else:
        try:
            expected_hashes = {
                name: _canonical_sha256(value) for name, value in hash_inputs.items()
            }
            if not _strict_json_equal(hashes_value, expected_hashes):
                errors.append("one or more experiential-memory component hashes do not match")
        except (TypeError, ValueError) as error:
            errors.append(f"experiential-memory components are noncanonical: {error}")

    optional = (memory, state, protocol)
    if any(value is not None for value in optional) and not all(
        value is not None for value in optional
    ):
        errors.append("memory, state, and protocol must be supplied together for live replay")
    elif (
        memory is not None
        and state is not None
        and protocol is not None
        and reconstructed_config is not None
        and reconstructed_protocol is not None
    ):
        before = frozen_experiential_memory_state_sha256(state)
        try:
            replay_memory_config = replay_memory.to_config() if replay_memory else None
            if not _strict_json_equal(memory.to_config(), replay_memory_config):
                raise ValueError("supplied memory construction does not match report")
            if not _strict_json_equal(protocol.to_config(), reconstructed_protocol.to_config()):
                raise ValueError("supplied protocol does not match report")
            live = build_experiential_memory_transfer_report(
                memory,
                state,
                reconstructed_config,
                protocol,
                root=root,
            )
            if not _strict_json_equal(live, report):
                raise ValueError("supplied live replay does not reproduce report")
            if frozen_experiential_memory_state_sha256(state) != before:
                raise ValueError("live replay mutated supplied memory snapshot")
        except (TypeError, ValueError, RuntimeError) as error:
            errors.append(f"experiential-memory transfer live replay failed: {error}")

    return ExperientialMemoryTransferValidation(
        valid=not errors,
        assessment_status="not-assessed",
        errors=tuple(errors),
    )


def canonical_experiential_memory_transfer_report_bytes(
    report: Mapping[str, object],
) -> bytes:
    """Return the sole accepted canonical JSON encoding."""
    return _canonical_json_bytes(report) + b"\n"


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_object(data: bytes) -> dict[str, object]:
    parsed = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("experiential-memory transfer report must be a JSON object")
    return cast(dict[str, object], parsed)


def load_experiential_memory_transfer_report(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load exact canonical JSON and require current-source validation."""
    source = Path(path)
    if source.stat().st_size > MAX_ABSOLUTE_REPORT_BYTES:
        raise ValueError("experiential-memory transfer report exceeds the hard byte ceiling")
    data = source.read_bytes()
    report = _strict_json_object(data)
    if data != canonical_experiential_memory_transfer_report_bytes(report):
        raise ValueError("experiential-memory transfer report is not exact canonical JSON")
    validation = validate_experiential_memory_transfer_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "invalid experiential-memory transfer report: " + "; ".join(validation.errors)
        )
    return report


def save_experiential_memory_transfer_report(
    path: str | Path,
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Atomically create one validated canonical report without overwrite."""
    validation = validate_experiential_memory_transfer_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "refusing to save invalid experiential-memory transfer report: "
            + "; ".join(validation.errors)
        )
    expanded = Path(path).expanduser()
    destination = expanded.resolve()
    if os.path.lexists(expanded) or os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite experiential-memory transfer report: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_experiential_memory_transfer_report_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite concurrently created experiential-memory "
                f"transfer report: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def save_experiential_memory_transfer_snapshot_checkpoint(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Create a source-, construction-, resource-, and state-bound checkpoint."""
    destination = Path(path).expanduser()
    if os.path.lexists(destination) or os.path.lexists(destination.resolve()):
        raise FileExistsError(
            f"refusing to overwrite experiential-memory transfer snapshot: {destination}"
        )
    snapshot = _snapshot_descriptor(memory, state)
    sources = experiential_memory_transfer_source_snapshot(root)
    memory_config = memory.to_config()
    save_checkpoint(
        state,
        destination,
        metadata={
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "transfer_claimed": False,
            "retention_claimed": False,
            "sota_claimed": False,
            "memory_config": memory_config,
            "memory_config_sha256": _canonical_sha256(memory_config),
            "snapshot": snapshot,
            "snapshot_sha256": _canonical_sha256(snapshot),
            "source_sha256": sources,
            "source_manifest_sha256": _canonical_sha256(sources),
        },
    )


def load_experiential_memory_transfer_snapshot_checkpoint(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> tuple[ExperientialMemory, ExperientialMemoryState]:
    """Restore only an exact current-source experiential-memory snapshot."""
    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "development_status",
        "assessment_status",
        "scientific_promotion_allowed",
        "performance_thresholds_applied",
        "transfer_claimed",
        "retention_claimed",
        "sota_claimed",
        "memory_config",
        "memory_config_sha256",
        "snapshot",
        "snapshot_sha256",
        "source_sha256",
        "source_manifest_sha256",
    }
    if set(metadata) != expected:
        raise ValueError("experiential-memory transfer snapshot metadata fields do not match v1")
    fixed = {
        "schema": EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "transfer_claimed": False,
        "retention_claimed": False,
        "sota_claimed": False,
    }
    for name, expected_value in fixed.items():
        if not _strict_json_equal(metadata.get(name), expected_value):
            raise ValueError(f"experiential-memory transfer snapshot {name} is invalid")
    sources = experiential_memory_transfer_source_snapshot(root)
    if not _strict_json_equal(metadata.get("source_sha256"), sources) or metadata.get(
        "source_manifest_sha256"
    ) != _canonical_sha256(sources):
        raise ValueError("experiential-memory transfer snapshot source hashes do not match")
    memory_config = _mapping(metadata.get("memory_config"), name="memory_config")
    if metadata.get("memory_config_sha256") != _canonical_sha256(memory_config):
        raise ValueError("experiential-memory transfer memory config digest does not match")
    memory = ExperientialMemory.from_config(dict(memory_config))
    if not _strict_json_equal(dict(memory_config), memory.to_config()):
        raise ValueError("experiential-memory transfer memory config is noncanonical")
    snapshot = _mapping(metadata.get("snapshot"), name="snapshot")
    if metadata.get("snapshot_sha256") != _canonical_sha256(snapshot):
        raise ValueError("experiential-memory transfer snapshot descriptor digest does not match")
    restored, restored_metadata = load_checkpoint(memory.init(), path)
    if not isinstance(restored, ExperientialMemoryState):
        raise ValueError("experiential-memory transfer restored state type is invalid")
    if restored_metadata != metadata:
        raise ValueError("experiential-memory transfer snapshot metadata changed between reads")
    descriptor = _snapshot_descriptor(memory, restored)
    if not _strict_json_equal(snapshot, descriptor):
        raise ValueError("restored experiential-memory snapshot does not match descriptor")
    return memory, restored


class ExperientialMemoryTransferEvaluator:
    """Immutable adapter binding one development config and A/B/A protocol."""

    def __init__(
        self,
        config: ExperientialMemoryTransferConfig,
        protocol: ExperientialMemoryTransferProtocol,
    ) -> None:
        if not isinstance(config, ExperientialMemoryTransferConfig):
            raise TypeError("config must be ExperientialMemoryTransferConfig")
        if not isinstance(protocol, ExperientialMemoryTransferProtocol):
            raise TypeError("protocol must be ExperientialMemoryTransferProtocol")
        _validate_protocol_dimensions(config, protocol)
        self._config = config
        self._protocol = protocol

    def to_config(self) -> dict[str, object]:
        return {
            "config": self._config.to_config(),
            "protocol": self._protocol.to_config(),
        }

    def build_report(
        self,
        memory: ExperientialMemory,
        state: ExperientialMemoryState,
        *,
        root: Path = REPO_ROOT,
    ) -> dict[str, object]:
        return build_experiential_memory_transfer_report(
            memory,
            state,
            self._config,
            self._protocol,
            root=root,
        )


__all__ = [
    "DEVELOPMENT_STATUS",
    "EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA",
    "EXPERIENTIAL_MEMORY_TRANSFER_CONFIG_SCHEMA",
    "EXPERIENTIAL_MEMORY_TRANSFER_PROTOCOL_SCHEMA",
    "EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA",
    "MAX_ABSOLUTE_EVENTS",
    "MAX_ABSOLUTE_PHASES",
    "MAX_ABSOLUTE_REPORT_BYTES",
    "MAX_ABSOLUTE_SNAPSHOT_BYTES",
    "SOURCE_PATHS",
    "ExperientialMemoryTransferConfig",
    "ExperientialMemoryTransferEvaluator",
    "ExperientialMemoryTransferEvent",
    "ExperientialMemoryTransferPhase",
    "ExperientialMemoryTransferProtocol",
    "ExperientialMemoryTransferValidation",
    "build_experiential_memory_transfer_report",
    "canonical_experiential_memory_transfer_report_bytes",
    "default_experiential_memory_transfer_config",
    "default_experiential_memory_transfer_protocol",
    "experiential_memory_transfer_source_snapshot",
    "frozen_experiential_memory_state_sha256",
    "load_experiential_memory_transfer_report",
    "load_experiential_memory_transfer_snapshot_checkpoint",
    "reconstruct_experiential_memory_transfer_summary",
    "save_experiential_memory_transfer_report",
    "save_experiential_memory_transfer_snapshot_checkpoint",
    "validate_experiential_memory_transfer_report",
]
