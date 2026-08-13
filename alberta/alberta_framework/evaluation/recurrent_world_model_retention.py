"""Strict development-only recurrent world-model retention diagnostics.

This evaluator adds an evaluator-only phase schedule to the bounded recurrent
world-model prequential probe contract.  The exact ordered grounded cases for a
recurring context must be reused after an intervening context and every phase
ends at a declared recurrent reset.  The first prediction in each occurrence
is therefore a directly comparable entry measurement; subsequent predictions
are explicitly pre-update measurements from an isolated copy that adapts once
per event.

No evaluator context identifier crosses the model boundary.  The supplied
snapshot is hashed before and after the run and never updated.  Reports retain
the complete source-bound prequential trace from the underlying diagnostic and
derive ID/OOD, per-phase, recurrence-entry, and within-occurrence measurements
from it.  All measurements are descriptive, development-only, and
``not-assessed``: they establish neither retention, efficacy, calibration,
scientific promotion, nor an Alberta Plan completion.
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
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleState,
)
from alberta_framework.evaluation.world_model_calibration import (
    REPO_ROOT,
    RecurrentWorldModelCalibrationEvent,
    RecurrentWorldModelCalibrationProbeSet,
    WorldModelCalibrationConfig,
    build_recurrent_world_model_calibration_report,
    frozen_world_model_state_sha256,
    validate_recurrent_world_model_calibration_report,
)
from alberta_framework.evaluation.world_model_calibration import (
    SOURCE_PATHS as CALIBRATION_SOURCE_PATHS,
)

RECURRENT_WORLD_MODEL_RETENTION_CONFIG_SCHEMA = (
    "alberta.recurrent-world-model-retention.config.v1"
)
RECURRENT_WORLD_MODEL_RETENTION_PROTOCOL_SCHEMA = (
    "alberta.recurrent-world-model-retention.protocol.v1"
)
RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA = (
    "alberta.recurrent-world-model-retention.report.v1"
)
RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA = (
    "alberta.recurrent-world-model-retention.snapshot.v1"
)
DEVELOPMENT_STATUS = "development-only-not-assessed"

SOURCE_PATHS = tuple(
    sorted(
        {
            *CALIBRATION_SOURCE_PATHS,
            Path("alberta_framework/evaluation/recurrent_world_model_retention.py"),
        },
        key=Path.as_posix,
    )
)

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "the fixed evaluator-owned A/B/A-style trace does not establish external validity",
    "recurrence entry compares exact reused cases after recurrent reset, but remains descriptive",
    "phase means include disclosed isolated adaptation and are not frozen-state held-out scores",
    "heteroscedastic Gaussian NLL is a training objective, not a calibration certificate",
    "one source-bound run supplies no efficacy, retention, scientific-promotion, or SOTA claim",
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
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a canonical finite JSON float")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recurrent_world_model_retention_source_snapshot(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Hash the complete local source closure used by the retention adapter."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


@dataclasses.dataclass(frozen=True)
class RecurrentWorldModelRetentionConfig:
    """Frozen diagnostic settings and hard report/snapshot resource limits."""

    diagnostic_config: WorldModelCalibrationConfig
    max_phases: int
    max_initial_snapshot_bytes: int
    max_report_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostic_config, WorldModelCalibrationConfig):
            raise TypeError("diagnostic_config must be WorldModelCalibrationConfig")
        _positive_int(self.max_phases, name="max_phases")
        _positive_int(
            self.max_initial_snapshot_bytes,
            name="max_initial_snapshot_bytes",
        )
        _positive_int(self.max_report_bytes, name="max_report_bytes")
        if (
            self.diagnostic_config.max_rollout_probes != 0
            or self.diagnostic_config.max_rollout_horizon != 0
        ):
            raise ValueError("recurrent retention v1 forbids open-loop rollout probes")

    @property
    def max_events(self) -> int:
        """Maximum prequential events delegated to the fixed diagnostic."""
        return self.diagnostic_config.max_one_step_cases

    def to_config(self) -> dict[str, object]:
        return {
            "schema": RECURRENT_WORLD_MODEL_RETENTION_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "performance_thresholds_applied": False,
            "diagnostic_config": self.diagnostic_config.to_config(),
            "max_phases": self.max_phases,
            "max_initial_snapshot_bytes": self.max_initial_snapshot_bytes,
            "max_report_bytes": self.max_report_bytes,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> RecurrentWorldModelRetentionConfig:
        expected = set(
            cls(
                diagnostic_config=WorldModelCalibrationConfig(),
                max_phases=1,
                max_initial_snapshot_bytes=1,
                max_report_bytes=1,
            ).to_config()
        )
        if set(payload) != expected:
            raise ValueError("recurrent retention config fields do not match v1")
        fixed = {
            "schema": RECURRENT_WORLD_MODEL_RETENTION_CONFIG_SCHEMA,
            "type": cls.__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "performance_thresholds_applied": False,
        }
        for name, value in fixed.items():
            if not _strict_json_equal(payload.get(name), value):
                raise ValueError(f"recurrent retention config {name} is invalid")
        nested = _mapping(payload.get("diagnostic_config"), name="diagnostic_config")
        result = cls(
            diagnostic_config=WorldModelCalibrationConfig.from_config(nested),
            max_phases=_positive_int(payload.get("max_phases"), name="max_phases"),
            max_initial_snapshot_bytes=_positive_int(
                payload.get("max_initial_snapshot_bytes"),
                name="max_initial_snapshot_bytes",
            ),
            max_report_bytes=_positive_int(
                payload.get("max_report_bytes"),
                name="max_report_bytes",
            ),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("recurrent retention config is noncanonical")
        return result


@dataclasses.dataclass(frozen=True)
class RecurrentWorldModelRetentionPhase:
    """Evaluator-only contiguous phase annotation over ordered probe events."""

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
    ) -> RecurrentWorldModelRetentionPhase:
        if set(payload) != {"phase_id", "evaluator_regime_id", "event_count"}:
            raise ValueError("recurrent retention phase fields do not match v1")
        result = cls(
            phase_id=_identifier(payload.get("phase_id"), name="phase_id"),
            evaluator_regime_id=_identifier(
                payload.get("evaluator_regime_id"),
                name="evaluator_regime_id",
            ),
            event_count=_positive_int(payload.get("event_count"), name="event_count"),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("recurrent retention phase is noncanonical")
        return result


def _event_case_signature(event: RecurrentWorldModelCalibrationEvent) -> dict[str, object]:
    payload = event.to_config()
    # A boundary's reset observation starts the next evaluator phase; it is not
    # a target or prediction input for the event whose recurrence is compared.
    return {
        key: value
        for key, value in payload.items()
        if key not in {"event_id", "next_decision_observation"}
    }


@dataclasses.dataclass(frozen=True)
class RecurrentWorldModelRetentionProtocol:
    """Fixed evaluator-owned ordered probes plus hidden phase annotations."""

    protocol_id: str
    probes: RecurrentWorldModelCalibrationProbeSet
    phases: tuple[RecurrentWorldModelRetentionPhase, ...]

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id")
        if not isinstance(self.probes, RecurrentWorldModelCalibrationProbeSet):
            raise TypeError("probes must be RecurrentWorldModelCalibrationProbeSet")
        if not isinstance(self.phases, tuple) or len(self.phases) < 3:
            raise ValueError("retention protocol requires at least three phases")
        if not all(isinstance(phase, RecurrentWorldModelRetentionPhase) for phase in self.phases):
            raise TypeError("phases must contain RecurrentWorldModelRetentionPhase")
        if len({phase.phase_id for phase in self.phases}) != len(self.phases):
            raise ValueError("phase_id values must be unique")
        if sum(phase.event_count for phase in self.phases) != len(self.probes.events):
            raise ValueError("phase event counts must exactly cover the ordered probes")
        if {event.partition for event in self.probes.events} != {
            "in_distribution",
            "ood",
        }:
            raise ValueError("retention protocol requires both ID and OOD probe events")
        if len({phase.evaluator_regime_id for phase in self.phases}) < 2:
            raise ValueError("retention protocol requires at least two evaluator regimes")

        phase_events = self.phase_events()
        for phase, events in zip(self.phases, phase_events, strict=True):
            if not (events[-1].terminated or events[-1].truncated):
                raise ValueError(f"phase {phase.phase_id} must end at a recurrent reset")

        positions: dict[str, list[int]] = {}
        for index, phase in enumerate(self.phases):
            positions.setdefault(phase.evaluator_regime_id, []).append(index)
        recurring_with_interference = False
        for regime_id, indices in positions.items():
            if len(indices) < 2:
                continue
            reference = [
                _event_case_signature(event) for event in phase_events[indices[0]]
            ]
            for index in indices[1:]:
                current = [_event_case_signature(event) for event in phase_events[index]]
                if not _strict_json_equal(current, reference):
                    raise ValueError(
                        f"recurring evaluator regime {regime_id} must reuse exact ordered cases"
                    )
            for left, right in zip(indices, indices[1:], strict=False):
                if any(
                    self.phases[index].evaluator_regime_id != regime_id
                    for index in range(left + 1, right)
                ):
                    recurring_with_interference = True
        if not recurring_with_interference:
            raise ValueError(
                "retention protocol requires recurrence after an intervening evaluator regime"
            )

    def phase_events(
        self,
    ) -> tuple[tuple[RecurrentWorldModelCalibrationEvent, ...], ...]:
        """Return exact contiguous event slices in evaluator phase order."""
        groups: list[tuple[RecurrentWorldModelCalibrationEvent, ...]] = []
        start = 0
        for phase in self.phases:
            end = start + phase.event_count
            groups.append(self.probes.events[start:end])
            start = end
        return tuple(groups)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": RECURRENT_WORLD_MODEL_RETENTION_PROTOCOL_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "ownership": "evaluator-owned-fixed-ordered-probes-and-phase-labels",
            "learner_visible_fields": [
                "observation",
                "action",
                "bootstrap_observation_target",
                "reward_target",
                "continuation_target",
                "terminated",
                "truncated",
                "next_decision_observation",
            ],
            "evaluator_only_fields": [
                "phase_id",
                "evaluator_regime_id",
                "partition",
            ],
            "regime_identifiers_visible_to_model": False,
            "protocol_id": self.protocol_id,
            "probes": self.probes.to_config(),
            "phases": [phase.to_config() for phase in self.phases],
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> RecurrentWorldModelRetentionProtocol:
        expected = {
            "schema",
            "type",
            "development_status",
            "ownership",
            "learner_visible_fields",
            "evaluator_only_fields",
            "regime_identifiers_visible_to_model",
            "protocol_id",
            "probes",
            "phases",
        }
        if set(payload) != expected:
            raise ValueError("recurrent retention protocol fields do not match v1")
        fixed = {
            "schema": RECURRENT_WORLD_MODEL_RETENTION_PROTOCOL_SCHEMA,
            "type": cls.__name__,
            "development_status": DEVELOPMENT_STATUS,
            "ownership": "evaluator-owned-fixed-ordered-probes-and-phase-labels",
            "learner_visible_fields": [
                "observation",
                "action",
                "bootstrap_observation_target",
                "reward_target",
                "continuation_target",
                "terminated",
                "truncated",
                "next_decision_observation",
            ],
            "evaluator_only_fields": [
                "phase_id",
                "evaluator_regime_id",
                "partition",
            ],
            "regime_identifiers_visible_to_model": False,
        }
        for name, value in fixed.items():
            if not _strict_json_equal(payload.get(name), value):
                raise ValueError(f"recurrent retention protocol {name} is invalid")
        raw_phases = _list(payload.get("phases"), name="phases")
        if not raw_phases or any(not isinstance(value, Mapping) for value in raw_phases):
            raise ValueError("phases must be a non-empty object array")
        result = cls(
            protocol_id=_identifier(payload.get("protocol_id"), name="protocol_id"),
            probes=RecurrentWorldModelCalibrationProbeSet.from_config(
                _mapping(payload.get("probes"), name="probes")
            ),
            phases=tuple(
                RecurrentWorldModelRetentionPhase.from_config(
                    cast(Mapping[str, object], phase)
                )
                for phase in raw_phases
            ),
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("recurrent retention protocol is noncanonical")
        return result


def _raw_events(base_report: Mapping[str, object]) -> list[object]:
    if set(base_report) != {"schema", "payload", "payload_sha256"}:
        raise ValueError("base prequential report fields do not match its schema")
    payload = _mapping(base_report.get("payload"), name="base report payload")
    raw = _mapping(payload.get("raw_trace"), name="base raw trace")
    if set(raw) != {"events"}:
        raise ValueError("base raw trace fields are invalid")
    return _list(raw.get("events"), name="base raw events")


def _event_primitives(
    record_value: object,
    probe: RecurrentWorldModelCalibrationEvent,
    *,
    observation_dim: int,
) -> dict[str, object]:
    record = _mapping(record_value, name="base raw event")
    if record.get("event_id") != probe.event_id:
        raise ValueError("base raw event identifier does not match the ordered probe")
    if record.get("partition") != probe.partition:
        raise ValueError("base raw event partition does not match the ordered probe")
    targets = _mapping(record.get("targets"), name="base raw targets")
    means = _mapping(record.get("mean_predictions"), name="base raw means")
    target_values = _list(targets.get("grounded_vector"), name="grounded targets")
    mean_values = _list(means.get("grounded_vector"), name="grounded means")
    target_dim = observation_dim + 2
    if len(target_values) != target_dim or len(mean_values) != target_dim:
        raise ValueError("base grounded vectors do not match model observation dimension")
    target = np.asarray(
        [
            _finite_float(value, name=f"grounded targets[{index}]")
            for index, value in enumerate(target_values)
        ],
        dtype=np.float64,
    )
    mean = np.asarray(
        [
            _finite_float(value, name=f"grounded means[{index}]")
            for index, value in enumerate(mean_values)
        ],
        dtype=np.float64,
    )
    squared = np.square(mean - target)
    update = _mapping(record.get("prequential_update"), name="base prequential update")
    if update.get("applied_to_isolated_copy") is not True or update.get(
        "recurrent_advanced_once"
    ) is not True:
        raise ValueError("base event must disclose one accepted isolated update")
    mean_nll = _finite_float(
        update.get("mean_negative_log_likelihood"),
        name="mean_negative_log_likelihood",
    )
    return {
        "event_id": probe.event_id,
        "partition": probe.partition,
        "mean_preupdate_nll": mean_nll,
        "mean_prediction_mse": float(np.mean(squared)),
        "dynamics_observation_mse": float(np.mean(squared[:observation_dim])),
        "reward_squared_error": float(squared[-2]),
        "continuation_squared_error": float(squared[-1]),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric group")
    return math.fsum(values) / len(values)


def _aggregate_metrics(primitives: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not primitives:
        raise ValueError("metric group must be non-empty")
    metric_names = (
        "mean_preupdate_nll",
        "mean_prediction_mse",
        "dynamics_observation_mse",
        "reward_squared_error",
        "continuation_squared_error",
    )
    return {
        "event_count": len(primitives),
        **{
            name: _mean([cast(float, primitive[name]) for primitive in primitives])
            for name in metric_names
        },
    }


def _evaluator_annotations(
    protocol: RecurrentWorldModelRetentionProtocol,
) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    event_index = 0
    occurrences: dict[str, int] = {}
    for phase_index, (phase, events) in enumerate(
        zip(protocol.phases, protocol.phase_events(), strict=True)
    ):
        occurrence = occurrences.get(phase.evaluator_regime_id, 0)
        occurrences[phase.evaluator_regime_id] = occurrence + 1
        for phase_event_index, event in enumerate(events):
            annotations.append(
                {
                    "event_index": event_index,
                    "event_id": event.event_id,
                    "phase_index": phase_index,
                    "phase_event_index": phase_event_index,
                    "phase_id": phase.phase_id,
                    "evaluator_regime_id": phase.evaluator_regime_id,
                    "evaluator_regime_occurrence_index": occurrence,
                    "partition": event.partition,
                    "evaluator_owned": True,
                    "learner_visible": False,
                }
            )
            event_index += 1
    return annotations


def reconstruct_recurrent_world_model_retention_summary(
    base_report: Mapping[str, object],
    protocol: RecurrentWorldModelRetentionProtocol,
) -> dict[str, object]:
    """Reconstruct all ID/OOD, phase, recurrence, and retention measurements."""
    if not isinstance(protocol, RecurrentWorldModelRetentionProtocol):
        raise TypeError("protocol must be RecurrentWorldModelRetentionProtocol")
    payload = _mapping(base_report.get("payload"), name="base report payload")
    initial = _mapping(payload.get("initial_snapshot"), name="initial snapshot")
    observation_dim = _positive_int(
        initial.get("observation_dim"),
        name="initial snapshot observation_dim",
    )
    raw_events = _raw_events(base_report)
    if len(raw_events) != len(protocol.probes.events):
        raise ValueError("base raw event count does not match retention protocol")
    primitives = [
        _event_primitives(record, probe, observation_dim=observation_dim)
        for record, probe in zip(raw_events, protocol.probes.events, strict=True)
    ]

    id_ood_metrics = []
    for partition in ("in_distribution", "ood"):
        selected = [item for item in primitives if item["partition"] == partition]
        id_ood_metrics.append(
            {
                "partition": partition,
                **_aggregate_metrics(selected),
            }
        )

    phase_metrics: list[dict[str, object]] = []
    offset = 0
    occurrence_counts: dict[str, int] = {}
    for phase_index, phase in enumerate(protocol.phases):
        selected = primitives[offset : offset + phase.event_count]
        occurrence_index = occurrence_counts.get(phase.evaluator_regime_id, 0)
        occurrence_counts[phase.evaluator_regime_id] = occurrence_index + 1
        metrics = _aggregate_metrics(selected)
        phase_metrics.append(
            {
                "phase_index": phase_index,
                "phase_id": phase.phase_id,
                "evaluator_regime_id": phase.evaluator_regime_id,
                "evaluator_regime_occurrence_index": occurrence_index,
                "event_start_index": offset,
                "event_stop_index_exclusive": offset + phase.event_count,
                **metrics,
                "entry_preupdate_nll": selected[0]["mean_preupdate_nll"],
                "exit_preupdate_nll": selected[-1]["mean_preupdate_nll"],
                "within_phase_nll_reduction": cast(
                    float, selected[0]["mean_preupdate_nll"]
                )
                - cast(float, selected[-1]["mean_preupdate_nll"]),
            }
        )
        offset += phase.event_count

    regime_order = tuple(dict.fromkeys(phase.evaluator_regime_id for phase in protocol.phases))
    retention_by_regime: list[dict[str, object]] = []
    for regime_id in regime_order:
        occurrences = [
            phase for phase in phase_metrics if phase["evaluator_regime_id"] == regime_id
        ]
        occurrence_payload = [
            {
                "phase_index": phase["phase_index"],
                "phase_id": phase["phase_id"],
                "occurrence_index": phase["evaluator_regime_occurrence_index"],
                "event_count": phase["event_count"],
                "entry_preupdate_nll": phase["entry_preupdate_nll"],
                "exit_preupdate_nll": phase["exit_preupdate_nll"],
                "mean_preupdate_nll": phase["mean_preupdate_nll"],
                "mean_prediction_mse": phase["mean_prediction_mse"],
                "dynamics_observation_mse": phase["dynamics_observation_mse"],
                "reward_squared_error": phase["reward_squared_error"],
                "continuation_squared_error": phase["continuation_squared_error"],
                "within_phase_nll_reduction": phase["within_phase_nll_reduction"],
            }
            for phase in occurrences
        ]
        recurring = len(occurrences) > 1
        recurrence_measurements: list[dict[str, object]] = []
        if recurring:
            first = occurrences[0]
            previous = first
            for current in occurrences[1:]:
                recurrence_measurements.append(
                    {
                        "phase_index": current["phase_index"],
                        "phase_id": current["phase_id"],
                        "occurrence_index": current["evaluator_regime_occurrence_index"],
                        "previous_occurrence_phase_index": previous["phase_index"],
                        "first_occurrence_phase_index": first["phase_index"],
                        "entry_nll_change_from_first_occurrence": cast(
                            float, current["entry_preupdate_nll"]
                        )
                        - cast(float, first["entry_preupdate_nll"]),
                        "entry_nll_change_from_previous_occurrence": cast(
                            float, current["entry_preupdate_nll"]
                        )
                        - cast(float, previous["entry_preupdate_nll"]),
                        "phase_mean_nll_change_from_first_occurrence": cast(
                            float, current["mean_preupdate_nll"]
                        )
                        - cast(float, first["mean_preupdate_nll"]),
                        "phase_mean_nll_change_from_previous_occurrence": cast(
                            float, current["mean_preupdate_nll"]
                        )
                        - cast(float, previous["mean_preupdate_nll"]),
                        "within_recurrence_phase_nll_reduction": current[
                            "within_phase_nll_reduction"
                        ],
                    }
                )
                previous = current
            entry_values = [cast(float, item["entry_preupdate_nll"]) for item in occurrences]
            latest_entry_minus_first: float | None = entry_values[-1] - entry_values[0]
            best_entry_to_latest: float | None = max(
                0.0,
                entry_values[-1] - min(entry_values),
            )
            unavailable_reason: str | None = None
        else:
            latest_entry_minus_first = None
            best_entry_to_latest = None
            unavailable_reason = "unavailable: evaluator regime has no repeated occurrence"
        retention_by_regime.append(
            {
                "evaluator_regime_id": regime_id,
                "occurrence_count": len(occurrences),
                "recurrence_available": recurring,
                "unavailable_reason": unavailable_reason,
                "exact_ordered_case_reuse": recurring,
                "occurrences": occurrence_payload,
                "latest_entry_minus_first_entry_nll": latest_entry_minus_first,
                "best_entry_to_latest_entry_forgetting_nll": best_entry_to_latest,
                "recurrence_measurements": recurrence_measurements,
                "assessment": "not-assessed",
            }
        )

    base_summary = _mapping(payload.get("summary"), name="base summary")
    base_accounting = _mapping(
        base_summary.get("prequential_transaction_accounting"),
        name="base transaction accounting",
    )
    return {
        "assessment_status": "not-assessed",
        "metric_direction": "lower_is_better",
        "evaluation_protocol": (
            "exact-case-recurrence-entry-plus-prequential-isolated-adaptation"
        ),
        "id_ood_metrics": id_ood_metrics,
        "phase_metrics": phase_metrics,
        "retention_by_regime": retention_by_regime,
        "transaction_accounting": dict(base_accounting),
        "claims": {
            "retention_established": False,
            "efficacy_established": False,
            "calibration_established": False,
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
        },
    }


def _snapshot_descriptor(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
) -> dict[str, object]:
    if not isinstance(model, RecurrentLatentWorldModelEnsemble):
        raise TypeError("model must be RecurrentLatentWorldModelEnsemble")
    if not isinstance(state, RecurrentLatentWorldModelEnsembleState):
        raise TypeError("state must be RecurrentLatentWorldModelEnsembleState")
    if not bool(jax.device_get(model.state_valid(state))):
        raise ValueError("recurrent retention snapshot state is invalid")
    model_config = model.to_config()
    resources = model.resource_budget(state).to_config()
    return {
        "model_config": model_config,
        "model_config_sha256": _canonical_sha256(model_config),
        "state_sha256": frozen_world_model_state_sha256(state),
        "resource_budget": resources,
        "resource_budget_sha256": _canonical_sha256(resources),
        "event_count": int(jax.device_get(state.event_count)),
        "recurrent_advance_count": int(jax.device_get(state.recurrent_advance_count)),
        "boundary_count": int(jax.device_get(state.boundary_count)),
        "zero_recurrent_context": bool(
            np.array_equal(
                np.asarray(jax.device_get(state.member_hidden_states)),
                np.zeros(
                    (model.config.ensemble_size, model.config.latent_dim),
                    dtype=np.float32,
                ),
            )
        ),
    }


def _validate_evaluation_inputs(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
    config: RecurrentWorldModelRetentionConfig,
    protocol: RecurrentWorldModelRetentionProtocol,
) -> dict[str, object]:
    descriptor = _snapshot_descriptor(model, state)
    if len(protocol.phases) > config.max_phases:
        raise ValueError("retention phase schedule exceeds max_phases")
    if len(protocol.probes.events) > config.max_events:
        raise ValueError("retention ordered probes exceed max_events")
    resources = cast(Mapping[str, object], descriptor["resource_budget"])
    state_bytes = _positive_int(
        resources.get("persistent_state_bytes"),
        name="persistent_state_bytes",
    )
    if state_bytes > config.max_initial_snapshot_bytes:
        raise ValueError("initial snapshot exceeds max_initial_snapshot_bytes")
    if descriptor["zero_recurrent_context"] is not True:
        raise ValueError("retention evaluation requires an exact zero recurrent context")
    remaining = model.config.max_updates - cast(int, descriptor["event_count"])
    if len(protocol.probes.events) > remaining:
        raise ValueError("retention ordered probes exceed snapshot update capacity")
    return descriptor


def _resource_accounting(
    *,
    model: RecurrentLatentWorldModelEnsemble,
    config: RecurrentWorldModelRetentionConfig,
    protocol: RecurrentWorldModelRetentionProtocol,
    base_report: Mapping[str, object],
    sources: Mapping[str, str],
    canonical_report_bytes: int,
) -> dict[str, object]:
    base_payload = _mapping(base_report.get("payload"), name="base report payload")
    base_resources = _mapping(
        base_payload.get("resource_accounting"),
        name="base resource accounting",
    )
    initial = _mapping(base_payload.get("initial_snapshot"), name="initial snapshot")
    event_count = len(protocol.probes.events)
    return {
        "initial_snapshot_state_bytes": _positive_int(
            initial.get("state_bytes"),
            name="initial snapshot state_bytes",
        ),
        "initial_snapshot_state_byte_limit": config.max_initial_snapshot_bytes,
        "bounded_event_capacity": config.max_events,
        "recorded_event_count": event_count,
        "bounded_phase_capacity": config.max_phases,
        "recorded_phase_count": len(protocol.phases),
        "predict_before_update_decide_calls": _nonnegative_int(
            base_resources.get("predict_before_update_decide_calls"),
            name="predict_before_update_decide_calls",
        ),
        "isolated_copy_update_calls": _nonnegative_int(
            base_resources.get("isolated_copy_update_calls"),
            name="isolated_copy_update_calls",
        ),
        "recurrent_advances": _nonnegative_int(
            base_resources.get("recurrent_advances"),
            name="recurrent_advances",
        ),
        "member_prediction_records": event_count * model.config.ensemble_size,
        "member_gradient_candidates": event_count * model.config.ensemble_size,
        "member_parameter_updates_applied": _nonnegative_int(
            base_resources.get("member_parameter_updates_applied"),
            name="member_parameter_updates_applied",
        ),
        "base_prequential_report_bytes": len(_canonical_json_bytes(base_report)) + 1,
        "source_manifest_entries": len(sources),
        "canonical_report_bytes": canonical_report_bytes,
        "canonical_report_byte_limit": config.max_report_bytes,
        "external_snapshot_mutations": 0,
        "learner_visible_evaluator_label_reads": 0,
        "regime_identifier_reads_by_model": 0,
        "persistent_evaluator_state_bytes": 0,
    }


def _assemble_report(
    *,
    config: RecurrentWorldModelRetentionConfig,
    protocol: RecurrentWorldModelRetentionProtocol,
    sources: Mapping[str, str],
    base_report: Mapping[str, object],
    annotations: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    base_payload = _mapping(base_report.get("payload"), name="base report payload")
    initial = base_payload.get("initial_snapshot")
    final = base_payload.get("final_isolated_state")
    config_payload = config.to_config()
    protocol_payload = protocol.to_config()
    annotation_payload = [dict(item) for item in annotations]
    hashes = {
        "config_sha256": _canonical_sha256(config_payload),
        "protocol_sha256": _canonical_sha256(protocol_payload),
        "source_manifest_sha256": _canonical_sha256(sources),
        "initial_snapshot_sha256": _canonical_sha256(initial),
        "final_isolated_state_sha256": _canonical_sha256(final),
        "base_prequential_report_sha256": _canonical_sha256(base_report),
        "evaluator_annotations_sha256": _canonical_sha256(annotation_payload),
        "summary_sha256": _canonical_sha256(summary),
        "resource_accounting_sha256": _canonical_sha256(resources),
    }
    payload: dict[str, object] = {
        "development_only": True,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "performance_thresholds_applied": False,
        "evaluation_protocol": (
            "exact-case-recurrence-entry-plus-prequential-isolated-adaptation"
        ),
        "config": config_payload,
        "source_sha256": dict(sources),
        "protocol": protocol_payload,
        "base_prequential_report": dict(base_report),
        "evaluator_annotations": annotation_payload,
        "summary": dict(summary),
        "resource_accounting": dict(resources),
        "hashes": hashes,
        "limitations": list(_LIMITATIONS),
    }
    return {
        "schema": RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def build_recurrent_world_model_retention_report(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
    config: RecurrentWorldModelRetentionConfig,
    protocol: RecurrentWorldModelRetentionProtocol,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Evaluate exact recurrence on an isolated copy of one frozen snapshot."""
    if not isinstance(config, RecurrentWorldModelRetentionConfig):
        raise TypeError("config must be RecurrentWorldModelRetentionConfig")
    if not isinstance(protocol, RecurrentWorldModelRetentionProtocol):
        raise TypeError("protocol must be RecurrentWorldModelRetentionProtocol")
    _validate_evaluation_inputs(model, state, config, protocol)
    snapshot_hash = frozen_world_model_state_sha256(state)
    base_report = build_recurrent_world_model_calibration_report(
        model,
        state,
        config.diagnostic_config,
        protocol.probes,
        root=root,
    )
    if frozen_world_model_state_sha256(state) != snapshot_hash:
        raise RuntimeError("recurrent retention evaluation mutated the supplied snapshot")
    base_validation = validate_recurrent_world_model_calibration_report(
        base_report,
        root=root,
    )
    if not base_validation.valid:
        raise ValueError(
            "internal base prequential report validation failed: "
            + "; ".join(base_validation.errors)
        )
    sources = recurrent_world_model_retention_source_snapshot(root)
    annotations = _evaluator_annotations(protocol)
    summary = reconstruct_recurrent_world_model_retention_summary(base_report, protocol)

    report_size = 0
    report: dict[str, object] | None = None
    for _ in range(16):
        resources = _resource_accounting(
            model=model,
            config=config,
            protocol=protocol,
            base_report=base_report,
            sources=sources,
            canonical_report_bytes=report_size,
        )
        report = _assemble_report(
            config=config,
            protocol=protocol,
            sources=sources,
            base_report=base_report,
            annotations=annotations,
            summary=summary,
            resources=resources,
        )
        measured = len(_canonical_json_bytes(report)) + 1
        if measured == report_size:
            break
        report_size = measured
    else:
        raise RuntimeError("canonical recurrent retention report size did not converge")
    assert report is not None
    if report_size > config.max_report_bytes:
        raise ValueError("canonical report exceeds configured report byte bound")
    validation = validate_recurrent_world_model_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "internal recurrent retention report validation failed: "
            + "; ".join(validation.errors)
        )
    return report


@dataclasses.dataclass(frozen=True)
class RecurrentWorldModelRetentionValidation:
    """Fail-closed structural result with no scientific assessment verdict."""

    valid: bool
    assessment_status: str
    errors: tuple[str, ...]


def _validate_annotation_binding(
    annotations_value: object,
    protocol: RecurrentWorldModelRetentionProtocol,
) -> list[dict[str, object]]:
    values = _list(annotations_value, name="evaluator_annotations")
    expected = _evaluator_annotations(protocol)
    if not _strict_json_equal(values, expected):
        raise ValueError("evaluator annotations do not reconstruct from protocol")
    return expected


def _model_from_base_report(
    base_report: Mapping[str, object],
) -> RecurrentLatentWorldModelEnsemble:
    payload = _mapping(base_report.get("payload"), name="base report payload")
    initial = _mapping(payload.get("initial_snapshot"), name="initial snapshot")
    model_config = _mapping(initial.get("model_config"), name="initial model config")
    model = RecurrentLatentWorldModelEnsemble.from_config(model_config)
    if not _strict_json_equal(dict(model_config), model.to_config()):
        raise ValueError("initial model config is noncanonical")
    return model


def validate_recurrent_world_model_retention_report(
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    model: RecurrentLatentWorldModelEnsemble | None = None,
    state: RecurrentLatentWorldModelEnsembleState | None = None,
    protocol: RecurrentWorldModelRetentionProtocol | None = None,
) -> RecurrentWorldModelRetentionValidation:
    """Reject schema, source, hash, reconstruction, resource, or replay drift."""
    errors: list[str] = []
    if set(report) != {"schema", "payload", "payload_sha256"}:
        errors.append("recurrent retention report top-level fields do not match v1")
    if report.get("schema") != RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA:
        errors.append("recurrent retention report schema is unsupported")
    payload_value = report.get("payload")
    if not isinstance(payload_value, Mapping):
        errors.append("recurrent retention report payload must be an object")
        payload: Mapping[str, object] = {}
    else:
        payload = cast(Mapping[str, object], payload_value)
    expected_payload = {
        "development_only",
        "assessment_status",
        "scientific_promotion_allowed",
        "efficacy_claimed",
        "calibration_claimed",
        "performance_thresholds_applied",
        "evaluation_protocol",
        "config",
        "source_sha256",
        "protocol",
        "base_prequential_report",
        "evaluator_annotations",
        "summary",
        "resource_accounting",
        "hashes",
        "limitations",
    }
    if set(payload) != expected_payload:
        errors.append("recurrent retention report payload fields do not match v1")
    fixed = {
        "development_only": True,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "performance_thresholds_applied": False,
        "evaluation_protocol": (
            "exact-case-recurrence-entry-plus-prequential-isolated-adaptation"
        ),
        "limitations": list(_LIMITATIONS),
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(payload.get(name), expected):
            errors.append(f"recurrent retention report {name} is not the fixed v1 value")
    try:
        if report.get("payload_sha256") != _canonical_sha256(payload):
            errors.append("recurrent retention report payload digest does not match")
    except (TypeError, ValueError) as error:
        errors.append(f"recurrent retention payload is not canonical JSON: {error}")

    config: RecurrentWorldModelRetentionConfig | None = None
    config_value = payload.get("config")
    if not isinstance(config_value, Mapping):
        errors.append("recurrent retention report config must be an object")
    else:
        try:
            config = RecurrentWorldModelRetentionConfig.from_config(config_value)
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention report config is invalid: {error}")

    reconstructed_protocol: RecurrentWorldModelRetentionProtocol | None = None
    protocol_value = payload.get("protocol")
    if not isinstance(protocol_value, Mapping):
        errors.append("recurrent retention report protocol must be an object")
    else:
        try:
            reconstructed_protocol = RecurrentWorldModelRetentionProtocol.from_config(
                protocol_value
            )
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention report protocol is invalid: {error}")

    source_value = payload.get("source_sha256")
    try:
        current_sources = recurrent_world_model_retention_source_snapshot(root)
    except OSError as error:
        errors.append(f"cannot hash recurrent retention sources: {error}")
        current_sources = {}
    if not _strict_json_equal(source_value, current_sources):
        errors.append("recurrent retention report source hashes do not match current sources")

    base_value = payload.get("base_prequential_report")
    if not isinstance(base_value, Mapping):
        errors.append("base prequential report must be an object")
        base_report: Mapping[str, object] = {}
    else:
        base_report = cast(Mapping[str, object], base_value)
        base_validation = validate_recurrent_world_model_calibration_report(
            base_report,
            root=root,
        )
        if not base_validation.valid:
            errors.append(
                "base prequential report is invalid: " + "; ".join(base_validation.errors)
            )

    reconstructed_summary: dict[str, object] | None = None
    report_model: RecurrentLatentWorldModelEnsemble | None = None
    if reconstructed_protocol is not None and base_report:
        try:
            base_payload = _mapping(base_report.get("payload"), name="base report payload")
            if not _strict_json_equal(
                base_payload.get("probe_set"),
                reconstructed_protocol.probes.to_config(),
            ):
                raise ValueError("base probe set does not match retention protocol")
            _validate_annotation_binding(
                payload.get("evaluator_annotations"),
                reconstructed_protocol,
            )
            reconstructed_summary = reconstruct_recurrent_world_model_retention_summary(
                base_report,
                reconstructed_protocol,
            )
            if not _strict_json_equal(payload.get("summary"), reconstructed_summary):
                raise ValueError("retention summary does not reconstruct from raw trace")
            report_model = _model_from_base_report(base_report)
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention reconstruction failed: {error}")
    else:
        if not isinstance(payload.get("evaluator_annotations"), list):
            errors.append("evaluator_annotations must be an array")
        if not isinstance(payload.get("summary"), Mapping):
            errors.append("retention summary must be an object")

    resources_value = payload.get("resource_accounting")
    if (
        config is not None
        and reconstructed_protocol is not None
        and report_model is not None
        and base_report
    ):
        try:
            supplied_resources = _mapping(
                resources_value,
                name="resource_accounting",
            )
            canonical_bytes = len(_canonical_json_bytes(report)) + 1
            expected_resources = _resource_accounting(
                model=report_model,
                config=config,
                protocol=reconstructed_protocol,
                base_report=base_report,
                sources=current_sources,
                canonical_report_bytes=canonical_bytes,
            )
            if not _strict_json_equal(supplied_resources, expected_resources):
                raise ValueError("resource accounting does not reconstruct exactly")
            if canonical_bytes > config.max_report_bytes:
                raise ValueError("canonical report exceeds configured report byte bound")
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention resources are invalid: {error}")
    elif not isinstance(resources_value, Mapping):
        errors.append("resource_accounting must be an object")

    hashes_value = payload.get("hashes")
    base_payload_value = (
        base_report.get("payload") if isinstance(base_report, Mapping) else None
    )
    if isinstance(base_payload_value, Mapping):
        initial_value = base_payload_value.get("initial_snapshot")
        final_value = base_payload_value.get("final_isolated_state")
    else:
        initial_value = None
        final_value = None
    hash_inputs = {
        "config_sha256": config_value,
        "protocol_sha256": protocol_value,
        "source_manifest_sha256": source_value,
        "initial_snapshot_sha256": initial_value,
        "final_isolated_state_sha256": final_value,
        "base_prequential_report_sha256": base_value,
        "evaluator_annotations_sha256": payload.get("evaluator_annotations"),
        "summary_sha256": payload.get("summary"),
        "resource_accounting_sha256": resources_value,
    }
    if not isinstance(hashes_value, Mapping) or set(hashes_value) != set(hash_inputs):
        errors.append("recurrent retention report hash fields do not match v1")
    else:
        try:
            expected_hashes = {
                name: _canonical_sha256(value) for name, value in hash_inputs.items()
            }
            if not _strict_json_equal(hashes_value, expected_hashes):
                errors.append("one or more recurrent retention component hashes do not match")
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention report components are noncanonical: {error}")

    optional = (model, state, protocol)
    if any(value is not None for value in optional) and not all(
        value is not None for value in optional
    ):
        errors.append("model, state, and protocol must be supplied together for live replay")
    elif (
        model is not None
        and state is not None
        and protocol is not None
        and reconstructed_protocol is not None
        and base_report
    ):
        before = frozen_world_model_state_sha256(state)
        try:
            if not _strict_json_equal(protocol.to_config(), reconstructed_protocol.to_config()):
                raise ValueError("supplied retention protocol does not match report")
            replay = validate_recurrent_world_model_calibration_report(
                base_report,
                root=root,
                model=model,
                state=state,
                probes=protocol.probes,
            )
            if not replay.valid:
                raise ValueError("; ".join(replay.errors))
            if frozen_world_model_state_sha256(state) != before:
                raise ValueError("live replay mutated the supplied recurrent snapshot")
        except (TypeError, ValueError) as error:
            errors.append(f"recurrent retention live replay failed: {error}")

    return RecurrentWorldModelRetentionValidation(
        valid=not errors,
        assessment_status="not-assessed",
        errors=tuple(errors),
    )


def canonical_recurrent_world_model_retention_report_bytes(
    report: Mapping[str, object],
) -> bytes:
    """Return the sole accepted canonical JSON report encoding."""
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
        raise ValueError("recurrent retention report must be a JSON object")
    return cast(dict[str, object], parsed)


def load_recurrent_world_model_retention_report(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load exact canonical JSON and require full source-bound validation."""
    data = Path(path).read_bytes()
    report = _strict_json_object(data)
    if data != canonical_recurrent_world_model_retention_report_bytes(report):
        raise ValueError("recurrent retention report is not exact canonical JSON")
    validation = validate_recurrent_world_model_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError("invalid recurrent retention report: " + "; ".join(validation.errors))
    return report


def save_recurrent_world_model_retention_report(
    path: str | Path,
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Atomically create one validated canonical report without overwrite."""
    validation = validate_recurrent_world_model_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "refusing to save invalid recurrent retention report: "
            + "; ".join(validation.errors)
        )
    expanded = Path(path).expanduser()
    destination = expanded.resolve()
    if os.path.lexists(expanded) or os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite recurrent retention report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_recurrent_world_model_retention_report_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created recurrent retention report: "
                f"{destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def save_recurrent_world_model_retention_snapshot_checkpoint(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Create a source-, construction-, resource-, and state-bound snapshot."""
    destination = Path(path).expanduser()
    if os.path.lexists(destination) or os.path.lexists(destination.resolve()):
        raise FileExistsError(f"refusing to overwrite recurrent retention snapshot: {destination}")
    descriptor = _snapshot_descriptor(model, state)
    sources = recurrent_world_model_retention_source_snapshot(root)
    save_checkpoint(
        state,
        destination,
        metadata={
            "schema": RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "snapshot": descriptor,
            "snapshot_sha256": _canonical_sha256(descriptor),
            "source_sha256": sources,
            "source_manifest_sha256": _canonical_sha256(sources),
        },
    )


def load_recurrent_world_model_retention_snapshot_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
    root: Path = REPO_ROOT,
) -> tuple[RecurrentLatentWorldModelEnsemble, RecurrentLatentWorldModelEnsembleState]:
    """Restore only the exact current-source recurrent retention snapshot."""
    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "development_status",
        "assessment_status",
        "scientific_promotion_allowed",
        "efficacy_claimed",
        "calibration_claimed",
        "snapshot",
        "snapshot_sha256",
        "source_sha256",
        "source_manifest_sha256",
    }
    if set(metadata) != expected:
        raise ValueError("recurrent retention snapshot metadata fields do not match v1")
    fixed = {
        "schema": RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
    }
    for name, expected_value in fixed.items():
        if not _strict_json_equal(metadata.get(name), expected_value):
            raise ValueError(f"recurrent retention snapshot {name} is invalid")
    sources = recurrent_world_model_retention_source_snapshot(root)
    if not _strict_json_equal(metadata.get("source_sha256"), sources) or metadata.get(
        "source_manifest_sha256"
    ) != _canonical_sha256(sources):
        raise ValueError("recurrent retention snapshot source hashes do not match")
    snapshot = _mapping(metadata.get("snapshot"), name="snapshot")
    if metadata.get("snapshot_sha256") != _canonical_sha256(snapshot):
        raise ValueError("recurrent retention snapshot descriptor digest does not match")
    model_config = _mapping(snapshot.get("model_config"), name="snapshot model_config")
    if snapshot.get("model_config_sha256") != _canonical_sha256(model_config):
        raise ValueError("recurrent retention snapshot model config digest does not match")
    model = RecurrentLatentWorldModelEnsemble.from_config(model_config)
    if not _strict_json_equal(dict(model_config), model.to_config()):
        raise ValueError("recurrent retention snapshot model config is noncanonical")
    key = jr.key(0) if template_key is None else template_key
    template = model.init(key)
    restored, restored_metadata = load_checkpoint(template, path)
    if not isinstance(restored, RecurrentLatentWorldModelEnsembleState):
        raise ValueError("recurrent retention snapshot state type is invalid")
    if restored_metadata != metadata:
        raise ValueError("recurrent retention snapshot metadata changed between reads")
    descriptor = _snapshot_descriptor(model, restored)
    if not _strict_json_equal(snapshot, descriptor):
        raise ValueError("restored recurrent retention snapshot does not match descriptor")
    return model, restored


class RecurrentWorldModelRetentionEvaluator:
    """Immutable adapter binding one fixed config and evaluator-only protocol."""

    def __init__(
        self,
        config: RecurrentWorldModelRetentionConfig,
        protocol: RecurrentWorldModelRetentionProtocol,
    ) -> None:
        if not isinstance(config, RecurrentWorldModelRetentionConfig):
            raise TypeError("config must be RecurrentWorldModelRetentionConfig")
        if not isinstance(protocol, RecurrentWorldModelRetentionProtocol):
            raise TypeError("protocol must be RecurrentWorldModelRetentionProtocol")
        if len(protocol.phases) > config.max_phases:
            raise ValueError("retention phase schedule exceeds max_phases")
        if len(protocol.probes.events) > config.max_events:
            raise ValueError("retention ordered probes exceed max_events")
        self._config = config
        self._protocol = protocol

    def to_config(self) -> dict[str, object]:
        """Return the exact evaluator construction without learned state."""
        return {
            "config": self._config.to_config(),
            "protocol": self._protocol.to_config(),
        }

    def build_report(
        self,
        model: RecurrentLatentWorldModelEnsemble,
        state: RecurrentLatentWorldModelEnsembleState,
        *,
        root: Path = REPO_ROOT,
    ) -> dict[str, object]:
        """Run the bound evaluator against one immutable supplied snapshot."""
        return build_recurrent_world_model_retention_report(
            model,
            state,
            self._config,
            self._protocol,
            root=root,
        )


__all__ = [
    "DEVELOPMENT_STATUS",
    "RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA",
    "RECURRENT_WORLD_MODEL_RETENTION_CONFIG_SCHEMA",
    "RECURRENT_WORLD_MODEL_RETENTION_PROTOCOL_SCHEMA",
    "RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA",
    "SOURCE_PATHS",
    "RecurrentWorldModelRetentionConfig",
    "RecurrentWorldModelRetentionEvaluator",
    "RecurrentWorldModelRetentionPhase",
    "RecurrentWorldModelRetentionProtocol",
    "RecurrentWorldModelRetentionValidation",
    "build_recurrent_world_model_retention_report",
    "canonical_recurrent_world_model_retention_report_bytes",
    "load_recurrent_world_model_retention_report",
    "load_recurrent_world_model_retention_snapshot_checkpoint",
    "reconstruct_recurrent_world_model_retention_summary",
    "recurrent_world_model_retention_source_snapshot",
    "save_recurrent_world_model_retention_report",
    "save_recurrent_world_model_retention_snapshot_checkpoint",
    "validate_recurrent_world_model_retention_report",
]
