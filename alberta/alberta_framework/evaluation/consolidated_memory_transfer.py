# mypy: disable-error-code="arg-type,attr-defined,call-arg,redundant-cast,type-var"
"""Development-only stress evaluation for consolidated memory.

The evaluator owns a fixed recurring schedule and starts every arm from one
immutable empty, source-bound snapshot.  Each event queries before it writes.
The full-memory and retrieval-ablation arms execute identical memory kernels;
the latter ignores the retrieved value.  A no-memory comparator receives the
same external events and opportunities but owns no persistent state.

Evaluator-only targets, phase labels, roles, and harm annotations never enter
the memory API.  Results are descriptive and remain ``not-assessed``: there
are no thresholds, promotion authority, efficacy claim, or SOTA conclusion.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, NoReturn, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Int

from alberta_framework.core.consolidated_memory import (
    SEMANTIC_KIND_AFFORDANCE,
    SEMANTIC_KIND_FACT,
    SEMANTIC_KIND_GVF,
    ConsolidatedMemory,
    ConsolidatedMemoryConfig,
    ConsolidatedMemoryState,
    ProceduralMemoryRecord,
    ProceduralMemoryRequest,
    SemanticMemoryRecord,
    SemanticMemoryRequest,
    canonical_memory_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CONSOLIDATED_MEMORY_TRANSFER_CONFIG_SCHEMA = "alberta.consolidated-memory-transfer.config.v1"
CONSOLIDATED_MEMORY_TRANSFER_PROTOCOL_SCHEMA = "alberta.consolidated-memory-transfer.protocol.v1"
CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA = "alberta.consolidated-memory-transfer.report.v1"
CONSOLIDATED_MEMORY_TRANSFER_CHECKPOINT_SCHEMA = (
    "alberta.consolidated-memory-transfer.checkpoint.v1"
)
CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS = "development-only-not-assessed"
CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS = "not-assessed"
CONSOLIDATED_MEMORY_TRANSFER_PROMOTION_AUTHORITY = False
CONSOLIDATED_MEMORY_TRANSFER_SCIENTIFIC_PROMOTION_ALLOWED = False

_MAX_EVENTS = 64
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
_SOURCE_PATHS = (
    Path("alberta_framework/core/consolidated_memory.py"),
    Path("alberta_framework/evaluation/consolidated_memory_transfer.py"),
)
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "the evaluator-owned finite schedule does not establish external validity",
    "exact-match precision is a diagnostic identity/target fact, not a calibrated threshold",
    "the no-memory comparator matches experience and opportunities but uses zero storage",
    "the retrieval ablation shares the same algorithm and capacity but masks its readout",
    "ConsolidatedMemory terminates updates at its configured signed-int32 max_operations cap",
    "one deterministic run establishes no efficacy, promotion, or SOTA claim",
)

RecordType = Literal["semantic", "procedural"]
_COMPILED_SCHEDULE_CACHE: dict[
    str, Callable[[ConsolidatedMemoryState], ConsolidatedMemoryState]
] = {}
_REPORT_CACHE: dict[str, dict[str, object]] = {}


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
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual), cast(list[object], expected), strict=True
                )
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


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented):
        raise ValueError(f"{name} must remain finite in float32")
    return value


def _float_tuple(value: object, *, name: str, allow_empty: bool = False) -> tuple[float, ...]:
    if type(value) not in {tuple, list}:
        raise ValueError(f"{name} must be a tuple or canonical JSON array")
    raw = cast(tuple[object, ...] | list[object], value)
    if not raw and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(raw))


def _optional_identifier(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name=name)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consolidated_memory_transfer_source_snapshot(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Return exact hashes for the complete source closure of this evaluator."""

    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


def consolidated_memory_transfer_runtime_identity() -> dict[str, object]:
    """Return the deterministic runtime identity recorded by development reports."""

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _tree_nbytes(tree: object) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(tree)
    )


def frozen_consolidated_memory_state_sha256(state: ConsolidatedMemoryState) -> str:
    """Hash every persistent leaf with explicit index, shape, dtype, and bytes."""

    if not isinstance(state, ConsolidatedMemoryState):
        raise TypeError("state must be ConsolidatedMemoryState")
    digest = hashlib.sha256()
    for index, raw_leaf in enumerate(jax.tree_util.tree_leaves(state)):
        leaf = np.asarray(jax.device_get(raw_leaf))
        digest.update(
            _canonical_json_bytes(
                {
                    "index": index,
                    "shape": list(leaf.shape),
                    "dtype": leaf.dtype.str,
                    "nbytes": int(leaf.nbytes),
                }
            )
        )
        digest.update(b"\0")
        digest.update(leaf.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _trees_exactly_equal(left: object, right: object) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(
            np.asarray(jax.device_get(left_leaf)), np.asarray(jax.device_get(right_leaf))
        )
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryTransferConfig:
    """Fixed memory construction, ablation readouts, bindings, and hard limits."""

    memory_config: ConsolidatedMemoryConfig
    semantic_reference: tuple[float, ...]
    procedural_reference: tuple[float, ...]
    source_binding: str
    semantic_namespace: str
    representation_revision: int
    source_revision: int
    max_events: int
    max_snapshot_bytes: int
    max_report_bytes: int

    SCHEMA_VERSION: ClassVar[str] = CONSOLIDATED_MEMORY_TRANSFER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.memory_config) is not ConsolidatedMemoryConfig:
            raise TypeError("memory_config must be an exact ConsolidatedMemoryConfig")
        ConsolidatedMemory(self.memory_config)
        if type(self.semantic_reference) is not tuple:
            raise TypeError("semantic_reference must be an exact tuple")
        if type(self.procedural_reference) is not tuple:
            raise TypeError("procedural_reference must be an exact tuple")
        if len(self.semantic_reference) != self.memory_config.semantic_payload_dim:
            raise ValueError("semantic_reference width differs from semantic payload width")
        if len(self.procedural_reference) != self.memory_config.procedural_outcome_dim:
            raise ValueError("procedural_reference width differs from procedural outcome width")
        _float_tuple(self.semantic_reference, name="semantic_reference")
        _float_tuple(self.procedural_reference, name="procedural_reference")
        _identifier(self.source_binding, name="source_binding")
        _identifier(self.semantic_namespace, name="semantic_namespace")
        _exact_int(self.representation_revision, name="representation_revision")
        _exact_int(self.source_revision, name="source_revision")
        _exact_int(self.max_events, name="max_events", minimum=1)
        _exact_int(self.max_snapshot_bytes, name="max_snapshot_bytes", minimum=1)
        _exact_int(self.max_report_bytes, name="max_report_bytes", minimum=1)
        if self.max_events > _MAX_EVENTS:
            raise ValueError("max_events exceeds the evaluator hard ceiling")
        if self.max_snapshot_bytes > _MAX_SNAPSHOT_BYTES:
            raise ValueError("max_snapshot_bytes exceeds the evaluator hard ceiling")
        if self.max_report_bytes > _MAX_REPORT_BYTES:
            raise ValueError("max_report_bytes exceeds the evaluator hard ceiling")

    @property
    def source_digest(self) -> Array:
        return canonical_memory_digest("consolidated-memory.source", self.source_binding)

    @property
    def semantic_namespace_digest(self) -> Array:
        return canonical_memory_digest("consolidated-memory.namespace", self.semantic_namespace)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "performance_thresholds_applied": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "memory_config": self.memory_config.to_config(),
            "semantic_reference": list(self.semantic_reference),
            "procedural_reference": list(self.procedural_reference),
            "source_binding": self.source_binding,
            "semantic_namespace": self.semantic_namespace,
            "representation_revision": self.representation_revision,
            "source_revision": self.source_revision,
            "max_events": self.max_events,
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "max_report_bytes": self.max_report_bytes,
        }

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedMemoryTransferConfig:
        raw = _mapping(value, name="config")
        expected = {
            "schema",
            "development_status",
            "assessment_status",
            "performance_thresholds_applied",
            "promotion_authority",
            "scientific_promotion_allowed",
            "memory_config",
            "semantic_reference",
            "procedural_reference",
            "source_binding",
            "semantic_namespace",
            "representation_revision",
            "source_revision",
            "max_events",
            "max_snapshot_bytes",
            "max_report_bytes",
        }
        if set(raw) != expected:
            raise ValueError("consolidated-memory transfer config fields differ from v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "performance_thresholds_applied": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }
        if any(
            not _strict_json_equal(raw[name], expected_value)
            for name, expected_value in fixed.items()
        ):
            raise ValueError("consolidated-memory transfer config fixed fields differ")
        result = cls(
            memory_config=ConsolidatedMemoryConfig.from_config(raw["memory_config"]),
            semantic_reference=_float_tuple(raw["semantic_reference"], name="semantic_reference"),
            procedural_reference=_float_tuple(
                raw["procedural_reference"], name="procedural_reference"
            ),
            source_binding=_identifier(raw["source_binding"], name="source_binding"),
            semantic_namespace=_identifier(raw["semantic_namespace"], name="semantic_namespace"),
            representation_revision=_exact_int(
                raw["representation_revision"], name="representation_revision"
            ),
            source_revision=_exact_int(raw["source_revision"], name="source_revision"),
            max_events=_exact_int(raw["max_events"], name="max_events", minimum=1),
            max_snapshot_bytes=_exact_int(
                raw["max_snapshot_bytes"], name="max_snapshot_bytes", minimum=1
            ),
            max_report_bytes=_exact_int(
                raw["max_report_bytes"], name="max_report_bytes", minimum=1
            ),
        )
        if not _strict_json_equal(dict(raw), result.to_config()):
            raise ValueError("consolidated-memory transfer config is noncanonical")
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryTransferEvent:
    """One evaluator-owned query-before-write stress event."""

    event_id: str
    phase_id: str
    regime_id: str
    role: str
    record_type: RecordType
    semantic_label: str
    query_generation: int
    write_generation: int
    semantic_kind: int
    query_provenance_label: str
    write_provenance_label: str
    record_payload: tuple[float, ...]
    confidence: float
    evidence: float
    expected_target: tuple[float, ...]
    succeeded: bool
    outcome: tuple[float, ...]
    query_lifecycle_label: str | None
    write_lifecycle_label: str | None
    query_lifecycle_generation: int
    write_lifecycle_generation: int
    query_lifecycle_revision: int
    write_lifecycle_revision: int

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "phase_id",
            "regime_id",
            "role",
            "semantic_label",
            "query_provenance_label",
            "write_provenance_label",
        ):
            _identifier(getattr(self, name), name=name)
        if self.record_type not in {"semantic", "procedural"}:
            raise ValueError("record_type must be semantic or procedural")
        _exact_int(self.query_generation, name="query_generation")
        _exact_int(self.write_generation, name="write_generation")
        if self.record_type == "semantic":
            if self.semantic_kind not in {
                SEMANTIC_KIND_GVF,
                SEMANTIC_KIND_FACT,
                SEMANTIC_KIND_AFFORDANCE,
            }:
                raise ValueError("semantic event kind is invalid")
        elif self.semantic_kind != -1:
            raise ValueError("procedural event semantic_kind must be -1")
        _float_tuple(self.record_payload, name="record_payload")
        _finite_float(self.confidence, name="confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        _finite_float(self.evidence, name="evidence")
        _float_tuple(self.expected_target, name="expected_target")
        if type(self.succeeded) is not bool:
            raise TypeError("succeeded must be an exact bool")
        _float_tuple(self.outcome, name="outcome")
        query_link = _optional_identifier(self.query_lifecycle_label, name="query_lifecycle_label")
        write_link = _optional_identifier(self.write_lifecycle_label, name="write_lifecycle_label")
        if self.record_type == "semantic":
            if query_link is not None or write_link is not None:
                raise ValueError("semantic events cannot carry lifecycle labels")
            if any(
                value != -1
                for value in (
                    self.query_lifecycle_generation,
                    self.write_lifecycle_generation,
                    self.query_lifecycle_revision,
                    self.write_lifecycle_revision,
                )
            ):
                raise ValueError("semantic lifecycle integers must be -1")
        else:
            if query_link is None or write_link is None:
                raise ValueError("procedural events require query and write lifecycle labels")
            for name in (
                "query_lifecycle_generation",
                "write_lifecycle_generation",
                "query_lifecycle_revision",
                "write_lifecycle_revision",
            ):
                _exact_int(getattr(self, name), name=name)

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["record_payload"] = list(self.record_payload)
        payload["expected_target"] = list(self.expected_target)
        payload["outcome"] = list(self.outcome)
        return payload

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedMemoryTransferEvent:
        raw = _mapping(value, name="event")
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(raw) != expected:
            raise ValueError("consolidated-memory transfer event fields differ from v1")
        kwargs = dict(raw)
        for name in ("record_payload", "expected_target", "outcome"):
            kwargs[name] = _float_tuple(raw[name], name=name)
        result = cls(**cast(Any, kwargs))
        if not _strict_json_equal(dict(raw), result.to_config()):
            raise ValueError("consolidated-memory transfer event is noncanonical")
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryTransferProtocol:
    """Frozen evaluator-owned recurring-regime schedule."""

    protocol_id: str
    events: tuple[ConsolidatedMemoryTransferEvent, ...]

    SCHEMA_VERSION: ClassVar[str] = CONSOLIDATED_MEMORY_TRANSFER_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id")
        if type(self.events) is not tuple or not self.events:
            raise TypeError("events must be a nonempty exact tuple")
        if any(type(event) is not ConsolidatedMemoryTransferEvent for event in self.events):
            raise TypeError("events must contain exact transfer events")
        event_ids = tuple(event.event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event ids must be unique")

    def to_config(self) -> dict[str, object]:
        events = [event.to_config() for event in self.events]
        return {
            "schema": self.SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "evaluator_owned_regime_labels": True,
            "regime_labels_visible_to_memory": False,
            "query_precedes_write": True,
            "events": events,
            "external_experience_sha256": _canonical_sha256(events),
        }

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedMemoryTransferProtocol:
        raw = _mapping(value, name="protocol")
        expected = {
            "schema",
            "protocol_id",
            "development_status",
            "assessment_status",
            "evaluator_owned_regime_labels",
            "regime_labels_visible_to_memory",
            "query_precedes_write",
            "events",
            "external_experience_sha256",
        }
        if set(raw) != expected:
            raise ValueError("consolidated-memory transfer protocol fields differ from v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "evaluator_owned_regime_labels": True,
            "regime_labels_visible_to_memory": False,
            "query_precedes_write": True,
        }
        if any(
            not _strict_json_equal(raw[name], expected_value)
            for name, expected_value in fixed.items()
        ):
            raise ValueError("consolidated-memory transfer protocol fixed fields differ")
        raw_events = _list(raw["events"], name="events")
        events = tuple(ConsolidatedMemoryTransferEvent.from_config(event) for event in raw_events)
        if raw["external_experience_sha256"] != _canonical_sha256(
            [event.to_config() for event in events]
        ):
            raise ValueError("external experience digest differs")
        result = cls(
            protocol_id=_identifier(raw["protocol_id"], name="protocol_id"),
            events=events,
        )
        if not _strict_json_equal(dict(raw), result.to_config()):
            raise ValueError("consolidated-memory transfer protocol is noncanonical")
        return result


def default_consolidated_memory_transfer_config() -> ConsolidatedMemoryTransferConfig:
    """Return the bounded development configuration; it contains no thresholds."""

    return ConsolidatedMemoryTransferConfig(
        memory_config=ConsolidatedMemoryConfig(
            semantic_capacity=2,
            procedural_capacity=2,
            semantic_payload_dim=2,
            procedural_payload_dim=2,
            procedural_outcome_dim=1,
            semantic_max_age=4,
            procedural_max_age=3,
            max_operations=128,
            semantic_min_confidence=0.5,
            procedural_min_confidence=0.5,
        ),
        semantic_reference=(0.0, 0.0),
        procedural_reference=(0.0,),
        source_binding="consolidated-memory-transfer-source-v1",
        semantic_namespace="consolidated-memory-transfer-semantics-v1",
        representation_revision=1,
        source_revision=1,
        max_events=24,
        max_snapshot_bytes=1_000_000,
        max_report_bytes=4_000_000,
    )


def _semantic_event(
    event_id: str,
    phase_id: str,
    regime_id: str,
    role: str,
    semantic_label: str,
    *,
    query_generation: int,
    write_generation: int | None = None,
    query_provenance: str,
    write_provenance: str | None = None,
    payload: tuple[float, float],
    target: tuple[float, float],
    confidence: float = 0.9,
    evidence: float = 1.0,
) -> ConsolidatedMemoryTransferEvent:
    return ConsolidatedMemoryTransferEvent(
        event_id=event_id,
        phase_id=phase_id,
        regime_id=regime_id,
        role=role,
        record_type="semantic",
        semantic_label=semantic_label,
        query_generation=query_generation,
        write_generation=(query_generation if write_generation is None else write_generation),
        semantic_kind=SEMANTIC_KIND_FACT,
        query_provenance_label=query_provenance,
        write_provenance_label=(query_provenance if write_provenance is None else write_provenance),
        record_payload=payload,
        confidence=confidence,
        evidence=evidence,
        expected_target=target,
        succeeded=False,
        outcome=(0.0,),
        query_lifecycle_label=None,
        write_lifecycle_label=None,
        query_lifecycle_generation=-1,
        write_lifecycle_generation=-1,
        query_lifecycle_revision=-1,
        write_lifecycle_revision=-1,
    )


def _procedural_event(
    event_id: str,
    phase_id: str,
    regime_id: str,
    role: str,
    semantic_label: str,
    *,
    query_generation: int,
    write_generation: int | None = None,
    query_provenance: str,
    write_provenance: str | None = None,
    outcome: float,
    target: float,
    succeeded: bool,
    query_lifecycle: str,
    write_lifecycle: str | None = None,
    query_lifecycle_generation: int,
    write_lifecycle_generation: int | None = None,
    query_lifecycle_revision: int = 0,
    write_lifecycle_revision: int | None = None,
    confidence: float = 0.9,
    evidence: float = 1.0,
) -> ConsolidatedMemoryTransferEvent:
    return ConsolidatedMemoryTransferEvent(
        event_id=event_id,
        phase_id=phase_id,
        regime_id=regime_id,
        role=role,
        record_type="procedural",
        semantic_label=semantic_label,
        query_generation=query_generation,
        write_generation=(query_generation if write_generation is None else write_generation),
        semantic_kind=-1,
        query_provenance_label=query_provenance,
        write_provenance_label=(query_provenance if write_provenance is None else write_provenance),
        record_payload=(outcome, 1.0),
        confidence=confidence,
        evidence=evidence,
        expected_target=(target,),
        succeeded=succeeded,
        outcome=(outcome,),
        query_lifecycle_label=query_lifecycle,
        write_lifecycle_label=(query_lifecycle if write_lifecycle is None else write_lifecycle),
        query_lifecycle_generation=query_lifecycle_generation,
        write_lifecycle_generation=(
            query_lifecycle_generation
            if write_lifecycle_generation is None
            else write_lifecycle_generation
        ),
        query_lifecycle_revision=query_lifecycle_revision,
        write_lifecycle_revision=(
            query_lifecycle_revision
            if write_lifecycle_revision is None
            else write_lifecycle_revision
        ),
    )


def default_consolidated_memory_transfer_protocol() -> ConsolidatedMemoryTransferProtocol:
    """Return the fixed v1 recurrence, interference, eviction, and recovery trace."""

    initial = (
        _semantic_event(
            "initial-semantic-a",
            "initial",
            "regime-a",
            "cold-semantic",
            "semantic-a",
            query_generation=0,
            query_provenance="semantic-a-v0",
            payload=(1.0, -1.0),
            target=(1.0, -1.0),
        ),
        _semantic_event(
            "initial-semantic-a-recurrence",
            "initial",
            "regime-a",
            "compatible-recurrence",
            "semantic-a",
            query_generation=0,
            query_provenance="semantic-a-v0",
            payload=(1.0, -1.0),
            target=(1.0, -1.0),
        ),
        _procedural_event(
            "initial-walk",
            "initial",
            "regime-a",
            "cold-procedural",
            "skill-walk",
            query_generation=0,
            query_provenance="walk-v0",
            outcome=1.0,
            target=1.0,
            succeeded=True,
            query_lifecycle="option-walk-v0",
            query_lifecycle_generation=0,
        ),
        _procedural_event(
            "initial-walk-recurrence",
            "initial",
            "regime-a",
            "compatible-recurrence",
            "skill-walk",
            query_generation=0,
            query_provenance="walk-v0",
            outcome=1.0,
            target=1.0,
            succeeded=True,
            query_lifecycle="option-walk-v0",
            query_lifecycle_generation=0,
        ),
    )
    interference = (
        _semantic_event(
            "interference-misleading-cold",
            "interference",
            "regime-b",
            "misleading-write",
            "semantic-b",
            query_generation=0,
            query_provenance="semantic-b-v0",
            payload=(-1.0, -1.0),
            target=(1.0, 1.0),
        ),
        _semantic_event(
            "interference-misleading-probe",
            "interference",
            "regime-b",
            "misleading-probe",
            "semantic-b",
            query_generation=0,
            query_provenance="semantic-b-v0",
            payload=(1.0, 1.0),
            target=(1.0, 1.0),
        ),
        _semantic_event(
            "interference-semantic-generation",
            "interference",
            "regime-b",
            "semantic-generation-shift",
            "semantic-a",
            query_generation=1,
            query_provenance="semantic-a-v1",
            payload=(2.0, -2.0),
            target=(2.0, -2.0),
        ),
        _semantic_event(
            "interference-semantic-generation-recurrence",
            "interference",
            "regime-b",
            "compatible-new-generation",
            "semantic-a",
            query_generation=1,
            query_provenance="semantic-a-v1",
            payload=(2.0, -2.0),
            target=(2.0, -2.0),
        ),
        _semantic_event(
            "interference-provenance-mismatch",
            "interference",
            "regime-b",
            "provenance-mismatch",
            "semantic-a",
            query_generation=1,
            query_provenance="semantic-a-counterfeit",
            write_provenance="semantic-a-v1",
            payload=(2.0, -2.0),
            target=(2.0, -2.0),
        ),
        _procedural_event(
            "interference-success-failure-shift",
            "interference",
            "regime-b",
            "procedural-outcome-shift",
            "skill-walk",
            query_generation=0,
            query_provenance="walk-v0",
            outcome=-1.0,
            target=-1.0,
            succeeded=False,
            query_lifecycle="option-walk-v0",
            query_lifecycle_generation=0,
        ),
        _procedural_event(
            "interference-procedural-generation",
            "interference",
            "regime-b",
            "procedural-generation-shift",
            "skill-walk",
            query_generation=1,
            query_provenance="walk-v1",
            outcome=-1.0,
            target=-1.0,
            succeeded=False,
            query_lifecycle="option-walk-v1",
            query_lifecycle_generation=1,
            query_lifecycle_revision=1,
        ),
        _semantic_event(
            "interference-eviction-c",
            "interference",
            "regime-b",
            "eviction-pressure",
            "semantic-c",
            query_generation=0,
            query_provenance="semantic-c-v0",
            payload=(3.0, 3.0),
            target=(3.0, 3.0),
        ),
        _semantic_event(
            "interference-eviction-d",
            "interference",
            "regime-b",
            "eviction-pressure",
            "semantic-d",
            query_generation=0,
            query_provenance="semantic-d-v0",
            payload=(4.0, 4.0),
            target=(4.0, 4.0),
        ),
    )
    # Exercise the outcome shift while the v0 skill is still fresh, then leave
    # the explicit v1 revision untouched long enough for the return-phase
    # stale-skill probe.  The remaining events create that exact elapsed age.
    interference = (
        interference[5],
        interference[6],
        *interference[:5],
        *interference[7:],
    )
    returned = (
        _semantic_event(
            "return-semantic-a",
            "return",
            "regime-a",
            "retained-semantic-return",
            "semantic-a",
            query_generation=1,
            query_provenance="semantic-a-v1",
            payload=(2.0, -2.0),
            target=(2.0, -2.0),
        ),
        _semantic_event(
            "return-semantic-a-recurrence",
            "return",
            "regime-a",
            "retained-semantic-recurrence",
            "semantic-a",
            query_generation=1,
            query_provenance="semantic-a-v1",
            payload=(2.0, -2.0),
            target=(2.0, -2.0),
        ),
        _procedural_event(
            "return-stale-walk",
            "return",
            "regime-a",
            "stale-skill-probe",
            "skill-walk",
            query_generation=1,
            write_generation=2,
            query_provenance="walk-v1",
            write_provenance="walk-v2",
            outcome=-1.0,
            target=-1.0,
            succeeded=True,
            query_lifecycle="option-walk-v1",
            write_lifecycle="option-walk-v2",
            query_lifecycle_generation=1,
            write_lifecycle_generation=2,
            query_lifecycle_revision=1,
            write_lifecycle_revision=2,
        ),
        _procedural_event(
            "return-walk-recovery",
            "return",
            "regime-a",
            "procedural-recovery",
            "skill-walk",
            query_generation=2,
            query_provenance="walk-v2",
            outcome=-1.0,
            target=-1.0,
            succeeded=True,
            query_lifecycle="option-walk-v2",
            query_lifecycle_generation=2,
            query_lifecycle_revision=2,
        ),
    )
    return ConsolidatedMemoryTransferProtocol(
        protocol_id="consolidated-memory-transfer-stress-v1",
        events=(*initial, *interference, *returned),
    )


def frozen_consolidated_memory_transfer_protocol_sha256() -> str:
    """Return the integrity digest of the evaluator-owned v1 schedule."""

    return _canonical_sha256(default_consolidated_memory_transfer_protocol().to_config())


def _validate_protocol_dimensions(
    config: ConsolidatedMemoryTransferConfig,
    protocol: ConsolidatedMemoryTransferProtocol,
) -> None:
    if len(protocol.events) > config.max_events:
        raise ValueError("protocol event count exceeds config.max_events")
    required_roles = {
        "compatible-recurrence",
        "misleading-probe",
        "semantic-generation-shift",
        "provenance-mismatch",
        "procedural-outcome-shift",
        "procedural-generation-shift",
        "eviction-pressure",
        "retained-semantic-return",
        "retained-semantic-recurrence",
        "stale-skill-probe",
        "procedural-recovery",
    }
    roles = {event.role for event in protocol.events}
    if not required_roles <= roles:
        raise ValueError("protocol omits required recurrence or stress roles")
    phases = tuple(dict.fromkeys(event.phase_id for event in protocol.events))
    if phases != ("initial", "interference", "return"):
        raise ValueError("protocol phases must be exactly initial/interference/return")
    cfg = config.memory_config
    for event in protocol.events:
        payload_width = (
            cfg.semantic_payload_dim
            if event.record_type == "semantic"
            else cfg.procedural_payload_dim
        )
        target_width = (
            cfg.semantic_payload_dim
            if event.record_type == "semantic"
            else cfg.procedural_outcome_dim
        )
        if len(event.record_payload) != payload_width:
            raise ValueError(f"event {event.event_id} record payload width differs")
        if len(event.expected_target) != target_width:
            raise ValueError(f"event {event.event_id} expected target width differs")
        if len(event.outcome) != cfg.procedural_outcome_dim:
            raise ValueError(f"event {event.event_id} procedural outcome width differs")


def _semantic_digest(event: ConsolidatedMemoryTransferEvent) -> Array:
    return canonical_memory_digest("consolidated-memory.semantic", event.semantic_label)


def _provenance_digest(label: str) -> Array:
    return canonical_memory_digest("consolidated-memory.provenance", label)


def _lifecycle_digest(label: str | None) -> Array:
    if label is None:
        return jnp.zeros((32,), dtype=jnp.uint8)
    return canonical_memory_digest("consolidated-memory.lifecycle", label)


def _semantic_request(
    config: ConsolidatedMemoryTransferConfig,
    event: ConsolidatedMemoryTransferEvent,
) -> SemanticMemoryRequest:
    return SemanticMemoryRequest(
        semantic_digest=_semantic_digest(event),
        generation=jnp.asarray(event.query_generation, dtype=jnp.int32),
        kind=jnp.asarray(event.semantic_kind, dtype=jnp.int32),
        provenance_digest=_provenance_digest(event.query_provenance_label),
        representation_revision=jnp.asarray(config.representation_revision, dtype=jnp.int32),
        source_revision=jnp.asarray(config.source_revision, dtype=jnp.int32),
    )


def _semantic_record(
    config: ConsolidatedMemoryTransferConfig,
    event: ConsolidatedMemoryTransferEvent,
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        semantic_digest=_semantic_digest(event),
        generation=jnp.asarray(event.write_generation, dtype=jnp.int32),
        kind=jnp.asarray(event.semantic_kind, dtype=jnp.int32),
        payload=jnp.asarray(event.record_payload, dtype=jnp.float32),
        confidence=jnp.asarray(event.confidence, dtype=jnp.float32),
        provenance_digest=_provenance_digest(event.write_provenance_label),
        representation_revision=jnp.asarray(config.representation_revision, dtype=jnp.int32),
        source_revision=jnp.asarray(config.source_revision, dtype=jnp.int32),
        evidence=jnp.asarray(event.evidence, dtype=jnp.float32),
    )


def _procedural_request(
    config: ConsolidatedMemoryTransferConfig,
    event: ConsolidatedMemoryTransferEvent,
) -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=_semantic_digest(event),
        generation=jnp.asarray(event.query_generation, dtype=jnp.int32),
        provenance_digest=_provenance_digest(event.query_provenance_label),
        representation_revision=jnp.asarray(config.representation_revision, dtype=jnp.int32),
        source_revision=jnp.asarray(config.source_revision, dtype=jnp.int32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_lifecycle_digest(event.query_lifecycle_label),
        lifecycle_generation=jnp.asarray(event.query_lifecycle_generation, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(event.query_lifecycle_revision, dtype=jnp.int32),
    )


def _procedural_record(
    config: ConsolidatedMemoryTransferConfig,
    event: ConsolidatedMemoryTransferEvent,
) -> ProceduralMemoryRecord:
    return ProceduralMemoryRecord(
        semantic_digest=_semantic_digest(event),
        generation=jnp.asarray(event.write_generation, dtype=jnp.int32),
        payload=jnp.asarray(event.record_payload, dtype=jnp.float32),
        confidence=jnp.asarray(event.confidence, dtype=jnp.float32),
        provenance_digest=_provenance_digest(event.write_provenance_label),
        representation_revision=jnp.asarray(config.representation_revision, dtype=jnp.int32),
        source_revision=jnp.asarray(config.source_revision, dtype=jnp.int32),
        evidence=jnp.asarray(event.evidence, dtype=jnp.float32),
        succeeded=jnp.asarray(event.succeeded, dtype=jnp.bool_),
        outcome=jnp.asarray(event.outcome, dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_lifecycle_digest(event.write_lifecycle_label),
        lifecycle_generation=jnp.asarray(event.write_lifecycle_generation, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(event.write_lifecycle_revision, dtype=jnp.int32),
    )


@chex.dataclass(frozen=True)
class ConsolidatedMemoryTransferRunState:
    """Checkpointable pair of matched memory arms and their next event index."""

    event_index: Int[Array, ""]
    full_memory_state: ConsolidatedMemoryState
    retrieval_ablation_state: ConsolidatedMemoryState


def frozen_consolidated_memory_transfer_run_state_sha256(
    state: ConsolidatedMemoryTransferRunState,
) -> str:
    """Hash a run state including exact event position and both arm states."""

    if not isinstance(state, ConsolidatedMemoryTransferRunState):
        raise TypeError("state must be ConsolidatedMemoryTransferRunState")
    digest = hashlib.sha256()
    for index, raw_leaf in enumerate(jax.tree_util.tree_leaves(state)):
        leaf = np.asarray(jax.device_get(raw_leaf))
        digest.update(
            _canonical_json_bytes(
                {"index": index, "shape": list(leaf.shape), "dtype": leaf.dtype.str}
            )
        )
        digest.update(leaf.tobytes(order="C"))
    return digest.hexdigest()


def _state_binding_kwargs(config: ConsolidatedMemoryTransferConfig) -> dict[str, object]:
    return {
        "source_digest": config.source_digest,
        "semantic_namespace_digest": config.semantic_namespace_digest,
        "representation_revision": config.representation_revision,
        "source_revision": config.source_revision,
    }


class ConsolidatedMemoryTransferEvaluator:
    """Strict runner for the fixed development-only transfer schedule."""

    def __init__(
        self,
        config: ConsolidatedMemoryTransferConfig | None = None,
        protocol: ConsolidatedMemoryTransferProtocol | None = None,
    ) -> None:
        self.config = config or default_consolidated_memory_transfer_config()
        self.protocol = protocol or default_consolidated_memory_transfer_protocol()
        if type(self.config) is not ConsolidatedMemoryTransferConfig:
            raise TypeError("config must be an exact ConsolidatedMemoryTransferConfig")
        if type(self.protocol) is not ConsolidatedMemoryTransferProtocol:
            raise TypeError("protocol must be an exact ConsolidatedMemoryTransferProtocol")
        _validate_protocol_dimensions(self.config, self.protocol)
        if (
            _canonical_sha256(self.protocol.to_config())
            != frozen_consolidated_memory_transfer_protocol_sha256()
        ):
            raise ValueError("protocol differs from the evaluator-owned frozen v1 schedule")
        self.memory = ConsolidatedMemory(self.config.memory_config)
        self._initial_memory_state = self.memory.init(
            source_digest=self.config.source_digest,
            semantic_namespace_digest=self.config.semantic_namespace_digest,
            representation_revision=self.config.representation_revision,
            source_revision=self.config.source_revision,
        )
        snapshot_bytes = _tree_nbytes(self._initial_memory_state)
        if snapshot_bytes > self.config.max_snapshot_bytes:
            raise ValueError("empty source-bound snapshot exceeds max_snapshot_bytes")
        if not bool(
            jax.device_get(
                self.memory.validate_state(
                    self._initial_memory_state,
                    **_state_binding_kwargs(self.config),
                )
            )
        ):
            raise ValueError("empty source-bound snapshot is invalid")

    @property
    def initial_memory_state(self) -> ConsolidatedMemoryState:
        """Return the immutable empty snapshot (the core state itself is frozen)."""

        return self._initial_memory_state

    def init(self) -> ConsolidatedMemoryTransferRunState:
        return ConsolidatedMemoryTransferRunState(
            event_index=jnp.asarray(0, dtype=jnp.int32),
            full_memory_state=self._initial_memory_state,
            retrieval_ablation_state=self._initial_memory_state,
        )

    def _apply_event(
        self,
        state: ConsolidatedMemoryState,
        event: ConsolidatedMemoryTransferEvent,
    ) -> object:
        if event.record_type == "semantic":
            return self.memory.semantic_step(
                state,
                _semantic_request(self.config, event),
                _semantic_record(self.config, event),
            )
        return self.memory.procedural_step(
            state,
            _procedural_request(self.config, event),
            _procedural_record(self.config, event),
        )

    def _run_state_valid(self, state: ConsolidatedMemoryTransferRunState) -> bool:
        if not isinstance(state, ConsolidatedMemoryTransferRunState):
            return False
        if state.event_index.shape != () or state.event_index.dtype != jnp.int32:
            return False
        index = int(jax.device_get(state.event_index))
        if not 0 <= index <= len(self.protocol.events):
            return False
        bindings = _state_binding_kwargs(self.config)
        full_valid = self.memory.validate_state(state.full_memory_state, **bindings)
        ablation_valid = self.memory.validate_state(state.retrieval_ablation_state, **bindings)
        return (
            bool(jax.device_get(full_valid))
            and bool(jax.device_get(ablation_valid))
            and _trees_exactly_equal(state.full_memory_state, state.retrieval_ablation_state)
        )

    def advance(
        self, state: ConsolidatedMemoryTransferRunState, *, steps: int
    ) -> ConsolidatedMemoryTransferRunState:
        """Advance a valid run by a bounded exact number of schedule events."""

        _exact_int(steps, name="steps")
        if not self._run_state_valid(state):
            raise ValueError("run state is invalid or its matched arms differ")
        start = int(jax.device_get(state.event_index))
        stop = min(start + steps, len(self.protocol.events))
        full = state.full_memory_state
        ablation = state.retrieval_ablation_state
        for event in self.protocol.events[start:stop]:
            full_result = cast(Any, self._apply_event(full, event))
            ablation_result = cast(Any, self._apply_event(ablation, event))
            full = full_result.state
            ablation = ablation_result.state
            if not _trees_exactly_equal(full_result, ablation_result):
                raise RuntimeError("matched full and retrieval-ablation kernels diverged")
        result = ConsolidatedMemoryTransferRunState(
            event_index=jnp.asarray(stop, dtype=jnp.int32),
            full_memory_state=full,
            retrieval_ablation_state=ablation,
        )
        if not self._run_state_valid(result):
            raise RuntimeError("advance produced an invalid matched run state")
        return result

    def checkpoint_payload(self, state: ConsolidatedMemoryTransferRunState) -> dict[str, object]:
        """Return a strict checkpoint bound to config, protocol, source, and position."""

        if not self._run_state_valid(state):
            raise ValueError("cannot checkpoint an invalid transfer run state")
        bindings = _state_binding_kwargs(self.config)
        return {
            "schema": CONSOLIDATED_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "config": self.config.to_config(),
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(self.protocol.to_config()),
            "source_sha256": consolidated_memory_transfer_source_snapshot(),
            "runtime_sha256": _canonical_sha256(consolidated_memory_transfer_runtime_identity()),
            "event_index": int(jax.device_get(state.event_index)),
            "full_memory": self.memory.checkpoint_payload(state.full_memory_state, **bindings),
            "retrieval_ablation": self.memory.checkpoint_payload(
                state.retrieval_ablation_state, **bindings
            ),
            "run_state_sha256": frozen_consolidated_memory_transfer_run_state_sha256(state),
        }

    def restore_checkpoint(self, payload: object) -> ConsolidatedMemoryTransferRunState:
        """Restore only an integrity-bound checkpoint equal to exact prefix replay."""

        raw = _mapping(payload, name="checkpoint")
        expected = {
            "schema",
            "development_status",
            "assessment_status",
            "config",
            "config_sha256",
            "protocol_sha256",
            "source_sha256",
            "runtime_sha256",
            "event_index",
            "full_memory",
            "retrieval_ablation",
            "run_state_sha256",
        }
        if set(raw) != expected:
            raise ValueError("transfer checkpoint fields differ from v1")
        fixed = {
            "schema": CONSOLIDATED_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(self.protocol.to_config()),
            "source_sha256": consolidated_memory_transfer_source_snapshot(),
            "runtime_sha256": _canonical_sha256(consolidated_memory_transfer_runtime_identity()),
        }
        if any(not _strict_json_equal(raw[name], value) for name, value in fixed.items()):
            raise ValueError("transfer checkpoint binding differs")
        if not _strict_json_equal(raw["config"], self.config.to_config()):
            raise ValueError("transfer checkpoint config differs")
        index = _exact_int(raw["event_index"], name="event_index")
        if index > len(self.protocol.events):
            raise ValueError("transfer checkpoint event_index is out of range")
        bindings = _state_binding_kwargs(self.config)
        full = self.memory.restore_checkpoint(raw["full_memory"], **bindings)
        ablation = self.memory.restore_checkpoint(raw["retrieval_ablation"], **bindings)
        state = ConsolidatedMemoryTransferRunState(
            event_index=jnp.asarray(index, dtype=jnp.int32),
            full_memory_state=full,
            retrieval_ablation_state=ablation,
        )
        if raw["run_state_sha256"] != frozen_consolidated_memory_transfer_run_state_sha256(state):
            raise ValueError("transfer checkpoint run-state SHA differs")
        expected_state = self.advance(self.init(), steps=index)
        if not _trees_exactly_equal(state, expected_state):
            raise ValueError("transfer checkpoint differs from exact causal prefix replay")
        return state


def _scalar_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)).reshape(()))


def _scalar_int(value: object) -> int:
    return int(np.asarray(jax.device_get(value)).reshape(()))


def _array_floats(value: object) -> list[float]:
    array = np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1)
    result = [float(item) for item in array]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("non-finite kernel output cannot enter the report")
    return result


def _digest_hex(value: object) -> str:
    array = np.asarray(jax.device_get(value), dtype=np.uint8).reshape((32,))
    return array.tobytes().hex()


def _mean_squared_error(prediction: Sequence[float], target: Sequence[float]) -> float:
    if not prediction or len(prediction) != len(target):
        raise ValueError("prediction and target widths must match and be nonempty")
    return math.fsum(
        (left - right) ** 2 for left, right in zip(prediction, target, strict=True)
    ) / len(target)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _pre_record_metadata(
    state: ConsolidatedMemoryState,
    event: ConsolidatedMemoryTransferEvent,
) -> dict[str, object]:
    identity = np.asarray(jax.device_get(_semantic_digest(event)), dtype=np.uint8)
    records = state.semantic if event.record_type == "semantic" else state.procedural
    occupied = np.asarray(jax.device_get(records.occupied), dtype=np.bool_)
    digests = np.asarray(jax.device_get(records.semantic_digests), dtype=np.uint8)
    matches = np.flatnonzero(occupied & np.all(digests == identity[None, :], axis=1))
    if len(matches) == 0:
        return {"identity_found": False}
    slot = int(matches[0])
    result: dict[str, object] = {
        "identity_found": True,
        "slot": slot,
        "generation": int(np.asarray(jax.device_get(records.generations[slot]))),
        "provenance_sha256": _digest_hex(records.provenance_digests[slot]),
        "valid": bool(np.asarray(jax.device_get(records.valid[slot]))),
        "stale": bool(np.asarray(jax.device_get(records.stale[slot]))),
        "invalidated": bool(np.asarray(jax.device_get(records.invalidated[slot]))),
    }
    if event.record_type == "procedural":
        procedural = state.procedural
        result.update(
            {
                "lifecycle_sha256": _digest_hex(procedural.lifecycle_digests[slot]),
                "lifecycle_generation": int(
                    np.asarray(jax.device_get(procedural.lifecycle_generations[slot]))
                ),
                "lifecycle_revision": int(
                    np.asarray(jax.device_get(procedural.lifecycle_revisions[slot]))
                ),
            }
        )
    return result


def _abstention_reason(
    event: ConsolidatedMemoryTransferEvent,
    retrieval: object,
    pre_metadata: Mapping[str, object],
) -> str | None:
    if _scalar_bool(cast(Any, retrieval).accepted):
        return None
    if not _scalar_bool(cast(Any, retrieval).identity_found):
        return "identity-missing"
    if not _scalar_bool(cast(Any, retrieval).compatible):
        if pre_metadata.get("generation") != event.query_generation:
            return "generation-mismatch"
        expected_provenance = _digest_hex(_provenance_digest(event.query_provenance_label))
        if pre_metadata.get("provenance_sha256") != expected_provenance:
            return "provenance-mismatch"
        if event.record_type == "procedural":
            expected_lifecycle = _digest_hex(_lifecycle_digest(event.query_lifecycle_label))
            if (
                pre_metadata.get("lifecycle_sha256") != expected_lifecycle
                or pre_metadata.get("lifecycle_generation") != event.query_lifecycle_generation
                or pre_metadata.get("lifecycle_revision") != event.query_lifecycle_revision
            ):
                return "lifecycle-mismatch"
        return "metadata-mismatch"
    if not _scalar_bool(cast(Any, retrieval).fresh):
        return "stale-or-invalidated"
    if not _scalar_bool(cast(Any, retrieval).confidence_ok):
        return "confidence-gate"
    if not _scalar_bool(cast(Any, retrieval).request_valid):
        return "invalid-request"
    if not _scalar_bool(cast(Any, retrieval).state_valid):
        return "invalid-state"
    return "transaction-unavailable"


def _trace_event(
    evaluator: ConsolidatedMemoryTransferEvaluator,
    event_index: int,
    event: ConsolidatedMemoryTransferEvent,
    pre_state: ConsolidatedMemoryState,
    result: object,
) -> dict[str, object]:
    retrieval = cast(Any, result).retrieval
    write = cast(Any, result).write
    post_state = cast(Any, result).state
    accepted = _scalar_bool(retrieval.accepted)
    if event.record_type == "semantic":
        retrieved = _array_floats(retrieval.payload)
        reference = list(evaluator.config.semantic_reference)
    else:
        retrieved = _array_floats(retrieval.outcome_mean)
        reference = list(evaluator.config.procedural_reference)
    prediction = retrieved if accepted else reference
    target = list(event.expected_target)
    memory_error = _mean_squared_error(prediction, target)
    ablation_error = _mean_squared_error(reference, target)
    signed_gain = ablation_error - memory_error
    pre_metadata = _pre_record_metadata(pre_state, event)
    reason = _abstention_reason(event, retrieval, pre_metadata)
    retrieved_provenance: str | None = None
    provenance_matches: bool | None = None
    if accepted:
        slot = _scalar_int(retrieval.slot)
        records = pre_state.semantic if event.record_type == "semantic" else pre_state.procedural
        retrieved_provenance = _digest_hex(records.provenance_digests[slot])
        provenance_matches = retrieved_provenance == _digest_hex(
            _provenance_digest(event.query_provenance_label)
        )
    evicted_provenance: str | None = None
    evicted_semantic: str | None = None
    if _scalar_bool(write.replaced):
        slot = _scalar_int(write.slot)
        records = pre_state.semantic if event.record_type == "semantic" else pre_state.procedural
        evicted_provenance = _digest_hex(records.provenance_digests[slot])
        evicted_semantic = _digest_hex(records.semantic_digests[slot])
    accounting = evaluator.memory.accounting(post_state)
    return {
        "event_index": event_index,
        "event_id": event.event_id,
        "phase_id": event.phase_id,
        "regime_id": event.regime_id,
        "role": event.role,
        "record_type": event.record_type,
        "evaluator_annotations_visible_to_memory": False,
        "query_precedes_write": True,
        "query_generation": event.query_generation,
        "write_generation": event.write_generation,
        "expected_target": target,
        "record_payload": list(event.record_payload),
        "procedural_succeeded": event.succeeded,
        "procedural_outcome": list(event.outcome),
        "retrieval": {
            "accepted": accepted,
            "slot": _scalar_int(retrieval.slot),
            "identity_found": _scalar_bool(retrieval.identity_found),
            "compatible": _scalar_bool(retrieval.compatible),
            "fresh": _scalar_bool(retrieval.fresh),
            "confidence_ok": _scalar_bool(retrieval.confidence_ok),
            "abstention_reason": reason,
            "prediction": prediction,
            "raw_retrieved_value": retrieved if accepted else None,
            "exact_target_match": accepted
            and np.array_equal(
                np.asarray(prediction, dtype=np.float32),
                np.asarray(target, dtype=np.float32),
            ),
            "retrieved_provenance_sha256": retrieved_provenance,
            "provenance_matches_request": provenance_matches,
        },
        "write": {
            "wrote": _scalar_bool(write.wrote),
            "merged": _scalar_bool(write.merged),
            "revised": _scalar_bool(write.revised),
            "replaced": _scalar_bool(write.replaced),
            "reset_evidence": _scalar_bool(write.reset_evidence),
            "slot": _scalar_int(write.slot),
            "evicted_semantic_sha256": evicted_semantic,
            "evicted_provenance_sha256": evicted_provenance,
        },
        "comparators": {
            "retrieval_ablation_prediction": reference,
            "no_memory_prediction": reference,
            "retrieval_ablation_error": ablation_error,
            "no_memory_error": ablation_error,
            "matched_external_event": True,
            "matched_query_opportunity": True,
            "matched_write_opportunity": True,
        },
        "memory_error": memory_error,
        "signed_transfer_gain": signed_gain,
        "retrieval_harm": accepted and memory_error > ablation_error,
        "retrieval_harm_excess": max(0.0, memory_error - ablation_error),
        "pre_state_sha256": frozen_consolidated_memory_state_sha256(pre_state),
        "post_state_sha256": frozen_consolidated_memory_state_sha256(post_state),
        "post_operation_count": _scalar_int(accounting.operation_count),
        "post_active_semantic_records": _scalar_int(accounting.active_semantic_records),
        "post_active_procedural_records": _scalar_int(accounting.active_procedural_records),
    }


def reconstruct_consolidated_memory_transfer_summary(
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct all descriptive metrics exclusively from the raw trace."""

    if not trace:
        raise ValueError("transfer trace must be nonempty")
    retrievals = [_mapping(event["retrieval"], name="trace.retrieval") for event in trace]
    writes = [_mapping(event["write"], name="trace.write") for event in trace]
    accepted_indices = [
        index for index, retrieval in enumerate(retrievals) if retrieval["accepted"] is True
    ]
    exact_matches = sum(
        retrievals[index]["exact_target_match"] is True for index in accepted_indices
    )
    abstention_reasons = Counter(
        cast(str, retrieval["abstention_reason"])
        for retrieval in retrievals
        if retrieval["accepted"] is False
    )
    harm_events = [event for event in trace if event["retrieval_harm"] is True]
    role_metrics: dict[str, dict[str, object]] = {}
    for role in sorted({cast(str, event["role"]) for event in trace}):
        events = [event for event in trace if event["role"] == role]
        role_metrics[role] = {
            "events": len(events),
            "accepted_retrievals": sum(
                _mapping(event["retrieval"], name="retrieval")["accepted"] is True
                for event in events
            ),
            "mean_memory_error": _mean([cast(float, event["memory_error"]) for event in events]),
            "mean_ablation_error": _mean(
                [
                    cast(
                        float,
                        _mapping(event["comparators"], name="comparators")[
                            "retrieval_ablation_error"
                        ],
                    )
                    for event in events
                ]
            ),
            "mean_signed_transfer_gain": _mean(
                [cast(float, event["signed_transfer_gain"]) for event in events]
            ),
            "harm_events": sum(event["retrieval_harm"] is True for event in events),
        }

    def role_error(role: str) -> float | None:
        value = role_metrics.get(role, {}).get("mean_memory_error")
        return cast(float | None, value)

    semantic_return_error = role_error("retained-semantic-return")
    semantic_recurrence_error = role_error("retained-semantic-recurrence")
    stale_skill_error = role_error("stale-skill-probe")
    procedural_recovery_error = role_error("procedural-recovery")
    semantic_recovery = (
        None
        if semantic_return_error is None or semantic_recurrence_error is None
        else semantic_return_error - semantic_recurrence_error
    )
    procedural_recovery = (
        None
        if stale_skill_error is None or procedural_recovery_error is None
        else stale_skill_error - procedural_recovery_error
    )
    stale_events = [event for event in trace if event["role"] == "stale-skill-probe"]
    retrieved_provenances = [
        cast(str, retrieval["retrieved_provenance_sha256"])
        for retrieval in retrievals
        if retrieval["retrieved_provenance_sha256"] is not None
    ]
    evicted_provenances = [
        cast(str, write["evicted_provenance_sha256"])
        for write in writes
        if write["evicted_provenance_sha256"] is not None
    ]
    return {
        "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
        "performance_thresholds_applied": False,
        "promotion_authority": False,
        "retrieval": {
            "opportunities": len(trace),
            "accepted": len(accepted_indices),
            "exact_target_matches": exact_matches,
            "exact_match_precision": (
                None if not accepted_indices else exact_matches / len(accepted_indices)
            ),
            "abstentions": len(trace) - len(accepted_indices),
            "abstention_rate": (len(trace) - len(accepted_indices)) / len(trace),
            "abstention_reasons": dict(sorted(abstention_reasons.items())),
            "provenance_matches": sum(
                retrieval["provenance_matches_request"] is True for retrieval in retrievals
            ),
            "provenance_mismatches_on_accepted_retrieval": sum(
                retrieval["provenance_matches_request"] is False for retrieval in retrievals
            ),
        },
        "harm": {
            "events": len(harm_events),
            "total_excess_squared_error": math.fsum(
                cast(float, event["retrieval_harm_excess"]) for event in harm_events
            ),
            "event_ids": [cast(str, event["event_id"]) for event in harm_events],
        },
        "forward_transfer_and_recovery": {
            "semantic_generation_shift_gain": role_metrics["semantic-generation-shift"][
                "mean_signed_transfer_gain"
            ],
            "semantic_new_generation_recurrence_gain": role_metrics["compatible-new-generation"][
                "mean_signed_transfer_gain"
            ],
            "semantic_return_to_recurrence_recovery": semantic_recovery,
            "procedural_stale_to_recurrence_recovery": procedural_recovery,
            "descriptive_only": True,
        },
        "retained_semantic_utility": {
            "return_events": role_metrics["retained-semantic-return"]["events"],
            "recurrence_events": role_metrics["retained-semantic-recurrence"]["events"],
            "return_signed_gain": role_metrics["retained-semantic-return"][
                "mean_signed_transfer_gain"
            ],
            "recurrence_signed_gain": role_metrics["retained-semantic-recurrence"][
                "mean_signed_transfer_gain"
            ],
            "recovery": semantic_recovery,
        },
        "stale_skill_harm": {
            "events": len(stale_events),
            "accepted_stale_retrievals": sum(
                _mapping(event["retrieval"], name="retrieval")["accepted"] is True
                for event in stale_events
            ),
            "harm_events": sum(event["retrieval_harm"] is True for event in stale_events),
            "total_excess_squared_error": math.fsum(
                cast(float, event["retrieval_harm_excess"]) for event in stale_events
            ),
        },
        "eviction_and_provenance": {
            "replacements": sum(write["replaced"] is True for write in writes),
            "evicted_provenance_sha256": evicted_provenances,
            "retrieved_provenance_sha256": retrieved_provenances,
            "unique_retrieved_provenance_count": len(set(retrieved_provenances)),
            "provenance_gate_abstentions": abstention_reasons.get("provenance-mismatch", 0),
        },
        "role_metrics": role_metrics,
    }


def _compiled_final_state(
    evaluator: ConsolidatedMemoryTransferEvaluator,
    initial: ConsolidatedMemoryState,
) -> ConsolidatedMemoryState:
    cache_key = _canonical_sha256(
        {
            "config": evaluator.config.to_config(),
            "protocol": evaluator.protocol.to_config(),
        }
    )
    compiled = _COMPILED_SCHEDULE_CACHE.get(cache_key)
    if compiled is None:

        def kernel(state: ConsolidatedMemoryState) -> ConsolidatedMemoryState:
            current = state
            for event in evaluator.protocol.events:
                current = cast(Any, evaluator._apply_event(current, event)).state
            return current

        compiled = jax.jit(kernel)
        _COMPILED_SCHEDULE_CACHE[cache_key] = compiled
    return cast(ConsolidatedMemoryState, compiled(initial))


def _execute_evaluator(
    evaluator: ConsolidatedMemoryTransferEvaluator,
) -> tuple[
    ConsolidatedMemoryState,
    ConsolidatedMemoryState,
    list[dict[str, object]],
    dict[str, object],
]:
    initial = evaluator.initial_memory_state
    initial_sha256 = frozen_consolidated_memory_state_sha256(initial)
    full = initial
    ablation = initial
    trace: list[dict[str, object]] = []
    for event_index, event in enumerate(evaluator.protocol.events):
        pre_full = full
        full_result = cast(Any, evaluator._apply_event(full, event))
        ablation_result = cast(Any, evaluator._apply_event(ablation, event))
        if not _trees_exactly_equal(full_result, ablation_result):
            raise RuntimeError("matched full and retrieval-ablation event kernels diverged")
        full = full_result.state
        ablation = ablation_result.state
        trace.append(_trace_event(evaluator, event_index, event, pre_full, full_result))
    if not _trees_exactly_equal(full, ablation):
        raise RuntimeError("matched full and retrieval-ablation final states diverged")
    compiled = _compiled_final_state(evaluator, initial)
    if not _trees_exactly_equal(full, compiled):
        raise RuntimeError("eager and compiled consolidated-memory schedules diverged")
    if frozen_consolidated_memory_state_sha256(initial) != initial_sha256:
        raise RuntimeError("evaluation mutated the immutable empty snapshot")
    return (
        full,
        ablation,
        trace,
        {
            "compiled_schedule_parity_checked": True,
            "compiled_schedule_parity_exact": True,
            "full_ablation_state_parity_exact": True,
            "external_snapshot_mutations": 0,
            "query_precedes_write_for_every_event": True,
        },
    )


def _accounting_payload(
    evaluator: ConsolidatedMemoryTransferEvaluator,
    full: ConsolidatedMemoryState,
    ablation: ConsolidatedMemoryState,
    *,
    report_bytes: int,
) -> dict[str, object]:
    events = len(evaluator.protocol.events)
    initial_bytes = _tree_nbytes(evaluator.initial_memory_state)
    full_accounting = evaluator.memory.accounting(full)
    ablation_accounting = evaluator.memory.accounting(ablation)

    def accounting(value: object) -> dict[str, int]:
        result: dict[str, int] = {}
        for field in dataclasses.fields(type(value)):
            result[field.name] = _scalar_int(getattr(value, field.name))
        return result

    counter_names = (
        "semantic_query_count",
        "semantic_accepted_query_count",
        "semantic_write_count",
        "semantic_merge_count",
        "semantic_revision_count",
        "semantic_replacement_count",
        "semantic_rejected_write_count",
        "semantic_invalidation_count",
        "semantic_retirement_count",
        "procedural_query_count",
        "procedural_accepted_query_count",
        "procedural_write_count",
        "procedural_merge_count",
        "procedural_revision_count",
        "procedural_replacement_count",
        "procedural_rejected_write_count",
        "procedural_invalidation_count",
        "procedural_retirement_count",
    )

    def lifetime_counters(state: ConsolidatedMemoryState) -> dict[str, int]:
        return {name: _scalar_int(getattr(state, name)) for name in counter_names}

    return {
        "memory_resource_budget": dataclasses.asdict(evaluator.memory.resource_budget),
        "persistent_state_bytes_per_memory_arm": initial_bytes,
        "logical_memory_arm_count": 2,
        "logical_memory_bytes_across_full_and_ablation": 2 * initial_bytes,
        "no_memory_persistent_state_bytes": 0,
        "dynamic_persistent_growth_bytes": 0,
        "external_event_count_per_arm": events,
        "query_opportunities_per_arm": events,
        "write_opportunities_per_arm": events,
        "full_memory_kernel_calls": events,
        "retrieval_ablation_kernel_calls": events,
        "no_memory_kernel_calls": 0,
        "compiled_parity_diagnostic_kernel_calls": events,
        "total_physical_memory_kernel_calls": 3 * events,
        "full_memory_accounting": accounting(full_accounting),
        "retrieval_ablation_accounting": accounting(ablation_accounting),
        "full_memory_lifetime_counters": lifetime_counters(full),
        "retrieval_ablation_lifetime_counters": lifetime_counters(ablation),
        "matched_external_experience": True,
        "matched_full_and_ablation_compute": True,
        "no_memory_compute_difference_declared": True,
        "evaluator_random_generator_calls": 0,
        "agent_parameter_mutations": 0,
        "action_selection_calls": 0,
        "promotion_decisions": 0,
        "canonical_report_bytes": report_bytes,
    }


def _assemble_report(
    evaluator: ConsolidatedMemoryTransferEvaluator,
    *,
    root: Path,
) -> dict[str, object]:
    full, ablation, trace, execution = _execute_evaluator(evaluator)
    config_payload = evaluator.config.to_config()
    protocol_payload = evaluator.protocol.to_config()
    source_snapshot = consolidated_memory_transfer_source_snapshot(root)
    runtime = consolidated_memory_transfer_runtime_identity()
    initial = evaluator.initial_memory_state
    empty_reconstructed = evaluator.memory.init(
        source_digest=evaluator.config.source_digest,
        semantic_namespace_digest=evaluator.config.semantic_namespace_digest,
        representation_revision=evaluator.config.representation_revision,
        source_revision=evaluator.config.source_revision,
    )
    payload: dict[str, object] = {
        "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
        "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
        "performance_thresholds_applied": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "config": config_payload,
        "protocol": protocol_payload,
        "bindings": {
            "config_sha256": _canonical_sha256(config_payload),
            "protocol_sha256": _canonical_sha256(protocol_payload),
            "frozen_protocol_sha256": frozen_consolidated_memory_transfer_protocol_sha256(),
            "external_experience_sha256": protocol_payload["external_experience_sha256"],
            "source_sha256": source_snapshot,
            "source_manifest_sha256": _canonical_sha256(source_snapshot),
            "runtime": runtime,
            "runtime_sha256": _canonical_sha256(runtime),
        },
        "initial_snapshot": {
            "empty": _trees_exactly_equal(initial, empty_reconstructed),
            "source_bound": True,
            "state_sha256": frozen_consolidated_memory_state_sha256(initial),
            "state_bytes": _tree_nbytes(initial),
            "source_digest_sha256": _digest_hex(evaluator.config.source_digest),
            "semantic_namespace_digest_sha256": _digest_hex(
                evaluator.config.semantic_namespace_digest
            ),
            "representation_revision": evaluator.config.representation_revision,
            "source_revision": evaluator.config.source_revision,
        },
        "raw_trace": trace,
        "summary": reconstruct_consolidated_memory_transfer_summary(trace),
        "execution": execution,
        "final_state": {
            "full_sha256": frozen_consolidated_memory_state_sha256(full),
            "retrieval_ablation_sha256": frozen_consolidated_memory_state_sha256(ablation),
            "states_exactly_equal": _trees_exactly_equal(full, ablation),
        },
        "resource_accounting": {},
        "limitations": list(_LIMITATIONS),
    }
    report_bytes = 0
    report: dict[str, object] = {}
    for _ in range(8):
        payload["resource_accounting"] = _accounting_payload(
            evaluator, full, ablation, report_bytes=report_bytes
        )
        report = {
            "schema": CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA,
            "payload": payload,
            "payload_sha256": _canonical_sha256(payload),
        }
        next_bytes = len(_canonical_json_bytes(report))
        if next_bytes == report_bytes:
            break
        report_bytes = next_bytes
    else:
        raise RuntimeError("canonical report byte accounting did not converge")
    if len(_canonical_json_bytes(report)) != report_bytes:
        raise RuntimeError("canonical report byte accounting is not exact")
    if report_bytes > evaluator.config.max_report_bytes:
        raise ValueError("canonical transfer report exceeds max_report_bytes")
    return report


def build_consolidated_memory_transfer_report(
    config: ConsolidatedMemoryTransferConfig | None = None,
    protocol: ConsolidatedMemoryTransferProtocol | None = None,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build the integrity-bound development report; no claim is assessed."""

    resolved_config = config or default_consolidated_memory_transfer_config()
    resolved_protocol = protocol or default_consolidated_memory_transfer_protocol()
    cache_key = _canonical_sha256(
        {
            "config": resolved_config.to_config(),
            "protocol": resolved_protocol.to_config(),
            "source": consolidated_memory_transfer_source_snapshot(root),
            "runtime": consolidated_memory_transfer_runtime_identity(),
            "root": root.resolve().as_posix(),
        }
    )
    cached = _REPORT_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    evaluator = ConsolidatedMemoryTransferEvaluator(
        config=resolved_config, protocol=resolved_protocol
    )
    report = _assemble_report(evaluator, root=root)
    _REPORT_CACHE[cache_key] = copy.deepcopy(report)
    return report


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryTransferValidation:
    valid: bool
    status: str
    errors: tuple[str, ...]
    exact_replay_checked: bool
    exact_replay_matches: bool


def validate_consolidated_memory_transfer_report(
    report: object,
    *,
    expected_config: ConsolidatedMemoryTransferConfig | None = None,
    root: Path = REPO_ROOT,
) -> ConsolidatedMemoryTransferValidation:
    """Fail closed on schema, hash, source, runtime, or replay differences."""

    errors: list[str] = []
    replay_checked = False
    replay_exact = False
    try:
        raw = _mapping(report, name="report")
        if set(raw) != {"schema", "payload", "payload_sha256"}:
            raise ValueError("report fields differ from v1")
        if raw["schema"] != CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA:
            raise ValueError("report schema differs")
        payload = _mapping(raw["payload"], name="report.payload")
        if raw["payload_sha256"] != _canonical_sha256(payload):
            raise ValueError("report payload SHA differs")
        expected_payload_fields = {
            "development_status",
            "assessment_status",
            "performance_thresholds_applied",
            "promotion_authority",
            "scientific_promotion_allowed",
            "config",
            "protocol",
            "bindings",
            "initial_snapshot",
            "raw_trace",
            "summary",
            "execution",
            "final_state",
            "resource_accounting",
            "limitations",
        }
        if set(payload) != expected_payload_fields:
            raise ValueError("report payload fields differ from v1")
        fixed = {
            "development_status": CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
            "assessment_status": CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
            "performance_thresholds_applied": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "limitations": list(_LIMITATIONS),
        }
        if any(not _strict_json_equal(payload[name], value) for name, value in fixed.items()):
            raise ValueError("report fixed development fields differ")
        config = ConsolidatedMemoryTransferConfig.from_config(payload["config"])
        expected = expected_config or default_consolidated_memory_transfer_config()
        if not _strict_json_equal(config.to_config(), expected.to_config()):
            raise ValueError("report config differs from the expected frozen config")
        protocol = ConsolidatedMemoryTransferProtocol.from_config(payload["protocol"])
        if (
            _canonical_sha256(protocol.to_config())
            != frozen_consolidated_memory_transfer_protocol_sha256()
        ):
            raise ValueError("report protocol differs from the frozen evaluator protocol")
        bindings = _mapping(payload["bindings"], name="bindings")
        expected_bindings = {
            "config_sha256": _canonical_sha256(config.to_config()),
            "protocol_sha256": _canonical_sha256(protocol.to_config()),
            "frozen_protocol_sha256": frozen_consolidated_memory_transfer_protocol_sha256(),
            "external_experience_sha256": protocol.to_config()["external_experience_sha256"],
            "source_sha256": consolidated_memory_transfer_source_snapshot(root),
            "source_manifest_sha256": _canonical_sha256(
                consolidated_memory_transfer_source_snapshot(root)
            ),
            "runtime": consolidated_memory_transfer_runtime_identity(),
            "runtime_sha256": _canonical_sha256(consolidated_memory_transfer_runtime_identity()),
        }
        if not _strict_json_equal(dict(bindings), expected_bindings):
            raise ValueError("report source, runtime, config, or protocol binding differs")
        trace = _list(payload["raw_trace"], name="raw_trace")
        parsed_trace = [
            _mapping(value, name=f"raw_trace[{index}]") for index, value in enumerate(trace)
        ]
        reconstructed = reconstruct_consolidated_memory_transfer_summary(parsed_trace)
        if not _strict_json_equal(payload["summary"], reconstructed):
            raise ValueError("report summary differs from raw-trace reconstruction")
        resources = _mapping(payload["resource_accounting"], name="resource_accounting")
        if resources.get("canonical_report_bytes") != len(_canonical_json_bytes(raw)):
            raise ValueError("report canonical byte accounting differs")
        replay_checked = True
        replay = build_consolidated_memory_transfer_report(
            config=config, protocol=protocol, root=root
        )
        replay_exact = _strict_json_equal(dict(raw), replay)
        if not replay_exact:
            raise ValueError("report differs from exact causal replay")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    return ConsolidatedMemoryTransferValidation(
        valid=not errors,
        status=(CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS if not errors else "invalid"),
        errors=tuple(errors),
        exact_replay_checked=replay_checked,
        exact_replay_matches=replay_exact,
    )


def canonical_consolidated_memory_transfer_report_bytes(report: object) -> bytes:
    """Return canonical bytes only for a valid integrity-bound report."""

    validation = validate_consolidated_memory_transfer_report(report)
    if not validation.valid:
        raise ValueError("cannot canonicalize invalid consolidated-memory transfer report")
    return _canonical_json_bytes(report)


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_consolidated_memory_transfer_report_bytes(data: bytes) -> dict[str, object]:
    """Strictly parse and validate canonical report bytes by exact replay."""

    if type(data) is not bytes:
        raise TypeError("report data must be exact bytes")
    parsed = json.loads(
        data.decode("utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if type(parsed) is not dict:
        raise ValueError("report JSON root must be an object")
    report = cast(dict[str, object], parsed)
    if _canonical_json_bytes(report) != data:
        raise ValueError("report bytes are not canonical JSON")
    validation = validate_consolidated_memory_transfer_report(report)
    if not validation.valid:
        raise ValueError(
            "report integrity/replay validation failed: "
            + "; ".join(validation.errors)
        )
    return report


__all__ = [
    "CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS",
    "CONSOLIDATED_MEMORY_TRANSFER_CHECKPOINT_SCHEMA",
    "CONSOLIDATED_MEMORY_TRANSFER_CONFIG_SCHEMA",
    "CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS",
    "CONSOLIDATED_MEMORY_TRANSFER_PROMOTION_AUTHORITY",
    "CONSOLIDATED_MEMORY_TRANSFER_PROTOCOL_SCHEMA",
    "CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA",
    "CONSOLIDATED_MEMORY_TRANSFER_SCIENTIFIC_PROMOTION_ALLOWED",
    "ConsolidatedMemoryTransferConfig",
    "ConsolidatedMemoryTransferEvaluator",
    "ConsolidatedMemoryTransferEvent",
    "ConsolidatedMemoryTransferProtocol",
    "ConsolidatedMemoryTransferRunState",
    "ConsolidatedMemoryTransferValidation",
    "build_consolidated_memory_transfer_report",
    "canonical_consolidated_memory_transfer_report_bytes",
    "consolidated_memory_transfer_runtime_identity",
    "consolidated_memory_transfer_source_snapshot",
    "default_consolidated_memory_transfer_config",
    "default_consolidated_memory_transfer_protocol",
    "frozen_consolidated_memory_state_sha256",
    "frozen_consolidated_memory_transfer_protocol_sha256",
    "frozen_consolidated_memory_transfer_run_state_sha256",
    "load_consolidated_memory_transfer_report_bytes",
    "reconstruct_consolidated_memory_transfer_summary",
    "validate_consolidated_memory_transfer_report",
]
