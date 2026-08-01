"""Pure, nonexecuting scan geometry for the noisy-world v6 design.

This module turns one already-initialized world's jittered segment geometry
into a fixed-shape, two-cycle scan plan.  It does not construct a world or an
agent, create random keys, execute a transition, choose thresholds, record an
outcome, or authorize evidence.  The serialized object is development-only
geometry that a future runner may consume only after its separate readiness
and governance gates are satisfied.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Literal, NoReturn, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    V6_DIAGNOSTIC_ORDER,
    build_v6_diagnostic_controls,
    build_v6_primary_controls,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackState,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.scan-plan-development.v1"
)
SCAN_PLAN_STATUS = "NONEXECUTING_DEVELOPMENT_SCAN_PLAN"
DEVELOPMENT_ONLY = True
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

CYCLE_COUNT = 2
ENTRY_WINDOW_STEPS = 128
TAIL_WINDOW_STEPS = 128
FINAL_WINDOW_STEPS = 256

_WORLD_CONFIG = HiddenPartnerWorldFeedbackConfig()
REGIME_SCHEDULE: tuple[int, ...] = _WORLD_CONFIG.regime_schedule
BASE_SEGMENT_LENGTHS: tuple[int, ...] = _WORLD_CONFIG.base_segment_lengths
JITTER_RADIUS = _WORLD_CONFIG.jitter_radius
N_SEGMENTS = len(REGIME_SCHEDULE)
MIN_CYCLE_LENGTH = sum(BASE_SEGMENT_LENGTHS) - (N_SEGMENTS * JITTER_RADIUS)
MAX_CYCLE_LENGTH = sum(BASE_SEGMENT_LENGTHS) + (N_SEGMENTS * JITTER_RADIUS)
MAX_SCAN_STEPS = CYCLE_COUNT * MAX_CYCLE_LENGTH
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_SERIALIZED_BYTES = 128 * 1024

if MAX_CYCLE_LENGTH != 15_159 or MAX_SCAN_STEPS != 30_318:
    raise RuntimeError("v6 scan bounds have drifted from the reviewed world geometry")
if MIN_CYCLE_LENGTH != 14_025:
    raise RuntimeError("v6 minimum cycle bound has drifted from the reviewed world geometry")
if MAX_SCAN_STEPS > _INT32_MAX:
    raise RuntimeError("the fixed v6 scan shape must fit in int32")
if min(BASE_SEGMENT_LENGTHS) - JITTER_RADIUS < FINAL_WINDOW_STEPS:
    raise RuntimeError("every jittered v6 segment must contain the largest required window")

WindowKind = Literal["entry", "tail", "final"]
ControlFamily = Literal["primary", "diagnostic"]


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6Window:
    """One exact half-open window in the active scan prefix."""

    kind: WindowKind
    start: int
    end_exclusive: int
    steps: int

    def to_config(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end_exclusive": self.end_exclusive,
            "steps": self.steps,
        }


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6SegmentOccurrence:
    """One of the nine world segments in one of the two complete cycles."""

    occurrence_index: int
    cycle_index: int
    segment_index: int
    regime_id: int
    start: int
    end_exclusive: int
    length: int
    entry_window: HiddenPartnerLifecycleWorldV6Window
    tail_window: HiddenPartnerLifecycleWorldV6Window

    def to_config(self) -> dict[str, object]:
        return {
            "occurrence_index": self.occurrence_index,
            "cycle_index": self.cycle_index,
            "segment_index": self.segment_index,
            "regime_id": self.regime_id,
            "start": self.start,
            "end_exclusive": self.end_exclusive,
            "length": self.length,
            "entry_window": self.entry_window.to_config(),
            "tail_window": self.tail_window.to_config(),
            "window_proofs": {
                "entry_within_one_segment": True,
                "tail_within_one_segment": True,
                "entry_within_active_prefix": True,
                "tail_within_active_prefix": True,
            },
        }


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6ScanPlan:
    """Validated two-cycle geometry with a static maximum scan shape."""

    segment_lengths: tuple[int, ...]
    segment_ends: tuple[int, ...]
    cycle_length: int
    run_steps: int
    segment_occurrences: tuple[HiddenPartnerLifecycleWorldV6SegmentOccurrence, ...]
    final_window: HiddenPartnerLifecycleWorldV6Window

    @property
    def scheduled_active(self) -> Array:
        """Return the exact bool[30318] active-prefix mask for this plan."""

        return v6_scheduled_active_mask(jnp.asarray(self.run_steps, dtype=jnp.int32))

    def to_config(self) -> dict[str, object]:
        """Return the exact authority-free, JSON-compatible development payload."""

        return {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA,
            "status": SCAN_PLAN_STATUS,
            "development_only": DEVELOPMENT_ONLY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "world_contract": HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION,
            "geometry": {
                "cycle_count": CYCLE_COUNT,
                "segment_count_per_cycle": N_SEGMENTS,
                "base_segment_lengths": list(BASE_SEGMENT_LENGTHS),
                "regime_schedule": list(REGIME_SCHEDULE),
                "jitter_radius": JITTER_RADIUS,
                "minimum_cycle_length": MIN_CYCLE_LENGTH,
                "maximum_cycle_length": MAX_CYCLE_LENGTH,
                "maximum_scan_steps": MAX_SCAN_STEPS,
                "segment_lengths": list(self.segment_lengths),
                "segment_ends": list(self.segment_ends),
                "cycle_length": self.cycle_length,
                "run_steps": self.run_steps,
            },
            "scheduled_active_contract": {
                "dtype": "bool",
                "shape": [MAX_SCAN_STEPS],
                "expression": "scan_index < run_steps",
                "active_start": 0,
                "active_end_exclusive": self.run_steps,
                "padding_start": self.run_steps,
                "padding_end_exclusive": MAX_SCAN_STEPS,
                "true_count": self.run_steps,
                "false_count": MAX_SCAN_STEPS - self.run_steps,
                "is_one_contiguous_prefix": True,
            },
            "window_contract": {
                "entry_window_steps": ENTRY_WINDOW_STEPS,
                "tail_window_steps": TAIL_WINDOW_STEPS,
                "final_window_steps": FINAL_WINDOW_STEPS,
                "segment_occurrence_count": CYCLE_COUNT * N_SEGMENTS,
                "segment_occurrences": [
                    occurrence.to_config() for occurrence in self.segment_occurrences
                ],
                "final_window": {
                    **self.final_window.to_config(),
                    "within_one_segment": True,
                    "within_active_prefix": True,
                },
            },
            "padding_contract": {
                "padding_is_execution": False,
                "padding_is_rejection": False,
                "padding_contributes_support": False,
                "world_step_called": False,
                "agent_decide_called": False,
                "agent_update_called": False,
                "carry_preserved_bit_exact": True,
                "canonical_sentinels": {
                    "bool": False,
                    "int32": -1,
                    "uint32": 0,
                    "float32": "positive_zero_bit_pattern_0x00000000",
                    "scheduled_active": False,
                },
            },
        }


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6BlockedControl:
    """One required control whose exact runner binding is still unavailable."""

    family: ControlFamily
    name: str
    reason: str

    def to_config(self) -> dict[str, object]:
        return {"family": self.family, "name": self.name, "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6ControlSuiteReadiness:
    """Non-authorizing source-binding readiness for required v6 controls."""

    primary_ready_count: int
    primary_required_count: int
    diagnostic_ready_count: int
    diagnostic_required_count: int
    blocked_controls: tuple[HiddenPartnerLifecycleWorldV6BlockedControl, ...]

    @property
    def all_controls_ready(self) -> bool:
        return (
            self.primary_ready_count == self.primary_required_count
            and self.diagnostic_ready_count == self.diagnostic_required_count
            and not self.blocked_controls
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA,
            "status": (
                "DEVELOPMENT_CONTROL_SUITE_READY"
                if self.all_controls_ready
                else "DEVELOPMENT_CONTROL_SUITE_BLOCKED"
            ),
            "development_only": DEVELOPMENT_ONLY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "primary_ready_count": self.primary_ready_count,
            "primary_required_count": self.primary_required_count,
            "diagnostic_ready_count": self.diagnostic_ready_count,
            "diagnostic_required_count": self.diagnostic_required_count,
            "all_controls_ready": self.all_controls_ready,
            "blocked_controls": [control.to_config() for control in self.blocked_controls],
        }


class HiddenPartnerLifecycleWorldV6ControlSuiteNotReadyError(RuntimeError):
    """Raised when runner work is requested while a required control is blocked."""


def v6_scheduled_active_mask(run_steps: Array) -> Array:
    """Construct a JIT-compatible, static bool[30318] active-prefix mask.

    ``run_steps`` must be an already validated scalar int32.  Geometry and
    capacity validation belongs to :func:`build_hidden_partner_lifecycle_world_v6_scan_plan`;
    keeping this primitive array-only lets it be used inside ``jax.jit``.
    """

    value = jnp.asarray(run_steps)
    if value.shape != ():
        raise ValueError("run_steps must be a scalar")
    if value.dtype != jnp.dtype(jnp.int32):
        raise TypeError("run_steps must have exact int32 dtype")
    return jnp.arange(MAX_SCAN_STEPS, dtype=jnp.int32) < value


def _exact_int32_vector(value: object, *, name: str) -> np.ndarray:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape != (N_SEGMENTS,):
        raise ValueError(f"{name} must have exact shape ({N_SEGMENTS},)")
    if dtype is None or np.dtype(dtype) != np.dtype(np.int32):
        raise TypeError(f"{name} must have exact int32 dtype")
    try:
        result = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a concrete int32 array") from exc
    if result.shape != (N_SEGMENTS,) or result.dtype != np.dtype(np.int32):
        raise TypeError(f"{name} must be a concrete exact int32[{N_SEGMENTS}] array")
    return result


def _exact_initialized_step_count(value: object) -> None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape != ():
        raise ValueError("world state step_count must be scalar")
    if dtype is None or np.dtype(dtype) != np.dtype(np.int32):
        raise TypeError("world state step_count must have exact int32 dtype")
    try:
        step_count = int(np.asarray(jax.device_get(value)).item())
    except (TypeError, ValueError) as exc:
        raise TypeError("world state step_count must be a concrete int32 scalar") from exc
    if step_count != 0:
        raise ValueError("scan plans must be derived from an initialized step-zero world state")


def _validate_segment_geometry(
    segment_lengths: object,
    segment_ends: object | None,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    lengths_array = _exact_int32_vector(segment_lengths, name="segment_lengths")
    lengths = tuple(int(value) for value in lengths_array.tolist())
    for index, (length, base) in enumerate(
        zip(lengths, BASE_SEGMENT_LENGTHS, strict=True)
    ):
        if length <= 0:
            raise ValueError(f"segment_lengths[{index}] must be positive")
        lower = base - JITTER_RADIUS
        upper = base + JITTER_RADIUS
        if not lower <= length <= upper:
            raise ValueError(
                f"segment_lengths[{index}] must lie in the exact jitter bound "
                f"[{lower}, {upper}]"
            )

    ends_int64 = np.cumsum(lengths_array, dtype=np.int64)
    if np.any(ends_int64 > _INT32_MAX):
        raise ValueError("segment_ends must fit in int32")
    expected_ends = tuple(int(value) for value in ends_int64.tolist())
    if segment_ends is not None:
        supplied_ends_array = _exact_int32_vector(segment_ends, name="segment_ends")
        supplied_ends = tuple(int(value) for value in supplied_ends_array.tolist())
        if supplied_ends != expected_ends:
            raise ValueError("segment_ends must be the exact cumulative segment lengths")

    cycle_length = expected_ends[-1]
    if not MIN_CYCLE_LENGTH <= cycle_length <= MAX_CYCLE_LENGTH:
        raise ValueError("cycle length lies outside the exact v6 jitter envelope")
    if cycle_length > _INT32_MAX:
        raise ValueError("cycle length must fit in int32")
    run_steps = CYCLE_COUNT * cycle_length
    if run_steps > _INT32_MAX:
        raise ValueError("run_steps must fit in int32")
    if run_steps > MAX_SCAN_STEPS:
        raise ValueError("run_steps exceeds the fixed v6 scan shape")
    return lengths, expected_ends, cycle_length, run_steps


def build_hidden_partner_lifecycle_world_v6_scan_plan(
    segment_lengths: object,
    *,
    segment_ends: object | None = None,
) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    """Build validated two-cycle geometry from exact concrete int32 arrays."""

    lengths, ends, cycle_length, run_steps = _validate_segment_geometry(
        segment_lengths,
        segment_ends,
    )
    occurrences: list[HiddenPartnerLifecycleWorldV6SegmentOccurrence] = []
    for cycle_index in range(CYCLE_COUNT):
        cycle_start = cycle_index * cycle_length
        for segment_index, (length, relative_end, regime_id) in enumerate(
            zip(lengths, ends, REGIME_SCHEDULE, strict=True)
        ):
            relative_start = 0 if segment_index == 0 else ends[segment_index - 1]
            start = cycle_start + relative_start
            end = cycle_start + relative_end
            entry = HiddenPartnerLifecycleWorldV6Window(
                kind="entry",
                start=start,
                end_exclusive=start + ENTRY_WINDOW_STEPS,
                steps=ENTRY_WINDOW_STEPS,
            )
            tail = HiddenPartnerLifecycleWorldV6Window(
                kind="tail",
                start=end - TAIL_WINDOW_STEPS,
                end_exclusive=end,
                steps=TAIL_WINDOW_STEPS,
            )
            if not start <= entry.start < entry.end_exclusive <= end:
                raise ValueError("entry window crosses a segment boundary")
            if not start <= tail.start < tail.end_exclusive <= end:
                raise ValueError("tail window crosses a segment boundary")
            if entry.end_exclusive > run_steps or tail.end_exclusive > run_steps:
                raise ValueError("segment window crosses the active scan prefix")
            occurrences.append(
                HiddenPartnerLifecycleWorldV6SegmentOccurrence(
                    occurrence_index=len(occurrences),
                    cycle_index=cycle_index,
                    segment_index=segment_index,
                    regime_id=regime_id,
                    start=start,
                    end_exclusive=end,
                    length=length,
                    entry_window=entry,
                    tail_window=tail,
                )
            )

    final_window = HiddenPartnerLifecycleWorldV6Window(
        kind="final",
        start=run_steps - FINAL_WINDOW_STEPS,
        end_exclusive=run_steps,
        steps=FINAL_WINDOW_STEPS,
    )
    final_occurrence = occurrences[-1]
    if not (
        final_occurrence.start
        <= final_window.start
        < final_window.end_exclusive
        <= final_occurrence.end_exclusive
    ):
        raise ValueError("final window crosses the final segment boundary")
    if len(occurrences) != CYCLE_COUNT * N_SEGMENTS:
        raise RuntimeError("v6 scan plan must reconstruct exactly eighteen segments")
    return HiddenPartnerLifecycleWorldV6ScanPlan(
        segment_lengths=lengths,
        segment_ends=ends,
        cycle_length=cycle_length,
        run_steps=run_steps,
        segment_occurrences=tuple(occurrences),
        final_window=final_window,
    )


def build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(
    state: HiddenPartnerWorldFeedbackState,
) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    """Build geometry from an exact initialized, step-zero world state."""

    if not isinstance(state, HiddenPartnerWorldFeedbackState):
        raise TypeError("state must be a HiddenPartnerWorldFeedbackState")
    _exact_initialized_step_count(state.step_count)
    return build_hidden_partner_lifecycle_world_v6_scan_plan(
        state.segment_lengths,
        segment_ends=state.segment_ends,
    )


def validate_hidden_partner_lifecycle_world_v6_scan_plan(
    plan: HiddenPartnerLifecycleWorldV6ScanPlan,
) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    """Reconstruct and accept only the exact canonical scan plan."""

    if type(plan) is not HiddenPartnerLifecycleWorldV6ScanPlan:
        raise TypeError("plan must be an exact HiddenPartnerLifecycleWorldV6ScanPlan")
    rebuilt = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(plan.segment_lengths, dtype=np.int32),
        segment_ends=np.asarray(plan.segment_ends, dtype=np.int32),
    )
    if plan != rebuilt:
        raise ValueError("scan plan differs from its exact reconstructed geometry")
    return rebuilt


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("scan plan must contain canonical finite JSON data") from exc


def canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(
    plan: HiddenPartnerLifecycleWorldV6ScanPlan,
) -> bytes:
    """Return canonical ASCII JSON bytes after exact plan reconstruction."""

    validated = validate_hidden_partner_lifecycle_world_v6_scan_plan(plan)
    return _canonical_json_bytes(validated.to_config())


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"canonical JSON cannot contain {value}")


def _reject_json_float(value: str) -> NoReturn:
    raise ValueError(f"v6 scan plans contain no floating-point value: {value}")


def _bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 10:
        raise ValueError("JSON integer exceeds the v6 scan-plan bound")
    return int(value)


def _plan_from_payload(payload: Mapping[str, object]) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    if type(payload) is not dict:
        raise ValueError("scan plan must be a plain JSON object")
    supplied = cast(dict[str, object], payload)
    geometry = supplied.get("geometry")
    if type(geometry) is not dict:
        raise ValueError("scan plan geometry must be a plain JSON object")
    geometry_dict = cast(dict[str, object], geometry)
    raw_lengths = geometry_dict.get("segment_lengths")
    raw_ends = geometry_dict.get("segment_ends")
    if type(raw_lengths) is not list or type(raw_ends) is not list:
        raise ValueError("serialized segment lengths and ends must be plain lists")
    lengths_list = cast(list[object], raw_lengths)
    ends_list = cast(list[object], raw_ends)
    if len(lengths_list) != N_SEGMENTS or any(type(value) is not int for value in lengths_list):
        raise ValueError("serialized segment_lengths must contain exactly nine integers")
    if len(ends_list) != N_SEGMENTS or any(type(value) is not int for value in ends_list):
        raise ValueError("serialized segment_ends must contain exactly nine integers")
    lengths = np.asarray(cast(list[int], lengths_list), dtype=np.int64)
    ends = np.asarray(cast(list[int], ends_list), dtype=np.int64)
    if np.any(lengths < np.iinfo(np.int32).min) or np.any(lengths > _INT32_MAX):
        raise ValueError("serialized segment_lengths must fit in int32")
    if np.any(ends < np.iinfo(np.int32).min) or np.any(ends > _INT32_MAX):
        raise ValueError("serialized segment_ends must fit in int32")
    rebuilt = build_hidden_partner_lifecycle_world_v6_scan_plan(
        lengths.astype(np.int32),
        segment_ends=ends.astype(np.int32),
    )
    if _canonical_json_bytes(supplied) != _canonical_json_bytes(rebuilt.to_config()):
        raise ValueError("serialized scan plan differs from exact reconstructed geometry")
    return rebuilt


def load_hidden_partner_lifecycle_world_v6_scan_plan(
    data: bytes | str,
) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    """Load only canonical, duplicate-free, finite, exactly reconstructable JSON."""

    if isinstance(data, str):
        try:
            raw = data.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("scan plan is not canonical ASCII JSON") from exc
    elif isinstance(data, bytes):
        raw = data
    else:
        raise TypeError("scan plan must be bytes or text")
    if len(raw) > _MAX_SERIALIZED_BYTES:
        raise ValueError("scan plan exceeds the canonical size limit")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("scan plan is not canonical ASCII JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_bounded_json_int,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("scan plan is not canonical JSON") from exc
    if type(value) is not dict:
        raise ValueError("scan plan must be a canonical JSON object")
    plan = _plan_from_payload(cast(dict[str, object], value))
    if raw != canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(plan):
        raise ValueError("scan plan bytes are not canonical")
    return plan


def build_v6_control_suite_readiness() -> HiddenPartnerLifecycleWorldV6ControlSuiteReadiness:
    """Report exact required-control bindings without granting execution authority."""

    primary = build_v6_primary_controls()
    diagnostics = build_v6_diagnostic_controls()
    if tuple(control.name for control in primary) != PRIMARY_CONDITION_ORDER:
        raise RuntimeError("v6 primary control order differs from the frozen design")
    if tuple(control.name for control in diagnostics) != V6_DIAGNOSTIC_ORDER:
        raise RuntimeError("v6 diagnostic control order differs from the declared controls")
    if not all(control.primary for control in primary):
        raise RuntimeError("every v6 primary binding must be marked primary")
    if any(control.primary for control in diagnostics):
        raise RuntimeError("v6 diagnostic bindings must not be marked primary")

    blocked: list[HiddenPartnerLifecycleWorldV6BlockedControl] = []
    for family, controls in (("primary", primary), ("diagnostic", diagnostics)):
        for control in controls:
            if control.execution_ready:
                continue
            reason = control.execution_blocker
            if not isinstance(reason, str) or not reason:
                raise RuntimeError(f"blocked v6 control {control.name!r} has no exact reason")
            blocked.append(
                HiddenPartnerLifecycleWorldV6BlockedControl(
                    family=cast(ControlFamily, family),
                    name=control.name,
                    reason=reason,
                )
            )
    return HiddenPartnerLifecycleWorldV6ControlSuiteReadiness(
        primary_ready_count=sum(control.execution_ready for control in primary),
        primary_required_count=len(PRIMARY_CONDITION_ORDER),
        diagnostic_ready_count=sum(control.execution_ready for control in diagnostics),
        diagnostic_required_count=len(V6_DIAGNOSTIC_ORDER),
        blocked_controls=tuple(blocked),
    )


def require_v6_control_suite_ready() -> HiddenPartnerLifecycleWorldV6ControlSuiteReadiness:
    """Fail visibly until every primary and diagnostic control is source-bound."""

    readiness = build_v6_control_suite_readiness()
    if not readiness.all_controls_ready:
        details = "; ".join(
            f"{blocked.family}:{blocked.name}: {blocked.reason}"
            for blocked in readiness.blocked_controls
        )
        raise HiddenPartnerLifecycleWorldV6ControlSuiteNotReadyError(
            "v6 control suite is not ready for runner binding: " + details
        )
    return readiness


__all__ = [
    "BASE_SEGMENT_LENGTHS",
    "CYCLE_COUNT",
    "DEVELOPMENT_ONLY",
    "ENTRY_WINDOW_STEPS",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "FINAL_WINDOW_STEPS",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA",
    "HiddenPartnerLifecycleWorldV6BlockedControl",
    "HiddenPartnerLifecycleWorldV6ControlSuiteNotReadyError",
    "HiddenPartnerLifecycleWorldV6ControlSuiteReadiness",
    "HiddenPartnerLifecycleWorldV6ScanPlan",
    "HiddenPartnerLifecycleWorldV6SegmentOccurrence",
    "HiddenPartnerLifecycleWorldV6Window",
    "JITTER_RADIUS",
    "MAX_CYCLE_LENGTH",
    "MAX_SCAN_STEPS",
    "MIN_CYCLE_LENGTH",
    "N_SEGMENTS",
    "REGIME_SCHEDULE",
    "SCAN_PLAN_STATUS",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TAIL_WINDOW_STEPS",
    "build_hidden_partner_lifecycle_world_v6_scan_plan",
    "build_hidden_partner_lifecycle_world_v6_scan_plan_from_state",
    "build_v6_control_suite_readiness",
    "canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes",
    "load_hidden_partner_lifecycle_world_v6_scan_plan",
    "require_v6_control_suite_ready",
    "v6_scheduled_active_mask",
    "validate_hidden_partner_lifecycle_world_v6_scan_plan",
]
