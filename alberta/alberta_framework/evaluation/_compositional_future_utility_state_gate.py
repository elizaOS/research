"""Bounded endpoint-state gate for contribution-mode future utility.

The gate validates already-completed arm state and telemetry.  It cannot build
experience, initialize or execute a learner, issue a root, produce a panel
result, write output, or authorize evidence or scientific promotion.  Its
claims are deliberately limited to exact endpoint invariants; the unavailable
per-step mechanism traces are explicit nonclaims.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from alberta_framework.core.compositional_features import CompositionalFeatureState
from alberta_framework.evaluation import compositional_control_life_development as control

DEVELOPMENT_ONLY: Final = True
PANEL_EXECUTION_AUTHORIZED: Final = False
ROOT_ISSUANCE_AUTHORIZED: Final = False
RESULT_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False

STATE_GATE_SCHEMA: Final = (
    "alberta.compositional-future-utility-state-gate.contribution-endpoint.v1"
)
NONCLAIMS: Final = (
    "per-step-contribution-transition-not-proven",
    "candidate-trace-transition-not-proven",
    "mixed-utility-equation-not-proven",
    "normalization-use-in-ranking-not-proven",
    "trace-reset-and-promotion-transfer-not-proven",
)
V3_TOTAL_STEPS: Final = 8_998
_DECAY_095_F32_BITS: Final = "3f733333"
_LONG_DECAY_F32_BITS: Final = "3f7fcc93"
V3_RAW_ENERGY_F32_BITS: Final = {
    _DECAY_095_F32_BITS: 0x419FFFF4,
    _LONG_DECAY_F32_BITS: 0x449F2936,
}
_SUPPORTED_MECHANISMS: Final = (
    (0.0, _DECAY_095_F32_BITS, "none"),
    (1.0, _DECAY_095_F32_BITS, "none"),
    (0.5, _DECAY_095_F32_BITS, "none"),
    (1.0, _DECAY_095_F32_BITS, "uncertainty_age"),
    (1.0, _LONG_DECAY_F32_BITS, "uncertainty_age"),
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _f32_bits(value: float | np.float32) -> int:
    return int(struct.unpack(">I", struct.pack(">f", float(value)))[0])


def _f32_hex(value: float) -> str:
    return struct.pack(">f", value).hex()


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityStateFieldSpec:
    """One exact binary32 field in the bounded future-state subset."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("state field name must be a nonempty exact string")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value < 1 for value in self.shape)
        ):
            raise ValueError("state field shape must contain positive exact integers")
        if self.dtype != "float32":
            raise ValueError("future-state fields must use exact binary32")

    def to_config(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


STATE_FIELD_MANIFEST: Final = (
    FutureUtilityStateFieldSpec("utilities", (control.ACTIVE_SLOTS,)),
    FutureUtilityStateFieldSpec(
        "utility_contribution_trace",
        (control.ACTION_HEADS, control.ACTIVE_SLOTS),
    ),
    FutureUtilityStateFieldSpec("utility_error_trace", (control.ACTION_HEADS,)),
    FutureUtilityStateFieldSpec("utility_feature_trace", (control.ACTIVE_SLOTS,)),
    FutureUtilityStateFieldSpec(
        "utility_feature_energy_trace",
        (control.ACTIVE_SLOTS,),
    ),
    FutureUtilityStateFieldSpec(
        "utility_signal_second_moment",
        (control.ACTIVE_SLOTS,),
    ),
    FutureUtilityStateFieldSpec("task_activity_ema", (control.ACTION_HEADS,)),
    FutureUtilityStateFieldSpec("candidate_utilities", (control.CANDIDATE_SLOTS,)),
    FutureUtilityStateFieldSpec(
        "candidate_utility_contribution_trace",
        (control.ACTION_HEADS, control.CANDIDATE_SLOTS),
    ),
    FutureUtilityStateFieldSpec(
        "candidate_utility_feature_trace",
        (control.CANDIDATE_SLOTS,),
    ),
    FutureUtilityStateFieldSpec(
        "candidate_utility_feature_energy_trace",
        (control.CANDIDATE_SLOTS,),
    ),
    FutureUtilityStateFieldSpec(
        "candidate_utility_signal_second_moment",
        (control.CANDIDATE_SLOTS,),
    ),
)
STATE_FIELD_MANIFEST_SHA256: Final = (
    "834498ba4ed937d814590c2852d756164a80377124ae11fc15ad22ed17cfc9bd"
)


def state_field_manifest_sha256() -> str:
    """Hash the exact ordered state-field manifest."""

    return _canonical_json_sha256([spec.to_config() for spec in STATE_FIELD_MANIFEST])


def _state_arrays(
    state: CompositionalFeatureState,
    *,
    label: str,
) -> dict[str, NDArray[np.float32]]:
    if type(state) is not CompositionalFeatureState:
        raise TypeError(f"{label} state must be an exact CompositionalFeatureState")
    arrays: dict[str, NDArray[np.float32]] = {}
    for spec in STATE_FIELD_MANIFEST:
        value = np.asarray(getattr(state, spec.name))
        if value.shape != spec.shape:
            raise ValueError(
                f"{label} future-state field {spec.name} has shape {value.shape}, "
                f"expected {spec.shape}"
            )
        if value.dtype != np.dtype(np.float32):
            raise TypeError(
                f"{label} future-state field {spec.name} has dtype {value.dtype}, "
                "expected float32"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} future-state field {spec.name} is not finite")
        arrays[spec.name] = cast(NDArray[np.float32], value)
    return arrays


def _all_positive_zero(value: NDArray[np.float32]) -> bool:
    bits = np.ascontiguousarray(value).view(np.uint32)
    return bool(np.all(bits == np.uint32(0)))


def _normalized_big_endian_bytes(
    value: NDArray[np.float32],
) -> bytes:
    return np.ascontiguousarray(
        value.astype(value.dtype.newbyteorder(">"), copy=False)
    ).tobytes(order="C")


def future_utility_state_subset_sha256(state: CompositionalFeatureState) -> str:
    """Hash the exact ordered twelve-field subset with dtype and shape framing."""

    arrays = _state_arrays(state, label="hashed")
    digest = hashlib.sha256()
    for spec in STATE_FIELD_MANIFEST:
        value = arrays[spec.name]
        raw = _normalized_big_endian_bytes(value)
        metadata = _canonical_json(
            {
                "name": spec.name,
                "dtype": ">f4",
                "shape": list(spec.shape),
                "nbytes": len(raw),
            }
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def expected_raw_energy_f32_bits(trace_decay: float, steps: int) -> int:
    """Reconstruct the exact CPU-binary32 raw-slot energy recurrence."""

    if type(trace_decay) is not float:
        raise TypeError("trace_decay must be an exact float")
    decay_bits = _f32_hex(trace_decay)
    if decay_bits not in V3_RAW_ENERGY_F32_BITS:
        raise ValueError("trace_decay is not one of the two frozen v3 decays")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive exact integer")
    decay = np.float32(trace_decay)
    value = np.float32(0.0)
    for _ in range(steps):
        value = np.float32(decay * value + np.float32(1.0))
    bits = _f32_bits(value)
    if steps == V3_TOTAL_STEPS and bits != V3_RAW_ENERGY_F32_BITS[decay_bits]:
        raise RuntimeError("the frozen v3 raw-energy bit pin does not reconstruct")
    return bits


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityStateGateReceipt:
    """JSON-safe receipt for the gate's exact, intentionally bounded claims."""

    schema: str
    steps: int
    trace_decay_f32_bits: str
    expected_raw_energy_f32_bits: int
    normalization_moment_policy: str
    field_manifest_sha256: str
    initial_subset_sha256: str
    final_subset_sha256: str
    initial_fields_all_zero: bool
    all_fields_finite: bool
    contribution_mode_zero_marginal_traces: bool
    raw_slots_untouched_by_curation: bool
    raw_energy_bits_exact: bool
    normalization_moment_policy_exact: bool
    utility_event_final_rows_exact: bool
    nonclaims: tuple[str, ...]
    development_only: bool
    panel_execution_authorized: bool
    result_authorized: bool
    output_writes_allowed: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        if not (
            self.schema == STATE_GATE_SCHEMA
            and type(self.steps) is int
            and self.steps > 0
            and self.trace_decay_f32_bits in V3_RAW_ENERGY_F32_BITS
            and type(self.expected_raw_energy_f32_bits) is int
            and self.normalization_moment_policy
            in {"disabled-exact-zero", "enabled-bounded-endpoint"}
            and self.field_manifest_sha256 == STATE_FIELD_MANIFEST_SHA256
            and _is_sha256(self.initial_subset_sha256)
            and _is_sha256(self.final_subset_sha256)
            and self.initial_subset_sha256 != self.final_subset_sha256
            and self.initial_fields_all_zero is True
            and self.all_fields_finite is True
            and self.contribution_mode_zero_marginal_traces is True
            and self.raw_slots_untouched_by_curation is True
            and self.raw_energy_bits_exact is True
            and self.normalization_moment_policy_exact is True
            and self.utility_event_final_rows_exact is True
            and self.nonclaims == NONCLAIMS
            and self.development_only is True
            and self.panel_execution_authorized is False
            and self.result_authorized is False
            and self.output_writes_allowed is False
            and self.evidence_authorized is False
            and self.scientific_promotion_allowed is False
        ):
            raise ValueError("future-utility state-gate receipt is not exact")

    def to_config(self) -> dict[str, object]:
        """Return a fresh strict-JSON representation of the bounded receipt."""

        return {
            "schema": self.schema,
            "steps": self.steps,
            "trace_decay_f32_bits": self.trace_decay_f32_bits,
            "expected_raw_energy_f32_bits": self.expected_raw_energy_f32_bits,
            "normalization_moment_policy": self.normalization_moment_policy,
            "field_manifest_sha256": self.field_manifest_sha256,
            "initial_subset_sha256": self.initial_subset_sha256,
            "final_subset_sha256": self.final_subset_sha256,
            "initial_fields_all_zero": self.initial_fields_all_zero,
            "all_fields_finite": self.all_fields_finite,
            "contribution_mode_zero_marginal_traces": (
                self.contribution_mode_zero_marginal_traces
            ),
            "raw_slots_untouched_by_curation": self.raw_slots_untouched_by_curation,
            "raw_energy_bits_exact": self.raw_energy_bits_exact,
            "normalization_moment_policy_exact": (
                self.normalization_moment_policy_exact
            ),
            "utility_event_final_rows_exact": self.utility_event_final_rows_exact,
            "nonclaims": list(self.nonclaims),
            "development_only": self.development_only,
            "panel_execution_authorized": self.panel_execution_authorized,
            "result_authorized": self.result_authorized,
            "output_writes_allowed": self.output_writes_allowed,
            "evidence_authorized": self.evidence_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


def _validate_mechanism(
    *,
    future_utility_mix: float,
    future_utility_trace_decay: float,
    future_utility_normalization: str,
) -> tuple[str, bool]:
    if type(future_utility_mix) is not float:
        raise TypeError("future_utility_mix must be an exact float")
    if type(future_utility_trace_decay) is not float:
        raise TypeError("future_utility_trace_decay must be an exact float")
    if type(future_utility_normalization) is not str:
        raise TypeError("future_utility_normalization must be an exact string")
    decay_bits = _f32_hex(future_utility_trace_decay)
    mechanism = (
        future_utility_mix,
        decay_bits,
        future_utility_normalization,
    )
    if mechanism not in _SUPPORTED_MECHANISMS:
        raise ValueError("mechanism is not one of the five frozen v3 intervention tuples")
    normalization_enabled = (
        future_utility_mix > 0.0 and future_utility_normalization != "none"
    )
    return decay_bits, normalization_enabled


def _exact_event_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"event field {name} has shape {array.shape}, expected {shape}")
    if array.dtype != dtype:
        raise TypeError(f"event field {name} has dtype {array.dtype}, expected {dtype}")
    return array


def validate_future_utility_state_gate(
    execution: control.CompositionalControlLifeArmExecution,
    *,
    future_utility_mix: float,
    future_utility_trace_decay: float,
    future_utility_normalization: str,
) -> FutureUtilityStateGateReceipt:
    """Validate exact endpoint-state invariants for one already-completed arm."""

    if type(execution) is not control.CompositionalControlLifeArmExecution:
        raise TypeError("execution must be an exact control-life arm execution")
    decay_bits, normalization_enabled = _validate_mechanism(
        future_utility_mix=future_utility_mix,
        future_utility_trace_decay=future_utility_trace_decay,
        future_utility_normalization=future_utility_normalization,
    )
    if state_field_manifest_sha256() != STATE_FIELD_MANIFEST_SHA256:
        raise RuntimeError("future-state field manifest digest does not reconstruct")

    initial = _state_arrays(execution.initial_state, label="initial")
    final = _state_arrays(execution.final_state, label="final")
    if any(not _all_positive_zero(value) for value in initial.values()):
        raise ValueError("future-state genesis is not exact positive zero")

    marginal_only_fields = (
        "utility_error_trace",
        "utility_feature_trace",
        "candidate_utility_feature_trace",
    )
    if any(not _all_positive_zero(final[name]) for name in marginal_only_fields):
        raise ValueError("contribution mode advanced a marginal-only trace")

    step_words = np.asarray(execution.final_state.step_words)
    if (
        step_words.shape != (2,)
        or step_words.dtype != np.dtype(np.uint32)
        or int(step_words[0]) != 0
        or int(step_words[1]) < 1
    ):
        raise ValueError("final exact lifetime words are invalid for the state gate")
    steps = int(step_words[1])

    trace = execution.events.curation_trace
    active_change = _exact_event_array(
        trace.active_change_mask,
        name="active_change_mask",
        shape=(steps, control.ACTIVE_SLOTS),
        dtype=np.dtype(np.bool_),
    )
    root_change = _exact_event_array(
        trace.root_change_mask,
        name="root_change_mask",
        shape=(steps, control.ACTIVE_SLOTS),
        dtype=np.dtype(np.bool_),
    )
    cascade_refill = _exact_event_array(
        trace.cascade_refill_mask,
        name="cascade_refill_mask",
        shape=(steps, control.ACTIVE_SLOTS),
        dtype=np.dtype(np.bool_),
    )
    if not np.array_equal(active_change, root_change | cascade_refill):
        raise ValueError("active curation masks do not reconstruct their exact union")
    if any(
        np.any(mask[:, : control.RAW_DIM])
        for mask in (active_change, root_change, cascade_refill)
    ):
        raise ValueError("curation mutated a reserved raw active slot")

    expected_energy_bits = expected_raw_energy_f32_bits(
        future_utility_trace_decay,
        steps,
    )
    observed_energy_bits = np.ascontiguousarray(
        final["utility_feature_energy_trace"][: control.RAW_DIM]
    ).view(np.uint32)
    if not np.all(observed_energy_bits == np.uint32(expected_energy_bits)):
        raise ValueError("raw active feature-energy bits do not match the frozen recurrence")

    active_moment = final["utility_signal_second_moment"]
    candidate_moment = final["candidate_utility_signal_second_moment"]
    if normalization_enabled:
        moment_bits = np.concatenate(
            (
                np.ascontiguousarray(active_moment).view(np.uint32),
                np.ascontiguousarray(candidate_moment).view(np.uint32),
            )
        )
        if np.any((moment_bits & np.uint32(0x80000000)) != 0):
            raise ValueError("enabled normalization produced a negative moment")
        if np.any(active_moment[: control.RAW_DIM] <= np.float32(0.0)):
            raise ValueError("enabled normalization lacks a positive raw active moment")
        normalization_policy = "enabled-bounded-endpoint"
    else:
        if not _all_positive_zero(active_moment) or not _all_positive_zero(
            candidate_moment
        ):
            raise ValueError("disabled normalization changed a second-moment field")
        normalization_policy = "disabled-exact-zero"

    active_utility_events = _exact_event_array(
        execution.events.raw_active_utilities,
        name="raw_active_utilities",
        shape=(steps, control.ACTIVE_SLOTS),
        dtype=np.dtype(np.float32),
    )
    candidate_utility_events = _exact_event_array(
        execution.events.raw_candidate_utilities,
        name="raw_candidate_utilities",
        shape=(steps, control.CANDIDATE_SLOTS),
        dtype=np.dtype(np.float32),
    )
    if not np.all(np.isfinite(active_utility_events)) or not np.all(
        np.isfinite(candidate_utility_events)
    ):
        raise ValueError("utility event telemetry is not finite")
    if not np.array_equal(
        np.ascontiguousarray(active_utility_events[-1]).view(np.uint32),
        np.ascontiguousarray(final["utilities"]).view(np.uint32),
    ) or not np.array_equal(
        np.ascontiguousarray(candidate_utility_events[-1]).view(np.uint32),
        np.ascontiguousarray(final["candidate_utilities"]).view(np.uint32),
    ):
        raise ValueError("utility event final row does not match final learner state")

    initial_sha256 = future_utility_state_subset_sha256(execution.initial_state)
    final_sha256 = future_utility_state_subset_sha256(execution.final_state)
    if initial_sha256 == final_sha256:
        raise ValueError("future-state subset did not advance")
    return FutureUtilityStateGateReceipt(
        schema=STATE_GATE_SCHEMA,
        steps=steps,
        trace_decay_f32_bits=decay_bits,
        expected_raw_energy_f32_bits=expected_energy_bits,
        normalization_moment_policy=normalization_policy,
        field_manifest_sha256=STATE_FIELD_MANIFEST_SHA256,
        initial_subset_sha256=initial_sha256,
        final_subset_sha256=final_sha256,
        initial_fields_all_zero=True,
        all_fields_finite=True,
        contribution_mode_zero_marginal_traces=True,
        raw_slots_untouched_by_curation=True,
        raw_energy_bits_exact=True,
        normalization_moment_policy_exact=True,
        utility_event_final_rows_exact=True,
        nonclaims=NONCLAIMS,
        development_only=DEVELOPMENT_ONLY,
        panel_execution_authorized=PANEL_EXECUTION_AUTHORIZED,
        result_authorized=RESULT_AUTHORIZED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "NONCLAIMS",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "RESULT_AUTHORIZED",
    "ROOT_ISSUANCE_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "STATE_FIELD_MANIFEST",
    "STATE_FIELD_MANIFEST_SHA256",
    "STATE_GATE_SCHEMA",
    "V3_RAW_ENERGY_F32_BITS",
    "FutureUtilityStateFieldSpec",
    "FutureUtilityStateGateReceipt",
    "expected_raw_energy_f32_bits",
    "future_utility_state_subset_sha256",
    "state_field_manifest_sha256",
    "validate_future_utility_state_gate",
]
