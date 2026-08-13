# mypy: disable-error-code="call-arg,operator"
"""Consumed, nonpromoting A-B-A diagnostic for the existing FastSlow learner.

One frozen evaluator-owned float32 stream drives an uninterrupted scalar
regression life with targets ``x -> -x -> x``.  The learner receives only the
current observation and target.  Phase identity, boundaries, clocks, and the
mapping sign remain evaluator-side; because every phase has the same input
distribution, a mapping switch is unidentifiable until an outcome arrives.

The exact ordinary :class:`FastSlowLearner` is compared with a same-shape
slow-only ablation whose fast and gate step sizes are zero.  This module owns
no writer, threshold, winner, promotion path, or default-selection mechanism.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxlib
import numpy as np

from alberta_framework.core.fast_slow import (
    FAST_SLOW_STATE_SCHEMA,
    FastSlowConfig,
    FastSlowLearner,
    FastSlowParams,
    FastSlowPredictionParts,
    FastSlowState,
    fast_slow_forward,
    measure_fast_slow_state_nbytes,
)

FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.fast-slow-recurrence-development.protocol.v1"
)
FAST_SLOW_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.fast-slow-recurrence-development.report.v1"
)
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ASSESSMENT_STATUS: Final = "not_assessed"

DEVELOPMENT_ROOT_SEED: Final = 20_260_802
PHASE_STEPS: Final = 512
SUMMARY_WINDOW: Final = 64
INPUT_DIM: Final = 1
OUTPUT_DIM: Final = 1
HIDDEN_DIM: Final = 64
PHASE_NAMES: Final = ("A1", "B", "A2")
ARM_NAMES: Final = ("ordinary_fast_slow", "slow_only_matched_state")
EXECUTION_ENGINES: Final = ("python_eager", "jax_jit_scan")
PARITY_FLOAT_MAX_ABS_TOLERANCE: Final = 1.0e-6

_SOURCE_FOLD_IN: Final = 0x53524345  # "SRCE"
_INITIALIZATION_FOLD_IN: Final = 0x494E4954  # "INIT"
_PARAMETER_FIELDS: Final = (
    "encoder_kernel",
    "encoder_bias",
    "slow_kernel",
    "slow_bias",
    "fast_kernel",
    "fast_bias",
    "gate_kernel",
    "gate_bias",
)
_TRACE_FLOAT_FIELDS: Final = (
    "observation",
    "target",
    "prediction",
    "squared_error",
    "slow_prediction",
    "fast_prediction",
    "gate",
    "gated_fast_contribution",
    "reconstruction_error",
)

LIMITATIONS: Final = (
    "one explicitly consumed development root and one scalar Gaussian life are not a "
    "population, robustness, or scale result",
    "the target mapping changes while the observation distribution does not, so the switch "
    "is unidentifiable until target outcomes arrive",
    "A probes reuse the frozen A1 observations and are descriptive read-only diagnostics",
    "the slow-only arm retains the fast and gate parameters but sets only their step sizes "
    "to zero; matched state shape does not imply matched gradient compute",
    "A2 improvement measures post-switch relearning and cannot establish retention through B",
    "FastSlowState has a finite uint32[2] authoritative lifetime and saturating int32 "
    "telemetry; this short 1536-update lane does not exercise horizon exhaustion or "
    "establish indefinite continual operation",
    "logical prediction/update counts and persistent bytes are not FLOP, allocator, latency, "
    "or energy measurements",
    "there are no thresholds, winner, default selection, artifact writer, held-out seeds, "
    "or scientific-promotion path",
)


@dataclasses.dataclass(frozen=True, slots=True)
class FastSlowRecurrenceProtocol:
    """One fully frozen, already-consumed development protocol."""

    schema_version: str = FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA
    development_root_seed: int = DEVELOPMENT_ROOT_SEED
    phase_steps: int = PHASE_STEPS
    summary_window: int = SUMMARY_WINDOW
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    hidden_dim: int = HIDDEN_DIM

    def __post_init__(self) -> None:
        expected = (
            FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA,
            DEVELOPMENT_ROOT_SEED,
            PHASE_STEPS,
            SUMMARY_WINDOW,
            INPUT_DIM,
            OUTPUT_DIM,
            HIDDEN_DIM,
        )
        actual = (
            self.schema_version,
            self.development_root_seed,
            self.phase_steps,
            self.summary_window,
            self.input_dim,
            self.output_dim,
            self.hidden_dim,
        )
        types_changed = any(
            type(value) is not type(reference)
            for value, reference in zip(actual, expected, strict=True)
        )
        if actual != expected or types_changed:
            raise ValueError("the consumed FastSlow recurrence protocol is frozen")

    @property
    def total_steps(self) -> int:
        return len(PHASE_NAMES) * self.phase_steps

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "output_writes_allowed": False,
            "assessment_status": ASSESSMENT_STATUS,
            "development_root_seed": self.development_root_seed,
            "development_root_consumed": True,
            "seed_or_hyperparameter_search_performed": False,
            "phase_names": list(PHASE_NAMES),
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "summary_window": self.summary_window,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "observation_stream": "jax.random.normal float32 from one folded source key",
            "target_mapping": {"A1": "x", "B": "-x", "A2": "x"},
            "learner_inputs": ["observation", "target"],
            "learner_metadata_exposed": [],
            "switch_identifiability": (
                "unidentifiable from observations; first target after each switch is the "
                "first learner-visible evidence"
            ),
            "a_probe_inputs": "the frozen A1 observation sequence, read-only",
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> FastSlowRecurrenceProtocol:
        protocol = cls()
        if not _exact_json_equal(dict(payload), protocol.to_config()):
            raise ValueError("protocol payload does not match the frozen consumed protocol")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class FastSlowRecurrenceValidation:
    """Strict in-memory validation result."""

    valid: bool
    errors: tuple[str, ...]


class _PhaseTraceArrays(NamedTuple):
    prediction: jax.Array
    squared_error: jax.Array
    slow_prediction: jax.Array
    fast_prediction: jax.Array
    gate: jax.Array
    gated_fast_contribution: jax.Array
    reconstruction_error: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutedArm:
    report: dict[str, object]
    checkpoints: tuple[FastSlowState, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        return set(left_dict) == set(right_dict) and all(
            _exact_json_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(a, b)
            for a, b in zip(left_list, right_list, strict=True)
        )
    return left == right


def _source_keys(protocol: FastSlowRecurrenceProtocol) -> tuple[jax.Array, jax.Array]:
    root = jr.key(protocol.development_root_seed)
    return (
        jr.fold_in(root, _SOURCE_FOLD_IN),
        jr.fold_in(root, _INITIALIZATION_FOLD_IN),
    )


def _array_sha256(*arrays: jax.Array) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        host = np.asarray(jax.device_get(array))
        canonical_dtype = host.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(host.astype(canonical_dtype, copy=False))
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _source_arrays(
    protocol: FastSlowRecurrenceProtocol,
) -> tuple[jax.Array, jax.Array, dict[str, object], jax.Array]:
    root_key = jr.key(protocol.development_root_seed)
    source_key, initialization_key = _source_keys(protocol)
    observations = jr.normal(
        source_key,
        (protocol.total_steps, protocol.input_dim),
        dtype=jnp.float32,
    )
    phase_signs = jnp.concatenate(
        (
            jnp.ones((protocol.phase_steps, 1), dtype=jnp.float32),
            -jnp.ones((protocol.phase_steps, 1), dtype=jnp.float32),
            jnp.ones((protocol.phase_steps, 1), dtype=jnp.float32),
        ),
        axis=0,
    )
    targets = observations * phase_signs
    source_manifest = {
        "generator": "jax.random.normal",
        "dtype": "float32",
        "shape": [protocol.total_steps, protocol.input_dim],
        "development_root_seed": protocol.development_root_seed,
        "root_key_data": [
            int(value) for value in np.asarray(jr.key_data(root_key))
        ],
        "source_fold_in": _SOURCE_FOLD_IN,
        "source_key_data": [int(value) for value in np.asarray(jr.key_data(source_key))],
        "initialization_fold_in": _INITIALIZATION_FOLD_IN,
        "initialization_key_data": [
            int(value) for value in np.asarray(jr.key_data(initialization_key))
        ],
        "one_source_draw_call": True,
        "source_float32_values_drawn": protocol.total_steps * protocol.input_dim,
        "probe_random_draws": 0,
        "input_sha256": _array_sha256(observations, targets),
    }
    source_manifest["manifest_sha256"] = _digest(source_manifest)
    return observations, targets, source_manifest, initialization_key


def _ordinary_config(protocol: FastSlowRecurrenceProtocol) -> FastSlowConfig:
    return FastSlowConfig(input_dim=protocol.input_dim)


def _arm_config(
    protocol: FastSlowRecurrenceProtocol,
    arm_name: str,
) -> FastSlowConfig:
    ordinary = _ordinary_config(protocol)
    if arm_name == ARM_NAMES[0]:
        return ordinary
    if arm_name == ARM_NAMES[1]:
        return FastSlowConfig(
            input_dim=ordinary.input_dim,
            output_dim=ordinary.output_dim,
            hidden_dim=ordinary.hidden_dim,
            encoder_step_size=ordinary.encoder_step_size,
            slow_step_size=ordinary.slow_step_size,
            fast_step_size=0.0,
            gate_step_size=0.0,
            fast_decay=ordinary.fast_decay,
            slow_weight_decay=ordinary.slow_weight_decay,
            gate_l2=ordinary.gate_l2,
            grad_clip=ordinary.grad_clip,
            init_scale=ordinary.init_scale,
        )
    raise ValueError("unsupported FastSlow recurrence arm")


def _state_resource_payload(state: FastSlowState) -> dict[str, object]:
    fields: dict[str, int] = {}
    for name in _PARAMETER_FIELDS:
        fields[name] = int(np.asarray(jax.device_get(getattr(state.params, name))).nbytes)
    fields["step_count"] = int(np.asarray(jax.device_get(state.step_count)).nbytes)
    fields["step_words"] = int(np.asarray(jax.device_get(state.step_words)).nbytes)
    total_nbytes = sum(fields.values())
    if total_nbytes != measure_fast_slow_state_nbytes(state):
        raise RuntimeError("FastSlow resource field accounting is incomplete")
    return {
        "state_schema": FAST_SLOW_STATE_SCHEMA,
        "fields_nbytes": fields,
        "parameter_nbytes": sum(fields[name] for name in _PARAMETER_FIELDS),
        "step_count_nbytes": fields["step_count"],
        "step_words_nbytes": fields["step_words"],
        "lifetime_counter_nbytes": fields["step_count"] + fields["step_words"],
        "exact_lifetime_identity_nbytes": fields["step_words"],
        "total_nbytes": total_nbytes,
        "step_count_dtype": str(state.step_count.dtype),
        "step_words_dtype": str(state.step_words.dtype),
        "step_words_order": "big-endian high-low",
        "step_count_indefinite_operation_established": False,
    }


def _state_sha256(state: FastSlowState) -> str:
    payload: list[dict[str, object]] = []
    for name, value in (
        *((name, getattr(state.params, name)) for name in _PARAMETER_FIELDS),
        ("step_count", state.step_count),
        ("step_words", state.step_words),
    ):
        host = np.asarray(jax.device_get(value))
        canonical_dtype = host.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(host.astype(canonical_dtype, copy=False))
        payload.append(
            {
                "name": name,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "data_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
            }
        )
    return _digest(payload)


def _group_norm(params: FastSlowParams, names: Sequence[str]) -> float:
    squared = 0.0
    for name in names:
        host = np.asarray(jax.device_get(getattr(params, name)), dtype=np.float64)
        squared += float(np.sum(np.square(host)))
    return math.sqrt(squared)


def _group_drift(
    params: FastSlowParams,
    reference: FastSlowParams,
    names: Sequence[str],
) -> float:
    squared = 0.0
    for name in names:
        current = np.asarray(jax.device_get(getattr(params, name)), dtype=np.float64)
        baseline = np.asarray(jax.device_get(getattr(reference, name)), dtype=np.float64)
        squared += float(np.sum(np.square(current - baseline)))
    return math.sqrt(squared)


_PARAMETER_GROUPS: Final = {
    "encoder": ("encoder_kernel", "encoder_bias"),
    "slow": ("slow_kernel", "slow_bias"),
    "fast": ("fast_kernel", "fast_bias"),
    "gate": ("gate_kernel", "gate_bias"),
    "all": _PARAMETER_FIELDS,
}


def _parameter_norms(params: FastSlowParams) -> dict[str, float]:
    return {
        name: _group_norm(params, fields)
        for name, fields in _PARAMETER_GROUPS.items()
    }


def _parameter_drift(
    params: FastSlowParams,
    reference: FastSlowParams,
) -> dict[str, float]:
    return {
        name: _group_drift(params, reference, fields)
        for name, fields in _PARAMETER_GROUPS.items()
    }


def _probe_payload(state: FastSlowState, observations: jax.Array) -> dict[str, float]:
    host_parts = jax.device_get(
        jax.vmap(lambda observation: fast_slow_forward(state.params, observation))(
            observations
        )
    )
    prediction = np.asarray(host_parts.prediction, dtype=np.float64).reshape(-1)
    slow = np.asarray(host_parts.slow_prediction, dtype=np.float64).reshape(-1)
    fast = np.asarray(host_parts.fast_prediction, dtype=np.float64).reshape(-1)
    gate = np.asarray(host_parts.gate, dtype=np.float64).reshape(-1)
    targets = np.asarray(jax.device_get(observations[:, 0]), dtype=np.float64)
    gated_fast = gate * fast
    reconstruction = prediction - slow - gated_fast
    return {
        "examples": int(targets.size),
        "full_a_mse": float(np.mean(np.square(prediction - targets))),
        "slow_component_a_mse": float(np.mean(np.square(slow - targets))),
        "gated_fast_only_a_mse": float(np.mean(np.square(gated_fast - targets))),
        "prediction_mean": float(np.mean(prediction)),
        "slow_prediction_mean": float(np.mean(slow)),
        "fast_prediction_mean": float(np.mean(fast)),
        "gated_fast_contribution_mean": float(np.mean(gated_fast)),
        "gated_fast_contribution_abs_mean": float(np.mean(np.abs(gated_fast))),
        "gate_mean": float(np.mean(gate)),
        "gate_minimum": float(np.min(gate)),
        "gate_maximum": float(np.max(gate)),
        "gate_l2_norm": float(np.linalg.norm(gate)),
        "decomposition_max_abs_error": float(np.max(np.abs(reconstruction))),
    }


def _checkpoint_payload(
    label: str,
    state: FastSlowState,
    initial_state: FastSlowState,
    previous_state: FastSlowState,
    probe_observations: jax.Array,
) -> dict[str, object]:
    return {
        "label": label,
        "step_count": int(np.asarray(jax.device_get(state.step_count))),
        "step_words": [
            int(value) for value in np.asarray(jax.device_get(state.step_words))
        ],
        "state_sha256": _state_sha256(state),
        "resources": _state_resource_payload(state),
        "parameter_norms": _parameter_norms(state.params),
        "parameter_drift_from_initial": _parameter_drift(
            state.params, initial_state.params
        ),
        "parameter_drift_from_previous_checkpoint": _parameter_drift(
            state.params, previous_state.params
        ),
        "a_probe": _probe_payload(state, probe_observations),
    }


def _trace_event(
    event_index: int,
    phase_index: int,
    phase_step: int,
    observation: object,
    target: object,
    parts: FastSlowPredictionParts,
) -> dict[str, object]:
    prediction = float(np.asarray(jax.device_get(parts.prediction))[0])
    slow = float(np.asarray(jax.device_get(parts.slow_prediction))[0])
    fast = float(np.asarray(jax.device_get(parts.fast_prediction))[0])
    gate = float(np.asarray(jax.device_get(parts.gate))[0])
    observation_value = float(np.asarray(jax.device_get(observation))[0])
    target_value = float(np.asarray(jax.device_get(target))[0])
    gated_fast = gate * fast
    return {
        "event_index": event_index,
        "phase": PHASE_NAMES[phase_index],
        "phase_step": phase_step,
        "observation": observation_value,
        "target": target_value,
        "prediction": prediction,
        "squared_error": (prediction - target_value) ** 2,
        "slow_prediction": slow,
        "fast_prediction": fast,
        "gate": gate,
        "gated_fast_contribution": gated_fast,
        "reconstruction_error": prediction - slow - gated_fast,
    }


def _run_phase_eager(
    learner: FastSlowLearner,
    state: FastSlowState,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> tuple[FastSlowState, list[dict[str, object]]]:
    trace: list[dict[str, object]] = []
    current = state
    for phase_step in range(observations.shape[0]):
        observation = observations[phase_step]
        target = targets[phase_step]
        parts = learner.predict_parts(current, observation)
        result = learner.update(current, observation, target)
        if not np.array_equal(
            np.asarray(jax.device_get(parts.prediction)),
            np.asarray(jax.device_get(result.prediction)),
        ):
            raise RuntimeError("FastSlow pre-update prediction did not reconstruct")
        trace.append(
            _trace_event(
                event_offset + phase_step,
                phase_index,
                phase_step,
                observation,
                target,
                parts,
            )
        )
        current = result.state
    return current, trace


@functools.partial(jax.jit, static_argnums=(0,))
def _run_phase_compiled(
    learner: FastSlowLearner,
    state: FastSlowState,
    observations: jax.Array,
    targets: jax.Array,
) -> tuple[FastSlowState, _PhaseTraceArrays]:
    def step(
        current: FastSlowState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[FastSlowState, _PhaseTraceArrays]:
        observation, target = inputs
        parts = learner.predict_parts(current, observation)
        result = learner.update(current, observation, target)
        prediction = parts.prediction[0]
        slow = parts.slow_prediction[0]
        fast = parts.fast_prediction[0]
        gate = parts.gate[0]
        gated_fast = gate * fast
        return result.state, _PhaseTraceArrays(
            prediction=prediction,
            squared_error=jnp.square(prediction - target[0]),
            slow_prediction=slow,
            fast_prediction=fast,
            gate=gate,
            gated_fast_contribution=gated_fast,
            reconstruction_error=prediction - slow - gated_fast,
        )

    return cast(
        tuple[FastSlowState, _PhaseTraceArrays],
        jax.lax.scan(step, state, (observations, targets)),
    )


def _compiled_trace_events(
    arrays: _PhaseTraceArrays,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> list[dict[str, object]]:
    host = cast(_PhaseTraceArrays, jax.device_get(arrays))
    host_observations = np.asarray(jax.device_get(observations), dtype=np.float32)
    host_targets = np.asarray(jax.device_get(targets), dtype=np.float32)
    return [
        {
            "event_index": event_offset + phase_step,
            "phase": PHASE_NAMES[phase_index],
            "phase_step": phase_step,
            "observation": float(host_observations[phase_step, 0]),
            "target": float(host_targets[phase_step, 0]),
            "prediction": float(np.asarray(host.prediction)[phase_step]),
            "squared_error": float(np.asarray(host.squared_error)[phase_step]),
            "slow_prediction": float(np.asarray(host.slow_prediction)[phase_step]),
            "fast_prediction": float(np.asarray(host.fast_prediction)[phase_step]),
            "gate": float(np.asarray(host.gate)[phase_step]),
            "gated_fast_contribution": float(
                np.asarray(host.gated_fast_contribution)[phase_step]
            ),
            "reconstruction_error": float(
                np.asarray(host.reconstruction_error)[phase_step]
            ),
        }
        for phase_step in range(observations.shape[0])
    ]


def _window_summary(events: Sequence[Mapping[str, object]]) -> dict[str, float]:
    squared_error = np.asarray(
        [cast(float, event["squared_error"]) for event in events],
        dtype=np.float64,
    )
    target = np.asarray(
        [cast(float, event["target"]) for event in events], dtype=np.float64
    )
    slow = np.asarray(
        [cast(float, event["slow_prediction"]) for event in events],
        dtype=np.float64,
    )
    fast = np.asarray(
        [cast(float, event["fast_prediction"]) for event in events],
        dtype=np.float64,
    )
    gate = np.asarray(
        [cast(float, event["gate"]) for event in events], dtype=np.float64
    )
    gated_fast = np.asarray(
        [cast(float, event["gated_fast_contribution"]) for event in events],
        dtype=np.float64,
    )
    reconstruction = np.asarray(
        [cast(float, event["reconstruction_error"]) for event in events],
        dtype=np.float64,
    )
    return {
        "prequential_mse": float(np.mean(squared_error)),
        "slow_component_mse": float(np.mean(np.square(slow - target))),
        "gated_fast_only_mse": float(np.mean(np.square(gated_fast - target))),
        "prediction_mean": float(
            np.mean([cast(float, event["prediction"]) for event in events])
        ),
        "slow_prediction_mean": float(np.mean(slow)),
        "fast_prediction_mean": float(np.mean(fast)),
        "gated_fast_contribution_mean": float(np.mean(gated_fast)),
        "gated_fast_contribution_abs_mean": float(np.mean(np.abs(gated_fast))),
        "gate_mean": float(np.mean(gate)),
        "gate_minimum": float(np.min(gate)),
        "gate_maximum": float(np.max(gate)),
        "decomposition_max_abs_error": float(np.max(np.abs(reconstruction))),
    }


def _metrics_from_trace(
    trace: Sequence[Mapping[str, object]],
    protocol: FastSlowRecurrenceProtocol,
) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        stop = start + protocol.phase_steps
        events = trace[start:stop]
        phases[phase_name] = {
            "start": start,
            "stop": stop,
            "early": _window_summary(events[: protocol.summary_window]),
            "tail": _window_summary(events[-protocol.summary_window :]),
            "mean": _window_summary(events),
        }
    a1 = cast(Mapping[str, Mapping[str, float]], phases["A1"])
    b = cast(Mapping[str, Mapping[str, float]], phases["B"])
    a2 = cast(Mapping[str, Mapping[str, float]], phases["A2"])
    a1_tail = a1["tail"]["prequential_mse"]
    b_early = b["early"]["prequential_mse"]
    b_tail = b["tail"]["prequential_mse"]
    a2_early = a2["early"]["prequential_mse"]
    a2_tail = a2["tail"]["prequential_mse"]
    return {
        "phase": phases,
        "switch_adaptation": {
            "b_entry_minus_a1_tail_mse": b_early - a1_tail,
            "b_tail_minus_b_entry_mse": b_tail - b_early,
            "a2_entry_minus_b_tail_mse": a2_early - b_tail,
            "a2_tail_minus_a2_entry_mse": a2_tail - a2_early,
        },
        "recurrence": {
            "a2_entry_minus_a1_tail_mse": a2_early - a1_tail,
            "a2_tail_minus_a1_tail_mse": a2_tail - a1_tail,
            "a2_within_phase_relearning_mse": a2_early - a2_tail,
            "relearning_is_not_retention": True,
        },
    }


def _slow_component_audit(checkpoints: Mapping[str, object]) -> dict[str, object]:
    def slow_mse(label: str) -> float:
        checkpoint = cast(Mapping[str, object], checkpoints[label])
        probe = cast(Mapping[str, object], checkpoint["a_probe"])
        return cast(float, probe["slow_component_a_mse"])

    a1 = slow_mse("A1_end")
    b = slow_mse("B_end")
    a2_entry = slow_mse("A2_entry")
    a2_tail = slow_mse("A2_tail")
    return {
        "question": "does the full arm slow component itself preserve A through B?",
        "a1_end_slow_component_a_mse": a1,
        "b_end_slow_component_a_mse": b,
        "b_end_minus_a1_end_slow_component_a_mse": b - a1,
        "b_end_over_a1_end_slow_component_a_mse": b / a1 if a1 > 0.0 else None,
        "a2_entry_slow_component_a_mse": a2_entry,
        "a2_entry_equals_b_end": a2_entry == b,
        "a2_tail_slow_component_a_mse": a2_tail,
        "a2_tail_relearning_reduction": a2_entry - a2_tail,
        "a2_tail_relearning_counted_as_retention": False,
        "threshold_or_verdict_applied": False,
    }


def _execute_arm(
    protocol: FastSlowRecurrenceProtocol,
    observations: jax.Array,
    targets: jax.Array,
    initialization_key: jax.Array,
    *,
    arm_name: str,
    engine: str,
) -> _ExecutedArm:
    config = _arm_config(protocol, arm_name)
    learner = FastSlowLearner(config)
    initial_state = learner.init(initialization_key)
    state = initial_state
    probe_observations = observations[: protocol.phase_steps]
    trace: list[dict[str, object]] = []
    checkpoint_states: list[FastSlowState] = [initial_state]
    checkpoints: dict[str, object] = {
        "initial": _checkpoint_payload(
            "initial", initial_state, initial_state, initial_state, probe_observations
        )
    }
    previous_checkpoint_state = initial_state
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        stop = start + protocol.phase_steps
        if phase_name == "A2":
            checkpoints["A2_entry"] = _checkpoint_payload(
                "A2_entry",
                state,
                initial_state,
                previous_checkpoint_state,
                probe_observations,
            )
            checkpoint_states.append(state)
            previous_checkpoint_state = state
        if engine == "python_eager":
            state, phase_trace = _run_phase_eager(
                learner,
                state,
                observations[start:stop],
                targets[start:stop],
                phase_index=phase_index,
                event_offset=start,
            )
        elif engine == "jax_jit_scan":
            state, arrays = _run_phase_compiled(
                learner,
                state,
                observations[start:stop],
                targets[start:stop],
            )
            phase_trace = _compiled_trace_events(
                arrays,
                observations[start:stop],
                targets[start:stop],
                phase_index=phase_index,
                event_offset=start,
            )
        else:
            raise ValueError("unsupported FastSlow recurrence engine")
        trace.extend(phase_trace)
        label = "A2_tail" if phase_name == "A2" else f"{phase_name}_end"
        checkpoints[label] = _checkpoint_payload(
            label,
            state,
            initial_state,
            previous_checkpoint_state,
            probe_observations,
        )
        checkpoint_states.append(state)
        previous_checkpoint_state = state

    b_end = cast(Mapping[str, object], checkpoints["B_end"])
    a2_entry = cast(Mapping[str, object], checkpoints["A2_entry"])
    if b_end["state_sha256"] != a2_entry["state_sha256"]:
        raise RuntimeError("A2 entry changed learner state before an outcome arrived")
    initial_resources = _state_resource_payload(initial_state)
    final_resources = _state_resource_payload(state)
    if initial_resources != final_resources:
        raise RuntimeError("FastSlow persistent state shape changed during the life")
    logical_probe_examples = len(checkpoints) * protocol.phase_steps
    metrics = _metrics_from_trace(trace, protocol)
    metrics["decisive_slow_component_audit"] = _slow_component_audit(checkpoints)
    report = {
        "arm": arm_name,
        "engine": engine,
        "learner_config": learner.to_config(),
        "learner_config_sha256": _digest(learner.to_config()),
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": metrics,
        "checkpoints": checkpoints,
        "resources": {
            "initial_state": initial_resources,
            "final_state": final_resources,
            "fixed_allocation": True,
            "logical_peak_state_nbytes": initial_resources["total_nbytes"],
        },
        "work": {
            "logical_updates": protocol.total_steps,
            "logical_online_predict_before_update_examples": protocol.total_steps,
            "logical_probe_prediction_examples": logical_probe_examples,
            "logical_explicit_prediction_examples": (
                protocol.total_steps + logical_probe_examples
            ),
            "logical_forward_examples_inside_updates": protocol.total_steps,
            "logical_total_forward_examples": (
                2 * protocol.total_steps + logical_probe_examples
            ),
            "update_random_draws": 0,
            "prediction_random_draws": 0,
        },
        "rng": {
            "initialization_owned_by": "FastSlowLearner.init",
            "common_initialization_key_across_arms": True,
            "encoder_float32_values_drawn": protocol.input_dim * protocol.hidden_dim,
            "gate_float32_values_drawn": protocol.hidden_dim * protocol.output_dim,
            "readout_initialization_random_draws": 0,
            "online_random_draws": 0,
        },
    }
    return _ExecutedArm(
        report=cast(dict[str, object], _json_clone(report)),
        checkpoints=tuple(checkpoint_states),
    )


def _tree_max_abs_difference(left: object, right: object) -> float:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        raise ValueError("parity states do not have the same tree")
    maximum = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_host = np.asarray(jax.device_get(left_leaf))
        right_host = np.asarray(jax.device_get(right_leaf))
        if left_host.shape != right_host.shape or left_host.dtype != right_host.dtype:
            raise ValueError("parity state leaves do not have the same shape and dtype")
        if np.issubdtype(left_host.dtype, np.integer):
            if not np.array_equal(left_host, right_host):
                return math.inf
        else:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            left_host.astype(np.float64)
                            - right_host.astype(np.float64)
                        )
                    )
                ),
            )
    return maximum


def _parity_payload(
    eager: Sequence[_ExecutedArm],
    compiled: Sequence[_ExecutedArm],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for eager_arm, compiled_arm in zip(eager, compiled, strict=True):
        if eager_arm.report["arm"] != compiled_arm.report["arm"]:
            raise RuntimeError("eager and compiled arm order diverged")
        eager_trace = cast(list[Mapping[str, object]], eager_arm.report["trace"])
        compiled_trace = cast(
            list[Mapping[str, object]], compiled_arm.report["trace"]
        )
        float_max: dict[str, float] = {}
        for name in _TRACE_FLOAT_FIELDS:
            float_max[name] = max(
                abs(cast(float, eager_event[name]) - cast(float, compiled_event[name]))
                for eager_event, compiled_event in zip(
                    eager_trace, compiled_trace, strict=True
                )
            )
        discrete_exact = all(
            eager_event["event_index"] == compiled_event["event_index"]
            and eager_event["phase"] == compiled_event["phase"]
            and eager_event["phase_step"] == compiled_event["phase_step"]
            for eager_event, compiled_event in zip(
                eager_trace, compiled_trace, strict=True
            )
        )
        checkpoint_state_max = max(
            _tree_max_abs_difference(eager_state, compiled_state)
            for eager_state, compiled_state in zip(
                eager_arm.checkpoints, compiled_arm.checkpoints, strict=True
            )
        )
        observed_max = max(*float_max.values(), checkpoint_state_max)
        arm_name = cast(str, eager_arm.report["arm"])
        arms[arm_name] = {
            "trace_discrete_fields_exact": discrete_exact,
            "trace_float_max_abs_difference": float_max,
            "checkpoint_state_max_abs_difference": checkpoint_state_max,
            "observed_max_abs_difference": observed_max,
            "declared_numeric_tolerance": PARITY_FLOAT_MAX_ABS_TOLERANCE,
            "within_declared_numeric_tolerance": (
                discrete_exact and observed_max <= PARITY_FLOAT_MAX_ABS_TOLERANCE
            ),
            "config_exact": eager_arm.report["learner_config"]
            == compiled_arm.report["learner_config"],
            "resources_exact": eager_arm.report["resources"]
            == compiled_arm.report["resources"],
            "work_exact": eager_arm.report["work"] == compiled_arm.report["work"],
            "full_state_digest_equality_claimed": False,
        }
    return {
        "technical_float_tolerance_not_an_outcome_threshold": True,
        "arms": arms,
    }


def _runtime_identity() -> dict[str, object]:
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": getattr(jaxlib, "__version__", "unknown"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "byteorder": sys.byteorder,
        "execution_engines": list(EXECUTION_ENGINES),
    }
    runtime["runtime_identity_sha256"] = _digest(runtime)
    return runtime


def _source_identity() -> dict[str, object]:
    evaluator_path = Path(__file__)
    core_path = evaluator_path.parents[1] / "core" / "fast_slow.py"
    payload: dict[str, object] = {
        "evaluator_module_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "fast_slow_core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
    }
    payload["source_identity_sha256"] = _digest(payload)
    return payload


def _build_report() -> dict[str, object]:
    protocol = FastSlowRecurrenceProtocol()
    observations, targets, source_manifest, initialization_key = _source_arrays(protocol)
    executions: dict[str, list[_ExecutedArm]] = {}
    for engine in EXECUTION_ENGINES:
        executions[engine] = [
            _execute_arm(
                protocol,
                observations,
                targets,
                initialization_key,
                arm_name=arm_name,
                engine=engine,
            )
            for arm_name in ARM_NAMES
        ]
    ordinary = _ordinary_config(protocol).to_config()
    slow_only = _arm_config(protocol, ARM_NAMES[1]).to_config()
    differing_fields = {
        name: {"ordinary_fast_slow": ordinary[name], "slow_only_matched_state": slow_only[name]}
        for name in ordinary
        if ordinary[name] != slow_only[name]
    }
    consumed_full = executions["jax_jit_scan"][0].report
    consumed_full_metrics = cast(Mapping[str, object], consumed_full["metrics"])
    consumed_audit = cast(
        Mapping[str, object],
        consumed_full_metrics["decisive_slow_component_audit"],
    )
    consumed_checkpoints = cast(Mapping[str, object], consumed_full["checkpoints"])
    consumed_a1_probe = cast(
        Mapping[str, object],
        cast(Mapping[str, object], consumed_checkpoints["A1_end"])["a_probe"],
    )
    consumed_b_probe = cast(
        Mapping[str, object],
        cast(Mapping[str, object], consumed_checkpoints["B_end"])["a_probe"],
    )
    consumed_a2_probe = cast(
        Mapping[str, object],
        cast(Mapping[str, object], consumed_checkpoints["A2_tail"])["a_probe"],
    )
    body: dict[str, object] = {
        "schema_version": FAST_SLOW_RECURRENCE_REPORT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "output_writes_allowed": False,
        "assessment_status": ASSESSMENT_STATUS,
        "consumed_development_result": True,
        "protocol": protocol.to_config(),
        "protocol_sha256": _digest(protocol.to_config()),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "arm_order": list(ARM_NAMES),
        "arm_comparison": {
            "ordinary_uses_core_defaults_except_required_input_dim": True,
            "slow_only_retains_all_parameter_arrays": True,
            "only_config_differences": differing_fields,
            "expected_difference_fields": ["fast_step_size", "gate_step_size"],
            "winner_selected": False,
            "default_selected": False,
        },
        "consumed_findings": {
            "source_engine": "jax_jit_scan",
            "ordinary_full_arm": {
                "a1_end_full_a_probe_mse": consumed_a1_probe["full_a_mse"],
                "b_end_full_a_probe_mse": consumed_b_probe["full_a_mse"],
                "a2_tail_full_a_probe_mse": consumed_a2_probe["full_a_mse"],
                "a1_end_slow_component_a_probe_mse": consumed_audit[
                    "a1_end_slow_component_a_mse"
                ],
                "b_end_slow_component_a_probe_mse": consumed_audit[
                    "b_end_slow_component_a_mse"
                ],
                "b_end_minus_a1_end_slow_component_a_probe_mse": consumed_audit[
                    "b_end_minus_a1_end_slow_component_a_mse"
                ],
                "b_end_over_a1_end_slow_component_a_probe_mse": consumed_audit[
                    "b_end_over_a1_end_slow_component_a_mse"
                ],
                "a2_tail_slow_component_a_probe_mse": consumed_audit[
                    "a2_tail_slow_component_a_mse"
                ],
                "slow_component_retention_through_b_demonstrated": False,
                "a2_relearning_used_as_retention_evidence": False,
                "scope": "this one consumed development life only",
                "outcome_threshold_applied": False,
            },
            "negative_finding": (
                "the ordinary full arm's slow component did not preserve the A mapping "
                "through B in this consumed life; later A2 improvement is relearning"
            ),
            "winner_or_default_selected": False,
        },
        "executions": {
            engine: [executed.report for executed in executions[engine]]
            for engine in EXECUTION_ENGINES
        },
        "eager_compiled_parity": _parity_payload(
            executions["python_eager"], executions["jax_jit_scan"]
        ),
        "limitations": list(LIMITATIONS),
    }
    body["causal_reconstruction_sha256"] = _digest(
        {
            "protocol_sha256": body["protocol_sha256"],
            "source_manifest_sha256": body["source_manifest_sha256"],
            "source_identity": body["source_identity"],
            "runtime_identity": body["runtime_identity"],
            "executions": body["executions"],
        }
    )
    report = {**body, "report_sha256": _digest(body)}
    return cast(dict[str, object], _json_clone(report))


@functools.lru_cache(maxsize=1)
def _expected_report_json() -> str:
    """Execute the one consumed life once per process and cache only its JSON."""

    return _canonical_json(_build_report())


def run_fast_slow_recurrence_development() -> dict[str, object]:
    """Return the one frozen in-memory consumed development result."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_fast_slow_recurrence_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated FastSlow recurrence report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def validate_fast_slow_recurrence_report(
    report: Mapping[str, object],
) -> FastSlowRecurrenceValidation:
    """Fail closed against the deterministic causal reconstruction."""

    errors: list[str] = []
    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return FastSlowRecurrenceValidation(False, (f"report is not canonical JSON: {error}",))
    expected = cast(dict[str, object], json.loads(_expected_report_json()))
    if not _exact_json_equal(candidate, expected):
        errors.append("report does not match the frozen causal reconstruction")
    body = {name: value for name, value in candidate.items() if name != "report_sha256"}
    if candidate.get("report_sha256") != _digest(body):
        errors.append("report_sha256 does not reconstruct")
    if candidate.get("causal_reconstruction_sha256") != _digest(
        {
            "protocol_sha256": candidate.get("protocol_sha256"),
            "source_manifest_sha256": candidate.get("source_manifest_sha256"),
            "source_identity": candidate.get("source_identity"),
            "runtime_identity": candidate.get("runtime_identity"),
            "executions": candidate.get("executions"),
        }
    ):
        errors.append("causal_reconstruction_sha256 does not reconstruct")
    return FastSlowRecurrenceValidation(not errors, tuple(errors))


def fast_slow_recurrence_report_json(report: Mapping[str, object]) -> str:
    """Serialize only a valid in-memory report; this function performs no write."""

    validation = validate_fast_slow_recurrence_report(report)
    if not validation.valid:
        raise ValueError("invalid FastSlow recurrence report: " + "; ".join(validation.errors))
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_ROOT_SEED",
    "FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA",
    "FAST_SLOW_RECURRENCE_REPORT_SCHEMA",
    "FastSlowRecurrenceProtocol",
    "FastSlowRecurrenceValidation",
    "LIMITATIONS",
    "OUTPUT_WRITES_ALLOWED",
    "PARITY_FLOAT_MAX_ABS_TOLERANCE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "fast_slow_recurrence_report_json",
    "run_fast_slow_recurrence_development",
    "validate_fast_slow_recurrence_report",
]
