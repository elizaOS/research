# mypy: disable-error-code="call-arg,operator"
"""Consumed-root diagnostic for an Alberta-derived permanent/transient learner.

The evaluator reuses the exact already-consumed FastSlow A/B/A observations,
targets, root seed, and initialization key.  It performs no tuning or new
random draw.  The implemented learner is explicitly not source-faithful; its
machine-readable design record enumerates departures from the OpenReview paper
and earlier public code.

Two equal-state/equal-work arms are run: ordinary online consolidation and a
no-consolidation ablation whose permanent step sizes are zero.  Permanent A
probes are recorded directly after A1 and directly after B, before any A2
update, so later reacquisition cannot be counted as retention.  This module
has no writer, threshold, winner, default, evidence, or promotion path.
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
import jaxlib
import numpy as np

from alberta_framework.core.fast_slow import FastSlowLearner
from alberta_framework.core.permanent_transient import (
    AlbertaPermanentTransientConfig,
    AlbertaPermanentTransientLearner,
    AlbertaPermanentTransientParams,
    AlbertaPermanentTransientPrediction,
    AlbertaPermanentTransientState,
    permanent_transient_design_record,
    permanent_transient_forward,
)
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    DEVELOPMENT_ROOT_SEED,
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    PHASE_NAMES,
    PHASE_STEPS,
    SUMMARY_WINDOW,
    FastSlowRecurrenceProtocol,
    _ordinary_config,
    _source_arrays,
)

PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.permanent-transient-recurrence-development.protocol.v1"
)
PERMANENT_TRANSIENT_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.permanent-transient-recurrence-development.report.v1"
)
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ASSESSMENT_STATUS: Final = "not_assessed"

ARM_NAMES: Final = (
    "alberta_pt_online_consolidation",
    "alberta_pt_no_consolidation",
)
EXECUTION_ENGINES: Final = ("python_eager", "jax_jit_scan")
# The compared trace includes squared errors, whose first-order discrepancy is
# about twice the underlying float32 prediction discrepancy.  This is a
# technical eager/XLA reconstruction bound, never an outcome threshold.
PARITY_FLOAT_MAX_ABS_TOLERANCE: Final = 2.0e-6
_PARAMETER_FIELDS: Final = (
    "permanent_encoder_kernel",
    "permanent_encoder_bias",
    "permanent_head_kernel",
    "permanent_head_bias",
    "transient_encoder_kernel",
    "transient_encoder_bias",
    "transient_head_kernel",
    "transient_head_bias",
)
_PERMANENT_FIELDS: Final = _PARAMETER_FIELDS[:4]
_TRANSIENT_FIELDS: Final = _PARAMETER_FIELDS[4:]
_TRACE_FLOAT_FIELDS: Final = (
    "observation",
    "target",
    "prediction",
    "squared_error",
    "permanent_squared_error",
    "permanent_prediction",
    "transient_prediction",
    "reconstruction_error",
)

LIMITATIONS: Final = (
    "this is an Alberta-derived supervised regression baseline, not a source-faithful "
    "implementation of the cited paper or its Craftax method",
    "the one consumed scalar Gaussian root is neither a population nor a robustness, control, "
    "or scale result",
    "the ordinary and no-consolidation arms are equal-state and equal-work, but comparison "
    "with FastSlow is resource-declared rather than shape- or gradient-work-matched",
    "the k=1 current-sample consolidation rule has no replay and is a deliberate departure "
    "from task-buffer and periodic-buffer source algorithms",
    "permanent A probes reuse frozen A1 observations and are read-only diagnostics",
    "the B-end probe occurs before any A2 update; A2 changes are reported only as reacquisition",
    "one finite uint32[2] lifetime does not establish indefinite continual operation",
    "there is no threshold, winner, default selection, writer, held-out seed, evidence, or "
    "scientific-promotion path",
)


@dataclasses.dataclass(frozen=True, slots=True)
class PermanentTransientRecurrenceProtocol:
    """Frozen sibling protocol bound to the consumed FastSlow root."""

    schema_version: str = PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA
    development_root_seed: int = DEVELOPMENT_ROOT_SEED
    phase_steps: int = PHASE_STEPS
    summary_window: int = SUMMARY_WINDOW
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    total_hidden_features: int = HIDDEN_DIM

    def __post_init__(self) -> None:
        expected = (
            PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA,
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
            self.total_hidden_features,
        )
        type_changed = any(
            type(value) is not type(reference)
            for value, reference in zip(actual, expected, strict=True)
        )
        if actual != expected or type_changed:
            raise ValueError("the consumed permanent/transient protocol is frozen")

    @property
    def total_steps(self) -> int:
        return len(PHASE_NAMES) * self.phase_steps

    def to_config(self) -> dict[str, object]:
        """Return the exact no-tuning protocol record."""

        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "output_writes_allowed": False,
            "assessment_status": ASSESSMENT_STATUS,
            "development_root_seed": self.development_root_seed,
            "development_root_already_consumed_by_fast_slow": True,
            "new_seed_drawn": False,
            "seed_or_hyperparameter_search_performed": False,
            "phase_names": list(PHASE_NAMES),
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "summary_window": self.summary_window,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "total_hidden_features": self.total_hidden_features,
            "permanent_hidden_features": self.total_hidden_features // 2,
            "transient_hidden_features": self.total_hidden_features // 2,
            "target_mapping": {"A1": "x", "B": "-x", "A2": "x"},
            "learner_inputs": ["observation", "target"],
            "learner_metadata_exposed": [],
            "a_probe_inputs": "the frozen A1 observations, read-only",
            "pre_post_b_permanent_probe_has_no_a2_updates": True,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PermanentTransientRecurrenceProtocol:
        protocol = cls()
        if not _exact_json_equal(dict(payload), protocol.to_config()):
            raise ValueError("protocol payload does not match the frozen consumed protocol")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class PermanentTransientRecurrenceValidation:
    """Strict in-memory reconstruction result."""

    valid: bool
    errors: tuple[str, ...]


class _TraceArrays(NamedTuple):
    prediction: jax.Array
    squared_error: jax.Array
    permanent_squared_error: jax.Array
    permanent_prediction: jax.Array
    transient_prediction: jax.Array
    reconstruction_error: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutedArm:
    report: dict[str, object]
    checkpoints: tuple[AlbertaPermanentTransientState, ...]


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


def _source_arrays_bound(
    protocol: PermanentTransientRecurrenceProtocol,
) -> tuple[jax.Array, jax.Array, dict[str, object], jax.Array]:
    fast_slow_protocol = FastSlowRecurrenceProtocol(
        development_root_seed=protocol.development_root_seed,
        phase_steps=protocol.phase_steps,
        summary_window=protocol.summary_window,
        input_dim=protocol.input_dim,
        output_dim=protocol.output_dim,
        hidden_dim=protocol.total_hidden_features,
    )
    return _source_arrays(fast_slow_protocol)


def _arm_config(
    protocol: PermanentTransientRecurrenceProtocol,
    arm_name: str,
) -> AlbertaPermanentTransientConfig:
    half = protocol.total_hidden_features // 2
    if arm_name == ARM_NAMES[0]:
        return AlbertaPermanentTransientConfig(
            input_dim=protocol.input_dim,
            output_dim=protocol.output_dim,
            permanent_hidden_dim=half,
            transient_hidden_dim=half,
            # Prespecified by inheriting the existing FastSlow diagnostic's
            # encoder, slow-head, fast-head, and decay values. No root result
            # was consulted to choose them.
            permanent_encoder_step_size=1e-3,
            permanent_head_step_size=1e-2,
            transient_encoder_step_size=1e-3,
            transient_head_step_size=5e-2,
            transient_decay=0.98,
            grad_clip=10.0,
            init_scale=1.0,
        )
    if arm_name == ARM_NAMES[1]:
        ordinary = _arm_config(protocol, ARM_NAMES[0])
        return AlbertaPermanentTransientConfig(
            input_dim=ordinary.input_dim,
            output_dim=ordinary.output_dim,
            permanent_hidden_dim=ordinary.permanent_hidden_dim,
            transient_hidden_dim=ordinary.transient_hidden_dim,
            permanent_encoder_step_size=0.0,
            permanent_head_step_size=0.0,
            transient_encoder_step_size=ordinary.transient_encoder_step_size,
            transient_head_step_size=ordinary.transient_head_step_size,
            transient_decay=ordinary.transient_decay,
            grad_clip=ordinary.grad_clip,
            init_scale=ordinary.init_scale,
        )
    raise ValueError("unsupported permanent/transient recurrence arm")


def _array_record(name: str, value: jax.Array) -> dict[str, object]:
    host = np.asarray(jax.device_get(value))
    canonical_dtype = host.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(host.astype(canonical_dtype, copy=False))
    return {
        "name": name,
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "data_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
    }


def _state_sha256(state: AlbertaPermanentTransientState) -> str:
    records = [
        _array_record(name, getattr(state.params, name)) for name in _PARAMETER_FIELDS
    ]
    records.extend(
        (
            _array_record("step_count", state.step_count),
            _array_record("step_words", state.step_words),
        )
    )
    return _digest(records)


def _group_sha256(params: AlbertaPermanentTransientParams, names: Sequence[str]) -> str:
    return _digest([_array_record(name, getattr(params, name)) for name in names])


def _group_norm(params: AlbertaPermanentTransientParams, names: Sequence[str]) -> float:
    squared = 0.0
    for name in names:
        host = np.asarray(jax.device_get(getattr(params, name)), dtype=np.float64)
        squared += float(np.sum(np.square(host)))
    return math.sqrt(squared)


def _probe_payload(
    state: AlbertaPermanentTransientState,
    observations: jax.Array,
) -> dict[str, object]:
    parts = jax.device_get(
        jax.vmap(lambda observation: permanent_transient_forward(state.params, observation))(
            observations
        )
    )
    targets = np.asarray(jax.device_get(observations[:, 0]), dtype=np.float64)
    combined = np.asarray(parts.prediction, dtype=np.float64).reshape(-1)
    permanent = np.asarray(parts.permanent_prediction, dtype=np.float64).reshape(-1)
    transient = np.asarray(parts.transient_prediction, dtype=np.float64).reshape(-1)
    reconstruction = combined - permanent - transient
    return {
        "examples": int(targets.size),
        "combined_a_mse": float(np.mean(np.square(combined - targets))),
        "permanent_a_mse": float(np.mean(np.square(permanent - targets))),
        "transient_a_mse": float(np.mean(np.square(transient - targets))),
        "combined_prediction_mean": float(np.mean(combined)),
        "permanent_prediction_mean": float(np.mean(permanent)),
        "transient_prediction_mean": float(np.mean(transient)),
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstruction))),
    }


def _checkpoint_payload(
    label: str,
    learner: AlbertaPermanentTransientLearner,
    state: AlbertaPermanentTransientState,
    probe_observations: jax.Array,
) -> dict[str, object]:
    resources = learner.resource_record(state)
    return {
        "label": label,
        "step_count": int(np.asarray(jax.device_get(state.step_count))),
        "step_words": [int(value) for value in np.asarray(jax.device_get(state.step_words))],
        "state_sha256": _state_sha256(state),
        "permanent_subtree_sha256": _group_sha256(state.params, _PERMANENT_FIELDS),
        "transient_subtree_sha256": _group_sha256(state.params, _TRANSIENT_FIELDS),
        "permanent_parameter_norm": _group_norm(state.params, _PERMANENT_FIELDS),
        "transient_parameter_norm": _group_norm(state.params, _TRANSIENT_FIELDS),
        "resources": resources.to_dict(),
        "a_probe": _probe_payload(state, probe_observations),
    }


def _trace_event(
    event_index: int,
    phase_index: int,
    phase_step: int,
    observation: jax.Array,
    target: jax.Array,
    parts: AlbertaPermanentTransientPrediction,
) -> dict[str, object]:
    observation_value = float(np.asarray(jax.device_get(observation))[0])
    target_value = float(np.asarray(jax.device_get(target))[0])
    prediction = float(np.asarray(jax.device_get(parts.prediction))[0])
    permanent = float(np.asarray(jax.device_get(parts.permanent_prediction))[0])
    transient = float(np.asarray(jax.device_get(parts.transient_prediction))[0])
    return {
        "event_index": event_index,
        "phase": PHASE_NAMES[phase_index],
        "phase_step": phase_step,
        "observation": observation_value,
        "target": target_value,
        "prediction": prediction,
        "squared_error": (prediction - target_value) ** 2,
        "permanent_squared_error": (permanent - target_value) ** 2,
        "permanent_prediction": permanent,
        "transient_prediction": transient,
        "reconstruction_error": prediction - permanent - transient,
    }


def _run_phase_eager(
    learner: AlbertaPermanentTransientLearner,
    state: AlbertaPermanentTransientState,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> tuple[AlbertaPermanentTransientState, list[dict[str, object]]]:
    trace: list[dict[str, object]] = []
    current = state
    for phase_step in range(observations.shape[0]):
        observation = observations[phase_step]
        target = targets[phase_step]
        parts = learner.predict_parts(current, observation)
        result = learner.update(current, observation, target)
        if not bool(result.update_applied):
            raise RuntimeError("prespecified permanent/transient update rejected")
        if not np.array_equal(
            np.asarray(jax.device_get(parts.prediction)),
            np.asarray(jax.device_get(result.prediction)),
        ):
            raise RuntimeError("pre-update permanent/transient prediction did not reconstruct")
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
    learner: AlbertaPermanentTransientLearner,
    state: AlbertaPermanentTransientState,
    observations: jax.Array,
    targets: jax.Array,
) -> tuple[AlbertaPermanentTransientState, _TraceArrays]:
    def step(
        current: AlbertaPermanentTransientState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[AlbertaPermanentTransientState, _TraceArrays]:
        observation, target = inputs
        parts = learner.predict_parts(current, observation)
        result = learner.update(current, observation, target)
        prediction = parts.prediction[0]
        permanent = parts.permanent_prediction[0]
        transient = parts.transient_prediction[0]
        return result.state, _TraceArrays(
            prediction=prediction,
            squared_error=(prediction - target[0]) ** 2,
            permanent_squared_error=(permanent - target[0]) ** 2,
            permanent_prediction=permanent,
            transient_prediction=transient,
            reconstruction_error=prediction - permanent - transient,
        )

    return jax.lax.scan(step, state, (observations, targets))


def _compiled_trace_events(
    arrays: _TraceArrays,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> list[dict[str, object]]:
    host = jax.device_get(arrays)
    host_observations = np.asarray(jax.device_get(observations)).reshape(-1)
    host_targets = np.asarray(jax.device_get(targets)).reshape(-1)
    events: list[dict[str, object]] = []
    for phase_step in range(observations.shape[0]):
        events.append(
            {
                "event_index": event_offset + phase_step,
                "phase": PHASE_NAMES[phase_index],
                "phase_step": phase_step,
                "observation": float(host_observations[phase_step]),
                "target": float(host_targets[phase_step]),
                "prediction": float(np.asarray(host.prediction)[phase_step]),
                "squared_error": float(np.asarray(host.squared_error)[phase_step]),
                "permanent_squared_error": float(
                    np.asarray(host.permanent_squared_error)[phase_step]
                ),
                "permanent_prediction": float(
                    np.asarray(host.permanent_prediction)[phase_step]
                ),
                "transient_prediction": float(
                    np.asarray(host.transient_prediction)[phase_step]
                ),
                "reconstruction_error": float(
                    np.asarray(host.reconstruction_error)[phase_step]
                ),
            }
        )
    return events


def _summary(values: Sequence[float], window: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "prequential_mse": float(np.mean(array)),
        "early_prequential_mse": float(np.mean(array[:window])),
        "tail_prequential_mse": float(np.mean(array[-window:])),
    }


def _metrics_from_trace(
    trace: Sequence[Mapping[str, object]],
    protocol: PermanentTransientRecurrenceProtocol,
) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        stop = start + protocol.phase_steps
        phase = trace[start:stop]
        phases[phase_name] = {
            "combined_readout": _summary(
                [cast(float, event["squared_error"]) for event in phase],
                protocol.summary_window,
            ),
            "permanent_only_readout": _summary(
                [cast(float, event["permanent_squared_error"]) for event in phase],
                protocol.summary_window,
            ),
        }
    return {"phase": phases}


def _execute_arm(
    protocol: PermanentTransientRecurrenceProtocol,
    observations: jax.Array,
    targets: jax.Array,
    initialization_key: jax.Array,
    *,
    arm_name: str,
    engine: str,
) -> _ExecutedArm:
    config = _arm_config(protocol, arm_name)
    learner = AlbertaPermanentTransientLearner(config)
    initial_state = learner.init(initialization_key)
    state = initial_state
    probe_observations = observations[: protocol.phase_steps]
    checkpoints: dict[str, object] = {
        "initial": _checkpoint_payload(
            "initial",
            learner,
            initial_state,
            probe_observations,
        )
    }
    checkpoint_states: list[AlbertaPermanentTransientState] = [initial_state]
    trace: list[dict[str, object]] = []
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        stop = start + protocol.phase_steps
        if phase_name == "A2":
            checkpoints["A2_entry"] = _checkpoint_payload(
                "A2_entry",
                learner,
                state,
                probe_observations,
            )
            checkpoint_states.append(state)
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
            raise ValueError("unsupported permanent/transient execution engine")
        trace.extend(phase_trace)
        label = "A2_tail" if phase_name == "A2" else f"{phase_name}_end"
        checkpoints[label] = _checkpoint_payload(
            label,
            learner,
            state,
            probe_observations,
        )
        checkpoint_states.append(state)

    b_end = cast(Mapping[str, object], checkpoints["B_end"])
    a2_entry = cast(Mapping[str, object], checkpoints["A2_entry"])
    if b_end["state_sha256"] != a2_entry["state_sha256"]:
        raise RuntimeError("A2 entry changed state before an A2 outcome")
    initial_resources = learner.resource_record(initial_state).to_dict()
    final_resources = learner.resource_record(state).to_dict()
    if initial_resources != final_resources:
        raise RuntimeError("permanent/transient state capacity changed during the life")
    report = {
        "arm": arm_name,
        "engine": engine,
        "learner_config": learner.to_config(),
        "learner_config_sha256": _digest(learner.to_config()),
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": _metrics_from_trace(trace, protocol),
        "checkpoints": checkpoints,
        "resources": {
            "initial_state": initial_resources,
            "final_state": final_resources,
            "fixed_allocation": True,
            "logical_peak_state_nbytes": initial_resources["state_nbytes"],
        },
        "work": {
            "logical_updates": protocol.total_steps,
            "maximum_gradient_evaluations_per_update": 2,
            "logical_gradient_evaluations": 2 * protocol.total_steps,
            "maximum_forward_evaluations_inside_update": 3,
            "logical_forward_evaluations_inside_updates": 3 * protocol.total_steps,
            "logical_explicit_pre_update_predictions": protocol.total_steps,
            "logical_probe_prediction_examples": len(checkpoints) * protocol.phase_steps,
            "replay_samples": 0,
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
        raise ValueError("parity state trees differ")
    maximum = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_host = np.asarray(jax.device_get(left_leaf))
        right_host = np.asarray(jax.device_get(right_leaf))
        if left_host.shape != right_host.shape or left_host.dtype != right_host.dtype:
            raise ValueError("parity state leaf contracts differ")
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
        eager_trace = cast(list[Mapping[str, object]], eager_arm.report["trace"])
        compiled_trace = cast(list[Mapping[str, object]], compiled_arm.report["trace"])
        float_max = {
            name: max(
                abs(cast(float, eager_event[name]) - cast(float, compiled_event[name]))
                for eager_event, compiled_event in zip(
                    eager_trace,
                    compiled_trace,
                    strict=True,
                )
            )
            for name in _TRACE_FLOAT_FIELDS
        }
        discrete_exact = all(
            eager_event["event_index"] == compiled_event["event_index"]
            and eager_event["phase"] == compiled_event["phase"]
            and eager_event["phase_step"] == compiled_event["phase_step"]
            for eager_event, compiled_event in zip(
                eager_trace,
                compiled_trace,
                strict=True,
            )
        )
        state_max = max(
            _tree_max_abs_difference(eager_state, compiled_state)
            for eager_state, compiled_state in zip(
                eager_arm.checkpoints,
                compiled_arm.checkpoints,
                strict=True,
            )
        )
        observed = max(*float_max.values(), state_max)
        arm_name = cast(str, eager_arm.report["arm"])
        arms[arm_name] = {
            "trace_discrete_fields_exact": discrete_exact,
            "trace_float_max_abs_difference": float_max,
            "checkpoint_state_max_abs_difference": state_max,
            "observed_max_abs_difference": observed,
            "declared_numeric_tolerance": PARITY_FLOAT_MAX_ABS_TOLERANCE,
            "within_declared_numeric_tolerance": (
                discrete_exact and observed <= PARITY_FLOAT_MAX_ABS_TOLERANCE
            ),
            "resources_exact": eager_arm.report["resources"]
            == compiled_arm.report["resources"],
            "work_exact": eager_arm.report["work"] == compiled_arm.report["work"],
            "technical_tolerance_is_not_an_outcome_threshold": True,
        }
    return {"arms": arms}


def _probe(
    report: Mapping[str, object],
    checkpoint: str,
    field: str,
) -> float:
    checkpoints = cast(Mapping[str, object], report["checkpoints"])
    payload = cast(Mapping[str, object], checkpoints[checkpoint])
    probe = cast(Mapping[str, object], payload["a_probe"])
    return cast(float, probe[field])


def _consumed_findings(report: Mapping[str, object]) -> dict[str, object]:
    permanent_a1 = _probe(report, "A1_end", "permanent_a_mse")
    permanent_b = _probe(report, "B_end", "permanent_a_mse")
    permanent_a2_entry = _probe(report, "A2_entry", "permanent_a_mse")
    permanent_a2_tail = _probe(report, "A2_tail", "permanent_a_mse")
    combined_a1 = _probe(report, "A1_end", "combined_a_mse")
    combined_b = _probe(report, "B_end", "combined_a_mse")
    combined_a2_entry = _probe(report, "A2_entry", "combined_a_mse")
    combined_a2_tail = _probe(report, "A2_tail", "combined_a_mse")
    transient_a1 = _probe(report, "A1_end", "transient_a_mse")
    transient_b = _probe(report, "B_end", "transient_a_mse")
    transient_a2_tail = _probe(report, "A2_tail", "transient_a_mse")
    return {
        "source_engine": "jax_jit_scan",
        "post_phase_a_probes": {
            "A1_end": {
                "combined_a_mse": combined_a1,
                "permanent_only_a_mse": permanent_a1,
            },
            "B_end": {
                "combined_a_mse": combined_b,
                "permanent_only_a_mse": permanent_b,
            },
            "A2_tail": {
                "combined_a_mse": combined_a2_tail,
                "permanent_only_a_mse": permanent_a2_tail,
            },
            "read_only_frozen_a1_observations": True,
        },
        "direct_pre_post_b_a_probe": {
            "a1_end_combined_a_mse": combined_a1,
            "b_end_combined_a_mse": combined_b,
            "b_end_minus_a1_end_combined_a_mse": combined_b - combined_a1,
            "a1_end_permanent_a_mse": permanent_a1,
            "b_end_permanent_a_mse": permanent_b,
            "b_end_minus_a1_end_permanent_a_mse": permanent_b - permanent_a1,
            "b_end_over_a1_end_permanent_a_mse": (
                permanent_b / permanent_a1 if permanent_a1 > 0.0 else None
            ),
            "a2_entry_combined_a_mse": combined_a2_entry,
            "a2_entry_permanent_a_mse": permanent_a2_entry,
            "a2_entry_equals_b_end": (
                combined_a2_entry == combined_b and permanent_a2_entry == permanent_b
            ),
            "contains_any_a2_update": False,
            "retention_threshold_or_verdict_applied": False,
        },
        "permanent_path_readout_ablation": {
            "intervention": "remove the additive permanent prediction at readout only",
            "counterfactual_retraining_claimed": False,
            "A1_end": {
                "combined_a_mse": combined_a1,
                "transient_only_a_mse": transient_a1,
                "transient_only_minus_combined_a_mse": transient_a1 - combined_a1,
            },
            "B_end": {
                "combined_a_mse": combined_b,
                "transient_only_a_mse": transient_b,
                "transient_only_minus_combined_a_mse": transient_b - combined_b,
            },
            "A2_tail": {
                "combined_a_mse": combined_a2_tail,
                "transient_only_a_mse": transient_a2_tail,
                "transient_only_minus_combined_a_mse": (
                    transient_a2_tail - combined_a2_tail
                ),
            },
            "success_threshold_or_verdict_applied": False,
        },
        "a2_reacquisition": {
            "a2_tail_combined_a_mse": combined_a2_tail,
            "a2_entry_minus_a2_tail_combined_a_mse": combined_a2_entry
            - combined_a2_tail,
            "a2_tail_permanent_a_mse": permanent_a2_tail,
            "a2_entry_minus_a2_tail_permanent_a_mse": permanent_a2_entry
            - permanent_a2_tail,
            "counted_as_pre_b_retention": False,
        },
        "winner_or_default_selected": False,
        "paper_reproduction_claimed": False,
    }


def _runtime_identity() -> dict[str, object]:
    payload = {
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
    payload["runtime_identity_sha256"] = _digest(payload)
    return payload


def _source_identity() -> dict[str, object]:
    evaluator_path = Path(__file__)
    core_path = evaluator_path.parents[1] / "core" / "permanent_transient.py"
    source_path = evaluator_path.parent / "fast_slow_recurrence_development.py"
    payload: dict[str, object] = {
        "evaluator_module_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "permanent_transient_core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "consumed_root_source_module_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    payload["source_identity_sha256"] = _digest(payload)
    return payload


def _build_report() -> dict[str, object]:
    protocol = PermanentTransientRecurrenceProtocol()
    observations, targets, source_manifest, initialization_key = _source_arrays_bound(protocol)
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
    ordinary_config = _arm_config(protocol, ARM_NAMES[0]).to_config()
    ablation_config = _arm_config(protocol, ARM_NAMES[1]).to_config()
    differing_fields = {
        name: {ARM_NAMES[0]: ordinary_config[name], ARM_NAMES[1]: ablation_config[name]}
        for name in ordinary_config
        if ordinary_config[name] != ablation_config[name]
    }
    ordinary_run = executions["jax_jit_scan"][0].report
    ablation_run = executions["jax_jit_scan"][1].report
    ordinary_initial = cast(Mapping[str, object], ordinary_run["checkpoints"])["initial"]
    ablation_initial = cast(Mapping[str, object], ablation_run["checkpoints"])["initial"]
    fast_slow = FastSlowLearner(_ordinary_config(FastSlowRecurrenceProtocol()))
    fast_slow_resources = fast_slow.resource_record().to_dict()
    body: dict[str, object] = {
        "schema_version": PERMANENT_TRANSIENT_RECURRENCE_REPORT_SCHEMA,
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
        "design_record": permanent_transient_design_record().to_dict(),
        "design_record_sha256": _digest(permanent_transient_design_record().to_dict()),
        "arm_order": list(ARM_NAMES),
        "arm_comparison": {
            "ordinary_and_ablation_initial_state_sha256_equal": cast(
                Mapping[str, object], ordinary_initial
            )["state_sha256"]
            == cast(Mapping[str, object], ablation_initial)["state_sha256"],
            "ordinary_and_ablation_resources_equal": ordinary_run["resources"]
            == ablation_run["resources"],
            "ordinary_and_ablation_work_equal": ordinary_run["work"]
            == ablation_run["work"],
            "only_config_differences": differing_fields,
            "expected_difference_fields": [
                "permanent_encoder_step_size",
                "permanent_head_step_size",
            ],
            "causal_intervention": "disable permanent consolidation only",
            "winner_selected": False,
        },
        "fast_slow_sibling_comparison_boundary": {
            "same_consumed_source_manifest": True,
            "same_input_and_output_dims": True,
            "same_total_hidden_feature_count": True,
            "permanent_transient_state_nbytes": cast(
                Mapping[str, object],
                cast(Mapping[str, object], ordinary_run["resources"])["initial_state"],
            )["state_nbytes"],
            "fast_slow_state_nbytes": fast_slow_resources["state_nbytes"],
            "permanent_transient_gradient_evaluations_per_update": 2,
            "fast_slow_gradient_evaluations_per_update": 1,
            "shape_or_gradient_work_matched": False,
            "cross_family_performance_winner_allowed": False,
        },
        "consumed_findings": _consumed_findings(ordinary_run),
        "no_consolidation_findings": _consumed_findings(ablation_run),
        "executions": {
            engine: [executed.report for executed in executions[engine]]
            for engine in EXECUTION_ENGINES
        },
        "eager_compiled_parity": _parity_payload(
            executions["python_eager"],
            executions["jax_jit_scan"],
        ),
        "limitations": list(LIMITATIONS),
    }
    body["causal_reconstruction_sha256"] = _digest(
        {
            "protocol_sha256": body["protocol_sha256"],
            "source_manifest_sha256": body["source_manifest_sha256"],
            "source_identity": body["source_identity"],
            "runtime_identity": body["runtime_identity"],
            "design_record_sha256": body["design_record_sha256"],
            "executions": body["executions"],
        }
    )
    report = {**body, "report_sha256": _digest(body)}
    return cast(dict[str, object], _json_clone(report))


@functools.lru_cache(maxsize=1)
def _expected_report_json() -> str:
    return _canonical_json(_build_report())


def run_permanent_transient_recurrence_development() -> dict[str, object]:
    """Return the deterministic in-memory consumed-root diagnostic."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_permanent_transient_recurrence_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated permanent/transient report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def validate_permanent_transient_recurrence_report(
    report: Mapping[str, object],
) -> PermanentTransientRecurrenceValidation:
    """Fail closed against full deterministic causal reconstruction."""

    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return PermanentTransientRecurrenceValidation(
            False,
            (f"report is not canonical JSON: {error}",),
        )
    expected = cast(dict[str, object], json.loads(_expected_report_json()))
    errors: list[str] = []
    if not _exact_json_equal(candidate, expected):
        errors.append("report does not match the frozen causal reconstruction")
    body = {name: value for name, value in candidate.items() if name != "report_sha256"}
    if candidate.get("report_sha256") != _digest(body):
        errors.append("report_sha256 does not reconstruct")
    return PermanentTransientRecurrenceValidation(not errors, tuple(errors))


def permanent_transient_recurrence_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize a valid report without writing it."""

    validation = validate_permanent_transient_recurrence_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid permanent/transient report: " + "; ".join(validation.errors)
        )
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_ONLY",
    "LIMITATIONS",
    "OUTPUT_WRITES_ALLOWED",
    "PARITY_FLOAT_MAX_ABS_TOLERANCE",
    "PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA",
    "PERMANENT_TRANSIENT_RECURRENCE_REPORT_SCHEMA",
    "PermanentTransientRecurrenceProtocol",
    "PermanentTransientRecurrenceValidation",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "permanent_transient_recurrence_report_json",
    "run_permanent_transient_recurrence_development",
    "validate_permanent_transient_recurrence_report",
]
