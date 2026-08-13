# mypy: disable-error-code="call-arg,operator"
"""Consumed-root dormancy/reactivation diagnostic for latent-context experts.

The evaluator reuses the exact already-consumed FastSlow A/B/A source.  The
learner receives no phase label, boundary, clock, mapping, replay item, reset,
or future datum.  Each event constructs a cache from the observation before
the target is supplied.  The target can select only the next owner and the
single committed expert candidate; it cannot relabel the current prequential
prediction or error.

The ordinary arm enables evidence-based selective gating.  Its same-state and
same-candidate-work ablation computes the same expert predictions, losses, and
candidate gradients but retains the cached owner.  Expert identity is learned:
the A expert is whichever slot owns prediction at A1 end, never a hard-coded
slot.  This module has no writer, threshold, winner, default selection,
evidence, or promotion path.
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
from typing import Final, cast

import jax
import jaxlib
import numpy as np

from alberta_framework.core.latent_context_experts import (
    LatentContextExpertConfig,
    LatentContextExpertLearner,
    LatentContextExpertLearningResult,
    LatentContextExpertState,
    latent_context_expert_design_record,
    latent_context_expert_forward,
    run_latent_context_expert_arrays,
)
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    DEVELOPMENT_ROOT_SEED,
    INPUT_DIM,
    OUTPUT_DIM,
    PHASE_NAMES,
    PHASE_STEPS,
    SUMMARY_WINDOW,
    FastSlowRecurrenceProtocol,
    _source_arrays,
)

LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.latent-context-expert-recurrence-development.protocol.v1"
)
LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.latent-context-expert-recurrence-development.report.v1"
)
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ASSESSMENT_STATUS: Final = "not_assessed"

ARM_NAMES: Final = (
    "latent_context_selective_gating",
    "latent_context_no_selective_gating",
)
EXECUTION_ENGINES: Final = ("python_eager", "jax_jit_scan")
PARITY_FLOAT_MAX_ABS_TOLERANCE: Final = 2.0e-6

LIMITATIONS: Final = (
    "this is a generic leakage-safe integration of the existing ContextInference "
    "active-only-freeze law, not a conceptually novel context-inference algorithm",
    "one consumed scalar Gaussian root is not a population, robustness, control, or scale result",
    "the selector uses one current outcome at a time and may fragment expert ownership; all "
    "fragmentation and dormant-slot writes are reported rather than hidden",
    "the target mapping is unidentifiable from observations, so the first prediction after a "
    "mapping change necessarily precedes the first evidence of that change",
    "expert selection and training on a switched-regime outcome occur only after that target "
    "is observed; no pre-outcome context-identification claim is made",
    "A probes reuse frozen A1 observations and are read-only diagnostics",
    "A2 reactivation latency counts observed A2 outcomes and is not a success threshold",
    "the finite uint32[2] lifetime does not establish indefinite continual operation",
    "there is no sweep, tuning, outcome threshold, verdict, winner, default, writer, held-out "
    "seed, evidence, or scientific-promotion path",
)

_TRACE_SCALAR_FLOAT_FIELDS: Final = (
    "observation",
    "target",
    "prediction",
    "squared_error",
    "owner_reconstruction_error",
)
_TRACE_VECTOR_FLOAT_FIELDS: Final = (
    "expert_predictions",
    "expert_losses",
    "candidate_gradient_norms",
)


@dataclasses.dataclass(frozen=True, slots=True)
class LatentContextExpertRecurrenceProtocol:
    """Frozen consumed source and prespecified learner construction."""

    schema_version: str = LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA
    development_root_seed: int = DEVELOPMENT_ROOT_SEED
    phase_steps: int = PHASE_STEPS
    summary_window: int = SUMMARY_WINDOW
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    max_experts: int = 2
    step_size: float = 0.05
    grad_clip: float = 10.0

    def __post_init__(self) -> None:
        expected = (
            LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA,
            DEVELOPMENT_ROOT_SEED,
            PHASE_STEPS,
            SUMMARY_WINDOW,
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            0.05,
            10.0,
        )
        actual = (
            self.schema_version,
            self.development_root_seed,
            self.phase_steps,
            self.summary_window,
            self.input_dim,
            self.output_dim,
            self.max_experts,
            self.step_size,
            self.grad_clip,
        )
        types_changed = any(
            type(value) is not type(reference)
            for value, reference in zip(actual, expected, strict=True)
        )
        if actual != expected or types_changed:
            raise ValueError("the consumed latent-context expert protocol is frozen")

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
            "development_root_already_consumed": True,
            "new_seed_or_initialization_drawn": False,
            "seed_or_hyperparameter_search_performed": False,
            "phase_names": list(PHASE_NAMES),
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "summary_window": self.summary_window,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "max_experts": self.max_experts,
            "step_size": self.step_size,
            "grad_clip": self.grad_clip,
            "initialization": "all expert weights and biases exactly zero",
            "target_mapping": {"A1": "x", "B": "-x", "A2": "x"},
            "learner_inputs": ["observation", "then cached prediction plus target"],
            "learner_metadata_exposed": [],
            "first_switched_regime_prediction_precedes_outcome": True,
            "a_probe_inputs": "the frozen A1 observations, read-only",
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> LatentContextExpertRecurrenceProtocol:
        protocol = cls()
        if not _exact_json_equal(dict(payload), protocol.to_config()):
            raise ValueError("protocol payload does not match the frozen consumed protocol")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class LatentContextExpertRecurrenceValidation:
    """Strict in-memory reconstruction result."""

    valid: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutedArm:
    report: dict[str, object]
    checkpoints: tuple[LatentContextExpertState, ...]


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
    protocol: LatentContextExpertRecurrenceProtocol,
) -> tuple[jax.Array, jax.Array, dict[str, object], jax.Array]:
    return _source_arrays(
        FastSlowRecurrenceProtocol(
            development_root_seed=protocol.development_root_seed,
            phase_steps=protocol.phase_steps,
            summary_window=protocol.summary_window,
            input_dim=protocol.input_dim,
            output_dim=protocol.output_dim,
        )
    )


def _arm_config(
    protocol: LatentContextExpertRecurrenceProtocol,
    arm_name: str,
) -> LatentContextExpertConfig:
    if arm_name not in ARM_NAMES:
        raise ValueError("unsupported latent-context recurrence arm")
    return LatentContextExpertConfig(
        input_dim=protocol.input_dim,
        output_dim=protocol.output_dim,
        max_experts=protocol.max_experts,
        step_size=protocol.step_size,
        grad_clip=protocol.grad_clip,
        selective_gating=arm_name == ARM_NAMES[0],
    )


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


def _state_sha256(state: LatentContextExpertState) -> str:
    return _digest(
        [
            _array_record("expert_weights", state.params.expert_weights),
            _array_record("expert_biases", state.params.expert_biases),
            _array_record("active_expert", state.active_expert),
            _array_record("step_count", state.step_count),
            _array_record("step_words", state.step_words),
        ]
    )


def _expert_sha256(state: LatentContextExpertState, expert: int) -> str:
    return _digest(
        [
            _array_record("expert_weights", state.params.expert_weights[expert]),
            _array_record("expert_biases", state.params.expert_biases[expert]),
        ]
    )


def _expert_norm(state: LatentContextExpertState, expert: int) -> float:
    weights = np.asarray(jax.device_get(state.params.expert_weights[expert]), dtype=np.float64)
    biases = np.asarray(jax.device_get(state.params.expert_biases[expert]), dtype=np.float64)
    return math.sqrt(float(np.sum(np.square(weights)) + np.sum(np.square(biases))))


def _a_probe(
    state: LatentContextExpertState,
    observations: jax.Array,
) -> dict[str, object]:
    predictions = np.asarray(
        jax.device_get(
            jax.vmap(
                lambda observation: latent_context_expert_forward(
                    state.params,
                    observation,
                )
            )(observations)
        ),
        dtype=np.float64,
    )
    targets = np.asarray(jax.device_get(observations), dtype=np.float64)
    losses = np.mean(np.square(predictions - targets[:, None, :]), axis=(0, 2))
    active = int(np.asarray(jax.device_get(state.active_expert)))
    return {
        "examples": int(observations.shape[0]),
        "expert_a_mse": [float(value) for value in losses],
        "active_expert": active,
        "active_expert_a_mse": float(losses[active]),
    }


def _checkpoint(
    label: str,
    learner: LatentContextExpertLearner,
    state: LatentContextExpertState,
    probe_observations: jax.Array,
) -> dict[str, object]:
    k = learner.config.max_experts
    resources = learner.resource_record(state)
    return {
        "label": label,
        "step_count": int(np.asarray(jax.device_get(state.step_count))),
        "step_words": [int(value) for value in np.asarray(jax.device_get(state.step_words))],
        "active_expert": int(np.asarray(jax.device_get(state.active_expert))),
        "state_sha256": _state_sha256(state),
        "expert_subtree_sha256": [_expert_sha256(state, index) for index in range(k)],
        "expert_parameter_norm": [_expert_norm(state, index) for index in range(k)],
        "resources": resources.to_dict(),
        "a_probe": _a_probe(state, probe_observations),
    }


def _event(
    *,
    event_index: int,
    phase_index: int,
    phase_step: int,
    observation: float,
    target: float,
    prediction: float,
    expert_predictions: Sequence[float],
    expert_losses: Sequence[float],
    candidate_gradient_norms: Sequence[float],
    pre_update_owner: int,
    evidence_best_expert: int,
    selected_next_expert: int,
    expert_update_mask: Sequence[bool],
    context_switched: bool,
) -> dict[str, object]:
    return {
        "event_index": event_index,
        "phase": PHASE_NAMES[phase_index],
        "phase_step": phase_step,
        "observation": observation,
        "target": target,
        "prediction": prediction,
        "squared_error": (target - prediction) ** 2,
        "expert_predictions": list(expert_predictions),
        "expert_losses": list(expert_losses),
        "candidate_gradient_norms": list(candidate_gradient_norms),
        "pre_update_owner": pre_update_owner,
        "evidence_best_expert": evidence_best_expert,
        "selected_next_expert": selected_next_expert,
        "expert_update_mask": list(expert_update_mask),
        "context_switched": context_switched,
        "owner_reconstruction_error": prediction - expert_predictions[pre_update_owner],
        "current_error_relabelled_after_target": False,
    }


def _run_phase_eager(
    learner: LatentContextExpertLearner,
    state: LatentContextExpertState,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> tuple[LatentContextExpertState, list[dict[str, object]]]:
    current = state
    trace: list[dict[str, object]] = []
    for phase_step in range(observations.shape[0]):
        observation = observations[phase_step]
        target = targets[phase_step]
        cache = learner.predict(current, observation)
        result = learner.update(current, cache, target)
        if not bool(result.update_applied):
            raise RuntimeError("prespecified latent-context expert update rejected")
        if not np.array_equal(
            np.asarray(jax.device_get(cache.prediction)),
            np.asarray(jax.device_get(result.prediction)),
        ):
            raise RuntimeError("target relabelled the cached prequential prediction")
        host = jax.device_get(result)
        trace.append(
            _event(
                event_index=event_offset + phase_step,
                phase_index=phase_index,
                phase_step=phase_step,
                observation=float(np.asarray(jax.device_get(observation))[0]),
                target=float(np.asarray(jax.device_get(target))[0]),
                prediction=float(np.asarray(host.prediction)[0]),
                expert_predictions=[
                    float(value) for value in np.asarray(host.expert_predictions).reshape(-1)
                ],
                expert_losses=[float(value) for value in np.asarray(host.expert_losses)],
                candidate_gradient_norms=[
                    float(value) for value in np.asarray(host.candidate_gradient_norms)
                ],
                pre_update_owner=int(np.asarray(host.pre_update_owner)),
                evidence_best_expert=int(np.asarray(host.evidence_best_expert)),
                selected_next_expert=int(np.asarray(host.selected_next_expert)),
                expert_update_mask=[
                    bool(value) for value in np.asarray(host.expert_update_mask)
                ],
                context_switched=bool(np.asarray(host.context_switched)),
            )
        )
        current = result.state
    return current, trace


@functools.partial(jax.jit, static_argnums=(0,))
def _run_phase_compiled(
    learner: LatentContextExpertLearner,
    state: LatentContextExpertState,
    observations: jax.Array,
    targets: jax.Array,
) -> LatentContextExpertLearningResult:
    return run_latent_context_expert_arrays(
        learner,
        observations,
        targets,
        state=state,
    )


def _compiled_events(
    result: LatentContextExpertLearningResult,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> list[dict[str, object]]:
    host = jax.device_get(result)
    host_observations = np.asarray(jax.device_get(observations)).reshape(-1)
    host_targets = np.asarray(jax.device_get(targets)).reshape(-1)
    events: list[dict[str, object]] = []
    for phase_step in range(observations.shape[0]):
        events.append(
            _event(
                event_index=event_offset + phase_step,
                phase_index=phase_index,
                phase_step=phase_step,
                observation=float(host_observations[phase_step]),
                target=float(host_targets[phase_step]),
                prediction=float(np.asarray(host.predictions)[phase_step, 0]),
                expert_predictions=[
                    float(value)
                    for value in np.asarray(host.expert_predictions)[phase_step].reshape(-1)
                ],
                expert_losses=[
                    float(value) for value in np.asarray(host.expert_losses)[phase_step]
                ],
                candidate_gradient_norms=[
                    float(value)
                    for value in np.asarray(host.candidate_gradient_norms)[phase_step]
                ],
                pre_update_owner=int(np.asarray(host.pre_update_owner)[phase_step]),
                evidence_best_expert=int(
                    np.asarray(host.evidence_best_expert)[phase_step]
                ),
                selected_next_expert=int(
                    np.asarray(host.selected_next_expert)[phase_step]
                ),
                expert_update_mask=[
                    bool(value)
                    for value in np.asarray(host.expert_update_mask)[phase_step]
                ],
                context_switched=bool(np.asarray(host.context_switched)[phase_step]),
            )
        )
    return events


def _phase_metrics(
    trace: Sequence[Mapping[str, object]],
    protocol: LatentContextExpertRecurrenceProtocol,
) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase_index, name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        phase = trace[start : start + protocol.phase_steps]
        losses = np.asarray(
            [cast(float, event["squared_error"]) for event in phase],
            dtype=np.float64,
        )
        switches = sum(bool(event["context_switched"]) for event in phase)
        phases[name] = {
            "prequential_mse": float(np.mean(losses)),
            "early_prequential_mse": float(np.mean(losses[: protocol.summary_window])),
            "tail_prequential_mse": float(np.mean(losses[-protocol.summary_window :])),
            "context_switch_count": switches,
            "distinct_pre_update_owners": sorted(
                {cast(int, event["pre_update_owner"]) for event in phase}
            ),
            "distinct_selected_next_experts": sorted(
                {cast(int, event["selected_next_expert"]) for event in phase}
            ),
        }
    return {"phase": phases}


def _execute_arm(
    protocol: LatentContextExpertRecurrenceProtocol,
    observations: jax.Array,
    targets: jax.Array,
    *,
    arm_name: str,
    engine: str,
) -> _ExecutedArm:
    learner = LatentContextExpertLearner(_arm_config(protocol, arm_name))
    initial_state = learner.init()
    state = initial_state
    probe_observations = observations[: protocol.phase_steps]
    checkpoints: dict[str, object] = {
        "initial": _checkpoint("initial", learner, state, probe_observations)
    }
    checkpoint_states: list[LatentContextExpertState] = [state]
    trace: list[dict[str, object]] = []
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        stop = start + protocol.phase_steps
        if phase_name == "A2":
            checkpoints["A2_entry"] = _checkpoint(
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
            result = _run_phase_compiled(
                learner,
                state,
                observations[start:stop],
                targets[start:stop],
            )
            state = result.state
            phase_trace = _compiled_events(
                result,
                observations[start:stop],
                targets[start:stop],
                phase_index=phase_index,
                event_offset=start,
            )
        else:
            raise ValueError("unsupported latent-context execution engine")
        trace.extend(phase_trace)
        label = "A2_tail" if phase_name == "A2" else f"{phase_name}_end"
        checkpoints[label] = _checkpoint(label, learner, state, probe_observations)
        checkpoint_states.append(state)

    b_end = cast(Mapping[str, object], checkpoints["B_end"])
    a2_entry = cast(Mapping[str, object], checkpoints["A2_entry"])
    if b_end["state_sha256"] != a2_entry["state_sha256"]:
        raise RuntimeError("A2 entry changed state before its first prediction")
    initial_resources = learner.resource_record(initial_state).to_dict()
    final_resources = learner.resource_record(state).to_dict()
    if initial_resources != final_resources:
        raise RuntimeError("latent-context state capacity changed during the life")
    k = protocol.max_experts
    checkpoint_count = len(checkpoints)
    report = {
        "arm": arm_name,
        "engine": engine,
        "learner_config": learner.to_config(),
        "learner_config_sha256": _digest(learner.to_config()),
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": _phase_metrics(trace, protocol),
        "checkpoints": checkpoints,
        "resources": {
            "initial_state": initial_resources,
            "final_state": final_resources,
            "fixed_allocation": True,
            "logical_peak_state_nbytes": initial_resources["state_nbytes"],
            "logical_prediction_cache_nbytes": initial_resources[
                "prediction_cache_nbytes"
            ],
        },
        "work": {
            "logical_updates": protocol.total_steps,
            "expert_predictions_per_update": 2 * k,
            "logical_expert_predictions": 2 * k * protocol.total_steps,
            "expert_losses_per_update": k,
            "logical_expert_losses": k * protocol.total_steps,
            "candidate_gradients_per_update": k,
            "logical_candidate_gradients": k * protocol.total_steps,
            "expert_subtree_commits_per_update": 1,
            "logical_expert_subtree_commits": protocol.total_steps,
            "logical_probe_observation_examples": checkpoint_count * protocol.phase_steps,
            "logical_probe_expert_predictions": (
                checkpoint_count * protocol.phase_steps * k
            ),
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


def _parity(
    eager: Sequence[_ExecutedArm],
    compiled: Sequence[_ExecutedArm],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for eager_arm, compiled_arm in zip(eager, compiled, strict=True):
        eager_trace = cast(list[Mapping[str, object]], eager_arm.report["trace"])
        compiled_trace = cast(list[Mapping[str, object]], compiled_arm.report["trace"])
        scalar_differences = {
            name: max(
                abs(cast(float, left[name]) - cast(float, right[name]))
                for left, right in zip(eager_trace, compiled_trace, strict=True)
            )
            for name in _TRACE_SCALAR_FLOAT_FIELDS
        }
        vector_differences: dict[str, float] = {}
        for name in _TRACE_VECTOR_FLOAT_FIELDS:
            maximum = 0.0
            for left, right in zip(eager_trace, compiled_trace, strict=True):
                left_values = cast(list[float], left[name])
                right_values = cast(list[float], right[name])
                maximum = max(
                    maximum,
                    max(
                        abs(a - b)
                        for a, b in zip(left_values, right_values, strict=True)
                    ),
                )
            vector_differences[name] = maximum
        discrete_exact = all(
            left["event_index"] == right["event_index"]
            and left["phase"] == right["phase"]
            and left["phase_step"] == right["phase_step"]
            and left["pre_update_owner"] == right["pre_update_owner"]
            and left["evidence_best_expert"] == right["evidence_best_expert"]
            and left["selected_next_expert"] == right["selected_next_expert"]
            and left["expert_update_mask"] == right["expert_update_mask"]
            and left["context_switched"] == right["context_switched"]
            and left["current_error_relabelled_after_target"]
            == right["current_error_relabelled_after_target"]
            for left, right in zip(eager_trace, compiled_trace, strict=True)
        )
        state_max = max(
            _tree_max_abs_difference(left, right)
            for left, right in zip(
                eager_arm.checkpoints,
                compiled_arm.checkpoints,
                strict=True,
            )
        )
        observed = max(
            *scalar_differences.values(),
            *vector_differences.values(),
            state_max,
        )
        arm_name = cast(str, eager_arm.report["arm"])
        arms[arm_name] = {
            "trace_discrete_fields_exact": discrete_exact,
            "trace_scalar_float_max_abs_difference": scalar_differences,
            "trace_vector_float_max_abs_difference": vector_differences,
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


def _arm_findings(report: Mapping[str, object]) -> dict[str, object]:
    trace = cast(list[Mapping[str, object]], report["trace"])
    checkpoints = cast(Mapping[str, object], report["checkpoints"])
    a1 = cast(Mapping[str, object], checkpoints["A1_end"])
    b = cast(Mapping[str, object], checkpoints["B_end"])
    a2_entry = cast(Mapping[str, object], checkpoints["A2_entry"])
    a2_tail = cast(Mapping[str, object], checkpoints["A2_tail"])
    learned_a = cast(int, a1["active_expert"])
    a1_hashes = cast(list[str], a1["expert_subtree_sha256"])
    b_hashes = cast(list[str], b["expert_subtree_sha256"])
    a2_hashes = cast(list[str], a2_tail["expert_subtree_sha256"])
    a1_probe = cast(Mapping[str, object], a1["a_probe"])
    b_probe = cast(Mapping[str, object], b["a_probe"])
    a2_probe = cast(Mapping[str, object], a2_tail["a_probe"])
    a1_probe_values = cast(list[float], a1_probe["expert_a_mse"])
    b_probe_values = cast(list[float], b_probe["expert_a_mse"])
    a2_probe_values = cast(list[float], a2_probe["expert_a_mse"])
    a1_trace = trace[:PHASE_STEPS]
    b_trace = trace[PHASE_STEPS : 2 * PHASE_STEPS]
    a2_trace = trace[2 * PHASE_STEPS :]
    b_selected_updates = sum(
        cast(list[bool], event["expert_update_mask"])[learned_a] for event in b_trace
    )
    b_a_update_steps = [
        cast(int, event["phase_step"])
        for event in b_trace
        if cast(list[bool], event["expert_update_mask"])[learned_a]
    ]
    b_a_update_steps_summary: dict[str, object] = {
        "count": len(b_a_update_steps),
        "first": b_a_update_steps[0] if b_a_update_steps else None,
        "last": b_a_update_steps[-1] if b_a_update_steps else None,
        "phase_steps_sha256": _digest(b_a_update_steps),
        "phase_steps_if_at_most_16": (
            b_a_update_steps if len(b_a_update_steps) <= 16 else None
        ),
    }
    b_preowners = [cast(int, event["pre_update_owner"]) for event in b_trace]
    b_selected = [cast(int, event["selected_next_expert"]) for event in b_trace]
    first_b = b_trace[0]
    first_b_selected = cast(int, first_b["selected_next_expert"])
    first_b_mask = cast(list[bool], first_b["expert_update_mask"])
    b_end_owner = cast(int, b["active_expert"])
    a_owner_dormant_at_a2_entry = b_end_owner != learned_a
    if b_end_owner == learned_a:
        latency: int | None = 0
    else:
        selected_steps = [
            index
            for index, event in enumerate(a2_trace)
            if cast(int, event["selected_next_expert"]) == learned_a
        ]
        latency = selected_steps[0] + 1 if selected_steps else None
    first_prediction_step = (
        0
        if latency == 0
        else latency
        if latency is not None and latency < PHASE_STEPS
        else None
    )
    return {
        "learned_a_expert_identity": learned_a,
        "identity_was_not_hard_coded": True,
        "direct_dormant_a_probe": {
            "a1_end_a_expert_mse": a1_probe_values[learned_a],
            "b_end_a_expert_mse": b_probe_values[learned_a],
            "b_end_minus_a1_end_a_expert_mse": (
                b_probe_values[learned_a] - a1_probe_values[learned_a]
            ),
            "a1_end_subtree_sha256": a1_hashes[learned_a],
            "b_end_subtree_sha256": b_hashes[learned_a],
            "subtree_bit_exact_across_b": a1_hashes[learned_a] == b_hashes[learned_a],
            "selected_update_count_during_b": b_selected_updates,
            "b_end_owner": b_end_owner,
            "contains_any_a2_update": False,
        },
        "first_b_outcome_routing": {
            "pre_update_owner": cast(int, first_b["pre_update_owner"]),
            "prequential_prediction": cast(float, first_b["prediction"]),
            "evidence_best_expert_after_target": cast(
                int,
                first_b["evidence_best_expert"],
            ),
            "selected_next_expert_after_target": first_b_selected,
            "selected_expert_received_current_outcome_update": first_b_mask[
                first_b_selected
            ],
            "selected_different_expert_than_pre_update_owner": (
                first_b_selected != cast(int, first_b["pre_update_owner"])
            ),
            "selection_and_training_are_post_outcome": True,
            "preoutcome_context_identification_claimed": False,
        },
        "a2_reactivation": {
            "a2_entry_state_equals_b_end": a2_entry["state_sha256"] == b["state_sha256"],
            "first_a2_pre_update_owner": cast(int, a2_trace[0]["pre_update_owner"]),
            "first_a2_prediction": cast(float, a2_trace[0]["prediction"]),
            "first_a2_squared_error": cast(float, a2_trace[0]["squared_error"]),
            "first_a2_selected_next_expert": cast(
                int,
                a2_trace[0]["selected_next_expert"],
            ),
            "observed_a2_outcomes_until_a1_owner_selected": latency,
            "first_a2_prediction_phase_step_using_reactivated_owner": first_prediction_step,
            "a1_owner_reactivated_during_a2": (
                a_owner_dormant_at_a2_entry and latency is not None
            ),
            "a1_owner_was_dormant_at_a2_entry": a_owner_dormant_at_a2_entry,
            "reactivation_term_applies": (
                a_owner_dormant_at_a2_entry and latency is not None
            ),
            "latency_is_descriptive_not_thresholded": True,
            "first_a2_prediction_precedes_first_a2_outcome": True,
            "a2_tail_a_expert_mse": a2_probe_values[learned_a],
            "a2_tail_subtree_sha256": a2_hashes[learned_a],
            "counted_as_retention_through_b": False,
        },
        "fragmentation_audit": {
            "a1_context_switch_count": sum(
                bool(event["context_switched"]) for event in a1_trace
            ),
            "a1_distinct_pre_update_owners": sorted(
                {cast(int, event["pre_update_owner"]) for event in a1_trace}
            ),
            "b_context_switch_count": sum(
                bool(event["context_switched"]) for event in b_trace
            ),
            "b_distinct_pre_update_owners": sorted(set(b_preowners)),
            "b_distinct_selected_next_experts": sorted(set(b_selected)),
            "b_phase_steps_selecting_a1_owner_for_update": b_a_update_steps_summary,
            "per_transaction_nonselected_subtree_preservation_contract": True,
            "clean_a_expert_dormancy_across_b": (
                b_selected_updates == 0
                and a1_hashes[learned_a] == b_hashes[learned_a]
                and b_end_owner != learned_a
            ),
            "fragmentation_threshold_or_verdict_applied": False,
        },
        "performance_threshold_or_verdict_applied": False,
        "winner_or_default_selected": False,
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
    core_path = evaluator_path.parents[1] / "core" / "latent_context_experts.py"
    source_path = evaluator_path.parent / "fast_slow_recurrence_development.py"
    context_path = evaluator_path.parents[1] / "core" / "context_inference.py"
    payload: dict[str, object] = {
        "evaluator_module_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "latent_context_core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "credited_context_inference_sha256": hashlib.sha256(
            context_path.read_bytes()
        ).hexdigest(),
        "consumed_root_source_module_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    }
    payload["source_identity_sha256"] = _digest(payload)
    return payload


def _build_report() -> dict[str, object]:
    protocol = LatentContextExpertRecurrenceProtocol()
    observations, targets, source_manifest, _unused_initialization_key = _source_arrays_bound(
        protocol
    )
    executions: dict[str, list[_ExecutedArm]] = {}
    for engine in EXECUTION_ENGINES:
        executions[engine] = [
            _execute_arm(
                protocol,
                observations,
                targets,
                arm_name=arm_name,
                engine=engine,
            )
            for arm_name in ARM_NAMES
        ]
    ordinary_config = _arm_config(protocol, ARM_NAMES[0]).to_config()
    ablation_config = _arm_config(protocol, ARM_NAMES[1]).to_config()
    differences = {
        name: {ARM_NAMES[0]: ordinary_config[name], ARM_NAMES[1]: ablation_config[name]}
        for name in ordinary_config
        if ordinary_config[name] != ablation_config[name]
    }
    ordinary = executions["jax_jit_scan"][0].report
    ablation = executions["jax_jit_scan"][1].report
    ordinary_initial = cast(Mapping[str, object], ordinary["checkpoints"])["initial"]
    ablation_initial = cast(Mapping[str, object], ablation["checkpoints"])["initial"]
    design = latent_context_expert_design_record().to_dict()
    body: dict[str, object] = {
        "schema_version": LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA,
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
        "design_record": design,
        "design_record_sha256": _digest(design),
        "arm_order": list(ARM_NAMES),
        "arm_comparison": {
            "initial_state_equal": cast(Mapping[str, object], ordinary_initial)[
                "state_sha256"
            ]
            == cast(Mapping[str, object], ablation_initial)["state_sha256"],
            "resources_equal": ordinary["resources"] == ablation["resources"],
            "work_equal": ordinary["work"] == ablation["work"],
            "only_config_differences": differences,
            "expected_difference_fields": ["selective_gating"],
            "causal_intervention": "disable evidence-based next-owner selection only",
            "winner_selected": False,
        },
        "consumed_findings": {
            "selective_gating": _arm_findings(ordinary),
            "no_selective_gating": _arm_findings(ablation),
        },
        "executions": {
            engine: [execution.report for execution in executions[engine]]
            for engine in EXECUTION_ENGINES
        },
        "eager_compiled_parity": _parity(
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
    return cast(dict[str, object], _json_clone({**body, "report_sha256": _digest(body)}))


@functools.lru_cache(maxsize=1)
def _expected_report_json() -> str:
    return _canonical_json(_build_report())


def run_latent_context_expert_recurrence_development() -> dict[str, object]:
    """Return the deterministic in-memory consumed-root diagnostic."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_latent_context_expert_recurrence_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated latent-context expert report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def validate_latent_context_expert_recurrence_report(
    report: Mapping[str, object],
) -> LatentContextExpertRecurrenceValidation:
    """Fail closed against full deterministic causal reconstruction."""

    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return LatentContextExpertRecurrenceValidation(
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
    return LatentContextExpertRecurrenceValidation(not errors, tuple(errors))


def latent_context_expert_recurrence_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize a valid report without writing it."""

    validation = validate_latent_context_expert_recurrence_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid latent-context expert report: " + "; ".join(validation.errors)
        )
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_ONLY",
    "LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA",
    "LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA",
    "LIMITATIONS",
    "LatentContextExpertRecurrenceProtocol",
    "LatentContextExpertRecurrenceValidation",
    "OUTPUT_WRITES_ALLOWED",
    "PARITY_FLOAT_MAX_ABS_TOLERANCE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "latent_context_expert_recurrence_report_json",
    "run_latent_context_expert_recurrence_development",
    "validate_latent_context_expert_recurrence_report",
]
