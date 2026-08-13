# mypy: disable-error-code="call-arg,operator"
"""Consumed-root Stage A diagnostic for H=2 quarantined latent experts.

The evaluator reuses the exact already-consumed FastSlow A/B/A source.  It is a
development-only, in-memory comparison between confirmation routing enabled and
disabled.  Both arms use the same fixed bank, H=2 relational law, opening and
pending transitions, candidate gradients, and source.  The intervention changes
only the committed owner after a confirmed quarantine.

Every event predicts before its target, records complete pending-state movement,
and identifies zero-parameter-commit openings or ambiguous-dormant abstentions.
There is no writer, search, margin, dwell parameter, winner, default, threshold,
evidence promotion, or new seed.
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

from alberta_framework.core.pairwise_dominance_quarantine import (
    TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
)
from alberta_framework.core.two_event_latent_context_experts import (
    ZERO_COMMIT_REASON_AMBIGUOUS_CHALLENGER,
    ZERO_COMMIT_REASON_NONE,
    ZERO_COMMIT_REASON_QUARANTINE_OPENED,
    ZERO_COMMIT_REASON_TRANSACTION_REJECTED,
    TwoEventLatentContextExpertConfig,
    TwoEventLatentContextExpertLearner,
    TwoEventLatentContextExpertLearningResult,
    TwoEventLatentContextExpertState,
    run_two_event_latent_context_expert_arrays,
    two_event_latent_context_expert_design_record,
    two_event_latent_context_expert_forward,
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

TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.two-event-latent-context-expert-recurrence-development.protocol.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.two-event-latent-context-expert-recurrence-development.report.v1"
)
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ASSESSMENT_STATUS: Final = "not_assessed"

ARM_NAMES: Final = (
    "two_event_confirmation_routing_enabled",
    "two_event_confirmation_routing_disabled",
)
EXECUTION_ENGINES: Final = ("python_eager", "jax_jit_scan")
PARITY_FLOAT_MAX_ABS_TOLERANCE: Final = 2.0e-6

LIMITATIONS: Final = (
    "this is one fixed H=2 sibling intervention, not a search over margins, dwell times, "
    "horizons, or learners",
    "one already-consumed scalar Gaussian root is not a population, robustness, control, "
    "or scale result",
    "the target mapping is unidentifiable from observations, so the first prediction after "
    "a mapping change necessarily precedes evidence of that change",
    "quarantine evidence is post-outcome and can route only future predictions; no "
    "pre-outcome context-identification claim is made",
    "opening and ambiguous dormant-challenger abstention advance the exact lifetime while "
    "committing zero parameter subtrees",
    "A probes reuse frozen A1 observations and are read-only diagnostics",
    "the finite uint32[2] lifetime does not establish indefinite continual operation",
    "there is no outcome threshold, winner, default, writer, held-out seed, evidence, or "
    "scientific-promotion path",
    "cross-birth prediction rescue is outside Stage A and is not executed here",
)

_ZERO_COMMIT_REASON_NAMES: Final = {
    ZERO_COMMIT_REASON_NONE: "none",
    ZERO_COMMIT_REASON_QUARANTINE_OPENED: "quarantine_opened",
    ZERO_COMMIT_REASON_AMBIGUOUS_CHALLENGER: "ambiguous_challenger_abstention",
    ZERO_COMMIT_REASON_TRANSACTION_REJECTED: "transaction_rejected",
}


@dataclasses.dataclass(frozen=True, slots=True)
class TwoEventLatentContextExpertRecurrenceProtocol:
    """Frozen consumed source and prespecified Stage A construction."""

    schema_version: str = TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA
    development_root_seed: int = DEVELOPMENT_ROOT_SEED
    phase_steps: int = PHASE_STEPS
    summary_window: int = SUMMARY_WINDOW
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    max_experts: int = 2
    step_size: float = 0.05
    grad_clip: float = 10.0
    confirmation_horizon: int = TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON

    def __post_init__(self) -> None:
        expected = (
            TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA,
            DEVELOPMENT_ROOT_SEED,
            PHASE_STEPS,
            SUMMARY_WINDOW,
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            0.05,
            10.0,
            2,
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
            self.confirmation_horizon,
        )
        types_changed = any(
            type(value) is not type(reference)
            for value, reference in zip(actual, expected, strict=True)
        )
        if actual != expected or types_changed:
            raise ValueError("the consumed two-event latent-context protocol is frozen")

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
            "confirmation_horizon": self.confirmation_horizon,
            "confirmation_horizon_structurally_fixed": True,
            "margin_or_dwell_parameter_present": False,
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
            "learner_inputs": ["observation", "then authenticated cache plus target"],
            "learner_metadata_exposed": [],
            "candidate_excluded_from_comparators": True,
            "first_switched_regime_prediction_precedes_outcome": True,
            "a_probe_inputs": "the frozen A1 observations, read-only",
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> TwoEventLatentContextExpertRecurrenceProtocol:
        protocol = cls()
        if not _exact_json_equal(dict(payload), protocol.to_config()):
            raise ValueError("protocol payload does not match the frozen consumed protocol")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class TwoEventLatentContextExpertRecurrenceValidation:
    """Strict in-memory reconstruction result."""

    valid: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutedArm:
    report: dict[str, object]
    checkpoints: tuple[TwoEventLatentContextExpertState, ...]


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
    protocol: TwoEventLatentContextExpertRecurrenceProtocol,
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
    protocol: TwoEventLatentContextExpertRecurrenceProtocol,
    arm_name: str,
) -> TwoEventLatentContextExpertConfig:
    if arm_name not in ARM_NAMES:
        raise ValueError("unsupported two-event latent-context recurrence arm")
    return TwoEventLatentContextExpertConfig(
        input_dim=protocol.input_dim,
        output_dim=protocol.output_dim,
        max_experts=protocol.max_experts,
        step_size=protocol.step_size,
        grad_clip=protocol.grad_clip,
        confirmation_routing_enabled=arm_name == ARM_NAMES[0],
    )


def _array_record(name: str, value: jax.Array) -> dict[str, object]:
    host = np.asarray(jax.device_get(value))
    canonical_dtype = host.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(host.astype(canonical_dtype, copy=False))
    return {
        "name": name,
        "shape": list(host.shape),
        "dtype": host.dtype.str,
        "nbytes": int(host.nbytes),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _state_record(state: TwoEventLatentContextExpertState) -> list[dict[str, object]]:
    return [
        _array_record("expert_weights", state.params.expert_weights),
        _array_record("expert_biases", state.params.expert_biases),
        _array_record("active_expert", state.active_expert),
        _array_record("step_count", state.step_count),
        _array_record("step_words", state.step_words),
        _array_record("pending_valid", state.pending_valid),
        _array_record("pending_owner", state.pending_owner),
        _array_record("pending_candidate", state.pending_candidate),
        _array_record("pending_birth_words", state.pending_birth_words),
        _array_record("pending_never_worse", state.pending_never_worse),
        _array_record("pending_ever_strict", state.pending_ever_strict),
    ]


def _state_sha256(state: TwoEventLatentContextExpertState) -> str:
    return _digest(_state_record(state))


def _expert_sha256(state: TwoEventLatentContextExpertState, expert: int) -> str:
    return _digest(
        [
            _array_record("expert_weights", state.params.expert_weights[expert]),
            _array_record("expert_biases", state.params.expert_biases[expert]),
        ]
    )


def _expert_norm(state: TwoEventLatentContextExpertState, expert: int) -> float:
    weights = np.asarray(jax.device_get(state.params.expert_weights[expert]), dtype=np.float64)
    biases = np.asarray(jax.device_get(state.params.expert_biases[expert]), dtype=np.float64)
    return math.sqrt(float(np.sum(np.square(weights)) + np.sum(np.square(biases))))


def _pending_record(state: TwoEventLatentContextExpertState) -> dict[str, object]:
    return {
        "valid": bool(np.asarray(jax.device_get(state.pending_valid))),
        "owner": int(np.asarray(jax.device_get(state.pending_owner))),
        "candidate": int(np.asarray(jax.device_get(state.pending_candidate))),
        "birth_words": [
            int(value)
            for value in np.asarray(jax.device_get(state.pending_birth_words))
        ],
        "never_worse": [
            bool(value)
            for value in np.asarray(jax.device_get(state.pending_never_worse))
        ],
        "ever_strict": [
            bool(value)
            for value in np.asarray(jax.device_get(state.pending_ever_strict))
        ],
    }


def _a_probe(
    state: TwoEventLatentContextExpertState,
    observations: jax.Array,
) -> dict[str, object]:
    before = _state_sha256(state)
    predictions = np.asarray(
        jax.device_get(
            jax.vmap(
                lambda observation: two_event_latent_context_expert_forward(
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
    after = _state_sha256(state)
    return {
        "examples": int(observations.shape[0]),
        "expert_a_mse": [float(value) for value in losses],
        "active_expert": active,
        "active_expert_a_mse": float(losses[active]),
        "read_only_state_sha256_before": before,
        "read_only_state_sha256_after": after,
        "state_unchanged": before == after,
    }


def _checkpoint(
    label: str,
    learner: TwoEventLatentContextExpertLearner,
    state: TwoEventLatentContextExpertState,
    probe_observations: jax.Array,
) -> dict[str, object]:
    k = learner.config.max_experts
    resources = learner.resource_record(state)
    return {
        "label": label,
        "step_count": int(np.asarray(jax.device_get(state.step_count))),
        "step_words": [
            int(value) for value in np.asarray(jax.device_get(state.step_words))
        ],
        "active_expert": int(np.asarray(jax.device_get(state.active_expert))),
        "pending": _pending_record(state),
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
    evidence_candidate_expert: int,
    selected_next_expert: int,
    expert_update_mask: Sequence[bool],
    parameter_subtree_commit_count: int,
    context_switched: bool,
    quarantine_opened: bool,
    quarantine_second_evidence: bool,
    quarantine_confirmed: bool,
    quarantine_rejected: bool,
    ambiguous_challenger_abstention: bool,
    zero_commit_reason: int,
    quarantine_never_worse: Sequence[bool],
    quarantine_ever_strict: Sequence[bool],
    pending_before_valid: bool,
    pending_after_valid: bool,
    pending_before_owner: int,
    pending_before_candidate: int,
    pending_before_birth_words: Sequence[int],
    pending_after_owner: int,
    pending_after_candidate: int,
    pending_after_birth_words: Sequence[int],
    pre_step_words: Sequence[int],
    post_step_words: Sequence[int],
) -> dict[str, object]:
    if zero_commit_reason not in _ZERO_COMMIT_REASON_NAMES:
        raise RuntimeError("unknown zero-commit reason")
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
        "evidence_candidate_expert": evidence_candidate_expert,
        "selected_next_expert": selected_next_expert,
        "expert_update_mask": list(expert_update_mask),
        "parameter_subtree_commit_count": parameter_subtree_commit_count,
        "context_switched": context_switched,
        "quarantine_opened": quarantine_opened,
        "quarantine_second_evidence": quarantine_second_evidence,
        "quarantine_confirmed": quarantine_confirmed,
        "quarantine_rejected": quarantine_rejected,
        "ambiguous_challenger_abstention": ambiguous_challenger_abstention,
        "zero_parameter_commit": parameter_subtree_commit_count == 0,
        "zero_commit_reason": _ZERO_COMMIT_REASON_NAMES[zero_commit_reason],
        "quarantine_never_worse": list(quarantine_never_worse),
        "quarantine_ever_strict": list(quarantine_ever_strict),
        "pending_before": {
            "valid": pending_before_valid,
            "owner": pending_before_owner,
            "candidate": pending_before_candidate,
            "birth_words": list(pending_before_birth_words),
        },
        "pending_after": {
            "valid": pending_after_valid,
            "owner": pending_after_owner,
            "candidate": pending_after_candidate,
            "birth_words": list(pending_after_birth_words),
        },
        "pre_step_words": list(pre_step_words),
        "post_step_words": list(post_step_words),
        "owner_reconstruction_error": prediction - expert_predictions[pre_update_owner],
        "current_error_relabelled_after_target": False,
        "update_applied": True,
    }


def _run_phase_eager(
    learner: TwoEventLatentContextExpertLearner,
    state: TwoEventLatentContextExpertState,
    observations: jax.Array,
    targets: jax.Array,
    *,
    phase_index: int,
    event_offset: int,
) -> tuple[TwoEventLatentContextExpertState, list[dict[str, object]]]:
    current = state
    trace: list[dict[str, object]] = []
    with jax.disable_jit():
        for phase_step in range(observations.shape[0]):
            observation = observations[phase_step]
            target = targets[phase_step]
            cache = learner.predict(current, observation)
            result = learner.update(current, cache, target)
            if not bool(result.update_applied):
                raise RuntimeError("prespecified two-event latent-context update rejected")
            if not np.array_equal(
                np.asarray(jax.device_get(cache.prediction)),
                np.asarray(jax.device_get(result.prediction)),
            ):
                raise RuntimeError("target relabelled the cached prequential prediction")
            host = jax.device_get(result)
            reason = int(np.asarray(host.zero_commit_reason))
            trace.append(
                _event(
                    event_index=event_offset + phase_step,
                    phase_index=phase_index,
                    phase_step=phase_step,
                    observation=float(np.asarray(jax.device_get(observation))[0]),
                    target=float(np.asarray(jax.device_get(target))[0]),
                    prediction=float(np.asarray(host.prediction)[0]),
                    expert_predictions=[
                        float(value)
                        for value in np.asarray(host.expert_predictions).reshape(-1)
                    ],
                    expert_losses=[
                        float(value) for value in np.asarray(host.expert_losses)
                    ],
                    candidate_gradient_norms=[
                        float(value)
                        for value in np.asarray(host.candidate_gradient_norms)
                    ],
                    pre_update_owner=int(np.asarray(host.pre_update_owner)),
                    evidence_best_expert=int(np.asarray(host.evidence_best_expert)),
                    evidence_candidate_expert=int(
                        np.asarray(host.evidence_candidate_expert)
                    ),
                    selected_next_expert=int(np.asarray(host.selected_next_expert)),
                    expert_update_mask=[
                        bool(value) for value in np.asarray(host.expert_update_mask)
                    ],
                    parameter_subtree_commit_count=int(
                        np.asarray(host.parameter_subtree_commit_count)
                    ),
                    context_switched=bool(np.asarray(host.context_switched)),
                    quarantine_opened=bool(np.asarray(host.quarantine_opened)),
                    quarantine_second_evidence=bool(
                        np.asarray(host.quarantine_second_evidence)
                    ),
                    quarantine_confirmed=bool(np.asarray(host.quarantine_confirmed)),
                    quarantine_rejected=bool(np.asarray(host.quarantine_rejected)),
                    ambiguous_challenger_abstention=bool(
                        np.asarray(host.ambiguous_challenger_abstention)
                    ),
                    zero_commit_reason=reason,
                    quarantine_never_worse=[
                        bool(value)
                        for value in np.asarray(host.quarantine_never_worse)
                    ],
                    quarantine_ever_strict=[
                        bool(value)
                        for value in np.asarray(host.quarantine_ever_strict)
                    ],
                    pending_before_valid=bool(np.asarray(host.pending_before_valid)),
                    pending_after_valid=bool(np.asarray(host.pending_after_valid)),
                    pending_before_owner=int(np.asarray(host.pending_before_owner)),
                    pending_before_candidate=int(
                        np.asarray(host.pending_before_candidate)
                    ),
                    pending_before_birth_words=[
                        int(value)
                        for value in np.asarray(host.pending_before_birth_words)
                    ],
                    pending_after_owner=int(np.asarray(host.pending_after_owner)),
                    pending_after_candidate=int(
                        np.asarray(host.pending_after_candidate)
                    ),
                    pending_after_birth_words=[
                        int(value)
                        for value in np.asarray(host.pending_after_birth_words)
                    ],
                    pre_step_words=[
                        int(value) for value in np.asarray(host.pre_step_words)
                    ],
                    post_step_words=[
                        int(value) for value in np.asarray(host.post_step_words)
                    ],
                )
            )
            current = result.state
    return current, trace


@functools.partial(jax.jit, static_argnums=(0,))
def _run_phase_compiled(
    learner: TwoEventLatentContextExpertLearner,
    state: TwoEventLatentContextExpertState,
    observations: jax.Array,
    targets: jax.Array,
) -> TwoEventLatentContextExpertLearningResult:
    return run_two_event_latent_context_expert_arrays(
        learner,
        observations,
        targets,
        state=state,
    )


def _compiled_events(
    result: TwoEventLatentContextExpertLearningResult,
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
        if not bool(np.asarray(host.update_applied)[phase_step]):
            raise RuntimeError("compiled two-event latent-context update rejected")
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
                evidence_candidate_expert=int(
                    np.asarray(host.evidence_candidate_expert)[phase_step]
                ),
                selected_next_expert=int(
                    np.asarray(host.selected_next_expert)[phase_step]
                ),
                expert_update_mask=[
                    bool(value)
                    for value in np.asarray(host.expert_update_mask)[phase_step]
                ],
                parameter_subtree_commit_count=int(
                    np.asarray(host.parameter_subtree_commit_count)[phase_step]
                ),
                context_switched=bool(np.asarray(host.context_switched)[phase_step]),
                quarantine_opened=bool(
                    np.asarray(host.quarantine_opened)[phase_step]
                ),
                quarantine_second_evidence=bool(
                    np.asarray(host.quarantine_second_evidence)[phase_step]
                ),
                quarantine_confirmed=bool(
                    np.asarray(host.quarantine_confirmed)[phase_step]
                ),
                quarantine_rejected=bool(
                    np.asarray(host.quarantine_rejected)[phase_step]
                ),
                ambiguous_challenger_abstention=bool(
                    np.asarray(host.ambiguous_challenger_abstention)[phase_step]
                ),
                zero_commit_reason=int(
                    np.asarray(host.zero_commit_reason)[phase_step]
                ),
                quarantine_never_worse=[
                    bool(value)
                    for value in np.asarray(host.quarantine_never_worse)[phase_step]
                ],
                quarantine_ever_strict=[
                    bool(value)
                    for value in np.asarray(host.quarantine_ever_strict)[phase_step]
                ],
                pending_before_valid=bool(
                    np.asarray(host.pending_before_valid)[phase_step]
                ),
                pending_after_valid=bool(
                    np.asarray(host.pending_after_valid)[phase_step]
                ),
                pending_before_owner=int(
                    np.asarray(host.pending_before_owner)[phase_step]
                ),
                pending_before_candidate=int(
                    np.asarray(host.pending_before_candidate)[phase_step]
                ),
                pending_before_birth_words=[
                    int(value)
                    for value in np.asarray(host.pending_before_birth_words)[phase_step]
                ],
                pending_after_owner=int(
                    np.asarray(host.pending_after_owner)[phase_step]
                ),
                pending_after_candidate=int(
                    np.asarray(host.pending_after_candidate)[phase_step]
                ),
                pending_after_birth_words=[
                    int(value)
                    for value in np.asarray(host.pending_after_birth_words)[phase_step]
                ],
                pre_step_words=[
                    int(value)
                    for value in np.asarray(host.pre_step_words)[phase_step]
                ],
                post_step_words=[
                    int(value)
                    for value in np.asarray(host.post_step_words)[phase_step]
                ],
            )
        )
    return events


def _phase_metrics(
    trace: Sequence[Mapping[str, object]],
    protocol: TwoEventLatentContextExpertRecurrenceProtocol,
) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase_index, name in enumerate(PHASE_NAMES):
        start = phase_index * protocol.phase_steps
        phase = trace[start : start + protocol.phase_steps]
        losses = np.asarray(
            [cast(float, event["squared_error"]) for event in phase],
            dtype=np.float64,
        )
        phases[name] = {
            "prequential_mse": float(np.mean(losses)),
            "early_prequential_mse": float(np.mean(losses[: protocol.summary_window])),
            "tail_prequential_mse": float(np.mean(losses[-protocol.summary_window :])),
            "context_switch_count": sum(
                bool(event["context_switched"]) for event in phase
            ),
            "quarantine_open_count": sum(
                bool(event["quarantine_opened"]) for event in phase
            ),
            "quarantine_second_evidence_count": sum(
                bool(event["quarantine_second_evidence"]) for event in phase
            ),
            "quarantine_confirmation_count": sum(
                bool(event["quarantine_confirmed"]) for event in phase
            ),
            "quarantine_rejection_count": sum(
                bool(event["quarantine_rejected"]) for event in phase
            ),
            "ambiguous_challenger_abstention_count": sum(
                bool(event["ambiguous_challenger_abstention"]) for event in phase
            ),
            "zero_parameter_commit_count": sum(
                bool(event["zero_parameter_commit"]) for event in phase
            ),
            "parameter_subtree_commit_count": sum(
                cast(int, event["parameter_subtree_commit_count"])
                for event in phase
            ),
            "distinct_pre_update_owners": sorted(
                {cast(int, event["pre_update_owner"]) for event in phase}
            ),
            "distinct_selected_next_experts": sorted(
                {cast(int, event["selected_next_expert"]) for event in phase}
            ),
        }
    return {"phase": phases}


def _validate_trace_contract(trace: Sequence[Mapping[str, object]]) -> None:
    for index, event in enumerate(trace):
        if cast(int, event["event_index"]) != index:
            raise RuntimeError("event index is not consecutive")
        if event["current_error_relabelled_after_target"] is not False:
            raise RuntimeError("current error was relabelled")
        commit_count = cast(int, event["parameter_subtree_commit_count"])
        mask = cast(list[bool], event["expert_update_mask"])
        if commit_count != sum(mask) or commit_count not in (0, 1):
            raise RuntimeError("parameter commit trace is inconsistent")
        opened = cast(bool, event["quarantine_opened"])
        ambiguous = cast(bool, event["ambiguous_challenger_abstention"])
        if (opened or ambiguous) and commit_count != 0:
            raise RuntimeError("opening or ambiguous abstention committed parameters")
        if opened and event["zero_commit_reason"] != "quarantine_opened":
            raise RuntimeError("opening zero-commit reason is missing")
        if ambiguous and event["zero_commit_reason"] != "ambiguous_challenger_abstention":
            raise RuntimeError("ambiguous zero-commit reason is missing")
        pending_after = cast(Mapping[str, object], event["pending_after"])
        if opened and pending_after["valid"] is not True:
            raise RuntimeError("opening did not install pending state")
        if cast(bool, event["quarantine_second_evidence"]) and pending_after["valid"]:
            raise RuntimeError("H=2 resolution did not clear pending state")
        if index:
            previous_after = cast(Mapping[str, object], trace[index - 1]["pending_after"])
            current_before = cast(Mapping[str, object], event["pending_before"])
            if not _exact_json_equal(previous_after, current_before):
                raise RuntimeError("pending transition is not consecutive")


def _execute_arm(
    protocol: TwoEventLatentContextExpertRecurrenceProtocol,
    observations: jax.Array,
    targets: jax.Array,
    *,
    arm_name: str,
    engine: str,
) -> _ExecutedArm:
    learner = TwoEventLatentContextExpertLearner(_arm_config(protocol, arm_name))
    initial_state = learner.init()
    state = initial_state
    probe_observations = observations[: protocol.phase_steps]
    checkpoints: dict[str, object] = {
        "initial": _checkpoint("initial", learner, state, probe_observations)
    }
    checkpoint_states: list[TwoEventLatentContextExpertState] = [state]
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
            raise ValueError("unsupported two-event latent-context execution engine")
        trace.extend(phase_trace)
        label = "A2_tail" if phase_name == "A2" else f"{phase_name}_end"
        checkpoints[label] = _checkpoint(label, learner, state, probe_observations)
        checkpoint_states.append(state)

    _validate_trace_contract(trace)
    b_end = cast(Mapping[str, object], checkpoints["B_end"])
    a2_entry = cast(Mapping[str, object], checkpoints["A2_entry"])
    if b_end["state_sha256"] != a2_entry["state_sha256"]:
        raise RuntimeError("A2 entry changed state before its first prediction")
    initial_resources = learner.resource_record(initial_state).to_dict()
    final_resources = learner.resource_record(state).to_dict()
    if initial_resources != final_resources:
        raise RuntimeError("two-event latent-context state capacity changed")
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
            "confirmation_horizon": 2,
            "pending_transition_evaluations_per_update": 1,
            "logical_pending_transition_evaluations": protocol.total_steps,
            "expert_predictions_per_update": 2 * k,
            "logical_expert_predictions": 2 * k * protocol.total_steps,
            "expert_losses_per_update": k,
            "logical_expert_losses": k * protocol.total_steps,
            "candidate_gradients_per_update": k,
            "logical_candidate_gradients": k * protocol.total_steps,
            "maximum_expert_subtree_commits_per_update": 1,
            "logical_expert_subtree_commit_slots_evaluated": protocol.total_steps,
            "logical_probe_observation_examples": checkpoint_count * protocol.phase_steps,
            "logical_probe_expert_predictions": (
                checkpoint_count * protocol.phase_steps * k
            ),
            "replay_samples": 0,
            "online_random_draws": 0,
        },
        "observed_routing_counts": {
            "parameter_subtree_commits": sum(
                cast(int, event["parameter_subtree_commit_count"])
                for event in trace
            ),
            "zero_parameter_commits": sum(
                bool(event["zero_parameter_commit"]) for event in trace
            ),
            "quarantine_openings": sum(
                bool(event["quarantine_opened"]) for event in trace
            ),
            "second_evidence_events": sum(
                bool(event["quarantine_second_evidence"]) for event in trace
            ),
            "confirmations": sum(
                bool(event["quarantine_confirmed"]) for event in trace
            ),
            "rejections": sum(bool(event["quarantine_rejected"]) for event in trace),
            "ambiguous_challenger_abstentions": sum(
                bool(event["ambiguous_challenger_abstention"]) for event in trace
            ),
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
        if not np.issubdtype(left_host.dtype, np.inexact):
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


def _json_parity(left: object, right: object) -> tuple[bool, float]:
    if type(left) is not type(right):
        return False, math.inf
    if type(left) is float:
        return True, abs(left - right)
    if type(left) is dict:
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        if set(left_dict) != set(right_dict):
            return False, math.inf
        results = [_json_parity(left_dict[key], right_dict[key]) for key in left_dict]
        return all(result[0] for result in results), max(
            (result[1] for result in results),
            default=0.0,
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        if len(left_list) != len(right_list):
            return False, math.inf
        results = [
            _json_parity(a, b)
            for a, b in zip(left_list, right_list, strict=True)
        ]
        return all(result[0] for result in results), max(
            (result[1] for result in results),
            default=0.0,
        )
    return left == right, 0.0 if left == right else math.inf


def _parity(
    eager: Sequence[_ExecutedArm],
    compiled: Sequence[_ExecutedArm],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for eager_arm, compiled_arm in zip(eager, compiled, strict=True):
        discrete_exact, trace_max = _json_parity(
            eager_arm.report["trace"],
            compiled_arm.report["trace"],
        )
        state_max = max(
            _tree_max_abs_difference(left, right)
            for left, right in zip(
                eager_arm.checkpoints,
                compiled_arm.checkpoints,
                strict=True,
            )
        )
        observed = max(trace_max, state_max)
        arm_name = cast(str, eager_arm.report["arm"])
        arms[arm_name] = {
            "trace_discrete_fields_exact": discrete_exact,
            "trace_float_max_abs_difference": trace_max,
            "checkpoint_state_max_abs_difference": state_max,
            "observed_max_abs_difference": observed,
            "declared_numeric_tolerance": PARITY_FLOAT_MAX_ABS_TOLERANCE,
            "within_declared_numeric_tolerance": (
                discrete_exact and observed <= PARITY_FLOAT_MAX_ABS_TOLERANCE
            ),
            "resources_exact": eager_arm.report["resources"]
            == compiled_arm.report["resources"],
            "work_exact": eager_arm.report["work"] == compiled_arm.report["work"],
            "observed_routing_counts_exact": eager_arm.report[
                "observed_routing_counts"
            ]
            == compiled_arm.report["observed_routing_counts"],
            "technical_tolerance_is_not_an_outcome_threshold": True,
        }
    return {"arms": arms}


def _phase_step_summary(steps: list[int]) -> dict[str, object]:
    return {
        "count": len(steps),
        "first": steps[0] if steps else None,
        "last": steps[-1] if steps else None,
        "phase_steps_sha256": _digest(steps),
        "phase_steps_if_at_most_16": steps if len(steps) <= 16 else None,
    }


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
    b_trace = trace[PHASE_STEPS : 2 * PHASE_STEPS]
    a2_trace = trace[2 * PHASE_STEPS :]
    b_a_update_steps = [
        cast(int, event["phase_step"])
        for event in b_trace
        if cast(list[bool], event["expert_update_mask"])[learned_a]
    ]
    a2_selected_steps = [
        cast(int, event["phase_step"])
        for event in a2_trace
        if cast(int, event["selected_next_expert"]) == learned_a
    ]
    b_end_owner = cast(int, b["active_expert"])
    latency = (
        0
        if b_end_owner == learned_a
        else a2_selected_steps[0] + 1
        if a2_selected_steps
        else None
    )
    return {
        "learned_a_expert_identity": learned_a,
        "identity_was_not_hard_coded": True,
        "direct_a_memory_probe": {
            "a1_end_a_expert_mse": a1_probe_values[learned_a],
            "b_end_a_expert_mse": b_probe_values[learned_a],
            "b_end_minus_a1_end_a_expert_mse": (
                b_probe_values[learned_a] - a1_probe_values[learned_a]
            ),
            "a2_tail_a_expert_mse": a2_probe_values[learned_a],
            "a1_end_subtree_sha256": a1_hashes[learned_a],
            "b_end_subtree_sha256": b_hashes[learned_a],
            "a2_tail_subtree_sha256": a2_hashes[learned_a],
            "subtree_bit_exact_across_b": a1_hashes[learned_a] == b_hashes[learned_a],
            "selected_update_count_during_b": len(b_a_update_steps),
            "b_phase_steps_updating_a_expert": _phase_step_summary(b_a_update_steps),
            "contains_any_a2_update_in_b_probe": False,
        },
        "first_b_two_event_window": [dict(event) for event in b_trace[:2]],
        "a2_reactivation": {
            "a2_entry_state_equals_b_end": a2_entry["state_sha256"] == b["state_sha256"],
            "b_end_owner": b_end_owner,
            "a1_owner_was_dormant_at_a2_entry": b_end_owner != learned_a,
            "observed_a2_outcomes_until_a1_owner_selected": latency,
            "latency_is_descriptive_not_thresholded": True,
            "first_a2_prediction_precedes_first_a2_outcome": True,
        },
        "routing_counts": report["observed_routing_counts"],
        "phase_metrics": report["metrics"],
        "a1_pending_at_end": cast(Mapping[str, object], a1["pending"])["valid"],
        "b_pending_at_end": cast(Mapping[str, object], b["pending"])["valid"],
        "a2_pending_at_end": cast(Mapping[str, object], a2_tail["pending"])["valid"],
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
    core_dir = evaluator_path.parents[1] / "core"
    source_path = evaluator_path.parent / "fast_slow_recurrence_development.py"
    payload: dict[str, object] = {
        "evaluator_module_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "two_event_latent_context_core_sha256": hashlib.sha256(
            (core_dir / "two_event_latent_context_experts.py").read_bytes()
        ).hexdigest(),
        "pairwise_dominance_helper_sha256": hashlib.sha256(
            (core_dir / "pairwise_dominance_quarantine.py").read_bytes()
        ).hexdigest(),
        "consumed_root_source_module_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    }
    payload["source_identity_sha256"] = _digest(payload)
    return payload


def _build_report() -> dict[str, object]:
    protocol = TwoEventLatentContextExpertRecurrenceProtocol()
    observations, targets, source_manifest, _unused_initialization_key = (
        _source_arrays_bound(protocol)
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
    enabled_config = _arm_config(protocol, ARM_NAMES[0]).to_config()
    disabled_config = _arm_config(protocol, ARM_NAMES[1]).to_config()
    differences = {
        name: {ARM_NAMES[0]: enabled_config[name], ARM_NAMES[1]: disabled_config[name]}
        for name in enabled_config
        if enabled_config[name] != disabled_config[name]
    }
    enabled = executions["jax_jit_scan"][0].report
    disabled = executions["jax_jit_scan"][1].report
    enabled_initial = cast(Mapping[str, object], enabled["checkpoints"])["initial"]
    disabled_initial = cast(Mapping[str, object], disabled["checkpoints"])["initial"]
    design = two_event_latent_context_expert_design_record().to_dict()
    body: dict[str, object] = {
        "schema_version": TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "output_writes_allowed": False,
        "assessment_status": ASSESSMENT_STATUS,
        "consumed_development_result": True,
        "stage": "A",
        "stage_b_executed": False,
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
            "initial_state_equal": cast(Mapping[str, object], enabled_initial)[
                "state_sha256"
            ]
            == cast(Mapping[str, object], disabled_initial)["state_sha256"],
            "resources_equal": enabled["resources"] == disabled["resources"],
            "fixed_work_equal": enabled["work"] == disabled["work"],
            "only_config_differences": differences,
            "expected_difference_fields": ["confirmation_routing_enabled"],
            "causal_intervention": (
                "route a confirmed H=2 quarantine to the candidate instead of the owner"
            ),
            "opening_evidence_candidate_work_matched_by_construction": True,
            "pending_transition_law_matched_by_construction": True,
            "winner_selected": False,
        },
        "consumed_findings": {
            "confirmation_routing_enabled": _arm_findings(enabled),
            "confirmation_routing_disabled": _arm_findings(disabled),
        },
        "executions": {
            engine: [execution.report for execution in executions[engine]]
            for engine in EXECUTION_ENGINES
        },
        "eager_compiled_parity": _parity(
            executions["python_eager"],
            executions["jax_jit_scan"],
        ),
        "descriptive_only": True,
        "winner_or_default_selected": False,
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


def run_two_event_latent_context_expert_recurrence_development() -> dict[str, object]:
    """Return the deterministic in-memory consumed-root Stage A diagnostic."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_two_event_latent_context_expert_recurrence_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated two-event latent-context report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def validate_two_event_latent_context_expert_recurrence_report(
    report: Mapping[str, object],
) -> TwoEventLatentContextExpertRecurrenceValidation:
    """Fail closed against full deterministic causal reconstruction."""

    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return TwoEventLatentContextExpertRecurrenceValidation(
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
    return TwoEventLatentContextExpertRecurrenceValidation(not errors, tuple(errors))


def two_event_latent_context_expert_recurrence_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize a valid report without writing it."""

    validation = validate_two_event_latent_context_expert_recurrence_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid two-event latent-context expert report: "
            + "; ".join(validation.errors)
        )
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_ONLY",
    "LIMITATIONS",
    "OUTPUT_WRITES_ALLOWED",
    "PARITY_FLOAT_MAX_ABS_TOLERANCE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA",
    "TwoEventLatentContextExpertRecurrenceProtocol",
    "TwoEventLatentContextExpertRecurrenceValidation",
    "run_two_event_latent_context_expert_recurrence_development",
    "two_event_latent_context_expert_recurrence_report_json",
    "validate_two_event_latent_context_expert_recurrence_report",
]
