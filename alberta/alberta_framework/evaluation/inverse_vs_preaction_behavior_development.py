"""Development-only pre-action behavior versus inverse-action localizer.

One deterministic common prefix trains three fixed-capacity online tables: a
partner behavior predictor, a partner-action-conditional physical world
predictor, and a consecutive-observation inverse-action classifier.  The
prefix state is copied into three matched counterfactual continuations:

* ``control`` preserves the partner policy and physical law;
* ``partner_policy_drift`` changes only the action distribution given the
  public cue;
* ``physical_law_drift`` changes only how an action moves the physical bit.

No branch, task, or regime identifier enters a learner call.  At every step,
the behavior distribution and action-marginal world prediction are frozen
from the pre-observation before the partner action exists.  The inverse head
requires both consecutive observations and is therefore constructed only
after outcome reveal.  It is measured as a retrospective representation
objective, never relabelled as a decision-time belief.

The source is analytic and deterministic, while the learners start from
uninformative pseudocounts and consume every transition once.  All outcomes
are raw descriptive measurements.  This L0 module has no threshold, verdict,
winner, default, output writer, artifact authority, evidence claim, benchmark
authority, or scientific-promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

DEVELOPMENT_SCHEMA: Final = "alberta.inverse-vs-preaction-behavior.development.v1"
CONFIG_SCHEMA: Final = "alberta.inverse-vs-preaction-behavior.config.v1"
DEVELOPMENT_ONLY: Final = True
ASSESSMENT_STATUS: Final = "not_assessed"
EVIDENCE_LEVEL: Final = "L0"
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
BENCHMARK_EXECUTION_AUTHORITY: Final = False
ARTIFACT_AUTHORITY: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_CLAIMED: Final = False
THRESHOLDS_DEFINED: Final = False
TASK_IDENTIFIERS_EXPOSED: Final = False
RESETS_EXPOSED: Final = False
RANDOMNESS_CALLS: Final = 0
PASSES_OVER_EACH_SOURCE_EVENT: Final = 1

type BranchName = Literal[
    "control",
    "partner_policy_drift",
    "physical_law_drift",
]

BRANCH_NAMES: Final[tuple[BranchName, ...]] = (
    "control",
    "partner_policy_drift",
    "physical_law_drift",
)

_MODULE_RELATIVE_PATH: Final = (
    "alberta_framework/evaluation/inverse_vs_preaction_behavior_development.py"
)
_SOURCE_GENERATOR_VERSION: Final = "matched-policy-physics-factorization-v1"
_N_ACTIONS: Final = 2
_N_CUES: Final = 2
_N_PHYSICAL_BITS: Final = 2
_N_OBSERVATIONS: Final = _N_CUES * _N_PHYSICAL_BITS
_BEHAVIOR_TABLE_CELLS: Final = _N_CUES * _N_ACTIONS
_INVERSE_TABLE_CELLS: Final = _N_OBSERVATIONS * _N_OBSERVATIONS * _N_ACTIONS
_WORLD_TABLE_CELLS: Final = _N_PHYSICAL_BITS * _N_ACTIONS
_PERSISTENT_INTEGER_SCALARS: Final = (
    _BEHAVIOR_TABLE_CELLS
    + _INVERSE_TABLE_CELLS
    + 2 * _WORLD_TABLE_CELLS
    + 1
)
_LOGICAL_INTEGER_NBYTES: Final = 8
_OPERATION_ORDER: Final = (
    "observe_pre_observation",
    "freeze_pre_action_behavior_distribution",
    "freeze_action_conditional_world_predictions",
    "freeze_causal_action_marginal_world_prediction",
    "reveal_partner_action",
    "reveal_post_observation",
    "form_retrospective_inverse_distribution",
    "score_frozen_predictions",
    "commit_one_update_per_model",
)
_TIMING_CONTRACT: Final = (
    "The learner first receives one ordinary pre-observation containing a public cue and a "
    "physical bit; it receives no branch, task, or regime identifier.",
    "The behavior distribution, both action-conditional world predictions, and their behavior-"
    "weighted marginal are immutable before partner-action reveal.",
    "The partner action is revealed before the post-observation only as evaluator feedback; "
    "neither reveal can alter an already-frozen pre-action prediction.",
    "The inverse distribution cannot be formed until the post-observation has been revealed, "
    "and is scored only as a retrospective consecutive-pair objective.",
    "All three models update once after every frozen prediction has been scored.",
)
_LIMITATIONS: Final = (
    "one deterministic binary analytic construction is not a seed, population, or "
    "robustness result",
    "matched branches are counterfactual continuations copied from one common learned prefix, not "
    "simultaneously realizable continuations of one physical world",
    "the partner policy is a scripted three-to-one cue-conditioned frequency and not a "
    "learned peer",
    "the behavior, inverse, and physical predictors are finite tables with exact "
    "observation routing",
    "the inverse table receives the full consecutive observation pair only after outcome reveal",
    "the physical world head predicts one binary coordinate for one step and has no reward or "
    "return head",
    "the causal action marginal measures realized one-step physical-bit error and includes both "
    "action uncertainty and conditional-world error",
    "preallocated logical table bytes exclude Python objects, allocator peaks, hash objects, and "
    "hardware work",
    "streamed trace accounting retains a digest and one transient canonical event, not the event "
    "sequence",
    "counterfactual branch execution scales linearly in branch count and is evaluator work, not "
    "one agent life",
    "no control policy, representation learner, feature discovery, replay, planning, visual "
    "input, or scale claim is tested",
    "raw branch-minus-control deltas have no direction requirement, threshold, ranking, or "
    "promotion meaning",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_nbytes(value: object) -> int:
    return len(_canonical_json(value).encode("ascii"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _strict_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _strict_positive_float(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class InverseVsPreactionBehaviorConfig:
    """Bounded deterministic configuration with no evidence role."""

    run_id: str = "development.inverse-vs-preaction-behavior.v1"
    prefix_steps: int = 64
    branch_steps: int = 32
    entry_window: int = 8
    pseudocount: float = 1.0
    max_total_source_events: int = 16_384
    max_logical_state_bytes: int = 16_384
    max_report_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id or ".v" not in self.run_id:
            raise ValueError("run_id must be an exact non-empty versioned string")
        for name in (
            "prefix_steps",
            "branch_steps",
            "entry_window",
            "max_total_source_events",
            "max_logical_state_bytes",
            "max_report_bytes",
        ):
            _strict_positive_int(getattr(self, name), name=name)
        if self.prefix_steps < 16 or self.prefix_steps % 16 != 0:
            raise ValueError("prefix_steps must be a positive multiple of sixteen")
        if self.branch_steps < 16 or self.branch_steps % 16 != 0:
            raise ValueError("branch_steps must be a positive multiple of sixteen")
        if self.entry_window > self.branch_steps:
            raise ValueError("entry_window must not exceed branch_steps")
        _strict_positive_float(self.pseudocount, name="pseudocount")
        if self.pseudocount > 1_000_000.0:
            raise ValueError("pseudocount exceeds the bounded analytic contract")
        if self.total_source_events > self.max_total_source_events:
            raise ValueError("source work exceeds max_total_source_events")
        if self.logical_state_nbytes > self.max_logical_state_bytes:
            raise ValueError("state exceeds max_logical_state_bytes")

    @property
    def total_source_events(self) -> int:
        """Return one prefix plus all matched counterfactual continuation events."""

        return self.prefix_steps + len(BRANCH_NAMES) * self.branch_steps

    @property
    def logical_state_nbytes(self) -> int:
        """Return fixed logical int64 table bytes, including the step counter."""

        return _PERSISTENT_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "type": type(self).__name__,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(cls, payload: object) -> InverseVsPreactionBehaviorConfig:
        if type(payload) is not dict:
            raise TypeError("config must be an exact JSON object")
        mapping = cast(dict[str, object], payload)
        fields = {field.name for field in dataclasses.fields(cls)}
        if set(mapping) != {"schema_version", "type", *fields}:
            raise ValueError("config fields differ from the v1 contract")
        if (
            type(mapping["schema_version"]) is not str
            or mapping["schema_version"] != CONFIG_SCHEMA
            or type(mapping["type"]) is not str
            or mapping["type"] != cls.__name__
        ):
            raise ValueError("config schema or type differs")
        kwargs = {field.name: mapping[field.name] for field in dataclasses.fields(cls)}
        try:
            config = cls(**kwargs)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise type(error)(f"invalid config payload: {error}") from error
        if _canonical_json(config.to_config()) != _canonical_json(mapping):
            raise ValueError("config canonical types or bytes differ")
        return config


@dataclasses.dataclass(frozen=True, slots=True)
class _Observation:
    cue: int
    physical_bit: int

    def to_data(self) -> dict[str, int]:
        return {"cue": self.cue, "physical_bit": self.physical_bit}


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceEvent:
    pre_observation: _Observation
    partner_action: int
    post_observation: _Observation

    def to_data(self) -> dict[str, object]:
        return {
            "pre_observation": self.pre_observation.to_data(),
            "partner_action": self.partner_action,
            "post_observation": self.post_observation.to_data(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _LearnerState:
    behavior_counts: tuple[int, ...]
    inverse_counts: tuple[int, ...]
    world_counts: tuple[int, ...]
    world_one_counts: tuple[int, ...]
    steps_consumed: int


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenPreActionPrediction:
    """Fields frozen before partner action or post-observation reveal."""

    behavior_probabilities: tuple[float, float]
    conditional_post_one_probabilities: tuple[float, float]
    causal_action_marginal_post_one_probability: float
    state_sha256: str
    partner_action_revealed: bool = False
    post_observation_revealed: bool = False
    inverse_distribution_available: bool = False

    def to_data(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class _RevealedObservationPair:
    """Post-reveal pair for inverse prediction; the true action is excluded."""

    pre_observation: _Observation
    post_observation: _Observation
    frozen_pre_action_sha256: str
    outcome_revealed: bool = True


class _CanonicalArrayAccumulator:
    """Stream the exact canonical JSON-array hash without storing old events."""

    def __init__(self) -> None:
        self._hasher: Any = hashlib.sha256()
        self._hasher.update(b"[")
        self.event_count = 0
        self.canonical_prefix_nbytes = 1
        self.maximum_event_canonical_nbytes = 0

    def append(self, value: Mapping[str, object]) -> None:
        encoded = _canonical_json(value).encode("ascii")
        if self.event_count:
            self._hasher.update(b",")
            self.canonical_prefix_nbytes += 1
        self._hasher.update(encoded)
        self.event_count += 1
        self.canonical_prefix_nbytes += len(encoded)
        self.maximum_event_canonical_nbytes = max(
            self.maximum_event_canonical_nbytes,
            len(encoded),
        )

    def descriptor(self) -> dict[str, object]:
        digest = self._hasher.copy()
        digest.update(b"]")
        return {
            "event_count": self.event_count,
            "stored_event_count": 0,
            "sha256": digest.hexdigest(),
            "canonical_nbytes": self.canonical_prefix_nbytes + 1,
            "maximum_transient_event_canonical_nbytes": (
                self.maximum_event_canonical_nbytes
            ),
            "logical_digest_state_nbytes": hashlib.sha256().digest_size,
        }


class _MetricAccumulator:
    """Bounded raw sums and two binary confusion matrices."""

    def __init__(self) -> None:
        self.steps = 0
        self.behavior_nll = 0.0
        self.behavior_brier = 0.0
        self.inverse_nll = 0.0
        self.inverse_brier = 0.0
        self.conditional_world_squared_error = 0.0
        self.causal_marginal_world_squared_error = 0.0
        self.behavior_confusion = [[0, 0], [0, 0]]
        self.inverse_confusion = [[0, 0], [0, 0]]

    def add(self, row: Mapping[str, object]) -> None:
        action = cast(int, row["partner_action"])
        behavior_probabilities = cast(list[float], row["behavior_probabilities"])
        inverse_probabilities = cast(list[float], row["inverse_probabilities"])
        self.steps += 1
        self.behavior_nll += cast(float, row["behavior_nll"])
        self.behavior_brier += cast(float, row["behavior_brier"])
        self.inverse_nll += cast(float, row["inverse_nll"])
        self.inverse_brier += cast(float, row["inverse_brier"])
        self.conditional_world_squared_error += cast(
            float,
            row["conditional_world_squared_error"],
        )
        self.causal_marginal_world_squared_error += cast(
            float,
            row["causal_action_marginal_world_squared_error"],
        )
        behavior_prediction = 0 if behavior_probabilities[0] >= behavior_probabilities[1] else 1
        inverse_prediction = 0 if inverse_probabilities[0] >= inverse_probabilities[1] else 1
        self.behavior_confusion[action][behavior_prediction] += 1
        self.inverse_confusion[action][inverse_prediction] += 1

    def summary(self) -> dict[str, object]:
        if self.steps <= 0:
            raise ValueError("cannot summarize an empty metric accumulator")
        divisor = float(self.steps)
        behavior_correct = sum(self.behavior_confusion[index][index] for index in range(2))
        inverse_correct = sum(self.inverse_confusion[index][index] for index in range(2))
        return {
            "steps": self.steps,
            "behavior": {
                "nll": self.behavior_nll / divisor,
                "brier": self.behavior_brier / divisor,
                "argmax_accuracy": behavior_correct / divisor,
                "confusion_rows_true_columns_predicted": self.behavior_confusion,
            },
            "retrospective_inverse": {
                "nll": self.inverse_nll / divisor,
                "brier": self.inverse_brier / divisor,
                "argmax_accuracy": inverse_correct / divisor,
                "confusion_rows_true_columns_predicted": self.inverse_confusion,
                "decision_time_available": False,
            },
            "world": {
                "conditional_post_bit_squared_error": (
                    self.conditional_world_squared_error / divisor
                ),
                "causal_action_marginal_post_bit_squared_error": (
                    self.causal_marginal_world_squared_error / divisor
                ),
            },
        }


def _initial_state() -> _LearnerState:
    return _LearnerState(
        behavior_counts=(0,) * _BEHAVIOR_TABLE_CELLS,
        inverse_counts=(0,) * _INVERSE_TABLE_CELLS,
        world_counts=(0,) * _WORLD_TABLE_CELLS,
        world_one_counts=(0,) * _WORLD_TABLE_CELLS,
        steps_consumed=0,
    )


def _state_data(state: _LearnerState) -> dict[str, object]:
    return {
        "behavior_counts": list(state.behavior_counts),
        "inverse_counts": list(state.inverse_counts),
        "world_counts": list(state.world_counts),
        "world_one_counts": list(state.world_one_counts),
        "steps_consumed": state.steps_consumed,
    }


def _state_record(state: _LearnerState) -> dict[str, object]:
    data = _state_data(state)
    return {
        "content": data,
        "content_sha256": _sha256(data),
        "canonical_nbytes": _canonical_nbytes(data),
        "logical_preallocated_int64_nbytes": (
            _PERSISTENT_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES
        ),
    }


def _replace_tuple(value: tuple[int, ...], index: int, replacement: int) -> tuple[int, ...]:
    mutable = list(value)
    mutable[index] = replacement
    return tuple(mutable)


def _observation_index(observation: _Observation) -> int:
    return observation.cue * _N_PHYSICAL_BITS + observation.physical_bit


def _behavior_index(cue: int, action: int) -> int:
    return cue * _N_ACTIONS + action


def _inverse_index(pre: _Observation, post: _Observation, action: int) -> int:
    pair_index = _observation_index(pre) * _N_OBSERVATIONS + _observation_index(post)
    return pair_index * _N_ACTIONS + action


def _world_index(physical_bit: int, action: int) -> int:
    return physical_bit * _N_ACTIONS + action


def _binary_probabilities(
    count_zero: int,
    count_one: int,
    pseudocount: float,
) -> tuple[float, float]:
    denominator = float(count_zero + count_one) + 2.0 * pseudocount
    return (
        (float(count_zero) + pseudocount) / denominator,
        (float(count_one) + pseudocount) / denominator,
    )


def _freeze_pre_action(
    state: _LearnerState,
    pre_observation: _Observation,
    pseudocount: float,
) -> _FrozenPreActionPrediction:
    """Freeze only quantities available before the partner action exists."""

    behavior_probabilities = _binary_probabilities(
        state.behavior_counts[_behavior_index(pre_observation.cue, 0)],
        state.behavior_counts[_behavior_index(pre_observation.cue, 1)],
        pseudocount,
    )
    conditional: list[float] = []
    for action in range(_N_ACTIONS):
        index = _world_index(pre_observation.physical_bit, action)
        denominator = float(state.world_counts[index]) + 2.0 * pseudocount
        conditional.append(
            (float(state.world_one_counts[index]) + pseudocount) / denominator
        )
    conditional_tuple = (conditional[0], conditional[1])
    marginal = sum(
        behavior_probabilities[action] * conditional_tuple[action]
        for action in range(_N_ACTIONS)
    )
    return _FrozenPreActionPrediction(
        behavior_probabilities=behavior_probabilities,
        conditional_post_one_probabilities=conditional_tuple,
        causal_action_marginal_post_one_probability=marginal,
        state_sha256=_sha256(_state_data(state)),
    )


def _retrospective_inverse_distribution(
    state: _LearnerState,
    revealed_pair: _RevealedObservationPair,
    pseudocount: float,
) -> tuple[float, float]:
    """Predict from a revealed observation pair whose data surface has no label."""

    if revealed_pair.outcome_revealed is not True:
        raise ValueError("inverse prediction requires an outcome-revealed transition")
    return _binary_probabilities(
        state.inverse_counts[
            _inverse_index(
                revealed_pair.pre_observation,
                revealed_pair.post_observation,
                0,
            )
        ],
        state.inverse_counts[
            _inverse_index(
                revealed_pair.pre_observation,
                revealed_pair.post_observation,
                1,
            )
        ],
        pseudocount,
    )


def _update_state(state: _LearnerState, event: _SourceEvent) -> _LearnerState:
    behavior_index = _behavior_index(event.pre_observation.cue, event.partner_action)
    inverse_index = _inverse_index(
        event.pre_observation,
        event.post_observation,
        event.partner_action,
    )
    world_index = _world_index(event.pre_observation.physical_bit, event.partner_action)
    return _LearnerState(
        behavior_counts=_replace_tuple(
            state.behavior_counts,
            behavior_index,
            state.behavior_counts[behavior_index] + 1,
        ),
        inverse_counts=_replace_tuple(
            state.inverse_counts,
            inverse_index,
            state.inverse_counts[inverse_index] + 1,
        ),
        world_counts=_replace_tuple(
            state.world_counts,
            world_index,
            state.world_counts[world_index] + 1,
        ),
        world_one_counts=_replace_tuple(
            state.world_one_counts,
            world_index,
            state.world_one_counts[world_index] + event.post_observation.physical_bit,
        ),
        steps_consumed=state.steps_consumed + 1,
    )


def _source_event(
    step: int,
    *,
    physical_bit: int,
    partner_policy_drift: bool,
    physical_law_drift: bool,
) -> _SourceEvent:
    cue = step % _N_CUES
    baseline_matches_cue = (step // 4) % 4 != 3
    baseline_action = cue if baseline_matches_cue else 1 - cue
    partner_action = 1 - baseline_action if partner_policy_drift else baseline_action
    post_physical_bit = physical_bit ^ partner_action ^ int(physical_law_drift)
    return _SourceEvent(
        pre_observation=_Observation(cue=cue, physical_bit=physical_bit),
        partner_action=partner_action,
        post_observation=_Observation(cue=1 - cue, physical_bit=post_physical_bit),
    )


def _score(
    frozen: _FrozenPreActionPrediction,
    inverse_probabilities: tuple[float, float],
    event: _SourceEvent,
) -> dict[str, object]:
    action = event.partner_action
    target_probabilities = (1.0, 0.0) if action == 0 else (0.0, 1.0)
    behavior_nll = -math.log(frozen.behavior_probabilities[action])
    inverse_nll = -math.log(inverse_probabilities[action])
    behavior_brier = sum(
        (frozen.behavior_probabilities[index] - target_probabilities[index]) ** 2
        for index in range(_N_ACTIONS)
    )
    inverse_brier = sum(
        (inverse_probabilities[index] - target_probabilities[index]) ** 2
        for index in range(_N_ACTIONS)
    )
    post_bit = float(event.post_observation.physical_bit)
    conditional = frozen.conditional_post_one_probabilities[action]
    return {
        "partner_action": action,
        "behavior_probabilities": list(frozen.behavior_probabilities),
        "inverse_probabilities": list(inverse_probabilities),
        "behavior_nll": behavior_nll,
        "behavior_brier": behavior_brier,
        "inverse_nll": inverse_nll,
        "inverse_brier": inverse_brier,
        "conditional_world_squared_error": (conditional - post_bit) ** 2,
        "causal_action_marginal_world_squared_error": (
            frozen.causal_action_marginal_post_one_probability - post_bit
        )
        ** 2,
    }


def _timing_witness(
    frozen: _FrozenPreActionPrediction,
    revealed_pair: _RevealedObservationPair,
    partner_action_after_reveal: int,
    inverse_probabilities: tuple[float, float],
) -> dict[str, object]:
    frozen_data = frozen.to_data()
    inverse_input_data = {
        "pre_observation": revealed_pair.pre_observation.to_data(),
        "post_observation": revealed_pair.post_observation.to_data(),
        "frozen_pre_action_sha256": revealed_pair.frozen_pre_action_sha256,
        "outcome_revealed": revealed_pair.outcome_revealed,
    }
    return {
        "operation_order": list(_OPERATION_ORDER),
        "pre_action_payload": frozen_data,
        "pre_action_payload_sha256": _sha256(frozen_data),
        "partner_action_revealed_for_scoring_and_update": partner_action_after_reveal,
        "inverse_input_payload_without_action_label": inverse_input_data,
        "inverse_input_payload_sha256": _sha256(inverse_input_data),
        "inverse_input_contains_partner_action_label": False,
        "retrospective_inverse_probabilities": list(inverse_probabilities),
        "retrospective_inverse_requires_post_observation": True,
        "inverse_decision_time_available": False,
    }


def _run_segment(
    initial_state: _LearnerState,
    *,
    steps: int,
    entry_window: int,
    pseudocount: float,
    partner_policy_drift: bool,
    physical_law_drift: bool,
) -> tuple[_LearnerState, dict[str, object]]:
    source_trace = _CanonicalArrayAccumulator()
    trajectory_trace = _CanonicalArrayAccumulator()
    full_metrics = _MetricAccumulator()
    entry_metrics = _MetricAccumulator()
    state = initial_state
    physical_bit = 0
    first_timing_witness: dict[str, object] | None = None

    for step in range(steps):
        event = _source_event(
            step,
            physical_bit=physical_bit,
            partner_policy_drift=partner_policy_drift,
            physical_law_drift=physical_law_drift,
        )
        source_trace.append(event.to_data())
        frozen = _freeze_pre_action(state, event.pre_observation, pseudocount)
        frozen_sha = _sha256(frozen.to_data())
        revealed_pair = _RevealedObservationPair(
            pre_observation=event.pre_observation,
            post_observation=event.post_observation,
            frozen_pre_action_sha256=frozen_sha,
        )
        inverse_probabilities = _retrospective_inverse_distribution(
            state,
            revealed_pair,
            pseudocount,
        )
        row = _score(frozen, inverse_probabilities, event)
        full_metrics.add(row)
        if step < entry_window:
            entry_metrics.add(row)
        updated = _update_state(state, event)
        trace_event: dict[str, object] = {
            "segment_step": step,
            "pre_observation": event.pre_observation.to_data(),
            "frozen_pre_action": frozen.to_data(),
            "frozen_pre_action_sha256": frozen_sha,
            "revealed_partner_action": event.partner_action,
            "revealed_post_observation": event.post_observation.to_data(),
            "retrospective_inverse_probabilities": list(inverse_probabilities),
            "raw_scores": row,
            "pre_state_sha256": _sha256(_state_data(state)),
            "post_state_sha256": _sha256(_state_data(updated)),
            "timing_contract_sha256": _sha256(list(_OPERATION_ORDER)),
            "behavior_updates_committed": 1,
            "inverse_updates_committed": 1,
            "world_updates_committed": 1,
        }
        trajectory_trace.append(trace_event)
        if first_timing_witness is None:
            first_timing_witness = _timing_witness(
                frozen,
                revealed_pair,
                event.partner_action,
                inverse_probabilities,
            )
        state = updated
        physical_bit = event.post_observation.physical_bit

    if first_timing_witness is None:
        raise AssertionError("bounded config guarantees at least one segment event")
    if physical_bit != 0:
        raise AssertionError("balanced sixteen-step segments must close the physical cycle")
    return state, {
        "source_trace": source_trace.descriptor(),
        "trajectory_trace": trajectory_trace.descriptor(),
        "entry": entry_metrics.summary(),
        "full": full_metrics.summary(),
        "first_event_timing_witness": first_timing_witness,
        "source_initial_physical_bit": 0,
        "source_final_physical_bit": physical_bit,
        "post_observation_equals_next_pre_observation_within_segment": True,
    }


def _metric_delta(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    left_behavior = cast(Mapping[str, object], left["behavior"])
    right_behavior = cast(Mapping[str, object], right["behavior"])
    left_inverse = cast(Mapping[str, object], left["retrospective_inverse"])
    right_inverse = cast(Mapping[str, object], right["retrospective_inverse"])
    left_world = cast(Mapping[str, object], left["world"])
    right_world = cast(Mapping[str, object], right["world"])

    def confusion_delta(
        left_matrix: object,
        right_matrix: object,
    ) -> list[list[int]]:
        lhs = cast(list[list[int]], left_matrix)
        rhs = cast(list[list[int]], right_matrix)
        return [[lhs[row][column] - rhs[row][column] for column in range(2)] for row in range(2)]

    return {
        "behavior_nll": cast(float, left_behavior["nll"])
        - cast(float, right_behavior["nll"]),
        "behavior_brier": cast(float, left_behavior["brier"])
        - cast(float, right_behavior["brier"]),
        "behavior_argmax_accuracy": cast(float, left_behavior["argmax_accuracy"])
        - cast(float, right_behavior["argmax_accuracy"]),
        "behavior_confusion_rows_true_columns_predicted": confusion_delta(
            left_behavior["confusion_rows_true_columns_predicted"],
            right_behavior["confusion_rows_true_columns_predicted"],
        ),
        "retrospective_inverse_nll": cast(float, left_inverse["nll"])
        - cast(float, right_inverse["nll"]),
        "retrospective_inverse_brier": cast(float, left_inverse["brier"])
        - cast(float, right_inverse["brier"]),
        "retrospective_inverse_argmax_accuracy": cast(
            float,
            left_inverse["argmax_accuracy"],
        )
        - cast(float, right_inverse["argmax_accuracy"]),
        "retrospective_inverse_confusion_rows_true_columns_predicted": confusion_delta(
            left_inverse["confusion_rows_true_columns_predicted"],
            right_inverse["confusion_rows_true_columns_predicted"],
        ),
        "conditional_world_squared_error": cast(
            float,
            left_world["conditional_post_bit_squared_error"],
        )
        - cast(float, right_world["conditional_post_bit_squared_error"]),
        "causal_action_marginal_world_squared_error": cast(
            float,
            left_world["causal_action_marginal_post_bit_squared_error"],
        )
        - cast(float, right_world["causal_action_marginal_post_bit_squared_error"]),
    }


def _source_contract(config: InverseVsPreactionBehaviorConfig) -> dict[str, object]:
    return {
        "generator_version": _SOURCE_GENERATOR_VERSION,
        "prefix_steps": config.prefix_steps,
        "branch_steps": config.branch_steps,
        "branch_names_evaluator_only": list(BRANCH_NAMES),
        "pre_observation": "public_cue_and_physical_bit",
        "baseline_partner_policy": "action_matches_cue_in_three_of_each_four_balanced_blocks",
        "policy_drift": "invert_every_baseline_partner_action",
        "baseline_physical_law": "post_bit=pre_bit_xor_partner_action",
        "physical_drift": "post_bit=pre_bit_xor_partner_action_xor_one",
        "post_cue": "one_minus_pre_cue",
        "learner_branch_identifiers": False,
        "learner_task_identifiers": False,
        "learner_resets": False,
        "randomness_calls": RANDOMNESS_CALLS,
        "passes_over_each_source_event": PASSES_OVER_EACH_SOURCE_EVENT,
    }


def _resource_summary(
    config: InverseVsPreactionBehaviorConfig,
    segment_results: list[dict[str, object]],
) -> dict[str, object]:
    maximum_transient = max(
        cast(int, cast(dict[str, object], result[trace_name])[
            "maximum_transient_event_canonical_nbytes"
        ])
        for result in segment_results
        for trace_name in ("source_trace", "trajectory_trace")
    )
    return {
        "behavior_table_cells": _BEHAVIOR_TABLE_CELLS,
        "inverse_table_cells": _INVERSE_TABLE_CELLS,
        "world_count_table_cells": _WORLD_TABLE_CELLS,
        "world_one_count_table_cells": _WORLD_TABLE_CELLS,
        "state_step_counter_scalars": 1,
        "persistent_integer_scalars": _PERSISTENT_INTEGER_SCALARS,
        "logical_bytes_per_integer_scalar": _LOGICAL_INTEGER_NBYTES,
        "logical_preallocated_state_nbytes": config.logical_state_nbytes,
        "state_size_fixed_across_steps": True,
        "state_copies_for_counterfactual_branches": len(BRANCH_NAMES),
        "total_source_events": config.total_source_events,
        "prefix_source_events": config.prefix_steps,
        "continuation_source_events_per_branch": config.branch_steps,
        "passes_over_each_source_event": PASSES_OVER_EACH_SOURCE_EVENT,
        "behavior_probability_cells_read_per_event": _N_ACTIONS,
        "conditional_world_cells_read_per_event": 2 * _N_ACTIONS,
        "retrospective_inverse_cells_read_per_event": _N_ACTIONS,
        "model_table_cells_addressed_for_update_per_event": 4,
        "behavior_predictions_per_event": 1,
        "conditional_world_predictions_per_event": _N_ACTIONS,
        "causal_action_marginals_per_event": 1,
        "retrospective_inverse_predictions_per_event": 1,
        "replay_capacity": 0,
        "stored_trace_events": 0,
        "logical_digest_state_nbytes_per_stream": hashlib.sha256().digest_size,
        "maximum_transient_event_canonical_nbytes": maximum_transient,
        "randomness_calls": RANDOMNESS_CALLS,
        "learner_resets": 0,
        "learner_task_or_branch_identifiers": 0,
        "persistent_state_scaling": "O(C*A + O^2*A + P*A)",
        "per_event_prediction_scaling": "O(A)",
        "evaluator_counterfactual_work_scaling": "O(prefix_steps + branches*branch_steps)",
        "symbols": {
            "C": "public cue cardinality",
            "A": "partner action cardinality",
            "O": "complete observation cardinality",
            "P": "physical pre-state cardinality",
        },
        "python_object_and_allocator_bytes_included": False,
    }


def _work_summary(config: InverseVsPreactionBehaviorConfig) -> dict[str, object]:
    contract: dict[str, object] = {
        "common_prefix_events_consumed": config.prefix_steps,
        "counterfactual_continuation_events_consumed": (
            len(BRANCH_NAMES) * config.branch_steps
        ),
        "total_evaluator_source_events_consumed": config.total_source_events,
        "behavior_distributions_frozen": config.total_source_events,
        "action_conditional_world_cells_predicted": (
            _N_ACTIONS * config.total_source_events
        ),
        "causal_action_marginals_frozen": config.total_source_events,
        "retrospective_inverse_distributions_formed_after_reveal": (
            config.total_source_events
        ),
        "behavior_updates_committed": config.total_source_events,
        "inverse_updates_committed": config.total_source_events,
        "world_updates_committed": config.total_source_events,
        "counterfactual_state_copies": len(BRANCH_NAMES),
        "source_event_replays": 0,
        "randomness_calls": RANDOMNESS_CALLS,
    }
    return {**contract, "work_contract_sha256": _sha256(contract)}


def _seal_report(report: dict[str, object]) -> dict[str, object]:
    resource = cast(dict[str, object], report["resource"])
    resource["final_report_canonical_nbytes"] = 0
    while True:
        payload_sha256 = _sha256(report)
        candidate = {
            **report,
            "integrity": {
                "report_without_integrity_sha256": payload_sha256,
            },
        }
        size = _canonical_nbytes(candidate)
        if resource["final_report_canonical_nbytes"] == size:
            return candidate
        resource["final_report_canonical_nbytes"] = size


def run_inverse_vs_preaction_behavior_development(
    config: InverseVsPreactionBehaviorConfig | None = None,
) -> dict[str, object]:
    """Run the bounded common-prefix and matched-branch L0 construction."""

    if config is None:
        cfg = InverseVsPreactionBehaviorConfig()
    elif type(config) is not InverseVsPreactionBehaviorConfig:
        raise TypeError("config must be an exact InverseVsPreactionBehaviorConfig")
    else:
        cfg = config
    initial_state = _initial_state()
    prefix_final_state, prefix_result = _run_segment(
        initial_state,
        steps=cfg.prefix_steps,
        entry_window=cfg.entry_window,
        pseudocount=cfg.pseudocount,
        partner_policy_drift=False,
        physical_law_drift=False,
    )

    branch_flags: dict[BranchName, tuple[bool, bool]] = {
        "control": (False, False),
        "partner_policy_drift": (True, False),
        "physical_law_drift": (False, True),
    }
    branch_results: list[dict[str, object]] = []
    branch_final_states: dict[str, object] = {}
    for branch in BRANCH_NAMES:
        policy_drift, physics_drift = branch_flags[branch]
        final_state, result = _run_segment(
            prefix_final_state,
            steps=cfg.branch_steps,
            entry_window=cfg.entry_window,
            pseudocount=cfg.pseudocount,
            partner_policy_drift=policy_drift,
            physical_law_drift=physics_drift,
        )
        result = {
            "branch": branch,
            "evaluator_only_intervention": {
                "partner_policy_mapping_changed": policy_drift,
                "physical_transition_law_changed": physics_drift,
            },
            "starts_from_common_prefix_state_sha256": _sha256(
                _state_data(prefix_final_state)
            ),
            "final_state_sha256": _sha256(_state_data(final_state)),
            **result,
        }
        branch_results.append(result)
        branch_final_states[branch] = _state_record(final_state)

    by_name = {
        cast(str, result["branch"]): result
        for result in branch_results
    }
    control = by_name["control"]
    deltas = {
        branch: {
            "entry": _metric_delta(
                cast(Mapping[str, object], by_name[branch]["entry"]),
                cast(Mapping[str, object], control["entry"]),
            ),
            "full": _metric_delta(
                cast(Mapping[str, object], by_name[branch]["full"]),
                cast(Mapping[str, object], control["full"]),
            ),
        }
        for branch in ("partner_policy_drift", "physical_law_drift")
    }

    source_contract = _source_contract(cfg)
    all_segments = [prefix_result, *branch_results]
    source_manifest = [
        cast(dict[str, object], segment["source_trace"])
        for segment in all_segments
    ]
    trajectory_manifest = [
        cast(dict[str, object], segment["trajectory_trace"])
        for segment in all_segments
    ]
    report: dict[str, object] = {
        "schema": DEVELOPMENT_SCHEMA,
        "status": "completed",
        "development_only": DEVELOPMENT_ONLY,
        "assessment_status": ASSESSMENT_STATUS,
        "evidence_level": EVIDENCE_LEVEL,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "benchmark_execution_authority": BENCHMARK_EXECUTION_AUTHORITY,
        "artifact_authority": ARTIFACT_AUTHORITY,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "evidence_claimed": EVIDENCE_CLAIMED,
        "thresholds_defined": THRESHOLDS_DEFINED,
        "task_identifiers_exposed": TASK_IDENTIFIERS_EXPOSED,
        "resets_exposed": RESETS_EXPOSED,
        "descriptive_claims_only": True,
        "config": cfg.to_config(),
        "source_contract": source_contract,
        "source_contract_sha256": _sha256(source_contract),
        "source_manifest_sha256": _sha256(source_manifest),
        "provenance": {
            "module_relative_path": _MODULE_RELATIVE_PATH,
            "module_sha256": _source_sha256(),
            "implementation": "pure_python_standard_library",
        },
        "timing_contract": list(_TIMING_CONTRACT),
        "timing_contract_sha256": _sha256(list(_OPERATION_ORDER)),
        "learner_visible_input_contract": {
            "pre_action": ["public_cue", "physical_bit"],
            "ordinary_feedback_after_action": ["partner_action", "post_observation"],
            "branch_or_task_identifier": False,
            "post_observation_available_to_pre_action_behavior_predictor": False,
            "post_observation_required_by_retrospective_inverse_head": True,
        },
        "states": {
            "initial": _state_record(initial_state),
            "after_common_prefix": _state_record(prefix_final_state),
            "branch_final": branch_final_states,
        },
        "common_prefix": prefix_result,
        "branch_results": branch_results,
        "trajectory_manifest_sha256": _sha256(trajectory_manifest),
        "branch_minus_control_deltas": deltas,
        "work": _work_summary(cfg),
        "resource": _resource_summary(cfg, all_segments),
        "limitations": list(_LIMITATIONS),
    }
    sealed = _seal_report(report)
    final_nbytes = cast(
        int,
        cast(dict[str, object], sealed["resource"])["final_report_canonical_nbytes"],
    )
    if final_nbytes > cfg.max_report_bytes:
        raise ValueError("report exceeds max_report_bytes")
    return sealed


def validate_inverse_vs_preaction_behavior_report(report: object) -> tuple[str, ...]:
    """Fail closed unless a report reconstructs with exact canonical types and bytes."""

    if type(report) is not dict:
        return ("report must be an exact JSON object",)
    candidate = cast(dict[str, object], report)
    errors: list[str] = []
    integrity = candidate.get("integrity")
    if type(integrity) is not dict:
        errors.append("integrity must be an exact JSON object")
    else:
        integrity_mapping = cast(dict[str, object], integrity)
        if set(integrity_mapping) != {"report_without_integrity_sha256"}:
            errors.append("integrity fields differ")
        claimed_sha = integrity_mapping.get("report_without_integrity_sha256")
        if type(claimed_sha) is not str or len(claimed_sha) != 64:
            errors.append("integrity digest type or length differs")
        else:
            unhashed = dict(candidate)
            unhashed.pop("integrity")
            if claimed_sha != _sha256(unhashed):
                errors.append("report integrity digest differs")

    config_payload = candidate.get("config")
    try:
        config = InverseVsPreactionBehaviorConfig.from_config(config_payload)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        return tuple(errors)
    try:
        expected = run_inverse_vs_preaction_behavior_development(config)
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"report reconstruction failed: {error}")
        return tuple(errors)
    if _canonical_json(candidate) != _canonical_json(expected):
        errors.append("report does not reconstruct with exact canonical types and bytes")
    resource = candidate.get("resource")
    if type(resource) is not dict:
        errors.append("resource must be an exact JSON object")
    else:
        claimed_nbytes = cast(dict[str, object], resource).get(
            "final_report_canonical_nbytes"
        )
        if type(claimed_nbytes) is not int or claimed_nbytes != _canonical_nbytes(candidate):
            errors.append("final report canonical byte count differs")
    return tuple(errors)


__all__ = [
    "ARTIFACT_AUTHORITY",
    "ASSESSMENT_STATUS",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "BRANCH_NAMES",
    "CONFIG_SCHEMA",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "EVIDENCE_CLAIMED",
    "EVIDENCE_LEVEL",
    "InverseVsPreactionBehaviorConfig",
    "OUTPUT_WRITES_ALLOWED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_DEFINED",
    "run_inverse_vs_preaction_behavior_development",
    "validate_inverse_vs_preaction_behavior_report",
]
