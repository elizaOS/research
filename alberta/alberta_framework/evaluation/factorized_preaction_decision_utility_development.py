"""Development-only factorized pre-action one-step decision-utility probe.

A deterministic common prefix trains three separate fixed-capacity tables:
cue-conditioned partner behavior, own-by-partner grounded reward/physical
outcomes, and retrospective inverse action.  Every own/partner conditional
cell has support before the learned state is frozen.  Three matched contextual
continuations then preserve both factors, change only the partner policy, or
change only the physical/reward law.  Branch identifiers remain evaluator
metadata and never enter a learner call.

For every continuation event, the immutable state freezes one partner belief
and the complete conditional model before simultaneous actions are revealed.
The learned-belief planner, a uniform-belief control, and an inverse-misuse
arm with a preregistered uniform fallback all act from that causal surface.
The inverse objective cannot provide a belief without a post-observation and
its later output is never fed back into the decision.  Label-aware model and
true-law comparators are computed only after reveal and are explicitly not
decision-time agents.  Every proposed own action is evaluated against the
same partner action and branch law without updating the snapshot.

This is an in-memory L0 mechanism localizer.  It provides raw descriptive
receipts only, with no acceptance rule, ranking, preferred arm, output writer,
artifact authority, benchmark authority, evidence claim, or promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast

DEVELOPMENT_SCHEMA: Final = "alberta.factorized-preaction-decision-utility.development.v1"
CONFIG_SCHEMA: Final = "alberta.factorized-preaction-decision-utility.config.v1"
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
RANDOMNESS_CALLS: Final = 0

type BranchName = Literal[
    "control",
    "partner_policy_drift",
    "physical_reward_law_drift",
]
type ArmName = Literal[
    "learned_behavior_marginal",
    "uniform_belief_control",
    "inverse_action_unavailable_fallback",
    "actual_partner_conditional_model_ceiling",
    "evaluator_true_reward_ceiling",
]

BRANCH_NAMES: Final[tuple[BranchName, ...]] = (
    "control",
    "partner_policy_drift",
    "physical_reward_law_drift",
)
ARM_NAMES: Final[tuple[ArmName, ...]] = (
    "learned_behavior_marginal",
    "uniform_belief_control",
    "inverse_action_unavailable_fallback",
    "actual_partner_conditional_model_ceiling",
    "evaluator_true_reward_ceiling",
)
PREACTION_ARM_NAMES: Final[tuple[ArmName, ...]] = ARM_NAMES[:3]
POST_REVEAL_COMPARATOR_NAMES: Final[tuple[ArmName, ...]] = ARM_NAMES[3:]

_MODULE_RELATIVE_PATH: Final = (
    "alberta_framework/evaluation/factorized_preaction_decision_utility_development.py"
)
_SOURCE_GENERATOR_VERSION: Final = "factorized-simultaneous-contextual-game-v1"
_N_CUES: Final = 2
_N_OWN_ACTIONS: Final = 2
_N_PARTNER_ACTIONS: Final = 2
_BEHAVIOR_CELLS: Final = _N_CUES * _N_PARTNER_ACTIONS
_CONDITIONAL_CELLS: Final = _N_OWN_ACTIONS * _N_PARTNER_ACTIONS
_INVERSE_CELLS: Final = _N_OWN_ACTIONS * 2 * _N_PARTNER_ACTIONS
_PERSISTENT_INTEGER_SCALARS: Final = (
    _BEHAVIOR_CELLS
    + 3 * _CONDITIONAL_CELLS
    + _INVERSE_CELLS
    + 1
)
_LOGICAL_INTEGER_NBYTES: Final = 8
_LOWER_ACTION_TIE_RULE: Final = "equal scores select the lower integer own action"
_DECISION_ORDER: Final = (
    "observe_public_cue",
    "freeze_behavior_belief",
    "freeze_complete_own_by_partner_conditional_model",
    "form_learned_behavior_marginal_action",
    "form_uniform_belief_control_action",
    "record_inverse_input_unavailable_and_use_fixed_uniform_fallback",
    "reveal_partner_action",
    "form_actual_partner_conditional_model_comparator",
    "reveal_counterfactual_outcomes_for_both_own_actions",
    "form_evaluator_true_reward_comparator",
    "score_every_proposed_action_on_same_revealed_event",
    "form_retrospective_inverse_diagnostics_without_decision_feedback",
)
_TIMING_CONTRACT: Final = (
    "The public cue is the only event input available before the simultaneous partner action.",
    "One behavior belief and all four own-by-partner conditional reward/physical cells are "
    "frozen from the immutable common-prefix state before partner-action reveal.",
    "The learned-belief, uniform-control, and inverse-unavailable fallback actions are committed "
    "before the partner action or any outcome is revealed.",
    "The actual-partner model comparator consumes the revealed partner action and is not a "
    "causally valid pre-action agent.",
    "The true-reward comparator consumes both the partner action and evaluator-owned branch law "
    "and is only a counterfactual utility ceiling.",
    "Retrospective inverse distributions require post-observations and never alter any action.",
    "The frozen decision state receives no continuation update in any branch.",
)
_LIMITATIONS: Final = (
    "one deterministic binary contextual game is not a seed, population, or robustness result",
    "matched continuations are counterfactual branches from one snapshot, not one "
    "uninterrupted life",
    "the scripted partner does not learn, so this is not a two-learning-agent result",
    "independent contextual trials and immutable evaluation state do not prove continuous-life "
    "learning",
    "finite tables use exact cue and action routing rather than learned general feature discovery",
    "the grounded model predicts one-step binary reward and one binary physical coordinate only",
    "the actual-partner model comparator is a behavior-information ceiling under the frozen model, "
    "not a true-environment ceiling when the law changes",
    "the true-reward comparator has evaluator-only access to the partner action and branch law",
    "the inverse fallback is fixed uniform belief and does not estimate partner behavior",
    "raw trajectories are intentionally retained, so report memory grows linearly with "
    "evaluation steps",
    "all arms score both own actions, but learned-table and evaluator-true-law primitive work "
    "are different operations",
    "logical bytes and work exclude Python objects, allocator peaks, hash objects, and hardware "
    "operations",
    "no exploration, temporal credit, replay, planning horizon, visual input, safety, or scaling "
    "claim is tested",
    "branch-minus-control values are descriptive arithmetic with no required direction or "
    "decision authority",
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
class FactorizedPreactionDecisionUtilityConfig:
    """Bounded deterministic development configuration."""

    run_id: str = "development.factorized-preaction-decision-utility.v1"
    prefix_steps: int = 64
    evaluation_steps: int = 32
    pseudocount: float = 1.0
    max_total_source_events: int = 16_384
    max_logical_state_bytes: int = 16_384
    max_raw_trajectory_bytes: int = 2_097_152
    max_report_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id or ".v" not in self.run_id:
            raise ValueError("run_id must be an exact non-empty versioned string")
        for name in (
            "prefix_steps",
            "evaluation_steps",
            "max_total_source_events",
            "max_logical_state_bytes",
            "max_raw_trajectory_bytes",
            "max_report_bytes",
        ):
            _strict_positive_int(getattr(self, name), name=name)
        if self.prefix_steps < 16 or self.prefix_steps % 16 != 0:
            raise ValueError("prefix_steps must be a positive multiple of sixteen")
        if self.evaluation_steps < 16 or self.evaluation_steps % 16 != 0:
            raise ValueError("evaluation_steps must be a positive multiple of sixteen")
        _strict_positive_float(self.pseudocount, name="pseudocount")
        if self.pseudocount > 1_000_000.0:
            raise ValueError("pseudocount exceeds the bounded analytic contract")
        if self.total_source_events > self.max_total_source_events:
            raise ValueError("source work exceeds max_total_source_events")
        if self.logical_state_nbytes > self.max_logical_state_bytes:
            raise ValueError("state exceeds max_logical_state_bytes")

    @property
    def total_source_events(self) -> int:
        return self.prefix_steps + len(BRANCH_NAMES) * self.evaluation_steps

    @property
    def logical_state_nbytes(self) -> int:
        return _PERSISTENT_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "type": type(self).__name__,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(cls, payload: object) -> FactorizedPreactionDecisionUtilityConfig:
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
class _LearnedState:
    behavior_counts: tuple[int, ...]
    conditional_counts: tuple[int, ...]
    reward_one_counts: tuple[int, ...]
    physical_one_counts: tuple[int, ...]
    inverse_counts: tuple[int, ...]
    steps_consumed: int


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenPreactionSurface:
    cue: int
    behavior_belief: tuple[float, float]
    complete_conditional_table: tuple[tuple[float, float], ...]
    learned_state_sha256: str
    partner_action_revealed: bool = False
    outcome_revealed: bool = False

    def to_data(self) -> dict[str, object]:
        return {
            "cue": self.cue,
            "behavior_belief": list(self.behavior_belief),
            "complete_conditional_table": [
                {
                    "own_action": index // _N_PARTNER_ACTIONS,
                    "partner_action": index % _N_PARTNER_ACTIONS,
                    "reward_one_probability": row[0],
                    "physical_one_probability": row[1],
                }
                for index, row in enumerate(self.complete_conditional_table)
            ],
            "learned_state_sha256": self.learned_state_sha256,
            "partner_action_revealed": self.partner_action_revealed,
            "outcome_revealed": self.outcome_revealed,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _InverseObservationPair:
    """Retrospective inverse input; deliberately excludes the action label."""

    own_action: int
    post_physical_bit: int
    outcome_revealed: bool = True


def _initial_state() -> _LearnedState:
    return _LearnedState(
        behavior_counts=(0,) * _BEHAVIOR_CELLS,
        conditional_counts=(0,) * _CONDITIONAL_CELLS,
        reward_one_counts=(0,) * _CONDITIONAL_CELLS,
        physical_one_counts=(0,) * _CONDITIONAL_CELLS,
        inverse_counts=(0,) * _INVERSE_CELLS,
        steps_consumed=0,
    )


def _state_data(state: _LearnedState) -> dict[str, object]:
    return {
        "behavior_counts": list(state.behavior_counts),
        "conditional_counts": list(state.conditional_counts),
        "reward_one_counts": list(state.reward_one_counts),
        "physical_one_counts": list(state.physical_one_counts),
        "inverse_counts": list(state.inverse_counts),
        "steps_consumed": state.steps_consumed,
    }


def _state_record(state: _LearnedState) -> dict[str, object]:
    content = _state_data(state)
    return {
        "content": content,
        "content_sha256": _sha256(content),
        "canonical_nbytes": _canonical_nbytes(content),
        "logical_preallocated_int64_nbytes": (
            _PERSISTENT_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES
        ),
    }


def _replace(value: tuple[int, ...], index: int, replacement: int) -> tuple[int, ...]:
    mutable = list(value)
    mutable[index] = replacement
    return tuple(mutable)


def _behavior_index(cue: int, partner_action: int) -> int:
    return cue * _N_PARTNER_ACTIONS + partner_action


def _conditional_index(own_action: int, partner_action: int) -> int:
    return own_action * _N_PARTNER_ACTIONS + partner_action


def _inverse_index(own_action: int, post_physical_bit: int, partner_action: int) -> int:
    pair = own_action * 2 + post_physical_bit
    return pair * _N_PARTNER_ACTIONS + partner_action


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


def _baseline_partner_action(step: int) -> int:
    cue = step % _N_CUES
    matches_cue = (step // 4) % 4 != 3
    return cue if matches_cue else 1 - cue


def _true_outcome(
    own_action: int,
    partner_action: int,
    *,
    physical_reward_law_drift: bool,
) -> tuple[int, int]:
    actions_differ = own_action ^ partner_action
    post_physical_bit = actions_differ ^ int(physical_reward_law_drift)
    reward = 1 - post_physical_bit
    return reward, post_physical_bit


def _update_prefix_state(
    state: _LearnedState,
    *,
    cue: int,
    own_action: int,
    partner_action: int,
    reward: int,
    post_physical_bit: int,
) -> _LearnedState:
    behavior_index = _behavior_index(cue, partner_action)
    conditional_index = _conditional_index(own_action, partner_action)
    inverse_index = _inverse_index(own_action, post_physical_bit, partner_action)
    return _LearnedState(
        behavior_counts=_replace(
            state.behavior_counts,
            behavior_index,
            state.behavior_counts[behavior_index] + 1,
        ),
        conditional_counts=_replace(
            state.conditional_counts,
            conditional_index,
            state.conditional_counts[conditional_index] + 1,
        ),
        reward_one_counts=_replace(
            state.reward_one_counts,
            conditional_index,
            state.reward_one_counts[conditional_index] + reward,
        ),
        physical_one_counts=_replace(
            state.physical_one_counts,
            conditional_index,
            state.physical_one_counts[conditional_index] + post_physical_bit,
        ),
        inverse_counts=_replace(
            state.inverse_counts,
            inverse_index,
            state.inverse_counts[inverse_index] + 1,
        ),
        steps_consumed=state.steps_consumed + 1,
    )


def _run_common_prefix(
    config: FactorizedPreactionDecisionUtilityConfig,
) -> tuple[_LearnedState, dict[str, object]]:
    state = _initial_state()
    source_events: list[dict[str, int]] = []
    for step in range(config.prefix_steps):
        cue = step % _N_CUES
        own_action = (step // 2) % _N_OWN_ACTIONS
        partner_action = _baseline_partner_action(step)
        reward, post_physical_bit = _true_outcome(
            own_action,
            partner_action,
            physical_reward_law_drift=False,
        )
        source_events.append(
            {
                "step": step,
                "cue": cue,
                "own_action": own_action,
                "partner_action": partner_action,
                "reward": reward,
                "post_physical_bit": post_physical_bit,
            }
        )
        state = _update_prefix_state(
            state,
            cue=cue,
            own_action=own_action,
            partner_action=partner_action,
            reward=reward,
            post_physical_bit=post_physical_bit,
        )
    return state, {
        "source_event_count": config.prefix_steps,
        "source_sha256": _sha256(source_events),
        "source_canonical_nbytes": _canonical_nbytes(source_events),
        "passes_over_source": 1,
        "learner_updates": config.prefix_steps,
        "source_events_retained_in_report": 0,
    }


def _freeze_preaction_surface(
    state: _LearnedState,
    cue: int,
    pseudocount: float,
) -> _FrozenPreactionSurface:
    """Freeze causal decision inputs without partner action, outcome, or branch id."""

    behavior_belief = _binary_probabilities(
        state.behavior_counts[_behavior_index(cue, 0)],
        state.behavior_counts[_behavior_index(cue, 1)],
        pseudocount,
    )
    complete_table: list[tuple[float, float]] = []
    for own_action in range(_N_OWN_ACTIONS):
        for partner_action in range(_N_PARTNER_ACTIONS):
            index = _conditional_index(own_action, partner_action)
            denominator = float(state.conditional_counts[index]) + 2.0 * pseudocount
            complete_table.append(
                (
                    (float(state.reward_one_counts[index]) + pseudocount) / denominator,
                    (float(state.physical_one_counts[index]) + pseudocount) / denominator,
                )
            )
    return _FrozenPreactionSurface(
        cue=cue,
        behavior_belief=behavior_belief,
        complete_conditional_table=tuple(complete_table),
        learned_state_sha256=_sha256(_state_data(state)),
    )


def _model_scores(
    surface: _FrozenPreactionSurface,
    belief: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    reward_scores: list[float] = []
    physical_scores: list[float] = []
    for own_action in range(_N_OWN_ACTIONS):
        reward_score = 0.0
        physical_score = 0.0
        for partner_action in range(_N_PARTNER_ACTIONS):
            row = surface.complete_conditional_table[
                _conditional_index(own_action, partner_action)
            ]
            reward_score += belief[partner_action] * row[0]
            physical_score += belief[partner_action] * row[1]
        reward_scores.append(reward_score)
        physical_scores.append(physical_score)
    return (
        (reward_scores[0], reward_scores[1]),
        (physical_scores[0], physical_scores[1]),
    )


def _lower_action_argmax(scores: tuple[float, float]) -> int:
    return 0 if scores[0] >= scores[1] else 1


def _model_arm(
    *,
    arm: ArmName,
    surface: _FrozenPreactionSurface,
    belief: tuple[float, float],
    belief_source: str,
    causally_valid_preaction: bool,
    partner_action_consumed: bool,
    inverse_fallback_used: bool,
) -> dict[str, object]:
    reward_scores, physical_scores = _model_scores(surface, belief)
    surface_data = surface.to_data()
    return {
        "arm": arm,
        "causally_valid_preaction": causally_valid_preaction,
        "evaluator_only": not causally_valid_preaction,
        "belief_source": belief_source,
        "belief": list(belief),
        "partner_action_consumed_before_action": partner_action_consumed,
        "post_observation_consumed_before_action": False,
        "retrospective_inverse_output_consumed": False,
        "fixed_inverse_unavailable_fallback_used": inverse_fallback_used,
        "learned_predictor_snapshot_consumed": True,
        "frozen_preaction_surface_sha256": _sha256(surface_data),
        "frozen_complete_conditional_table_sha256": _sha256(
            surface_data["complete_conditional_table"]
        ),
        "joint_cell_evaluation_order": [
            [own_action, partner_action]
            for own_action in range(_N_OWN_ACTIONS)
            for partner_action in range(_N_PARTNER_ACTIONS)
        ],
        "reward_scores_by_own_action": list(reward_scores),
        "physical_one_predictions_by_own_action": list(physical_scores),
        "chosen_action": _lower_action_argmax(reward_scores),
        "tie_rule": _LOWER_ACTION_TIE_RULE,
    }


def _form_preaction_decisions(
    surface: _FrozenPreactionSurface,
) -> dict[ArmName, dict[str, object]]:
    """Form every causally valid action; no action label or outcome is accepted."""

    uniform = (0.5, 0.5)
    return {
        "learned_behavior_marginal": _model_arm(
            arm="learned_behavior_marginal",
            surface=surface,
            belief=surface.behavior_belief,
            belief_source="learned_cue_conditioned_preaction_behavior",
            causally_valid_preaction=True,
            partner_action_consumed=False,
            inverse_fallback_used=False,
        ),
        "uniform_belief_control": _model_arm(
            arm="uniform_belief_control",
            surface=surface,
            belief=uniform,
            belief_source="fixed_uniform_matched_control",
            causally_valid_preaction=True,
            partner_action_consumed=False,
            inverse_fallback_used=False,
        ),
        "inverse_action_unavailable_fallback": _model_arm(
            arm="inverse_action_unavailable_fallback",
            surface=surface,
            belief=uniform,
            belief_source="fixed_uniform_because_inverse_requires_unrevealed_post_observation",
            causally_valid_preaction=True,
            partner_action_consumed=False,
            inverse_fallback_used=True,
        ),
    }


def _form_post_reveal_comparators(
    surface: _FrozenPreactionSurface,
    partner_action: int,
    *,
    physical_reward_law_drift: bool,
) -> dict[ArmName, dict[str, object]]:
    """Form label-aware evaluator comparators only after partner-action reveal."""

    actual_belief = (1.0, 0.0) if partner_action == 0 else (0.0, 1.0)
    model_comparator = _model_arm(
        arm="actual_partner_conditional_model_ceiling",
        surface=surface,
        belief=actual_belief,
        belief_source="revealed_actual_partner_action_one_hot",
        causally_valid_preaction=False,
        partner_action_consumed=True,
        inverse_fallback_used=False,
    )
    true_rows = tuple(
        _true_outcome(
            own_action,
            partner_action,
            physical_reward_law_drift=physical_reward_law_drift,
        )
        for own_action in range(_N_OWN_ACTIONS)
    )
    true_rewards = tuple(float(row[0]) for row in true_rows)
    true_physical = tuple(float(row[1]) for row in true_rows)
    true_comparator: dict[str, object] = {
        "arm": "evaluator_true_reward_ceiling",
        "causally_valid_preaction": False,
        "evaluator_only": True,
        "belief_source": "revealed_partner_action_and_evaluator_true_branch_law",
        "belief": list(actual_belief),
        "partner_action_consumed_before_action": True,
        "post_observation_consumed_before_action": False,
        "retrospective_inverse_output_consumed": False,
        "fixed_inverse_unavailable_fallback_used": False,
        "learned_predictor_snapshot_consumed": False,
        "frozen_preaction_surface_sha256": None,
        "frozen_complete_conditional_table_sha256": None,
        "joint_cell_evaluation_order": [],
        "reward_scores_by_own_action": list(true_rewards),
        "physical_one_predictions_by_own_action": list(true_physical),
        "chosen_action": _lower_action_argmax(cast(tuple[float, float], true_rewards)),
        "tie_rule": _LOWER_ACTION_TIE_RULE,
    }
    return {
        "actual_partner_conditional_model_ceiling": model_comparator,
        "evaluator_true_reward_ceiling": true_comparator,
    }


def _retrospective_inverse_distribution(
    state: _LearnedState,
    observation_pair: _InverseObservationPair,
    pseudocount: float,
) -> tuple[float, float]:
    """Form a post-outcome representation diagnostic without receiving its label."""

    if observation_pair.outcome_revealed is not True:
        raise ValueError("retrospective inverse prediction requires a revealed outcome")
    return _binary_probabilities(
        state.inverse_counts[
            _inverse_index(
                observation_pair.own_action,
                observation_pair.post_physical_bit,
                0,
            )
        ],
        state.inverse_counts[
            _inverse_index(
                observation_pair.own_action,
                observation_pair.post_physical_bit,
                1,
            )
        ],
        pseudocount,
    )


def _score_arm(
    proposal: Mapping[str, object],
    *,
    partner_action: int,
    counterfactual_outcomes: tuple[tuple[int, int], tuple[int, int]],
    surface: _FrozenPreactionSurface,
    learned_action: int,
    previous_action: int | None,
) -> dict[str, object]:
    chosen_action = cast(int, proposal["chosen_action"])
    reward, post_physical_bit = counterfactual_outcomes[chosen_action]
    true_best_reward = max(outcome[0] for outcome in counterfactual_outcomes)
    reward_scores = cast(list[float], proposal["reward_scores_by_own_action"])
    physical_scores = cast(
        list[float],
        proposal["physical_one_predictions_by_own_action"],
    )
    conditional_row = surface.complete_conditional_table[
        _conditional_index(chosen_action, partner_action)
    ]
    return {
        **proposal,
        "action_changed_from_previous_event": (
            False if previous_action is None else chosen_action != previous_action
        ),
        "action_differs_from_learned_behavior_marginal": chosen_action != learned_action,
        "realized_reward": reward,
        "realized_regret": true_best_reward - reward,
        "realized_post_physical_bit": post_physical_bit,
        "decision_reward_prediction": reward_scores[chosen_action],
        "decision_reward_prediction_squared_error": (
            reward_scores[chosen_action] - float(reward)
        )
        ** 2,
        "decision_physical_prediction": physical_scores[chosen_action],
        "decision_physical_prediction_squared_error": (
            physical_scores[chosen_action] - float(post_physical_bit)
        )
        ** 2,
        "learned_conditional_reward_prediction_for_actual_partner": conditional_row[0],
        "learned_conditional_reward_prediction_squared_error": (
            conditional_row[0] - float(reward)
        )
        ** 2,
        "learned_conditional_physical_prediction_for_actual_partner": conditional_row[1],
        "learned_conditional_physical_prediction_squared_error": (
            conditional_row[1] - float(post_physical_bit)
        )
        ** 2,
    }


def _branch_flags(branch: BranchName) -> tuple[bool, bool]:
    return (
        branch == "partner_policy_drift",
        branch == "physical_reward_law_drift",
    )


def _run_evaluation_branch(
    state: _LearnedState,
    config: FactorizedPreactionDecisionUtilityConfig,
    branch: BranchName,
) -> dict[str, object]:
    partner_policy_drift, physical_reward_law_drift = _branch_flags(branch)
    state_sha = _sha256(_state_data(state))
    raw_events: list[dict[str, object]] = []
    source_events: list[dict[str, object]] = []
    previous_actions: dict[str, int | None] = {arm: None for arm in ARM_NAMES}

    for step in range(config.evaluation_steps):
        cue = step % _N_CUES
        baseline_partner = _baseline_partner_action(step)
        partner_action = 1 - baseline_partner if partner_policy_drift else baseline_partner
        surface = _freeze_preaction_surface(state, cue, config.pseudocount)
        surface_data = surface.to_data()
        preaction = _form_preaction_decisions(surface)
        comparators = _form_post_reveal_comparators(
            surface,
            partner_action,
            physical_reward_law_drift=physical_reward_law_drift,
        )
        proposals = {**preaction, **comparators}
        counterfactual_outcomes = cast(
            tuple[tuple[int, int], tuple[int, int]],
            tuple(
                _true_outcome(
                    own_action,
                    partner_action,
                    physical_reward_law_drift=physical_reward_law_drift,
                )
                for own_action in range(_N_OWN_ACTIONS)
            ),
        )
        learned_action = cast(int, preaction["learned_behavior_marginal"]["chosen_action"])
        scored_arms: dict[str, object] = {}
        for arm in ARM_NAMES:
            scored = _score_arm(
                proposals[arm],
                partner_action=partner_action,
                counterfactual_outcomes=counterfactual_outcomes,
                surface=surface,
                learned_action=learned_action,
                previous_action=previous_actions[arm],
            )
            scored_arms[arm] = scored
            previous_actions[arm] = cast(int, scored["chosen_action"])

        inverse_diagnostics: list[dict[str, object]] = []
        for own_action, (_, post_physical_bit) in enumerate(counterfactual_outcomes):
            inverse_probabilities = _retrospective_inverse_distribution(
                state,
                _InverseObservationPair(
                    own_action=own_action,
                    post_physical_bit=post_physical_bit,
                ),
                config.pseudocount,
            )
            target = (1.0, 0.0) if partner_action == 0 else (0.0, 1.0)
            inverse_diagnostics.append(
                {
                    "own_action": own_action,
                    "post_physical_bit": post_physical_bit,
                    "probabilities": list(inverse_probabilities),
                    "partner_action_label_used_only_for_scoring": partner_action,
                    "nll": -math.log(inverse_probabilities[partner_action]),
                    "brier": sum(
                        (inverse_probabilities[index] - target[index]) ** 2
                        for index in range(_N_PARTNER_ACTIONS)
                    ),
                    "formed_after_outcome_reveal": True,
                    "fed_back_into_any_decision": False,
                }
            )

        behavior_target = (1.0, 0.0) if partner_action == 0 else (0.0, 1.0)
        source_event: dict[str, object] = {
            "step": step,
            "cue": cue,
            "partner_action": partner_action,
            "partner_policy_mapping_changed": partner_policy_drift,
            "physical_reward_law_changed": physical_reward_law_drift,
        }
        source_events.append(source_event)
        raw_events.append(
            {
                "step": step,
                "cue": cue,
                "decision_operation_order": list(_DECISION_ORDER),
                "frozen_preaction_surface_sha256": _sha256(surface_data),
                "frozen_complete_conditional_table_sha256": _sha256(
                    surface_data["complete_conditional_table"]
                ),
                "frozen_behavior_belief": list(surface.behavior_belief),
                "partner_action_revealed_after_preaction_decisions": partner_action,
                "behavior_nll": -math.log(surface.behavior_belief[partner_action]),
                "behavior_brier": sum(
                    (surface.behavior_belief[index] - behavior_target[index]) ** 2
                    for index in range(_N_PARTNER_ACTIONS)
                ),
                "counterfactual_true_outcomes_by_own_action": [
                    {
                        "own_action": own_action,
                        "reward": outcome[0],
                        "post_physical_bit": outcome[1],
                    }
                    for own_action, outcome in enumerate(counterfactual_outcomes)
                ],
                "arms": scored_arms,
                "retrospective_inverse_diagnostics_by_own_action": inverse_diagnostics,
                "frozen_state_unchanged_after_event": True,
            }
        )

    return {
        "branch": branch,
        "evaluator_only_intervention": {
            "partner_policy_mapping_changed": partner_policy_drift,
            "physical_reward_law_changed": physical_reward_law_drift,
        },
        "starts_from_state_sha256": state_sha,
        "ends_with_state_sha256": state_sha,
        "evaluation_updates": 0,
        "source_event_count": config.evaluation_steps,
        "source_sha256": _sha256(source_events),
        "raw_events": raw_events,
        "raw_trajectory_sha256": _sha256(raw_events),
        "raw_trajectory_canonical_nbytes": _canonical_nbytes(raw_events),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / float(len(values))


def _summarize_branch(
    branch: Mapping[str, object],
    state: _LearnedState,
    config: FactorizedPreactionDecisionUtilityConfig,
) -> dict[str, object]:
    events = cast(list[dict[str, object]], branch["raw_events"])
    intervention = cast(dict[str, object], branch["evaluator_only_intervention"])
    law_drift = cast(bool, intervention["physical_reward_law_changed"])
    behavior_nll = [cast(float, event["behavior_nll"]) for event in events]
    behavior_brier = [cast(float, event["behavior_brier"]) for event in events]
    inverse_rows = [
        row
        for event in events
        for row in cast(
            list[dict[str, object]],
            event["retrospective_inverse_diagnostics_by_own_action"],
        )
    ]
    arm_summaries: dict[str, object] = {}
    for arm in ARM_NAMES:
        rows = [
            cast(dict[str, object], cast(dict[str, object], event["arms"])[arm])
            for event in events
        ]
        chosen_actions = [cast(int, row["chosen_action"]) for row in rows]
        arm_summaries[arm] = {
            "event_count": len(rows),
            "chosen_actions": chosen_actions,
            "chosen_action_counts": [chosen_actions.count(0), chosen_actions.count(1)],
            "action_changes_from_previous_event": sum(
                int(cast(bool, row["action_changed_from_previous_event"])) for row in rows
            ),
            "actions_different_from_learned_behavior_marginal": sum(
                int(cast(bool, row["action_differs_from_learned_behavior_marginal"]))
                for row in rows
            ),
            "mean_realized_reward": _mean(
                [float(cast(int, row["realized_reward"])) for row in rows]
            ),
            "mean_realized_regret": _mean(
                [float(cast(int, row["realized_regret"])) for row in rows]
            ),
            "mean_decision_reward_prediction_squared_error": _mean(
                [cast(float, row["decision_reward_prediction_squared_error"]) for row in rows]
            ),
            "mean_decision_physical_prediction_squared_error": _mean(
                [cast(float, row["decision_physical_prediction_squared_error"]) for row in rows]
            ),
            "mean_learned_conditional_reward_prediction_squared_error": _mean(
                [
                    cast(float, row["learned_conditional_reward_prediction_squared_error"])
                    for row in rows
                ]
            ),
            "mean_learned_conditional_physical_prediction_squared_error": _mean(
                [
                    cast(float, row["learned_conditional_physical_prediction_squared_error"])
                    for row in rows
                ]
            ),
            "causally_valid_preaction": rows[0]["causally_valid_preaction"],
            "evaluator_only": rows[0]["evaluator_only"],
        }

    table_reward_errors: list[float] = []
    table_physical_errors: list[float] = []
    surface = _freeze_preaction_surface(state, 0, config.pseudocount)
    for own_action in range(_N_OWN_ACTIONS):
        for partner_action in range(_N_PARTNER_ACTIONS):
            reward, post = _true_outcome(
                own_action,
                partner_action,
                physical_reward_law_drift=law_drift,
            )
            row = surface.complete_conditional_table[
                _conditional_index(own_action, partner_action)
            ]
            table_reward_errors.append((row[0] - float(reward)) ** 2)
            table_physical_errors.append((row[1] - float(post)) ** 2)
    return {
        "behavior": {
            "mean_nll": _mean(behavior_nll),
            "mean_brier": _mean(behavior_brier),
        },
        "grounded_complete_conditional_table": {
            "reward_mse_over_all_own_partner_cells": _mean(table_reward_errors),
            "physical_mse_over_all_own_partner_cells": _mean(table_physical_errors),
        },
        "retrospective_inverse_over_all_counterfactual_own_actions": {
            "row_count": len(inverse_rows),
            "mean_nll": _mean([cast(float, row["nll"]) for row in inverse_rows]),
            "mean_brier": _mean([cast(float, row["brier"]) for row in inverse_rows]),
            "decision_feedback_count": sum(
                int(cast(bool, row["fed_back_into_any_decision"])) for row in inverse_rows
            ),
        },
        "arms": arm_summaries,
    }


def _summary_delta(
    left: Mapping[str, object],
    control: Mapping[str, object],
) -> dict[str, object]:
    left_behavior = cast(dict[str, object], left["behavior"])
    control_behavior = cast(dict[str, object], control["behavior"])
    left_table = cast(dict[str, object], left["grounded_complete_conditional_table"])
    control_table = cast(dict[str, object], control["grounded_complete_conditional_table"])
    left_inverse = cast(
        dict[str, object],
        left["retrospective_inverse_over_all_counterfactual_own_actions"],
    )
    control_inverse = cast(
        dict[str, object],
        control["retrospective_inverse_over_all_counterfactual_own_actions"],
    )
    left_arms = cast(dict[str, dict[str, object]], left["arms"])
    control_arms = cast(dict[str, dict[str, object]], control["arms"])
    return {
        "behavior_mean_nll": cast(float, left_behavior["mean_nll"])
        - cast(float, control_behavior["mean_nll"]),
        "behavior_mean_brier": cast(float, left_behavior["mean_brier"])
        - cast(float, control_behavior["mean_brier"]),
        "grounded_table_reward_mse": cast(
            float,
            left_table["reward_mse_over_all_own_partner_cells"],
        )
        - cast(float, control_table["reward_mse_over_all_own_partner_cells"]),
        "grounded_table_physical_mse": cast(
            float,
            left_table["physical_mse_over_all_own_partner_cells"],
        )
        - cast(float, control_table["physical_mse_over_all_own_partner_cells"]),
        "retrospective_inverse_mean_nll": cast(float, left_inverse["mean_nll"])
        - cast(float, control_inverse["mean_nll"]),
        "retrospective_inverse_mean_brier": cast(float, left_inverse["mean_brier"])
        - cast(float, control_inverse["mean_brier"]),
        "arms": {
            arm: {
                "chosen_action_count_deltas": [
                    cast(list[int], left_arms[arm]["chosen_action_counts"])[index]
                    - cast(list[int], control_arms[arm]["chosen_action_counts"])[index]
                    for index in range(_N_OWN_ACTIONS)
                ],
                "action_change_count": cast(
                    int,
                    left_arms[arm]["action_changes_from_previous_event"],
                )
                - cast(
                    int,
                    control_arms[arm]["action_changes_from_previous_event"],
                ),
                "mean_realized_reward": cast(
                    float,
                    left_arms[arm]["mean_realized_reward"],
                )
                - cast(float, control_arms[arm]["mean_realized_reward"]),
                "mean_realized_regret": cast(
                    float,
                    left_arms[arm]["mean_realized_regret"],
                )
                - cast(float, control_arms[arm]["mean_realized_regret"]),
                "decision_reward_prediction_mse": cast(
                    float,
                    left_arms[arm]["mean_decision_reward_prediction_squared_error"],
                )
                - cast(
                    float,
                    control_arms[arm]["mean_decision_reward_prediction_squared_error"],
                ),
                "decision_physical_prediction_mse": cast(
                    float,
                    left_arms[arm]["mean_decision_physical_prediction_squared_error"],
                )
                - cast(
                    float,
                    control_arms[arm]["mean_decision_physical_prediction_squared_error"],
                ),
            }
            for arm in ARM_NAMES
        },
    }


def _source_contract(config: FactorizedPreactionDecisionUtilityConfig) -> dict[str, object]:
    return {
        "generator_version": _SOURCE_GENERATOR_VERSION,
        "prefix_steps": config.prefix_steps,
        "evaluation_steps_per_branch": config.evaluation_steps,
        "branch_names_evaluator_only": list(BRANCH_NAMES),
        "baseline_partner_policy": "partner_matches_public_cue_in_three_of_four_balanced_blocks",
        "partner_policy_drift": "invert_every_baseline_partner_action",
        "prefix_own_action_schedule": "two-step-block alternation",
        "control_law": "reward=1-(own_xor_partner);physical=own_xor_partner",
        "drift_law": "reward=own_xor_partner;physical=1-(own_xor_partner)",
        "simultaneous_actions": True,
        "learner_branch_or_task_identifiers": False,
        "evaluation_updates": 0,
        "randomness_calls": RANDOMNESS_CALLS,
    }


def _work_summary(config: FactorizedPreactionDecisionUtilityConfig) -> dict[str, object]:
    evaluation_events = len(BRANCH_NAMES) * config.evaluation_steps
    contract: dict[str, object] = {
        "prefix_source_events_consumed": config.prefix_steps,
        "evaluation_source_events_consumed": evaluation_events,
        "total_source_events_consumed": config.total_source_events,
        "prefix_behavior_updates": config.prefix_steps,
        "prefix_grounded_model_updates": config.prefix_steps,
        "prefix_inverse_updates": config.prefix_steps,
        "evaluation_model_updates": 0,
        "preaction_surfaces_frozen": evaluation_events,
        "complete_conditional_cells_frozen": evaluation_events * _CONDITIONAL_CELLS,
        "causal_preaction_arm_decisions": evaluation_events * len(PREACTION_ARM_NAMES),
        "post_reveal_comparator_decisions": (
            evaluation_events * len(POST_REVEAL_COMPARATOR_NAMES)
        ),
        "own_action_scores_per_arm_per_event": _N_OWN_ACTIONS,
        "equal_own_action_score_count_across_arms": True,
        "learned_model_arms_per_event": 4,
        "learned_joint_cells_scored_per_model_arm": _CONDITIONAL_CELLS,
        "learned_joint_cell_score_evaluations": (
            evaluation_events * 4 * _CONDITIONAL_CELLS
        ),
        "evaluator_true_law_cells_scored_per_event": _N_OWN_ACTIONS,
        "own_actions_counterfactually_evaluated": evaluation_events * _N_OWN_ACTIONS,
        "retrospective_inverse_distributions": evaluation_events * _N_OWN_ACTIONS,
        "retrospective_inverse_decision_feedbacks": 0,
        "randomness_calls": RANDOMNESS_CALLS,
    }
    return {**contract, "work_contract_sha256": _sha256(contract)}


def _resource_summary(
    config: FactorizedPreactionDecisionUtilityConfig,
    branches: list[dict[str, object]],
) -> dict[str, object]:
    raw_nbytes = sum(cast(int, branch["raw_trajectory_canonical_nbytes"]) for branch in branches)
    if raw_nbytes > config.max_raw_trajectory_bytes:
        raise ValueError("raw trajectories exceed max_raw_trajectory_bytes")
    return {
        "behavior_count_cells": _BEHAVIOR_CELLS,
        "conditional_count_cells": _CONDITIONAL_CELLS,
        "conditional_reward_one_cells": _CONDITIONAL_CELLS,
        "conditional_physical_one_cells": _CONDITIONAL_CELLS,
        "retrospective_inverse_count_cells": _INVERSE_CELLS,
        "state_step_counter_scalars": 1,
        "persistent_integer_scalars": _PERSISTENT_INTEGER_SCALARS,
        "logical_bytes_per_integer_scalar": _LOGICAL_INTEGER_NBYTES,
        "logical_preallocated_state_nbytes": config.logical_state_nbytes,
        "state_size_fixed": True,
        "frozen_state_copies_for_evaluation_branches": len(BRANCH_NAMES),
        "raw_evaluation_events_retained": len(BRANCH_NAMES) * config.evaluation_steps,
        "raw_trajectory_canonical_nbytes": raw_nbytes,
        "raw_trajectory_memory_scaling": "O(branches*evaluation_steps*arms)",
        "persistent_state_scaling": "O(C*P + U*P + U*Z*P)",
        "per_event_decision_scaling": "O(arms*U*P)",
        "symbols": {
            "C": "public cue cardinality",
            "P": "partner action cardinality",
            "U": "own action cardinality",
            "Z": "post-physical observation cardinality",
        },
        "replay_capacity": 0,
        "randomness_calls": RANDOMNESS_CALLS,
        "python_object_and_allocator_bytes_included": False,
    }


def _seal_report(report: dict[str, object]) -> dict[str, object]:
    resource = cast(dict[str, object], report["resource"])
    resource["final_report_canonical_nbytes"] = 0
    while True:
        candidate = {
            **report,
            "integrity": {"report_without_integrity_sha256": _sha256(report)},
        }
        final_nbytes = _canonical_nbytes(candidate)
        if resource["final_report_canonical_nbytes"] == final_nbytes:
            return candidate
        resource["final_report_canonical_nbytes"] = final_nbytes


def run_factorized_preaction_decision_utility_development(
    config: FactorizedPreactionDecisionUtilityConfig | None = None,
) -> dict[str, object]:
    """Run one learned prefix and three immutable matched decision branches."""

    if config is None:
        cfg = FactorizedPreactionDecisionUtilityConfig()
    elif type(config) is not FactorizedPreactionDecisionUtilityConfig:
        raise TypeError("config must be an exact FactorizedPreactionDecisionUtilityConfig")
    else:
        cfg = config
    initial_state = _initial_state()
    frozen_state, prefix = _run_common_prefix(cfg)
    support = list(frozen_state.conditional_counts)
    if min(support) <= 0:
        raise AssertionError("common prefix must cover every own-partner conditional cell")
    branches = [
        _run_evaluation_branch(frozen_state, cfg, branch)
        for branch in BRANCH_NAMES
    ]
    summaries = {
        cast(str, branch["branch"]): _summarize_branch(branch, frozen_state, cfg)
        for branch in branches
    }
    control_summary = summaries["control"]
    deltas = {
        branch: _summary_delta(
            summaries[branch],
            control_summary,
        )
        for branch in ("partner_policy_drift", "physical_reward_law_drift")
    }
    source_contract = _source_contract(cfg)
    frozen_surface = _freeze_preaction_surface(frozen_state, 0, cfg.pseudocount)
    inverse_fallback_contract = {
        "candidate_signal": "retrospective_inverse_action_distribution",
        "available_before_simultaneous_action": False,
        "missing_preaction_input": "post_physical_observation",
        "fixed_fallback_belief": [0.5, 0.5],
        "fallback_uses_post_outcome": False,
        "later_inverse_output_reused_for_decision": False,
        "fallback_matches_uniform_belief_control": True,
    }
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
        "descriptive_claims_only": True,
        "config": cfg.to_config(),
        "provenance": {
            "module_relative_path": _MODULE_RELATIVE_PATH,
            "module_sha256": _source_sha256(),
            "implementation": "pure_python_standard_library",
        },
        "source_contract": source_contract,
        "source_contract_sha256": _sha256(source_contract),
        "decision_timing_contract": list(_TIMING_CONTRACT),
        "decision_operation_order_sha256": _sha256(list(_DECISION_ORDER)),
        "arm_contract": {
            "all_arms": list(ARM_NAMES),
            "causally_valid_preaction_arms": list(PREACTION_ARM_NAMES),
            "post_reveal_evaluator_comparators": list(POST_REVEAL_COMPARATOR_NAMES),
            "actual_partner_model_ceiling_scope": (
                "behavior_information_ceiling_under_frozen_learned_conditional_model"
            ),
            "true_reward_ceiling_scope": "evaluator_true_one_step_counterfactual_utility",
            "tie_rule": _LOWER_ACTION_TIE_RULE,
        },
        "inverse_action_misuse_contract": inverse_fallback_contract,
        "inverse_action_misuse_contract_sha256": _sha256(inverse_fallback_contract),
        "states": {
            "initial": _state_record(initial_state),
            "after_common_prefix_frozen": _state_record(frozen_state),
            "after_all_evaluation_branches": _state_record(frozen_state),
        },
        "conditional_support_receipt": {
            "own_partner_counts": support,
            "minimum_cell_count": min(support),
            "complete_before_evaluation": all(count > 0 for count in support),
            "support_cell_order": [
                [own_action, partner_action]
                for own_action in range(_N_OWN_ACTIONS)
                for partner_action in range(_N_PARTNER_ACTIONS)
            ],
        },
        "common_prefix": prefix,
        "frozen_model_snapshot": {
            "state_sha256": _sha256(_state_data(frozen_state)),
            "complete_conditional_table": frozen_surface.to_data()[
                "complete_conditional_table"
            ],
            "complete_conditional_table_sha256": _sha256(
                frozen_surface.to_data()["complete_conditional_table"]
            ),
        },
        "branches": branches,
        "branch_summaries": summaries,
        "branch_minus_control_deltas": deltas,
        "source_manifest_sha256": _sha256(
            [prefix["source_sha256"], *[branch["source_sha256"] for branch in branches]]
        ),
        "trajectory_manifest_sha256": _sha256(
            [branch["raw_trajectory_sha256"] for branch in branches]
        ),
        "work": _work_summary(cfg),
        "resource": _resource_summary(cfg, branches),
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


def validate_factorized_preaction_decision_utility_report(
    report: object,
) -> tuple[str, ...]:
    """Require exact canonical reconstruction, including raw trajectories."""

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
        claimed = integrity_mapping.get("report_without_integrity_sha256")
        if type(claimed) is not str or len(claimed) != 64:
            errors.append("integrity digest type or length differs")
        else:
            unhashed = dict(candidate)
            unhashed.pop("integrity")
            if claimed != _sha256(unhashed):
                errors.append("report integrity digest differs")
    try:
        config = FactorizedPreactionDecisionUtilityConfig.from_config(
            candidate.get("config")
        )
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        return tuple(errors)
    try:
        expected = run_factorized_preaction_decision_utility_development(config)
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
    "ARM_NAMES",
    "ARTIFACT_AUTHORITY",
    "ASSESSMENT_STATUS",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "BRANCH_NAMES",
    "CONFIG_SCHEMA",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "EVIDENCE_CLAIMED",
    "EVIDENCE_LEVEL",
    "FactorizedPreactionDecisionUtilityConfig",
    "OUTPUT_WRITES_ALLOWED",
    "POST_REVEAL_COMPARATOR_NAMES",
    "PREACTION_ARM_NAMES",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_DEFINED",
    "run_factorized_preaction_decision_utility_development",
    "validate_factorized_preaction_decision_utility_report",
]
