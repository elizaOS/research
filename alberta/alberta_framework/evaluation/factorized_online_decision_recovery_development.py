"""Development-only factorized online decision-recovery localizer.

The same deterministic, exhaustively supported common prefix used by the
static factorized decision probe is copied into four online arms for each of
three matched continuations.  Every arm freezes a genuinely pre-action
behavior belief and complete own-by-partner grounded reward/physical table,
then computes both update candidates after ordinary feedback.  Only routing
differs: both candidates, world only, behavior only, or neither.

A disclosed branch-independent periodic exploration instruction replaces the
proposed action on one event per four-event block.  Across each sixteen-event
cycle its four forced actions cover every own/partner cell under both partner
policy mappings.  Receipts keep the proposed counterfactual utility separate
from actually executed utility, so support acquisition is not attributed to
the planner.  Exploit-only, exploration-only, and full raw results are all
reported.  Evaluation branches are counterfactual state copies, not one
uninterrupted life.

This pure-standard-library L0 evaluator is descriptive and in-memory.  It has
no threshold, ranking, preferred arm, output writer, artifact authority,
benchmark authority, evidence claim, or scientific-promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast

DEVELOPMENT_SCHEMA: Final = "alberta.factorized-online-decision-recovery.development.v1"
CONFIG_SCHEMA: Final = "alberta.factorized-online-decision-recovery.config.v1"
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
HARD_MAX_PREFIX_STEPS: Final = 65_536
HARD_MAX_TOTAL_ARM_EVENTS: Final = 65_536
HARD_MAX_LOGICAL_STATE_BYTES: Final = 1_048_576
HARD_MAX_RAW_TRAJECTORY_BYTES: Final = 67_108_864
HARD_MAX_REPORT_BYTES: Final = 100_663_296

type BranchName = Literal[
    "control",
    "partner_policy_drift",
    "physical_reward_law_drift",
]
type ArmName = Literal[
    "both_adaptive",
    "behavior_frozen_world_adaptive",
    "behavior_adaptive_world_frozen",
    "both_frozen",
]

BRANCH_NAMES: Final[tuple[BranchName, ...]] = (
    "control",
    "partner_policy_drift",
    "physical_reward_law_drift",
)
ARM_NAMES: Final[tuple[ArmName, ...]] = (
    "both_adaptive",
    "behavior_frozen_world_adaptive",
    "behavior_adaptive_world_frozen",
    "both_frozen",
)
_ROUTING_MASKS: Final[dict[ArmName, tuple[bool, bool]]] = {
    "both_adaptive": (True, True),
    "behavior_frozen_world_adaptive": (False, True),
    "behavior_adaptive_world_frozen": (True, False),
    "both_frozen": (False, False),
}

_MODULE_RELATIVE_PATH: Final = (
    "alberta_framework/evaluation/factorized_online_decision_recovery_development.py"
)
_SOURCE_GENERATOR_VERSION: Final = "factorized-online-simultaneous-contextual-game-v1"
_N_CUES: Final = 2
_N_OWN_ACTIONS: Final = 2
_N_PARTNER_ACTIONS: Final = 2
_BEHAVIOR_CELLS: Final = _N_CUES * _N_PARTNER_ACTIONS
_WORLD_CELLS: Final = _N_OWN_ACTIONS * _N_PARTNER_ACTIONS
_STATE_INTEGER_SCALARS: Final = _BEHAVIOR_CELLS + 3 * _WORLD_CELLS
_LOGICAL_INTEGER_NBYTES: Final = 8
_EXPLORATION_OFFSETS: Final = (0, 1, 2, 0)
_EXPLORATION_FORCED_ACTIONS: Final = (0, 0, 1, 1)
_LOWER_ACTION_TIE_RULE: Final = "equal expected rewards select the lower integer own action"
_EVENT_ORDER: Final = (
    "observe_public_cue",
    "freeze_behavior_belief_and_complete_conditional_table",
    "form_exploit_action_proposal",
    "read_branch_independent_periodic_exploration_instruction",
    "commit_executed_action",
    "reveal_partner_action_and_outcome",
    "score_proposed_and_executed_counterfactual_utility",
    "compute_behavior_update_candidate",
    "compute_conditional_world_update_candidate",
    "route_candidates_under_fixed_arm_mask",
)
_TIMING_CONTRACT: Final = (
    "The pre-action API receives only frozen state, public cue, and pseudocount.",
    "The exploit proposal is frozen before exploration replacement, partner action, reward, or "
    "post-physical observation is available.",
    "The periodic exploration instruction depends only on the within-branch step and is identical "
    "for every branch and arm.",
    "A scheduled forced action replaces the proposal only for execution; both utilities remain "
    "separately scored on the same revealed event.",
    "Both update candidates are computed after feedback in every arm before either routing bit "
    "is read.",
    "Applied candidates affect only the next event; no update can alter its own pre-action "
    "decision.",
)
_LIMITATIONS: Final = (
    "one deterministic binary contextual game is not a seed, population, or robustness result",
    "branches and arms are counterfactual copies from one prefix, not one uninterrupted life",
    "the scripted partner does not learn, so this is not a two-learning-agent result",
    "independent contextual events do not establish continuous-life or long-horizon control",
    "finite count tables use exact cue/action routing rather than general feature discovery",
    "periodic forced exploration is externally scheduled and not learned from experience",
    "prefix-retaining cumulative counts adapt slowly and do not implement explicit forgetting",
    "reward and physical predictions are one-step binary means without temporal credit",
    "raw curves are descriptive; late-minus-early arithmetic has no acceptance meaning",
    "retained raw trajectories make report memory linear in branches, arms, and events",
    "logical work and bytes exclude Python objects, allocator peaks, and hardware operations",
    "no replay, planning horizon, safety, visual input, or population scaling claim is tested",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int or tuple/list aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[object, object], left)
        right_mapping = cast(dict[object, object], right)
        if len(left_mapping) != len(right_mapping):
            return False
        for right_key, right_value in right_mapping.items():
            matching_keys = [
                left_key
                for left_key in left_mapping
                if type(left_key) is type(right_key) and left_key == right_key
            ]
            if len(matching_keys) != 1:
                return False
            if not _exact_json_equal(left_mapping[matching_keys[0]], right_value):
                return False
        return True
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return left == right


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
class FactorizedOnlineDecisionRecoveryConfig:
    """Bounded deterministic online-recovery configuration."""

    run_id: str = "development.factorized-online-decision-recovery.v1"
    prefix_steps: int = 64
    evaluation_steps: int = 96
    summary_window: int = 16
    pseudocount: float = 1.0
    max_total_arm_events: int = 65_536
    max_logical_state_bytes: int = 16_384
    max_raw_trajectory_bytes: int = 8_388_608
    max_report_bytes: int = 12_582_912

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id or ".v" not in self.run_id:
            raise ValueError("run_id must be an exact non-empty versioned string")
        for name in (
            "prefix_steps",
            "evaluation_steps",
            "summary_window",
            "max_total_arm_events",
            "max_logical_state_bytes",
            "max_raw_trajectory_bytes",
            "max_report_bytes",
        ):
            _strict_positive_int(getattr(self, name), name=name)
        if self.prefix_steps < 16 or self.prefix_steps % 16 != 0:
            raise ValueError("prefix_steps must be a positive multiple of sixteen")
        if self.prefix_steps > HARD_MAX_PREFIX_STEPS:
            raise ValueError("prefix_steps exceeds the hard development bound")
        if self.evaluation_steps < 16 or self.evaluation_steps % 16 != 0:
            raise ValueError("evaluation_steps must be a positive multiple of sixteen")
        if self.summary_window > self.evaluation_steps:
            raise ValueError("summary_window must not exceed evaluation_steps")
        if self.summary_window < 16 or self.summary_window % 16 != 0:
            raise ValueError("summary_window must be a positive multiple of sixteen")
        _strict_positive_float(self.pseudocount, name="pseudocount")
        if self.pseudocount > 1_000_000.0:
            raise ValueError("pseudocount exceeds the bounded analytic contract")
        hard_caps = (
            ("max_total_arm_events", self.max_total_arm_events, HARD_MAX_TOTAL_ARM_EVENTS),
            (
                "max_logical_state_bytes",
                self.max_logical_state_bytes,
                HARD_MAX_LOGICAL_STATE_BYTES,
            ),
            (
                "max_raw_trajectory_bytes",
                self.max_raw_trajectory_bytes,
                HARD_MAX_RAW_TRAJECTORY_BYTES,
            ),
            ("max_report_bytes", self.max_report_bytes, HARD_MAX_REPORT_BYTES),
        )
        for name, value, hard_cap in hard_caps:
            if value > hard_cap:
                raise ValueError(f"{name} exceeds the hard development bound")
        if self.total_arm_events > self.max_total_arm_events:
            raise ValueError("arm work exceeds max_total_arm_events")
        if self.logical_state_nbytes > self.max_logical_state_bytes:
            raise ValueError("state exceeds max_logical_state_bytes")

    @property
    def total_arm_events(self) -> int:
        return len(BRANCH_NAMES) * len(ARM_NAMES) * self.evaluation_steps

    @property
    def logical_state_nbytes(self) -> int:
        return _STATE_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "type": type(self).__name__,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(cls, payload: object) -> FactorizedOnlineDecisionRecoveryConfig:
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
        if not _exact_json_equal(config.to_config(), mapping) or (
            _canonical_json(config.to_config()) != _canonical_json(mapping)
        ):
            raise ValueError("config canonical types or bytes differ")
        return config


@dataclasses.dataclass(frozen=True, slots=True)
class _OnlineState:
    behavior_counts: tuple[int, ...]
    conditional_counts: tuple[int, ...]
    reward_one_counts: tuple[int, ...]
    physical_one_counts: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenPreactionDecision:
    cue: int
    behavior_belief: tuple[float, float]
    conditional_table: tuple[tuple[float, float], ...]
    reward_scores: tuple[float, float]
    proposed_action: int
    state_sha256: str
    partner_action_revealed: bool = False
    outcome_revealed: bool = False

    def to_data(self) -> dict[str, object]:
        return {
            "cue": self.cue,
            "behavior_belief": list(self.behavior_belief),
            "conditional_table": [list(row) for row in self.conditional_table],
            "reward_scores": list(self.reward_scores),
            "proposed_action": self.proposed_action,
            "state_sha256": self.state_sha256,
            "partner_action_revealed": self.partner_action_revealed,
            "outcome_revealed": self.outcome_revealed,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _BehaviorCandidate:
    cue: int
    partner_action: int
    count_increment: int = 1

    def to_data(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class _WorldCandidate:
    own_action: int
    partner_action: int
    reward: int
    post_physical_bit: int
    count_increment: int = 1

    def to_data(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def _initial_state() -> _OnlineState:
    return _OnlineState(
        behavior_counts=(0,) * _BEHAVIOR_CELLS,
        conditional_counts=(0,) * _WORLD_CELLS,
        reward_one_counts=(0,) * _WORLD_CELLS,
        physical_one_counts=(0,) * _WORLD_CELLS,
    )


def _state_data(state: _OnlineState) -> dict[str, object]:
    return {
        "behavior_counts": list(state.behavior_counts),
        "conditional_counts": list(state.conditional_counts),
        "reward_one_counts": list(state.reward_one_counts),
        "physical_one_counts": list(state.physical_one_counts),
    }


def _state_record(state: _OnlineState) -> dict[str, object]:
    content = _state_data(state)
    return {
        "content": content,
        "content_sha256": _sha256(content),
        "canonical_nbytes": _canonical_nbytes(content),
        "logical_preallocated_int64_nbytes": (
            _STATE_INTEGER_SCALARS * _LOGICAL_INTEGER_NBYTES
        ),
    }


def _replace(value: tuple[int, ...], index: int, replacement: int) -> tuple[int, ...]:
    mutable = list(value)
    mutable[index] = replacement
    return tuple(mutable)


def _behavior_index(cue: int, partner_action: int) -> int:
    return cue * _N_PARTNER_ACTIONS + partner_action


def _world_index(own_action: int, partner_action: int) -> int:
    return own_action * _N_PARTNER_ACTIONS + partner_action


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
    post_physical_bit = own_action ^ partner_action ^ int(physical_reward_law_drift)
    return 1 - post_physical_bit, post_physical_bit


def _prefix_state(
    config: FactorizedOnlineDecisionRecoveryConfig,
) -> tuple[_OnlineState, dict[str, object]]:
    state = _initial_state()
    events: list[dict[str, int]] = []
    for step in range(config.prefix_steps):
        cue = step % _N_CUES
        own_action = (step // 2) % _N_OWN_ACTIONS
        partner_action = _baseline_partner_action(step)
        reward, post = _true_outcome(
            own_action,
            partner_action,
            physical_reward_law_drift=False,
        )
        behavior = _BehaviorCandidate(cue=cue, partner_action=partner_action)
        world = _WorldCandidate(
            own_action=own_action,
            partner_action=partner_action,
            reward=reward,
            post_physical_bit=post,
        )
        state = _apply_candidates(
            state,
            behavior,
            world,
            apply_behavior=True,
            apply_world=True,
        )
        events.append(
            {
                "step": step,
                "cue": cue,
                "own_action": own_action,
                "partner_action": partner_action,
                "reward": reward,
                "post_physical_bit": post,
            }
        )
    return state, {
        "source_event_count": config.prefix_steps,
        "source_sha256": _sha256(events),
        "source_canonical_nbytes": _canonical_nbytes(events),
        "passes_over_source": 1,
        "behavior_updates": config.prefix_steps,
        "world_updates": config.prefix_steps,
        "source_events_retained_in_report": 0,
    }


def _freeze_preaction_decision(
    state: _OnlineState,
    cue: int,
    pseudocount: float,
) -> _FrozenPreactionDecision:
    """Freeze one causal action proposal without branch, action label, or outcome."""

    behavior_belief = _binary_probabilities(
        state.behavior_counts[_behavior_index(cue, 0)],
        state.behavior_counts[_behavior_index(cue, 1)],
        pseudocount,
    )
    table: list[tuple[float, float]] = []
    for own_action in range(_N_OWN_ACTIONS):
        for partner_action in range(_N_PARTNER_ACTIONS):
            index = _world_index(own_action, partner_action)
            denominator = float(state.conditional_counts[index]) + 2.0 * pseudocount
            table.append(
                (
                    (float(state.reward_one_counts[index]) + pseudocount) / denominator,
                    (float(state.physical_one_counts[index]) + pseudocount) / denominator,
                )
            )
    reward_scores = tuple(
        sum(
            behavior_belief[partner_action]
            * table[_world_index(own_action, partner_action)][0]
            for partner_action in range(_N_PARTNER_ACTIONS)
        )
        for own_action in range(_N_OWN_ACTIONS)
    )
    scores = cast(tuple[float, float], reward_scores)
    proposed_action = 0 if scores[0] >= scores[1] else 1
    return _FrozenPreactionDecision(
        cue=cue,
        behavior_belief=behavior_belief,
        conditional_table=tuple(table),
        reward_scores=scores,
        proposed_action=proposed_action,
        state_sha256=_sha256(_state_data(state)),
    )


def _exploration_instruction(step: int) -> dict[str, object]:
    """Return the fixed support instruction using no branch, arm, or learner state."""

    block = step // 4
    within_cycle_block = block % 4
    scheduled = step % 4 == _EXPLORATION_OFFSETS[within_cycle_block]
    forced_action = _EXPLORATION_FORCED_ACTIONS[within_cycle_block]
    return {
        "scheduled": scheduled,
        "forced_action": forced_action if scheduled else None,
        "cycle_length_events": 16,
        "events_per_four_event_block": 1,
        "within_cycle_block": within_cycle_block,
        "depends_on_branch": False,
        "depends_on_arm": False,
        "depends_on_prediction": False,
        "randomness_calls": 0,
    }


def _compute_behavior_candidate(cue: int, partner_action: int) -> _BehaviorCandidate:
    return _BehaviorCandidate(cue=cue, partner_action=partner_action)


def _compute_world_candidate(
    executed_action: int,
    partner_action: int,
    reward: int,
    post_physical_bit: int,
) -> _WorldCandidate:
    return _WorldCandidate(
        own_action=executed_action,
        partner_action=partner_action,
        reward=reward,
        post_physical_bit=post_physical_bit,
    )


def _apply_candidates(
    state: _OnlineState,
    behavior: _BehaviorCandidate,
    world: _WorldCandidate,
    *,
    apply_behavior: bool,
    apply_world: bool,
) -> _OnlineState:
    behavior_counts = state.behavior_counts
    conditional_counts = state.conditional_counts
    reward_one_counts = state.reward_one_counts
    physical_one_counts = state.physical_one_counts
    if apply_behavior:
        index = _behavior_index(behavior.cue, behavior.partner_action)
        behavior_counts = _replace(
            behavior_counts,
            index,
            behavior_counts[index] + behavior.count_increment,
        )
    if apply_world:
        index = _world_index(world.own_action, world.partner_action)
        conditional_counts = _replace(
            conditional_counts,
            index,
            conditional_counts[index] + world.count_increment,
        )
        reward_one_counts = _replace(
            reward_one_counts,
            index,
            reward_one_counts[index] + world.reward,
        )
        physical_one_counts = _replace(
            physical_one_counts,
            index,
            physical_one_counts[index] + world.post_physical_bit,
        )
    return _OnlineState(
        behavior_counts=behavior_counts,
        conditional_counts=conditional_counts,
        reward_one_counts=reward_one_counts,
        physical_one_counts=physical_one_counts,
    )


def _table_errors(
    frozen: _FrozenPreactionDecision,
    *,
    physical_reward_law_drift: bool,
) -> tuple[float, float]:
    reward_errors: list[float] = []
    physical_errors: list[float] = []
    for own_action in range(_N_OWN_ACTIONS):
        for partner_action in range(_N_PARTNER_ACTIONS):
            true_reward, true_post = _true_outcome(
                own_action,
                partner_action,
                physical_reward_law_drift=physical_reward_law_drift,
            )
            prediction = frozen.conditional_table[_world_index(own_action, partner_action)]
            reward_errors.append((prediction[0] - float(true_reward)) ** 2)
            physical_errors.append((prediction[1] - float(true_post)) ** 2)
    return (
        sum(reward_errors) / float(len(reward_errors)),
        sum(physical_errors) / float(len(physical_errors)),
    )


def _branch_flags(branch: BranchName) -> tuple[bool, bool]:
    return (
        branch == "partner_policy_drift",
        branch == "physical_reward_law_drift",
    )


def _run_arm(
    initial_state: _OnlineState,
    config: FactorizedOnlineDecisionRecoveryConfig,
    branch: BranchName,
    arm: ArmName,
) -> dict[str, object]:
    policy_drift, law_drift = _branch_flags(branch)
    state = initial_state
    raw_events: list[dict[str, object]] = []
    behavior_candidates = 0
    world_candidates = 0
    behavior_updates = 0
    world_updates = 0
    candidate_support = [0] * _WORLD_CELLS
    applied_world_support = [0] * _WORLD_CELLS
    exploration_candidate_support = [0] * _WORLD_CELLS
    exploration_applied_world_support = [0] * _WORLD_CELLS

    for step in range(config.evaluation_steps):
        cue = step % _N_CUES
        frozen = _freeze_preaction_decision(state, cue, config.pseudocount)
        exploration = _exploration_instruction(step)
        scheduled = cast(bool, exploration["scheduled"])
        forced_action = exploration["forced_action"]
        executed_action = (
            cast(int, forced_action) if scheduled else frozen.proposed_action
        )
        baseline_partner = _baseline_partner_action(step)
        partner_action = 1 - baseline_partner if policy_drift else baseline_partner
        outcomes = cast(
            tuple[tuple[int, int], tuple[int, int]],
            tuple(
                _true_outcome(
                    own_action,
                    partner_action,
                    physical_reward_law_drift=law_drift,
                )
                for own_action in range(_N_OWN_ACTIONS)
            ),
        )
        proposed_reward, proposed_post = outcomes[frozen.proposed_action]
        executed_reward, executed_post = outcomes[executed_action]
        behavior_candidate = _compute_behavior_candidate(cue, partner_action)
        world_candidate = _compute_world_candidate(
            executed_action,
            partner_action,
            executed_reward,
            executed_post,
        )
        behavior_candidates += 1
        world_candidates += 1
        apply_behavior, apply_world = _ROUTING_MASKS[arm]
        support_index = _world_index(executed_action, partner_action)
        candidate_support[support_index] += 1
        if scheduled:
            exploration_candidate_support[support_index] += 1
        pre_state_sha = _sha256(_state_data(state))
        updated = _apply_candidates(
            state,
            behavior_candidate,
            world_candidate,
            apply_behavior=apply_behavior,
            apply_world=apply_world,
        )
        if apply_behavior:
            behavior_updates += 1
        if apply_world:
            world_updates += 1
            applied_world_support[support_index] += 1
            if scheduled:
                exploration_applied_world_support[support_index] += 1
        table_reward_mse, table_physical_mse = _table_errors(
            frozen,
            physical_reward_law_drift=law_drift,
        )
        target = (1.0, 0.0) if partner_action == 0 else (0.0, 1.0)
        selected_prediction = frozen.conditional_table[
            _world_index(executed_action, partner_action)
        ]
        raw_events.append(
            {
                "step": step,
                "event_operation_order": list(_EVENT_ORDER),
                "cue": cue,
                "frozen_preaction": frozen.to_data(),
                "exploration": {
                    **exploration,
                    "proposed_action": frozen.proposed_action,
                    "executed_action": executed_action,
                    "forced_action_replaced_proposal": (
                        scheduled and executed_action != frozen.proposed_action
                    ),
                },
                "partner_action_revealed_after_action": partner_action,
                "counterfactual_outcomes_by_own_action": [
                    {
                        "own_action": own_action,
                        "reward": outcome[0],
                        "post_physical_bit": outcome[1],
                    }
                    for own_action, outcome in enumerate(outcomes)
                ],
                "proposed_counterfactual_reward": proposed_reward,
                "proposed_counterfactual_regret": 1 - proposed_reward,
                "proposed_counterfactual_post_physical_bit": proposed_post,
                "executed_reward": executed_reward,
                "executed_regret": 1 - executed_reward,
                "executed_post_physical_bit": executed_post,
                "behavior_nll": -math.log(frozen.behavior_belief[partner_action]),
                "behavior_brier": sum(
                    (frozen.behavior_belief[index] - target[index]) ** 2
                    for index in range(_N_PARTNER_ACTIONS)
                ),
                "complete_table_reward_mse": table_reward_mse,
                "complete_table_physical_mse": table_physical_mse,
                "executed_conditional_reward_squared_error": (
                    selected_prediction[0] - float(executed_reward)
                )
                ** 2,
                "executed_conditional_physical_squared_error": (
                    selected_prediction[1] - float(executed_post)
                )
                ** 2,
                "behavior_candidate": {
                    "content": behavior_candidate.to_data(),
                    "content_sha256": _sha256(behavior_candidate.to_data()),
                    "computed": True,
                    "applied": apply_behavior,
                },
                "world_candidate": {
                    "content": world_candidate.to_data(),
                    "content_sha256": _sha256(world_candidate.to_data()),
                    "computed": True,
                    "applied": apply_world,
                },
                "routing_mask": {
                    "apply_behavior": apply_behavior,
                    "apply_world": apply_world,
                },
                "pre_state_sha256": pre_state_sha,
                "post_state_sha256": _sha256(_state_data(updated)),
            }
        )
        state = updated

    apply_behavior, apply_world = _ROUTING_MASKS[arm]
    return {
        "arm": arm,
        "routing_mask": {
            "apply_behavior": apply_behavior,
            "apply_world": apply_world,
        },
        "initial_state_sha256": _sha256(_state_data(initial_state)),
        "final_state": _state_record(state),
        "candidate_counts": {
            "behavior": behavior_candidates,
            "world": world_candidates,
        },
        "applied_update_counts": {
            "behavior": behavior_updates,
            "world": world_updates,
        },
        "world_candidate_support_counts": candidate_support,
        "world_applied_support_counts": applied_world_support,
        "exploration_world_candidate_support_counts": exploration_candidate_support,
        "exploration_world_applied_support_counts": exploration_applied_world_support,
        "raw_events": raw_events,
        "raw_trajectory_sha256": _sha256(raw_events),
        "raw_trajectory_canonical_nbytes": _canonical_nbytes(raw_events),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / float(len(values))


def _metric_means(events: list[dict[str, object]]) -> dict[str, float]:
    return {
        "behavior_nll": _mean([cast(float, event["behavior_nll"]) for event in events]),
        "behavior_brier": _mean([cast(float, event["behavior_brier"]) for event in events]),
        "complete_table_reward_mse": _mean(
            [cast(float, event["complete_table_reward_mse"]) for event in events]
        ),
        "complete_table_physical_mse": _mean(
            [cast(float, event["complete_table_physical_mse"]) for event in events]
        ),
        "executed_reward": _mean(
            [float(cast(int, event["executed_reward"])) for event in events]
        ),
        "executed_regret": _mean(
            [float(cast(int, event["executed_regret"])) for event in events]
        ),
        "proposed_counterfactual_reward": _mean(
            [float(cast(int, event["proposed_counterfactual_reward"])) for event in events]
        ),
        "proposed_counterfactual_regret": _mean(
            [float(cast(int, event["proposed_counterfactual_regret"])) for event in events]
        ),
        "executed_conditional_reward_squared_error": _mean(
            [
                cast(float, event["executed_conditional_reward_squared_error"])
                for event in events
            ]
        ),
        "executed_conditional_physical_squared_error": _mean(
            [
                cast(float, event["executed_conditional_physical_squared_error"])
                for event in events
            ]
        ),
    }


def _subtract_metrics(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {field: left[field] - right[field] for field in left}


def _summarize_arm(
    arm: Mapping[str, object],
    config: FactorizedOnlineDecisionRecoveryConfig,
) -> dict[str, object]:
    events = cast(list[dict[str, object]], arm["raw_events"])
    exploration_events = [
        event
        for event in events
        if cast(bool, cast(dict[str, object], event["exploration"])["scheduled"])
    ]
    exploit_events = [
        event
        for event in events
        if not cast(bool, cast(dict[str, object], event["exploration"])["scheduled"])
    ]
    early = events[: config.summary_window]
    late = events[-config.summary_window :]
    early_metrics = _metric_means(early)
    late_metrics = _metric_means(late)
    proposed_actions = [
        cast(int, cast(dict[str, object], event["frozen_preaction"])["proposed_action"])
        for event in events
    ]
    executed_actions = [
        cast(int, cast(dict[str, object], event["exploration"])["executed_action"])
        for event in events
    ]
    return {
        "event_count": len(events),
        "exploration_event_count": len(exploration_events),
        "exploit_event_count": len(exploit_events),
        "proposed_actions": proposed_actions,
        "executed_actions": executed_actions,
        "proposed_action_counts": [proposed_actions.count(0), proposed_actions.count(1)],
        "executed_action_counts": [executed_actions.count(0), executed_actions.count(1)],
        "forced_action_replacement_count": sum(
            int(
                cast(
                    bool,
                    cast(dict[str, object], event["exploration"])[
                        "forced_action_replaced_proposal"
                    ],
                )
            )
            for event in exploration_events
        ),
        "full": _metric_means(events),
        "exploration_only": _metric_means(exploration_events),
        "exploit_only": _metric_means(exploit_events),
        "early_window": early_metrics,
        "late_window": late_metrics,
        "late_minus_early": _subtract_metrics(late_metrics, early_metrics),
        "raw_curves": {
            "behavior_nll": [cast(float, event["behavior_nll"]) for event in events],
            "behavior_brier": [cast(float, event["behavior_brier"]) for event in events],
            "complete_table_reward_mse": [
                cast(float, event["complete_table_reward_mse"]) for event in events
            ],
            "complete_table_physical_mse": [
                cast(float, event["complete_table_physical_mse"]) for event in events
            ],
            "executed_reward": [cast(int, event["executed_reward"]) for event in events],
            "executed_regret": [cast(int, event["executed_regret"]) for event in events],
            "proposed_counterfactual_reward": [
                cast(int, event["proposed_counterfactual_reward"]) for event in events
            ],
            "proposed_counterfactual_regret": [
                cast(int, event["proposed_counterfactual_regret"]) for event in events
            ],
        },
        "candidate_counts": arm["candidate_counts"],
        "applied_update_counts": arm["applied_update_counts"],
        "world_candidate_support_counts": arm["world_candidate_support_counts"],
        "world_applied_support_counts": arm["world_applied_support_counts"],
        "exploration_world_candidate_support_counts": arm[
            "exploration_world_candidate_support_counts"
        ],
        "exploration_world_applied_support_counts": arm[
            "exploration_world_applied_support_counts"
        ],
    }


def _summary_delta(left: Mapping[str, object], control: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for scope in ("full", "exploration_only", "exploit_only", "early_window", "late_window"):
        result[scope] = _subtract_metrics(
            cast(dict[str, float], left[scope]),
            cast(dict[str, float], control[scope]),
        )
    result["proposed_action_count_deltas"] = [
        cast(list[int], left["proposed_action_counts"])[index]
        - cast(list[int], control["proposed_action_counts"])[index]
        for index in range(_N_OWN_ACTIONS)
    ]
    result["executed_action_count_deltas"] = [
        cast(list[int], left["executed_action_counts"])[index]
        - cast(list[int], control["executed_action_counts"])[index]
        for index in range(_N_OWN_ACTIONS)
    ]
    return result


def _source_contract(config: FactorizedOnlineDecisionRecoveryConfig) -> dict[str, object]:
    return {
        "generator_version": _SOURCE_GENERATOR_VERSION,
        "prefix_steps": config.prefix_steps,
        "evaluation_steps_per_arm": config.evaluation_steps,
        "branch_names_evaluator_only": list(BRANCH_NAMES),
        "arm_routing_masks": {
            arm: {"apply_behavior": mask[0], "apply_world": mask[1]}
            for arm, mask in _ROUTING_MASKS.items()
        },
        "baseline_partner_policy": "partner_matches_cue_in_three_of_four_balanced_blocks",
        "partner_policy_drift": "invert_every_baseline_partner_action",
        "control_law": "reward=1-(own_xor_partner);physical=own_xor_partner",
        "drift_law": "reward=own_xor_partner;physical=1-(own_xor_partner)",
        "exploration_offsets_by_four_event_block_in_sixteen_event_cycle": list(
            _EXPLORATION_OFFSETS
        ),
        "exploration_forced_actions_by_block": list(_EXPLORATION_FORCED_ACTIONS),
        "learner_branch_or_task_identifiers": False,
        "randomness_calls": RANDOMNESS_CALLS,
    }


def _work_summary(config: FactorizedOnlineDecisionRecoveryConfig) -> dict[str, object]:
    arm_events = config.total_arm_events
    total_events = config.prefix_steps + arm_events
    arm_candidates_applied = len(BRANCH_NAMES) * 2 * config.evaluation_steps
    contract: dict[str, object] = {
        "prefix_events": config.prefix_steps,
        "arm_events": arm_events,
        "total_events_evaluated": total_events,
        "arm_preaction_decisions": arm_events,
        "arm_complete_world_cells_frozen": arm_events * _WORLD_CELLS,
        "prefix_behavior_candidates_computed": config.prefix_steps,
        "prefix_world_candidates_computed": config.prefix_steps,
        "arm_behavior_candidates_computed": arm_events,
        "arm_world_candidates_computed": arm_events,
        "total_behavior_candidates_computed": total_events,
        "total_world_candidates_computed": total_events,
        "prefix_behavior_candidates_applied": config.prefix_steps,
        "prefix_world_candidates_applied": config.prefix_steps,
        "arm_behavior_candidates_applied": arm_candidates_applied,
        "arm_world_candidates_applied": arm_candidates_applied,
        "total_behavior_candidates_applied": config.prefix_steps + arm_candidates_applied,
        "total_world_candidates_applied": config.prefix_steps + arm_candidates_applied,
        "counterfactual_own_actions_scored": arm_events * _N_OWN_ACTIONS,
        "scheduled_exploration_events": arm_events // 4,
        "randomness_calls": RANDOMNESS_CALLS,
    }
    return {**contract, "work_contract_sha256": _sha256(contract)}


def _resource_summary(
    config: FactorizedOnlineDecisionRecoveryConfig,
    branches: list[dict[str, object]],
) -> dict[str, object]:
    arm_records = [
        arm
        for branch in branches
        for arm in cast(list[dict[str, object]], branch["arms"])
    ]
    raw_nbytes = sum(cast(int, arm["raw_trajectory_canonical_nbytes"]) for arm in arm_records)
    if raw_nbytes > config.max_raw_trajectory_bytes:
        raise ValueError("raw trajectories exceed max_raw_trajectory_bytes")
    return {
        "behavior_count_cells": _BEHAVIOR_CELLS,
        "conditional_count_cells": _WORLD_CELLS,
        "reward_one_count_cells": _WORLD_CELLS,
        "physical_one_count_cells": _WORLD_CELLS,
        "persistent_integer_scalars_per_arm": _STATE_INTEGER_SCALARS,
        "logical_bytes_per_integer_scalar": _LOGICAL_INTEGER_NBYTES,
        "logical_state_nbytes_per_arm": config.logical_state_nbytes,
        "counterfactual_arm_state_copies": len(BRANCH_NAMES) * len(ARM_NAMES),
        "state_size_fixed": True,
        "raw_arm_events_retained": config.total_arm_events,
        "raw_trajectory_canonical_nbytes": raw_nbytes,
        "persistent_state_scaling_per_arm": "O(C*P + U*P)",
        "per_event_prediction_and_candidate_scaling": "O(U*P)",
        "raw_report_memory_scaling": "O(branches*arms*evaluation_steps*U*P)",
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
        size = _canonical_nbytes(candidate)
        if resource["final_report_canonical_nbytes"] == size:
            return candidate
        resource["final_report_canonical_nbytes"] = size


def run_factorized_online_decision_recovery_development(
    config: FactorizedOnlineDecisionRecoveryConfig | None = None,
) -> dict[str, object]:
    """Run a common prefix and all factor-by-routing counterfactual copies."""

    if config is None:
        cfg = FactorizedOnlineDecisionRecoveryConfig()
    elif type(config) is not FactorizedOnlineDecisionRecoveryConfig:
        raise TypeError("config must be an exact FactorizedOnlineDecisionRecoveryConfig")
    else:
        cfg = FactorizedOnlineDecisionRecoveryConfig.from_config(config.to_config())
    initial = _initial_state()
    prefix_state, prefix = _prefix_state(cfg)
    if min(prefix_state.conditional_counts) <= 0:
        raise AssertionError("common prefix must support every own-partner world cell")
    branches: list[dict[str, object]] = []
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for branch in BRANCH_NAMES:
        policy_drift, law_drift = _branch_flags(branch)
        arms = [_run_arm(prefix_state, cfg, branch, arm) for arm in ARM_NAMES]
        branch_summary: dict[str, dict[str, object]] = {
            cast(str, arm["arm"]): _summarize_arm(arm, cfg)
            for arm in arms
        }
        summaries[branch] = branch_summary
        branches.append(
            {
                "branch": branch,
                "evaluator_only_intervention": {
                    "partner_policy_mapping_changed": policy_drift,
                    "physical_reward_law_changed": law_drift,
                },
                "common_prefix_state_sha256": _sha256(_state_data(prefix_state)),
                "source_sha256": _sha256(
                    [
                        {
                            "step": step,
                            "cue": step % _N_CUES,
                            "partner_action": (
                                1 - _baseline_partner_action(step)
                                if policy_drift
                                else _baseline_partner_action(step)
                            ),
                            "physical_reward_law_changed": law_drift,
                        }
                        for step in range(cfg.evaluation_steps)
                    ]
                ),
                "arms": arms,
            }
        )
    control = summaries["control"]
    deltas = {
        branch: {
            arm: _summary_delta(summaries[branch][arm], control[arm])
            for arm in ARM_NAMES
        }
        for branch in ("partner_policy_drift", "physical_reward_law_drift")
    }
    source_contract = _source_contract(cfg)
    exploration_contract = {
        "cycle_length_events": 16,
        "one_scheduled_event_per_four_event_block": True,
        "offsets_by_block": list(_EXPLORATION_OFFSETS),
        "forced_actions_by_block": list(_EXPLORATION_FORCED_ACTIONS),
        "replacement_semantics": "forced action replaces proposal only for execution",
        "proposed_counterfactual_utility_retained": True,
        "executed_utility_retained": True,
        "joint_candidate_support_per_cycle_under_each_partner_mapping": True,
        "randomness_calls": 0,
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
        "event_timing_contract": list(_TIMING_CONTRACT),
        "event_operation_order_sha256": _sha256(list(_EVENT_ORDER)),
        "exploration_contract": exploration_contract,
        "exploration_contract_sha256": _sha256(exploration_contract),
        "routing_contract": {
            arm: {
                "apply_behavior": _ROUTING_MASKS[arm][0],
                "apply_world": _ROUTING_MASKS[arm][1],
                "behavior_candidate_computed_every_event": True,
                "world_candidate_computed_every_event": True,
            }
            for arm in ARM_NAMES
        },
        "states": {
            "initial": _state_record(initial),
            "after_common_prefix": _state_record(prefix_state),
        },
        "common_prefix": prefix,
        "common_prefix_support": {
            "own_partner_counts": list(prefix_state.conditional_counts),
            "complete": all(count > 0 for count in prefix_state.conditional_counts),
        },
        "branches": branches,
        "branch_summaries": summaries,
        "branch_minus_control_deltas": deltas,
        "localization_receipt": {
            "policy_factor_metric": "prequential_behavior_nll_and_brier",
            "policy_factor_routed_arms": [
                "both_adaptive",
                "behavior_adaptive_world_frozen",
            ],
            "policy_factor_frozen_arms": [
                "behavior_frozen_world_adaptive",
                "both_frozen",
            ],
            "law_factor_metric": "complete_conditional_reward_and_physical_mse",
            "law_factor_routed_arms": [
                "both_adaptive",
                "behavior_frozen_world_adaptive",
            ],
            "law_factor_frozen_arms": [
                "behavior_adaptive_world_frozen",
                "both_frozen",
            ],
            "outcomes_are_raw_descriptive_values_not_gates": True,
        },
        "source_manifest_sha256": _sha256(
            [prefix["source_sha256"], *[branch["source_sha256"] for branch in branches]]
        ),
        "trajectory_manifest_sha256": _sha256(
            [
                arm["raw_trajectory_sha256"]
                for branch in branches
                for arm in cast(list[dict[str, object]], branch["arms"])
            ]
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


def validate_factorized_online_decision_recovery_report(report: object) -> tuple[str, ...]:
    """Require exact canonical reconstruction of every state and raw curve."""

    if type(report) is not dict:
        return ("report must be an exact JSON object",)
    candidate = cast(dict[str, object], report)
    errors: list[str] = []
    integrity = candidate.get("integrity")
    if type(integrity) is not dict:
        errors.append("integrity must be an exact JSON object")
    else:
        mapping = cast(dict[str, object], integrity)
        if set(mapping) != {"report_without_integrity_sha256"}:
            errors.append("integrity fields differ")
        claimed = mapping.get("report_without_integrity_sha256")
        if type(claimed) is not str or len(claimed) != 64:
            errors.append("integrity digest type or length differs")
        else:
            unhashed = dict(candidate)
            unhashed.pop("integrity")
            if claimed != _sha256(unhashed):
                errors.append("report integrity digest differs")
    try:
        config = FactorizedOnlineDecisionRecoveryConfig.from_config(candidate.get("config"))
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        return tuple(errors)
    try:
        expected = run_factorized_online_decision_recovery_development(config)
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"report reconstruction failed: {error}")
        return tuple(errors)
    if not _exact_json_equal(candidate, expected) or (
        _canonical_json(candidate) != _canonical_json(expected)
    ):
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
    "FactorizedOnlineDecisionRecoveryConfig",
    "OUTPUT_WRITES_ALLOWED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_DEFINED",
    "run_factorized_online_decision_recovery_development",
    "validate_factorized_online_decision_recovery_report",
]
