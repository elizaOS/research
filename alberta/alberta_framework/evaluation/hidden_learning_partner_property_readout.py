"""Threshold-free Alberta-property readout for the hidden co-learning dyad.

The hidden-learning-partner development kernel already records one
uninterrupted, predict-before-update trace for two online learning roles, a
beneficiary-action model, and a one-step grounded world model.  This module
only reconstructs descriptive measurements from that existing run.  It does
not execute a campaign, choose a seed, write an artifact, apply a threshold,
or turn the development trace into scientific evidence.

Several Alberta Plan requirements are not measured by this dyad.  In
particular, the trace has no individual helper-only reward, helper self-model,
long-horizon probabilistic world prediction, learned general features, or
explicit memory-selection mechanism.  The report records those absences
instead of substituting a proxy or claiming completion.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    HiddenLearningPartnerPlanningBridge,
    HiddenLearningPartnerPlanningRun,
    condition_spec,
    validate_hidden_learning_partner_planning_run,
)

HIDDEN_LEARNING_PARTNER_PROPERTY_READOUT_SCHEMA = (
    "alberta.hidden-learning-partner-property-readout.development.v1"
)
ASSESSMENT_STATUS = "descriptive-only-not-assessed"

_SHARED_CONTEXT_ROW = 0
_GROUND_REWARD_INDEX = 1


class HiddenLearningPartnerPropertyReadoutError(ValueError):
    """Raised when a source run cannot support a trustworthy readout."""


@dataclasses.dataclass(frozen=True, slots=True)
class ScalarReadout:
    """One finite sample mean, with absence represented explicitly."""

    count: int
    value: float | None
    available: bool
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("ScalarReadout.count must be a non-negative integer")
        if type(self.available) is not bool:
            raise ValueError("ScalarReadout.available must be a boolean")
        if self.available:
            if self.count == 0:
                raise ValueError("an available scalar must have a positive count")
            if type(self.value) is not float or not math.isfinite(self.value):
                raise ValueError("an available scalar must have one finite float value")
            if self.unavailable_reason is not None:
                raise ValueError("an available scalar cannot have an unavailable reason")
        elif (
            self.count != 0
            or self.value is not None
            or type(self.unavailable_reason) is not str
            or not self.unavailable_reason
        ):
            raise ValueError(
                "an unavailable scalar requires count zero, value None, and one reason"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class UnavailablePropertyReadout:
    """One property that the source trace does not actually identify."""

    property_name: str
    available: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.property_name) is not str or not self.property_name:
            raise ValueError("property_name must be a non-empty string")
        if self.available is not False:
            raise ValueError("UnavailablePropertyReadout.available must be false")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("an unavailable property requires one reason")


@dataclasses.dataclass(frozen=True, slots=True)
class PrequentialPerformanceReadout:
    """Shared outcome and randomized planner/control performance."""

    shared_dyad_mean_reward: ScalarReadout
    planner_treated_eligible_mean_reward: ScalarReadout
    ordinary_control_eligible_mean_reward: ScalarReadout
    treated_minus_control_mean_reward: ScalarReadout
    individual_helper_reward: UnavailablePropertyReadout


@dataclasses.dataclass(frozen=True, slots=True)
class AgentPredictionReadout:
    """Predict-before-update beneficiary-action diagnostics."""

    partner_action_mean_nll: ScalarReadout
    partner_action_mean_brier: ScalarReadout
    partner_action_argmax_accuracy: ScalarReadout
    helper_self_action_prediction: UnavailablePropertyReadout


@dataclasses.dataclass(frozen=True, slots=True)
class WorldPlanningReadout:
    """Available one-step grounded predictions and planner contrasts."""

    grounded_reward_mse: ScalarReadout
    grounded_next_observation_mse: ScalarReadout
    randomized_planner_advantage: ScalarReadout
    direct_channel_potential_planner_advantage: ScalarReadout
    long_horizon_distributional_world_prediction: UnavailablePropertyReadout


@dataclasses.dataclass(frozen=True, slots=True)
class RoleUpdateReadout:
    """Committed writes and observed selected-cell changes for one role."""

    role: str
    update_opportunities: int
    committed_writes: int
    effective_selected_value_changes: int
    initial_to_final_table_changed: bool
    committed_update_observed: bool
    effective_value_change_observed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PhasePropertyReadout:
    """Prequential entry, full-phase, and exit summaries for one hidden phase."""

    phase_index: int
    hidden_context: int
    count: int
    window_count: int
    mean_reward: float
    entry_reward: float
    exit_reward: float
    mean_partner_nll: float
    entry_partner_nll: float
    exit_partner_nll: float
    mean_partner_brier: float
    entry_partner_brier: float
    exit_partner_brier: float
    mean_grounded_reward_mse: float
    entry_grounded_reward_mse: float
    exit_grounded_reward_mse: float
    mean_grounded_next_observation_mse: float
    entry_grounded_next_observation_mse: float
    exit_grounded_next_observation_mse: float


@dataclasses.dataclass(frozen=True, slots=True)
class RecurrencePropertyReadout:
    """Same-context entry retention and within-phase recovery contrasts.

    Reward deltas are positive when performance is higher.  NLL and MSE entry
    costs are positive when recurrence entry is worse than the prior exit;
    their recovery reductions are positive when the loss falls within the
    recurrent phase.
    """

    phase_index: int
    hidden_context: int
    reference_phase_index: int | None
    available: bool
    unavailable_reason: str | None
    reward_entry_minus_reference_exit: float | None
    reward_exit_minus_entry: float | None
    reward_exit_minus_reference_exit: float | None
    partner_nll_entry_cost: float | None
    partner_nll_recovery_reduction: float | None
    partner_brier_entry_cost: float | None
    partner_brier_recovery_reduction: float | None
    grounded_reward_mse_entry_cost: float | None
    grounded_reward_mse_recovery_reduction: float | None
    grounded_next_observation_mse_entry_cost: float | None
    grounded_next_observation_mse_recovery_reduction: float | None


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenLearningPartnerPropertyReadout:
    """Strict in-memory development readout; never an acceptance verdict."""

    schema: str
    assessment_status: str
    development_only: bool
    thresholds_applied: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    alberta_plan_completion_claimed: bool
    source_condition: str
    num_steps: int
    prequential_performance: PrequentialPerformanceReadout
    agent_prediction: AgentPredictionReadout
    world_and_planning: WorldPlanningReadout
    helper_updates: RoleUpdateReadout
    beneficiary_updates: RoleUpdateReadout
    both_roles_committed_updates_observed: bool
    both_roles_effective_value_changes_observed: bool
    phases: tuple[PhasePropertyReadout, ...]
    recurrences: tuple[RecurrencePropertyReadout, ...]
    unavailable_alberta_properties: tuple[UnavailablePropertyReadout, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible mapping with no NaN sentinels."""

        encoded = json.dumps(
            dataclasses.asdict(self),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cast(dict[str, object], json.loads(encoded))


def _available_mean(values: np.ndarray) -> ScalarReadout:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.all(np.isfinite(flattened)):
        raise HiddenLearningPartnerPropertyReadoutError(
            "an available prequential mean must contain finite observations"
        )
    return ScalarReadout(
        count=int(flattened.size),
        value=float(np.mean(flattened)),
        available=True,
        unavailable_reason=None,
    )


def _conditional_mean(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    reason: str,
) -> ScalarReadout:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=np.bool_)]
    if selected.size == 0:
        return ScalarReadout(
            count=0,
            value=None,
            available=False,
            unavailable_reason=reason,
        )
    return _available_mean(selected)


def _difference(
    left: ScalarReadout,
    right: ScalarReadout,
    *,
    reason: str,
) -> ScalarReadout:
    if not left.available or not right.available:
        return ScalarReadout(
            count=0,
            value=None,
            available=False,
            unavailable_reason=reason,
        )
    assert left.value is not None and right.value is not None
    return ScalarReadout(
        count=left.count + right.count,
        value=float(left.value - right.value),
        available=True,
        unavailable_reason=None,
    )


def _array_bits_equal(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _tree_bits_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):  # type: ignore[operator]
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            left_leaf.dtype, jax.dtypes.prng_key
        ):
            if not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                right_leaf.dtype, jax.dtypes.prng_key
            ):
                return False
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        if not _array_bits_equal(left_leaf, right_leaf):
            return False
    return True


def _tree_schema_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):  # type: ignore[operator]
        return False
    return all(
        getattr(left_leaf, "shape", None) == getattr(right_leaf, "shape", None)
        and getattr(left_leaf, "dtype", None) == getattr(right_leaf, "dtype", None)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_all_finite_host(tree: object) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype, jax.dtypes.prng_key
        ):
            continue
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            return False
    return True


def _validate_role_value_continuity(
    run: HiddenLearningPartnerPlanningRun,
) -> tuple[int, int]:
    """Replay the two tiny value tables from selected-cell pre/post records."""

    trace = run.trace
    rewards = np.asarray(trace.reward, dtype=np.float32)
    helper_cues = np.asarray(trace.helper_cue, dtype=np.int64)
    helper_actions = np.asarray(trace.helper_message, dtype=np.int64)
    delivered = np.asarray(trace.delivered_message, dtype=np.int64)
    beneficiary_actions = np.asarray(trace.beneficiary_action, dtype=np.int64)
    helper_writes = np.asarray(trace.helper_write, dtype=np.bool_)
    beneficiary_writes = np.asarray(trace.beneficiary_write, dtype=np.bool_)
    helper_pre = np.asarray(trace.helper_value_pre, dtype=np.float32)
    helper_post = np.asarray(trace.helper_value_post, dtype=np.float32)
    beneficiary_pre = np.asarray(trace.beneficiary_value_pre, dtype=np.float32)
    beneficiary_post = np.asarray(trace.beneficiary_value_post, dtype=np.float32)
    helper_values = np.asarray(run.initial_state.learner.helper.values).copy()
    beneficiary_values = np.asarray(run.initial_state.learner.beneficiary.values).copy()
    learning_rate = np.float32(run.config.learning_rate)
    helper_changes = 0
    beneficiary_changes = 0

    for step in range(run.config.num_steps):
        helper_index = (
            _SHARED_CONTEXT_ROW,
            int(helper_cues[step]),
            int(helper_actions[step]),
        )
        beneficiary_index = (
            _SHARED_CONTEXT_ROW,
            int(delivered[step]),
            int(beneficiary_actions[step]),
        )
        if not _array_bits_equal(helper_values[helper_index], helper_pre[step]):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"helper selected-value continuity failed at step {step}"
            )
        if not _array_bits_equal(
            beneficiary_values[beneficiary_index], beneficiary_pre[step]
        ):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"beneficiary selected-value continuity failed at step {step}"
            )

        expected_helper = helper_pre[step]
        if helper_writes[step]:
            expected_helper = np.float32(
                helper_pre[step]
                + learning_rate * np.float32(rewards[step] - helper_pre[step])
            )
        expected_beneficiary = beneficiary_pre[step]
        if beneficiary_writes[step]:
            expected_beneficiary = np.float32(
                beneficiary_pre[step]
                + learning_rate * np.float32(rewards[step] - beneficiary_pre[step])
            )
        if not _array_bits_equal(expected_helper, helper_post[step]):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"helper update equation failed at step {step}"
            )
        if not _array_bits_equal(expected_beneficiary, beneficiary_post[step]):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"beneficiary update equation failed at step {step}"
            )
        helper_changes += int(
            helper_writes[step]
            and not _array_bits_equal(helper_pre[step], helper_post[step])
        )
        beneficiary_changes += int(
            beneficiary_writes[step]
            and not _array_bits_equal(beneficiary_pre[step], beneficiary_post[step])
        )
        helper_values[helper_index] = helper_post[step]
        beneficiary_values[beneficiary_index] = beneficiary_post[step]

    if not _array_bits_equal(helper_values, run.final_state.learner.helper.values):
        raise HiddenLearningPartnerPropertyReadoutError(
            "helper final table is not continuous with selected-cell updates"
        )
    if not _array_bits_equal(
        beneficiary_values, run.final_state.learner.beneficiary.values
    ):
        raise HiddenLearningPartnerPropertyReadoutError(
            "beneficiary final table is not continuous with selected-cell updates"
        )
    return helper_changes, beneficiary_changes


def _validate_source_run(run: object) -> HiddenLearningPartnerPlanningRun:
    if type(run) is not HiddenLearningPartnerPlanningRun:
        raise HiddenLearningPartnerPropertyReadoutError(
            "run must be an exact HiddenLearningPartnerPlanningRun"
        )
    checked = run
    try:
        validation_errors = validate_hidden_learning_partner_planning_run(checked)
    except Exception as exc:  # fail closed on malformed nested arrays/types
        raise HiddenLearningPartnerPropertyReadoutError(
            f"source run validation failed closed: {exc}"
        ) from exc
    if validation_errors:
        joined = "; ".join(validation_errors)
        raise HiddenLearningPartnerPropertyReadoutError(
            f"source run validation failed: {joined}"
        )

    bridge = HiddenLearningPartnerPlanningBridge(checked.config, checked.condition)
    expected_initial = bridge.initialize(jr.key(checked.seed, impl="threefry2x32"))
    if not _tree_bits_equal(expected_initial, checked.initial_state):
        raise HiddenLearningPartnerPropertyReadoutError(
            "initial state is not the exact seed/config/condition initialization"
        )
    if not _tree_schema_equal(expected_initial, checked.final_state):
        raise HiddenLearningPartnerPropertyReadoutError(
            "final state differs from the exact persistent-state schema"
        )
    if not _tree_all_finite_host(checked.initial_state) or not _tree_all_finite_host(
        checked.final_state
    ):
        raise HiddenLearningPartnerPropertyReadoutError(
            "initial/final persistent state contains a non-finite value"
        )
    n = checked.config.num_steps
    trace = checked.trace
    neutral_trace = bridge._neutral_trace(expected_initial)
    for field in dataclasses.fields(neutral_trace):  # type: ignore[arg-type]
        expected_leaf = getattr(neutral_trace, field.name)
        observed_leaf = getattr(trace, field.name)
        if (
            not isinstance(observed_leaf, jax.Array)
            or observed_leaf.shape != (n, *expected_leaf.shape)
            or observed_leaf.dtype != expected_leaf.dtype
        ):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"trace.{field.name} differs from the exact array schema"
            )
    helper_cues = np.asarray(trace.helper_cue)
    next_helper_cues = np.asarray(trace.next_helper_cue)
    if (
        int(checked.initial_state.step_count) != 0
        or int(checked.initial_state.world.step_count) != 0
        or int(checked.final_state.step_count) != n
        or int(checked.final_state.world.step_count) != n
    ):
        raise HiddenLearningPartnerPropertyReadoutError(
            "initial/final state counters do not bind one uninterrupted life"
        )
    if not _array_bits_equal(helper_cues[0], checked.initial_state.world.cue):
        raise HiddenLearningPartnerPropertyReadoutError(
            "first trace observation is not bound to the initial world state"
        )
    if n > 1 and not _array_bits_equal(next_helper_cues[:-1], helper_cues[1:]):
        raise HiddenLearningPartnerPropertyReadoutError(
            "world observation continuity failed between trace steps"
        )
    if not _array_bits_equal(next_helper_cues[-1], checked.final_state.world.cue):
        raise HiddenLearningPartnerPropertyReadoutError(
            "last trace observation is not bound to the final world state"
        )
    phases = checked.metrics.phase_diagnostics
    if (
        phases.phase_index != tuple(range(checked.config.n_phases))
        or phases.phase_counts != (checked.config.phase_length,) * checked.config.n_phases
        or phases.phase_valid != (True,) * checked.config.n_phases
        or phases.leading_counts != (phases.window_steps,) * checked.config.n_phases
        or phases.trailing_counts != (phases.window_steps,) * checked.config.n_phases
    ):
        raise HiddenLearningPartnerPropertyReadoutError(
            "phase accounting does not describe complete contiguous phases"
        )
    spec = condition_spec(checked.condition)
    expected_behavior_updates = n if spec.behavior_write else 0
    expected_grounded_updates = n if spec.grounded_write else 0
    for name, state, expected_updates in (
        ("initial behavior", checked.initial_state.behavior, 0),
        ("final behavior", checked.final_state.behavior, expected_behavior_updates),
    ):
        if int(state.step_count) != expected_updates or not _array_bits_equal(
            state.step_words,
            np.asarray((0, expected_updates), dtype=np.uint32),
        ):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"{name} exact update clock is not continuous"
            )
    for name, grounded_state, expected_updates in (
        ("initial grounded model", checked.initial_state.grounded, 0),
        ("final grounded model", checked.final_state.grounded, expected_grounded_updates),
    ):
        if int(grounded_state.update_count) != expected_updates or not _array_bits_equal(
            grounded_state.update_words,
            np.asarray((0, expected_updates), dtype=np.uint32),
        ):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"{name} exact update clock is not continuous"
            )
    _validate_role_value_continuity(checked)
    return checked


def _phase_readouts(
    run: HiddenLearningPartnerPlanningRun,
    *,
    partner_brier: np.ndarray,
    grounded_next_mse: np.ndarray,
) -> tuple[PhasePropertyReadout, ...]:
    trace = run.trace
    rewards = np.asarray(trace.reward, dtype=np.float64)
    partner_nll = np.asarray(trace.behavior_nll, dtype=np.float64)
    grounded_reward_mse = np.square(
        np.asarray(trace.grounded_reward_error, dtype=np.float64)
    )
    window = run.metrics.phase_diagnostics.window_steps
    results: list[PhasePropertyReadout] = []
    for phase in range(run.config.n_phases):
        start = phase * run.config.phase_length
        stop = start + run.config.phase_length
        entry = slice(start, start + window)
        exit_window = slice(stop - window, stop)
        whole = slice(start, stop)
        context = int(np.asarray(trace.oracle_context)[start])
        values = (
            rewards,
            partner_nll,
            partner_brier,
            grounded_reward_mse,
            grounded_next_mse,
        )
        if any(not np.all(np.isfinite(value[whole])) for value in values):
            raise HiddenLearningPartnerPropertyReadoutError(
                f"phase {phase} contains a non-finite readout primitive"
            )
        results.append(
            PhasePropertyReadout(
                phase_index=phase,
                hidden_context=context,
                count=run.config.phase_length,
                window_count=window,
                mean_reward=float(np.mean(rewards[whole])),
                entry_reward=float(np.mean(rewards[entry])),
                exit_reward=float(np.mean(rewards[exit_window])),
                mean_partner_nll=float(np.mean(partner_nll[whole])),
                entry_partner_nll=float(np.mean(partner_nll[entry])),
                exit_partner_nll=float(np.mean(partner_nll[exit_window])),
                mean_partner_brier=float(np.mean(partner_brier[whole])),
                entry_partner_brier=float(np.mean(partner_brier[entry])),
                exit_partner_brier=float(np.mean(partner_brier[exit_window])),
                mean_grounded_reward_mse=float(np.mean(grounded_reward_mse[whole])),
                entry_grounded_reward_mse=float(np.mean(grounded_reward_mse[entry])),
                exit_grounded_reward_mse=float(np.mean(grounded_reward_mse[exit_window])),
                mean_grounded_next_observation_mse=float(
                    np.mean(grounded_next_mse[whole])
                ),
                entry_grounded_next_observation_mse=float(
                    np.mean(grounded_next_mse[entry])
                ),
                exit_grounded_next_observation_mse=float(
                    np.mean(grounded_next_mse[exit_window])
                ),
            )
        )
    return tuple(results)


def _recurrence_readouts(
    phases: tuple[PhasePropertyReadout, ...],
) -> tuple[RecurrencePropertyReadout, ...]:
    latest_by_context: dict[int, PhasePropertyReadout] = {}
    results: list[RecurrencePropertyReadout] = []
    for phase in phases:
        reference = latest_by_context.get(phase.hidden_context)
        if reference is None:
            results.append(
                RecurrencePropertyReadout(
                    phase_index=phase.phase_index,
                    hidden_context=phase.hidden_context,
                    reference_phase_index=None,
                    available=False,
                    unavailable_reason=(
                        "no earlier complete phase has the same hidden context"
                    ),
                    reward_entry_minus_reference_exit=None,
                    reward_exit_minus_entry=None,
                    reward_exit_minus_reference_exit=None,
                    partner_nll_entry_cost=None,
                    partner_nll_recovery_reduction=None,
                    partner_brier_entry_cost=None,
                    partner_brier_recovery_reduction=None,
                    grounded_reward_mse_entry_cost=None,
                    grounded_reward_mse_recovery_reduction=None,
                    grounded_next_observation_mse_entry_cost=None,
                    grounded_next_observation_mse_recovery_reduction=None,
                )
            )
        else:
            results.append(
                RecurrencePropertyReadout(
                    phase_index=phase.phase_index,
                    hidden_context=phase.hidden_context,
                    reference_phase_index=reference.phase_index,
                    available=True,
                    unavailable_reason=None,
                    reward_entry_minus_reference_exit=float(
                        phase.entry_reward - reference.exit_reward
                    ),
                    reward_exit_minus_entry=float(
                        phase.exit_reward - phase.entry_reward
                    ),
                    reward_exit_minus_reference_exit=float(
                        phase.exit_reward - reference.exit_reward
                    ),
                    partner_nll_entry_cost=float(
                        phase.entry_partner_nll - reference.exit_partner_nll
                    ),
                    partner_nll_recovery_reduction=float(
                        phase.entry_partner_nll - phase.exit_partner_nll
                    ),
                    partner_brier_entry_cost=float(
                        phase.entry_partner_brier - reference.exit_partner_brier
                    ),
                    partner_brier_recovery_reduction=float(
                        phase.entry_partner_brier - phase.exit_partner_brier
                    ),
                    grounded_reward_mse_entry_cost=float(
                        phase.entry_grounded_reward_mse
                        - reference.exit_grounded_reward_mse
                    ),
                    grounded_reward_mse_recovery_reduction=float(
                        phase.entry_grounded_reward_mse
                        - phase.exit_grounded_reward_mse
                    ),
                    grounded_next_observation_mse_entry_cost=float(
                        phase.entry_grounded_next_observation_mse
                        - reference.exit_grounded_next_observation_mse
                    ),
                    grounded_next_observation_mse_recovery_reduction=float(
                        phase.entry_grounded_next_observation_mse
                        - phase.exit_grounded_next_observation_mse
                    ),
                )
            )
        latest_by_context[phase.hidden_context] = phase
    return tuple(results)


def build_hidden_learning_partner_property_readout(
    *,
    run: HiddenLearningPartnerPlanningRun,
) -> HiddenLearningPartnerPropertyReadout:
    """Validate one run and reconstruct only properties identified by its trace."""

    checked = _validate_source_run(run)
    trace = checked.trace
    rewards = np.asarray(trace.reward, dtype=np.float64)
    probabilities = np.asarray(trace.behavior_probabilities_pre, dtype=np.float64)
    actions = np.asarray(trace.beneficiary_action, dtype=np.int64)
    one_hot = np.eye(2, dtype=np.float64)[actions]
    partner_brier = np.sum(np.square(probabilities - one_hot), axis=1)
    partner_accuracy = (np.argmax(probabilities, axis=1) == actions).astype(np.float64)
    eligible = np.asarray(trace.action_changed, dtype=np.bool_)
    consumed = np.asarray(trace.planner_consumed, dtype=np.bool_)
    treated_reward = _conditional_mean(
        rewards,
        eligible & consumed,
        reason="no action-changing step consumed the planner proposal",
    )
    control_reward = _conditional_mean(
        rewards,
        eligible & ~consumed,
        reason="no action-changing step executed the ordinary control proposal",
    )
    randomized_advantage = _difference(
        treated_reward,
        control_reward,
        reason=(
            "the randomized contrast requires both eligible planner-treated and "
            "ordinary-control observations"
        ),
    )

    grounded_raw = np.asarray(trace.grounded_raw_prediction_pre, dtype=np.float64)
    next_target = 2.0 * np.asarray(trace.next_helper_cue, dtype=np.float64) - 1.0
    grounded_next_mse = np.square(grounded_raw[:, 0] - next_target)
    grounded_reward_mse = np.square(
        grounded_raw[:, _GROUND_REWARD_INDEX] - rewards
    )
    potential = np.asarray(trace.delivered_potential_rewards, dtype=np.float64)
    planner = np.asarray(trace.planner_message, dtype=np.int64)
    ordinary = np.asarray(trace.ordinary_message, dtype=np.int64)
    indices = np.arange(rewards.size)
    potential_delta = potential[indices, planner] - potential[indices, ordinary]
    spec = condition_spec(checked.condition)
    if spec.channel == "direct":
        potential_advantage = _conditional_mean(
            potential_delta,
            eligible,
            reason="no action-changing planner proposal occurred",
        )
    else:
        potential_advantage = ScalarReadout(
            count=0,
            value=None,
            available=False,
            unavailable_reason=(
                "delivered-message potential outcomes identify this contrast only "
                "under the direct channel"
            ),
        )

    helper_changes, beneficiary_changes = _validate_role_value_continuity(checked)
    helper_writes = int(np.count_nonzero(np.asarray(trace.helper_write)))
    beneficiary_writes = int(np.count_nonzero(np.asarray(trace.beneficiary_write)))
    helper_table_changed = not _array_bits_equal(
        checked.initial_state.learner.helper.values,
        checked.final_state.learner.helper.values,
    )
    beneficiary_table_changed = not _array_bits_equal(
        checked.initial_state.learner.beneficiary.values,
        checked.final_state.learner.beneficiary.values,
    )
    helper_updates = RoleUpdateReadout(
        role="helper",
        update_opportunities=checked.config.num_steps,
        committed_writes=helper_writes,
        effective_selected_value_changes=helper_changes,
        initial_to_final_table_changed=helper_table_changed,
        committed_update_observed=helper_writes > 0,
        effective_value_change_observed=helper_changes > 0,
    )
    beneficiary_updates = RoleUpdateReadout(
        role="beneficiary",
        update_opportunities=checked.config.num_steps,
        committed_writes=beneficiary_writes,
        effective_selected_value_changes=beneficiary_changes,
        initial_to_final_table_changed=beneficiary_table_changed,
        committed_update_observed=beneficiary_writes > 0,
        effective_value_change_observed=beneficiary_changes > 0,
    )
    phases = _phase_readouts(
        checked,
        partner_brier=partner_brier,
        grounded_next_mse=grounded_next_mse,
    )

    individual_helper_reward = UnavailablePropertyReadout(
        property_name="individual_helper_reward",
        available=False,
        reason=(
            "the kernel exposes one shared dyadic reward, not a separate helper-only outcome"
        ),
    )
    helper_self_prediction = UnavailablePropertyReadout(
        property_name="helper_self_action_prediction",
        available=False,
        reason=(
            "the trace predicts the beneficiary action but contains no prequential helper "
            "self-action model"
        ),
    )
    long_horizon_world = UnavailablePropertyReadout(
        property_name="long_horizon_distributional_world_prediction",
        available=False,
        reason=(
            "the grounded model exposes one-step point predictions for reward and next cue "
            "only"
        ),
    )
    unavailable_alberta = (
        UnavailablePropertyReadout(
            property_name="general_feature_discovery",
            available=False,
            reason="the dyad uses fixed one-dimensional inputs and fixed value-table cells",
        ),
        UnavailablePropertyReadout(
            property_name="learned_memory_selection_and_forgetting",
            available=False,
            reason="the dyad has no learned remember/forget or memory-allocation mechanism",
        ),
        UnavailablePropertyReadout(
            property_name="catastrophic_forgetting_resistance",
            available=False,
            reason=(
                "phase recurrence is descriptive on one short trace and cannot establish "
                "absence of catastrophic forgetting"
            ),
        ),
        UnavailablePropertyReadout(
            property_name="scaling_behavior",
            available=False,
            reason="one fixed-size binary dyad contains no scale sweep",
        ),
    )
    return HiddenLearningPartnerPropertyReadout(
        schema=HIDDEN_LEARNING_PARTNER_PROPERTY_READOUT_SCHEMA,
        assessment_status=ASSESSMENT_STATUS,
        development_only=True,
        thresholds_applied=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        alberta_plan_completion_claimed=False,
        source_condition=checked.condition,
        num_steps=checked.config.num_steps,
        prequential_performance=PrequentialPerformanceReadout(
            shared_dyad_mean_reward=_available_mean(rewards),
            planner_treated_eligible_mean_reward=treated_reward,
            ordinary_control_eligible_mean_reward=control_reward,
            treated_minus_control_mean_reward=randomized_advantage,
            individual_helper_reward=individual_helper_reward,
        ),
        agent_prediction=AgentPredictionReadout(
            partner_action_mean_nll=_available_mean(
                np.asarray(trace.behavior_nll, dtype=np.float64)
            ),
            partner_action_mean_brier=_available_mean(partner_brier),
            partner_action_argmax_accuracy=_available_mean(partner_accuracy),
            helper_self_action_prediction=helper_self_prediction,
        ),
        world_and_planning=WorldPlanningReadout(
            grounded_reward_mse=_available_mean(grounded_reward_mse),
            grounded_next_observation_mse=_available_mean(grounded_next_mse),
            randomized_planner_advantage=randomized_advantage,
            direct_channel_potential_planner_advantage=potential_advantage,
            long_horizon_distributional_world_prediction=long_horizon_world,
        ),
        helper_updates=helper_updates,
        beneficiary_updates=beneficiary_updates,
        both_roles_committed_updates_observed=(
            helper_updates.committed_update_observed
            and beneficiary_updates.committed_update_observed
        ),
        both_roles_effective_value_changes_observed=(
            helper_updates.effective_value_change_observed
            and beneficiary_updates.effective_value_change_observed
        ),
        phases=phases,
        recurrences=_recurrence_readouts(phases),
        unavailable_alberta_properties=unavailable_alberta,
    )


__all__ = [
    "ASSESSMENT_STATUS",
    "HIDDEN_LEARNING_PARTNER_PROPERTY_READOUT_SCHEMA",
    "AgentPredictionReadout",
    "HiddenLearningPartnerPropertyReadout",
    "HiddenLearningPartnerPropertyReadoutError",
    "PhasePropertyReadout",
    "PrequentialPerformanceReadout",
    "RecurrencePropertyReadout",
    "RoleUpdateReadout",
    "ScalarReadout",
    "UnavailablePropertyReadout",
    "WorldPlanningReadout",
    "build_hidden_learning_partner_property_readout",
]
