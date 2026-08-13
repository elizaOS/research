# mypy: disable-error-code="type-var"
"""Focused contracts for the hidden-dyad Alberta-property readout."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import alberta_framework.evaluation.hidden_learning_partner_planning_development as source
from alberta_framework.evaluation.hidden_learning_partner_property_readout import (
    ASSESSMENT_STATUS,
    HIDDEN_LEARNING_PARTNER_PROPERTY_READOUT_SCHEMA,
    HiddenLearningPartnerPropertyReadoutError,
    build_hidden_learning_partner_property_readout,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def source_run() -> source.HiddenLearningPartnerPlanningRun:
    """Build one real tiny source run with no resource-contract shim."""

    config = source.HiddenLearningPartnerPlanningConfig(
        phase_length=2,
        n_phases=4,
        learning_rate=0.2,
        epsilon=0.2,
        behavior_step_size=0.1,
        grounded_step_size=0.1,
    )
    return source.run_hidden_learning_partner_planning(
        source.JOINT_ADAPTIVE,
        seed=13,
        config=config,
        jit_compile=False,
    )


def test_readout_matches_direct_hand_calculation(
    source_run: source.HiddenLearningPartnerPlanningRun,
) -> None:
    report = build_hidden_learning_partner_property_readout(run=source_run)
    trace = source_run.trace
    rewards = np.asarray(trace.reward, dtype=np.float64)
    probabilities = np.asarray(trace.behavior_probabilities_pre, dtype=np.float64)
    actions = np.asarray(trace.beneficiary_action, dtype=np.int64)
    one_hot = np.eye(2, dtype=np.float64)[actions]
    brier = np.sum(np.square(probabilities - one_hot), axis=1)
    eligible = np.asarray(trace.action_changed, dtype=np.bool_)
    consumed = np.asarray(trace.planner_consumed, dtype=np.bool_)
    treated = eligible & consumed
    control = eligible & ~consumed

    assert report.prequential_performance.shared_dyad_mean_reward.value == pytest.approx(
        float(np.mean(rewards))
    )
    assert report.prequential_performance.planner_treated_eligible_mean_reward.value == (
        pytest.approx(float(np.mean(rewards[treated])))
    )
    assert report.prequential_performance.ordinary_control_eligible_mean_reward.value == (
        pytest.approx(float(np.mean(rewards[control])))
    )
    assert report.prequential_performance.treated_minus_control_mean_reward.value == (
        pytest.approx(float(np.mean(rewards[treated]) - np.mean(rewards[control])))
    )
    assert report.agent_prediction.partner_action_mean_nll.value == pytest.approx(
        float(np.mean(np.asarray(trace.behavior_nll, dtype=np.float64)))
    )
    assert report.agent_prediction.partner_action_mean_brier.value == pytest.approx(
        float(np.mean(brier))
    )
    assert report.agent_prediction.partner_action_argmax_accuracy.value == pytest.approx(
        float(np.mean(np.argmax(probabilities, axis=1) == actions))
    )

    grounded = np.asarray(trace.grounded_raw_prediction_pre, dtype=np.float64)
    next_target = 2.0 * np.asarray(trace.next_helper_cue, dtype=np.float64) - 1.0
    assert report.world_and_planning.grounded_reward_mse.value == pytest.approx(
        float(np.mean(np.square(grounded[:, 1] - rewards)))
    )
    assert report.world_and_planning.grounded_next_observation_mse.value == pytest.approx(
        float(np.mean(np.square(grounded[:, 0] - next_target)))
    )


def test_phase_retention_and_recovery_are_exact_window_contrasts(
    source_run: source.HiddenLearningPartnerPlanningRun,
) -> None:
    report = build_hidden_learning_partner_property_readout(run=source_run)
    rewards = np.asarray(source_run.trace.reward, dtype=np.float64).reshape(4, 2)
    nll = np.asarray(source_run.trace.behavior_nll, dtype=np.float64).reshape(4, 2)

    assert tuple(phase.hidden_context for phase in report.phases) == (0, 1, 0, 1)
    assert tuple(phase.window_count for phase in report.phases) == (1, 1, 1, 1)
    for phase_index, phase in enumerate(report.phases):
        assert phase.mean_reward == pytest.approx(float(np.mean(rewards[phase_index])))
        assert phase.entry_reward == float(rewards[phase_index, 0])
        assert phase.exit_reward == float(rewards[phase_index, 1])
        assert phase.entry_partner_nll == float(nll[phase_index, 0])
        assert phase.exit_partner_nll == float(nll[phase_index, 1])

    assert not report.recurrences[0].available
    assert not report.recurrences[1].available
    recurrence = report.recurrences[2]
    assert recurrence.available
    assert recurrence.reference_phase_index == 0
    assert recurrence.reward_entry_minus_reference_exit == pytest.approx(
        rewards[2, 0] - rewards[0, 1]
    )
    assert recurrence.reward_exit_minus_entry == pytest.approx(
        rewards[2, 1] - rewards[2, 0]
    )
    assert recurrence.reward_exit_minus_reference_exit == pytest.approx(
        rewards[2, 1] - rewards[0, 1]
    )
    assert recurrence.partner_nll_entry_cost == pytest.approx(
        nll[2, 0] - nll[0, 1]
    )
    assert recurrence.partner_nll_recovery_reduction == pytest.approx(
        nll[2, 0] - nll[2, 1]
    )


def test_both_role_updates_are_counted_without_conflating_writes_and_changes(
    source_run: source.HiddenLearningPartnerPlanningRun,
) -> None:
    report = build_hidden_learning_partner_property_readout(run=source_run)
    trace = source_run.trace
    helper_changes = int(
        np.count_nonzero(
            np.asarray(trace.helper_write)
            & (
                np.asarray(trace.helper_value_pre).view(np.uint32)
                != np.asarray(trace.helper_value_post).view(np.uint32)
            )
        )
    )
    beneficiary_changes = int(
        np.count_nonzero(
            np.asarray(trace.beneficiary_write)
            & (
                np.asarray(trace.beneficiary_value_pre).view(np.uint32)
                != np.asarray(trace.beneficiary_value_post).view(np.uint32)
            )
        )
    )

    assert report.helper_updates.update_opportunities == source_run.config.num_steps
    assert report.helper_updates.committed_writes == source_run.config.num_steps
    assert report.helper_updates.effective_selected_value_changes == helper_changes
    assert report.beneficiary_updates.committed_writes == source_run.config.num_steps
    assert (
        report.beneficiary_updates.effective_selected_value_changes
        == beneficiary_changes
    )
    assert report.both_roles_committed_updates_observed
    assert report.both_roles_effective_value_changes_observed


@pytest.mark.parametrize(
    "corruption",
    (
        "phase-accounting",
        "trace-dtype",
        "helper-selected-value",
        "final-helper-table",
        "final-behavior-state",
        "initial-state",
    ),
)
def test_malformed_accounting_and_state_continuity_fail_closed(
    source_run: source.HiddenLearningPartnerPlanningRun,
    corruption: str,
) -> None:
    if corruption == "phase-accounting":
        phases = source_run.metrics.phase_diagnostics
        bad_phases = dataclasses.replace(
            phases,
            phase_counts=(phases.phase_counts[0] - 1, *phases.phase_counts[1:]),
        )
        malformed = dataclasses.replace(
            source_run,
            metrics=dataclasses.replace(
                source_run.metrics,
                phase_diagnostics=bad_phases,
            ),
        )
    elif corruption == "trace-dtype":
        malformed = dataclasses.replace(
            source_run,
            trace=dataclasses.replace(
                source_run.trace,
                reward=source_run.trace.reward.astype(np.int32),
            ),
        )
    elif corruption == "helper-selected-value":
        malformed = dataclasses.replace(
            source_run,
            trace=dataclasses.replace(
                source_run.trace,
                helper_value_pre=source_run.trace.helper_value_pre.at[0].add(0.125)
            ),
        )
    elif corruption == "final-helper-table":
        bad_helper = dataclasses.replace(
            source_run.final_state.learner.helper,
            values=source_run.final_state.learner.helper.values.at[0, 0, 0].add(0.125)
        )
        malformed = dataclasses.replace(
            source_run,
            final_state=dataclasses.replace(
                source_run.final_state,
                learner=dataclasses.replace(
                    source_run.final_state.learner,
                    helper=bad_helper,
                ),
            ),
        )
    elif corruption == "final-behavior-state":
        malformed = dataclasses.replace(
            source_run,
            final_state=dataclasses.replace(
                source_run.final_state,
                behavior=dataclasses.replace(
                    source_run.final_state.behavior,
                    weights=source_run.final_state.behavior.weights.at[0, 0].set(
                        np.nan
                    ),
                ),
            ),
        )
    else:
        malformed = dataclasses.replace(
            source_run,
            initial_state=dataclasses.replace(
                source_run.initial_state,
                learner=dataclasses.replace(
                    source_run.initial_state.learner,
                    helper=dataclasses.replace(
                        source_run.initial_state.learner.helper,
                        values=source_run.initial_state.learner.helper.values.at[
                            0, 0, 0
                        ].set(0.125)
                    )
                )
            ),
        )

    with pytest.raises(HiddenLearningPartnerPropertyReadoutError):
        build_hidden_learning_partner_property_readout(run=malformed)


def test_unsupported_properties_and_serialization_remain_explicitly_nonpromoting(
    source_run: source.HiddenLearningPartnerPlanningRun,
) -> None:
    report = build_hidden_learning_partner_property_readout(run=source_run)
    payload = report.to_dict()
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert json.loads(serialized) == payload
    assert payload["schema"] == HIDDEN_LEARNING_PARTNER_PROPERTY_READOUT_SCHEMA
    assert payload["assessment_status"] == ASSESSMENT_STATUS
    assert payload["development_only"] is True
    assert payload["thresholds_applied"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["alberta_plan_completion_claimed"] is False
    assert "source_seed" not in payload

    unavailable = {
        item.property_name: item.reason
        for item in report.unavailable_alberta_properties
    }
    assert set(unavailable) == {
        "general_feature_discovery",
        "learned_memory_selection_and_forgetting",
        "catastrophic_forgetting_resistance",
        "scaling_behavior",
    }
    assert not report.prequential_performance.individual_helper_reward.available
    assert not report.agent_prediction.helper_self_action_prediction.available
    assert (
        not report.world_and_planning.long_horizon_distributional_world_prediction.available
    )


def test_wrong_source_type_fails_closed_without_execution() -> None:
    with pytest.raises(
        HiddenLearningPartnerPropertyReadoutError,
        match="exact HiddenLearningPartnerPlanningRun",
    ):
        build_hidden_learning_partner_property_readout(run=object())  # type: ignore[arg-type]
