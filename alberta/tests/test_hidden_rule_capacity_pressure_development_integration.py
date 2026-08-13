"""Consumed-root regression for hidden-rule selective retention."""

from __future__ import annotations

import dataclasses

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_rule_capacity_pressure_development import (
    BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
    EPSILON_GRID,
    LINEAGE_CACHE_CAPACITY,
    LINEAGE_CACHE_NO_SIGNAL,
    LINEAGE_CACHE_PREDICTIVE_RESCUE,
    MAX_CONTEXTS,
    NUM_STEPS,
    PHASE_LENGTH,
    POST_AUDIT_BASELINE,
    SELECTIVE_RETENTION_NO_SIGNAL,
    SELECTIVE_RETENTION_PAST_RECURRENCE,
    SUMMARY_WINDOW,
    CapacityPressurePanel,
    ContextLineageCacheState,
    LineageCachePairedPanel,
    PostAuditPairedPanel,
    SelectiveRetentionPairedPanel,
    run_lineage_cache_paired_intervention,
    run_post_audit_paired_intervention,
    run_selective_retention_paired_intervention,
    summarize_capacity_pressure_run,
)

pytestmark = [pytest.mark.integration, pytest.mark.development]


@pytest.fixture(scope="module")
def paired_panel() -> PostAuditPairedPanel:
    return run_post_audit_paired_intervention()


@pytest.fixture(scope="module")
def selective_panel() -> SelectiveRetentionPairedPanel:
    return run_selective_retention_paired_intervention()


@pytest.fixture(scope="module")
def lineage_panel() -> LineageCachePairedPanel:
    return run_lineage_cache_paired_intervention()


@pytest.fixture(scope="module")
def panel(paired_panel: PostAuditPairedPanel) -> CapacityPressurePanel:
    return paired_panel.baseline_calibration_panel


def _by_epsilon(panel: CapacityPressurePanel) -> dict[float, object]:
    return {summary.epsilon: summary for summary in panel.summaries}


def test_grid_is_descriptive_only_and_reproduces_consumed_root_facts(
    panel: CapacityPressurePanel,
) -> None:
    assert tuple(run.epsilon for run in panel.runs) == EPSILON_GRID
    assert tuple(summary.epsilon for summary in panel.summaries) == EPSILON_GRID
    assert panel.selection_performed is False
    assert panel.selected_epsilon is None
    summaries = {summary.epsilon: summary for summary in panel.summaries}

    low = summaries[0.05]
    assert float(low.phase_early_reward[9]) == 3.0 / SUMMARY_WINDOW
    assert not bool(jnp.any(low.final_abc_births_distinct))
    assert not bool(jnp.any(low.recurrence_birth_reuse))
    assert bool(jnp.all(jnp.sum(low.phase_eviction_counts, axis=0) == 2))

    medium_low = summaries[0.1]
    assert bool(jnp.all(medium_low.final_abc_births_distinct))
    assert not bool(jnp.any(medium_low.recurrence_birth_reuse[0]))
    assert bool(jnp.all(medium_low.recurrence_birth_reuse[1:]))

    medium = summaries[0.2]
    assert bool(jnp.all(medium.final_abc_births_distinct))
    assert bool(jnp.all(medium.recurrence_birth_reuse))
    assert float(jnp.min(medium.phase_tail_reward)) == 0.65625
    assert float(jnp.max(medium.phase_tail_reward)) == 0.8125

    high = summaries[0.4]
    assert bool(jnp.all(high.final_abc_births_distinct))
    assert bool(jnp.all(high.recurrence_birth_reuse))
    assert high.overall_reward < medium.overall_reward


def test_life_is_uninterrupted_oracle_free_and_all_clocks_are_exact(
    panel: CapacityPressurePanel,
) -> None:
    expected_pre = jnp.stack(
        (
            jnp.zeros((NUM_STEPS,), dtype=jnp.uint32),
            jnp.arange(NUM_STEPS, dtype=jnp.uint32),
        ),
        axis=1,
    )
    expected_post = expected_pre.at[:, 1].add(jnp.asarray(1, dtype=jnp.uint32))
    terminal = jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32)
    for run in panel.runs:
        trace = run.trace
        assert trace.reward.shape == (NUM_STEPS,)
        assert trace.actions.shape == (NUM_STEPS, 2)
        assert trace.pre_context_birth_words.shape == (NUM_STEPS, 2, 2)
        assert trace.post_context_birth_words.shape == (NUM_STEPS, 2, 2)
        assert bool(jnp.all(trace.update_applied))
        assert bool(jnp.all(trace.environment_update_proposed))
        assert bool(jnp.all(trace.context_updates_proposed))
        assert bool(jnp.all(trace.controller_updates_proposed))
        assert bool(jnp.all(trace.source_clocks_aligned))
        assert bool(jnp.all(trace.candidate_clocks_aligned))
        assert bool(jnp.all(trace.source_state_finite))
        assert bool(jnp.all(trace.candidate_state_finite))
        chex.assert_trees_all_equal(trace.pre_step_words, expected_pre)
        chex.assert_trees_all_equal(trace.post_step_words, expected_post)
        chex.assert_trees_all_equal(
            trace.pre_context_slots[1:],
            trace.post_context_slots[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_context_birth_words[1:],
            trace.post_context_birth_words[:-1],
        )
        for boundary in range(PHASE_LENGTH, NUM_STEPS, PHASE_LENGTH):
            chex.assert_trees_all_equal(
                trace.pre_context_birth_words[boundary],
                trace.post_context_birth_words[boundary - 1],
            )
        chex.assert_trees_all_equal(run.final_state.environment.step_words, terminal)
        for controller in (run.final_state.controller_0, run.final_state.controller_1):
            chex.assert_trees_all_equal(controller.step_words, terminal)
            assert int(controller.step_count) == NUM_STEPS
        for context in (run.final_state.context_0, run.final_state.context_1):
            chex.assert_trees_all_equal(context.step_words, terminal)
            assert int(context.step_count) == NUM_STEPS


def test_both_agents_and_both_context_banks_learn_independent_state(
    panel: CapacityPressurePanel,
) -> None:
    for run in panel.runs:
        left = run.final_state.controller_0
        right = run.final_state.controller_1
        assert bool(jnp.any(left.q_weights != 0.0))
        assert bool(jnp.any(right.q_weights != 0.0))
        assert bool(jnp.any(left.q_weights != right.q_weights))
        left_context = run.final_state.context_0
        right_context = run.final_state.context_1
        assert bool(jnp.any(left_context.reward_weights != 0.5))
        assert bool(jnp.any(right_context.reward_weights != 0.5))
        assert bool(jnp.any(left_context.reward_weights != right_context.reward_weights))


def test_birth_ledger_exactly_stamps_allocations_and_separates_reuse(
    panel: CapacityPressurePanel,
) -> None:
    for run, summary in zip(panel.runs, panel.summaries, strict=True):
        trace = run.trace
        allocations = np.asarray(trace.allocations)
        switches = np.asarray(trace.switches)
        evictions = np.asarray(trace.evictions)
        reuses = np.asarray(trace.reuses)
        post_births = np.asarray(trace.post_context_birth_words)
        post_words = np.broadcast_to(
            np.asarray(trace.post_step_words)[:, None, :],
            post_births.shape,
        )
        assert np.array_equal(switches, allocations | reuses)
        assert not np.any(allocations & reuses)
        assert not np.any(evictions & ~allocations)
        assert np.array_equal(post_births[allocations], post_words[allocations])
        for agent_index in range(2):
            allocated_births = post_births[allocations[:, agent_index], agent_index]
            assert len({tuple(row) for row in allocated_births.tolist()}) == len(
                allocated_births
            )
        assert np.array_equal(
            np.asarray(summary.phase_switch_counts).sum(axis=0),
            switches.sum(axis=0),
        )
        assert np.array_equal(
            np.asarray(summary.phase_allocation_counts).sum(axis=0),
            allocations.sum(axis=0),
        )
        assert np.array_equal(
            np.asarray(summary.phase_eviction_counts).sum(axis=0),
            evictions.sum(axis=0),
        )
        assert np.array_equal(
            np.asarray(summary.phase_reuse_counts).sum(axis=0),
            reuses.sum(axis=0),
        )


def test_resources_work_and_common_random_numbers_are_exactly_matched(
    panel: CapacityPressurePanel,
) -> None:
    assert panel.resources_matched is True
    assert panel.work_matched is True
    assert panel.common_random_numbers.key_streams_equal_across_arms is True
    assert panel.common_random_numbers.selection_calls_per_agent == NUM_STEPS + 1
    assert panel.common_random_numbers.branch_independent_key_advance is True
    assert panel.common_random_numbers.environment_randomness_consumed is False
    assert len(set(panel.common_random_numbers.agent_key_stream_sha256)) == 2
    for digest in panel.common_random_numbers.agent_key_stream_sha256:
        assert len(digest) == 64

    budget = panel.runs[0].resource_budget
    assert budget.fixed_shape is True
    assert budget.replay_capacity == 0
    assert budget.max_context_slots_per_agent == MAX_CONTEXTS
    assert budget.joint_agent_environment_nbytes == (
        budget.environment_nbytes
        + 2 * budget.per_agent_controller_nbytes
        + 2 * budget.per_agent_context_nbytes
    )
    assert budget.joint_evaluator_birth_ledger_nbytes == (
        2 * budget.per_agent_evaluator_birth_ledger_nbytes
    )
    assert budget.total_scan_carry_nbytes == (
        budget.joint_agent_environment_nbytes
        + budget.joint_evaluator_birth_ledger_nbytes
    )
    assert budget.joint_agent_environment_clock_nbytes == (
        budget.environment_exact_clock_nbytes
        + 2 * budget.per_agent_controller_clock_nbytes
        + 2 * budget.per_agent_context_clock_nbytes
    )
    assert dataclasses.asdict(budget) == budget.to_dict()

    work = panel.runs[0].work_budget
    assert work.transitions == NUM_STEPS
    assert work.environment_transition_proposals == NUM_STEPS
    assert work.context_update_proposals == 2 * NUM_STEPS
    assert work.controller_update_proposals == 2 * NUM_STEPS
    assert work.context_event_audits == 2 * NUM_STEPS
    assert work.atomic_commit_decisions == NUM_STEPS
    assert work.action_selection_calls == 2 * (NUM_STEPS + 1)
    assert work.replay_updates == work.reset_callbacks == 0
    assert work.fixed_work_across_epsilon_grid is True
    assert work.exploration_and_greedy_draws_both_generated is True


def test_prefix_limit_and_next_panel_are_reported_without_overclaim(
    panel: CapacityPressurePanel,
) -> None:
    boundary = panel.prefix_twin_boundary
    assert boundary.deterministic_online_guarantee_possible is False
    assert boundary.counterfactual_future_executed is False
    assert boundary.stochastic_optimality_claimed is False
    assert "unknowable without a prior" in panel.causal_gap
    design = panel.next_frozen_panel
    assert design.status == "designed-not-issued-not-executed"
    assert design.root_zero_excluded is True
    assert design.epsilon_arms == EPSILON_GRID
    assert design.promotion_claimed is False
    assert any("all four" in action for action in design.required_protocol_actions)


def test_post_audit_panel_is_eight_paired_winner_free_consumed_root_lives(
    paired_panel: PostAuditPairedPanel,
) -> None:
    assert len(paired_panel.runs) == 8
    assert tuple(
        (run.epsilon, run.condition) for run in paired_panel.runs
    ) == tuple(
        (epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in (POST_AUDIT_BASELINE, BIRTH_AUTHENTICATED_CONTROLLER_SCRUB)
    )
    assert paired_panel.selection_performed is False
    assert paired_panel.selected_epsilon is None
    assert paired_panel.resources_matched is True
    assert paired_panel.work_matched is True
    assert paired_panel.common_random_numbers.key_streams_equal_across_all_eight is True
    assert paired_panel.post_audit_only is True
    assert paired_panel.scientific_promotion_allowed is False


def test_post_audit_baseline_is_exact_original_and_scrub_events_are_authenticated(
    paired_panel: PostAuditPairedPanel,
) -> None:
    for pair in paired_panel.pairs:
        baseline = pair.baseline
        scrub = pair.scrub
        original = next(
            run
            for run in paired_panel.baseline_calibration_panel.runs
            if run.epsilon == pair.epsilon
        )
        chex.assert_trees_all_equal(baseline.trace, original.trace)
        chex.assert_trees_all_equal(baseline.final_state, original.final_state)
        assert bool(jnp.all(scrub.scrub.preparation_valid))
        assert bool(jnp.all(scrub.scrub.biases_untouched))
        assert bool(jnp.all(scrub.scrub.average_reward_untouched))
        assert bool(jnp.all(scrub.scrub.rng_untouched_before_update))
        assert bool(jnp.all(scrub.scrub.clock_untouched_before_update))
        assert bool(jnp.all(scrub.scrub.survivor_rows_untouched))
        assert bool(
            jnp.all(
                jnp.where(
                    scrub.trace.reuses,
                    scrub.scrub.prepared_controller_unchanged,
                    True,
                )
            )
        )
        assert bool(jnp.all(scrub.scrub.scrub_applied == scrub.trace.allocations))
        assert not bool(jnp.any(scrub.scrub.cross_birth_contamination_consumed))
        assert not bool(jnp.any(scrub.scrub.authentication_failed))


def test_post_audit_effects_are_reported_for_every_epsilon_without_thresholds(
    paired_panel: PostAuditPairedPanel,
) -> None:
    assert tuple(effect.epsilon for effect in paired_panel.effects) == EPSILON_GRID
    for effect in paired_panel.effects:
        assert effect.baseline_scrub_count == 0
        assert effect.scrub_scrub_count > 0
        assert effect.baseline_contamination_count >= effect.scrub_contamination_count
        assert effect.scrub_contamination_count == 0
        assert effect.scrub_prevented_count >= 0
        assert effect.baseline_phase_early_reward.shape == (10,)
        assert effect.scrub_phase_early_reward.shape == (10,)
        assert effect.baseline_phase_tail_reward.shape == (10,)
        assert effect.scrub_phase_tail_reward.shape == (10,)
        assert effect.baseline_tail_birth_modes.shape == (10, 2, 2)
        assert effect.scrub_tail_birth_modes.shape == (10, 2, 2)
        assert effect.baseline_phase_distinct_birth_counts.shape == (10, 2)
        assert effect.scrub_phase_distinct_birth_counts.shape == (10, 2)
    assert paired_panel.thresholds_used is False
    assert paired_panel.conclusion.startswith("Descriptive consumed-root post-audit")


def test_post_audit_consumed_root_effect_table_is_exact(
    paired_panel: PostAuditPairedPanel,
) -> None:
    expected = {
        0.05: {
            "rewards": (0.8567500114440918, 0.8670000433921814),
            "scrubs_crossings_prevented": (10, 4, 0, 6),
            "lifecycle": (16, 4, 2, 12, 18, 5, 3, 13),
            "birth_modes_equal": False,
            "churn": (23, 25),
        },
        0.1: {
            "rewards": (0.8167500495910645, 0.8287500143051147),
            "scrubs_crossings_prevented": (6, 4, 0, 2),
            "lifecycle": (15, 4, 2, 11, 13, 3, 1, 10),
            "birth_modes_equal": False,
            "churn": (25, 23),
        },
        0.2: {
            "rewards": (0.706250011920929, 0.7072500586509705),
            "scrubs_crossings_prevented": (6, 2, 0, 2),
            "lifecycle": (13, 3, 1, 10, 13, 3, 1, 10),
            "birth_modes_equal": True,
            "churn": (22, 22),
        },
        0.4: {
            "rewards": (0.5160000324249268, 0.5202500224113464),
            "scrubs_crossings_prevented": (6, 2, 0, 2),
            "lifecycle": (12, 3, 1, 9, 12, 3, 1, 9),
            "birth_modes_equal": True,
            "churn": (21, 21),
        },
    }
    for effect in paired_panel.effects:
        record = expected[effect.epsilon]
        assert (
            effect.baseline_overall_reward,
            effect.scrub_overall_reward,
        ) == record["rewards"]
        assert (
            effect.scrub_scrub_count,
            effect.baseline_contamination_count,
            effect.scrub_contamination_count,
            effect.scrub_prevented_count,
        ) == record["scrubs_crossings_prevented"]
        lifecycle = tuple(
            int(jnp.sum(values[:, 0]))
            for values in (
                effect.baseline_switch_counts,
                effect.baseline_allocation_counts,
                effect.baseline_eviction_counts,
                effect.baseline_reuse_counts,
                effect.scrub_switch_counts,
                effect.scrub_allocation_counts,
                effect.scrub_eviction_counts,
                effect.scrub_reuse_counts,
            )
        )
        assert lifecycle == record["lifecycle"]
        assert bool(
            jnp.all(effect.baseline_tail_birth_modes == effect.scrub_tail_birth_modes)
        ) is record["birth_modes_equal"]
        assert (
            int(jnp.sum(effect.baseline_phase_distinct_birth_counts[:, 0])),
            int(jnp.sum(effect.scrub_phase_distinct_birth_counts[:, 0])),
        ) == record["churn"]


def test_post_audit_exact_clocks_rng_resources_and_work_remain_matched(
    paired_panel: PostAuditPairedPanel,
) -> None:
    terminal = jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32)
    for run in paired_panel.runs:
        chex.assert_trees_all_equal(run.final_state.environment.step_words, terminal)
        chex.assert_trees_all_equal(run.final_state.controller_0.step_words, terminal)
        chex.assert_trees_all_equal(run.final_state.controller_1.step_words, terminal)
        chex.assert_trees_all_equal(run.final_state.context_0.step_words, terminal)
        chex.assert_trees_all_equal(run.final_state.context_1.step_words, terminal)
        assert run.resource_budget.intervention_persistent_nbytes == 0
        assert run.resource_budget.logical_transient_scrub_candidate_nbytes == (
            2 * run.resource_budget.base.per_agent_controller_nbytes
        )
        assert run.work_budget.birth_authentication_audits == 2 * NUM_STEPS
        assert run.work_budget.scrub_candidate_constructions == 2 * NUM_STEPS
        assert run.work_budget.baseline_controller_updates == 2 * NUM_STEPS
        assert run.work_budget.scrub_controller_updates == 2 * NUM_STEPS
        assert run.work_budget.context_event_audits == 2 * NUM_STEPS
        assert run.work_budget.action_selection_calls == 4 * NUM_STEPS + 2
        assert run.work_budget.fixed_work_across_conditions is True
    for pair in paired_panel.pairs:
        chex.assert_trees_all_equal(
            pair.baseline.trace.controller_rng_key_words,
            pair.scrub.trace.controller_rng_key_words,
        )


def test_selective_panel_is_eight_matched_consumed_root_lives_without_selection(
    selective_panel: SelectiveRetentionPairedPanel,
) -> None:
    assert len(selective_panel.runs) == 8
    assert tuple((run.epsilon, run.condition) for run in selective_panel.runs) == tuple(
        (epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in (
            SELECTIVE_RETENTION_NO_SIGNAL,
            SELECTIVE_RETENTION_PAST_RECURRENCE,
        )
    )
    assert selective_panel.resources_matched is True
    assert selective_panel.work_matched is True
    assert selective_panel.no_signal_is_controller_scrub_baseline is True
    assert selective_panel.past_only_score == (
        "authenticated_current_birth_occurrences_minus_one"
    )
    assert selective_panel.prefix_twin_first_eviction_resolved is False
    assert selective_panel.development_only is True
    assert selective_panel.scientific_promotion_allowed is False
    assert selective_panel.thresholds_used is False
    assert selective_panel.selection_performed is False
    assert selective_panel.selected_epsilon is None
    assert selective_panel.common_random_numbers.key_streams_equal_across_all_eight
    assert "no threshold, tuning, winner" in selective_panel.conclusion


def test_selective_no_signal_is_bit_exact_controller_scrub_baseline(
    paired_panel: PostAuditPairedPanel,
    selective_panel: SelectiveRetentionPairedPanel,
) -> None:
    for post_pair, selective_pair in zip(
        paired_panel.pairs,
        selective_panel.pairs,
        strict=True,
    ):
        assert post_pair.epsilon == selective_pair.epsilon
        chex.assert_trees_all_equal(
            selective_pair.no_signal.capacity_run.trace,
            post_pair.scrub.capacity_run.trace,
        )
        chex.assert_trees_all_equal(
            selective_pair.no_signal.capacity_run.final_state,
            post_pair.scrub.capacity_run.final_state,
        )


def test_selective_history_exactly_tracks_birth_resets_and_stored_recurrences(
    selective_panel: SelectiveRetentionPairedPanel,
) -> None:
    for run in selective_panel.runs:
        trace = run.trace
        capacity = trace.capacity
        assert bool(jnp.all(capacity.update_applied))
        assert bool(jnp.all(trace.history_source_valid))
        assert bool(jnp.all(trace.history_candidate_valid))
        assert bool(jnp.all(trace.history_capacity_available))
        assert bool(jnp.all(trace.history_updates_proposed))
        assert bool(jnp.all(trace.priority_inputs_valid))
        assert not bool(jnp.any(trace.scrub.authentication_failed))
        chex.assert_trees_all_equal(
            trace.history_allocation_resets,
            capacity.allocations,
        )
        chex.assert_trees_all_equal(
            trace.history_stored_recurrences,
            capacity.reuses,
        )
        chex.assert_trees_all_equal(
            trace.pre_occurrence_words[1:],
            trace.post_occurrence_words[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_last_entry_words[1:],
            trace.post_last_entry_words[:-1],
        )
        occurrence_low = trace.pre_occurrence_words[..., 1]
        expected_scores = jnp.where(
            occurrence_low > 0,
            occurrence_low - jnp.asarray(1, dtype=jnp.uint32),
            jnp.asarray(0, dtype=jnp.uint32),
        ).astype(jnp.float32)
        chex.assert_trees_all_equal(
            trace.raw_completed_recurrence_scores,
            expected_scores,
        )
        if run.condition == SELECTIVE_RETENTION_NO_SIGNAL:
            assert not bool(jnp.any(trace.protection_enabled))
            assert not bool(jnp.any(trace.dispatched_eviction_protection))
            assert not bool(jnp.any(trace.eviction_targets_adjusted))
        else:
            assert bool(jnp.all(trace.protection_enabled))
            chex.assert_trees_all_equal(
                trace.dispatched_eviction_protection,
                trace.raw_completed_recurrence_scores,
            )
        assert not bool(
            jnp.any(trace.eviction_targets_adjusted & ~trace.eviction_protection_used)
        )
        assert not bool(
            jnp.any(trace.eviction_protection_used & ~trace.full_bank_evictions_requested)
        )

        allocations = np.asarray(capacity.allocations)
        reuses = np.asarray(capacity.reuses)
        slots = np.asarray(capacity.post_context_slots)
        pre_count = np.asarray(trace.pre_occurrence_words)
        post_count = np.asarray(trace.post_occurrence_words)
        pre_entry = np.asarray(trace.pre_last_entry_words)
        post_entry = np.asarray(trace.post_last_entry_words)
        post_interval = np.asarray(trace.post_last_interval_words)
        post_step = np.asarray(capacity.post_step_words)
        for step in range(NUM_STEPS):
            for agent_index in range(2):
                target = int(slots[step, agent_index])
                if allocations[step, agent_index]:
                    assert np.array_equal(post_count[step, agent_index, target], (0, 1))
                    assert np.array_equal(
                        post_entry[step, agent_index, target],
                        post_step[step],
                    )
                    assert np.array_equal(
                        post_interval[step, agent_index, target],
                        (0, 0),
                    )
                elif reuses[step, agent_index]:
                    assert int(post_count[step, agent_index, target, 0]) == 0
                    assert int(pre_count[step, agent_index, target, 0]) == 0
                    assert int(post_count[step, agent_index, target, 1]) == (
                        int(pre_count[step, agent_index, target, 1]) + 1
                    )
                    assert np.array_equal(
                        post_entry[step, agent_index, target],
                        post_step[step],
                    )
                    assert int(post_interval[step, agent_index, target, 0]) == 0
                    assert int(post_interval[step, agent_index, target, 1]) == (
                        int(post_step[step, 1])
                        - int(pre_entry[step, agent_index, target, 1])
                    )
                else:
                    assert np.array_equal(
                        post_count[step, agent_index],
                        pre_count[step, agent_index],
                    )

        final = run.final_state
        for context, ledger, history in (
            (final.base.context_0, final.base.ledger_0, final.recurrence_0),
            (final.base.context_1, final.base.ledger_1, final.recurrence_1),
        ):
            chex.assert_trees_all_equal(
                history.bound_birth_words,
                ledger.slot_birth_words,
            )
            live_counts = jnp.any(history.occurrence_words != 0, axis=1)
            chex.assert_trees_all_equal(live_counts, context.in_use)


def test_selective_consumed_root_effect_table_is_exact_and_bounded(
    selective_panel: SelectiveRetentionPairedPanel,
) -> None:
    expected = {
        0.05: {
            "rewards": (0.8670000433921814, 0.8695000410079956),
            "lifecycle": (18, 5, 3, 13, 16, 4, 2, 12),
            "full_adjust_avoid": (6, 4, 2, 8.0),
            "birth_modes_equal": False,
            "churn": (25, 24),
        },
        0.1: {
            "rewards": (0.8287500143051147, 0.8287500143051147),
            "lifecycle": (13, 3, 1, 10, 13, 3, 1, 10),
            "full_adjust_avoid": (2, 2, 0, 0.0),
            "birth_modes_equal": True,
            "churn": (23, 23),
        },
        0.2: {
            "rewards": (0.7072500586509705, 0.7072500586509705),
            "lifecycle": (13, 3, 1, 10, 13, 3, 1, 10),
            "full_adjust_avoid": (2, 2, 0, 0.0),
            "birth_modes_equal": True,
            "churn": (22, 22),
        },
        0.4: {
            "rewards": (0.5202500224113464, 0.5202500224113464),
            "lifecycle": (12, 3, 1, 9, 12, 3, 1, 9),
            "full_adjust_avoid": (2, 2, 0, 0.0),
            "birth_modes_equal": True,
            "churn": (21, 21),
        },
    }
    for pair, effect in zip(selective_panel.pairs, selective_panel.effects, strict=True):
        record = expected[effect.epsilon]
        assert (
            effect.no_signal_overall_reward,
            effect.past_recurrence_overall_reward,
        ) == record["rewards"]
        lifecycle = tuple(
            int(jnp.sum(values[:, 0]))
            for values in (
                effect.no_signal_phase_switch_counts,
                effect.no_signal_phase_allocation_counts,
                effect.no_signal_phase_eviction_counts,
                effect.no_signal_phase_reuse_counts,
                effect.past_recurrence_phase_switch_counts,
                effect.past_recurrence_phase_allocation_counts,
                effect.past_recurrence_phase_eviction_counts,
                effect.past_recurrence_phase_reuse_counts,
            )
        )
        assert lifecycle == record["lifecycle"]
        assert (
            effect.no_signal_full_bank_eviction_count,
            effect.past_recurrence_full_bank_eviction_count,
            effect.past_recurrence_adjusted_target_count,
            effect.avoided_completed_recurrence_intervals,
        ) == record["full_adjust_avoid"]
        assert effect.no_signal_adjusted_target_count == 0
        assert effect.nonzero_selected_eviction_score_count == 0
        assert bool(
            jnp.all(
                effect.no_signal_tail_birth_modes
                == effect.past_recurrence_tail_birth_modes
            )
        ) is record["birth_modes_equal"]
        no_signal_summary = summarize_capacity_pressure_run(pair.no_signal.capacity_run)
        recurrence_summary = summarize_capacity_pressure_run(
            pair.past_recurrence.capacity_run
        )
        assert (
            int(jnp.sum(no_signal_summary.phase_distinct_birth_counts[:, 0])),
            int(jnp.sum(recurrence_summary.phase_distinct_birth_counts[:, 0])),
        ) == record["churn"]
    assert selective_panel.effects[0].past_recurrence_minus_no_signal_overall_reward == (
        0.002499997615814209
    )
    assert all(
        effect.past_recurrence_minus_no_signal_overall_reward == 0.0
        for effect in selective_panel.effects[1:]
    )


def test_selective_first_capacity_decision_remains_prefix_limited_and_resources_exact(
    selective_panel: SelectiveRetentionPairedPanel,
) -> None:
    terminal = jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32)
    for pair in selective_panel.pairs:
        trace = pair.past_recurrence.trace
        first = int(jnp.where(jnp.any(trace.full_bank_evictions_requested, axis=1))[0][0])
        assert not bool(jnp.any(trace.eviction_targets_adjusted[first]))
        assert bool(
            jnp.all(
                trace.ordinary_lru_completed_recurrence_scores[first]
                == trace.selected_completed_recurrence_scores[first]
            )
        )
    for run in selective_panel.runs:
        chex.assert_trees_all_equal(run.final_state.base.environment.step_words, terminal)
        budget = run.resource_budget
        assert budget.per_agent_recurrence_history_nbytes == 96
        assert budget.joint_recurrence_history_nbytes == 192
        assert budget.total_scan_carry_nbytes == (
            budget.base.total_scan_carry_nbytes + 192
        )
        assert budget.logical_transient_protection_nbytes == 48
        assert budget.logical_transient_scrub_candidate_nbytes == (
            2 * budget.base.per_agent_controller_nbytes
        )
        assert budget.replay_capacity == 0
        assert budget.fixed_shape is True
        work = run.work_budget
        assert work.prioritized_context_update_proposals == 2 * NUM_STEPS
        assert work.recurrence_score_computations == 2 * NUM_STEPS
        assert work.recurrence_history_audits == 2 * NUM_STEPS
        assert work.recurrence_history_proposals == 2 * NUM_STEPS
        assert work.birth_authentication_audits == 2 * NUM_STEPS
        assert work.scrub_candidate_constructions == 2 * NUM_STEPS
        assert work.controller_update_proposals == 2 * NUM_STEPS
        assert work.action_selection_calls == 2 * (NUM_STEPS + 1)
        assert work.fixed_work_across_conditions is True
        assert work.no_signal_still_computes_history is True
        assert work.replay_updates == work.reset_callbacks == 0
    for pair in selective_panel.pairs:
        chex.assert_trees_all_equal(
            pair.no_signal.capacity_run.trace.controller_rng_key_words,
            pair.past_recurrence.capacity_run.trace.controller_rng_key_words,
        )


def test_lineage_panel_is_exactly_eight_bounded_nonpromoting_lives(
    lineage_panel: LineageCachePairedPanel,
) -> None:
    assert len(lineage_panel.runs) == 8
    assert tuple((run.epsilon, run.condition) for run in lineage_panel.runs) == tuple(
        (epsilon, condition)
        for epsilon in EPSILON_GRID
        for condition in (
            LINEAGE_CACHE_NO_SIGNAL,
            LINEAGE_CACHE_PREDICTIVE_RESCUE,
        )
    )
    assert lineage_panel.resources_matched is True
    assert lineage_panel.work_matched is True
    assert lineage_panel.no_signal_is_controller_scrub_baseline is True
    assert lineage_panel.cache_capacity_per_agent == LINEAGE_CACHE_CAPACITY == 1
    assert lineage_panel.task_labels_used is False
    assert lineage_panel.configurable_match_threshold_used is False
    assert lineage_panel.score_source == (
        "exact_cross_birth_strict_predictive_rescue_count"
    )
    assert lineage_panel.prefix_twin_first_eviction_resolved is False
    assert lineage_panel.development_only is True
    assert lineage_panel.scientific_promotion_allowed is False
    assert lineage_panel.selection_performed is False
    assert lineage_panel.selected_epsilon is None
    assert lineage_panel.common_random_numbers.key_streams_equal_across_all_eight
    assert "No threshold, task label, tuning" in lineage_panel.conclusion
    assert "one just-observed transition is insufficient" in lineage_panel.conclusion

    # Agent namespace is an external component of identity.  Each independent
    # sidecar stores only exact words and never pays bytes for a task/agent ID.
    sidecar_fields = set(ContextLineageCacheState.__annotations__)
    assert "namespace" not in sidecar_fields
    assert "agent_id" not in sidecar_fields


def test_lineage_no_signal_is_bit_exact_controller_scrub_baseline(
    paired_panel: PostAuditPairedPanel,
    lineage_panel: LineageCachePairedPanel,
) -> None:
    for post_pair, lineage_pair in zip(
        paired_panel.pairs,
        lineage_panel.pairs,
        strict=True,
    ):
        assert post_pair.epsilon == lineage_pair.epsilon
        chex.assert_trees_all_equal(
            lineage_pair.no_signal.capacity_run.trace,
            post_pair.scrub.capacity_run.trace,
        )
        chex.assert_trees_all_equal(
            lineage_pair.no_signal.capacity_run.final_state,
            post_pair.scrub.capacity_run.final_state,
        )


def test_lineage_trace_is_atomic_past_only_and_birth_authenticated(
    lineage_panel: LineageCachePairedPanel,
) -> None:
    for run in lineage_panel.runs:
        trace = run.trace
        capacity = trace.capacity
        assert bool(jnp.all(capacity.update_applied))
        assert bool(jnp.all(trace.source_scores_fixed_before_outcome))
        assert not bool(jnp.any(trace.outcome_routed_to_current_protection))
        assert bool(jnp.all(trace.lineage_source_valid))
        assert bool(jnp.all(trace.lineage_candidate_valid))
        assert bool(jnp.all(trace.rescue_capacity_available))
        assert bool(jnp.all(trace.lineage_updates_proposed))
        assert bool(jnp.all(trace.scrub_preparations_valid))
        chex.assert_trees_all_equal(
            trace.pre_live_lineage_words[1:],
            trace.post_live_lineage_words[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_live_rescue_words[1:],
            trace.post_live_rescue_words[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_cache_valid[1:],
            trace.post_cache_valid[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_cache_source_birth_words[1:],
            trace.post_cache_source_birth_words[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_cache_lineage_words[1:],
            trace.post_cache_lineage_words[:-1],
        )
        chex.assert_trees_all_equal(
            trace.pre_cache_rescue_words[1:],
            trace.post_cache_rescue_words[:-1],
        )
        expected_scores = trace.pre_live_rescue_words[..., 1].astype(jnp.float32)
        chex.assert_trees_all_equal(trace.raw_predictive_rescue_scores, expected_scores)
        if run.condition == LINEAGE_CACHE_NO_SIGNAL:
            assert not bool(jnp.any(trace.protection_enabled))
            assert not bool(jnp.any(trace.dispatched_eviction_protection))
        else:
            assert bool(jnp.all(trace.protection_enabled))
            chex.assert_trees_all_equal(
                trace.dispatched_eviction_protection,
                trace.raw_predictive_rescue_scores,
            )
        assert not bool(jnp.any(trace.cache_matched ^ trace.lineage_transferred))
        assert not bool(jnp.any(trace.cache_matched ^ trace.rescue_incremented))
        assert not bool(
            jnp.any(trace.cache_matched & ~trace.strict_predictive_dominance)
        )
        assert not bool(
            jnp.any(trace.eviction_protection_used & ~trace.full_bank_evictions_requested)
        )
        assert not bool(
            jnp.any(trace.eviction_targets_adjusted & ~trace.eviction_protection_used)
        )

        final = run.final_state
        for context, ledger, sidecar in (
            (final.base.context_0, final.base.ledger_0, final.lineage_0),
            (final.base.context_1, final.base.ledger_1, final.lineage_1),
        ):
            chex.assert_trees_all_equal(
                sidecar.bound_birth_words,
                ledger.slot_birth_words,
            )
            live_lineages = np.asarray(sidecar.live_lineage_words)[
                np.asarray(context.in_use)
            ]
            assert len({tuple(row) for row in live_lineages.tolist()}) == len(
                live_lineages
            )
            if bool(sidecar.cache_valid):
                cache_lineage = tuple(np.asarray(sidecar.cache_lineage_words).tolist())
                assert cache_lineage not in {
                    tuple(row) for row in live_lineages.tolist()
                }


def test_lineage_consumed_panel_is_an_exact_null_strict_match_result(
    lineage_panel: LineageCachePairedPanel,
) -> None:
    expected = {
        0.05: {
            "reward": 0.8670000433921814,
            "tested_archived_retained": (4, 4, 2),
            "decomposition": (6, 4, 4, 2, 2, 2, 4, 4, 2),
            "lifecycle": (10, 6, 26),
        },
        0.1: {
            "reward": 0.8287500143051147,
            "tested_archived_retained": (0, 2, 0),
            "decomposition": (2, 0, 0, 0, 0, 0, 0, 2, 0),
            "lifecycle": (6, 2, 20),
        },
        0.2: {
            "reward": 0.7072500586509705,
            "tested_archived_retained": (0, 2, 0),
            "decomposition": (2, 0, 0, 0, 0, 0, 0, 2, 0),
            "lifecycle": (6, 2, 20),
        },
        0.4: {
            "reward": 0.5202500224113464,
            "tested_archived_retained": (0, 2, 0),
            "decomposition": (2, 0, 0, 0, 0, 0, 0, 2, 0),
            "lifecycle": (6, 2, 18),
        },
    }
    for pair, effect in zip(lineage_panel.pairs, lineage_panel.effects, strict=True):
        record = expected[effect.epsilon]
        assert (
            effect.no_signal_overall_reward,
            effect.predictive_rescue_overall_reward,
        ) == (record["reward"], record["reward"])
        assert effect.predictive_rescue_minus_no_signal_overall_reward == 0.0
        assert (
            effect.no_signal_cache_match_count,
            effect.predictive_rescue_cache_match_count,
            effect.no_signal_rescue_increment_count,
            effect.predictive_rescue_rescue_increment_count,
            effect.no_signal_adjusted_target_count,
            effect.predictive_rescue_adjusted_target_count,
            effect.nonzero_selected_predictive_rescue_count,
            effect.avoided_predictive_rescues,
        ) == (0, 0, 0, 0, 0, 0, 0, 0.0)
        assert dataclasses.astuple(effect.no_signal_failure_decomposition) == record[
            "decomposition"
        ]
        assert dataclasses.astuple(
            effect.predictive_rescue_failure_decomposition
        ) == record["decomposition"]
        for run in (pair.no_signal, pair.predictive_rescue):
            trace = run.trace
            assert (
                int(jnp.sum(trace.cache_tested)),
                int(jnp.sum(trace.victim_archived)),
                int(jnp.sum(trace.old_cache_retained)),
            ) == record["tested_archived_retained"]
            assert not bool(jnp.any(trace.strict_predictive_dominance))
            assert not bool(jnp.any(trace.cache_matched))
            assert (
                int(jnp.sum(trace.capacity.allocations)),
                int(jnp.sum(trace.capacity.evictions)),
                int(jnp.sum(trace.capacity.reuses)),
            ) == record["lifecycle"]
        chex.assert_trees_all_equal(
            pair.no_signal.capacity_run.trace,
            pair.predictive_rescue.capacity_run.trace,
        )
        chex.assert_trees_all_equal(
            pair.no_signal.capacity_run.final_state,
            pair.predictive_rescue.capacity_run.final_state,
        )


def test_lineage_resource_work_key_parity_and_prefix_limit_are_exact(
    lineage_panel: LineageCachePairedPanel,
) -> None:
    terminal = jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32)
    for pair in lineage_panel.pairs:
        trace = pair.predictive_rescue.trace
        first = int(jnp.where(jnp.any(trace.full_bank_evictions_requested, axis=1))[0][0])
        assert not bool(jnp.any(trace.raw_predictive_rescue_scores[first]))
        assert not bool(jnp.any(trace.eviction_targets_adjusted[first]))
        chex.assert_trees_all_equal(
            pair.no_signal.capacity_run.trace.controller_rng_key_words,
            pair.predictive_rescue.capacity_run.trace.controller_rng_key_words,
        )
    for run in lineage_panel.runs:
        chex.assert_trees_all_equal(run.final_state.base.environment.step_words, terminal)
        budget = run.resource_budget
        assert budget.cache_capacity_per_agent == 1
        assert budget.per_agent_lineage_cache_nbytes == 161
        assert budget.joint_lineage_cache_nbytes == 322
        assert budget.total_scan_carry_nbytes == (
            budget.base.total_scan_carry_nbytes + 322
        )
        assert budget.logical_transient_protection_nbytes == 48
        assert budget.logical_transient_prediction_nbytes == 80
        assert budget.logical_transient_lineage_candidate_nbytes == 322
        assert budget.logical_transient_scrub_candidate_nbytes == (
            2 * budget.base.per_agent_controller_nbytes
        )
        assert budget.replay_capacity == 0
        assert budget.fixed_shape is True
        work = run.work_budget
        assert work.prioritized_context_update_proposals == 2 * NUM_STEPS
        assert work.source_rescue_score_computations == 2 * NUM_STEPS
        assert work.cache_match_computations == 2 * NUM_STEPS
        assert work.live_model_error_comparisons == 2 * MAX_CONTEXTS * NUM_STEPS
        assert work.lineage_cache_audits == 2 * NUM_STEPS
        assert work.lineage_cache_proposals == 2 * NUM_STEPS
        assert work.birth_authentication_audits == 2 * NUM_STEPS
        assert work.scrub_candidate_constructions == 2 * NUM_STEPS
        assert work.controller_update_proposals == 2 * NUM_STEPS
        assert work.action_selection_calls == 2 * (NUM_STEPS + 1)
        assert work.atomic_commit_decisions == NUM_STEPS
        assert work.fixed_work_across_conditions is True
        assert work.no_signal_still_computes_cache_match is True
        assert work.replay_updates == work.reset_callbacks == 0
