"""Acceptance tests for the isolated recurring pair-feature probe."""

from dataclasses import replace

import pytest

from alberta_framework.recurring_feature_gate import (
    CRITICAL_TASKS,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    PAIRWISE_PROBE_SCOPE,
    PHASE_TASKS,
    TASK_PAIRS,
    RecurringFeatureGateCriteria,
    RecurringFeatureGateError,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    run_recurring_feature_gate,
)


@pytest.fixture(scope="module")
def gate_result() -> RecurringFeatureGateResult:
    """Run the frozen protocol on thirty disjoint held-out seeds."""
    protocol = replace(RecurringFeatureProtocol(), heldout_samples=512)
    return run_recurring_feature_gate(EVIDENCE_SEEDS, protocol=protocol)


def test_canonical_probe_passes_and_beats_matched_no_retention(
    gate_result: RecurringFeatureGateResult,
) -> None:
    decision = gate_result.require_pass()

    assert decision.accepted
    assert not decision.failures
    assert gate_result.retained.all_critical_retention_rate == 1.0
    assert gate_result.retained.obsolete_eviction_rate == 1.0
    assert (
        gate_result.retained.all_critical_retention_rate
        > gate_result.no_retention.all_critical_retention_rate
    )
    assert (
        gate_result.retained.maximum_critical_task_median_nmse
        < gate_result.no_retention.maximum_critical_task_median_nmse
    )
    assert gate_result.retained.maximum_critical_task_median_nmse < 0.05
    assert gate_result.retained.median_heldout_nmse_by_task[3] > 0.8


def test_probe_is_one_uninterrupted_budgeted_online_run(
    gate_result: RecurringFeatureGateResult,
) -> None:
    protocol = gate_result.protocol
    budget = gate_result.memory_budget
    critical_pairs = set(TASK_PAIRS[:3])

    assert protocol.total_steps == 9 * 400
    assert budget.active_pair_slots == 3
    assert budget.candidate_pair_slots == 15
    assert budget.total_pair_slots == 18
    assert budget.active_output_weight_slots == 12
    assert budget.candidate_output_weight_slots == 60
    assert budget.total_output_weight_slots == 72
    assert tuple(seed.seed for seed in gate_result.retained.seeds) == EVIDENCE_SEEDS
    assert set(EVIDENCE_SEEDS).isdisjoint(DEVELOPMENT_SEEDS)
    for seed in gate_result.retained.seeds:
        assert seed.steps_seen == protocol.total_steps
        assert set(seed.active_pairs) == critical_pairs
        assert len(seed.candidate_pairs) == budget.candidate_pair_slots
        assert len(set(seed.candidate_pairs)) == budget.candidate_pair_slots
        # Eviction is from the deployed bank only: the exhaustive candidate
        # archive remains counted memory and still contains obsolete pair D.
        assert TASK_PAIRS[3] in set(seed.candidate_pairs)


def test_prequential_phases_and_recovery_are_reported(
    gate_result: RecurringFeatureGateResult,
) -> None:
    for seed in gate_result.retained.seeds:
        assert tuple(phase.task for phase in seed.phase_evidence) == PHASE_TASKS
        assert all(phase.prequential_nmse >= 0.0 for phase in seed.phase_evidence)
        assert tuple(recovery.task for recovery in seed.task_recovery[:3]) == CRITICAL_TASKS

    assert (
        gate_result.retained.median_recurrence_recovery_steps
        < gate_result.retained.median_acquisition_recovery_steps
    )


def test_acceptance_recomputes_and_fails_closed_with_evidence(
    gate_result: RecurringFeatureGateResult,
) -> None:
    impossible = RecurringFeatureGateCriteria(maximum_median_critical_nmse=0.0)
    decision = gate_result.decision(impossible)

    assert not decision.accepted
    assert any("held-out NMSE" in failure for failure in decision.failures)
    with pytest.raises(
        RecurringFeatureGateError,
        match=r"(?s)Recurring pairwise feature gate FAILED.*held-out NMSE",
    ):
        gate_result.require_pass(impossible)


def test_runner_is_deterministic_for_the_same_ordered_seeds() -> None:
    protocol = replace(RecurringFeatureProtocol(), heldout_samples=128)

    first = run_recurring_feature_gate((7, 11), protocol=protocol)
    second = run_recurring_feature_gate((7, 11), protocol=protocol)

    assert first == second


def test_underpowered_exploratory_run_fails_default_gate() -> None:
    protocol = replace(RecurringFeatureProtocol(), heldout_samples=512)
    result = run_recurring_feature_gate(range(5), protocol=protocol)

    decision = result.decision()
    assert not decision.accepted
    assert any("only 5 matched seeds" in failure for failure in decision.failures)


def test_scope_does_not_overclaim_general_continual_learning(
    gate_result: RecurringFeatureGateResult,
) -> None:
    assert gate_result.scope == PAIRWISE_PROBE_SCOPE
    assert "Pairwise Gaussian/L2" in gate_result.scope
    assert "candidate archive remains counted memory" in gate_result.scope
    assert "does not establish" in gate_result.scope
    assert "Alberta Plan completion" in gate_result.scope
