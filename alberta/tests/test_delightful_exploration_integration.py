"""Integration diagnostics for fixed-budget WP5.6 exploration selection."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.prospective_exploration import (
    PROSPECTIVE_EXPLORATION_MODES,
    ExplorationCandidateBatch,
    ProspectiveExploration,
    ProspectiveExplorationConfig,
    run_prospective_exploration_from_batches,
)

pytestmark = pytest.mark.integration


def _digest(index: int) -> tuple[int, ...]:
    return (index, index + 1, index + 2, index + 3, 0, 0, 0, 0)


def _config(mode: str = "expected_improvement_surprisal", epsilon: float = 0.0):
    return ProspectiveExplorationConfig(
        n_actions=4,
        candidate_budget=3,
        mode=mode,  # type: ignore[arg-type]
        epsilon=epsilon,
        host_surprisal_cap=4.0,
        max_expected_improvement=100.0,
        max_ensemble_disagreement=100.0,
        max_information_gain=100.0,
        max_learning_progress=100.0,
        source_owner_digest=_digest(10),
        host_policy_owner_digest=_digest(20),
        candidate_owner_digest=_digest(30),
        score_owner_digest=_digest(40),
        safety_owner_digest=_digest(50),
    )


def _batch(
    config: ProspectiveExplorationConfig,
    event: int,
    *,
    host_policy: tuple[float, float, float, float] = (0.6, 0.3, 0.1, 0.0),
    expected: tuple[float, float, float] = (1.0, 2.0, 1.0),
    disagreement: tuple[float, float, float] = (3.0, 2.0, 1.0),
    information_gain: tuple[float, float, float] = (1.0, 3.0, 2.0),
    progress: tuple[float, float, float] = (1.0, 2.0, 3.0),
    safety: tuple[bool, bool, bool] = (True, True, True),
) -> ExplorationCandidateBatch:
    source = jnp.asarray([0, event], dtype=jnp.uint32)
    return ExplorationCandidateBatch(
        candidate_actions=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        candidate_identity_words=jnp.asarray(
            [[event, 101], [event, 102], [event, 103]],
            dtype=jnp.uint32,
        ),
        candidate_valid=jnp.asarray([True, True, True], dtype=jnp.bool_),
        host_policy=jnp.asarray(host_policy, dtype=jnp.float32),
        host_action=jnp.asarray(0, dtype=jnp.int32),
        expected_improvement=jnp.asarray(expected, dtype=jnp.float32),
        ensemble_disagreement=jnp.asarray(disagreement, dtype=jnp.float32),
        information_gain=jnp.asarray(information_gain, dtype=jnp.float32),
        learning_progress=jnp.asarray(progress, dtype=jnp.float32),
        candidate_safety_allowed=jnp.asarray(safety, dtype=jnp.bool_),
        host_action_safety_allowed=jnp.asarray(True, dtype=jnp.bool_),
        source_event_words=source,
        candidate_source_event_words=source,
        score_source_event_words=source,
        host_policy_source_event_words=source,
        safety_source_event_words=source,
        host_policy_revision_words=jnp.asarray([0, 5], dtype=jnp.uint32),
        candidate_revision_words=jnp.asarray([0, 7], dtype=jnp.uint32),
        score_revision_words=jnp.asarray([0, event], dtype=jnp.uint32),
        safety_revision_words=jnp.asarray([0, 9], dtype=jnp.uint32),
        source_owner_digest=jnp.asarray(config.source_owner_digest, dtype=jnp.uint32),
        host_policy_owner_digest=jnp.asarray(
            config.host_policy_owner_digest,
            dtype=jnp.uint32,
        ),
        candidate_owner_digest=jnp.asarray(
            config.candidate_owner_digest,
            dtype=jnp.uint32,
        ),
        score_owner_digest=jnp.asarray(config.score_owner_digest, dtype=jnp.uint32),
        safety_owner_digest=jnp.asarray(config.safety_owner_digest, dtype=jnp.uint32),
        causal_pre_decision_attested=jnp.asarray(True, dtype=jnp.bool_),
    )


def _stack_batches(batches: tuple[ExplorationCandidateBatch, ...]) -> ExplorationCandidateBatch:
    return jax.tree.map(lambda *values: jnp.stack(values), *batches)


def test_scan_matches_eager_transactions_and_compiled_scan_bit_exactly() -> None:
    config = _config("expected_improvement_surprisal")
    controller = ProspectiveExploration(config)
    initial = controller.init(jr.key(100))
    items = tuple(
        _batch(
            config,
            event,
            expected=(1.0 + event, 2.0, 0.5 * event),
            safety=(True, event % 2 == 0, True),
        )
        for event in range(1, 6)
    )
    stacked = _stack_batches(items)

    state = initial
    indices: list[jax.Array] = []
    actions: list[jax.Array] = []
    proposals: list[jax.Array] = []
    selected_scores: list[jax.Array] = []
    applied: list[jax.Array] = []
    shielded: list[jax.Array] = []
    available: list[jax.Array] = []
    for item in items:
        result = controller.select(state, item)
        state = result.state
        indices.append(result.selected_index)
        actions.append(result.selected_candidate_action)
        proposals.append(result.proposed_executable_action)
        selected_scores.append(result.selected_expected_improvement_surprisal_score)
        applied.append(result.decision_applied)
        shielded.append(result.candidate_passed_hard_shield)
        available.append(result.proposed_executable_action_available)

    scanned = run_prospective_exploration_from_batches(controller, initial, stacked)
    compiled = jax.jit(
        lambda carry, receipts: run_prospective_exploration_from_batches(
            controller,
            carry,
            receipts,
        )
    )(initial, stacked)
    chex.assert_trees_all_equal(scanned, compiled)
    chex.assert_trees_all_equal(scanned.state, state)
    np.testing.assert_array_equal(scanned.selected_indices, jnp.stack(indices))
    np.testing.assert_array_equal(scanned.selected_candidate_actions, jnp.stack(actions))
    np.testing.assert_array_equal(scanned.proposed_executable_actions, jnp.stack(proposals))
    np.testing.assert_array_equal(
        scanned.selected_expected_improvement_surprisal_score,
        jnp.stack(selected_scores),
    )
    np.testing.assert_array_equal(scanned.decision_applied, jnp.stack(applied))
    np.testing.assert_array_equal(scanned.candidate_passed_hard_shield, jnp.stack(shielded))
    np.testing.assert_array_equal(
        scanned.proposed_executable_action_available,
        jnp.stack(available),
    )


def test_all_comparators_consume_one_identical_fixed_budget_receipt() -> None:
    selected: dict[str, int] = {}
    resources: list[tuple[int, int, int]] = []
    for mode in PROSPECTIVE_EXPLORATION_MODES:
        epsilon = 0.25 if mode == "epsilon_greedy" else 0.0
        config = _config(mode, epsilon)
        controller = ProspectiveExploration(config)
        state = controller.init(jr.key(200))
        receipt = _batch(config, 1)
        result = controller.select(state, receipt)
        budget = controller.resource_budget(state)
        assert bool(result.decision_applied)
        selected[mode] = int(result.selected_index)
        resources.append(
            (
                budget.fixed_candidate_budget,
                budget.logical_uniform_draws_per_decision,
                budget.candidate_metric_scalars_per_decision,
            )
        )
    assert set(selected) == set(PROSPECTIVE_EXPLORATION_MODES)
    assert len(set(resources)) == 1


def test_stochastic_trap_pattern_is_a_threshold_free_mechanism_diagnostic() -> None:
    # This supplies, rather than learns, a pattern in which candidate 2 has
    # large disagreement but zero expected improvement. It checks routing only;
    # it is not evidence that the controller avoids stochastic traps in an
    # environment.
    prospective_config = _config("expected_improvement_surprisal")
    disagreement_config = _config("ensemble_disagreement")
    prospective = ProspectiveExploration(prospective_config).select(
        ProspectiveExploration(prospective_config).init(jr.key(300)),
        _batch(
            prospective_config,
            1,
            host_policy=(0.65, 0.30, 0.05, 0.0),
            expected=(0.1, 1.2, 0.0),
            disagreement=(0.1, 0.2, 9.0),
        ),
    )
    disagreement_controller = ProspectiveExploration(disagreement_config)
    disagreement = disagreement_controller.select(
        disagreement_controller.init(jr.key(300)),
        _batch(
            disagreement_config,
            1,
            host_policy=(0.65, 0.30, 0.05, 0.0),
            expected=(0.1, 1.2, 0.0),
            disagreement=(0.1, 0.2, 9.0),
        ),
    )
    assert int(prospective.selected_index) == 1
    assert int(disagreement.selected_index) == 2


def test_long_horizon_value_must_be_supplied_upstream_and_is_not_inferred() -> None:
    # Candidate 2's value is declared to include a long-horizon improvement.
    # The selector honors that receipt but performs no rollout or VOI estimate.
    config = _config("expected_improvement_surprisal")
    controller = ProspectiveExploration(config)
    base = _batch(
        config,
        1,
        host_policy=(0.7, 0.2, 0.1, 0.0),
        expected=(0.4, 0.5, 2.0),
    )
    long_horizon = controller.select(controller.init(jr.key(400)), base)
    without_upstream_value = controller.select(
        controller.init(jr.key(400)),
        dataclasses.replace(
            base,
            expected_improvement=jnp.asarray([0.4, 0.5, 0.0], dtype=jnp.float32),
        ),
    )
    assert int(long_horizon.selected_index) == 2
    assert int(without_upstream_value.selected_index) != 2
