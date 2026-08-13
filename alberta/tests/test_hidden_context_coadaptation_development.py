"""Development-only hidden-rule coadaptation feasibility contracts."""

from __future__ import annotations

import dataclasses

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_context_coadaptation_development import (
    CONDITIONS,
    DEVELOPMENT_NAMESPACE,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SEEDS,
    ENVIRONMENT_RANDOMNESS_CONSUMED,
    FORBIDDEN_LEARNER_CHANNELS,
    HIDDEN_INFERENCE_UNROUTED,
    HIDDEN_INFERRED,
    LEARNER_POST_ACTION_CHANNELS,
    LEARNER_PRE_ACTION_CHANNELS,
    NUM_STEPS,
    ORACLE_VISIBLE_CEILING,
    OUTPUT_WRITES_ALLOWED,
    PHASE_LENGTH,
    PROTOCOL,
    SCIENTIFIC_PROMOTION_ALLOWED,
    HiddenContextCoadaptationSmoke,
    derive_development_seeds,
    run_development_smoke,
    run_hidden_context_coadaptation,
    summarize_run,
    validate_static_contract,
)

pytestmark = [pytest.mark.development, pytest.mark.integration]


@pytest.fixture(scope="module")
def smoke() -> HiddenContextCoadaptationSmoke:
    return run_development_smoke()


def test_static_contract_is_namespaced_nonpromoting_and_oracle_free() -> None:
    assert DEVELOPMENT_NAMESPACE.endswith("feasibility-v1")
    assert (DEVELOPMENT_ONLY, SCIENTIFIC_PROMOTION_ALLOWED, OUTPUT_WRITES_ALLOWED) == (
        True,
        False,
        False,
    )
    assert DEVELOPMENT_SEEDS == derive_development_seeds(DEVELOPMENT_NAMESPACE, 2)
    assert ENVIRONMENT_RANDOMNESS_CONSUMED is False
    assert CONDITIONS == (
        HIDDEN_INFERRED,
        HIDDEN_INFERENCE_UNROUTED,
        ORACLE_VISIBLE_CEILING,
    )
    learner_channels = set(LEARNER_PRE_ACTION_CHANNELS) | set(
        LEARNER_POST_ACTION_CHANNELS
    )
    assert learner_channels.isdisjoint(FORBIDDEN_LEARNER_CHANNELS)
    assert PROTOCOL.boundary_callbacks_used is False
    assert validate_static_contract() == ()


def test_hidden_inference_is_one_uninterrupted_causal_life(
    smoke: HiddenContextCoadaptationSmoke,
) -> None:
    run = smoke.runs[HIDDEN_INFERRED]
    assert run.rewards.shape == (NUM_STEPS,)
    assert run.actions.shape == (NUM_STEPS, 2)
    assert run.pre_action_contexts.shape == (NUM_STEPS, 2)
    assert run.post_action_contexts.shape == (NUM_STEPS, 2)
    assert run.contexts_in_use.shape == (NUM_STEPS, 2)
    assert bool(jnp.all(run.pre_action_contexts[1:] == run.post_action_contexts[:-1]))
    for boundary in range(PHASE_LENGTH, NUM_STEPS, PHASE_LENGTH):
        chex.assert_trees_all_equal(
            run.pre_action_contexts[boundary],
            run.post_action_contexts[boundary - 1],
        )
    assert bool(jnp.all(run.controller_updates_applied))
    assert bool(jnp.all(run.context_updates_applied))
    chex.assert_trees_all_equal(
        run.final_game_state.step_words,
        jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32),
    )
    assert int(run.final_game_state.step_count) == NUM_STEPS
    for controller_state in run.final_controller_states:
        chex.assert_trees_all_equal(
            controller_state.step_words,
            jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32),
        )
    assert run.final_context_states is not None
    for context_state in run.final_context_states:
        chex.assert_trees_all_equal(
            context_state.step_words,
            jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32),
        )


def test_both_agents_update_independently_from_ordinary_experience(
    smoke: HiddenContextCoadaptationSmoke,
) -> None:
    run = smoke.runs[HIDDEN_INFERRED]
    left, right = run.final_controller_states
    assert bool(jnp.any(left.q_weights != 0.0))
    assert bool(jnp.any(right.q_weights != 0.0))
    assert not np.array_equal(
        np.asarray(jr.key_data(left.rng_key)),
        np.asarray(jr.key_data(right.rng_key)),
    )
    assert run.final_context_states is not None
    left_context, right_context = run.final_context_states
    assert bool(jnp.any(left_context.reward_weights != 0.5))
    assert bool(jnp.any(right_context.reward_weights != 0.5))
    assert run.experience_contract.current_rule_visible is False
    assert run.experience_contract.partner_action_available_only_after_acting is True


def test_recurrence_diagnostics_measure_distinct_slot_reuse_without_label_oracle(
    smoke: HiddenContextCoadaptationSmoke,
) -> None:
    run = smoke.runs[HIDDEN_INFERRED]
    diagnostic = summarize_run(run)
    assert diagnostic.phase_early_rewards.shape == (PROTOCOL.num_phases,)
    assert diagnostic.phase_tail_rewards.shape == (PROTOCOL.num_phases,)
    assert diagnostic.pre_action_switch_lags.shape == (
        PROTOCOL.num_phases - 1,
        2,
    )
    assert diagnostic.tail_context_modes.shape == (PROTOCOL.num_phases, 2)
    assert diagnostic.distinct_rule_slots.shape == (2,)
    assert diagnostic.recurrence_slot_reuse.shape == (2,)
    assert bool(jnp.all(diagnostic.distinct_rule_slots))
    assert bool(jnp.all(diagnostic.recurrence_slot_reuse))
    assert bool(jnp.all(diagnostic.max_contexts_in_use <= PROTOCOL.max_contexts))
    assert bool(jnp.all(diagnostic.max_contexts_in_use == 2))


def test_matched_hidden_ablation_and_labeled_oracle_share_initialization_and_schedule(
    smoke: HiddenContextCoadaptationSmoke,
) -> None:
    inferred = smoke.runs[HIDDEN_INFERRED]
    hidden = smoke.runs[HIDDEN_INFERENCE_UNROUTED]
    oracle = smoke.runs[ORACLE_VISIBLE_CEILING]
    assert inferred.seed == hidden.seed == oracle.seed == DEVELOPMENT_SEEDS[0]
    assert inferred.resource_budget.control_feature_dim == 2
    assert hidden.resource_budget.control_feature_dim == 2
    assert oracle.resource_budget.control_feature_dim == 2
    chex.assert_trees_all_equal(inferred.actions[0], hidden.actions[0])
    chex.assert_trees_all_equal(inferred.actions[0], oracle.actions[0])
    assert bool(jnp.all(hidden.pre_action_contexts == -1))
    assert bool(jnp.all(hidden.context_updates_applied))
    assert inferred.experience_contract.inference_routed_to_control is True
    assert hidden.experience_contract.inference_routed_to_control is False
    assert hidden.final_context_states is not None
    for context_state in hidden.final_context_states:
        chex.assert_trees_all_equal(
            context_state.step_words,
            jnp.asarray((0, NUM_STEPS), dtype=jnp.uint32),
        )
    assert oracle.experience_contract.current_rule_visible is True
    assert oracle.experience_contract.diagnostic_ceiling_only is True
    expected_gap = (
        summarize_run(inferred).recurrent_early_reward
        - summarize_run(hidden).recurrent_early_reward
    )
    assert smoke.inferred_minus_unrouted_recurrent_early == pytest.approx(
        expected_gap
    )


def test_run_is_deterministic_for_the_fixed_seed() -> None:
    first = run_hidden_context_coadaptation(HIDDEN_INFERRED, DEVELOPMENT_SEEDS[0])
    second = run_hidden_context_coadaptation(HIDDEN_INFERRED, DEVELOPMENT_SEEDS[0])
    chex.assert_trees_all_equal(first.rewards, second.rewards)
    chex.assert_trees_all_equal(first.actions, second.actions)
    chex.assert_trees_all_equal(first.pre_action_contexts, second.pre_action_contexts)
    chex.assert_trees_all_equal(first.post_action_contexts, second.post_action_contexts)
    chex.assert_trees_all_equal(first.final_controller_states, second.final_controller_states)
    chex.assert_trees_all_equal(first.final_context_states, second.final_context_states)


def test_resource_bounds_are_exact_fixed_and_report_environment_clock_limit(
    smoke: HiddenContextCoadaptationSmoke,
) -> None:
    inferred = smoke.runs[HIDDEN_INFERRED].resource_budget
    hidden = smoke.runs[HIDDEN_INFERENCE_UNROUTED].resource_budget
    oracle = smoke.runs[ORACLE_VISIBLE_CEILING].resource_budget
    assert inferred.fixed_shape and hidden.fixed_shape and oracle.fixed_shape
    assert inferred.replay_capacity == hidden.replay_capacity == oracle.replay_capacity == 0
    assert inferred.per_agent_control_nbytes == hidden.per_agent_control_nbytes
    assert inferred.per_agent_control_nbytes == oracle.per_agent_control_nbytes
    assert inferred.per_agent_context_nbytes > 0
    assert hidden.per_agent_context_nbytes == inferred.per_agent_context_nbytes
    assert oracle.per_agent_context_nbytes == 0
    assert inferred.joint_persistent_jax_array_nbytes == (
        inferred.environment_nbytes
        + 2 * inferred.per_agent_control_nbytes
        + 2 * inferred.per_agent_context_nbytes
    )
    assert hidden.joint_persistent_jax_array_nbytes == (
        hidden.environment_nbytes
        + 2 * hidden.per_agent_control_nbytes
        + 2 * hidden.per_agent_context_nbytes
    )
    assert hidden.joint_persistent_jax_array_nbytes == (
        inferred.joint_persistent_jax_array_nbytes
    )
    assert hidden.joint_clock_nbytes == inferred.joint_clock_nbytes
    assert inferred.environment_clock_nbytes == 8
    assert hidden.environment_clock_nbytes == 8
    assert oracle.environment_clock_nbytes == 8
    assert inferred.joint_clock_nbytes == (
        inferred.environment_clock_nbytes
        + 2 * inferred.per_agent_control_clock_nbytes
        + 2 * inferred.per_agent_context_clock_nbytes
    )
    assert inferred.per_agent_context_updates_per_transition == 1
    assert hidden.per_agent_context_updates_per_transition == 1
    assert oracle.per_agent_context_updates_per_transition == 0
    assert inferred.max_context_slots_per_agent == 2
    assert hidden.max_context_slots_per_agent == 2
    assert inferred.controller_and_context_clocks_exact is True
    assert inferred.environment_schedule_clock_exact is True
    assert inferred.environment_schedule_max_steps == 2**64 - 1
    assert inferred.environment_randomness_consumed is False
    assert hidden.environment_randomness_consumed is False
    assert oracle.environment_randomness_consumed is False
    assert dataclasses.asdict(inferred) == inferred.to_dict()
