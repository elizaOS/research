"""Focused contracts for the hidden-rule capacity-pressure calibration."""

from __future__ import annotations

import inspect
from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import alberta_framework.evaluation.hidden_rule_capacity_pressure_development as lane
from alberta_framework.core.context_inference import ContextInference, ContextInferenceState
from alberta_framework.evaluation.hidden_rule_capacity_pressure_development import (
    ARBITRARY_ROOT_EXECUTION_ALLOWED,
    BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
    CALIBRATION_ROOT,
    CALIBRATION_ROOT_CONSUMED,
    CONTEXT_CONFIG,
    DEVELOPMENT_NAMESPACE,
    DEVELOPMENT_ONLY,
    EPSILON_GRID,
    FORBIDDEN_LEARNER_CHANNELS,
    GAME_CONFIG,
    LEARNER_POST_ACTION_CHANNELS,
    LEARNER_PRE_ACTION_CHANNELS,
    LINEAGE_CACHE_CAPACITY,
    LINEAGE_CACHE_CONDITIONS,
    LINEAGE_CACHE_NO_SIGNAL,
    LINEAGE_CACHE_PREDICTIVE_RESCUE,
    MAX_CONTEXTS,
    OFFSETS,
    OUTPUT_WRITES_ALLOWED,
    POST_AUDIT_BASELINE,
    POST_AUDIT_CONDITIONS,
    PROTOCOL,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SELECTIVE_RETENTION_CONDITIONS,
    SELECTIVE_RETENTION_NO_SIGNAL,
    SELECTIVE_RETENTION_PAST_RECURRENCE,
    SEMANTIC_CONTEXT_IDENTITY,
    CapacityPressureState,
    advance_consumed_capacity_pressure_state,
    advance_consumed_lineage_cache_retention_state,
    advance_consumed_selective_retention_state,
    build_prefix_twin_boundary,
    control_config,
    initialize_capacity_pressure_state,
    initialize_lineage_cache_retention_state,
    initialize_selective_retention_state,
    run_consumed_calibration_arm,
    step_capacity_pressure,
    step_lineage_cache_retention,
    step_post_audit_intervention,
    step_selective_retention,
    validate_static_contract,
)
from alberta_framework.streams.matrix_game import RecurringConventionGame

pytestmark = [pytest.mark.unit, pytest.mark.development]


def test_lineage_cache_intervention_is_defaults_off_and_minimally_bounded() -> None:
    assert LINEAGE_CACHE_CONDITIONS == (
        LINEAGE_CACHE_NO_SIGNAL,
        LINEAGE_CACHE_PREDICTIVE_RESCUE,
    )
    assert LINEAGE_CACHE_CAPACITY == 1
    state = initialize_lineage_cache_retention_state(0.2)
    for sidecar in (state.lineage_0, state.lineage_1):
        assert not bool(sidecar.cache_valid)
        assert not bool(jnp.any(sidecar.live_rescue_words))
        assert not bool(jnp.any(sidecar.cache_reward_weights))


def test_static_contract_is_consumed_root_nonpromoting_and_oracle_free() -> None:
    assert validate_static_contract() == ()
    assert DEVELOPMENT_NAMESPACE.endswith("consumed-calibration-root-0-v1")
    assert (DEVELOPMENT_ONLY, SCIENTIFIC_PROMOTION_ALLOWED, OUTPUT_WRITES_ALLOWED) == (
        True,
        False,
        False,
    )
    assert CALIBRATION_ROOT_CONSUMED is True
    assert ARBITRARY_ROOT_EXECUTION_ALLOWED is False
    assert CALIBRATION_ROOT.index == CALIBRATION_ROOT.key_seed == 0
    assert PROTOCOL.offsets == OFFSETS == (0, 1, 0, 3, 0, 2, 0, 1, 2, 0)
    assert PROTOCOL.phase_length == 400
    assert PROTOCOL.max_contexts == MAX_CONTEXTS == 3
    assert len(set(OFFSETS)) == 4
    assert GAME_CONFIG.feature_mode == "plain"
    assert GAME_CONFIG.observation_dim == 1
    assert CONTEXT_CONFIG.observation_dim == 4
    assert PROTOCOL.boundary_callbacks_used is False
    assert PROTOCOL.resets_after_initialization == PROTOCOL.replay_capacity == 0
    learner_channels = set(LEARNER_PRE_ACTION_CHANNELS) | set(
        LEARNER_POST_ACTION_CHANNELS
    )
    assert learner_channels.isdisjoint(FORBIDDEN_LEARNER_CHANNELS)
    assert PROTOCOL.current_rule_visible_to_learners is False
    assert PROTOCOL.schedule_visible_to_learners is False
    assert PROTOCOL.evaluator_birth_ledger_routed_to_learners is False
    assert SELECTIVE_RETENTION_CONDITIONS == (
        SELECTIVE_RETENTION_NO_SIGNAL,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
    )


def test_eviction_recurrence_protection_is_an_explicit_defaults_off_api() -> None:
    inference = ContextInference(CONTEXT_CONFIG)
    state = inference.init()
    default = inference.update_result(
        state,
        jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    prioritized = inference.update_result_with_eviction_protection(
        state,
        jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(prioritized.state, default.state)
    chex.assert_trees_all_equal(prioritized.context_onehot, default.context_onehot)
    assert not bool(prioritized.eviction_protection_used)
    assert not bool(prioritized.eviction_target_adjusted)


def _full_bank_context_state() -> ContextInferenceState:
    inference = ContextInference(CONTEXT_CONFIG)
    return cast(
        ContextInferenceState,
        inference.init().replace(  # type: ignore[attr-defined]
            in_use=jnp.ones((MAX_CONTEXTS,), dtype=jnp.bool_),
            error_ema=jnp.ones((MAX_CONTEXTS,), dtype=jnp.float32),
            last_active_step=jnp.asarray((9, 2, 5), dtype=jnp.int32),
            dwell=jnp.asarray(10, dtype=jnp.int32),
            step_count=jnp.asarray(10, dtype=jnp.int32),
            last_active_words=jnp.asarray(
                ((0, 9), (0, 2), (0, 5)),
                dtype=jnp.uint32,
            ),
            dwell_words=jnp.asarray((0, 10), dtype=jnp.uint32),
            step_words=jnp.asarray((0, 10), dtype=jnp.uint32),
        ),
    )


def test_priority_changes_only_a_valid_full_bank_fresh_allocation() -> None:
    inference = ContextInference(CONTEXT_CONFIG)
    state = _full_bank_context_state()
    assert bool(inference.state_is_valid(state))
    observation = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    action = jnp.asarray(0, dtype=jnp.int32)
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    default = inference.update_result(state, observation, action, reward)
    zero = inference.update_result_with_eviction_protection(
        state,
        observation,
        action,
        reward,
        jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
    )
    protected = inference.update_result_with_eviction_protection(
        state,
        observation,
        action,
        reward,
        jnp.asarray((0.0, 10.0, 0.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(zero.state, default.state)
    assert int(default.state.active_context) == int(zero.state.active_context) == 1
    assert int(protected.ordinary_lru_slot) == 1
    assert int(protected.protected_lru_slot) == 2
    assert int(protected.selected_eviction_slot) == 2
    assert int(protected.state.active_context) == 2
    assert bool(protected.full_bank_eviction_requested)
    assert bool(protected.eviction_protection_used)
    assert bool(protected.eviction_target_adjusted)

    # A stored model that explains the reward is reused regardless of scores.
    reuse_state = state.replace(  # type: ignore[attr-defined]
        reward_weights=state.reward_weights.at[1, 0, 0].set(1.0),
        error_ema=state.error_ema.at[1].set(0.0),
    )
    reuse_default = inference.update_result(reuse_state, observation, action, 1.0)
    reuse_protected = inference.update_result_with_eviction_protection(
        reuse_state,
        observation,
        action,
        1.0,
        jnp.asarray((0.0, 100.0, 0.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(reuse_protected.state, reuse_default.state)
    assert int(reuse_protected.state.active_context) == 1
    assert not bool(reuse_protected.allocation_requested)
    assert not bool(reuse_protected.eviction_protection_used)

    # Allocation into a free slot also remains exact and ignores protection.
    free_state = state.replace(  # type: ignore[attr-defined]
        in_use=jnp.asarray((True, True, False), dtype=jnp.bool_),
        error_ema=jnp.asarray((1.0, 1.0, 0.5), dtype=jnp.float32),
        last_active_step=jnp.asarray((9, 2, -1), dtype=jnp.int32),
        last_active_words=jnp.asarray(
            ((0, 9), (0, 2), (0, 0)),
            dtype=jnp.uint32,
        ),
    )
    assert bool(inference.state_is_valid(free_state))
    free_default = inference.update_result(free_state, observation, action, reward)
    free_protected = inference.update_result_with_eviction_protection(
        free_state,
        observation,
        action,
        reward,
        jnp.asarray((0.0, 100.0, 1000.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(free_protected.state, free_default.state)
    assert int(free_protected.state.active_context) == 2
    assert bool(free_protected.allocation_requested)
    assert not bool(free_protected.full_bank_eviction_requested)
    assert not bool(free_protected.eviction_protection_used)


def test_priority_api_is_exact_under_eager_jit_and_scan_and_rejects_bad_scores() -> None:
    inference = ContextInference(CONTEXT_CONFIG)
    state = _full_bank_context_state()
    observation = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    action = jnp.asarray(0, dtype=jnp.int32)
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    protection = jnp.asarray((0.0, 10.0, 0.0), dtype=jnp.float32)
    with jax.disable_jit():
        eager = inference.update_result_with_eviction_protection(
            state,
            observation,
            action,
            reward,
            protection,
        )
    compiled = jax.jit(
        lambda source, scores: inference.update_result_with_eviction_protection(
            source,
            observation,
            action,
            reward,
            scores,
        )
    )(state, protection)
    chex.assert_trees_all_equal(eager, compiled)

    observations = jnp.eye(MAX_CONTEXTS + 1, dtype=jnp.float32)[:3]
    actions = jnp.zeros((3,), dtype=jnp.int32)
    rewards = jnp.asarray((1.0, 0.0, 1.0), dtype=jnp.float32)

    def default_scan_step(
        source: ContextInferenceState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[ContextInferenceState, Array]:
        obs, act, rew = inputs
        result = inference.update_result(source, obs, act, rew)
        return result.state, result.context_onehot

    def zero_scan_step(
        source: ContextInferenceState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[ContextInferenceState, Array]:
        obs, act, rew = inputs
        result = inference.update_result_with_eviction_protection(
            source,
            obs,
            act,
            rew,
            jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
        )
        return result.state, result.context_onehot

    default_final, default_contexts = jax.lax.scan(
        default_scan_step,
        inference.init(),
        (observations, actions, rewards),
    )
    zero_final, zero_contexts = jax.lax.scan(
        zero_scan_step,
        inference.init(),
        (observations, actions, rewards),
    )
    chex.assert_trees_all_equal(zero_final, default_final)
    chex.assert_trees_all_equal(zero_contexts, default_contexts)

    for bad in (
        jnp.asarray((0.0, jnp.nan, 0.0), dtype=jnp.float32),
        jnp.asarray((0.0, -1.0, 0.0), dtype=jnp.float32),
    ):
        rejected = inference.update_result_with_eviction_protection(
            state,
            observation,
            action,
            reward,
            bad,
        )
        assert not bool(rejected.eviction_protection_input_valid)
        assert not bool(rejected.update_applied)
        chex.assert_trees_all_equal(rejected.state, state)
    with pytest.raises(TypeError, match="dtype float32"):
        inference.update_result_with_eviction_protection(
            state,
            observation,
            action,
            reward,
            jnp.zeros((MAX_CONTEXTS,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="shape"):
        inference.update_result_with_eviction_protection(
            state,
            observation,
            action,
            reward,
            jnp.zeros((MAX_CONTEXTS - 1,), dtype=jnp.float32),
        )


def test_only_predeclared_epsilon_grid_is_executable_and_no_root_surface_exists() -> None:
    for epsilon in EPSILON_GRID:
        config = control_config(epsilon)
        assert config.epsilon_start == config.epsilon_end == epsilon
        assert config.epsilon_decay_steps == 0
        assert config.q_step_size == 0.15
        assert config.average_reward_step_size == 0.01
        assert config.use_bias is False
    with pytest.raises(ValueError, match="predeclared"):
        control_config(0.3)
    assert tuple(inspect.signature(run_consumed_calibration_arm).parameters) == (
        "epsilon",
    )


def test_birth_words_are_agent_namespaced_not_recyclable_slot_identity() -> None:
    assert SEMANTIC_CONTEXT_IDENTITY == "(agent_namespace, exact_birth_words)"
    assert PROTOCOL.semantic_context_identity == SEMANTIC_CONTEXT_IDENTITY
    assert PROTOCOL.slot_indices_are_semantic_identities is False
    assert PROTOCOL.cross_agent_birth_words_are_comparable_identities is False


def test_joint_step_rolls_every_child_and_ledger_back_on_clock_misalignment() -> None:
    state = initialize_capacity_pressure_state(0.2)
    misaligned = state.replace(  # type: ignore[attr-defined]
        controller_0=state.controller_0.replace(  # type: ignore[attr-defined]
            step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            step_count=jnp.asarray(1, dtype=jnp.int32),
        )
    )
    result = step_capacity_pressure(0.2, misaligned)
    assert bool(result.trace.environment_update_proposed)
    assert bool(jnp.all(result.trace.context_updates_proposed))
    assert bool(jnp.all(result.trace.controller_updates_proposed))
    assert not bool(result.trace.source_clocks_aligned)
    assert not bool(result.trace.candidate_clocks_aligned)
    assert not bool(result.trace.update_applied)
    assert float(result.trace.reward) == 0.0
    assert not bool(jnp.any(result.trace.switches))
    assert not bool(jnp.any(result.trace.allocations))
    assert not bool(jnp.any(result.trace.evictions))
    assert not bool(jnp.any(result.trace.reuses))
    chex.assert_trees_all_equal(result.state, misaligned)
    chex.assert_trees_all_equal(
        result.trace.pre_step_words,
        result.trace.post_step_words,
    )


def test_prefix_twin_binds_opposite_evictions_without_executing_future() -> None:
    boundary = build_prefix_twin_boundary()
    assert boundary.actual_schedule[: boundary.common_prefix_phases] == (
        boundary.counterfactual_schedule[: boundary.common_prefix_phases]
    )
    assert boundary.actual_prefix_sha256 == boundary.counterfactual_prefix_sha256
    assert boundary.common_prefix_sha256 == boundary.actual_prefix_sha256
    assert boundary.differing_phase_indices == (7,)
    assert boundary.first_divergent_phase == 7
    assert (
        boundary.actual_divergent_offset,
        boundary.counterfactual_divergent_offset,
    ) == (1, 3)
    assert set(boundary.regimes_seen_by_first_c_admission) == {"A", "B", "C", "D"}
    assert boundary.capacity == 3
    assert boundary.actual_zero_recurrence_loss_set == ("A", "B", "C")
    assert boundary.counterfactual_zero_recurrence_loss_set == ("A", "D", "C")
    assert boundary.correct_actual_eviction == "D"
    assert boundary.correct_counterfactual_eviction == "B"
    assert boundary.same_policy_rng_implies_identical_prefix_history is True
    assert boundary.future_schedule_only_divergence is True
    assert boundary.deterministic_online_guarantee_possible is False
    assert boundary.future_schedule_revealed_to_learners is False
    assert boundary.counterfactual_future_executed is False
    assert boundary.stochastic_optimality_claimed is False


def _state_before_first_allocation() -> CapacityPressureState:
    state = advance_consumed_capacity_pressure_state(0.2, 404)
    probe = step_capacity_pressure(0.2, state)
    assert bool(jnp.all(probe.trace.allocations))
    return state


def test_post_audit_scrub_removes_stale_destination_before_q_next_and_action() -> None:
    assert POST_AUDIT_CONDITIONS == (
        POST_AUDIT_BASELINE,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
    )
    state = _state_before_first_allocation()
    stale_q = jnp.asarray((10.0, 20.0, 30.0, 40.0), dtype=jnp.float32)
    stale_trace = jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float32)
    destination = 1
    state = state.replace(  # type: ignore[attr-defined]
        controller_0=state.controller_0.replace(  # type: ignore[attr-defined]
            q_weights=state.controller_0.q_weights.at[:, destination].set(stale_q),
            q_trace_weights=state.controller_0.q_trace_weights.at[:, destination].set(
                stale_trace
            ),
        ),
        controller_1=state.controller_1.replace(  # type: ignore[attr-defined]
            q_weights=state.controller_1.q_weights.at[:, destination].set(stale_q),
            q_trace_weights=state.controller_1.q_trace_weights.at[:, destination].set(
                stale_trace
            ),
        ),
    )
    baseline = step_post_audit_intervention(0.2, POST_AUDIT_BASELINE, state)
    original = step_capacity_pressure(0.2, state)
    scrub = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        state,
    )
    assert bool(baseline.trace.update_applied)
    assert bool(scrub.trace.update_applied)
    chex.assert_trees_all_equal(baseline.trace, original.trace)
    chex.assert_trees_all_equal(baseline.state, original.state)
    assert bool(jnp.all(baseline.scrub.scrub_required))
    assert not bool(jnp.any(baseline.scrub.scrub_applied))
    assert bool(jnp.all(scrub.scrub.scrub_applied))
    assert bool(jnp.all(scrub.scrub.scrubbed_parameter_scalars == 8))
    assert bool(jnp.all(scrub.scrub.pre_destination_q_weight_l1 == 100.0))
    assert bool(jnp.all(scrub.scrub.prepared_destination_q_weight_l1 == 0.0))
    assert bool(jnp.all(scrub.scrub.pre_destination_q_trace_l1 == 10.0))
    assert bool(jnp.all(scrub.scrub.prepared_destination_q_trace_l1 == 0.0))
    assert bool(jnp.all(baseline.scrub.cross_birth_contamination_consumed))
    assert bool(jnp.all(scrub.scrub.cross_birth_contamination_prevented))
    for agent_index, (baseline_controller, scrub_controller, prepared) in enumerate(
        zip(
            (baseline.state.controller_0, baseline.state.controller_1),
            (scrub.state.controller_0, scrub.state.controller_1),
            scrub.prepared_controllers,
            strict=True,
        )
    ):
        chex.assert_trees_all_equal(
            baseline_controller.q_weights[:, destination],
            stale_q,
        )
        chex.assert_trees_all_equal(
            scrub_controller.q_weights[:, destination],
            jnp.zeros_like(stale_q),
        )
        chex.assert_trees_all_equal(
            prepared.q_weights[:, destination],
            jnp.zeros_like(stale_q),
        )
        chex.assert_trees_all_equal(
            prepared.q_trace_weights[:, destination],
            jnp.zeros_like(stale_trace),
        )
        chex.assert_trees_all_equal(
            baseline.trace.controller_next_q_values[agent_index],
            stale_q,
        )
        chex.assert_trees_all_equal(
            scrub.trace.controller_next_q_values[agent_index],
            jnp.zeros_like(stale_q),
        )
    chex.assert_trees_all_equal(
        baseline.state.controller_0.rng_key,
        scrub.state.controller_0.rng_key,
    )
    chex.assert_trees_all_equal(
        baseline.state.controller_1.rng_key,
        scrub.state.controller_1.rng_key,
    )
    chex.assert_trees_all_equal(
        baseline.state.environment.step_words,
        scrub.state.environment.step_words,
    )


def test_same_source_binding_and_corrupt_birth_ledger_fail_atomically() -> None:
    state = _state_before_first_allocation()
    source_slot = int(state.context_0.active_context)
    same_source = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        state,
        audit_destination_override=(source_slot, source_slot),
    )
    assert not bool(same_source.trace.update_applied)
    assert not bool(jnp.any(same_source.scrub.source_destination_separated))
    chex.assert_trees_all_equal(same_source.state, state)

    corrupt = state.replace(  # type: ignore[attr-defined]
        ledger_0=state.ledger_0.replace(  # type: ignore[attr-defined]
            slot_birth_words=state.ledger_0.slot_birth_words.at[0].set(
                jnp.asarray((0, 10_000), dtype=jnp.uint32)
            )
        )
    )
    rejected = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        corrupt,
    )
    assert not bool(rejected.trace.update_applied)
    assert not bool(rejected.scrub.pre_ledger_valid[0])
    chex.assert_trees_all_equal(rejected.state, corrupt)


def test_post_audit_shape_finite_and_nonzero_bias_contracts_roll_back() -> None:
    initial = initialize_capacity_pressure_state(0.2)
    malformed = initial.replace(  # type: ignore[attr-defined]
        controller_0=initial.controller_0.replace(  # type: ignore[attr-defined]
            q_weights=jnp.zeros((4, 2), dtype=jnp.float32),
            q_trace_weights=jnp.zeros((4, 2), dtype=jnp.float32),
            last_observation=jnp.zeros((2,), dtype=jnp.float32),
        )
    )
    shape_rejected = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        malformed,
    )
    assert not bool(shape_rejected.trace.update_applied)
    assert not bool(shape_rejected.scrub.controller_shape_valid[0])
    chex.assert_trees_all_equal(shape_rejected.state, malformed)

    nonfinite = initial.replace(  # type: ignore[attr-defined]
        controller_0=initial.controller_0.replace(  # type: ignore[attr-defined]
            q_weights=initial.controller_0.q_weights.at[0, 0].set(jnp.inf)
        )
    )
    finite_rejected = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        nonfinite,
    )
    assert not bool(finite_rejected.trace.update_applied)
    assert not bool(finite_rejected.trace.source_state_finite)
    chex.assert_trees_all_equal(finite_rejected.state, nonfinite)

    allocation_state = _state_before_first_allocation()
    bad_bias = allocation_state.replace(  # type: ignore[attr-defined]
        controller_0=allocation_state.controller_0.replace(  # type: ignore[attr-defined]
            q_bias=allocation_state.controller_0.q_bias.at[0].set(1.0)
        )
    )
    bias_rejected = step_post_audit_intervention(
        0.2,
        BIRTH_AUTHENTICATED_CONTROLLER_SCRUB,
        bad_bias,
    )
    assert not bool(bias_rejected.trace.update_applied)
    assert not bool(bias_rejected.scrub.biases_zero_before[0])
    chex.assert_trees_all_equal(bias_rejected.state, bad_bias)


def test_selective_history_is_past_only_defaults_off_and_prefix_twin_limited() -> None:
    initial = initialize_selective_retention_state(0.05)
    no_signal = step_selective_retention(
        0.05,
        SELECTIVE_RETENTION_NO_SIGNAL,
        initial,
    )
    recurrence = step_selective_retention(
        0.05,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
        initial,
    )
    chex.assert_trees_all_equal(no_signal.state, recurrence.state)
    assert bool(no_signal.trace.capacity.update_applied)
    assert bool(recurrence.trace.capacity.update_applied)
    assert not bool(no_signal.trace.protection_enabled)
    assert bool(recurrence.trace.protection_enabled)
    assert not bool(jnp.any(no_signal.trace.raw_completed_recurrence_scores))
    assert not bool(jnp.any(recurrence.trace.raw_completed_recurrence_scores))
    assert not bool(jnp.any(no_signal.trace.dispatched_eviction_protection))
    assert not bool(jnp.any(recurrence.trace.dispatched_eviction_protection))
    assert tuple(inspect.signature(step_selective_retention).parameters) == (
        "epsilon",
        "condition",
        "state",
    )
    source = inspect.getsource(step_selective_retention)
    assert "future" not in source
    boundary = build_prefix_twin_boundary()
    assert boundary.deterministic_online_guarantee_possible is False


def test_past_recurrence_signal_changes_only_the_eviction_target_and_resets_birth() -> None:
    source = advance_consumed_selective_retention_state(
        0.05,
        SELECTIVE_RETENTION_NO_SIGNAL,
        3216,
    )
    no_signal = step_selective_retention(
        0.05,
        SELECTIVE_RETENTION_NO_SIGNAL,
        source,
    )
    protected = step_selective_retention(
        0.05,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
        source,
    )
    assert bool(no_signal.trace.capacity.update_applied)
    assert bool(protected.trace.capacity.update_applied)
    chex.assert_trees_all_equal(
        no_signal.trace.capacity.actions,
        protected.trace.capacity.actions,
    )
    chex.assert_trees_all_equal(
        no_signal.trace.capacity.reward,
        protected.trace.capacity.reward,
    )
    chex.assert_trees_all_equal(
        no_signal.trace.raw_completed_recurrence_scores,
        jnp.asarray(((4.0, 6.0, 0.0), (4.0, 6.0, 0.0)), dtype=jnp.float32),
    )
    assert not bool(jnp.any(no_signal.trace.dispatched_eviction_protection))
    chex.assert_trees_all_equal(
        protected.trace.dispatched_eviction_protection,
        protected.trace.raw_completed_recurrence_scores,
    )
    assert bool(jnp.all(no_signal.trace.full_bank_evictions_requested))
    assert bool(jnp.all(protected.trace.full_bank_evictions_requested))
    assert not bool(jnp.any(no_signal.trace.eviction_targets_adjusted))
    assert bool(jnp.all(protected.trace.eviction_targets_adjusted))
    assert bool(jnp.all(protected.trace.ordinary_lru_slots == 0))
    assert bool(jnp.all(protected.trace.selected_eviction_slots == 2))
    assert bool(
        jnp.all(protected.trace.ordinary_lru_completed_recurrence_scores == 4.0)
    )
    assert bool(jnp.all(protected.trace.selected_completed_recurrence_scores == 0.0))
    assert bool(jnp.all(protected.trace.history_allocation_resets))
    post_step = protected.trace.capacity.post_step_words
    for history in (
        protected.state.recurrence_0,
        protected.state.recurrence_1,
    ):
        chex.assert_trees_all_equal(
            history.occurrence_words[2],
            jnp.asarray((0, 1), dtype=jnp.uint32),
        )
        chex.assert_trees_all_equal(history.bound_birth_words[2], post_step)
        chex.assert_trees_all_equal(history.last_entry_words[2], post_step)
        chex.assert_trees_all_equal(
            history.last_interval_words[2],
            jnp.zeros((2,), dtype=jnp.uint32),
        )


def test_selective_history_corruption_nonfinite_state_and_bad_shape_fail_closed() -> None:
    initial = initialize_selective_retention_state(0.2)
    corrupt = initial.replace(  # type: ignore[attr-defined]
        recurrence_0=initial.recurrence_0.replace(  # type: ignore[attr-defined]
            bound_birth_words=initial.recurrence_0.bound_birth_words.at[0].set(
                jnp.asarray((0, 1), dtype=jnp.uint32)
            )
        )
    )
    rejected = step_selective_retention(
        0.2,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
        corrupt,
    )
    assert not bool(rejected.trace.capacity.update_applied)
    assert not bool(rejected.trace.history_source_valid[0])
    chex.assert_trees_all_equal(rejected.state, corrupt)

    nonfinite = initial.replace(  # type: ignore[attr-defined]
        base=initial.base.replace(  # type: ignore[attr-defined]
            controller_0=initial.base.controller_0.replace(  # type: ignore[attr-defined]
                q_weights=initial.base.controller_0.q_weights.at[0, 0].set(jnp.nan)
            )
        )
    )
    nonfinite_rejected = step_selective_retention(
        0.2,
        SELECTIVE_RETENTION_PAST_RECURRENCE,
        nonfinite,
    )
    assert not bool(nonfinite_rejected.trace.capacity.update_applied)
    assert not bool(nonfinite_rejected.trace.capacity.source_state_finite)
    chex.assert_trees_all_equal(nonfinite_rejected.state, nonfinite)

    malformed = initial.replace(  # type: ignore[attr-defined]
        recurrence_0=initial.recurrence_0.replace(  # type: ignore[attr-defined]
            occurrence_words=jnp.zeros((MAX_CONTEXTS - 1, 2), dtype=jnp.uint32)
        )
    )
    with pytest.raises(ValueError, match="static shapes"):
        step_selective_retention(
            0.2,
            SELECTIVE_RETENTION_PAST_RECURRENCE,
            malformed,
        )


def _upcoming_lineage_experience(
    state: lane.LineageCacheRetentionState,
) -> tuple[Array, Array, tuple[Array, Array]]:
    actions = jnp.stack(
        (state.base.controller_0.last_action, state.base.controller_1.last_action)
    ).astype(jnp.int32)
    result = RecurringConventionGame(GAME_CONFIG).step_result(
        state.base.environment,
        actions[0],
        actions[1],
    )
    observations = (
        jax.nn.one_hot(actions[1], lane.N_ACTIONS, dtype=jnp.float32),
        jax.nn.one_hot(actions[0], lane.N_ACTIONS, dtype=jnp.float32),
    )
    return actions, result.reward, observations


def _perfect_upcoming_cache(
    state: lane.LineageCacheRetentionState,
) -> lane.LineageCacheRetentionState:
    actions, reward, _ = _upcoming_lineage_experience(state)
    lineage_0 = state.lineage_0.replace(  # type: ignore[attr-defined]
        cache_reward_weights=state.lineage_0.cache_reward_weights.at[
            actions[0], actions[1]
        ].set(reward)
    )
    lineage_1 = state.lineage_1.replace(  # type: ignore[attr-defined]
        cache_reward_weights=state.lineage_1.cache_reward_weights.at[
            actions[1], actions[0]
        ].set(reward)
    )
    return cast(
        lane.LineageCacheRetentionState,
        state.replace(  # type: ignore[attr-defined]
            lineage_0=lineage_0,
            lineage_1=lineage_1,
        ),
    )


def _set_lineage_rescue_values(
    state: lane.LineageCacheRetentionState,
    *,
    cache_value: int,
    victim_value: int,
) -> lane.LineageCacheRetentionState:
    # The target is obtained from the actual next step because an allocation
    # need not replace the currently active slot.
    probe = step_lineage_cache_retention(0.05, LINEAGE_CACHE_NO_SIGNAL, state)
    targets = (
        int(probe.trace.capacity.post_context_slots[0]),
        int(probe.trace.capacity.post_context_slots[1]),
    )
    cache_words = jnp.asarray((0, cache_value), dtype=jnp.uint32)
    victim_words = jnp.asarray((0, victim_value), dtype=jnp.uint32)
    lineage_0 = state.lineage_0.replace(  # type: ignore[attr-defined]
        cache_rescue_words=cache_words,
        live_rescue_words=state.lineage_0.live_rescue_words.at[targets[0]].set(
            victim_words
        ),
    )
    lineage_1 = state.lineage_1.replace(  # type: ignore[attr-defined]
        cache_rescue_words=cache_words,
        live_rescue_words=state.lineage_1.live_rescue_words.at[targets[1]].set(
            victim_words
        ),
    )
    return cast(
        lane.LineageCacheRetentionState,
        state.replace(  # type: ignore[attr-defined]
            lineage_0=lineage_0,
            lineage_1=lineage_1,
        ),
    )


def test_lineage_free_allocation_never_queries_or_mutates_the_valid_cache() -> None:
    source = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        403,
    )
    cache_identity = jnp.asarray((0, 1), dtype=jnp.uint32)
    weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16.0
    cache_kwargs = {
        "cache_valid": jnp.asarray(True, dtype=jnp.bool_),
        "cache_source_birth_words": cache_identity,
        "cache_lineage_words": cache_identity,
        "cache_rescue_words": jnp.zeros((2,), dtype=jnp.uint32),
        "cache_reward_weights": weights,
    }
    source = source.replace(  # type: ignore[attr-defined]
        lineage_0=source.lineage_0.replace(**cache_kwargs),  # type: ignore[attr-defined]
        lineage_1=source.lineage_1.replace(**cache_kwargs),  # type: ignore[attr-defined]
    )
    result = step_lineage_cache_retention(0.05, LINEAGE_CACHE_NO_SIGNAL, source)
    assert bool(jnp.all(result.trace.capacity.allocations))
    assert not bool(jnp.any(result.trace.capacity.evictions))
    assert not bool(jnp.any(result.trace.cache_tested))
    assert not bool(jnp.any(result.trace.victim_archived))
    for before, after in (
        (source.lineage_0, result.state.lineage_0),
        (source.lineage_1, result.state.lineage_1),
    ):
        chex.assert_trees_all_equal(after.cache_valid, before.cache_valid)
        chex.assert_trees_all_equal(
            after.cache_source_birth_words,
            before.cache_source_birth_words,
        )
        chex.assert_trees_all_equal(after.cache_lineage_words, before.cache_lineage_words)
        chex.assert_trees_all_equal(after.cache_rescue_words, before.cache_rescue_words)
        chex.assert_trees_all_equal(after.cache_reward_weights, before.cache_reward_weights)


def test_strict_cache_match_transfers_lineage_and_archives_source_victim() -> None:
    source = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        3216,
    )
    source = _perfect_upcoming_cache(source)
    no_signal = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        source,
    )
    signal = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_PREDICTIVE_RESCUE,
        source,
    )
    assert bool(jnp.all(no_signal.trace.capacity.update_applied))
    assert bool(jnp.all(no_signal.trace.full_bank_evictions_requested))
    assert bool(jnp.all(no_signal.trace.cache_tested))
    assert bool(jnp.all(no_signal.trace.strict_predictive_dominance))
    assert bool(jnp.all(no_signal.trace.cache_matched))
    assert bool(jnp.all(no_signal.trace.lineage_transferred))
    assert bool(jnp.all(no_signal.trace.rescue_incremented))
    assert bool(jnp.all(no_signal.trace.victim_archived))
    assert bool(no_signal.trace.source_scores_fixed_before_outcome)
    assert not bool(no_signal.trace.outcome_routed_to_current_protection)
    # The just-observed match cannot change this event's source-score victim.
    chex.assert_trees_all_equal(
        no_signal.trace.selected_eviction_slots,
        signal.trace.selected_eviction_slots,
    )
    assert not bool(jnp.any(signal.trace.eviction_targets_adjusted))
    for agent_index, (before, after, source_context, post_context) in enumerate(
        zip(
            (source.lineage_0, source.lineage_1),
            (no_signal.state.lineage_0, no_signal.state.lineage_1),
            (source.base.context_0, source.base.context_1),
            (no_signal.state.base.context_0, no_signal.state.base.context_1),
            strict=True,
        )
    ):
        target = int(no_signal.trace.capacity.post_context_slots[agent_index])
        chex.assert_trees_all_equal(
            after.live_lineage_words[target],
            before.cache_lineage_words,
        )
        chex.assert_trees_all_equal(
            after.live_rescue_words[target],
            before.cache_rescue_words + jnp.asarray((0, 1), dtype=jnp.uint32),
        )
        chex.assert_trees_all_equal(
            after.cache_lineage_words,
            before.live_lineage_words[target],
        )
        chex.assert_trees_all_equal(
            after.cache_source_birth_words,
            before.bound_birth_words[target],
        )
        # The frozen archive is the pre-event victim, never the reset/updated
        # reward model present in the post-allocation context slot.
        chex.assert_trees_all_equal(
            after.cache_reward_weights,
            source_context.reward_weights[target],
        )
        assert bool(
            jnp.any(after.cache_reward_weights != post_context.reward_weights[target])
        )


def test_cache_ties_and_every_nonfinite_predictive_input_abstain() -> None:
    source = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        3216,
    )
    fresh_tie = step_lineage_cache_retention(0.05, LINEAGE_CACHE_NO_SIGNAL, source)
    assert bool(jnp.all(fresh_tie.trace.cache_tested))
    chex.assert_trees_all_equal(fresh_tie.trace.cache_errors, fresh_tie.trace.fresh_errors)
    assert not bool(jnp.any(fresh_tie.trace.strict_predictive_dominance))
    assert not bool(jnp.any(fresh_tie.trace.cache_matched))

    perfect = _perfect_upcoming_cache(source)
    actions, reward, observations = _upcoming_lineage_experience(perfect)
    full_result = step_lineage_cache_retention(0.05, LINEAGE_CACHE_NO_SIGNAL, perfect)
    target = int(full_result.trace.capacity.post_context_slots[0])
    tied_context = perfect.base.context_0.replace(  # type: ignore[attr-defined]
        reward_weights=perfect.base.context_0.reward_weights.at[
            target, actions[0], actions[1]
        ].set(reward)
    )
    inference = ContextInference(CONTEXT_CONFIG)
    live_tie = lane._propose_lineage_cache(
        inference,
        tied_context,
        full_result.state.base.context_0,
        perfect.base.ledger_0,
        full_result.state.base.ledger_0,
        perfect.lineage_0,
        observations[0],
        actions[0],
        reward,
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    assert bool(live_tie.cache_tested)
    assert float(live_tie.cache_error) == 0.0
    assert float(live_tie.fresh_error) == 0.5
    assert float(live_tie.live_errors[target]) == 0.0
    assert not bool(live_tie.strict_predictive_dominance)
    assert not bool(live_tie.cache_matched)

    for nonfinite in (jnp.nan, jnp.inf):
        for bad_observation, bad_reward in (
            (observations[0].at[0].set(nonfinite), reward),
            (observations[0], jnp.asarray(nonfinite, dtype=jnp.float32)),
        ):
            proposal = lane._propose_lineage_cache(
                inference,
                perfect.base.context_0,
                full_result.state.base.context_0,
                perfect.base.ledger_0,
                full_result.state.base.ledger_0,
                perfect.lineage_0,
                bad_observation,
                actions[0],
                bad_reward,
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
            )
            assert not bool(proposal.strict_predictive_dominance)
            assert not bool(proposal.cache_matched)
        bad_cache = perfect.lineage_0.replace(  # type: ignore[attr-defined]
            cache_reward_weights=perfect.lineage_0.cache_reward_weights.at[0, 0].set(
                nonfinite
            )
        )
        proposal = lane._propose_lineage_cache(
            inference,
            perfect.base.context_0,
            full_result.state.base.context_0,
            perfect.base.ledger_0,
            full_result.state.base.ledger_0,
            bad_cache,
            observations[0],
            actions[0],
            reward,
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(True),
        )
        assert not bool(proposal.source_valid)
        assert not bool(proposal.strict_predictive_dominance)
        assert not bool(proposal.cache_matched)


def test_unmatched_cache_retention_is_value_then_exact_source_recency() -> None:
    newer_victim = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        3646,
    )
    higher_cache = _set_lineage_rescue_values(
        newer_victim,
        cache_value=2,
        victim_value=1,
    )
    retained = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        higher_cache,
    )
    assert not bool(jnp.any(retained.trace.cache_matched))
    assert bool(jnp.all(retained.trace.old_cache_retained))
    assert not bool(jnp.any(retained.trace.victim_archived))

    higher_victim = _set_lineage_rescue_values(
        newer_victim,
        cache_value=1,
        victim_value=2,
    )
    archived = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        higher_victim,
    )
    assert bool(jnp.all(archived.trace.victim_archived))
    assert not bool(jnp.any(archived.trace.old_cache_retained))

    equal_newer = _set_lineage_rescue_values(
        newer_victim,
        cache_value=0,
        victim_value=0,
    )
    equal_newer_result = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        equal_newer,
    )
    assert bool(jnp.all(equal_newer_result.trace.victim_archived))

    older_victim = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        3216,
    )
    equal_older_result = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        older_victim,
    )
    assert bool(jnp.all(equal_older_result.trace.old_cache_retained))
    assert not bool(jnp.any(equal_older_result.trace.victim_archived))


def test_invalid_lineage_sidecar_and_terminal_rescue_overflow_roll_back_jointly() -> None:
    initial = initialize_lineage_cache_retention_state(0.2)
    invalid = initial.replace(  # type: ignore[attr-defined]
        lineage_0=initial.lineage_0.replace(  # type: ignore[attr-defined]
            bound_birth_words=initial.lineage_0.bound_birth_words.at[0].set(
                jnp.asarray((0, 1), dtype=jnp.uint32)
            )
        )
    )
    rejected = step_lineage_cache_retention(0.2, LINEAGE_CACHE_NO_SIGNAL, invalid)
    assert not bool(rejected.trace.capacity.update_applied)
    assert not bool(rejected.trace.lineage_source_valid[0])
    chex.assert_trees_all_equal(rejected.state, invalid)

    source = advance_consumed_lineage_cache_retention_state(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        3216,
    )
    source = _perfect_upcoming_cache(source)
    terminal = jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
    overflow = source.replace(  # type: ignore[attr-defined]
        lineage_0=source.lineage_0.replace(  # type: ignore[attr-defined]
            cache_rescue_words=terminal
        ),
        lineage_1=source.lineage_1.replace(  # type: ignore[attr-defined]
            cache_rescue_words=terminal
        ),
    )
    overflow_result = step_lineage_cache_retention(
        0.05,
        LINEAGE_CACHE_NO_SIGNAL,
        overflow,
    )
    assert bool(jnp.all(overflow_result.trace.cache_tested))
    assert not bool(jnp.any(overflow_result.trace.lineage_source_valid))
    assert not bool(jnp.any(overflow_result.trace.rescue_capacity_available))
    assert not bool(overflow_result.trace.capacity.update_applied)
    chex.assert_trees_all_equal(overflow_result.state, overflow)
