# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for the live persistent STOMP lifecycle composition."""

from __future__ import annotations

import copy
import dataclasses
import functools

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    option_semantic_digest,
)
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.core.stomp_option_lifecycle import (
    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED,
    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY,
    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE,
    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID,
    STOMP_OPTION_LIFECYCLE_CURATION_AUTHORITY,
    STOMP_OPTION_LIFECYCLE_DISPATCH_AUTHORITY,
    STOMP_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY,
    STOMP_OPTION_LIFECYCLE_PROMOTION_AUTHORITY,
    STOMP_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED,
    STOMPOptionLifecycle,
    STOMPOptionLifecycleConfig,
    STOMPOptionLifecycleState,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]

SOURCE = option_semantic_digest({"source": "stomp-live-test"})
REPRESENTATION = option_semantic_digest({"representation": "obs2-v1"})
LIFECYCLE = jnp.asarray([0xA17E, 0x51DE], dtype=jnp.uint32)


def _stomp_config(
    *,
    first_threshold: float = 0.8,
    backups: int = 0,
    second_feature: int = 1,
) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=first_threshold,
                pseudo_reward_scale=1.0,
                max_option_steps=5,
            ),
            SubtaskSpec(
                feature_index=second_feature,
                threshold=10.0,
                pseudo_reward_scale=1.0,
                max_option_steps=5,
            ),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=0.0,
        base_avg_reward_step_size=0.0,
        base_trace_decay=0.0,
        option_step_size=0.0,
        option_avg_reward_step_size=0.0,
        option_trace_decay=0.0,
        option_gamma=0.5,
        option_model_decay=0.0,
        option_model_step_size=1.0,
        option_planning_backups_per_step=backups,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _audit_config(
    *,
    fixed_horizon: int = 1,
    max_observations: int = 64,
) -> OptionLifecycleAuditConfig:
    return OptionLifecycleAuditConfig(
        n_options=2,
        n_contexts=2,
        outcome_dim=2,
        fixed_horizon=fixed_horizon,
        maintenance_budget=1,
        signature_scales=(1.0,) * 7,
        initiation_opportunity_floor=1,
        completion_evidence_floor=1,
        model_error_evidence_floor=1,
        comparison_treatment_evidence_floor=1,
        comparison_primitive_evidence_floor=1,
        signature_evidence_floor_per_context=1,
        redundancy_shared_context_floor=1,
        max_planning_uses_per_observation=8,
        max_compute_cost_per_observation=10.0,
        max_observations=max_observations,
    )


@functools.cache
def _wrapper(
    *,
    enabled: bool = True,
    first_threshold: float = 0.8,
    backups: int = 0,
    second_feature: int = 1,
    max_observations: int = 64,
) -> STOMPOptionLifecycle:
    agent = STOMPAgent(
        _stomp_config(
            first_threshold=first_threshold,
            backups=backups,
            second_feature=second_feature,
        )
    )
    audit = OptionLifecycleAudit(_audit_config(max_observations=max_observations))
    return STOMPOptionLifecycle(
        agent,
        audit,
        STOMPOptionLifecycleConfig(audit_enabled=enabled),
    )


def _init(wrapper: STOMPOptionLifecycle, seed: int = 0) -> STOMPOptionLifecycleState:
    return wrapper.init(
        jr.key(seed),
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        lifecycle_id=LIFECYCLE,
    )


def _force_extended_action(
    wrapper: STOMPOptionLifecycle,
    state: STOMPOptionLifecycleState,
    extended_action: int,
) -> STOMPOptionLifecycleState:
    total = wrapper.stomp_agent.config.n_total_actions
    weights = tuple(
        jnp.asarray(
            [[
                10.0 if index == extended_action else -10.0,
                10.0 if index == extended_action else -10.0,
            ]],
            dtype=jnp.float32,
        )
        for index in range(total)
    )
    learner = state.stomp_state.base_learner_state
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=weights,
            biases=tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in range(total)),
        )
    )
    stomp = state.stomp_state.replace(base_learner_state=learner)
    return wrapper._with_checksum(dataclasses.replace(state, stomp_state=stomp))


def _start_option(
    wrapper: STOMPOptionLifecycle,
    state: STOMPOptionLifecycleState,
    option: int = 0,
) -> STOMPOptionLifecycleState:
    state = _force_extended_action(
        wrapper,
        state,
        wrapper.stomp_agent.config.n_primitive_actions + option,
    )
    result = wrapper.start(state, jnp.asarray([0.0, 1.0], dtype=jnp.float32))
    assert bool(result.applied)
    assert int(result.state.stomp_state.executing_option) == option
    return result.state


def test_config_semantics_resources_and_authority_are_exact() -> None:
    wrapper = _wrapper()
    state = _init(wrapper)
    assert bool(wrapper.state_valid(state))
    assert wrapper.semantic_digests.shape == (2, 8)
    assert not bool(jnp.array_equal(wrapper.semantic_digests[0], wrapper.semantic_digests[1]))
    budget = wrapper.resource_budget(state)
    assert budget.wrapped_persistent_state_nbytes == (
        budget.stomp_persistent_state_nbytes
        + budget.audit_persistent_state_nbytes
        + budget.composition_binding_nbytes
    )
    assert budget.additional_rng_draws_per_update == 0
    assert budget.additional_backward_passes_per_update == 0
    assert budget.additional_consumer_calls_per_update == 0
    assert budget.composition_binding_nbytes == 58
    assert budget.max_audited_observations == 64
    assert budget.stomp_lifetime_identity_bits == 64
    assert budget.wrapper_revision_saturation == 2_147_483_647
    assert budget.audit_capacity_can_block_stomp is False
    assert budget.curation_authority is False
    assert budget.dispatch_authority is False
    assert STOMP_OPTION_LIFECYCLE_CURATION_AUTHORITY is False
    assert STOMP_OPTION_LIFECYCLE_PROMOTION_AUTHORITY is False
    assert STOMP_OPTION_LIFECYCLE_DISPATCH_AUTHORITY is False
    assert STOMP_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY is False
    assert STOMP_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED is False
    restored = STOMPOptionLifecycleConfig.from_config(wrapper.config.to_config())
    assert restored == wrapper.config
    with pytest.raises(ValueError, match="exact Python bool"):
        STOMPOptionLifecycleConfig(audit_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="planning budget"):
        STOMPOptionLifecycle(
            STOMPAgent(_stomp_config(backups=9)),
            OptionLifecycleAudit(_audit_config()),
        )


def test_disabled_wrapper_preserves_exact_stomp_rng_and_learning_state() -> None:
    wrapper = _wrapper(enabled=False)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    direct_started = wrapper.stomp_agent.start(
        state.stomp_state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    wrapped_started = wrapper.start(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(wrapped_started.state.stomp_state, direct_started)
    audit_before = wrapped_started.state.audit_state
    direct = wrapper.stomp_agent.update(
        direct_started,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    wrapped = wrapper.update(
        wrapped_started.state,
        1.0,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        0.9,
    )
    assert bool(wrapped.transaction_applied)
    chex.assert_trees_all_equal(wrapped.state.stomp_state, direct.state)
    chex.assert_trees_all_equal(wrapped.state.stomp_state.rng_key, direct.state.rng_key)
    chex.assert_trees_all_equal(wrapped.state.audit_state, audit_before)


def test_actual_option_lifecycle_drives_discounted_signature_and_natural_reasons() -> None:
    wrapper = _wrapper()
    state = _start_option(wrapper, _init(wrapper))
    first = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.2, 1.0], dtype=jnp.float32),
        0.5,
        context=1,
        comparator_randomized=True,
        treatment_propensity=0.5,
    )
    assert bool(first.transaction_applied)
    assert int(first.state.audit_state.active_option) == 0
    assert int(first.state.audit_state.active_context) == 1
    assert not bool(first.option_terminated)
    second = wrapper.update(
        first.state,
        2.0,
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        0.5,
    )
    assert bool(second.transaction_applied)
    assert bool(second.natural_completion)
    assert bool(second.option_terminated)
    assert not bool(second.censor_only_ending)
    audit_state = second.state.audit_state
    # env return=1 + 0.5*2; pseudo=.2+1; duration=2;
    # baseline mass=1+.5; discount=.25; final-start delta=[1,0].
    expected = jnp.asarray([2.0, 1.2, 2.0, 1.5, 0.25, 1.0, 0.0], jnp.float32)
    np.testing.assert_allclose(audit_state.completion_signature_sums[0], expected)
    frozen = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], jnp.float32)
    np.testing.assert_allclose(
        audit_state.model_squared_error_sums[0],
        (expected - frozen) ** 2,
    )
    assert int(audit_state.goal_terminations[0]) == 1
    assert int(audit_state.timeout_terminations[0]) == 0
    assert int(audit_state.comparison_treatment_counts[0, 1]) == 1
    assert int(audit_state.comparison_primitive_counts[0, 1]) == 0
    assert int(second.state.stomp_state.option_models.n_completions[0]) == 1
    np.testing.assert_array_equal(second.transaction_identity[:2], LIFECYCLE)
    np.testing.assert_array_equal(
        second.transaction_identity[2:],
        jnp.asarray([0, 2], dtype=jnp.uint32),
    )


def test_execution_boundary_is_censor_only_and_never_a_model_completion() -> None:
    wrapper = _wrapper(first_threshold=100.0)
    state = _start_option(wrapper, _init(wrapper))
    result = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.2, 1.0], dtype=jnp.float32),
        0.9,
        context=0,
        execution_boundary=True,
    )
    assert bool(result.transaction_applied)
    assert bool(result.option_terminated)
    assert bool(result.censor_only_ending)
    assert not bool(result.natural_completion)
    assert int(result.state.audit_state.censor_only_endings[0]) == 1
    assert int(result.state.audit_state.model_error_counts[0]) == 0
    assert int(result.state.stomp_state.option_models.n_completions[0]) == 0


def test_corrupt_persistent_sidecar_fails_closed_for_checkpoint_recovery() -> None:
    wrapper = _wrapper(enabled=True)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    corrupt_audit = dataclasses.replace(
        state.audit_state,
        state_checksum=jnp.zeros((2,), dtype=jnp.uint32),
    )
    corrupt = wrapper._with_checksum(dataclasses.replace(state, audit_state=corrupt_audit))
    direct = wrapper.stomp_agent.update(
        corrupt.stomp_state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(direct.update_applied)
    result = wrapper.update(
        corrupt,
        1.0,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        0.9,
    )
    assert bool(result.stomp_update_applied)
    assert not bool(result.audit_applied)
    assert not bool(result.audit_sidecar_accepted)
    assert not bool(result.transaction_applied)
    assert bool(result.rolled_back)
    assert int(result.audit_error) == (
        STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID
    )
    chex.assert_trees_all_equal(result.state, corrupt)


def test_invalid_external_attribution_stops_only_the_observer() -> None:
    wrapper = _wrapper(enabled=True)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    audit_before = state.audit_state
    direct = wrapper.stomp_agent.update(
        state.stomp_state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    result = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        0.9,
        context=99,
    )
    assert bool(result.transaction_applied)
    assert not bool(result.rolled_back)
    assert not bool(result.audit_applied)
    assert bool(result.audit_capacity_available)
    assert bool(result.audit_unavailable_noop)
    assert int(result.audit_error) == (
        STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED
    )
    assert bool(result.state.audit_unavailable)
    assert bool(wrapper.state_valid(result.state))
    chex.assert_trees_all_equal(result.state.stomp_state, direct.state)
    chex.assert_trees_all_equal(result.state.audit_state, audit_before)

    second_direct = wrapper.stomp_agent.update(
        result.state.stomp_state,
        jnp.asarray(2.0, dtype=jnp.float32),
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    second = wrapper.update(
        result.state,
        2.0,
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        0.8,
        context=0,
    )
    assert bool(second.transaction_applied)
    assert not bool(second.audit_applied)
    assert bool(second.audit_unavailable_noop)
    assert int(second.audit_error) == (
        STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED
    )
    chex.assert_trees_all_equal(second.state.stomp_state, second_direct.state)
    chex.assert_trees_all_equal(second.state.audit_state, audit_before)


def test_audit_capacity_exhaustion_never_freezes_real_stomp() -> None:
    wrapper = _wrapper(enabled=True, max_observations=1)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    first = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        0.9,
        context=0,
    )
    assert bool(first.transaction_applied)
    assert bool(first.audit_applied)
    assert bool(first.audit_capacity_available)
    assert not bool(first.audit_unavailable_noop)
    assert bool(first.state.audit_unavailable)
    assert int(first.state.audit_error) == STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY
    assert int(first.state.audit_state.observation_count) == 1
    terminal_audit = first.state.audit_state

    direct = wrapper.stomp_agent.update(
        first.state.stomp_state,
        jnp.asarray(2.0, dtype=jnp.float32),
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    second = wrapper.update(
        first.state,
        2.0,
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        0.8,
        context=0,
    )
    assert bool(second.transaction_applied)
    assert not bool(second.rolled_back)
    assert not bool(second.audit_applied)
    assert not bool(second.audit_capacity_available)
    assert bool(second.audit_unavailable_noop)
    assert int(second.audit_error) == STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY
    assert bool(wrapper.state_valid(second.state))
    chex.assert_trees_all_equal(second.state.stomp_state, direct.state)
    chex.assert_trees_all_equal(second.state.audit_state, terminal_audit)


def test_invalid_real_stomp_transaction_is_an_exact_composed_noop() -> None:
    wrapper = _wrapper(enabled=True)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    result = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        context=0,
    )
    assert not bool(result.stomp_update_applied)
    assert not bool(result.transaction_applied)
    assert not bool(result.audit_applied)
    assert not bool(result.audit_unavailable_noop)
    assert int(result.audit_error) == STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE
    chex.assert_trees_all_equal(result.state, state)


def test_randomized_primitive_declaration_is_bound_to_actual_idle_owner() -> None:
    wrapper = _wrapper()
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    assert int(state.stomp_state.executing_option) == -1
    result = wrapper.update(
        state,
        3.0,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        1.0,
        context=1,
        idle_candidate_option=1,
        idle_initiation_eligible=True,
        comparator_randomized=True,
        treatment_propensity=0.5,
    )
    assert bool(result.transaction_applied)
    assert int(result.state.audit_state.comparison_primitive_counts[1, 1]) == 1
    assert int(result.state.audit_state.comparison_treatment_counts[1, 1]) == 0


def test_planning_backups_are_attributed_to_the_actual_completed_models() -> None:
    wrapper = _wrapper(backups=2)
    state = _force_extended_action(wrapper, _init(wrapper), 0)
    models = state.stomp_state.option_models.replace(
        n_completions=jnp.asarray([1, 1], dtype=jnp.int32),
        env_return_ema=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        duration_ema=jnp.ones((2,), dtype=jnp.float32),
        baseline_mass_ema=jnp.ones((2,), dtype=jnp.float32),
        discount_ema=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    stomp = state.stomp_state.replace(option_models=models)
    state = wrapper._with_checksum(dataclasses.replace(state, stomp_state=stomp))
    state = wrapper.start(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32)).state
    result = wrapper.update(
        state,
        0.0,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        1.0,
    )
    assert bool(result.transaction_applied)
    assert int(jnp.sum(result.planning_usage)) == 2
    np.testing.assert_array_equal(
        result.state.audit_state.planning_use_counts,
        result.planning_usage,
    )


def test_mid_option_checkpoint_resume_has_exact_next_update_parity() -> None:
    wrapper = _wrapper(first_threshold=100.0)
    state = _start_option(wrapper, _init(wrapper))
    state = wrapper.update(
        state,
        1.0,
        jnp.asarray([0.2, 1.0], dtype=jnp.float32),
        0.5,
        context=1,
    ).state
    assert int(state.stomp_state.executing_option) == 0
    assert int(state.audit_state.active_option) == 0
    payload = wrapper.checkpoint_payload(state)
    resumed = wrapper.restore_checkpoint(
        copy.deepcopy(payload),
        expected_source_digest=SOURCE,
        expected_representation_digest=REPRESENTATION,
        expected_lifecycle_id=LIFECYCLE,
    )
    chex.assert_trees_all_equal(resumed, state)
    direct = wrapper.update(
        state,
        2.0,
        jnp.asarray([0.4, 1.0], dtype=jnp.float32),
        0.5,
    )
    after_resume = wrapper.update(
        resumed,
        2.0,
        jnp.asarray([0.4, 1.0], dtype=jnp.float32),
        0.5,
    )
    chex.assert_trees_all_equal(after_resume, direct)

    tampered = copy.deepcopy(payload)
    tampered["state_digest"] = jnp.zeros((32,), dtype=jnp.uint8)
    with pytest.raises(ValueError, match="state digest"):
        wrapper.restore_checkpoint(
            tampered,
            expected_source_digest=SOURCE,
            expected_representation_digest=REPRESENTATION,
            expected_lifecycle_id=LIFECYCLE,
        )


def _filled_idle_state(
    wrapper: STOMPOptionLifecycle,
    state: STOMPOptionLifecycleState,
) -> STOMPOptionLifecycleState:
    policies = state.stomp_state.option_policies.replace(
        q_weights=state.stomp_state.option_policies.q_weights.at[1].set(9.0),
        traces=state.stomp_state.option_policies.traces.at[1].set(8.0),
        average_rewards=state.stomp_state.option_policies.average_rewards.at[1].set(7.0),
    )
    models = state.stomp_state.option_models.replace(
        cumreward_ema=state.stomp_state.option_models.cumreward_ema.at[1].set(6.0),
        env_return_ema=state.stomp_state.option_models.env_return_ema.at[1].set(5.0),
        duration_ema=state.stomp_state.option_models.duration_ema.at[1].set(4.0),
        baseline_mass_ema=state.stomp_state.option_models.baseline_mass_ema.at[1].set(3.0),
        discount_ema=state.stomp_state.option_models.discount_ema.at[1].set(0.5),
        next_state_weights=state.stomp_state.option_models.next_state_weights.at[1].set(2.0),
        n_completions=state.stomp_state.option_models.n_completions.at[1].set(1),
    )
    stomp = state.stomp_state.replace(option_policies=policies, option_models=models)
    return wrapper._with_checksum(dataclasses.replace(state, stomp_state=stomp))


def test_new_shape_compatible_wrapper_resets_only_changed_semantic_slot() -> None:
    old_wrapper = _wrapper(second_feature=1)
    old = _filled_idle_state(old_wrapper, _init(old_wrapper))
    new_wrapper = _wrapper(second_feature=0)
    fresh_key = jr.key(99)
    fresh = new_wrapper.stomp_agent.init(fresh_key)
    result = new_wrapper.rebind(
        old,
        fresh_key,
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
    )
    assert bool(result.applied)
    np.testing.assert_array_equal(result.preserved_slots, [True, False])
    np.testing.assert_array_equal(result.reset_slots, [False, True])
    np.testing.assert_array_equal(
        result.state.stomp_state.option_policies.q_weights[0],
        old.stomp_state.option_policies.q_weights[0],
    )
    np.testing.assert_array_equal(
        result.state.stomp_state.option_policies.q_weights[1],
        fresh.option_policies.q_weights[1],
    )
    np.testing.assert_array_equal(
        result.state.stomp_state.option_models.n_completions[1],
        fresh.option_models.n_completions[1],
    )
    changed_head = new_wrapper.stomp_agent.config.n_primitive_actions + 1
    chex.assert_trees_all_equal(
        result.state.stomp_state.base_learner_state.head_optimizer_states[changed_head],
        fresh.base_learner_state.head_optimizer_states[changed_head],
    )
    assert int(result.state.audit_state.semantic_generations[1]) == 1
    chex.assert_trees_all_equal(result.state.stomp_state.rng_key, old.stomp_state.rng_key)


def test_semantic_rebind_defers_while_real_option_is_in_flight() -> None:
    old_wrapper = _wrapper(second_feature=1)
    active = _start_option(old_wrapper, _init(old_wrapper), option=1)
    new_wrapper = _wrapper(second_feature=0)
    result = new_wrapper.rebind(
        active,
        jr.key(5),
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
    )
    assert bool(result.deferred)
    assert not bool(result.applied)
    np.testing.assert_array_equal(result.preserved_slots, [True, False])
    np.testing.assert_array_equal(result.reset_slots, [False, False])
    chex.assert_trees_all_equal(result.state, active)
