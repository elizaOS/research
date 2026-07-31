"""Unit tests for the fixed-capacity hidden-regime signaling learners."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    N_SLOTS,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    SLOT_VACANT,
    SLOT_VALUE_SHAPE,
    SlotSignalingAgent,
    SlotSignalingConfig,
    SlotSignalingState,
    greedy_slot_action,
    slot_role_resource_budget,
    slot_signaling_keys,
    slot_signaling_resource_budget,
)

pytestmark = pytest.mark.unit


def _agent(
    **config_overrides: object,
) -> tuple[SlotSignalingAgent, SlotSignalingState]:
    config = SlotSignalingConfig(**config_overrides)  # type: ignore[arg-type]
    agent = SlotSignalingAgent(config)
    return agent, agent.init(slot_signaling_keys(jr.key(17)))


def _replace_both(
    state: SlotSignalingState,
    **role_changes: object,
) -> SlotSignalingState:
    return SlotSignalingState(
        helper=dataclasses.replace(state.helper, **role_changes),
        beneficiary=dataclasses.replace(state.beneficiary, **role_changes),
    )


def _one_update(
    agent: SlotSignalingAgent,
    state: SlotSignalingState,
    reward: float,
    *,
    helper_write: bool = True,
    beneficiary_write: bool = True,
):
    helper = agent.select_helper(state.helper, jnp.int32(0))
    beneficiary = agent.select_beneficiary(state.beneficiary, helper.action)
    return agent.update(
        state,
        helper,
        beneficiary,
        jnp.float32(reward),
        helper_write=helper_write,
        beneficiary_write=beneficiary_write,
    )


def test_config_is_strict_and_explicitly_development_only() -> None:
    invalid = (
        {"learning_rate": True},
        {"learning_rate": 0.0},
        {"epsilon": float("nan")},
        {"relevance_rate": 1.1},
        {"lease_length": True},
        {"lease_length": 0},
        {"confirmation_steps": 0},
        {"lease_length": 2, "confirmation_steps": 3},
        {"durable_retrieval_threshold": float("inf")},
        {"durable_retrieval_threshold": -0.1},
        {"candidate_confirmation_threshold": float("nan")},
        {"candidate_confirmation_threshold": 1.1},
        {"durable_retrieval_threshold": 0.75, "candidate_confirmation_threshold": 0.75},
        {"durable_retrieval_threshold": 0.8, "candidate_confirmation_threshold": 0.75},
        {"candidate_confirmation_leases": False},
        {"candidate_confirmation_leases": 0},
        {"scratch_training_leases_before_retest": False},
        {"scratch_training_leases_before_retest": 0},
        {"scratch_training_leases_before_retest": np.iinfo(np.int32).max + 1},
        {"writable_lru_ablation": 1},
        {"lease_length": np.iinfo(np.int32).max + 1},
        {"candidate_confirmation_leases": np.iinfo(np.int32).max + 1},
        {"durable_write_policy": "oracle"},
        {"replacement_target_policy": "oracle"},
        {"durable_write_policy": DURABLE_WRITE_SELECTIVE},
        {"replacement_target_policy": REPLACEMENT_TARGET_EVIDENCE},
        {
            "writable_lru_ablation": True,
            "durable_write_policy": DURABLE_WRITE_WRITABLE,
            "replacement_target_policy": REPLACEMENT_TARGET_LRU,
        },
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            SlotSignalingConfig(**kwargs)  # type: ignore[arg-type]
    payload = SlotSignalingConfig().to_dict()
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["durable_retrieval_threshold"] == 0.5
    assert payload["candidate_confirmation_threshold"] == 0.75
    assert payload["candidate_confirmation_leases"] == 2
    assert payload["scratch_training_leases_before_retest"] == 1
    assert payload["requested_durable_write_policy"] is None
    assert payload["requested_replacement_target_policy"] is None
    assert payload["effective_durable_write_policy"] == DURABLE_WRITE_SELECTIVE
    assert payload["effective_replacement_target_policy"] == REPLACEMENT_TARGET_EVIDENCE
    assert "retirement_failures" not in payload
    assert "retirement_grace_leases" not in payload


def test_explicit_factorial_policies_are_independent_and_legacy_resolves_exactly() -> None:
    configs = {
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE): SlotSignalingConfig(
            durable_write_policy=DURABLE_WRITE_SELECTIVE,
            replacement_target_policy=REPLACEMENT_TARGET_EVIDENCE,
        ),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_EVIDENCE): SlotSignalingConfig(
            durable_write_policy=DURABLE_WRITE_WRITABLE,
            replacement_target_policy=REPLACEMENT_TARGET_EVIDENCE,
        ),
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_LRU): SlotSignalingConfig(
            durable_write_policy=DURABLE_WRITE_SELECTIVE,
            replacement_target_policy=REPLACEMENT_TARGET_LRU,
        ),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU): SlotSignalingConfig(
            durable_write_policy=DURABLE_WRITE_WRITABLE,
            replacement_target_policy=REPLACEMENT_TARGET_LRU,
        ),
    }
    assert {
        (config.effective_durable_write_policy, config.effective_replacement_target_policy)
        for config in configs.values()
    } == set(configs)
    legacy = SlotSignalingConfig(writable_lru_ablation=True)
    assert legacy.effective_durable_write_policy == DURABLE_WRITE_WRITABLE
    assert legacy.effective_replacement_target_policy == REPLACEMENT_TARGET_LRU


def _assert_pytree_bit_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jr.key_data(left_leaf), jr.key_data(right_leaf))
        else:
            np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


@pytest.mark.parametrize(
    "compatibility,explicit",
    [
        (
            SlotSignalingConfig(lease_length=1, confirmation_steps=1),
            SlotSignalingConfig(
                lease_length=1,
                confirmation_steps=1,
                durable_write_policy=DURABLE_WRITE_SELECTIVE,
                replacement_target_policy=REPLACEMENT_TARGET_EVIDENCE,
            ),
        ),
        (
            SlotSignalingConfig(
                lease_length=1,
                confirmation_steps=1,
                writable_lru_ablation=True,
            ),
            SlotSignalingConfig(
                lease_length=1,
                confirmation_steps=1,
                durable_write_policy=DURABLE_WRITE_WRITABLE,
                replacement_target_policy=REPLACEMENT_TARGET_LRU,
            ),
        ),
    ],
)
def test_compatibility_and_explicit_policies_are_transition_bit_exact(
    compatibility: SlotSignalingConfig,
    explicit: SlotSignalingConfig,
) -> None:
    agents = (SlotSignalingAgent(compatibility), SlotSignalingAgent(explicit))
    states = [agent.init(slot_signaling_keys(jr.key(103))) for agent in agents]
    for reward in (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0):
        updates = []
        for agent, state in zip(agents, states, strict=True):
            helper = agent.select_helper(state.helper, jnp.int32(1))
            beneficiary = agent.select_beneficiary(state.beneficiary, helper.action)
            updates.append(
                agent.update(state, helper, beneficiary, jnp.float32(reward))
            )
        _assert_pytree_bit_equal(updates[0], updates[1])
        states = [update.state for update in updates]


def test_zero_state_has_three_vacancies_one_scratch_and_exact_resources() -> None:
    _, state = _agent()
    for role in (state.helper, state.beneficiary):
        np.testing.assert_array_equal(role.values, np.zeros(SLOT_VALUE_SHAPE, np.float32))
        np.testing.assert_array_equal(
            role.status,
            np.asarray((SLOT_SCRATCH, SLOT_VACANT, SLOT_VACANT, SLOT_VACANT)),
        )
        assert int(role.active_slot) == SCRATCH_SLOT
        budget = slot_role_resource_budget(role)
        assert budget.value_scalars == 36
        assert budget.relevance_scalars == 8
        assert budget.lifecycle_scalars == 23
        assert budget.key_scalars == 2
        assert budget.state_scalars == 69
        assert budget.state_bytes == 276
        np.testing.assert_array_equal(role.generation, np.zeros((N_SLOTS,), np.int32))
        assert int(role.remaining_durable_tests) == 0
        assert int(role.search_cursor) == 1
        assert int(role.candidate_successful_leases) == 0
        assert int(role.next_generation) == 1
    assert not np.array_equal(
        jr.key_data(state.helper.key),
        jr.key_data(state.beneficiary.key),
    )
    joint = slot_signaling_resource_budget(state)
    assert joint.state_scalars == 138
    assert joint.state_bytes == 552


def test_policy_api_has_no_regime_target_schedule_or_other_role_input() -> None:
    helper = inspect.signature(SlotSignalingAgent.select_helper).parameters
    beneficiary = inspect.signature(SlotSignalingAgent.select_beneficiary).parameters
    update = inspect.signature(SlotSignalingAgent.update).parameters
    assert set(helper) == {"self", "state", "private_cue"}
    assert set(beneficiary) == {"self", "state", "delivered_message"}
    for forbidden in ("regime", "target", "schedule", "oracle"):
        assert forbidden not in helper
        assert forbidden not in beneficiary
        assert forbidden not in update


def test_success_stays_then_failure_exhausts_stored_slots_before_scratch() -> None:
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    state = _replace_both(initial, status=status, active_slot=jnp.int32(1))

    success = _one_update(agent, state, 1.0)
    assert int(success.state.helper.active_slot) == 1
    assert int(success.state.helper.remaining_durable_tests) == 0

    first_failure = _one_update(agent, success.state, 0.0)
    assert int(first_failure.state.helper.active_slot) == 2
    assert int(first_failure.state.helper.remaining_durable_tests) == 2
    second_failure = _one_update(agent, first_failure.state, 0.0)
    assert int(second_failure.state.helper.active_slot) == 3
    assert int(second_failure.state.helper.remaining_durable_tests) == 1
    final_failure = _one_update(agent, second_failure.state, 0.0)
    assert int(final_failure.state.helper.active_slot) == SCRATCH_SLOT
    assert int(final_failure.state.helper.remaining_durable_tests) == 0

    scratch = _one_update(agent, final_failure.state, 1.0)
    assert bool(scratch.helper.value_write)
    assert np.count_nonzero(np.asarray(scratch.state.helper.values[SCRATCH_SLOT])) > 0
    assert bool(scratch.lifecycle_synchronized)


def test_default_one_scratch_lease_reproduces_immediate_retest() -> None:
    common = {
        "relevance_rate": 1.0,
        "lease_length": 1,
        "confirmation_steps": 1,
    }
    implicit, initial = _agent(**common)
    explicit = SlotSignalingAgent(
        SlotSignalingConfig(
            **common,  # type: ignore[arg-type]
            scratch_training_leases_before_retest=1,
        )
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)
    state = _replace_both(initial, status=status, generation=generation)
    implicit_update = _one_update(implicit, state, 0.0)
    explicit_update = _one_update(explicit, state, 0.0)
    for update in (implicit_update, explicit_update):
        assert int(update.state.helper.active_slot) == 1
        assert int(update.state.helper.remaining_durable_tests) == 3
        assert int(update.state.helper.failed_leases[SCRATCH_SLOT]) == 0
        assert int(update.helper.scratch_failed_leases_pre) == 0
        assert int(update.helper.scratch_failed_leases_post) == 0
        assert bool(update.helper.scratch_retest_started)
        assert bool(update.lifecycle_synchronized)
    np.testing.assert_array_equal(
        implicit_update.state.helper.failed_leases,
        explicit_update.state.helper.failed_leases,
    )
    np.testing.assert_array_equal(
        implicit_update.state.helper.status,
        explicit_update.state.helper.status,
    )


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("helper_write,beneficiary_write", [(True, True), (False, True)])
def test_scratch_training_residency_is_write_mask_and_ablation_neutral(
    writable: bool,
    helper_write: bool,
    beneficiary_write: bool,
) -> None:
    # Three is an arbitrary adversarial unit-test value, not a selected protocol value.
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        scratch_training_leases_before_retest=3,
        writable_lru_ablation=writable,
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)
    state = _replace_both(initial, status=status, generation=generation)
    for expected_count in (1, 2):
        update = _one_update(
            agent,
            state,
            0.0,
            helper_write=helper_write,
            beneficiary_write=beneficiary_write,
        )
        state = update.state
        assert int(state.helper.active_slot) == SCRATCH_SLOT
        assert int(state.helper.failed_leases[SCRATCH_SLOT]) == expected_count
        assert int(update.helper.scratch_failed_leases_post) == expected_count
        assert not bool(update.helper.scratch_retest_started)
        assert bool(update.lifecycle_synchronized)
    retest = _one_update(
        agent,
        state,
        0.0,
        helper_write=helper_write,
        beneficiary_write=beneficiary_write,
    )
    assert int(retest.state.helper.active_slot) == 1
    assert int(retest.state.helper.remaining_durable_tests) == 3
    assert int(retest.state.helper.failed_leases[SCRATCH_SLOT]) == 0
    assert bool(retest.helper.scratch_retest_started)
    assert bool(retest.lifecycle_synchronized)
    assert slot_signaling_resource_budget(retest.state).state_bytes == 552


def test_scratch_failure_counter_resets_on_exhaustion_success_and_commit() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=2,
        scratch_training_leases_before_retest=3,
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)
    failed = jnp.asarray((2, 0, 0, 0), dtype=jnp.int32)

    searching = _replace_both(
        initial,
        status=status,
        generation=generation,
        failed_leases=failed,
        active_slot=jnp.int32(1),
    )
    for expected_slot in (2, 3, SCRATCH_SLOT):
        search_update = _one_update(agent, searching, 0.0)
        searching = search_update.state
        assert int(searching.helper.active_slot) == expected_slot
    assert int(searching.helper.failed_leases[SCRATCH_SLOT]) == 0
    assert int(search_update.helper.scratch_failed_leases_post) == 0

    candidate_state = _replace_both(
        initial,
        status=status,
        generation=generation,
        failed_leases=failed,
    )
    candidate_success = _one_update(agent, candidate_state, 1.0)
    assert bool(candidate_success.helper.candidate_lease_success)
    assert int(candidate_success.state.helper.active_slot) == SCRATCH_SLOT
    assert int(candidate_success.state.helper.candidate_successful_leases) == 1
    assert int(candidate_success.state.helper.failed_leases[SCRATCH_SLOT]) == 0

    search_success_state = _replace_both(
        initial,
        status=status,
        generation=generation,
        failed_leases=failed,
        active_slot=jnp.int32(2),
    )
    search_success = _one_update(agent, search_success_state, 1.0)
    assert bool(search_success.helper.durable_relevant)
    assert int(search_success.state.helper.active_slot) == 2
    assert int(search_success.state.helper.failed_leases[SCRATCH_SLOT]) == 0

    commit_state = _replace_both(
        initial,
        status=status,
        generation=generation,
        failed_leases=failed,
        candidate_successful_leases=jnp.int32(1),
        next_generation=jnp.int32(4),
    )
    commit = _one_update(agent, commit_state, 1.0)
    assert int(commit.helper.committed_generation) == 4
    assert int(commit.helper.retired_generation) > 0
    assert int(commit.state.helper.failed_leases[SCRATCH_SLOT]) == 0
    assert not bool(commit.helper.scratch_retest_started)


def test_scratch_failure_counter_saturates_safely_and_routes_under_jit() -> None:
    maximum = np.iinfo(np.int32).max
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        scratch_training_leases_before_retest=maximum,
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_VACANT, SLOT_VACANT))
    generation = jnp.asarray((0, 1, 0, 0), dtype=jnp.int32)
    state = _replace_both(
        initial,
        status=status,
        generation=generation,
        failed_leases=jnp.asarray((maximum - 2, 0, 0, 0), dtype=jnp.int32),
    )
    penultimate = _one_update(agent, state, 0.0)
    assert int(penultimate.state.helper.failed_leases[SCRATCH_SLOT]) == maximum - 1
    assert not bool(penultimate.helper.scratch_retest_started)
    saturated = _one_update(agent, penultimate.state, 0.0)
    assert int(saturated.helper.scratch_failed_leases_pre) == maximum - 1
    assert int(saturated.helper.scratch_failed_leases_post) == 0
    assert bool(saturated.helper.scratch_retest_started)
    assert int(saturated.state.helper.active_slot) == 1

    routed_agent, routed_initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        scratch_training_leases_before_retest=3,
    )
    full_status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    full_generation = jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)
    routed_initial = _replace_both(
        routed_initial,
        status=full_status,
        generation=full_generation,
    )

    @jax.jit
    def run(old_state):
        def body(carry, _):
            update = _one_update(routed_agent, carry, 0.0)
            diagnostics = jnp.stack(
                (
                    update.helper.scratch_failed_leases_post,
                    update.helper.scratch_retest_started.astype(jnp.int32),
                    update.state.helper.active_slot,
                )
            )
            return update.state, diagnostics

        return jax.lax.scan(body, old_state, xs=None, length=4)

    final_state, diagnostics = run(routed_initial)
    np.testing.assert_array_equal(
        diagnostics,
        np.asarray(((1, 0, 0), (2, 0, 0), (0, 1, 1), (0, 0, 2)), dtype=np.int32),
    )
    assert slot_signaling_resource_budget(final_state).state_bytes == 552


def test_search_skips_vacancies_and_never_retests_before_scratch() -> None:
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
    )
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_VACANT, SLOT_DURABLE))
    state = _replace_both(initial, status=status, active_slot=jnp.int32(1))
    first = _one_update(agent, state, 0.0)
    assert int(first.state.helper.active_slot) == 3
    assert int(first.state.helper.remaining_durable_tests) == 1
    second = _one_update(agent, first.state, 0.0)
    assert int(second.state.helper.active_slot) == SCRATCH_SLOT
    assert int(second.state.helper.remaining_durable_tests) == 0


def test_candidate_is_always_formed_while_external_mask_preserves_values() -> None:
    agent, state = _agent(
        learning_rate=0.25,
        lease_length=4,
        confirmation_steps=2,
    )
    before_key = np.asarray(jr.key_data(state.helper.key)).copy()
    update = _one_update(
        agent,
        state,
        1.0,
        helper_write=False,
        beneficiary_write=True,
    )
    assert float(update.helper.value_pre) == 0.0
    assert float(update.helper.candidate_value) == 0.25
    assert not bool(update.helper.value_write)
    assert float(update.helper.value_post) == 0.0
    np.testing.assert_array_equal(update.state.helper.values, state.helper.values)
    assert not np.array_equal(jr.key_data(update.state.helper.key), before_key)
    assert float(update.state.helper.relevance_mean[0]) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        update.state.helper.relevance_mean,
        update.state.beneficiary.relevance_mean,
    )
    assert bool(update.lifecycle_synchronized)


def test_scratch_learns_and_sustained_reward_commits_then_resets_it() -> None:
    agent, state = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        lease_length=2,
        confirmation_steps=2,
        candidate_confirmation_leases=1,
    )
    first = _one_update(agent, state, 1.0)
    assert bool(first.helper.value_write)
    assert int(first.helper.committed_slot) == -1
    second = _one_update(agent, first.state, 1.0)
    assert bool(second.helper.lease_boundary)
    assert int(second.helper.committed_slot) == 1
    assert int(second.beneficiary.committed_slot) == 1
    assert int(second.state.helper.status[1]) == SLOT_DURABLE
    assert int(second.state.beneficiary.status[1]) == SLOT_DURABLE
    assert np.count_nonzero(np.asarray(second.state.helper.values[1])) > 0
    assert np.count_nonzero(np.asarray(second.state.beneficiary.values[1])) > 0
    np.testing.assert_array_equal(second.state.helper.values[0], np.zeros((3, 3)))
    np.testing.assert_array_equal(second.state.beneficiary.values[0], np.zeros((3, 3)))
    assert int(second.state.helper.active_slot) == 1
    assert int(second.helper.committed_generation) == 1
    assert int(second.state.helper.generation[1]) == 1
    assert int(second.state.helper.next_generation) == 2


def test_commit_activates_the_actual_nonfirst_vacancy() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
    )
    status = initial.helper.status.at[1].set(SLOT_DURABLE)
    generation = initial.helper.generation.at[1].set(7)
    state = _replace_both(
        initial,
        status=status,
        generation=generation,
        next_generation=jnp.int32(8),
    )
    update = _one_update(agent, state, 1.0)
    assert int(update.helper.committed_slot) == 2
    assert int(update.helper.committed_generation) == 8
    assert int(update.state.helper.active_slot) == 2
    assert int(update.state.helper.generation[2]) == 8


def test_selective_durable_values_close_but_relevance_remains_separate() -> None:
    selective, initial = _agent(
        learning_rate=0.5,
        epsilon=0.0,
        lease_length=4,
        confirmation_steps=2,
    )
    values = initial.helper.values.at[1, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
    statuses = initial.helper.status.at[1].set(SLOT_DURABLE)
    state = _replace_both(
        initial,
        values=values,
        status=statuses,
        active_slot=jnp.int32(1),
    )
    update = _one_update(selective, state, 0.0)
    assert float(update.helper.candidate_value) == 0.5
    assert not bool(update.helper.value_write)
    np.testing.assert_array_equal(update.state.helper.values, state.helper.values)
    assert float(update.state.helper.relevance_mass[1]) == 1.0
    assert float(update.state.helper.relevance_mean[1]) == 0.0

    writable = SlotSignalingAgent(dataclasses.replace(selective.config, writable_lru_ablation=True))
    ablation = _one_update(writable, state, 0.0)
    assert bool(ablation.helper.value_write)
    assert float(ablation.helper.value_post) == 0.5
    assert slot_signaling_resource_budget(ablation.state).state_bytes == 552


def test_relevance_history_materially_decides_stay_or_search() -> None:
    agent, initial = _agent(
        relevance_rate=0.1,
        lease_length=1,
        confirmation_steps=1,
    )
    status = initial.helper.status.at[1].set(SLOT_DURABLE)
    common = {
        "status": status,
        "active_slot": jnp.int32(1),
        "relevance_mass": initial.helper.relevance_mass.at[1].set(100.0),
    }
    high = _replace_both(
        initial,
        **common,
        relevance_mean=initial.helper.relevance_mean.at[1].set(1.0),
    )
    low = _replace_both(
        initial,
        **common,
        relevance_mean=initial.helper.relevance_mean.at[1].set(0.0),
    )
    high_update = _one_update(agent, high, 0.0)
    low_update = _one_update(agent, low, 0.0)
    assert float(high_update.helper.lease_reward_mean) == 0.0
    assert float(low_update.helper.lease_reward_mean) == 0.0
    assert bool(high_update.helper.durable_relevant)
    assert not bool(low_update.helper.durable_relevant)
    assert int(high_update.state.helper.active_slot) == 1
    assert int(low_update.state.helper.active_slot) == SCRATCH_SLOT


def test_hysteresis_relocks_moderate_durable_but_only_high_scratch_confirms() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=0.1,
        lease_length=5,
        confirmation_steps=5,
        durable_retrieval_threshold=0.5,
        candidate_confirmation_threshold=0.75,
        candidate_confirmation_leases=2,
    )
    moderate_rewards = (1.0, 1.0, 1.0, 0.0, 0.0)
    status = initial.helper.status.at[1].set(SLOT_DURABLE).at[2].set(SLOT_DURABLE)
    generation = initial.helper.generation.at[1].set(1).at[2].set(2)
    durable_state = _replace_both(
        initial,
        status=status,
        generation=generation,
        active_slot=jnp.int32(2),
        next_generation=jnp.int32(3),
    )
    durable_bits = np.asarray(durable_state.helper.values[1]).view(np.uint32).copy()
    for _ in range(5):
        search_update = _one_update(agent, durable_state, 0.0)
        durable_state = search_update.state
    assert int(durable_state.helper.active_slot) == 1
    assert int(durable_state.helper.remaining_durable_tests) == 1
    for reward in moderate_rewards:
        durable_update = _one_update(agent, durable_state, reward)
        durable_state = durable_update.state
    assert float(durable_update.helper.lease_reward_mean) == pytest.approx(0.6)
    assert bool(durable_update.helper.durable_relevant)
    assert not bool(durable_update.helper.candidate_relevant)
    assert int(durable_state.helper.active_slot) == 1
    assert int(durable_state.helper.remaining_durable_tests) == 0
    assert int(durable_state.helper.failed_leases[1]) == 0
    assert int(durable_update.helper.committed_slot) == -1
    assert int(durable_update.helper.retired_slot) == -1
    np.testing.assert_array_equal(
        np.asarray(durable_state.helper.values[1]).view(np.uint32),
        durable_bits,
    )

    scratch_state = _replace_both(
        initial,
        status=status,
        generation=generation,
        next_generation=jnp.int32(3),
    )
    for reward in moderate_rewards:
        scratch_update = _one_update(agent, scratch_state, reward)
        scratch_state = scratch_update.state
    assert bool(scratch_update.helper.durable_relevant)
    assert not bool(scratch_update.helper.candidate_relevant)
    assert not bool(scratch_update.helper.candidate_lease_success)
    assert int(scratch_update.helper.committed_slot) == -1
    assert int(scratch_state.helper.candidate_successful_leases) == 0
    assert int(scratch_state.helper.active_slot) == 1

    high_state = initial
    for _ in range(5):
        first_high_lease = _one_update(agent, high_state, 1.0)
        high_state = first_high_lease.state
    assert bool(first_high_lease.helper.candidate_relevant)
    assert bool(first_high_lease.helper.candidate_lease_success)
    assert int(high_state.helper.candidate_successful_leases) == 1
    assert int(high_state.helper.active_slot) == SCRATCH_SLOT
    for _ in range(5):
        second_high_lease = _one_update(agent, high_state, 1.0)
        high_state = second_high_lease.state
    assert bool(second_high_lease.helper.candidate_lease_success)
    assert int(second_high_lease.helper.committed_slot) == 1
    assert int(high_state.helper.generation[1]) == 1


def test_one_frozen_role_blocks_vacancy_commit_and_atomic_replacement() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
    )
    helper_bits = np.asarray(initial.helper.values).view(np.uint32).copy()
    blocked_commit = _one_update(
        agent,
        initial,
        1.0,
        helper_write=False,
        beneficiary_write=True,
    )
    assert int(blocked_commit.helper.committed_slot) == -1
    assert int(blocked_commit.beneficiary.committed_slot) == -1
    for before, after in (
        (initial.helper, blocked_commit.state.helper),
        (initial.beneficiary, blocked_commit.state.beneficiary),
    ):
        np.testing.assert_array_equal(after.status, before.status)
        np.testing.assert_array_equal(after.generation, before.generation)
    np.testing.assert_array_equal(
        np.asarray(blocked_commit.state.helper.values).view(np.uint32),
        helper_bits,
    )
    assert np.count_nonzero(np.asarray(blocked_commit.state.beneficiary.values[0])) > 0
    assert bool(blocked_commit.lifecycle_synchronized)

    values = initial.helper.values.at[0, 0, 0].set(0.5)
    values = values.at[1:].set(jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3))
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 5, 6, 7), dtype=jnp.int32)
    replacement_candidate = _replace_both(
        initial,
        values=values,
        status=status,
        generation=generation,
        failed_leases=jnp.asarray((0, 1, 8, 2), dtype=jnp.int32),
        idle_leases=jnp.asarray((0, 2, 9, 4), dtype=jnp.int32),
        next_generation=jnp.int32(8),
    )
    frozen_durable_bits = np.asarray(replacement_candidate.helper.values[1:]).view(np.uint32).copy()
    blocked_replacement = _one_update(
        agent,
        replacement_candidate,
        1.0,
        helper_write=False,
        beneficiary_write=True,
    )
    assert int(blocked_replacement.helper.retired_slot) == -1
    assert int(blocked_replacement.beneficiary.retired_slot) == -1
    assert int(blocked_replacement.helper.committed_slot) == -1
    assert int(blocked_replacement.beneficiary.committed_slot) == -1
    for role in (
        blocked_replacement.state.helper,
        blocked_replacement.state.beneficiary,
    ):
        np.testing.assert_array_equal(role.status, status)
        np.testing.assert_array_equal(role.generation, generation)
    np.testing.assert_array_equal(
        np.asarray(blocked_replacement.state.helper.values[1:]).view(np.uint32),
        frozen_durable_bits,
    )
    assert bool(blocked_replacement.lifecycle_synchronized)


def test_selective_durable_table_is_bitwise_immutable_while_being_tested() -> None:
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
    )
    durable = jnp.asarray(
        ((-0.0, 0.25, -1.0), (2.0, -3.5, 4.0), (5.5, -6.0, 7.25)),
        dtype=jnp.float32,
    )
    values = initial.helper.values.at[1].set(durable)
    status = initial.helper.status.at[1].set(SLOT_DURABLE)
    state = _replace_both(
        initial,
        values=values,
        status=status,
        active_slot=jnp.int32(1),
    )
    before = np.asarray(state.helper.values[1]).view(np.uint32).copy()
    update = _one_update(agent, state, 0.0)
    assert not bool(update.helper.value_write)
    assert int(update.state.helper.status[1]) == SLOT_DURABLE
    np.testing.assert_array_equal(
        np.asarray(update.state.helper.values[1]).view(np.uint32),
        before,
    )


def test_durable_mismatch_records_failure_without_deletion_or_value_change() -> None:
    agent, initial = _agent(
        learning_rate=0.5,
        epsilon=0.0,
        lease_length=1,
        confirmation_steps=1,
        relevance_rate=1.0,
    )
    values = initial.helper.values.at[1].set(jnp.ones((3, 3), dtype=jnp.float32))
    status = initial.helper.status.at[1].set(SLOT_DURABLE)
    state = _replace_both(
        initial,
        values=values,
        relevance_mean=initial.helper.relevance_mean.at[1].set(0.8),
        relevance_mass=initial.helper.relevance_mass.at[1].set(7.0),
        status=status,
        generation=initial.helper.generation.at[1].set(9),
        active_slot=jnp.int32(1),
    )
    durable_bits = np.asarray(state.helper.values[1]).view(np.uint32).copy()
    update = _one_update(agent, state, 0.0)
    for role_update in (update.helper, update.beneficiary):
        assert int(role_update.retired_slot) == -1
        assert int(role_update.retired_generation) == -1
        role = role_update.state
        assert int(role.status[1]) == SLOT_DURABLE
        np.testing.assert_array_equal(
            np.asarray(role.values[1]).view(np.uint32),
            durable_bits,
        )
        assert int(role.failed_leases[1]) == 1
        assert int(role.idle_leases[1]) == 1
        assert int(role.generation[1]) == 9
        assert int(role.active_slot) == SCRATCH_SLOT


def test_confirmed_candidate_atomically_replaces_stale_generation() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=2,
    )
    values = initial.helper.values.at[0, 0, 0].set(0.25)
    values = values.at[1:].set(jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3))
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 11, 12, 13), dtype=jnp.int32)
    failed = jnp.asarray((0, 2, 7, 7), dtype=jnp.int32)
    idle = jnp.asarray((0, 10, 4, 9), dtype=jnp.int32)
    state = _replace_both(
        initial,
        values=values,
        status=status,
        generation=generation,
        failed_leases=failed,
        idle_leases=idle,
        next_generation=jnp.int32(14),
    )
    durable_bits = np.asarray(state.helper.values[1:]).view(np.uint32).copy()

    first_lease = _one_update(agent, state, 1.0)
    assert int(first_lease.helper.committed_slot) == -1
    assert int(first_lease.helper.retired_slot) == -1
    assert int(first_lease.state.helper.active_slot) == SCRATCH_SLOT
    assert int(first_lease.state.helper.candidate_successful_leases) == 1
    np.testing.assert_array_equal(
        np.asarray(first_lease.state.helper.values[1:]).view(np.uint32),
        durable_bits,
    )

    replacement = _one_update(agent, first_lease.state, 1.0)
    for role_update in (replacement.helper, replacement.beneficiary):
        assert int(role_update.retired_slot) == 3
        assert int(role_update.retired_generation) == 13
        assert int(role_update.committed_slot) == 3
        assert int(role_update.committed_generation) == 14
        role = role_update.state
        np.testing.assert_array_equal(
            role.status,
            np.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE)),
        )
        np.testing.assert_array_equal(role.generation, np.asarray((0, 11, 12, 14)))
        assert int(role.active_slot) == 3
        assert int(role.candidate_successful_leases) == 0
        assert int(role.failed_leases[3]) == 0
        assert int(role.idle_leases[3]) == 0
    np.testing.assert_array_equal(
        np.asarray(replacement.state.helper.values[1:3]).view(np.uint32),
        durable_bits[:2],
    )
    assert not np.array_equal(
        np.asarray(replacement.state.helper.values[3]).view(np.uint32),
        durable_bits[2],
    )


def test_default_short_transient_cannot_replace_full_durable_bank() -> None:
    agent, initial = _agent()
    values = initial.helper.values.at[1:].set(jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3))
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    generation = jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)
    state = _replace_both(
        initial,
        values=values,
        status=status,
        generation=generation,
        lease_offset=jnp.int32(16),
        lease_reward_sum=jnp.float32(16.0),
        next_generation=jnp.int32(4),
    )
    durable_bits = np.asarray(state.helper.values[1:]).view(np.uint32).copy()
    for _ in range(16):
        update = _one_update(agent, state, 1.0)
        assert int(update.helper.committed_slot) == -1
        assert int(update.helper.retired_slot) == -1
        state = update.state
    assert int(state.helper.active_slot) == SCRATCH_SLOT
    assert int(state.helper.candidate_successful_leases) == 1
    np.testing.assert_array_equal(state.helper.generation, generation)
    np.testing.assert_array_equal(
        np.asarray(state.helper.values[1:]).view(np.uint32),
        durable_bits,
    )


def test_ab_recurrence_relocks_stored_module_without_scratch_replacement() -> None:
    agent, initial = _agent(
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
    )
    durable_a = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    durable_b = durable_a + 20.0
    values = initial.helper.values.at[1].set(durable_a).at[2].set(durable_b)
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_VACANT))
    generation = jnp.asarray((0, 1, 2, 0), dtype=jnp.int32)
    state = _replace_both(
        initial,
        values=values,
        status=status,
        generation=generation,
        active_slot=jnp.int32(1),
        next_generation=jnp.int32(3),
    )
    durable_bits = np.asarray(state.helper.values[1:3]).view(np.uint32).copy()

    find_b = _one_update(agent, state, 0.0)
    assert int(find_b.state.helper.active_slot) == 2
    lock_b = _one_update(agent, find_b.state, 1.0)
    assert int(lock_b.state.helper.active_slot) == 2
    find_a = _one_update(agent, lock_b.state, 0.0)
    assert int(find_a.state.helper.active_slot) == 1
    lock_a = _one_update(agent, find_a.state, 1.0)
    assert int(lock_a.state.helper.active_slot) == 1
    assert int(lock_a.helper.committed_slot) == -1
    assert int(lock_a.helper.retired_slot) == -1
    np.testing.assert_array_equal(lock_a.state.helper.generation, generation)
    np.testing.assert_array_equal(
        np.asarray(lock_a.state.helper.values[1:3]).view(np.uint32),
        durable_bits,
    )


def test_writable_lru_ablation_replaces_oldest_slot_with_identical_resources() -> None:
    config = SlotSignalingConfig(
        learning_rate=1.0,
        epsilon=0.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
        writable_lru_ablation=True,
    )
    agent = SlotSignalingAgent(config)
    initial = agent.init(slot_signaling_keys(jr.key(22)))
    values = initial.helper.values
    values = values.at[0, 0, 0].set(0.25)
    values = values.at[1:].set(jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3))
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    idle = jnp.asarray((0, 1, 7, 3), dtype=jnp.int32)
    state = _replace_both(initial, values=values, status=status, idle_leases=idle)
    update = _one_update(agent, state, 1.0)
    assert int(update.helper.committed_slot) == 2
    assert int(update.beneficiary.committed_slot) == 2
    assert int(update.helper.retired_slot) == 2
    assert int(update.beneficiary.retired_slot) == 2
    assert int(update.state.helper.status[2]) == SLOT_DURABLE
    assert int(update.state.helper.idle_leases[2]) == 0
    assert np.count_nonzero(np.asarray(update.state.helper.values[2])) > 0
    np.testing.assert_array_equal(update.state.helper.values[0], np.zeros((3, 3)))
    assert slot_signaling_resource_budget(update.state).state_bytes == 552


@pytest.mark.parametrize(
    "durable_policy,replacement_policy,expected_write,expected_target",
    [
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE, False, 1),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_EVIDENCE, True, 1),
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_LRU, False, 2),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU, True, 2),
    ],
)
def test_factorial_axes_independently_control_durable_writes_and_replacement_target(
    durable_policy: str,
    replacement_policy: str,
    expected_write: bool,
    expected_target: int,
) -> None:
    config = SlotSignalingConfig(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
        durable_write_policy=durable_policy,  # type: ignore[arg-type]
        replacement_target_policy=replacement_policy,  # type: ignore[arg-type]
    )
    agent = SlotSignalingAgent(config)
    initial = agent.init(slot_signaling_keys(jr.key(221)))
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))

    durable_state = _replace_both(
        initial,
        status=status,
        active_slot=jnp.int32(3),
        values=initial.helper.values.at[3, 0, 0].set(0.5),
    )
    durable_update = _one_update(agent, durable_state, 0.0)
    assert bool(durable_update.helper.value_write) is expected_write

    # Evidence targets slot 1 (largest failed streak); LRU targets slot 2.
    candidate_state = _replace_both(
        initial,
        status=status,
        values=initial.helper.values.at[0, 0, 0].set(0.5),
        failed_leases=jnp.asarray((0, 9, 0, 0), dtype=jnp.int32),
        idle_leases=jnp.asarray((0, 1, 8, 3), dtype=jnp.int32),
    )
    commit = _one_update(agent, candidate_state, 1.0)
    assert int(commit.helper.committed_slot) == expected_target
    assert int(commit.helper.retired_slot) == expected_target
    budget = slot_signaling_resource_budget(commit.state)
    assert budget.helper.state_scalars == 69
    assert budget.helper.state_bytes == 276
    assert budget.beneficiary.state_scalars == 69
    assert budget.beneficiary.state_bytes == 276
    assert budget.state_scalars == 138
    assert budget.state_bytes == 552


def test_generation_exhaustion_blocks_commit_without_reusing_identity() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
    )
    state = _replace_both(
        initial,
        next_generation=jnp.int32(np.iinfo(np.int32).max),
    )
    update = _one_update(agent, state, 1.0)
    for role in (update.helper, update.beneficiary):
        assert bool(role.generation_exhausted)
        assert int(role.committed_slot) == -1
        assert int(role.committed_generation) == -1
        assert int(role.retired_slot) == -1
        assert int(role.state.next_generation) == np.iinfo(np.int32).max
        np.testing.assert_array_equal(role.state.generation, state.helper.generation)


def test_public_role_transition_exactly_reproduces_joint_role_updates() -> None:
    agent, state = _agent(
        learning_rate=0.5,
        epsilon=0.0,
        relevance_rate=1.0,
        lease_length=1,
        confirmation_steps=1,
        candidate_confirmation_leases=1,
    )
    helper = agent.select_helper(state.helper, jnp.int32(2))
    beneficiary = agent.select_beneficiary(state.beneficiary, helper.action)
    joint = agent.update(state, helper, beneficiary, jnp.float32(1.0))
    helper_local = agent.update_role(
        state.helper,
        helper,
        jnp.float32(1.0),
        value_write=True,
        lifecycle_write=True,
    )
    beneficiary_local = agent.update_role(
        state.beneficiary,
        beneficiary,
        jnp.float32(1.0),
        value_write=True,
        lifecycle_write=True,
    )
    _assert_pytree_bit_equal(helper_local, joint.helper)
    _assert_pytree_bit_equal(beneficiary_local, joint.beneficiary)


def test_separate_role_instances_reproduce_a_full_joint_lifecycle() -> None:
    agent, joint_state = _agent(
        learning_rate=0.5,
        epsilon=0.1,
        relevance_rate=0.5,
        lease_length=4,
        confirmation_steps=1,
        durable_retrieval_threshold=0.25,
        candidate_confirmation_threshold=0.75,
        candidate_confirmation_leases=2,
        scratch_training_leases_before_retest=2,
    )
    helper_state = joint_state.helper
    beneficiary_state = joint_state.beneficiary
    commit_events = 0
    active_slot_changes = 0

    for step in range(192):
        cue = jnp.int32(step % 3)
        joint_helper = agent.select_helper(joint_state.helper, cue)
        local_helper = agent.select_helper(helper_state, cue)
        _assert_pytree_bit_equal(local_helper, joint_helper)

        joint_beneficiary = agent.select_beneficiary(
            joint_state.beneficiary,
            joint_helper.action,
        )
        local_beneficiary = agent.select_beneficiary(
            beneficiary_state,
            local_helper.action,
        )
        _assert_pytree_bit_equal(local_beneficiary, joint_beneficiary)

        # Alternating successful and failed leases exercises commit, durable
        # failure/search, and scratch-retest paths without either local role
        # reading the other role's values, observation, action, or key.
        reward = jnp.float32(1.0 if (step // 16) % 2 == 0 else 0.0)
        joint = agent.update(
            joint_state,
            joint_helper,
            joint_beneficiary,
            reward,
        )
        helper_local = agent.update_role(
            helper_state,
            local_helper,
            reward,
            value_write=True,
            lifecycle_write=True,
        )
        beneficiary_local = agent.update_role(
            beneficiary_state,
            local_beneficiary,
            reward,
            value_write=True,
            lifecycle_write=True,
        )

        _assert_pytree_bit_equal(helper_local, joint.helper)
        _assert_pytree_bit_equal(beneficiary_local, joint.beneficiary)
        commit_events += int(helper_local.committed_slot >= 0)
        active_slot_changes += int(
            helper_local.state.active_slot != helper_state.active_slot
        )
        joint_state = joint.state
        helper_state = helper_local.state
        beneficiary_state = beneficiary_local.state

    _assert_pytree_bit_equal(helper_state, joint_state.helper)
    _assert_pytree_bit_equal(beneficiary_state, joint_state.beneficiary)
    assert commit_events > 0
    assert active_slot_changes > 0


def test_selective_full_bank_waits_for_candidate_confirmation() -> None:
    agent, initial = _agent(
        learning_rate=1.0,
        epsilon=0.0,
        lease_length=1,
        confirmation_steps=1,
    )
    values = initial.helper.values.at[0, 0, 0].set(0.5)
    values = values.at[1:].set(0.75)
    status = jnp.asarray((SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE))
    state = _replace_both(initial, values=values, status=status)
    update = _one_update(agent, state, 1.0)
    assert int(update.helper.committed_slot) == -1
    assert int(update.helper.retired_slot) == -1
    assert int(update.state.helper.active_slot) == SCRATCH_SLOT
    assert int(update.state.helper.candidate_successful_leases) == 1
    np.testing.assert_array_equal(update.state.helper.values[1:], state.helper.values[1:])
    assert np.count_nonzero(np.asarray(update.state.helper.values[0])) > 0

    failed_candidate = _one_update(agent, update.state, 0.0)
    assert int(failed_candidate.helper.committed_slot) == -1
    assert int(failed_candidate.helper.retired_slot) == -1
    assert int(failed_candidate.state.helper.active_slot) == 1
    assert int(failed_candidate.state.helper.remaining_durable_tests) == 3
    assert int(failed_candidate.state.helper.candidate_successful_leases) == 0
    np.testing.assert_array_equal(
        failed_candidate.state.helper.values[1:],
        state.helper.values[1:],
    )


def test_named_keys_randomize_zero_ties_and_greedy_probe_is_read_only() -> None:
    agent = SlotSignalingAgent(SlotSignalingConfig(epsilon=0.0))
    actions = {
        int(
            agent.select_helper(
                agent.init(slot_signaling_keys(jr.key(seed))).helper,
                jnp.int32(0),
            ).action
        )
        for seed in range(24)
    }
    assert actions == {0, 1, 2}
    state = agent.init(slot_signaling_keys(jr.key(8))).helper
    values_before = np.asarray(state.values).copy()
    key_before = np.asarray(jr.key_data(state.key)).copy()
    assert int(greedy_slot_action(state.values, jnp.int32(0), jnp.int32(0))) == 0
    np.testing.assert_array_equal(state.values, values_before)
    np.testing.assert_array_equal(jr.key_data(state.key), key_before)


def test_joint_agent_is_jittable_scannable_finite_and_resource_constant() -> None:
    agent, state = _agent(
        learning_rate=0.2,
        epsilon=0.1,
        lease_length=4,
        confirmation_steps=3,
    )
    budget_before = slot_signaling_resource_budget(state)

    @jax.jit
    def run(initial_state):
        def body(old_state, inputs):
            cue, reward = inputs
            helper = agent.select_helper(old_state.helper, cue)
            beneficiary = agent.select_beneficiary(old_state.beneficiary, helper.action)
            update = agent.update(old_state, helper, beneficiary, reward)
            diagnostics = jnp.stack(
                (
                    update.helper.candidate_value,
                    update.beneficiary.candidate_value,
                    update.helper.lease_reward_mean,
                    update.lifecycle_synchronized.astype(jnp.float32),
                )
            )
            return update.state, diagnostics

        steps = jnp.arange(128, dtype=jnp.int32)
        inputs = (
            steps % 3,
            ((steps % 5) != 0).astype(jnp.float32),
        )
        return jax.lax.scan(body, initial_state, inputs)

    final_state, diagnostics = run(state)
    assert diagnostics.shape == (128, 4)
    assert bool(jnp.all(jnp.isfinite(diagnostics)))
    assert bool(jnp.all(diagnostics[:, 3] == 1.0))
    for role in (final_state.helper, final_state.beneficiary):
        assert role.values.shape == SLOT_VALUE_SHAPE
        assert bool(jnp.all(jnp.isfinite(role.values)))
        assert bool(jnp.all(jnp.isfinite(role.relevance_mean)))
        assert 0 <= int(role.active_slot) < N_SLOTS
        assert 0 <= int(role.lease_offset) < agent.config.lease_length
        assert 0 <= int(role.remaining_durable_tests) <= 3
        assert 1 <= int(role.search_cursor) <= 3
        assert (
            0 <= int(role.candidate_successful_leases) <= agent.config.candidate_confirmation_leases
        )
        assert (
            0
            <= int(role.failed_leases[SCRATCH_SLOT])
            < agent.config.scratch_training_leases_before_retest
        )
        active = int(role.active_slot)
        assert active == SCRATCH_SLOT or int(role.status[active]) == SLOT_DURABLE
        np.testing.assert_array_equal(
            np.asarray(role.generation)[np.asarray(role.status) != SLOT_DURABLE],
            0,
        )
    assert slot_signaling_resource_budget(final_state) == budget_before
