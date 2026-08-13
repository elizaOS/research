# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,operator,type-var"
"""Red-first contracts for one HCCL owner and two live learned memories.

The HCCL causal world fixes the learner observation width at 16, so this is
the smallest compatible external-state composition.  The tests intentionally
exercise mechanism and ownership only.  ``delight_or_actor_backward=False``
is a protocol fact, not a claim that a gradient did or did not "spark joy."
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_hccl_external_coordinator_base_bridge import _coordinator_config

from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapterConfig,
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    HCCL_TWO_LIVE_MEMORY_STATUS,
    HCCLTwoLiveMemoryActionBinding,
    HCCLTwoLiveMemoryBridge,
    HCCLTwoLiveMemoryBridgeConfig,
    HCCLTwoLiveMemoryBridgeState,
    load_hccl_two_live_memory_checkpoint,
    measure_hccl_two_live_memory_state_nbytes,
    save_hccl_two_live_memory_checkpoint,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterConfig,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
N_ACTIONS = 2
RAW_DIM = 16
MM_SLOT = 0
B0M1_SLOT = 1
M0B1_SLOT = 2
BB_SLOT = 3
PP_SLOT = 4


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> object:
    if request.node.name == "test_resources_checkpoint_and_host_only_bounds_are_strict":
        yield
    else:
        with jax.disable_jit():
            yield


def _live_config(*, max_events: int = 8) -> ExternalLearnedStateLiveMemoryAdapterConfig:
    memory = ExperientialMemoryConfig(
        capacity=2,
        observation_dim=RAW_DIM,
        key_dim=RAW_DIM,
        action_dim=N_ACTIONS,
        outcome_dim=RAW_DIM,
        top_k=1,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=1.0e-6,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=8,
        staleness_scale=8.0,
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=2.0,
    )
    return ExternalLearnedStateLiveMemoryAdapterConfig(
        coordinator=_coordinator_config(max_events=max_events),
        learned_memory=LearnedExperientialMemoryControllerConfig(
            memory=memory,
            admission_step_size=0.1,
            retention_step_size=0.1,
            admission_threshold=0.0,
            initial_admission_bias=0.0,
            max_abs_counterfactual_delta=100.0,
        ),
    )


def _config(
    *,
    agent_0_max_events: int = 8,
    agent_1_max_events: int = 8,
) -> HCCLTwoLiveMemoryBridgeConfig:
    return HCCLTwoLiveMemoryBridgeConfig(
        hccl=HCCLWorldAttributionAdapterConfig(
            proposal_owner_digest=(
                0x10203040,
                0x50607080,
                0x90A0B0C0,
                0xD0E0F001,
                0x12345678,
                0x9ABCDEF0,
                0x0F1E2D3C,
                0x4B5A6978,
            )
        ),
        agent_0=_live_config(max_events=agent_0_max_events),
        agent_1=_live_config(max_events=agent_1_max_events),
        binding_owner_digest=(
            0xCAFEBABE,
            0x0BADF00D,
            0x13579BDF,
            0x2468ACE1,
            0x31415927,
            0x27182819,
            0x11235813,
            0x21345591,
        ),
    )


def _bridge() -> tuple[HCCLTwoLiveMemoryBridge, HCCLTwoLiveMemoryBridgeState]:
    bridge = HCCLTwoLiveMemoryBridge(_config())
    return bridge, bridge.init(jr.key(7))


def _event_input(provenance: int) -> ExternalLearnedStateLiveMemoryEventInput:
    return ExternalLearnedStateLiveMemoryEventInput(
        query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        entry_safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_reliability=jnp.asarray(1.0, dtype=jnp.float32),
        provenance_id=jnp.asarray(provenance, dtype=jnp.int32),
        source_id=jnp.asarray(19, dtype=jnp.int32),
    )


def _inputs(provenance: int) -> tuple[
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryEventInput,
]:
    return _event_input(2 * provenance), _event_input(2 * provenance + 1)


def _stage(
    bridge: HCCLTwoLiveMemoryBridge,
    state: HCCLTwoLiveMemoryBridgeState,
    event: object,
    binding: HCCLTwoLiveMemoryActionBinding,
    event_inputs: tuple[
        ExternalLearnedStateLiveMemoryEventInput,
        ExternalLearnedStateLiveMemoryEventInput,
    ],
    next_masks: jax.Array,
    *,
    gate: bool = True,
) -> Any:
    return bridge.stage(
        state,
        event,
        binding,
        event_inputs[0],
        event_inputs[1],
        next_decision_hard_action_masks=next_masks,
        downstream_candidate_valid=jnp.asarray(gate, dtype=jnp.bool_),
    )


def _child_states(
    state: HCCLTwoLiveMemoryBridgeState,
) -> tuple[
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryAdapterState,
]:
    return state.agent_0_state, state.agent_1_state


def _one_hot_masks(actions: jax.Array) -> jax.Array:
    return jax.nn.one_hot(actions, N_ACTIONS, dtype=jnp.bool_)


def _pp(result: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf[PP_SLOT], result.hccl_result.world_proposals)


def _tree_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        np.testing.assert_array_equal(np.asarray(left_array), np.asarray(right_array))


def _replace_stored_record(
    bridge: HCCLTwoLiveMemoryBridge,
    state: HCCLTwoLiveMemoryBridgeState,
    *,
    agent: int,
    slot: int,
    query_key: jax.Array,
    action: int,
) -> HCCLTwoLiveMemoryBridgeState:
    child = _child_states(state)[agent]
    learned = child.learned_memory_state
    memory = learned.memory
    entries = dataclasses.replace(
        memory.entries,
        keys=memory.entries.keys.at[slot].set(query_key),
        actions=memory.entries.actions.at[slot].set(
            jax.nn.one_hot(action, N_ACTIONS, dtype=jnp.float32)
        ),
    )
    changed_child = child.replace(
        learned_memory_state=learned.replace(
            memory=dataclasses.replace(memory, entries=entries)
        )
    )
    changed = (
        state.replace(agent_0_state=changed_child)
        if agent == 0
        else state.replace(agent_1_state=changed_child)
    )
    assert bool(bridge.state_valid(changed))
    return cast(HCCLTwoLiveMemoryBridgeState, changed)


def _contrast_exact_zero(contrast: object) -> bool:
    return all(
        bool(jnp.all(jax.lax.bitcast_convert_type(jnp.asarray(leaf), jnp.uint32) == 0))
        for leaf in jax.tree.leaves(contrast)
    )


def test_config_and_exact_owner_topology_are_nonduplicating() -> None:
    bridge, state = _bridge()
    payload = bridge.to_config()
    assert payload["mechanism_status"] == HCCL_TWO_LIVE_MEMORY_STATUS
    assert payload["mechanism_status"] == (
        "l0-development-hccl-two-live-memory-causal-feedback"
    )
    assert payload["hccl_state_owners"] == 1
    assert payload["live_memory_adapter_state_owners"] == 2
    assert payload["external_coordinator_state_owners"] == 2
    assert payload["learned_memory_controller_state_owners"] == 2
    assert payload["prototype_state_owners"] == 2
    assert payload["additional_coordinator_state_owners"] == 0
    assert payload["additional_memory_controller_state_owners"] == 0
    assert payload["additional_prototype_state_owners"] == 0
    assert payload["current_hard_action_mask_bindings"] == 1
    assert payload["planner_action_relation"] == "P=M-no-planner-rung"
    assert payload["per_agent_memory_feedback"] == ["M0B1-BB", "B0M1-BB"]
    assert payload["memory_interaction_usage"] == "separate-audit-fact-only"
    assert payload["delight_or_actor_backward"] is False
    assert payload["delight_interpretation"] == "protocol-fact-only-not-evaluated"
    assert payload["composite_jit_supported"] is False
    assert payload["scan_supported"] is False
    for name in (
        "caller_identity_authenticated",
        "planner_authority",
        "dispatch_authority",
        "safety_authority",
        "schedule_execution_authorized",
        "seed_authority",
        "output_writes_authorized",
        "artifact_authorized",
        "threshold_authorized",
        "evidence_authorized",
        "promotion_authorized",
    ):
        assert payload[name] is False
    assert HCCLTwoLiveMemoryBridge.from_config(payload).to_config() == payload

    assert tuple(state.__dataclass_fields__) == (
        "hccl_state",
        "agent_0_state",
        "agent_1_state",
        "current_hard_action_masks",
    )
    chex.assert_trees_all_equal(
        state.current_hard_action_masks,
        jnp.ones((N_AGENTS, N_ACTIONS), dtype=jnp.bool_),
    )
    assert not bool(state.agent_0_state.pending_binding.available)
    assert not bool(state.agent_1_state.pending_binding.available)
    assert bool(bridge.state_valid(state))
    explicit = bridge.init(
        jr.key(7),
        initial_hard_action_masks=jnp.ones(
            (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
        ),
    )
    _tree_exact(explicit, state)


@pytest.mark.parametrize(
    ("field", "alias"),
    (
        ("delight_or_actor_backward", 0),
        ("hccl_state_owners", True),
        ("planner_authority", 0),
    ),
)
def test_config_rejects_bool_integer_canonical_type_aliases(
    field: str,
    alias: object,
) -> None:
    bridge, _ = _bridge()
    payload = bridge.to_config()
    payload[field] = alias

    with pytest.raises(ValueError, match="unsupported"):
        HCCLTwoLiveMemoryBridge.from_config(payload)


@pytest.mark.parametrize("field", ("per_agent_memory_feedback", "limitations"))
def test_config_rejects_tuple_for_canonical_list(field: str) -> None:
    bridge, _ = _bridge()
    payload = bridge.to_config()
    value = payload[field]
    assert type(value) is list
    payload[field] = tuple(cast(list[object], value))

    with pytest.raises(ValueError, match="unsupported"):
        HCCLTwoLiveMemoryBridge.from_config(payload)


def test_checkpoint_rejects_resealed_boolean_integer_alias() -> None:
    bridge, state = _bridge()
    checkpoint = save_hccl_two_live_memory_checkpoint(bridge, state)
    aliased = dataclasses.replace(
        checkpoint,
        output_writes_authorized=cast(Any, 0),
    )
    from alberta_framework.core import hccl_two_live_memory_bridge as module

    resealed = dataclasses.replace(
        aliased,
        checkpoint_sha256=module._checkpoint_digest(aliased),
    )
    with pytest.raises(ValueError, match="output_writes_authorized"):
        load_hccl_two_live_memory_checkpoint(resealed)


def test_checkpoint_rejects_resealed_resource_boolean_integer_alias() -> None:
    bridge, state = _bridge()
    checkpoint = save_hccl_two_live_memory_checkpoint(bridge, state)
    resource = dict(checkpoint.resource_budget)
    resource["hccl_state_owners"] = True
    aliased = dataclasses.replace(checkpoint, resource_budget=resource)
    from alberta_framework.core import hccl_two_live_memory_bridge as module

    resealed = dataclasses.replace(
        aliased,
        checkpoint_sha256=module._checkpoint_digest(aliased),
    )
    with pytest.raises(ValueError, match="resource budget"):
        load_hccl_two_live_memory_checkpoint(resealed)


def test_first_abstention_then_two_b_to_m_bindings_feed_only_unilateral_effects() -> None:
    bridge, initial = _bridge()
    all_true = jnp.ones((N_AGENTS, N_ACTIONS), dtype=jnp.bool_)

    event_0 = bridge.prepare_event(initial)
    binding_0 = bridge.bind_live_memory_actions(initial, event_0)
    current_actions = jnp.stack(
        (
            initial.agent_0_state.coordinator_state.current_action,
            initial.agent_1_state.coordinator_state.current_action,
        )
    )
    assert not bool(jnp.any(binding_0.feedback_binding_available))
    chex.assert_trees_all_equal(binding_0.base_actions, current_actions)
    chex.assert_trees_all_equal(binding_0.memory_actions, current_actions)
    chex.assert_trees_all_equal(binding_0.current_hard_action_masks, all_true)
    for receipt in (binding_0.base, binding_0.memory, binding_0.planner):
        chex.assert_trees_all_equal(receipt.actions_before_mask, current_actions)
        chex.assert_trees_all_equal(receipt.actions_after_mask, current_actions)
        chex.assert_trees_all_equal(receipt.hard_action_masks, all_true)

    inputs_0 = _inputs(1)
    preview_0 = _stage(
        bridge,
        initial,
        event_0,
        binding_0,
        inputs_0,
        all_true,
        gate=False,
    )
    preview_actions = jnp.stack(
        (
            preview_0.agent_0_result.prepared.coordinator_result.state.current_action,
            preview_0.agent_1_result.prepared.coordinator_result.state.current_action,
        )
    )
    abstaining_next_masks = _one_hot_masks(preview_actions)

    first = _stage(
        bridge,
        initial,
        event_0,
        binding_0,
        inputs_0,
        abstaining_next_masks,
    )
    assert bool(first.update_applied)
    chex.assert_trees_all_equal(
        first.prior_feedback_required,
        jnp.zeros((N_AGENTS,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(first.prior_feedback_supplied, first.prior_feedback_required)
    assert not bool(first.agent_0_result.prepared.feedback_supplied)
    assert not bool(first.agent_1_result.prepared.feedback_supplied)
    assert not bool(first.state.agent_0_state.pending_binding.available)
    assert not bool(first.state.agent_1_state.pending_binding.available)
    chex.assert_trees_all_equal(
        first.state.current_hard_action_masks,
        abstaining_next_masks,
    )
    assert bool(first.next_decision_masks_installed)
    assert bool(first.current_event_masks_bound)
    assert bool(first.planner_equals_memory)
    assert bool(first.pp_executes_memory_actions)
    assert bool(first.no_planner_rung_valid)
    assert not bool(first.delight_or_actor_backward)
    pp_0 = _pp(first)
    chex.assert_trees_all_equal(pp_0.joint_action_ids, binding_0.memory_actions)
    for index, transition in enumerate(
        (first.agent_0_transition, first.agent_1_transition)
    ):
        chex.assert_trees_all_equal(transition.action, binding_0.memory_actions[index])

    # The second event is previewed without adoption so its exact next raw
    # query keys and coordinator-only base actions can seed a categorical
    # witness without consuming another HCCL event.
    event_1 = bridge.prepare_event(first.state)
    binding_1 = bridge.bind_live_memory_actions(first.state, event_1)
    chex.assert_trees_all_equal(
        binding_1.current_hard_action_masks,
        abstaining_next_masks,
    )
    inputs_1 = _inputs(2)
    preview_1 = _stage(
        bridge,
        first.state,
        event_1,
        binding_1,
        inputs_1,
        all_true,
        gate=False,
    )
    base_next_actions = jnp.stack(
        (
            preview_1.agent_0_result.prepared.coordinator_result.state.current_action,
            preview_1.agent_1_result.prepared.coordinator_result.state.current_action,
        )
    )
    retrieved_actions = 1 - base_next_actions
    seeded = first.state
    for agent, result in enumerate((first.agent_0_result, first.agent_1_result)):
        seeded = _replace_stored_record(
            bridge,
            seeded,
            agent=agent,
            slot=int(result.prepared.learned_memory_result.slot),
            query_key=(
                preview_1.agent_0_result.prepared.query_key
                if agent == 0
                else preview_1.agent_1_result.prepared.query_key
            ),
            action=int(retrieved_actions[agent]),
        )

    binding_1 = bridge.bind_live_memory_actions(seeded, event_1)
    armed = _stage(
        bridge,
        seeded,
        event_1,
        binding_1,
        inputs_1,
        all_true,
    )
    assert bool(armed.update_applied)
    chex.assert_trees_all_equal(armed.state.current_hard_action_masks, all_true)
    for agent, child in enumerate(_child_states(armed.state)):
        pending = child.pending_binding
        assert bool(pending.available)
        assert bool(pending.categorical_retrieval)
        assert bool(pending.retrieval_used_expected)
        assert int(pending.base_action_before_retrieval) == int(
            base_next_actions[agent]
        )
        assert int(pending.effective_action) == int(retrieved_actions[agent])
        assert int(pending.base_action_before_retrieval) != int(
            pending.effective_action
        )
        chex.assert_trees_all_equal(pending.hard_action_mask, all_true[agent])
        chex.assert_trees_all_equal(
            pending.hard_action_mask,
            armed.state.current_hard_action_masks[agent],
        )

    mask_corruption = armed.state.replace(
        current_hard_action_masks=armed.state.current_hard_action_masks.at[
            0, retrieved_actions[0]
        ].set(False)
    )
    assert not bool(bridge.state_valid(mask_corruption))

    # Matching mask metadata is not sufficient: the mask must admit both the
    # coordinator-only base action and the retrieved effective action.
    pending_0 = armed.state.agent_0_state.pending_binding
    masked_base = pending_0.hard_action_mask.at[
        pending_0.base_action_before_retrieval
    ].set(False)
    base_mask_corruption = armed.state.replace(
        agent_0_state=armed.state.agent_0_state.replace(
            pending_binding=pending_0.replace(hard_action_mask=masked_base)
        ),
        current_hard_action_masks=armed.state.current_hard_action_masks.at[0].set(
            masked_base
        ),
    )
    assert not bool(bridge.state_valid(base_mask_corruption))

    event_2 = bridge.prepare_event(armed.state)
    binding_2 = bridge.bind_live_memory_actions(armed.state, event_2)
    assert bool(jnp.all(binding_2.feedback_binding_available))
    expected_transactions = jnp.stack(
        tuple(
            child.pending_binding.memory_transaction_words
            for child in _child_states(armed.state)
        )
    )
    expected_decisions = jnp.stack(
        tuple(
            child.pending_binding.prototype_decision_id
            for child in _child_states(armed.state)
        )
    )
    expected_base = jnp.stack(
        tuple(
            child.pending_binding.base_action_before_retrieval
            for child in _child_states(armed.state)
        )
    )
    expected_memory = jnp.stack(
        tuple(
            child.pending_binding.effective_action
            for child in _child_states(armed.state)
        )
    )
    chex.assert_trees_all_equal(
        binding_2.live_memory_transaction_words,
        expected_transactions,
    )
    chex.assert_trees_all_equal(binding_2.prototype_decision_words, expected_decisions)
    chex.assert_trees_all_equal(binding_2.base_actions, expected_base)
    chex.assert_trees_all_equal(binding_2.memory_actions, expected_memory)
    chex.assert_trees_all_equal(binding_2.current_hard_action_masks, all_true)
    chex.assert_trees_all_equal(
        binding_2.planner.actions_after_mask,
        binding_2.memory.actions_after_mask,
    )

    inputs_2 = _inputs(3)
    missing_binding = dataclasses.replace(
        binding_2,
        feedback_binding_available=binding_2.feedback_binding_available.at[0].set(False),
    )
    missing = _stage(
        bridge,
        armed.state,
        event_2,
        missing_binding,
        inputs_2,
        all_true,
    )
    assert not bool(missing.feedback_bindings_complete)
    assert not bool(missing.update_applied)
    _tree_exact(missing.state, armed.state)

    stale_binding = dataclasses.replace(
        binding_2,
        live_memory_transaction_words=binding_2.live_memory_transaction_words.at[
            1, 1
        ].add(jnp.uint32(1)),
    )
    stale_feedback = _stage(
        bridge,
        armed.state,
        event_2,
        stale_binding,
        inputs_2,
        all_true,
    )
    assert not bool(stale_feedback.feedback_bindings_match_children)
    assert not bool(stale_feedback.update_applied)
    _tree_exact(stale_feedback.state, armed.state)

    final = _stage(
        bridge,
        armed.state,
        event_2,
        binding_2,
        inputs_2,
        all_true,
    )
    assert bool(final.update_applied)
    proposals = final.hccl_result.world_proposals
    expected_agent_0 = (
        proposals.signals.net_reward[M0B1_SLOT, 0]
        - proposals.signals.net_reward[BB_SLOT, 0]
    )
    expected_agent_1 = (
        proposals.signals.net_reward[B0M1_SLOT, 1]
        - proposals.signals.net_reward[BB_SLOT, 1]
    )
    chex.assert_trees_all_equal(
        final.agent_unilateral_counterfactual_delta,
        jnp.stack((expected_agent_0, expected_agent_1)),
    )
    chex.assert_trees_all_equal(
        final.agent_0_feedback.counterfactual_delta,
        expected_agent_0,
    )
    chex.assert_trees_all_equal(
        final.agent_1_feedback.counterfactual_delta,
        expected_agent_1,
    )
    assert bool(final.agent_0_feedback_is_m0b1_minus_bb)
    assert bool(final.agent_1_feedback_is_b0m1_minus_bb)
    assert not bool(final.mm_minus_bb_broadcast_to_both_agents)
    assert not bool(final.memory_interaction_used_for_agent_feedback)
    chex.assert_trees_all_equal(
        final.memory_interaction_audit,
        final.hccl_result.attribution.contrasts.memory_interaction,
    )
    interaction = final.hccl_result.attribution.contrasts.memory_interaction
    if not _contrast_exact_zero(interaction):
        mm_total = final.hccl_result.attribution.contrasts.memory_total.net_reward
        assert not np.array_equal(
            np.asarray(final.agent_unilateral_counterfactual_delta),
            np.asarray(mm_total),
        )
    assert bool(final.planner_equals_memory)
    assert bool(final.no_planner_rung_valid)
    assert _contrast_exact_zero(final.hccl_result.attribution.contrasts.planner_total)
    assert _contrast_exact_zero(
        final.hccl_result.attribution.contrasts.planner_interaction
    )
    pp_2 = _pp(final)
    chex.assert_trees_all_equal(pp_2.joint_action_ids, binding_2.memory_actions)
    for index, (transition, child_result, feedback) in enumerate(
        (
            (final.agent_0_transition, final.agent_0_result, final.agent_0_feedback),
            (final.agent_1_transition, final.agent_1_result, final.agent_1_feedback),
        )
    ):
        chex.assert_trees_all_equal(transition.action, binding_2.memory_actions[index])
        chex.assert_trees_all_equal(child_result.prepared.feedback, feedback)
        assert bool(child_result.prepared.feedback_supplied)
        assert bool(child_result.diagnostics.prior_feedback_settled)
        assert bool(child_result.diagnostics.prior_feedback_learning_applied)

    reused = _stage(
        bridge,
        final.state,
        event_2,
        binding_2,
        inputs_2,
        all_true,
    )
    assert not bool(reused.binding_matches_source)
    assert not bool(reused.update_applied)
    _tree_exact(reused.state, final.state)


def test_either_live_child_failure_rolls_hccl_and_both_children_back() -> None:
    bridge, state = _bridge()
    event = bridge.prepare_event(state)
    binding = bridge.bind_live_memory_actions(state, event)
    valid_inputs = _inputs(10)
    all_true = jnp.ones((N_AGENTS, N_ACTIONS), dtype=jnp.bool_)
    for failed_agent in range(N_AGENTS):
        bad = list(valid_inputs)
        bad[failed_agent] = dataclasses.replace(
            bad[failed_agent],
            entry_reliability=jnp.asarray(jnp.nan, dtype=jnp.float32),
        )
        result = _stage(
            bridge,
            state,
            event,
            binding,
            cast(
                tuple[
                    ExternalLearnedStateLiveMemoryEventInput,
                    ExternalLearnedStateLiveMemoryEventInput,
                ],
                tuple(bad),
            ),
            all_true,
        )
        assert bool(result.hccl_result.update_applied)
        child_results = (result.agent_0_result, result.agent_1_result)
        assert not bool(child_results[failed_agent].diagnostics.transaction_applied)
        assert bool(child_results[1 - failed_agent].diagnostics.transaction_applied)
        assert not bool(result.hccl_commit_applied)
        assert not bool(result.agent_0_update_applied)
        assert not bool(result.agent_1_update_applied)
        assert not bool(result.next_decision_masks_installed)
        assert not bool(result.update_applied)
        _tree_exact(result.state, state)


def test_resources_checkpoint_and_host_only_bounds_are_strict() -> None:
    bridge, state = _bridge()
    payload = bridge.to_config()
    budget = bridge.resource_budget(state)
    measured = measure_hccl_two_live_memory_state_nbytes(state)
    assert budget.total_persistent_state_nbytes == measured
    assert budget.hccl_state_owners == 1
    assert budget.live_memory_adapter_state_owners == 2
    assert budget.external_coordinator_state_owners == 2
    assert budget.learned_memory_controller_state_owners == 2
    assert budget.prototype_state_owners == 2
    assert budget.current_hard_action_mask_bindings == 1
    assert budget.max_world_proposal_calls_per_transaction == 8
    assert budget.max_attribution_proposal_calls_per_transaction == 8
    assert budget.live_adapter_step_calls_per_transaction == 2
    assert budget.maximum_feedback_settlements_per_transaction == 2
    assert budget.maximum_cached_action_replacements_per_transaction == 2
    assert budget.planner_calls_per_transaction == 0
    assert budget.delight_or_actor_backward_calls_per_transaction == 0
    assert budget.composite_jit_supported is False
    assert budget.scan_supported is False
    assert budget.output_write_calls == 0
    assert budget.artifact_bytes_written == 0

    checkpoint = save_hccl_two_live_memory_checkpoint(bridge, state)
    assert checkpoint.output_writes_authorized is False
    assert checkpoint.artifact_authorized is False
    restored_bridge, restored = load_hccl_two_live_memory_checkpoint(checkpoint)
    assert restored_bridge.to_config() == payload
    _tree_exact(restored, state)
    tampered = dataclasses.replace(
        checkpoint,
        state=checkpoint.state.replace(
            current_hard_action_masks=checkpoint.state.current_hard_action_masks.at[
                0, 0
            ].set(False)
        ),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_hccl_two_live_memory_checkpoint(tampered)

    event = bridge.prepare_event(state)
    binding = bridge.bind_live_memory_actions(state, event)
    with pytest.raises((TypeError, RuntimeError), match="host|eager|JIT"):
        jax.jit(
            lambda source: bridge.stage(
                source,
                event,
                binding,
                _event_input(90),
                _event_input(91),
                next_decision_hard_action_masks=jnp.ones(
                    (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
                ),
                downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
            )
        )(state)
    assert not hasattr(bridge, "scan")
