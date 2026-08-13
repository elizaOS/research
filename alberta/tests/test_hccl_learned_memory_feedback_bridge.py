"""Integration contracts for HCCL causal feedback into learned memory."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.core as core_api
import alberta_framework.core.hccl_learned_memory_feedback_bridge as bridge_module
from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCLActionLayer,
    HCCLActionReceipt,
)
from alberta_framework.core.hccl_learned_memory_feedback_bridge import (
    HCCL_LEARNED_MEMORY_FEEDBACK_STATUS,
    HCCLLearnedMemoryFeedbackBridge,
    HCCLLearnedMemoryFeedbackBridgeConfig,
    HCCLLearnedMemoryFeedbackBridgeState,
    load_hccl_learned_memory_feedback_checkpoint,
    measure_hccl_learned_memory_feedback_state_nbytes,
    run_hccl_learned_memory_feedback_scan,
    save_hccl_learned_memory_feedback_checkpoint,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterConfig,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _controller_config(
    *,
    capacity: int = 4,
    top_k: int = 1,
    max_delta: float = 1.0,
) -> LearnedExperientialMemoryControllerConfig:
    return LearnedExperientialMemoryControllerConfig(
        memory=ExperientialMemoryConfig(
            capacity=capacity,
            observation_dim=2,
            key_dim=2,
            action_dim=2,
            outcome_dim=1,
            top_k=top_k,
            min_neighbors=1,
            distance_scale=1.0,
            min_similarity=0.0,
            min_effective_reliability=1.0e-6,
            max_uncertainty=1.0,
            max_safety_cost=1.0,
            max_age=100,
            staleness_scale=100.0,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=0.0,
            recency_scale=10.0,
        ),
        admission_step_size=0.5,
        retention_step_size=1.0,
        admission_threshold=0.0,
        initial_admission_bias=0.0,
        max_abs_admission_weight=8.0,
        max_abs_counterfactual_delta=max_delta,
        retention_prior=0.5,
    )


def _bridge_config(
    *,
    capacity: int = 4,
    top_k: int = 1,
    max_delta: float = 1.0,
    max_scan_steps: int = 4,
) -> HCCLLearnedMemoryFeedbackBridgeConfig:
    return HCCLLearnedMemoryFeedbackBridgeConfig(
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
        controller=_controller_config(
            capacity=capacity,
            top_k=top_k,
            max_delta=max_delta,
        ),
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
        max_host_scan_steps=max_scan_steps,
    )


def _entry(
    provenance: int,
    *,
    action_id: int,
    key: tuple[float, float] = (0.0, 0.0),
) -> ExperientialMemoryEntry:
    action = jnp.zeros((2,), dtype=jnp.float32).at[action_id].set(jnp.float32(1.0))
    return ExperientialMemoryEntry(
        observation=jnp.asarray(key, dtype=jnp.float32),
        key=jnp.asarray(key, dtype=jnp.float32),
        action=action,
        outcome=jnp.asarray((1.0,), dtype=jnp.float32),
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(0.5, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
    )


def _prime_controller(
    config: LearnedExperientialMemoryControllerConfig,
    *,
    action_id: int,
    capacity_entries: tuple[tuple[int, int], ...] | None = None,
) -> Any:
    controller = LearnedExperientialMemoryController(config)
    state = controller.init()
    entries = capacity_entries or ((1, action_id),)
    for provenance, stored_action in entries:
        result = controller.step(
            state,
            jnp.asarray((20.0, 20.0), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            _entry(provenance, action_id=stored_action),
        )
        assert bool(result.diagnostics.transaction_applied)
        assert not bool(result.state.pending.available)
        state = result.state
    return state


def _bridge(
    *,
    action_id: int,
    capacity: int = 4,
    top_k: int = 1,
    max_delta: float = 1.0,
    seed: int = 0,
    capacity_entries: tuple[tuple[int, int], ...] | None = None,
) -> tuple[HCCLLearnedMemoryFeedbackBridge, HCCLLearnedMemoryFeedbackBridgeState]:
    config = _bridge_config(capacity=capacity, top_k=top_k, max_delta=max_delta)
    bridge = HCCLLearnedMemoryFeedbackBridge(config)
    controller_state = _prime_controller(
        config.controller,
        action_id=action_id,
        capacity_entries=capacity_entries,
    )
    return bridge, bridge.init(jr.key(seed), controller_state=controller_state)


def _identity_rows(offset: int) -> jax.Array:
    return jnp.arange(offset + 1, offset + 9, dtype=jnp.uint32).reshape((2, 4))


def _receipts(
    bridge: HCCLLearnedMemoryFeedbackBridge,
    state: HCCLLearnedMemoryFeedbackBridgeState,
    event: Any,
    *,
    agent: int,
    base_action: int,
    retrieved_action: int,
    routed: bool = True,
    retrieved_safe: bool = True,
) -> tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt]:
    other = 1 - agent
    base_actions = [0, 0]
    base_actions[agent] = base_action
    memory_before = list(base_actions)
    memory_after = list(base_actions)
    masks = np.ones((2, 2), dtype=np.bool_)
    if routed:
        memory_before[agent] = retrieved_action
        masks[agent, retrieved_action] = retrieved_safe
        memory_after[agent] = retrieved_action if retrieved_safe else base_action
    else:
        masks[agent, retrieved_action] = retrieved_safe
    memory_after[other] = base_actions[other]
    planner_actions = list(memory_after)
    all_masks = jnp.asarray(masks, dtype=jnp.bool_)
    source = state.hccl_state
    base = bridge.hccl.bind_action_receipt(
        source,
        event,
        layer=HCCLActionLayer.BASE,
        actions_before_mask=jnp.asarray(base_actions, dtype=jnp.int32),
        actions_after_mask=jnp.asarray(base_actions, dtype=jnp.int32),
        hard_action_masks=all_masks,
        action_receipt_identity_words=_identity_rows(100),
    )
    memory = bridge.hccl.bind_action_receipt(
        source,
        event,
        layer=HCCLActionLayer.MEMORY,
        actions_before_mask=jnp.asarray(memory_before, dtype=jnp.int32),
        actions_after_mask=jnp.asarray(memory_after, dtype=jnp.int32),
        hard_action_masks=all_masks,
        action_receipt_identity_words=_identity_rows(140),
    )
    planner = bridge.hccl.bind_action_receipt(
        source,
        event,
        layer=HCCLActionLayer.PLANNER,
        actions_before_mask=jnp.asarray(planner_actions, dtype=jnp.int32),
        actions_after_mask=jnp.asarray(planner_actions, dtype=jnp.int32),
        hard_action_masks=all_masks,
        action_receipt_identity_words=_identity_rows(180),
    )
    return base, memory, planner


def _prepare(
    bridge: HCCLLearnedMemoryFeedbackBridge,
    state: HCCLLearnedMemoryFeedbackBridgeState,
    event: Any,
    receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    *,
    agent: int,
    routed: bool = True,
    entry_action: int = 0,
    entry_key: tuple[float, float] = (1.0, 0.0),
    provenance: int = 2,
) -> Any:
    return bridge.prepare_retrieval(
        state,
        event,
        receipts[0],
        receipts[1],
        agent_index=jnp.asarray(agent, dtype=jnp.int32),
        retrieval_routed=jnp.asarray(routed, dtype=jnp.bool_),
        query_key=jnp.asarray((0.0, 0.0), dtype=jnp.float32),
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry=_entry(provenance, action_id=entry_action, key=entry_key),
    )


def _settle(
    bridge: HCCLLearnedMemoryFeedbackBridge,
    prepared: Any,
    event: Any,
    receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    *,
    gate: bool = True,
) -> Any:
    return bridge.stage_feedback(
        prepared.state,
        event,
        *receipts,
        downstream_candidate_valid=jnp.asarray(gate, dtype=jnp.bool_),
    )


def _stack(*items: object) -> object:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *items)


def test_config_state_ownership_exports_and_authority_are_strict() -> None:
    config = _bridge_config()
    bridge = HCCLLearnedMemoryFeedbackBridge(config)
    payload = bridge.to_config()
    assert payload["mechanism_status"] == HCCL_LEARNED_MEMORY_FEEDBACK_STATUS
    assert payload["mechanism_status"] == "l0-development-hccl-learned-memory-feedback-only"
    assert payload["hccl_state_owners"] == 1
    assert payload["controller_state_owners"] == 1
    assert payload["fixed_pending_bindings"] == 1
    assert payload["composite_jit_supported"] is False
    assert payload["delight_or_actor_backward"] is False
    for name in (
        "caller_identity_authenticated",
        "agent_implementation_present",
        "schedule_execution_authorized",
        "output_writes_authorized",
        "artifact_authorized",
        "threshold_authorized",
        "seed_authority",
        "evidence_authorized",
        "promotion_authorized",
    ):
        assert payload[name] is False
    assert HCCLLearnedMemoryFeedbackBridge.from_config(payload).to_config() == payload
    assert alberta_framework.HCCLLearnedMemoryFeedbackBridge is HCCLLearnedMemoryFeedbackBridge
    assert core_api.HCCLLearnedMemoryFeedbackBridgeConfig is HCCLLearnedMemoryFeedbackBridgeConfig

    state = bridge.init(jr.key(0))
    assert tuple(state.__dataclass_fields__) == (
        "hccl_state",
        "controller_state",
        "pending_binding",
    )
    assert not bool(state.pending_binding.available)
    assert bool(bridge.state_valid(state))
    malformed = dict(payload)
    malformed["evidence_authorized"] = True
    with pytest.raises(ValueError, match="config|unsupported"):
        HCCLLearnedMemoryFeedbackBridge.from_config(malformed)


@pytest.mark.parametrize(
    ("agent", "base_action", "retrieved_action", "sign"),
    ((0, 1, 0, 1), (1, 0, 1, -1), (0, 0, 0, 0)),
)
def test_committed_attribution_settles_positive_negative_and_zero_effects(
    agent: int,
    base_action: int,
    retrieved_action: int,
    sign: int,
) -> None:
    bridge, state = _bridge(action_id=retrieved_action)
    event = bridge.hccl.world.prepare_event(state.hccl_state.world_state)
    receipts = _receipts(
        bridge,
        state,
        event,
        agent=agent,
        base_action=base_action,
        retrieved_action=retrieved_action,
    )
    prepared = _prepare(bridge, state, event, receipts, agent=agent)
    assert bool(prepared.update_applied)
    assert bool(prepared.state.pending_binding.available)
    assert int(prepared.state.pending_binding.agent_index) == agent
    assert int(prepared.state.pending_binding.retrieved_action_id) == retrieved_action
    chex.assert_trees_all_equal(
        prepared.state.pending_binding.controller_transaction_words,
        prepared.state.controller_state.pending.transaction_words,
    )
    chex.assert_trees_all_equal(
        prepared.state.pending_binding.hccl_source_words,
        state.hccl_state.world_state.step_words,
    )
    chex.assert_trees_all_equal(
        prepared.state.pending_binding.base_action_receipt_identity_words,
        receipts[0].action_receipt_identity_words,
    )
    chex.assert_trees_all_equal(
        prepared.state.pending_binding.memory_action_receipt_identity_words,
        receipts[1].action_receipt_identity_words,
    )

    result = _settle(bridge, prepared, event, receipts)
    assert bool(result.update_applied)
    delta = float(result.counterfactual_delta)
    assert (delta > 0.0) if sign > 0 else (delta < 0.0) if sign < 0 else (delta == 0.0)
    assert int(result.state.controller_state.feedback_count) == 1
    assert int(result.state.controller_state.learned_feedback_count) == 1
    assert not bool(result.state.controller_state.pending.available)
    assert not bool(result.state.pending_binding.available)
    if sign > 0:
        assert int(result.state.controller_state.positive_feedback_count) == 1
    else:
        assert int(result.state.controller_state.nonpositive_feedback_count) == 1
    if sign == 0:
        chex.assert_trees_all_equal(
            result.state.controller_state.admission_weights,
            prepared.state.controller_state.admission_weights,
        )
    chex.assert_trees_all_equal(
        result.state.hccl_state.world_state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )


@pytest.mark.parametrize(
    ("routed", "retrieved_safe"),
    ((False, True), (True, False)),
)
def test_unrouted_or_masked_retrieval_clears_only_via_no_learning_settlement(
    routed: bool,
    retrieved_safe: bool,
) -> None:
    bridge, state = _bridge(action_id=1)
    event = bridge.hccl.world.prepare_event(state.hccl_state.world_state)
    receipts = _receipts(
        bridge,
        state,
        event,
        agent=0,
        base_action=0,
        retrieved_action=1,
        routed=routed,
        retrieved_safe=retrieved_safe,
    )
    prepared = _prepare(bridge, state, event, receipts, agent=0, routed=routed)
    assert bool(prepared.update_applied)
    assert not bool(prepared.state.pending_binding.retrieval_used)
    result = _settle(bridge, prepared, event, receipts)
    assert bool(result.update_applied)
    assert bool(result.controller_feedback.diagnostics.transaction_applied)
    assert not bool(result.controller_feedback.diagnostics.learning_eligible)
    assert int(result.state.controller_state.feedback_count) == 1
    assert int(result.state.controller_state.learned_feedback_count) == 0
    assert not bool(result.state.controller_state.pending.available)
    assert not bool(result.state.pending_binding.available)


def test_categorical_action_mask_and_single_agent_timing_fail_closed() -> None:
    bridge, state = _bridge(action_id=1)
    event = bridge.hccl.world.prepare_event(state.hccl_state.world_state)
    receipts = _receipts(
        bridge,
        state,
        event,
        agent=0,
        base_action=0,
        retrieved_action=1,
    )
    mismatched_memory = dataclasses.replace(  # type: ignore[type-var]
        receipts[1],
        actions_before_mask=receipts[1].actions_before_mask.at[0].set(0),
    )
    rejected = _prepare(
        bridge,
        state,
        event,
        (receipts[0], mismatched_memory, receipts[2]),
        agent=0,
    )
    assert not bool(rejected.categorical_action_timing_valid)
    chex.assert_trees_all_equal(rejected.state, state)

    changed_other = bridge.hccl.bind_action_receipt(
        state.hccl_state,
        event,
        layer=HCCLActionLayer.MEMORY,
        actions_before_mask=jnp.asarray((1, 1), dtype=jnp.int32),
        actions_after_mask=jnp.asarray((1, 1), dtype=jnp.int32),
        hard_action_masks=jnp.ones((2, 2), dtype=jnp.bool_),
        action_receipt_identity_words=_identity_rows(140),
    )
    rejected = _prepare(
        bridge,
        state,
        event,
        (receipts[0], changed_other, receipts[2]),
        agent=0,
    )
    assert not bool(rejected.unbound_agent_unchanged)
    chex.assert_trees_all_equal(rejected.state, state)

    soft_bridge, soft_state = _bridge(
        action_id=0,
        capacity=3,
        top_k=2,
        capacity_entries=((1, 0), (2, 1)),
    )
    soft_event = soft_bridge.hccl.world.prepare_event(soft_state.hccl_state.world_state)
    soft_receipts = _receipts(
        soft_bridge,
        soft_state,
        soft_event,
        agent=0,
        base_action=1,
        retrieved_action=0,
    )
    soft = _prepare(soft_bridge, soft_state, soft_event, soft_receipts, agent=0)
    assert not bool(soft.retrieved_action_categorical)
    chex.assert_trees_all_equal(soft.state, soft_state)


def test_stale_tampered_cross_event_failed_child_and_same_event_retry_are_atomic() -> None:
    bridge, state = _bridge(action_id=0)
    event = bridge.hccl.world.prepare_event(state.hccl_state.world_state)
    receipts = _receipts(
        bridge,
        state,
        event,
        agent=0,
        base_action=1,
        retrieved_action=0,
    )
    prepared = _prepare(bridge, state, event, receipts, agent=0)

    alternate_base = bridge.hccl.bind_action_receipt(
        state.hccl_state,
        event,
        layer=HCCLActionLayer.BASE,
        actions_before_mask=receipts[0].actions_before_mask,
        actions_after_mask=receipts[0].actions_after_mask,
        hard_action_masks=receipts[0].hard_action_masks,
        action_receipt_identity_words=_identity_rows(220),
    )
    mismatched = _settle(
        bridge,
        prepared,
        event,
        (alternate_base, receipts[1], receipts[2]),
    )
    assert bool(mismatched.hccl_result.update_applied)
    assert bool(mismatched.controller_feedback.diagnostics.transaction_applied)
    assert not bool(mismatched.binding_matches_action_receipts)
    assert not bool(mismatched.attribution_source_bound_and_committed)
    assert not bool(mismatched.controller_settlement_applied)
    assert not bool(mismatched.update_applied)
    chex.assert_trees_all_equal(mismatched.state, prepared.state)

    rejected = _settle(bridge, prepared, event, receipts, gate=False)
    assert not bool(rejected.update_applied)
    assert not bool(rejected.attribution_source_bound_and_committed)
    assert not bool(rejected.controller_settlement_applied)
    assert bool(rejected.controller_feedback.diagnostics.transaction_applied)
    chex.assert_trees_all_equal(rejected.state, prepared.state)
    chex.assert_trees_all_equal(
        bridge.hccl.world.prepare_event(rejected.state.hccl_state.world_state),
        event,
    )
    retry = _settle(bridge, rejected, event, receipts)
    assert bool(retry.update_applied)

    other = bridge.init(
        jr.key(3),
        controller_state=state.controller_state,
    )
    other_event = bridge.hccl.world.prepare_event(other.hccl_state.world_state)
    crossed = _settle(bridge, prepared, other_event, receipts)
    assert not bool(crossed.binding_matches_event)
    chex.assert_trees_all_equal(crossed.state, prepared.state)

    tampered_binding = dataclasses.replace(  # type: ignore[type-var]
        prepared.state.pending_binding,
        retrieved_action_id=jnp.asarray(1, dtype=jnp.int32),
    )
    tampered_state = dataclasses.replace(  # type: ignore[type-var]
        prepared.state,
        pending_binding=tampered_binding,
    )
    tampered = bridge.stage_feedback(
        tampered_state,
        event,
        *receipts,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(tampered.source_state_valid)
    chex.assert_trees_all_equal(tampered.state, tampered_state)


def test_counterfactual_bound_and_slot_reuse_are_donor_checked() -> None:
    bounded, bounded_state = _bridge(action_id=0, max_delta=0.05)
    event = bounded.hccl.world.prepare_event(bounded_state.hccl_state.world_state)
    receipts = _receipts(
        bounded,
        bounded_state,
        event,
        agent=0,
        base_action=1,
        retrieved_action=0,
    )
    prepared = _prepare(bounded, bounded_state, event, receipts, agent=0)
    rejected = _settle(bounded, prepared, event, receipts)
    assert not bool(rejected.counterfactual_within_controller_bound)
    assert not bool(rejected.update_applied)
    assert bool(rejected.hccl_result.update_applied)
    assert not bool(rejected.controller_feedback.diagnostics.transaction_applied)
    assert not bool(rejected.attribution_source_bound_and_committed)
    assert not bool(rejected.controller_settlement_applied)
    chex.assert_trees_all_equal(rejected.state, prepared.state)

    bridge, state = _bridge(action_id=0, capacity=1)
    event = bridge.hccl.world.prepare_event(state.hccl_state.world_state)
    receipts = _receipts(
        bridge,
        state,
        event,
        agent=0,
        base_action=1,
        retrieved_action=0,
    )
    prepared = _prepare(bridge, state, event, receipts, agent=0)
    assert bool(prepared.controller_step.evicted)
    settled = _settle(bridge, prepared, event, receipts)
    assert bool(settled.update_applied)
    assert bool(settled.controller_feedback.diagnostics.admission_updated)
    assert int(settled.controller_feedback.diagnostics.retention_rows_updated) == 0


def test_resources_and_in_memory_checkpoint_are_strict() -> None:
    bridge, state = _bridge(action_id=0)
    budget = bridge.resource_budget(state)
    measured = measure_hccl_learned_memory_feedback_state_nbytes(state)
    assert budget.total_persistent_state_nbytes == measured
    assert budget.hccl_state_owners == 1
    assert budget.controller_state_owners == 1
    assert budget.fixed_pending_bindings == 1
    assert budget.max_world_proposal_calls_per_feedback == 8
    assert budget.max_controller_settlements_per_feedback == 1
    assert budget.output_write_calls == 0
    checkpoint = save_hccl_learned_memory_feedback_checkpoint(bridge, state)
    restored_bridge, restored = load_hccl_learned_memory_feedback_checkpoint(checkpoint)
    assert restored_bridge.to_config() == bridge.to_config()
    chex.assert_trees_all_equal(restored, state)
    tampered = dataclasses.replace(
        checkpoint,
        state=cast(Any, checkpoint.state).replace(
            controller_state=cast(Any, checkpoint.state.controller_state).replace(
                admission_weights=checkpoint.state.controller_state.admission_weights.at[0].add(
                    jnp.float32(0.1)
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_hccl_learned_memory_feedback_checkpoint(tampered)


def test_config_checkpoint_and_public_resource_types_are_canonical() -> None:
    bridge, state = _bridge(action_id=0)
    config = bridge.to_config()

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(TypeError, match="exact dict"):
        HCCLLearnedMemoryFeedbackBridge.from_config(DictSubclass(config))
    for field, alias in (
        ("hccl_state_owners", True),
        ("output_writes_authorized", 0),
    ):
        aliased = dict(config)
        aliased[field] = alias
        with pytest.raises(ValueError, match="unsupported"):
            HCCLLearnedMemoryFeedbackBridge.from_config(aliased)

    budget = bridge.resource_budget(state)
    with pytest.raises(TypeError, match="exact int"):
        dataclasses.replace(budget, hccl_state_owners=True)
    with pytest.raises(ValueError, match="byte total"):
        dataclasses.replace(
            budget,
            total_persistent_state_nbytes=budget.total_persistent_state_nbytes + 1,
        )

    checkpoint = save_hccl_learned_memory_feedback_checkpoint(bridge, state)

    def reseal(**changes: object) -> object:
        changed = dataclasses.replace(checkpoint, **changes)  # type: ignore[arg-type]
        return dataclasses.replace(
            changed,
            checkpoint_sha256=bridge_module._checkpoint_digest(changed),
        )

    with pytest.raises(ValueError, match="output_writes_authorized"):
        load_hccl_learned_memory_feedback_checkpoint(
            cast(Any, reseal(output_writes_authorized=0))
        )
    with pytest.raises(TypeError, match="config must be an exact dict"):
        load_hccl_learned_memory_feedback_checkpoint(
            cast(Any, reseal(config=DictSubclass(checkpoint.config)))
        )
    aliased_budget = dict(checkpoint.resource_budget)
    aliased_budget["hccl_state_owners"] = True
    with pytest.raises(ValueError, match="resource budget"):
        load_hccl_learned_memory_feedback_checkpoint(
            cast(Any, reseal(resource_budget=aliased_budget))
        )
    with pytest.raises(TypeError, match="state_nbytes must be an exact int"):
        load_hccl_learned_memory_feedback_checkpoint(
            cast(Any, reseal(state_nbytes=True))
        )


def test_bounded_host_prebound_scan_matches_manual_and_rejects_composite_jit() -> None:
    bridge, initial = _bridge(action_id=0)
    event_0 = bridge.hccl.world.prepare_event(initial.hccl_state.world_state)
    receipts_0 = _receipts(
        bridge,
        initial,
        event_0,
        agent=0,
        base_action=1,
        retrieved_action=0,
    )
    entry_0 = _entry(2, action_id=1, key=(1.0, 0.0))
    prepared_0 = bridge.prepare_retrieval(
        initial,
        event_0,
        receipts_0[0],
        receipts_0[1],
        agent_index=jnp.asarray(0, dtype=jnp.int32),
        retrieval_routed=jnp.asarray(True, dtype=jnp.bool_),
        query_key=jnp.asarray((0.0, 0.0), dtype=jnp.float32),
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry=entry_0,
    )
    manual_0 = _settle(bridge, prepared_0, event_0, receipts_0)

    event_1 = bridge.hccl.world.prepare_event(manual_0.state.hccl_state.world_state)
    receipts_1 = _receipts(
        bridge,
        manual_0.state,
        event_1,
        agent=1,
        base_action=0,
        retrieved_action=1,
    )
    entry_1 = _entry(3, action_id=0, key=(2.0, 0.0))
    prepared_1 = bridge.prepare_retrieval(
        manual_0.state,
        event_1,
        receipts_1[0],
        receipts_1[1],
        agent_index=jnp.asarray(1, dtype=jnp.int32),
        retrieval_routed=jnp.asarray(True, dtype=jnp.bool_),
        query_key=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry=entry_1,
    )
    manual_1 = _settle(bridge, prepared_1, event_1, receipts_1)
    assert bool(manual_1.update_applied)

    scanned = run_hccl_learned_memory_feedback_scan(
        bridge,
        initial,
        cast(Any, _stack(event_0, event_1)),
        cast(Any, _stack(receipts_0[0], receipts_1[0])),
        cast(Any, _stack(receipts_0[1], receipts_1[1])),
        cast(Any, _stack(receipts_0[2], receipts_1[2])),
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((True, True), dtype=jnp.bool_),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((0.1, 0.1), dtype=jnp.float32),
        jnp.asarray((True, True), dtype=jnp.bool_),
        cast(Any, _stack(entry_0, entry_1)),
        jnp.asarray((True, True), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(scanned.state, manual_1.state)
    np.testing.assert_array_equal(np.asarray(scanned.update_applied), (True, True))
    np.testing.assert_array_equal(
        np.asarray(scanned.hccl_post_transaction_words),
        ((0, 1), (0, 2)),
    )
    with pytest.raises(TypeError, match="host/eager only"):
        jax.jit(run_hccl_learned_memory_feedback_scan, static_argnums=(0,))(
            bridge,
            initial,
            cast(Any, _stack(event_0, event_1)),
            cast(Any, _stack(receipts_0[0], receipts_1[0])),
            cast(Any, _stack(receipts_0[1], receipts_1[1])),
            cast(Any, _stack(receipts_0[2], receipts_1[2])),
            jnp.asarray((0, 1), dtype=jnp.int32),
            jnp.asarray((True, True), dtype=jnp.bool_),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float32),
            jnp.asarray((0, 0), dtype=jnp.int32),
            jnp.asarray((0.1, 0.1), dtype=jnp.float32),
            jnp.asarray((True, True), dtype=jnp.bool_),
            cast(Any, _stack(entry_0, entry_1)),
            jnp.asarray((True, True), dtype=jnp.bool_),
        )
